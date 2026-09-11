"""Beaker repository integration for the AutomationBench + skills recipe.

The Beaker agent's editable surface is ``automationbench/skills`` — the folder
of ``SKILL.md`` guides the agent reads at runtime through ``list_skills`` /
``read_skill``. Everything else (the harness, the benchmark, this integration)
is off limits, so a candidate can only improve by writing better skills.

One case is one real AutomationBench rollout: ``run_one_async`` drives the
unmodified ``verifiers`` loop against the task's simulated workspace, pointed
at the candidate's ``skills/`` directory, and the benchmark's own deterministic
assertions score the final world state.

Dataset rows name tasks the way ``splits/{train,test}.txt`` do — by
``task_name``, the benchmark's only globally-unique stable id. The prompt, the
simulated initial state, and the ground-truth assertions all come from the
pinned ``automation-bench`` dependency, so the rows stay valid exactly as long
as that pin and the frozen splits move together.

The objective is ``partial_credit`` (the fraction of scored assertions the run
satisfied). The strict 0/1 ``task_completed_correctly`` is reported alongside it
but does not drive optimization: at the recipe's baseline it is 0 on almost
every task, which leaves an optimizer nothing to climb.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path
from typing import Any

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    JsonValue,
    RepositoryRunSetup,
    RetryableCaseError,
    RolloutRuntime,
    SetupRuntime,
    inference_target,
    repository,
)
from pydantic import BaseModel, Field


# The recipe's project root — the directory holding skills/ and pyproject.toml.
# In a hosted run this module is imported from the candidate's own checkout, so
# this resolves to that candidate's tree and picks up its edited skill files.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"

# Defaults for one rollout; override per run through the launch `extra` mapping.
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_MAX_STEPS = 50  # the benchmark's own --max-turns default
DEFAULT_TIMEOUT_SECONDS = 900.0

# Where a Beaker-selected comparison model's run-scoped key is placed for the
# application's own `api_key_var` lookup. Evaluator-process only.
GATEWAY_API_KEY_VAR = "BEAKER_GATEWAY_API_KEY"


class Row(BaseModel):
    """One task from a frozen split, keyed the way the split files key them."""

    task_name: str = Field(min_length=1)  # e.g. "sales.multi_hop_lookup"
    domain: str = Field(min_length=1)  # sales | marketing | operations | support | finance | hr
    task_family: str = Field(min_length=1)  # split-stratification family, e.g. "docusign"


@cache
def _samples_by_name() -> dict[str, Any]:
    from automationbench_skills.data.tasks import load_samples

    return {sample.task_name: sample for sample in load_samples()}


def _sample_for(task_name: str) -> Any:
    """Resolve a row's task against the pinned benchmark task set."""
    sample = _samples_by_name().get(task_name)
    if sample is None:
        raise ValueError(
            f"Task {task_name!r} is not in the pinned automation-bench task set. The "
            "dataset, the dependency pin, and the frozen split files must move together."
        )
    return sample


def _run_options(config: Any) -> dict[str, JsonValue]:
    """Per-run rollout knobs from the launch `extra` mapping."""
    extra = config if isinstance(config, dict) else {}
    return {
        "model": str(extra.get("model") or DEFAULT_MODEL),
        "max_steps": int(extra.get("max_steps") or DEFAULT_MAX_STEPS),
        "timeout_seconds": float(extra.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS),
    }


class Setup(RepositoryRunSetup[Row]):
    """Rows in, cases out. No clients to open: the benchmark is in-process."""

    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        sample = _sample_for(row.task_name)
        yield Case(
            id=row.task_name,
            input={
                "task_name": sample.task_name,
                "domain": sample.domain,
                # What the agent is actually given: the benchmark's own domain
                # system prompt plus the user's automation request.
                "prompt": sample.prompt,
                "available_tools": sample.info.get("zapier_tools", []),
                "run_options": _run_options(runtime.config),
            },
            # Ground truth: the assertion specs the benchmark checks against the
            # final world state. Held in the controller; never sent to run_case.
            expected={"assertions": sample.info.get("assertions", [])},
            metadata={"domain": row.domain, "task_family": row.task_family},
        )


def _model_spec(runtime: RolloutRuntime, requested_model: str) -> Any:
    """The application's own model selection, or a Beaker-selected one.

    Without ``runtime.model`` the recipe keeps its production routing and
    defaults, and Beaker's hosted provider routing serves the call. With one —
    an explicitly requested model comparison — the run-scoped gateway endpoint
    is threaded through the recipe's existing ``ModelSpec`` fields, pinned to
    Chat Completions because that is the shape the gateway serves.
    """
    from automationbench_skills.runner import ModelSpec

    if runtime.model is None:
        return ModelSpec(name=requested_model)
    target = inference_target(runtime)
    os.environ[GATEWAY_API_KEY_VAR] = target.api_key
    return ModelSpec(
        name=target.model,
        base_url=target.base_url,
        api_key_var=GATEWAY_API_KEY_VAR,
        api="chat_completions",
    )


def _record_names(end_state: dict[str, Any] | None) -> dict[str, str]:
    """Map record ids in the final world state to the names people call them.

    Assertion parameters address records by opaque id; this lets the scorer put
    "Meridian Corp - Platform Deal" in a check name instead of 006xx000004MER1.
    """
    names: dict[str, str] = {}
    if not end_state:
        return names
    label_keys = ("name", "title", "subject", "full_name", "email", "display_name")

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            identifier = node.get("id")
            if isinstance(identifier, str):
                for key in label_keys:
                    label = node.get(key)
                    if isinstance(label, str) and label.strip():
                        names.setdefault(identifier, label)
                        break
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(end_state)
    return names


def _simulated_apps() -> list[str]:
    """The simulated apps, longest name first, read off the world state itself.

    Used to group checks by the app an assertion inspects, without keeping a
    hand-written list in step with the pinned benchmark.
    """
    from automationbench.schema.world import WorldState

    return sorted((f for f in WorldState.model_fields if f != "meta"), key=len, reverse=True)


def _app_for(assertion_type: str, apps: list[str]) -> str | None:
    for app in apps:
        if assertion_type.startswith(f"{app}_"):
            return app
        # A few apps are named in the plural but assert in the singular
        # (facebook_pages -> facebook_page_post_exists).
        if app.endswith("s") and assertion_type.startswith(f"{app[:-1]}_"):
            return app
    return None


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime) -> CaseResult:
    """Run one AutomationBench rollout against the candidate's skills folder."""
    from beaker_tracing import instrument_env, traced_client
    from verifiers.legacy.errors import InfraError, ModelError

    from automationbench_skills.runner import get_env, run_one_async
    from automationbench_skills.vendored.model_setup import build_client, resolve_api_key_var

    assert isinstance(case_input, dict)
    options = case_input["run_options"]
    assert isinstance(options, dict)
    max_steps = int(options["max_steps"])  # type: ignore[arg-type]
    timeout = float(options["timeout_seconds"])  # type: ignore[arg-type]

    if not SKILLS_DIR.is_dir():
        raise RuntimeError(
            f"No skills directory at {SKILLS_DIR}. The candidate checkout is not laid "
            "out as this integration expects, so the case could not run."
        )

    sample = _sample_for(str(case_input["task_name"]))
    spec = _model_spec(runtime, str(options["model"]))
    resolved_api = spec.resolved_api()

    # Tool spans for the simulated workspace calls. The environment is cached and
    # shared across cases, so instrument it unpinned: each call records under
    # whichever case's capture is active.
    instrument_env(get_env(toolset="zapier", skills=True, max_steps=max_steps))

    plain = build_client(resolved_api, resolve_api_key_var(resolved_api, spec.api_key_var), spec.base_url)
    client = traced_client(plain, resolved_api=resolved_api)
    try:
        with runtime.trace.stage("automationbench.rollout") as stage:
            result = await run_one_async(
                sample,
                model=spec,
                skills_dir=SKILLS_DIR,
                max_steps=max_steps,
                timeout=timeout,
                client=client,
            )
            stage.output({"partial_credit": result.partial_credit, "error": str(result.error or "")})
    finally:
        await client.close()

    # A provider or infrastructure failure means the rollout never really ran;
    # the harness stores it in the rollout state and hands back an untouched
    # world, which would otherwise score a silent zero. An agent-side failure
    # (a bad tool call, an overlong prompt) is a legitimate zero and stays in
    # the output for the scorer to explain.
    error = result.error
    if isinstance(error, (ModelError, InfraError)):
        raise RetryableCaseError(f"{type(error).__name__}: {error}") from error

    return CaseResult(
        output={
            "task_name": result.task_name,
            "domain": result.domain,
            "partial_credit": result.partial_credit,
            "task_completed_correctly": result.task_completed_correctly,
            # Per-assertion outcomes from the benchmark's own rubric: type,
            # params, whether it passed, and whether it was excluded from
            # scoring (it already held before the agent acted).
            "assertion_results": result.assertion_results,
            "turns": len(result.trajectory),
            "error": str(error) if error is not None else None,
        },
        output_kind="record",
        # Scorer-only evidence: readable names for the ids the assertions
        # address. Droppable — the scorer falls back to the ids.
        context={"record_names": _record_names(result.end_state)},
    )


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    """Score one rollout with the benchmark's own partial-credit rule.

    ``partial_credit`` is passed / scored, where "scored" excludes assertions
    that already held in the task's initial state — the benchmark's own guard
    against earning credit for doing nothing. ``task_completed_correctly`` is
    1.0 only when every scored assertion passed.
    """
    del case_files_dir
    output = result.output
    assert isinstance(output, dict)
    context = result.context or {}
    names = context.get("record_names") or {}
    assert isinstance(names, dict)

    apps = _simulated_apps()
    outcomes = output.get("assertion_results") or []
    assert isinstance(outcomes, list)

    expected = case.expected if isinstance(case.expected, dict) else {}
    declared = expected.get("assertions") or []
    run_error = output.get("error")

    checks: list[Check] = []
    scored_passed = 0
    scored_total = 0

    for outcome in outcomes:
        assert isinstance(outcome, dict)
        kind = str(outcome.get("type", "assertion"))
        params = outcome.get("params") or {}
        assert isinstance(params, dict)
        excluded = bool(outcome.get("excluded"))
        passed = bool(outcome.get("passed"))
        if not excluded:
            scored_total += 1
            scored_passed += int(passed)

        # Name the assertion by what it checks, with ids swapped for the names
        # the workspace shows, so ten checks on one app stay distinguishable.
        readable = [names.get(v, v) if isinstance(v, str) else v for v in params.values()]
        label = " · ".join([kind, *(str(v) for v in readable)])
        checks.append(
            Check(
                name=label[:300],
                verdict="pass" if passed else "fail",
                group=_app_for(kind, apps),
                informational=excluded,
                message=(
                    "Already held before the agent acted, so it earns no credit."
                    if excluded and passed
                    else "Excluded from scoring by the task author."
                    if excluded
                    else None
                ),
            )
        )

    if not outcomes:
        # Nothing was evaluated: the rollout ended before the rubric could run.
        checks.append(
            Check(
                name="Rollout produced a scorable end state",
                verdict="fail",
                message=str(run_error) if run_error else "The rollout returned no assertion results.",
            )
        )
    elif run_error:
        checks.append(
            Check(
                name="Rollout finished without an agent-side failure",
                verdict="fail",
                message=str(run_error),
            )
        )

    partial = scored_passed / scored_total if scored_total else 0.0
    strict = 1.0 if scored_total and scored_passed == scored_total else 0.0

    if declared and len(declared) != len(outcomes):
        checks.append(
            Check(
                name="Every declared assertion was evaluated",
                verdict="fail",
                expected=len(declared),
                predicted=len(outcomes),
                message="The dataset row and the pinned benchmark task set disagree.",
            )
        )

    return CaseScore(
        objective=partial,
        field_scores={"partial_credit": partial, "task_completed_correctly": strict},
        checks=tuple(checks),
    )


integration = Integration(
    # The candidate may only edit the skill guides. The harness, the benchmark
    # pin, the frozen splits, and this evaluation policy stay fixed, so a higher
    # score can only come from better skills.
    targets=repository(("skills",)),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
