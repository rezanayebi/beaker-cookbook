"""Beaker repository optimization spec for the AutomationBench skills harness.

The candidate surface is ``skills/`` -- the folder of ``SKILL.md`` guides the
agent reads through its ``list_skills`` / ``read_skill`` tools. This is the lever
the recipe was built for: the tools re-read the directory on every call, so
editing a guide changes agent behaviour with no environment rebuild. Everything
that decides *what a good rollout is* stays here under ``.beaker/``: the row
contract, the rollout call, and the scorer. The score itself comes from the
benchmark's own deterministic assertion rubric, which this spec never
re-implements and which lives in the pinned ``automation-bench`` dependency
rather than in editable source.

Each dataset row names one AutomationBench task by its globally-unique
``task_name``, exactly as the repository's frozen ``splits/*.txt`` files do. The
task's prompt, simulated world, and assertions are rebuilt from that pinned
dependency at rollout time, so a 15-80 KB world state never travels in the
dataset while the labels stay byte-identical to the benchmark's own. Row parsing
checks each row against the pin, so a dependency bump that changes a task's
assertions fails validation instead of silently re-scoring against new labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import beaker_support
from beaker import (
    Case,
    CaseDataLoader,
    CaseResult,
    CaseScore,
    DatasetRowContext,
    DatasetSchema,
    OptimizationContext,
    Spec,
    objective_score,
    spec,
)


# The two numbers every AutomationBench rollout produces. `partial_credit` is
# the fraction of the task's scored assertions that passed and is the metric
# this agent hill-climbs; `task_completed_correctly` is the benchmark's strict
# 0/1 headline metric, carried at zero weight so runs still report it.
OBJECTIVE_FIELD = "partial_credit"
STRICT_FIELD = "task_completed_correctly"
FIELD_WEIGHTS: Mapping[str, float] = {OBJECTIVE_FIELD: 1.0, STRICT_FIELD: 0.0}

# A stuck provider request would otherwise hang a rollout indefinitely; the
# harness scores an expired task 0 with an ``error`` note, which is how the
# recipe's own CLI treats it too.
CASE_TIMEOUT_SECONDS = 900.0

DATASET_SCHEMA = DatasetSchema(
    json_schema={
        "type": "object",
        "required": ["id", "input", "expected"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "input": {
                "type": "object",
                "required": ["task_name"],
                "properties": {
                    "task_name": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "AutomationBench task id, '<domain>.<name>'. Resolved against the "
                            "pinned automation-bench dependency to rebuild the task's prompt, "
                            "simulated world, and assertions."
                        ),
                    }
                },
                "additionalProperties": False,
            },
            "expected": {
                "type": "object",
                "required": ["partial_credit", "num_assertions"],
                "properties": {
                    "partial_credit": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Target score: 1.0 means every scored assertion passes.",
                    },
                    "num_assertions": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Assertion count recorded from the pin, to detect label drift.",
                    },
                    "assertions": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "The task's assertion list, as labelled by the benchmark.",
                    },
                },
                "additionalProperties": True,
            },
            "metadata": {"type": "object"},
            "group_key": {
                "type": "string",
                "description": "Business domain (sales, marketing, operations, support, finance, hr).",
            },
        },
        "additionalProperties": False,
    }
)


# ── Dataset loading ──────────────────────────────────────────────────────────


@cache
def _samples_by_name() -> Mapping[str, Any]:
    """Index every public task from the pinned dependency by ``task_name``."""
    from automationbench_skills.data import load_samples

    return {sample.task_name: sample for sample in load_samples()}


@dataclass(frozen=True)
class _TaskRow:
    """One validated dataset row: a task id plus its recorded label."""

    id: str
    task_name: str
    target_partial_credit: float
    num_assertions: int
    metadata: dict[str, Any] = field(default_factory=dict)
    group_key: str = "default"


class _AutomationBenchDataLoader(CaseDataLoader[_TaskRow]):
    """Validate rows against both the JSONL contract and the pinned task set.

    Uploaded datasets need ``train`` and ``val`` splits; ``test`` is optional and
    is evaluated only for the winning candidate. The repository's own leakage
    policy (``src/automationbench_skills/splits/README.md``) is what decides
    which frozen split each of those may draw from: optimization and model
    selection happen on ``splits/train.txt``, and ``splits/test.txt`` is for
    final evaluation only.
    """

    dataset_schema = DATASET_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> _TaskRow:
        del context
        missing = [name for name in ("id", "input", "expected") if name not in raw]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")

        row_id = str(raw["id"]).strip()
        if not row_id:
            raise ValueError("id must be non-empty")

        input_payload = raw["input"]
        expected = raw["expected"]
        metadata = raw.get("metadata") or {}
        for name, value in (("input", input_payload), ("expected", expected), ("metadata", metadata)):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a JSON object")

        task_name = str(input_payload.get("task_name") or "").strip()
        if not task_name:
            raise ValueError("input.task_name must be a non-empty task id")

        sample = _samples_by_name().get(task_name)
        if sample is None:
            raise ValueError(
                f"task_name {task_name!r} is not in the pinned automation-bench task set. "
                "The dependency pin and the dataset must move together."
            )

        num_assertions = expected.get("num_assertions")
        if not isinstance(num_assertions, int) or num_assertions < 1:
            raise ValueError("expected.num_assertions must be a positive integer")
        actual = len(sample.info.get("assertions") or [])
        if num_assertions != actual:
            raise ValueError(
                f"label drift for {task_name!r}: the dataset records {num_assertions} assertion(s) "
                f"but the pinned dependency now defines {actual}. Re-derive the dataset."
            )

        target = expected.get("partial_credit")
        if not isinstance(target, int | float) or not 0.0 <= float(target) <= 1.0:
            raise ValueError("expected.partial_credit must be a number in [0, 1]")

        return _TaskRow(
            id=row_id,
            task_name=task_name,
            target_partial_credit=float(target),
            num_assertions=num_assertions,
            metadata=dict(metadata),
            group_key=str(raw.get("group_key") or sample.domain),
        )

    def iter_cases(self, row: _TaskRow, context: DatasetRowContext) -> Iterable[Case]:
        del context
        yield Case(
            input={"task_name": row.task_name},
            case_id=row.id,
            ground_truth={
                OBJECTIVE_FIELD: row.target_partial_credit,
                STRICT_FIELD: 1.0,
                "num_assertions": row.num_assertions,
            },
            group_key=row.group_key,
            metadata=row.metadata,
        )


# ── Rollout ──────────────────────────────────────────────────────────────────


async def _run_case(*, case: Case, targets: None, runtime: Any) -> CaseResult:
    """Run ONE AutomationBench task against the candidate's ``skills/`` folder.

    ``run_one_async`` is the recipe's own single-sample entrypoint: it opens a
    fresh simulated world, runs the agent loop, and scores the end state with the
    benchmark's deterministic rubric. Only ``skills_dir`` and the model client
    vary here, and ``skills_dir`` resolves inside this candidate's checkout, so
    each candidate is judged on its own guides.
    """
    del targets
    from automationbench_skills.runner import run_one_async

    task_name = str(case.input["task_name"])
    sample = _samples_by_name().get(task_name)
    if sample is None:
        return CaseResult.failed(
            f"task_name {task_name!r} is absent from the pinned automation-bench task set.",
            retryable=False,
        )

    try:
        skills_dir = beaker_support.resolve_skills_dir()
        model = beaker_support.build_model_spec(runtime)
        client = beaker_support.build_client(
            model,
            trace=runtime.trace,
            provider=getattr(runtime, "provider", None),
        )
    except Exception as exc:  # noqa: BLE001 - setup failure, not a bad answer
        return CaseResult.failed(f"could not prepare the rollout: {exc}", retryable=False)

    with runtime.trace.stage(
        "automationbench.rollout",
        inputs={"task_name": task_name, "skills_dir": str(skills_dir)},
        attributes={"domain": sample.domain, "model": model.name},
    ) as stage:
        try:
            result = await run_one_async(
                sample,
                model=model,
                skills_dir=skills_dir,
                timeout=CASE_TIMEOUT_SECONDS,
                client=client,
            )
        except Exception as exc:  # noqa: BLE001 - the rollout could not run
            return CaseResult.failed(f"rollout raised {type(exc).__name__}: {exc}", retryable=True)
        stage.output(
            {
                OBJECTIVE_FIELD: result.partial_credit,
                STRICT_FIELD: result.task_completed_correctly,
                "error": str(result.error) if result.error is not None else None,
            }
        )

    # A timed-out or aborted rollout is a real 0, not an execution failure: it
    # ran, and the benchmark scores it. Only an exception above prevents scoring.
    return CaseResult(
        output={
            OBJECTIVE_FIELD: float(result.partial_credit),
            STRICT_FIELD: float(result.task_completed_correctly),
        },
        context={
            "task_name": result.task_name,
            "domain": result.domain,
            "num_assertions_expected": case.ground_truth.get("num_assertions"),
            "num_assertions_scored": len(result.assertion_results),
            "assertion_results": result.assertion_results,
            "error": str(result.error) if result.error is not None else None,
            "trajectory": result.trajectory,
        },
    )


# ── Scoring ──────────────────────────────────────────────────────────────────


class _AssertionRubricScorer:
    """Report both benchmark metrics; hill-climb ``partial_credit``.

    The numbers are produced inside the rollout by AutomationBench's own
    deterministic rubric, which asserts against the simulated world's final
    state. This scorer only selects which of them the optimizer maximizes, so
    the scoring rule stays immutable evaluation policy under ``.beaker/``.
    """

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        del case
        output = result.output if isinstance(result.output, Mapping) else {}
        field_scores = {
            OBJECTIVE_FIELD: _bounded(output.get(OBJECTIVE_FIELD)),
            STRICT_FIELD: _bounded(output.get(STRICT_FIELD)),
        }
        return CaseScore(
            field_scores=field_scores,
            objective=objective_score(field_scores, field_weights=FIELD_WEIGHTS),
            key="assertion_rubric",
        )


def _bounded(value: Any) -> float:
    """Coerce a reported metric into [0, 1]; anything unusable scores 0."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


# ── Spec wiring ──────────────────────────────────────────────────────────────


@spec(
    dataset_schema=DATASET_SCHEMA,
    # The lever this recipe exists for. Narrowing the surface to the skill
    # guides also keeps the runner, the split files, and the scoring path out of
    # reach, so a candidate can only raise its score by writing better guides.
    repository=("skills",),
)
def build_spec(ctx: OptimizationContext) -> Spec:
    """Assemble the repository optimization spec Beaker runs."""
    del ctx
    return Spec(
        data_loader=_AutomationBenchDataLoader(),
        run_case=_run_case,
        scorer=_AssertionRubricScorer(),
    )
