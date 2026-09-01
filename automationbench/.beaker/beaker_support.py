"""Beaker-owned helpers for the AutomationBench skills optimization spec.

Everything in this module is evaluation policy: it lives under ``.beaker/``,
outside the candidate surface Beaker may edit, and is imported only by
``beaker_spec.py``. It holds the three things the spec needs from the outside
world:

* where the *candidate's* ``skills/`` directory is (:func:`resolve_skills_dir`);
* how to point the benchmark's own model client at the Beaker gateway when a
  rollout model is selected (:func:`build_model_spec`);
* a type-preserving tracing adapter for that client (:func:`build_client`).

The adapter subclasses AutomationBench's own Chat Completions client rather
than wrapping it, because ``verifiers`` dispatches on the client's real type;
a proxy or ``__getattr__`` forwarder would be rejected before any case runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from automationbench.clients import RetryingOpenAIChatCompletionsClient
from beaker import inference_target
from verifiers.types import ClientConfig


if TYPE_CHECKING:
    from automationbench_skills.runner import ModelSpec


# This module sits at <project>/.beaker/, so the project root -- and with it the
# skills/ directory the optimizer edits -- is one level up. Resolving it from
# __file__ rather than the process CWD is what makes every rollout read the
# skills/ of the candidate repository copy it was launched for.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"

# The benchmark's client factory takes the *name* of the variable holding its
# API key, not the key itself, so gateway routing has to publish the run-scoped
# credential under some name. This one is set in-process for the lifetime of the
# rollout and is deliberately absent from `spec.required_env`: it is never a
# hosted setting, and BEAKER_* names are reserved there anyway.
GATEWAY_API_KEY_VAR = "AUTOMATIONBENCH_BEAKER_GATEWAY_KEY"

# The gateway speaks the OpenAI Chat Completions shape, which is also the only
# shape this tracing adapter can preserve the client type for.
CHAT_COMPLETIONS = "chat_completions"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def resolve_skills_dir() -> Path:
    """Return the candidate's ``skills/`` directory, or fail loudly.

    A missing directory would otherwise silently degrade every rollout to the
    baseline arm (``skills_dir=None``), which scores without the tools the
    optimization is about -- a wrong number rather than an error.
    """
    if not SKILLS_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected the candidate skills directory at {SKILLS_DIR}. "
            "The spec resolves it relative to .beaker/, so this means the "
            "repository layout moved or skills/ is absent from the checkout."
        )
    return SKILLS_DIR


def build_model_spec(runtime: Any) -> ModelSpec:
    """Map the rollout's selected model onto the benchmark's own routing.

    With a selected model, the request goes through the Beaker gateway: only the
    endpoint and credential change, and the benchmark's own ``_resolve_api``
    then lands on Chat Completions for any model name because the base URL is
    neither Anthropic's nor Google's. With no selected model -- ordinary local
    runs -- the application's existing default model and credentials are used
    unchanged.
    """
    from automationbench_skills.runner import DEFAULT_MODEL, ModelSpec

    if not getattr(runtime, "model", None):
        return ModelSpec(name=DEFAULT_MODEL)

    target = inference_target(runtime)
    os.environ[GATEWAY_API_KEY_VAR] = target.api_key
    return ModelSpec(
        name=target.model,
        base_url=target.base_url,
        api_key_var=GATEWAY_API_KEY_VAR,
        api=CHAT_COMPLETIONS,
    )


def _response_payload(response: Any) -> Any:
    """Summarize one native response for the trace without copying the world."""
    choices = getattr(response, "choices", None)
    if not choices:
        return {"raw": str(response)[:2000]}
    message = choices[0].message
    dump = getattr(message, "model_dump", None)
    payload = dump(mode="json") if callable(dump) else {"content": getattr(message, "content", None)}
    return {"message": payload, "finish_reason": getattr(choices[0], "finish_reason", None)}


class TracedChatCompletionsClient(RetryingOpenAIChatCompletionsClient):
    """The benchmark's Chat Completions client, with the model call traced.

    Subclassing keeps the exact type ``verifiers`` and AutomationBench expect
    (including the retry behaviour, which stays inside the span so a retried
    turn is still one logical model call). ``trace`` is a no-op outside a
    Beaker capture, so this client behaves identically when untraced.
    """

    def __init__(self, config: ClientConfig, *, trace: Any, provider: str) -> None:
        super().__init__(config)
        self._beaker_trace = trace
        self._beaker_provider = provider

    async def get_native_response(self, *args: Any, **kwargs: Any) -> Any:
        prompt = args[0] if args else kwargs.get("prompt")
        model = args[1] if len(args) > 1 else kwargs.get("model")
        with self._beaker_trace.model_call(
            operation="chat.completions.create",
            provider=self._beaker_provider,
            model=str(model),
            input_messages=prompt,
        ) as operation:
            response = await super().get_native_response(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                operation.usage(
                    input_tokens=getattr(usage, "prompt_tokens", None),
                    output_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                )
            operation.output(_response_payload(response))
            return response


def build_client(model: ModelSpec, *, trace: Any, provider: str | None = None) -> Any:
    """Build the client for ``model``, traced when its API shape allows it.

    Only the Chat Completions path -- every gateway-routed model, and the
    application's own OpenAI default -- can be traced while preserving the
    client type. Anything else (first-party Anthropic or Gemini, selected
    locally) keeps the application's untraced client rather than risking a
    client the harness rejects.
    """
    from automationbench_skills.vendored.model_setup import build_client as build_plain_client
    from automationbench_skills.vendored.model_setup import resolve_api_key_var

    resolved = model.resolved_api()
    key_var = resolve_api_key_var(resolved, model.api_key_var)
    if resolved != CHAT_COMPLETIONS:
        return build_plain_client(resolved, key_var, model.base_url)

    if not os.environ.get(key_var):
        raise ValueError(f"No API key found. Set the {key_var} environment variable.")
    config = ClientConfig(
        api_key_var=key_var,
        api_base_url=model.base_url or DEFAULT_OPENAI_BASE_URL,
        extra_headers={},
    )
    return TracedChatCompletionsClient(config, trace=trace, provider=provider or "openai")
