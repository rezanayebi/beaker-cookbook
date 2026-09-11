"""Candidate-workflow tracing for one AutomationBench rollout.

Two halves, both scoped to the candidate workflow inside
``Integration.run_case`` — the rollout itself and nothing else:

* **Tool spans** come from Beaker's built-in ``verifiers`` integration,
  attached to the shared ``AutomationBenchEnv`` instance. Left unpinned, it
  resolves ``current_trace()`` per call, so the one cached environment records
  under whichever case is running and records nothing outside a capture.
* **Model spans** need hand wiring: that integration covers tool calls only.
  The benchmark harness owns the turn loop and takes a ``verifiers`` ``Client``
  *object* as a parameter, so no framework adapter ever sees the calls. We
  therefore subclass the exact client class the harness built and override
  ``get_response`` — the single method every turn goes through — wrapping only
  the call to the original implementation. The result is still an instance of
  the harness's own client class, so ``verifiers``' client resolution and any
  ``isinstance`` check keep passing; a proxy or ``__getattr__`` forwarder would
  fail them.

Nothing here reaches the scorer: ``score_case`` runs in the controller from the
JSON result and makes no model calls of its own.
"""

from __future__ import annotations

from typing import Any

from beaker.tracing import current_trace
from beaker.tracing.integrations import verifiers as beaker_verifiers
from verifiers.legacy.clients.client import Client
from verifiers.legacy.types import Messages, Response, SamplingArgs, Tool


# The wire protocol each resolved API speaks, for the span's provider label.
_PROVIDERS = {
    "anthropic": "anthropic",
    "gemini_interactions": "google",
    "responses": "openai",
    "chat_completions": "openai",
}

_TRACED_CLASSES: dict[type[Client], type[Client]] = {}


def _traced_class(cls: type[Client]) -> type[Client]:
    """Build (once) a subclass of ``cls`` that records one span per turn.

    A single base, so instances keep ``cls``'s exact memory layout and an
    existing client can be promoted into it.
    """
    traced = _TRACED_CLASSES.get(cls)
    if traced is not None:
        return traced

    async def get_response(
        self: Client,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Response:
        with current_trace().model_call(
            operation="chat",
            provider=getattr(self, "beaker_provider", "openai"),
            model=model,
            # One span per turn carrying that turn's full message list, so the
            # agent's tool calls and the simulator's tool results are captured.
            input_messages=prompt,
        ) as call:
            response = await cls.get_response(self, prompt, model, sampling_args, tools, **kwargs)
            call.output(response.model_dump(mode="json"))
            usage = response.usage
            if usage is not None:
                # verifiers normalizes every provider's usage block into its own
                # Usage model, so read that instead of sniffing native shapes.
                call.usage(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )
            return response

    traced = type(f"Traced{cls.__name__}", (cls,), {"get_response": get_response})
    _TRACED_CLASSES[cls] = traced
    return traced


def traced_client(plain: Client, *, resolved_api: str) -> Client:
    """Promote a freshly built client to its tracing subclass, in place.

    ``plain`` must be a client this integration just built for one case and owns
    outright. Promotion re-points it at a strict subclass of its own class, so
    it stays the same object with the same connection pool and satisfies every
    ``isinstance`` check it satisfied before, now with ``get_response``
    instrumented. Promoting rather than reconstructing keeps this correct for
    whichever client class the benchmark's routing picks — including one with
    its own constructor — and builds no second connection pool per case.
    """
    plain.__class__ = _traced_class(type(plain))
    plain.beaker_provider = _PROVIDERS.get(resolved_api, "openai")  # type: ignore[attr-defined]
    return plain


def instrument_env(env: Any) -> Any:
    """Record ``env``'s tool executions as ``tool_call`` spans."""
    return beaker_verifiers.instrument(env)
