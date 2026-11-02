"""Telling a local model not to think, by the backend it runs on (#339, #428).

A short JSON job (a step review, Hippo's encoder) needs no thinking, and a local
model whose chat template thinks by default would spend the whole reply budget on
it. Which request field turns thinking off depends on the server, so the rule is
here once, for every caller that builds such a request.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

# The backends whose OpenAI-compatible servers take chat_template_kwargs
# (None: a --base-url server, a local one).
TEMPLATE_KWARGS_BACKENDS: Final = frozenset({"vllm", "sglang", None})
# The local backends: every other one is a cloud backend, whose models answer
# fast and some of which refuse these fields.
LOCAL_BACKENDS: Final = frozenset({"vllm", "sglang", "mlc", "ollama", None})


def thinking_off_fields(
    llm: BaseLLMProvider,
    backend: str | None,
    own: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The request fields that tell *llm*'s model on *backend* not to think.

    Ollama's native API: ``think`` off. An OpenAI-compatible server that takes
    it (vllm, sglang, a ``--base-url`` server, ``backend`` ``None``):
    ``chat_template_kwargs.enable_thinking`` off, over the ``chat_template_kwargs``
    of *own*, the provider's own request body (``llm.extra_body`` when not given),
    whose other keys stay. Nothing for mlc and for a cloud backend.
    """
    if backend not in LOCAL_BACKENDS:
        return {}
    if isinstance(llm, OllamaNativeChat):
        return {"think": False}
    if backend in TEMPLATE_KWARGS_BACKENDS:
        body = own if own is not None else getattr(llm, "extra_body", None) or {}
        kwargs = body.get("chat_template_kwargs")
        kwargs = dict(kwargs) if isinstance(kwargs, Mapping) else {}
        return {"chat_template_kwargs": {**kwargs, "enable_thinking": False}}
    return {}
