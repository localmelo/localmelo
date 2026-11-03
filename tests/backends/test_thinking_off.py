"""The request fields that turn a local model's thinking off, by backend (#339, #428).

One rule for every short JSON request (a step review, Hippo's encoder). No server:
the fields are read off the function's result.
"""

from __future__ import annotations

import pytest

from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.providers.llm.thinking import thinking_off_fields

OWN = {"chat_template_kwargs": {"keep": 1, "enable_thinking": True}}


def openai_llm() -> OpenAICompatLLM:
    return OpenAICompatLLM("http://127.0.0.1:9/v1", "m", extra_body=OWN)


@pytest.mark.parametrize("backend", ["vllm", "sglang", None])
def test_template_kwargs_are_set_over_the_providers_own(backend: str | None) -> None:
    assert thinking_off_fields(openai_llm(), backend) == {
        "chat_template_kwargs": {"keep": 1, "enable_thinking": False}
    }
    assert thinking_off_fields(OpenAICompatLLM("http://x/v1", "m"), backend) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_the_body_given_replaces_the_providers_extra_body_as_the_kwargs_source() -> (
    None
):
    own = {"chat_template_kwargs": {"other": 2}}
    assert thinking_off_fields(openai_llm(), None, own) == {
        "chat_template_kwargs": {"other": 2, "enable_thinking": False}
    }
    # A kwargs value that is not a table is left out.
    assert thinking_off_fields(openai_llm(), None, {"chat_template_kwargs": 3}) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_ollama_native_is_asked_with_think_off() -> None:
    ollama = OllamaNativeChat("http://127.0.0.1:9", "m")
    assert thinking_off_fields(ollama, "ollama") == {"think": False}


@pytest.mark.parametrize("backend", ["mlc", "openai", "gemini", "anthropic", "nvidia"])
def test_mlc_and_cloud_backends_get_nothing(backend: str) -> None:
    assert thinking_off_fields(openai_llm(), backend) == {}
