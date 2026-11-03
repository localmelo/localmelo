"""How a hosted service's thinking switch reaches its requests (#469).

DeepSeek's preset turns thinking off for ``deepseek-flash`` and leaves it on for
the other models. A model that thinks keeps the switch for one retry, as
``thinking_off_body``; one that already has thinking off has none to give.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from localmelo.melo.contracts.providers import LLM_REQUEST_EXTRA_BODY
from localmelo.support import config
from localmelo.support.backends.registry import get_backend
from localmelo.support.gateway.chat_gate import GatedLLM

OFF = {"thinking": {"type": "disabled"}}
DEEPSEEK = "https://api.deepseek.com/v1"


def _provider(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    *,
    thinking: str = "",
    extra_body: dict[str, Any] | None = None,
) -> Any:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-0123456789")
    cfg = config.Config(chat_backend="openai")
    cfg.openai.base_url = DEEPSEEK
    cfg.openai.api_key_env = "DEEPSEEK_API_KEY"
    cfg.openai.chat_model = model
    cfg.openai.thinking = thinking
    cfg.openai.extra_body = extra_body or {}
    return get_backend("openai").build_chat_provider(cfg)


def _sent(provider: Any) -> dict[str, Any]:
    body: dict[str, Any] = provider._payload([], None)
    return body


def test_the_flash_model_starts_with_thinking_off_and_has_no_retry_to_make(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch, "deepseek-flash")
    assert provider.thinking_off is True
    assert _sent(provider)["thinking"] == OFF["thinking"]
    assert dict(provider.thinking_off_body) == {}


def test_a_thinking_model_keeps_the_switch_for_one_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch, "deepseek-v4-pro")
    assert provider.thinking_off is False
    assert "thinking" not in _sent(provider)
    assert dict(provider.thinking_off_body) == OFF
    token = LLM_REQUEST_EXTRA_BODY.set(provider.thinking_off_body)
    try:
        assert _sent(provider)["thinking"] == OFF["thinking"]  # that one request
    finally:
        LLM_REQUEST_EXTRA_BODY.reset(token)
    assert "thinking" not in _sent(provider)


@pytest.mark.parametrize(
    ("model", "thinking", "off"),
    [
        ("deepseek-flash", "on", False),
        ("deepseek-flash", "off", True),
        ("deepseek-v4-pro", "off", True),
        ("deepseek-v4-pro", "on", False),
    ],
)
def test_the_per_model_setting_decides_over_the_default(
    monkeypatch: pytest.MonkeyPatch, model: str, thinking: str, off: bool
) -> None:
    provider = _provider(monkeypatch, model, thinking=thinking)
    assert provider.thinking_off is off
    assert (provider.thinking_off_body == {}) is off


def test_a_field_the_config_sets_itself_is_never_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    own = {"thinking": {"type": "enabled"}}
    provider = _provider(monkeypatch, "deepseek-v4-pro", extra_body=own)
    assert _sent(provider)["thinking"] == own["thinking"]
    assert dict(provider.thinking_off_body) == {}


def test_a_service_with_no_switch_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789")
    cfg = config.Config(chat_backend="openai")
    cfg.openai.api_key_env = "OPENAI_API_KEY"
    provider = get_backend("openai").build_chat_provider(cfg)
    assert provider.thinking_off is False
    assert dict(provider.thinking_off_body) == {}


def test_the_gate_passes_the_switch_on(monkeypatch: pytest.MonkeyPatch) -> None:
    inner = _provider(monkeypatch, "deepseek-v4-pro")
    gated = GatedLLM(inner, SimpleNamespace())  # type: ignore[arg-type]
    assert dict(gated.thinking_off_body) == OFF
