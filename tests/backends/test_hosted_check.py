"""The Test button for a hosted service: model list, then one short chat (``support/hosted_check.py``)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from localmelo.support import hosted_check, keychain
from localmelo.support.config import CloudVendorConfig, Config
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.tests.fake_keychain import FakeSecurity

KEY = "sk-test-0123456789abcdef"
BASE = "https://api.example.com/v1"

Handler = Callable[[httpx.Request], httpx.Response]


def config(model: str = "deepseek-flash") -> Config:
    keychain.write("localmelo.deepseek", KEY)
    return Config(
        chat_backend="openai",
        embedding_backend="none",
        openai=CloudVendorConfig(
            api_key="keychain:localmelo.deepseek",
            chat_model=model,
            base_url="https://api.deepseek.com/v1",
        ),
    )


def reply(*, tool: bool = True, content: str = "") -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": json.dumps({"text": "hello"}),
                },
            }
        ]
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


def service(
    models: list[str] | int | None = None,
    chat: int | dict[str, Any] | Exception = 200,
    seen: list[httpx.Request] | None = None,
) -> Handler:
    """A service whose `/models` answers *models* (a list, or a status) and whose chat answers *chat*."""

    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if request.url.path.endswith("/models"):
            if isinstance(models, int):
                return httpx.Response(models, json={"error": {"message": "no"}})
            rows = [{"id": name} for name in (models or [])]
            return httpx.Response(200, json={"data": rows})
        if isinstance(chat, Exception):
            raise chat
        if isinstance(chat, int) and chat != 200:
            return httpx.Response(chat, json={"error": {"message": "refused"}})
        return httpx.Response(200, json=reply() if chat == 200 else chat)

    return handle


async def run(cfg: Config, handler: Handler) -> hosted_check.HostedCheck:
    transport = httpx.MockTransport(handler)
    provider = OpenAICompatLLM(
        "https://api.deepseek.com/v1",
        cfg.openai.chat_model,
        api_key=KEY,
        transport=transport,
    )
    try:
        return await hosted_check.check_hosted(
            cfg, transport=transport, provider=provider
        )
    finally:
        await provider.close()


async def test_a_working_service_lists_its_models_and_answers_with_a_tool_call(
    fake_keychain: FakeSecurity,
) -> None:
    seen: list[httpx.Request] = []
    found = await run(
        config(),
        service(["deepseek-v4-pro", "deepseek-flash", "text-embedding-3"], seen=seen),
    )

    assert (found.ok, found.result, found.tools) == (True, "works", True)
    assert found.models == ("deepseek-flash", "deepseek-v4-pro")  # chat models, sorted
    assert [request.url.path for request in seen] == [
        "/v1/models",
        "/v1/chat/completions",
    ]
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"


async def test_a_model_that_does_not_call_the_tool_is_said_so(
    fake_keychain: FakeSecurity,
) -> None:
    found = await run(
        config(), service(["deepseek-flash"], chat=reply(tool=False, content="hi"))
    )
    assert (found.ok, found.tools) == (True, False)


@pytest.mark.parametrize(
    ("models", "chat", "result", "status"),
    [
        (401, 200, "wrong_key", 401),
        (402, 200, "refused", 402),
        (403, 200, "refused", 403),
        (429, 200, "busy", 429),
        (500, 200, "error", 500),
        # No list is no failure: the chat decides.
        (404, 401, "wrong_key", 401),
        (404, 402, "refused", 402),
        (404, 403, "refused", 403),
        (404, 404, "no_model", 404),
        (405, 429, "busy", 429),
    ],
)
async def test_each_refusal_becomes_a_plain_result(
    fake_keychain: FakeSecurity, models: int, chat: int, result: str, status: int
) -> None:
    found = await run(config(), service(models, chat=chat))
    assert (found.ok, found.result, found.status) == (False, result, status)


async def test_a_model_the_service_does_not_list_is_not_asked(
    fake_keychain: FakeSecurity,
) -> None:
    seen: list[httpx.Request] = []
    found = await run(config("deepseek-big"), service(["deepseek-flash"], seen=seen))
    assert (found.ok, found.result) == (False, "no_model")
    assert found.models == ("deepseek-flash",)
    assert [request.url.path for request in seen] == ["/v1/models"]  # no chat was spent


async def test_a_service_nobody_answers_is_unreachable(
    fake_keychain: FakeSecurity,
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    found = await run(config(), refuse)
    assert (found.ok, found.result) == (False, "unreachable")


async def test_a_slow_list_and_a_slow_chat_are_timeouts_not_unreachable(
    fake_keychain: FakeSecurity,
) -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    assert (await run(config(), slow)).result == "timeout"
    found = await run(
        config(), service(["deepseek-flash"], chat=httpx.ReadTimeout("t"))
    )
    assert found.result == "timeout"


async def test_a_missing_or_locked_key_is_said_before_any_request(
    fake_keychain: FakeSecurity,
) -> None:
    cfg = config()
    fake_keychain.items.clear()
    seen: list[httpx.Request] = []
    found = await run(cfg, service(["deepseek-flash"], seen=seen))
    assert found.result == "keychain_missing"
    fake_keychain.locked = True
    assert (
        await run(cfg, service(["deepseek-flash"], seen=seen))
    ).result == "keychain_locked"
    cfg.openai.api_key = ""
    assert (await run(cfg, service(["deepseek-flash"], seen=seen))).result == "no_key"
    assert seen == []


async def test_the_result_never_carries_the_key(fake_keychain: FakeSecurity) -> None:
    def echo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad key {request.headers['authorization']}")

    found = await run(config(), echo)
    assert KEY not in repr(found)
