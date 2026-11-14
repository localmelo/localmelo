"""The Test button for a hosted service: model list, then one short chat (``support/hosted_check.py``)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from localmelo.support import hosted, hosted_check, keychain
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


async def test_a_models_prefix_is_dropped_from_the_listed_names(
    fake_keychain: FakeSecurity,
) -> None:
    found = await run(
        config(), service(["models/deepseek-flash", "models/text-embedding-3"])
    )
    assert (found.ok, found.models) == (True, ("deepseek-flash",))


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


# -- The listing as the pages ask for it ------------------------------------


async def test_a_listing_is_one_get_that_never_follows_a_redirect_and_cleans_names() -> (
    None
):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = {
            "data": [{"id": "models/ok-model"}, {"id": "bad\nname"}, {"id": "x" * 300}]
        }
        return httpx.Response(200, json=body)

    models, failed = await hosted_check.list_models(
        BASE, KEY, transport=httpx.MockTransport(handler)
    )
    assert (models, failed) == (["ok-model"], None)
    assert [(r.method, r.url.path) for r in seen] == [("GET", "/v1/models")]

    moved = httpx.MockTransport(
        lambda request: httpx.Response(302, headers={"location": "https://evil.test/"})
    )
    models, failed = await hosted_check.list_models(BASE, KEY, transport=moved)
    assert models == [] and failed is not None and failed.status == 302


async def test_a_listing_answer_past_the_limit_or_compressed_is_a_failure_not_a_list() -> (
    None
):
    big = {"data": [{"id": f"model-{i}"} for i in range(hosted_check.MAX_BODY // 8)]}
    assert len(json.dumps(big)) > hosted_check.MAX_BODY

    models, failed = await hosted_check.list_models(
        BASE,
        KEY,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=big)),
    )

    assert models == [] and failed is not None and failed.result == "error"
    assert failed.status == 200

    # An answer that is compressed although it was asked not to be is refused before it is read.
    class Unread(httpx.AsyncByteStream):
        entered = False

        async def __aiter__(self) -> Any:
            Unread.entered = True
            yield b"x"

    models, failed = await hosted_check.list_models(
        BASE,
        KEY,
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, headers={"content-encoding": "gzip"}, stream=Unread()
            )
        ),
    )
    assert models == [] and failed is not None and failed.result == "error"
    assert failed.detail == "Encoded" and not Unread.entered


async def test_the_anthropic_list_is_asked_in_its_own_format_and_others_are_unchanged(
    fake_keychain: FakeSecurity,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "claude-sonnet-5-5"}]})

    transport = httpx.MockTransport(handler)
    anthropic, openai = hosted.get("anthropic"), hosted.get("openai")
    assert anthropic is not None and openai is not None
    keychain.write(anthropic.service, KEY)

    # The key comes through the preset's reader and goes in its own header, alone.
    found = await hosted_check.service_models(
        anthropic,
        anthropic.base_url,
        transport=transport,
        listings=hosted_check.Listings(),
    )
    assert found.models == ("claude-sonnet-5-5",) and found.check is None
    sent = seen[-1]
    assert str(sent.url) == "https://api.anthropic.com/v1/models?limit=1000"
    assert sent.headers["anthropic-version"] == "2026-10-21"
    assert sent.headers["x-api-key"] == KEY and "authorization" not in sent.headers

    # Another service gets the request it always got.
    await hosted_check.list_models(openai.base_url, KEY, transport=transport)
    other = seen[-1]
    async with httpx.AsyncClient(transport=transport) as client:
        await client.get(
            f"{openai.base_url}/models",
            headers={"Accept-Encoding": "identity", "Authorization": f"Bearer {KEY}"},
        )
    before = seen[-1]
    assert str(other.url) == str(before.url) == "https://api.openai.com/v1/models"
    assert list(other.headers.items()) == list(before.headers.items())


async def test_a_json_error_body_gives_its_message_as_one_short_line() -> None:
    said = "anthropic-version: header is required"
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": said},
        "request_id": "req_1",
    }
    models, failed = await hosted_check.list_models(
        BASE,
        KEY,
        transport=httpx.MockTransport(lambda r: httpx.Response(400, json=body)),
    )
    assert models == [] and failed is not None and failed.result == "error"
    assert failed.detail == said and "{" not in failed.detail

    # The sentence is one line, cut at the limit, with the key and known secrets out of it.
    def detail(payload: Any) -> str:
        return hosted_check.failure(500, json.dumps(payload), KEY)[1]

    assert detail({"message": "top\nlevel   words"}) == "top level words"
    assert detail({"error": {"message": f"bad {KEY} here"}}) == "bad [redacted] here"
    long = detail({"error": {"message": "word " * 100}})
    assert len(long) == hosted_check.MAX_DETAIL and long.endswith("…")
    for odd in ({"error": "just a string"}, {"errors": ["x"]}, ["x"], "x", 5):
        assert detail(odd) == ""


@pytest.mark.parametrize(
    "body",
    [
        "<html><body><h1>502 Bad Gateway</h1> {not json} nginx</body></html>",
        "upstream connect error or disconnect/reset before headers",
        '<html>{"error": {"message": "inside a page"}}</html>',
        "",
    ],
)
async def test_an_error_body_that_is_not_json_gives_no_text(body: str) -> None:
    models, failed = await hosted_check.list_models(
        BASE,
        KEY,
        transport=httpx.MockTransport(lambda r: httpx.Response(502, text=body)),
    )
    assert models == [] and failed is not None and failed.status == 502
    assert (failed.result, failed.detail) == ("error", "")


@pytest.mark.parametrize("status", [403, 500])
async def test_an_error_body_that_repeats_the_key_never_reaches_a_result(
    status: int,
) -> None:
    odd = "zz9-" + "q7" * 12  # a shape no redaction pattern knows
    body = json.dumps({"error": {"message": f"denied for key {odd}, again {odd}"}})
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, text=body)

    models, failed = await hosted_check.list_models(
        BASE, odd, transport=httpx.MockTransport(handler)
    )

    assert models == [] and failed is not None and failed.status == status
    assert odd not in repr(failed) and "denied for key" in failed.detail
    assert (
        seen[0].headers["authorization"] == f"Bearer {odd}"
    )  # it was sent, only there

    # The Test button's chat: the error the provider raises is cut the same way.
    def chat_fails(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "deepseek-flash"}]})
        return httpx.Response(status, text=body)

    cfg = config()
    keychain.write("localmelo.deepseek", odd)
    keychain._cache.clear()
    checked = await run(cfg, chat_fails)
    assert checked.status == status and odd not in repr(checked)


async def test_listings_are_reused_for_a_while_and_shared_while_in_flight() -> None:
    calls: list[tuple[str, str]] = []
    gate = asyncio.Event()
    now = [0.0]

    async def slow(address: str, key: str, **_: Any) -> tuple[list[str], None]:
        calls.append((address, key))
        await gate.wait()
        return ["m"], None

    listings = hosted_check.Listings(ttl_s=60, clock=lambda: now[0])
    original = hosted_check.list_models
    hosted_check.list_models = slow  # type: ignore[assignment]
    try:
        both = asyncio.gather(listings.get(BASE, KEY), listings.get(BASE, KEY))
        await asyncio.sleep(0)
        gate.set()
        assert await both == [(["m"], None), (["m"], None)]
        assert len(calls) == 1  # one in flight for the service
        await listings.get(BASE, KEY)
        assert len(calls) == 1  # reused
        now[0] = 61.0
        await listings.get(BASE, KEY)
        assert len(calls) == 2  # past the minute
        await listings.get(BASE, KEY + "x")
        assert len(calls) == 3  # another key asks again
    finally:
        hosted_check.list_models = original  # type: ignore[assignment]


async def test_a_transient_failure_is_not_kept() -> None:
    answers: list[tuple[list[str], hosted_check.HostedCheck | None]] = [
        ([], hosted_check.HostedCheck(False, hosted_check.TIMEOUT)),
        ([], hosted_check.HostedCheck(False, hosted_check.WRONG_KEY, status=401)),
    ]

    async def next_answer(address: str, key: str, **_: Any) -> Any:
        return answers.pop(0) if len(answers) > 1 else answers[0]

    listings = hosted_check.Listings()
    original = hosted_check.list_models
    hosted_check.list_models = next_answer  # type: ignore[assignment]
    try:
        assert (await listings.get(BASE, KEY))[1].result == hosted_check.TIMEOUT  # type: ignore[union-attr]
        assert (await listings.get(BASE, KEY))[1].result == hosted_check.WRONG_KEY  # type: ignore[union-attr]
        assert (await listings.get(BASE, KEY))[1].result == hosted_check.WRONG_KEY  # type: ignore[union-attr]
        assert len(answers) == 1  # the definite answer was kept, the timeout was not
    finally:
        hosted_check.list_models = original  # type: ignore[assignment]
