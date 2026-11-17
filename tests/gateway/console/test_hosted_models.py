"""The models a hosted service lists, several keys at once, and the one switch list.

No test calls a service: ``hosted_check.list_models`` and the local server's listing are
stubs, and every call they get is recorded so a test can say what was never asked.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient

from localmelo.support import config, hosted_check, keychain
from localmelo.support.backends import discovery
from localmelo.support.config import Config
from localmelo.support.gateway.console import Console
from localmelo.tests.fake_keychain import FakeSecurity

REAL_LIST_MODELS = hosted_check.list_models  # the real one, before a fixture stubs it
KEY = "sk-test-secret-value-9876543210abcdef"
DEEPSEEK = "https://api.deepseek.com/v1"
OPENAI = "https://api.openai.com/v1"
EVIL = "https://evil.example.com/v1"
GATEWAY_A = "https://llm.example.com/v1"
GATEWAY_B = "https://other.example.com/v1"
LOCAL_URL = "http://127.0.0.1:9/v1"


class Stub:
    """The service listings: what each address answers, and what was asked."""

    def __init__(self) -> None:
        self.answers: dict[str, tuple[list[str], hosted_check.HostedCheck | None]] = {}
        self.calls: list[tuple[str, str]] = []
        self.local: list[str] | Exception = ["qwen-small", "qwen-big"]
        self.local_calls: list[str] = []

    async def list_models(
        self, address: str, key: str, *, transport: Any = None
    ) -> tuple[list[str], hosted_check.HostedCheck | None]:
        self.calls.append((address, key))
        return self.answers.get(address, ([], None))

    async def discover(self, backend: str, url: str, **_: Any) -> list[str]:
        self.local_calls.append(url)
        if isinstance(self.local, Exception):
            raise self.local
        return list(self.local)


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path) -> Iterator[None]:
    with (
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path / "cfg")),
        mock.patch.object(config, "CONFIG_PATH", str(tmp_path / "cfg" / "config.toml")),
    ):
        yield


@pytest.fixture()
def stub() -> Iterator[Stub]:
    found = Stub()
    hosted_check.LISTINGS.clear()
    with (
        mock.patch.object(hosted_check, "list_models", found.list_models),
        mock.patch.object(discovery, "list_models", found.discover),
    ):
        yield found
    hosted_check.LISTINGS.clear()


def save_local(embedding: str = "none") -> None:
    cfg = Config(chat_backend="vllm", embedding_backend=embedding)
    cfg.vllm.chat_url, cfg.vllm.chat_model = LOCAL_URL, "qwen-small"
    cfg.ollama.embedding_url, cfg.ollama.embedding_model = LOCAL_URL, "embed-1"
    config.save(cfg)


def listing(client: TestClient, **query: str) -> dict[str, Any]:
    response = client.get("/api/hosted/models", params=query)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# -- M1: the list comes by itself once a key exists ---------------------------


def test_a_service_lists_its_models_once_a_key_exists(
    client: TestClient, stub: Stub, fake_keychain: FakeSecurity
) -> None:
    stub.answers[DEEPSEEK] = (["deepseek-flash", "deepseek-v4-pro"], None)

    before = listing(client, preset="deepseek")
    assert (before["key"], before["models"]) == (False, [])
    assert stub.calls == []  # no key: nothing was asked

    keychain.write("localmelo.deepseek", KEY)
    body = listing(client, preset="deepseek", model="deepseek-flash")
    assert body["key"] is True and body["listed"] is True
    assert body["models"] == ["deepseek-flash", "deepseek-v4-pro"]
    assert (body["gone"], body["message"]) == (False, "")
    assert stub.calls == [(DEEPSEEK, KEY)]
    assert KEY not in str(body)


def test_a_wrong_key_and_a_service_with_no_list_say_so(
    client: TestClient, stub: Stub, fake_keychain: FakeSecurity
) -> None:
    keychain.write("localmelo.deepseek", KEY)
    stub.answers[DEEPSEEK] = (
        [],
        hosted_check.HostedCheck(False, "wrong_key", status=401),
    )
    wrong = listing(client, preset="deepseek")
    assert wrong["result"] == "wrong_key" and wrong["models"] == []
    assert "did not accept the key (401)" in wrong["message"]

    hosted_check.LISTINGS.clear()
    stub.answers[DEEPSEEK] = ([], None)  # a service with no list is no failure
    none = listing(client, preset="deepseek")
    assert (none["key"], none["listed"], none["result"], none["message"]) == (
        True,
        False,
        "",
        "",
    )


def test_a_saved_model_the_service_no_longer_has_is_said_without_a_test(
    client: TestClient, stub: Stub, fake_keychain: FakeSecurity
) -> None:
    keychain.write("localmelo.deepseek", KEY)
    stub.answers[DEEPSEEK] = (["deepseek-v4-pro"], None)
    body = listing(client, preset="deepseek", model="deepseek-old")
    assert body["gone"] is True
    assert body["line"] == (
        "The service has no model called deepseek-old. Choose one from the list."
    )
    assert listing(client, preset="deepseek", model="deepseek-v4-pro")["gone"] is False


def test_a_key_in_the_presets_own_variable_counts(
    client: TestClient, stub: Stub
) -> None:
    stub.answers[OPENAI] = (["gpt-4o"], None)
    with mock.patch.dict("os.environ", {"OPENAI_API_KEY": KEY}):
        body = listing(client, preset="openai")
    assert body["models"] == ["gpt-4o"] and stub.calls == [(OPENAI, KEY)]


# -- K1: the route that sends a key, and where it refuses ---------------------


def test_a_built_in_services_key_is_not_sent_to_another_address(
    client: TestClient, stub: Stub, fake_keychain: FakeSecurity
) -> None:
    keychain.write("localmelo.openai", KEY)
    fake_keychain.calls.clear()

    refused = client.get(
        "/api/hosted/models", params={"preset": "openai", "address": EVIL}
    )

    assert (
        refused.status_code == 409 and refused.json()["error"]["code"] == "key_address"
    )
    assert stub.calls == [] and fake_keychain.calls == []
    # The address it does belong to is accepted when named.
    stub.answers[OPENAI] = (["gpt-4o"], None)
    ok = client.get(
        "/api/hosted/models", params={"preset": "openai", "address": OPENAI + "/"}
    )
    assert ok.status_code == 200 and stub.calls == [(OPENAI, KEY)]


def test_a_free_form_services_key_goes_only_to_the_address_it_was_saved_with(
    client: TestClient, stub: Stub, fake_keychain: FakeSecurity
) -> None:
    saved = client.put(
        "/api/settings/key", json={"preset": "other", "key": KEY, "address": GATEWAY_A}
    )
    assert saved.status_code == 200
    assert saved.json()["api_key"].startswith("keychain:localmelo.other-")

    stub.answers[GATEWAY_A] = (["m1"], None)
    elsewhere = listing(client, preset="other", address=GATEWAY_B)
    assert elsewhere["key"] is False and stub.calls == []
    assert listing(client, preset="other", address=GATEWAY_A)["models"] == ["m1"]
    assert stub.calls == [(GATEWAY_A, KEY)]
    assert (
        client.get("/api/hosted/models", params={"preset": "other"}).status_code == 400
    )


def test_the_test_button_refuses_an_address_the_key_does_not_belong_to(
    client: TestClient, stub: Stub, fake_keychain: FakeSecurity
) -> None:
    keychain.write("localmelo.openai", KEY)
    body = {
        "chat_backend": "openai",
        "embedding_backend": "none",
        "workspace": "~/ws",
        "sections": {
            "openai": {
                "api_key": "keychain:localmelo.openai",
                "base_url": EVIL,
                "chat_model": "gpt-4o",
            }
        },
    }
    fake_keychain.calls.clear()

    response = client.post("/api/settings/test", json=body)

    # The settings check refuses it before the Test runs, so no request is made.
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_settings"
    assert "never sends a key anywhere else" in error["message"]
    assert stub.calls == []
    assert not any("-w" in args for args, _ in fake_keychain.calls)


def test_the_key_routes_refuse_an_address_for_a_built_in_service(
    client: TestClient, fake_keychain: FakeSecurity
) -> None:
    refused = client.put(
        "/api/settings/key", json={"preset": "openai", "key": KEY, "address": EVIL}
    )
    assert refused.status_code == 409 and fake_keychain.items == {}
    no_address = client.put("/api/settings/key", json={"preset": "other", "key": KEY})
    assert no_address.status_code == 400


# -- M5: the switch list ------------------------------------------------------


def keep_two_services(fake_keychain: FakeSecurity) -> None:
    keychain.write("localmelo.deepseek", KEY)
    keychain.write("localmelo.openai", KEY + "2")


def test_with_no_hosted_key_there_is_no_switch_list_and_nothing_is_asked(
    client: TestClient, stub: Stub
) -> None:
    save_local()
    body = client.get("/api/chat-models").json()
    assert body["hosted"] is False and body["groups"] == []
    assert stub.calls == [] and stub.local_calls == []


def test_the_list_holds_the_local_server_and_each_service_that_has_a_key(
    client: TestClient, stub: Stub, fake_keychain: FakeSecurity
) -> None:
    save_local()
    keep_two_services(fake_keychain)
    stub.answers[DEEPSEEK] = (["deepseek-flash"], None)
    stub.answers[OPENAI] = (["gpt-4o", "gpt-4o-mini"], None)

    body = client.get("/api/chat-models").json()

    assert body["hosted"] is True
    assert [(g["id"], g["kind"]) for g in body["groups"]] == [
        ("local", "local"),
        ("openai", "hosted"),
        ("deepseek", "hosted"),
    ]
    groups = {g["id"]: g for g in body["groups"]}
    assert groups["local"]["models"] == ["qwen-small", "qwen-big"]
    assert groups["local"]["current"] == "qwen-small"  # the one in use is marked
    assert groups["openai"]["models"] == ["gpt-4o", "gpt-4o-mini"]
    assert groups["deepseek"]["current"] == ""
    assert "anthropic" not in groups  # no key, not listed


def test_the_quiet_list_names_the_groups_and_asks_no_server(
    client: TestClient, stub: Stub, fake_keychain: FakeSecurity
) -> None:
    save_local()
    keep_two_services(fake_keychain)

    body = client.get("/api/chat-models", params={"models": 0}).json()

    assert body["hosted"] is True
    assert [g["id"] for g in body["groups"]] == ["local", "openai", "deepseek"]
    assert all(g["models"] == [] and g["problem"] == "" for g in body["groups"])
    assert (
        stub.calls == [] and stub.local_calls == []
    )  # nothing was asked of any server


def test_a_failing_service_is_one_row_that_says_why_and_hides_nothing(
    client: TestClient, stub: Stub, fake_keychain: FakeSecurity
) -> None:
    save_local()
    keep_two_services(fake_keychain)
    stub.answers[DEEPSEEK] = (
        [],
        hosted_check.HostedCheck(False, "timeout", detail="ReadTimeout"),
    )
    stub.answers[OPENAI] = (["gpt-4o"], None)

    groups = {g["id"]: g for g in client.get("/api/chat-models").json()["groups"]}

    assert groups["deepseek"]["models"] == []
    assert "did not answer" in groups["deepseek"]["problem"]
    assert (
        groups["openai"]["models"] == ["gpt-4o"] and groups["openai"]["problem"] == ""
    )
    assert groups["local"]["models"]

    stub.local = discovery.DiscoveryError("connection refused", reachable=False)
    local = {g["id"]: g for g in client.get("/api/chat-models").json()["groups"]}[
        "local"
    ]
    assert local["models"] == [] and "connection refused" in local["problem"]


# -- M4, M5, M6: a switch -----------------------------------------------------


def switch(client: TestClient, group: str, model: str) -> Any:
    return client.post("/api/chat-models/switch", json={"group": group, "model": model})


def test_a_switch_changes_the_model_for_the_next_message_and_keeps_the_rest(
    client: TestClient,
    console: Console,
    stub: Stub,
    fake_keychain: FakeSecurity,
) -> None:
    save_local(embedding="ollama")
    keep_two_services(fake_keychain)
    stub.answers[DEEPSEEK] = (["deepseek-flash", "deepseek-v4-pro"], None)
    stub.answers[OPENAI] = (["gpt-4o", "gpt-4o-mini"], None)
    changed = mock.AsyncMock()
    console.sessions.settings_changed = changed  # type: ignore[method-assign]

    first = switch(client, "deepseek", "deepseek-v4-pro")
    assert first.status_code == 200, first.text
    assert first.json()["model"]["name"] == "deepseek-v4-pro"
    assert changed.await_count == 1  # idle chats are rebuilt, a running turn is left
    assert console.cfg.openai.chat_model == "deepseek-v4-pro"
    assert console.cfg.openai.base_url == DEEPSEEK

    assert switch(client, "openai", "gpt-4o-mini").status_code == 200
    assert switch(client, "local", "qwen-big").status_code == 200
    assert switch(client, "deepseek", "deepseek-v4-pro").status_code == 200
    back = switch(client, "openai", "gpt-4o-mini")
    assert back.status_code == 200

    saved = config.load()
    assert saved.chat_backend == "openai" and saved.openai.chat_model == "gpt-4o-mini"
    assert saved.vllm.chat_model == "qwen-big"
    assert config.load().services["deepseek"] == {"chat_model": "deepseek-v4-pro"}
    # M6: the embedding choice is the one it was.
    assert (saved.embedding_backend, saved.ollama.embedding_model) == (
        "ollama",
        "embed-1",
    )
    assert (saved.ollama.embedding_url) == LOCAL_URL
    assert {item[0] for item in fake_keychain.items} == {
        "localmelo.deepseek",
        "localmelo.openai",
    }


@pytest.mark.parametrize("status", [403, 500])
def test_a_services_error_text_that_repeats_the_key_is_not_in_the_answer(
    client: TestClient, fake_keychain: FakeSecurity, status: int
) -> None:
    odd = "zz9-" + "q7" * 12  # a shape no redaction pattern knows
    keychain.write("localmelo.openai", odd)
    real = REAL_LIST_MODELS

    async def asks(address: str, key: str, **_: Any) -> Any:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                status, json={"error": {"message": f"no access for {odd}"}}
            )
        )
        return await real(address, key, transport=transport)

    hosted_check.LISTINGS.clear()
    with mock.patch.object(hosted_check, "list_models", asks):
        body = client.get("/api/hosted/models", params={"preset": "openai"})

    assert body.status_code == 200 and body.json()["message"]
    if (
        status == 500
    ):  # the words for an error carry the service's text, so this is the case that matters
        assert "no access for" in body.json()["message"]
    assert odd not in body.text


@pytest.mark.parametrize(
    ("status", "answer", "words"),
    [
        (
            400,
            {
                "type": "error",
                "error": {"message": "anthropic-version: header is required"},
            },
            "The service answered with an error (400): anthropic-version: header is required",
        ),
        (
            502,
            "<html><body>502 Bad Gateway {x}</body></html>",
            "The service answered with an error (502).",
        ),
    ],
)
def test_a_services_error_is_one_short_line_and_ends_cleanly(
    client: TestClient,
    fake_keychain: FakeSecurity,
    status: int,
    answer: Any,
    words: str,
) -> None:
    keychain.write("localmelo.openai", KEY)
    real = REAL_LIST_MODELS

    async def asks(address: str, key: str, **_: Any) -> Any:
        body = {"json": answer} if isinstance(answer, dict) else {"text": answer}
        transport = httpx.MockTransport(lambda r: httpx.Response(status, **body))
        return await real(address, key, transport=transport)

    save_local()
    hosted_check.LISTINGS.clear()
    with mock.patch.object(hosted_check, "list_models", asks):
        shown = client.get("/api/hosted/models", params={"preset": "openai"}).json()[
            "message"
        ]
        row = {g["id"]: g for g in client.get("/api/chat-models").json()["groups"]}[
            "openai"
        ]

    assert shown == words and row["problem"] == words


def test_the_anthropic_service_lists_its_models_on_the_page_and_in_the_switch_list(
    client: TestClient, fake_keychain: FakeSecurity
) -> None:
    keychain.write("localmelo.anthropic", KEY)
    save_local()
    real = REAL_LIST_MODELS
    sent: list[httpx.Request] = []

    def answers(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if "x-api-key" not in request.headers:
            return httpx.Response(
                400,
                json={"error": {"message": "anthropic-version: header is required"}},
            )
        return httpx.Response(
            200, json={"data": [{"id": "claude-sonnet-5-5"}, {"id": "claude-opus-5-5"}]}
        )

    async def asks(address: str, key: str, **_: Any) -> Any:
        return await real(address, key, transport=httpx.MockTransport(answers))

    hosted_check.LISTINGS.clear()
    with mock.patch.object(hosted_check, "list_models", asks):
        gone = listing(client, preset="anthropic", model="claude-retired")
        groups = {g["id"]: g for g in client.get("/api/chat-models").json()["groups"]}

    assert (
        gone["models"] == ["claude-opus-5-5", "claude-sonnet-5-5"]
        and gone["gone"] is True
    )
    assert groups["anthropic"]["models"] == ["claude-opus-5-5", "claude-sonnet-5-5"]
    assert groups["anthropic"]["problem"] == ""
    # Every request went to the service's own address, in its own format.
    assert {str(r.url) for r in sent} == {
        "https://api.anthropic.com/v1/models?limit=1000"
    }


def test_the_switch_list_sends_a_key_only_to_the_address_it_belongs_to(
    client: TestClient, stub: Stub, fake_keychain: FakeSecurity
) -> None:
    save_local()
    keychain.write("localmelo.openai", KEY)
    saved = client.put(
        "/api/settings/key",
        json={"preset": "other", "key": KEY + "2", "address": GATEWAY_A},
    )
    assert saved.status_code == 200
    # A file that names another address for a built-in service, and for "Other" an
    # address its key was not saved for.
    with open(config.config_path(), "a", encoding="utf-8") as handle:
        handle.write(
            f'\n[services.openai]\nchat_model = "gpt-4o"\nbase_url = "{EVIL}"\n'
            f'\n[services.other]\nchat_model = "m1"\nbase_url = "{GATEWAY_B}"\n'
        )
    stub.answers[OPENAI] = (["gpt-4o"], None)

    body = client.get("/api/chat-models").json()
    refused = switch(client, "other", "m1")

    assert [group["id"] for group in body["groups"]] == ["local", "openai"]
    assert refused.status_code == 409
    assert stub.calls == [(OPENAI, KEY)]  # nothing went to EVIL or to GATEWAY_B


def test_a_switch_that_cannot_be_made_changes_nothing(
    client: TestClient, console: Console, stub: Stub, fake_keychain: FakeSecurity
) -> None:
    save_local()
    keychain.write("localmelo.deepseek", KEY)
    stub.answers[DEEPSEEK] = (["deepseek-flash"], None)
    before = Path(config.config_path()).read_text()

    gone = switch(client, "deepseek", "deepseek-old")
    assert gone.status_code == 400 and "no model called deepseek-old" in gone.text
    no_key = switch(client, "openai", "gpt-4o")
    assert no_key.status_code == 409 and "no key yet" in no_key.text
    assert switch(client, "nowhere", "x").status_code == 400
    assert switch(client, "deepseek", "").status_code == 400
    assert Path(config.config_path()).read_text() == before
    assert console.cfg.chat_backend != "openai"


def test_a_service_with_no_list_takes_a_typed_name(
    client: TestClient, stub: Stub, fake_keychain: FakeSecurity
) -> None:
    save_local()
    keychain.write("localmelo.deepseek", KEY)
    stub.answers[DEEPSEEK] = ([], None)
    assert switch(client, "deepseek", "any-name").status_code == 200
    assert config.load().openai.chat_model == "any-name"
