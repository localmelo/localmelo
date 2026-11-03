"""Hosted services in Settings: presets, the key in the keychain, and the Test button."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from localmelo.support import config, hosted_check, keychain
from localmelo.support.config import CloudVendorConfig, Config
from localmelo.support.gateway.console import Console, routes_settings
from localmelo.tests.fake_keychain import FakeSecurity

KEY = "sk-test-secret-value-9876543210abcdef"
REF = "keychain:localmelo.deepseek"
DEEPSEEK = "https://api.deepseek.com/v1"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path) -> Iterator[Path]:
    cfg_dir = tmp_path / "cfg"
    cfg_path = cfg_dir / "config.toml"
    with (
        mock.patch.object(config, "CONFIG_DIR", str(cfg_dir)),
        mock.patch.object(config, "CONFIG_PATH", str(cfg_path)),
    ):
        yield cfg_path


def deepseek_cfg(**fields: str) -> Config:
    return Config(
        chat_backend="openai",
        embedding_backend="none",
        openai=CloudVendorConfig(
            api_key=REF, chat_model="deepseek-flash", base_url=DEEPSEEK, **fields
        ),
    )


def put_body(revision: str, **section: str) -> dict[str, object]:
    return {
        "chat_backend": "openai",
        "embedding_backend": "none",
        "workspace": "~/ws",
        "revision": revision,
        "sections": {"openai": section},
    }


# -- Presets ------------------------------------------------------------------


def test_the_settings_offer_the_presets_in_order(client: TestClient) -> None:
    hosted_list = client.get("/api/settings").json()["choices"]["hosted"]

    assert [row["id"] for row in hosted_list] == [
        "openai",
        "anthropic",
        "gemini",
        "nvidia",
        "deepseek",
        "other",
    ]
    deepseek = hosted_list[4]
    assert deepseek["backend"] == "openai"
    assert deepseek["base_url"] == DEEPSEEK
    assert deepseek["models"] == ["deepseek-flash", "deepseek-v4-pro"]
    assert deepseek["key_ref"] == REF
    assert deepseek["thinking_switch"] is True
    assert [row["id"] for row in hosted_list if row["thinking_switch"]] == ["deepseek"]
    other = hosted_list[5]
    assert (other["backend"], other["free_address"], other["models"]) == (
        "openai",
        True,
        [],
    )


def test_the_free_form_preset_is_named_in_the_pages_language(
    client: TestClient,
) -> None:
    rows = client.get(
        "/api/settings", headers={"X-LocalMelo-Language": "zh-Hans"}
    ).json()["choices"]["hosted"]
    assert rows[-1]["label"] == "其他（OpenAI 格式）"
    assert rows[0]["label"] == "OpenAI"


def test_a_section_says_which_preset_it_is(client: TestClient) -> None:
    config.save(deepseek_cfg())
    sections = client.get("/api/settings").json()["sections"]
    assert sections["openai"]["preset"] == "deepseek"
    assert sections["openai"]["api_key"] == REF
    assert sections["anthropic"]["preset"] == "anthropic"

    cfg = deepseek_cfg()
    cfg.openai.base_url = "https://llm.example.com/v1"
    config.save(cfg)
    assert client.get("/api/settings").json()["sections"]["openai"]["preset"] == "other"
    cfg.openai.base_url = ""
    config.save(cfg)
    assert (
        client.get("/api/settings").json()["sections"]["openai"]["preset"] == "openai"
    )


def test_a_preset_is_saved_as_its_address_a_key_reference_and_a_model(
    client: TestClient, isolated_config: Path, fake_keychain: FakeSecurity
) -> None:
    client.put("/api/settings/key", json={"preset": "deepseek", "key": KEY})
    revision = client.get("/api/settings").json()["revision"]
    resp = client.put(
        "/api/settings",
        json=put_body(
            revision,
            api_key=REF,
            base_url=DEEPSEEK,
            chat_model="deepseek-v4-pro",
            thinking="on",
        ),
    )
    assert resp.status_code == 200, resp.text
    saved = config.load()
    assert (saved.openai.api_key, saved.openai.base_url) == (REF, DEEPSEEK)
    assert (saved.openai.chat_model, saved.openai.thinking) == ("deepseek-v4-pro", "on")
    assert "thinking" in isolated_config.read_text()


@pytest.mark.parametrize(
    ("field", "value"),
    [("api_key", "keychain:github"), ("api_key", KEY), ("thinking", "maybe")],
)
def test_a_bad_reference_or_thinking_value_is_refused(
    client: TestClient, field: str, value: str
) -> None:
    resp = client.put("/api/settings", json=put_body("", **{field: value}))
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"
    assert KEY not in resp.text


def test_the_banner_names_the_service_not_the_backend() -> None:
    from localmelo.support.backends import get_backend

    assert get_backend("openai").label(deepseek_cfg()) == "DeepSeek"
    assert get_backend("openai").label(Config()) == "OpenAI"
    custom = deepseek_cfg()
    custom.openai.base_url = "https://llm.example.com/v1"
    assert get_backend("openai").label(custom) == "llm.example.com"


# -- The key ------------------------------------------------------------------


def test_the_pasted_key_goes_to_the_keychain_and_nowhere_else(
    client: TestClient,
    console: Console,
    isolated_config: Path,
    fake_keychain: FakeSecurity,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        saved = client.put("/api/settings/key", json={"preset": "deepseek", "key": KEY})
        config.save(deepseek_cfg())
        console.cfg = deepseek_cfg()
        replies = [
            saved,
            client.get("/api/settings"),
            client.get("/api/health"),
            client.get("/api/settings/key-status?preset=deepseek"),
        ]

    assert saved.status_code == 200
    assert saved.json() == {"preset": "deepseek", "api_key": REF, "keychain": "saved"}
    assert fake_keychain.items == {("localmelo.deepseek", "localmelo"): KEY}
    assert KEY not in fake_keychain.argv_text()
    for reply in replies:
        assert KEY not in reply.text
    assert KEY not in isolated_config.read_text()
    assert KEY not in caplog.text
    assert replies[1].json()["sections"]["openai"]["key"] == "keychain"
    assert replies[3].json()["keychain"] == "saved"
    assert replies[2].json()["config_errors"] == []


def test_a_key_the_page_sends_wrong_is_refused_without_repeating_it(
    client: TestClient, fake_keychain: FakeSecurity
) -> None:
    pasted = "two words and a long enough tail"
    resp = client.put("/api/settings/key", json={"preset": "deepseek", "key": pasted})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_key"
    assert pasted not in resp.text
    assert fake_keychain.items == {}
    extra = client.put(
        "/api/settings/key", json={"preset": "deepseek", "key": KEY, "more": "x"}
    )
    assert extra.status_code == 400
    assert KEY not in extra.text
    unknown = client.put("/api/settings/key", json={"preset": "nope", "key": KEY})
    assert unknown.status_code == 400
    assert KEY not in unknown.text


def test_a_locked_keychain_gets_a_plain_fix_in_the_pages_language(
    client: TestClient, fake_keychain: FakeSecurity
) -> None:
    fake_keychain.locked = True
    resp = client.put("/api/settings/key", json={"preset": "deepseek", "key": KEY})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "keychain_locked"
    assert "Unlock it" in resp.json()["error"]["message"]
    zh = client.put(
        "/api/settings/key",
        json={"preset": "deepseek", "key": KEY},
        headers={"X-LocalMelo-Language": "zh-Hans"},
    )
    assert "解锁" in zh.json()["error"]["message"]
    assert KEY not in resp.text + zh.text


def test_remove_deletes_the_saved_key(
    client: TestClient, fake_keychain: FakeSecurity
) -> None:
    client.put("/api/settings/key", json={"preset": "deepseek", "key": KEY})
    assert fake_keychain.items
    resp = client.delete("/api/settings/key?preset=deepseek")
    assert resp.status_code == 200
    assert resp.json() == {"preset": "deepseek", "keychain": "missing"}
    assert fake_keychain.items == {}
    assert client.get("/api/settings/key-status?preset=deepseek").json()[
        "keychain"
    ] == ("missing")
    assert client.delete("/api/settings/key?preset=nope").status_code == 400
    assert client.delete("/api/settings/key").status_code == 400


def test_a_missing_item_shows_in_the_settings_and_the_health_check(
    client: TestClient, console: Console, fake_keychain: FakeSecurity
) -> None:
    config.save(deepseek_cfg())
    console.cfg = deepseek_cfg()
    assert client.get("/api/settings").json()["sections"]["openai"]["key"] == "missing"
    errors = client.get("/api/health").json()["config_errors"]
    assert len(errors) == 1 and "not in the keychain" in errors[0]


def test_an_environment_variable_named_in_the_config_wins(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_keychain: FakeSecurity
) -> None:
    client.put("/api/settings/key", json={"preset": "deepseek", "key": KEY})
    cfg = deepseek_cfg()
    cfg.openai.api_key_env = "DEEPSEEK_API_KEY"
    config.save(cfg)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-the-environment")
    section = client.get("/api/settings").json()["sections"]["openai"]
    assert (section["key"], section["api_key_set"]) == ("env", True)
    assert "from-the-environment" not in client.get("/api/settings").text


# -- The Test button ----------------------------------------------------------

TEST_BODY = {
    "chat_backend": "openai",
    "embedding_backend": "none",
    "workspace": "~/ws",
    "sections": {
        "openai": {"api_key": REF, "base_url": DEEPSEEK, "chat_model": "deepseek-flash"}
    },
}


def press_test(
    client: TestClient, found: hosted_check.HostedCheck
) -> dict[str, object]:
    keychain.write("localmelo.deepseek", KEY)

    async def fake(cfg: Config) -> hosted_check.HostedCheck:
        assert cfg.openai.base_url == DEEPSEEK
        return found

    with mock.patch.object(routes_settings, "check_hosted", fake):
        resp = client.post("/api/settings/test", json=TEST_BODY)
    assert resp.status_code == 200
    return resp.json()  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("found", "words"),
    [
        (
            hosted_check.HostedCheck(False, "wrong_key", status=401),
            "did not accept the key (401)",
        ),
        (
            hosted_check.HostedCheck(False, "refused", status=402),
            "refused this account (402)",
        ),
        (
            hosted_check.HostedCheck(False, "refused", status=403),
            "refused this account (403)",
        ),
        (
            hosted_check.HostedCheck(False, "no_model", status=404),
            "no model called deepseek-flash",
        ),
        (hosted_check.HostedCheck(False, "unreachable"), "Cannot reach the service"),
        (hosted_check.HostedCheck(False, "timeout"), "did not answer within 60 s"),
        (hosted_check.HostedCheck(False, "busy", status=429), "busy"),
        (hosted_check.HostedCheck(False, "no_key"), "no key yet"),
        (hosted_check.HostedCheck(False, "keychain_locked"), "Unlock it"),
        (hosted_check.HostedCheck(False, "keychain_missing"), "not in the keychain"),
        (
            hosted_check.HostedCheck(False, "error", status=500, detail="boom"),
            "error (500): boom",
        ),
    ],
)
def test_each_result_of_the_test_is_said_plainly(
    client: TestClient,
    fake_keychain: FakeSecurity,
    found: hosted_check.HostedCheck,
    words: str,
) -> None:
    body = press_test(client, found)
    assert body["ok"] is False and body["failed"] == "chat"
    assert body["result"] == found.result
    assert words in str(body["error"]) and body["error"] == body["message"]


def test_a_working_service_returns_its_models(
    client: TestClient, fake_keychain: FakeSecurity
) -> None:
    body = press_test(
        client,
        hosted_check.HostedCheck(
            True, "works", models=("deepseek-flash", "deepseek-v4-pro"), tools=True
        ),
    )
    assert body["ok"] is True and body["error"] == ""
    assert body["models"] == ["deepseek-flash", "deepseek-v4-pro"]
    assert "deepseek-flash answered and used a tool" in str(body["message"])
    quiet = press_test(client, hosted_check.HostedCheck(True, "works", tools=False))
    assert quiet["ok"] is True and "did not use the test tool" in str(quiet["message"])


def test_the_test_reads_the_key_from_the_keychain_not_the_page(
    client: TestClient, fake_keychain: FakeSecurity
) -> None:
    keychain.write("localmelo.deepseek", KEY)
    seen: list[str] = []

    async def fake(cfg: Config) -> hosted_check.HostedCheck:
        from localmelo.support import hosted

        seen.append(hosted.api_key(cfg.openai))
        return hosted_check.HostedCheck(True, "works", tools=True)

    with mock.patch.object(routes_settings, "check_hosted", fake):
        resp = client.post("/api/settings/test", json=TEST_BODY)
    assert seen == [KEY]
    assert KEY not in resp.text
