"""A stored key goes only to its own service, and each service keeps its last model.

The key rule (``support/hosted.py``): one place reads a key for a request
(:func:`hosted.api_key` for a section, :func:`hosted.preset_key` for a preset), and it
refuses an address the key does not belong to before the keychain is read.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from localmelo.support import config, hosted, keychain
from localmelo.support.config import CloudVendorConfig, Config
from localmelo.tests.fake_keychain import FakeSecurity

KEY = "sk-test-0123456789abcdef"
EVIL = "https://evil.example.com/v1"
GATEWAY_A = "https://llm.example.com/v1"
GATEWAY_B = "https://other.example.com/v1"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path) -> Any:
    with (
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path / "cfg")),
        mock.patch.object(config, "CONFIG_PATH", str(tmp_path / "cfg" / "config.toml")),
        mock.patch.dict(os.environ, {}, clear=False),
    ):
        yield


def reads(fake: FakeSecurity) -> int:
    """How many times a secret was read from the keychain (``-w``)."""
    return sum(
        "-w" in args and args[0] == "find-generic-password" for args, _ in fake.calls
    )


def preset(name: str) -> hosted.Preset:
    found = hosted.get(name)
    assert found is not None
    return found


# -- K1: a key is sent only to the address it belongs to ----------------------


@pytest.mark.parametrize(
    ("backend", "service", "address"),
    [
        ("openai", "localmelo.openai", EVIL),
        ("anthropic", "localmelo.anthropic", "https://proxy.example.com/v1"),
        ("gemini", "localmelo.gemini", EVIL),
        ("nvidia", "localmelo.nvidia", EVIL),
        # The DeepSeek item on OpenAI's address, and the reverse.
        ("openai", "localmelo.deepseek", ""),
        ("openai", "localmelo.openai", "https://api.deepseek.com/v1"),
    ],
)
def test_a_section_whose_key_belongs_to_another_address_is_refused_before_a_read(
    fake_keychain: FakeSecurity, backend: str, service: str, address: str
) -> None:
    keychain.write(service, KEY)
    fake_keychain.calls.clear()
    keychain._cache.clear()
    section = CloudVendorConfig(
        api_key=keychain.ref_for(service), base_url=address, chat_model="m"
    )

    with pytest.raises(keychain.KeychainError) as caught:
        hosted.api_key(backend, section)

    assert caught.value.kind == "address"
    assert (
        reads(fake_keychain) == 0
    )  # refused before the keychain was asked for the key
    assert hosted.key_state(backend, section) == "address"
    assert (
        "never sends a key anywhere else" in hosted.key_errors(backend, section, "X")[0]
    )


def test_a_chat_provider_is_not_built_with_a_key_for_another_address(
    fake_keychain: FakeSecurity,
) -> None:
    from localmelo.support.backends.registry import get_backend

    keychain.write("localmelo.openai", KEY)
    fake_keychain.calls.clear()
    keychain._cache.clear()
    cfg = Config(chat_backend="openai", embedding_backend="none")
    cfg.openai = CloudVendorConfig(
        api_key="keychain:localmelo.openai", base_url=EVIL, chat_model="m"
    )

    with pytest.raises(keychain.KeychainError) as caught:
        get_backend("openai").build_chat_provider(cfg)

    assert caught.value.kind == "address" and reads(fake_keychain) == 0


async def test_a_listing_for_another_address_is_refused_before_any_request(
    fake_keychain: FakeSecurity,
) -> None:
    from localmelo.support import hosted_check

    keychain.write("localmelo.openai", KEY)
    fake_keychain.calls.clear()
    keychain._cache.clear()
    asked: list[str] = []

    async def list_models(address: str, key: str, **_: Any) -> tuple[list[str], None]:
        asked.append(address)
        return ["m"], None

    with mock.patch.object(hosted_check, "list_models", list_models):
        found = await hosted_check.service_models(preset("openai"), EVIL)

    assert found.check is not None and found.check.result == "keychain_address"
    assert asked == [] and reads(fake_keychain) == 0


PROXY = "https://proxy.example/v1"


@pytest.mark.parametrize("in_use", [True, False])
@pytest.mark.parametrize("name", ["nvidia", "anthropic"])
def test_a_switch_does_not_carry_a_proxys_variable_to_the_built_in_address(
    fake_keychain: FakeSecurity, name: str, in_use: bool
) -> None:
    vendor = preset(name)
    cfg = Config(chat_backend=name if in_use else "vllm", embedding_backend="none")
    setattr(
        cfg,
        vendor.backend,
        CloudVendorConfig(base_url=PROXY, api_key_env="MY_PROXY_TOKEN", chat_model="m"),
    )
    with mock.patch.dict(os.environ, {"MY_PROXY_TOKEN": "the-proxys-token"}):
        before = hosted.api_key(vendor.backend, getattr(cfg, vendor.backend))
        assert (before.address, before.key) == (PROXY, "the-proxys-token")

        hosted.use_service(cfg, vendor, "m")

        after = hosted.api_key(vendor.backend, getattr(cfg, vendor.backend))
    assert after.address == vendor.base_url and after.key == ""


def test_a_presets_own_variable_goes_only_to_its_address() -> None:
    section = CloudVendorConfig(api_key_env="OPENAI_API_KEY", base_url=EVIL)
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "from-the-environment"}):
        with pytest.raises(keychain.KeychainError) as caught:
            hosted.api_key("openai", section)
        assert caught.value.kind == "address"
        assert hosted.key_state("openai", section) == "address"
        section.base_url = ""
        assert hosted.api_key("openai", section).key == "from-the-environment"


def test_a_key_saved_for_the_right_address_is_read(
    fake_keychain: FakeSecurity,
) -> None:
    keychain.write("localmelo.deepseek", KEY)
    section = CloudVendorConfig(
        api_key="keychain:localmelo.deepseek", base_url="https://api.deepseek.com/v1/"
    )
    assert hosted.api_key("openai", section).key == KEY
    assert hosted.key_state("openai", section) == "keychain"


def test_a_free_form_key_is_kept_for_the_address_it_was_saved_with(
    fake_keychain: FakeSecurity,
) -> None:
    other = preset("other")
    keychain.write(hosted.service_for(other, GATEWAY_A), KEY)
    keychain._cache.clear()
    fake_keychain.calls.clear()

    assert hosted.preset_key_state(other, GATEWAY_A + "/") == "keychain"
    assert hosted.preset_key(other, GATEWAY_A).key == KEY
    # Another address finds no key: the item is named by the address.
    assert hosted.service_for(other, GATEWAY_A) != hosted.service_for(other, GATEWAY_B)
    assert hosted.preset_key_state(other, GATEWAY_B) == "missing"
    assert hosted.preset_key(other, GATEWAY_B).key == ""
    # And a section that names the item of one address, on another address, is refused.
    ref = keychain.ref_for(hosted.service_for(other, GATEWAY_A))
    section = CloudVendorConfig(api_key=ref, base_url=GATEWAY_B)
    with pytest.raises(keychain.KeychainError) as caught:
        hosted.api_key("openai", section)
    assert caught.value.kind == "address"
    # The item the earlier versions wrote is no address's: it is not used.
    keychain.write("localmelo.other", KEY)
    old = CloudVendorConfig(api_key="keychain:localmelo.other", base_url=GATEWAY_A)
    assert hosted.key_state("openai", old) == "address"


def test_a_preset_key_is_never_read_for_another_address(
    fake_keychain: FakeSecurity,
) -> None:
    openai = preset("openai")
    keychain.write(openai.service, KEY)
    fake_keychain.calls.clear()
    with pytest.raises(keychain.KeychainError) as caught:
        hosted.preset_key(openai, EVIL)
    assert caught.value.kind == "address"
    assert hosted.preset_key_state(openai, EVIL) == "address"
    assert fake_keychain.calls == []  # no call to the keychain at all
    assert hosted.preset_key(openai, openai.base_url).key == KEY


# -- M4: several services at once, each with its last model --------------------


def on_service(name: str, model: str, address: str = "") -> Config:
    cfg = Config(chat_backend="openai", embedding_backend="none")
    found = preset(name)
    hosted.use_service(cfg, found, model, address=address)
    return cfg


def test_choosing_another_service_keeps_the_models_of_the_others(
    fake_keychain: FakeSecurity,
) -> None:
    keychain.write("localmelo.deepseek", KEY)
    keychain.write("localmelo.openai", KEY + "2")
    cfg = on_service("deepseek", "deepseek-v4-pro")
    hosted.use_service(cfg, preset("openai"), "gpt-4o-mini")
    hosted.use_service(cfg, preset("other"), "m1", address=GATEWAY_A)
    hosted.use_service(cfg, preset("deepseek"), "deepseek-v4-pro")
    hosted.use_service(cfg, preset("openai"), "gpt-4o-mini")

    config.save(cfg)
    kept = config.load()

    assert kept.chat_backend == "openai"
    assert hosted.last_models(kept) == {
        "deepseek": {"chat_model": "deepseek-v4-pro"},
        "openai": {"chat_model": "gpt-4o-mini"},
        "other": {"chat_model": "m1", "base_url": GATEWAY_A},
    }
    # Both keys are still in the keychain, each in its own item.
    assert {item[0] for item in fake_keychain.items} == {
        "localmelo.deepseek",
        "localmelo.openai",
    }


def test_the_section_takes_the_key_of_the_service_chosen_and_no_other(
    fake_keychain: FakeSecurity,
) -> None:
    keychain.write("localmelo.deepseek", KEY)
    keychain.write("localmelo.openai", KEY + "2")
    cfg = on_service("deepseek", "deepseek-flash")
    assert (cfg.openai.base_url, cfg.openai.api_key) == (
        "https://api.deepseek.com/v1",
        "keychain:localmelo.deepseek",
    )
    hosted.use_service(cfg, preset("openai"), "gpt-4o")
    assert (cfg.openai.base_url, cfg.openai.api_key) == (
        "",
        "keychain:localmelo.openai",
    )
    assert hosted.api_key("openai", cfg.openai).key == KEY + "2"
    hosted.use_service(cfg, preset("deepseek"), "deepseek-flash")
    assert hosted.api_key("openai", cfg.openai).key == KEY


def test_a_service_with_a_section_of_its_own_keeps_its_section(
    fake_keychain: FakeSecurity,
) -> None:
    cfg = Config(chat_backend="anthropic", embedding_backend="none")
    cfg.anthropic = CloudVendorConfig(
        api_key_env="MY_ANTHROPIC_KEY", chat_model="claude-sonnet-4-20250514"
    )
    keychain.write("localmelo.deepseek", KEY)
    with mock.patch.dict(os.environ, {"MY_ANTHROPIC_KEY": "v"}):
        hosted.use_service(cfg, preset("deepseek"), "deepseek-flash")
        assert cfg.chat_backend == "openai"
        assert cfg.anthropic.chat_model == "claude-sonnet-4-20250514"
        hosted.use_service(cfg, preset("anthropic"), "claude-x")
    assert cfg.anthropic.api_key_env == "MY_ANTHROPIC_KEY"
    assert cfg.anthropic.chat_model == "claude-x"
    assert hosted.last_models(cfg)["deepseek"] == {"chat_model": "deepseek-flash"}


def test_the_last_local_server_is_kept_while_a_hosted_one_is_in_use() -> None:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.vllm.chat_url = "http://127.0.0.1:9/v1"
    cfg.vllm.chat_model = "qwen"
    hosted.use_service(cfg, preset("openai"), "gpt-4o")
    assert hosted.last_models(cfg)["local"] == {"backend": "vllm"}
    hosted.use_local(cfg, "vllm", "qwen")
    assert cfg.chat_backend == "vllm"
    assert hosted.last_models(cfg)["openai"] == {"chat_model": "gpt-4o"}


def test_the_stored_state_is_additive_and_holds_no_key(
    fake_keychain: FakeSecurity, tmp_path: Path
) -> None:
    keychain.write("localmelo.deepseek", KEY)
    cfg = on_service("deepseek", "deepseek-flash")
    config.save(cfg)
    text = Path(config.config_path()).read_text()

    assert "[services.deepseek]" in text and 'chat_model = "deepseek-flash"' in text
    assert KEY not in text
    # A file with no such table, and one with junk in it, load.
    Path(config.config_path()).write_text(
        'chat_backend = "openai"\nembedding_backend = "none"\n\n[services.bogus]\nchat_model = "x"\n'
        '\n[services.openai]\nchat_model = 7\nbase_url = "x"\n'
    )
    assert config.load().services == {}
    Path(config.config_path()).write_text('chat_backend = "openai"\n')
    assert config.load().services == {}


def test_a_setup_that_never_used_a_hosted_service_writes_no_services_table() -> None:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.vllm.chat_url, cfg.vllm.chat_model = "http://127.0.0.1:9/v1", "qwen"
    config.save(cfg)

    assert "services" not in Path(config.config_path()).read_text()
    assert config.load().services == {}
