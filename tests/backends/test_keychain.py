"""The keychain route of a hosted service's API key (``support/keychain.py``)."""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

import pytest

from localmelo.support import hosted, keychain
from localmelo.support.config import CloudVendorConfig
from localmelo.tests.fake_keychain import FakeSecurity

KEY = "sk-test-0123456789abcdef"
SERVICE = "localmelo.deepseek"


def test_a_key_is_written_through_stdin_and_never_a_command_line(
    fake_keychain: FakeSecurity,
) -> None:
    keychain.write(SERVICE, KEY)

    assert fake_keychain.items == {(SERVICE, "localmelo"): KEY}
    assert KEY not in fake_keychain.argv_text()
    args, stdin = fake_keychain.calls[0]
    assert args == ["-i"]
    assert stdin is not None and KEY in stdin and stdin.endswith("\n")


def test_a_key_with_a_quote_or_a_backslash_is_stored_as_it_was_pasted(
    fake_keychain: FakeSecurity,
) -> None:
    odd = 'ab"cd\\ef-0123456789'
    keychain.write(SERVICE, odd)
    assert keychain.read(SERVICE) == odd


def test_writing_again_replaces_the_key(fake_keychain: FakeSecurity) -> None:
    keychain.write(SERVICE, KEY)
    keychain.write(SERVICE, "sk-second-0123456789")
    assert keychain.read(SERVICE) == "sk-second-0123456789"
    assert len(fake_keychain.items) == 1


@pytest.mark.parametrize(
    "text",
    [
        "",
        "short",
        "two words in a key",
        "line\nbreak-0123456789",
        "k" * 600,
        "kéy-0123456789",
    ],
)
def test_text_that_is_not_a_key_is_refused_before_anything_is_written(
    fake_keychain: FakeSecurity, text: str
) -> None:
    with pytest.raises(keychain.KeychainError) as caught:
        keychain.write(SERVICE, text)
    assert caught.value.kind == "bad"
    assert fake_keychain.calls == []
    assert text not in str(caught.value) or text == ""


def test_a_pasted_bearer_prefix_and_spaces_are_trimmed(
    fake_keychain: FakeSecurity,
) -> None:
    keychain.write(SERVICE, f"  Bearer {KEY}\n")
    assert keychain.read(SERVICE) == KEY


def test_remove_deletes_the_item_and_is_quiet_when_it_is_gone(
    fake_keychain: FakeSecurity,
) -> None:
    keychain.write(SERVICE, KEY)
    keychain.remove(SERVICE)
    assert fake_keychain.items == {}
    keychain.remove(SERVICE)  # already gone: no error
    with pytest.raises(keychain.KeychainError) as caught:
        keychain.read(SERVICE)
    assert caught.value.kind == "missing"


def test_a_missing_item_has_a_plain_fix(fake_keychain: FakeSecurity) -> None:
    with pytest.raises(keychain.KeychainError) as caught:
        keychain.read(SERVICE)
    assert "not in the keychain" in str(caught.value)
    assert "Paste it again" in str(caught.value)


def test_a_locked_keychain_has_a_plain_fix(fake_keychain: FakeSecurity) -> None:
    fake_keychain.locked = True
    with pytest.raises(keychain.KeychainError) as caught:
        keychain.read(SERVICE)
    assert caught.value.kind == "locked"
    assert "Unlock it" in str(caught.value)
    assert keychain.status(SERVICE) == "locked"


def test_a_computer_without_a_keychain_says_to_use_a_variable() -> None:
    with mock.patch.object(keychain, "available", lambda: False):
        with pytest.raises(keychain.KeychainError) as caught:
            keychain.write(SERVICE, KEY)
        assert caught.value.kind == "unavailable"
        assert "environment variable" in str(caught.value)
        assert keychain.status(SERVICE) == "unavailable"


def test_the_status_reads_no_secret_and_is_reused_for_a_moment(
    fake_keychain: FakeSecurity,
) -> None:
    assert keychain.status(SERVICE) == "missing"
    keychain.write(SERVICE, KEY)  # a write forgets the answer
    assert keychain.status(SERVICE) == "saved"
    calls = len(fake_keychain.calls)
    assert keychain.status(SERVICE) == "saved"
    assert len(fake_keychain.calls) == calls
    assert all("-w" not in args for args, _ in fake_keychain.calls[1:2])


@pytest.mark.parametrize(
    ("text", "service"),
    [("", None), ("keychain:localmelo.deepseek", "localmelo.deepseek")],
)
def test_a_reference_names_one_of_melos_own_items(
    text: str, service: str | None
) -> None:
    assert keychain.parse_ref(text) == service


@pytest.mark.parametrize(
    "text",
    ["keychain:github", "keychain:localmelo.", "deepseek", "keychain:localmelo.A B"],
)
def test_any_other_reference_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        keychain.parse_ref(text)


# -- Which key a section uses ------------------------------------------------


def section(**fields: Any) -> CloudVendorConfig:
    return CloudVendorConfig(chat_model="m", **fields)


def test_the_environment_variable_wins_when_it_is_set(
    fake_keychain: FakeSecurity,
) -> None:
    keychain.write(SERVICE, KEY)
    chosen = section(api_key_env="SOME_HOSTED_KEY", api_key=f"keychain:{SERVICE}")
    with mock.patch.dict(os.environ, {"SOME_HOSTED_KEY": "from-the-environment"}):
        assert hosted.api_key(chosen) == "from-the-environment"
        assert hosted.key_state(chosen) == "env"
    with mock.patch.dict(os.environ, {}, clear=True):
        assert hosted.api_key(chosen) == KEY
        assert hosted.key_state(chosen) == "keychain"


def test_a_section_with_no_key_named_has_none() -> None:
    assert hosted.api_key(section()) == ""
    assert hosted.key_state(section()) == "none"


def test_an_old_section_with_only_a_variable_keeps_working(
    fake_keychain: FakeSecurity,
) -> None:
    old = section(api_key_env="OLD_STYLE_KEY")
    with mock.patch.dict(os.environ, {"OLD_STYLE_KEY": "old-key-value"}):
        assert hosted.key_errors("openai", old, "OPENAI_API_KEY") == []
        assert hosted.api_key(old) == "old-key-value"
    with mock.patch.dict(os.environ, {}, clear=True):
        errors = hosted.key_errors("openai", old, "OPENAI_API_KEY")
    assert errors == [
        "[openai] environment variable $OLD_STYLE_KEY is not set. "
        "Export it before starting the gateway, or paste the key in Settings."
    ]


def test_validation_names_the_fix_for_each_way_a_saved_key_can_fail(
    fake_keychain: FakeSecurity,
) -> None:
    saved = section(api_key=f"keychain:{SERVICE}")
    assert "not in the keychain" in hosted.key_errors("openai", saved, "X")[0]
    keychain.write(SERVICE, KEY)
    assert hosted.key_errors("openai", saved, "X") == []
    keychain._cache.clear()
    fake_keychain.locked = True
    assert "locked" in hosted.key_errors("openai", saved, "X")[0]
    assert "api_key_env is empty" in hosted.key_errors("openai", section(), "X")[0]


def test_a_reference_to_someone_elses_item_fails_validation() -> None:
    errors = hosted.hosted_fields_errors("openai", section(api_key="keychain:github"))
    assert len(errors) == 1 and "keychain:localmelo.name" in errors[0]
