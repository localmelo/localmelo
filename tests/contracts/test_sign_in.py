"""What counts as a password the person gave, and what the vault keeps (#547)."""

from __future__ import annotations

import pytest

from localmelo.melo.contracts.sign_in import (
    HANDLE,
    MAX_KEPT,
    VAULT,
    handles_in,
    hide_passwords,
)

GIVEN = [
    # (message, what reaches everything but the page)
    ("sign in, my password is hunter2", "sign in, my password is [password-1]"),
    ("password: hunter2.", "password: [password-1]."),
    ('password="two words here" please', 'password="[password-1]" please'),
    ("pwd = 'p@ss word'", "pwd = '[password-1]'"),
    ("密码是 hunter2 谢谢", "密码是 [password-1] 谢谢"),
    ("密码：abc12345", "密码：[password-1]"),
    ("my password Hunter2!", "my password [password-1]"),
    # A card number after a password word is hidden too: it is only ever typed
    # into a password box.
    ("password: 4111111111111111", "password: [password-1]"),
]
NOT_GIVEN = [
    "I forgot my password",
    "the password is wrong",
    "use the password manager",
    "password: abc",  # shorter than a password
    "keypassword: abcd1234",  # not the word itself
    "OTP code: 123456",  # a one-time code is no password: left alone
    "my password is [password-1]",  # a handle is not a value
]


@pytest.mark.parametrize(("text", "masked"), GIVEN)
def test_a_password_the_person_gives_is_cut_out_and_a_handle_stands_in(
    text: str, masked: str
) -> None:
    out = hide_passwords(text)
    assert out == masked and handles_in(out) == {"[password-1]"}
    kept = VAULT.value("[password-1]")
    assert kept is not None and kept not in out
    assert hide_passwords(out) == out, "hiding a hidden text changes nothing"


@pytest.mark.parametrize("text", NOT_GIVEN)
def test_a_text_that_is_not_plainly_a_password_given_is_left_alone(text: str) -> None:
    assert hide_passwords(text) == text and not handles_in(text)


def test_the_same_value_has_one_handle_and_the_vault_is_bounded_and_shows_nothing() -> (
    None
):
    out = hide_passwords("password: hunter2 and password: hunter2 and pw=other99")
    assert (
        out == "password: [password-1] and password: [password-1] and pw=[password-2]"
    )
    assert "hunter2" not in repr(VAULT) and "other99" not in repr(VAULT)
    for n in range(MAX_KEPT):
        hide_passwords(f"password: value{n}x")
    # The newest are kept; an older handle stands for nothing and is never reused
    # for another value.
    assert VAULT.value("[password-1]") is None and not handles_in(out)
    newest = hide_passwords("password: onemore1")
    assert newest == f"password: [password-{MAX_KEPT + 3}]"
    assert VAULT.value(f"[password-{MAX_KEPT + 3}]") == "onemore1"
    assert HANDLE.fullmatch(newest[10:]) and not HANDLE.fullmatch("[password-0]")
    VAULT.forget()
    assert VAULT.value(f"[password-{MAX_KEPT + 3}]") is None
