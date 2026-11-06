"""What counts as a password the person gave, and what the vault keeps (#547)."""

from __future__ import annotations

import pytest

from localmelo.melo.contracts.sign_in import HANDLE, MAX_HANDLES, UNUSABLE, Vault

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
    vault = Vault()
    out, handles = vault.hide(text)
    assert out == masked and handles == ("[password-1]",)
    kept = vault.value("[password-1]")
    assert kept is not None and kept not in out
    assert vault.hide(out) == (out, ()), "hiding a hidden text changes nothing"


@pytest.mark.parametrize("text", NOT_GIVEN)
def test_a_text_that_is_not_plainly_a_password_given_is_left_alone(text: str) -> None:
    assert Vault().hide(text) == (text, ())


def test_the_same_value_has_one_handle_and_the_vault_is_bounded_and_shows_nothing() -> (
    None
):
    vault = Vault()
    out, handles = vault.hide("password: hunter2 and password: hunter2 and pw=other99")
    assert handles == ("[password-1]", "[password-1]", "[password-2]")
    assert (
        out == "password: [password-1] and password: [password-1] and pw=[password-2]"
    )
    assert "hunter2" not in repr(vault) and "other99" not in repr(vault)
    for n in range(MAX_HANDLES):
        vault.hide(f"password: value{n}x")
    over, _ = vault.hide("password: onemore1")
    assert over == f"password: {UNUSABLE}" and vault.value(UNUSABLE) is None
    assert HANDLE.fullmatch(UNUSABLE) and not HANDLE.fullmatch("[password-0]")
    vault.forget()
    assert vault.value("[password-1]") is None
