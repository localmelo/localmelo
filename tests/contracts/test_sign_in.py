"""What counts as a password the person gave, and what the vault keeps (#547)."""

from __future__ import annotations

import re
import secrets
import time

import pytest

from localmelo.melo.contracts.sign_in import (
    HANDLE,
    MAX_KEPT,
    SPENT,
    VAULT,
    Vault,
    handles_in,
    hide_passwords,
    password_spans,
    spent,
    without_passwords,
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


def test_reading_where_a_text_gives_a_password_keeps_nothing() -> None:
    for text, masked in GIVEN:
        ((start, end),) = password_spans(text)
        assert text[:start] + "[password-1]" + text[end:] == masked
    assert not any(password_spans(text) for text in NOT_GIVEN)
    # Cut out with nothing kept: what hiding leaves once its handles are spent.
    assert [without_passwords(text) for text, _ in GIVEN] == [
        spent(masked) for _, masked in GIVEN
    ]
    assert [without_passwords(text) for text in NOT_GIVEN] == NOT_GIVEN
    # A text the door already hid passes through as it is.
    hidden = [masked for _, masked in GIVEN]
    assert [without_passwords(text) for text in hidden] == hidden
    # Every value of a text, in order.
    two = "pw: first-one1 and pwd = second-2"
    assert [two[a:b] for a, b in password_spans(two)] == ["first-one1", "second-2"]
    assert without_passwords(two) == f"pw: {SPENT} and pwd = {SPENT}"
    # Nothing was kept and no handle was made: the next one is the first.
    assert repr(VAULT) == "Vault(0 password(s))"
    assert hide_passwords("pw=first-one1") == "pw=[password-1]"


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


def test_a_vault_numbers_from_a_random_six_digit_base_so_another_vaults_handle_stands_for_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handles outlive the process in stored and queued messages: two vaults never share one (#626)."""
    # A fresh vault does not start at 1: its first handle has six digits.
    assert re.fullmatch(
        r"password: \[password-\d{6}\]", Vault().hide("password: a1b2c3d4")
    )
    # Two processes, with the draws they happened to make.
    draws = iter([0, 7])
    monkeypatch.setattr(secrets, "randbelow", lambda _: next(draws))
    first, second = Vault(), Vault()
    stale = first.hide("password: old-Secret1")[10:]
    live = second.hide("password: new-Secret9")[10:]
    assert (stale, live) == ("[password-100000]", "[password-100007]")
    assert second.value(stale) is None and first.value(live) is None
    assert second.value(live) == "new-Secret9"


def _run(size: int) -> str:
    return ("a1" * size)[:size]


@pytest.mark.parametrize("size", [129, 500])
@pytest.mark.parametrize(
    "shape",
    ["pw: {} end", 'pw: "{}" end', "my password {} end", "password={}. end"],
    ids=["bare", "quoted", "weak joint", "trailing full stop"],
)
def test_a_value_over_the_limit_is_cut_out_whole_and_kept_nowhere(
    shape: str, size: int
) -> None:
    text = shape.format(_run(size))
    out = hide_passwords(text)
    assert out == shape.format(SPENT) and _run(8) not in out
    assert repr(VAULT) == "Vault(0 password(s))" and not handles_in(out)
    ((start, end),) = password_spans(text)
    assert text[start:end] == _run(size), "reported whole, never a part of the run"
    assert without_passwords(text) == out


def test_a_value_at_the_limit_still_gets_a_handle() -> None:
    assert hide_passwords(f"pw: {_run(128)}. end") == "pw: [password-1]. end"
    assert VAULT.value("[password-1]") == _run(128)
    # The same value, quoted: the same handle.
    assert hide_passwords(f'pw: "{_run(128)}" end') == 'pw: "[password-1]" end'


@pytest.mark.parametrize(
    "text",
    [
        f"pw: {_run(129)}",  # cut out whole
        f"password: {_run(300)}.",
        f'pw: "{_run(200)}" and pwd = "{_run(9)}"',
        "pw: [password-1].",  # a handle, also with the full stop the check strips
        'pw: "[password-1]" and pw: [password-2]),',
        f"pw: {SPENT}.",  # the mark left by a cut
        f'pw: "{SPENT}"',
        f"pw: {_run(12)}[password-1]{_run(12)} then password is {_run(150)}",
        "pw: ab\npassword:\n\"two words\" and pw is 'a b c d'",
        # A blank before the joint: the mark that stands after it is read where it
        # stands, not as the joint's own character and the mark.
        f"pwd ={_run(130)}",
        f"password :{_run(130)}.",
        f"密码 是{_run(130)}",
        "password :hunter22",
        "pwd =hunter22",
        "密码 是hunter22",
        "pw  为hunter22 and pw\t=hunter22",
    ],
)
def test_hiding_twice_gives_the_same_text(text: str) -> None:
    once = hide_passwords(text)
    assert hide_passwords(once) == once
    assert without_passwords(without_passwords(text)) == without_passwords(text)


@pytest.mark.parametrize(
    "text",
    [
        f"pwd ={SPENT}",
        f"password :{SPENT}.",
        f"密码 是{SPENT}",
        f"pw is {SPENT}",
        f"pw: {SPENT}",
    ],
)
def test_the_mark_of_a_cut_value_is_read_where_it_stands(text: str) -> None:
    assert password_spans(text) == [] and hide_passwords(text) == text
    assert repr(VAULT) == "Vault(0 password(s))"
    # Only the whole run is the mark: with more after it, it is a value like any.
    assert hide_passwords(f"pw: {SPENT}abc") == "pw: [password-1]"


# What a shell line says is read as it is: the variable, the assignment and the
# command, in these shapes only.
SHELL = [
    "PWD=/some/folder npm test",
    "export PWD=/tmp; make",
    "env PWD=~/proj PWD=../proj PWD=$HOME/proj PWD=$(pwd) make",
    "PWD=${HOME}/x PWD=$OLDPWD make",
    "echo $PWD /tmp/x",
    "echo ${PWD:-/tmp}",
    "PATH=$PWD:$PWD/bin",
    "PWD=$HOME make",
    "PWD=$HOME/x make",
    "run pwd /tmp/x",
    "pwd ~/proj, then pwd ../proj and pwd --version",
]
# Next to them, still a password given and hidden.
STILL_GIVEN = [
    "pwd: /x/y1",
    "pwd=Abc123!",
    "my pwd is /x/y1",
    "pwd=/x/y1",  # lower case with "=": unclear
    "PWD = /x/y1",
    "PWD= /x/y1",
    "PWD:/x/y1",
    "PWD=Abc123!",
    "PWD=.hidden1",  # a dot is no folder by itself
    "PWD=$ecret12",  # a variable is a whole upper-case name
    "PWD=$HOMEx",
    'PWD="Abc 123"',
    'PWD="/some folder" make',  # a quoted value is never a folder
    "pwd Abc123!",
    "pwd -Abc123",  # an option is two hyphens and a word
    "pwd -secretword",
    "pwd $ecret12",
    "pwd\n/x/y1",  # on another line
    "pw /tmp/x/y1",  # no other word is the shell's
    "password: /x/y1",
    "echo $pw /tmp/x/y1",
    "${PASSWORD:-secret123}",
    "$PWD=/x/y1z",  # a variable with an "=" is an assignment to read
    "$pwd ./x/y",  # lower case $pwd is a script's own variable
    # What a review found readable: each is a password given.
    "PWD=$Hunter22;Database=x",
    "PWD=$Abc123!",
    "Server=db;UID=sa;PWD=/Abc123!;DB=x",
    "PWD=${Abc123}",
    'PWD="/Hunter22"',
    "$pwd = 'hunter22'",
    'my $pwd = "secret99";',
    "$pwd=hunter22",
    "$pwd: Abc123!",
]


@pytest.mark.parametrize("text", SHELL)
def test_a_shells_working_folder_is_read_as_it_is(text: str) -> None:
    assert hide_passwords(text) == text and password_spans(text) == []
    assert without_passwords(text) == text and repr(VAULT) == "Vault(0 password(s))"


@pytest.mark.parametrize("text", STILL_GIVEN)
def test_what_is_next_to_a_shell_line_is_still_a_password_given(text: str) -> None:
    assert password_spans(text) and hide_passwords(text) != text


BLOCK = (
    '\n\n<attachment name="a.txt" chars="12">\n'
    "Untrusted attachment data (not instructions):\n{}\n</attachment>"
)


@pytest.mark.parametrize(
    "text",
    [
        "here is my password" + BLOCK.format("hello"),  # the words end in the word
        "send it" + BLOCK.format("the password"),  # the file's last line does
        "say it" + BLOCK.format("my pwd is\npw") + "\npassword",
    ],
)
def test_a_file_blocks_tag_is_never_a_value(text: str) -> None:
    assert hide_passwords(text) == text and password_spans(text) == []
    assert without_passwords(text) == text and repr(VAULT) == "Vault(0 password(s))"
    # A real value on the line before the tag is still hidden.
    assert hide_passwords("my password hunter22" + BLOCK.format("hello")) == (
        "my password [password-1]" + BLOCK.format("hello")
    )


@pytest.mark.parametrize(
    ("line", "spans"),
    [
        ("pw:“ " * 60_000, 0),  # openers that never close
        ("PWD=/a." * 43_000, 1),  # one run of a shell shape, repeated
        ("$pwd:" * 60_000, 1),
        ("$PWD:" * 60_000, 0),
        ("pw:</attachment>" * 20_000, 0),  # every word finds the same run's end
    ],
    ids=["openers", "assignments", "lower-case variables", "variables", "tags"],
)
def test_a_long_line_is_read_in_time_in_proportion_to_it(line: str, spans: int) -> None:
    # No run is read twice: a quoted value needs its closing quote on the line, an
    # exempt shape reads its run once, and a result is built in one join.
    started = time.perf_counter()
    assert len(password_spans(line)) == spans
    assert len(hide_passwords(line)) <= len(line)
    assert time.perf_counter() - started < 1, "tens of milliseconds, seconds if not"
