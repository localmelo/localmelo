"""What counts as a password the person gave, and what the vault keeps (#547)."""

from __future__ import annotations

import time

import pytest

from localmelo.melo.contracts.sign_in import (
    HANDLE,
    MAX_KEPT,
    SPENT,
    VAULT,
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
    # A quote or a backtick inside the value is part of it; a backtick pair is a
    # quote pair; an opening quote with no closing one on its line is no quote.
    ('password: ab"cd1234', "password: [password-1]"),
    ('pw: hunter"22xx', "pw: [password-1]"),
    ("password: `hunter22`", "password: `[password-1]`"),
    ('password: "hunter22', 'password: "[password-1]'),
    ('my password "hunter22', 'my password "[password-1]'),
    # ... and the rest of a later line is no part of it.
    ('password: "hunter22\nthen "ok" here', 'password: "[password-1]\nthen "ok" here'),
    ('he wrote "my password is hunter22".', 'he wrote "my password is [password-1]".'),
    # Closing brackets after a closing quote are no part of the value, any others are.
    ('{"hint": "my password is hunter22"}', '{"hint": "my password is [password-1]"}'),
    ('["pw: hunter22"]', '["pw: [password-1]"]'),
    ("password: x1y2}", "password: [password-1]"),
    # Emphasis after the colon is part of a value that begins with it.
    ("password:**hunter22", "password:[password-1]"),
    # One question mark ends a plain word, nothing else does.
    ("my password is hunter??", "my password is [password-1]"),
    ("my password hunter!", "my password [password-1]"),
    ("password: hunter?", "password: [password-1]"),
    # The password word wrapped in quotes or emphasis, or the end of a longer name.
    ('{"password": "hunter22"}', '{"password": "[password-1]"}'),
    ("`password`: hunter22", "`password`: [password-1]"),
    ("**Password:** hunter22", "**Password:** [password-1]"),
    ("DB_PASSWORD=hunter22", "DB_PASSWORD=[password-1]"),
    ("db-password: hunter22", "db-password: [password-1]"),
    ('"dbPassword": "hunter22"', '"dbPassword": "[password-1]"'),
    ("userPwd：hunter22", "userPwd：[password-1]"),
    # A password word inside a quoted value is read for its own value: a quoted value
    # that starts there and reaches past the first one is one value with it, from the
    # first quote, so no quote is left to pair differently.
    (
        '"password": "abcd1234 then pw="hunter22" ok',
        '"password": [password-1]" ok',
    ),
    # A run that is no value by its shape (a plain word) holds the words of a value
    # that is read the same way: the run is one value with it.
    ('Pwd 我\'pw是"abcd" x', 'Pwd [password-1]" x'),
]
NOT_GIVEN = [
    "I forgot my password",
    "the password is wrong",
    "use the password manager",
    "password: abc",  # shorter than a password
    "keypassword: abcd1234",  # not the word itself
    "OTP code: 123456",  # a one-time code is no password: left alone
    "my password is [password-1]",  # a handle is not a value
    'password: "abc" please',  # fewer than MIN_CHARS between the quotes
    # Quote characters are no symbols for the plain-word test.
    "the password isn't working",
    "my password doesn't work",
    "the password won’t sync",
    'my password "secret',
    # A longer name counts only when the word ends it, as a name.
    "password_hash=x1y2z3",
    "keypassword: hunter22",
    "my_password is hunter22",  # only a ":" or "=" joint
    "OLDPWD=/tmp",
    # A question is no password.
    "what's my password again?",
    "密码是对的吗？",
]


@pytest.mark.parametrize(("text", "masked"), GIVEN)
def test_a_password_the_person_gives_is_cut_out_and_a_handle_stands_in(
    text: str, masked: str
) -> None:
    out = hide_passwords(text)
    assert out == masked and handles_in(out) == {"[password-1]"}
    kept = VAULT.value("[password-1]")
    assert kept is not None and kept not in out
    head, tail = masked.split("[password-1]")
    assert kept == text[len(head) : len(text) - len(tail)], "the whole value is kept"
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
        f'pw: {SPENT}".',  # and the marks after it
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
        # A quote inside the value, an opening quote never closed, a long value
        # then trailing marks, and a closing quote that stands first.
        'password: ab"cd1234 and pw: "hunter22',
        f'pw: {_run(200)}".',
        f'pw: "{_run(200)}',
        "\tpw are 「口令：」[…].",
        'my password [password-1]".',
        # A word read inside an earlier value skips nothing it does not hide.
        "dbPw:\npassword PASSWORD\tx1y2z3",
        # A quoted value that starts inside a run and reaches past it.
        'PWD=/tmpPWD="\t = "PASSWORD',
        # A run that is no value holds a word whose value is quoted.
        'Pwd 口令是"hunter 22"',
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
        f'pw: {SPENT}".',
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
    '"pwd /tmp/x"',  # marks at the end of the run do not count
    "`pwd /tmp/x`",
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


@pytest.mark.parametrize(
    ("text", "values"),
    [
        # A quote ended a run before, so the password word after it was read.
        ('pw: abcd"pw: efgh1234', ['abcd"pw:', "efgh1234"]),
        ('pw: abcd"pw:"efgh 1234" x', ['abcd"pw:"efgh 1234']),
        ('my password "abcd.pw: efgh1234', ["abcd.pw:", "efgh1234"]),
        # Likewise a longer name or a wrapped word.
        ("DB_PASSWORD=abcd,pw: efgh1234", ["abcd,pw:", "efgh1234"]),
        ('"password":abcd,pw: efgh1234', ["abcd,pw:", "efgh1234"]),
    ],
)
def test_a_password_word_inside_a_longer_run_still_gives_its_own_value(
    text: str, values: list[str]
) -> None:
    assert [text[a:b] for a, b in password_spans(text)] == values
    assert "efgh" not in without_passwords(text)


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
        ("pw:" + "“" * 300_000, 0),  # a run of quote characters after a cue
        ('pw:ab"' * 50_000, 1),  # one run, a quote and a word in every part of it
        # Values that merge, again and again: no value is copied for each merge.
        ("pw:a" + '"pw:"bcde f' * 55_000, 1),
    ],
    ids=[
        "openers",
        "assignments",
        "lower-case variables",
        "variables",
        "tags",
        "quotes",
        "run",
        "merges",
    ],
)
def test_a_long_line_is_read_in_time_in_proportion_to_it(line: str, spans: int) -> None:
    # No run is read twice: a quoted value needs its closing quote on the line, an
    # exempt shape reads its run once, and a result is built in one join.
    started = time.perf_counter()
    assert len(password_spans(line)) == spans
    assert len(hide_passwords(line)) <= len(line)
    assert time.perf_counter() - started < 1, "tens of milliseconds, seconds if not"
