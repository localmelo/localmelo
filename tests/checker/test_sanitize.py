"""sanitize_assistant_text: removals, escapes, nesting and idempotence."""

from __future__ import annotations

import time

import pytest

from localmelo.melo.checker.sanitize import sanitize_assistant_text

ROLE_TOKENS = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<|developer|>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|im_start|>",
    "<|im_end|>",
)
TAG_NAMES = ("tool_call", "tool_response", "arg_key", "arg_value", "note")

# (input, expected output) for every rule; each is also checked for
# idempotence below.
CASES: list[tuple[str, str]] = [
    # Role tokens, escaped in exact case only.
    *[(f"a {token} b", f"a &lt;{token[1:-1]}&gt; b") for token in ROLE_TOKENS],
    ("<|USER|> and <|Im_Start|>", "<|USER|> and <|Im_Start|>"),
    (
        "<|im_start|>assistant\nhi<|im_end|>",
        "&lt;|im_start|&gt;assistant\nhi&lt;|im_end|&gt;",
    ),
    # Think tags, removed in exact case only.
    ("a<think>b</think>c", "abc"),
    ("<think></think><think>", ""),
    ("<THINK>kept</Think>", "<THINK>kept</Think>"),
    # NUL characters, removed before the think tags.
    ("a\x00b\x00", "ab"),
    ("<thi\x00nk>", ""),
    ("</think\x00>x", "x"),
    # Nested think tags.
    ("<thi<think>nk>", ""),
    ("<</think>think>", ""),
    ("</thi</think>nk>", ""),
    ("<th<thi<think>nk>ink>x", "x"),
    ("a<thi<think>nk>b", "ab"),
    # Tool-call and note tags, both forms, any case; the case is kept.
    *[(f"<{name}>x</{name}>", f"&lt;{name}&gt;x&lt;/{name}&gt;") for name in TAG_NAMES],
    *[
        (
            f"<{name.upper()}>x</{name.title()}>",
            f"&lt;{name.upper()}&gt;x&lt;/{name.title()}&gt;",
        )
        for name in TAG_NAMES
    ],
    # Function and parameter elements: the opening up to "=", and the
    # closing tag.
    (
        "<function=file_read><parameter=path>a.py</parameter></function>",
        "&lt;function=file_read>&lt;parameter=path>a.py&lt;/parameter&gt;&lt;/function&gt;",
    ),
    (
        "<Function=x><PARAMETER=y>z</Parameter></FUNCTION>",
        "&lt;Function=x>&lt;PARAMETER=y>z&lt;/Parameter&gt;&lt;/FUNCTION&gt;",
    ),
    # Everything at once. Only the think tags go, not the text between them.
    (
        "<think>plan</think><tool_call>\x00<|user|></tool_call>",
        "plan&lt;tool_call&gt;&lt;|user|&gt;&lt;/tool_call&gt;",
    ),
    # A nested note tag is escaped once its think tag is gone.
    ("<no<think>te>x", "&lt;note&gt;x"),
]

# Text that holds no token or tag of any rule.
ORDINARY = [
    "a < b > c",
    "<div>hello</div>",
    "&lt;tool_call&gt; is already escaped",
    "x <= y && y >= z",
    "<thinking> and <think > and <notes> and <function> and <parameter>",
    "<|unknown|> and <| user |>",
    "读取 list_ops.py 文件，然后运行测试。",
    "",
]


@pytest.mark.parametrize("token", ROLE_TOKENS)
def test_each_role_token_is_escaped(token: str) -> None:
    name = token[2:-2]
    assert sanitize_assistant_text(token) == f"&lt;|{name}|&gt;"


@pytest.mark.parametrize("name", TAG_NAMES)
def test_each_tag_is_escaped_in_both_forms_and_any_case(name: str) -> None:
    for tag in (name, name.upper(), name.capitalize()):
        assert sanitize_assistant_text(f"<{tag}>") == f"&lt;{tag}&gt;"
        assert sanitize_assistant_text(f"</{tag}>") == f"&lt;/{tag}&gt;"


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_cases(text: str, expected: str) -> None:
    assert sanitize_assistant_text(text) == expected


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_idempotent(text: str, expected: str) -> None:
    once = sanitize_assistant_text(text)
    assert sanitize_assistant_text(once) == once


@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_text_is_unchanged(text: str) -> None:
    assert sanitize_assistant_text(text) == text


def test_nul_removal_can_complete_a_think_tag() -> None:
    assert sanitize_assistant_text("<thi\x00nk>secret</thi\x00nk>") == "secret"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Many nested tags side by side: 14 x 71,429 characters.
        ("<thi<think>nk>" * 71_429, ""),
        # One tag nested 142,857 deep: 7 x 142,857 + 7 characters.
        ("<thi" * 142_857 + "<think>" + "nk>" * 142_857, ""),
        # Closing tags nested 62,500 deep, after 500,000 characters of text.
        ("x" * 500_000 + "</thi" * 62_500 + "</think>" + "nk>" * 62_500, "x" * 500_000),
    ],
    ids=["side-by-side", "deep", "deep-closing"],
)
def test_a_million_characters_of_nested_tags_take_under_a_second(
    text: str, expected: str
) -> None:
    assert len(text) >= 1_000_000
    start = time.perf_counter()
    result = sanitize_assistant_text(text)
    elapsed = time.perf_counter() - start
    assert result == expected
    assert elapsed < 1.0
