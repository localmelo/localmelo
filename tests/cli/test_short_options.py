"""Letters and one-word names are more ways to say what an old option says.

Each new spelling is another action on the old option's dest, type and choices, so it
gives the namespace the old spelling gives, in both parsers. The old option keeps its
own name in its own errors, and a prefix of an old option keeps its meaning.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

import localmelo.__main__ as cli

# (new spelling, old spelling) with a value that is valid for both.
ALIASES = [
    ("-m qwen3", "--chat-model qwen3"),
    ("--model qwen3", "--chat-model qwen3"),
    ("--model=qwen3", "--chat-model=qwen3"),
    ("-m=qwen3", "--chat-model qwen3"),
    ("-u http://127.0.0.1:8000/v1", "--base-url http://127.0.0.1:8000/v1"),
    ("--url=http://x/v1", "--base-url http://x/v1"),
    ("-w /tmp", "--workspace /tmp"),
    ("-s 7", "--max-steps 7"),
    ("--steps 7", "--max-steps 7"),
    ("-t 2.5", "--timeout 2.5"),
    ("-o json", "--output-format json"),
    ("--output json", "--output-format json"),
    ("-o stream-json", "--output-format stream-json"),
    ("--mcp servers.json", "--mcp-config servers.json"),
    ("-y", "--yolo"),
    ("-m a -u http://x/v1 -s 3 -t 9 -o json", "--chat-model a --base-url http://x/v1 "
     "--max-steps 3 --timeout 9 --output-format json"),
]  # fmt: skip


def _parse(kind: str, argv: str) -> dict[str, Any]:
    parser = cli._build_parser(direct_only=kind == "legacy")
    return vars(parser.parse_known_args([*argv.split(), "--", "task"])[0])


@pytest.mark.parametrize("kind", ["legacy", "full"])
@pytest.mark.parametrize(("new", "old"), ALIASES, ids=[n for n, _ in ALIASES])
def test_a_new_spelling_gives_the_namespace_of_the_old_one(
    kind: str, new: str, old: str
) -> None:
    if kind == "legacy" and "-y" in new.split():
        pytest.skip("--yolo is for the interactive UI; the direct parser has no -y")
    assert _parse(kind, new) == _parse(kind, old)


def test_version_prints_the_same_with_a_letter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    outputs = []
    for flag in ("--version", "-V"):
        with pytest.raises(SystemExit) as exc:
            cli._build_parser().parse_args([flag])
        assert exc.value.code == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1] and outputs[0].startswith("melo ")


@pytest.mark.parametrize(
    ("typed", "name"),
    [
        ("-t", "-t"),
        ("--timeout", "--timeout"),
        ("-s", "-s/--steps"),
        ("--steps", "-s/--steps"),
        ("--max-steps", "--max-steps"),
    ],
)
def test_an_error_names_the_option_as_it_was_typed(typed: str, name: str) -> None:
    parser = cli._build_parser(direct_only=True)
    with pytest.raises(argparse.ArgumentError, match=rf"^argument {name}: "):
        parser.parse_known_args([typed, "soon"])


@pytest.mark.parametrize(
    "token", ["-s5", "-unknown", "-mfoo", "-tx", "--mod", "--ste", "--ur"]
)
def test_a_letter_takes_its_value_apart_and_a_new_name_is_not_an_abbreviation(
    token: str,
) -> None:
    """``-unknown`` is not ``-u nknown`` and ``--mod`` is not ``--model``; only the
    old names keep their prefixes (``--out`` is still ``--output-format``)."""
    namespace, unknown = cli._build_parser().parse_known_args([token])
    assert unknown == [token]
    assert namespace.chat_model is None and namespace.max_steps is None


def test_the_prefixes_that_mean_something_still_do() -> None:
    parser = cli._build_parser()
    assert parser.parse_args(["--s"]).serve is True
    assert parser.parse_args(["--out", "json", "q"]).output_format == "json"
    assert parser.parse_args(["--base", "http://x/v1"]).base_url == "http://x/v1"
    assert parser.parse_args(["--chat", "m"]).chat_model == "m"
