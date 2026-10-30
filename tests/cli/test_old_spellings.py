"""Every spelling the command line took before the short commands still parses the same.

``golden/old_spellings.json`` was recorded from the parser at 66c22f85, before any
command word, letter or ``-X`` existed: one valid spelling of each of the 65 options,
the benchmark's, the macOS app's and a worker launcher's command lines, abbreviations
that mean something, and values that are errors. Each case holds the namespace
(changed fields only) or the error text of both parsers. A new spelling may add a way
to say something; it may not change what an old one meant, so the benchmark, the app
and the launchers need no new baseline.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

import localmelo.__main__ as cli

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "old_spellings.json").read_text("utf-8")
)
CASES = [pytest.param(c, id=" ".join(c["argv"]) or "no-args") for c in GOLDEN["cases"]]


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))  # help reads [ui] mode from the config


def _plain(value: object) -> object:
    return {"set": sorted(value)} if isinstance(value, frozenset | set) else value


def _probe(kind: str, argv: list[str]) -> dict[str, Any]:
    parser = cli._build_parser(direct_only=kind == "legacy")
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            ns, unknown = parser.parse_known_args(argv)
    except SystemExit as exc:
        lines = err.getvalue().strip().splitlines()
        return {"exit": exc.code, "error": lines[-1] if lines else ""}
    except argparse.ArgumentError as exc:
        return {"error": str(exc)}
    defaults = GOLDEN["defaults"][kind]
    changed = {
        key: _plain(value)
        for key, value in vars(ns).items()
        if _plain(defaults.get(key, "<absent>")) != _plain(value)
    }
    return {"ns": changed, **({"unknown": unknown} if unknown else {})}


@pytest.mark.parametrize("case", CASES)
def test_an_old_spelling_parses_as_it_did(case: dict[str, Any]) -> None:
    for kind in ("legacy", "full"):
        assert _probe(kind, case["argv"]) == case[kind], kind


@pytest.mark.parametrize("kind", ["legacy", "full"])
def test_the_namespace_gains_no_field_and_no_default_changes(kind: str) -> None:
    parser = cli._build_parser(direct_only=kind == "legacy")
    ns, unknown = parser.parse_known_args([])
    assert unknown == []
    assert {k: _plain(v) for k, v in vars(ns).items()} == GOLDEN["defaults"][kind]


def test_every_old_option_string_is_still_an_option() -> None:
    parser = cli._build_parser()
    assert set(GOLDEN["options"]) <= set(parser._option_string_actions)


@pytest.mark.parametrize("kind", ["legacy", "full"])
def test_a_prefix_of_an_old_option_means_what_it_meant(kind: str) -> None:
    """``--s`` is ``--serve`` and ``--out`` is ``--output-format``; the new names join
    the options that need their full spelling, so no prefix gains or loses a match."""
    parser = cli._build_parser(direct_only=kind == "legacy")
    old = {o for o in GOLDEN["options"] if o.startswith("--")}
    if kind == "legacy":
        old = {o for o in old if o in parser._option_string_actions}
    hidden = set(GOLDEN["full_name_only"])
    for name in sorted(old):
        for end in range(3, len(name)):
            prefix = name[:end]
            expected = sorted(
                o for o in old if o.startswith(prefix) and o not in hidden
            )
            got = sorted(item[1] for item in parser._get_option_tuples(prefix))
            assert got == expected, prefix


def test_the_benchmark_still_finds_no_save_session_in_main() -> None:
    """The benchmark adds ``--no-save-session`` only when ``__main__.py`` names it
    (its harness greps that file), so the literal must stay in that module."""
    assert "--no-save-session" in Path(cli.__file__).read_text(encoding="utf-8")
