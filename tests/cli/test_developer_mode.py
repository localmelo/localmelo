"""Default mode and developer mode (#186): the tier table and what ``--help`` shows.

The mode decides only what help prints; every option parses in both modes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import cast

import pytest

import localmelo.__main__ as cli
from localmelo.support import config
from localmelo.support.option_tiers import (
    ADVANCED,
    EXPERIMENTAL,
    EXPERIMENTAL_ISSUES,
    OPTION_TIERS,
    USER,
)
from localmelo.tests.cli.test_direct_mode import _main

# Adding a user option is a deliberate change: update this number with it.
USER_OPTION_COUNT = 15

HEADINGS = (
    "Options:",
    "Advanced:",
    "Experimental (compared on the benchmark, may change or go away):",
)


def _long_names(parser: argparse.ArgumentParser) -> set[str]:
    return {
        next(s for s in action.option_strings if s.startswith("--"))
        for action in parser._actions
        if action.option_strings
    }


def _help(*argv: str, capsys: pytest.CaptureFixture[str]) -> str:
    assert _main(*argv).code == 0
    return capsys.readouterr().out


def _shown_options(text: str) -> list[str]:
    """The long option of each entry of a help text (an entry starts at column 2)."""
    return re.findall(r"^  (?:-\w, )?(--[a-z-]+)", text, re.M)


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))


# -- the table -----------------------------------------------------------------


def test_the_table_and_the_parser_name_the_same_options() -> None:
    assert set(OPTION_TIERS) == _long_names(cli._build_parser())


def test_every_option_has_a_known_tier() -> None:
    assert set(OPTION_TIERS.values()) == {USER, ADVANCED, EXPERIMENTAL}


def test_an_experimental_option_names_the_issue_that_decides_it() -> None:
    for name, issue in EXPERIMENTAL_ISSUES.items():
        assert OPTION_TIERS[name] == EXPERIMENTAL
        assert re.fullmatch(r"#\d+", issue), name


def test_the_user_tier_has_the_pinned_count() -> None:
    users = [name for name, tier in OPTION_TIERS.items() if tier == USER]
    assert len(users) == USER_OPTION_COUNT


# -- default mode --------------------------------------------------------------


def test_default_help_shows_only_the_user_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = _help("--help", capsys=capsys)
    shown = _shown_options(text)
    assert len(shown) == USER_OPTION_COUNT
    assert set(shown) == {n for n, tier in OPTION_TIERS.items() if tier == USER}
    assert HEADINGS[0] in text
    assert not any(heading in text for heading in HEADINGS[1:])
    assert text.rstrip().endswith(
        'More options: melo --developer --help (or set [ui] mode = "developer").'
    )


def test_the_parser_default_is_default_help() -> None:
    text = cli._build_parser(developer=False).format_help()
    assert len(_shown_options(text)) == USER_OPTION_COUNT


@pytest.mark.parametrize(
    "argv",
    [
        ["--reasoning", "compress", "--retriever", "bm25", "--recall-k", "2", "q"],
        ["--window-eviction", "chunk", "--fixed-prefix", "on", "q"],
        ["--attempt-policy", "none", "--reflection-gate", "utility", "q"],
        ["--progress", "off", "--builtins", "none", "--memory-dir", "m", "q"],
    ],
)
def test_a_hidden_option_still_parses_in_default_mode(argv: list[str]) -> None:
    args = cli._build_parser().parse_args(argv)
    assert args.query == ["q"]


def test_a_hidden_option_still_reaches_the_run() -> None:
    ran = _main("--window-eviction", "chunk", "--retriever", "bm25", "q")
    assert ran.code == 0
    options = ran.build.call_args.kwargs["options"]
    assert options.window_eviction == "chunk"


# -- developer mode ------------------------------------------------------------


def test_developer_help_lists_every_option_under_its_tier() -> None:
    text = cli._build_parser(developer=True).format_help()
    assert sorted(_shown_options(text)) == sorted(OPTION_TIERS)
    positions = [text.index(heading) for heading in HEADINGS]
    assert positions == sorted(positions)
    for name, tier in OPTION_TIERS.items():
        start = positions[(USER, ADVANCED, EXPERIMENTAL).index(tier)]
        end = min([p for p in positions if p > start], default=len(text))
        assert re.search(rf"^  (-\w, )?{name}\b", text[start:end], re.M), name


def test_an_experimental_line_names_its_issue() -> None:
    text = cli._build_parser(developer=True).format_help()
    entries = re.split(r"\n(?=  -)", text[text.index(HEADINGS[2]) :])
    for name, issue in EXPERIMENTAL_ISSUES.items():
        entry = next(e for e in entries if e.startswith(f"  {name}"))
        assert f"decided in {issue}" in " ".join(entry.split())
    assert "decided in #" not in text[: text.index(HEADINGS[2])]


def test_the_flag_shows_developer_help_before_or_after_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for argv in (["--developer", "--help"], ["--help", "--developer"]):
        text = _help(*argv, capsys=capsys)
        assert sorted(_shown_options(text)) == sorted(OPTION_TIERS)


def test_the_help_text_of_an_experimental_option_is_not_changed_by_a_second_call() -> (
    None
):
    parser = cli._build_parser(developer=True)
    assert parser.format_help() == parser.format_help()
    window = next(a for a in parser._actions if a.dest == "window_eviction")
    assert "decided in" not in str(window.help)


def test_developer_flag_is_not_an_abbreviation_and_keeps_old_ones() -> None:
    parser = cli._build_parser()
    assert parser.parse_args(["--d", "status"]).daemon == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["--dev"])


def test_a_developer_after_the_separator_is_task_text() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["--", "--developer"])
    assert args.query == ["--developer"]
    assert cast(cli._WorkerFlagParser, parser).developer is None


def test_the_direct_parser_has_no_developer_flag() -> None:
    assert (
        "--developer" not in cli._build_parser(direct_only=True)._option_string_actions
    )


def test_a_task_with_the_flag_runs_as_before() -> None:
    ran = _main("--developer", "--", "q")
    assert ran.code == 0
    ran.agent.run_task.assert_awaited_once_with("q")


# -- [ui] mode -----------------------------------------------------------------


def test_the_config_mode_shows_developer_help(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(tmp_path, monkeypatch, '[ui]\nmode = "developer"\n')
    text = _help("--help", capsys=capsys)
    assert sorted(_shown_options(text)) == sorted(OPTION_TIERS)


@pytest.mark.parametrize("text", ["", '[ui]\nmode = "default"\n', "[ui]\n"])
def test_default_or_absent_mode_shows_default_help(
    text: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(tmp_path, monkeypatch, text)
    assert len(_shown_options(_help("--help", capsys=capsys))) == USER_OPTION_COUNT


@pytest.mark.parametrize("text", ['[ui]\nmode = "dev"\n', "[ui]\nmode = 3\n"])
def test_a_bad_mode_warns_and_shows_default_help(
    text: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(tmp_path, monkeypatch, text)
    assert _main("--help").code == 0
    out, err = capsys.readouterr()
    assert len(_shown_options(out)) == USER_OPTION_COUNT
    assert err.startswith("[ui] mode=") and "default help" in err


def test_an_unreadable_config_still_shows_default_help(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(tmp_path, monkeypatch, "[ui\nmode = ")
    text = _help("--help", capsys=capsys)
    assert len(_shown_options(text)) == USER_OPTION_COUNT


def test_the_mode_is_validated_and_saved() -> None:
    assert config.Config().ui.mode == "default"
    assert config.Config().ui_errors() == []
    bad = config.Config()
    bad.ui.mode = "expert"
    assert "[ui] mode='expert'" in bad.ui_errors()[0]
    # A display preference: it never stops a run or the gateway.
    assert not any("[ui]" in error for error in bad.validate())


def test_the_mode_survives_a_save_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.toml"))
    cfg = config.Config()
    cfg.ui.mode = "developer"
    config.save(cfg)
    assert '[ui]\nmode = "developer"' in (tmp_path / "config.toml").read_text()
    assert config.load().ui.mode == "developer"
    config.save(config.Config())
    assert "[ui]" not in (tmp_path / "config.toml").read_text()
