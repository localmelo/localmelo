"""Default mode and developer mode (#186): the tier table and what ``-h`` shows.

``melo -h`` is the short help (20 lines); developer mode and ``-h all`` print every
option, command and setting. The mode decides only what help prints; every option
parses in both modes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import cast
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.support import command_words, config
from localmelo.support.option_tiers import (
    ADVANCED,
    EXPERIMENTAL,
    EXPERIMENTAL_ISSUES,
    OLD_SPELLINGS,
    OPTION_TIERS,
    SETTINGS,
    USER,
)
from localmelo.tests.cli.test_direct_mode import _main

# Adding a user option is a deliberate change: update this number with it.
USER_OPTION_COUNT = 11


def _long_names(parser: argparse.ArgumentParser) -> set[str]:
    """The first long name of each option; a letter-only spelling has none."""
    return {
        next(s for s in action.option_strings if s.startswith("--"))
        for action in parser._actions
        if any(s.startswith("--") for s in action.option_strings)
    }


def _help(*argv: str, capsys: pytest.CaptureFixture[str]) -> str:
    assert _main(*argv).code == 0
    return capsys.readouterr().out


def _shown_options(text: str) -> list[str]:
    """The long option of each row of a help text (a row starts at column 2)."""
    return re.findall(r"^  (?:-\w, )?(--[a-z-]+)(?= |$)", text, re.M)


def _is_short(text: str) -> bool:
    return "Commands: setup" in text and "Settings:" not in text


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


def test_the_cli_reference_table_matches_the_code() -> None:
    text = (Path(cli.__file__).parent / "docs/reference/cli.md").read_text("utf-8")
    table = text[text.index("| Option | Tier | Decided in |") :]
    rows = re.findall(
        r"^\| (?:`-\w`, )?`(--[a-z-]+)` \| (\w+) \| (#\d+)? ?\|$", table, re.M
    )
    assert {name: tier for name, tier, _ in rows} == OPTION_TIERS
    assert {name: issue for name, _, issue in rows if issue} == EXPERIMENTAL_ISSUES


def test_the_old_spelling_table_matches_the_code() -> None:
    text = (Path(cli.__file__).parent / "docs/reference/cli.md").read_text("utf-8")
    start = text.index("| Old spelling | New |")
    table = text[start : text.index("\n## ", start)]
    rows = re.findall(r"^\| `(--[a-z-]+)` \| `([^`]+)` \|$", table, re.M)
    assert dict(rows) == OLD_SPELLINGS


# -- default mode --------------------------------------------------------------


def test_default_help_fits_twenty_lines_and_names_no_old_spelling(
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = _help("-h", capsys=capsys)
    lines = text.splitlines()
    assert len(lines) <= 20 and max(map(len, lines)) <= 80
    assert _is_short(text)
    shown = set(_shown_options(text))
    assert shown == {n for n, tier in OPTION_TIERS.items() if tier == USER} - {"--help"}
    assert not shown & set(OLD_SPELLINGS)
    assert not any(" " + old in text for old in OLD_SPELLINGS)
    # No option is alone on its row: the row's text follows on the same line.
    assert all(re.match(r"^  -.* {2,}\S", row) for row in lines[6:16])


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


def test_the_full_help_lists_every_option_command_and_setting_without_wrapping(
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = _help("-h", "all", capsys=capsys)
    assert max(map(len, text.splitlines())) <= 80
    shown = set(_shown_options(text))
    settings = {item.flag for item in SETTINGS}
    visible = set(OPTION_TIERS) - set(OLD_SPELLINGS) - settings - {"--help"}
    assert visible <= shown
    assert not shown & set(OLD_SPELLINGS)
    for word in ("setup", "config", "serve", "console", "task", "worker", "import"):
        assert re.search(rf"^  {word} ", text, re.M), word
    assert "melo-retrieve" in text and "docs/reference/cli.md" in text
    for item in SETTINGS:
        assert re.search(rf"^  {re.escape(item.key)} +\S.* {item.flag}\b", text, re.M)


def test_an_experimental_setting_names_its_issue() -> None:
    text = cli._build_parser(developer=True).format_help()
    for item in SETTINGS:
        row = next(r for r in text.splitlines() if r.startswith(f"  {item.key} "))
        issue = EXPERIMENTAL_ISSUES.get(item.flag)
        assert (f"({issue})" in row) == bool(issue), item.key
    assert "decided in" not in text


def test_every_setting_flag_is_a_parser_option_with_a_tier() -> None:
    assert {item.flag for item in SETTINGS} <= set(OPTION_TIERS)
    assert len({item.key for item in SETTINGS}) == len(SETTINGS) == 31
    assert sum(not item.file for item in SETTINGS) == 9


@pytest.mark.parametrize("topic", [[], ["all"]])
def test_the_flag_shows_the_full_help_before_or_after_help(
    topic: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    for argv in (["--developer", "--help", *topic], ["--help", *topic, "--developer"]):
        assert not _is_short(_help(*argv, capsys=capsys)), argv


def test_a_command_word_after_h_shows_that_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _help("-h", "serve", capsys=capsys).startswith("usage: melo serve ")


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
    assert not _is_short(_help("--help", capsys=capsys))


@pytest.mark.parametrize("text", ["", '[ui]\nmode = "default"\n', "[ui]\n"])
def test_default_or_absent_mode_shows_default_help(
    text: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(tmp_path, monkeypatch, text)
    assert _is_short(_help("--help", capsys=capsys))


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
    assert _is_short(out)
    assert err.startswith("[ui] mode=") and "default help" in err


def test_an_unreadable_config_still_shows_default_help(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(tmp_path, monkeypatch, "[ui\nmode = ")
    assert _is_short(_help("--help", capsys=capsys))


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


# -- --version and usage errors ------------------------------------


def test_version_prints_the_banner_version_and_exits_0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(tmp_path, monkeypatch, "")
    with mock.patch(
        "localmelo.support.sessions.display.display_version",
        return_value="0.1.0 (baab21e)",
    ):
        assert _main("--version").code == 0
    out, err = capsys.readouterr()
    assert (out, err) == ("melo 0.1.0 (baab21e)\n", "")


def test_a_usage_error_in_default_mode_prints_a_short_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(tmp_path, monkeypatch, "")
    assert _main("--no-such-flag").code == 2
    assert capsys.readouterr().err == (
        "usage: melo [options] [--] [task ...]\n"
        "Run melo --help for options.\n"
        "melo: error: unrecognized arguments: --no-such-flag\n"
    )


@pytest.mark.parametrize("developer", ["flag", "config"])
def test_a_usage_error_in_developer_mode_keeps_the_long_usage(
    developer: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        '[ui]\nmode = "developer"\n' if developer == "config" else "",
    )
    argv = ["--no-such-flag"] + (["--developer"] if developer == "flag" else [])
    assert _main(*argv).code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: melo [-h]") and "--reflection-gate" in err
    assert "Run melo --help" not in err


def test_the_worker_help_says_a_failed_check_exits_4() -> None:
    """The worker help tells a launcher's author when Check: lines run and what a failure does."""
    text = " ".join(command_words.help_text("worker").split())
    assert "Check: lines run when the model is done, and a failed check exits 4" in text
