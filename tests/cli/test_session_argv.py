"""U2b argv compatibility against recorded historical parsers, not itself.

The golden corpus was generated from AST-extracted parser/check functions in
``git show d232b9e:__main__.py`` (before the UI), plus ``a4599e0`` for R2a.
Only shared namespace defaults are factored out; every case's result is recorded.
The three explicit R4 overrides were captured from upstream cb250c6 before U2b,
not from the new parser: compress became valid, and two replay errors name it.
R2b intentionally retires the no-query reasoning-field refusal: that one case
now checks UI dispatch and its complete namespace; the original golden remains.
Dispatch is mocked: no model, MCP process, session store or gateway is started.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

import localmelo.__main__ as cli

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "direct_argv.json").read_text(encoding="utf-8")
)
SOURCES = GOLDEN["sources"]
R2B_OVERRIDES = {
    GOLDEN["reasoning_baseline"]: {
        "r2a-16": {"result": "ui", "namespace": {"reasoning_field": "auto"}}
    }
}
CASES = [
    pytest.param(
        source,
        {
            **case,
            **GOLDEN["upstream_overrides"].get(source, {}).get(case["id"], {}),
            **R2B_OVERRIDES.get(source, {}).get(case["id"], {}),
        },
        id=f"{source[:7]}-{case['id']}",
    )
    for source, record in SOURCES.items()
    for case in record["cases"]
]
UI_FIELDS = {"yolo", "continue_session", "resume"}
R2A_DEFAULTS = {
    key: SOURCES[GOLDEN["reasoning_baseline"]]["namespace_defaults"][key]
    for key in ("reasoning", "reasoning_field", "replay_turns")
}
R4_DEFAULTS = GOLDEN["upstream_namespace_defaults"]  # recorded from upstream #85
# R5 (#94) added seven direct-mode options, all None unless given.
R5_DEFAULTS = dict.fromkeys(
    (
        "retriever",
        "recall_at",
        "recall_k",
        "recall_kinds",
        "recall_placement",
        "recall_min_score",
        "retrieval_extra_body",
    )
)
# P5 moved --workspace into direct mode: the direct parser carries it now.
P5_DEFAULTS = {"workspace": None}
# R6-lite added two direct-mode options.
R6_DEFAULTS = {"admit_on_outcome": None, "keep_decision_text": False}
# U3 PR 4 added --allow-unconfigured for --serve.
U3_PR4_DEFAULTS = {"allow_unconfigured": False}
WORKER_DEFAULTS = dict.fromkeys(
    ("save_session", "parent_run", "import_session", "title", "brief", "inbox")
)


@pytest.fixture
def dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[SimpleNamespace]:
    monkeypatch.setenv("HOME", str(tmp_path))
    with (
        mock.patch.object(cli, "_register_backends"),
        mock.patch.object(cli, "_build_direct_mode_agent") as build,
        mock.patch.object(cli, "_direct_tool_providers", return_value=[]) as providers,
        mock.patch.object(cli, "_run_direct", return_value=0) as run,
        mock.patch("localmelo.support.cli.run_interactive", return_value=0) as ui,
        mock.patch("localmelo.support.cli.ui_terminal", return_value=False),
        mock.patch("localmelo.support.config.load") as load,
        mock.patch.object(cli, "_start_gateway") as gateway,
        mock.patch("localmelo.support.gateway.daemon.status", return_value={}),
    ):
        load.return_value.is_configured = True
        yield SimpleNamespace(
            build=build, providers=providers, run=run, ui=ui, gateway=gateway
        )


@pytest.mark.parametrize(("source", "case"), CASES)
def test_historical_dispatch(
    source: str,
    case: dict[str, Any],
    dispatch: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["melo", *case["argv"]])
    if case["result"] == "error":
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 2
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        error = (
            GOLDEN["error_variants"]
            .get(version, {})
            .get(source, {})
            .get(case["id"], case["error"])
        )
        assert capsys.readouterr().err.endswith(f"melo: error: {error}\n")
        dispatch.build.assert_not_called()
        dispatch.run.assert_not_called()
        dispatch.ui.assert_not_called()
        dispatch.gateway.assert_not_called()
        return

    cli.main()
    if case["result"] == "ui":
        dispatch.ui.assert_called_once()
        assert vars(dispatch.ui.call_args.args[0]) == {
            **SOURCES[source]["namespace_defaults"],
            **R2A_DEFAULTS,
            **R4_DEFAULTS,
            **R5_DEFAULTS,
            **P5_DEFAULTS,
            **R6_DEFAULTS,
            **U3_PR4_DEFAULTS,
            **WORKER_DEFAULTS,
            "continue_session": False,
            "resume": None,
            **case["namespace"],
        }
        dispatch.build.assert_not_called()
        dispatch.providers.assert_not_called()
        dispatch.run.assert_not_called()
        dispatch.gateway.assert_not_called()
        assert capsys.readouterr() == ("", "")
        return
    if case["result"] != "direct":
        dispatch.build.assert_not_called()
        dispatch.run.assert_not_called()
        return

    dispatch.build.assert_called_once()
    args = dispatch.build.call_args.args[0]
    expected = {
        **SOURCES[source]["namespace_defaults"],
        **R2A_DEFAULTS,
        **R4_DEFAULTS,
        **R5_DEFAULTS,
        **P5_DEFAULTS,
        **R6_DEFAULTS,
        **U3_PR4_DEFAULTS,
        **WORKER_DEFAULTS,
        **case["namespace"],
    }
    for field in UI_FIELDS:
        expected.pop(field, None)
    assert vars(args) == expected
    assert not UI_FIELDS.intersection(vars(args))
    assert set(dispatch.build.call_args.kwargs) == {"options", "tool_providers"}
    dispatch.run.assert_called_once_with(
        dispatch.build.return_value,
        " ".join(expected["query"]),
        expected["output_format"] or "text",
        [],
        progress=expected["progress"],
        routine=expected["routine"],
    )
    dispatch.ui.assert_not_called()
    dispatch.gateway.assert_not_called()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    ("argv", "continuing", "resume"),
    [
        ([], False, None),
        (["-c"], True, None),
        (["--continue"], True, None),
        (["-r"], False, ""),
        (["--resume"], False, ""),
        (["-r", "fix tests"], False, "fix tests"),
        (["--resume", "session-id"], False, "session-id"),
        (["--resume="], False, ""),
        (["-r", "--chat-model", "M"], False, ""),
        (["--workspace", "/tmp", "--yolo", "-c"], True, None),
    ],
)
def test_session_flags_without_a_query_reach_only_the_ui(
    argv: list[str],
    continuing: bool,
    resume: str | None,
    dispatch: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["melo", *argv])
    cli.main()
    dispatch.ui.assert_called_once()
    args = dispatch.ui.call_args.args[0]
    assert args.query == []
    assert args.continue_session is continuing
    assert args.resume == resume
    dispatch.build.assert_not_called()
    dispatch.run.assert_not_called()
    dispatch.gateway.assert_not_called()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["-c", "-r"], "-c and -r cannot be combined"),
        (["-r", "-c"], "-c and -r cannot be combined"),
        (["--continue", "--resume", "title"], "-c and -r cannot be combined"),
        (["--mcp-only", "-c", "-r"], "--mcp-only requires --mcp-config"),
        (["--serve", "-c"], "UI: -c"),
        (["--serve", "-r"], "UI: -r"),
        (["--daemon", "status", "--resume"], "UI: -r"),
        (["--daemon", "install", "--continue"], "UI: -c"),
        (["-c", "task"], "UI: -c"),
        (["--continue", ""], "UI: -c"),
        (["-r", "title", "--", "task"], "UI: -r"),
        (["-r", "title", ""], "UI: -r"),
        (["--yolo", "task"], "UI: --yolo"),
        (
            ["--serve", "--workspace", "/tmp", "--yolo", "-c", "-r"],
            "UI: --yolo, -c, -r",
        ),
        (["--serve", "--workspace", "/tmp"], "direct: --workspace"),
        (["--workspace", "/tmp", ""], "direct: --workspace"),
    ],
)
def test_invalid_session_flags_fail_before_dispatch(
    argv: list[str],
    message: str,
    dispatch: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if message.startswith("UI: "):
        message = (
            "only for the interactive UI (melo without a query, --serve or --daemon): "
            + message.removeprefix("UI: ")
        )
    elif message.startswith("direct: "):
        message = (
            "only for direct mode (a query without --serve): "
            + message.removeprefix("direct: ")
        )
    monkeypatch.setattr(sys, "argv", ["melo", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert capsys.readouterr().err.endswith(f"melo: error: {message}\n")
    dispatch.build.assert_not_called()
    dispatch.run.assert_not_called()
    dispatch.ui.assert_not_called()
    dispatch.gateway.assert_not_called()


def test_c_is_ambiguous_only_without_a_legacy_query(
    dispatch: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["melo", "--c", "M"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "ambiguous option: --c could match --chat-model, --continue" in (
        capsys.readouterr().err
    )
    dispatch.ui.assert_not_called()
    dispatch.build.assert_not_called()


def test_unambiguous_chat_abbreviation_still_opens_the_ui(
    dispatch: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["melo", "--chat-mode", "M"])
    cli.main()
    assert dispatch.ui.call_args.args[0].chat_model == "M"
    dispatch.build.assert_not_called()


def test_the_legacy_probe_has_no_ui_options_help_or_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser(direct_only=True)
    args, unknown = parser.parse_known_args(["--help", "-c", "-r", "task"])
    assert unknown == ["--help", "-c", "-r"]
    assert not UI_FIELDS.intersection(vars(args))
    assert not any(action.dest == "help" for action in parser._actions)
    with pytest.raises(argparse.ArgumentError, match="must be at least 1"):
        parser.parse_known_args(["--max-steps", "0"])
    assert capsys.readouterr() == ("", "")


def test_the_default_parser_keeps_session_help() -> None:
    text = " ".join(cli._build_parser().format_help().split())
    assert "-c, --continue" in text
    assert "--resume [ID|TITLE]" in text
    resume = next(
        action for action in cli._build_parser()._actions if action.dest == "resume"
    )
    assert resume.option_strings == ["-r", "--resume"]
    assert (resume.nargs, resume.const, resume.default) == ("?", "", None)


@pytest.mark.parametrize(
    "flag,value,expected",
    [
        ("--max-steps", "0", "argument --max-steps: must be at least 1, got 0"),
        ("--attempt-policy", "invalid", "argument --attempt-policy: invalid choice:"),
        ("--port", "invalid", "argument --port: invalid int value:"),
    ],
)
def test_legacy_value_error_is_not_shadowed_by_full_parser_ambiguity(
    dispatch: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
    expected: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["melo", "--c", "M", flag, value, "task"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert expected in error
    assert "ambiguous option" not in error
    dispatch.run.assert_not_called()
    dispatch.ui.assert_not_called()
