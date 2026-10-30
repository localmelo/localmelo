"""What a turn's events print, and what the live area and the banner show."""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.text import Text

from localmelo.melo.schema import RunResult, ToolInfo, ToolTier
from localmelo.support.cli.output import RecordingOut, TerminalOut
from localmelo.support.cli.render import LiveState, TurnRenderer, live_rows
from localmelo.support.cli.text import cells
from localmelo.support.cli.theme import ASCII, UNICODE, Symbols
from localmelo.support.cli.views import BackendInfo, banner
from localmelo.support.sessions.agents import SESSION_TOOL_TIMEOUT_S
from localmelo.support.sessions.store import StoredEvent

URL = "http://127.0.0.1:18431/v1"


def _backend(**fields: Any) -> BackendInfo:
    values: dict[str, Any] = {
        "key": "openai-compatible",
        "label": "OpenAI-compatible",
        "model": "stub-model",
        "base_url": URL,
    }
    values.update(fields)
    return BackendInfo(**values)


class Turn:
    """A renderer on a recording, fed events by type and data."""

    def __init__(
        self,
        width: int = 80,
        symbols: Symbols = UNICODE,
        snapshots: Any = None,
        **backend: Any,
    ) -> None:
        self.out = RecordingOut(width)
        self.now = [100.0]
        self.renderer = TurnRenderer(
            self.out,
            symbols=symbols,
            backend=lambda: _backend(**backend),
            snapshots=snapshots,
            clock=lambda: self.now[0],
        )
        self.renderer.begin_turn("What is in this folder?")

    def event(self, type_: str, **data: Any) -> Turn:
        self.renderer.on_event(
            StoredEvent(
                id=1, session_id="s", turn=1, event={"type": type_, "data": data}
            )
        )
        return self

    def tool(
        self, name: str, preview: str, call_id: str = "c1", **finished: Any
    ) -> Turn:
        self.event(
            "tool.started",
            step=1,
            call_id=call_id,
            name=name,
            arguments={},
            preview=preview,
        )
        data: dict[str, Any] = {
            "step": 1,
            "call_id": call_id,
            "name": name,
            "status": "success",
            "error_category": "none",
            "output": "",
            "output_chars": 0,
            "error": "",
            "duration_s": 0.1,
            "images": [],
            "approval": None,
            "exit_code": None,
        }
        data.update(finished)
        return self.event("tool.finished", **data)

    def finish(self, status: str = "completed", **result: Any) -> Turn:
        values = {
            "task_id": "t",
            "status": status,
            "result": "",
            "error": "",
            "n_tool": 0,
            "duration_s": 14.2,
        }
        values.update(result)
        return self.event("run.finished", result=values)

    def text(self) -> str:
        return self.out.text()


def test_the_echo_shows_six_lines_of_a_long_message() -> None:
    out = RecordingOut(80)
    renderer = TurnRenderer(out, symbols=UNICODE, backend=_backend)
    renderer.begin_turn("\n".join(f"line {n}" for n in range(1, 9)))
    text = out.text()
    assert "› line 1\n  line 2\n" in text
    assert "line 6\n  … +2 lines" in text and "line 7" not in text


def test_a_tool_block_shows_three_lines_then_the_rest_count() -> None:
    output = "\n".join(["app.py", "notes.txt", "tests", "todo.md", "zzz"])
    text = (
        Turn()
        .tool(
            "shell_exec",
            "ls -1",
            output=output,
            approval={
                "request_id": "r",
                "allowed": True,
                "reason": "",
                "remember": True,
                "origin": "user",
            },
        )
        .text()
    )
    assert "✓ shell_exec  ls -1 · 0.1s · allowed for the session\n" in text
    assert (
        "  └ app.py\n    notes.txt\n    tests\n    … +2 lines (/expand 1.1)\n" in text
    )


def test_a_single_line_left_prints_instead_of_the_marker() -> None:
    text = Turn().tool("shell_exec", "ls", output="a\nb\nc\nd").text()
    assert "    c\n    d\n" in text and "+1 lines" not in text


def test_a_nonzero_exit_of_a_command_that_ran_is_a_tag_not_a_failure() -> None:
    text = (
        Turn()
        .tool(
            "shell_exec",
            "pytest -q tests",
            output="[exit code 1]\n....F\nFAILED tests/test_app.py",
            exit_code=1,
            duration_s=4.1,
        )
        .text()
    )
    assert "✓ shell_exec  pytest -q tests · 4.1s · exit 1\n" in text
    assert "  └ [exit code 1]\n" in text


def test_a_failed_call_shows_its_error_then_its_output() -> None:
    text = (
        Turn()
        .tool(
            "python_exec",
            "import app",
            status="error",
            error="Python exited with code 1:",
            output="Traceback (most recent call last):",
            duration_s=0.2,
        )
        .text()
    )
    assert "✗ python_exec  import app · 0.2s\n" in text
    assert "  └ Python exited with code 1:\n    Traceback" in text


def test_an_empty_output_says_so() -> None:
    assert "  └ (no output)\n" in Turn().tool("shell_exec", "true").text()


@pytest.mark.parametrize(
    ("approval", "tag"),
    [
        ({"allowed": True, "origin": "user", "remember": False}, "allowed"),
        (
            {"allowed": True, "origin": "session", "remember": False},
            "allowed by a session grant",
        ),
        ({"allowed": True, "origin": "policy", "remember": False}, "auto-allowed"),
        (
            {"allowed": False, "origin": "timeout", "remember": False},
            "not answered in time",
        ),
        (
            {"allowed": False, "origin": "error", "remember": False, "reason": "boom"},
            "approval failed",
        ),
    ],
)
def test_approval_tags(approval: dict[str, Any], tag: str) -> None:
    status = "success" if approval["allowed"] else "blocked"
    text = (
        Turn()
        .tool("file_write", "todo.md (68 chars)", status=status, approval=approval)
        .text()
    )
    assert "file_write  todo.md (68 chars)" in text
    header = next(line for line in text.split("\n") if "file_write  todo.md" in line)
    assert header.endswith(tag)


def test_a_denial_shows_its_reason() -> None:
    text = (
        Turn()
        .tool(
            "file_write",
            "todo.md (68 chars)",
            status="blocked",
            error_category="approval_denied",
            approval={
                "allowed": False,
                "origin": "user",
                "reason": "keep the\n item first",
            },
        )
        .text()
    )
    assert "⊘ file_write  todo.md (68 chars) · denied: keep the item first\n" in text


def test_a_blocked_call_without_an_approval_names_the_block() -> None:
    text = (
        Turn()
        .tool(
            "file_read",
            "/etc/hosts",
            status="blocked",
            error="Path '/etc/hosts' is outside allowed workspace\nmore",
        )
        .text()
    )
    line = next(row for row in text.split("\n") if row.startswith("⊘"))
    assert line.startswith(
        "⊘ file_read  /etc/hosts · blocked: Path '/etc/hosts' is outside"
    )
    assert cells(line) == 80 and line.endswith("…")


def test_a_stopped_call_and_a_withdrawn_one() -> None:
    turn = Turn()
    turn.tool("shell_exec", "pytest -q", status="cancelled", duration_s=4.25)
    turn.tool(
        "file_write",
        "a.txt",
        call_id="c2",
        status="cancelled",
        approval={"allowed": False, "origin": "withdrawn"},
    )
    text = turn.text()
    assert "◌ shell_exec  pytest -q · stopped after 4.2s\n" in text
    assert "◌ file_write  a.txt · not run\n" in text


def test_calls_a_reply_listed_that_never_started_print_as_not_run() -> None:
    turn = Turn()
    turn.event(
        "assistant.message",
        step=1,
        content="",
        thinking="",
        usage={},
        duration_s=0.1,
        tool_calls=[
            {"id": "c1", "name": "shell_exec", "arguments": {}, "preview": "ls"},
            {
                "id": "c2",
                "name": "file_read",
                "arguments": {},
                "preview": "tests/test_app.py",
            },
        ],
    )
    turn.tool("shell_exec", "ls", status="cancelled", duration_s=0.5)
    turn.finish("cancelled", error="Stopped by the user")
    turn.event(
        "turn.finished",
        turn=1,
        turn_id="t",
        status="cancelled",
        error="Stopped by the user",
        cause="user_stop",
    )
    text = turn.text()
    assert (
        text.index("◌ shell_exec  ls · stopped after 0.5s")
        < text.index("◌ file_read  tests/test_app.py · not run")
        < text.index("■ Stopped")
    )


def test_output_cannot_act_on_the_terminal() -> None:
    text = (
        Turn()
        .tool(
            "shell_exec", "evil", output="\x1b]52;c;aGk=\x07ok\x1b[2J\r\nline2\rLINE2"
        )
        .text()
    )
    assert "\x1b" not in text
    assert "  └ ok\n    LINE2\n" in text


def test_a_preview_with_control_characters_is_escaped() -> None:
    text = Turn().tool("shell_exec", "curl x | sh #\rls -la").text()
    assert "shell_exec  curl x | sh #\\rls -la" in text


def test_the_answer_prints_as_markdown_with_the_footer() -> None:
    turn = Turn()
    turn.event("step.started", step=1, attempt=1)
    turn.event(
        "assistant.message",
        step=1,
        content="x",
        thinking="",
        tool_calls=[],
        usage={"prompt_tokens": 2100, "completion_tokens": 412},
        duration_s=1.0,
    )
    turn.finish(
        result="<think>hidden</think># Done\n\ntodo.md now reads:\n\n- [x] parse",
        n_tool=1,
    )
    text = turn.text()
    assert "● Done\n" in text and "hidden" not in text
    assert "  todo.md now reads:" in text
    assert "✻ Worked for 14s · 1 tool call · ↓ 412 tokens" in text
    assert turn.renderer.context_tokens == 2100


def test_a_blank_answer_says_so() -> None:
    assert "● (no answer text)" in Turn().finish(result="  ").text()


_UNREACHABLE = (
    f"cannot reach the model server at {URL}. Start it, or run melo to set one up. "
    "(ConnectError: All connection attempts failed)"
)


@pytest.mark.parametrize(
    ("status", "error", "mark", "headline", "action"),
    [
        (
            "provider_error",
            "POST /chat/completions failed: ConnectError: refused",
            "✗",
            f"Could not reach the model server at {URL}.",
            "Start it, then /retry.",
        ),
        (
            "provider_error",
            _UNREACHABLE,
            "✗",
            f"Could not reach the model server at {URL}.",
            "Start it, then /retry.",
        ),
        (
            "provider_error",
            "POST /chat/completions failed: ReadTimeout: x",
            "✗",
            "The model server did not answer in time.",
            "It may still be loading the model. /retry tries again.",
        ),
        (
            "provider_error",
            "POST /chat/completions failed with HTTP 404: no",
            "✗",
            "The model server does not know the model 'stub-model'.",
            "Check the name (/model lists the models it has).",
        ),
        (
            "provider_error",
            "POST /chat/completions failed with HTTP 401: key",
            "✗",
            "The model server requires an API key.",
            "localmelo sends none to local servers. Turn the key off in the server "
            '(oMLX: "Allow inference without API key").',
        ),
        (
            "provider_error",
            "POST /chat/completions failed with HTTP 400: tools",
            "✗",
            "The model server refused the request: HTTP 400: tools.",
            "If it says the model does not support tools, choose one that does.",
        ),
        (
            "provider_error",
            "POST /chat/completions failed with HTTP 500: boom",
            "✗",
            "Could not get an answer from the model server: HTTP 500: boom.",
            f"/status checks the model server at {URL}; /retry tries again.",
        ),
        (
            "budget_exhausted",
            "Max steps reached",
            "⚠",
            "Ran out of steps.",
            "/retry tries again; for more steps, quit and run melo -c --max-steps 60.",
        ),
        (
            "stuck",
            "repeated",
            "⚠",
            "Stopped: the same calls kept repeating.",
            "Rephrase the request, or /retry.",
        ),
        (
            "failed",
            "empty replies",
            "⚠",
            "The model returned only empty replies.",
            "/retry, or try another model.",
        ),
        (
            "failed",
            "post_plan refused",
            "⚠",
            "A check stopped the run: post_plan refused.",
            "/retry, or try another model.",
        ),
        (
            "gate_stopped",
            "Trying the same command again would not help",
            "⚠",
            "Stopped after reflection: Trying the same command again would not help.",
            "melo judged that another attempt would not help. "
            "Rephrase the request, or /retry.",
        ),
        (
            "context_budget_exceeded",
            "1.2k tokens over",
            "⚠",
            "Context budget exceeded: 1.2k tokens over.",
            "Reduce the task or increase the configured context window; then /retry.",
        ),
        (
            "setup_error",
            "MCP server 'notes' exited (code 1)",
            "✗",
            "Tools could not start: MCP server 'notes' exited (code 1).",
            "/mcp shows why.",
        ),
        (
            "crashed",
            "KeyError: x",
            "✗",
            "Something went wrong inside localmelo: KeyError: x.",
            "This is a bug; the log is ~/.cache/localmelo/logs/cli.log.",
        ),
    ],
)
def test_a_run_that_did_not_complete_says_what_happened_and_what_to_do(
    status: str, error: str, mark: str, headline: str, action: str
) -> None:
    text = Turn().finish(status, error=error, result="partial").text()
    assert "● Best effort: partial" in text
    assert f"{mark} {headline} {action} ✻ Stopped after 14s" in " ".join(text.split())
    # Direct mode's machine line and the status word are not for this screen.
    assert "melo: run ended" not in text and status not in text
    assert "Worked for" not in text


def test_a_server_that_cannot_be_reached_reads_in_two_rows_that_are_not_cut() -> None:
    turn = Turn()
    turn.event(
        "run.started", query="q", max_steps=30, workspace_root="/w", approval=True
    )
    for delay in (1.0, 4.0):
        turn.event(
            "notice",
            kind="retry",
            message=f"Model request failed; retrying in {delay:g} s: {_UNREACHABLE}",
            detail={"request_attempt": 1, "delay_s": delay, "status_code": None},
            step=1,
            attempt=1,
        )
    # Only the first retry prints; the live row counts down the second.
    assert turn.renderer.live is not None and turn.renderer.live.retry_until == 104.0
    turn.finish("provider_error", error=_UNREACHABLE)
    assert turn.text().split("\n")[2:] == [
        f"↻ Could not reach {URL}; trying again in 1 s",
        f"✗ Could not reach the model server at {URL}.",
        "  Start it, then /retry.",
        "✻ Stopped after 14s",
        "",
    ]


def test_the_retry_line_gives_up_its_wait_before_its_url() -> None:
    rows = {}
    for width in (80, 60, 30):
        turn = Turn(width)
        turn.event(
            "notice",
            kind="retry",
            message=f"Model request failed; retrying in 1 s: {_UNREACHABLE}",
            detail={"request_attempt": 1, "delay_s": 1.0},
            step=1,
            attempt=1,
        )
        [rows[width]] = [row for row in turn.text().split("\n") if "↻" in row]
    assert rows[80] == f"↻ Could not reach {URL}; trying again in 1 s"
    assert rows[60] == f"↻ Could not reach {URL}"  # the wait goes, the URL stays
    assert cells(rows[30]) <= 30


def test_a_run_that_ran_out_of_steps_names_the_limit() -> None:
    turn = Turn()
    turn.event(
        "run.started", query="q", max_steps=30, workspace_root="/w", approval=True
    )
    turn.finish("budget_exhausted", error="Max steps reached", n_tool=29)
    assert (
        "⚠ Ran out of steps (30).\n  /retry tries again; for more steps," in turn.text()
    )
    assert "✻ Stopped after 14s · 29 tool calls\n" in turn.text()


def test_ollama_and_cloud_hints() -> None:
    text = (
        Turn(key="ollama", label="Ollama")
        .finish("provider_error", error="POST /api/chat failed with HTTP 404: x")
        .text()
    )
    assert (
        "Check the name (/model lists the models it has). ollama pull stub-model fetches it."
        in " ".join(text.split())
    )
    text = (
        Turn(key="openai", label="OpenAI", base_url="", key_env="OPENAI_API_KEY")
        .finish(
            "provider_error", error="POST /chat/completions failed with HTTP 401: x"
        )
        .text()
    )
    assert "The server refused the key in OPENAI_API_KEY." in text
    text = (
        Turn(key="openai", label="OpenAI", base_url="")
        .finish("provider_error", error=_UNREACHABLE)
        .text()
    )
    assert "✗ Could not reach OpenAI.\n  Check the network, then /retry.\n" in text


@pytest.mark.parametrize(
    ("cause", "line"), [("user_stop", "■ Stopped"), ("shutdown", "■ Interrupted")]
)
def test_a_stopped_turn(cause: str, line: str) -> None:
    turn = Turn()
    turn.renderer.stopped_by(cause)
    assert turn.renderer.live is not None and turn.renderer.live.activity == "Stopping"
    turn.finish("cancelled", error="Stopped by the user")
    # The cause is turn.finished's, which the runner forwards after the run.
    turn.event(
        "turn.finished",
        turn=1,
        turn_id="t",
        status="cancelled",
        error="Stopped by the user",
        cause=cause,
    )
    assert line in turn.text() and "Worked for" not in turn.text()


def test_a_turn_stopped_before_its_run_prints_the_stopped_line_at_its_end() -> None:
    turn = Turn()
    turn.renderer.end_turn(
        RunResult(task_id="", status="cancelled"), None, cause="user_stop"
    )
    assert "■ Stopped" in turn.text()
    assert turn.renderer.live is None


def test_a_turn_that_could_not_run_says_why() -> None:
    turn = Turn()
    turn.renderer.end_turn(None, OSError("disk full"))
    assert "✗ The turn could not run: OSError: disk full" in turn.text()


def test_a_yolo_turn_says_so() -> None:
    text = (
        Turn()
        .event(
            "run.started", query="q", max_steps=30, workspace_root="/w", approval=False
        )
        .text()
    )
    assert "  No approvals for this turn (--yolo)." in text


def test_the_live_state_follows_the_events() -> None:
    turn = Turn()
    live = turn.renderer.live
    assert live is not None and live.activity == "Starting"
    turn.event("step.started", step=2, attempt=2)
    assert (live.activity, live.step, live.attempt) == ("Thinking", 2, 2)
    turn.event(
        "tool.started",
        step=2,
        call_id="c",
        name="shell_exec",
        arguments={},
        preview="pytest",
    )
    assert (live.activity, live.tool, live.tool_limit_s) == (
        "Running",
        "shell_exec",
        600.0,
    )
    turn.event("reflection.started", attempt=1, failure_type="x")
    assert live.activity == "Reflecting"
    turn.finish()
    assert (live.activity, live.finished) == ("Wrapping up", True)
    assert turn.renderer.finished


def test_the_spinner_row_never_shows_a_step_budget() -> None:
    live = LiveState(
        activity="Thinking", started=94.0, step=4, attempt=1, output_tokens=412
    )
    rows = live_rows(live, 100.0, 80, UNICODE, frame=0)
    assert rows == [
        [("class:accent", "· Thinking… · step 4 · 6s · ↓ 412 tokens · esc to stop")]
    ]
    live.attempt = 2
    text = live_rows(live, 100.0, 80, UNICODE, frame=0)[0][0][1]
    assert "step 4 · attempt 2 · 6s" in text and not re.search(r"step 4/\d", text)


def test_the_spinner_row_drops_the_hint_then_the_tokens_when_narrow() -> None:
    live = LiveState(activity="Thinking", started=94.0, step=4, output_tokens=412)
    assert (
        live_rows(live, 100.0, 50, UNICODE, frame=0)[0][0][1]
        == "· Thinking… · step 4 · 6s · ↓ 412 tokens"
    )
    assert (
        live_rows(live, 100.0, 34, UNICODE, frame=0)[0][0][1]
        == "· Thinking… · step 4 · 6s"
    )
    for width in range(10, 80):
        for row in live_rows(live, 100.0, width, UNICODE, frame=0):
            assert cells(row[0][1]) <= width


def test_a_running_tool_shows_its_clock_and_limit() -> None:
    live = LiveState(
        activity="Running",
        started=60.0,
        step=4,
        tool="shell_exec",
        preview="pytest -q tests",
        tool_started=75.0,
        tool_limit_s=600.0,
    )
    rows = live_rows(live, 100.0, 80, UNICODE, frame=1)
    assert rows[1] == [
        ("class:muted", "  └ shell_exec  pytest -q tests · 0:25 of 10:00")
    ]


def test_waiting_and_wrapping_up() -> None:
    live = LiveState(activity="Thinking", started=97.0, step=1, waiting=True)
    assert live_rows(live, 100.0, 80, UNICODE)[0][0][1].startswith(
        "⚠ Waiting for your approval · step 1 · 3s"
    )
    live.finished = True
    assert live_rows(live, 100.0, 80, UNICODE) == [[("class:muted", "Wrapping up…")]]


def test_the_ascii_symbols() -> None:
    turn = Turn(symbols=ASCII)
    turn.tool("shell_exec", "ls", output="a")
    turn.finish(result="done")
    text = turn.text()
    assert "> What is in this folder?" in text
    assert "+ shell_exec  ls - 0.1s\n  | a\n" in text
    assert "* done" in text


# ── The banner ──


def _tools() -> list[ToolInfo]:
    rows: tuple[tuple[str, bool, ToolTier], ...] = (
        ("file_read", False, "read"),
        ("file_write", True, "write"),
        ("shell_exec", True, "exec"),
        ("python_exec", True, "exec"),
    )
    return [
        ToolInfo(
            name=name,
            description="",
            parameters={},
            source="builtin",
            requires_approval=asks,
            tier=tier,
        )
        for name, asks, tier in rows
    ]


def _banner(
    width: int = 80,
    workspace: str = "/tmp/demo-project",
    *,
    developer: bool = True,
    version: str = "0.1.0",
    yolo: bool = False,
    symbols: Symbols = UNICODE,
    **backend: Any,
) -> str:
    out = RecordingOut(width)
    out.print(
        banner(
            version=version,
            backend=_backend(**backend),
            workspace=workspace,
            tools=_tools(),
            memory=True,
            yolo=yolo,
            width=width,
            symbols=symbols,
            developer=developer,
        )
    )
    return out.text()


def _welcome(width: int = 80, **kwargs: Any) -> list[str]:
    """The rows of the default banner, without the blank rows around them."""
    text = _banner(width, developer=False, **kwargs)
    assert text.startswith("\n") and text.endswith("\n\n")
    return text.strip("\n").split("\n")


def test_the_banner_box() -> None:
    lines = _banner(source="config").split("\n")
    assert lines[0].startswith("╭─ melo 0.1.0 ─") and lines[0].endswith("╮")
    assert lines[1].startswith("│ model      stub-model · OpenAI-compatible · " + URL)
    assert lines[1].endswith(" │")
    assert "│ tools      4 built-in (3 ask first)" in lines[3]
    assert (
        "│ memory     no long-term memory · --embedding-base-url adds one" in lines[4]
    )
    assert "│ session    new" in lines[5]
    assert all(cells(line) == 80 for line in lines[:7])
    assert lines[7] == "  / commands · esc stop · ctrl+d quit"


def test_the_welcome_is_three_rows_that_start_the_conversation() -> None:
    assert _welcome() == [
        "melo 0.1.0 · stub-model (/model to change)",
        "/tmp/demo-project",
        "Try: What is this project? · / commands · ctrl+d quit",
    ]
    assert _welcome(symbols=ASCII)[2].endswith("? - / commands - ctrl+d quit")


@pytest.mark.parametrize("width", [80, 60, 44])
def test_the_welcome_keeps_three_rows_and_wraps_none(width: int) -> None:
    rows = _welcome(width, version="0.1.0 (6a400ec)")
    assert len(rows) == 3 and all(cells(row) <= width for row in rows), rows
    assert "(/model to change)" in rows[0] and "ctrl+d quit" in rows[2]
    assert rows[2].startswith("Try: What is this project?")


def test_the_welcome_names_the_server_only_for_a_model_that_is_not_saved() -> None:
    for source in ("config", "saved"):
        assert len(_welcome(source=source)) == 3
    wide = _welcome(120, source="--base-url", listed=3)
    assert wide[2] == (
        f"{URL} · from --base-url (not saved) · the loaded one of 3; /model lists them"
    )
    assert _welcome(80, source="--base-url", listed=3)[2] == (
        f"{URL} · from --base-url (not saved)"
    )
    assert _welcome(44, source="unsaved")[2] == URL  # the longer notes give way


def test_the_welcome_keeps_the_warnings_and_cleans_the_model_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    rows = _welcome(
        workspace=os.path.realpath(tmp_path), yolo=True, model=f"qwen{OSC_TITLE}3"
    )
    assert rows[0] == "melo 0.1.0 · qwen3 (/model to change)" and rows[1] == "~"
    assert rows[3:] == [
        "⚠ YOLO: tools run without asking.",
        "⚠ The workspace is ~: file tools can reach every file in it.",
    ]


def test_the_banner_names_the_configured_embedding_model() -> None:
    text = _banner(key="vllm", label="vLLM", embedding="vllm · Qwen3-Embedding-0.6B")
    assert "│ memory     long-term memory · vllm · Qwen3-Embedding-0.6B" in text


def test_the_banner_without_an_embedding_names_the_wizard() -> None:
    text = _banner(key="vllm", label="vLLM")
    assert "│ memory     no long-term memory · melo --reconfigure adds one" in text


def test_the_banner_after_the_first_run_does_not_repeat_how_to_add_memory() -> None:
    text = _banner(key="vllm", label="vLLM", memory_told=True)
    assert "│ memory     no long-term memory " in text
    assert "adds one" not in text


def test_the_banner_cleans_an_embedding_model_id() -> None:
    text = _banner(key="vllm", label="vLLM", embedding="vllm · bad\x1b[31m")
    assert "\x1b" not in text and "31m" not in text and "bad" in text


def test_a_boxed_banner_wraps_the_memory_text_inside_the_box() -> None:
    text = _banner(
        width=60,
        key="vllm",
        label="vLLM",
        embedding="vllm · Qwen3-Embedding-0.6B",
    )
    lines = text.split("\n")
    assert lines[0].startswith("╭")
    boxed = [line for line in lines if line.startswith("│")]
    assert boxed and all(cells(line) == 60 for line in boxed)
    box_text = "\n".join(boxed)
    assert "long-term memory · vllm ·" in box_text
    assert "Qwen3-Embedding-0.6B" in box_text


@pytest.mark.parametrize(
    "backend",
    [
        {},
        {"key": "vllm", "label": "vLLM", "embedding": "vllm · Qwen3-Embedding-0.6B"},
    ],
    ids=["no-embedding", "configured"],
)
def test_a_narrow_banner_keeps_the_memory_text_in_full(
    backend: dict[str, Any],
) -> None:
    text = _banner(width=45, **backend)
    assert "╭" not in text
    if backend:
        assert "long-term memory" in text
        assert "Qwen3-Embedding-0.6B" in text
    else:
        assert "no long-term memory" in text
        assert "--embedding-base-url adds one" in text
    assert all(cells(line) <= 45 for line in text.split("\n"))


def test_the_banner_names_where_the_model_came_from() -> None:
    text = _banner(source="found on this machine", listed=3)
    assert (
        "│            found on this machine · the loaded one of 3; /model lists them"
        in text
    )


def test_the_banner_border_stays_straight_with_wide_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    lines = _banner(workspace=str(tmp_path / "项目")).split("\n")
    assert "│ workspace  ~/项目" in lines[2]
    assert {cells(line) for line in lines[:7]} == {80}


def test_a_narrow_banner_has_no_box() -> None:
    text = _banner(width=45)
    assert "╭" not in text and "model      stub-model" in text


def test_the_banner_warns_about_a_home_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert "⚠ The workspace is ~: file tools can reach every file in it." in _banner(
        workspace=os.path.realpath(tmp_path)
    )


def test_no_color_keeps_bold_and_drops_colors() -> None:
    output = Vt100_Output(
        io.StringIO(), lambda: Size(rows=24, columns=80), term="xterm"
    )
    styled = Text("✗ failed", style="error")
    styled.append(" bold", style="bold")
    colored = TerminalOut(output, no_color=False).render(styled)
    plain = TerminalOut(output, no_color=True).render(styled)
    assert "\x1b[31m" in colored
    assert not re.search(r"\x1b\[3\dm", plain) and "\x1b[1m" in plain


# ── Review fixes: text from the model or the server never acts on the terminal ──

OSC_TITLE = "\x1b]0;PWNED\x1b\\"
OSC_CLIPBOARD = "\x1b]52;c;cm0gLXJmIH4=\x07"


def test_an_answer_cannot_set_the_title_or_write_the_clipboard() -> None:
    text = Turn().finish(result=f"Done.{OSC_TITLE}{OSC_CLIPBOARD} all good").text()
    assert "\x1b" not in text and "PWNED" not in text
    assert "● Done. all good" in text


def test_a_best_effort_and_its_error_are_cleaned() -> None:
    text = (
        Turn(width=200)
        .finish(
            "provider_error",
            error=f"POST /chat/completions failed with HTTP 400: \x1b[3J{OSC_TITLE}bad",
            result=f"partial{OSC_TITLE}",
        )
        .text()
    )
    assert "\x1b" not in text
    assert "refused the request: HTTP 400: bad." in text


def test_a_tool_name_from_the_model_is_cleaned() -> None:
    turn = Turn()
    turn.tool(f"evil{OSC_TITLE}", "{}", status="error", error="Unknown tool")
    turn.event(
        "assistant.message",
        step=1,
        content="",
        thinking="",
        usage={},
        duration_s=0.1,
        tool_calls=[
            {"id": "c9", "name": f"never{OSC_TITLE}", "arguments": {}, "preview": ""}
        ],
    )
    turn.finish("cancelled")
    text = turn.text()
    assert "\x1b" not in text and "PWNED" not in text
    assert "✗ evil  {}" in text and "◌ never" in text


def test_model_ids_in_the_banner_are_cleaned() -> None:
    text = _banner(model=f"qwen{OSC_TITLE}3")
    assert "\x1b" not in text and "qwen3 · OpenAI-compatible" in text


# ── Review fixes: long text wraps, every character kept ──

LONG_ZH = "这个函数的问题在于它没有检查输入是否为空，所以当用户传入空列表时会抛出索引错误。建议在函数开头加一个检查。"
LONG_URL = "https://example.com/" + "a1b2c3d4" * 12


def _joined(text: str) -> str:
    return "".join(text.split())


def test_a_long_chinese_answer_and_a_long_url_wrap_whole() -> None:
    text = Turn().finish(result=f"{LONG_ZH}\n\n{LONG_URL}").text()
    assert "…" not in text.split("✻")[0]
    assert LONG_ZH in _joined(text) and LONG_URL in _joined(text)


def test_a_long_echo_wraps_whole() -> None:
    out = RecordingOut(40)
    TurnRenderer(out, symbols=UNICODE, backend=_backend).begin_turn(LONG_ZH)
    assert LONG_ZH in _joined(out.text())


def test_a_best_effort_keeps_its_code_block() -> None:
    answer = "```python\ndef f(x):\n    return x * 2\n```\nThat should work."
    text = Turn().finish("budget_exhausted", error="Max steps", result=answer).text()
    lines = text.split("\n")
    assert "● Best effort:" in lines
    code = lines.index(next(line for line in lines if "def f(x):" in line))
    assert lines[code + 1].strip() == "return x * 2"
    assert lines[code + 1].index("return") > lines[code].index("def")


@pytest.mark.parametrize(
    ("width", "row"),
    [
        # Whole parts go first; below that the facts themselves are cut.
        (20, "✳ Thinking… · step …"),
        (30, "✳ Thinking… · step 3 · 1h22m"),
        (41, "✳ Thinking… · step 3 · 1h22m"),
        (50, "✳ Thinking… · step 3 · 1h22m · ↓ 1.2k tokens"),
        (60, "✳ Thinking… · step 3 · 1h22m · ↓ 1.2k tokens · esc to stop"),
    ],
)
def test_the_spinner_row_drops_whole_parts_then_cuts_the_facts(
    width: int, row: str
) -> None:
    live = LiveState(activity="Thinking", started=0.0, step=3, output_tokens=1200)
    assert live_rows(live, 4920.0, width, UNICODE, frame=2)[0] == [
        ("class:accent", row)
    ]


@pytest.mark.parametrize("width", [30, 45, 50, 60])
def test_banner_values_wrap_under_themselves(width: int) -> None:
    lines = _banner(width=width, model="a-model-name-that-is-rather-long").split("\n")
    rows = [line for line in lines if line.strip()]
    assert all(cells(line) <= width for line in rows), rows
    joined = "".join(line.strip("│ ") for line in rows)
    assert URL in joined  # the URL shows whole, never cut
    body = [line.strip("│") for line in rows[1:]]
    tools = next(i for i, line in enumerate(body) if line.lstrip().startswith("tools"))
    assert body[tools + 1].startswith(" " * 11) or body[tools + 1].lstrip().startswith(
        "memory"
    )


def test_an_output_line_rich_measures_wider_is_not_wrapped() -> None:
    # "⚠️" is one cell for wcwidth, two for rich: the cut line must stay one row.
    text = Turn(width=60).tool("shell_exec", "lint", output="⚠️ " * 40 + "END").text()
    [line] = [row for row in text.split("\n") if "⚠" in row]
    assert line.startswith("  └ ⚠")


@pytest.mark.parametrize(
    ("approval", "tag"),
    [
        (
            {"allowed": True, "origin": "user", "remember": True},
            "allowed for the session",
        ),
        ({"allowed": True, "origin": "session"}, "allowed by a session grant"),
        ({"allowed": True, "origin": "policy"}, "auto-allowed"),
        ({"allowed": False, "origin": "timeout"}, "not answered in time"),
    ],
)
def test_a_long_preview_never_cuts_the_approval_tag(
    approval: dict[str, Any], tag: str
) -> None:
    preview = "printf '" + "x" * 200 + "'"
    text = Turn(width=100).tool("shell_exec", preview, approval=approval).text()
    [header] = [line for line in text.split("\n") if "shell_exec" in line]
    assert header.endswith(f" · {tag}") and cells(header) <= 100, header


def test_a_long_denial_reason_is_still_cut_at_the_end() -> None:
    approval = {"allowed": False, "origin": "user", "reason": "no " * 60}
    text = Turn(width=100).tool("shell_exec", "rm -rf build", approval=approval).text()
    [header] = [line for line in text.split("\n") if "shell_exec" in line]
    assert "rm -rf build" in header and header.endswith("…") and cells(header) == 100


# ── The turn's work as it happens (U2a.2) ──

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "events"
GOLDEN = Path(__file__).resolve().parent / "golden"

STREAMS = sorted(path.stem for path in FIXTURES.glob("*.jsonl"))
VERBOSE_STREAMS = (
    "tools",
    "thinking",
    "budget_reflect",
    "long_output",
    "file_write_overwrite",
    "history",
)
NARROW_STREAMS = ("tools", "allowed", "provider_errors", "history")
ASCII_THINKING_OFF_STREAMS = ("tools", "history")

GOLDEN_CASES = (
    [(name, "default", 80, UNICODE, False, True) for name in STREAMS]
    + [(name, "verbose", 80, UNICODE, True, True) for name in VERBOSE_STREAMS]
    + [(name, "narrow", 40, UNICODE, False, True) for name in NARROW_STREAMS]
    + [(name, "ascii", 80, ASCII, False, False) for name in ASCII_THINKING_OFF_STREAMS]
)


def _record(name: str) -> list[dict[str, Any]]:
    lines = (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def render_stream(
    name: str,
    *,
    width: int = 80,
    symbols: Symbols = UNICODE,
    verbose: bool = False,
    thinking: bool = True,
) -> str:
    """*name*'s recorded events through a renderer, as a session would feed them."""
    out = RecordingOut(width)
    renderer = TurnRenderer(
        out,
        symbols=symbols,
        backend=_backend,
        verbose=verbose,
        thinking=thinking,
    )
    turn = 0
    for record in _record(name):
        event = record["event"]
        kind = event["type"]
        data = event["data"]
        if kind == "turn.started":
            turn += 1
            renderer.begin_turn(str(data.get("user_text") or ""), first_turn=turn == 1)
        renderer.on_event(
            StoredEvent(
                id=record["turn"], session_id="s", turn=record["turn"], event=event
            )
        )
        if kind == "turn.finished":
            renderer.end_turn(None, None, cause=str(data.get("cause") or ""))
    return out.text()


@pytest.mark.parametrize(
    ("name", "mode", "width", "symbols", "verbose", "thinking"),
    GOLDEN_CASES,
    ids=[f"{name}-{mode}-{width}" for name, mode, width, _, _, _ in GOLDEN_CASES],
)
def test_a_recorded_stream_renders_its_golden(
    name: str,
    mode: str,
    width: int,
    symbols: Symbols,
    verbose: bool,
    thinking: bool,
) -> None:
    text = render_stream(
        name, width=width, symbols=symbols, verbose=verbose, thinking=thinking
    )
    for line in text.split("\n"):
        assert cells(line) <= width, (name, mode, cells(line), line)
    path = GOLDEN / f"{name}-{mode}-{width}.txt"
    if os.environ.get("LOCALMELO_UPDATE_GOLDEN") == "1":
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    assert path.exists(), f"missing {path.name}; run with LOCALMELO_UPDATE_GOLDEN=1"
    assert text == path.read_text(encoding="utf-8")


# ── LiveState, event by event ──


def test_live_state_follows_a_turn() -> None:
    turn = Turn()
    renderer = turn.renderer
    assert renderer.live is not None
    assert (renderer.live.activity, renderer.live.step, renderer.live.attempt) == (
        "Starting",
        0,
        1,
    )
    turn.event("step.started", step=2, attempt=1)
    assert renderer.live.activity == "Thinking" and renderer.live.step == 2
    turn.event(
        "tool.started",
        step=2,
        call_id="c1",
        name="shell_exec",
        arguments={"command": "ls"},
        preview="ls",
    )
    assert (renderer.live.activity, renderer.live.tool, renderer.live.preview) == (
        "Running",
        "shell_exec",
        "ls",
    )
    assert renderer.live.tool_limit_s == SESSION_TOOL_TIMEOUT_S
    turn.event(
        "tool.finished",
        step=2,
        call_id="c1",
        name="shell_exec",
        status="success",
        output="x",
        approval={"allowed": True, "origin": "user", "remember": False},
        duration_s=0.1,
    )
    assert renderer.live.activity == "Thinking" and renderer.live.tool == ""
    turn.event("run.progress", level=1, state="running", step=2, attempt=1)
    assert renderer.live.progress is not None
    assert renderer.live.progress["level"] == 1
    turn.event("approval.requested", request_id="r", tool_name="shell_exec")
    assert (
        renderer.live.waiting and renderer.live.activity == "Waiting for your approval"
    )
    turn.event("approval.resolved", request_id="r", outcome="allowed")
    assert not renderer.live.waiting
    turn.finish("completed", result="done", n_tool=1)
    assert renderer.live.finished and renderer.live.activity == "Wrapping up"
    assert live_rows(renderer.live, 100.0, 80, UNICODE)[0][0][1] == "Wrapping up…"


def test_a_retry_counts_down_in_the_live_area() -> None:
    turn = Turn()
    turn.event(
        "notice",
        kind="retry",
        message="Model request failed; retrying in 4 s: HTTP 503 busy",
        detail={"request_attempt": 1, "delay_s": 4.0, "status_code": 503},
        step=1,
        attempt=1,
    )
    assert "↻ Model request failed; retrying in 4 s: HTTP 503 busy" in turn.text()
    assert turn.renderer.live is not None
    assert turn.renderer.live.activity == "Retrying"
    rows = live_rows(turn.renderer.live, turn.now[0] + 2.5, 80, UNICODE, frame=0)
    assert "Retrying in 2s" in rows[0][0][1]
    rows = live_rows(turn.renderer.live, turn.now[0] + 9.0, 80, UNICODE, frame=0)
    assert "Retrying in" not in rows[0][0][1] and "Retrying…" in rows[0][0][1]


def test_an_empty_reply_is_a_dim_notice() -> None:
    turn = Turn()
    turn.event(
        "notice",
        kind="empty_reply",
        message="The model returned an empty reply; asking it to continue.",
        detail={"count": 1},
        step=1,
        attempt=1,
    )
    assert "· The model returned an empty reply; asking it to continue." in turn.text()


def test_a_budget_and_a_stuck_notice_are_warnings() -> None:
    text = (
        Turn()
        .event(
            "notice",
            kind="budget",
            message="The attempt used its step budget.",
            detail={},
            step=1,
            attempt=1,
        )
        .event(
            "notice",
            kind="stuck",
            message="The same tool calls or errors repeated; the attempt ends.",
            detail={},
            step=2,
            attempt=1,
        )
        .text()
    )
    assert "⚠ The attempt used its step budget." in text
    assert "⚠ The same tool calls or errors repeated; the attempt ends." in text


# ── Thinking, intermediate and held text ──


def _message(turn: Turn, content: str = "", **data: Any) -> Turn:
    fields: dict[str, Any] = {
        "step": 1,
        "content": content,
        "thinking": "",
        "tool_calls": [],
        "usage": {},
        "duration_s": 0.1,
    }
    fields.update(data)
    return turn.event("assistant.message", **fields)


def test_the_thinking_excerpt_is_three_rows_then_the_marker() -> None:
    # Single line breaks inside a paragraph are joined, then the rows wrap.
    thinking = " ".join(f"word{number}" for number in range(1, 400))
    text = _message(
        Turn(),
        thinking=thinking,
        tool_calls=[{"id": "c1", "name": "file_read", "preview": "a"}],
    ).text()
    assert "✻ Thought for 0.1s" in text
    body = [row for row in text.split("\n") if row.startswith("  ")]
    assert len(body) == 4  # three rows and the marker row
    assert body[3].strip().startswith("… +") and " lines" in body[3]


def test_verbose_shows_forty_thinking_rows_and_thinking_off_none() -> None:
    thinking = " ".join(f"w{number}" for number in range(1, 2500))
    verbose = Turn()
    verbose.renderer.verbose = True
    text = _message(verbose, thinking=thinking).text()
    body = [row for row in text.split("\n") if row.startswith("  ")]
    assert len(body) == 41  # 40 rows and the marker row
    assert body[40].strip().startswith("… +")
    off = Turn()
    off.renderer.thinking = False
    text = _message(off, thinking=thinking).text()
    assert "✻ Thought for 0.1s" in text
    assert not any(row.startswith("  w") for row in text.split("\n"))


def test_an_ollama_thinking_block_is_split_once() -> None:
    content = "<think>\nweighing the options\n</think>\n\nI will read the file."
    text = _message(
        Turn(),
        content=content,
        tool_calls=[{"id": "c1", "name": "file_read", "preview": "a"}],
    ).text()
    assert "weighing the options" in text
    assert "I will read the file." in text
    assert "<think>" not in text and "</think>" not in text


def test_intermediate_text_follows_its_tool_calls() -> None:
    text = _message(
        Turn(),
        content="Looking first.",
        tool_calls=[{"id": "c1", "name": "file_read", "preview": "a"}],
    ).text()
    assert "● Looking first." in text


def test_held_text_prints_at_the_next_step() -> None:
    text = (
        _message(Turn(), content="An intermediate thought.")
        .event("step.started", step=2, attempt=1)
        .text()
    )
    assert "● An intermediate thought." in text


def test_a_held_message_that_is_the_answer_prints_once() -> None:
    text = _message(Turn(), content="The answer.").finish(result="The answer.").text()
    assert text.count("The answer.") == 1


def test_a_held_message_that_differs_prints_before_the_status() -> None:
    text = (
        _message(Turn(), content="My plan was to read it.")
        .finish("failed", error="post_plan refused", result="")
        .text()
    )
    assert text.index("● My plan was to read it.") < text.index(
        "A check stopped the run"
    )


# ── Reflections ──


def test_a_readable_reflection_names_its_action_and_rationale() -> None:
    text = (
        Turn()
        .event(
            "reflection",
            attempt=1,
            action="continue",
            continue_run=True,
            rationale="Read the test before editing.",
            summary="s",
            best_effort_result=None,
            requests=1,
            readable=True,
            usage={},
            duration_s=2.3,
        )
        .text()
    )
    assert (
        "✻ Reflected after attempt 1: continue · Read the test before editing. (2.3s)"
        in text
    )


def test_verbose_reflection_adds_the_summary_and_asked_twice() -> None:
    turn = Turn()
    turn.renderer.verbose = True
    text = turn.event(
        "reflection",
        attempt=1,
        action="continue",
        continue_run=True,
        rationale="The full rationale, in one line.",
        summary="The summary.",
        best_effort_result=None,
        requests=2,
        readable=True,
        usage={},
        duration_s=0.5,
    ).text()
    assert "· asked twice" in text
    assert "The summary." in text and "The full rationale, in one line." in text


def test_an_unreadable_reflection_says_whether_the_run_goes_on() -> None:
    text = (
        Turn()
        .event(
            "reflection",
            attempt=2,
            action=None,
            continue_run=True,
            rationale=None,
            summary=None,
            best_effort_result=None,
            requests=2,
            readable=False,
            usage={},
            duration_s=0.2,
        )
        .event(
            "reflection",
            attempt=3,
            action=None,
            continue_run=False,
            rationale=None,
            summary=None,
            best_effort_result=None,
            requests=2,
            readable=False,
            usage={},
            duration_s=0.2,
        )
        .text()
    )
    assert "Could not read the reflection after attempt 2; the run continues" in text
    assert (
        "Could not read the reflection after attempt 3; the budget is used up" in text
    )


# ── /verbose's blocks ──


def test_verbose_blocks_add_arguments_and_twenty_lines() -> None:
    output = "\n".join(f"line {number}" for number in range(1, 31))
    turn = Turn()
    turn.renderer.verbose = True
    turn.event(
        "tool.started",
        step=1,
        call_id="c1",
        name="shell_exec",
        arguments={"command": "grep -n x notes.txt"},
        preview="grep -n x notes.txt",
    )
    text = turn.event(
        "tool.finished",
        step=1,
        call_id="c1",
        name="shell_exec",
        status="success",
        output=output,
        approval={"allowed": True, "origin": "user", "remember": False},
        duration_s=0.1,
    ).text()
    assert '"command": "grep -n x notes.txt"' in text
    assert "line 20" in text and "line 21" not in text
    assert "… +10 lines" in text


def test_images_are_named_not_drawn() -> None:
    text = (
        Turn()
        .tool(
            "screenshot",
            "{}",
            images=[{"mime_type": "image/png", "size_bytes": 312_000}],
        )
        .text()
    )
    assert "    ▪ image/png · 312 KB · not shown" in text


# ── The live area's third row ──


def _past_progress(**basis: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "routine_id": "r",
        "template_id": "r:t",
        "version": 1,
        "match": "routine",
        "model": "stub-model",
        "title": None,
    }
    fields.update(basis)
    return {
        "level": 1,
        "state": "running",
        "step": 1,
        "attempt": 1,
        "limits": {},
        "phase": None,
        "cycle": 0,
        "tests": None,
        "items": [{"label": "ls", "state": "done", "repeat": None}],
        "fraction": None,
        "expected_steps": None,
        "past": {
            "runs": 1,
            "steps_lo": 4,
            "steps_median": 4.0,
            "steps_hi": 4,
            "wall_lo_s": 10.0,
            "wall_hi_s": 10.0,
            "last_steps": 4,
            "last_wall_s": 10.0,
        },
        "off_reason": None,
        "basis": fields,
    }


def test_the_past_row_shows_on_a_first_turn_with_past_runs() -> None:
    live = LiveState(
        activity="Thinking",
        started=0.0,
        step=1,
        attempt=1,
        first_turn=True,
        progress=_past_progress(),
    )
    rows = live_rows(live, 5.0, 80, UNICODE, frame=0)
    assert len(rows) == 2
    assert rows[1][0][0] == "class:muted"
    assert rows[1][0][1] == "last time 4 steps, 10s: ls ✓"


def test_the_past_row_hides_on_a_later_turn_or_a_new_match() -> None:
    live = LiveState(
        activity="Thinking",
        started=0.0,
        step=1,
        attempt=1,
        first_turn=False,
        progress=_past_progress(),
    )
    assert len(live_rows(live, 5.0, 80, UNICODE, frame=0)) == 1
    live.first_turn = True
    live.progress = _past_progress(match="new")
    assert len(live_rows(live, 5.0, 80, UNICODE, frame=0)) == 1


def test_a_stale_turn_takes_its_status_from_turn_finished() -> None:
    text = (
        Turn()
        .event(
            "turn.finished",
            turn=1,
            turn_id="t",
            status="crashed",
            error="The process running this turn stopped",
            cause="crash",
        )
        .text()
    )
    assert "melo: run ended" not in text and "crashed" not in text
    assert "The process was killed or the computer lost power." in text
    # A crash cause is a warning, not a bug: yellow, not the red failed mark.
    assert "⚠ Interrupted: melo stopped unexpectedly." in text
    assert "✗ Interrupted" not in text


def test_a_wrapped_thinking_row_never_starts_with_a_space() -> None:
    # The reply's line breaks are joined into one paragraph; at this width
    # the wrap boundary falls on a space, which must not lead the next row.
    thinking = (
        "app.py is the single script. Reading it shows what the project does\n"
        "and whether it needs data.\nThe tests will say which behaviour matters."
    )
    text = _message(
        Turn(width=100),
        thinking=thinking,
        tool_calls=[{"id": "c1", "name": "file_read", "preview": "a"}],
    ).text()
    rows = [row for row in text.split("\n") if row.startswith("  ")]
    assert len(rows) >= 2, text
    assert not any(row.startswith("   ") for row in rows), rows


# ── Review fixes (T4R) ──


def test_intermediate_and_held_text_cannot_act_on_the_terminal() -> None:
    for content in ("A\x1b[2JB", "A\x1b]0;PWNED\x1b\\B", "A\x1b]52;c;aGk=\x07B"):
        with_calls = _message(
            Turn(),
            content=content,
            tool_calls=[{"id": "c1", "name": "file_read", "preview": "a"}],
        ).text()
        held = _message(Turn(), content=content).text()
        assert "\x1b" not in with_calls and "\x1b" not in held, content
        assert "PWNED" not in held.split("›", 1)[-1] or "\x1b" not in held


def test_an_image_mime_type_cannot_act_on_the_terminal() -> None:
    text = (
        Turn()
        .tool(
            "screenshot",
            "{}",
            images=[{"mime_type": "image/png\x1b]0;PWNED\x07", "size_bytes": 3}],
        )
        .text()
    )
    assert "\x1b" not in text and "\x07" not in text


def test_a_cancelled_run_waits_for_the_turn_finished_cause() -> None:
    turn = Turn()
    turn.finish("cancelled", error="Another window took over")
    assert "Stopped" not in turn.text()  # deferred until the cause arrives
    turn.event(
        "turn.finished",
        turn=1,
        turn_id="t",
        status="cancelled",
        error="Another window took over",
        cause="takeover",
    )
    assert "■ Stopped: another melo window took over this session" in turn.text()


def test_a_long_reflection_and_a_footer_are_one_line_each() -> None:
    rationale = "Run it once more, then report the result to the user."
    text = (
        Turn(width=40)
        .event(
            "reflection",
            attempt=1,
            action="continue",
            continue_run=True,
            rationale=rationale,
            summary=None,
            best_effort_result=None,
            requests=1,
            readable=True,
            usage={"completion_tokens": 76},
            duration_s=0.4,
        )
        .finish("completed", result="done", n_tool=3, duration_s=0.5)
        .text()
    )
    lines = text.split("\n")
    [reflection] = [
        index for index, line in enumerate(lines) if "Reflected after" in line
    ]
    assert lines[reflection].endswith("…") and "(0.4s)" not in lines[reflection]
    [footer] = [index for index, line in enumerate(lines) if "Worked for" in line]
    assert "tokens" not in lines[footer] and len(lines[footer]) <= 40
    assert lines[footer + 1] == ""


def test_an_empty_rationale_has_one_separator() -> None:
    text = (
        Turn()
        .event(
            "reflection",
            attempt=1,
            action="continue",
            continue_run=True,
            rationale="",
            summary=None,
            best_effort_result=None,
            requests=1,
            readable=True,
            usage={},
            duration_s=0.1,
        )
        .text()
    )
    assert "Reflected after attempt 1: continue (0.1s)" in text


def test_a_retry_ends_at_the_next_event() -> None:
    turn = Turn()
    turn.event(
        "notice",
        kind="retry",
        message="Model request failed; retrying in 4 s",
        detail={"delay_s": 4.0},
        step=1,
        attempt=1,
    )
    assert turn.renderer.live is not None
    assert turn.renderer.live.activity == "Retrying"
    _message(
        turn, content="", tool_calls=[{"id": "c1", "name": "file_read", "preview": "a"}]
    )
    assert turn.renderer.live.activity == "Thinking"


# ── U2a.3: a file write's change from its snapshot ──


def _snapshots(
    request_id: str, *, old: str | None, new: str, added: int = 1, removed: int = 0
) -> Any:
    from localmelo.support.cli import edits
    from localmelo.support.cli.approvals import EditSnapshots, Snapshot

    snapshots = EditSnapshots()
    target = (
        edits.Target(path="todo.md", exists=False)
        if old is None
        else edits.Target(path="todo.md", exists=True, text=old)
    )
    snapshots._snapshots[request_id] = Snapshot(request_id, target, new, added, removed)
    return snapshots


def test_a_file_write_block_shows_the_counts_from_the_snapshot() -> None:
    snapshots = _snapshots("r1", old="a\n", new="b\n", added=1, removed=1)
    text = (
        Turn(snapshots=snapshots)
        .tool(
            "file_write",
            "todo.md (2 chars)",
            request_id="r1",
            status="success",
            output="Written 2 bytes to todo.md",
            approval={
                "request_id": "r1",
                "allowed": True,
                "origin": "user",
                "remember": False,
            },
            duration_s=0.1,
        )
        .text()
    )
    assert "  └ +1 −1 lines" in text
    assert "Written 2 bytes" not in text


def test_a_new_file_block_shows_its_line_count() -> None:
    snapshots = _snapshots("r1", old=None, new="a\nb\n", added=2, removed=0)
    text = (
        Turn(snapshots=snapshots)
        .tool(
            "file_write",
            "new.md",
            request_id="r1",
            status="success",
            output="Written 4 bytes",
            approval={"request_id": "r1", "allowed": True, "origin": "user"},
        )
        .text()
    )
    assert "  └ new file, 2 lines" in text


def test_verbose_shows_the_change_hunks() -> None:
    snapshots = _snapshots("r1", old="a\n", new="b\n", added=1, removed=1)
    turn = Turn(snapshots=snapshots)
    turn.renderer.verbose = True
    text = turn.tool(
        "file_write",
        "todo.md (2 chars)",
        request_id="r1",
        status="success",
        output="Written 2 bytes",
        approval={"request_id": "r1", "allowed": True, "origin": "user"},
    ).text()
    assert "     1 − a" in text and "     1 + b" in text


def test_a_write_without_a_snapshot_shows_its_output() -> None:
    text = (
        Turn()
        .tool(
            "file_write",
            "todo.md (2 chars)",
            request_id="r1",
            status="success",
            output="Written 2 bytes to todo.md",
            approval={"request_id": "r1", "allowed": True, "origin": "user"},
        )
        .text()
    )
    assert "  └ Written 2 bytes to todo.md" in text


# ── Review fixes ──


def test_a_file_write_block_uses_the_snapshots_counts() -> None:
    from localmelo.support.cli import edits
    from localmelo.support.cli.approvals import EditSnapshots, Snapshot

    snapshots = EditSnapshots()
    target = edits.Target(path="todo.md", exists=True, text="a\n")
    snapshots._snapshots["r1"] = Snapshot("r1", target, "b\n", 7, 9)
    text = (
        Turn(snapshots=snapshots)
        .tool(
            "file_write",
            "todo.md",
            request_id="r1",
            status="success",
            output="Written",
            approval={"request_id": "r1", "allowed": True, "origin": "user"},
        )
        .text()
    )
    assert "  └ +7 −9 lines" in text


def test_a_write_with_no_diff_shows_that() -> None:
    from localmelo.support.cli import edits
    from localmelo.support.cli.approvals import EditSnapshots, Snapshot

    snapshots = EditSnapshots()
    target = edits.Target(
        path="big.bin", exists=True, problem="overwrites a 1.2 MB file"
    )
    snapshots._snapshots["r1"] = Snapshot("r1", target, "x\ny\n", 2, 0)
    text = (
        Turn(snapshots=snapshots)
        .tool(
            "file_write",
            "big.bin",
            request_id="r1",
            status="success",
            output="Written",
            approval={"request_id": "r1", "allowed": True, "origin": "user"},
        )
        .text()
    )
    assert "  └ changed; no diff shown" in text


def test_verbose_hunks_cannot_act_on_the_terminal() -> None:
    from localmelo.support.cli import edits
    from localmelo.support.cli.approvals import EditSnapshots, Snapshot

    snapshots = EditSnapshots()
    target = edits.Target(path="todo.md", exists=True, text="old\x1b[2J\n")
    snapshots._snapshots["r1"] = Snapshot("r1", target, "new\x1b]0;PWNED\x07\n", 1, 1)
    turn = Turn(snapshots=snapshots)
    turn.renderer.verbose = True
    text = turn.tool(
        "file_write",
        "todo.md",
        request_id="r1",
        status="success",
        output="Written",
        approval={"request_id": "r1", "allowed": True, "origin": "user"},
    ).text()
    assert "\x1b" not in text and "\x07" not in text


def test_events_feed_the_usage_snapshot_until_the_session_is_cleared() -> None:
    turn = Turn()
    turn.event("run.started", query="q", max_steps=30, workspace_root="", approval=True)
    turn.event("assistant.message", usage={"prompt_tokens": 200, "cached_tokens": 50})
    turn.event("run.progress", level=0, step=1, attempt=1)
    usage = turn.renderer.usage
    assert (usage.last_prompt, usage.last_cached, usage.run_active) == (200, 50, True)
    assert usage.progress is not None
    turn.renderer.clear_session()
    assert turn.renderer.usage.requests == 0
