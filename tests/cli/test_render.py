"""What a turn's events print, and what the live area and the banner show."""

from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.text import Text

from localmelo.melo.schema import RunResult, ToolInfo
from localmelo.support.cli.output import RecordingOut, TerminalOut
from localmelo.support.cli.render import LiveState, TurnRenderer, live_rows
from localmelo.support.cli.text import cells
from localmelo.support.cli.theme import ASCII, UNICODE, Symbols
from localmelo.support.cli.views import BackendInfo, banner
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
        self, width: int = 80, symbols: Symbols = UNICODE, **backend: Any
    ) -> None:
        self.out = RecordingOut(width)
        self.now = [100.0]
        self.renderer = TurnRenderer(
            self.out,
            symbols=symbols,
            backend=lambda: _backend(**backend),
            clock=lambda: self.now[0],
        )
        self.renderer.begin_turn("What is in this folder?")

    def event(self, kind: str, **data: Any) -> Turn:
        self.renderer.on_event(
            StoredEvent(
                id=1, session_id="s", turn=1, event={"type": kind, "data": data}
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
    assert "  └ app.py\n    notes.txt\n    tests\n    … +2 lines\n" in text


def test_a_fourth_line_becomes_the_marker() -> None:
    text = Turn().tool("shell_exec", "ls", output="a\nb\nc\nd").text()
    assert "    c\n    … +1 lines\n" in text and "    d\n" not in text


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


@pytest.mark.parametrize(
    ("status", "error", "mark", "hint"),
    [
        (
            "provider_error",
            "POST /chat/completions failed: ConnectError: refused",
            "✗",
            f"Is the model server running at {URL}?",
        ),
        (
            "provider_error",
            "POST /chat/completions failed: ReadTimeout: x",
            "✗",
            "may still be loading the model.",
        ),
        (
            "provider_error",
            "POST /chat/completions failed with HTTP 404: no",
            "✗",
            "The server does not know the model 'stub-model'; check the name (/model lists the models it has).",
        ),
        (
            "provider_error",
            "POST /chat/completions failed with HTTP 401: key",
            "✗",
            'Turn the key off in the server (oMLX: "Allow inference without API key").',
        ),
        (
            "provider_error",
            "POST /chat/completions failed with HTTP 400: tools",
            "✗",
            "if it says the model does not support tools",
        ),
        ("budget_exhausted", "Max steps reached", "⚠", None),
        ("stuck", "repeated", "⚠", "The model repeated itself; rephrase the request."),
        (
            "failed",
            "empty replies",
            "⚠",
            "The model returned only empty replies; try another model.",
        ),
        (
            "gate_stopped",
            "stop",
            "⚠",
            "melo judged that another attempt would not help.",
        ),
        (
            "crashed",
            "KeyError: x",
            "✗",
            "This is a bug in localmelo; the log is ~/.cache/localmelo/logs/cli.log.",
        ),
    ],
)
def test_a_run_that_did_not_complete_prints_its_status_line_and_hint(
    status: str, error: str, mark: str, hint: str | None
) -> None:
    text = Turn().finish(status, error=error, result="partial").text()
    assert "● Best effort: partial" in text
    assert f"{mark} melo: run ended with status {status}: " in text
    if hint:
        assert " ".join(hint.split()) in " ".join(text.split())
    else:
        assert "\n  " not in text.split("melo: run ended")[1].split("✻")[0]
    assert "✻ Worked for" in text


def test_ollama_and_cloud_hints() -> None:
    text = (
        Turn(key="ollama", label="Ollama")
        .finish("provider_error", error="POST /api/chat failed with HTTP 404: x")
        .text()
    )
    assert "ollama pull stub-model fetches it." in text
    text = (
        Turn(key="openai", label="OpenAI", base_url="", key_env="OPENAI_API_KEY")
        .finish(
            "provider_error", error="POST /chat/completions failed with HTTP 401: x"
        )
        .text()
    )
    assert "The server refused the key in OPENAI_API_KEY;" in text


@pytest.mark.parametrize(
    ("cause", "line"), [("user_stop", "■ Stopped"), ("shutdown", "■ Interrupted")]
)
def test_a_stopped_turn(cause: str, line: str) -> None:
    turn = Turn()
    turn.renderer.stopped_by(cause)
    assert turn.renderer.live is not None and turn.renderer.live.activity == "Stopping"
    turn.finish("cancelled", error="Stopped by the user")
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
    return [
        ToolInfo(
            name=name,
            description="",
            parameters={},
            source="builtin",
            requires_approval=asks,
        )
        for name, asks in (
            ("file_read", False),
            ("file_write", True),
            ("shell_exec", True),
            ("python_exec", True),
        )
    ]


def _banner(
    width: int = 80, workspace: str = "/tmp/demo-project", **backend: Any
) -> str:
    out = RecordingOut(width)
    out.print(
        banner(
            version="0.1.0",
            backend=_backend(**backend),
            workspace=workspace,
            tools=_tools(),
            memory=True,
            yolo=False,
            width=width,
            symbols=UNICODE,
        )
    )
    return out.text()


def test_the_banner_box() -> None:
    lines = _banner(source="config").split("\n")
    assert lines[0].startswith("╭─ melo 0.1.0 ─") and lines[0].endswith("╮")
    assert lines[1].startswith("│ model      stub-model · OpenAI-compatible · " + URL)
    assert lines[1].endswith(" │")
    assert "│ tools      4 built-in (3 ask first)" in lines[3]
    assert "│ memory     no long-term memory (no embedding backend)" in lines[4]
    assert "│ session    new" in lines[5]
    assert all(cells(line) == 80 for line in lines[:7])
    assert lines[7] == "  / commands · esc stop · ctrl+d quit"


def test_the_banner_names_where_the_model_came_from() -> None:
    text = _banner(source="found on this machine", listed=3)
    assert "│            found on this machine · first of 3; /model lists them" in text


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
    assert "failed with HTTP 400: bad" in text


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
        (20, "✳ Thinking… · step 3"),
        (30, "✳ Thinking… · step 3 · 1h22m"),
        (41, "✳ Thinking… · step 3 · 1h22m"),
        (50, "✳ Thinking… · step 3 · 1h22m · ↓ 1.2k tokens"),
        (60, "✳ Thinking… · step 3 · 1h22m · ↓ 1.2k tokens · esc to stop"),
    ],
)
def test_the_spinner_row_drops_whole_facts_rather_than_cut_them(
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
