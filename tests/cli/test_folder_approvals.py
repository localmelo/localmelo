"""The terminal's option list for a folder outside the workspace (#405).

A read offers Yes, this folder for this chat, always this folder (kept in
Settings) and No; a write offers Yes and No and starts on No. The panel
names the real folder. HOME is a temporary folder, so paths read ``~/``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    DenyAllPolicy,
    find_folder_asker,
)
from localmelo.melo.contracts.folders import OutsideAccess
from localmelo.support.cli.approvals import (
    Approvals,
    TerminalApprover,
    folder_options,
    panel_text,
)
from localmelo.support.cli.main_chat import approval_request
from localmelo.tests.cli.ui_helpers import ENTER, running_app


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("HOME", str(folder))
    return Path(os.path.realpath(folder))


@pytest.fixture
def now() -> list[float]:
    return [100.0]


def _request(
    home: Path, tool: str = "file_read", access: str = "read"
) -> ApprovalRequest:
    career = home / "developer" / "career"
    path = career / ("cv.md" if tool != "list_files" else "")
    grantable = access == "read"
    arguments = {"path": str(path)}
    if tool == "file_write":
        arguments["content"] = "new line\n"
    return ApprovalRequest(
        request_id="r1",
        run_id="run",
        call_id="c1",
        tool_name=tool,
        arguments=arguments,
        preview="",
        workspace_root=str(home / "workspace"),
        tier="read" if access == "read" else "write",
        outside=OutsideAccess(
            str(path).rstrip("/"), str(career), access, grantable  # type: ignore[arg-type]
        ),
    )


def test_a_read_panel_names_the_folder_and_says_read_only(home: Path) -> None:
    text = panel_text(_request(home), width=80, show_tier=False)

    assert text.title == "Let melo read in ~/developer/career?"
    assert text.header == "file_read · outside ~/workspace · read only"
    assert [line for _, line in text.body] == [
        "melo wants to read ~/developer/career/cv.md."
    ]


def test_a_list_panel_says_look_in(home: Path) -> None:
    text = panel_text(_request(home, "list_files"), width=80, show_tier=False)
    assert text.title == "Let melo look in ~/developer/career?"


def test_a_write_panel_keeps_the_file_body_and_says_it_asks_every_time(
    home: Path,
) -> None:
    text = panel_text(_request(home, "file_write", "write"), width=80, show_tier=False)

    assert text.title == "Write outside the workspace, in ~/developer/career?"
    assert text.notes == (
        "~/developer/career/cv.md is outside ~/workspace. A write there asks every "
        "time.",
    )
    assert any("new line" in line for _, line in text.body)


def test_the_labels_of_the_folder_options(home: Path) -> None:
    assert folder_options(_request(home)) == (
        "Yes, and allow ~/developer/career for this chat",
        "Yes, and always allow ~/developer/career (kept in Settings)",
    )


async def test_the_answers_become_folder_decisions(home: Path) -> None:
    approver = TerminalApprover(clock=lambda: 0.0)
    for choice, decision in (
        ("once", ApprovalDecision(True)),
        ("session", ApprovalDecision(True, remember=True)),
        ("always", ApprovalDecision(True, remember=True, always=True)),
        ("deny", ApprovalDecision(False, "no")),
    ):
        asking = asyncio.ensure_future(approver.ask(_request(home)))
        await asyncio.sleep(0)
        approver.answer(choice, "no")  # type: ignore[arg-type]
        assert await asking == decision


async def test_a_write_offers_neither_chat_nor_always(home: Path) -> None:
    approver = TerminalApprover(clock=lambda: 0.0)
    asking = asyncio.ensure_future(approver.ask(_request(home, "file_write", "write")))
    for _ in range(200):  # the file is read first, in a thread
        if approver.pending is not None:
            break
        await asyncio.sleep(0.01)
    pending = approver.pending
    assert pending is not None
    assert (pending.offer_session, pending.offer_always, pending.selected) == (
        False,
        False,
        2,
    )
    approver.answer("always")
    approver.answer("session")
    assert not asking.done()
    approver.answer("once")
    assert await asking == ApprovalDecision(True)


async def test_the_option_list_on_screen_and_key_three_keeps_the_folder(
    home: Path, now: list[float]
) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request(home)))
        await app.wait(lambda: "Let melo read in ~/developer/career?" in app.text())
        text = app.text()
        assert "❯ 1. Yes" in text
        assert "2. Yes, and allow ~/developer/career for this chat" in text
        assert "3. Yes, and always allow ~/developer/career (kept in Settings)" in text
        assert "4. No, and tell melo why" in text
        assert "Always keeps the folder in Settings" in text
        assert "still asks every time." in text
        assert "1-4 · v view all" in text
        now[0] += 1
        app.type("3")
        assert await asyncio.wait_for(asking, 3) == ApprovalDecision(
            True, remember=True, always=True
        )


async def test_key_four_asks_for_a_reason(home: Path, now: list[float]) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(approver.ask(_request(home)))
        await app.wait(lambda: "Let melo read in" in app.text())
        now[0] += 1
        app.type("4")
        await app.wait(lambda: "Reason (optional" in app.text(), timeout_s=3)
        app.type("not that folder")
        app.type(ENTER)
        assert await asyncio.wait_for(asking, 3) == ApprovalDecision(
            False, "not that folder"
        )


async def test_a_write_panel_starts_on_no(home: Path, now: list[float]) -> None:
    approver = TerminalApprover(clock=lambda: now[0])
    async with running_app(approver=approver, clock=lambda: now[0]) as app:
        asking = asyncio.ensure_future(
            approver.ask(_request(home, "file_write", "write"))
        )
        await app.wait(lambda: "Write outside the workspace" in app.text())
        text = app.text()
        assert "❯ 2. No, and tell melo why" in text
        assert "always allow" not in text and "for this chat" not in text
        now[0] += 1
        app.type("1")
        assert await asyncio.wait_for(asking, 3) == ApprovalDecision(True)


def test_ask_and_auto_ask_about_folders_deny_and_yolo_do_not() -> None:
    async def ask(request: ApprovalRequest) -> ApprovalDecision:
        raise AssertionError("never asked")

    approvals = Approvals(ask, "ask")
    assert find_folder_asker(approvals.policy()) is not None
    approvals.set_mode("auto")
    assert find_folder_asker(approvals.policy()) is not None
    approvals.set_mode("deny")
    assert isinstance(approvals.policy(), DenyAllPolicy)
    assert find_folder_asker(approvals.policy()) is None
    assert Approvals(ask, "yolo").policy() is None


def test_the_main_chat_reads_the_folder_from_the_gateway(home: Path) -> None:
    pending = {
        "request_id": "r1",
        "tool_name": "file_read",
        "arguments": {"path": "/x/y/cv.md"},
        "workspace_root": "/w",
        "outside": {
            "path": "/x/y/cv.md",
            "folder": "/x/y",
            "access": "read",
            "grantable": True,
        },
    }
    request = approval_request(pending)
    assert request.outside == OutsideAccess("/x/y/cv.md", "/x/y", "read", True)
    # A write is never grantable, whatever the gateway says.
    pending["outside"] = {**pending["outside"], "access": "write"}  # type: ignore[dict-item]
    assert approval_request(pending).outside == OutsideAccess(
        "/x/y/cv.md", "/x/y", "write", False
    )
    assert approval_request({"request_id": "r2"}).outside is None
