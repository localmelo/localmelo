"""The terminal in the main chat, against a fake gateway, and finding the gateway.

:class:`FakeGateway` is API v2 in memory behind ``httpx.MockTransport``: the
main chat, side chats, their events on a stream, approvals and cancels. The
session is the real :class:`CliSession` of ``test_session``'s harness (a
real store, a recording terminal, a fake app). :func:`connect` is tested
against small HTTP servers on ports the system picks, with HOME in a
temporary folder.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import http.server
import json
import os
import socket
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.support import config, gateway_endpoint
from localmelo.support.cli import main_chat
from localmelo.support.cli.main_chat import (
    GatewayClient,
    Link,
    MainChat,
    Offline,
    approval_request,
    card_line,
    echo_line,
)
from localmelo.support.cli.theme import UNICODE

from .test_session import Setup, _until
from .test_session import setup as setup

MAIN = "main00000001"


class FakeGateway:
    """The gateway's API v2 in memory (see the module docstring)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.sessions: dict[str, dict[str, Any]] = {
            MAIN: {"id": MAIN, "kind": "front", "main": True, "title": "Main chat"}
        }
        self.turns: dict[str, list[dict[str, Any]]] = {MAIN: []}
        self.pending: dict[str, list[dict[str, Any]]] = {}
        self.requests: list[tuple[str, str, Any]] = []
        self.closed = False
        self.lose_stream = False
        self._running: dict[str, tuple[str, int]] = {}  # turn id -> session, number

    # ── Events ──

    def emit(
        self, session_id: str, type_: str, data: dict[str, Any], turn: int | None
    ) -> int:
        event_id = len(self.events) + 1
        self.events.append(
            {"id": event_id, "session_id": session_id, "turn": turn, "v": 1,
             "type": type_, "run_id": "", "seq": None, "ts": 1000.0 + event_id,
             "data": data}
        )  # fmt: skip
        return event_id

    def start_turn(self, session_id: str, text: str, key: str | None) -> dict[str, Any]:
        turns = self.turns.setdefault(session_id, [])
        number = len(turns) + 1
        turn_id = f"{session_id}-t{number}"
        turns.append({"id": turn_id, "turn": number, "user_text": text,
                      "status": "running", "answer": "", "started_at": 1000.0})  # fmt: skip
        self.emit(session_id, "turn.started", {"turn": number, "turn_id": turn_id,
                  "user_text": text, "client_request_id": key}, number)  # fmt: skip
        self.emit(session_id, "run.started", {"max_steps": 30}, number)
        if text.startswith("approve"):
            request = {"request_id": "r1", "call_id": "c1", "tool_name": "shell_exec",
                       "arguments": {"command": "ls"}, "arguments_complete": True,
                       "preview": "ls", "workspace_root": "/tmp/w",
                       "review": None}  # fmt: skip
            if "big" in text:
                request["arguments_complete"] = False
            if "high" in text:
                request["review"] = {"verdict": "high", "title": "Install software?",
                                     "reasons": ["Installs x with pip."],
                                     "categories": ["install"]}  # fmt: skip
            self.pending[session_id] = [request]
            self.emit(session_id, "approval.requested",
                      {k: request[k] for k in ("request_id", "call_id", "tool_name",
                       "arguments", "preview", "workspace_root")}, number)  # fmt: skip
            self._running[turn_id] = (session_id, number)
        elif text.startswith("long"):
            self._running[turn_id] = (session_id, number)
        else:
            self.finish(session_id, number, "completed", f"Answer to {text}.")
        return {"turn_id": turn_id, "turn": {"turn": number, "id": turn_id}}

    def finish(self, session_id: str, number: int, status: str, answer: str = "",
               cause: str = "run") -> None:  # fmt: skip
        result = {"task_id": "", "status": status, "result": answer, "error": "",
                  "n_tool": 0, "duration_s": 0.1}  # fmt: skip
        self.emit(session_id, "run.finished", {"result": result}, number)
        turn = self.turns[session_id][number - 1]
        turn.update(status=status, answer=answer)
        self.emit(session_id, "turn.finished", {"turn": number, "turn_id": turn["id"],
                  "status": status, "error": "", "cause": cause}, number)  # fmt: skip

    # ── Routes ──

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        body: Any = json.loads(request.content) if request.content else None
        self.requests.append((method, path, body))
        parts = path.strip("/").split("/")
        if path == "/api/sessions" and method == "GET":
            side = [s for s in self.sessions.values() if not s["main"]]
            return httpx.Response(
                200, json={"sessions": side, "main": self.sessions[MAIN]}
            )
        if path == "/api/sessions" and method == "POST":
            session_id = f"side{len(self.sessions):08d}"
            self.sessions[session_id] = {"id": session_id, "kind": "chat", "main": False,
                                         "title": body["turn"]["text"],
                                         "workspace_label": "~/w"}  # fmt: skip
            started = self.start_turn(session_id, body["turn"]["text"],
                                      body["turn"]["client_request_id"])  # fmt: skip
            return httpx.Response(
                201, json={"session": self.sessions[session_id], **started}
            )
        if parts[:2] == ["api", "sessions"] and len(parts) == 3:
            session_id = parts[2]
            if session_id not in self.sessions:
                return _error(404, "not_found", "Session not found.")
            if method == "PATCH":
                self.sessions[session_id]["title"] = body["title"]
                return httpx.Response(200, json={"session": self.sessions[session_id]})
            last = max(
                (e["id"] for e in self.events if e["session_id"] == session_id),
                default=0,
            )
            return httpx.Response(200, json={"session": self.sessions[session_id],
                                  "pending_approvals": self.pending.get(session_id, []),
                                  "last_event_id": last})  # fmt: skip
        if parts[:2] == ["api", "sessions"] and parts[3:] == ["turns"]:
            session_id = parts[2]
            if method == "POST":
                text = body["text"]
                if text.startswith("/task"):
                    data = {"echo": "started", "task_id": "t1", "title": text[6:],
                            "line": f"Started a task: {text[6:]} · Normal"}  # fmt: skip
                    echo_id = self.emit(session_id, "attention.echo", data, None)
                    return httpx.Response(200, json={"echo": {"id": echo_id, **data}})
                if any(
                    t["status"] == "running" for t in self.turns.get(session_id, [])
                ):
                    return _error(
                        409,
                        "session_busy",
                        "A turn is already running in this session.",
                    )
                started = self.start_turn(session_id, text, body["client_request_id"])
                return httpx.Response(202, json=started)
            limit = int(request.url.params.get("limit", 10))
            turns = self.turns.get(session_id, [])[-limit:]
            cards = [{"id": e["id"], "ts": e["ts"], **e["data"]} for e in self.events
                     if e["session_id"] == session_id and e["type"] == "attention.card"]  # fmt: skip
            first = next((e["id"] for e in self.events if e["session_id"] == session_id
                          and turns and e["turn"] == turns[0]["turn"]), 1)  # fmt: skip
            return httpx.Response(200, json={"turns": turns, "cards": cards,
                                  "replay_from": first - 1})  # fmt: skip
        if parts[:2] == ["api", "sessions"] and parts[3:] == ["retry"]:
            turns = self.turns[parts[2]]
            if not turns or turns[-1]["id"] != body["turn_id"]:
                return _error(409, "not_latest_turn", "Only the newest finished turn.")
            turns[-1]["undone"] = True
            started = self.start_turn(parts[2], turns[-1]["user_text"], None)
            return httpx.Response(202, json=started)
        if parts[:2] == ["api", "sessions"] and parts[3:] == ["events"]:
            if self.lose_stream:
                return _error(503, "store_unavailable", "The store is not open.")
            after = int(request.url.params.get("after", 0))
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=self._stream(parts[2], after))  # fmt: skip
        if parts[:2] == ["api", "approvals"]:
            for session_id, asked in self.pending.items():
                if asked and asked[0]["request_id"] == parts[2]:
                    self.pending[session_id] = []
                    turn_id = next(
                        t for t, (s, _) in self._running.items() if s == session_id
                    )
                    _, number = self._running.pop(turn_id)
                    outcome = "denied" if body["choice"] == "deny" else "allowed"
                    self.emit(session_id, "approval.resolved", {"request_id": parts[2],
                              "outcome": outcome}, number)  # fmt: skip
                    self.finish(session_id, number, "completed", "Ran it.")
                    return httpx.Response(200, json={"outcome": outcome})
            return _error(404, "not_found", "Approval not found.")
        if parts[:2] == ["api", "turns"] and parts[3:] == ["cancel"]:
            found = self._running.pop(parts[2], None)
            if found is None:
                return _error(409, "turn_not_running", "This turn is not running.")
            self.finish(found[0], found[1], "cancelled", cause="user_stop")
            return httpx.Response(202, json={"cancelling": True})
        return _error(404, "not_found", "Not found.")

    async def _stream(self, session_id: str, after: int) -> AsyncIterator[bytes]:
        while not self.closed:
            for event in [
                e
                for e in self.events
                if e["session_id"] == session_id and e["id"] > after
            ]:
                after = event["id"]
                yield f"id: {after}\ndata: {json.dumps(event)}\n\n".encode()
            if self.lose_stream:
                return
            await asyncio.sleep(0.01)


def _error(status: int, code: str, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": message}})


@pytest.fixture
async def chat(setup: Setup) -> AsyncIterator[tuple[Setup, FakeGateway]]:
    gateway = FakeGateway()
    client = GatewayClient(
        "http://127.0.0.1:1", transport=httpx.MockTransport(gateway.handle), headers={}
    )
    setup.session.remote = MainChat(
        setup.session, client, Link("http://127.0.0.1:1", 1)
    )
    setup.session.title = main_chat.MAIN_TITLE
    picks: list[int | None] = []
    offered: list[tuple[str, list[str], int]] = []

    async def choose_from(title: str, options: Any, *, default: int = 0) -> int | None:
        offered.append((title, list(options), default))
        return picks.pop(0)

    setup.app.choose_from = choose_from  # type: ignore[attr-defined]
    setup.app.picks_from = picks  # type: ignore[attr-defined]
    setup.app.offered_from = offered  # type: ignore[attr-defined]
    await setup.session.open_main_chat()
    try:
        yield setup, gateway
    finally:
        gateway.closed = True


async def test_a_message_in_the_main_chat_is_a_turn_of_the_gateway(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session = setup.session
    assert session.status_bar().title == "Main chat"
    assert session.enter("hello") is None
    assert session.state == "running"
    await setup.idle()
    await _until(lambda: "Answer to hello." in setup.out.text())
    sent = [r for r in gateway.requests if r[0] == "POST"]
    assert sent[0][1] == f"/api/sessions/{MAIN}/turns"
    assert sent[0][2]["text"] == "hello" and sent[0][2]["client_request_id"]
    # The terminal's own store holds nothing: the turn is the gateway's.
    assert await setup.store.list_sessions() == []
    assert session.status_bar().title == "Main chat"


async def test_retry_runs_the_newest_turn_again_through_the_gateway(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session = setup.session
    assert session.enter("/retry") is None
    await _until(lambda: "Nothing to retry" in setup.out.text())
    await setup.idle()
    assert session.enter("hello") is None
    await setup.idle()
    assert session.enter("/retry") is None
    await _until(lambda: "↻ Retrying: hello" in setup.out.text())
    await setup.idle()
    # The retried turn's events come before the gateway's answer: all are drawn.
    assert setup.out.text().count("Answer to hello.") == 2
    assert ("POST", f"/api/sessions/{MAIN}/retry", {"turn_id": f"{MAIN}-t1"}) in (
        gateway.requests
    )


async def test_task_goes_to_the_main_chat_and_its_echo_is_one_line(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    assert setup.session.enter("/task write the slides") is None
    await setup.idle()
    await _until(lambda: "Started a task: write the slides" in setup.out.text())
    assert setup.out.text().count("Started a task: write the slides") == 1
    assert "Unknown command" not in setup.out.text()
    assert gateway.requests[-1][2]["text"] == "/task write the slides"


async def test_commands_of_the_terminals_own_chat_say_where_they_work(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    for command in ("/model", "/undo", "/resume x"):
        assert setup.session.enter(command) is None
    text = " ".join(setup.out.text().split())
    assert (
        text.count("works only in the terminal's own chat (melo -X cli.chat=local)")
        == 3
    )
    assert not [r for r in gateway.requests if r[0] == "POST"]
    # /help lists the main chat's commands, not those.
    assert setup.session.enter("/help") is None
    await setup.idle()
    await _until(lambda: "/chats" in setup.out.text())
    assert "/undo" not in setup.out.text().split("/help", 1)[1]


async def test_an_approval_is_the_terminals_list_answered_through_the_gateway(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session = setup.session
    assert session.enter("approve it") is None
    await _until(lambda: session.approver.pending is not None)
    pending = session.approver.pending
    assert pending is not None and pending.request.tool_name == "shell_exec"
    assert pending.request.arguments == {"command": "ls"} and pending.offer_session
    session.approver.answer("session")
    await setup.idle()
    answered = [r for r in gateway.requests if r[1] == "/api/approvals/r1"]
    assert answered == [
        ("POST", "/api/approvals/r1", {"choice": "session", "reason": ""})
    ]
    await _until(lambda: "Ran it." in setup.out.text())


async def test_a_high_impact_call_offers_no_grant_and_starts_on_no(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, _ = chat
    session = setup.session
    assert session.enter("approve high") is None
    await _until(lambda: session.approver.pending is not None)
    pending = session.approver.pending
    assert pending is not None and not pending.offer_session and pending.selected == 2
    review = pending.request.review
    assert review is not None and review.categories == ("install",)
    session.approver.answer("deny", "not now")
    await setup.idle()


async def test_a_call_too_large_to_show_is_left_to_the_console(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session = setup.session
    assert session.enter("approve big") is None
    await _until(lambda: "too large to show here" in setup.out.text())
    assert session.approver.pending is None
    assert not [r for r in gateway.requests if r[1].startswith("/api/approvals")]
    session.stop()
    await setup.idle()


async def test_an_approval_answered_elsewhere_closes_the_list(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session = setup.session
    assert session.enter("approve it") is None
    await _until(lambda: session.approver.pending is not None)
    # The console answers it.
    await GatewayClient(
        "http://127.0.0.1:1", transport=httpx.MockTransport(gateway.handle), headers={}
    ).answer("r1", "once")
    await _until(lambda: session.approver.pending is None)
    await setup.idle()
    assert "Ran it." in setup.out.text()


async def test_esc_stops_the_gateways_turn(chat: tuple[Setup, FakeGateway]) -> None:
    setup, gateway = chat
    session = setup.session
    assert session.enter("long work") is None
    await _until(
        lambda: any(r[1].endswith("/turns") for r in gateway.requests if r[0] == "POST")
    )
    await _until(lambda: session.remote is not None and session.remote._turn is not None
                 and session.remote._turn.number is not None)  # fmt: skip
    assert session.stop() is True
    await setup.idle()
    assert ("POST", f"/api/turns/{MAIN}-t1/cancel", None) in gateway.requests
    assert "■ Stopped · /retry runs it again" in setup.out.text()


async def test_a_turn_sent_in_another_window_draws_here(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session = setup.session
    gateway.start_turn(MAIN, "long from the console", "other")
    await _until(lambda: session.state == "running")  # drawn as it runs
    # A message typed meanwhile waits for it, as during a turn of this terminal.
    assert session.enter("next") is None
    assert session.queued == "next"
    gateway._running.clear()
    gateway.finish(MAIN, 1, "completed", "Console answer.")
    await _until(lambda: "Answer to next." in setup.out.text())
    await setup.idle()
    text = setup.out.text()
    assert text.index("long from the console") < text.index("Console answer.")
    assert text.index("Console answer.") < text.index("Answer to next.")
    sent = [r[2]["text"] for r in gateway.requests if r[0] == "POST"]
    assert sent == ["next"]


async def test_cards_are_one_coloured_line_with_how_to_open_them(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    gateway.emit(MAIN, "attention.card", {"state": "done", "source_id": "t42",
                 "source_kind": "task", "title": "Fix CI", "line": "2 tests fixed."}, None)  # fmt: skip
    gateway.emit(MAIN, "attention.card", {"state": "waiting", "source_id": "s1",
                 "source_kind": "chat", "title": "Trip", "line": "Needs your approval"}, None)  # fmt: skip
    await _until(lambda: "Trip" in setup.out.text())
    text = setup.out.text()
    assert "✓ Done: Fix CI · 2 tests fixed.  melo task show t42" in text
    assert "⚠ Waits for you: Trip · Needs your approval  /chats opens it" in text


async def test_side_starts_a_side_chat_made_by_its_first_message(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session = setup.session
    assert session.enter("/side Trip plans") is None
    await _until(lambda: "first message starts it" in setup.out.text())
    await setup.idle()
    assert session.status_bar().title == "Trip plans"
    assert session.enter("plan it") is None
    await setup.idle()
    await _until(lambda: "Answer to plan it." in setup.out.text())
    created = [r for r in gateway.requests if r[:2] == ("POST", "/api/sessions")]
    assert created[0][2]["workspace"] == session.workspace
    assert (
        "PATCH",
        "/api/sessions/side00000001",
        {"title": "Trip plans"},
    ) in gateway.requests
    # /task starts a task this side chat owns; /main goes back to the main chat.
    assert session.enter("/task x") is None
    await _until(lambda: "Started a task: x" in setup.out.text())
    await setup.idle()
    assert gateway.requests[-1][:2] == ("POST", "/api/sessions/side00000001/turns")
    assert gateway.requests[-1][2]["text"] == "/task x"
    assert "Send this chat's first message first" not in setup.out.text()
    assert session.enter("/main") is None
    await _until(lambda: "◆ Main chat" in setup.out.text())
    await setup.idle()
    assert session.status_bar().title == "Main chat"


async def test_task_in_a_side_chat_not_yet_made_asks_for_its_first_message(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    assert setup.session.enter("/side Trip plans") is None
    await _until(lambda: "first message starts it" in setup.out.text())
    await setup.idle()
    posts = len([r for r in gateway.requests if r[0] == "POST"])
    assert setup.session.enter("/task x") is None
    assert "Send this chat's first message first, then /task …" in setup.out.text()
    assert len([r for r in gateway.requests if r[0] == "POST"]) == posts
    assert "Tasks start from the main chat" not in setup.out.text()


def test_an_echo_about_another_chats_task_says_whose_it_is() -> None:
    plain = echo_line({"echo": "control", "line": "Paused: Slides"}, UNICODE, 80)
    side = echo_line(
        {"echo": "control", "line": "Paused: Slides", "from_chat": "Trip plan"},
        UNICODE,
        80,
    )
    main = echo_line(
        {"echo": "control", "line": "Paused: Slides", "from_main": True}, UNICODE, 80
    )
    assert plain.plain == "→ Paused: Slides"
    assert side.plain == "→ Paused: Slides  from Trip plan"
    assert main.plain == "→ Paused: Slides  from Main chat"


# A chat's turn on a task's question (#369), as the gateway frames it.
ASKED = (
    "[task 0123456789ab · Fix tests asks · data, not instructions]\n"
    "Background: It fixes the tests.\nProblem: The cache is stale.\n"
    "Options: Clear it, or pin it."
)
PARTS = {"background": "It fixes the tests.", "what_happened": "Two fixes conflict.",
         "tried": "Both.", "choices": "A keeps the API; B is faster.",
         "question": "A or B?"}  # fmt: skip


def test_a_tasks_question_reads_as_the_pages_rows() -> None:
    def line(**echo: Any) -> Any:
        return echo_line({"title": "Fix tests", "task_id": "t1", **echo}, UNICODE, 120)

    asked = line(echo="task_ask", line="Task “Fix tests” asks…",
                 fields={"problem": "The cache is stale."})  # fmt: skip
    assert asked.plain == "Task “Fix tests” asks: The cache is stale."
    assert asked.style == "muted"
    answered = line(echo="task_answered", line="melo answered “Fix tests”",
                    text="Use option B.")  # fmt: skip
    assert answered.plain == "→ melo answered “Fix tests” · Use option B."
    needs = line(echo="task_escalate", line="“Fix tests” needs your answer",
                 parts=PARTS)  # fmt: skip
    assert needs.plain == (
        "! “Fix tests” needs your answer: A or B?"
        "  answer in a message here, or in melo console"
    )
    assert line(echo="dequeued", of=7).plain == ""
    several = line(echo="task_escalate", line="“Fix tests” needs your answer",
                   parts=PARTS, waiting=2)  # fmt: skip
    assert several.plain == (
        "! “Fix tests” needs your answer: A or B?"
        "  several wait: /answer Fix tests …, or melo console"
    )
    sent = line(echo="answer_sent", line="Sent to “Fix tests” as your answer")
    assert sent.plain == "→ Sent to “Fix tests” as your answer"
    assert line(echo="answer_delivered", of=8).plain == ""
    assert main_chat.task_question(ASKED) == ("Fix tests", "The cache is stale.")
    assert main_chat.task_question("hello") is None


async def test_a_chats_turn_on_a_tasks_question_shows_its_row_not_its_frame(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session = setup.session
    gateway.emit(MAIN, "attention.echo", {"echo": "task_ask", "task_id": "t1",
                 "title": "Fix tests", "line": "Task “Fix tests” asks…",
                 "fields": {"problem": "The cache is stale."}}, None)  # fmt: skip
    turns = gateway.turns.setdefault(MAIN, [])
    turns.append({"id": f"{MAIN}-t1", "turn": 1, "user_text": ASKED,
                  "origin": "ask_owner", "status": "running", "answer": "",
                  "started_at": 1000.0})  # fmt: skip
    gateway.emit(MAIN, "turn.started", {"turn": 1, "turn_id": f"{MAIN}-t1",
                 "user_text": ASKED, "client_request_id": None,
                 "origin": "ask_owner"}, 1)  # fmt: skip
    gateway.emit(MAIN, "run.started", {"max_steps": 8}, 1)
    await _until(lambda: session.state == "running")
    gateway.emit(MAIN, "attention.echo", {"echo": "task_answered", "task_id": "t1",
                 "title": "Fix tests", "line": "melo answered “Fix tests”",
                 "text": "Clear the cache."}, None)  # fmt: skip
    gateway.finish(MAIN, 1, "completed", "I told the task to clear the cache.")
    await setup.idle()
    text = setup.out.text()
    assert "Task “Fix tests” asks: The cache is stale." in text
    assert "→ melo answered “Fix tests” · Clear the cache." in text
    assert "I told the task to clear the cache." in text
    assert "[task" not in text and "data, not instructions" not in text
    # The recap of a reopened chat shows it the same way, and /retry leaves it.
    remote = session.remote
    assert isinstance(remote, MainChat)
    remote._cursors.clear()
    await remote.open(remote.chat)
    assert setup.out.text().count("Task “Fix tests” asks: The cache is stale.") == 2
    assert "[task" not in setup.out.text()
    assert session.enter("/retry") is None
    await _until(lambda: "Nothing to retry" in setup.out.text())
    assert not [r for r in gateway.requests if r[1].endswith("/retry")]


async def test_chats_lists_the_main_chat_first_and_opens_the_one_picked(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    gateway.sessions["side00000009"] = {"id": "side00000009", "kind": "chat", "main": False,
                                        "title": "Trip", "workspace_label": "~/w",
                                        "pending_approvals": 1}  # fmt: skip
    gateway.turns["side00000009"] = []
    setup.app.picks_from.append(1)  # type: ignore[attr-defined]
    assert setup.session.enter("/chats") is None
    await _until(lambda: "Side chat: Trip" in setup.out.text())
    await setup.idle()
    title, rows, default = setup.app.offered_from[0]  # type: ignore[attr-defined]
    assert (title, default) == ("Chats", 0)
    assert rows == [
        "Main chat (open)",
        "Trip · waits for your approval · ~/w",
        "New side chat",
    ]
    assert setup.session.status_bar().title == "Trip"


async def test_a_lost_stream_ends_the_turn_with_a_line(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session = setup.session
    assert session.enter("long work") is None
    await _until(lambda: session.remote is not None and session.remote._turn is not None
                 and session.remote._turn.number is not None)  # fmt: skip
    gateway.lose_stream = True
    await setup.idle()
    text = " ".join(setup.out.text().split())
    assert "The gateway is not answering" in text
    assert "The gateway stopped answering during this turn" in text
    assert "back in the input box" in text
    assert setup.app.buffer.text == "long work"


def test_lines_cut_to_the_width_and_keep_how_to_open() -> None:
    card = {"state": "failed", "source_id": "t1", "source_kind": "task",
            "title": "A very long title " * 5, "line": "Stopped before it finished"}  # fmt: skip
    line = card_line(card, UNICODE, 60)
    assert line.cell_len <= 60 + len("  melo task show t1")
    assert line.plain.endswith("  melo task show t1")
    assert line.plain.startswith("✗ Not finished: A very long")
    asked = echo_line({"echo": "ask", "line": "Start this as a task?"}, UNICODE, 80)
    assert asked.plain == "→ Start this as a task?  /task and your message starts it"


def test_a_task_that_ran_out_of_steps_is_a_stopped_card_not_a_failed_one() -> None:
    card = {"state": "stopped", "source_id": "t1", "source_kind": "task",
            "title": "Fix it", "line": "Remaining: run the tests"}  # fmt: skip
    line = card_line(card, UNICODE, 80)
    assert line.plain.startswith("■ Stopped: Fix it")
    assert line.plain.endswith("Remaining: run the tests  melo task show t1")
    assert "Not finished" not in line.plain


def test_an_approval_request_from_the_gateways_list() -> None:
    request = approval_request(
        {"request_id": "r", "call_id": "c", "tool_name": "file_write",
         "arguments": {"path": "/w/a.txt", "content": "x"}, "preview": "a.txt",
         "workspace_root": "/w", "review": {"verdict": "routine", "reasons": []}}
    )  # fmt: skip
    assert request.tool_name == "file_write" and request.workspace_root == "/w"
    assert request.review is not None and request.review.verdict == "routine"
    assert approval_request({"review": {"verdict": "odd"}}).review is None


# ── Finding the gateway ──


class _Gateway(http.server.BaseHTTPRequestHandler):
    """Answers ``GET /api/sessions`` as a gateway with its main chat does."""

    status = 200
    body: dict[str, Any] = {"sessions": [], "main": {"id": MAIN, "main": True}}

    def do_GET(self) -> None:  # noqa: N802 - the stdlib's name
        data = json.dumps(self.body).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextlib.contextmanager
def _serving(
    handler: type[http.server.BaseHTTPRequestHandler], port: int = 0
) -> Iterator[int]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """HOME in *tmp_path*, with a gateway token as the gateway writes it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    auth = tmp_path / ".cache" / "localmelo" / "gateway" / "auth"
    auth.mkdir(parents=True, mode=0o700)
    os.chmod(auth, 0o700)
    (auth / "token").write_text("ab" * 32)
    os.chmod(auth / "token", 0o600)
    return tmp_path


def _cfg(port: int) -> config.Config:
    cfg = config.Config()
    cfg.gateway.port = port
    return cfg


class _Process:
    def __init__(self, code: int | None = None) -> None:
        self.code = code

    def poll(self) -> int | None:
        return self.code


def _no_spawn(host: str, port: int) -> Any:
    raise AssertionError("no gateway may start here")


def test_connect_uses_the_gateway_on_the_configured_port(home: Path) -> None:
    with _serving(_Gateway) as port:
        found = main_chat.connect(_cfg(port), spawn=_no_spawn)
    assert found == Link(f"http://127.0.0.1:{port}", port)
    assert found.line() == f"Connected to the gateway on port {port}."


def test_connect_starts_a_gateway_when_none_runs(home: Path) -> None:
    port = _free_port()
    said: list[str] = []
    servers: list[Any] = []

    def spawn(host: str, at: int) -> Any:
        assert (host, at) == ("127.0.0.1", port)
        servers.append(_serving(_Gateway, at))
        servers[-1].__enter__()
        return _Process()

    try:
        found = main_chat.connect(
            _cfg(port), on_start=lambda: said.append("start"), spawn=spawn
        )
    finally:
        for server in servers:
            server.__exit__(None, None, None)
    assert said == ["start"]
    assert isinstance(found, Link) and found.started and found.port == port
    assert "Started the gateway in the background on port" in found.line()


def test_connect_says_why_when_the_gateway_stops_as_it_starts(home: Path) -> None:
    log = home / ".cache" / "localmelo" / "logs" / "gateway.log"
    log.parent.mkdir(parents=True)
    log.write_text("Starting...\n  Configuration errors: no chat model\n\n")
    found = main_chat.connect(_cfg(_free_port()), spawn=lambda h, p: _Process(1))
    assert found == Offline(
        "the gateway stopped as it started (exit 1: Configuration errors: no chat model)"
    )
    assert found.line().endswith("; this is the terminal's own chat.")


def test_connect_gives_up_on_a_gateway_that_never_answers(home: Path) -> None:
    ticks = iter(range(100))
    found = main_chat.connect(
        _cfg(_free_port()), spawn=lambda h, p: _Process(), start_s=3,
        clock=lambda: float(next(ticks)),
    )  # fmt: skip
    assert isinstance(found, Offline)
    assert found.reason == (
        "the gateway did not answer in 3 s; log: ~/.cache/localmelo/logs/gateway.log"
    )


def test_a_port_held_by_another_program_is_said(home: Path) -> None:
    class Other(_Gateway):
        status = 404
        body = {"detail": "nope"}

    with _serving(Other) as port:
        found = main_chat.connect(_cfg(port), spawn=_no_spawn)
    assert found == Offline(f"port {port} is used by another program")


def test_a_gateway_that_refuses_the_token_is_said(home: Path) -> None:
    class Refusing(_Gateway):
        status = 401
        body = {
            "error": {"code": "unauthorized", "message": "Authentication required."}
        }

    with _serving(Refusing) as port:
        found = main_chat.connect(_cfg(port), spawn=_no_spawn)
    assert found == Offline(f"the gateway on port {port} refused this computer's token")


@contextlib.contextmanager
def _owner(home: Path, pid: int) -> Iterator[None]:
    """The main chat's file held, as a running gateway holds it."""
    folder = home / ".cache" / "localmelo" / "gateway"
    fd = os.open(folder / "sessions.db.owner", os.O_RDWR | os.O_CREAT, 0o600)
    os.pwrite(fd, f"{pid} host 2026\n".encode(), 0)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        yield
    finally:
        os.close(fd)


def test_connect_finds_the_gateway_that_holds_the_main_chat_on_its_own_port(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _serving(_Gateway) as other, _owner(home, 4242):
        monkeypatch.setattr(
            main_chat, "listening_ports", lambda pid: [other] if pid == 4242 else []
        )
        found = main_chat.connect(_cfg(_free_port()), spawn=_no_spawn)
    assert found == Link(f"http://127.0.0.1:{other}", other)


def test_connect_never_starts_a_second_gateway_on_the_same_file(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter(range(100))
    monkeypatch.setattr(main_chat, "listening_ports", lambda pid: [])
    with _owner(home, 4242):
        assert main_chat.store_owner() == 4242
        found = main_chat.connect(
            _cfg(_free_port()), spawn=_no_spawn, clock=lambda: float(next(ticks))
        )
    assert found == Offline(
        "another gateway (pid 4242) holds it on a port melo cannot find"
    )
    assert main_chat.store_owner() is None  # released: nobody holds it


def _published(port: int, gateway_id: str) -> None:
    """The endpoint file as a running gateway (this process, alive) writes it."""
    gateway_endpoint.write(
        gateway_endpoint.Endpoint(
            pid=os.getpid(), host="127.0.0.1", port=port, instance="",
            id=gateway_id, started=1.0,
        )
    )  # fmt: skip


class _Named(_Gateway):
    """A gateway whose page carries its endpoint file's id."""

    gateway_id = "c" * 32

    def end_headers(self) -> None:
        self.send_header(gateway_endpoint.GATEWAY_HEADER, self.gateway_id)
        super().end_headers()


def test_connect_finds_the_gateway_the_endpoint_file_names_first(home: Path) -> None:
    with _serving(_Named) as named, _serving(_Gateway) as configured:
        _published(named, _Named.gateway_id)
        found = main_chat.connect(_cfg(configured), spawn=_no_spawn)
    assert found == Link(f"http://127.0.0.1:{named}", named)


def test_connect_ignores_an_endpoint_file_whose_port_says_otherwise(
    home: Path,
) -> None:
    with _serving(_Named) as named, _serving(_Gateway) as configured:
        _published(named, "d" * 32)  # a newer gateway reuses the port
        found = main_chat.connect(_cfg(configured), spawn=_no_spawn)
    assert found == Link(f"http://127.0.0.1:{configured}", configured)


def test_the_chat_setting_and_its_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from argparse import Namespace

    from localmelo.support.cli.launch import Choice, wants_main_chat
    from localmelo.support.cli.views import BackendInfo

    cfg = config.Config()
    choice = Choice(BackendInfo("vllm", "vLLM", "m", "http://x/v1"), cfg)
    plain = Namespace(yolo=False, continue_session=False, resume=None)
    monkeypatch.delenv("LOCALMELO_CHAT")
    assert wants_main_chat(plain, choice, "main")
    assert not wants_main_chat(plain, choice, "local")  # -X cli.chat=local
    assert not wants_main_chat(Namespace(yolo=True), choice, "main")
    assert not wants_main_chat(Namespace(resume=""), choice, "main")
    assert not wants_main_chat(plain, Choice(choice.backend, None), "main")  # --url
    monkeypatch.setenv("LOCALMELO_CHAT", "local")
    assert not wants_main_chat(plain, choice, "main")
    assert config.Config().cli.chat == "main"
    bad = config.Config()
    bad.cli.chat = "both"
    assert any("[cli] chat" in error for error in bad.cli_errors())


async def test_answer_goes_to_the_open_chat_as_a_message(
    chat: tuple[Setup, FakeGateway],
) -> None:
    setup, gateway = chat
    session = setup.session
    assert session.enter("/answer Fix tests: B") is None
    await setup.idle()
    sent = [r[2]["text"] for r in gateway.requests if r[0] == "POST"]
    assert sent == ["/answer Fix tests: B"]
