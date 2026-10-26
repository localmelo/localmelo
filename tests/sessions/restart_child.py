"""A gateway process that dies at a chosen point, for test_restart_process.py.

Usage: ``python -I restart_child.py SCENARIO PATH WORKSPACE``

It opens a gateway-writer store at PATH and runs turns of a scripted model
in WORKSPACE through a SessionRunner, then ends with ``os._exit(73)`` at
the SCENARIO's crash point: no finally block, no clean-up. Before it dies
it prints one JSON line: the session id, the turn id and the id of the
last event its store's listener saw.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import deque
from collections.abc import Iterable
from typing import Any

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolCall, ToolDef
from localmelo.support.sessions.approvals import RememberingPolicy
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore, SessionStoreError

KEY = "8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07"
TEXT = "Crash during this request"


class Model(BaseLLMProvider):
    """Scripted replies, then a request that never returns."""

    def __init__(self, replies: Iterable[Message] = ()) -> None:
        self.replies = deque(replies)
        self.blocked = asyncio.Event()

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        if self.replies:
            return self.replies.popleft()
        self.blocked.set()
        await asyncio.Event().wait()
        raise AssertionError("not reached")

    async def close(self) -> None:
        pass


def tool(name: str, arguments: dict[str, Any]) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(name, arguments, id="call_1")],
    )


def die(store_state: dict[str, Any]) -> None:
    print(json.dumps(store_state), flush=True)
    os._exit(73)


async def main(scenario: str, path: str, root: str) -> None:
    store = SessionStore(path, writer="gateway")
    state: dict[str, Any] = {"last_id": 0}

    def heard(event: Any) -> None:
        state["last_id"] = event.id

    store.add_listener(heard)
    replies: list[Message] = [Message(role="assistant", content="AKITA-RECOVERY-527")]
    if scenario in ("c2_approval", "arms"):
        replies = [tool("shell_exec", {"command": "echo hi"})]
    if scenario == "arms":
        replies = [
            tool("file_read", {"path": "notes.txt"}),
            Message(role="assistant", content="Buy milk."),
        ]
    if scenario in ("c2", "c3_cancelled", "shutdown"):
        replies = []  # the model blocks: the run is under way
    model = Model(replies)
    agent = Agent(
        llm=model, options=AgentOptions(workspace_root=root, record_answers=True)
    )
    await agent.start()
    session = await store.create_session()
    state["session_id"] = session.id
    runner = SessionRunner(agent, store, session.id)

    original_append = store.append_events
    batches = 0

    async def append(*args: Any, **kwargs: Any) -> Any:
        nonlocal batches
        stored = await original_append(*args, **kwargs)
        batches += 1
        types = [event["type"] for event in args[2]]
        if scenario == "c2" and batches == 1:
            die(state)
        if scenario == "c2_approval" and "approval.requested" in types:
            die(state)
        if scenario in ("c3", "c3_cancelled") and "run.finished" in types:
            die(state)
        return stored

    store.append_events = append  # type: ignore[method-assign]

    def before_commit(label: str) -> None:
        if scenario == "c5_begin" and label == "begin_turn":
            die(state)
        if scenario == "c5_finish" and label == "finish_turn":
            die(state)
        if scenario == "c5_append" and label == "append_events":
            die(state)

    store._before_commit = before_commit  # type: ignore[method-assign]

    if scenario == "arms":
        await runner.run_turn("Read my notes")
        turn = await runner.start_turn("Now wait for ever")
        state["turn_id"] = turn.id
        await model.blocked.wait()
        await runner.flush()
        die(state)

    if scenario == "shutdown":
        turn = await runner.start_turn(TEXT, client_request_id=KEY)
        state["turn_id"] = turn.id
        await model.blocked.wait()
        agent.begin_exit("Session closed", within_s=5.0)
        await runner.close(reason="Session closed", cause="shutdown")
        # A turn whose end comes too late: the store is closed by then.
        other = await store.begin_turn(
            None, "late", holder="h:1:x", new_session=NewSession()
        )
        state["late_session_id"] = other.session_id
        await store.aclose()
        try:
            await store.finish_turn(
                other.session_id,
                1,
                holder="h:1:x",
                status="completed",
                cause="run",
                result=None,
                answer="late",
            )
        except SessionStoreError as exc:
            state["late_error"] = str(exc)
        die(state)

    approval = None
    if scenario == "c2_approval":

        async def never(request: ApprovalRequest) -> ApprovalDecision:
            await asyncio.Event().wait()
            raise AssertionError("not reached")

        approval = RememberingPolicy(runner.observe(never))
    turn = await runner.start_turn(TEXT, approval=approval, client_request_id=KEY)
    state["turn_id"] = turn.id
    if scenario == "c1":
        die(state)  # the turn row is committed; run.started is not stored yet
    if scenario == "c3_cancelled":
        await model.blocked.wait()
        runner.cancel()
    await runner.wait()
    if scenario == "c4":
        die(state)
    await asyncio.Event().wait()


async def recover_twice(path: str) -> None:
    """Make two running turns, then die in the recovery of the second."""
    store = SessionStore(path, writer="gateway")
    ids = []
    for text in ("one", "two"):
        turn = await store.begin_turn(
            None, text, holder="h:1:x", new_session=NewSession()
        )
        ids.append(turn.session_id)
    print(json.dumps({"session_ids": ids}), flush=True)
    os._exit(73)


async def die_in_recovery(path: str) -> None:
    store = SessionStore(path, writer="gateway")
    recovered = 0

    def before_commit(label: str) -> None:
        nonlocal recovered
        if label == "recover":
            recovered += 1
            if recovered == 2:
                print(json.dumps({"recovered": 1}), flush=True)
                os._exit(73)

    store._before_commit = before_commit  # type: ignore[method-assign]
    await store.open()


if __name__ == "__main__":
    if sys.argv[1] == "c6_make":
        asyncio.run(recover_twice(sys.argv[2]))
    elif sys.argv[1] == "c6_recover":
        asyncio.run(die_in_recovery(sys.argv[2]))
    else:
        asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3]))
