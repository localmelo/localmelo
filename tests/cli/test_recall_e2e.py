"""Recall reaches a real planning request, end to end.

The benchmark's ``retry`` stub exercises R5 through the real CLI: ten steps
fill the window, the first reflection continues, and the early note that
described the failure leaves the window in the second attempt. Without the
step trigger ranking a superseded note of an evicted turn, and without the
note's ``found:`` items in the state query, every later step is empty. These
runs go through ``melo``'s own JSON line and the stub server's recorded
request bodies, not through ``Agent`` directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.memory.notes import RECALL_START
from localmelo.tests.fixtures.stub_llm import StubServer

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX subprocesses and shell commands"
)

TASK = "track down the parser build failure"
EARLY = "Step 1: build.log says the error is in module parser."
LATE = "Step 12: back to the parser module error from build.log."
PROXY_VARS = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}


def _echo_reply(step: int) -> dict[str, Any]:
    return {
        "reasoning": EARLY if step == 1 else f"Step {step}: keep echoing.",
        "tool_calls": [
            {"name": "shell_exec", "arguments": {"command": f"echo step {step}"}}
        ],
    }


def _continue() -> dict[str, Any]:
    return {
        "content": json.dumps(
            {
                "summary": "ten steps done; build.log is still unread",
                "next_promising_directions": ["read build.log for the parser error"],
                "recommended_action": "continue",
                "task_still_feasible": True,
                "next_step_is_concrete": True,
                "next_step_is_novel": True,
            }
        )
    }


def _answer() -> dict[str, Any]:
    return {"content": "Done: build.log names the parser module."}


def _step_scenario() -> dict[str, Any]:
    """Ten steps, the first reflection continues, then the early note is gone."""
    replies: list[dict[str, Any]] = [_echo_reply(i) for i in range(1, 11)]
    replies.append(_continue())
    replies.append(_echo_reply(11))
    replies.append(
        {
            "reasoning": LATE,
            "tool_calls": [
                {"name": "shell_exec", "arguments": {"command": "echo step 12"}}
            ],
        }
    )
    replies.append(_answer())
    return {"models": ["stub-model"], "sequential": True, "replies": replies}


def _tool_scenario() -> dict[str, Any]:
    """The model asks the ``recall`` tool once the early note has left."""
    replies: list[dict[str, Any]] = [_echo_reply(i) for i in range(1, 11)]
    replies.append(_continue())
    replies.append(_echo_reply(11))
    replies.append(
        {
            "tool_calls": [
                {
                    "name": "recall",
                    "arguments": {"query": "parser module error in build.log"},
                }
            ]
        }
    )
    replies.append(_answer())
    return {"models": ["stub-model"], "sequential": True, "replies": replies}


@pytest.fixture
def stub() -> Iterator[StubServer]:
    server = StubServer({"step": _step_scenario(), "tool": _tool_scenario()}).start()
    try:
        yield server
    finally:
        server.close()


def _env(home: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LOCALMELO_") and key.lower() not in PROXY_VARS
    }
    env.update(HOME=str(home), NO_PROXY="127.0.0.1,localhost")
    return env


def _run(tmp_path: Path, stub: StubServer, scenario: str, *args: str) -> dict[str, Any]:
    home = tmp_path / f"{scenario}-home"
    home.mkdir()
    base_url = stub.base_url.replace("/v1", f"/s/{scenario}/v1")
    argv = [
        sys.executable,
        "-I",
        "-m",
        "localmelo",
        "--base-url",
        base_url,
        "--reasoning",
        "compress",
        "--retriever",
        "bm25",
        "--memory",
        "off",
        "--output-format",
        "json",
        *args,
        "--",
        TASK,
    ]
    done = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        env=_env(home),
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.count("\n") == 1, done.stdout
    return json.loads(done.stdout)


def _blocks(requests: list[dict[str, Any]], role: str) -> list[str]:
    """The contents of *role* messages that carry a recall block."""
    return [
        message["content"]
        for body in requests
        for message in body.get("messages", [])
        if message.get("role") == role
        and RECALL_START in str(message.get("content") or "")
    ]


def test_a_step_recalls_a_superseded_note_the_window_left(
    tmp_path: Path, stub: StubServer
) -> None:
    result = _run(tmp_path, stub, "step")
    stats = result["memory_stats"]
    assert stats["recalls"] >= 1
    assert stats["recalls_by_trigger"]["step"] >= 1
    assert stats["recall_empty"] >= 1
    blocks = _blocks(stub.requests, "tool")
    assert blocks
    assert any("[n1]" in block and "parser" in block for block in blocks)


def test_message_placement_puts_the_step_block_in_a_user_message(
    tmp_path: Path, stub: StubServer
) -> None:
    result = _run(tmp_path, stub, "step", "--recall-placement", "message")
    assert result["memory_stats"]["recalls"] >= 1
    blocks = _blocks(stub.requests, "user")
    assert blocks
    assert any("[n1]" in block and "parser" in block for block in blocks)


def test_the_recall_tool_returns_the_superseded_note(
    tmp_path: Path, stub: StubServer
) -> None:
    result = _run(tmp_path, stub, "tool", "--recall-at", "tool")
    stats = result["memory_stats"]
    assert stats["recalls_by_trigger"]["tool"] >= 1
    blocks = _blocks(stub.requests, "tool")
    assert any("[n1]" in block and "parser" in block for block in blocks)
