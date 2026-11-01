"""API v2 details: tools, grants, memory and runs, and the gate behind a turn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from localmelo.support.gateway.chat_gate import ChatGate, GatedLLM
from localmelo.support.gateway.console.registry import default_builder
from localmelo.tests.sessions.helpers import answer, tool_call

from .conftest import make_cfg
from .test_turn_writes import (
    _create,
    _post_turn,
    _wait_pending,
    _wait_turn,
    console_with,
)

# ── Tools ──


async def test_tools_list_each_tool_and_whether_it_is_granted(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ws" / "todo.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "Write the todo file")
        session_id = body["session"]["id"]
        [pending] = await _wait_pending(client, session_id)
        await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "session"}
        )
        await _wait_turn(client, session_id)

        listed = (await client.get(f"/api/sessions/{session_id}/tools")).json()

        assert listed["mcp_config_label"] == ""
        # MCP in the console is deferred: the tab says so (decision 5B).
        assert listed["mcp_error"] == "MCP is not built in the console yet"
        by_name = {tool["name"]: tool for tool in listed["tools"]}
        # A console session is gateway mode: no shell_exec, but the other
        # built-ins stay, the panel still asks about python_exec, and the
        # read-only listing never asks.
        assert set(by_name) == {"file_read", "file_write", "list_files", "python_exec"}
        assert by_name["list_files"]["requires_approval"] is False
        assert by_name["file_write"]["granted"] is True
        assert by_name["python_exec"]["granted"] is False
        assert by_name["python_exec"]["requires_approval"] is True
        assert by_name["python_exec"]["source"] == "builtin"
        assert by_name["python_exec"]["parameters"]["type"] == "object"


async def test_tools_open_a_session_its_agent_was_kept_from(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hello")) as (console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)
        await console.sessions.keep(session_id)
        assert (await client.get(f"/api/sessions/{session_id}")).json()["agent"] == (
            "paused"
        )

        listed = await client.get(f"/api/sessions/{session_id}/tools")

        assert listed.status_code == 200
        assert (await client.get(f"/api/sessions/{session_id}")).json()["agent"] == (
            "open"
        )


async def test_an_unknown_session_has_no_tools(tmp_path: Path) -> None:
    async with console_with(tmp_path) as (_console, client, _):
        response = await client.get("/api/sessions/0123456789ab/tools")
        assert response.status_code == 404


# ── Grants ──


async def test_forgetting_a_grant_takes_it_back(tmp_path: Path) -> None:
    target = tmp_path / "ws" / "todo.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "Write the todo file")
        session_id = body["session"]["id"]
        [pending] = await _wait_pending(client, session_id)
        await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "session"}
        )
        await _wait_turn(client, session_id)
        assert (await client.get(f"/api/sessions/{session_id}")).json()["grants"] == (
            ["file_write"]
        )

        forgotten = await client.delete(f"/api/sessions/{session_id}/grants/file_write")

        assert forgotten.status_code == 204
        assert forgotten.content == b""
        detail = (await client.get(f"/api/sessions/{session_id}")).json()
        assert detail["grants"] == [] and detail["granted_at"] == {}
        listed = (await client.get(f"/api/sessions/{session_id}/tools")).json()
        by_name = {tool["name"]: tool for tool in listed["tools"]}
        assert by_name["file_write"]["granted"] is False


async def test_forgetting_an_unknown_grant_is_404(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("hello")) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)

        response = await client.delete(f"/api/sessions/{session_id}/grants/shell_exec")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


async def test_forgetting_every_grant_takes_them_back(tmp_path: Path) -> None:
    target = tmp_path / "ws" / "todo.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "Write the todo file")
        session_id = body["session"]["id"]
        [pending] = await _wait_pending(client, session_id)
        await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "session"}
        )
        await _wait_turn(client, session_id)
        assert (await client.get(f"/api/sessions/{session_id}")).json()["grants"] == (
            ["file_write"]
        )

        forgotten = await client.delete(f"/api/sessions/{session_id}/grants")

        assert forgotten.status_code == 204
        detail = (await client.get(f"/api/sessions/{session_id}")).json()
        assert detail["grants"] == [] and detail["granted_at"] == {}
        listed = (await client.get(f"/api/sessions/{session_id}/tools")).json()
        assert all(tool["granted"] is False for tool in listed["tools"])


async def test_forgetting_every_grant_is_204_even_without_any(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hello")) as (_console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]

        response = await client.delete(f"/api/sessions/{session_id}/grants")

        assert response.status_code == 204
        assert response.content == b""


# ── Memory ──


async def test_memory_never_opens_a_session(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("hello")) as (console, client, _):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)
        # A session this gateway never opened, with a turn in its file.
        seeded = await seed_session(console, tmp_path)

        response = await client.get(f"/api/sessions/{seeded}/memory")

        assert response.status_code == 200
        body = response.json()
        assert body["agent"] == "not_loaded"
        assert body["working"] == []
        assert body["working_max"] == 64  # the local backend's default
        assert body["long_term"] == {"enabled": False, "backend": "none"}


async def test_memory_reports_the_cap_of_the_open_agent(tmp_path: Path) -> None:
    async with console_with(tmp_path, answer("hello"), window_messages=32) as (
        _console,
        client,
        _,
    ):
        body = await _create(client, tmp_path, "Hi there")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)

        memory = (await client.get(f"/api/sessions/{session_id}/memory")).json()

        assert memory["agent"] == "open"
        assert memory["working_max"] == 32


async def test_memory_of_a_session_not_open_reports_the_settings_cap(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hello")) as (console, client, _):
        console.cfg.agent.window_messages = 32
        seeded = await seed_session(console, tmp_path)

        memory = (await client.get(f"/api/sessions/{seeded}/memory")).json()

        assert memory["agent"] == "not_loaded"
        assert memory["working_max"] == 32


async def test_memory_of_a_session_not_open_reports_a_cloud_backends_default(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("hello")) as (console, client, _):
        console.cfg.chat_backend = "openai"
        seeded = await seed_session(console, tmp_path)

        memory = (await client.get(f"/api/sessions/{seeded}/memory")).json()

        assert memory["working_max"] == 20


async def test_memory_shows_the_conversation_without_thinking(
    tmp_path: Path,
) -> None:
    thinking = "<think>weigh the options</think>The answer is 42."
    async with console_with(tmp_path, answer(thinking)) as (_console, client, _):
        body = await _create(client, tmp_path, "What is it?")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)

        memory = (await client.get(f"/api/sessions/{session_id}/memory")).json()

        assert memory["agent"] == "open"
        roles = [row["role"] for row in memory["working"]]
        assert "user" in roles and "assistant" in roles
        answers = [
            row["text"] for row in memory["working"] if row["role"] == "assistant"
        ]
        assert any("The answer is 42." in text for text in answers)
        assert all("weigh the options" not in text for text in answers)
        assert all("images" in row and "tool_calls" in row for row in memory["working"])


async def test_memory_shows_the_kept_copy_and_cuts_long_texts(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("x" * 2_500)) as (console, client, _):
        body = await _create(client, tmp_path, "How long?")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)
        await console.sessions.keep(session_id)

        memory = (await client.get(f"/api/sessions/{session_id}/memory")).json()

        assert memory["agent"] == "paused"
        [reply] = [row for row in memory["working"] if row["role"] == "assistant"]
        assert reply["text"].endswith("… (+500 chars)")
        assert len(reply["text"]) == 2_500 - 500 + len("… (+500 chars)")


async def test_memory_redacts_a_secret(tmp_path: Path) -> None:
    secret = "sk-" + "a" * 24
    async with console_with(tmp_path, answer(f"the key is {secret}")) as (
        _console,
        client,
        _,
    ):
        body = await _create(client, tmp_path, "Which key?")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)

        memory = (await client.get(f"/api/sessions/{session_id}/memory")).json()

        [reply] = [row for row in memory["working"] if row["role"] == "assistant"]
        assert secret not in reply["text"]
        assert "[redacted]" in reply["text"]


# ── Runs ──


async def test_runs_list_finished_turns_without_their_answer(
    tmp_path: Path,
) -> None:
    async with console_with(tmp_path, answer("first"), answer("second")) as (
        _console,
        client,
        _,
    ):
        body = await _create(client, tmp_path, "One")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)
        await client.post(f"/api/sessions/{session_id}/turns", json={"text": "Two"})
        await _wait_turn(client, session_id)

        listed = (await client.get(f"/api/sessions/{session_id}/runs")).json()

        assert [run["turn"] for run in listed["runs"]] == [1, 2]
        assert listed["has_earlier"] is False
        [newest] = [run for run in listed["runs"] if run["turn"] == 2]
        assert newest["status"] == "completed"
        assert newest["result"]["status"] == "completed"
        assert "result" not in newest["result"]
        assert "user_text" not in newest


async def test_runs_report_whether_earlier_turns_exist(tmp_path: Path) -> None:
    async with console_with(
        tmp_path, answer("one"), answer("two"), answer("three")
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "One")
        session_id = body["session"]["id"]
        await _wait_turn(client, session_id)
        await _post_turn(client, session_id, "Two")
        await _wait_turn(client, session_id)
        await _post_turn(client, session_id, "Three")
        await _wait_turn(client, session_id)

        page = (await client.get(f"/api/sessions/{session_id}/runs?limit=2")).json()
        assert [run["turn"] for run in page["runs"]] == [2, 3]
        assert page["has_earlier"] is True
        older = (
            await client.get(f"/api/sessions/{session_id}/runs?limit=2&before=2")
        ).json()
        assert [run["turn"] for run in older["runs"]] == [1]
        assert older["has_earlier"] is False


async def test_runs_leave_a_running_turn_out(tmp_path: Path) -> None:
    target = tmp_path / "ws" / "todo.md"
    async with console_with(
        tmp_path,
        tool_call("file_write", {"path": str(target), "content": "# TODO"}),
        answer("done"),
    ) as (_console, client, _):
        body = await _create(client, tmp_path, "Write the todo file")
        session_id = body["session"]["id"]
        [pending] = await _wait_pending(client, session_id)

        listed = (await client.get(f"/api/sessions/{session_id}/runs")).json()

        assert listed["runs"] == []
        await client.post(
            f"/api/approvals/{pending['request_id']}", json={"choice": "deny"}
        )
        await _wait_turn(client, session_id)


# ── The gate ──


async def test_default_builder_puts_the_gate_in_front_of_the_model(
    tmp_path: Path,
) -> None:
    gate = ChatGate(policy="off", call_log_dir=None)
    agent = await default_builder(make_cfg(), str(tmp_path), None, "3f9c2a7b1d4e", gate)

    try:
        # The Chat is built from the provider the hook returned, so every
        # model call of this session goes through the gate, tagged with it.
        assert isinstance(agent._llm, GatedLLM)
        assert agent.chat.llm is agent._llm
        assert agent._llm.client.gate is gate
        assert agent._llm.client.session_id == "3f9c2a7b1d4e"
    finally:
        await agent.close()


async def test_default_builder_without_a_gate_wraps_nothing(
    tmp_path: Path,
) -> None:
    agent = await default_builder(make_cfg(), str(tmp_path), None, "3f9c2a7b1d4e")

    try:
        assert not isinstance(agent._llm, GatedLLM)
        # The production builder, not only the test one, is gateway mode.
        assert agent.options.mode == "gateway"
        names = {tool.name for tool in agent.list_tools()}
        # web_fetch is on by default ([tools] web_fetch).
        assert names == {
            "file_read",
            "file_write",
            "list_files",
            "python_exec",
            "web_fetch",
        }
    finally:
        await agent.close()


async def seed_session(console: Any, tmp_path: Path) -> str:
    """A session in the store that this console has never opened."""
    from .conftest import seed_turn

    workspace = tmp_path / "seeded"
    workspace.mkdir(exist_ok=True)
    turn = await seed_turn(console.store, str(workspace))
    return turn.session_id
