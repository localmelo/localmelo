from __future__ import annotations

import os
from pathlib import Path

import pytest

from localmelo.melo.contracts.events import AgentEvent
from localmelo.support.cli.worker_inbox import MAX_INBOX_BYTES, Inbox, InboxError


def test_missing_then_complete_messages_and_partial_utf8(tmp_path: Path) -> None:
    path = tmp_path / "inbox"
    inbox = Inbox(path)
    assert inbox.poll() == []
    path.write_bytes(b"first\n\xe2\x82")
    assert inbox.poll() == [(6, "first")]
    inbox.accept(6)
    with path.open("ab") as file:
        file.write(b"\xac\nsecond\npartial")
    assert inbox.poll() == [(10, "€"), (17, "second")]
    inbox.accept(10)
    inbox.accept(17)
    assert inbox.poll() == []
    with path.open("ab") as file:
        file.write(b"!\n")
    assert inbox.poll() == [(26, "partial!")]
    inbox.accept(26)
    assert inbox.poll() == []


def test_unaccepted_message_is_not_consumed(tmp_path: Path) -> None:
    path = tmp_path / "inbox"
    path.write_text("one\ntwo\n", encoding="utf-8")
    inbox = Inbox(path)
    assert inbox.poll() == [(4, "one"), (8, "two")]
    inbox.accept(4)
    assert inbox.poll() == [(8, "two")]
    with pytest.raises(InboxError):
        inbox.accept(4)


@pytest.mark.parametrize("content", [b"\n", b" \n", b"\xff\n", b"x" * 16385 + b"\n"])
def test_invalid_complete_message_fails(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "inbox"
    path.write_bytes(content)
    with pytest.raises(InboxError):
        Inbox(path).poll()


def test_overlong_partial_and_file_fail(tmp_path: Path) -> None:
    path = tmp_path / "inbox"
    path.write_bytes(b"x" * 16385)
    with pytest.raises(InboxError):
        Inbox(path).poll()
    path.write_bytes(b"a" * (MAX_INBOX_BYTES + 1))
    with pytest.raises(InboxError):
        Inbox(path).poll()


@pytest.mark.parametrize("change", ["truncate", "rewrite", "replace", "delete"])
def test_changed_inbox_fails(tmp_path: Path, change: str) -> None:
    path = tmp_path / "inbox"
    path.write_bytes(b"first\npartial")
    inbox = Inbox(path)
    inbox.poll()
    inbox.accept(6)
    if change == "truncate":
        path.write_bytes(b"first\n")
    elif change == "rewrite":
        path.write_bytes(b"first\nchanged")
    elif change == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"first\npartial\n")
        replacement.replace(path)
    else:
        path.unlink()
    with pytest.raises(InboxError):
        inbox.poll()


def test_refuses_symlink_and_nonregular_file(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("task\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for path in (link, fifo, tmp_path):
        with pytest.raises(InboxError):
            Inbox(path).poll()


@pytest.mark.parametrize("fixed_prefix", [False, True])
async def test_agent_accepts_at_boundaries_without_adjacent_users(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_prefix: bool
) -> None:
    import json

    import httpx

    from localmelo.melo.agent import Agent, AgentOptions
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "inbox"
    path.write_text("first instruction\n", encoding="utf-8")
    requests = []
    events: list[AgentEvent] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            with path.open("ab") as file:
                file.write(b"second instruction\n")
            text = ""
        else:
            text = "Done"
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": text}}]}
        )

    agent = Agent(
        llm=OpenAICompatLLM(
            base_url="http://llm.test/v1",
            model="stub",
            transport=httpx.MockTransport(reply),
        ),
        options=AgentOptions(
            memory=False, max_steps=2, attempt_policy="none", fixed_prefix=fixed_prefix
        ),
    )
    try:
        result = await agent.run_task(
            "original task", observe=events.append, inbox=Inbox(path)
        )
    finally:
        await agent.close()
    assert result.status == "completed"
    assert len(requests) == 2
    for request in requests:
        messages = request["messages"]
        assert all(
            not (left["role"] == right["role"] == "user")
            for left, right in zip(messages, messages[1:], strict=False)
        )
    assert "first instruction" in json.dumps(requests[0])
    assert "second instruction" not in json.dumps(requests[0])
    assert "second instruction" in json.dumps(requests[1])
    accepted = [event for event in events if event.type == "inbox.accepted"]
    assert [event.data["text"] for event in accepted] == [
        "first instruction",
        "second instruction",
    ]
    assert [event.data["offset"] for event in accepted] == [18, 37]


async def test_inbox_after_tool_keeps_tool_call_result_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import httpx

    from localmelo.melo.agent import Agent, AgentOptions
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "inbox"
    requests = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            path.write_text("after tool\n", encoding="utf-8")
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {
                            "name": "python_exec",
                            "arguments": '{"code":"print(1)"}',
                        },
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "Done"}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    agent = Agent(
        llm=OpenAICompatLLM(
            base_url="http://llm.test/v1",
            model="stub",
            transport=httpx.MockTransport(reply),
        ),
        options=AgentOptions(memory=False, max_steps=2, attempt_policy="none"),
    )
    try:
        result = await agent.run_task("calculate", inbox=Inbox(path))
    finally:
        await agent.close()
    assert result.status == "completed"
    messages = requests[1]["messages"]
    tool_index = next(i for i, msg in enumerate(messages) if msg["role"] == "tool")
    assert messages[tool_index - 1]["role"] == "assistant"
    assert (
        messages[tool_index - 1]["tool_calls"][0]["id"]
        == messages[tool_index]["tool_call_id"]
    )
    assert "after tool" in messages[tool_index + 1]["content"]


async def test_inbox_batch_that_cannot_fit_is_not_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    from localmelo.melo.agent import Agent, AgentOptions
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "inbox"
    path.write_bytes((b"x" * 11000 + b"\n") * 10)
    events: list[AgentEvent] = []
    transport = httpx.MockTransport(lambda request: pytest.fail("model must not run"))
    agent = Agent(
        llm=OpenAICompatLLM(
            base_url="http://llm.test/v1", model="stub", transport=transport
        ),
        options=AgentOptions(memory=False),
    )
    source = Inbox(path)
    try:
        result = await agent.run_task("task", observe=events.append, inbox=source)
    finally:
        await agent.close()
    assert result.status != "completed"
    assert source.offset == 0
    assert not [event for event in events if event.type == "inbox.accepted"]


async def test_inbox_is_not_acknowledged_if_context_budget_rejects_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    from localmelo.melo.agent import Agent, AgentOptions
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "inbox"
    path.write_text("x" * 1000 + "\n", encoding="utf-8")
    events: list[AgentEvent] = []
    agent = Agent(
        llm=OpenAICompatLLM(
            base_url="http://llm.test/v1",
            model="stub",
            transport=httpx.MockTransport(
                lambda request: pytest.fail("model must not run")
            ),
        ),
        options=AgentOptions(
            memory=False, context_window=100, output_reserve=10, safety_margin=10
        ),
    )
    source = Inbox(path)
    try:
        result = await agent.run_task("task", observe=events.append, inbox=source)
    finally:
        await agent.close()
    assert result.status != "completed"
    assert source.offset == 0
    assert not [event for event in events if event.type == "inbox.accepted"]


def test_external_stream_omits_inbox_text(tmp_path: Path) -> None:
    import io
    import json

    from localmelo.support.cli.worker_contract import WorkerObserver

    out = io.StringIO()
    observer = WorkerObserver(out, str(tmp_path), stream=True)
    observer(
        AgentEvent(
            type="inbox.accepted",
            run_id="r",
            seq=1,
            ts=0,
            data={"offset": 5, "text": "private instruction", "step": 1, "attempt": 1},
        )
    )
    observer.close()
    data = json.loads(out.getvalue())["event"]["data"]
    assert data == {"offset": 5, "step": 1, "attempt": 1}


async def test_saved_worker_records_redacted_inbox_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    from localmelo.melo.agent import Agent, AgentOptions
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
    from localmelo.support.sessions.store import SessionStore
    from localmelo.support.sessions.worker import run_worker

    monkeypatch.setenv("HOME", str(tmp_path))
    secret = "sk-" + "f00dfeed12" * 3
    path = tmp_path / "inbox"
    path.write_text(f"Remember {secret}\n", encoding="utf-8")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
        )
    )
    agent = Agent(
        llm=OpenAICompatLLM(
            base_url="http://llm.test/v1", model="stub", transport=transport
        ),
        options=AgentOptions(memory=False),
    )
    try:
        result = await run_worker(
            agent, "Task", title="", parent_run=None, inbox=Inbox(path)
        )
        assert result.status == "completed"
    finally:
        await agent.close()
    store = SessionStore(writer="cli")
    try:
        sessions = await store.list_sessions(kinds=("worker",))
        events = await store.list_events(sessions[0].id)
        accepted = [e.event for e in events if e.event["type"] == "inbox.accepted"]
        assert len(accepted) == 1
        assert "[redacted]" in accepted[0]["data"]["text"]
        assert secret not in accepted[0]["data"]["text"]
    finally:
        await store.aclose()


@pytest.mark.parametrize("save", [False, True])
def test_cli_brief_inbox_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    save: bool,
) -> None:
    import json
    import sys

    import httpx

    import localmelo.__main__ as cli
    from localmelo.melo.agent import Agent, AgentOptions
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "brief").write_text("Task", encoding="utf-8")
    (tmp_path / "inbox").write_text("Instruction\n", encoding="utf-8")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
        )
    )
    monkeypatch.setattr(
        cli,
        "_build_direct_mode_agent",
        lambda *args, **kwargs: Agent(
            llm=OpenAICompatLLM(
                base_url="http://llm.test/v1", model="stub", transport=transport
            ),
            options=AgentOptions(memory=False),
        ),
    )
    args = [
        "melo",
        "--brief",
        "brief",
        "--inbox",
        "inbox",
        "--output-format",
        "stream-json",
    ]
    if save:
        args.append("--save-session")
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    frames = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    acknowledgements = [
        f["event"]["data"]
        for f in frames
        if f.get("type") == "event" and f["event"]["type"] == "inbox.accepted"
    ]
    assert acknowledgements == [{"offset": 12, "step": 1, "attempt": 1}]
    assert frames[-1]["result"]["status"] == "completed"
