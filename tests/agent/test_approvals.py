"""Approval policies in a run (Agent.run_task's approval) — no real model.

Every run is scripted. ``deploy`` is a custom tool that requires approval
and ``status`` one that does not, both offered on every step
(``tool_exposure="all"``); history is in memory.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import deque
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.checker import DEFAULT_BUILTINS
from localmelo.melo.contracts.approvals import (
    AllowAllPolicy,
    ApprovalDecision,
    ApprovalRequest,
    DenyAllPolicy,
)
from localmelo.melo.contracts.events import EVENT_FIELDS, AgentEvent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.contracts.review import ActionReview
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.schema import CALL_NOT_STARTED, Message, ToolCall, ToolDef

# The texts the model reads, pinned literally.
ENDING = (
    "Do not run it again or reach the same result another way unless the user asks."
)
NOT_IN_TIME = (
    "The call was not approved in time, so it did not run. Silence is not "
    "consent: do not retry it without asking the user."
)
FAILED = "Approval failed, so the call did not run: "

# ── Fakes ───────────────────────────────────────────────────────────────────


class ScriptedLLM(BaseLLMProvider):
    """Answers each request with the next scripted reply; records each request."""

    def __init__(self, replies: Iterable[Message]) -> None:
        self.replies: deque[Message] = deque(replies)
        self.requests: list[list[Message]] = []
        self.offered: list[list[str]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        self.offered.append([tool.name for tool in tools or []])
        if not self.replies:
            raise AssertionError("no reply is scripted for this request")
        return self.replies.popleft()

    async def close(self) -> None:
        pass

    def tool_messages(self, request: int) -> list[str]:
        """The contents of the tool messages that request *request* carried."""
        return [m.content for m in self.requests[request] if m.role == "tool"]


class Policy:
    """An approval policy that records each request and gives *answer*.

    *answer* is a decision or an exception to raise; a callable is called
    inside ``decide`` and its result given (or what it raises raised).
    With *wait*, ``decide`` first waits until it is set, and a cancellation
    there sets ``withdrawn``. ``asked`` is set once a request arrives.
    """

    def __init__(self, answer: object, *, wait: asyncio.Event | None = None) -> None:
        self.answer = answer
        self.wait = wait
        self.requests: list[ApprovalRequest] = []
        self.asked = asyncio.Event()
        self.withdrawn = False

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        self.asked.set()
        if self.wait is not None:
            try:
                await self.wait.wait()
            except asyncio.CancelledError:
                self.withdrawn = True
                raise
        answer = self.answer() if callable(self.answer) else self.answer
        if isinstance(answer, BaseException):
            raise answer
        return answer  # type: ignore[return-value]


class Recorder:
    """A sink that keeps every event."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.events.append(event)

    def of(self, type_: str) -> list[AgentEvent]:
        return [event for event in self.events if event.type == type_]


class Tools:
    """The custom tools; ``ran`` records each call that ran."""

    def __init__(self) -> None:
        self.ran: list[tuple[str, dict[str, Any]]] = []

    async def deploy(self, target: str) -> str:
        self.ran.append(("deploy", {"target": target}))
        return f"deployed to {target}"

    async def status(self) -> str:
        self.ran.append(("status", {}))
        return "all good"


DEPLOY = ToolDef(
    name="deploy",
    description="Deploy the site to a target",
    parameters={
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
    },
)
STATUS = ToolDef(
    name="status",
    description="Report the status of the site",
    parameters={"type": "object", "properties": {}},
)


class ReleaseTools(BaseToolProvider):
    """Provides ``release``, which requires approval, and ``changelog``."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    async def start(self) -> list[ProvidedTool]:
        return [
            ProvidedTool(
                definition=ToolDef(
                    name="release",
                    description="Publish a new version",
                    parameters={
                        "type": "object",
                        "properties": {"version": {"type": "string"}},
                    },
                ),
                function=self._release,
                requires_approval=True,
            ),
            ProvidedTool(
                definition=ToolDef(
                    name="changelog",
                    description="Show the changelog",
                    parameters={"type": "object", "properties": {}},
                    annotations={"readOnlyHint": True},
                ),
                function=self._changelog,
            ),
        ]

    async def _release(self, version: str = "") -> str:
        self.ran.append(f"release {version}")
        return f"released {version}"

    async def _changelog(self) -> str:
        self.ran.append("changelog")
        return "- fixes"

    async def close(self) -> None:
        pass


class TieredTools(BaseToolProvider):
    """Provides an unannotated write tool and a read-only one, neither marked."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    async def start(self) -> list[ProvidedTool]:
        return [
            ProvidedTool(
                definition=ToolDef(
                    name="lookup",
                    description="Look a fact up",
                    parameters={"type": "object", "properties": {}},
                    annotations={"readOnlyHint": True},
                ),
                function=self._lookup,
            ),
            ProvidedTool(
                definition=ToolDef(
                    name="remember",
                    description="Remember a fact",
                    parameters={"type": "object", "properties": {}},
                ),
                function=self._remember,
            ),
        ]

    async def _lookup(self) -> str:
        self.ran.append("lookup")
        return "a fact"

    async def _remember(self) -> str:
        self.ran.append("remember")
        return "remembered"

    async def close(self) -> None:
        pass


async def _no_wait(delay: float) -> None:
    pass


@pytest.fixture(autouse=True)
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


def _agent(
    llm: BaseLLMProvider,
    *,
    tools: Tools | None = None,
    providers: Sequence[BaseToolProvider] = (),
    **options: Any,
) -> tuple[Agent, Tools]:
    """An agent with ``deploy`` (marked) and ``status``, and no waits."""
    options.setdefault("builtins", frozenset())
    options.setdefault("tool_exposure", "all")
    agent = Agent(llm=llm, options=AgentOptions(**options), tool_providers=providers)
    tools = tools or Tools()
    agent.hippo.register_tool(DEPLOY)
    agent.executor.register("deploy", tools.deploy, requires_approval=True)
    agent.hippo.register_tool(STATUS)
    agent.executor.register("status", tools.status)
    agent.chat.sleep = _no_wait
    return agent, tools


# ── Replies and checks ──────────────────────────────────────────────────────


def _answer(text: str) -> Message:
    return Message(role="assistant", content=text)


def _calls(*calls: tuple[str, dict[str, Any], str]) -> Message:
    """A reply with tool calls, each ``(name, arguments, id)``."""
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(tool_name=n, arguments=a, id=i) for n, a, i in calls],
    )


def _deploy(call_id: str = "call_1", target: str = "prod") -> Message:
    return _calls(("deploy", {"target": target}, call_id))


def _reflection(action: str) -> Message:
    decision = {
        "summary": "Deploying was denied each time.",
        "progress_made": False,
        "task_still_feasible": False,
        "new_information_gained": False,
        "next_step_is_concrete": False,
        "next_step_is_novel": False,
        "recommended_action": action,
        "rationale": "The user does not want a deploy.",
        "best_effort_result": "",
    }
    return _answer(json.dumps(decision))


def _approval(
    request: ApprovalRequest,
    *,
    origin: str,
    allowed: bool = False,
    reason: str = "",
    remember: bool = False,
) -> dict[str, Any]:
    """The ``approval`` of a tool.finished event for *request*."""
    return {
        "request_id": request.request_id,
        "allowed": allowed,
        "reason": reason,
        "remember": remember,
        "origin": origin,
    }


def check_events(recorder: Recorder) -> None:
    """Pinned keys, JSON envelopes, and one tool.finished per tool.started."""
    open_call: str | None = None
    for event in recorder.events:
        assert set(event.data) == EVENT_FIELDS[event.type], event.type
        json.dumps(event.to_dict(), allow_nan=False)
        if event.type == "tool.started":
            assert open_call is None
            open_call = event.data["call_id"]
        elif event.type == "tool.finished":
            assert event.data["call_id"] == open_call
            open_call = None
        elif event.type in ("step.started", "run.finished"):
            assert open_call is None


# ── Who is asked ────────────────────────────────────────────────────────────


class TestWhoIsAsked:
    @pytest.mark.asyncio
    async def test_without_a_policy_marked_tools_run(self, tmp_path: Path) -> None:
        target = tmp_path / "notes.txt"
        write = {"path": str(target), "content": "hi"}
        llm = ScriptedLLM(
            [
                _calls(
                    ("deploy", {"target": "prod"}, "c1"), ("file_write", write, "c2")
                ),
                _answer("done"),
            ]
        )
        agent, tools = _agent(llm, builtins=frozenset({"file_write"}))
        recorder = Recorder()

        result = await agent.run_task("deploy", on_event=recorder)

        assert agent.executor.requires_approval("deploy")
        assert agent.executor.requires_approval("file_write")
        assert result.status == "completed"
        assert tools.ran == [("deploy", {"target": "prod"})]
        assert target.read_text() == "hi"
        finished = recorder.of("tool.finished")
        assert [e.data["status"] for e in finished] == ["success", "success"]
        assert [e.data["approval"] for e in finished] == [None, None]
        assert recorder.events[0].data["approval"] is False
        check_events(recorder)

    @pytest.mark.asyncio
    async def test_unmarked_tools_never_ask(self) -> None:
        policy = Policy(ApprovalDecision(False, "never"))
        llm = ScriptedLLM([_calls(("status", {}, "c1")), _answer("all good")])
        agent, tools = _agent(llm)
        recorder = Recorder()

        await agent.run_task("status?", on_event=recorder, approval=policy)

        assert policy.requests == []
        assert tools.ran == [("status", {})]
        [finished] = recorder.of("tool.finished")
        assert (finished.data["status"], finished.data["approval"]) == (
            "success",
            None,
        )

    @pytest.mark.asyncio
    async def test_approval_is_null_when_the_policy_was_not_asked(self) -> None:
        # Refused before the policy would be asked: an unknown tool, and a
        # marked tool's arguments that do not match its parameters.
        llm = ScriptedLLM(
            [
                _calls(("nope", {}, "c1"), ("deploy", {"where": "prod"}, "c2")),
                _answer("done"),
            ]
        )
        policy = Policy(ApprovalDecision(True))
        agent, tools = _agent(llm)
        recorder = Recorder()

        await agent.run_task("deploy", on_event=recorder, approval=policy)

        assert (policy.requests, tools.ran) == ([], [])
        finished = [e.data for e in recorder.of("tool.finished")]
        assert [(d["error_category"], d["approval"]) for d in finished] == [
            ("tool_not_found", None),
            ("invalid_arguments", None),
        ]
        assert finished[1]["error"] == (
            "Arguments for deploy do not match its parameters: missing a "
            "required argument: 'target'"
        )
        check_events(recorder)

    @pytest.mark.asyncio
    async def test_a_marked_provided_tool_asks(self) -> None:
        provider = ReleaseTools()
        llm = ScriptedLLM(
            [
                _calls(("changelog", {}, "c1"), ("release", {"version": "1.2"}, "c2")),
                _answer("released"),
            ]
        )
        policy = Policy(ApprovalDecision(True))
        agent, _ = _agent(llm, providers=[provider])

        result = await agent.run_task("release 1.2", approval=policy)

        assert result.status == "completed"
        assert agent.executor.requires_approval("release")
        assert not agent.executor.requires_approval("changelog")
        assert [(r.tool_name, dict(r.arguments), r.tier) for r in policy.requests] == [
            ("release", {"version": "1.2"}, "write")
        ]
        assert provider.ran == ["changelog", "release 1.2"]

    @pytest.mark.asyncio
    async def test_a_provided_tool_is_marked_by_its_tier(self) -> None:
        # Neither tool sets requires_approval: the readOnlyHint makes one
        # read and the other write, so only the write one asks.
        provider = TieredTools()
        llm = ScriptedLLM(
            [
                _calls(("lookup", {}, "c1"), ("remember", {}, "c2")),
                _answer("done"),
            ]
        )
        policy = Policy(ApprovalDecision(True))
        agent, _ = _agent(llm, providers=[provider])

        result = await agent.run_task("look and remember", approval=policy)

        assert result.status == "completed"
        assert agent.executor.requires_approval("remember")
        assert not agent.executor.requires_approval("lookup")
        assert {
            tool.name: tool.tier
            for tool in agent.list_tools()
            if tool.name in {"lookup", "remember"}
        } == {"lookup": "read", "remember": "write"}
        assert [request.tool_name for request in policy.requests] == ["remember"]
        assert provider.ran == ["lookup", "remember"]

    @pytest.mark.asyncio
    async def test_retrieve_mode_asks_about_builtins_and_offered_tools_only(
        self, tmp_path: Path
    ) -> None:
        # The query names no tool. Since C1 every step offers the built-ins,
        # so file_write reaches the policy; release, which retrieval did not
        # offer, is refused before anyone is asked.
        target = tmp_path / "notes.txt"
        write = {"path": str(target), "content": "hi"}
        provider = ReleaseTools()
        llm = ScriptedLLM(
            [
                _calls(
                    ("file_write", write, "c1"), ("release", {"version": "2"}, "c2")
                ),
                _answer("done"),
            ]
        )
        policy = Policy(ApprovalDecision(True))
        agent = Agent(
            llm=llm,
            options=AgentOptions(tool_exposure="retrieve"),
            tool_providers=[provider],
        )

        result = await agent.run_task("Tidy up", approval=policy)

        assert result.status == "completed"
        assert llm.offered[0] == sorted(DEFAULT_BUILTINS["python"])
        assert [r.tool_name for r in policy.requests] == ["file_write"]
        assert target.read_text() == "hi"
        assert provider.ran == []
        assert llm.tool_messages(1) == [
            f"Written 2 bytes to {target}",
            "Error: Tool 'release' was not offered in this step.",
        ]


# ── The request ─────────────────────────────────────────────────────────────


class TestRequest:
    @pytest.mark.asyncio
    async def test_the_request_names_the_call_exactly(self, tmp_path: Path) -> None:
        target = tmp_path / "notes.txt"
        write = {"path": str(target), "content": "hello"}
        llm = ScriptedLLM(
            [
                _calls(
                    ("file_write", write, "call_w"),
                    ("deploy", {"target": "prod"}, "call_d"),
                ),
                _answer("done"),
                _deploy("call_d"),  # a backend's call ids repeat across runs
                _answer("done again"),
            ]
        )
        policy = Policy(ApprovalDecision(True))
        agent, _ = _agent(llm, builtins=frozenset({"file_write"}))
        recorder = Recorder()

        first = await agent.run_task(
            "write, then deploy", on_event=recorder, approval=policy
        )
        second = await agent.run_task("deploy again", approval=policy)

        write_request, deploy_request, again = policy.requests
        # The path is the one the model gave: without a workspace root the
        # executor resolves no path.
        assert write_request == ApprovalRequest(
            request_id=write_request.request_id,
            run_id=first.task_id,
            call_id="call_w",
            tool_name="file_write",
            arguments=write,
            preview=f"{target} (5 chars)",
            workspace_root=None,
            tier="write",
            review=ActionReview("routine"),
        )
        assert deploy_request == ApprovalRequest(
            request_id=deploy_request.request_id,
            run_id=first.task_id,
            call_id="call_d",
            tool_name="deploy",
            arguments={"target": "prod"},
            preview='{"target":"prod"}',
            workspace_root=None,
            tier="write",
            review=ActionReview("routine"),
        )
        assert (again.run_id, again.call_id) == (second.task_id, "call_d")
        ids = [request.request_id for request in policy.requests]
        assert all(re.fullmatch(r"[0-9a-f]{32}", i) for i in ids)
        assert len(set(ids)) == 3
        approvals = [e.data["approval"] for e in recorder.of("tool.finished")]
        assert approvals == [
            _approval(write_request, origin="user", allowed=True),
            _approval(deploy_request, origin="user", allowed=True),
        ]

    @pytest.mark.asyncio
    async def test_a_rooted_request_names_the_resolved_path(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "root"
        root.mkdir()
        write = {"path": "notes.txt", "content": "hi"}
        llm = ScriptedLLM([_calls(("file_write", write, "call_w")), _answer("done")])
        policy = Policy(ApprovalDecision(True))
        agent, _ = _agent(
            llm, builtins=frozenset({"file_write"}), workspace_root=str(root)
        )
        recorder = Recorder()

        result = await agent.run_task("write", on_event=recorder, approval=policy)

        # The policy sees the real path that will be written, not the
        # model's relative one, and the root it lies in.
        resolved = os.path.join(os.path.realpath(root), "notes.txt")
        [request] = policy.requests
        assert request == ApprovalRequest(
            request_id=request.request_id,
            run_id=result.task_id,
            call_id="call_w",
            tool_name="file_write",
            arguments={"path": resolved, "content": "hi"},
            preview=f"{resolved} (2 chars)",
            workspace_root=agent.workspace_root,
            tier="write",
            review=ActionReview("routine"),
        )
        assert agent.workspace_root == os.path.realpath(root)
        [started] = recorder.of("run.started")
        assert started.data["workspace_root"] == agent.workspace_root
        assert Path(resolved).read_text() == "hi"

    @pytest.mark.asyncio
    async def test_a_shell_exec_request_is_exec(self) -> None:
        llm = ScriptedLLM(
            [
                _calls(("shell_exec", {"command": "printf hi"}, "call_s")),
                _answer("done"),
            ]
        )
        policy = Policy(ApprovalDecision(True))
        agent, _ = _agent(llm, builtins=frozenset({"shell_exec"}))

        result = await agent.run_task("say hi", approval=policy)

        assert result.status == "completed"
        [request] = policy.requests
        assert (request.tool_name, request.tier) == ("shell_exec", "exec")

    @pytest.mark.asyncio
    async def test_the_arguments_are_a_copy(self, tmp_path: Path) -> None:
        target = tmp_path / "notes.txt"

        def tamper() -> ApprovalDecision:
            arguments = policy.requests[-1].arguments
            assert isinstance(arguments, dict)
            arguments["content"] = "tampered"
            return ApprovalDecision(True)

        policy = Policy(tamper)
        llm = ScriptedLLM(
            [
                _calls(("file_write", {"path": str(target), "content": "hi"}, "c1")),
                _answer("done"),
            ]
        )
        agent, _ = _agent(llm, builtins=frozenset({"file_write"}))

        await agent.run_task("write", approval=policy)

        assert target.read_text() == "hi"


# ── Decisions ───────────────────────────────────────────────────────────────


DENIALS = [
    pytest.param(
        ApprovalDecision(False), f"The user denied this call. {ENDING}", id="user"
    ),
    pytest.param(
        ApprovalDecision(False, " not in this folder. "),
        f"The user denied this call: not in this folder. {ENDING}",
        id="user_with_a_reason",
    ),
    pytest.param(
        ApprovalDecision(False, origin="policy"),
        f"The approval policy denied this call. {ENDING}",
        id="policy",
    ),
    pytest.param(
        ApprovalDecision(False, "approvals are set to deny", origin="policy"),
        f"The approval policy denied this call: approvals are set to deny. {ENDING}",
        id="policy_with_a_reason",
    ),
    pytest.param(ApprovalDecision(False, origin="timeout"), NOT_IN_TIME, id="timeout"),
    pytest.param(
        ApprovalDecision(False, "No answer within 300 s", origin="timeout"),
        NOT_IN_TIME,
        id="timeout_with_a_reason",
    ),
]


class TestDecisions:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("decision", "text"), DENIALS)
    async def test_each_denial_reaches_the_model(
        self, decision: ApprovalDecision, text: str
    ) -> None:
        llm = ScriptedLLM([_deploy(), _answer("Not deploying, then.")])
        policy = Policy(decision)
        agent, tools = _agent(llm)
        recorder = Recorder()

        result = await agent.run_task("deploy", on_event=recorder, approval=policy)

        assert (result.status, result.result) == ("completed", "Not deploying, then.")
        assert tools.ran == []
        assert llm.tool_messages(1) == [f"Error: {text}"]
        [finished] = recorder.of("tool.finished")
        data = finished.data
        assert (data["status"], data["error_category"], data["error"]) == (
            "blocked",
            "approval_denied",
            text,
        )
        assert (data["output"], data["duration_s"], data["exit_code"]) == ("", 0, None)
        assert data["approval"] == _approval(
            policy.requests[0], origin=decision.origin, reason=decision.reason
        )
        check_events(recorder)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("answer", "failure"),
        [
            pytest.param(
                RuntimeError("the policy is down"),
                "RuntimeError: the policy is down",
                id="raises",
            ),
            pytest.param(
                lambda: None,
                "TypeError: the approval policy returned NoneType, not an "
                "ApprovalDecision",
                id="returns_no_decision",
            ),
            pytest.param(
                asyncio.CancelledError(),
                "RuntimeError: the approval policy raised CancelledError while the "
                "run was not cancelled",
                id="spurious_cancelled_error",
            ),
            pytest.param(
                lambda: ApprovalDecision(False, origin="session"),
                "ValueError: ApprovalDecision.origin 'session' needs allow=True: a "
                "session grant only allows",
                id="session_denial",
            ),
            pytest.param(
                lambda: ApprovalDecision(True, origin="timeout"),
                "ValueError: ApprovalDecision.origin 'timeout' needs allow=False: a "
                "timeout only denies",
                id="timeout_allow",
            ),
            pytest.param(
                lambda: ApprovalDecision(False, remember=True),
                "ValueError: ApprovalDecision.remember needs allow=True",
                id="remembered_denial",
            ),
        ],
    )
    async def test_a_failing_policy_fails_closed(
        self, answer: object, failure: str
    ) -> None:
        llm = ScriptedLLM([_deploy(), _answer("ok")])
        policy = Policy(answer)
        agent, tools = _agent(llm)
        recorder = Recorder()

        result = await agent.run_task("deploy", on_event=recorder, approval=policy)

        # The call is refused, and the run goes on.
        assert (result.status, result.result) == ("completed", "ok")
        assert tools.ran == []
        assert llm.tool_messages(1) == [f"Error: {FAILED}{failure}"]
        [finished] = recorder.of("tool.finished")
        assert (finished.data["status"], finished.data["error_category"]) == (
            "blocked",
            "approval_denied",
        )
        assert finished.data["approval"] == _approval(
            policy.requests[0], origin="error", reason=failure
        )
        check_events(recorder)

    @pytest.mark.asyncio
    async def test_remember_is_reported_not_acted_on(self) -> None:
        llm = ScriptedLLM([_deploy("c1", "a"), _deploy("c2", "b"), _answer("done")])
        policy = Policy(ApprovalDecision(True, remember=True))
        agent, tools = _agent(llm)
        recorder = Recorder()

        await agent.run_task("deploy twice", on_event=recorder, approval=policy)

        # Asked about both calls: the run keeps no grant.
        assert len(policy.requests) == 2
        assert tools.ran == [("deploy", {"target": "a"}), ("deploy", {"target": "b"})]
        assert [e.data["approval"] for e in recorder.of("tool.finished")] == [
            _approval(request, origin="user", allowed=True, remember=True)
            for request in policy.requests
        ]

    @pytest.mark.asyncio
    async def test_allow_all_policy(self) -> None:
        llm = ScriptedLLM([_deploy(), _answer("done")])
        agent, tools = _agent(llm)
        recorder = Recorder()

        await agent.run_task("deploy", on_event=recorder, approval=AllowAllPolicy())

        assert tools.ran == [("deploy", {"target": "prod"})]
        [finished] = recorder.of("tool.finished")
        assert finished.data["status"] == "success"
        approval = finished.data["approval"]
        assert (approval["allowed"], approval["origin"]) == (True, "policy")

    @pytest.mark.asyncio
    async def test_deny_all_policy(self) -> None:
        llm = ScriptedLLM([_deploy(), _answer("done")])
        agent, tools = _agent(llm)

        await agent.run_task(
            "deploy", approval=DenyAllPolicy("approvals are set to deny")
        )

        assert tools.ran == []
        assert llm.tool_messages(1) == [
            "Error: The approval policy denied this call: approvals are set to "
            f"deny. {ENDING}"
        ]

    @pytest.mark.asyncio
    async def test_the_policy_is_checked_before_the_run(self) -> None:
        llm = ScriptedLLM([_answer("never sent")])
        agent, _ = _agent(llm)

        with pytest.raises(TypeError) as caught:
            await agent.run_task("q", approval=object())  # type: ignore[arg-type]

        assert str(caught.value) == "approval must have an async decide(request) method"
        assert llm.requests == []
        assert agent.hippo.history._tasks == {}  # type: ignore[attr-defined]
        assert agent.hippo.working.get_window() == []


# ── Denials and the rest of the run ─────────────────────────────────────────


class TestDenialLoop:
    @pytest.mark.asyncio
    async def test_three_identical_denials_end_the_attempt_as_stuck(self) -> None:
        llm = ScriptedLLM([_deploy(f"c{i}") for i in range(3)])
        policy = Policy(ApprovalDecision(False, "no deploys today"))
        agent, tools = _agent(llm, attempt_policy="none")
        recorder = Recorder()

        result = await agent.run_task("deploy", on_event=recorder, approval=policy)

        assert result.status == "stuck"
        assert (result.steps, result.n_tool, len(policy.requests)) == (3, 3, 3)
        assert tools.ran == []
        assert [e.data["kind"] for e in recorder.of("notice")] == ["stuck"]

    @pytest.mark.asyncio
    async def test_a_reflection_follows_and_the_gate_decides(self) -> None:
        llm = ScriptedLLM([*(_deploy(f"c{i}") for i in range(3)), _reflection("stop")])
        policy = Policy(ApprovalDecision(False, "no deploys today"))
        agent, _ = _agent(llm, max_attempts=2)
        recorder = Recorder()

        result = await agent.run_task("deploy", on_event=recorder, approval=policy)

        [started] = recorder.of("reflection.started")
        assert started.data["failure_type"] == "stuck"
        assert (result.status, result.error) == (
            "gate_stopped",
            "The user does not want a deploy.",
        )

    @pytest.mark.asyncio
    async def test_an_approval_wait_counts_toward_timeout_s(self) -> None:
        llm = ScriptedLLM([_deploy()])
        policy = Policy(ApprovalDecision(True), wait=asyncio.Event())
        agent, tools = _agent(llm, timeout_s=0.3)
        recorder = Recorder()

        result = await agent.run_task("deploy", on_event=recorder, approval=policy)

        assert (result.status, result.error) == ("timeout", "Timed out after 0.3 s")
        assert policy.withdrawn
        assert tools.ran == []
        [finished] = recorder.of("tool.finished")
        assert (finished.data["status"], finished.data["error"]) == (
            "cancelled",
            CALL_NOT_STARTED,
        )
        assert finished.data["approval"] == _approval(
            policy.requests[0], origin="withdrawn"
        )
        check_events(recorder)
