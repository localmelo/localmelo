"""Scripted runs on the wire: a stub chat server, a run's document and the golden check.

A run goes through ``Agent.run_task`` as direct mode runs it, without an event
sink or an approval policy, through the real :class:`OpenAICompatLLM` or
:class:`OllamaNativeChat` against an ``httpx.MockTransport`` server, with SQLite
history in a temporary ``memory_dir``. Its document holds:

- ``result``: ``RunResult.to_dict()`` as a whole, except ``duration_s``;
- ``requests``: every request body as sent (``json.loads`` of its content),
  reflection requests included. Headers are not recorded;
- ``window``: working memory once the run is over, by ``Message`` field;
- ``history``: the ``tasks`` and ``steps`` rows of ``history.db``, read by column
  name.

Before a golden is compared, generated ids (task, step and tool-call ids) become
ordinals in order of first appearance, such as ``<task:1>``; timestamps and
durations are dropped; ``config.memory_dir``, a temporary directory, becomes
``<memory_dir>``; and a text longer than 10,000 characters keeps its first 100
characters, its length and its SHA-256.

``LOCALMELO_RECORD_GOLDENS=1`` writes the golden files instead of comparing them.
It is for recording a run again; the suite never sets it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sqlite3
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, RunResult
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

# Set to "1" to write the golden files instead of comparing them.
RECORD_ENV = "LOCALMELO_RECORD_GOLDENS"

# ── The serializer ──────────────────────────────────────────────────────────

# The fields and columns a golden holds, by name.
WINDOW_MESSAGE_FIELDS = (
    "role",
    "content",
    "thinking",
    "tool_calls",
    "tool_call_id",
    "images",
)
TOOL_CALL_FIELDS = ("tool_name", "arguments", "id")
IMAGE_FIELDS = ("mime_type", "sha256", "size_bytes")
TASK_COLUMNS = ("task_id", "query", "status", "result")
STEP_COLUMNS = (
    "step_id",
    "task_id",
    "thought",
    "tool_call_json",
    "tool_result_json",
    "seq",
)


def _message_field(message: Message, name: str) -> Any:
    value = getattr(message, name)
    if name == "tool_calls":
        return [
            {f: copy.deepcopy(getattr(call, f)) for f in TOOL_CALL_FIELDS}
            for call in value
        ]
    if name == "images":
        return [{f: getattr(image, f) for f in IMAGE_FIELDS} for image in value]
    return value


def window_json(messages: Iterable[Message]) -> list[dict[str, Any]]:
    """The working-memory window, by the fields of WINDOW_MESSAGE_FIELDS."""
    return [
        {name: _message_field(m, name) for name in WINDOW_MESSAGE_FIELDS}
        for m in messages
    ]


def result_json(result: RunResult, memory_dir: str) -> dict[str, Any]:
    """``RunResult.to_dict()``, in its key order, without ``duration_s``."""
    data = result.to_dict()
    del data["duration_s"]
    if data["config"].get("memory_dir") == memory_dir:
        data["config"]["memory_dir"] = "<memory_dir>"
    return data


def history_json(db_path: Path) -> dict[str, list[dict[str, Any]]]:
    """The ``tasks`` and ``steps`` rows of *db_path*, by the named columns.

    The JSON columns of ``steps`` are parsed, and ``duration_ms`` is left
    out of ``tool_result_json``.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tasks = conn.execute(
            f"SELECT {', '.join(TASK_COLUMNS)} FROM tasks ORDER BY rowid"
        ).fetchall()
        steps = conn.execute(
            f"SELECT {', '.join(STEP_COLUMNS)} FROM steps ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    return {
        "tasks": [{name: row[name] for name in TASK_COLUMNS} for row in tasks],
        "steps": [_step_json(row) for row in steps],
    }


def _step_json(row: sqlite3.Row) -> dict[str, Any]:
    step = {name: row[name] for name in STEP_COLUMNS}
    step["tool_call_json"] = _stored_json(step["tool_call_json"])
    result = _stored_json(step["tool_result_json"])
    if isinstance(result, dict):
        result.pop("duration_ms", None)
    step["tool_result_json"] = result
    return step


def _stored_json(text: str | None) -> Any:
    """The value of a JSON column, which must hold exactly its ``json.dumps``.

    So the parsed value, whose key order a golden keeps, pins the text.
    """
    if text is None:
        return None
    value = json.loads(text)
    assert json.dumps(value) == text, f"not as json.dumps writes it: {text}"
    return value


# A tool-call id that ToolCall generated, and a task or step id (UUID4 hex).
_CALL_ID = re.compile(r"call_[0-9a-f]{24}(?![0-9a-f])")
_RECORD_ID = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")

# A longer text keeps only its first LONG_TEXT_KEPT characters, its length
# and its digest in a golden.
LONG_TEXT_CHARS = 10_000
LONG_TEXT_KEPT = 100


class Normalizer:
    """Makes a run's document comparable across runs.

    Calling it returns a normalized copy of a JSON-like value. Ids become
    ordinals in the order it meets them: ``<task:n>`` and ``<step:n>`` for
    the ids in *tasks* and *steps*, ``<call:n>`` for generated tool-call
    ids and ``<id:n>`` for any other UUID4 hex. So one normalizer serves
    one document. A text longer than LONG_TEXT_CHARS is shortened (see the
    module docstring).
    """

    def __init__(self, *, tasks: Iterable[str] = (), steps: Iterable[str] = ()) -> None:
        self._kinds = {**dict.fromkeys(tasks, "task"), **dict.fromkeys(steps, "step")}
        self._labels: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def __call__(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._text(value)
        if isinstance(value, list):
            return [self(item) for item in value]
        if isinstance(value, dict):
            return {key: self(item) for key, item in value.items()}
        return value

    def _label(self, kind: str, value: str) -> str:
        if value not in self._labels:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            self._labels[value] = f"<{kind}:{self._counts[kind]}>"
        return self._labels[value]

    def _text(self, text: str) -> str:
        text = _CALL_ID.sub(lambda m: self._label("call", m.group()), text)
        text = _RECORD_ID.sub(
            lambda m: self._label(self._kinds.get(m.group(), "id"), m.group()), text
        )
        if len(text) > LONG_TEXT_CHARS:
            digest = hashlib.sha256(text.encode()).hexdigest()
            text = f"{text[:LONG_TEXT_KEPT]}… [{len(text)} chars, sha256 {digest}]"
        return text


def check_golden(path: Path, document: Any) -> None:
    """Compare *document* with the golden file *path*, or record it there.

    The text is compared too, so key order counts: the CLI's JSON line
    prints ``RunResult.to_dict()`` in its order.
    """
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if os.environ.get(RECORD_ENV) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        pytest.skip(f"recorded {path.name}")
    assert path.is_file(), f"no golden file {path}"
    golden = path.read_text(encoding="utf-8")
    assert json.loads(text) == json.loads(golden)
    assert text == golden


# ── The scripted server ─────────────────────────────────────────────────────


class ScriptedServer:
    """Answers each chat request with the next scripted body; keeps each body.

    A request body is kept as ``json.loads`` of its content, as the provider
    sent it.
    """

    def __init__(self, replies: Iterable[dict[str, Any]]) -> None:
        self.replies: deque[dict[str, Any]] = deque(replies)
        self.bodies: list[Any] = []
        self.raw_bodies: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.raw_bodies.append(request.content.decode("utf-8"))
        self.bodies.append(json.loads(request.content))
        if not self.replies:
            raise AssertionError("no reply is scripted for this request")
        return httpx.Response(200, json=self.replies.popleft())


@dataclass(frozen=True)
class Call:
    """One tool call of a scripted reply."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Reply:
    """One scripted model reply, written out per provider by :func:`_body`."""

    content: str = ""
    thinking: str = ""
    calls: tuple[Call, ...] = ()
    prompt_tokens: int = 100
    completion_tokens: int = 10
    # Usage fields an OpenAI-compatible server adds, such as
    # prompt_tokens_details; Ollama's native API has none.
    usage_details: dict[str, Any] = field(default_factory=dict)


def _openai_body(reply: Reply, call_ids: Iterable[str]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": reply.content}
    if reply.thinking:
        message["reasoning_content"] = reply.thinking
    if reply.calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call, call_id in zip(reply.calls, call_ids, strict=True)
        ]
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "stub-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if reply.calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "total_tokens": reply.prompt_tokens + reply.completion_tokens,
            **reply.usage_details,
        },
    }


def _ollama_body(reply: Reply) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": reply.content}
    if reply.thinking:
        message["thinking"] = reply.thinking
    if reply.calls:
        # The native API sends no call ids: the provider generates them.
        message["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in reply.calls
        ]
    return {
        "model": "stub-model",
        "message": message,
        "done": True,
        "prompt_eval_count": reply.prompt_tokens,
        "eval_count": reply.completion_tokens,
    }


def _bodies(provider: str, replies: Iterable[Reply]) -> list[dict[str, Any]]:
    """The server's bodies for *replies*; OpenAI call ids are unique per run."""
    bodies: list[dict[str, Any]] = []
    count = 0
    for reply in replies:
        if provider == "ollama":
            bodies.append(_ollama_body(reply))
            continue
        ids = [f"call_{count + i + 1}" for i in range(len(reply.calls))]
        count += len(reply.calls)
        bodies.append(_openai_body(reply, ids))
    return bodies


def _llm(provider: str, transport: httpx.MockTransport) -> BaseLLMProvider:
    if provider == "openai":
        return OpenAICompatLLM(
            base_url="http://llm.test/v1", model="stub-model", transport=transport
        )
    return OllamaNativeChat(
        base_url="http://ollama.test", model="stub-model", transport=transport
    )


# ── Replies ─────────────────────────────────────────────────────────────────


def _python(code: str, *, content: str = "", thinking: str = "") -> Reply:
    return Reply(
        content=content,
        thinking=thinking,
        calls=(Call("python_exec", {"code": code}),),
    )


def _reflection(action: str, *, thinking: str = "", **fields: Any) -> Reply:
    """A reflection reply with every key REFLECTION_PROMPT asks for."""
    data: dict[str, Any] = {
        "summary": "Ran the code once.",
        "failed_hypotheses": [],
        "useful_evidence": [],
        "recommended_avoids": [],
        "next_promising_directions": [],
        "progress_made": True,
        "task_still_feasible": True,
        "new_information_gained": True,
        "next_step_is_concrete": True,
        "next_step_is_novel": True,
        "recommended_action": action,
        "rationale": "",
        "best_effort_result": "",
        "estimated_info_gain": 0.8,
        "estimated_cost": 0.1,
        "repeat_risk": 0.1,
        "novelty": 0.8,
        "feasibility": 0.8,
    }
    data.update(fields)
    return Reply(
        content=json.dumps(data),
        thinking=thinking,
        prompt_tokens=900,
        completion_tokens=80,
    )


@dataclass
class Scenario:
    """One scripted run: options, the query and the model's replies.

    *status* is the status the run must end with.
    """

    status: str
    query: str
    replies: list[Reply]
    options: dict[str, Any] = field(default_factory=dict)


def reflection_continues() -> Scenario:
    return Scenario(
        status="completed",
        query="What is 2 to the power of 10? Check it with Python.",
        options={"steps_per_attempt": 1},
        replies=[
            _python(
                "print(2 ** 10)",
                content="Computing it.",
                thinking="Python gives the exact power.",
            ),
            _reflection(
                "continue",
                thinking="The output is known; one more step answers.",
                summary="Python printed 1024.",
                next_promising_directions=["Answer with 1024."],
                rationale="The answer is known.",
            ),
            Reply(
                content="2 to the power of 10 is 1024.",
                thinking="The reflection says the answer is 1024.",
            ),
        ],
    )


async def record_run(
    provider: str,
    scenario: Scenario,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_bodies: list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run *scenario* on *provider* and return its RunResult and document."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    memory_dir = str(tmp_path / "memory")
    server = ScriptedServer(_bodies(provider, scenario.replies))
    agent = Agent(
        llm=_llm(provider, httpx.MockTransport(server.handler)),
        options=AgentOptions(memory_dir=memory_dir, **scenario.options),
    )
    try:
        result = await agent.run_task(scenario.query)
        window = window_json(agent.hippo.working.get_window())
    finally:
        await agent.close()
    # Every scripted reply was used: the script and the run agree.
    assert not server.replies, f"replies left: {list(server.replies)}"

    history = history_json(Path(memory_dir) / "history.db")
    normalize = Normalizer(
        tasks=[result.task_id, *(row["task_id"] for row in history["tasks"])],
        steps=[row["step_id"] for row in history["steps"]],
    )
    document = normalize(
        {
            "result": result_json(result, memory_dir),
            "requests": server.bodies,
            "window": window,
            "history": history,
        }
    )
    if raw_bodies is not None:
        raw_bodies.extend(server.raw_bodies)
    return result, document
