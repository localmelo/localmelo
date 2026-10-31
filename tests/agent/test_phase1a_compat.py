"""Golden fixtures of non-interactive runs (T-COMPAT) — no real model.

Every run here is scripted and goes through ``Agent.run_task`` as direct
mode runs it: without an event sink or an approval policy, on an Agent
built without a workspace root, with SQLite history in a temporary
``memory_dir``. Its golden file, ``goldens/phase1a_compat/<run>.json``,
holds what the run sent and what it kept, each by the fields named in
this module, so that a field, column or table added elsewhere changes no
golden:

- ``result``: ``RunResult.to_dict()`` as a whole, except ``duration_s``;
- ``requests``: every request the model received, retries of a
  reflection included: the offered tool names, and each message's role,
  content, tool calls and ``tool_call_id``;
- ``window``: working memory once the run is over, by ``Message`` field:
  role, content, thinking, tool calls, ``tool_call_id`` and the images'
  MIME type, digest and size;
- ``history``: the ``tasks`` and ``steps`` rows of ``history.db``, read by
  column name.

The goldens were recorded at c295ce2, which holds Phase 1a (#30), the
reflection retry (#31), the built-ins offered on every step (#32) and the
reflection gate (#34) with its default ``reflection_gate="flags"``. Every
later change must leave them as they are.

Before a golden is compared, generated ids (task, step and tool-call ids)
become ordinals in order of first appearance, such as ``<task:1>``;
timestamps and durations are dropped; ``config.memory_dir``, a temporary
directory, becomes ``<memory_dir>``; and a text longer than 10,000
characters keeps its first 100 characters, its length and its SHA-256.

``LOCALMELO_RECORD_GOLDENS=1`` writes the golden files instead of
comparing them. It is for recording new runs; the suite never sets it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sqlite3
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    ProviderError,
)
from localmelo.melo.contracts.tools import BaseToolProvider, ProvidedTool
from localmelo.melo.executor import ToolOutput
from localmelo.melo.schema import ImagePart, Message, RunResult, ToolCall, ToolDef
from localmelo.tests.legacy_options import LEGACY_ANSWER_OPTIONS, LEGACY_CACHE_OPTIONS

GOLDEN_DIR = Path(__file__).parent / "goldens" / "phase1a_compat"

# Set to "1" to write the golden files instead of comparing them.
RECORD_ENV = "LOCALMELO_RECORD_GOLDENS"

# ── The serializer ──────────────────────────────────────────────────────────

# The fields and columns a golden holds, by name.
REQUEST_MESSAGE_FIELDS = ("role", "content", "tool_calls", "tool_call_id")
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


def messages_json(
    messages: Iterable[Message], fields: Sequence[str]
) -> list[dict[str, Any]]:
    """*messages*, each as a dict of the named *fields*."""
    return [{name: _message_field(m, name) for name in fields} for m in messages]


def request_json(
    messages: Sequence[Message], tools: Sequence[ToolDef] | None
) -> dict[str, Any]:
    """One request as the model receives it: offered tool names, then messages."""
    return {
        "tools": None if tools is None else [tool.name for tool in tools],
        "messages": messages_json(messages, REQUEST_MESSAGE_FIELDS),
    }


def window_json(messages: Iterable[Message]) -> list[dict[str, Any]]:
    """The working-memory window, by the fields of WINDOW_MESSAGE_FIELDS."""
    return messages_json(messages, WINDOW_MESSAGE_FIELDS)


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


# ── Fakes ───────────────────────────────────────────────────────────────────


class ScriptedLLM(BaseLLMProvider):
    """Answers each request with the next scripted reply; records each request.

    A scripted exception is raised instead of returned. A request is
    recorded as it arrives, before anything can change its messages.
    """

    def __init__(self, replies: Iterable[Message | BaseException]) -> None:
        self.replies: deque[Message | BaseException] = deque(replies)
        self.requests: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append(request_json(messages, tools))
        if not self.replies:
            raise AssertionError("no reply is scripted for this request")
        reply = self.replies.popleft()
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def close(self) -> None:
        pass


class SameVectorEmbedding(BaseEmbeddingProvider):
    """Gives every text the same vector, so recall keeps the order of storage."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def close(self) -> None:
        pass


FORECAST_MAP = ImagePart.from_bytes(
    b"\x89PNG\r\n\x1a\n a map of the forecast", "image/png"
)


async def _weather(city: str = "") -> ToolOutput:
    return ToolOutput(text=f"Sunny, 21 °C in {city}", images=[FORECAST_MAP])


async def _stock_price(ticker: str = "") -> str:
    return f"{ticker}: 123.45"


def _one_string(name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {name: {"type": "string"}},
        "required": [name],
    }


class WeatherTools(BaseToolProvider):
    """Provides ``weather``, which returns text and an image, and ``stock_price``.

    No word of stock_price's name or description is in a query below, so
    ``tool_exposure="retrieve"`` offers it only once a hint names it. With
    *error*, start() raises it.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def start(self) -> list[ProvidedTool]:
        if self.error is not None:
            raise self.error
        return [
            ProvidedTool(
                definition=ToolDef(
                    name="weather",
                    description="Forecast for a city",
                    parameters=_one_string("city"),
                ),
                function=_weather,
            ),
            ProvidedTool(
                definition=ToolDef(
                    name="stock_price",
                    description="Latest quote for a ticker symbol",
                    parameters=_one_string("ticker"),
                ),
                function=_stock_price,
            ),
        ]

    async def close(self) -> None:
        pass


# ── Replies ─────────────────────────────────────────────────────────────────


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _reply(
    content: str, *, thinking: str = "", usage: dict[str, int] | None = None
) -> Message:
    return Message(role="assistant", content=content, thinking=thinking, usage=usage)


def _tool_calls(
    *calls: tuple[str, dict[str, Any]],
    content: str = "",
    thinking: str = "",
    usage: dict[str, int] | None = None,
) -> Message:
    return Message(
        role="assistant",
        content=content,
        thinking=thinking,
        usage=usage,
        tool_calls=[ToolCall(tool_name=name, arguments=args) for name, args in calls],
    )


def _python(
    code: str, *, content: str = "", usage: dict[str, int] | None = None
) -> Message:
    return _tool_calls(("python_exec", {"code": code}), content=content, usage=usage)


# Estimates whose utility, 0.8 * 0.8 * 0.8 - 0.1 - 0.1, passes either gate.
HIGH_UTILITY: dict[str, Any] = {
    "estimated_info_gain": 0.8,
    "estimated_cost": 0.1,
    "repeat_risk": 0.1,
    "novelty": 0.8,
    "feasibility": 0.8,
}
# Estimates whose utility, 0.0 * 0.8 * 0.8 - 0.25 - 0.25, only "flags" passes:
# a run that continues after them would stop under reflection_gate="utility".
LOW_UTILITY: dict[str, Any] = {
    "estimated_info_gain": 0.0,
    "estimated_cost": 0.25,
    "repeat_risk": 0.25,
    "novelty": 0.8,
    "feasibility": 0.8,
}


def _decision(action: str, **fields: Any) -> dict[str, Any]:
    """A reflection reply's object, with every key REFLECTION_PROMPT asks for."""
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
        **HIGH_UTILITY,
    }
    data.update(fields)
    return data


def _reflection(
    action: str, *, usage: dict[str, int] | None = None, **fields: Any
) -> Message:
    return _reply(json.dumps(_decision(action, **fields)), usage=usage)


PROSE = "I would rather describe it in words than in JSON."
STILL_PROSE = "Still no JSON, sorry."


# ── Runs ────────────────────────────────────────────────────────────────────


@dataclass
class Scenario:
    """One scripted run: the agent's set-up, the query and the model's replies.

    *status* is the status the run must end with. The memory directory is
    a temporary one, and the embedding provider is a SameVectorEmbedding
    unless *embedding* is false.
    """

    status: str
    query: str
    replies: list[Message | BaseException]
    options: dict[str, Any] = field(default_factory=dict)
    providers: list[BaseToolProvider] = field(default_factory=list)
    embedding: bool = True


def _completed_with_an_answer() -> Scenario:
    return Scenario(
        status="completed",
        query="What is six times seven?",
        replies=[_reply("42", usage=_usage(120, 2))],
    )


def _completed_after_tools_retrieve() -> Scenario:
    # The default tool_exposure, "retrieve": the built-ins, then the
    # provided tools that hints or BM25 choose.
    return Scenario(
        status="completed",
        query="What is the weather in Paris?",
        providers=[WeatherTools()],
        replies=[
            _tool_calls(
                ("weather", {"city": "Paris"}),
                content="Checking the forecast.",
                thinking="The user wants the forecast, so I call the weather tool.",
                usage=_usage(300, 25),
            ),
            # stock_price is not offered on this step: refused, not run.
            _tool_calls(
                ("python_exec", {"code": "print(6 * 7)"}),
                ("stock_price", {"ticker": "ACME"}),
                usage=_usage(420, 30),
            ),
            _reply("It is sunny and 21 °C in Paris.", usage=_usage(520, 12)),
        ],
    )


ALL_BUILTINS = frozenset({"file_read", "file_write", "python_exec", "shell_exec"})


def _completed_after_tools_all() -> Scenario:
    return Scenario(
        status="completed",
        query="Check the weather in Oslo and save it to notes.txt.",
        options={
            "tool_exposure": "all",
            # P13b leaves shell_exec out of direct and python defaults; this
            # scenario names every built-in so the legacy run still runs it.
            "builtins": ALL_BUILTINS,
        },
        providers=[WeatherTools()],
        embedding=False,
        replies=[
            _tool_calls(
                ("weather", {"city": "Oslo"}),
                ("shell_exec", {"command": "echo checked"}),
                usage=_usage(310, 40),
            ),
            _tool_calls(
                ("file_write", {"path": "notes.txt", "content": "Sunny in Oslo"}),
                ("file_read", {"path": "notes.txt"}),
                usage=_usage(450, 45),
            ),
            _reply("Saved: it is sunny in Oslo.", usage=_usage(560, 10)),
        ],
    )


def _completed_by_a_reflection_done() -> Scenario:
    return Scenario(
        status="completed",
        query="What is 2 + 2? Check it with Python.",
        options={"steps_per_attempt": 1},
        replies=[
            _python(
                "print(2 + 2)", content="Let me compute it.", usage=_usage(200, 15)
            ),
            _reflection(
                "done",
                summary="Python printed 4.",
                rationale="The code printed the answer.",
                best_effort_result="2 + 2 = 4",
                usage=_usage(900, 80),
            ),
        ],
    )


def _failed_by_pre_plan() -> Scenario:
    # Over the 100,000-character prompt cap on its own: no planning request.
    return Scenario(
        status="failed",
        query="Summarize this text: " + "lorem ipsum " * 9_000,
        replies=[],
    )


def _failed_by_post_plan() -> Scenario:
    return Scenario(
        status="failed",
        query="Run the right tool.",
        replies=[_tool_calls(("", {}), usage=_usage(90, 5))],
    )


def _failed_after_two_empty_replies() -> Scenario:
    return Scenario(
        status="failed",
        query="Name a prime number.",
        replies=[
            _reply("", usage=_usage(80, 0)),
            _reply("", thinking="Maybe 7, but I will not say it.", usage=_usage(95, 9)),
        ],
    )


def _gate_stopped_by_a_check() -> Scenario:
    # A continue whose utility passes either gate, but one check is false.
    return Scenario(
        status="gate_stopped",
        query="Find the largest prime below 10 with Python.",
        options={"steps_per_attempt": 1},
        replies=[
            _python("print(7)", content="Trying a guess."),
            _reflection(
                "continue",
                summary="Printed 7 without checking it.",
                next_step_is_novel=False,
                rationale="Running the same code again would not help.",
                best_effort_result="7",
            ),
        ],
    )


def _unreadable_reflection_continues() -> Scenario:
    return Scenario(
        status="completed",
        query="What is 3 + 4? Check it with Python.",
        options={"steps_per_attempt": 1},
        replies=[
            _python("print(3 + 4)"),
            _reply(PROSE),
            _reply(STILL_PROSE),
            _reply("3 + 4 = 7"),
        ],
    )


def _unreadable_reflection_at_the_final_boundary() -> Scenario:
    return Scenario(
        status="budget_exhausted",
        query="What is 3 + 4? Check it with Python.",
        options={"steps_per_attempt": 1, "max_attempts": 1},
        replies=[
            _python("print(3 + 4)", content="The answer is probably 7."),
            _reply(PROSE),
            _reply(STILL_PROSE),
        ],
    )


def _over_budget_reflection_trimmed() -> Scenario:
    # A continue of about 2,500 serialized characters, over the 2,000 of
    # MAX_REFLECTION_CHARS: trimmed, and the next attempt starts.
    bulky = _reflection(
        "continue",
        summary="Tried rhymes that Python printed.",
        failed_hypotheses=[f"hypothesis {i}: " + "f" * 280 for i in range(5)],
        useful_evidence=[f"evidence {i}: " + "e" * 200 for i in range(4)],
        recommended_avoids=["printing guesses", "near rhymes"],
        next_promising_directions=["answer with a two-word rhyme", "say none"],
        rationale="A two-word rhyme exists.",
        **LOW_UTILITY,
    )
    return Scenario(
        status="completed",
        query="Find a word that rhymes with orange.",
        options={"steps_per_attempt": 1},
        replies=[_python("print('door hinge')"), bulky, _reply("Door hinge.")],
    )


def _budget_exhausted_after_reflections() -> Scenario:
    return Scenario(
        status="budget_exhausted",
        query="Give the square root of 2 to ten decimal places.",
        options={"steps_per_attempt": 1, "max_attempts": 2},
        replies=[
            _python("print(2 ** 0.5)", content="A float first.", usage=_usage(210, 20)),
            _reflection(
                "continue",
                summary="The float has more places than asked.",
                next_promising_directions=["Format it to ten places."],
                rationale="Formatting gives exactly ten places.",
                usage=_usage(950, 90),
                **LOW_UTILITY,
            ),
            _python("print(f'{2 ** 0.5:.10f}')", usage=_usage(330, 25)),
            _reflection(
                "stop",
                summary="Printed it to ten places.",
                rationale="No attempt is left.",
                best_effort_result="1.4142135624",
                usage=_usage(1200, 70),
            ),
        ],
    )


def _budget_exhausted_without_reflection() -> Scenario:
    return Scenario(
        status="budget_exhausted",
        query="Count to two with Python.",
        options={"attempt_policy": "none", "max_steps": 2},
        replies=[_python("print(1)"), _python("print(2)")],
    )


def _stuck() -> Scenario:
    return Scenario(
        status="stuck",
        query="Wait until the counter changes.",
        options={"attempt_policy": "none"},
        embedding=False,
        replies=[_python("print('counter: 1')") for _ in range(3)],
    )


def _reflection_asked_again() -> Scenario:
    # The first reflection reply is a tool call, so it is asked again (#31).
    return Scenario(
        status="completed",
        query="What is 2 to the power of 10? Check it with Python.",
        options={"steps_per_attempt": 1},
        replies=[
            _python("print(2 ** 10)", content="Computing it."),
            _python("print(2 ** 10)"),
            _reflection(
                "continue",
                summary="Python printed 1024.",
                next_promising_directions=["Answer with 1024."],
                rationale="The answer is known.",
                **LOW_UTILITY,
            ),
            _reply("2 to the power of 10 is 1024."),
        ],
    )


def _provider_error() -> Scenario:
    refused = ProviderError(
        "POST /chat/completions failed with HTTP 400: context length exceeded",
        status_code=400,
        retryable=False,
    )
    return Scenario(
        status="provider_error", query="Summarize the news.", replies=[refused]
    )


def _setup_error() -> Scenario:
    return Scenario(
        status="setup_error",
        query="What is the weather in Rome?",
        providers=[WeatherTools(error=RuntimeError("cannot reach the weather server"))],
        replies=[],
    )


def _timeout() -> Scenario:
    # timeout_s cuts the run short while the shell command runs.
    return Scenario(
        status="timeout",
        query="Wait for the build to finish.",
        options={"timeout_s": 1.0, "builtins": ALL_BUILTINS},
        replies=[_tool_calls(("shell_exec", {"command": "sleep 30"}))],
    )


def _crashed() -> Scenario:
    return Scenario(
        status="crashed", query="Say hello.", replies=[RuntimeError("boom")]
    )


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "completed_with_an_answer": _completed_with_an_answer,
    "completed_after_tools_retrieve": _completed_after_tools_retrieve,
    "completed_after_tools_all": _completed_after_tools_all,
    "completed_by_a_reflection_done": _completed_by_a_reflection_done,
    "failed_by_pre_plan": _failed_by_pre_plan,
    "failed_by_post_plan": _failed_by_post_plan,
    "failed_after_two_empty_replies": _failed_after_two_empty_replies,
    "gate_stopped_by_a_check": _gate_stopped_by_a_check,
    "unreadable_reflection_continues": _unreadable_reflection_continues,
    "unreadable_reflection_at_the_final_boundary": (
        _unreadable_reflection_at_the_final_boundary
    ),
    "over_budget_reflection_trimmed": _over_budget_reflection_trimmed,
    "budget_exhausted_after_reflections": _budget_exhausted_after_reflections,
    "budget_exhausted_without_reflection": _budget_exhausted_without_reflection,
    "stuck": _stuck,
    "reflection_asked_again": _reflection_asked_again,
    "provider_error": _provider_error,
    "setup_error": _setup_error,
    "timeout": _timeout,
    "crashed": _crashed,
}


async def record_run(
    scenario: Scenario, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RunResult, dict[str, Any]]:
    """Run *scenario* and return its RunResult and normalized document."""
    # The file tools resolve relative paths against the process directory.
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    memory_dir = str(tmp_path / "memory")
    llm = ScriptedLLM(scenario.replies)
    agent = Agent(
        llm=llm,
        embedding=SameVectorEmbedding() if scenario.embedding else None,
        options=AgentOptions(
            memory_dir=memory_dir,
            **{**LEGACY_CACHE_OPTIONS, **LEGACY_ANSWER_OPTIONS, **scenario.options},
        ),
        tool_providers=scenario.providers,
    )
    try:
        result = await agent.run_task(scenario.query)
        window = window_json(agent.hippo.working.get_window())
    finally:
        await agent.close()
    # Every scripted reply was used: the script and the run agree.
    assert not llm.replies, f"replies left: {list(llm.replies)}"

    history = history_json(Path(memory_dir) / "history.db")
    normalize = Normalizer(
        tasks=[result.task_id, *(row["task_id"] for row in history["tasks"])],
        steps=[row["step_id"] for row in history["steps"]],
    )
    document = normalize(
        {
            "result": result_json(result, memory_dir),
            "requests": llm.requests,
            "window": window,
            "history": history,
        }
    )
    return result, document


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_run_matches_its_golden(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = SCENARIOS[name]()
    result, document = await record_run(scenario, tmp_path, monkeypatch)
    assert (result.status, result.config["reflection_gate"]) == (
        scenario.status,
        "flags",
    )
    check_golden(GOLDEN_DIR / f"{name}.json", document)


def test_every_golden_file_has_its_run() -> None:
    assert sorted(path.stem for path in GOLDEN_DIR.glob("*.json")) == sorted(SCENARIOS)


# ── The serializer itself ───────────────────────────────────────────────────


def test_ids_become_ordinals_in_order_of_first_appearance() -> None:
    task, step_1, step_2 = "a" * 32, "b" * 32, "c" * 32
    call = "call_" + "d" * 24
    normalize = Normalizer(tasks=[task], steps=[step_2, step_1])
    assert normalize(
        {
            "calls": [call, f"answers {call}"],
            "task": task,
            "steps": [step_1, step_2, step_1],
            "other": "e" * 32,
            # A digest is not an id.
            "sha256": "f" * 64,
        }
    ) == {
        "calls": ["<call:1>", "answers <call:1>"],
        "task": "<task:1>",
        "steps": ["<step:1>", "<step:2>", "<step:1>"],
        "other": "<id:1>",
        "sha256": "f" * 64,
    }


def test_a_long_text_keeps_its_start_length_and_digest() -> None:
    text = "x" * (LONG_TEXT_CHARS + 1)
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert Normalizer()([text, "x" * LONG_TEXT_CHARS]) == [
        f"{'x' * LONG_TEXT_KEPT}… [{LONG_TEXT_CHARS + 1} chars, sha256 {digest}]",
        "x" * LONG_TEXT_CHARS,
    ]
