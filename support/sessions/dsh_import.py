"""Import a dsh JSONL event stream as one read-only worker session.

The import is one SQLite transaction. The content-addressed session id is also
its concurrency key: no session or partial turn is visible until commit.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localmelo.melo.contracts.events import EVENT_SCHEMA_VERSION, json_safe_payload
from localmelo.melo.schema import RunResult
from localmelo.support.sessions.store import (
    SessionStore,
    _dumps,
    _session_event,
    title_from_text,
)
from localmelo.support.sessions.worker import redact_worker, sensitive_read

_FALLBACK = "Imported dsh worker run"


@dataclass(frozen=True)
class ImportResult:
    session_id: str
    imported: bool
    skipped: int


def _safe(value: Any) -> Any:
    return redact_worker(value)


def _text(value: Any) -> str:
    return _safe(value) if isinstance(value, str) else ""


def _usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    keys = {
        "inputTokens": "prompt_tokens",
        "outputTokens": "completion_tokens",
        "totalTokens": "total_tokens",
        "cacheReadTokens": "cache_read_tokens",
        "cacheWriteTokens": "cache_write_tokens",
    }
    return {
        target: value[source]
        for source, target in keys.items()
        if isinstance(value.get(source), int)
        and not isinstance(value[source], bool)
        and value[source] >= 0
    }


class _Mapper:
    def __init__(self, run_id: str, now: float) -> None:
        self.run_id = run_id
        self.now = now
        self.events: list[dict[str, Any]] = []
        self.step = 1
        self.started_step = False
        self.thinking = ""
        self.text = ""
        self.calls: dict[str, tuple[str, int, str | None]] = {}
        self.usage: dict[str, int] = {}
        self.last_usage: dict[str, int] = {}
        self.final: str | None = None
        self.skipped = 0
        self.tool_count = 0
        self.emit(
            "run.started",
            query=_FALLBACK,
            max_steps=0,
            workspace_root=None,
            approval=False,
        )

    def emit(self, kind: str, **data: Any) -> None:
        self.events.append(
            {
                "v": EVENT_SCHEMA_VERSION,
                "type": kind,
                "run_id": self.run_id,
                "seq": len(self.events) + 1,
                "ts": self.now,
                "data": json_safe_payload(_safe(data)),
            }
        )

    def start_step(self, step: int) -> None:
        if self.started_step and step == self.step:
            return
        self.flush()
        self.step = step
        self.started_step = True
        self.emit("step.started", step=step, attempt=1)

    def flush(self, usage: dict[str, int] | None = None) -> None:
        if self.thinking or self.text or usage:
            self.emit(
                "assistant.message",
                step=self.step,
                content=self.text,
                thinking=self.thinking,
                tool_calls=[],
                usage=usage or {},
                duration_s=None,
            )
            self.thinking = self.text = ""

    def add(self, row: Any) -> None:
        if not isinstance(row, dict):
            self.skipped += 1
            return
        kind = row.get("type")
        if not isinstance(kind, str):
            self.skipped += 1
            return
        step = row.get("step")
        if isinstance(step, int) and not isinstance(step, bool) and step > 0:
            self.start_step(step)
        elif (
            kind in {"thinking", "text", "tool_call", "tool_result"}
            and not self.started_step
        ):
            self.start_step(1)
        if kind == "thinking":
            self.thinking += _text(row.get("text"))
        elif kind == "text":
            self.text += _text(row.get("text"))
        elif kind == "tool_call":
            self.flush()
            call_id = _text(row.get("callId")) or f"call-{self.tool_count + 1}"
            name = _text(row.get("tool")) or "unknown"
            args = row.get("input") if isinstance(row.get("input"), dict) else {}
            sensitive = sensitive_read(name, args)
            if call_id in self.calls and self.calls[call_id][2]:
                sensitive = self.calls[call_id][2]
            args = _safe(args)
            self.calls[call_id] = (name, self.step, sensitive)
            self.tool_count += 1
            self.emit(
                "tool.started",
                step=self.step,
                call_id=call_id,
                name=name,
                arguments=args,
                preview="",
            )
        elif kind == "tool_result":
            self.flush()
            call_id = _text(row.get("callId"))
            if not call_id and len(self.calls) == 1:
                call_id = next(iter(self.calls))
            name, call_step, sensitive = self.calls.pop(
                call_id, ("unknown", self.step, None)
            )
            if name == "unknown" and sensitive is None:
                sensitive = next(
                    (path for _, _, path in self.calls.values() if path), None
                )
            output = (
                f"[Content of {sensitive} omitted]"
                if sensitive is not None
                else _text(row.get("result"))
            )
            status = str(row.get("status") or "").lower()
            status = "success" if status in {"success", "ok", "completed"} else "error"
            self.emit(
                "tool.finished",
                step=call_step,
                call_id=call_id,
                name=name,
                status=status,
                error_category="none" if status == "success" else "unknown",
                output=output,
                output_chars=len(output),
                error="",
                duration_s=None,
                images=[],
                approval=None,
                exit_code=None,
            )
        elif kind == "final":
            self.final = _text(row.get("text"))
        elif kind != "usage":
            self.skipped += 1
        # dsh's usage in the sampled log is cumulative, including on step
        # records; emit only its delta, allowing a counter reset.
        current = _usage(row.get("usage") if kind != "usage" else row.get("usage", row))
        if current:
            delta = {
                k: (
                    n - self.last_usage.get(k, 0)
                    if n >= self.last_usage.get(k, 0)
                    else n
                )
                for k, n in current.items()
            }
            for k, n in delta.items():
                self.usage[k] = self.usage.get(k, 0) + n
            self.last_usage.update(current)
            self.flush(delta)

    def finish(self) -> dict[str, Any]:
        self.flush()
        result = RunResult(
            task_id=self.run_id,
            status="completed" if self.final is not None else "crashed",
            result=self.final or "",
            error="" if self.final is not None else "Import has no final record.",
            steps=self.step if self.started_step else 0,
            attempts=1 if self.started_step else 0,
            n_tool=self.tool_count,
            usage=self.usage,
        ).to_dict()
        self.emit("run.finished", result=result)
        return result


async def import_dsh(
    path: str | Path,
    *,
    title: str | None = None,
    store: SessionStore | None = None,
) -> ImportResult:
    """Import *path* once by SHA-256; invalid lines are counted, never echoed.

    A supplied store must be a CLI store. When omitted, the default CLI file is
    opened and closed here. A no-op repeat does not modify the original title.
    """
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    session_id = "dsh-" + digest
    mapper = _Mapper(session_id, 0.0)
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError, RecursionError):
            mapper.skipped += 1
            continue
        mapper.add(row)
    result = mapper.finish()
    safe_title = title_from_text(_text(title) or _FALLBACK)
    own = store is None
    store = store or SessionStore(writer="cli")
    if store.writer != "cli":
        raise ValueError("dsh imports require a CLI session store")
    if own:
        await store.open()
    try:
        async with store._lock:
            conn = await store._ready()
            store._check_writable()
            async with store._transaction(conn, "import_dsh") as stored:
                rows = await conn.execute_fetchall(
                    "SELECT id FROM sessions WHERE id = ?", (session_id,)
                )
                if rows:
                    return ImportResult(session_id, False, mapper.skipped)
                now = store._clock()
                turn_id = uuid.uuid4().hex
                holder = "dsh-import"
                await conn.execute(
                    "INSERT INTO sessions (id, kind, title, created_at, updated_at, model, workspace, meta) "
                    "VALUES (?, 'worker', ?, ?, ?, '', NULL, ?)",
                    (session_id, safe_title, now, now, _dumps({"source": "dsh"})),
                )
                await conn.execute(
                    "INSERT INTO turns (session_id, turn, id, run_id, user_text, typed_text, holder, started_at) "
                    "VALUES (?, 1, ?, ?, ?, '', ?, ?)",
                    (session_id, turn_id, session_id, _FALLBACK, holder, now),
                )
                await store._insert_event(
                    conn,
                    stored,
                    session_id,
                    1,
                    _session_event(
                        "turn.started",
                        {
                            "turn": 1,
                            "turn_id": turn_id,
                            "user_text": _FALLBACK,
                            "client_request_id": None,
                        },
                        now,
                    ),
                )
                for event in mapper.events:
                    event["ts"] = now
                    await store._insert_event(
                        conn, stored, session_id, 1, store._prepared_event(event)
                    )
                await store._end_turn(
                    conn,
                    stored,
                    session_id,
                    1,
                    holder=holder,
                    status=result["status"],
                    cause="run",
                    answer=result["result"],
                    note="",
                    result=_dumps(result),
                    error=result["error"],
                    now=now,
                    approval_reason="",
                )
        return ImportResult(session_id, True, mapper.skipped)
    finally:
        if own:
            await store.aclose()
