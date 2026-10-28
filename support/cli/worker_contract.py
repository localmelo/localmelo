"""Opt-in direct-worker result and event stream support."""

from __future__ import annotations

import json
import os
import queue
import re
import stat
import subprocess
import threading
from pathlib import Path
from typing import Any, TextIO, cast

from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.memory.notes import _call_text, _digest_verified, _is_check
from localmelo.melo.schema import ToolCall, ToolResult
from localmelo.support.sessions.worker import redact_worker, sensitive_read

MAX_BRIEF_BYTES = 1024 * 1024
MAX_EVENT_BYTES = 1024 * 1024
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b((?=[A-Za-z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD|PASSWD|CREDENTIAL))[A-Za-z_][A-Za-z0-9_]*[ \t]*[:=][ \t]*)[^\r\n]*"
)


def redact_output(value: Any) -> Any:
    """Mask common secret assignments in addition to worker redaction."""

    def mask(item: Any) -> Any:
        if isinstance(item, str):
            return _SECRET_ASSIGNMENT.sub(r"\1[redacted]", item)
        if isinstance(item, dict):
            return {mask(key): mask(child) for key, child in item.items()}
        if isinstance(item, list):
            return [mask(child) for child in item]
        return item

    return mask(redact_worker(value))


def read_brief(path: str | Path) -> str:
    """Read one bounded UTF-8 brief without opening a FIFO or device."""
    path = Path(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(f"{path}: brief is not a regular file")
            with os.fdopen(fd, "rb", closefd=False) as source:
                raw = source.read(MAX_BRIEF_BYTES + 1)
        finally:
            os.close(fd)
        if len(raw) > MAX_BRIEF_BYTES:
            raise ValueError("brief exceeds 1 MiB")
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read brief: {exc}") from exc
    if not text.strip():
        raise ValueError("brief is empty")
    return text


def workspace_head(workspace: str) -> tuple[str | None, bool | None]:
    """Return the workspace HEAD and dirty flag, or unknown on failure."""
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env["GIT_OPTIONAL_LOCKS"] = "0"

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", workspace, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout.strip()

    try:
        head = git("rev-parse", "--verify", "HEAD") or None
    except (OSError, subprocess.SubprocessError):
        head = None
    try:
        dirty: bool | None = bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return head, dirty


class WorkerObserver:
    """Collect check evidence and, optionally, write bounded redacted NDJSON."""

    def __init__(
        self, out: TextIO, workspace: str, *, stream: bool, on_failure: Any = None
    ) -> None:
        self.out = out
        self.on_failure = on_failure
        self.workspace = workspace
        self.stream = stream
        self.error: Exception | None = None
        self.pending: dict[str, tuple[ToolCall, str | None]] = {}
        self.checks: list[tuple[ToolCall, ToolResult]] = []
        self.check_exit: int | None = None
        self.attempt = 0
        self.lines: queue.Queue[str | None] = queue.Queue(maxsize=64)
        self.thread: threading.Thread | None = None
        if stream:
            self.thread = threading.Thread(target=self._write, daemon=True)
            self.thread.start()

    def _fail(self, exc: Exception) -> None:
        if self.error is None:
            self.error = exc
            if self.on_failure is not None:
                self.on_failure()

    def _write(self) -> None:
        while True:
            line = self.lines.get()
            if line is None:
                return
            try:
                self.out.write(line)
                self.out.flush()
            except Exception as exc:
                self._fail(exc)
                return

    def _enqueue(self, value: dict[str, Any]) -> None:
        if self.error is not None:
            return
        line = json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n"
        if len(line.encode("utf-8")) > MAX_EVENT_BYTES:
            self._fail(ValueError("worker event exceeds 1 MiB"))
            return
        try:
            self.lines.put_nowait(line)
        except queue.Full:
            self._fail(ValueError("worker event output queue is full"))

    def __call__(self, event: AgentEvent) -> None:
        try:
            self._observe(event)
        except Exception as exc:
            self._fail(exc)

    def _observe(self, event: AgentEvent) -> None:
        data = event.data
        if event.type == "step.started":
            attempt = data.get("attempt", 0)
            if attempt != self.attempt:
                self.attempt = attempt
                self.checks.clear()
                self.check_exit = None
        if event.type == "tool.started":
            call = ToolCall(
                str(data.get("name", "")),
                data.get("arguments") or {},
                id=str(data.get("call_id", "")),
            )
            path = sensitive_read(call.tool_name, call.arguments, self.workspace)
            self.pending[str(data.get("call_id", ""))] = (call, path)
        elif event.type == "tool.finished":
            pair = self.pending.pop(str(data.get("call_id", "")), None)
            if pair and not pair[1]:
                call = pair[0]
                result = ToolResult(
                    call.tool_name,
                    str(data.get("output", "")),
                    str(data.get("error", "")),
                )
                if data.get("status") != "success" or data.get("exit_code") not in (
                    None,
                    0,
                ):
                    result.error = result.error or "check did not pass"
                if _is_check(call, (), self.workspace):
                    self.checks.append((call, result))
                    exit_code = data.get("exit_code")
                    self.check_exit = exit_code if type(exit_code) is int else None
        if not self.stream:
            return
        payload = redact_output(event.to_dict())
        if event.type == "run.finished" and isinstance(
            payload.get("data", {}).get("result"), dict
        ):
            payload["data"]["result"].pop("config", None)
        if event.type == "tool.finished":
            path = pair[1] if pair else None
            if path:
                safe = payload["data"]
                safe["output"] = f"[Sensitive file content omitted: {path}]"
                safe["output_chars"] = len(safe["output"])
                safe["error"] = ""
                safe["images"] = []
        self._enqueue({"v": 1, "type": "event", "event": payload})

    def tests(self) -> dict[str, Any]:
        evidence = _digest_verified(self.checks, self.workspace)
        if not evidence:
            return {"state": "not_recorded", "items": []}
        command = redact_output(_call_text(self.checks[-1][0]))
        return {
            "state": "recorded",
            "items": [
                {
                    "command": command[:2000],
                    "command_truncated": len(command) > 2000,
                    "exit_code": self.check_exit,
                    "source": "tool_event",
                }
            ],
        }

    def close(self) -> None:
        if self.thread is None:
            return
        try:
            self.lines.put(None, timeout=1)
        except queue.Full:
            self.error = self.error or ValueError("worker event output queue is full")
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            self.error = self.error or TimeoutError("worker event output did not drain")


def result_record(
    outcome: Any, observer: WorkerObserver, workspace: str, code: int
) -> dict[str, Any]:
    head, dirty = workspace_head(workspace)
    run = outcome.to_dict()
    run.pop("config", None)
    return cast(
        dict[str, Any],
        redact_output(
            {
                "v": 1,
                "status": outcome.status,
                "run_id": outcome.task_id,
                "exit_code": code,
                "head_sha": head,
                "git_dirty": dirty,
                "tests": observer.tests(),
                "run": run,
                "error": outcome.error or None,
            }
        ),
    )
