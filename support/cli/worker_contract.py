"""Opt-in direct-worker result and event stream support."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import re
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, TextIO, cast

from localmelo.melo.agent._brief_gate import CHECK_TAIL_LINES, BriefCheckResult
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.executor import TOOL_EXIT_CODE, Executor
from localmelo.melo.executor.models import (
    ErrorCategory,
    ExecutionRequest,
    ExecutionStatus,
)
from localmelo.melo.memory.notes import _call_text, _digest_verified, _is_check
from localmelo.melo.schema import ToolCall, ToolResult
from localmelo.support.sessions.worker import redact_worker, sensitive_read

MAX_BRIEF_BYTES = 1024 * 1024
MAX_EVENT_BYTES = 1024 * 1024

# Appended to every brief-mode query (the same text on every call, so the
# task query still matches when the window drops it).
WORKER_PROTOCOL = (
    "Make the smallest change that satisfies the task as soon as you know where it "
    "goes; do not read more first. After each change, run the task's check commands. "
    "If a check fails, fix what you broke, including tests that assert the old "
    "behaviour. Before you edit a file, read the lines you will change: earlier tool "
    "output leaves your context after about ten steps. In your final answer, say "
    "which checks you ran and their result."
)
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


def _git(workspace: str, *args: str) -> str:
    """Run one read-only git command in *workspace*, raising when it fails."""
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", "-C", workspace, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=3,
        check=True,
    ).stdout.strip()


def workspace_head(workspace: str) -> tuple[str | None, bool | None]:
    """Return the workspace HEAD and dirty flag, or unknown on failure."""
    try:
        head = _git(workspace, "rev-parse", "--verify", "HEAD") or None
    except (OSError, subprocess.SubprocessError):
        head = None
    try:
        dirty: bool | None = bool(_git(workspace, "status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return head, dirty


# At most this many untracked files are looked at by workspace_fingerprint.
_FINGERPRINT_UNTRACKED = 200


def workspace_fingerprint(workspace: str) -> str | None:
    """A digest of the workspace's changes, ``None`` outside a git tree.

    It covers the status, the diff of tracked files against HEAD and the
    size and mtime of untracked files, so two equal digests mean that
    nothing was edited in between, also in a tree that was dirty already.
    """
    try:
        parts = [_git(workspace, "status", "--porcelain")]
        try:
            parts.append(_git(workspace, "diff", "HEAD"))
        except subprocess.CalledProcessError:  # no commit yet
            parts.append(_git(workspace, "diff"))
        names = _git(workspace, "ls-files", "-o", "--exclude-standard").splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    for name in names[:_FINGERPRINT_UNTRACKED]:
        try:
            info = os.stat(os.path.join(workspace, name))
        except OSError:
            continue
        parts.append(f"{name} {info.st_size} {info.st_mtime_ns}")
    return hashlib.sha256("\0".join(parts).encode("utf-8", "replace")).hexdigest()


# Printed after a check that ran without a workspace root, whose shell_exec
# records no exit status, so the gate can read the status from the output.
_EXIT_MARK = "__melo_check_exit__"
_EXIT_LINE = re.compile(rf"\n?{_EXIT_MARK}(\d+)\n?\Z")
_REFUSED = (
    ErrorCategory.TOOL_NOT_FOUND,
    ErrorCategory.TOOL_NOT_OFFERED,
    ErrorCategory.INVALID_ARGUMENTS,
)


async def run_brief_check(executor: Executor, command: str) -> BriefCheckResult:
    """Run a brief's check as a ``shell_exec`` call of the run's executor.

    The call takes the executor's whole path, so the mode's policy, the
    workspace root and the shell timeout apply as they do to the model's
    own calls. A call that the path refuses does not run and is reported
    in ``refused``. The check passes on exit status 0.
    """
    wrapped = command
    if executor.workspace_root is None:
        wrapped = (
            f"(\n{command}\n)\n__melo_rc=$?\n"
            f"printf '\\n{_EXIT_MARK}%s\\n' \"$__melo_rc\"\nexit $__melo_rc"
        )
    codes: list[int] = []
    slot = TOOL_EXIT_CODE.set(codes)
    started = time.perf_counter()
    try:
        outcome = await executor.execute_structured(
            ExecutionRequest(tool_name="shell_exec", arguments={"command": wrapped})
        )
    finally:
        TOOL_EXIT_CODE.reset(slot)
    duration = time.perf_counter() - started
    ran = outcome.status is ExecutionStatus.SUCCESS
    text = outcome.output if ran else outcome.error
    exit_code = codes[-1] if codes else None
    if ran and exit_code is None:
        found = _EXIT_LINE.search(text)
        if found is not None:
            exit_code, text = int(found.group(1)), text[: found.start()]
    refused = ""
    if outcome.status is ExecutionStatus.BLOCKED or outcome.error_category in _REFUSED:
        refused = outcome.error
    # The shell_exec line that repeats the status the result already holds.
    text = re.sub(r"\A\[exit code -?\d+\]\n?", "", text)
    tail = "\n".join(text.splitlines()[-CHECK_TAIL_LINES:])
    return BriefCheckResult(
        command=redact_output(command),
        exit_code=exit_code,
        duration_s=duration,
        output=redact_output(tail),
        refused=redact_output(refused),
    )


def failure_record(status: str, workspace: str, error: str) -> dict[str, Any]:
    """Build a redacted report when there is no completed run result."""
    head, dirty = workspace_head(workspace)
    return cast(
        dict[str, Any],
        redact_output(
            {
                "v": 1,
                "status": status,
                "run_id": None,
                "exit_code": 1,
                "head_sha": head,
                "git_dirty": dirty,
                "tests": {"state": "not_recorded", "items": []},
                "run": None,
                "error": error,
            }
        ),
    )


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
        # The detail of each brief_check notice (see run_brief_check).
        self.gate: list[dict[str, Any]] = []
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
                written = self.out.write(line)
                if written != len(line):
                    raise OSError("short worker output write")
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
        # Backpressure, never loss: a slow reader stalls the worker at the
        # planning boundary, while a failed writer sets ``error`` and lets the
        # blocked producer fall through to the transport failure path.
        while self.error is None:
            try:
                self.lines.put(line, timeout=0.05)
                return
            except queue.Full:
                continue

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
        if event.type == "notice" and data.get("kind") == "brief_check":
            self.gate.append(dict(data.get("detail") or {}))
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
        if event.type == "inbox.accepted":
            payload["data"].pop("text", None)
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

    def _gate_tests(self) -> dict[str, Any]:
        """The checks of the brief gate's last round: ``passed`` or ``failed``."""
        last = max(detail.get("round", 0) for detail in self.gate)
        latest = [detail for detail in self.gate if detail.get("round", 0) == last]
        items = []
        for detail in latest:
            command = redact_output(str(detail.get("command", "")))
            item = {
                "command": command[:2000],
                "command_truncated": len(command) > 2000,
                "exit_code": detail.get("exit_code"),
                "duration_s": detail.get("duration_s"),
                "source": "brief_check",
            }
            if detail.get("refused"):
                item["refused"] = detail["refused"]
            items.append(item)
        passed = all(
            detail.get("exit_code") == 0 and not detail.get("refused")
            for detail in latest
        )
        return {"state": "passed" if passed else "failed", "items": items}

    def tests(self) -> dict[str, Any]:
        if self.gate:
            return self._gate_tests()
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

    def transport_result(self, record: dict[str, Any]) -> None:
        """Try to append a failure record after an event sink failure."""
        if self.thread is None or not self.thread.is_alive():
            return
        line = (
            json.dumps(
                {"v": 1, "type": "result", "result": record},
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        with contextlib.suppress(queue.Full):
            self.lines.put(line, timeout=1)

    def close(self) -> None:
        if self.thread is None:
            return
        # Wait for the writer rather than give up on a full queue: a slow
        # reader stalls the worker, and a failed writer has set ``error``.
        while self.thread.is_alive():
            try:
                self.lines.put(None, timeout=0.05)
                break
            except queue.Full:
                continue
        self.thread.join()
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
