"""Clipboard tests never write the person's real macOS clipboard."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest

from localmelo.support.cli import terminal


class FakeProcess:
    def __init__(self, code: int = 0, *, block: bool = False) -> None:
        self.code, self.block = code, block
        self.returncode: int | None = None
        self.data = b""
        self.started = asyncio.Event()
        self.killed = self.waited = False

    async def communicate(self, data: bytes) -> tuple[None, None]:
        self.data = data
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        self.returncode = self.code
        return None, None

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.waited = True
        return self.returncode or 0


@pytest.fixture
async def native(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(terminal.sys, "platform", "darwin")
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    state: dict[str, Any] = {"process": FakeProcess(), "calls": []}

    async def spawn(*args: Any, **kwargs: Any) -> FakeProcess:
        state["calls"].append((args, kwargs))
        if "error" in state:
            raise state["error"]
        return state["process"]

    monkeypatch.setattr(terminal.asyncio, "create_subprocess_exec", spawn)
    return state


def decoded(sequence: str) -> str:
    assert sequence.startswith("\x1b]52;c;") and sequence.endswith("\x07")
    assert sequence.count("\x1b") == 1
    return base64.b64decode(sequence[len("\x1b]52;c;") : -1], validate=True).decode(
        "utf-8"
    )


async def test_local_mac_uses_bounded_pbcopy_without_terminal_controls(
    native: dict[str, Any],
) -> None:
    controls: list[str] = []
    assert (
        await terminal.copy_text(
            "```python\nprint('你好')\n```", lambda s: controls.append(s) or True
        )
        == "pbcopy"
    )
    process = native["process"]
    assert process.data.decode() == "```python\nprint('你好')\n```"
    assert not controls and not process.killed
    [(args, kwargs)] = native["calls"]
    assert args == (terminal.PBCOPY,)
    assert kwargs == {
        "stdin": asyncio.subprocess.PIPE,
        "stdout": asyncio.subprocess.DEVNULL,
        "stderr": asyncio.subprocess.DEVNULL,
    }


@pytest.mark.parametrize("environment", ["SSH_TTY", "SSH_CONNECTION"])
async def test_ssh_uses_encoded_osc52_not_the_remote_macos_clipboard(
    native: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
) -> None:
    monkeypatch.setenv(environment, "test-session")
    controls: list[str] = []
    text = "markdown\n\x1b]52;c;not-a-second-control\x07\n你好"
    assert (
        await terminal.copy_text(text, lambda s: controls.append(s) or True) == "osc52"
    )
    assert native["calls"] == []
    assert decoded(controls[0]) == text


async def test_other_platforms_request_terminal_clipboard(
    native: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(terminal.sys, "platform", "linux")
    controls: list[str] = []
    assert (
        await terminal.copy_text("answer", lambda s: controls.append(s) or True)
        == "osc52"
    )
    assert not native["calls"] and decoded(controls[0]) == "answer"


@pytest.mark.parametrize("failure", ["exit", "missing"])
async def test_native_failure_falls_back_without_claiming_native_success(
    native: dict[str, Any], failure: str
) -> None:
    if failure == "exit":
        native["process"] = FakeProcess(code=1)
    else:
        native["error"] = FileNotFoundError("fake pbcopy is missing")
    controls: list[str] = []
    assert (
        await terminal.copy_text("stored answer", lambda s: controls.append(s) or True)
        == "osc52"
    )
    assert decoded(controls[0]) == "stored answer"


async def test_timeout_kills_and_reaps_native_child_before_fallback(
    native: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native["process"] = FakeProcess(block=True)
    monkeypatch.setattr(terminal, "CLIPBOARD_TIMEOUT_S", 0.01)
    controls: list[str] = []
    assert (
        await terminal.copy_text("answer", lambda s: controls.append(s) or True)
        == "osc52"
    )
    assert native["process"].killed and native["process"].waited
    assert decoded(controls[0]) == "answer"


async def test_cancellation_kills_native_child_and_never_falls_back(
    native: dict[str, Any],
) -> None:
    process = FakeProcess(block=True)
    native["process"] = process
    controls: list[str] = []
    task = asyncio.create_task(
        terminal.copy_text("answer", lambda s: controls.append(s) or True)
    )
    await process.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.killed and process.waited and not controls


async def test_detached_terminal_does_not_report_a_sent_request(
    native: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSH_TTY", "test-session")
    with pytest.raises(ValueError, match="Could not send"):
        await terminal.copy_text("answer", lambda _: False)
