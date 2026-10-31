"""The work beside the chat: the gateway's tasks, in the status bar and a list.

The terminal reads and changes tasks only through ``melo task``'s own client
(:func:`~localmelo.support.task_cli.connect` and
:func:`~localmelo.support.task_cli.execute`): the same requests and checks,
and no other way in. The gateway decides what a change may do; the list
offers an action only where the task's state allows it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import httpx

from localmelo.support import task_cli
from localmelo.support.usage_snapshot import BarPart

# The key that opens the list, as the bar names it and prompt_toolkit binds it.
WORK_KEY: Final = "ctrl+t"
WORK_BINDING: Final = "c-t"
# Seconds between two looks at the gateway's tasks while the UI is open.
POLL_S: Final = 5.0
# How long one look may take: a loopback gateway answers at once.
TIMEOUT_S: Final = 2.0
PAUSED: Final = frozenset({"parked", "interrupted"})
QUEUED: Final = frozenset({"queued", "waiting"})
LIVE: Final = frozenset({"running", "held", *QUEUED, *PAUSED})
# P5 is the paused level: lowering stops at P4, and Pause goes further.
LOWEST_RUNNING: Final = 4


class WorkError(Exception):
    """The gateway could not be asked, or refused; the message says why."""


@dataclass(frozen=True)
class WorkTask:
    """One task that has not finished, as the gateway lists it."""

    task_id: str
    title: str
    state: str
    level: int | None = None

    @property
    def paused(self) -> bool:
        return self.state in PAUSED

    @property
    def state_word(self) -> str:
        return "paused" if self.state == "parked" else self.state

    @property
    def level_text(self) -> str:
        return "" if self.level is None else f"P{self.level}"

    @classmethod
    def from_json(cls, data: object) -> WorkTask | None:
        if not isinstance(data, Mapping):
            return None
        task_id, state, level = (
            data.get("task_id"),
            data.get("state"),
            data.get("level"),
        )
        if not isinstance(task_id, str) or not isinstance(state, str):
            return None
        title = data.get("title")
        return cls(
            task_id,
            title if isinstance(title, str) and title else task_id,
            state,
            level if type(level) is int and 0 <= level <= 5 else None,
        )


@dataclass(frozen=True)
class WorkSummary:
    """The tasks beside the chat by state, and the top priority of those not paused."""

    running: int = 0
    queued: int = 0
    held: int = 0
    paused: int = 0
    top: int | None = None

    @property
    def any(self) -> bool:
        return bool(self.running or self.queued or self.held or self.paused)


def summarize(tasks: Sequence[WorkTask]) -> WorkSummary:
    levels = [t.level for t in tasks if not t.paused and t.level is not None]
    return WorkSummary(
        running=sum(t.state == "running" for t in tasks),
        queued=sum(t.state in QUEUED for t in tasks),
        held=sum(t.state == "held" for t in tasks),
        paused=sum(t.paused for t in tasks),
        top=min(levels, default=None),
    )


def work_parts(summary: WorkSummary) -> list[BarPart]:
    """``2 running · 1 queued · P1 · ctrl+t tasks``; nothing without tasks.

    A held task waits for the person (an approval), so it is drawn as a
    warning; the counts are figures, the priority and the key details.
    """
    if not summary.any:
        return []
    parts = [
        BarPart(f"{count} {word}", role)
        for count, word, role in (
            (summary.running, "running", "live"),
            (summary.queued, "queued", ""),
            (summary.held, "held", "warn"),
            (summary.paused, "paused", ""),
        )
        if count
    ]
    if summary.top is not None:
        parts.append(BarPart(f"P{summary.top}", "figure", detail=True))
    return [*parts, BarPart(f"{WORK_KEY} tasks", "key", detail=True)]


@dataclass(frozen=True)
class Action:
    """One thing the list can do to a task: ``melo task <command> <id> [level]``."""

    label: str
    command: str
    level: str | None = None


def actions(task: WorkTask) -> list[Action]:
    """What the task's state allows, in the order the list shows it."""
    if task.state not in LIVE:
        return []
    found = [Action("Resume", "resume")] if task.paused else [Action("Pause", "pause")]
    level = task.level
    if not task.paused and level is not None:
        for step, verb in ((-1, "Raise"), (1, "Lower")):
            to = level + step
            if 0 <= to <= LOWEST_RUNNING:
                found.append(Action(f"{verb} to P{to}", "level", f"P{to}"))
    return [*found, Action("Stop", "stop")]


class TaskSource(Protocol):
    """Where the tasks come from: the gateway, or a fake in a test."""

    def available(self) -> bool: ...

    def list(self) -> list[WorkTask]: ...

    def act(self, task: WorkTask, action: Action) -> WorkTask | None: ...


class GatewayTasks:
    """The local gateway's tasks, read and changed as ``melo task`` does it."""

    def __init__(
        self,
        connect: Callable[..., httpx.Client] = task_cli.connect,
        token: Callable[[], Path] = task_cli.token_path,
    ) -> None:
        self._connect, self._token = connect, token

    def available(self) -> bool:
        """Whether a gateway has ever run here (its token exists): no request."""
        try:
            return self._token().exists()
        except OSError:
            return False

    def _run(self, **fields: Any) -> dict[str, Any]:
        args = argparse.Namespace(json=True, **fields)
        out = io.StringIO()
        try:
            with self._connect(timeout=TIMEOUT_S) as client:
                task_cli.execute(args, client, out=out, stdin=io.StringIO())
            data = json.loads(out.getvalue())
        except Exception as exc:  # a bad config or gateway never ends the UI
            raise WorkError(str(exc) or type(exc).__name__) from exc
        if not isinstance(data, dict):
            raise WorkError("gateway returned an invalid task response")
        return data

    def list(self) -> list[WorkTask]:
        """The tasks that have not finished (blocking: call it in a thread)."""
        data = self._run(command="list", state=None, level=None)
        found = [WorkTask.from_json(item) for item in data.get("tasks") or []]
        return [task for task in found if task is not None and task.state in LIVE]

    def act(self, task: WorkTask, action: Action) -> WorkTask | None:
        """Do *action* to *task* (blocking); the task as the gateway now has it."""
        data = self._run(command=action.command, id=task.task_id, level=action.level)
        return WorkTask.from_json(data.get("task"))


def task_rows(tasks: Sequence[WorkTask]) -> list[str]:
    """The list's rows: ``P1  running  Fix the parser``, columns lined up."""
    states = max((len(task.state_word) for task in tasks), default=0)
    labels = [task.level_text or "--" for task in tasks]
    width = max((len(label) for label in labels), default=2)
    return [
        f"{label:<{width}}  {task.state_word:<{states}}  {task.title}"
        for label, task in zip(labels, tasks, strict=True)
    ]


def done_text(task: WorkTask, action: Action, after: WorkTask | None) -> str:
    """What the hint says once the gateway took *action*."""
    if action.command == "stop" and after is not None and after.state in LIVE:
        # A running task's stop lands when its worker has ended.
        return f"Stop requested: {task.title} (it ends when its worker does)"
    if action.command == "level":
        return f"{task.title} is now {action.level}"
    word = {"pause": "Paused", "resume": "Resumed", "stop": "Stopped"}
    return f"{word[action.command]}: {task.title}"


class _App(Protocol):
    def hint(self, text: str, seconds: float = ...) -> None: ...

    async def choose_from(
        self, title: str, options: Sequence[str], *, default: int = 0
    ) -> int | None: ...


class WorkBeside:
    """The tasks the bar counts and ctrl+t lists, kept fresh by a slow look.

    :meth:`watch` asks the gateway every :data:`POLL_S` seconds while the UI
    is open, and only when a gateway token exists; *on_change* (the bar's
    redraw) runs only when the counts change. Levels read as P-codes.
    """

    def __init__(
        self,
        source: TaskSource,
        *,
        poll_s: float = POLL_S,
        on_change: Callable[[], None] = lambda: None,
    ) -> None:
        self.source, self.poll_s, self.on_change = source, poll_s, on_change
        self.tasks: list[WorkTask] = []
        self.summary = WorkSummary()

    async def refresh(self) -> None:
        tasks: list[WorkTask] = []
        if self.source.available():
            with contextlib.suppress(WorkError):
                tasks = await asyncio.to_thread(self.source.list)
        self.tasks = tasks
        if summarize(tasks) != self.summary:
            self.summary = summarize(tasks)
            self.on_change()

    async def watch(self) -> None:
        while True:
            await self.refresh()
            await asyncio.sleep(self.poll_s)

    async def open(self, app: _App) -> None:
        """The list of tasks; Enter on one lists what can be done to it."""
        await self.refresh()
        tasks = list(self.tasks)
        if not tasks:
            app.hint("No tasks beside this chat: melo task new adds one", 3.0)
            return
        rows = task_rows(tasks)
        index = await app.choose_from("Tasks beside this chat", rows)
        if index is None:
            return
        task = tasks[index]
        choices = actions(task)
        title = " · ".join(
            p for p in (task.title, task.level_text, task.state_word) if p
        )
        picked = await app.choose_from(title, [choice.label for choice in choices])
        if picked is None:
            return
        action = choices[picked]
        if action.command == "stop":
            sure = await app.choose_from(
                f"Stop {task.title}? A stopped task cannot be resumed.",
                ["Stop it", "Keep it"],
                default=1,
            )
            if sure != 0:
                return
        try:
            after = await asyncio.to_thread(self.source.act, task, action)
        except WorkError as exc:
            app.hint(f"Could not {action.label.lower()}: {exc}", 4.0)
            return
        app.hint(done_text(task, action, after), 3.0)
        await self.refresh()
