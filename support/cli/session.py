"""One terminal session: its turns, its commands, and how it stops.

:class:`CliSession` is what the app drives. It owns the session store, the
Agent, the runner and the approval policy, and runs each turn, command and
switch as a task of its own (never as one of the app's), so leaving the
app never cancels a turn behind shutdown's back.

A stop goes through ``runner.cancel``, never ``task.cancel()``: while the
runner still writes the turn's row, ``cancel`` returns False, so the stop
is kept and tried again on every tick until it lands; the turn then ends
``cancelled`` with cause ``user_stop``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import sys
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from typing import Any, Final, Literal, cast

from prompt_toolkit.application.application import attach_winch_signal_handler
from prompt_toolkit.application.current import set_app
from prompt_toolkit.application.run_in_terminal import in_terminal
from prompt_toolkit.document import Document
from rich.text import Text

from localmelo.melo.agent import Agent
from localmelo.melo.agent.agent import FINALIZE_TIMEOUT_S
from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.melo.contracts.streaming import STREAM_DISPLAY
from localmelo.melo.schema import RunResult
from localmelo.support import config
from localmelo.support.backends import discovery
from localmelo.support.backends.openai_compat import normalize_url
from localmelo.support.cli import browsing, commands, usershell
from localmelo.support.cli.agents import AgentFactory
from localmelo.support.cli.app import InteractiveApp, StatusBar
from localmelo.support.cli.approvals import (
    Approvals,
    Mode,
    TerminalApprover,
    approval_words,
)
from localmelo.support.cli.attachments import (
    Attachment,
)
from localmelo.support.cli.attachments import (
    add as add_attachment,
)
from localmelo.support.cli.attachments import (
    load as load_attachment,
)
from localmelo.support.cli.output import Out
from localmelo.support.cli.picker import ModelRow
from localmelo.support.cli.render import (
    EXPAND_LINES,
    STREAM_BUFFERED,
    ToolOutput,
    TurnRenderer,
)
from localmelo.support.cli.session_tools import SessionTools
from localmelo.support.cli.sessions import BUSY, DELETED, SessionNavigation
from localmelo.support.cli.steps import STEPS_KEY, learned_total, step_figure
from localmelo.support.cli.text import expand_tabs, wrap_cells
from localmelo.support.cli.theme import ASCII, in_symbols
from localmelo.support.cli.views import SOURCE_TEXT, BackendInfo, hang
from localmelo.support.cli.work import WorkBeside, work_parts
from localmelo.support.progress_line import run_elapsed, wall_range
from localmelo.support.sessions.attachments import TextAttachment
from localmelo.support.sessions.display import (
    clean_output,
    format_duration,
    format_tokens,
    home_label,
)
from localmelo.support.sessions.runner import ResumeLoader, ResumeMode, SessionRunner
from localmelo.support.sessions.store import (
    NewSession,
    SessionBusyError,
    SessionStore,
    SessionStoreError,
    StoredEvent,
    TurnInfo,
    UnknownSessionError,
    title_from_text,
)
from localmelo.support.sessions.sync import SessionMark
from localmelo.support.usage_snapshot import (
    BarPart,
    bar_usage,
    clock_running,
    context_summary,
    usage_lines,
)

logger = logging.getLogger(__name__)


def _line(text: str = "", style: str | None = None) -> Text:
    """A line to print, with what came from a server (model ids, errors) cleaned."""
    return Text(clean_output(text), style=style or "")


STOP_REASON: Final = "Stopped by the user"


def _toggle(arguments: str, current: bool) -> bool | None:
    """``/verbose``'s or ``/thinking``'s new state, or None for a bad argument."""
    value = arguments.strip().lower()
    if not value:
        return not current
    if value == "on":
        return True
    if value == "off":
        return False
    return None


def _progress_lines(payload: Mapping[str, Any], symbols: Any) -> list[str]:
    """A run.progress payload as /progress prints it, one fact per line."""
    dot = f" {symbols.dot} "
    level = payload.get("level") or 0
    step = payload.get("step") or 0
    attempt = payload.get("attempt") or 1
    head = [f"level {level}", f"step {step}"]
    if attempt >= 2:
        head.append(f"attempt {attempt}")
    lines = [dot.join(head)]
    second: list[str] = []
    phase = payload.get("phase")
    if isinstance(phase, str) and phase:
        cycle = payload.get("cycle") or 0
        second.append(f"phase {phase}" + (f" (cycle {cycle})" if cycle else ""))
    tests = payload.get("tests")
    if isinstance(tests, dict):
        passed, total = tests.get("passed"), tests.get("total")
        if isinstance(passed, int) and isinstance(total, int):
            second.append(f"tests {passed}/{total}")
    if second:
        lines.append(dot.join(second))
    past = payload.get("past")
    if isinstance(past, dict):
        runs = past.get("runs") or 0
        lo, hi = past.get("steps_lo"), past.get("steps_hi")
        last = past.get("last_steps")
        facts = [f"past {runs} run{'' if runs == 1 else 's'}"]
        if isinstance(lo, int) and isinstance(hi, int):
            facts.append(f"{lo}-{hi} steps" if lo != hi else f"{lo} steps")
        wall_lo, wall_hi = past.get("wall_lo_s"), past.get("wall_hi_s")
        if isinstance(wall_lo, int | float) and isinstance(wall_hi, int | float):
            facts.append(wall_range(float(wall_lo), float(wall_hi)))
        if isinstance(last, int):
            last_fact = f"last {last} steps"
            last_wall = past.get("last_wall_s")
            if isinstance(last_wall, int | float):
                last_fact += f", {run_elapsed(float(last_wall))}"
            facts.append(last_fact)
        lines.append(dot.join(facts))
    items = payload.get("items")
    if isinstance(items, list) and items:
        lines.append("items")
        for item in items:
            if not isinstance(item, dict):
                continue
            label = " ".join(str(item.get("label") or "").split())
            if not label:
                continue
            state = str(item.get("state") or "pending")
            repeat = item.get("repeat")
            if isinstance(repeat, int) and repeat > 1:
                times = "x" if symbols is ASCII else "×"
                label += f" ({times}{repeat})"
            lines.append(f"  {label}{dot}{state}")
    basis = payload.get("basis")
    if isinstance(basis, dict):
        facts = []
        if basis.get("routine_id"):
            facts.append(str(basis["routine_id"]))
        if basis.get("version") is not None:
            facts.append(f"v{basis['version']}")
        if basis.get("match"):
            facts.append(f"match {basis['match']}")
        if basis.get("model"):
            facts.append(f"model {basis['model']}")
        if facts:
            lines.append("basis " + dot.join(facts))
    return lines


# A turn that ran this long rings the bell when it ends.
BELL_AFTER_S: Final = 30.0
UNDO_HINT: Final = "Undoing the last turn; wait for it to finish."

State = Literal["idle", "running", "stopping", "switching", "shell"]

_STATE_HINTS: Final = {
    "stopping": "Stopping the turn; enter sends once it has stopped.",
    "switching": "Switching the chat or model; esc cancels it.",
    "shell": "A ! command is running.",
}
# Why a "! command" draft waits: the draft is kept.
_SHELL_REFUSED: Final = {
    "running": "! runs only when melo is idle (esc stops the turn).",
    "stopping": "! runs only when melo is idle; the turn is stopping.",
    "switching": "! runs only when melo is idle; wait for the switch to finish.",
    "shell": "A ! command is already running.",
}


class CliSession:
    """A session in the terminal (see the module docstring)."""

    def __init__(
        self,
        *,
        store: SessionStore,
        factory: AgentFactory,
        agent: Agent,
        renderer: TurnRenderer,
        approver: TerminalApprover,
        mode: Mode,
        out: Out,
        resume_mode: ResumeMode = "pairs",
        stream: bool = False,
        reasoning_tags: Mapping[str, str] | None = None,
        status_line: str = "auto",
        clock: Callable[[], float] = time.monotonic,
        developer: bool = False,
        pricing: Mapping[str, Mapping[str, float]] | None = None,
        work: WorkBeside | None = None,
    ) -> None:
        self.developer = developer
        # [pricing.<model>] for a cloud backend's session cost.
        self.pricing = dict(pricing or {})
        # The gateway's tasks beside the chat (none: the bar has no work zone).
        self.work = work
        self.store = store
        self.factory = factory
        self.agent = agent
        self.renderer = renderer
        self.approver = approver
        self.out = out
        # [cli] stream, flipped by /stream for later turns. The per-model
        # framing hints belong to the renderer, which applies the same rule
        # to the stream and to the whole reply.
        self.stream = stream
        # [cli] status_line: "auto" adds the usage segments to the status bar.
        self.status_line = status_line
        self._default_stream = stream
        self.renderer.reasoning_tags = dict(reasoning_tags or {})
        # A tool row's kind comes from the tool's tier.
        self.renderer.tool_tiers = lambda: {
            tool.name: tool.tier for tool in self.agent.list_tools()
        }
        self.approvals = Approvals(self._ask, mode, snapshots=self.approver.snapshots)
        self.runner: SessionRunner | None = None
        self.app: InteractiveApp | None = None
        self.queued: str | None = None
        self._attachments: tuple[Attachment, ...] = ()
        # The last ! command's output, as an attachment to offer; never stored.
        self._shell_output: Attachment | None = None
        self._shell: usershell.ShellRun | None = None
        self.shell_ran = False
        self.activity: str | None = None
        self.title = ""
        self.models: list[str] = []
        self._clock = clock
        self._state: State = "idle"
        self._work: asyncio.Task[Any] | None = None
        self._pending_command: str | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._stop_wanted = False
        self._stop_landed = False
        self._closed = False
        self.navigation = SessionNavigation(self, resume_mode)
        self.session_tools = SessionTools(self)
        self.exit_code = 0
        self.saved_resume_hint = ""

    # ── What the app reads ──

    @property
    def state(self) -> State:
        return self._state

    def attach(self, app: InteractiveApp) -> None:
        self.app = app

    def status_bar(self) -> StatusBar:
        backend = self.factory.backend
        mode = self.approvals.mode
        yolo = mode == "yolo"
        shown: str = mode
        if yolo:
            shown = f"{self.renderer.symbols.warn} YOLO: no approvals"
        elif mode == "ask" and not self.developer:
            shown = ""  # what everyone expects needs no word
        # The model and the folder are in the welcome and /status, not here.
        last = self.renderer.context_tokens
        usage: tuple[BarPart, ...] = ()
        work: tuple[BarPart, ...] = ()
        if self.status_line == "auto":
            usage = tuple(
                bar_usage(
                    self.renderer.usage,
                    local=backend.local,
                    now=time.time(),
                    prices=self.pricing,
                    limits=self._limits(),
                    last_prompt=last,
                    ascii=self.renderer.symbols is ASCII,
                )
            )
            if self.work is not None:
                work = tuple(work_parts(self.work.summary))
        elif last:
            usage = (BarPart(f"ctx {format_tokens(last)}"),)
        return StatusBar(
            title=self.title or "New chat",
            mode=shown,
            mode_style="class:yolo" if yolo else "class:muted",
            usage=usage,
            work=work,
            run=self._run_zone(),
        )

    def _limits(self) -> tuple[int, float | None] | None:
        """The rate limit the chat provider's last response reported, if any."""
        limits = getattr(getattr(self.agent, "_llm", None), "last_limits", None)
        return limits if isinstance(limits, tuple) and len(limits) == 2 else None

    def clock_visible(self) -> bool:
        """Whether the bar shows a clock (idle time, a limit's reset) that moves."""
        return self.status_line == "auto" and clock_running(
            self.renderer.usage,
            local=self.factory.backend.local,
            now=time.time(),
            limits=self._limits(),
        )

    async def open_work(self) -> None:
        """ctrl+t: the tasks beside the chat, and what can be done to one."""
        if self.app is None:
            return
        if self.work is None:
            self.app.hint("No tasks beside this chat: melo task new adds one", 3.0)
            return
        await self.work.open(self.app)

    def _run_zone(self) -> str:
        """While a turn runs: ``2/~8 · ctrl+o steps``, its steps in brief and the key."""
        live = self.renderer.live
        if live is None or self._state not in ("running", "stopping"):
            return ""
        progress = self.renderer.progress
        step = live.step
        if isinstance(progress, Mapping) and isinstance(progress.get("step"), int):
            step = max(step, progress["step"])
        figure = step_figure(step, learned=learned_total(progress))
        dot = f" {self.renderer.symbols.dot} "
        return dot.join(part for part in (figure, f"{STEPS_KEY} steps") if part)

    def on_event(self, stored: StoredEvent) -> None:
        """Render one event of a turn, and redraw the bar its usage may change."""
        # Its requests went to the session's model: their cost uses its price.
        self.renderer.usage.model = self.factory.backend.model
        self.renderer.on_event(stored)
        if self.status_line == "auto" and self.app is not None:
            self.app.invalidate()

    @property
    def attachment_line(self) -> str:
        if not self._attachments:
            return ""
        return "Attached: " + " · ".join(
            item.path.replace("\n", " ").replace("\t", " ")
            for item in self._attachments
        )

    def attach_path(self, path: str) -> str | None:
        """Stage a path explicitly selected from completion, never parsed from prose."""
        if self._state != "idle":
            return "Wait for the current operation before attaching a file."
        self._state = "switching"
        self._pending_command = "attach"
        self._work = self.spawn(
            self._run_command(
                "attach",
                shlex.quote(path),
                "/attach " + shlex.quote(path),
                restore=False,
            )
        )
        return None

    def _hint(self, text: str) -> None:
        if self.app is not None:
            self.app.hint(text)

    def _text(self, text: str = "", style: str | None = None) -> Text:
        """A line of melo's, in the session's symbols, cleaned (:func:`_line`)."""
        return _line(in_symbols(text, self.renderer.symbols), style)

    # ── Enter ──

    def enter(self, text: str) -> str | None:
        """Act on Enter: run a command, send, or queue (see the app's contract)."""
        shell = usershell.parse(text)
        if shell is not None:
            if shell.kind != "literal":
                return self._enter_shell(shell, text)
            text = shell.text  # the escape: the message "! text"
        parsed = commands.parse_command(text)
        if parsed is not None:
            command, name, arguments = parsed
            if command is None:
                self.out.print(
                    self._text(
                        f"Unknown command /{name}. /help lists the commands.",
                        style="error",
                    )
                )
                return ""
            if self._attachments and command.name in ("new", "resume", "sessions"):
                return "Detach staged files before switching chats."
            allowed = command.during_turn is True or (
                command.during_turn == "show" and not arguments
            )
            if self._state != "idle" and not allowed:
                if self._pending_command == "undo":
                    return UNDO_HINT
                return f"/{command.name} is not available while a turn runs (esc stops it)."
            if command.name == "title" and arguments and self._state == "switching":
                return "Wait for the chat switch to finish before changing its title."
            if command.name in (
                "sessions",
                "resume",
                "retry",
                "undo",
                "new",
                "attach",
                "attachments",
                "detach",
            ):
                self._state = "switching"
                self._pending_command = command.name
                self._work = self.spawn(
                    self._run_command(command.name, arguments, text)
                )
            else:
                self.spawn(self._run_command(command.name, arguments, text))
            return None
        if self._state == "idle":
            self.submit(text)
            return None
        if self._state == "running":
            if self.queued is not None:
                return "One message is already queued (↑ edits it)."
            self.queued = text
            self._hint("Sent when this turn completes · ↑ edits it · ctrl+c clears it")
            return None
        return (
            UNDO_HINT if self._pending_command == "undo" else _STATE_HINTS[self._state]
        )

    def _enter_shell(self, shell: usershell.ShellInput, text: str) -> str | None:
        """``! command`` and ``!``: only when idle, else the draft stays."""
        if self._state != "idle":
            return _SHELL_REFUSED[self._state]
        if shell.kind == "help":
            self.renderer.echo(text)
            for line in usershell.HELP:
                self.out.print(self._text(line, "muted"))
            self.out.print(Text(""))
            return None
        if sys.platform == "win32":
            return "! needs a POSIX terminal."
        self._state = "shell"
        self._work = self.spawn(self._run_shell(shell.text))
        return None

    def take_queued(self) -> str | None:
        text, self.queued = self.queued, None
        return text

    def clear_queued(self) -> bool:
        had, self.queued = self.queued is not None, None
        return had

    # ── Tasks ──

    def spawn(self, coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Run *coroutine* as a session task, kept until it ends."""
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            if self._work is task and self._state in ("switching", "shell"):
                self._work, self._state = None, "idle"
                self._pending_command = None
                if self.app is not None:
                    self.app.invalidate()
            return
        exc = task.exception()
        if exc is not None:
            logger.error("A session task failed", exc_info=exc)
            self.out.print(
                self._text(
                    f"⚠ {type(exc).__name__}: {exc} (details in "
                    "~/.cache/localmelo/logs/cli.log)",
                    style="muted",
                )
            )

    # ── Turns ──

    def submit(self, text: str) -> None:
        """Send *text* as the next turn (idle only)."""
        self._state = "running"
        self._stop_wanted = self._stop_landed = False
        if self.runner is None or self.runner.session_id is None:
            self.title = self.session_tools.pending_title or title_from_text(text)
        first_turn = self.runner is None or self.runner.session_id is None
        self.renderer.begin_turn(text, first_turn=first_turn)
        staged, self._attachments = self._attachments, ()
        for item in staged:
            self.out.print(
                self._text(
                    f"  └ attached {item.path} ({len(item.content.encode('utf-8')):,} bytes)",
                    "muted",
                )
            )
        self._work = self.spawn(self._turn(text, attachments=staged))

    async def _ask(self, request: ApprovalRequest) -> ApprovalDecision:
        """Ask through the panel, recorded in the session's events during a turn."""
        if self.runner is None:
            return await self.approver.ask(request)
        return await self.runner.observe(self.approver.ask)(request)

    async def _prepare_turn(self, reload: ResumeLoader) -> None:
        await self.navigation.synchronize(reload)
        await self.session_tools.apply_title()

    async def _turn(
        self,
        text: str,
        *,
        typed_text: str | None = None,
        attachments: tuple[Attachment, ...] = (),
    ) -> None:
        started = self._clock()
        if self.runner is None:
            self.runner = SessionRunner(
                self.agent,
                self.store,
                None,
                new_session=NewSession(
                    model=self.factory.backend.model,
                    workspace=self.agent.workspace_root,
                ),
                on_event=self.on_event,
                progress=True,
                resume_mode=self.navigation.mode,
                before_turn=self._prepare_turn,
            )
        runner = self.runner
        result: RunResult | None = None
        error: BaseException | None = None
        try:
            kwargs: dict[str, Any] = (
                {
                    "attachments": tuple(
                        TextAttachment(item.path, item.content) for item in attachments
                    )
                }
                if attachments
                else {}
            )
            token = (
                STREAM_DISPLAY.set(self.renderer.stream_feed()) if self.stream else None
            )
            try:
                result = await runner.run_turn(
                    text,
                    approval=self.approvals.policy(),
                    title=text if typed_text is None else typed_text,
                    **kwargs,
                )
            finally:
                if token is not None:
                    STREAM_DISPLAY.reset(token)
        except UnknownSessionError:
            error = UnknownSessionError(DELETED)
        except SessionBusyError:
            error = SessionBusyError(BUSY)
        except Exception as exc:
            # The ✗ line says it on the screen; an ERROR would add a ⚠ line.
            logger.warning("The turn could not run", exc_info=True)
            error = exc
        # A second stop can make run_turn return while the runner still
        # finishes the turn: the next message must not meet it.
        while runner.busy:
            await asyncio.sleep(0.05)
        with contextlib.suppress(SessionStoreError):
            await self.session_tools.apply_title()
        cause = "user_stop" if self._stop_wanted else ""
        self.renderer.end_turn(result, error, cause=cause)
        with contextlib.suppress(SessionStoreError):
            await self.navigation.refresh_refs()
        if error is not None and attachments:
            self._attachments = attachments + self._attachments
        self._state = "idle"
        self._work = None
        if self.app is not None:
            if self._clock() - started >= BELL_AFTER_S:
                self.app.bell()
            self.app.invalidate()
        if self._closed:
            return  # the session shuts down: a queued message is not sent
        queued = self.take_queued()
        completed = result is not None and result.status == "completed"
        if queued is not None and completed and error is None:
            self.submit(queued)
            return
        # A turn that could not run gives its message back, then the queued one.
        back = [part for part in (text if error else None, queued) if part]
        if not back or self.app is None:
            return
        self._give_back("\n".join(back))
        if error and queued:
            line = "Your message and the queued one are back in the input box."
        elif error:
            line = "Your message is back in the input box."
        else:
            line = "Your queued message is back in the input box."
        self.out.print(self._text(line, style="muted"))

    def _give_back(self, text: str) -> None:
        """Put *text* in front of the draft; the cursor stays where it was.

        In an empty draft the cursor goes after *text*, so what is typed
        next follows it.
        """
        assert self.app is not None
        buffer = self.app.buffer
        draft = buffer.document
        if draft.text:
            buffer.document = Document(
                text + "\n" + draft.text, len(text) + 1 + draft.cursor_position
            )
        else:
            buffer.document = Document(text, len(text))

    def stop(self, reason: str = STOP_REASON) -> bool:
        """Stop the turn or the switch, once; Esc and Ctrl+C."""
        if self._state == "switching":
            if self._work is None:
                return False  # the new Agent is in place; the switch ends
            self._work.cancel()
            return True
        if self._state != "running":
            return False
        live = self.renderer.live
        ended = self.runner is not None and self.runner.run_ended
        if ended or (live is not None and live.finished):
            # The run has ended; the turn is wrapping up.
            self._hint("Finishing the turn…")
            return False
        self._state = "stopping"
        self._stop_wanted = True
        self.renderer.stopped_by("user_stop")
        self._try_cancel(reason, "user_stop")
        return True

    def _try_cancel(self, reason: str, cause: Literal["user_stop", "shutdown"]) -> None:
        if self._stop_landed or self.runner is None:
            return
        if self.runner.cancel(reason, cause=cause):
            self._stop_landed = True

    def tick(self) -> None:
        """The app's tick: try a stop again until the runner takes it."""
        if self._state == "stopping" and not self._stop_landed:
            self._try_cancel(STOP_REASON, "user_stop")

    # ── Commands ──

    async def _run_command(
        self, name: str, arguments: str, text: str, *, restore: bool = True
    ) -> None:
        """Echo the command as a message is echoed, run it, then a blank line.

        ``/exit`` prints nothing.
        """
        if name == "exit":
            await self._command(name, arguments)
            return
        self.renderer.echo(text)
        try:
            await self._command(name, arguments)
        except (ValueError, SessionStoreError, SessionBusyError) as exc:
            message = (
                DELETED
                if isinstance(exc, UnknownSessionError)
                else BUSY if isinstance(exc, SessionBusyError) else str(exc)
            )
            self.out.print(self._text(f"✗ {message}", style="error"))
            if self.app is not None and restore:
                self._give_back(text)
        finally:
            if self._work is asyncio.current_task():
                self._work, self._state = None, "idle"
            if self._pending_command == name:
                self._pending_command = None
            self.out.print(Text(""))

    # ── ! commands ──

    @contextlib.asynccontextmanager
    async def _terminal_handed_over(self) -> AsyncIterator[None]:
        """Pause the prompt and give the terminal to the child.

        The renderer and the input pause (prompt_toolkit's ``in_terminal``)
        and come back once, whatever ends the command; what is printed
        meanwhile waits. The prompt's resize handler would draw into the
        child's screen and ask the terminal for the cursor position, whose
        answer the child would read, so it is off meanwhile: the relay passes
        the new size on, and the prompt reads it again when it comes back.
        Without an app there is no terminal.
        """
        ui = getattr(self.app, "app", None)
        if ui is None:
            yield
            return
        held = getattr(self.out, "held", contextlib.nullcontext)
        with set_app(ui), held(), attach_winch_signal_handler(lambda: None):
            async with in_terminal():
                yield

    async def _run_shell(self, command: str) -> None:
        cwd = self.agent.workspace_root or os.getcwd()
        self.shell_ran = True
        header = Text()
        header.append("! user shell", style="bold")
        header.append(f" · {home_label(cwd)}", style="muted")
        self.out.print(self._text(), header)
        for number, line in enumerate(command.splitlines() or [""]):
            self.out.print(self._text(("$ " if number == 0 else "  ") + line))
        await asyncio.sleep(0)  # the header is printed before the prompt is erased
        ui = getattr(self.app, "app", None)
        run = usershell.ShellRun(
            command,
            cwd,
            stdin_fd=None if ui is None else ui.input.fileno(),
            stdout_fd=None if ui is None else ui.output.fileno(),
            clock=self._clock,
        )
        self._shell = run
        try:
            if self._closed:  # melo is exiting: nothing starts now
                return
            async with self._terminal_handed_over():
                result = await run.run()
            self._shell_output = usershell.snapshot(result)
            line = f"  └ {result.status()} · {format_duration(result.elapsed_s)}"
            self.out.print(
                self._text(line, "muted" if result.returncode == 0 else "error")
            )
            if self._shell_output is not None:
                self.out.print(
                    self._text(
                        "  └ /attach ! shows this output (redacted, "
                        f"{self._shell_output.size_bytes:,} bytes) and asks before attaching it",
                        "muted",
                    )
                )
            self.out.print(Text(""))
        finally:
            self._shell = None
            if self._work is asyncio.current_task():
                self._work, self._state = None, "idle"
            if self.app is not None:
                self.app.invalidate()

    async def _command(self, name: str, arguments: str) -> None:
        if name == "help":
            self.out.print(
                commands.help_text(
                    arguments,
                    self.renderer.symbols,
                    everything=self.developer,
                    width=self.out.width,
                )
            )
        elif name == "attach":
            await self._attach_file(arguments)
        elif name == "attachments":
            self._list_attachments(arguments)
        elif name == "detach":
            self._detach(arguments)
        elif name in ("exit", "quit"):
            if self.app is not None:
                self.app.exit()
        elif name in ("new", "clear"):
            await self.new()
        elif name == "model":
            await self.model(arguments)
        elif name == "backend":
            await self.backend(arguments)
        elif name in ("title", "undo", "usage", "memory", "copy"):
            await self.session_tools.run(name, arguments)
        elif name == "verbose":
            self.toggle_verbose(arguments)
        elif name == "thinking":
            self.toggle_thinking(arguments)
        elif name == "stream":
            self.toggle_stream(arguments)
        elif name == "expand":
            await self.expand(arguments)
        elif name == "sessions":
            await self.navigation.select(query=arguments)
        elif name == "resume":
            await self.navigation.select(ref=arguments)
        elif name == "progress":
            self.progress()
        elif name == "approvals":
            self.approvals_command(arguments)
        elif name == "status":
            await self.status(arguments)
        elif name == "retry":
            await self.retry()
        elif name == "mcp":
            self.mcp()
        elif name == "browser":
            await browsing.command(self, arguments)

    async def _attach_file(self, arguments: str) -> None:
        try:
            paths = shlex.split(arguments)
        except ValueError as exc:
            raise ValueError(
                "Use /attach PATH (quote paths containing spaces)."
            ) from exc
        if len(paths) != 1:
            raise ValueError("Use /attach PATH (quote paths containing spaces).")
        workspace = self.agent.workspace_root or os.getcwd()
        if paths[0] == "!":  # the last ! output (a file of that name is ./!)
            if self._shell_output is None:
                raise ValueError("No ! output to attach: run ! COMMAND first.")
            item = self._shell_output
        else:
            item = await asyncio.to_thread(load_attachment, workspace, paths[0])
        updated = add_attachment(self._attachments, item)
        from_shell = paths[0] == "!"
        if item.changed or from_shell:
            self.out.print(
                self._text(
                    (
                        "Attachment was redacted or cleaned; review the exact snapshot:"
                        if item.changed
                        else "Review the exact output that would be sent:"
                    ),
                    "warn",
                )
            )
            self.out.print(Text(item.content))
            yes = (
                self.app is not None
                and await self.app.pick(
                    (
                        "Attach this redacted snapshot?"
                        if item.changed
                        else "Attach this output?"
                    ),
                    ["Attach it", "Don't attach it"],
                    default=1,
                    escape=1,
                )
                == 0
            )
            if not yes:
                raise ValueError("Attachment not added.")
        self._attachments = updated
        self.out.print(
            self._text(
                f"✓ Attached {item.path} ({len(item.content.encode('utf-8')):,} bytes)."
            )
        )
        if self.app is not None:
            self.app.invalidate()

    def _list_attachments(self, arguments: str) -> None:
        if arguments:
            raise ValueError("Use /attachments without arguments.")
        if not self._attachments:
            self.out.print(self._text("No staged attachments.", "muted"))
        for index, item in enumerate(self._attachments, 1):
            self.out.print(
                self._text(
                    f"{index}. {item.path} ({len(item.content.encode('utf-8')):,} bytes)",
                    "muted",
                )
            )

    def _detach(self, arguments: str) -> None:
        if arguments.strip() == "all":
            self._attachments = ()
        elif arguments.strip().isdigit() and 1 <= int(arguments.strip()) <= len(
            self._attachments
        ):
            i = int(arguments.strip()) - 1
            self._attachments = self._attachments[:i] + self._attachments[i + 1 :]
        else:
            raise ValueError("Use /detach N or /detach all.")
        self.out.print(self._text("Attachment removed.", "muted"))
        if self.app is not None:
            self.app.invalidate()

    def toggle_verbose(self, arguments: str) -> None:
        """``/verbose [on|off]``: verbose rendering, or the other way."""
        state = _toggle(arguments, self.renderer.verbose)
        if state is None:
            self.out.print(
                self._text("✗ Use /verbose on or /verbose off.", style="error")
            )
            return
        self.renderer.verbose = state
        self.out.print(
            self._text(f"Verbose rendering {'on' if state else 'off'}.", "muted")
        )

    def toggle_thinking(self, arguments: str) -> None:
        """``/thinking [on|off]``: thinking excerpts, or the other way."""
        state = _toggle(arguments, self.renderer.thinking)
        if state is None:
            self.out.print(
                self._text("✗ Use /thinking on or /thinking off.", style="error")
            )
            return
        self.renderer.thinking = state
        self.out.print(
            self._text(f"Thinking excerpts {'on' if state else 'off'}.", "muted")
        )

    def toggle_stream(self, arguments: str) -> None:
        """``/stream [on|off]``: live streamed replies, for the next turn."""
        state = _toggle(arguments, self.stream)
        if state is None:
            self.out.print(
                self._text("✗ Use /stream on or /stream off.", style="error")
            )
            return
        self.stream = state
        if not state:
            words, style = "Streaming off. Replies appear when complete.", "muted"
        elif self.renderer.stream_hint():
            words, style = (
                "Streaming on. Replies appear as the model writes them.",
                "muted",
            )
        else:  # the advice is default text; a confirmation is dim
            words, style = f"Streaming on. {STREAM_BUFFERED}", None
        self.out.print(self._text(words, style))

    async def expand(self, arguments: str) -> None:
        """``/expand [turn.call]``: a tool call's whole output or error."""
        calls = await self.navigation.tool_outputs(arguments)
        if not calls:
            self.out.print(self._text("✗ No tool calls yet.", style="error"))
            return
        latest_turn = calls[-1].ref.split(".", 1)[0]
        call: ToolOutput | None
        argument = arguments.strip()
        if not argument:
            call = calls[-1]
        elif argument.startswith("!"):
            self.out.print(self._text("✗ No tool calls yet.", style="error"))
            return
        elif "." in argument:
            turn, _, number = argument.partition(".")
            if not (turn.isdigit() and number.isdigit()):
                self.out.print(self._text("✗ Use /expand [turn.call].", style="error"))
                return
            call = next(
                (c for c in calls if c.ref == f"{int(turn)}.{int(number)}"), None
            )
            if call is None:
                self._expand_missing(int(turn), calls)
                return
        elif argument.isdigit():
            call = next(
                (c for c in calls if c.ref == f"{latest_turn}.{int(argument)}"), None
            )
            if call is None:
                self._expand_missing(int(latest_turn), calls)
                return
        else:
            self.out.print(self._text("✗ Use /expand [turn.call].", style="error"))
            return
        self._print_expanded(call.text)

    def _expand_missing(self, turn: int, calls: list[ToolOutput]) -> None:
        count = sum(1 for call in calls if call.ref.startswith(f"{turn}."))
        if count:
            self.out.print(
                self._text(
                    f"✗ Turn {turn} has {count} tool call{'' if count == 1 else 's'}.",
                    style="error",
                )
            )
        else:
            self.out.print(self._text("✗ No tool calls yet.", style="error"))

    def _print_expanded(self, text: str) -> None:
        lines = [
            expand_tabs(line) for line in clean_output(text).rstrip("\n").split("\n")
        ]
        if not "".join(lines).strip():
            lines = ["(no output)"]
        rows: list[str] = []
        for line in lines:
            rows.extend(wrap_cells(line, max(1, self.out.width)) or [""])
        for row in rows[:EXPAND_LINES]:
            self.out.print(Text(row))
        if len(rows) > EXPAND_LINES:
            ellipsis = self.renderer.symbols.ellipsis
            self.out.print(
                Text(
                    f"{ellipsis} +{len(rows) - EXPAND_LINES} lines not shown "
                    f"({len(text)} chars in all)",
                    style="muted",
                )
            )

    def _row(self, label: str, value: str | Text, style: str | None = None) -> None:
        """One ``/status`` row; a value too long for it continues under itself."""
        text = value if isinstance(value, Text) else self._text(value, style)
        self.out.print(hang(label, text, self.out.width))

    async def status(self, arguments: str = "") -> None:
        """``/status``: whether the model answers, the folder, session, approvals and context.

        ``/status all``, and ``/status`` in developer mode, print every row:
        the workspace's fence, tools, memory, limits, usage, files and version.
        """
        from localmelo.support.cli.render import LOG_PATH
        from localmelo.support.sessions.agents import SESSION_TOOL_TIMEOUT_S
        from localmelo.support.sessions.display import display_version

        word = arguments.strip().lower()
        if word not in ("", "all"):
            self.out.print(self._text("✗ Use /status or /status all.", style="error"))
            return
        backend = self.factory.backend
        checked = self.factory.listable()
        failure: discovery.DiscoveryError | None = None
        if checked:
            try:
                await discovery.list_models(
                    discovery.OPENAI_COMPATIBLE, backend.base_url, timeout_s=2
                )
            except discovery.DiscoveryError as exc:
                failure = exc
            self.factory.relabel()  # the listing may have said the server is oMLX
            backend = self.factory.backend
        turns = []
        if self.runner is not None and self.runner.session_id is not None:
            turns = await self.store.list_turns(self.runner.session_id)
        where = home_label(self.agent.workspace_root or "")
        url = redact_secrets(backend.base_url or backend.label)[0]
        if not (self.developer or word == "all"):
            self._short_status(backend, where, len(turns), checked, failure)
            return
        state = "not checked"
        if checked:
            state = "reachable"
            if failure is not None:
                state = (
                    str(failure) if failure.reachable else f"not reachable: {failure}"
                )
        source = "" if backend.source == "config" else f" · {backend.source}"
        self._row(
            "model", f"{backend.model} · {backend.label} · {url} · {state}{source}"
        )
        self._row("workspace", where)
        self._row(
            "fence",
            "file tools stay here; commands and Python can reach other files",
            "muted",
        )
        title = self.title or "New chat"
        session_id = self.runner.session_id if self.runner is not None else ""
        short = (session_id or "")[:12]
        count = len(turns)
        self._row(
            "session",
            f"{title!r} · {short or 'new'} · {count} turn"
            + ("" if count == 1 else "s"),
        )
        grants = ", ".join(sorted(self.approvals.allowed_tools)) or "none"
        self._row(
            "approvals", f"{self.approvals.mode} · allowed in this session: {grants}"
        )
        tools = self.agent.list_tools()
        builtin = [tool for tool in tools if tool.source == "builtin"]
        provided = [tool for tool in tools if tool.source == "provided"]
        asking = sum(1 for tool in builtin if tool.requires_approval)
        self._row(
            "tools",
            f"{len(builtin)} built-in ({asking} ask first)"
            + (f" · {len(provided)} MCP" if provided else ""),
        )
        messages = len(self.agent.conversation())
        window = self.agent.options.window_messages
        if window == config.default_window_messages(backend.key):
            window_text = f"{window} ({config.window_reason(backend.key)})"
        else:
            window_text = str(window)
        embedding = backend.embedding
        self._row(
            "memory",
            f"{messages} of {window_text} messages · "
            + (
                f"long-term memory · {embedding}"
                if embedding
                else "no long-term memory (no embedding backend)"
            ),
        )
        options = self.agent.options
        limit = format_duration(SESSION_TOOL_TIMEOUT_S)
        self._row(
            "limits",
            f"{options.max_steps} steps a turn, "
            f"{options.steps_per_attempt} an attempt, "
            f"{options.max_attempts} attempts · commands stop after {limit}",
        )
        for row in usage_lines(
            self.renderer.usage,
            local=backend.local,
            now=time.time(),
            steps=lambda payload: _progress_lines(payload, self.renderer.symbols),
            hide_empty=not self.developer,
        ):
            self._row(row[:11], row[11:])
        files = [getattr(self.store, "path", ""), LOG_PATH]
        if os.environ.get("LOCALMELO_PERSIST_MEMORY") == "1" or getattr(
            self.agent.options, "memory_dir", None
        ):
            memory_dir = os.environ.get("LOCALMELO_MEMORY_DIR") or os.path.join(
                os.path.expanduser("~"), ".cache", "localmelo", "memory"
            )
            files += [
                os.path.join(memory_dir, "procedures.db"),
                os.path.join(memory_dir, "history.db"),
            ]
        files = [home_label(str(path)) for path in files if path]
        self._row("files", " · ".join(files))
        self._row("version", f"melo {display_version()}")

    def _short_status(
        self,
        backend: BackendInfo,
        where: str,
        turns: int,
        checked: bool,
        failure: discovery.DiscoveryError | None,
    ) -> None:
        """The five rows of ``/status``; only the mark of the model's row has a color."""
        url = redact_secrets(backend.base_url)[0] if backend.base_url else ""
        model = [backend.model, backend.label, url]
        if backend.source not in ("config", "saved"):
            model.append(SOURCE_TEXT.get(backend.source, backend.source))
        parts: list[tuple[str, str | None]] = [
            (" · ".join(p for p in model if p), None)
        ]
        if checked and failure is None:
            parts += [(" · ", None), ("✓", "ok"), (" answers", None)]
        elif failure is not None:
            said = (
                f" replied with an error ({failure})."
                if failure.reachable
                else f" does not answer ({failure}). Start it, then /status checks again."
            )
            parts += [(" · ", None), ("✗", "error"), (said, None)]
        row = Text()
        for piece, style in parts:
            row.append_text(self._text(piece, style))
        self._row("model", row)
        self._row("folder", where)
        title = " ".join(self.title.split())
        count = f"{turns} turn" + ("" if turns == 1 else "s") if turns else "new"
        self._row("chat", f'"{title}" · {count}' if title else count)
        mode, grants = self.approvals.mode, self.approvals.allowed_tools
        self._row("approvals", approval_words(mode, grants))
        usage, last = self.renderer.usage, self.renderer.context_tokens
        self._row("context", context_summary(usage, last))

    async def retry(self) -> None:
        """``/retry``: re-run the newest turn, undoing it only when still live."""
        if self.runner is None:
            self.out.print(self._text("✗ Nothing to retry.", style="error"))
            return
        self._state = "switching"
        did_undo = False
        turn: TurnInfo | None
        try:
            await self.navigation.synchronize()
            sid = self.runner.session_id
            newest = (
                await self.store.list_turns(sid, limit=1) if sid is not None else []
            )
            if newest and newest[0].undone:
                turn = newest[0]
            else:
                did_undo = True
                undo = asyncio.ensure_future(self.runner.undo())
                try:
                    turn = await asyncio.shield(undo)
                except asyncio.CancelledError:
                    turn = await undo
                    if turn is not None:
                        mark = self.navigation.mark
                        self.navigation.mark = SessionMark(
                            mark.foreign_newest, mark.undone + 1
                        )
                        if self.app is not None:
                            self._give_back(turn.user_text)
                    raise
        except SessionBusyError:
            self.out.print(self._text("✗ Nothing to retry.", style="error"))
            return
        if turn is None:
            self.out.print(self._text("✗ Nothing to retry.", style="error"))
            return
        if did_undo:
            mark = self.navigation.mark
            self.navigation.mark = SessionMark(mark.foreign_newest, mark.undone + 1)
        first = " ".join(turn.typed_text.split())[:60]
        self.out.print(self._text(f"↻ Retrying turn {turn.turn}: {first}"))
        self._state = "running"
        self._stop_wanted = self._stop_landed = False
        staged = tuple(
            Attachment(a.name, a.text, len(a.text.encode("utf-8")), len(a.text), False)
            for a in (turn.attachments or ())
        )
        message = turn.typed_text if staged else turn.user_text
        self.renderer.begin_turn(message, first_turn=False)
        self._work = self.spawn(
            self._turn(message, typed_text=turn.typed_text, attachments=staged)
        )

    def mcp(self) -> None:
        """``/mcp``: the MCP servers, their tools, and where the file came from."""
        factory = self.factory
        if not factory.mcp_config:
            self.out.print(
                self._text(
                    "No MCP servers. Name a config file with mcp_config in "
                    "config.toml, or start melo with --mcp FILE "
                    "(the mcpServers JSON format)."
                )
            )
            return
        from localmelo.melo.checker.redaction import redact_secrets

        path = home_label(os.path.expanduser(factory.mcp_config))
        self.out.print(
            self._text(
                f"MCP servers from {redact_secrets(path)[0]} ({factory.mcp_source})"
            )
        )
        tools = self.agent.list_tools() if factory.mcp_error is None else []
        for server in factory.mcp_servers:
            prefix = f"mcp__{server.namespace}__"
            count = sum(1 for tool in tools if tool.name.startswith(prefix))
            command = redact_secrets(" ".join([server.command, *server.args]))[0]
            if factory.mcp_error is not None:
                line = f"  {server.name}: not running · {command}"
            else:
                line = (
                    f"  {server.name}: running, {count} tool"
                    + ("" if count == 1 else "s")
                    + f" · {command}"
                )
            self.out.print(self._text(line, "muted"))
        if factory.mcp_error:
            problem = redact_secrets(factory.mcp_error)[0]
            self.out.print(
                self._text(
                    f"MCP is off for this chat: {problem}. "
                    "Fix the config, then restart melo.",
                    "warn",
                )
            )

    def approvals_command(self, arguments: str) -> None:
        """``/approvals``: the mode and grants, or switch, or forget one."""
        approvals = self.approvals
        mode = approvals.mode
        if mode == "yolo":
            self.out.print(self._text("Approvals: YOLO (--yolo); no grants.", "muted"))
            if arguments:
                self.out.print(
                    self._text(
                        "✗ YOLO is on; /approvals ask turns prompts on.",
                        style="error",
                    )
                )
            return
        parts = arguments.split()
        if parts:
            what = parts[0].lower()
            if what in ("ask", "auto", "deny"):
                approvals.set_mode(cast(Literal["ask", "auto", "deny"], what))
            elif what == "forget":
                approvals.forget(parts[1] if len(parts) > 1 else None)
            else:
                self.out.print(
                    self._text(
                        "✗ Use /approvals ask, auto, deny, or forget [tool].",
                        style="error",
                    )
                )
                return
        grants = ", ".join(sorted(approvals.allowed_tools)) or "none"
        self.out.print(self._text(f"Approvals: {approvals.mode} · grants: {grants}"))

    def progress(self) -> None:
        """``/progress``: the latest run.progress of this session, in full."""
        payload = self.renderer.progress
        if not isinstance(payload, dict):
            self.out.print(self._text("No progress facts for this turn yet.", "muted"))
            return
        for line in _progress_lines(payload, self.renderer.symbols):
            self.out.print(self._text(line))

    async def new(self) -> None:
        """``/new``: start over, grants and display preferences included."""
        if self.runner is not None:
            await self.runner.close()
        self.runner = None
        self.session_tools.reset()
        self.navigation.mark = SessionMark(0, 0)
        self.agent.load_conversation([])
        self.approvals.reset()
        browsing.forget(self)
        self.title = ""
        self.renderer.context_tokens = 0
        self.renderer.new_session()
        self.stream = self._default_stream
        label = " New chat "
        side = max(2, (self.out.width - len(label)) // 2)
        rule = self.renderer.symbols.rule
        self.out.print(self._text(rule * side + label + rule * side, style="muted"))

    async def model(self, arguments: str) -> None:
        """``/model``: pick one of the server's models, or switch to one by name and save it."""
        backend = self.factory.backend
        if not arguments:
            await self._pick_current_server_model()
            return
        name = arguments
        if not self.models and self.factory.listable():
            # Startup may not have listed them: /model picks from the list.
            try:
                self.models = await self.factory.list_models()
            except discovery.DiscoveryError as exc:
                if name.isdigit():
                    self.out.print(
                        self._text(f"✗ Could not list the models: {exc}", style="error")
                    )
                    return
                self.out.print(
                    self._text(
                        f"⚠ Could not list the models ({exc}); switching anyway.",
                        style="warn",
                    )
                )
        if self.models and name not in self.models:
            # A listed server: only its models, by id or by number.
            if not name.isdigit():
                self.out.print(
                    self._text(
                        f"✗ {backend.base_url} lists no model named '{name}'. "
                        "/model lists them.",
                        style="error",
                    )
                )
                return
            if not 1 <= int(name) <= len(self.models):
                self.out.print(
                    self._text(
                        f"✗ Type a number from 1 to {len(self.models)}, or a model id.",
                        style="error",
                    )
                )
                return
            name = self.models[int(name) - 1]
        await self._switch(name, url=None)

    async def _pick_current_server_model(self) -> None:
        """``/model`` alone: a list of the server's chat models, the current one marked."""
        backend = self.factory.backend
        if self.app is None:
            return
        if not self.factory.listable():  # a cloud API has no list to pick from
            self.out.print(self._text(f"Model: {backend.model} · {backend.label}"))
            self.out.print(self._text("/model NAME switches and saves.", "muted"))
            return
        try:
            self.models = await self.factory.list_models()
        except discovery.DiscoveryError as exc:
            self.out.print(
                self._text(f"✗ Could not list the models: {exc}", style="error")
            )
            return
        backend = self.factory.backend  # the listing may have said the server is oMLX
        if not self.models:
            self.out.print(
                self._text(f"✗ {backend.base_url} lists no models.", style="error")
            )
            return
        kinds = {model: discovery.model_kind(model) for model in self.models}
        chat = [model for model in self.models if kinds[model] == "chat"]
        loaded = (
            await discovery.loaded_models(backend.base_url)
            if len(chat) > 1
            else discovery.LoadedModels()
        )
        rows = [
            ModelRow(model, model in loaded.ids, kinds[model])
            for model in (chat or self.models)
        ]
        where = redact_secrets(f"{backend.label} · {backend.base_url}")[0]
        picked = await self.app.choose_model(rows, where=where, current=backend.model)
        if picked is None:
            self.out.print(
                self._text(f"Cancelled. Still using {backend.model}.", "muted")
            )
        elif picked == backend.model:
            self.out.print(self._text(f"Still using {picked}.", "muted"))
        else:
            await self._switch(picked, url=None)

    async def backend(self, arguments: str) -> None:
        """``/backend``: ask for a server and a model, check them, switch and save."""
        if self.app is None:
            return
        old = self.factory.backend
        url = arguments or await self.app.ask(
            "Model server URL", old.base_url or "http://127.0.0.1:8000/v1"
        )
        if not url:
            return
        self._state, self.activity = "switching", f"Checking {url}"
        self._work = asyncio.current_task()
        try:
            models = await self.factory.list_models(url)
        except discovery.DiscoveryError as exc:
            self.out.print(
                self._text(
                    f"✗ Could not reach {url}: {exc}. Still using {old.base_url or old.label}.",
                    style="error",
                )
            )
            return
        except asyncio.CancelledError:
            self.out.print(
                self._text(
                    f"Cancelled. Still using {old.base_url or old.label}.",
                    style="muted",
                )
            )
            return
        finally:
            self._state, self.activity, self._work = "idle", None, None
        if not models:
            self.out.print(
                self._text(
                    f"✗ {url} lists no models. Still using {old.base_url or old.label}.",
                    style="error",
                )
            )
            return
        model = await self._pick_model(url, models)
        if model is None:
            return
        # It answered with /v1 added, so that is the URL to chat with too.
        url = normalize_url(url)
        if await self._switch(model, url=url):
            self.models = models

    async def _pick_model(self, url: str, models: list[str]) -> str | None:
        """The model to use on *url*, from the list the first run shows; None on Esc."""
        kinds = {model: discovery.model_kind(model) for model in models}
        chat = [model for model in models if kinds[model] == "chat"]
        if len(models) == 1 and chat:
            return models[0]
        loaded = (
            await discovery.loaded_models(url)
            if len(chat) > 1
            else discovery.LoadedModels()
        )
        shown = chat or models
        pick = discovery.pick_model(shown, loaded)
        if len(chat) > 1 and pick is None:  # nothing is highlighted: say why
            self.out.print(self._text(discovery.LOAD_COST_NOTE, style="muted"))
        assert self.app is not None
        return await self.app.choose_model(
            [ModelRow(model, model in loaded.ids, kinds[model]) for model in shown],
            where=redact_secrets(url)[0],
            current="",
            start=pick,
            title="Choose a model",
        )

    async def _switch(
        self, model: str, *, url: str | None, note: str | None = None
    ) -> bool:
        """Move the conversation to a new Agent on *model*, then save it.

        With *note*, the switch is not a model change: *note* is printed in
        place of the saved-model line (``/browser`` rebuilds the Agent).
        False when it failed. The session stays "switching" until the
        switch is complete: a message sent meanwhile would start on the old
        model. Esc cancels only the start of the new Agent.
        """
        old = self.agent
        self._state, self.activity = "switching", f"Starting {model}"
        self._work = asyncio.current_task()
        try:
            try:
                new = await self.factory.build(model, url=url)
            except asyncio.CancelledError:
                self.out.print(
                    self._text(
                        f"Switch cancelled. Still using {self.factory.backend.model}.",
                        style="muted",
                    )
                )
                return False
            except Exception as exc:
                self.out.print(
                    self._text(
                        f"✗ Could not start {model}: {exc}. Still using {self.factory.backend.model}.",
                        style="error",
                    )
                )
                return False
            self._work = None  # past this point the switch completes
            new.load_conversation(old.conversation())
            new.load_notes(old.notes())
            if self.runner is not None:
                self.runner.replace_agent(new)
            self.agent = new
            if url is None:
                self.factory.use_model(model)
                done = f"Model: {model} ({self.factory.backend.label})."
            else:
                await self.factory.use_url(url, model)
                done = f"Backend: {url} · {model}."
            if self.runner is not None and self.runner.session_id is not None:
                await self.store.update_session(self.runner.session_id, model=model)
            await self.factory.close(old)
            if note is None:
                await self._report_switch(done)
            else:
                self.out.print(self._text(note, "ok"))
            return True
        finally:
            self._state, self.activity, self._work = "idle", None, None

    async def _report_switch(self, done: str) -> None:
        problem = await self.factory.save()
        if problem:
            self.out.print(
                self._text(f"{done} {problem[0].upper()}{problem[1:]}.", style="warn")
            )
        else:
            self.out.print(self._text(f"{done} Saved as the default.", style="ok"))

    # ── The model server at startup ──

    async def check_server(self) -> None:
        """Warn when the server does not answer, or does not list the model."""
        backend = self.factory.backend
        try:
            models = await self.factory.list_models()
        except discovery.DiscoveryError as exc:
            if self.factory.backend.base_url != backend.base_url:
                return  # /backend moved to another server meanwhile
            if not exc.reachable:
                text = f"Cannot reach the model server at {backend.base_url} ({exc}). Start it, then send your message."
            else:
                text = f"{backend.base_url} could not list its models ({exc}); the first message may fail."
            self.out.print(self._text(f"⚠ {text}", style="warn"))
            return
        if self.factory.backend.base_url != backend.base_url:
            return  # /backend moved to another server meanwhile
        self.models = models
        if models and backend.model not in models:
            self.out.print(
                self._text(
                    f"⚠ {backend.base_url} lists no model named '{backend.model}'; the first message may fail.",
                    style="warn",
                )
            )
        if (
            models
            and not backend.embedding
            and self.factory.memory
            and backend.key != discovery.OPENAI_COMPATIBLE
        ):
            candidate = discovery.recommend_embedding(
                discovery.embedding_models(models)
            )
            if candidate:
                self.out.print(
                    self._text(
                        f"This server lists {candidate}; melo setup adds it for long-term memory.",
                        style="muted",
                    )
                )

    # ── Shutdown ──

    async def shutdown(
        self,
        reason: str = STOP_REASON,
        *,
        cause: Literal["user_stop", "shutdown"] = "user_stop",
    ) -> None:
        """Stop what runs with *reason* and *cause*, then close everything once."""
        if self._closed:
            return
        self._closed = True
        if self._state in ("running", "stopping"):
            self._stop_wanted = True
            self.renderer.stopped_by(cause)
            self._try_cancel(reason, cause)
        if self._shell is not None:
            self._shell.stop()  # its group ends with melo
        work = self._work
        if work is not None and self._state == "switching":
            work.cancel()
        if work is not None:
            deadline = FINALIZE_TIMEOUT_S + 2
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                async with asyncio.timeout(deadline):
                    while not work.done():
                        self._try_cancel(reason, cause)
                        await asyncio.wait({work}, timeout=0.1)
        for task in list(self._tasks):
            if task is not work and not task.done():
                task.cancel()
        if self.runner is not None:
            await self.runner.close(reason=reason, cause=cause)
        with contextlib.suppress(SessionStoreError):
            self.saved_resume_hint = await self.navigation.saved_hint()
        for agent in self.factory.live_agents():
            with contextlib.suppress(Exception):
                await self.factory.close(agent)
        await self.store.aclose()
