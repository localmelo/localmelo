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
import time
from collections.abc import Callable, Coroutine
from typing import Any, Final, Literal

from rich.text import Text

from localmelo.melo.agent import Agent
from localmelo.melo.agent.agent import FINALIZE_TIMEOUT_S
from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.melo.schema import RunResult
from localmelo.support.backends import discovery
from localmelo.support.backends.openai_compat import normalize_url
from localmelo.support.cli import commands
from localmelo.support.cli.agents import AgentFactory
from localmelo.support.cli.app import InteractiveApp, StatusBar
from localmelo.support.cli.approvals import Approvals, Mode, TerminalApprover
from localmelo.support.cli.output import Out
from localmelo.support.cli.render import TurnRenderer
from localmelo.support.sessions.display import (
    clean_output,
    format_tokens,
    home_label,
)
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import (
    NewSession,
    SessionBusyError,
    SessionStore,
    title_from_text,
)

logger = logging.getLogger(__name__)


def _text(text: str = "", style: str | None = None) -> Text:
    """A line to print, with what came from a server (model ids, errors) cleaned."""
    return Text(clean_output(text), style=style or "")


STOP_REASON: Final = "Stopped by the user"
# A turn that ran this long rings the bell when it ends.
BELL_AFTER_S: Final = 30.0

State = Literal["idle", "running", "stopping", "switching"]

_STATE_HINTS: Final = {
    "stopping": "Stopping the turn; enter sends once it has stopped.",
    "switching": "Switching the model; esc cancels it.",
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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.factory = factory
        self.agent = agent
        self.renderer = renderer
        self.approver = approver
        self.out = out
        self.approvals = Approvals(self._ask, mode)
        self.runner: SessionRunner | None = None
        self.app: InteractiveApp | None = None
        self.queued: str | None = None
        self.activity: str | None = None
        self.title = ""
        self.models: list[str] = []
        self._clock = clock
        self._state: State = "idle"
        self._work: asyncio.Task[Any] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._stop_wanted = False
        self._stop_landed = False
        self._closed = False

    # ── What the app reads ──

    @property
    def state(self) -> State:
        return self._state

    def attach(self, app: InteractiveApp) -> None:
        self.app = app

    def status_bar(self) -> StatusBar:
        backend = self.factory.backend
        context = self.renderer.context_tokens
        yolo = self.approvals.mode == "yolo"
        return StatusBar(
            model=backend.model,
            workspace=home_label(self.agent.workspace_root or ""),
            context=f"ctx {format_tokens(context)}" if context else "",
            title=self.title or "New session",
            mode="⚠ YOLO: no approvals" if yolo else "ask",
            mode_style="class:yolo" if yolo else "class:muted",
        )

    def _hint(self, text: str) -> None:
        if self.app is not None:
            self.app.hint(text)

    # ── Enter ──

    def enter(self, text: str) -> str | None:
        """Act on Enter: run a command, send, or queue (see the app's contract)."""
        parsed = commands.parse_command(text)
        if parsed is not None:
            command, name, arguments = parsed
            if command is None:
                self.out.print(
                    _text(
                        f"Unknown command /{name}. /help lists the commands.",
                        style="error",
                    )
                )
                return ""
            allowed = command.during_turn is True or (
                command.during_turn == "show" and not arguments
            )
            if self._state != "idle" and not allowed:
                return f"/{command.name} is not available while a turn runs (esc stops it)."
            self.spawn(self._command(command.name, arguments))
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
        return _STATE_HINTS[self._state]

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
            return
        exc = task.exception()
        if exc is not None:
            logger.error("A session task failed", exc_info=exc)
            self.out.print(
                _text(
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
        if self.runner is None:
            self.title = title_from_text(text)
        self.renderer.begin_turn(text)
        self._work = self.spawn(self._turn(text))

    async def _ask(self, request: ApprovalRequest) -> ApprovalDecision:
        """Ask through the panel, recorded in the session's events during a turn."""
        if self.runner is None:
            return await self.approver.ask(request)
        return await self.runner.observe(self.approver.ask)(request)

    async def _turn(self, text: str) -> None:
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
                on_event=self.renderer.on_event,
            )
        runner = self.runner
        result: RunResult | None = None
        error: BaseException | None = None
        try:
            result = await runner.run_turn(
                text, approval=self.approvals.policy(), title=text
            )
        except SessionBusyError as exc:
            error = exc
        except Exception as exc:
            logger.exception("The turn could not run")
            error = exc
        # A second stop can make run_turn return while the runner still
        # finishes the turn: the next message must not meet it.
        while runner.busy:
            await asyncio.sleep(0.05)
        cause = "user_stop" if self._stop_wanted else ""
        self.renderer.end_turn(result, error, cause=cause)
        self._state = "idle"
        self._work = None
        if self.app is not None:
            if self._clock() - started >= BELL_AFTER_S:
                self.app.bell()
            self.app.invalidate()
        queued = self.take_queued()
        if queued is None:
            return
        if result is not None and result.status == "completed" and error is None:
            self.submit(queued)
        elif self.app is not None:
            buffer = self.app.buffer
            buffer.text = queued + ("\n" + buffer.text if buffer.text else "")
            self.out.print(
                _text("Your queued message is back in the input box.", style="muted")
            )

    def stop(self, reason: str = STOP_REASON) -> bool:
        """Stop the turn or the switch, once; Esc and Ctrl+C."""
        if self._state == "switching":
            if self._work is not None:
                self._work.cancel()
            return True
        if self._state != "running":
            return False
        live = self.renderer.live
        if live is not None and live.finished:
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

    async def _command(self, name: str, arguments: str) -> None:
        if name == "help":
            self.out.print(commands.help_text(arguments))
        elif name in ("exit", "quit"):
            if self.app is not None:
                self.app.exit()
        elif name in ("new", "clear"):
            await self.new()
        elif name == "model":
            await self.model(arguments)
        elif name == "backend":
            await self.backend(arguments)

    async def new(self) -> None:
        """``/new``: the next message starts a new session, with no grants."""
        if self.runner is not None:
            await self.runner.close()
        self.runner = None
        self.agent.load_conversation([])
        self.approvals.reset()
        self.title = ""
        self.renderer.context_tokens = 0
        label = " New session "
        side = max(2, (self.out.width - len(label)) // 2)
        self.out.print(_text("─" * side + label + "─" * side, style="muted"))

    async def model(self, arguments: str) -> None:
        """``/model``: list the server's models, or switch to one and save it."""
        backend = self.factory.backend
        if not arguments:
            self.out.print(
                _text(
                    f"Model: {backend.model} · {backend.label} · {backend.base_url or backend.label}"
                )
            )
            if not self.factory.listable():
                return
            try:
                self.models = await self.factory.list_models()
            except discovery.DiscoveryError as exc:
                self.out.print(
                    _text(f"✗ Could not list the models: {exc}", style="error")
                )
                return
            lines = _text()
            for number, model in enumerate(self.models, 1):
                mark = "❯" if model == backend.model else " "
                lines.append(
                    clean_output(f"{mark} {number}. {model}") + "\n",
                    style="accent" if mark != " " else None,
                )
            lines.append("/model NAME or /model N switches and saves.", style="muted")
            self.out.print(lines)
            return
        name = arguments
        if (
            name.isdigit()
            and name not in self.models
            and 1 <= int(name) <= len(self.models)
        ):
            name = self.models[int(name) - 1]
        if self.models and name not in self.models:
            self.out.print(
                _text(
                    f"⚠ {backend.base_url} lists no model named '{name}'; switching anyway.",
                    style="warn",
                )
            )
        if await self._switch(name, url=None):
            self.factory.use_model(name)
            self._report_switch(f"Model: {name} ({self.factory.backend.label}).")

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
                _text(
                    f"✗ Could not reach {url}: {exc}. Still using {old.base_url or old.label}.",
                    style="error",
                )
            )
            return
        except asyncio.CancelledError:
            self.out.print(
                _text(
                    f"Cancelled. Still using {old.base_url or old.label}.",
                    style="muted",
                )
            )
            return
        finally:
            self._state, self.activity, self._work = "idle", None, None
        if not models:
            self.out.print(
                _text(
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
            self.factory.use_url(url, model)
            self._report_switch(f"Backend: {url} · {model}.")

    async def _pick_model(self, url: str, models: list[str]) -> str | None:
        chat = [model for model in models if discovery.is_chat_model(model)]
        if len(models) == 1 and chat:
            return models[0]
        lines = _text(f"Models on {url}:\n")
        for number, model in enumerate(models, 1):
            lines.append(clean_output(f"  {number}. {model}") + "\n")
        self.out.print(lines)
        default = models.index(discovery.pick_model(models) or models[0]) + 1
        assert self.app is not None
        while True:
            answer = await self.app.ask("Model", str(default))
            if answer is None:
                return None
            if answer in models:
                return answer
            if answer.isdigit() and 1 <= int(answer) <= len(models):
                return models[int(answer) - 1]
            self.out.print(
                _text(
                    f"Type a number from 1 to {len(models)}, or a model id.",
                    style="muted",
                )
            )

    async def _switch(self, model: str, *, url: str | None) -> bool:
        """Move the conversation to a new Agent on *model*; False when it failed."""
        old = self.agent
        self._state, self.activity = "switching", f"Starting {model}"
        self._work = asyncio.current_task()
        try:
            new = await self.factory.build(model, url=url)
        except asyncio.CancelledError:
            self.out.print(
                _text(
                    f"Switch cancelled. Still using {self.factory.backend.model}.",
                    style="muted",
                )
            )
            return False
        except Exception as exc:
            self.out.print(
                _text(
                    f"✗ Could not start {model}: {exc}. Still using {self.factory.backend.model}.",
                    style="error",
                )
            )
            return False
        finally:
            self._state, self.activity, self._work = "idle", None, None
        new.load_conversation(old.conversation())
        if self.runner is not None:
            self.runner.replace_agent(new)
            if self.runner.session_id is not None:
                await self.store.update_session(self.runner.session_id, model=model)
        self.agent = new
        await self.factory.close(old)
        return True

    def _report_switch(self, done: str) -> None:
        problem = self.factory.save()
        if problem:
            self.out.print(
                _text(f"{done} {problem[0].upper()}{problem[1:]}.", style="warn")
            )
        else:
            self.out.print(_text(f"{done} Saved as the default.", style="ok"))

    # ── The model server at startup ──

    async def check_server(self) -> None:
        """Warn when the server does not answer, or does not list the model."""
        backend = self.factory.backend
        try:
            models = await self.factory.list_models()
        except discovery.DiscoveryError as exc:
            if not exc.reachable:
                text = f"Cannot reach the model server at {backend.base_url} ({exc}). Start it, then send your message."
            else:
                text = f"{backend.base_url} could not list its models ({exc}); the first message may fail."
            self.out.print(_text(f"⚠ {text}", style="warn"))
            return
        self.models = models
        if models and backend.model not in models:
            self.out.print(
                _text(
                    f"⚠ {backend.base_url} lists no model named '{backend.model}'; the first message may fail.",
                    style="warn",
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
        for agent in self.factory.live_agents():
            with contextlib.suppress(Exception):
                await self.factory.close(agent)
        await self.store.aclose()
