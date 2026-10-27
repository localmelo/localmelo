"""Start the terminal UI: the terminal, the workspace, the backend, then the app.

The backend is the first that applies: ``--reconfigure`` runs the setup
wizard; ``--base-url`` is used as given (the model from ``--chat-model`` or
the server's list, and, with no config file, one question whether to save
it); a configured backend is used as it is, with no check before the UI;
with no config, the first run asks for a server's URL (prefilled with one
found on this machine) and a model from its list, and saves them. Every
question here is a plain line, before the app starts.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import logging.handlers
import os
import signal
import sys
import time
import tomllib
from dataclasses import dataclass, replace
from typing import Any, Final

from localmelo.melo.agent import AgentOptions
from localmelo.support import config, onboard
from localmelo.support.backends import discovery
from localmelo.support.backends.openai_compat import normalize_url
from localmelo.support.cli.views import BackendInfo

LOG_DIR: Final = "~/.cache/localmelo/logs"
LOG_FILE: Final = "cli.log"
HISTORY_FILE: Final = "~/.cache/localmelo/cli_history"
# At most one line on screen for errors that go to the log, per this many seconds.
SCREEN_ERRORS_S: Final = 5.0
# After a signal or a hangup, how long the app gets to end on its own.
APP_END_S: Final = 1.0

NO_TERMINAL: Final = (
    "melo: the interactive UI needs a terminal (stdin and stdout must be TTYs).\n"
    '  Run one task:      melo "your task"\n'
    "  Start the gateway: melo --serve"
)


@dataclass(frozen=True)
class Choice:
    """The backend startup chose, and what the session may do with it.

    *checked* says the server was listed at startup, so the banner's check
    does not list it again; *models* is what it listed then; *warning* is
    a line for under the banner.
    """

    backend: BackendInfo
    cfg: config.Config | None
    saves: bool = True
    checked: bool = False
    warning: str = ""
    models: tuple[str, ...] = ()


def ui_terminal() -> bool:
    """Whether stdin and stdout are a terminal the UI can draw on (not TERM=dumb)."""
    try:
        ttys = sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False
    return ttys and os.environ.get("TERM") != "dumb"


def _configured(cfg: config.Config, model: str | None) -> Choice | int:
    try:
        cfg.validate_or_raise()
    except config.ConfigError as exc:
        print(f"\n  {exc}\n", file=sys.stderr)
        return 1
    from localmelo.support.cli.agents import config_backend

    return Choice(config_backend(cfg, model), cfg)


def _with_url(args: argparse.Namespace) -> Choice:
    """``--base-url``: the model from the flag or the server's list; maybe saved."""
    from localmelo.support.cli.agents import url_backend

    url: str = args.base_url
    offer = not config.load().is_configured
    models: list[str] = []
    reason = ""
    listed = not args.chat_model or offer
    if listed:
        try:
            models = asyncio.run(
                discovery.list_models(discovery.OPENAI_COMPATIBLE, url)
            )
        except discovery.DiscoveryError as exc:
            reason = str(exc)
        else:
            url, reason = normalize_url(url), "" if models else "it lists no models"
    model = args.chat_model or discovery.pick_model(models)
    saved = offer and model in models and onboard.offer_to_save(url, model)
    warning = ""
    if reason and not args.chat_model:
        warning = (
            f"Could not list the models at {url} ({reason}); asking for the model "
            "'default' (/model picks one)."
        )
    picked = len(models) if not args.chat_model and len(models) > 1 else 0
    backend = url_backend(
        url, model, source="saved" if saved else "--base-url", listed=picked
    )
    return Choice(
        backend,
        None,
        saves=bool(saved),
        checked=listed,
        warning=warning,
        models=tuple(models),
    )


def choose_backend(args: argparse.Namespace) -> Choice | int:
    """The backend for the session, or an exit code (see the module docstring).

    EOFError and KeyboardInterrupt from a question reach the caller.
    """
    if args.reconfigure:
        cfg = onboard.run_backend_setup()
        if cfg is None:
            return 0
        return _configured(cfg, args.chat_model)
    if args.base_url:
        return _with_url(args)
    cfg = config.load()
    if cfg.is_configured:
        return _configured(cfg, args.chat_model)
    setup = onboard.run_first_setup()
    if setup is None:
        return 0
    if setup.saved:
        chosen = _configured(config.load(), args.chat_model)
        if isinstance(chosen, Choice):
            return replace(
                chosen, backend=replace(chosen.backend, source="saved"), checked=True
            )
        return chosen
    from localmelo.support.cli.agents import url_backend

    return Choice(
        url_backend(setup.url, args.chat_model or setup.model, source="unsaved"), None
    )


class _ScreenErrors(logging.Handler):
    """Shows an ERROR of localmelo's as one dim line, at most once per 5 s."""

    def __init__(self, show: Any) -> None:
        super().__init__(logging.ERROR)
        self._show = show
        self._last = -SCREEN_ERRORS_S

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith("localmelo"):
            return
        now = time.monotonic()
        if now - self._last < SCREEN_ERRORS_S:
            return
        self._last = now
        text = record.getMessage().split("\n", 1)[0]
        if record.exc_info and record.exc_info[1] is not None:
            exc = record.exc_info[1]
            text = f"{type(exc).__name__}: {exc}"
        self._show(text)


class _PrivateRotatingHandler(logging.handlers.RotatingFileHandler):
    """A rotating log whose files are created 0600: the log holds tool output."""

    def _open(self) -> Any:
        fd = os.open(self.baseFilename, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        os.close(fd)
        return super()._open()


def configure_logging() -> str:
    """Send all logging to ``cli.log`` (0600, in a 0700 folder); return its path."""
    directory = os.path.expanduser(LOG_DIR)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    path = os.path.join(directory, LOG_FILE)
    handler = _PrivateRotatingHandler(
        path, maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.captureWarnings(True)
    return path


def run_interactive(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    options: AgentOptions,
) -> int:
    """Open the terminal UI for bare melo; return the exit code.

    A signal that ended the session ends the process by that signal.
    """
    if not ui_terminal():
        if args.reconfigure:
            # A script can pipe the wizard's answers in; nothing opens after.
            try:
                onboard.run_backend_setup()
            except EOFError:
                print("\nSetup cancelled.")
            except KeyboardInterrupt:
                print("\nSetup cancelled.")
                return 130
            return 0
        print(NO_TERMINAL, file=sys.stderr)
        return 2
    workspace = os.path.realpath(os.path.expanduser(args.workspace or os.getcwd()))
    if not os.path.isdir(workspace):
        print(f"melo: --workspace: not a directory: {args.workspace}", file=sys.stderr)
        return 2
    try:
        choice = choose_backend(args)
    except EOFError:
        print("\nSetup cancelled.")
        return 0
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        return 130
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"melo: cannot read {config.CONFIG_PATH}: {exc}", file=sys.stderr)
        return 1
    if isinstance(choice, int):
        return choice
    try:
        log_path = configure_logging()
    except OSError as exc:
        print(f"melo: cannot open the log in {LOG_DIR}: {exc}", file=sys.stderr)
        return 1
    from localmelo.support.cli.signals import exit_by_signal

    started = time.monotonic()
    try:
        code, signum = asyncio.run(
            main_async(
                choice,
                args=args,
                options=options,
                workspace=workspace,
                log_path=log_path,
            )
        )
    except KeyboardInterrupt:
        return 130
    logging.getLogger(__name__).info(
        "Interactive session ended after %.1f s", time.monotonic() - started
    )
    if signum is not None:
        exit_by_signal(signum)
    return code


async def main_async(
    choice: Choice,
    *,
    args: argparse.Namespace,
    options: AgentOptions,
    workspace: str,
    log_path: str,
) -> tuple[int, int | None]:
    """Run the session: open the store, build the Agent, show the banner, run the app.

    Returns the exit code and the signal that ended the session, if one did.
    """
    from prompt_toolkit.history import History, InMemoryHistory
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.patch_stdout import patch_stdout
    from rich.text import Text

    from localmelo.support.cli.agents import AgentFactory
    from localmelo.support.cli.app import InteractiveApp, open_history, terminal_input
    from localmelo.support.cli.approvals import TerminalApprover
    from localmelo.support.cli.completion import SlashCompleter
    from localmelo.support.cli.output import TerminalOut
    from localmelo.support.cli.render import TurnRenderer
    from localmelo.support.cli.session import STOP_REASON, CliSession
    from localmelo.support.cli.signals import ExitSignals, save_terminal
    from localmelo.support.cli.theme import symbols_for, wants_no_color
    from localmelo.support.cli.views import banner
    from localmelo.support.sessions.display import home_label, localmelo_version
    from localmelo.support.sessions.store import SessionStore, SessionStoreError

    log_label = home_label(log_path)
    no_color = wants_no_color()
    output = Vt100_Output.from_pty(sys.stdout, term=os.environ.get("TERM"))
    output.enable_cpr = os.environ.get("LOCALMELO_CPR") == "1"
    out = TerminalOut(output, no_color=no_color)
    symbols = symbols_for(sys.stdout.encoding)

    def show_error(text: str) -> None:
        out.print(
            Text(f"{symbols.warn} {text} (details in {log_label})", style="muted")
        )

    loop = asyncio.get_running_loop()
    last_shown = [-SCREEN_ERRORS_S]

    def on_exception(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        logging.getLogger("localmelo.cli").error(
            "Unhandled: %s", context.get("message"), exc_info=exc
        )
        if time.monotonic() - last_shown[0] >= SCREEN_ERRORS_S:
            last_shown[0] = time.monotonic()
            text = (
                f"{type(exc).__name__}: {exc}" if exc else str(context.get("message"))
            )
            show_error(text)

    loop.set_exception_handler(on_exception)
    logging.getLogger().addHandler(_ScreenErrors(show_error))

    store = SessionStore(writer="cli")
    factory: AgentFactory | None = None
    try:
        try:
            await store.open()
        except SessionStoreError as exc:
            print(f"melo: {exc}", file=sys.stderr)
            return 1, None
        factory = AgentFactory(
            backend=choice.backend,
            cfg=choice.cfg,
            options=options,
            workspace=workspace,
            embedding_base_url=args.embedding_base_url,
            embedding_model=args.embedding_model,
            request_timeout_s=args.request_timeout,
            saves=choice.saves,
        )
        try:
            agent = await factory.build()
        except Exception as exc:
            print(f"melo: could not start the agent: {exc}", file=sys.stderr)
            return 1, None

        renderer = TurnRenderer(out, symbols=symbols, backend=lambda: factory.backend)
        approver = TerminalApprover()
        session = CliSession(
            store=store,
            factory=factory,
            agent=agent,
            renderer=renderer,
            approver=approver,
            mode="yolo" if args.yolo else "ask",
            out=out,
        )
        out.print(
            banner(
                version=localmelo_version(),
                backend=factory.backend,
                workspace=agent.workspace_root or workspace,
                tools=agent.list_tools(),
                memory=factory.memory,
                yolo=args.yolo,
                width=out.width,
                symbols=symbols,
            )
        )
        if choice.warning:
            out.print(Text(f"{symbols.warn} {choice.warning}", style="warn"))
        history: History
        try:
            history = open_history(os.path.expanduser(HISTORY_FILE))
        except OSError as exc:
            history = InMemoryHistory()
            out.print(
                Text(
                    f"{symbols.warn} The input history is not saved this time: {exc}",
                    style="warn",
                )
            )

        def terminal_closed() -> None:
            loop.call_soon(signals.hangup)  # signals is set below

        ui = InteractiveApp(
            session,
            renderer,
            symbols=symbols,
            history=history,
            completer=SlashCompleter(lambda: session.models),
            input=terminal_input(sys.stdin, on_close=terminal_closed),
            output=output,
            approver=approver,
            no_color=no_color,
        )
        session.attach(ui)
        session.models = list(choice.models)
        restore = save_terminal()

        signals = ExitSignals(
            factory,
            shutdown=lambda reason: session.shutdown(reason, cause="shutdown"),
            end_app=ui.exit,
            on_hangup=out.detach,
            restore_terminal=restore,
        )
        signals.install()
        if not choice.checked and factory.listable():
            session.spawn(session.check_server())
        app = asyncio.ensure_future(ui.run())
        arrived = asyncio.ensure_future(signals.arrived.wait())
        try:
            with patch_stdout(raw=True):
                await asyncio.wait({app, arrived}, return_when=asyncio.FIRST_COMPLETED)
                # A signal or a hangup ends the app; on a terminal that is
                # gone prompt_toolkit can wait forever for its own drawing,
                # so the app gets a moment, then is left behind.
                if not app.done():
                    await asyncio.wait({app}, timeout=APP_END_S)
                if not app.done():
                    app.cancel()
                    await asyncio.wait({app}, timeout=APP_END_S)
            arrived.cancel()
            if app.done() and not app.cancelled():
                app.result()
        except EOFError:
            signals.hangup()  # the terminal closed
        except OSError:
            # Reading or drawing failed: the terminal is gone, with or without
            # a SIGHUP (nohup ignores it).
            logging.getLogger(__name__).info("The terminal failed", exc_info=True)
            signals.hangup()
        # The app is gone: Ctrl+C now says "Closing…", and a second one ends melo
        # at once, killing the tool providers.
        presses = [0]

        def on_sigint() -> None:
            presses[0] += 1
            if presses[0] == 1:
                out.print(Text("Closing…", style="muted"))
                return
            for agent in factory.live_agents():
                for provider in agent._tool_providers:
                    with contextlib.suppress(Exception):
                        provider.kill()
            os._exit(130)

        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signal.SIGINT, on_sigint)
        try:
            if signals.shutting_down is not None:
                await signals.shutting_down
            else:
                await session.shutdown(STOP_REASON, cause="user_stop")
        finally:
            signals.uninstall()
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(signal.SIGINT)
        return 0, signals.received
    finally:
        # Whatever ended the session (a signal, a closed terminal, Ctrl+C
        # while the Agent starts), nothing stays open: the store's thread
        # would keep the process alive.
        if factory is not None:
            for live in factory.live_agents():
                with contextlib.suppress(Exception):
                    await factory.close(live)
        await store.aclose()
