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
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Final, cast

from localmelo.melo.agent import AgentOptions
from localmelo.melo.agent.options import reasoning_refusal
from localmelo.melo.contracts import own_gateway
from localmelo.support import config, gateway_endpoint, onboard
from localmelo.support.backends import discovery
from localmelo.support.cli.views import BackendInfo
from localmelo.support.sessions.agents import (
    configured_cache_options,
    configured_context_options,
    configured_image_options,
    configured_session_options,
)

LOG_DIR: Final = "~/.cache/localmelo/logs"
LOG_FILE: Final = "cli.log"
HISTORY_FILE: Final = "~/.cache/localmelo/cli_history"
# At most one line on screen for errors that go to the log, per this many seconds.
SCREEN_ERRORS_S: Final = 5.0
# After a signal or a hangup, how long the app gets to end on its own.
APP_END_S: Final = 1.0

# Overrides [cli] chat for one process: "local" keeps the terminal's own chat
# (a script or a test harness that must never reach a gateway).
CHAT_ENV: Final = "LOCALMELO_CHAT"

NO_TERMINAL: Final = (
    "melo: the interactive UI needs a terminal (stdin and stdout must be TTYs).\n"
    '  Run one task:      melo "your task"\n'
    "  Start the gateway: melo serve"
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


def expect_gateway(cfg: config.Config, endpoint_path: str | None = None) -> None:
    """Tell auto mode the ports melo's gateway may listen on (#339).

    The terminal's own chat runs no gateway, so its Auto steps are kept off
    the one the app or ``melo --serve`` runs, in the review and in the
    sandbox: the port the settings give it, and the running gateway's own
    (its endpoint file), when that differs.
    """
    port = getattr(getattr(cfg, "gateway", None), "port", None)
    live = gateway_endpoint.read(endpoint_path)
    own_gateway.expect(
        port if isinstance(port, int) else None,
        live.port if live is not None else None,
    )


def ui_terminal() -> bool:
    """Whether stdin and stdout are a terminal the UI can draw on (not TERM=dumb)."""
    try:
        ttys = sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False
    return ttys and os.environ.get("TERM") != "dumb"


def wants_main_chat(args: argparse.Namespace, choice: Choice, chat: str) -> bool:
    """Whether bare melo opens the main chat, as ``[cli] chat`` (*chat*) says.

    ``--url``, ``-c``, ``-r`` and ``--yolo`` choose what only the terminal's
    own chat does, so they keep it; so does ``LOCALMELO_CHAT=local``.
    """
    override = os.environ.get(CHAT_ENV, "")
    chosen = override if override in config.CLI_CHATS else chat
    return (
        chosen == "main"
        and choice.cfg is not None
        and not getattr(args, "yolo", False)
        and not getattr(args, "continue_session", False)
        and getattr(args, "resume", None) is None
    )


def _configured(
    cfg: config.Config, model: str | None, args: argparse.Namespace | None = None
) -> Choice | int:
    if args is not None:
        for name in ("context_window", "output_reserve", "safety_margin"):
            value = getattr(args, name, None)
            if value is not None:
                setattr(cfg.agent, name, value)
        if any(
            getattr(args, name, None) is not None
            for name in ("context_window", "output_reserve", "safety_margin")
        ):
            try:
                configured_context_options(cfg, AgentOptions(), {})
            except ValueError as exc:
                print(f"melo: error: {exc}", file=sys.stderr)
                return 2
    try:
        cfg.validate_or_raise()
    except config.ConfigError as exc:
        print(f"\n  {exc}\n", file=sys.stderr)
        return 1
    from localmelo.support.cli.agents import config_backend

    return Choice(config_backend(cfg, model), cfg)


def _with_url(args: argparse.Namespace) -> Choice | int:
    """``--base-url``: the model from the flag or the server's list; maybe saved."""
    from localmelo.support.cli.agents import url_backend, url_embedding

    offer = not config.load().is_configured
    listed = not args.chat_model or offer
    found = asyncio.run(
        discovery.choose_model(args.base_url, args.chat_model, list_anyway=offer)
    )
    url, model, models = found.url, found.model, found.models
    if model is None and models:
        print(f"melo: error: {found.refusal()}", file=sys.stderr)
        return 2
    saved = (
        offer
        and model is not None
        and model in models
        and onboard.offer_to_save(url, model)
    )
    warning = ""
    if found.reason and not args.chat_model:
        warning = found.warning(" (/model picks one)")
    picked = len(models) if not args.chat_model and len(models) > 1 else 0
    backend = url_backend(
        url,
        model,
        source="saved" if saved else "--base-url",
        listed=picked,
        embedding=url_embedding(args.embedding_base_url, args.embedding_model),
    )
    return Choice(
        backend,
        None,
        saves=bool(saved),
        checked=listed,
        warning=warning,
        models=tuple(models),
    )


def _developer(args: argparse.Namespace) -> bool:
    """Whether ``--developer`` or ``[ui] mode = "developer"`` is on."""
    if getattr(args, "developer", False):
        return True
    try:
        return bool(config.load().ui.mode == "developer")
    except Exception:  # a broken config file must not stop the wizard
        return False


def choose_backend(args: argparse.Namespace) -> Choice | int:
    """The backend for the session, or an exit code (see the module docstring).

    EOFError and KeyboardInterrupt from a question reach the caller.
    """
    if args.reconfigure:
        cfg = onboard.run_backend_setup(guided=not _developer(args))
        if cfg is None:
            return 0
        return _configured(cfg, args.chat_model, args)
    if args.base_url:
        return _with_url(args)
    cfg = config.load()
    if cfg.is_configured:
        return _configured(cfg, args.chat_model, args)
    setup = onboard.run_first_setup()
    if setup is None:
        return 0
    if setup.saved:
        chosen = _configured(config.load(), args.chat_model, args)
        if isinstance(chosen, Choice):
            # The first run just said how long-term memory is added, if it is off.
            backend = replace(chosen.backend, source="saved", memory_told=True)
            return replace(chosen, backend=backend, checked=True)
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
    validate_options: Callable[[AgentOptions], None] | None = None,
) -> int:
    """Open the terminal UI for bare melo; return the exit code.

    A signal that ended the session ends the process by that signal.
    """
    if not ui_terminal():
        if args.reconfigure:
            # A script can pipe the wizard's answers in; nothing opens after.
            try:
                onboard.run_backend_setup(guided=not _developer(args))
            except onboard.ModelChoiceError as exc:
                print(f"melo: error: {exc}", file=sys.stderr)
                return 2
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
    except onboard.ModelChoiceError as exc:
        print(f"melo: error: {exc}", file=sys.stderr)
        return 2
    except EOFError:
        print("\nSetup cancelled.")
        return 0
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        return 130
    except (OSError, tomllib.TOMLDecodeError) as exc:
        path = os.path.expanduser(config.CONFIG_PATH)
        print(f"melo: cannot read {path}: {exc}", file=sys.stderr)
        return 1
    if isinstance(choice, int):
        return choice
    explicit = {
        name
        for name in (
            "fixed_prefix",
            "window_eviction",
            "reflect_tool_guard",
            "cache_layout",
            "window_messages",
            "captured_history",
            "reasoning",
            "replay_turns",
            "note_writer",
            "retriever",
            "recall_at",
            "recall_k",
            "recall_kinds",
            "recall_placement",
            "recall_min_score",
        )
        if getattr(args, name, None) is not None
    }
    try:
        if choice.cfg is None:
            loaded = config.load()
            options = configured_cache_options(
                loaded, options, keep=explicit, backend=choice.backend.key
            )
            if "captured_history" not in explicit and loaded.agent.captured_history:
                options = replace(options, captured_history=True)
            options = configured_image_options(loaded, options)
            options = configured_context_options(
                loaded,
                options,
                {
                    name: getattr(args, name, None)
                    for name in ("context_window", "output_reserve", "safety_margin")
                },
            )
        options = configured_session_options(
            choice.cfg,
            options,
            explicit=explicit,
            budget_overrides={
                name: getattr(args, name, None)
                for name in ("context_window", "output_reserve", "safety_margin")
            },
        )
    except (ValueError, config.ConfigError) as exc:
        parser.error(str(exc))
    if (
        getattr(args, "reasoning_field", None) is not None
        and choice.backend.key == "ollama"
    ):
        parser.error(
            "argument --reasoning-field: the ollama backend always reads and sends the thinking field"
        )
    section = (
        None
        if choice.cfg is None
        else getattr(choice.cfg, choice.cfg.chat_backend, None)
    )
    field = getattr(args, "reasoning_field", None)
    if field is None:
        field = (
            "thinking"
            if choice.backend.key == "ollama"
            else getattr(section, "reasoning_field", None)
        )
    required = bool(getattr(section, "requires_reasoning_replay", False))
    effective_mode = options.reasoning
    if effective_mode == "compress" and required:
        effective_mode = "compress+replay"
    if refusal := reasoning_refusal(effective_mode, field, required):
        parser.error("argument --reasoning: " + refusal)
    if validate_options is not None:
        validate_options(replace(options, reasoning=effective_mode))
    if options.retriever == "embedding":
        if not choice.backend.embedding:
            parser.error("argument --retriever: embedding needs an embedding provider")
        if not options.memory:
            parser.error("argument --retriever: embedding needs --memory on")
    # Step 4: the MCP config, --mcp-config first, else the config file's key.
    requested = getattr(args, "mcp_config", None)
    mcp_config = requested or config.load().mcp_config or None
    mcp_source = "--mcp" if requested else ("mcp_config" if mcp_config else "")
    mcp_servers: list[Any] = []
    if mcp_config is not None:
        from localmelo.support.mcp import McpConfigError, load_mcp_config

        try:
            mcp_servers = load_mcp_config(os.path.expanduser(mcp_config))
        except McpConfigError as exc:
            print(f"melo: error: {exc}", file=sys.stderr)
            return 2
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
                mcp_config=mcp_config,
                mcp_source=mcp_source,
                mcp_servers=mcp_servers,
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
    mcp_config: str | None = None,
    mcp_source: str = "",
    mcp_servers: list[Any] | None = None,
) -> tuple[int, int | None]:
    """Run the session: open the store, build the Agent, show the banner, run the app.

    Returns the exit code and the signal that ended the session, if one did.
    """

    from prompt_toolkit.history import History, InMemoryHistory
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.patch_stdout import patch_stdout
    from rich.text import Text

    from localmelo.support import task_cli
    from localmelo.support.cli import main_chat
    from localmelo.support.cli.agents import AgentFactory
    from localmelo.support.cli.app import InteractiveApp, open_history, terminal_input
    from localmelo.support.cli.approvals import TerminalApprover
    from localmelo.support.cli.completion import SlashCompleter
    from localmelo.support.cli.output import TerminalOut
    from localmelo.support.cli.render import TurnRenderer
    from localmelo.support.cli.session import STOP_REASON, CliSession
    from localmelo.support.cli.signals import ExitSignals, save_terminal
    from localmelo.support.cli.termcolors import query_colors
    from localmelo.support.cli.theme import (
        activate,
        active,
        detect,
        in_symbols,
        symbols_for,
        wants_no_color,
    )
    from localmelo.support.cli.views import banner
    from localmelo.support.cli.work import GatewayTasks, WorkBeside
    from localmelo.support.sessions.display import display_version, home_label
    from localmelo.support.sessions.runner import ResumeMode
    from localmelo.support.sessions.store import (
        SessionInfo,
        SessionStore,
        SessionStoreError,
    )

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
    browser_driver = config.load().browser.driver
    if browser_driver not in config.BROWSER_DRIVERS:
        logging.getLogger(__name__).warning(
            "Invalid [browser] driver %r; browsing is off", browser_driver
        )
        browser_driver = "off"
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
            reasoning_field=getattr(args, "reasoning_field", None),
            retrieval_extra_body=getattr(args, "retrieval_extra_body", None),
            mcp_config=mcp_config,
            mcp_source=mcp_source,
            mcp_servers=mcp_servers,
            saves=choice.saves,
            browser_driver=browser_driver,
        )
        try:
            agent = await factory.build()
        except Exception as exc:
            print(f"melo: could not start the agent: {exc}", file=sys.stderr)
            return 1, None

        approver = TerminalApprover()
        session_config = config.load()
        expect_gateway(session_config)
        # Config validation never fails on [cli] (a display preference), and a
        # --base-url session validates nothing at all, so both paths land here:
        # a bad value falls back with a warning rather than turning streaming
        # on or keeping an unknown hint.
        if errors := session_config.cli_errors():
            logging.getLogger(__name__).warning(
                "Invalid [cli] settings; using defaults: %s", "; ".join(errors)
            )
            session_config.cli = config.CliConfig()
        # Before prompt_toolkit takes the terminal: ask it for its colours.
        activate(
            detect(
                colors=session_config.cli.colors,
                bg=session_config.cli.background,
                ask=lambda: query_colors(
                    sys.stdin.fileno(), sys.stdout.fileno(), os.environ
                ),
            )
        )
        developer = _developer(args)
        renderer = TurnRenderer(
            out,
            symbols=symbols,
            backend=lambda: factory.backend,
            developer=developer,
            reasoning_tags=session_config.cli.reasoning_tags,
            snapshots=approver.snapshots,
        )
        raw_mode = session_config.sessions.resume_mode
        invalid_mode = raw_mode not in ("pairs", "window")
        if invalid_mode:
            logging.getLogger(__name__).warning(
                "Invalid [sessions] resume_mode %r; using pairs", raw_mode
            )
        pricing = session_config.pricing
        if errors := session_config.pricing_errors():
            logging.getLogger(__name__).warning(
                "Invalid [pricing]; showing tokens instead of cost: %s",
                "; ".join(errors),
            )
            pricing = {}

        starting = (
            getattr(args, "continue_session", False)
            or getattr(args, "resume", None) is not None
        )
        main_wanted = not starting and wants_main_chat(
            args, choice, session_config.cli.chat
        )

        def draw_banner(selected: SessionInfo | None = None) -> None:
            out.print(
                banner(
                    version=display_version(),
                    backend=factory.backend,
                    workspace=agent.workspace_root or workspace,
                    tools=agent.list_tools(),
                    memory=factory.memory,
                    yolo=args.yolo,
                    width=out.width,
                    symbols=symbols,
                    session_title=selected.title if selected is not None else None,
                    developer=developer,
                    palette=active(),
                    chat=main_chat.MAIN_TITLE if main_wanted else None,
                )
            )

        session = CliSession(
            store=store,
            factory=factory,
            agent=agent,
            renderer=renderer,
            approver=approver,
            # [approvals] mode: the terminal's own chat starts as a new chat does.
            mode=(
                "yolo"
                if args.yolo
                else "auto" if session_config.approvals.mode == "auto" else "ask"
            ),
            resume_mode=cast(ResumeMode, "pairs" if invalid_mode else raw_mode),
            stream=session_config.cli.stream,
            reasoning_tags=session_config.cli.reasoning_tags,
            status_line=session_config.cli.status_line,
            out=out,
            developer=developer,
            pricing=pricing,
            work=WorkBeside(GatewayTasks()),
            welcome=draw_banner,
        )

        def show_banner(selected: SessionInfo | None = None) -> None:
            draw_banner(selected)
            if invalid_mode:
                out.print(
                    Text(
                        f"{symbols.warn} Invalid [sessions] resume_mode; using pairs.",
                        style="warn",
                    )
                )
            if choice.warning:
                out.print(Text(f"{symbols.warn} {choice.warning}", style="warn"))
            if factory.mcp_error:
                from localmelo.melo.checker.redaction import redact_secrets

                problem = redact_secrets(factory.mcp_error)[0]
                out.print(
                    Text(
                        f"{symbols.warn} MCP servers failed to start: {problem}. "
                        "Continuing without MCP tools; /mcp has details.",
                        style="warn",
                    )
                )
            if not choice.checked and factory.listable():
                session.spawn(session.check_server())

        connected: main_chat.Link | main_chat.Offline | None = None
        if main_wanted:
            # The main chat lives in the gateway: find it, or start it here.
            def starting_gateway() -> None:
                line = Text(
                    in_symbols("Starting the gateway in the background…", symbols),
                    style="muted",
                )
                loop.call_soon_threadsafe(out.print, line)

            found = connected = await asyncio.to_thread(
                main_chat.connect, session_config, on_start=starting_gateway
            )
            if isinstance(found, main_chat.Link):
                session.remote = main_chat.MainChat(
                    session, main_chat.GatewayClient(found.url), found
                )
                session.title = main_chat.MAIN_TITLE
                url = found.url
                session.work = WorkBeside(
                    GatewayTasks(
                        connect=lambda timeout: task_cli.connect(url, timeout=timeout)
                    )
                )
            else:
                main_wanted = False  # the welcome does not name the main chat
        if not starting:
            show_banner()
        if isinstance(connected, main_chat.Link):
            out.print(Text(connected.line(), style="muted"))
        elif isinstance(connected, main_chat.Offline):
            out.print(Text(f"{symbols.warn} {connected.line()}", style="warn"))
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
            completer=SlashCompleter(
                lambda: session.models, lambda: session.navigation.refs
            ),
            input=terminal_input(sys.stdin, on_close=terminal_closed),
            output=output,
            approver=approver,
            no_color=no_color,
        )
        session.attach(ui)
        if session.work is not None:
            session.work.on_change = ui.invalidate
            session.spawn(session.work.watch())
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
        app = asyncio.ensure_future(ui.run())
        session.spawn(session.navigation.refresh_refs())
        if session.remote is not None:
            session.spawn(session.open_main_chat())
        if starting:
            session._state = "switching"
            session._work = session.spawn(
                session.navigation.startup(
                    getattr(args, "continue_session", False),
                    getattr(args, "resume", None),
                    on_choice=show_banner,
                )
            )
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
                out.print(Text(in_symbols("Closing…", symbols), style="muted"))
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
        if session.shell_ran:
            restore()  # a ! command's end may have re-applied the prompt's raw mode
        if session.saved_resume_hint:
            out.print(session._text(session.saved_resume_hint))
        return session.exit_code, signals.received
    finally:
        # Whatever ended the session (a signal, a closed terminal, Ctrl+C
        # while the Agent starts), nothing stays open: the store's thread
        # would keep the process alive.
        if factory is not None:
            for live in factory.live_agents():
                with contextlib.suppress(Exception):
                    await factory.close(live)
        await store.aclose()
