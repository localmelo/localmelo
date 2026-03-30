from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING

from localmelo.melo.agent import Agent

if TYPE_CHECKING:
    from localmelo.support.config import Config


def _register_backends() -> None:
    """Ensure built-in backend adapters are registered.

    Delegates to :func:`~localmelo.support.backends.ensure_defaults_registered`
    which is idempotent — safe to call multiple times and also called
    automatically by :func:`get_backend` / :func:`list_backends`.
    """
    from localmelo.support.backends import ensure_defaults_registered

    ensure_defaults_registered()


async def _run(agent: Agent, query: str | None) -> None:
    try:
        if query:
            result = await agent.run(query)
            print(result)
        else:
            print("melo agent (type 'exit' to quit)")
            while True:
                try:
                    q = input("\n> ")
                except (EOFError, KeyboardInterrupt):
                    break
                if q.strip().lower() in ("exit", "quit"):
                    break
                if not q.strip():
                    continue
                result = await agent.run(q)
                print(f"\n{result}")
    finally:
        await agent.close()


def main() -> None:
    _register_backends()

    parser = argparse.ArgumentParser(prog="melo", description="localmelo agent")
    parser.add_argument("query", nargs="*", help="Task to execute (direct mode)")
    parser.add_argument("--base-url", default=None, help="LLM API base URL")
    parser.add_argument("--chat-model", default=None, help="Chat model name")

    # gateway mode
    parser.add_argument("--serve", action="store_true", help="Run as gateway server")
    parser.add_argument("--host", default=None, help="Gateway listen host")
    parser.add_argument("--port", type=int, default=None, help="Gateway listen port")

    # setup
    parser.add_argument(
        "--reconfigure", action="store_true", help="Re-run backend setup"
    )
    parser.add_argument(
        "--daemon",
        choices=["install", "uninstall", "status"],
        help="Manage launchd daemon",
    )

    args = parser.parse_args()

    # daemon commands
    if args.daemon:
        from localmelo.support.gateway import daemon

        if args.daemon == "install":
            kwargs: dict = {}
            if args.port is not None:
                kwargs["port"] = args.port
            path = daemon.install(**kwargs)
            print(f"Daemon installed: {path}")
        elif args.daemon == "uninstall":
            if daemon.uninstall():
                print("Daemon uninstalled")
            else:
                print("Daemon not installed")
        elif args.daemon == "status":
            info = daemon.status()
            for k, v in info.items():
                print(f"  {k}: {v}")
        return

    # direct mode (no gateway, original behavior)
    if args.query and not args.serve:
        query = " ".join(args.query)
        agent = _build_direct_mode_agent(args)
        asyncio.run(_run(agent, query))
        return

    # gateway mode: load or create config
    from localmelo.support import config
    from localmelo.support.onboard import run_backend_setup

    cfg = config.load()

    if args.reconfigure or not cfg.is_configured:
        setup_cfg = run_backend_setup()
        if setup_cfg is None:
            return
        cfg = setup_cfg

    # apply CLI overrides
    if args.host is not None:
        cfg.gateway.host = args.host
    if args.port is not None:
        cfg.gateway.port = args.port

    # validate before anything else — fail fast with actionable errors
    from localmelo.support.config import ConfigError

    try:
        cfg.validate_or_raise()
    except ConfigError as e:
        print(f"\n  {e}\n", file=__import__("sys").stderr)
        raise SystemExit(1) from e

    # start gateway
    _start_gateway(cfg)


def _build_direct_mode_agent(args: argparse.Namespace) -> Agent:
    """Build an Agent for direct CLI mode from CLI flags.

    Two cases:
    1. --base-url provided: generic OpenAI-compatible endpoint injection,
       no embedding (preserves the exact URL as given).
    2. No --base-url: use configured backends from the config file.
    """
    base_url = args.base_url

    if base_url:
        # Generic OpenAI-compatible endpoint — no embedding in direct mode
        from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

        model = args.chat_model or "default"
        llm = OpenAICompatLLM(base_url=base_url, model=model)
        return Agent(llm=llm, embedding=None)

    # No --base-url: use configured backends from config file
    from localmelo.support import config

    cfg = config.load()
    if not cfg.is_configured:
        import sys

        print(
            "No backend configured. Run 'melo --serve --reconfigure' to set up,\n"
            "or use --base-url to connect to an OpenAI-compatible endpoint.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return Agent(config=cfg)


def _start_gateway(cfg: Config) -> None:
    """Thin seam for gateway startup — keeps the lazy import local."""
    from localmelo.support.gateway import start_gateway

    print("\n  Starting localmelo gateway ...")
    print(f"  Chat: {cfg.chat_backend}")
    print(f"  Embedding: {cfg.embedding_backend}")
    print(f"  Gateway: http://{cfg.gateway.host}:{cfg.gateway.port}")
    print(f"  Web UI:  http://{cfg.gateway.host}:{cfg.gateway.port}/")
    print()

    start_gateway(cfg)


if __name__ == "__main__":
    main()
