"""Recording helpers for one-shot worker sessions."""

from __future__ import annotations

import asyncio
import os
import subprocess
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from localmelo.melo.checker.redaction import redact_secrets

if TYPE_CHECKING:
    from localmelo.melo.agent import Agent
    from localmelo.melo.contracts.events import EventSink
    from localmelo.melo.contracts.inbox import InboxSource
    from localmelo.melo.schema import RunResult


def redact_worker(value: Any) -> Any:
    """Redact text throughout a JSON-shaped worker record."""
    if isinstance(value, str):
        return redact_secrets(value)[0]
    if isinstance(value, dict):
        return {redact_worker(key): redact_worker(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_worker(item) for item in value]
    return value


def sensitive_read(
    tool: str, arguments: Any, workspace: str | None = None
) -> str | None:
    """Identify direct reads of common secret files, including symlink targets."""
    name = tool.lower()
    if name.startswith("mcp__") and "__" in name[5:]:
        name = name.rsplit("__", 1)[-1]
    if name not in {
        "file_read",
        "read",
        "read_file",
        "read_text_file",
    } or not isinstance(arguments, dict):
        return None
    path = next(
        (
            arguments.get(k)
            for k in ("path", "file_path", "filename")
            if isinstance(arguments.get(k), str)
        ),
        None,
    )
    if not path:
        return None

    def matches(candidate: str) -> bool:
        parts = candidate.replace("\\", "/").lower().split("/")
        name = parts[-1]
        return (
            ".ssh" in parts
            or name == ".env"
            or name.startswith(".env.")
            or "credential" in name
            or "secret" in name
            or "settings" in name
            or name in {"config.toml", "config.json", "id_rsa", "id_ed25519"}
            or name.endswith((".pem", ".key", ".p12", ".pfx"))
        )

    resolved = os.path.realpath(os.path.join(workspace or os.getcwd(), path))
    return redact_worker(path) if matches(path) or matches(resolved) else None


def _git_metadata(workspace: str) -> dict[str, str]:
    result = {}
    for key, arg in (("git_branch", "--abbrev-ref"), ("git_head", "--verify")):
        try:
            process = subprocess.run(
                ["git", "-C", workspace, "rev-parse", arg, "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            result[key] = redact_worker(process.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            return {}
    return result


async def run_worker(
    agent: Agent,
    query: str,
    *,
    title: str,
    parent_run: str | None,
    observe: EventSink | None = None,
    routine: str | None = None,
    inbox: InboxSource | None = None,
) -> RunResult:
    """Record a direct run in the CLI store without changing its options."""
    from localmelo.support.sessions.runner import SessionRunner
    from localmelo.support.sessions.store import (
        NewSession,
        SessionStore,
        title_from_text,
    )

    workspace = agent.workspace_root or os.getcwd()
    provider = agent._llm
    model = str(getattr(provider, "model", ""))
    http = getattr(provider, "_http", None)
    host = urlsplit(str(getattr(http, "base_url", ""))).hostname or ""
    meta: dict[str, Any] = {
        "source": "localmelo",
        "base_url_host": host,
        "parent_run": parent_run,
        "tools": [tool.name for tool in agent.list_tools()],
    }
    if workspace:
        meta.update(await asyncio.to_thread(_git_metadata, workspace))
    meta = redact_worker(meta)
    store = SessionStore(writer="cli")
    runner = SessionRunner(
        agent,
        store,
        None,
        new_session=NewSession(
            model=redact_worker(model),
            workspace=redact_worker(workspace),
            meta=meta,
            kind="worker",
        ),
        progress=observe is not None,
        observe=observe,
        inbox=inbox,
    )
    try:
        outcome = await runner.run_turn(
            query,
            title=title or title_from_text(query.splitlines()[0]),
            routine=routine,
        )
        meta.update(
            {
                "tools": [tool.name for tool in agent.list_tools()],
                "status": outcome.status,
                "exit_code": (
                    0
                    if outcome.status == "completed"
                    else 1 if outcome.status == "crashed" else 2
                ),
                "steps": outcome.steps,
                "usage": outcome.usage,
            }
        )
        if runner.session_id:
            await store.update_session(runner.session_id, meta=redact_worker(meta))
        return outcome
    finally:
        await runner.close()
        await store.aclose()
