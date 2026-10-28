"""Recording helpers for one-shot worker sessions."""

from __future__ import annotations

import ast
import asyncio
import os
import shlex
import subprocess
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from localmelo.melo.checker.redaction import redact_secrets

if TYPE_CHECKING:
    from localmelo.melo.agent import Agent
    from localmelo.melo.contracts.events import EventSink
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
    """Identify simple reads of common secret files, including symlink targets.

    This is best-effort detection, not a boundary against arbitrary commands.
    """
    name = tool.lower()
    if name.startswith("mcp__") and "__" in name[5:]:
        name = name.rsplit("__", 1)[-1]
    if not isinstance(arguments, dict):
        return None
    paths: list[str] = []
    if name in {"file_read", "read", "read_file", "read_text_file"}:
        path = next(
            (
                arguments.get(k)
                for k in ("path", "file_path", "filename")
                if isinstance(arguments.get(k), str)
            ),
            None,
        )
        if path:
            paths.append(path)
    elif name == "shell_exec" and isinstance(arguments.get("command"), str):
        paths = _shell_read_paths(arguments["command"])
    elif name == "python_exec" and isinstance(arguments.get("code"), str):
        try:
            tree = ast.parse(arguments["code"])
        except (SyntaxError, ValueError, RecursionError):
            return None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id != "open":
                    continue
                path_node = (
                    node.args[0]
                    if node.args
                    else next(
                        (kw.value for kw in node.keywords if kw.arg == "file"), None
                    )
                )
                if isinstance(path_node, ast.Constant) and isinstance(
                    path_node.value, str
                ):
                    paths.append(path_node.value)
    else:
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

    for path in paths:
        if not path or "\x00" in path:
            continue
        resolved = os.path.realpath(os.path.join(workspace or os.getcwd(), path))
        if matches(path) or matches(resolved):
            return redact_secrets(path)[0]
    return None


def _shell_read_paths(command: str) -> list[str]:
    """Extract operands from a small subset of shell reads, without executing it."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return []
    if not tokens or tokens[0] not in {
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "sed",
        "grep",
    }:
        return []
    # Shell syntax and expansion require an actual shell parser; don't guess.
    if any(any(c in token for c in "|&;<>$`\n") for token in tokens):
        return []
    tool, args = tokens[0], tokens[1:]
    if tool == "sed" and "-n" not in args:
        return []
    operands: list[str] = []
    pattern_supplied = False
    options = True
    i = 0
    while i < len(args):
        arg = args[i]
        if options and arg == "--":
            options = False
        elif options and arg.startswith("-") and arg != "-":
            if arg in {"-e", "-f"} and tool in {"grep", "sed"}:
                pattern_supplied = True
                i += 1
            elif (
                arg in {"-n", "-c"}
                and tool in {"head", "tail"}
                or arg in {"-m", "-A", "-B", "-C"}
                and tool == "grep"
            ):
                i += 1
        else:
            operands.append(arg)
        i += 1
    if tool in {"grep", "sed"} and not pattern_supplied:
        operands = operands[1:]
    return operands


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
