"""MCP server configuration in the common ``mcpServers`` JSON format.

The file format is the one Claude Code, Cursor and other clients read::

    {
      "mcpServers": {
        "cua": {"command": "cua-driver", "args": ["mcp"], "env": {"K": "V"}},
        "pab": {"command": "python", "args": ["-m", "pab.mcp"], "cwd": "/srv"}
      }
    }

Only stdio servers are supported. localmelo also reads these optional
per-server keys:

- ``timeout``: seconds per tool call (default 120)
- ``startup_timeout``: seconds for start-up and tool listing (default 30)
- ``include_tools``: MCP tool names to expose; others are not registered
- ``disabled``: ``true`` skips the server

Other keys are ignored so files written for other clients still load.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TOOL_TIMEOUT_S = 120.0
DEFAULT_STARTUP_TIMEOUT_S = 30.0

# OpenAI function names allow [A-Za-z0-9_-]; MCP names may also use dots.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")


class McpConfigError(ValueError):
    """An MCP config file is missing or malformed."""


def sanitize_name(name: str) -> str:
    """Replace characters that tool names sent to the model cannot contain."""
    return _UNSAFE_NAME_CHARS.sub("_", name)


@dataclass
class McpServerConfig:
    """One stdio MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S
    include_tools: list[str] | None = None

    @property
    def namespace(self) -> str:
        """Server part of the exposed tool names (``mcp__<namespace>__<tool>``)."""
        return sanitize_name(self.name)


def load_mcp_config(path: str | Path) -> list[McpServerConfig]:
    """Read and validate an MCP config file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise McpConfigError(f"Cannot read MCP config {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"MCP config {path} is not valid JSON: {exc}") from exc
    return parse_mcp_config(data, source=str(path))


def parse_mcp_config(data: Any, source: str = "MCP config") -> list[McpServerConfig]:
    """Validate parsed config data (``mcpServers``, or ``servers`` as an alias)."""
    if not isinstance(data, dict):
        raise McpConfigError(f"{source}: top level must be a JSON object")
    servers = data.get("mcpServers", data.get("servers"))
    if not isinstance(servers, dict):
        raise McpConfigError(f"{source}: expected an object under 'mcpServers'")

    configs: list[McpServerConfig] = []
    owners: dict[str, str] = {}
    for name, raw in servers.items():
        cfg = _parse_server(name, raw, source)
        if cfg is None:
            continue
        other = owners.get(cfg.namespace)
        if other is not None:
            raise McpConfigError(
                f"{source}: servers {other!r} and {name!r} both map to the tool "
                f"prefix mcp__{cfg.namespace}__"
            )
        owners[cfg.namespace] = name
        configs.append(cfg)
    return configs


def _parse_server(name: Any, raw: Any, source: str) -> McpServerConfig | None:
    if not isinstance(name, str) or not name:
        raise McpConfigError(f"{source}: server names must be non-empty strings")
    where = f"{source}: server {name!r}"
    if not isinstance(raw, dict):
        raise McpConfigError(f"{where} must be an object")
    if raw.get("disabled") is True:
        return None

    transport = raw.get("type", raw.get("transport", "stdio"))
    if transport != "stdio":
        raise McpConfigError(
            f"{where}: only stdio servers are supported (got type {transport!r})"
        )
    command = raw.get("command")
    if not isinstance(command, str) or not command:
        hint = " (HTTP/SSE servers are not supported)" if "url" in raw else ""
        raise McpConfigError(f"{where}: 'command' must be a non-empty string{hint}")

    args = raw.get("args", [])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise McpConfigError(f"{where}: 'args' must be a list of strings")
    env = raw.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise McpConfigError(f"{where}: 'env' must map strings to strings")
    cwd = raw.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise McpConfigError(f"{where}: 'cwd' must be a string")
    include = raw.get("include_tools")
    if include is not None and (
        not isinstance(include, list) or not all(isinstance(t, str) for t in include)
    ):
        raise McpConfigError(f"{where}: 'include_tools' must be a list of strings")

    return McpServerConfig(
        name=name,
        command=command,
        args=list(args),
        env=dict(env),
        cwd=cwd,
        timeout_s=_seconds(raw, "timeout", DEFAULT_TOOL_TIMEOUT_S, where),
        startup_timeout_s=_seconds(
            raw, "startup_timeout", DEFAULT_STARTUP_TIMEOUT_S, where
        ),
        include_tools=list(include) if include is not None else None,
    )


def _seconds(raw: dict[str, Any], key: str, default: float, where: str) -> float:
    value = raw.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise McpConfigError(f"{where}: {key!r} must be a positive number of seconds")
    return float(value)
