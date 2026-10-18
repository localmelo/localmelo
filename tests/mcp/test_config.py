"""MCP config file parsing and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.mcp import (
    McpConfigError,
    load_mcp_config,
    parse_mcp_config,
    sanitize_name,
)
from localmelo.support.mcp.config import (
    DEFAULT_STARTUP_TIMEOUT_S,
    DEFAULT_TOOL_TIMEOUT_S,
)


def _one(server: dict[str, Any], name: str = "srv") -> Any:
    (cfg,) = parse_mcp_config({"mcpServers": {name: server}})
    return cfg


class TestParse:
    def test_full_entry(self) -> None:
        cfg = _one(
            {
                "command": "cua-driver",
                "args": ["mcp", "--stdio"],
                "env": {"CUA_TELEMETRY": "0"},
                "cwd": "/tmp",
                "timeout": 300,
                "startup_timeout": 10.5,
                "include_tools": ["screenshot", "click"],
            },
            name="cua",
        )
        assert cfg.name == "cua"
        assert cfg.command == "cua-driver"
        assert cfg.args == ["mcp", "--stdio"]
        assert cfg.env == {"CUA_TELEMETRY": "0"}
        assert cfg.cwd == "/tmp"
        assert cfg.timeout_s == 300.0
        assert cfg.startup_timeout_s == 10.5
        assert cfg.include_tools == ["screenshot", "click"]

    def test_defaults(self) -> None:
        cfg = _one({"command": "pab-mcp"})
        assert cfg.args == []
        assert cfg.env == {}
        assert cfg.cwd is None
        assert cfg.timeout_s == DEFAULT_TOOL_TIMEOUT_S
        assert cfg.startup_timeout_s == DEFAULT_STARTUP_TIMEOUT_S
        assert cfg.include_tools is None

    def test_servers_alias_and_explicit_stdio(self) -> None:
        (cfg,) = parse_mcp_config({"servers": {"a": {"type": "stdio", "command": "x"}}})
        assert cfg.command == "x"

    def test_disabled_servers_are_skipped(self) -> None:
        cfgs = parse_mcp_config(
            {
                "mcpServers": {
                    "on": {"command": "x"},
                    "off": {"command": "y", "disabled": True},
                }
            }
        )
        assert [c.name for c in cfgs] == ["on"]

    def test_unknown_keys_are_ignored(self) -> None:
        assert _one({"command": "x", "autoApprove": ["*"]}).command == "x"

    def test_namespace_is_sanitized(self) -> None:
        assert _one({"command": "x"}, name="macos.apps mcp").namespace == (
            "macos_apps_mcp"
        )
        assert sanitize_name("read.file") == "read_file"

    @pytest.mark.parametrize(
        ("server", "message"),
        [
            ({"url": "http://localhost:3000/mcp"}, "HTTP/SSE"),
            ({"type": "http", "command": "x"}, "only stdio"),
            ({}, "'command'"),
            ({"command": ""}, "'command'"),
            ({"command": "x", "args": "--flag"}, "'args'"),
            ({"command": "x", "env": {"N": 1}}, "'env'"),
            ({"command": "x", "cwd": 3}, "'cwd'"),
            ({"command": "x", "timeout": 0}, "'timeout'"),
            ({"command": "x", "timeout": True}, "'timeout'"),
            ({"command": "x", "startup_timeout": "5"}, "'startup_timeout'"),
            ({"command": "x", "include_tools": "screenshot"}, "'include_tools'"),
        ],
    )
    def test_invalid_entries(self, server: dict[str, Any], message: str) -> None:
        with pytest.raises(McpConfigError, match=message):
            _one(server)

    def test_top_level_shape(self) -> None:
        with pytest.raises(McpConfigError, match="mcpServers"):
            parse_mcp_config({"tools": []})
        with pytest.raises(McpConfigError, match="object"):
            parse_mcp_config([])

    def test_prefix_collision_is_rejected(self) -> None:
        with pytest.raises(McpConfigError, match="mcp__a_b__"):
            parse_mcp_config(
                {"mcpServers": {"a.b": {"command": "x"}, "a_b": {"command": "y"}}}
            )


class TestLoad:
    def test_load_file(self, tmp_path: Path) -> None:
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({"mcpServers": {"pab": {"command": "pab"}}}))
        assert [c.name for c in load_mcp_config(path)] == ["pab"]

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(McpConfigError, match="Cannot read"):
            load_mcp_config(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "mcp.json"
        path.write_text("{not json")
        with pytest.raises(McpConfigError, match="not valid JSON"):
            load_mcp_config(path)
