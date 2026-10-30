"""A direct run whose MCP tool returns a screenshot, against the stub model server.

The bench's plumbing run: ``melo --mcp FILE --mcp-only -o json -- TASK``. The
fake MCP server's ``screenshot`` returns a picture and repeats its base64 in
``structuredContent``. With ``-X agent.tool_images=on`` (or ``auto`` on a
server that says the model reads images) the second request carries the
picture as an ``image_url`` data URL in a user message after the tool result;
off, it carries only the line that names it. The base64 never reaches the
model as text.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.tests.fixtures.stub_llm import StubServer
from localmelo.tests.mcp.fake_mcp_server import PNG_1X1

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX subprocesses")

FAKE_MCP_SERVER = Path(__file__).resolve().parents[1] / "mcp" / "fake_mcp_server.py"
TASK = "take a screenshot"
B64 = base64.b64encode(PNG_1X1).decode()
REPLIES = [
    {
        "tool_calls": [
            {"name": "mcp__fake__screenshot", "arguments": {"structured": True}}
        ]
    },
    {"content": "I see one pixel."},
]
# Sequential: the picture's user message would restart the stub's turn count.
SCENARIOS: dict[str, dict[str, Any]] = {
    "plain": {"replies": REPLIES, "sequential": True},
    "vlm": {
        "omlx": [{"id": "vl", "model_alias": "stub-vl", "model_type": "vlm"}],
        "replies": REPLIES,
        "sequential": True,
    },
}


@pytest.fixture
def stub() -> Iterator[StubServer]:
    server = StubServer(SCENARIOS, default="plain").start()
    try:
        yield server
    finally:
        server.close()


def run(tmp_path: Path, stub: StubServer, scenario: str, *setting: str) -> None:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    config = tmp_path / "mcp.json"
    server = {"command": sys.executable, "args": ["-I", str(FAKE_MCP_SERVER)]}
    config.write_text(json.dumps({"mcpServers": {"fake": server}}), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALMELO_")}
    env.update(HOME=str(home), NO_PROXY="127.0.0.1,localhost")
    url = stub.base_url.removesuffix("/v1") + f"/s/{scenario}/v1"
    model = "stub-vl" if scenario == "vlm" else "stub-model"
    argv = [sys.executable, "-I", "-m", "localmelo", *setting, "--base-url", url]
    argv += ["--chat-model", model, "--mcp-only", "--mcp", str(config), "-o", "json"]
    done = subprocess.run(
        [*argv, "--", TASK],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["status"] == "completed"


def second_request(stub: StubServer) -> list[dict[str, Any]]:
    assert len(stub.requests) == 2
    messages: list[dict[str, Any]] = stub.requests[1]["messages"]
    # The tool message names the picture; its base64 is never text.
    tool = next(m for m in messages if m["role"] == "tool")
    assert tool["content"].startswith("[1 image: sha256:")
    assert all(B64 not in str(m["content"]) for m in messages if m["role"] != "user")
    return messages


def image_parts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        part
        for m in messages
        if isinstance(m["content"], list)
        for part in m["content"]
        if part.get("type") == "image_url"
    ]


@pytest.mark.parametrize(
    ("scenario", "setting"),
    [("plain", ["-X", "agent.tool_images=on"]), ("vlm", [])],
)
def test_the_picture_reaches_the_model(
    tmp_path: Path, stub: StubServer, scenario: str, setting: list[str]
) -> None:
    run(tmp_path, stub, scenario, *setting)
    messages = second_request(stub)
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    label, part = messages[-1]["content"]
    assert label["text"].startswith(
        "Output of tool call call_1 (mcp__fake__screenshot): [1 image: sha256:"
    )
    assert part == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{B64}"},
    }


@pytest.mark.parametrize(
    ("scenario", "setting"),
    [("plain", []), ("vlm", ["-X", "agent.tool_images=off"])],
)
def test_without_the_switch_only_the_name_goes(
    tmp_path: Path, stub: StubServer, scenario: str, setting: list[str]
) -> None:
    run(tmp_path, stub, scenario, *setting)
    messages = second_request(stub)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert image_parts(messages) == []


class TestSetting:
    def test_the_keys_are_written_only_when_changed_and_checked(self) -> None:
        from localmelo.support import config

        assert config.AgentConfig().tool_images == "auto"
        config.save(config.Config())
        assert "tool_images" not in Path(config.config_path()).read_text()
        cfg = config.Config()
        cfg.agent.tool_images, cfg.agent.tool_images_keep = "on", 4
        config.save(cfg)
        text = Path(config.config_path()).read_text()
        assert 'tool_images = "on"' in text and "tool_images_keep = 4" in text
        loaded = config.load().agent
        assert (loaded.tool_images, loaded.tool_images_keep) == ("on", 4)
        cfg.agent.tool_images, cfg.agent.tool_images_retire = "yes", 0
        errors = cfg.validate()
        assert any("tool_images 'yes' is not recognised" in e for e in errors)
        assert any("tool_images_retire must be an int >= 1" in e for e in errors)

    def test_x_sets_it_for_one_run_and_a_session_reads_it(self) -> None:
        import localmelo.__main__ as cli
        from localmelo.support import config
        from localmelo.support.sessions.agents import configured_session_options

        cli._parse(["-X", "agent.tool_images=on", "-u", "http://127.0.0.1:1/v1", "t"])
        assert config.overrides() == {"agent.tool_images": "on"}
        assert configured_session_options(config.load()).tool_images == "on"
        with pytest.raises(SystemExit):
            cli._parse(["-X", "agent.tool_images=maybe", "t"])
