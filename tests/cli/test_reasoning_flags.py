"""R2's direct-mode flags, refusals, config table and JSON line."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support import config
from localmelo.support.config import (
    AgentConfig,
    Config,
    LocalBackendConfig,
    save,
)

CONFIG = """\
chat_backend = "{backend}"
embedding_backend = "none"

[{backend}]
chat_url = "http://127.0.0.1:1234/v1"
chat_model = "stub-model"
embedding_url = ""
embedding_model = ""
{extra}
"""


def _write_config(
    path: Path, backend: str = "vllm", extra: str = "", agent: str = ""
) -> None:
    path.write_text(
        CONFIG.format(backend=backend, extra=extra) + agent, encoding="utf-8"
    )


def _parse(argv: list[str]) -> Any:
    parser = cli._build_parser()
    args = parser.parse_args(argv)
    cli._check_args(parser, args)
    return parser, args


def _main(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(sys, "argv", ["melo", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert isinstance(exc.value.code, int)
    return exc.value.code


def test_replay_with_base_url_needs_a_reasoning_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        cli._check_args(
            parser,
            parser.parse_args(
                ["--base-url", "http://127.0.0.1:1/v1", "--reasoning", "replay", "q"]
            ),
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert (
        "melo: error: argument --reasoning: replay needs a reasoning field: give "
        "--reasoning-field (auto, reasoning_content or reasoning) with --base-url, "
        "or set reasoning_field in the backend's config\n"
    ) in err


def test_the_configured_backend_needs_a_reasoning_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path)
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _main(["--reasoning", "replay", "q"], monkeypatch) == 2
    assert (
        "melo: error: argument --reasoning: replay needs a reasoning field: set "
        "reasoning_field in the chat backend's config section\n"
    ) in capsys.readouterr().err


def test_drop_on_a_replay_required_backend_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(
        path,
        extra='reasoning_field = "auto"\nrequires_reasoning_replay = true\n',
    )
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _main(["--reasoning", "drop", "q"], monkeypatch) == 2
    assert (
        "melo: error: argument --reasoning: this backend requires reasoning "
        "replay, so drop is not possible: use replay\n"
    ) in capsys.readouterr().err


def test_reasoning_field_on_ollama_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path, backend="ollama")
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _main(["--reasoning-field", "auto", "q"], monkeypatch) == 2
    assert (
        "melo: error: argument --reasoning-field: the ollama backend always "
        "reads and sends the thinking field\n"
    ) in capsys.readouterr().err


def test_invalid_reasoning_values_are_argparse_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--reasoning", "compress", "q"])
    assert exc.value.code == 2
    assert (
        "argument --reasoning: invalid choice: 'compress' "
        "(choose from 'legacy', 'drop', 'replay')"
    ) in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--replay-turns", "0", "q"])
    assert exc.value.code == 2
    assert (
        "argument --replay-turns: must be at least 1, got 0" in capsys.readouterr().err
    )


def test_the_agent_table_applies_only_when_the_flag_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    _write_config(
        path,
        extra='reasoning_field = "auto"\n',
        agent='\n[agent]\nreasoning = "replay"\nreplay_turns = 2\n',
    )
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    parser, args = _parse(["q"])
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    built = agent_cls.call_args.kwargs["options"]
    assert (built.reasoning, built.replay_turns) == ("replay", 2)

    parser, args = _parse(["--reasoning", "drop", "--replay-turns", "5", "q"])
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    built = agent_cls.call_args.kwargs["options"]
    assert (built.reasoning, built.replay_turns) == ("drop", 5)


def test_base_url_does_not_read_the_agent_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path, agent='\n[agent]\nreasoning = "replay"\n')
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    parser, args = _parse(["--base-url", "http://127.0.0.1:1/v1", "q"])
    options = cli._direct_options(parser, args)
    with (
        mock.patch("localmelo.support.providers.llm.openai_compat.OpenAICompatLLM"),
        mock.patch.object(config, "load") as load,
    ):
        cli._build_direct_mode_agent(args, options=options)
    load.assert_not_called()


def test_the_agent_table_round_trips_and_is_dropped_at_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.agent = AgentConfig(reasoning="replay", replay_turns=3)
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
        text = path.read_text(encoding="utf-8")
        assert "[agent]" in text
        loaded = config.load()
        assert (loaded.agent.reasoning, loaded.agent.replay_turns) == ("replay", 3)
        loaded.agent = AgentConfig()
        save(loaded)
        assert "[agent]" not in path.read_text(encoding="utf-8")


def test_the_backend_reasoning_keys_round_trip_only_when_set(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    cfg = Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(
            chat_url="http://127.0.0.1:1/v1",
            chat_model="m",
            reasoning_field="reasoning_content",
            requires_reasoning_replay=True,
        ),
    )
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
        text = path.read_text(encoding="utf-8")
        assert 'reasoning_field = "reasoning_content"' in text
        assert "requires_reasoning_replay = true" in text
        loaded = config.load()
        assert loaded.vllm.reasoning_field == "reasoning_content"
        assert loaded.vllm.requires_reasoning_replay is True
        loaded.vllm.reasoning_field = ""
        loaded.vllm.requires_reasoning_replay = False
        save(loaded)
        text = path.read_text(encoding="utf-8")
        assert "reasoning_field" not in text
        assert "requires_reasoning_replay" not in text


def test_an_invalid_agent_value_exits_through_the_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path, agent='\n[agent]\nreasoning = "compress"\n')
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _main(["q"], monkeypatch) == 1
    assert "[agent] reasoning 'compress' is not recognised" in capsys.readouterr().err


class FakeLLM(BaseLLMProvider):
    reasoning_field = "auto"

    def __init__(self) -> None:
        self.last_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cached_tokens": 5,
            "reasoning_tokens": 1,
        }

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        return Message(
            role="assistant",
            content="done",
            thinking="draft",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )

    async def close(self) -> None:
        pass


def _fake_agent(options: AgentOptions | None) -> Agent:
    return Agent(llm=FakeLLM(), options=options)


def test_the_json_line_reports_the_mode_and_counters(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def build(
        args: Any,
        *,
        parser: Any = None,
        options: AgentOptions,
        tool_providers: Any,
    ) -> Agent:
        return _fake_agent(options)

    monkeypatch.setattr(cli, "_build_direct_mode_agent", build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "melo",
            "--base-url",
            "http://x/v1",
            "--reasoning",
            "drop",
            "--output-format",
            "json",
            "q",
        ],
    )
    cli.main()
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    record = json.loads(out[0])
    assert record["config"]["reasoning"] == "drop"
    assert record["usage"]["reasoning_chars"] == 5
    assert record["memory_stats"] == {
        "replayed_chars": 0,
        "replay_skipped": 0,
        "replay_stripped": 0,
    }


def test_the_legacy_json_line_has_none_of_the_new_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def build(
        args: Any,
        *,
        parser: Any = None,
        options: AgentOptions,
        tool_providers: Any,
    ) -> Agent:
        return _fake_agent(options)

    monkeypatch.setattr(cli, "_build_direct_mode_agent", build)
    monkeypatch.setattr(
        sys,
        "argv",
        ["melo", "--base-url", "http://x/v1", "--output-format", "json", "q"],
    )
    cli.main()
    record = json.loads(capsys.readouterr().out.strip())
    assert "memory_stats" not in record
    assert "reasoning" not in record["config"]
    assert "reasoning_chars" not in record["usage"]


def test_the_interactive_help_lists_the_flags() -> None:
    text = cli._build_parser().format_help()
    assert "--reasoning {legacy,drop,replay}" in text
    assert "--reasoning-field {auto,reasoning_content,reasoning}" in text
    assert "--replay-turns N" in text
