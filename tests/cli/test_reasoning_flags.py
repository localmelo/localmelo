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


def _error_line(capsys: pytest.CaptureFixture[str]) -> str:
    """The single ``melo: error:`` line argparse printed."""
    lines = capsys.readouterr().err.splitlines()
    errors = [line for line in lines if line.startswith("melo: error:")]
    assert len(errors) == 1
    assert lines[-1] == errors[0]
    return errors[0]


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
    assert _error_line(capsys) == (
        "melo: error: argument --reasoning: replay and compress+replay need a "
        "reasoning field: give --reasoning-field (auto, reasoning_content or "
        "reasoning) with --base-url, or set reasoning_field in the backend's "
        "config"
    )


def test_the_configured_backend_needs_a_reasoning_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path)
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _main(["--reasoning", "replay", "q"], monkeypatch) == 2
    assert _error_line(capsys) == (
        "melo: error: argument --reasoning: replay and compress+replay need a "
        "reasoning field: set reasoning_field in the chat backend's config section"
    )


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
    assert _error_line(capsys) == (
        "melo: error: argument --reasoning: this backend requires reasoning "
        "replay, so drop is not possible: use replay or compress+replay"
    )


def test_reasoning_field_on_ollama_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path, backend="ollama")
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _main(["--reasoning-field", "auto", "q"], monkeypatch) == 2
    assert _error_line(capsys) == (
        "melo: error: argument --reasoning-field: the ollama backend always "
        "reads and sends the thinking field"
    )


def test_invalid_reasoning_values_are_argparse_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--reasoning", "summarize", "q"])
    assert exc.value.code == 2
    assert _error_line(capsys) == (
        "melo: error: argument --reasoning: invalid choice: 'summarize' "
        "(choose from 'legacy', 'drop', 'replay', 'compress', "
        "'compress+replay')"
    )

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
        mock.patch(
            "localmelo.support.providers.llm.openai_compat.OpenAICompatLLM"
        ) as provider,
        mock.patch.object(cli, "Agent") as agent_cls,
        mock.patch.object(config, "load") as load,
    ):
        cli._build_direct_mode_agent(args, options=options)
    load.assert_not_called()
    assert agent_cls.call_args.kwargs["options"].reasoning == "legacy"

    parser, args = _parse(
        [
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--reasoning",
            "replay",
            "--reasoning-field",
            "auto",
            "q",
        ]
    )
    options = cli._direct_options(parser, args)
    with (
        mock.patch(
            "localmelo.support.providers.llm.openai_compat.OpenAICompatLLM"
        ) as provider,
        mock.patch.object(cli, "Agent") as agent_cls,
    ):
        cli._build_direct_mode_agent(args, options=options)
    assert provider.call_args.kwargs["reasoning_field"] == "auto"
    assert agent_cls.call_args.kwargs["options"].reasoning == "replay"

    parser, args = _parse(
        ["--base-url", "http://127.0.0.1:1/v1", "--reasoning-field", "auto", "q"]
    )
    options = cli._direct_options(parser, args)
    with (
        mock.patch(
            "localmelo.support.providers.llm.openai_compat.OpenAICompatLLM"
        ) as provider,
        mock.patch.object(cli, "Agent"),
    ):
        cli._build_direct_mode_agent(args, options=options)
    assert "reasoning_field" not in provider.call_args.kwargs
    assert options.reasoning == "legacy"


def test_a_config_reasoning_field_is_inert_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path, extra='reasoning_field = "auto"\n')
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    parser, args = _parse(["q"])
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    built_cfg = agent_cls.call_args.kwargs["config"]
    assert built_cfg.vllm.reasoning_field == ""


def test_replay_passes_the_section_field_to_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    _write_config(
        path,
        extra='reasoning_field = "reasoning_content"\n',
        agent='\n[agent]\nreasoning = "replay"\n',
    )
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    parser, args = _parse(["q"])
    options = cli._direct_options(parser, args)
    with mock.patch(
        "localmelo.support.backends.openai_compat.OpenAICompatLLM"
    ) as provider:
        cli._build_direct_mode_agent(args, options=options)
    # The table's mode applies (legacy would have cleared the section's key)
    # and the section's key reaches the builder.
    assert provider.call_args.kwargs["reasoning_field"] == "reasoning_content"

    parser, args = _parse(
        ["--reasoning", "replay", "--reasoning-field", "reasoning", "q"]
    )
    options = cli._direct_options(parser, args)
    with mock.patch(
        "localmelo.support.backends.openai_compat.OpenAICompatLLM"
    ) as provider:
        cli._build_direct_mode_agent(args, options=options)
    # The flag wins over the section's value for this run.
    assert provider.call_args.kwargs["reasoning_field"] == "reasoning"


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
        assert text.index("[gateway]") < text.index("[agent]")
        loaded = config.load()
        assert (loaded.agent.reasoning, loaded.agent.replay_turns) == ("replay", 3)
        loaded.agent = AgentConfig(replay_turns=3)
        save(loaded)
        text = path.read_text(encoding="utf-8")
        assert 'reasoning = "legacy"' in text
        assert "replay_turns = 3" in text
        loaded.agent = AgentConfig(reasoning="replay")
        save(loaded)
        text = path.read_text(encoding="utf-8")
        assert 'reasoning = "replay"' in text
        # TOML has no null: the cap is written only when it is set.
        assert "replay_turns" not in text
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


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        (
            'reasoning = "summarize"',
            "[agent] reasoning 'summarize' is not recognised",
        ),
        ("replay_turns = 0", "[agent] replay_turns must be an int >= 1"),
        ('replay_turns = "three"', "[agent] replay_turns must be an int >= 1"),
    ],
)
def test_an_invalid_agent_value_exits_through_the_config_path(
    argument: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path, agent=f"\n[agent]\n{argument}\n")
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _main(["q"], monkeypatch) == 1
    assert message in capsys.readouterr().err


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
        "notes": 0,
        "notes_model": 0,
        "notes_heuristic": 0,
        "note_chars": 0,
        "drafts": 0,
        "drafts_written": 0,
        "candidates": 0,
        "remember_model": 0,
        "remember_heuristic": 0,
        "candidates_refused": 0,
        "redactions": 0,
        "rehearsed": {"restated": 0, "acted": 0, "cited": 0},
        "digests": 0,
        "recalls": 0,
        "recalls_by_trigger": {"start": 0, "attempt": 0, "step": 0, "tool": 0},
        "recall_hits": 0,
        "recall_chars": 0,
        "recall_ms": 0,
        "recall_fallbacks": 0,
        "recall_skipped": 0,
        "recall_refused": 0,
        "recall_empty": 0,
        "aux_retrieve_calls": 0,
        "retrieve_errors": 0,
        "retrieve_prompt_tokens": 0,
        "retrieve_completion_tokens": 0,
        "thought_chunks": 0,
        "recall_log": [],
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
    assert "--reasoning {legacy,drop,replay,compress,compress+replay}" in text
    assert "--reasoning-field {auto,reasoning_content,reasoning}" in text
    assert "--replay-turns N" in text
    assert "--note-writer {auto,heuristic}" in text


def test_a_config_refusal_leaves_no_memory_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path)
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    memory_dir = tmp_path / "memory"
    assert (
        _main(
            ["--memory-dir", str(memory_dir), "--reasoning", "replay", "q"],
            monkeypatch,
        )
        == 2
    )
    assert "replay and compress+replay need a reasoning field" in _error_line(capsys)
    assert not memory_dir.exists()


def test_a_serve_conflict_wins_over_the_reasoning_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        cli._check_args(
            parser,
            parser.parse_args(
                [
                    "--serve",
                    "--base-url",
                    "http://127.0.0.1:1/v1",
                    "--reasoning",
                    "replay",
                ]
            ),
        )
    assert exc.value.code == 2
    assert "are for a task or the interactive UI" in _error_line(capsys)


def test_the_interactive_ui_refuses_the_reasoning_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        cli._check_args(parser, parser.parse_args(["--reasoning-field", "auto"]))
    assert exc.value.code == 2
    assert _error_line(capsys) == (
        "melo: error: --reasoning-field is not in the interactive UI yet; "
        'give it with a task (melo -- "<task>")'
    )


# ── R4: compress modes and the note writer ──


def test_compress_replay_with_base_url_needs_a_reasoning_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        cli._check_args(
            parser,
            parser.parse_args(
                [
                    "--base-url",
                    "http://127.0.0.1:1/v1",
                    "--reasoning",
                    "compress+replay",
                    "q",
                ]
            ),
        )
    assert exc.value.code == 2
    assert _error_line(capsys) == (
        "melo: error: argument --reasoning: replay and compress+replay need a "
        "reasoning field: give --reasoning-field (auto, reasoning_content or "
        "reasoning) with --base-url, or set reasoning_field in the backend's "
        "config"
    )


def test_compress_with_base_url_needs_no_reasoning_field() -> None:
    parser, args = _parse(
        ["--base-url", "http://127.0.0.1:1/v1", "--reasoning", "compress", "q"]
    )
    options = cli._direct_options(parser, args)
    assert options.reasoning == "compress"
    assert options.note_writer == "auto"


def test_the_note_writer_flag_reaches_the_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser, args = _parse(
        [
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--reasoning",
            "compress",
            "--note-writer",
            "heuristic",
            "q",
        ]
    )
    options = cli._direct_options(parser, args)
    assert (options.reasoning, options.note_writer) == ("compress", "heuristic")


def test_the_note_writer_is_accepted_in_the_interactive_ui() -> None:
    parser, args = _parse(["--note-writer", "heuristic"])
    assert args.note_writer == "heuristic"


def test_the_agent_table_passes_the_note_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    _write_config(
        path,
        agent='\n[agent]\nreasoning = "compress"\nnote_writer = "heuristic"\n',
    )
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    parser, args = _parse(["q"])
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    built = agent_cls.call_args.kwargs["options"]
    assert (built.reasoning, built.note_writer) == ("compress", "heuristic")

    parser, args = _parse(["--note-writer", "auto", "q"])
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    assert agent_cls.call_args.kwargs["options"].note_writer == "auto"


def test_an_unknown_note_writer_is_a_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path, agent='\n[agent]\nnote_writer = "mystery"\n')
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _main(["q"], monkeypatch) == 1
    assert "[agent] note_writer 'mystery' is not recognised" in (
        capsys.readouterr().err
    )


def test_the_note_writer_round_trips_only_when_set(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.agent = AgentConfig(reasoning="compress", note_writer="heuristic")
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
        text = path.read_text(encoding="utf-8")
        assert 'reasoning = "compress"' in text
        assert 'note_writer = "heuristic"' in text
        loaded = config.load()
        assert (loaded.agent.reasoning, loaded.agent.note_writer) == (
            "compress",
            "heuristic",
        )
        loaded.agent = AgentConfig()
        save(loaded)
        assert "[agent]" not in path.read_text(encoding="utf-8")


def test_the_compress_json_line_has_its_config_and_counters(
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
            "compress",
            "--note-writer",
            "heuristic",
            "--output-format",
            "json",
            "q",
        ],
    )
    cli.main()
    record = json.loads(capsys.readouterr().out.strip())
    assert record["config"]["reasoning"] == "compress"
    assert record["config"]["note_writer"] == "heuristic"
    assert record["memory_stats"]["notes"] == 1
    assert record["memory_stats"]["note_chars"] == len("found: done")
    assert record["memory_stats"]["candidates_refused"] == 0
    assert record["memory_stats"]["digests"] == 1


def test_the_note_writer_is_inert_in_legacy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def build(
        args: Any,
        *,
        parser: Any = None,
        options: AgentOptions,
        tool_providers: Any,
    ) -> Agent:
        assert options.note_writer == "heuristic"
        return _fake_agent(options)

    monkeypatch.setattr(cli, "_build_direct_mode_agent", build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "melo",
            "--base-url",
            "http://x/v1",
            "--note-writer",
            "heuristic",
            "--output-format",
            "json",
            "q",
        ],
    )
    cli.main()
    record = json.loads(capsys.readouterr().out.strip())
    assert "note_writer" not in record["config"]
    assert "memory_stats" not in record


def test_compress_on_a_replay_required_backend_becomes_compress_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(
        path,
        extra='reasoning_field = "auto"\nrequires_reasoning_replay = true\n',
        agent='\n[agent]\nreasoning = "compress"\n',
    )
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    parser, args = _parse(["q"])
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    built = agent_cls.call_args.kwargs["options"]
    assert (built.reasoning, built.replay_turns) == ("compress+replay", None)

    path.write_text(
        CONFIG.format(backend="vllm", extra="requires_reasoning_replay = true\n")
        + '\n[agent]\nreasoning = "compress"\n',
        encoding="utf-8",
    )
    assert _main(["q"], monkeypatch) == 2
    assert "replay and compress+replay need a reasoning field" in (_error_line(capsys))
