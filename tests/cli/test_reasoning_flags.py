"""R2's direct-mode flags, refusals, config table and JSON line."""

from __future__ import annotations

import json
import re
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
        "(choose from 'legacy', 'compress')"
    )


def test_the_agent_table_applies_only_when_the_flag_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path, agent='\n[agent]\nreasoning = "compress"\n')
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    parser, args = _parse(["q"])
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    assert agent_cls.call_args.kwargs["options"].reasoning == "compress"

    parser, args = _parse(["--reasoning", "legacy", "q"])
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    assert agent_cls.call_args.kwargs["options"].reasoning == "legacy"


def test_base_url_ignores_config_reasoning_but_reads_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path, agent='\n[agent]\nreasoning = "compress"\n')
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    parser, args = _parse(["--base-url", "http://127.0.0.1:1/v1", "q"])
    options = cli._direct_options(parser, args)
    with (
        mock.patch(
            "localmelo.support.providers.llm.openai_compat.OpenAICompatLLM"
        ) as provider,
        mock.patch.object(cli, "Agent") as agent_cls,
        mock.patch.object(config, "load", wraps=config.load) as load,
    ):
        cli._build_direct_mode_agent(args, options=options)
    load.assert_called_once()
    assert agent_cls.call_args.kwargs["options"].reasoning == "legacy"

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
    assert agent_cls.call_args.kwargs["llm"].reasoning_field is None


def test_compress_on_a_replay_required_backend_passes_the_section_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    _write_config(
        path,
        extra='reasoning_field = "reasoning_content"\nrequires_reasoning_replay = true\n',
        agent='\n[agent]\nreasoning = "compress"\n',
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
        ["--reasoning", "compress", "--reasoning-field", "reasoning", "q"]
    )
    options = cli._direct_options(parser, args)
    with mock.patch(
        "localmelo.support.backends.openai_compat.OpenAICompatLLM"
    ) as provider:
        cli._build_direct_mode_agent(args, options=options)
    # The flag wins over the section's value for this run.
    assert provider.call_args.kwargs["reasoning_field"] == "reasoning"


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
            "compress",
            "--output-format",
            "json",
            "q",
        ],
    )
    cli.main()
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    record = json.loads(out[0])
    assert record["config"]["reasoning"] == "compress"
    assert record["usage"]["reasoning_chars"] == 5
    assert record["memory_stats"] == {
        "replayed_chars": 0,
        "replay_skipped": 0,
        "replay_stripped": 0,
        "notes": 1,
        "notes_model": 0,
        "notes_heuristic": 1,
        "note_chars": 11,
        "drafts": 0,
        "drafts_written": 0,
        "candidates": 4,
        "remember_model": 0,
        "remember_heuristic": 1,
        "candidates_refused": 0,
        "redactions": 0,
        "rehearsed": {"restated": 0, "acted": 0, "cited": 0},
        "digests": 1,
        "recalls": 0,
        "recalls_by_trigger": {"start": 0, "attempt": 0, "step": 0, "tool": 0},
        "recall_hits": 0,
        "recall_same_task": 0,
        "recall_cross_task": 0,
        "recall_task_notes": 0,
        "recall_chars": 0,
        "recall_ms": 0,
        "recall_fallbacks": 0,
        "recall_skipped": 0,
        "recall_refused": 0,
        "recall_empty": 1,
        "aux_retrieve_calls": 0,
        "retrieve_errors": 0,
        "retrieve_prompt_tokens": 0,
        "retrieve_completion_tokens": 0,
        "thought_chunks": 0,
        "recall_log": [],
        "admitted": {
            "rehearsal": 0,
            "user_request": 0,
            "user_statement": 0,
            "outcome": 0,
            "sleep": 0,
        },
        "written": {
            "semantic": 0,
            "episodic": 0,
            "profile": 0,
            "reflection": 0,
        },
        "ignored": 0,
    }
    assert record["config"]["keep_decision_text"] is False


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
    text = cli._build_parser(developer=True).format_help()
    for key, flag in (
        ("agent.reasoning", "--reasoning"),
        ("endpoint.reasoning_field", "--reasoning-field"),
        ("agent.note_writer", "--note-writer"),
    ):
        assert re.search(rf"^  {key} .* {flag} \(#183\)$", text, re.M)


def test_a_config_refusal_leaves_no_memory_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(path, extra="requires_reasoning_replay = true\n")
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    memory_dir = tmp_path / "memory"
    assert (
        _main(
            ["--memory-dir", str(memory_dir), "--reasoning", "compress", "q"],
            monkeypatch,
        )
        == 2
    )
    assert "replay and compress+replay need a reasoning field" in _error_line(capsys)
    assert not memory_dir.exists()


def test_the_interactive_ui_accepts_the_reasoning_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["--reasoning-field", "auto"])
    cli._check_args(parser, args)
    assert args.reasoning_field == "auto"
    assert capsys.readouterr().err == ""


# ── R4: compress modes and the note writer ──


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
    assert built.reasoning == "compress+replay"

    path.write_text(
        CONFIG.format(backend="vllm", extra="requires_reasoning_replay = true\n")
        + '\n[agent]\nreasoning = "compress"\n',
        encoding="utf-8",
    )
    assert _main(["q"], monkeypatch) == 2
    assert "replay and compress+replay need a reasoning field" in (_error_line(capsys))
