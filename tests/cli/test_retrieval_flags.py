"""R5's direct-mode retrieval flags, refusals, config table and extra body."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.melo.contracts.providers import LLM_REQUEST_KIND
from localmelo.melo.schema import Message
from localmelo.support import config
from localmelo.support.config import Config, RetrievalConfig, save

from .test_reasoning_flags import (
    _error_line,
    _main,
    _parse,
    _write_config,
)


def _options(argv: list[str], **kwargs: Any) -> AgentOptions:
    parser, args = _parse(argv)
    return cli._direct_options(parser, args)


def test_the_seven_flags_parse_into_agent_options() -> None:
    options = _options(
        [
            "--reasoning",
            "compress",
            "--retriever",
            "llm",
            "--recall-at",
            "tool,step,step",
            "--recall-k",
            "5",
            "--recall-kinds",
            "long,note",
            "--recall-placement",
            "message",
            "--recall-min-score",
            "0.5",
            "--retrieval-extra-body",
            '{"chat_template_kwargs": {"enable_thinking": false}}',
        ]
    )
    assert options.retriever == "llm"
    assert options.recall_at == frozenset({"step", "tool"})
    assert options.recall_k == 5
    assert options.recall_kinds == frozenset({"note", "long"})
    assert options.recall_placement == "message"
    assert options.recall_min_score == 0.5


def test_the_defaults_are_the_specs() -> None:
    options = _options(["--reasoning", "compress"])
    assert options.retriever == "auto"
    assert options.recall_at is None
    assert options.recall_k == 3
    assert options.recall_kinds is None
    assert options.recall_placement == "tool"
    assert options.recall_min_score == 0.35


def test_an_unknown_trigger_kind_or_range_is_exit_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    for argv, needle in (
        (["--recall-at", "start,bogus", "q"], "unknown recall trigger"),
        (["--recall-kinds", "note,bogus", "q"], "unknown recall kind"),
        (["--recall-k", "0", "q"], "must be an integer from 1 to 10"),
        (["--recall-k", "11", "q"], "must be an integer from 1 to 10"),
        (["--recall-min-score", "1.5", "q"], "must be a number from 0 to 1"),
    ):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(argv)
        assert exc.value.code == 2
        assert needle in capsys.readouterr().err


def test_legacy_refuses_every_retrieval_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        cli._check_args(
            parser,
            parser.parse_args(["--reasoning", "legacy", "--retriever", "bm25", "q"]),
        )
    assert exc.value.code == 2
    assert "not available in legacy" in _error_line(capsys)

    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        cli._check_args(
            parser,
            parser.parse_args(
                [
                    "--reasoning",
                    "compress",
                    "--retriever",
                    "off",
                    "--recall-k",
                    "4",
                    "q",
                ]
            ),
        )
    assert exc.value.code == 2
    assert "do not give --recall-k" in _error_line(capsys)


def test_message_placement_is_refused_in_replay_modes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        cli._check_args(
            parser,
            parser.parse_args(
                [
                    "--reasoning",
                    "replay",
                    "--reasoning-field",
                    "auto",
                    "--recall-placement",
                    "message",
                    "q",
                ]
            ),
        )
    assert exc.value.code == 2
    assert "message is not available in replay modes" in _error_line(capsys)


def test_the_tool_trigger_is_refused_with_mcp_only_or_without_recall(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        cli._check_args(
            parser,
            parser.parse_args(
                [
                    "--reasoning",
                    "compress",
                    "--mcp-config",
                    "x.json",
                    "--mcp-only",
                    "--recall-at",
                    "tool",
                    "q",
                ]
            ),
        )
    assert exc.value.code == 2
    assert "tool is not available with --mcp-only" in _error_line(capsys)

    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        cli._check_args(
            parser,
            parser.parse_args(
                [
                    "--reasoning",
                    "compress",
                    "--builtins",
                    "file_read",
                    "--recall-at",
                    "tool",
                    "q",
                ]
            ),
        )
    assert exc.value.code == 2
    assert "--builtins must include recall" in _error_line(capsys)


def test_an_extra_body_needs_llm(capsys: pytest.CaptureFixture[str]) -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        cli._check_args(
            parser,
            parser.parse_args(
                [
                    "--reasoning",
                    "compress",
                    "--retriever",
                    "bm25",
                    "--retrieval-extra-body",
                    "{}",
                    "q",
                ]
            ),
        )
    assert exc.value.code == 2
    assert "it needs --retriever llm" in _error_line(capsys)


def test_a_base_url_embedding_retriever_needs_a_provider(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser, args = _parse(
        [
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--reasoning",
            "compress",
            "--retriever",
            "embedding",
            "q",
        ]
    )
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent"), pytest.raises(SystemExit) as exc:
        cli._build_direct_mode_agent(args, options=options)
    assert exc.value.code == 2
    assert "embedding needs an embedding provider" in _error_line(capsys)


def test_the_config_table_applies_only_when_the_flag_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    _write_config(
        path,
        agent=(
            '\n[agent]\nreasoning = "compress"\n'
            "\n[retrieval]\n"
            'retriever = "bm25"\n'
            'recall_at = ["step"]\n'
            "k = 4\n"
            'kinds = ["note"]\n'
            'placement = "message"\n'
            "min_score = 0.2\n"
        ),
    )
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    parser, args = _parse(["q"])
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    built = agent_cls.call_args.kwargs["options"]
    assert built.retriever == "bm25"
    assert built.recall_at == frozenset({"step"})
    assert built.recall_k == 4
    assert built.recall_kinds == frozenset({"note"})
    assert built.recall_placement == "message"
    assert built.recall_min_score == 0.2

    parser, args = _parse(
        [
            "--retriever",
            "llm",
            "--recall-k",
            "2",
            "--recall-placement",
            "tool",
            "q",
        ]
    )
    options = cli._direct_options(parser, args)
    with mock.patch.object(cli, "Agent") as agent_cls:
        cli._build_direct_mode_agent(args, options=options)
    built = agent_cls.call_args.kwargs["options"]
    assert (built.retriever, built.recall_k, built.recall_placement) == (
        "llm",
        2,
        "tool",
    )


def test_a_bad_retrieval_value_is_a_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(
        path,
        agent='\n[agent]\nreasoning = "compress"\n\n[retrieval]\nretriever = "bogus"\n',
    )
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _main(["q"], monkeypatch) == 1
    assert "[retrieval] retriever 'bogus' is not recognised" in (
        capsys.readouterr().err
    )


def test_the_table_round_trips_only_when_set(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.retrieval = RetrievalConfig(
        retriever="llm",
        recall_at=["step"],
        k=4,
        kinds=["note", "long"],
        placement="message",
        min_score=0.2,
    )
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
    text = path.read_text()
    assert "[retrieval]" in text
    with mock.patch.object(config, "CONFIG_PATH", str(path)):
        loaded = config.load()
    assert loaded.retrieval == cfg.retrieval

    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(Config(chat_backend="vllm", embedding_backend="none"))
    assert "[retrieval]" not in path.read_text()


def test_the_interactive_ui_accepts_the_flags() -> None:
    parser, args = _parse(
        ["--reasoning", "compress", "--retriever", "bm25", "--recall-k", "2"]
    )
    assert args.retriever == "bm25"
    assert args.recall_k == 2


@pytest.mark.asyncio
async def test_the_extra_body_reaches_retrieve_requests_only() -> None:
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "a"}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    llm = OpenAICompatLLM(
        base_url="http://stub.test/v1",
        model="m",
        transport=httpx.MockTransport(handler),
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        extra_body_by_kind={
            "retrieve": {"chat_template_kwargs": {"enable_thinking": False}}
        },
    )
    try:
        for kind in ("retrieve", "plan"):
            token = LLM_REQUEST_KIND.set(kind)
            try:
                await llm.chat([Message(role="user", content="q")])
            finally:
                LLM_REQUEST_KIND.reset(token)
    finally:
        await llm.close()
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert bodies[1]["chat_template_kwargs"] == {"enable_thinking": True}


def test_the_extra_body_flag_reaches_the_provider_by_kind() -> None:
    parser, args = _parse(
        [
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--reasoning",
            "compress",
            "--retriever",
            "llm",
            "--retrieval-extra-body",
            '{"chat_template_kwargs": {"enable_thinking": false}}',
            "q",
        ]
    )
    options = cli._direct_options(parser, args)
    with (
        mock.patch(
            "localmelo.support.providers.llm.openai_compat.OpenAICompatLLM"
        ) as provider,
        mock.patch.object(cli, "Agent"),
    ):
        cli._build_direct_mode_agent(args, options=options)
    assert provider.call_args.kwargs["extra_body_by_kind"] == {
        "retrieve": {"chat_template_kwargs": {"enable_thinking": False}}
    }


def test_a_config_table_message_placement_is_refused_in_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(
        path,
        extra='reasoning_field = "auto"\n',
        agent=(
            '\n[agent]\nreasoning = "replay"\n' '\n[retrieval]\nplacement = "message"\n'
        ),
    )
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _main(["q"], monkeypatch) == 2
    assert "message is not available in replay modes" in _error_line(capsys)


def test_a_config_retriever_off_refuses_a_recall_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.toml"
    _write_config(
        path,
        agent=(
            '\n[agent]\nreasoning = "compress"\n' '\n[retrieval]\nretriever = "off"\n'
        ),
    )
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    assert _main(["--recall-k", "4", "q"], monkeypatch) == 2
    assert "do not give the --recall-* flags" in _error_line(capsys)
