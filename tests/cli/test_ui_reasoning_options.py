"""Reasoning and recall flags reach UI/session consumers, not just argparse."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import AgentOptions
from localmelo.support import config
from localmelo.support.cli import launch
from localmelo.support.cli.agents import AgentFactory, config_backend, url_backend
from localmelo.support.gateway.console.registry import default_builder


def configured() -> config.Config:
    return config.Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=config.LocalBackendConfig(
            chat_url="http://127.0.0.1:9/v1",
            chat_model="stub",
            reasoning_field="reasoning_content",
        ),
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--reasoning-field", "auto"],
        [
            "--reasoning",
            "compress",
            "--retriever",
            "llm",
            "--retrieval-extra-body",
            '{"temperature":0.1}',
        ],
    ],
)
def test_interactive_parser_accepts_the_newly_wired_provider_options(
    argv: list[str],
) -> None:
    parser = cli._build_parser()
    cli._check_args(parser, parser.parse_args(argv))


@pytest.mark.parametrize("override", [False, True])
def test_launch_resolves_config_defaults_and_explicit_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: bool,
) -> None:
    cfg = configured()
    cfg.agent.reasoning = "compress"
    cfg.agent.note_writer = "heuristic"
    cfg.agent.replay_turns = 4
    cfg.agent.tool_env = ["MELO_TEST_TOOL_VAR"]
    cfg.retrieval.retriever = "llm"
    cfg.retrieval.k = 7
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MELO_TEST_TOOL_VAR", "synthetic")
    monkeypatch.chdir(tmp_path)
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "--reasoning",
            "drop",
            "--note-writer",
            "auto",
            "--retriever",
            "bm25",
            "--recall-k",
            "2",
        ]
        if override
        else []
    )
    consumer = mock.AsyncMock(return_value=(0, None))
    with (
        mock.patch.object(launch, "ui_terminal", return_value=True),
        mock.patch.object(
            launch,
            "choose_backend",
            return_value=launch.Choice(config_backend(cfg, None), cfg),
        ),
        mock.patch.object(config, "load", return_value=cfg),
        mock.patch.object(launch, "configure_logging", return_value="test.log"),
        mock.patch.object(launch, "main_async", consumer),
    ):
        assert (
            launch.run_interactive(
                args, parser=parser, options=cli._direct_options(parser, args)
            )
            == 0
        )
    options = consumer.call_args.kwargs["options"]
    assert options.reasoning == ("drop" if override else "compress")
    assert options.note_writer == ("auto" if override else "heuristic")
    assert options.replay_turns == 4
    assert options.retriever == ("bm25" if override else "llm")
    assert options.recall_k == (2 if override else 7)
    assert options.tool_env == {"MELO_TEST_TOOL_VAR": "synthetic"}


@pytest.mark.parametrize("from_config", [False, True])
async def test_factory_forwards_provider_options_on_both_backend_paths(
    tmp_path: Path,
    from_config: bool,
) -> None:
    cfg = configured() if from_config else None
    factory = AgentFactory(
        backend=(
            config_backend(cfg, None)
            if cfg is not None
            else url_backend("http://127.0.0.1:9/v1", "stub", source="--base-url")
        ),
        cfg=cfg,
        options=AgentOptions(reasoning="replay", retriever="llm"),
        workspace=str(tmp_path),
        reasoning_field="auto",
        retrieval_extra_body={"temperature": 0.1},
    )
    build = mock.AsyncMock(return_value=mock.sentinel.agent)
    with mock.patch("localmelo.support.cli.agents.build_session_agent", build):
        assert await factory._build("next-model", None, None) is mock.sentinel.agent
    assert build.call_args.kwargs["reasoning_field"] == "auto"
    assert build.call_args.kwargs["retrieval_extra_body"] == {"temperature": 0.1}


@pytest.mark.parametrize("mode", ["legacy", "drop", "compress"])
async def test_console_uses_configured_mode_but_defaults_to_legacy(
    tmp_path: Path,
    mode: str,
) -> None:
    cfg = configured()
    cfg.agent.reasoning = mode
    cfg.retrieval.retriever = "bm25"
    agent = await default_builder(cfg, str(tmp_path), None, "synthetic-session")
    try:
        assert agent.options.reasoning == mode
        assert agent.options.mode == "gateway"
        assert "shell_exec" not in {tool.name for tool in agent.list_tools()}
        assert agent.options.retriever == ("auto" if mode == "legacy" else "bm25")
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("mode", "retriever", "argv", "error"),
    [
        ("legacy", "auto", ["--recall-k", "2"], "not available in legacy"),
        ("compress", "off", ["--recall-k", "2"], "do not give the --recall-*"),
        (
            "compress",
            "bm25",
            ["--retrieval-extra-body", '{"temperature":0.1}'],
            "it needs --retriever llm",
        ),
        ("compress", "embedding", [], "embedding needs an embedding provider"),
        (
            "compress",
            "embedding",
            ["--memory", "off"],
            "embedding needs --memory on",
        ),
        (
            "legacy",
            "auto",
            ["--reasoning-field", "auto"],
            "ollama backend always reads and sends the thinking field",
        ),
    ],
)
def test_launch_refuses_invalid_effective_options_before_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    retriever: str,
    argv: list[str],
    error: str,
) -> None:
    cfg = configured()
    cfg.agent.reasoning = mode
    cfg.retrieval.retriever = retriever
    if "--memory" in argv:
        cfg.embedding_backend = "vllm"
        cfg.vllm.embedding_model = "synthetic-embedding"
        cfg.vllm.embedding_url = "http://127.0.0.1:9/v1"
    if "--reasoning-field" in argv:
        cfg.chat_backend = "ollama"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    parser = cli._build_parser()
    args = parser.parse_args(argv)
    with (
        mock.patch.object(launch, "ui_terminal", return_value=True),
        mock.patch.object(
            launch,
            "choose_backend",
            return_value=launch.Choice(config_backend(cfg, None), cfg),
        ),
        mock.patch.object(launch, "configure_logging") as logging,
        mock.patch.object(launch, "main_async") as startup,
        pytest.raises(SystemExit) as exc,
    ):
        launch.run_interactive(
            args,
            parser=parser,
            options=cli._direct_options(parser, args),
            validate_options=lambda resolved: cli._check_ui_options(
                parser, args, resolved
            ),
        )
    assert exc.value.code == 2
    assert error in capsys.readouterr().err
    logging.assert_not_called()
    startup.assert_not_called()


async def test_direct_config_exposes_only_explicitly_named_tool_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = configured()
    cfg.agent.tool_env = ["MELO_TEST_TOOL_VAR", "MELO_TEST_MISSING_VAR"]
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MELO_TEST_TOOL_VAR", "synthetic")
    monkeypatch.delenv("MELO_TEST_MISSING_VAR", raising=False)
    monkeypatch.setenv("MELO_TEST_PRIVATE_VAR", "not-inherited")
    parser = cli._build_parser()
    args = parser.parse_args(["a task"])
    with mock.patch.object(config, "load", return_value=cfg):
        agent = cli._build_direct_mode_agent(
            args,
            parser=parser,
            options=cli._direct_options(parser, args),
            tool_providers=[],
        )
    try:
        assert agent.options.tool_env == {"MELO_TEST_TOOL_VAR": "synthetic"}
    finally:
        await agent.close()
