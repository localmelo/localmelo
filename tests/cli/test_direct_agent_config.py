"""Direct mode checks ``[agent]`` once, with the config's own sentences.

The same refusal covers a ``--base-url`` run and a run on the configured backend,
a plain run and a ``--brief`` one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo import __main__ as cli
from localmelo.support import config

BACKEND = (
    'chat_backend = "vllm"\nembedding_backend = "none"\n'
    '[vllm]\nchat_url = "http://127.0.0.1:2/v1"\nchat_model = "m"\n'
)


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    path = tmp_path / "config.toml"
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    return path


def _build(argv: list[str]) -> cli.Agent:
    parser = cli._build_parser()
    args = parser.parse_args(argv)
    options = cli._direct_options(parser, args)
    return cli._build_direct_mode_agent(args, parser=parser, options=options)


@pytest.mark.parametrize("url", [True, False], ids=["base-url", "backend"])
@pytest.mark.parametrize(
    "line",
    [
        'answer_on_stop = "yes"',  # the wrong kind of value
        'window_eviction = "sliding"',  # not one of the choices
        "window_messages = 3",  # out of range
        "max_calls = 1",
        'tool_images = "maybe"',
        # Not checked in a --base-url run before the one check.
        'reasoning = "summarize"',
        'note_writer = "mystery"',
        "replay_turns = 0",
        'tool_env = ["not a name"]',
        "reflection = 2",
    ],
)
def test_a_bad_agent_value_is_refused_with_the_configs_sentence(
    line: str,
    url: bool,
    config_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_file.write_text(f"{BACKEND}[agent]\n{line}\n")
    sentence = config.load().agent_errors()[0]
    assert sentence.startswith("[agent] ")
    argv = [*(["--base-url", "http://127.0.0.1:1/v1"] if url else []), "task"]
    with pytest.raises(SystemExit) as exit_:
        _build(argv)
    assert exit_.value.code == 2
    assert capsys.readouterr().err.splitlines()[-1] == f"melo: error: {sentence}"


def test_a_brief_reports_it_as_a_startup_error(
    config_file: Path, tmp_path: Path
) -> None:
    config_file.write_text(f"{BACKEND}[agent]\nreasoning = 'summarize'\n")
    brief = tmp_path / "brief.md"
    brief.write_text("do it\n")
    with pytest.raises(cli._StartupConfigError) as error:
        _build(["--base-url", "http://127.0.0.1:1/v1", "--brief", str(brief)])
    assert str(error.value) == config.load().agent_errors()[0]


async def test_flags_complete_a_budget_the_file_leaves_partial(
    config_file: Path,
) -> None:
    config_file.write_text(f"{BACKEND}[agent]\ncontext_window = 20000\n")
    flags = ["--output-reserve", "1000", "--safety-margin", "500"]
    agent = _build(
        ["--base-url", "http://127.0.0.1:1/v1", "--chat-model", "m", *flags, "task"]
    )
    try:
        assert agent.options.context_window == 20000
        assert (agent.options.output_reserve, agent.options.safety_margin) == (
            1000,
            500,
        )
    finally:
        await agent.close()
