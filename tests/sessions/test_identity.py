"""What a chat tells its model about itself: melo, the model, the server, the place.

The text is built from the backend an Agent is built on (``config_identity`` for
the config file's backend, ``url_identity`` for a URL), so a model switch, which
builds a new Agent, names the new model. Nothing here reaches a server: the
console's builder and the terminal's factory build Agents on addresses that
nothing listens on, and ``start()`` does not connect.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.support.backends import discovery
from localmelo.support.cli.agents import AgentFactory, config_backend, url_backend
from localmelo.support.config import CloudVendorConfig, Config, LocalBackendConfig
from localmelo.support.gateway.console.registry import default_builder
from localmelo.support.sessions.agents import build_session_agent
from localmelo.support.sessions.identity import (
    config_identity,
    host_of,
    identity_text,
    url_identity,
)

URL = "http://127.0.0.1:9/v1"
MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"
WHO = "You are melo, the assistant built into localmelo"


def local(url: str = URL, model: str = MODEL, **fields: str) -> Config:
    cfg = Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(chat_url=url, chat_model=model),
    )
    for name, value in fields.items():
        setattr(cfg, name, value)
    return cfg


def line(text: str, start: str) -> str:
    return next(row for row in text.split("\n") if row.startswith(start))


def test_the_text_says_who_melo_is_and_how_to_answer() -> None:
    text = config_identity(local())
    assert text.startswith(WHO + ": an open-source agent that runs on the user's own")
    assert "works in their folder with tools" in text
    assert "asks before it changes anything" in text
    assert "say that you are melo from localmelo and name the model you run on" in text
    assert "never present yourself as the model's own assistant" in text
    assert "or as made by its maker" in text
    assert "Answer in the user's language." in text
    assert text.endswith(
        "For a greeting or a question you can answer directly, reply without using tools."
    )
    assert len(text) < 900


@pytest.mark.parametrize(
    ("cfg", "server", "where"),
    [
        (local(chat_alias="omlx"), "oMLX", " on this computer"),
        (local(), "an OpenAI-compatible server", " on this computer"),
        (
            local("http://localhost:11434"),
            "an OpenAI-compatible server",
            " on this computer",
        ),
        (
            local("http://192.168.1.20:8000/v1"),
            "an OpenAI-compatible server",
            " at 192.168.1.20",
        ),
        (
            local("http://user:secret@studio.example:8500/v1", chat_alias="omlx"),
            "oMLX",
            " at studio.example",
        ),
    ],
)
def test_a_local_backend_names_its_server_and_where_it_is(
    cfg: Config, server: str, where: str
) -> None:
    text = config_identity(cfg)
    assert line(text, "You are running on") == (
        f"You are running on the model {MODEL}, served by {server}{where}."
    )
    assert "secret" not in text and "8500" not in text and "8000" not in text


def test_ollama_and_the_other_local_servers_are_named_as_they_are_labelled() -> None:
    cfg = Config(
        chat_backend="ollama",
        embedding_backend="none",
        ollama=LocalBackendConfig(
            chat_url="http://127.0.0.1:11434", chat_model="qwen3"
        ),
    )
    assert line(config_identity(cfg), "You are running on") == (
        "You are running on the model qwen3, served by Ollama on this computer."
    )
    cfg = Config(
        chat_backend="sglang",
        embedding_backend="none",
        sglang=LocalBackendConfig(chat_url="http://10.0.0.5:30000/v1", chat_model="m"),
    )
    assert line(config_identity(cfg), "You are running on") == (
        "You are running on the model m, served by SGLang at 10.0.0.5."
    )


@pytest.mark.parametrize(
    ("backend", "label"),
    [
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"),
        ("gemini", "Gemini"),
        ("nvidia", "NVIDIA"),
    ],
)
def test_a_cloud_provider_has_no_place(backend: str, label: str) -> None:
    cfg = Config(chat_backend=backend, embedding_backend="none")
    setattr(
        cfg, backend, CloudVendorConfig(chat_model="big-model", base_url="http://x/v1")
    )
    text = config_identity(cfg)
    assert line(text, "You are running on") == (
        f"You are running on the model big-model, served by {label}."
    )
    assert "on this computer" not in line(text, "You are running on")
    assert "http://x" not in text


def test_a_url_session_is_a_local_server_and_omlx_when_a_listing_said_so() -> None:
    assert line(url_identity(URL, MODEL), "You are running on") == (
        f"You are running on the model {MODEL}, served by "
        "an OpenAI-compatible server on this computer."
    )
    discovery._OMLX_URLS.add(discovery.normalize_url(URL))
    assert "served by oMLX on this computer." in url_identity(URL, MODEL)
    assert "served by an OpenAI-compatible server at gpu.lan." in url_identity(
        "http://gpu.lan:8000/v1", MODEL
    )


@pytest.mark.parametrize("name", [None, "", "default"])
def test_an_unnamed_model_names_the_server_only(name: str | None) -> None:
    text = url_identity(URL, name)
    assert line(text, "You are running on") == (
        "You are running on a model served by "
        "an OpenAI-compatible server on this computer."
    )
    assert "name the server that serves you" in text
    assert "name the model you run on" not in text


def test_a_name_is_one_short_line() -> None:
    text = identity_text("odd\nname  \t" + "x" * 500, "S\nrv", local=True)
    assert text.count("\n") == 3  # four lines
    assert "the model odd name x" in text
    assert len(line(text, "You are running on")) < 200
    assert host_of("http://u:p@Example.org:99/x") == "example.org"
    assert host_of("not a url") == "" and host_of("http://[bad") == ""


async def test_the_consoles_agent_carries_the_identity_in_every_planning_request(
    tmp_path: Path,
) -> None:
    cfg = local(chat_alias="omlx")
    agent = await default_builder(cfg, str(tmp_path), None, "synthetic-session")
    try:
        assert agent.options.identity == config_identity(cfg)
        assert agent.chat.identity == agent.options.identity
    finally:
        await agent.close()


async def test_a_session_builder_without_identify_leaves_the_options_alone(
    tmp_path: Path,
) -> None:
    for identify in (False, True):
        agent = await build_session_agent(
            cfg=local(),
            base_url=None,
            model=None,
            workspace=str(tmp_path),
            options=AgentOptions(),
            identify=identify,
        )
        try:
            assert (agent.options.identity is not None) is identify
            assert (agent.chat.identity != "") is identify
        finally:
            await agent.close()


async def test_an_identity_the_caller_set_is_kept(tmp_path: Path) -> None:
    agent = await build_session_agent(
        cfg=local(),
        base_url=None,
        model=None,
        workspace=str(tmp_path),
        options=AgentOptions(identity="You are Test."),
        identify=True,
    )
    try:
        assert agent.chat.identity == "You are Test."
    finally:
        await agent.close()


@pytest.mark.parametrize("from_config", [True, False])
async def test_the_terminals_factory_names_the_model_it_builds_on(
    tmp_path: Path, from_config: bool
) -> None:
    cfg = local() if from_config else None
    factory = AgentFactory(
        backend=(
            config_backend(cfg, None)
            if cfg is not None
            else url_backend(URL, MODEL, source="--base-url")
        ),
        cfg=cfg,
        options=AgentOptions(),
        workspace=str(tmp_path),
        saves=False,
    )
    first = await factory.build()
    second = await factory.build("Other-Model-4bit")
    try:
        assert f"the model {MODEL}," in first.chat.identity
        assert "Other-Model-4bit" not in first.chat.identity
        # /model builds a new Agent: the next turn names the new model.
        assert "the model Other-Model-4bit," in second.chat.identity
        assert MODEL not in second.chat.identity
        assert first.chat.identity.replace(MODEL, "X") == second.chat.identity.replace(
            "Other-Model-4bit", "X"
        )
    finally:
        await factory.close(first)
        await factory.close(second)


def test_the_option_must_be_a_text() -> None:
    assert AgentOptions().identity is None
    for bad in ("", "  \n", 3):
        with pytest.raises(ValueError, match="identity"):
            AgentOptions(identity=bad)  # type: ignore[arg-type]
