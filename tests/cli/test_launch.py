"""Bare melo's startup: the terminal, the workspace and the backend, before the UI.

The backend choice keeps U2.0's rules (ported from the `melo ""` loop's
tests): model listing, offering to save and the first run are faked, so
nothing reaches a model server.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.support.backends.discovery import DiscoveryError, LoadedModels
from localmelo.support.cli import launch
from localmelo.support.cli.launch import Choice, choose_backend, run_interactive
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.onboard import FirstSetup

URL = "http://127.0.0.1:8500/v1"


def _args(**fields: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "reconfigure": False,
        "base_url": None,
        "chat_model": None,
        "workspace": None,
        "yolo": False,
        "embedding_base_url": None,
        "embedding_model": None,
        "request_timeout": None,
    }
    values.update(fields)
    return argparse.Namespace(**values)


def _configured(model: str = "configured") -> Config:
    return Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(chat_url=URL, chat_model=model),
    )


class Faked:
    """choose_backend with the listing, the offer and the setups faked."""

    def __init__(
        self,
        *,
        cfg: Config | None = None,
        listed: list[str] | Exception | None = None,
        offer: bool = True,
        first: FirstSetup | None | type[BaseException] = None,
        wizard: Config | None = None,
    ) -> None:
        self.listing = mock.AsyncMock(
            side_effect=listed if isinstance(listed, Exception) else None,
            return_value=listed if isinstance(listed, list) else [],
        )
        self.offer = mock.MagicMock(return_value=offer)
        first_kwargs: dict[str, Any] = (
            {"side_effect": first}
            if isinstance(first, type)
            else {"return_value": first}
        )
        self.first = mock.MagicMock(**first_kwargs)
        self.wizard = mock.MagicMock(return_value=wizard)
        self.cfg = cfg or Config()

    def choose(self, loaded: LoadedModels | None = None, **fields: Any) -> Choice | int:
        self.status = mock.AsyncMock(return_value=loaded or LoadedModels())
        with (
            mock.patch(
                "localmelo.support.backends.discovery.loaded_models", self.status
            ),
            mock.patch("localmelo.support.config.load", return_value=self.cfg),
            mock.patch(
                "localmelo.support.backends.discovery.list_models", self.listing
            ),
            mock.patch("localmelo.support.onboard.offer_to_save", self.offer),
            mock.patch("localmelo.support.onboard.run_first_setup", self.first),
            mock.patch("localmelo.support.onboard.run_backend_setup", self.wizard),
        ):
            return choose_backend(_args(**fields))


def _choice(value: Choice | int) -> Choice:
    assert isinstance(value, Choice), value
    return value


# ── --base-url ──


def test_a_base_url_takes_the_only_chat_model_it_lists() -> None:
    faked = Faked(cfg=_configured(), listed=["bge-embed", "Qwen3-8B", "nomic-embed"])
    choice = _choice(faked.choose(base_url=URL))
    faked.listing.assert_awaited_once_with("openai-compatible", URL)
    assert (choice.backend.base_url, choice.backend.model) == (URL, "Qwen3-8B")
    assert (choice.backend.source, choice.backend.listed) == ("--base-url", 3)
    assert choice.cfg is None and choice.checked and not choice.saves
    faked.offer.assert_not_called()  # a config exists


def test_a_base_url_keeps_the_models_it_listed_for_model_n() -> None:
    faked = Faked(cfg=_configured(), listed=["embed-x", "chat-a", "chat-b"])
    loaded = LoadedModels(frozenset({"chat-b"}), True)
    choice = _choice(faked.choose(base_url=URL, loaded=loaded))
    assert choice.models == ("embed-x", "chat-a", "chat-b")
    assert choice.backend.model == "chat-b" and choice.backend.listed == 3


def test_a_base_url_with_a_model_sends_no_listing_when_a_config_exists() -> None:
    faked = Faked(cfg=_configured())
    choice = _choice(faked.choose(base_url=URL, chat_model="M"))
    faked.listing.assert_not_called()
    assert choice.backend.model == "M" and not choice.checked


def test_a_server_that_cannot_list_its_models_gets_the_model_default() -> None:
    faked = Faked(listed=DiscoveryError("connection refused", reachable=False))
    choice = _choice(faked.choose(base_url=URL))
    assert choice.backend.model == "default"
    faked.offer.assert_not_called()
    assert choice.warning == (
        f"Could not list the models at {URL} (connection refused); "
        "asking for the model 'default' (/model picks one)."
    )


def test_a_server_that_lists_no_models_says_so() -> None:
    choice = _choice(Faked(cfg=_configured(), listed=[]).choose(base_url=URL))
    assert "(it lists no models)" in choice.warning
    assert choice.backend.model == "default"


def test_with_no_config_the_listed_model_is_offered_and_saved() -> None:
    faked = Faked(listed=["m1"], offer=True)
    choice = _choice(faked.choose(base_url=URL))
    faked.offer.assert_called_once_with(URL, "m1")
    assert (choice.backend.source, choice.saves) == ("saved", True)


def test_a_base_url_takes_the_one_loaded_chat_model_of_several() -> None:
    faked = Faked(cfg=_configured(), listed=["big", "small"])
    loaded = LoadedModels(frozenset({"small"}), True)
    choice = _choice(faked.choose(base_url=URL, loaded=loaded))
    faked.status.assert_awaited_once_with(URL)
    assert choice.backend.model == "small"


@pytest.mark.parametrize(
    "loaded",
    [
        LoadedModels(),  # no status route, or a 401
        LoadedModels(frozenset(), True, False),  # none in memory
        LoadedModels(frozenset({"big", "small"}), True),  # more than one
    ],
)
def test_a_base_url_with_several_models_and_no_loaded_one_exits_2(
    loaded: LoadedModels, capsys: pytest.CaptureFixture[str]
) -> None:
    faked = Faked(cfg=_configured(), listed=["big", "small"], offer=True)
    assert faked.choose(base_url=URL, loaded=loaded) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"melo: error: {URL} serves several models")
    assert "\n  big" in err and "\n  small" in err
    assert err.endswith("Pass --chat-model <id>.\n")
    faked.offer.assert_not_called()


def test_a_refusal_says_when_the_status_route_needs_a_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    faked = Faked(cfg=_configured(), listed=["big", "small"])
    loaded = LoadedModels(frozenset(), False, True)  # answered 401
    assert faked.choose(base_url=URL, loaded=loaded) == 2
    note = "The server did not say which models are loaded (it needs a key for that)."
    assert capsys.readouterr().err.endswith(
        f"\n  big\n  small\n{note}\nPass --chat-model <id>.\n"
    )


def test_a_named_model_skips_the_loaded_lookup() -> None:
    faked = Faked(cfg=_configured(), listed=["big", "small"])
    assert _choice(faked.choose(base_url=URL, chat_model="big")).backend.model == "big"
    faked.status.assert_not_called()


def test_a_declined_offer_saves_nothing_and_later_switches_do_not_save() -> None:
    choice = _choice(Faked(listed=["m1"], offer=False).choose(base_url=URL))
    assert (choice.backend.source, choice.saves) == ("--base-url", False)


def test_with_no_config_a_named_model_is_offered_when_the_server_lists_it() -> None:
    faked = Faked(listed=["m1", "m2"])
    _choice(faked.choose(base_url=URL, chat_model="m2"))
    faked.listing.assert_awaited_once()
    faked.offer.assert_called_once_with(URL, "m2")


def test_a_model_the_server_does_not_list_is_not_offered() -> None:
    faked = Faked(listed=["m1"])
    choice = _choice(faked.choose(base_url=URL, chat_model="typo"))
    faked.offer.assert_not_called()
    assert choice.warning == ""  # the listing was only for the offer


def test_a_url_without_v1_is_used_with_it_once_the_server_lists_there() -> None:
    faked = Faked(cfg=_configured(), listed=["m"])
    choice = _choice(faked.choose(base_url="http://127.0.0.1:8500/"))
    faked.listing.assert_awaited_once_with(
        "openai-compatible", "http://127.0.0.1:8500/"
    )
    assert choice.backend.base_url == URL


def test_a_url_that_does_not_list_is_used_as_given() -> None:
    faked = Faked(listed=DiscoveryError("HTTP 404", reachable=True, status_code=404))
    choice = _choice(faked.choose(base_url="http://127.0.0.1:8500/api"))
    assert choice.backend.base_url == "http://127.0.0.1:8500/api"


# ── The config, the wizard and the first run ──


def test_a_configured_backend_is_used_without_a_check() -> None:
    faked = Faked(cfg=_configured())
    choice = _choice(faked.choose())
    faked.listing.assert_not_called()
    faked.first.assert_not_called()
    faked.wizard.assert_not_called()
    assert (choice.backend.model, choice.backend.base_url) == ("configured", URL)
    assert choice.backend.source == "config" and not choice.checked


def test_a_chat_model_replaces_the_configured_one_for_the_session() -> None:
    choice = _choice(Faked(cfg=_configured()).choose(chat_model="other"))
    assert choice.backend.model == "other"


def test_an_invalid_config_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    broken = Config(chat_backend="vllm", embedding_backend="bogus")
    assert Faked(cfg=broken).choose() == 1
    assert "embedding" in capsys.readouterr().err


def test_reconfigure_runs_the_wizard_and_cancelling_it_starts_nothing() -> None:
    faked = Faked(cfg=_configured(), wizard=None)
    assert faked.choose(reconfigure=True) == 0
    faked.wizard.assert_called_once_with(guided=True)
    faked = Faked(cfg=_configured(), wizard=_configured("new"))
    choice = _choice(faked.choose(reconfigure=True, chat_model="M"))
    assert choice.backend.model == "M"


def test_reconfigure_in_developer_mode_runs_the_full_wizard() -> None:
    faked = Faked(cfg=_configured(), wizard=None)
    assert faked.choose(reconfigure=True, developer=True) == 0
    faked.wizard.assert_called_once_with(guided=False)


def test_with_no_config_the_first_run_saves_and_the_session_uses_the_config() -> None:
    saved = FirstSetup(url=URL, model="m", saved=True)
    faked = Faked(first=saved)
    faked.cfg = Config()
    with (
        mock.patch(
            "localmelo.support.config.load", side_effect=[Config(), _configured("m")]
        ),
        mock.patch("localmelo.support.onboard.run_first_setup", faked.first),
    ):
        choice = _choice(choose_backend(_args()))
    faked.first.assert_called_once_with()
    assert (choice.backend.model, choice.backend.source) == ("m", "saved")
    assert choice.cfg is not None and choice.checked


def test_a_server_started_anyway_is_used_unsaved_with_the_chat_model_first() -> None:
    unsaved = FirstSetup(url=URL, model="typed", saved=False)
    choice = _choice(Faked(first=unsaved).choose())
    assert (choice.backend.base_url, choice.backend.model) == (URL, "typed")
    assert choice.backend.source == "unsaved" and not choice.checked
    choice = _choice(Faked(first=unsaved).choose(chat_model="M"))
    assert choice.backend.model == "M"


def test_quitting_the_first_run_starts_nothing() -> None:
    assert Faked(first=None).choose() == 0


# ── run_interactive before the UI ──


def _run(monkeypatch: pytest.MonkeyPatch, *, terminal: bool, **fields: Any) -> int:
    monkeypatch.setattr(launch, "ui_terminal", lambda: terminal)
    with mock.patch.object(launch, "main_async") as main:
        code = run_interactive(
            _args(**fields), parser=argparse.ArgumentParser(), options=AgentOptions()
        )
    main.assert_not_called()
    return code


def test_without_a_terminal_the_ui_explains_and_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(monkeypatch, terminal=False) == 2
    assert capsys.readouterr() == (
        "",
        "melo: the interactive UI needs a terminal (stdin and stdout must be TTYs).\n"
        '  Run one task:      melo "your task"\n'
        "  Start the gateway: melo --serve\n",
    )


def test_reconfigure_without_a_terminal_runs_the_wizard_then_exits_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with mock.patch("localmelo.support.onboard.run_backend_setup") as wizard:
        assert _run(monkeypatch, terminal=False, reconfigure=True) == 0
    wizard.assert_called_once_with(guided=True)


@pytest.mark.parametrize(("raised", "code"), [(EOFError, 0), (KeyboardInterrupt, 130)])
def test_leaving_the_wizard_without_a_terminal_starts_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raised: type[BaseException],
    code: int,
) -> None:
    with mock.patch("localmelo.support.onboard.run_backend_setup", side_effect=raised):
        assert _run(monkeypatch, terminal=False, reconfigure=True) == code
    assert capsys.readouterr().out == "\nSetup cancelled.\n"


@pytest.mark.parametrize(
    ("stdin", "stdout"), [(True, True), (False, True), (True, False)]
)
def test_the_ui_needs_both_stdin_and_stdout_on_a_terminal(
    monkeypatch: pytest.MonkeyPatch, stdin: bool, stdout: bool
) -> None:
    class Stream:
        def __init__(self, tty: bool) -> None:
            self.tty = tty

        def isatty(self) -> bool:
            return self.tty

    monkeypatch.setattr("sys.stdin", Stream(stdin))
    monkeypatch.setattr("sys.stdout", Stream(stdout))
    monkeypatch.setenv("TERM", "xterm-256color")
    assert launch.ui_terminal() is (stdin and stdout)


@pytest.mark.parametrize(
    ("term", "expected"), [("xterm-256color", True), ("dumb", False)]
)
def test_the_ui_needs_a_terminal_that_is_not_dumb(
    monkeypatch: pytest.MonkeyPatch, term: str, expected: bool
) -> None:
    class Tty:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", Tty())
    monkeypatch.setattr("sys.stdout", Tty())
    monkeypatch.setenv("TERM", term)
    assert launch.ui_terminal() is expected


def test_a_workspace_that_is_not_a_directory_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = str(tmp_path / "nope")
    assert _run(monkeypatch, terminal=True, workspace=missing) == 2
    assert capsys.readouterr().err == f"melo: --workspace: not a directory: {missing}\n"


@pytest.mark.parametrize(("raised", "code"), [(EOFError, 0), (KeyboardInterrupt, 130)])
def test_leaving_a_setup_question_starts_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raised: type[BaseException],
    code: int,
) -> None:
    monkeypatch.setattr(launch, "choose_backend", mock.MagicMock(side_effect=raised))
    assert _run(monkeypatch, terminal=True) == code
    assert capsys.readouterr().out == "\nSetup cancelled.\n"


def test_the_log_is_private(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import logging

    monkeypatch.setenv("HOME", str(tmp_path))
    root = logging.getLogger()
    saved = list(root.handlers), root.level
    try:
        path = Path(launch.configure_logging())
        logging.getLogger("localmelo.test").info("x" * 600_000)
        logging.getLogger("localmelo.test").info("y" * 600_000)  # rolls over
        logging.getLogger("localmelo.test").info("z")
        for handler in root.handlers:
            handler.flush()
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in saved[0]:
            root.addHandler(handler)
        root.setLevel(saved[1])
    assert path.parent.stat().st_mode & 0o777 == 0o700
    files = sorted(path.parent.glob("cli.log*"))
    assert [file.name for file in files] == ["cli.log", "cli.log.1"]
    assert all(file.stat().st_mode & 0o777 == 0o600 for file in files)


# ── Review fixes: whatever ends the session, the store closes ──


async def test_a_cancel_while_the_agent_starts_closes_the_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    from localmelo.support.cli.agents import AgentFactory, url_backend
    from localmelo.support.sessions import store as store_module

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        Vt100_Output,
        "from_pty",
        classmethod(
            lambda cls, stdout, term=None: cls(
                io.StringIO(), lambda: Size(24, 80), term="xterm"
            )
        ),
    )
    stores: list[store_module.SessionStore] = []

    class Store(store_module.SessionStore):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            stores.append(self)

    monkeypatch.setattr(store_module, "SessionStore", Store)
    starting = asyncio.Event()

    async def build(self: AgentFactory, *args: Any, **kwargs: Any) -> Any:
        starting.set()
        await asyncio.sleep(30)

    monkeypatch.setattr(AgentFactory, "build", build)
    choice = Choice(backend=url_backend(URL, "m", source="config"), cfg=None)
    main = asyncio.ensure_future(
        launch.main_async(
            choice,
            args=_args(),
            options=AgentOptions(),
            workspace=str(tmp_path),
            log_path=str(tmp_path / "cli.log"),
        )
    )
    try:
        await asyncio.wait_for(starting.wait(), 10)
        main.cancel()  # Ctrl+C: asyncio.run cancels the main task
        with pytest.raises(asyncio.CancelledError):
            await main
        [store] = stores
        with pytest.raises(store_module.SessionStoreError, match="closed"):
            await store.list_sessions()
    finally:
        for opened in stores:  # an open store's thread would keep pytest alive
            await opened.aclose()


def test_a_malformed_config_is_one_line_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import tomllib

    monkeypatch.setattr(
        launch,
        "choose_backend",
        mock.MagicMock(side_effect=tomllib.TOMLDecodeError("Invalid value", "x=", 2)),
    )
    assert _run(monkeypatch, terminal=True) == 1
    err = capsys.readouterr().err
    assert err.startswith("melo: cannot read ") and "Invalid value" in err
    assert err.count("\n") == 1


def test_a_log_folder_that_cannot_be_made_is_one_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    choice = Choice(backend=mock.MagicMock(), cfg=None)
    monkeypatch.setattr(launch, "choose_backend", lambda args: choice)
    monkeypatch.setattr(
        launch,
        "configure_logging",
        mock.MagicMock(side_effect=FileExistsError(17, "File exists")),
    )
    assert _run(monkeypatch, terminal=True) == 1
    err = capsys.readouterr().err
    assert err.startswith("melo: cannot open the log in ") and err.count("\n") == 1


def test_a_base_url_with_an_embedding_server_shows_long_term_memory() -> None:
    from localmelo.support.cli.views import memory_text

    faked = Faked(cfg=_configured(), listed=["chat-a"])
    choice = _choice(
        faked.choose(
            base_url=URL,
            chat_model="chat-a",
            embedding_base_url=URL,
            embedding_model="nomic-embed-text",
        )
    )
    assert choice.backend.embedding == "OpenAI-compatible · nomic-embed-text"
    assert (
        memory_text(choice.backend, memory=True)
        == "long-term memory · OpenAI-compatible · nomic-embed-text"
    )
    plain = _choice(Faked(cfg=_configured(), listed=["chat-a"]).choose(base_url=URL))
    assert plain.backend.embedding == ""
    assert (
        memory_text(plain.backend, memory=True)
        == "no long-term memory · --embedding-base-url adds one"
    )


def test_memory_text_names_a_configured_embedding_backend_and_model() -> None:
    from localmelo.support.cli.agents import config_backend
    from localmelo.support.cli.views import memory_text

    cfg = _configured()
    cfg.embedding_backend = "vllm"
    cfg.vllm.embedding_url = URL
    cfg.vllm.embedding_model = "Qwen3-Embedding-0.6B"
    backend = config_backend(cfg, None)
    assert backend.embedding == "vLLM · Qwen3-Embedding-0.6B"
    assert (
        memory_text(backend, memory=True)
        == "long-term memory · vLLM · Qwen3-Embedding-0.6B"
    )


def test_the_embedding_name_uses_the_backends_display_name() -> None:
    from localmelo.support.cli.agents import _embedding_name

    cfg = _configured()
    cfg.embedding_backend = "vllm"
    cfg.vllm.embedding_url = URL
    cfg.vllm.embedding_model = "Qwen3-Embedding-0.6B"
    assert _embedding_name(cfg) == "vLLM · Qwen3-Embedding-0.6B"
    cfg.embedding_backend = "ollama"
    cfg.ollama.embedding_url = URL
    cfg.ollama.embedding_model = "bge-m3"
    assert _embedding_name(cfg) == "Ollama · bge-m3"


async def test_a_new_server_keeps_the_sessions_embedding_server() -> None:
    from localmelo.support.cli.agents import AgentFactory, url_backend

    factory = AgentFactory(
        backend=url_backend(URL, "m", source="--base-url", embedding="E · e"),
        cfg=None,
        options=AgentOptions(),
        workspace="/tmp",
        embedding_base_url=URL,
        embedding_model="e",
    )
    await factory.use_url("http://127.0.0.1:19/v1", "n")
    assert factory.backend.embedding == "E · e"


@pytest.mark.parametrize(
    "content",
    [b"{", '{"mcpServers": {}}'.encode("utf-16")],
    ids=["broken-json", "utf-16"],
)
def test_a_bad_mcp_config_file_is_one_line_and_exit_2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: bytes,
) -> None:
    path = tmp_path / "mcp.json"
    path.write_bytes(content)
    monkeypatch.setattr(
        launch,
        "choose_backend",
        lambda args: launch.Choice(backend=mock.MagicMock(), cfg=None),
    )
    assert _run(monkeypatch, terminal=True, mcp_config=str(path)) == 2
    err = capsys.readouterr().err
    assert err.startswith("melo: error: ") and err.count("\n") == 1
