"""The first-run questions and the setup wizard's defaults and retries.

Answers are scripted through input(); model listing and the backend checks
are faked, so nothing reaches a model server. The config file lives in a
temporary directory.
"""

from __future__ import annotations

import asyncio
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import pytest

import localmelo.support.onboard as onboard
from localmelo.melo.contracts.providers import ProviderError
from localmelo.support import config
from localmelo.support.backends import discovery
from localmelo.support.backends.discovery import DiscoveryError, FoundServer
from localmelo.support.config import Config, LocalBackendConfig
from localmelo.support.onboard import FirstSetup, run_backend_setup, run_first_setup
from localmelo.support.onboard_probe import ProbeError

URL = "http://127.0.0.1:8500/v1"

# The real function, kept for tests that answer over an httpx.MockTransport.
_REAL_IS_OLLAMA = discovery.is_ollama
_REAL_LOADED_MODELS = discovery.loaded_models
_REAL_LIST_MODELS = discovery.list_models
_REAL_DETECT = discovery.detect_local_server


@pytest.fixture(autouse=True)
def _server_is_not_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving asks whether the server is Ollama; these fake servers are not."""

    async def is_ollama(url: str, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(discovery, "is_ollama", is_ollama)


@pytest.fixture(autouse=True)
def _nothing_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first run asks which models are in memory; these servers do not say."""

    async def loaded_models(url: str, **kwargs: object) -> discovery.LoadedModels:
        return discovery.LoadedModels()

    monkeypatch.setattr(discovery, "loaded_models", loaded_models)


@pytest.fixture(autouse=True)
def _no_models_for_the_wizard(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wizard lists nothing unless a test overrides this (the first run fakes it)."""

    async def list_models(key: str, url: str, **kwargs: object) -> list[str]:
        return []

    monkeypatch.setattr(discovery, "list_models", list_models)


@pytest.fixture
def config_path(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "config.toml"
    with (
        mock.patch("localmelo.support.config.CONFIG_PATH", str(path)),
        mock.patch("localmelo.support.config.CONFIG_DIR", str(tmp_path)),
    ):
        yield path


def _answered_by(transport: httpx.MockTransport) -> Any:
    """discovery.is_ollama over *transport*: the real question, no network."""

    async def is_ollama(url: str, **kwargs: object) -> bool:
        return await _REAL_IS_OLLAMA(url, transport=transport)

    return is_ollama


class _Answers:
    """input() that answers from a script and records each prompt."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"no answer left for {prompt!r}")
        return self.answers.pop(0)


def _servers(
    found: FoundServer | None = None,
    lists: dict[str, list[str] | Exception] | None = None,
) -> tuple[mock.AsyncMock, mock.AsyncMock]:
    """Fakes of detect_local_server and list_models."""
    lists = lists or {}

    async def list_models(key: str, url: str, **kwargs: object) -> list[str]:
        assert key == "openai-compatible"
        result = lists[url]
        if isinstance(result, Exception):
            raise result
        return result

    return (
        mock.AsyncMock(return_value=found),
        mock.AsyncMock(side_effect=list_models),
    )


def _first_setup(
    answers: _Answers,
    *,
    found: FoundServer | None = None,
    lists: dict[str, list[str] | Exception] | None = None,
    urls: tuple[str, ...] = ("http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1"),
) -> tuple[FirstSetup | None, mock.AsyncMock, mock.AsyncMock]:
    detect, list_models = _servers(found, lists)
    with (
        mock.patch.object(discovery, "detect_urls", return_value=urls),
        mock.patch.object(discovery, "detect_local_server", detect),
        mock.patch.object(discovery, "list_models", list_models),
        mock.patch("builtins.input", answers),
    ):
        result = run_first_setup()
    return result, detect, list_models


# ── First run ──


def test_a_found_server_prefills_the_url_and_its_models_are_offered(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(onboard.sys, "platform", "darwin")
    models = ["bge-m3-embed", "Qwen3-8B"]
    answers = _Answers("", "", "n")
    result, detect, list_models = _first_setup(
        answers, found=FoundServer(url=URL, models=models)
    )
    assert result == FirstSetup(url=URL, model="Qwen3-8B", saved=True)
    assert answers.prompts == [
        f"Server URL [{URL}]: ",
        "Model [1]: ",
        "Use bge-m3-embed for long-term memory? [Y/n]: ",
    ]
    detect.assert_awaited_once_with(
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    )
    list_models.assert_not_called()  # the found server's list is used
    assert capsys.readouterr().out == (
        "melo uses a model server on this computer (oMLX, Ollama, LM Studio…).\n"
        f"Found a model server at {URL} (2 models).\n"
        f"Models on {URL}:\n"
        "  [1] Qwen3-8B\n"
        "  [2] bge-m3-embed  (embedding)\n"
        f"✓ {URL} lists 2 models; using Qwen3-8B.\n"
        "Long-term memory off (melo setup sets it again).\n"
        f"Saved to {config_path} (melo setup changes it).\n"
    )
    cfg = config.load()
    assert (cfg.chat_backend, cfg.embedding_backend) == ("vllm", "none")
    assert (cfg.vllm.chat_url, cfg.vllm.chat_model) == (URL, "Qwen3-8B")


def _listed(out: str) -> list[str]:
    """The rows of the model list the first run printed, without their numbers."""
    return re.findall(r"^  \[\d+\] (.*)$", out, re.M)


@pytest.mark.parametrize(
    ("models", "loaded", "rows", "prompt"),
    [
        (
            ["nomic-embed-text", "bge-reranker-v2-m3", "Qwen3-8B"],
            [],
            [
                "Qwen3-8B",
                "nomic-embed-text  (embedding)",
                "bge-reranker-v2-m3  (reranking)",
            ],
            "Model [1]: ",
        ),
        (  # still no default among several chat models (#231)
            ["nomic-embed-text", "chat-a", "chat-b"],
            [],
            ["chat-a", "chat-b", "nomic-embed-text  (embedding)"],
            "Model: ",
        ),
        (  # an embedding family whose id does not say "embed"
            ["bge-m3", "Qwen3-8B"],
            [],
            ["Qwen3-8B", "bge-m3  (embedding)"],
            "Model [1]: ",
        ),
        (  # the loaded chat model is the default, an embedding model never is
            ["nomic-embed", "chat-a", "chat-b"],
            ["nomic-embed", "chat-b"],
            ["chat-a", "chat-b  (loaded)", "nomic-embed  (embedding, loaded)"],
            "Model [2]: ",
        ),
    ],
    ids=["one-chat-model", "no-loaded-signal", "embedding-family", "loaded"],
)
def test_chat_models_come_first_and_the_others_are_tagged_and_never_the_default(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    models: list[str],
    loaded: list[str],
    rows: list[str],
    prompt: str,
) -> None:
    _terminal(monkeypatch, True)

    async def loaded_models(url: str, **kwargs: object) -> discovery.LoadedModels:
        return discovery.LoadedModels(frozenset(loaded), bool(loaded))

    monkeypatch.setattr(discovery, "loaded_models", loaded_models)
    answers = _Answers("", "", "1", "n")
    _first_setup(answers, found=FoundServer(url=URL, models=models))
    assert answers.prompts[1] == prompt
    assert _listed(capsys.readouterr().out) == rows


def test_a_model_number_counts_in_the_listed_order(config_path: Path) -> None:
    models = ["nomic-embed-text", "chat-a", "chat-b"]
    result, _, _ = _first_setup(
        _Answers("", "2", "n"), found=FoundServer(url=URL, models=models)
    )
    assert result is not None and result.model == "chat-b"


@pytest.mark.parametrize(
    ("platform", "named"),
    [
        ("darwin", "oMLX, Ollama, LM Studio…"),
        ("linux", "Ollama, LM Studio, vLLM…"),
    ],
)
def test_the_first_line_says_what_the_first_run_needs(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    named: str,
) -> None:
    monkeypatch.setattr(onboard.sys, "platform", platform)
    _first_setup(_Answers(URL), lists={URL: ["only"]}, urls=())
    first = capsys.readouterr().out.split("\n")[0]
    assert first == f"melo uses a model server on this computer ({named})."


def test_a_guided_reconfigure_does_not_repeat_the_first_line(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config.save(_configured())
    _guided(_Answers("", "", "n"), found=FoundServer(url=URL, models=["Qwen3-8B"]))
    assert "melo uses a model server" not in capsys.readouterr().out


def test_a_guided_first_run_says_what_melo_needs_then_offers_the_server(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`melo serve` before any setup asks what a first run asks, not the full menu."""
    answers = _Answers("")
    cfg = _guided(answers, found=FoundServer(url=URL, models=["Qwen3-8B"]))
    assert cfg is not None
    assert (cfg.chat_backend, cfg.vllm.chat_url, cfg.vllm.chat_model) == (
        "vllm",
        URL,
        "Qwen3-8B",
    )
    assert answers.prompts == ["Use it? [Y/n]: "]
    lines = capsys.readouterr().out.split("\n")
    assert lines[0].startswith("melo uses a model server on this computer (")
    assert lines[1] == f"Found a model server at {URL} (1 model)."
    assert not any("Welcome" in line for line in lines)


def test_hints_are_wrapped_at_word_boundaries_to_the_terminal_width(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _terminal(monkeypatch, True)
    monkeypatch.setattr(onboard, "_columns", lambda: 40)
    models = ["Qwen3-27B", "Qwen3-Flash"]
    result, _, _ = _first_setup(
        _Answers("", "", "1"), found=FoundServer(url=URL, models=models)
    )
    assert result is not None
    out = capsys.readouterr().out
    hint = out[out.index("  Type the number") : out.index("✓")]
    rows = hint.split("\n")[:-1]
    assert len(rows) > 2 and all(len(row) < 40 for row in rows)
    assert all(
        row.startswith("  ") for row in rows
    )  # continuation rows keep the indent
    assert " ".join(row.strip() for row in rows) == discovery.NO_DEFAULT_HINT


def test_a_loaded_model_is_marked_and_is_the_default(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def loaded_models(url: str, **kwargs: object) -> discovery.LoadedModels:
        assert url == URL
        return discovery.LoadedModels(frozenset({"Qwen3-Flash"}), True)

    monkeypatch.setattr(discovery, "loaded_models", loaded_models)
    models = ["Qwen3-27B", "Qwen3-Flash"]
    answers = _Answers("", "", "n")
    result, _, _ = _first_setup(answers, found=FoundServer(url=URL, models=models))
    assert result == FirstSetup(url=URL, model="Qwen3-Flash", saved=True)
    assert answers.prompts[1] == "Model [2]: "
    assert (
        f"Models on {URL}:\n  [1] Qwen3-27B\n  [2] Qwen3-Flash  (loaded)\n"
        in capsys.readouterr().out
    )


def _terminal(monkeypatch: pytest.MonkeyPatch, tty: bool) -> None:
    class Stdin:
        def isatty(self) -> bool:
            return tty

    monkeypatch.setattr("sys.stdin", Stdin())


# ── On a terminal the model is chosen with the arrow keys ──


def _arrow_keys(monkeypatch: pytest.MonkeyPatch, choose: int | None) -> list[Any]:
    """The first run on a terminal; the list is built, then *choose* picks a row.

    Returns the pickers it built, so a test can read what the list offered.
    """
    from localmelo.support.cli import picker as pickers

    built: list[Any] = []

    def run_inline(build: Any, *, no_color: bool = False) -> Any:
        async def make() -> Any:
            return build(lambda: 80, lambda: None)

        asked = asyncio.run(make())
        built.append(asked)
        return None if choose is None else asked.models[choose]

    monkeypatch.setattr(onboard, "_arrow_keys", lambda: True)
    monkeypatch.setattr(pickers, "run_inline", run_inline)
    return built


@pytest.mark.parametrize(
    ("models", "loaded", "offered", "start", "said"),
    [
        (  # several chat models and none known loaded: nothing is highlighted
            ["nomic-embed-text", "chat-a", "chat-b"],
            [],
            [("chat-a", False, "chat"), ("chat-b", False, "chat")],
            None,
            discovery.LOAD_COST_NOTE,
        ),
        (  # the loaded chat model is where the highlight starts
            ["chat-a", "chat-b", "bge-m3"],
            ["chat-b"],
            [("chat-a", False, "chat"), ("chat-b", True, "chat")],
            "chat-b",
            "",
        ),
        (  # no chat model: the others are offered, tagged, with a note
            ["nomic-embed-text", "bge-reranker-v2-m3"],
            [],
            [
                ("nomic-embed-text", False, "embedding"),
                ("bge-reranker-v2-m3", False, "reranking"),
            ],
            "nomic-embed-text",
            "These look like embedding or reranking models",
        ),
    ],
    ids=["no-default", "loaded", "no-chat-model"],
)
def test_on_a_terminal_the_models_are_a_list_to_choose_from(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    models: list[str],
    loaded: list[str],
    offered: list[tuple[str, bool, str]],
    start: str | None,
    said: str,
) -> None:
    async def loaded_models(url: str, **kwargs: object) -> discovery.LoadedModels:
        return discovery.LoadedModels(frozenset(loaded), bool(loaded))

    monkeypatch.setattr(discovery, "loaded_models", loaded_models)
    built = _arrow_keys(monkeypatch, choose=len(offered) - 1)
    answers = _Answers("", "n")  # the URL; then, with embedding models, memory
    result, _, _ = _first_setup(answers, found=FoundServer(url=URL, models=models))
    [asked] = built
    assert [(m.id, m.loaded, m.kind) for m in asked.models] == offered
    assert (asked.start, asked.where, asked.title) == (start, URL, "Choose a model")
    assert result is not None and result.model == offered[-1][0]
    assert not any(prompt.startswith("Model") for prompt in answers.prompts)
    out = capsys.readouterr().out
    assert "Models on" not in out and "[1]" not in out and said in out
    assert f"✓ {URL} lists {len(models)} models; using {result.model}.\n" in out


def test_closing_the_model_list_cancels_the_first_run(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arrow_keys(monkeypatch, choose=None)
    with pytest.raises(KeyboardInterrupt):
        _first_setup(
            _Answers(""), found=FoundServer(url=URL, models=["chat-a", "chat-b"])
        )
    assert not config_path.exists()


def test_a_server_with_one_model_needs_no_list_even_on_a_terminal(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _arrow_keys(monkeypatch, choose=None)
    answers = _Answers("")
    result, _, _ = _first_setup(
        answers, found=FoundServer(url=URL, models=["only-model"])
    )
    assert result == FirstSetup(url=URL, model="only-model", saved=True)
    assert built == []
    assert not any(prompt.startswith("Model") for prompt in answers.prompts)


@pytest.mark.parametrize(
    ("stdin", "stdout", "term", "expected"),
    [
        (True, True, "xterm-256color", True),
        (True, True, "dumb", False),
        (True, False, "xterm-256color", False),
        (False, True, "xterm-256color", False),
    ],
)
def test_the_list_needs_a_terminal_on_both_sides_that_is_not_dumb(
    monkeypatch: pytest.MonkeyPatch,
    stdin: bool,
    stdout: bool,
    term: str,
    expected: bool,
) -> None:
    class Stream:
        def __init__(self, tty: bool) -> None:
            self.tty = tty

        def isatty(self) -> bool:
            return self.tty

    monkeypatch.setattr("sys.stdin", Stream(stdin))
    monkeypatch.setattr("sys.stdout", Stream(stdout))
    monkeypatch.setenv("TERM", term)
    assert onboard._arrow_keys() is expected


def test_without_a_loaded_answer_there_is_no_default_and_enter_asks_again(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _terminal(monkeypatch, True)
    models = ["Qwen3-27B", "Qwen3-Flash"]
    answers = _Answers("", "", "", "2", "n")
    result, _, _ = _first_setup(answers, found=FoundServer(url=URL, models=models))
    assert result == FirstSetup(url=URL, model="Qwen3-Flash", saved=True)
    assert answers.prompts[1:4] == ["Model: "] * 3
    out = capsys.readouterr().out
    assert "(loaded)" not in out
    assert out.count(discovery.NO_DEFAULT_HINT) == 2
    assert "needs a key" not in out  # no status route answered 401


def test_a_loaded_chat_model_among_several_is_the_default_even_with_embeddings(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def loaded_models(url: str, **kwargs: object) -> discovery.LoadedModels:
        return discovery.LoadedModels(frozenset({"nomic-embed", "b"}), True)

    monkeypatch.setattr(discovery, "loaded_models", loaded_models)
    answers = _Answers("", "", "n")
    result, _, _ = _first_setup(
        answers, found=FoundServer(url=URL, models=["a", "nomic-embed", "b"])
    )
    assert result is not None and result.model == "b"
    assert answers.prompts[1] == "Model [2]: "  # a, b, then nomic-embed


def test_a_status_that_lists_none_loaded_gives_no_default(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _terminal(monkeypatch, True)

    async def loaded_models(url: str, **kwargs: object) -> discovery.LoadedModels:
        return discovery.LoadedModels(frozenset(), True)

    monkeypatch.setattr(discovery, "loaded_models", loaded_models)
    answers = _Answers("", "Qwen3-27B", "n")
    result, _, _ = _first_setup(
        answers, found=FoundServer(url=URL, models=["Qwen3-27B", "Qwen3-Flash"])
    )
    assert result is not None and result.model == "Qwen3-27B"
    assert answers.prompts[1] == "Model: "


def test_without_a_terminal_an_empty_answer_with_no_default_is_an_error(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _terminal(monkeypatch, False)
    answers = _Answers("", "")
    with pytest.raises(onboard.ModelChoiceError, match="no default"):
        _first_setup(
            answers, found=FoundServer(url=URL, models=["Qwen3-27B", "Qwen3-Flash"])
        )
    assert not config_path.exists()


def test_one_chat_model_is_not_looked_up(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def loaded_models(url: str, **kwargs: object) -> discovery.LoadedModels:
        raise AssertionError("no choice to make")

    monkeypatch.setattr(discovery, "loaded_models", loaded_models)
    answers = _Answers("", "", "n")
    result, _, _ = _first_setup(
        answers, found=FoundServer(url=URL, models=["Qwen3-27B", "nomic-embed"])
    )
    assert result == FirstSetup(url=URL, model="Qwen3-27B", saved=True)


@pytest.mark.parametrize(
    ("status", "says_key"), [(401, True), (404, False), (500, False)]
)
def test_a_server_that_does_not_say_what_is_loaded_gives_no_default(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    says_key: bool,
) -> None:
    """Without a key the status route answers 401 as the listing would."""
    _terminal(monkeypatch, True)
    real = _REAL_LOADED_MODELS
    transport = httpx.MockTransport(lambda request: httpx.Response(status))

    async def loaded_models(url: str, **kwargs: object) -> discovery.LoadedModels:
        return await real(url, transport=transport)

    monkeypatch.setattr(discovery, "loaded_models", loaded_models)
    answers = _Answers("", "", "1", "n")
    result, _, _ = _first_setup(
        answers, found=FoundServer(url=URL, models=["Qwen3-27B", "Qwen3-Flash"])
    )
    assert result == FirstSetup(url=URL, model="Qwen3-27B", saved=True)
    assert answers.prompts[1:3] == ["Model: "] * 2
    note = "The server did not say which models are loaded (it needs a key for that)."
    assert (note in capsys.readouterr().out) is says_key


def test_a_first_run_offers_the_servers_embedding_model(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = ["Qwen3-8B", "Qwen3-Embedding-0.6B"]
    answers = _Answers("", "", "")  # URL, model, then Enter: yes
    probe = mock.AsyncMock()
    is_ollama = mock.AsyncMock(return_value=False)
    with (
        mock.patch.object(onboard, "probe_embedding", probe),
        mock.patch.object(discovery, "is_ollama", is_ollama),
    ):
        result, _, _ = _first_setup(answers, found=FoundServer(url=URL, models=models))
    assert result == FirstSetup(url=URL, model="Qwen3-8B", saved=True)
    assert answers.prompts == [
        f"Server URL [{URL}]: ",
        "Model [1]: ",
        "Use Qwen3-Embedding-0.6B for long-term memory? [Y/n]: ",
    ]
    probe.assert_awaited_once()
    # The probe path answers GET /api/version once and passes it to the save.
    is_ollama.assert_awaited_once_with(URL)
    assert probe.await_args is not None
    probed = probe.await_args.args[0]
    assert (
        probed.embedding_backend,
        probed.vllm.embedding_url,
        probed.vllm.embedding_model,
    ) == ("vllm", URL, "Qwen3-Embedding-0.6B")
    cfg = config.load()
    assert (cfg.chat_backend, cfg.embedding_backend) == ("vllm", "vllm")
    assert (cfg.vllm.chat_url, cfg.vllm.chat_model) == (URL, "Qwen3-8B")
    assert (cfg.vllm.embedding_url, cfg.vllm.embedding_model) == (
        URL,
        "Qwen3-Embedding-0.6B",
    )
    out = capsys.readouterr().out
    assert "Long-term memory off" not in out


def test_a_first_run_on_ollama_saves_the_embedding_in_its_table(
    config_path: Path,
) -> None:
    models = ["qwen3:8b", "qwen3-embedding:0.6b"]
    is_ollama = mock.AsyncMock(return_value=True)
    probe = mock.AsyncMock()
    with (
        mock.patch.object(onboard, "probe_embedding", probe),
        mock.patch.object(discovery, "is_ollama", is_ollama),
    ):
        result, _, _ = _first_setup(
            _Answers("", "", ""), found=FoundServer(url=URL, models=models)
        )
    assert result == FirstSetup(url=URL, model="qwen3:8b", saved=True)
    is_ollama.assert_awaited_once_with(URL)
    cfg = config.load()
    assert (cfg.chat_backend, cfg.embedding_backend) == ("ollama", "ollama")
    assert (cfg.ollama.embedding_url, cfg.ollama.embedding_model) == (
        URL,
        "qwen3-embedding:0.6b",
    )
    assert (cfg.vllm.embedding_url, cfg.vllm.embedding_model) == ("", "")


def test_a_first_run_that_declines_saves_embedding_off(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = ["Qwen3-8B", "Qwen3-Embedding-0.6B"]
    answers = _Answers("", "", "n")
    probe = mock.AsyncMock()
    with mock.patch.object(onboard, "probe_embedding", probe):
        result, _, _ = _first_setup(answers, found=FoundServer(url=URL, models=models))
    assert result == FirstSetup(url=URL, model="Qwen3-8B", saved=True)
    probe.assert_not_awaited()
    cfg = config.load()
    assert (cfg.chat_backend, cfg.embedding_backend) == ("vllm", "none")
    assert "Long-term memory off (melo setup sets it again).\n" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("ollama", "hint"),
    [
        (
            False,
            "No embedding model on this server. Serve Qwen3-Embedding-0.6B "
            "on it, then melo setup.",
        ),
        (
            True,
            "No embedding model on this server. ollama pull "
            "qwen3-embedding:0.6b, then melo setup.",
        ),
    ],
)
def test_a_first_run_without_a_candidate_prints_the_hint(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    ollama: bool,
    hint: str,
) -> None:
    async def is_ollama(url: str, **kwargs: object) -> bool:
        return ollama

    monkeypatch.setattr(discovery, "is_ollama", is_ollama)
    result, _, _ = _first_setup(_Answers(URL), lists={URL: ["only"]})
    assert result == FirstSetup(url=URL, model="only", saved=True)
    out = capsys.readouterr().out
    assert f"{hint}\n" in out
    assert config.load().embedding_backend == "none"


def test_a_failed_embedding_probe_saves_with_embedding_off(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = ["Qwen3-8B", "Qwen3-Embedding-0.6B"]
    probe = mock.AsyncMock(side_effect=ProbeError("embedding probe failed: refused"))
    with mock.patch.object(onboard, "probe_embedding", probe):
        result, _, _ = _first_setup(
            _Answers("", "", "y"), found=FoundServer(url=URL, models=models)
        )
    assert result == FirstSetup(url=URL, model="Qwen3-8B", saved=True)
    cfg = config.load()
    assert (cfg.chat_backend, cfg.embedding_backend) == ("vllm", "none")
    assert (cfg.vllm.chat_url, cfg.vllm.chat_model) == (URL, "Qwen3-8B")
    out = capsys.readouterr().out
    assert "  Probe failed: embedding probe failed: refused\n" in out
    assert "  Saving with embedding off: no long-term memory" in out


def test_without_a_found_server_the_typed_url_is_listed(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(onboard.sys, "platform", "linux")
    answers = _Answers(URL)
    result, _, list_models = _first_setup(answers, lists={URL: ["only"]})
    assert result == FirstSetup(url=URL, model="only", saved=True)
    # One model: nothing to choose.
    assert answers.prompts == ["Server URL [http://127.0.0.1:8000/v1]: "]
    list_models.assert_awaited_once_with("openai-compatible", URL)
    out = capsys.readouterr().out
    assert out.startswith(
        "melo uses a model server on this computer (Ollama, LM Studio, vLLM…).\n"
        "No model server found on this computer (looked at ports 8000, 11434). "
        "Start one (for example ollama serve), or enter its URL.\n"
        f"✓ {URL} lists 1 model; using only.\n"
    )


@pytest.mark.parametrize(
    ("platform", "servers"),
    [
        ("darwin", "omlx serve or ollama serve"),
        ("linux", "ollama serve"),
        ("win32", "ollama serve"),
    ],
)
def test_the_no_server_hint_names_omlx_first_on_a_mac(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    servers: str,
) -> None:
    monkeypatch.setattr(onboard.sys, "platform", platform)
    _first_setup(_Answers(URL), lists={URL: ["only"]})
    assert (
        "No model server found on this computer (looked at ports 8000, 11434). "
        f"Start one (for example {servers}), or enter its URL.\n"
    ) in capsys.readouterr().out


def test_a_model_is_chosen_by_number_or_id(config_path: Path) -> None:
    lists: dict[str, list[str] | Exception] = {URL: ["a", "b", "c"]}
    answers = _Answers(URL, "4", "x", "3")
    result, _, _ = _first_setup(answers, lists=lists)
    assert result is not None and result.model == "c"
    assert answers.prompts[1:] == ["Model: "] * 3

    answers = _Answers(URL, "b")
    result, _, _ = _first_setup(answers, lists=lists)
    assert result is not None and result.model == "b"


def test_a_failed_check_retries_with_the_typed_url_as_the_default(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = "http://127.0.0.1:9000/v1"
    lists: dict[str, list[str] | Exception] = {
        URL: DiscoveryError("connection refused", reachable=False),
        other: ["m"],
    }
    answers = _Answers(URL, "", other)
    result, _, _ = _first_setup(answers, lists=lists)
    assert result == FirstSetup(url=other, model="m", saved=True)
    assert answers.prompts == [
        "Server URL [http://127.0.0.1:8000/v1]: ",
        "Enter tries again · s starts without saving · q quits: ",
        f"Server URL [{URL}]: ",
    ]
    assert f"✗ {URL}: connection refused.\n" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("listed", "reason"),
    [
        ([], "it lists no models"),
        (
            DiscoveryError(
                f"HTTP 401: {discovery.NO_API_KEY_HINT}",
                reachable=True,
                status_code=401,
            ),
            "HTTP 401: the server requires an API key; localmelo sends none to "
            'local servers (oMLX: "Allow inference without API key")',
        ),
    ],
)
def test_starting_anyway_asks_for_the_model_and_saves_nothing(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    listed: list[str] | Exception,
    reason: str,
) -> None:
    answers = _Answers(URL, "s", "my-model")
    result, _, _ = _first_setup(answers, lists={URL: listed})
    assert result == FirstSetup(url=URL, model="my-model", saved=False)
    assert answers.prompts[-1] == "Model name [default]: "
    out = capsys.readouterr().out
    assert f"✗ {URL}: {reason}.\n" in out
    assert "Not saved: melo asks again next time.\n" in out
    assert not config_path.exists()


def test_quitting_saves_nothing(config_path: Path) -> None:
    lists: dict[str, list[str] | Exception] = {
        URL: DiscoveryError("connection refused", reachable=False)
    }
    result, _, _ = _first_setup(_Answers(URL, "q"), lists=lists)
    assert result is None
    assert not config_path.exists()


def test_detection_turned_off_sends_nothing(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result, detect, _ = _first_setup(_Answers(URL), lists={URL: ["m"]}, urls=())
    assert result is not None
    detect.assert_not_called()
    assert "No model server found" not in capsys.readouterr().out


def test_a_save_that_fails_keeps_the_choice_for_this_run(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with mock.patch.object(config, "save", side_effect=OSError("read-only")):
        result, _, _ = _first_setup(_Answers(URL), lists={URL: ["m"]})
    assert result == FirstSetup(url=URL, model="m", saved=False)
    assert f"Could not save {config_path}: read-only\n" in capsys.readouterr().out


def test_the_first_run_keeps_the_configs_other_values(config_path: Path) -> None:
    cfg = Config(request_timeout_s=600)
    cfg.ollama.chat_url = "http://127.0.0.1:11434"
    config.save(cfg)
    _first_setup(_Answers(URL), lists={URL: ["m"]})
    saved = config.load()
    assert saved.request_timeout_s == 600
    assert saved.ollama.chat_url == "http://127.0.0.1:11434"
    assert (saved.chat_backend, saved.vllm.chat_model) == ("vllm", "m")


def test_the_first_run_saves_an_ollama_server_as_ollama(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def is_ollama(url: str, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(discovery, "is_ollama", is_ollama)
    result, _, _ = _first_setup(_Answers(URL), lists={URL: ["qwen3:8b"]})
    assert result == FirstSetup(url=URL, model="qwen3:8b", saved=True)
    saved = config.load()
    assert (saved.chat_backend, saved.embedding_backend) == ("ollama", "none")
    assert (saved.ollama.chat_url, saved.ollama.chat_model) == (URL, "qwen3:8b")
    assert (saved.vllm.chat_url, saved.vllm.chat_model) == ("", "")


@pytest.mark.parametrize(("answer", "saved"), [("", True), ("y", True), ("n", False)])
def test_offer_to_save(config_path: Path, answer: str, saved: bool) -> None:
    answers = _Answers(answer)
    with mock.patch("builtins.input", answers):
        assert onboard.offer_to_save(URL, "m") is saved
    assert answers.prompts == [f"Save {URL} · m as the default backend? [Y/n] "]
    assert config_path.exists() is saved


# ── The wizard ──


def _configured() -> Config:
    return Config(
        chat_backend="vllm",
        embedding_backend="none",
        vllm=LocalBackendConfig(chat_url=URL, chat_model="Qwen3-8B"),
    )


def test_enter_at_every_wizard_question_keeps_the_config(config_path: Path) -> None:
    config.save(_configured())
    answers = _Answers(*[""] * 10)
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "_run_probes", return_value=True),
    ):
        cfg = run_backend_setup()
    assert cfg is not None
    assert (cfg.chat_backend, cfg.embedding_backend) == ("vllm", "none")
    assert (cfg.vllm.chat_url, cfg.vllm.chat_model) == (URL, "Qwen3-8B")
    assert answers.prompts == [
        "  Chat backend [3]: ",
        f"  Base URL [{URL}]: ",
        "  Chat model name [Qwen3-8B]: ",
        "  Save and continue? [Y]: ",
        "  Embedding backend [5]: ",
    ]
    assert not any("key" in prompt.lower() for prompt in answers.prompts)


def test_the_wizard_offers_the_chat_servers_embedding_model(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    listing = mock.AsyncMock(
        return_value=["Qwen3-8B", "nomic-embed-text", "Qwen3-Embedding-0.6B"]
    )
    config.save(_configured())
    answers = _Answers("", "", "", "", "6")
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "_run_probes", return_value=True),
        mock.patch.object(discovery, "list_models", listing),
    ):
        cfg = run_backend_setup()
    assert cfg is not None
    assert answers.prompts == [
        "  Chat backend [3]: ",
        f"  Base URL [{URL}]: ",
        "  Chat model name [Qwen3-8B]: ",
        "  Save and continue? [Y]: ",
        "  Embedding backend [6]: ",
    ]
    listing.assert_awaited_once_with("vllm", URL)
    assert (cfg.embedding_backend, cfg.vllm.embedding_url) == ("vllm", URL)
    assert cfg.vllm.embedding_model == "Qwen3-Embedding-0.6B"
    out = capsys.readouterr().out
    assert (
        f"    [6] Qwen3-Embedding-0.6B on this server ({URL})\n"
        f"    [7] nomic-embed-text on this server ({URL})\n"
    ) in out


def test_embedding_choices_are_pure_and_in_recommendation_order() -> None:
    cfg = _configured()
    before = (
        cfg.embedding_backend,
        cfg.vllm.embedding_url,
        cfg.vllm.embedding_model,
        cfg.vllm.chat_model,
    )
    choices = onboard._embedding_choices(
        cfg, ["bge-m3", "Qwen3-Embedding-0.6B", "chat-a", "nomic-embed-text"]
    )
    assert [(choice.key, choice.url, choice.model) for choice in choices] == [
        ("vllm", URL, "Qwen3-Embedding-0.6B"),
        ("vllm", URL, "nomic-embed-text"),
        ("vllm", URL, "bge-m3"),
    ]
    assert (
        cfg.embedding_backend,
        cfg.vllm.embedding_url,
        cfg.vllm.embedding_model,
        cfg.vllm.chat_model,
    ) == before


def test_the_wizard_default_without_a_candidate_is_none(
    config_path: Path,
) -> None:
    config.save(_configured())
    answers = _Answers(*[""] * 10)
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "_run_probes", return_value=True),
        mock.patch.object(discovery, "list_models", mock.AsyncMock(return_value=[])),
    ):
        cfg = run_backend_setup()
    assert cfg is not None and cfg.embedding_backend == "none"
    assert answers.prompts[4] == "  Embedding backend [5]: "


def test_the_wizard_falls_back_when_the_listing_fails(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    listing = mock.AsyncMock(
        side_effect=DiscoveryError("connection refused", reachable=False)
    )
    config.save(_configured())
    answers = _Answers(*[""] * 10)
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "_run_probes", return_value=True),
        mock.patch.object(discovery, "list_models", listing),
    ):
        cfg = run_backend_setup()
    assert cfg is not None and cfg.embedding_backend == "none"
    assert answers.prompts == [
        "  Chat backend [3]: ",
        f"  Base URL [{URL}]: ",
        "  Chat model name [Qwen3-8B]: ",
        "  Save and continue? [Y]: ",
        "  Embedding backend [5]: ",
    ]
    assert "Traceback" not in capsys.readouterr().out


def test_a_failed_wizard_embedding_probe_does_not_save_the_model(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config.save(_configured())
    answers = _Answers("", "", "", "", "6")
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "probe_chat", mock.AsyncMock()),
        mock.patch.object(
            onboard,
            "probe_embedding",
            mock.AsyncMock(side_effect=ProbeError("embedding probe failed: refused")),
        ),
        mock.patch.object(
            discovery,
            "list_models",
            mock.AsyncMock(return_value=["Qwen3-Embedding-0.6B"]),
        ),
    ):
        cfg = run_backend_setup()
    assert cfg is not None and cfg.embedding_backend == "none"
    assert (cfg.vllm.embedding_url, cfg.vllm.embedding_model) == ("", "")
    saved = config.load()
    assert (saved.embedding_backend, saved.vllm.embedding_url) == ("none", "")
    assert saved.vllm.embedding_model == ""
    out = capsys.readouterr().out
    assert "Probe failed: embedding probe failed: refused" in out
    assert "Saving with embedding off: no long-term memory" in out


def test_the_wizard_keeps_a_saved_embedding(config_path: Path) -> None:
    cfg = _configured()
    cfg.embedding_backend = "ollama"
    cfg.ollama.embedding_url = "http://127.0.0.1:11434"
    cfg.ollama.embedding_model = "bge-m3"
    config.save(cfg)
    listing = mock.AsyncMock(return_value=["Qwen3-Embedding-0.6B"])
    answers = _Answers(*[""] * 10)
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "_run_probes", return_value=True),
        mock.patch.object(discovery, "list_models", listing),
    ):
        result = run_backend_setup()
    assert result is not None and result.embedding_backend == "ollama"
    listing.assert_not_called()
    assert answers.prompts[4:] == [
        "  Embedding backend [2]: ",
        "  Embedding endpoint URL [http://127.0.0.1:11434]: ",
        "  Embedding model name [bge-m3]: ",
    ]


def test_the_embedding_question_defaults_to_the_configs_backend(
    config_path: Path,
) -> None:
    cfg = _configured()
    cfg.embedding_backend = "ollama"
    cfg.ollama.embedding_url = "http://127.0.0.1:11434"
    cfg.ollama.embedding_model = "bge-m3"
    config.save(cfg)
    answers = _Answers(*[""] * 10)
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "_run_probes", return_value=True),
    ):
        result = run_backend_setup()
    assert result is not None and result.embedding_backend == "ollama"
    assert answers.prompts[4:] == [
        "  Embedding backend [2]: ",
        "  Embedding endpoint URL [http://127.0.0.1:11434]: ",
        "  Embedding model name [bge-m3]: ",
    ]


def test_without_a_config_the_wizard_offers_the_backends_defaults(
    config_path: Path,
) -> None:
    answers = _Answers("3", "", "m", "", "")
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "_run_probes", return_value=True),
    ):
        run_backend_setup()
    assert answers.prompts[:2] == [
        "  Chat backend [1]: ",
        "  Base URL [http://localhost:8000/v1]: ",
    ]
    assert answers.prompts[4] == "  Embedding backend [5]: "


def _probe_failure(status: int | None = None) -> ProbeError:
    try:
        raise ProviderError("POST /chat/completions failed", status_code=status)
    except ProviderError as cause:
        try:
            raise ProbeError(f"chat probe failed: {cause}") from cause
        except ProbeError as exc:
            return exc


def test_retry_asks_again_with_what_was_typed(config_path: Path) -> None:
    config.save(Config())
    chat = mock.AsyncMock(side_effect=[_probe_failure(), None])
    answers = _Answers(
        *(
            "3",
            "http://127.0.0.1:9000/v1",
            "typo",
            "",
        ),  # backend, URL, model, Save and continue?
        "",  # embedding: none
        "r",  # the check fails: retry
        *("", "fixed", ""),  # URL kept, model fixed, Save and continue?
    )
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "probe_chat", chat),
        mock.patch.object(onboard, "probe_embedding", mock.AsyncMock()),
    ):
        cfg = run_backend_setup()
    assert cfg is not None
    assert (cfg.vllm.chat_url, cfg.vllm.chat_model) == (
        "http://127.0.0.1:9000/v1",
        "fixed",
    )
    assert answers.prompts[6:8] == [
        "  Base URL [http://127.0.0.1:9000/v1]: ",
        "  Chat model name [typo]: ",
    ]
    assert config.load().vllm.chat_model == "fixed"


@pytest.mark.parametrize(("answer", "saved"), [("s", True), ("", False), ("n", False)])
def test_a_failed_check_may_be_saved_anyway(
    config_path: Path, capsys: pytest.CaptureFixture[str], answer: str, saved: bool
) -> None:
    with (
        mock.patch("builtins.input", _Answers(answer)),
        mock.patch.object(
            onboard, "probe_chat", mock.AsyncMock(side_effect=_probe_failure())
        ),
        mock.patch.object(onboard, "probe_embedding", mock.AsyncMock()),
    ):
        assert onboard._run_probes(_configured()) is saved
    out = capsys.readouterr().out
    assert ("config not saved" in out) is not saved


@pytest.mark.parametrize(
    ("backend", "status", "noted"),
    [
        ("vllm", 401, True),
        ("vllm", 403, True),
        ("vllm", 404, False),
        ("vllm", None, False),
        ("openai", 401, False),
    ],
)
def test_a_local_server_that_wants_a_key_says_so(
    capsys: pytest.CaptureFixture[str], backend: str, status: int | None, noted: bool
) -> None:
    cfg = _configured()
    cfg.chat_backend = backend
    with (
        mock.patch("builtins.input", _Answers("")),
        mock.patch.object(
            onboard, "probe_chat", mock.AsyncMock(side_effect=_probe_failure(status))
        ),
    ):
        onboard._run_probes(cfg)
    note = (
        "  Note: the server requires an API key; localmelo sends none to local "
        'servers (oMLX: "Allow inference without API key").\n'
    )
    assert (note in capsys.readouterr().out) is noted


def test_a_failed_embedding_check_saves_with_embedding_off(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _configured()
    cfg.embedding_backend = "ollama"
    with (
        mock.patch.object(onboard, "probe_chat", mock.AsyncMock()),
        mock.patch.object(
            onboard,
            "probe_embedding",
            mock.AsyncMock(side_effect=ProbeError("embedding probe failed: refused")),
        ),
        mock.patch("builtins.input", _Answers()),
    ):
        assert onboard._run_probes(cfg) is True
    assert cfg.embedding_backend == "none"
    out = capsys.readouterr().out
    assert "Probe failed: embedding probe failed: refused\n" in out
    assert "Saving with embedding off: no long-term memory" in out


# ── Review fixes ──


def test_a_lone_embedding_model_is_asked_about_not_used_silently(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = _Answers(URL, "")
    result, _, _ = _first_setup(answers, lists={URL: ["nomic-embed-text"]})
    assert result is not None and result.model == "nomic-embed-text"
    assert answers.prompts[1] == "Model [1]: "
    assert (
        "  These look like embedding or reranking models; chat needs a chat model.\n"
        in capsys.readouterr().out
    )


def test_a_server_that_answered_without_a_list_is_named_and_prefilled(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = "http://127.0.0.1:9100/v1"
    reason = f"HTTP 401: {discovery.NO_API_KEY_HINT}"
    found = FoundServer(url=other, models=[], problem=reason)
    lists: dict[str, list[str] | Exception] = {
        other: DiscoveryError(reason, reachable=True, status_code=401)
    }
    answers = _Answers("", "q")
    result, _, list_models = _first_setup(answers, found=found, lists=lists)
    assert result is None
    assert answers.prompts[0] == f"Server URL [{other}]: "
    list_models.assert_awaited_once_with("openai-compatible", other)  # asked again
    out = capsys.readouterr().out
    assert f"A model server at {other} answered, but {reason}.\n" in out


def test_urls_from_the_environment_prefill_the_question(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(discovery.DETECT_URLS_ENV, "http://127.0.0.1:9100/v1")
    answers = _Answers("http://127.0.0.1:9100/v1")
    _first_setup(
        answers,
        lists={"http://127.0.0.1:9100/v1": ["m"]},
        urls=("http://127.0.0.1:9100/v1",),
    )
    assert answers.prompts[0] == "Server URL [http://127.0.0.1:9100/v1]: "


@pytest.mark.parametrize(
    ("answer", "tail", "expected"),
    [
        ("start without saving", ("m",), FirstSetup(url=URL, model="m", saved=False)),
        ("Quit", (), None),
        ("", ("", "q"), None),  # Enter retries: the URL question comes again
    ],
)
def test_an_option_counts_by_its_first_letter(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    answer: str,
    tail: tuple[str, ...],
    expected: FirstSetup | None,
) -> None:
    lists: dict[str, list[str] | Exception] = {
        URL: DiscoveryError("connection refused", reachable=False)
    }
    answers = _Answers(URL, "x", answer, *tail)
    result, _, _ = _first_setup(answers, lists=lists)
    assert result == expected
    failed = "Enter tries again · s starts without saving · q quits: "
    assert answers.prompts[1:3] == [failed] * 2
    assert "  Press Enter, s or q.\n" in capsys.readouterr().out
    if answer == "":
        assert answers.prompts[3] == f"Server URL [{URL}]: "


@pytest.mark.parametrize(("answer", "saved"), [("save anyway", True), ("No", False)])
def test_the_wizards_options_count_by_their_first_letter(
    config_path: Path, capsys: pytest.CaptureFixture[str], answer: str, saved: bool
) -> None:
    answers = _Answers("y", answer)
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(
            onboard, "probe_chat", mock.AsyncMock(side_effect=_probe_failure())
        ),
        mock.patch.object(onboard, "probe_embedding", mock.AsyncMock()),
    ):
        assert onboard._run_probes(_configured()) is saved
    # "y" was the old "Save config anyway?" answer; it is asked again.
    assert answers.prompts == ["  [s]ave anyway, [N]o [N]: "] * 2
    assert "  Type s or n.\n" in capsys.readouterr().out


def test_offer_to_save_asks_again_for_an_unknown_answer(config_path: Path) -> None:
    answers = _Answers("maybe", "no")
    with mock.patch("builtins.input", answers):
        assert onboard.offer_to_save(URL, "m") is False
    assert len(answers.prompts) == 2


async def test_an_embedding_url_that_cannot_be_used_fails_the_check() -> None:
    from localmelo.support.onboard_probe import probe_embedding

    cfg = _configured()
    cfg.embedding_backend = "vllm"
    cfg.vllm.embedding_url = "http://127.0.0.1:80O0/v1"  # a letter O in the port
    cfg.vllm.embedding_model = "bge-m3"
    with pytest.raises(ProbeError, match=r"^embedding probe failed: \[vllm\] "):
        await probe_embedding(cfg)


def test_the_cloud_wizard_offers_the_configs_key_and_model(config_path: Path) -> None:
    cfg = Config(chat_backend="openai", embedding_backend="none")
    cfg.openai.api_key_env = "MY_OPENAI_KEY"
    cfg.openai.chat_model = "gpt-x"
    config.save(cfg)
    answers = _Answers(*[""] * 6)
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "_run_probes", return_value=True),
    ):
        result = run_backend_setup()
    assert result is not None
    assert (result.openai.api_key_env, result.openai.chat_model) == (
        "MY_OPENAI_KEY",
        "gpt-x",
    )
    assert answers.prompts[:3] == [
        "  Chat backend [5]: ",
        "  API key env var [MY_OPENAI_KEY]: ",
        "  Chat model [gpt-x]: ",
    ]


@pytest.mark.parametrize(
    "model", ["C:\\models\\a", 'quote " inside', "two \\\\ slashes", "tab\tand ünïcode"]
)
async def test_a_listed_model_id_survives_the_config_file(
    config_path: Path, model: str
) -> None:
    assert await onboard.save_chat_server('http://h/v1?x="y"', model)
    saved = config.load()
    assert (saved.vllm.chat_url, saved.vllm.chat_model) == ('http://h/v1?x="y"', model)


async def test_a_server_saved_from_a_session_keeps_embedding_and_the_rest(
    config_path: Path,
) -> None:
    cfg = config.Config(
        chat_backend="sglang",
        embedding_backend="sglang",
        request_timeout_s=30,
        sglang=config.LocalBackendConfig(
            chat_url="http://127.0.0.1:30000/v1",
            chat_model="old",
            embedding_url="http://127.0.0.1:30000/v1",
            embedding_model="embed-x",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        vllm=config.LocalBackendConfig(extra_body={"top_k": 20}),
    )
    config.save(cfg)
    assert await onboard.save_chat_server("http://h:9/v1", "new", keep_embedding=True)
    saved = config.load()
    assert (saved.chat_backend, saved.vllm.chat_url, saved.vllm.chat_model) == (
        "vllm",
        "http://h:9/v1",
        "new",
    )
    assert saved.embedding_backend == "sglang" and saved.has_embedding
    assert saved.sglang == cfg.sglang and saved.request_timeout_s == 30
    assert saved.vllm.extra_body == {"top_k": 20}
    assert saved.validate() == []
    # The first run's save still turns embedding off.
    assert await onboard.save_chat_server("http://h:9/v1", "new")
    assert config.load().embedding_backend == "none"


# ── Telling an Ollama server from any other ──


async def test_saving_an_ollama_server_writes_the_ollama_table(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"version": "0.12.3"})

    monkeypatch.setattr(
        discovery, "is_ollama", _answered_by(httpx.MockTransport(handle))
    )
    assert await onboard.save_chat_server(URL, "qwen3:8b")
    saved = config.load()
    assert (saved.chat_backend, saved.embedding_backend) == ("ollama", "none")
    assert (saved.ollama.chat_url, saved.ollama.chat_model) == (URL, "qwen3:8b")
    assert (saved.vllm.chat_url, saved.vllm.chat_model) == ("", "")
    # Asked once, at the root: the chat URL ends in /v1, this one does not.
    assert seen == ["http://127.0.0.1:8500/api/version"]


def _refusing(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out")


@pytest.mark.parametrize(
    "transport",
    [
        httpx.MockTransport(lambda request: httpx.Response(404)),
        httpx.MockTransport(_refusing),
    ],
    ids=["no-version-endpoint", "timeout"],
)
async def test_another_server_is_saved_as_vllm(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport
) -> None:
    monkeypatch.setattr(discovery, "is_ollama", _answered_by(transport))
    assert await onboard.save_chat_server(URL, "m")
    saved = config.load()
    assert (saved.chat_backend, saved.vllm.chat_url, saved.vllm.chat_model) == (
        "vllm",
        URL,
        "m",
    )
    assert (saved.ollama.chat_url, saved.ollama.chat_model) == ("", "")


def test_the_embedding_guides_toml_examples_load(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every TOML block in the embedding guide is a file config.load() reads."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")  # the cloud example names it
    guide = Path(__file__).resolve().parents[2] / "docs" / "guides" / "embedding.md"
    blocks = re.findall(r"```toml\n(.*?)```", guide.read_text(), re.S)
    assert len(blocks) == 4
    for block in blocks:
        config_path.write_text(block)
        cfg = config.load()
        if "chat_backend" in tomllib.loads(block):
            assert cfg.validate() == []


# ── --reconfigure in default mode (guided) ──


def _guided(
    answers: _Answers, *, found: FoundServer | None, lists: dict | None = None
) -> Config | None:
    detect, list_models = _servers(found, lists)
    with (
        mock.patch.object(discovery, "detect_urls", return_value=(URL,)),
        mock.patch.object(discovery, "detect_local_server", detect),
        mock.patch.object(onboard, "_run_probes", return_value=True),
        mock.patch("builtins.input", answers),
    ):
        return run_backend_setup(guided=True)


def test_a_guided_reconfigure_keeps_a_found_server_with_one_question(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config.save(_configured())
    answers = _Answers("", "", "n")  # keep it, model, no embedding
    cfg = _guided(answers, found=FoundServer(url=URL, models=["Qwen3-8B"]))
    assert cfg is not None and (cfg.vllm.chat_url, cfg.vllm.chat_model) == (
        URL,
        "Qwen3-8B",
    )
    assert answers.prompts[0] == "Use it? [Y/n]: "
    assert not any("backend" in p.lower() or "URL" in p for p in answers.prompts)
    out = capsys.readouterr().out
    assert out.startswith(f"Found a model server at {URL} (1 model).\n")
    assert "Welcome" not in out


def test_a_guided_reconfigure_shows_the_full_menu_after_no(
    config_path: Path,
) -> None:
    config.save(_configured())
    answers = _Answers("n", *[""] * 10)
    cfg = _guided(answers, found=FoundServer(url=URL, models=["Qwen3-8B"]))
    assert cfg is not None
    assert answers.prompts[0] == "Use it? [Y/n]: "
    assert answers.prompts[1] == "  Chat backend [3]: "
    assert "  Save and continue? [Y]: " in answers.prompts


def test_a_guided_reconfigure_with_no_server_goes_straight_to_the_menu(
    config_path: Path,
) -> None:
    config.save(_configured())
    answers = _Answers(*[""] * 10)
    assert _guided(answers, found=None) is not None
    assert answers.prompts[0] == "  Chat backend [3]: "


@pytest.mark.parametrize(
    ("saved", "ollama", "default"),
    [(False, False, "3"), (False, True, "2"), (True, True, "3")],
)
def test_a_declined_server_is_the_menus_default_unless_a_backend_is_saved(
    config_path: Path, saved: bool, ollama: bool, default: str
) -> None:
    if saved:
        config.save(_configured())

    async def is_ollama(url: str, **kwargs: object) -> bool:
        return ollama

    answers = _Answers("n", *[""] * 10)
    with mock.patch.object(discovery, "is_ollama", is_ollama):
        _guided(answers, found=FoundServer(url=URL, models=["Qwen3-8B"]))
    assert answers.prompts[1] == f"  Chat backend [{default}]: "


# ── oMLX in the menus and the saved file ──

_VLLM_ENTRY = "oMLX or vLLM (OpenAI-compatible, local)"


def test_the_backend_menus_name_omlx_with_vllm() -> None:
    assert dict(onboard.CHAT_BACKENDS)["vllm"] == _VLLM_ENTRY
    assert dict(onboard.EMBEDDING_BACKENDS)["vllm"] == _VLLM_ENTRY
    assert dict(onboard.CHAT_BACKENDS)["ollama"] == "Ollama (local endpoint)"
    assert dict(onboard.EMBEDDING_BACKENDS)["mlc"] == "MLC (local endpoint)"


def test_the_wizard_shows_the_vllm_entry_as_omlx_or_vllm(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = _Answers("3", "", "m", "", "")
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "_run_probes", return_value=True),
    ):
        run_backend_setup()
    out = capsys.readouterr().out
    # The chat menu and the embedding menu both list it as entry 3.
    assert out.count(f"    [3] {_VLLM_ENTRY}\n") == 2
    assert "  -- oMLX or vLLM (local endpoint) --\n" in out
    assert "[3] vLLM (local endpoint)" not in out
    assert "-- vllm (local endpoint) --" not in out


def test_the_other_local_backends_keep_their_wizard_heading(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = _Answers("1", "", "m", "", "")
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "_run_probes", return_value=True),
    ):
        run_backend_setup()
    assert "  -- mlc (local endpoint) --\n" in capsys.readouterr().out


_OMLX_FILE = (
    f'chat_backend = "omlx"\nembedding_backend = "none"\n\n'
    f'[vllm]\nchat_url = "{URL}"\nchat_model = "Qwen3-8B-4bit"\n'
)


async def test_the_same_server_is_saved_under_the_spelling_the_file_used(
    config_path: Path,
) -> None:
    config_path.write_text(_OMLX_FILE, encoding="utf-8")
    assert await onboard.save_chat_server(URL, "Qwen3-32B-4bit", keep_embedding=True)
    assert config_path.read_text(encoding="utf-8").startswith(
        'chat_backend = "omlx"\nembedding_backend = "none"\n'
    )
    assert config.load().vllm.chat_model == "Qwen3-32B-4bit"


@pytest.mark.parametrize(
    ("url", "ollama"),
    [("http://127.0.0.1:9000/v1", False), (URL, True)],
    ids=["another-server", "an-ollama-server"],
)
async def test_another_server_is_not_saved_as_omlx(
    config_path: Path, url: str, ollama: bool
) -> None:
    config_path.write_text(_OMLX_FILE, encoding="utf-8")
    assert await onboard.save_chat_server(url, "m", ollama=ollama)
    text = config_path.read_text(encoding="utf-8")
    assert text.startswith(f'chat_backend = "{"ollama" if ollama else "vllm"}"\n')
    assert "omlx" not in text


def test_a_first_run_that_chooses_an_embedding_model_chooses_its_spelling_again(
    config_path: Path,
) -> None:
    cfg = config.Config(chat_backend="vllm", embedding_backend="vllm")
    cfg.chat_alias = cfg.embedding_alias = "omlx"
    cfg.vllm.chat_url = URL
    onboard._with_chat_server(cfg, URL, "m", ollama=False, embedding_model="e")
    assert (cfg.chat_alias, cfg.embedding_alias) == ("omlx", "")


def test_the_first_run_saves_omlx_for_a_server_that_names_itself_omlx(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first run lists the server to find it; the file then says oMLX."""
    from localmelo.support.cli.agents import config_backend

    reply = {"data": [{"id": "Qwen3-8B-4bit", "owned_by": "omlx"}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=reply))

    async def detect(urls: Any, **kwargs: Any) -> FoundServer | None:
        return await _REAL_DETECT(urls, transport=transport)

    monkeypatch.setattr(discovery, "list_models", _REAL_LIST_MODELS)
    monkeypatch.setattr(discovery, "detect_local_server", detect)
    monkeypatch.setattr(discovery, "detect_urls", lambda: (URL,))
    monkeypatch.setattr("builtins.input", _Answers(""))
    result = run_first_setup()
    assert result == FirstSetup(url=URL, model="Qwen3-8B-4bit", saved=True)
    assert config_path.read_text(encoding="utf-8").startswith(
        'chat_backend = "omlx"\nembedding_backend = "none"\n'
    )
    saved = config.load()
    assert (saved.chat_backend, saved.vllm.chat_model) == ("vllm", "Qwen3-8B-4bit")
    # A later process has listed nothing yet; the file alone names the server.
    discovery.forget_servers()
    assert config_backend(saved, None).label == "oMLX"


# ── A listing that names oMLX decides how a server is saved ──


def _list_as_omlx(url: str) -> None:
    """List the models of *url*, which answers as oMLX does (no network)."""
    reply = {"data": [{"id": "Qwen3-8B-4bit", "owned_by": "omlx"}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=reply))
    asyncio.run(_REAL_LIST_MODELS("vllm", url, transport=transport))


def test_a_listed_omlx_server_is_saved_as_omlx(config_path: Path) -> None:
    _list_as_omlx(URL)
    assert asyncio.run(onboard.save_chat_server(URL, "Qwen3-8B-4bit"))
    assert config_path.read_text(encoding="utf-8").startswith(
        'chat_backend = "omlx"\nembedding_backend = "none"\n'
    )
    saved = config.load()
    assert (saved.chat_backend, saved.vllm.chat_url) == ("vllm", URL)
    assert saved.validate() == []


def test_its_embedding_model_is_saved_as_omlx_too(config_path: Path) -> None:
    _list_as_omlx(URL)
    assert asyncio.run(
        onboard.save_chat_server(
            URL, "Qwen3-8B-4bit", embedding_model="Qwen3-Embedding-0.6B"
        )
    )
    assert config_path.read_text(encoding="utf-8").startswith(
        'chat_backend = "omlx"\nembedding_backend = "omlx"\n'
    )
    saved = config.load()
    assert saved.has_embedding and saved.vllm.embedding_url == URL
    assert saved.validate() == []


@pytest.mark.parametrize(
    ("listed", "url", "ollama", "key"),
    [
        (None, URL, False, "vllm"),  # nothing listed: unknown is not oMLX
        (URL, "http://127.0.0.1:9000/v1", False, "vllm"),  # another server
        (URL, URL, True, "ollama"),  # it answered like Ollama
    ],
    ids=["not-listed", "another-server", "ollama"],
)
def test_any_other_server_is_saved_under_its_own_key(
    config_path: Path, listed: str | None, url: str, ollama: bool, key: str
) -> None:
    if listed:
        _list_as_omlx(listed)
    assert asyncio.run(onboard.save_chat_server(url, "m", ollama=ollama))
    text = config_path.read_text(encoding="utf-8")
    assert text.startswith(f'chat_backend = "{key}"\n') and "omlx" not in text


def test_a_server_that_names_another_owner_is_saved_as_vllm(config_path: Path) -> None:
    reply = {"data": [{"id": "m", "owned_by": "vllm"}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=reply))
    asyncio.run(_REAL_LIST_MODELS("vllm", URL, transport=transport))
    assert asyncio.run(onboard.save_chat_server(URL, "m"))
    assert config_path.read_text(encoding="utf-8").startswith('chat_backend = "vllm"\n')


def test_a_switch_that_keeps_the_embedding_leaves_its_spelling(
    config_path: Path,
) -> None:
    config_path.write_text(
        'chat_backend = "vllm"\nembedding_backend = "omlx"\n\n[vllm]\n'
        f'chat_url = "http://127.0.0.1:9000/v1"\nchat_model = "a"\n'
        f'embedding_url = "{URL}"\nembedding_model = "e"\n',
        encoding="utf-8",
    )
    assert asyncio.run(
        onboard.save_chat_server("http://127.0.0.1:9001/v1", "b", keep_embedding=True)
    )
    text = config_path.read_text(encoding="utf-8")
    assert text.startswith('chat_backend = "vllm"\nembedding_backend = "omlx"\n')


def test_the_wizard_spells_a_listed_omlx_server_omlx(config_path: Path) -> None:
    _list_as_omlx(URL)
    answers = _Answers("3", URL, "Qwen3-8B-4bit", "", "")
    with (
        mock.patch("builtins.input", answers),
        mock.patch.object(onboard, "_run_probes", return_value=True),
    ):
        assert run_backend_setup() is not None
    assert config_path.read_text(encoding="utf-8").startswith(
        'chat_backend = "omlx"\nembedding_backend = "none"\n'
    )
