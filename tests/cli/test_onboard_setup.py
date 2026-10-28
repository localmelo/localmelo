"""The first-run questions and the setup wizard's defaults and retries.

Answers are scripted through input(); model listing and the backend checks
are faked, so nothing reaches a model server. The config file lives in a
temporary directory.
"""

from __future__ import annotations

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


@pytest.fixture(autouse=True)
def _server_is_not_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving asks whether the server is Ollama; these fake servers are not."""

    async def is_ollama(url: str, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(discovery, "is_ollama", is_ollama)


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
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = ["bge-m3-embed", "Qwen3-8B", "Qwen3-32B"]
    answers = _Answers("", "", "n")
    result, detect, list_models = _first_setup(
        answers, found=FoundServer(url=URL, models=models)
    )
    assert result == FirstSetup(url=URL, model="Qwen3-8B", saved=True)
    assert answers.prompts == [
        f"Model server URL [{URL}]: ",
        "Model [2]: ",
        "Use bge-m3-embed for long-term memory? [Y/n]: ",
    ]
    detect.assert_awaited_once_with(
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    )
    list_models.assert_not_called()  # the found server's list is used
    assert capsys.readouterr().out == (
        f"Found a model server at {URL} (3 models).\n"
        f"Models on {URL}:\n"
        "  [1] bge-m3-embed\n"
        "  [2] Qwen3-8B\n"
        "  [3] Qwen3-32B\n"
        f"✓ {URL} lists 3 models; using Qwen3-8B.\n"
        "Long-term memory off (melo --reconfigure sets it again).\n"
        f"Saved to {config_path} (melo --reconfigure changes it).\n"
    )
    cfg = config.load()
    assert (cfg.chat_backend, cfg.embedding_backend) == ("vllm", "none")
    assert (cfg.vllm.chat_url, cfg.vllm.chat_model) == (URL, "Qwen3-8B")


def test_a_first_run_offers_the_servers_embedding_model(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = ["Qwen3-8B", "Qwen3-Embedding-0.6B"]
    answers = _Answers("", "", "")  # URL, model, then Enter: yes
    probe = mock.AsyncMock()
    with mock.patch.object(onboard, "probe_embedding", probe):
        result, _, _ = _first_setup(answers, found=FoundServer(url=URL, models=models))
    assert result == FirstSetup(url=URL, model="Qwen3-8B", saved=True)
    assert answers.prompts == [
        f"Model server URL [{URL}]: ",
        "Model [1]: ",
        "Use Qwen3-Embedding-0.6B for long-term memory? [Y/n]: ",
    ]
    probe.assert_awaited_once()
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
    assert "Long-term memory off (melo --reconfigure sets it again).\n" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("ollama", "hint"),
    [
        (
            False,
            "No embedding model on this server. Serve Qwen3-Embedding-0.6B "
            "on it, then melo --reconfigure.",
        ),
        (
            True,
            "No embedding model on this server. ollama pull "
            "qwen3-embedding:0.6b, then melo --reconfigure.",
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
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = _Answers(URL)
    result, _, list_models = _first_setup(answers, lists={URL: ["only"]})
    assert result == FirstSetup(url=URL, model="only", saved=True)
    # One model: nothing to choose.
    assert answers.prompts == ["Model server URL [http://127.0.0.1:8000/v1]: "]
    list_models.assert_awaited_once_with("openai-compatible", URL)
    out = capsys.readouterr().out
    assert out.startswith(
        "No model server answered on this machine (ports 8000, 11434).\n"
        f"✓ {URL} lists 1 model; using only.\n"
    )


def test_a_model_is_chosen_by_number_or_id(config_path: Path) -> None:
    lists: dict[str, list[str] | Exception] = {URL: ["a", "b", "c"]}
    answers = _Answers(URL, "4", "x", "3")
    result, _, _ = _first_setup(answers, lists=lists)
    assert result is not None and result.model == "c"
    assert answers.prompts[1:] == ["Model [1]: "] * 3

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
        "Model server URL [http://127.0.0.1:8000/v1]: ",
        "[r]etry, [s]tart anyway, [q]uit [r]: ",
        f"Model server URL [{URL}]: ",
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
    assert "No model server answered" not in capsys.readouterr().out


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
        "  Start? [Y]: ",
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
        "  Start? [Y]: ",
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
    choices = onboard._embedding_choices(
        cfg, ["bge-m3", "Qwen3-Embedding-0.6B", "chat-a", "nomic-embed-text"]
    )
    assert [(choice.key, choice.url, choice.model) for choice in choices] == [
        ("vllm", URL, "Qwen3-Embedding-0.6B"),
        ("vllm", URL, "nomic-embed-text"),
        ("vllm", URL, "bge-m3"),
    ]
    assert cfg.embedding_backend == "none"  # nothing was written


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
        assert run_backend_setup() is not None
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
        "  Start? [Y]: ",
        "  Embedding backend [5]: ",
    ]
    assert "Traceback" not in capsys.readouterr().out


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
        *("3", "http://127.0.0.1:9000/v1", "typo", ""),  # backend, URL, model, Start?
        "",  # embedding: none
        "r",  # the check fails: retry
        *("", "fixed", ""),  # URL kept, model fixed, Start?
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
    assert answers.prompts[0] == f"Model server URL [{other}]: "
    list_models.assert_awaited_once_with("openai-compatible", other)  # asked again
    out = capsys.readouterr().out
    assert out.startswith(f"A model server at {other} answered, but {reason}.\n")


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
    assert answers.prompts[0] == "Model server URL [http://127.0.0.1:9100/v1]: "


@pytest.mark.parametrize(
    ("answer", "tail", "expected"),
    [
        ("start anyway", ("m",), FirstSetup(url=URL, model="m", saved=False)),
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
    assert answers.prompts[1:3] == ["[r]etry, [s]tart anyway, [q]uit [r]: "] * 2
    assert "  Type r, s or q.\n" in capsys.readouterr().out
    if answer == "":
        assert answers.prompts[3] == f"Model server URL [{URL}]: "


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
    """Every TOML block in docs/embedding.md is a file config.load() reads."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")  # the cloud example names it
    guide = Path(__file__).resolve().parents[2] / "docs" / "embedding.md"
    blocks = re.findall(r"```toml\n(.*?)```", guide.read_text(), re.S)
    assert len(blocks) == 4
    for block in blocks:
        config_path.write_text(block)
        cfg = config.load()
        if "chat_backend" in tomllib.loads(block):
            assert cfg.validate() == []
