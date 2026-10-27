"""The first-run questions and the setup wizard's defaults and retries.

Answers are scripted through input(); model listing and the backend checks
are faked, so nothing reaches a model server. The config file lives in a
temporary directory.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest import mock

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


@pytest.fixture
def config_path(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "config.toml"
    with (
        mock.patch("localmelo.support.config.CONFIG_PATH", str(path)),
        mock.patch("localmelo.support.config.CONFIG_DIR", str(tmp_path)),
    ):
        yield path


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
    answers = _Answers("", "")
    result, detect, list_models = _first_setup(
        answers, found=FoundServer(url=URL, models=models)
    )
    assert result == FirstSetup(url=URL, model="Qwen3-8B", saved=True)
    assert answers.prompts == [f"Model server URL [{URL}]: ", "Model [2]: "]
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
        f"Saved to {config_path} (melo --reconfigure changes it).\n"
    )
    cfg = config.load()
    assert (cfg.chat_backend, cfg.embedding_backend) == ("vllm", "none")
    assert (cfg.vllm.chat_url, cfg.vllm.chat_model) == (URL, "Qwen3-8B")


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
def test_a_listed_model_id_survives_the_config_file(
    config_path: Path, model: str
) -> None:
    assert onboard.save_chat_server('http://h/v1?x="y"', model)
    saved = config.load()
    assert (saved.vllm.chat_url, saved.vllm.chat_model) == ('http://h/v1?x="y"', model)
