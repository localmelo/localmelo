"""Every question with a fixed set of answers is a list: move to a row, press Enter.

The first run's lists are drawn by prompt_toolkit's own application on a pipe
input (the keys are queued before the list opens); the attach question runs in
melo's own screen. Each question's typed answers, for a pipe or a script, keep
working: their tests are the ones in test_onboard_setup.py and test_session.py,
which feed `input()` and run unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

import localmelo.support.onboard as onboard
from localmelo.support import config
from localmelo.support.backends import discovery
from localmelo.support.backends.discovery import DiscoveryError, FoundServer
from localmelo.support.cli.theme import ASCII
from localmelo.support.onboard import FirstSetup
from localmelo.support.onboard_probe import ProbeError
from localmelo.tests.cli.test_onboard_setup import (  # noqa: F401  (fixtures)
    URL,
    _Answers,
    _first_setup,
    _no_models_for_the_wizard,
    _nothing_is_loaded,
    _server_is_not_ollama,
    _setup_metadata,
    config_path,
)
from localmelo.tests.cli.ui_helpers import CTRL_C, CTRL_D, ENTER, ESC, UP, running_app

DOWN = "\x1b[B"

# A config file in a temporary folder for every test here.
pytestmark = pytest.mark.usefixtures("config_path")
_ESCAPES = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[()][0-9A-B]|\x1b[=>78]")


@contextlib.contextmanager
def typed(monkeypatch: pytest.MonkeyPatch, keys: str) -> Iterator[io.StringIO]:
    """A terminal that has *keys* queued; the yielded stream is what the list drew."""
    drawn = io.StringIO()

    def from_pty(cls: Any, stdout: Any, term: str | None = None, **kwargs: Any) -> Any:
        return Vt100_Output(drawn, lambda: Size(24, 80), term="xterm-256color")

    monkeypatch.setattr(Vt100_Output, "from_pty", classmethod(from_pty))
    monkeypatch.setattr(onboard, "_arrow_keys", lambda: True)
    with create_pipe_input() as pipe, create_app_session(input=pipe):
        pipe.send_text(keys)
        yield drawn


def seen(drawn: io.StringIO) -> str:
    """The text the list drew, without the control sequences."""
    return _ESCAPES.sub("", drawn.getvalue())


# ── The wizard: the backend lists and "Save these settings?" ──


@pytest.mark.parametrize(
    ("keys", "chosen"),
    [
        (ENTER, 3),  # the current backend is selected first
        (DOWN + ENTER, 4),
        (UP + ENTER, 2),
        ("5", 5),  # a number chooses that row at once
        ("9" + ENTER, 3),  # there is no ninth row: the key does nothing
    ],
)
def test_the_backend_list_starts_on_the_current_backend(
    monkeypatch: pytest.MonkeyPatch, keys: str, chosen: int
) -> None:
    names = [name for _, name in onboard.CHAT_BACKENDS]
    with typed(monkeypatch, keys) as drawn:
        assert onboard._ask_choice("Chat backend", names, default=3) == chosen
    text = seen(drawn)
    assert "Chat backend" in text and "❯ 3. oMLX or vLLM" in text
    assert "↑↓ select · enter confirm · esc cancel" in text


@pytest.mark.parametrize(
    ("keys", "raised"),
    [(ESC, KeyboardInterrupt), (CTRL_C, KeyboardInterrupt), (CTRL_D, EOFError)],
)
def test_closing_a_backend_list_cancels_the_setup(
    monkeypatch: pytest.MonkeyPatch, keys: str, raised: type[BaseException]
) -> None:
    with typed(monkeypatch, keys), pytest.raises(raised):
        onboard._ask_choice("Chat backend", ["a", "b"])


@pytest.mark.parametrize(
    ("keys", "saved"),
    [(ENTER, True), (DOWN + ENTER, False), ("1", True), ("2", False), (ESC, False)],
)
def test_save_and_continue_is_a_list_that_starts_on_saving(
    monkeypatch: pytest.MonkeyPatch, keys: str, saved: bool
) -> None:
    with typed(monkeypatch, keys) as drawn:
        assert onboard._confirm() is saved
    text = seen(drawn)
    assert "Save these settings?" in text and "❯ 1. Save and continue" in text
    assert "2. Cancel setup" in text and "esc cancel setup" in text


# ── The failed check ──


def _probes(monkeypatch: pytest.MonkeyPatch, *, fail: int) -> None:
    """The chat check fails *fail* times, then answers; the embedding check passes."""
    left = [fail]

    async def probe_chat(cfg: object) -> None:
        if left[0]:
            left[0] -= 1
            raise ProbeError("no answer")

    async def probe_embedding(cfg: object) -> None:
        return None

    monkeypatch.setattr(onboard, "probe_chat", probe_chat)
    monkeypatch.setattr(onboard, "probe_embedding", probe_embedding)


@pytest.mark.parametrize(
    ("keys", "result", "retried"),
    [
        (ENTER, False, 0),  # the default is the last row: cancel, as [N] was
        ("1", True, 1),  # Try again asks the setup's questions again, and passes
        ("2", True, 0),  # Save anyway
        (UP + ENTER, True, 0),
        (ESC, False, 0),
    ],
)
def test_a_failed_check_offers_try_again_save_anyway_and_cancel(
    monkeypatch: pytest.MonkeyPatch, keys: str, result: bool, retried: int
) -> None:
    _probes(monkeypatch, fail=1)
    again = mock.Mock(return_value=True)
    with typed(monkeypatch, keys) as drawn:
        assert onboard._run_probes(config.Config(), retry=again) is result
    assert again.call_count == retried
    text = seen(drawn)
    assert "The check failed. What now?" in text and "❯ 3. Cancel setup" in text
    assert "1. Try again" in text and "esc cancel setup" in text


@pytest.mark.parametrize(
    ("keys", "result"), [(ENTER, False), ("1", True), (ESC, False)]
)
def test_without_a_retry_the_failed_check_offers_two_rows(
    monkeypatch: pytest.MonkeyPatch, keys: str, result: bool
) -> None:
    _probes(monkeypatch, fail=1)
    with typed(monkeypatch, keys) as drawn:
        assert onboard._run_probes(config.Config()) is result
    text = seen(drawn)
    assert "1. Save anyway" in text and "❯ 2. Cancel setup" in text
    assert "Try again" not in text


# ── Long-term memory ──


@pytest.mark.parametrize(
    ("keys", "model"),
    [
        (ENTER, "nomic-embed-text"),  # use it: the default
        (DOWN + ENTER, None),
        ("2", None),
        (ESC, None),  # esc leaves memory off
    ],
)
def test_long_term_memory_is_a_list_that_starts_on_using_it(
    monkeypatch: pytest.MonkeyPatch, keys: str, model: str | None
) -> None:
    _probes(monkeypatch, fail=0)
    models = ["chat-a", "nomic-embed-text"]
    with typed(monkeypatch, keys) as drawn:
        offered = onboard._offer_embedding(URL, "chat-a", models, ollama=False)
    assert offered == (model, False)
    text = seen(drawn)
    assert "Remember across chats with nomic-embed-text?" in text
    assert "❯ 1. Yes, with it" in text and "esc no, by words only" in text


# ── The first run: a found server, a failed address, no server ──


@pytest.mark.parametrize(
    ("keys", "prompts"),
    [
        (ENTER, []),
        (DOWN + ENTER, [f"Server URL [{URL}]: "]),
        ("2", [f"Server URL [{URL}]: "]),
    ],
)
def test_a_found_server_is_a_list_that_starts_on_using_it(
    monkeypatch: pytest.MonkeyPatch, keys: str, prompts: list[str]
) -> None:
    answers = _Answers("")
    with typed(monkeypatch, keys) as drawn:
        result, _, _ = _first_setup(
            answers, found=FoundServer(url=URL, models=["only-model"])
        )
    assert result == FirstSetup(url=URL, model="only-model", saved=True)
    assert answers.prompts == prompts
    text = seen(drawn)
    assert "Use this model server?" in text and "❯ 1. Use it" in text
    assert "2. Use another server" in text and "esc cancel" in text


def test_closing_the_found_server_list_cancels_the_first_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with typed(monkeypatch, ESC), pytest.raises(KeyboardInterrupt):
        _first_setup(_Answers(), found=FoundServer(url=URL, models=["only-model"]))
    assert not Path(config.CONFIG_PATH).exists()


def _first_run_where_the_address_fails_once(answers: _Answers) -> FirstSetup | None:
    """Detection is off; the address typed first fails, a second try works."""
    listed = mock.AsyncMock(
        side_effect=[DiscoveryError("connection refused", reachable=False), ["m"]]
    )
    with (
        mock.patch.object(discovery, "detect_urls", return_value=()),
        mock.patch.object(discovery, "list_models", listed),
        mock.patch("builtins.input", answers),
    ):
        return onboard.run_first_setup()


@pytest.mark.parametrize(
    ("keys", "saved", "prompts"),
    [
        # Enter tries again: the address is asked again, with the failed one.
        (ENTER, True, 2),
        ("2", False, 2),  # starts without saving: asks the model name
        ("3", None, 1),  # quits
        (ESC, None, 1),
    ],
)
def test_a_failed_address_offers_try_again_start_without_saving_and_quit(
    monkeypatch: pytest.MonkeyPatch,
    keys: str,
    saved: bool | None,
    prompts: int,
) -> None:
    answers = _Answers("http://a/v1", "")
    with typed(monkeypatch, keys) as drawn:
        result = _first_run_where_the_address_fails_once(answers)
    assert (None if result is None else result.saved) is saved
    assert len(answers.prompts) == prompts
    text = seen(drawn)
    assert "What now?" in text and "❯ 1. Try again" in text
    assert "2. Start without saving" in text and "esc quit" in text
    if saved is False:
        assert result == FirstSetup(url="http://a/v1", model="default", saved=False)


@pytest.mark.parametrize(
    ("keys", "result", "prompts"),
    [
        (ENTER, FoundServer(url=URL, models=["m"]), []),  # Check again finds one
        (
            DOWN + DOWN + ENTER,
            "http://x/v1",
            [f"Server URL [{onboard._FIRST_RUN_URL}]: "],
        ),
        ("3", "http://x/v1", [f"Server URL [{onboard._FIRST_RUN_URL}]: "]),
        ("4", None, []),
        (ESC, None, []),
    ],
)
def test_with_no_server_the_list_checks_again_takes_an_address_or_quits(
    monkeypatch: pytest.MonkeyPatch, keys: str, result: object, prompts: list[str]
) -> None:
    answers = _Answers("http://x/v1")
    found = mock.AsyncMock(return_value=FoundServer(url=URL, models=["m"]))
    with (
        typed(monkeypatch, keys) as drawn,
        mock.patch.object(discovery, "detect_local_server", found),
        mock.patch("builtins.input", answers),
    ):
        assert onboard._wait_for_a_server((URL,)) == result
    assert answers.prompts == prompts
    text = seen(drawn)
    assert "No model server found. What now?" in text and "❯ 1. Check again" in text
    assert "2. Review setup plan" in text and "3. Enter an address" in text
    assert "4. Quit" in text
    assert "esc quit" in text


# ── Save as the default ──


@pytest.mark.parametrize(
    ("keys", "saved"),
    [(ENTER, True), ("1", True), (DOWN + ENTER, False), ("2", False), (ESC, False)],
)
def test_saving_the_default_backend_is_a_list_that_starts_on_saving(
    monkeypatch: pytest.MonkeyPatch, keys: str, saved: bool
) -> None:
    save = mock.AsyncMock(return_value=True)
    with (
        typed(monkeypatch, keys) as drawn,
        mock.patch.object(onboard, "save_chat_server", save),
    ):
        assert onboard.offer_to_save(URL, "m") is saved
    assert save.await_count == int(saved)
    text = seen(drawn)
    assert f"Save {URL} · m as the default backend?" in text
    assert "❯ 1. Save it" in text and "2. Not now" in text and "esc not now" in text


# ── Without a terminal the typed answers stay ──


def test_without_a_terminal_every_question_still_reads_its_typed_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipes, scripts and these tests: no list is built, `input()` is asked."""
    monkeypatch.setattr(onboard, "_arrow_keys", lambda: False)
    monkeypatch.setattr(
        onboard,
        "_choose_from",
        mock.Mock(side_effect=AssertionError("a list was built without a terminal")),
    )
    answers = _Answers("", "", "n", "s", "n")
    with mock.patch("builtins.input", answers):
        assert onboard._confirm() is True  # "" is yes
        assert onboard._ask_choice("Chat backend", ["a", "b", "c"], default=2) == 2
        assert onboard._choose("Use it? [Y/n]: ", "yn", "y", rows=["a", "b"]) == "n"
        assert onboard._choose("[s]ave, [N]o ", "sn", "n", rows=["a", "b"]) == "s"
        assert onboard._choose("Keep? [Y/n]: ", "yn", "y", rows=["a", "b"]) == "n"
    assert len(answers.prompts) == 5


# ── In melo's own screen: the attach question ──


def _pick(driver: Any) -> asyncio.Task[int]:
    return asyncio.create_task(
        driver.ui.pick(
            "Attach this output?",
            ["Attach it", "Don't attach it"],
            default=1,
            escape=1,
        )
    )


@pytest.mark.parametrize(
    ("keys", "chosen"),
    [
        (ENTER, 1),  # the safe answer is selected first
        (UP + ENTER, 0),
        (UP + DOWN + ENTER, 1),
        ("1", 0),
        ("2", 1),
        ("3" + UP + ENTER, 0),  # there is no third row
        (ESC, 1),
        (CTRL_C, 1),  # inside melo's screen ctrl+c closes the list, as esc does
        (CTRL_D, 1),
    ],
)
async def test_the_attach_question_is_a_list_that_starts_on_not_attaching(
    keys: str, chosen: int
) -> None:
    async with running_app() as driver:
        task = _pick(driver)
        await driver.wait(lambda: "Attach this output?" in driver.text())
        shown = driver.text()
        assert "❯ 2. Don't attach it" in shown and "1. Attach it" in shown
        assert "↑↓ select · enter confirm · esc don't attach it" in shown
        driver.type(keys)
        assert await asyncio.wait_for(task, 3) == chosen
        assert driver.ui.app.is_running and driver.ui.picker is None


async def test_a_question_list_has_no_filter_and_ignores_typed_text() -> None:
    async with running_app() as driver:
        task = _pick(driver)
        await driver.wait(lambda: "Attach this output?" in driver.text())
        assert "Filter" not in driver.text()
        driver.type("hello")
        await driver.settle()
        assert not task.done()  # the letters went nowhere
        driver.type(ENTER)
        assert await asyncio.wait_for(task, 3) == 1


async def test_a_long_question_wraps_and_the_ascii_symbols_have_no_arrows() -> None:
    async with running_app(columns=40) as driver:
        driver.ui.symbols = ASCII
        title = "Save http://127.0.0.1:8000/v1 - a-long-model-name as the default?"
        task = asyncio.create_task(
            driver.ui.pick(title, ["Save it", "Not now"], default=0, escape=1)
        )
        await driver.wait(lambda: "up/down select" in driver.text())
        screen = driver.text()
        assert title in "".join(screen.splitlines())  # wrapped, never cut
        assert "↑↓" not in screen
        driver.type(ESC)
        assert await asyncio.wait_for(task, 3) == 1
