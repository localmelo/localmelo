"""ExitSignals: a signal bounds the clean-up, and an overdue one ends melo."""

from __future__ import annotations

import asyncio
import signal
from types import SimpleNamespace
from typing import Any

import pytest

from localmelo.support.cli import signals as signals_module
from localmelo.support.cli.signals import SIGNAL_CLEANUP_S, ExitSignals

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="POSIX signals")


class Seen:
    def __init__(self) -> None:
        self.killed = 0
        self.begun: list[tuple[str, float]] = []
        self.restored = 0
        self.hung_up = 0
        self.ended = 0
        self.shutdowns: list[str] = []
        self.exits: list[int] = []
        self.later: list[tuple[float, Any]] = []

    def factory(self) -> Any:
        provider = SimpleNamespace(kill=self._kill)
        agent = SimpleNamespace(
            _tool_providers=[provider],
            begin_exit=lambda reason, within_s: self.begun.append((reason, within_s)),
        )
        return SimpleNamespace(live_agents=lambda: [agent])

    def _kill(self) -> None:
        self.killed += 1

    async def shutdown(self, reason: str) -> None:
        self.shutdowns.append(reason)


def _signals(seen: Seen) -> ExitSignals:
    def restore() -> None:
        seen.restored += 1

    def hang_up() -> None:
        seen.hung_up += 1

    def end_app() -> None:
        seen.ended += 1

    return ExitSignals(
        seen.factory(),
        shutdown=seen.shutdown,
        end_app=end_app,
        on_hangup=hang_up,
        restore_terminal=restore,
    )


@pytest.mark.parametrize(
    ("signum", "restored"), [(signal.SIGTERM, 1), (signal.SIGHUP, 0)]
)
async def test_a_signal_bounds_the_cleanup_and_an_overdue_one_ends_melo(
    monkeypatch: pytest.MonkeyPatch, signum: int, restored: int
) -> None:
    seen = Seen()
    monkeypatch.setattr(signals_module, "exit_by_signal", seen.exits.append)
    loop = asyncio.get_running_loop()
    exit_signals = _signals(seen)
    with monkeypatch.context() as patch:
        patch.setattr(
            loop, "call_later", lambda delay, fn, *args: seen.later.append((delay, fn))
        )
        exit_signals.receive(signum)
        exit_signals.receive(signal.SIGTERM)  # a second signal changes nothing
    assert SIGNAL_CLEANUP_S == 8.0
    name = signal.Signals(signum).name
    assert seen.begun == [(f"Cancelled by {name}", 7.0)]
    assert seen.ended == 1 and seen.hung_up == (signum == signal.SIGHUP)
    assert exit_signals.shutting_down is not None
    await exit_signals.shutting_down
    assert seen.shutdowns == [f"Cancelled by {name}"]
    [(delay, overdue)] = seen.later
    assert delay == 8.0
    overdue()
    # The tool providers are killed; the terminal's settings come back,
    # unless the terminal is gone.
    assert (seen.killed, seen.restored, seen.exits) == (1, restored, [signum])


def test_hangup_takes_the_sighup_path() -> None:
    seen = Seen()
    exit_signals = _signals(seen)

    async def run() -> None:
        with pytest.MonkeyPatch.context() as patch:
            loop = asyncio.get_running_loop()
            patch.setattr(loop, "call_later", lambda *args: None)
            exit_signals.hangup()
        assert exit_signals.shutting_down is not None
        await exit_signals.shutting_down

    asyncio.run(run())
    assert exit_signals.received == signal.SIGHUP and seen.hung_up == 1
