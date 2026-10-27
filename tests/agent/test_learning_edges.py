"""Edge cases of a run's procedural side, found in review.

A folder whose name is not UTF-8, a shell command of a megabyte, a run
cut short while it looks its routine up, and a display helper that
fails: none may cost a run its learning, its first progress event, or
its end.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import agent as agent_module
from localmelo.melo.memory.procedures.classify import label_for, shell_label
from localmelo.melo.memory.procedures.normalize import key_for_query, run_place
from localmelo.melo.memory.procedures.store import MemoryProcedures
from localmelo.melo.memory.procedures.template import TemplateView
from localmelo.tests.agent.procedural_helpers import (
    QUERY,
    Recorder,
    ScriptedLLM,
    answer,
    make_agent,
    read,
)


@pytest.fixture
def odd_folder(tmp_path: Path) -> Any:
    """The working directory set to a folder whose name is not UTF-8."""
    raw = os.fsencode(tmp_path) + b"/caf\xe9"
    try:
        os.mkdir(raw)
    except OSError:  # a file system that takes UTF-8 names only
        pytest.skip("this file system refuses names that are not UTF-8")
    cwd = os.getcwd()
    os.chdir(raw)
    try:
        yield raw
    finally:
        os.chdir(cwd)


def test_run_place_gives_text_that_encodes(odd_folder: bytes) -> None:
    place = run_place(None)
    assert place.encode("utf-8")  # no lone surrogate left
    assert place.endswith("caf\\xe9")
    assert key_for_query(QUERY, place).place == place


async def test_a_folder_whose_name_is_not_utf8_is_learned(
    odd_folder: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    agent = make_agent(ScriptedLLM([answer("one"), answer("two")]))
    try:
        await agent.run_task(QUERY)
        recorder = Recorder()
        await agent.run_task(QUERY, on_event=recorder, progress=True)
    finally:
        await agent.close()
    episodes = [
        row
        for row in await agent.hippo.procedures.export_rows()
        if row["table"] == "episodes"
    ]
    assert [e["match"] for e in episodes] == ["new", "exact"]
    assert recorder.of("run.progress")[0].data["level"] == 1
    assert not [r for r in caplog.records if "learn" in r.getMessage()]


def test_a_huge_shell_command_is_labelled_quickly() -> None:
    body = "print('x' * 10)\n" * 60_000  # about 1 MB
    command = f"python3 - <<'EOF'\n{body}EOF"
    started = time.perf_counter()
    label = label_for("shell_exec", {"command": command})
    assert time.perf_counter() - started < 0.05
    assert label == shell_label("python3 - <<'EOF'") == "python3"
    assert shell_label("cd /p && " + "x=1 " * 5 + "git status " + "a " * 300_000) == (
        "git status"
    )


class HangingStore(MemoryProcedures):
    async def lookup(self, *args: Any, **kwargs: Any) -> TemplateView | None:
        await asyncio.Event().wait()
        return None


async def test_the_first_progress_event_comes_when_a_timeout_cuts_the_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "LOOKUP_TIMEOUT_S", 30.0)
    recorder = Recorder()
    agent = make_agent(ScriptedLLM([answer("never")]), timeout_s=0.1)
    agent.hippo.procedures = HangingStore()
    result = await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert result.status == "timeout"
    assert recorder.types == [
        "run.started",
        "run.progress",
        "run.progress",
        "run.finished",
    ]
    states = [e.data["state"] for e in recorder.of("run.progress")]
    assert states == ["running", "ended"]


async def test_the_first_progress_event_comes_when_a_stop_cuts_the_lookup() -> None:
    recorder = Recorder()
    store = HangingStore()
    agent = make_agent(ScriptedLLM([answer("never")]))
    agent.hippo.procedures = store
    running = asyncio.create_task(
        agent.run_task(QUERY, on_event=recorder, progress=True)
    )
    while "run.started" not in recorder.types:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert recorder.types == [
        "run.started",
        "run.progress",
        "run.progress",
        "run.finished",
    ]
    assert recorder.events[-1].data["result"]["status"] == "cancelled"


async def test_a_display_helper_that_fails_ends_progress_not_the_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = 0

    def failing(view: TemplateView | None) -> TemplateView | None:
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise RuntimeError("display bug")
        return view

    monkeypatch.setattr(agent_module, "shown_view", failing)
    caplog.set_level(logging.DEBUG)
    recorder = Recorder()
    agent = make_agent(ScriptedLLM([read("/p/a.py"), answer("done")]))
    result = await agent.run_task(QUERY, on_event=recorder, progress=True)
    assert result.status == "completed"
    assert recorder.types[-1] == "run.finished"
    assert recorder.types.count("run.progress") == 2
    assert [r.levelno for r in caplog.records if "display bug" in r.getMessage()] == [
        logging.DEBUG
    ]
    episodes = [
        row
        for row in await agent.hippo.procedures.export_rows()
        if row["table"] == "episodes"
    ]
    assert len(episodes) == 1
