"""Edge cases of a run's procedural side, found in review.

A folder whose name is not UTF-8 and a shell command of a megabyte may
cost a run neither its learning nor time.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.memory.procedures.classify import label_for, shell_label
from localmelo.melo.memory.procedures.normalize import key_for_query, run_place
from localmelo.tests.agent.procedural_helpers import (
    QUERY,
    Recorder,
    ScriptedLLM,
    answer,
    make_agent,
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
