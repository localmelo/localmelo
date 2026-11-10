"""The requests of a run with the defaults, as sha256 digests.

The run is the scripted one of the default golden (a Python call, a reflection that
continues, an answer) through the OpenAI-compatible provider, with the defaults.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import _chat
from localmelo.tests.agent.wire_runs import record_run, reflection_continues

# The sha256 of the run's three request bodies (plan, reflection, plan) as main sends
# them. Recorded at 139992b5; re-recorded when the fixed system
# section lost its ``Prefix: fixed-prefix-v2`` line: with that line put back,
# the digests of 139992b5 come out again. Re-recorded again when file_read gained
# start_line and end_line: with its earlier description and parameters put back
# in the tool list, the digests before this change come out. Re-recorded again when
# shell_exec and python_exec gained the optional ``why`` (and a last sentence in their
# descriptions): with the earlier tool definitions put back, the digests before this
# change come out. Re-recorded again when the reflection prompt lost the seven keys
# nothing read (UNREAD): with them put back, the digests before this change come out.
OFF = (
    "5ccaf2488d634fd83628cf6db020d3b930be27752f8d97733af26ea4b154939f",
    "eb0e359a146cfc949ce6ca9c4798cb2bedd78c41b10c1d09e331011d6e4dd708",
    "8edfe1a513570859ae8c8231b34f9ca84a4bc1b87be1c8ff189244f0535f8042",
)


async def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **options: Any
) -> tuple[Any, list[dict[str, Any]], tuple[str, ...]]:
    """The result, the request bodies and their sha256 of one run."""
    scenario = reflection_continues()
    scenario.options.update(options)
    monkeypatch.setattr(
        _chat, "local_date_line", lambda: "Date: 2026-10-28 (UTC-04:00)", raising=False
    )
    raw: list[str] = []
    run_dir = Path(tempfile.mkdtemp(dir=tmp_path))
    result, _ = await record_run("openai", scenario, run_dir, monkeypatch, raw)
    digests = tuple(hashlib.sha256(body.encode()).hexdigest() for body in raw)
    return result, [json.loads(body) for body in raw], digests


async def test_every_request_of_a_default_run_is_the_one_main_sends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (await _run(tmp_path, monkeypatch))[2] == OFF
