"""The real-backend smoke tests are opt-in, and collecting them is offline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Run pytest in a fresh process with connect() broken before collection starts.
_GUARDED_PYTEST = """
import pytest
import socket

def _fail(*args, **kwargs):
    raise AssertionError("pytest collection opened a socket")

socket.create_connection = _fail
socket.socket.connect = _fail

raise SystemExit(pytest.main({args!r}))
"""


def _run_guarded_pytest(
    args: list[str], *, opt_in: bool, cache_dir: Path
) -> subprocess.CompletedProcess[str]:
    """Run pytest with sockets disabled, and ``LOCALMELO_SMOKE`` as asked."""
    env = os.environ.copy()
    env.pop("LOCALMELO_SMOKE", None)
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    if opt_in:
        env["LOCALMELO_SMOKE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _GUARDED_PYTEST.format(
                args=["-o", f"cache_dir={cache_dir}", *args],
            ),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("opt_in", [False, True], ids=["off", "on"])
def test_collecting_the_smoke_tests_opens_no_socket(
    opt_in: bool, tmp_path: Path
) -> None:
    result = _run_guarded_pytest(
        ["--collect-only", "-q", "tests/smoke"],
        opt_in=opt_in,
        cache_dir=tmp_path / "pytest-cache",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "test_smoke_ollama_chat_round_trip" in output
    assert "test_smoke_mlc_chat_round_trip" in output


def test_smoke_tests_are_skipped_with_a_hint_when_not_opted_in(
    tmp_path: Path,
) -> None:
    result = _run_guarded_pytest(
        ["-q", "-rs", "tests/smoke/test_backend_smoke.py"],
        opt_in=False,
        cache_dir=tmp_path / "pytest-cache",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "2 skipped" in output, output
    assert "LOCALMELO_SMOKE=1" in output, output
