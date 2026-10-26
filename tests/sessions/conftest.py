"""Fixtures for the session tests: a temporary HOME for every test.

With HOME in a temporary directory, the default sessions files and the CLI
host id resolve there, so no test opens ``~/.cache/localmelo``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path
