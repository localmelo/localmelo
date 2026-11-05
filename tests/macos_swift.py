"""Build and run Swift test programs against the macOS app's sources (``macos/LocalMeloApp``).

Every macOS test file compiles some of the app's Swift files with a test ``main.swift`` and
runs the result. This is the one place that does it: ``run_swift`` builds and runs,
``build_swift`` only builds, and ``typecheck_swift`` checks that files compile together.

Builds go to one temporary folder for the test session.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "macos/LocalMeloApp"
DEVELOPER_DIR = "/Applications/Xcode.app/Contents/Developer"
COMPILE_TIMEOUT_S = 300

_folder: tempfile.TemporaryDirectory[str] | None = None
_builds = 0


def _swiftc(*args: str) -> None:
    subprocess.run(
        ["/usr/bin/xcrun", "swiftc", *args],
        env={**os.environ, "DEVELOPER_DIR": DEVELOPER_DIR},
        check=True,
        timeout=COMPILE_TIMEOUT_S,
    )


def _paths(files: Sequence[str | Path]) -> list[str]:
    """The files as paths: a bare name is a file of the app."""
    return [str(APP / file) for file in files]


def app_sources(*, but: str = "") -> list[Path]:
    """Every Swift file of the app, as the build finds them, except the one named *but*."""
    return [path for path in sorted(APP.glob("*.swift")) if path.name != but]


def build_swift(
    files: Sequence[str | Path], source: str, *, to: Path | None = None
) -> Path:
    """The program of *files* and the test ``main.swift`` *source*, built.

    A name in *files* is a file of the app; a path stands for itself. The binary is the
    build's own unless *to* is given: then it is copied there (a program that must
    sit at a given place, such as an app bundle's executable) and that path is returned.
    """
    global _folder, _builds
    if _folder is None:
        _folder = tempfile.TemporaryDirectory(prefix="localmelo-swift-")
    _builds += 1
    build = Path(_folder.name) / str(_builds)
    build.mkdir()
    (build / "main.swift").write_text(source, encoding="utf-8")
    binary = build / "Test"
    _swiftc(*_paths(files), str(build / "main.swift"), "-o", str(binary))
    if to is None:
        return binary
    shutil.copy2(binary, to)
    return to


def run_swift(
    files: Sequence[str | Path], source: str, *args: str, timeout: float = 30
) -> str:
    """Build the program (see :func:`build_swift`), run it with *args* and return its output.

    A program that exits non-zero, or runs past *timeout* seconds, fails the test; its
    error output goes to the test's own.
    """
    done = subprocess.run(
        [str(build_swift(files, source)), *args],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        timeout=timeout,
    )
    return done.stdout


def typecheck_swift(files: Sequence[str | Path]) -> None:
    """Fail unless *files* compile together."""
    _swiftc("-typecheck", *_paths(files))
