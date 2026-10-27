"""Direct mode never loads the terminal UI's libraries.

``prompt_toolkit`` and ``rich`` are dependencies for bare ``melo``, which
imports them lazily; ``melo "<task>"`` and ``import localmelo.__main__``
load neither, so a task starts as fast as before and its output cannot
depend on them.

httpx imports ``rich`` itself, for its command-line client, whenever
``click`` is installed (the gateway and dev extras pull it in). The
children here make ``click`` unimportable, as in an install with no extras,
so only localmelo's own imports count.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from localmelo.tests.cli.test_direct_mode_process import (
    FakeChatServer,
    _env,
    chat_server,  # noqa: F401  (a fixture)
)

UI_LIBRARIES = ("prompt_toolkit", "rich")

# Prints, at exit, which of the UI's libraries the process loaded; click is
# left out, as without extras (see the module's docstring).
REPORT = (
    "import atexit, sys\n"
    "sys.modules['click'] = None\n"
    "atexit.register(lambda: print('LOADED', sorted(m for m in {names!r} "
    "if m in sys.modules), file=sys.stderr))\n"
)


def _child(code: str, home: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", REPORT.format(names=UI_LIBRARIES) + code],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        cwd=home,
        env=_env(home),
        timeout=60,
        check=False,
    )


def test_importing_the_cli_loads_neither_library(tmp_path: Path) -> None:
    done = _child("import localmelo.__main__\n", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stderr.endswith("LOADED []\n"), done.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="a local HTTP server")
def test_a_direct_run_loads_neither_library(
    chat_server: FakeChatServer, tmp_path: Path  # noqa: F811
) -> None:
    code = (
        "import localmelo.__main__ as m\n"
        f"sys.argv = ['melo', '--base-url', {chat_server.base_url!r}, 'say done']\n"
        "m.main()\n"
    )
    done = _child(code, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout == "done\n"
    assert done.stderr.endswith("LOADED []\n"), done.stderr
    assert chat_server.requests == 1


def test_without_click_httpx_loads_no_ui_library(tmp_path: Path) -> None:
    done = _child("import httpx\n", tmp_path)
    assert done.stderr.endswith("LOADED []\n"), done.stderr


def test_the_report_names_a_library_that_was_loaded(tmp_path: Path) -> None:
    done = _child("import rich.console\n", tmp_path)
    assert done.stderr.endswith("LOADED ['rich']\n"), done.stderr
