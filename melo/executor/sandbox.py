"""Keep the file writes of a run's shell and Python children in its workspace.

With :attr:`~localmelo.melo.agent.AgentOptions.confine_writes` on, the
``shell_exec`` and ``python_exec`` children of a run with a workspace root
start, on macOS, under ``sandbox-exec`` with the profile :func:`profile`
builds: every file write is refused except in the paths of
:func:`allowed_paths` and the devices of :data:`DEVICES`. Reads, process
starts and the network stay as they are; children (``git``, a child
Python) inherit the profile.

The run's temporary folder is :data:`TMP_DIR` inside the workspace, so it
goes when the workspace goes and stays when the run is kept for review; a
``.gitignore`` in it keeps it out of ``git status`` and the brief's digest.
macOS's ``mktemp`` ignores ``TMPDIR`` and writes in the user's own
temporary folder, which stays writable (see :func:`user_temp_dir`).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
TMP_DIR = ".melo-tmp"
OFF_HINT = (
    "set agent.confine_writes to false to run without it "
    "(-X agent.confine_writes=false for one run)"
)

# Devices a confined child may write, by exact path: the null and zero
# devices, the terminal, the standard streams by name, DTrace's helper,
# which every program's loader opens, and the pseudo-terminal master
# (pty.openpty() in tests). /dev/fd and the pseudo-terminal slaves follow.
DEVICES = (
    "/dev/null",
    "/dev/zero",
    "/dev/tty",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/dtracehelper",
    "/dev/ptmx",
)
_DEVICE_RULES = (r'(subpath "/dev/fd")', r'(regex #"^/dev/ttys[0-9]+$")')

# confstr()'s name for the user's temporary folder (<unistd.h>), which
# os.confstr_names does not list, and the folder every such path is under.
_CS_DARWIN_USER_TEMP_DIR = 65537
_USER_FOLDERS = "/private/var/folders/"

_REFUSED = re.compile(r"operation not permitted", re.IGNORECASE)
_QUOTED = re.compile(r"""['"`](/[^'"`\n]*)['"`]""")
_NOTE_PATHS = 3


class ConfinementError(RuntimeError):
    """Confinement is on but cannot be applied, so the run must not start."""


def cache_dirs(env: Mapping[str, str]) -> list[str]:
    """The caches that the project's tools write outside a workspace.

    Measured on macOS for ``uv`` (``uv sync``, ``uv run``), ``black`` and
    ``npm``; ``pytest``, ``ruff``, ``mypy`` and ``git status``/``add``
    write only in the project. Each follows the variable its tool reads in
    *env* (the child's environment), else the tool's default under
    ``HOME``. A path that is not absolute is left out.
    """
    home = env.get("HOME", "")
    xdg = env.get("XDG_CACHE_HOME", "")
    uv = env.get("UV_CACHE_DIR") or (
        os.path.join(xdg, "uv") if xdg else os.path.join(home, ".cache", "uv")
    )
    black = env.get("BLACK_CACHE_DIR") or os.path.join(
        home, "Library", "Caches", "black"
    )
    npm = env.get("npm_config_cache") or env.get("NPM_CONFIG_CACHE")
    npm = npm or os.path.join(home, ".npm")
    return [path for path in (uv, black, npm) if os.path.isabs(path)]


def git_dirs(root: str) -> list[str]:
    """What git writes outside *root* when *root* is a linked worktree.

    A linked worktree's ``.git`` is a file naming its own git directory in
    another repository (``<common>/worktrees/<name>``): its index, ``HEAD``
    and logs are there, and ``git add`` writes objects into
    ``<common>/objects``. Both are allowed, nothing else of that
    repository: its refs, config and index, and the main checkout's files,
    stay unwritable, so ``git status``, ``diff`` and ``add`` work and a
    commit to a branch is refused. Objects are content-addressed and
    append-only, so a new one changes no ref, index or file of any
    checkout. The git directory counts only when its ``gitdir`` file names
    this ``.git`` back, a file the confined run cannot write; any other
    ``.git`` file (a submodule's, or one the model wrote) adds nothing.
    """
    dot_git = os.path.join(root, ".git")
    try:
        with open(dot_git, encoding="utf-8") as file:
            line = file.readline().strip()
        if not line.startswith("gitdir:"):
            return []
        gitdir = os.path.realpath(os.path.join(root, line[7:].strip()))
        with open(os.path.join(gitdir, "gitdir"), encoding="utf-8") as file:
            back = os.path.realpath(os.path.join(gitdir, file.readline().strip()))
        with open(os.path.join(gitdir, "commondir"), encoding="utf-8") as file:
            common = os.path.realpath(os.path.join(gitdir, file.readline().strip()))
    except (OSError, UnicodeDecodeError, ValueError):
        return []
    if back != os.path.realpath(dot_git):
        return []
    return [gitdir, os.path.join(common, "objects")]


def user_temp_dir() -> str | None:
    """The user's own temporary folder on macOS (``/var/folders/.../T``), or ``None``.

    It is ``confstr(_CS_DARWIN_USER_TEMP_DIR)``, else ``getconf
    DARWIN_USER_TEMP_DIR``, as a real path. macOS's ``mktemp`` and
    ``mktemp -d`` write there whatever ``TMPDIR`` says; the folder is
    private to the user and holds no project. Anything that is not under
    ``/private/var/folders/`` (so never ``/tmp``, the home folder or a
    project) is ``None``.
    """
    try:
        path = os.confstr(_CS_DARWIN_USER_TEMP_DIR)
    except (ValueError, OSError):
        path = None
    if not path:
        try:
            path = subprocess.run(
                ["/usr/bin/getconf", "DARWIN_USER_TEMP_DIR"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    real = os.path.realpath(path) if path and os.path.isabs(path) else ""
    return real if real.startswith(_USER_FOLDERS) else None


def allowed_paths(root: str, env: Mapping[str, str]) -> list[str]:
    """The folders a confined child may write in, real paths, each once.

    The workspace *root* (and so the run's temporary folder in it), the git
    directories of :func:`git_dirs`, the user's temporary folder of
    :func:`user_temp_dir` and the caches of :func:`cache_dirs`; a path
    inside one listed before it is left out.
    """
    temp = user_temp_dir()
    out: list[str] = []
    for path in [root, *git_dirs(root), *([temp] if temp else []), *cache_dirs(env)]:
        real = os.path.realpath(path)
        if not any(_inside(real, kept) for kept in out):
            out.append(real)
    return out


def _inside(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def quote(path: str) -> str:
    """*path* as a string of the profile language, quoted safely.

    A backslash and a double quote are escaped; a control character, which
    no workspace or cache path needs, raises :class:`ConfinementError`.
    """
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in path):
        raise ConfinementError(f"a path has a control character: {path!r}")
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def profile(paths: Sequence[str]) -> str:
    """The ``sandbox-exec`` profile that allows writes only in *paths*."""
    rules = [
        *(f"(literal {quote(device)})" for device in DEVICES),
        *_DEVICE_RULES,
        *(f"(subpath {quote(path)})" for path in paths),
    ]
    return (
        "(version 1)\n(allow default)\n(deny file-write*)\n(allow file-write*\n  "
        + "\n  ".join(rules)
        + ")\n"
    )


@dataclass
class WriteConfinement:
    """The confinement of one run: its profile, temp folder and refusals."""

    root: str
    tmp: str
    paths: tuple[str, ...]
    profile: str
    refused: int = 0

    def argv(self, argv: Sequence[str]) -> list[str]:
        """*argv* started under the profile."""
        return [SANDBOX_EXEC, "-p", self.profile, *argv]

    def env(self) -> dict[str, str]:
        """The variables that point a child's temporary files at the run's folder."""
        return {"TMPDIR": self.tmp, "TMP": self.tmp, "TEMP": self.tmp}

    def note(self, text: str) -> str:
        """The line, with its line break, for the refused writes *text* reports.

        It is ``""`` when *text* reports none.

        A line of *text* with "Operation not permitted" (EPERM, the error a
        refused write gets) counts as one refused write, unless the path it
        names is in an allowed folder, where the refusal was not the
        profile's. Each refused write adds to :attr:`refused`.
        """
        paths: list[str] = []
        count = 0
        for line in text.splitlines():
            found = _REFUSED.search(line)
            if found is None:
                continue
            path = _refused_path(line, found.start())
            if path and any(_inside(os.path.realpath(path), p) for p in self.paths):
                continue
            count += 1
            if path and path not in paths:
                paths.append(path)
        if not count:
            return ""
        self.refused += count
        named = ", ".join(paths[:_NOTE_PATHS])
        if len(paths) > _NOTE_PATHS:
            named += f" and {len(paths) - _NOTE_PATHS} more"
        line = (
            "melo: writing outside the workspace is not allowed in this run"
            + (f" (refused: {named})" if named else "")
            + f"; write in {self.root}, and put scratch files in $TMPDIR."
        )
        return f"{line}\n"


class Confinement(Protocol):
    """What a shell or Python child's start asks of a confinement."""

    def argv(self, argv: Sequence[str]) -> list[str]: ...

    def env(self) -> dict[str, str]: ...

    def note(self, text: str) -> str: ...


class AutoConfinement:
    """The confinement of the calls auto mode decides (#339); off for the rest.

    An Agent that confines nothing of its own still keeps the writes of an
    auto-mode call in its workspace: nobody approves those calls, so the
    profile is the floor under the review. The Agent sets :attr:`on` for
    each call and calls :meth:`ready` (blocking: run it in a thread) before
    one runs with it on; the profile is made once, by :func:`confine`. While
    :attr:`on` is false a child starts as it would without it. When the
    profile cannot be made, a call with it on fails with why, never
    unconfined.
    """

    def __init__(self, root: str, env: Callable[[], Mapping[str, str]]) -> None:
        self.root = root
        self.on = False
        self._env = env
        self._made: WriteConfinement | None = None
        self._error = ""
        self._tried = False

    def ready(self) -> None:
        """Make the profile, the first time only (off macOS there is none)."""
        if self._tried:
            return
        self._tried = True
        try:
            self._made = confine(self.root, self._env())
        except ConfinementError as exc:
            self._error = str(exc)

    @property
    def refused(self) -> int:
        return 0 if self._made is None else self._made.refused

    def argv(self, argv: Sequence[str]) -> list[str]:
        if not self.on:
            return list(argv)
        if self._error or not self._tried:
            raise ConfinementError(
                self._error or "auto mode's confinement is not ready"
            )
        return list(argv) if self._made is None else self._made.argv(argv)

    def env(self) -> dict[str, str]:
        return self._made.env() if self.on and self._made is not None else {}

    def note(self, text: str) -> str:
        if not self.on or self._made is None:
            return ""
        return self._made.note(text).replace("in this run", "in auto mode")


def _refused_path(line: str, at: int) -> str | None:
    """The path a line with an EPERM error at *at* names, or ``None``."""
    quoted = _QUOTED.search(line)
    if quoted is not None:
        return quoted.group(1)
    head = line[:at].rstrip().removesuffix(":").rpartition(": ")[2].strip()
    return head if head.startswith("/") else None


def platform_note() -> str:
    """Why :func:`confine` confines nothing on this platform."""
    return f"writes are confined only on macOS, not {sys.platform}"


def confine(root: str, env: Mapping[str, str]) -> WriteConfinement | None:
    """The confinement of a run in the workspace *root*, or ``None`` off macOS.

    *env* is the environment the run's children get. It makes the run's
    temporary folder and checks, by starting ``/usr/bin/true`` under the
    profile, that the profile applies here. Raises
    :class:`ConfinementError`, one line that says why and how to turn
    confinement off, when ``sandbox-exec`` is missing, the folder cannot be
    made, or the profile does not apply (as inside a sandbox that refuses
    another).
    """
    if sys.platform != "darwin":
        return None
    try:
        return _confine(root, env)
    except ConfinementError as exc:
        raise ConfinementError(
            f"writes cannot be confined to the workspace: {exc}; {OFF_HINT}"
        ) from None


def _confine(root: str, env: Mapping[str, str]) -> WriteConfinement:
    if not os.access(SANDBOX_EXEC, os.X_OK):
        raise ConfinementError(f"{SANDBOX_EXEC} is missing")
    root = os.path.realpath(root)
    tmp = os.path.join(root, TMP_DIR)
    try:
        os.makedirs(tmp, exist_ok=True)
        ignore = os.path.join(tmp, ".gitignore")
        if not os.path.exists(ignore):
            with open(ignore, "w", encoding="utf-8") as file:
                file.write("*\n")
    except OSError as exc:
        raise ConfinementError(f"cannot make {tmp}: {exc.strerror}") from None
    paths = tuple(allowed_paths(root, env))
    text = profile(paths)
    try:
        probe = subprocess.run(
            [SANDBOX_EXEC, "-p", text, "/usr/bin/true"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfinementError(f"sandbox-exec did not start: {exc}") from None
    if probe.returncode:
        why = (probe.stderr.strip().splitlines() or [f"exit {probe.returncode}"])[0]
        raise ConfinementError(f"sandbox-exec refused the profile: {why}")
    return WriteConfinement(root=root, tmp=tmp, paths=paths, profile=text)
