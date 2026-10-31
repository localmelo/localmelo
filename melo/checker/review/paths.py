"""What the review knows about one call, what it found, and where a path lies."""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import os
import re
from dataclasses import dataclass, field, replace

from localmelo.melo.contracts.review import TITLES, ActionReview

# Under the home folder. Reads count too: passwords, keys and cookies. An
# entry that starts with "**/" is a file name, anywhere (shell patterns, any
# case), that a read through a helper may not name unasked (see secret_name).
_CREDENTIALS = [
    ".ssh",
    ".aws",
    ".netrc",
    ".config/gh",
    ".docker/config.json",
    ".gnupg",
    ".kube",
    ".pypirc",
    ".npmrc",
    ".git-credentials",
    ".password-store",
    "Library/Keychains",
    "Library/Cookies",
    "Library/Safari",
    "Library/Application Support/Google/Chrome",
    "Library/Application Support/Firefox",
    ".cache/localmelo",
    "**/.env*",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/id_rsa*",
    "**/id_ed25519*",
    "**/id_ecdsa*",
    "**/.netrc",
    "**/.npmrc",
    "**/.pypirc",
    "**/.git-credentials",
    "**/credentials*",
    "**/secrets*",
    "**/*token",
    "**/*.token",
    "**/token.*",
    "**/*token.json",
    "**/*token.txt",
]
_ANYWHERE = "**/"
# Writes here make something run later on its own.
_PERSISTENCE = [
    "Library/LaunchAgents",
    ".zshrc",
    ".zprofile",
    ".zshenv",
    ".zlogin",
    ".bashrc",
    ".bash_profile",
    ".profile",
    ".config/fish",
    ".ssh/config",
    ".gitconfig",
    ".config/git",
]
_SYSTEM = ["/etc", "/private/etc", "/System", "/Library", "/usr", "/bin", "/sbin"]
# Writable without asking: temporary folders, a few devices, the tools' caches.
_HARMLESS = [
    "/tmp",
    "/private/tmp",
    "/var/folders",
    "/private/var/folders",
    "/dev/null",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/tty",
    "/dev/fd",
]
_CACHES = [".cache/uv", "Library/Caches/black", ".npm"]
# Where the running localmelo is imported from: an editable install's
# source, or the package inside the app bundle (see _running).
_SOURCE = os.path.realpath(os.path.join(os.path.dirname(__file__), *[".."] * 3))
# Local files the review reads for one call, at most: those it runs, and
# the changed files a runner (pytest…) runs.
MAX_READS = 10
MAX_SCANS = 40
_LAUNCHD = ("/Library/LaunchAgents", "/Library/LaunchDaemons")
_SAY = {
    "self": "Changes {}, part of localmelo itself.",
    "source": "Changes {}, part of the running localmelo.",
    "persistence": "Changes {}, which makes something run later on its own.",
    "system": "Changes {}, a system folder.",
}


@dataclass(frozen=True)
class Context:
    """The call's surroundings: the workspace, the home folder, melo's Python.

    ``here`` is the folder of the file under review (its imports start
    there); ``since`` the wall-clock time the chat began, or ``None``: a
    file changed after it counts as written in this chat.
    """

    root: str
    home: str
    executable: str
    prefix: str
    depth: int = 0
    here: str = ""
    since: float | None = None

    @property
    def folder(self) -> str:
        return os.path.basename(self.root.rstrip("/")) or self.root

    def deeper(self, here: str | None = None) -> Context:
        return replace(
            self, depth=self.depth + 1, here=self.here if here is None else here
        )


@dataclass
class Findings:
    """High-impact categories with their reasons; the undecided reasons.

    ``hits`` has one ``(category, site, reason)`` for each finding as the
    rules met it, ``?`` for a doubt, repeats kept; ``site`` is the call or
    command the rules were reading (``Findings.site``), else the reason. An
    edit is judged by the hits it adds (see ``files.written``).
    """

    high: dict[str, list[str]] = field(default_factory=dict)
    unsure: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    seen: set[tuple[str, str, str | None]] = field(default_factory=set)
    # The local files the call runs, read for it: path -> digest (the key's).
    read: dict[str, str] = field(default_factory=dict)
    # Files a runner runs that changed in this chat, read for it (not the key's).
    scanned: dict[str, str] = field(default_factory=dict)
    # Changed files a runner runs that melo did not read, as they are now.
    changed: list[str] = field(default_factory=list)
    ran_runner: bool = False
    site: str = ""
    hits: list[tuple[str, str, str]] = field(default_factory=list)

    def take(self, path: str, *, key: bool = True) -> str | None:
        """*path*'s text when the review may read it (once, at most
        :data:`MAX_READS` files, or :data:`MAX_SCANS` without *key*, 64 KB
        each), else ``None``.

        A file read before gives ``""``: it was reviewed already. A file
        that cannot be read is a doubt. With *key*, its digest joins the
        step's key (a file the call runs); without, it does not (a file a
        runner runs, which changes with each edit).
        """
        if path in self.read or path in self.scanned:
            return ""
        if len(self.read if key else self.scanned) >= (MAX_READS if key else MAX_SCANS):
            self.doubt("Runs more local files than melo checks.")
            return None
        try:
            if os.path.getsize(path) > 64 * 1024:
                raise OSError
            with open(path, "rb") as file:
                data = file.read()
            text = data.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            self.doubt(f"Runs {os.path.basename(path)}, which melo could not read.")
            return None
        digest = hashlib.sha256(data).hexdigest()[:16]
        (self.read if key else self.scanned)[path] = digest
        return text

    def add(self, category: str, reason: str) -> None:
        self.hits.append((category, self.site or reason, reason))
        self.high.setdefault(category, [])
        if reason not in self.high[category]:
            self.high[category].append(reason)

    def doubt(self, reason: str) -> None:
        self.hits.append(("?", self.site or reason, reason))
        if reason not in self.unsure:
            self.unsure.append(reason)

    def review(self) -> ActionReview:
        def cut(text: str) -> str:
            return text if len(text) <= 120 else text[:119] + "…"

        inputs = tuple(f"{path}:{digest}" for path, digest in sorted(self.read.items()))
        unsure = tuple(map(cut, self.unsure[:3]))
        changed = tuple(self.changed)
        if not self.high:
            if unsure:
                return ActionReview(
                    "undecided", (), unsure, inputs=inputs, changed=changed
                )
            return ActionReview("routine", inputs=inputs)
        cats = tuple(sorted(self.high, key=list(TITLES).index))
        reasons = tuple(dict.fromkeys(cut(r) for c in cats for r in self.high[c]))
        paths = tuple(dict.fromkeys(self.paths))
        return ActionReview(
            "high",
            cats,
            reasons[:3],
            paths,
            doubts=unsure,
            inputs=inputs,
            changed=changed,
        )


def under(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def resolve(ctx: Context, raw: str, cwd: str | None = None) -> str | None:
    """*raw* as an absolute real path, or None when it holds a variable."""
    text = raw.replace("${HOME}", ctx.home).replace("$HOME", ctx.home)
    if text == "~" or text.startswith("~/"):
        text = ctx.home + text[1:]
    if not text or text.startswith("~") or re.search(r"[$`]", text):
        return None
    return os.path.realpath(os.path.join(cwd or ctx.root, text))


def _running(source: str) -> str:
    """The folder of the running localmelo: the app bundle around *source*, else *source*."""
    parts = source.split(os.sep)
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].endswith(".app"):
            return os.sep.join(parts[: i + 1])
    return source


@functools.lru_cache(maxsize=32)
def _places(home: str, root: str, source: str) -> dict[str, tuple[str, ...]]:
    """The folders of :func:`where` for one home, workspace and running localmelo.

    Only the running localmelo counts as itself (*source*, the folder it is
    imported from, or the app bundle around it), with its settings: another
    copy of the source or another app is a folder like any other. It does
    not count when the workspace is inside it: the person works on it.
    """
    later = [*(os.path.join(home, p) for p in _PERSISTENCE), f"{root}/.git/hooks"]
    running = _running(source)
    own = () if under(root, running) else (running,)
    return {
        "self": (f"{home}/.cache/localmelo", *own),
        "persistence": (*later, f"{root}/.envrc", f"{root}/.git/config", *_LAUNCHD),
        "credentials": tuple(
            os.path.join(home, p) for p in _CREDENTIALS if not p.startswith(_ANYWHERE)
        ),
        "temp": tuple(os.path.realpath(p) for p in _HARMLESS),
        "caches": tuple(os.path.join(home, p) for p in _CACHES),
    }


def secret_name(text: str) -> bool:
    """Whether a part of *text* (a path, ``REV:path`` or ``--opt=path``) names a secret file."""
    names = [p.lower() for p in re.split(r"[/:=\\]", text) if p]
    return any(
        fnmatch.fnmatchcase(name, pattern[len(_ANYWHERE) :].lower())
        for pattern in _CREDENTIALS
        if pattern.startswith(_ANYWHERE)
        for name in names
    )


def where(ctx: Context, path: str, *, write: bool) -> str:
    """self, credentials, persistence, system, harmless, root, inside or outside."""
    home, root = ctx.home.rstrip("/"), ctx.root
    places = _places(home, root, _SOURCE)
    if write and any(under(path, p) for p in (ctx.prefix, *places["self"])):
        return "self"
    if write and any(under(path, p) for p in places["persistence"]):
        return "persistence"
    inside = under(path, root)
    secret = any(under(path, p) for p in places["credentials"])
    if secret or (os.path.basename(path) == ".env" and not inside):
        return "credentials"
    if inside:
        return "root" if path in (root, f"{root}/.git") else "inside"
    # A temporary folder holds no home folder (but a test's might).
    temp = not under(path, home) and any(under(path, p) for p in places["temp"])
    if temp or any(under(path, p) for p in places["caches"]):
        return "harmless"
    if any(under(path, p) for p in _SYSTEM) and not under(path, "/usr/local"):
        return "system"
    return "outside"


def shown(ctx: Context, path: str) -> str:
    """*path* for the person: relative inside the workspace, ``~/…`` in the home."""
    if under(path, ctx.root) and path != ctx.root:
        return os.path.relpath(path, ctx.root)
    return "~" + path[len(ctx.home.rstrip("/")) :] if under(path, ctx.home) else path


def touch(ctx: Context, f: Findings, raw: str, op: str, cwd: str | None = None) -> None:
    """Record what a ``read``, ``write`` or ``delete`` (*op*) of *raw* means."""
    if (raw, op, cwd) in f.seen:
        return
    f.seen.add((raw, op, cwd))
    path = resolve(ctx, raw, cwd)
    if path is None:
        if op != "read":
            f.doubt(f"Changes a path melo cannot resolve: {raw}")
        return
    kind, name = where(ctx, path, write=op != "read"), shown(ctx, path)
    if kind == "credentials":
        f.add("credentials", f"Uses {name}, which holds passwords or keys.")
    elif op == "read" or kind in ("harmless", "inside"):
        return
    elif kind == "self" and under(path, _SOURCE):
        f.add(kind, _SAY["source"].format(name))
    elif kind in _SAY:
        f.add(kind, _SAY[kind].format(name))
    elif kind == "root":
        if op == "delete":
            f.add("delete", f"Deletes {name}: the workspace or its git history.")
    else:
        verb = "Deletes" if op == "delete" else "Writes"
        category = "delete" if op == "delete" else "outside_write"
        f.add(category, f"{verb} {name}, outside {ctx.folder}.")
        f.paths.append(path)
