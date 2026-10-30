"""What the review knows about one call, what it found, and where a path lies."""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass, field

from localmelo.melo.contracts.review import TITLES, ActionReview

# Under the home folder. Reads count too: passwords, keys and cookies.
_CREDENTIALS = [
    ".ssh",
    ".aws",
    ".netrc",
    ".config/gh",
    ".docker/config.json",
    ".gnupg",
    ".kube",
    ".pypirc",
    ".password-store",
    "Library/Keychains",
    "Library/Cookies",
    "Library/Safari",
    "Library/Application Support/Google/Chrome",
    "Library/Application Support/Firefox",
    ".cache/localmelo",
]
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
_APP = ("/Applications/LocalMelo.app",)
_LAUNCHD = ("/Library/LaunchAgents", "/Library/LaunchDaemons")
_SAY = {
    "self": "Changes {}, part of localmelo itself.",
    "persistence": "Changes {}, which makes something run later on its own.",
    "system": "Changes {}, a system folder.",
}


@dataclass(frozen=True)
class Context:
    """The call's surroundings: the workspace, the home folder, melo's Python."""

    root: str
    home: str
    executable: str
    prefix: str
    depth: int = 0

    @property
    def folder(self) -> str:
        return os.path.basename(self.root.rstrip("/")) or self.root

    def deeper(self) -> Context:
        return Context(
            self.root, self.home, self.executable, self.prefix, self.depth + 1
        )


@dataclass
class Findings:
    """High-impact categories with their reasons; the undecided reasons."""

    high: dict[str, list[str]] = field(default_factory=dict)
    unsure: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    seen: set[tuple[str, str, str | None]] = field(default_factory=set)

    def add(self, category: str, reason: str) -> None:
        self.high.setdefault(category, [])
        if reason not in self.high[category]:
            self.high[category].append(reason)

    def doubt(self, reason: str) -> None:
        if reason not in self.unsure:
            self.unsure.append(reason)

    def review(self) -> ActionReview:
        def cut(text: str) -> str:
            return text if len(text) <= 120 else text[:119] + "…"

        if not self.high:
            unsure = tuple(map(cut, self.unsure[:3]))
            return ActionReview("undecided" if unsure else "routine", (), unsure)
        cats = tuple(sorted(self.high, key=list(TITLES).index))
        reasons = tuple(dict.fromkeys(cut(r) for c in cats for r in self.high[c]))
        return ActionReview("high", cats, reasons[:3], tuple(dict.fromkeys(self.paths)))


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


@functools.lru_cache(maxsize=32)
def _places(home: str, root: str) -> dict[str, tuple[str, ...]]:
    """The folders of :func:`where` for one home and workspace."""
    later = [*(os.path.join(home, p) for p in _PERSISTENCE), f"{root}/.git/hooks"]
    return {
        "self": (f"{home}/.cache/localmelo", *_APP, f"{home}{_APP[0]}"),
        "persistence": (*later, f"{root}/.envrc", *_LAUNCHD),
        "credentials": tuple(os.path.join(home, p) for p in _CREDENTIALS),
        "temp": tuple(os.path.realpath(p) for p in _HARMLESS),
        "caches": tuple(os.path.join(home, p) for p in _CACHES),
    }


def where(ctx: Context, path: str, *, write: bool) -> str:
    """self, credentials, persistence, system, harmless, root, inside or outside."""
    home, root = ctx.home.rstrip("/"), ctx.root
    places = _places(home, root)
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
