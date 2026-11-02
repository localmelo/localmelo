"""Commands that destroy data, discard work or stop other programs.

Each function reads the arguments of one command (the program already found,
wrappers such as ``sudo`` or ``env`` already taken off) and says in a sentence
what it would destroy, or returns ``None`` for a command that only reads or
does something recoverable. When the arguments leave it open, a function
answers as the stricter reading.

- :func:`diskutil_erases`: ``diskutil eraseDisk``, ``partitionDisk``, …;
- :func:`git_discards`: ``git checkout -- <paths>``, ``git restore``,
  ``git reset --hard``, ``git clean``, ``git stash drop``, …, with the paths
  they name (:func:`git_discards` answers a :class:`Discard`);
- :func:`stops_others`: ``pkill``, ``killall`` and ``kill`` of a process this
  step did not start.

The category is the caller's: erasing a disk and stopping other programs are
``system`` changes, discarding work is ``delete``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

# diskutil's verbs that erase, repartition or overwrite what a disk or a
# volume holds (they are not case sensitive).
_DISK_VERBS: Final = frozenset(
    ["erasedisk", "erasevolume", "reformat", "partitiondisk", "zerodisk"]
    + ["randomdisk", "secureerase", "mergepartitions", "splitpartition"]
    + ["resizevolume"]
)
# What ``diskutil apfs`` and ``diskutil coreStorage`` can delete or erase.
_APFS_VERBS: Final = frozenset(
    ["deletecontainer", "deletevolume", "erasevolume", "resizecontainer"]
    + ["delete", "deletelvg", "deletelv"]
)
_DISK_GROUPS: Final = frozenset(["apfs", "ap", "corestorage", "cs"])

# git's options that take a value before the subcommand.
_GIT_VALUED: Final = frozenset(["-C", "-c", "--git-dir", "--work-tree", "--namespace"])
# Words of ``git checkout`` that make it overwrite files or ask what to keep.
_CHECKOUT_FORCES: Final = frozenset(
    ["--force", "--ours", "--theirs", "--merge", "--patch", "--discard-changes"]
)
_CHECKOUT_LETTERS: Final = frozenset("fpm")
# The options of ``git checkout`` / ``switch`` that name a new branch.
_NEW_BRANCH: Final = frozenset(["-b", "-B", "--orphan", "-c", "-C"])
# ``kill`` and its relatives: signal 0 only asks whether the process exists.
_SIGNAL_WORDS: Final = frozenset(["-s", "-n", "--signal"])
_DOLLAR_BANG: Final = re.compile(r"\$!|\$\{!\}")
_SHELL_PID: Final = re.compile(r"\$\$|\$\{\$\}")
_VARIABLE: Final = re.compile(r"\$\{?([A-Za-z_]\w*)\}?")


@dataclass(frozen=True)
class Discard:
    """What a git command discards: the sentence, and the paths it names."""

    reason: str
    paths: tuple[str, ...] = ()


def diskutil_erases(args: Sequence[str]) -> str | None:
    """The sentence for a ``diskutil`` call that erases or repartitions, else ``None``.

    ``list``, ``info`` and the other verbs that only read or mount stay
    ``None``. A group (``apfs``, ``coreStorage``) is read for its own verbs.
    """
    words = [a for a in args if not a.startswith("-")]
    if not words:
        return None
    verb = words[0].casefold()
    if verb in _DISK_GROUPS and len(words) > 1:
        sub = words[1].casefold()
        if sub in _APFS_VERBS:
            return f"Erases or deletes what a disk holds (diskutil {verb} {words[1]})."
        return None
    if verb in _DISK_VERBS:
        return f"Erases or repartitions a disk (diskutil {words[0]})."
    return None


def git_sub(args: Sequence[str]) -> tuple[str, list[str]]:
    """The subcommand of a ``git`` call, past its options, and the words after it."""
    at = 0
    while at < len(args):
        word = args[at]
        if word in _GIT_VALUED:
            at += 2
        elif word.startswith("-"):
            at += 1
        else:
            return word, list(args[at + 1 :])
    return "", []


def _letters(args: Sequence[str]) -> set[str]:
    """The letters of the single-dash options in *args* (``-fd`` gives ``f`` and ``d``)."""
    found: set[str] = set()
    for arg in args:
        if arg.startswith("-") and not arg.startswith("--") and len(arg) > 1:
            found |= set(arg[1:])
    return found


def _paths(args: Sequence[str]) -> tuple[str, ...]:
    """The path words of a git command: those after ``--``, else every plain word."""
    if "--" in args:
        return tuple(args[list(args).index("--") + 1 :])
    return tuple(a for a in args if not a.startswith("-"))


def _checkout(args: list[str]) -> Discard | None:
    """``git checkout`` that overwrites files: paths, ``.``, ``-f``, ``--ours``, ``-p``."""
    reason = "Discards uncommitted changes (git checkout)."
    if "--" in args and args.index("--") + 1 < len(args):
        return Discard(reason, tuple(args[args.index("--") + 1 :]))
    if _CHECKOUT_FORCES & set(args):
        return Discard(reason)
    for arg in args:
        if arg.startswith("-") and not arg.startswith("--"):
            for letter in arg[1:]:
                if letter in "bB":
                    break  # the rest of the word is the new branch's name
                if letter in _CHECKOUT_LETTERS:
                    return Discard(reason)
    if any(a in _NEW_BRANCH or a.startswith(("-b", "-B")) for a in args):
        return None  # a new branch from the current files overwrites nothing
    plain = [a for a in args if not a.startswith("-")]
    if len(plain) >= 2 or "." in plain:
        # A tree-ish and paths, or the whole tree: the paths are overwritten.
        return Discard(reason, tuple(plain if "." in plain else plain[1:]))
    return None


def _restore(args: list[str]) -> Discard | None:
    """``git restore`` that writes the working tree: all but ``--staged`` alone."""
    letters = _letters(args)
    staged = "--staged" in args or "S" in letters
    worktree = "--worktree" in args or "W" in letters
    if staged and not worktree:
        return None
    return Discard("Discards uncommitted changes (git restore).", _paths(args))


def git_discards(args: Sequence[str], exists: bool = False) -> Discard | None:
    """What a ``git`` call discards, or ``None`` for one that cannot lose work.

    Covers ``checkout`` with paths, ``.``, ``-f``, ``--ours``/``--theirs`` or
    ``-p``; ``switch -f``; ``restore`` unless it only unstages; ``reset
    --hard``; ``clean`` unless it is a dry run; ``stash drop`` and ``stash
    clear``. ``exists`` says the one plain word of a ``checkout`` names a file
    that is there (git reads it as a path then, not a branch).
    """
    sub, rest = git_sub(args)
    if sub == "checkout":
        found = _checkout(rest)
        if found is None and exists:
            plain = [a for a in rest if not a.startswith("-")]
            if len(plain) == 1 and not any(a in _NEW_BRANCH for a in rest):
                return Discard(
                    "Discards uncommitted changes (git checkout).", (plain[0],)
                )
        return found
    if sub == "switch":
        if {"--force", "--discard-changes"} & set(rest) or "f" in _letters(rest):
            return Discard("Discards uncommitted changes (git switch).")
        return None
    if sub == "restore":
        return _restore(rest)
    if sub == "reset":
        if "--hard" in rest:
            return Discard("Resets with --hard, which discards uncommitted changes.")
        return None
    if sub == "clean":
        if "--dry-run" in rest or "n" in _letters(rest):
            return None
        if any(re.fullmatch(r"-\w*[xX]\w*", a) for a in rest):
            return Discard(
                "Deletes untracked and ignored files (git clean -x).", _paths(rest)
            )
        return Discard("Deletes untracked files (git clean).", _paths(rest))
    if sub == "stash" and rest[:1] in (["drop"], ["clear"]):
        return Discard(f"Deletes stashed work (git stash {rest[0]}).")
    return None


def _signal_zero(args: Sequence[str]) -> bool:
    """Whether the signal is ``0`` (``-0``, ``-s 0``, ``-n 0``): no signal is sent."""
    for at, arg in enumerate(args):
        if arg == "-0":
            return True
        if arg in _SIGNAL_WORDS and args[at + 1 : at + 2] == ["0"]:
            return True
    return False


def _started(word: str, started: set[str]) -> bool:
    """Whether a target of ``kill`` is a process this step started or the step itself."""
    text = word.strip("\"'")
    if (
        text.startswith("%")
        or _DOLLAR_BANG.fullmatch(text)
        or _SHELL_PID.fullmatch(text)
    ):
        return True
    named = _VARIABLE.fullmatch(text)
    return named is not None and named.group(1) in started


def stops_others(prog: str, args: Sequence[str], started: set[str]) -> str | None:
    """The sentence for ``pkill``, ``killall`` or a ``kill`` of another process, else ``None``.

    Signal ``0`` (an existence check), listing the signals (``-l``) and
    ``killall -s`` (print only) stay ``None``. ``kill`` of ``$!``, a job
    (``%1``), the step's own shell (``$$``) or a variable *started* holds
    (``pid=$!``) is a process this step started; any other target is not.
    """
    if prog in ("pkill", "killall", "skill", "killall5"):
        if _signal_zero(args) or {"-l", "--list", "-L"} & set(args):
            return None
        if prog == "killall" and "-s" in args:
            return None
        return f"Stops other programs by signalling them ({prog})."
    if prog != "kill":
        return None
    if _signal_zero(args) or {"-l", "-L"} & set(args):
        return None
    targets = list(args)
    while (
        targets and targets[0].startswith("-") and not re.fullmatch(r"-\d+", targets[0])
    ):
        # A signal's name (-KILL, -TERM) or -s / -n with its value.
        takes = targets[0] in _SIGNAL_WORDS
        targets = targets[2:] if takes else targets[1:]
    if targets[:1] and re.fullmatch(r"-\d+", targets[0]) and len(targets) > 1:
        targets = targets[1:]  # -9 is the signal when a target follows
    if targets[:1] == ["--"]:
        targets = targets[1:]
    if not targets:
        return None
    if all(_started(word, started) for word in targets):
        return None
    return "Signals a process this step did not start (kill)."
