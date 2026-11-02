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
# What ``diskutil apfs``, ``coreStorage`` and ``appleRAID`` can delete or erase.
_APFS_VERBS: Final = frozenset(
    ["deletecontainer", "deletevolume", "erasevolume", "resizecontainer"]
    + ["delete", "deletelvg", "deletelv", "destroy"]
)
_DISK_GROUPS: Final = frozenset(["apfs", "ap", "corestorage", "cs", "appleraid", "ar"])

# git's options that take a value before the subcommand.
_GIT_VALUED: Final = frozenset(
    ["-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"]
    + ["--super-prefix", "--attr-source"]
)
# Words of ``git checkout`` that make it overwrite files or ask what to keep.
_CHECKOUT_FORCES: Final = frozenset(
    ["--force", "--ours", "--theirs", "--merge", "--patch", "--discard-changes"]
)
# The letters of ``git clean``, ``restore`` and ``switch`` that take a value.
_CLEAN_VALUED: Final = "e"
_RESTORE_VALUED: Final = "sU"
_SWITCH_VALUED: Final = "cC"
_CHECKOUT_LETTERS: Final = frozenset("fpm")
# The options of ``git checkout`` / ``switch`` that name a new branch.
_NEW_BRANCH: Final = frozenset(["-b", "-B", "--orphan", "-c", "-C"])
# ``kill`` and its relatives: signal 0 only asks whether the process exists.
_SIGNAL_WORDS: Final = frozenset(["-s", "-n", "--signal"])
# The options of each that take a value in the next word, and the signals'
# names (a name is one word of its own, not a cluster of letters).
_VALUED_WORDS: Final[dict[str, frozenset[str]]] = {
    "killall": frozenset(["-u", "-t", "-c"]),
}
_PKILL_VALUED: Final = "FGPUdgstu"
_SIGNAL_NAMES: Final = frozenset(
    ["HUP", "INT", "QUIT", "ILL", "TRAP", "ABRT", "EMT", "FPE", "KILL", "BUS"]
    + ["SEGV", "SYS", "PIPE", "ALRM", "TERM", "URG", "STOP", "TSTP", "CONT"]
    + ["CHLD", "TTIN", "TTOU", "IO", "XCPU", "XFSZ", "VTALRM", "PROF", "WINCH"]
    + ["INFO", "USR1", "USR2", "EXIT"]
)
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


def _options(args: Sequence[str], valued: str = "") -> tuple[set[str], list[str]]:
    """The letters of the single-dash options of *args*, and the plain words before ``--``.

    A cluster is read left to right: a letter of *valued* takes the rest of
    the cluster as its value (``-enode_modules``), or the next word when it
    ends the cluster (``-e node_modules``), so the letters of a value are
    never read as options.
    """
    letters: set[str] = set()
    words: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg == "--":
            break
        elif arg.startswith("-") and not arg.startswith("--") and len(arg) > 1:
            for at, char in enumerate(arg[1:], 1):
                letters.add(char)
                if char in valued:
                    skip = at == len(arg) - 1
                    break
        elif not arg.startswith("-"):
            words.append(arg)
    return letters, words


def _paths(args: Sequence[str], valued: str = "") -> tuple[str, ...]:
    """The path words of a git command: those after ``--``, else every plain word."""
    if "--" in args:
        return tuple(args[list(args).index("--") + 1 :])
    return tuple(_options(args, valued)[1])


def _abbreviates(arg: str, option: str) -> bool:
    """Whether *arg* is *option* or an unambiguous shortening git accepts (``--har``)."""
    return arg == option or (len(arg) >= 5 and option.startswith(arg))


def _checkout(args: list[str]) -> Discard | None:
    """``git checkout`` that overwrites files: paths, ``.``, ``-f``, ``--ours``, ``-p``."""
    reason = "Discards uncommitted changes (git checkout)."
    if "--" in args and args.index("--") + 1 < len(args):
        return Discard(reason, tuple(args[args.index("--") + 1 :]))
    if _CHECKOUT_FORCES & set(args) or any(
        a.startswith("--pathspec-from-file") for a in args
    ):
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
    letters, _ = _options(args, _RESTORE_VALUED)
    staged = "--staged" in args or "S" in letters
    worktree = "--worktree" in args or "W" in letters
    if staged and not worktree:
        return None
    return Discard(
        "Discards uncommitted changes (git restore).", _paths(args, _RESTORE_VALUED)
    )


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
        letters, _ = _options(rest, _SWITCH_VALUED)
        if {"--force", "--discard-changes"} & set(rest) or "f" in letters:
            return Discard("Discards uncommitted changes (git switch).")
        return None
    if sub == "restore":
        return _restore(rest)
    if sub == "reset":
        if any(_abbreviates(a, "--hard") for a in rest):
            return Discard("Resets with --hard, which discards uncommitted changes.")
        return None
    if sub == "clean":
        letters, _ = _options(rest, _CLEAN_VALUED)
        if "--dry-run" in rest or "n" in letters:
            return None
        paths = _paths(rest, _CLEAN_VALUED)
        if "x" in letters or "X" in letters:
            return Discard("Deletes untracked and ignored files (git clean -x).", paths)
        return Discard("Deletes untracked files (git clean).", paths)
    if sub == "stash":
        verb = next((a for a in rest if not a.startswith("-")), "")
        if verb in ("drop", "clear"):
            return Discard(f"Deletes stashed work (git stash {verb}).")
    return None


def _leading(prog: str, args: Sequence[str]) -> tuple[list[str], list[str]]:
    """The options before the first target of ``kill``, ``pkill`` or ``killall``, and the targets.

    These programs stop reading options at the first target: a ``-0`` or
    ``-s`` after it is one more target, not an option. A ``--`` ends the
    options. An option that takes a value takes the next word as well. For
    ``kill`` only the first word can be an option (a signal, ``-s sig``,
    ``-l``): a later ``-123`` is a process group.
    """
    if prog == "kill":
        first = args[0] if args else ""
        if first in _SIGNAL_WORDS:
            options, at = list(args[:2]), 2
        elif first.startswith("-") and len(first) > 1 and first != "--":
            options, at = [first], 1
        else:
            options, at = [], 0
        if args[at : at + 1] == ["--"]:
            at += 1
        return options, list(args[at:])
    valued = _VALUED_WORDS.get(prog, frozenset())
    options = []
    at = 0
    while at < len(args):
        word = args[at]
        if word == "--":
            at += 1
            break
        if not word.startswith("-") or len(word) < 2:
            break
        options.append(word)
        at += 1
        takes = word in valued
        if prog != "killall" and word[1:2] != "-" and not _is_signal(word):
            # pkill and its relatives: a cluster of letters, one of which may
            # take the rest of the word, or the next word when it ends it.
            for place, char in enumerate(word[1:], 1):
                if char in _PKILL_VALUED:
                    takes = place == len(word) - 1
                    break
        if takes and at < len(args):
            options.append(args[at])
            at += 1
    return options, list(args[at:])


def _is_signal(word: str) -> bool:
    """Whether *word* is a signal given as an option: ``-9``, ``-HUP``, ``-SIGKILL``."""
    name = word[1:]
    return name.isdigit() or name.removeprefix("SIG") in _SIGNAL_NAMES


def _signal_zero(options: Sequence[str], valued: bool) -> bool:
    """Whether the signal is ``0``: ``-0``, or (for ``kill``) ``-s 0`` / ``-n 0``.

    For ``pkill`` and ``killall`` the ``-s`` option is something else (a
    session, print only), so only ``-0`` counts there.
    """
    for at, option in enumerate(options):
        if option == "-0":
            return True
        if valued and option in _SIGNAL_WORDS and options[at + 1 : at + 2] == ["0"]:
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

    Only the options before the first target count (see :func:`_leading`).
    Signal ``0`` (an existence check), ``kill -l`` and ``killall -l`` (list
    the signals) and ``killall -s`` (print only) stay ``None``; ``pkill -l``
    is long output and still kills. ``kill`` of ``$!``, a job (``%1``), the
    step's own shell (``$$``) or a variable *started* holds (``pid=$!``) is a
    process this step started; any other target is not.
    """
    if prog not in ("kill", "pkill", "killall", "skill", "killall5"):
        return None
    options, targets = _leading(prog, args)
    if _signal_zero(options, prog == "kill"):
        return None
    if prog == "kill":
        if {"-l", "-L", "--list"} & set(options) or not targets:
            return None
        if all(_started(word, started) for word in targets):
            return None
        return "Signals a process this step did not start (kill)."
    if prog == "killall" and {"-l", "-s"} & set(options):
        return None
    return f"Stops other programs by signalling them ({prog})."
