"""The one plain sentence an approval card starts with (#454).

:func:`step_gist` says what a call will do. For a command or a script the
sentence is built from the call itself and names every effect it finds, in
order ("Operate Safari, then delete files"); a command too long to show says
how many more there are ("... and 2 more"). Only the model's own words about
a call (a ``why`` argument, or the leading comment of its code or command)
are ever shown beside it, always labelled as the model's:

- a routine step with words: the words are the sentence;
- a step the rules found risky: the sentence stays the code's, and the words
  go below it as a note, so they can never make the step look safer;
- words that clearly disagree with what the code does (it deletes files, it
  sends data out): a warning that quotes them. When the words do agree on the
  effect but name a file, folder or host, and none of the places the code
  deletes or sends to is what they name, a second warning says where the code
  acts (``delete_target``, ``send_out_target``).

The checks are a few clear patterns on a program's name or a call in the
syntax tree, never on text in a string.

A :class:`Gist` holds the ids of catalog entries (``gist.say.<id>``,
``gist.warn.<id>``) and their parameters, so every surface says it in its own
language. Imports only the standard library, the redaction, the sibling rules
and the contracts.
"""

from __future__ import annotations

import ast
import copy
import re
import shlex
import unicodedata
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal
from urllib.parse import urlsplit

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.checker.review import apps as app_names
from localmelo.melo.checker.review import destroys, targets
from localmelo.melo.checker.review.paths import loopback
from localmelo.melo.contracts import shell_lexer
from localmelo.melo.contracts.review import MAX_TEXT, ActionReview, command_steps

# The argument that carries the model's own words about a call. Another
# argument (``description``, ``purpose``) is often the content a tool sends,
# so only ``why`` counts.
WORD_FIELDS: Final = ("why",)
_RUNNERS: Final = frozenset(["shell_exec", "python_exec"])
# The longest sentence of the model's words, and of a command in a code-built one.
WORDS_CHARS: Final = 120
COMMAND_CHARS: Final = 80
# The most parts a code-built sentence lists before it says "and N more".
MAX_STEPS: Final = 4
# The ids of the sentences and warnings; the catalog holds one entry for each
# (``gist.say.<id>``, ``gist.warn.<id>``). ``then`` and ``more`` hold other
# sentences in their parameters.
SAY_IDS: Final = (
    "model",
    "command",
    "script",
    "script_command",
    "write",
    "edit",
    "tool",
    "operate",
    "effect_delete",
    "effect_send_out",
    "script_effect_delete",
    "script_effect_send_out",
    "then",
    "more",
)
WARN_IDS: Final = ("delete", "send_out", "delete_target", "send_out_target")
# The most places a target warning lists before it says "…".
MAX_TARGETS: Final = 3
# The longest a place in a target warning is.
TARGET_CHARS: Final = 60
# Sentences built by code that a card's question ("Create notes.md?") already says.
PLAIN_IDS: Final = frozenset(["write", "edit", "tool"])

_CJK: Final = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
# A comment that is a pragma, a shebang or a header, not a description.
_NOT_WORDS: Final = re.compile(
    r"(?i)^(?:!|noqa\b|type:|pylint\b|pragma\b|fmt:|isort\b|mypy\b|-\*-|coding[:=]"
    r"|todo\b|fixme\b|xxx\b)"
)
# A commented-out line of code is not a description.
_CODE_LIKE: Final = re.compile(
    r"^[\w.\[\]'\"]+\s*[-+*/]?=[^=]|^(?:import|from|def|class|return|print|if|for)\b"
    r"|\(\s*\)|\w\(.*\)$"
)
# The words of a description that cover what the code does, per effect. Only
# words that say the same thing: "copy" does not cover sending, "reset" or
# "clear" does not cover deleting.
_AGREES: Final[dict[str, re.Pattern[str]]] = {
    "delete": re.compile(
        r"(?i)\b(?:delet|remov|eras|clean|purg|prun|trash|discard|destroy"
        r"|uninstall|unlink|overwrit)\w*|\brm\b|\bwip(?:e|es|ed|ing)\b"
        r"|\bdrop(?:s|ped|ping)?\b"
        r"|删|移除|清理|擦除|丢弃|卸载|清掉|覆盖"
    ),
    "send_out": re.compile(
        r"(?i)\b(?:send|upload|push|submit|transmit|publish|e-?mail|mail"
        r"|back\s?up|export|deliver|notify|transfer|deploy)\w*"
        r"|\bpost(?:s|ed|ing)?\b"
        r"|发送|发给|上传|提交|推送|发布|邮件|备份|导出|通知|传输|部署"
    ),
}
_EFFECTS: Final = ("delete", "send_out")
# What a command clearly does, strongest first, as the card's titles go
# (TITLES: controlling an app, deleting, sending data out).
_STRONGEST: Final = ("operate", *_EFFECTS)
# What an honest description of a git command that discards work says besides
# "delete": reverting, restoring, resetting, rolling back. These cover only a
# step that is nothing but such commands (checkout, restore, switch, stash):
# those were not read as deleting before, so no older warning is hidden. A
# ``git clean`` or a ``reset --hard`` was, so only the words for deleting cover it.
_GIT_AGREES: Final = re.compile(
    r"(?i)\b(?:revert|restor|reset|undo|roll\s?back|throw\s+away|check\s?out"
    r"|clear|drop|abandon|reject)\w*|还原|撤销|回滚|放弃|重置|恢复"
)
# Words that start a command line and run the command after them: the
# wrappers (with the options of each that take a value, so the value is not
# read as the program), the shell keywords and a subshell's parenthesis.
_WRAPPERS: Final = frozenset([*shell_lexer.WRAPPERS, *shell_lexer.KEYWORDS, "("])
# The ends of a block, which name no command of their own.
# ``command -v rm``: a lookup of a program, not a run of it.
_LOOKUP: Final = re.compile(r"-[pvV]*[vV][pvV]*")
_CLOSERS: Final = frozenset(["done", "fi", "esac", "}", ")", ";;"])
_DELETE_PROGRAMS: Final = frozenset(["rm", "rmdir", "unlink", "shred", "srm", "trash"])
_SEND_PROGRAMS: Final = frozenset(
    ["mail", "mailx", "sendmail", "msmtp", "sftp", "ftp", "lftp"]
)
_SOCKET_PROGRAMS: Final = frozenset(["nc", "ncat", "netcat", "socat"])
_CURL_LONG_SENDS: Final = re.compile(
    r"^--(?:data(?:-[\w]+)?|form(?:-string)?|upload-file|json)$"
)
# curl's short options that take a value (the rest of a cluster is that value).
_CURL_VALUED: Final = frozenset("AbcCdDeEFHKmoPrTuUwxXyYz")
_METHOD_SENDS: Final = frozenset(["POST", "PUT", "PATCH", "DELETE"])
_REMOTE: Final = re.compile(r"^(?:[\w.-]+@)?[\w.-]+:(?!//)")
_BARE_LOCAL: Final = re.compile(
    r"^(?:localhost|127(?:\.\d{1,3}){3}|\[::1\]|0\.0\.0\.0)(?::\d+)?(?:/|$)"
)
_PY_SENDERS: Final = frozenset(["requests", "httpx", "session", "client", "aiohttp"])
_PY_RUNS: Final = frozenset(
    ["run", "call", "check_call", "check_output", "Popen", "system", "popen"]
    + ["getoutput", "getstatusoutput"]
)
_PY_SOCKETS: Final = frozenset(["sendall", "sendto"])
_SUBSTITUTION: Final = re.compile(r"\$\(|\$\{|`")
# A disk or volume a disk command names: disk2, disk2s1, /dev/rdisk3, /Volumes/Backup.
_DISK: Final = re.compile(r"(?:/dev/)?r?disk\d+(?:s\d+)*|/Volumes/[^\s]+")
# git options before the subcommand that take a value.
_GIT_VALUED: Final = frozenset(
    ["-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"]
    + ["--super-prefix", "--attr-source"]
)
# The git commands that discard work and were not read as deleting before.
_GIT_UNREAD: Final = frozenset(["checkout", "restore", "switch", "stash"])


@dataclass(frozen=True)
class Gist:
    """What a call does, as catalog ids and parameters.

    ``id`` names the sentence (``gist.say.<id>``) and ``by`` says who wrote
    it: ``model`` (its own words, in ``params["text"]``; only for a routine
    step) or ``code``. A parameter may itself be a sentence
    (``{"id", "params"}``, as in ``then`` and ``more``) or a list of names
    (``apps``). A ``warn`` id (``gist.warn.<id>``) is set when the model's
    words clearly disagree with the code: the sentence is then the code's own
    and ``warn_params["said"]`` holds the words. ``note`` holds the model's
    words when the rules found the step risky: they stand below the sentence,
    labelled as the model's. ``apps`` names the apps a script operates,
    whoever wrote the sentence, so a reason line can name them. ``attach``
    has the file, its folder, the site and whether the person named the file
    for a question about attaching a file to a page (#551), so the reason
    line says them in the page's own language.
    """

    id: str
    params: dict[str, Any] = field(default_factory=dict)
    by: Literal["model", "code"] = "code"
    warn: str = ""
    warn_params: dict[str, str] = field(default_factory=dict)
    note: str = ""
    apps: tuple[str, ...] = ()
    attach: dict[str, Any] = field(default_factory=dict)

    @property
    def plain(self) -> bool:
        """Whether the card's own question says as much (a write, an edit, a tool)."""
        return self.by == "code" and not self.warn and self.id in PLAIN_IDS

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "say": {"id": self.id, "params": copy.deepcopy(self.params)},
            "by": self.by,
        }
        if self.warn:
            out["warn"] = {"id": self.warn, "params": dict(self.warn_params)}
        if self.note:
            out["note"] = self.note
        if self.apps:
            out["apps"] = list(self.apps)
        if self.attach:
            out["attach"] = dict(self.attach)
        return out


@dataclass(frozen=True)
class _Item:
    """One thing a call does: ``operate`` an app, ``delete``, ``send_out`` or run a ``command``."""

    kind: str
    text: str = ""
    apps: tuple[str, ...] = ()
    # An effect only the rules saw, in a script whose code shows nothing: the
    # sentence says it is a script that does it.
    script: bool = False
    # The places an effect acts on as written (deleted paths, hosts sent to),
    # and whether some place melo cannot read (a variable, the folder itself).
    places: tuple[str, ...] = ()
    blind: bool = False
    # A delete that is a git command discarding work and was not read as a
    # delete before (see :data:`_GIT_AGREES`).
    git: bool = False


def step_gist(
    tool_name: str, arguments: Mapping[str, Any], review: ActionReview | None = None
) -> Gist | None:
    """The sentence for a call of *tool_name*, or ``None`` for a tool whose card has its own.

    The browser tools (``browser:<site>``) ask with a sentence of their own.
    """
    if tool_name.startswith("browser:"):
        return None
    if tool_name.startswith("browser-attach:"):
        return _attach(tool_name, arguments)
    words = _words(tool_name, arguments)
    # A text over the limit is not read by the rules either (see MAX_TEXT).
    items = (
        _items(tool_name, arguments)
        if tool_name in _RUNNERS and len(_source(tool_name, arguments)) <= MAX_TEXT
        else []
    )
    for effect in _EFFECTS:
        # What the rules found that the code does not show (a string run with
        # exec, a command built at run time) is part of the sentence.
        if (
            review is not None
            and effect in review.categories
            and not any(item.kind == effect for item in items)
        ):
            items = [
                *items,
                _Item(
                    effect,
                    script=tool_name == "python_exec" and not items,
                    blind=True,
                ),
            ]
    gone, where = _mismatch(items, review, words) if words else ("", {})
    apps = _apps(review)
    if apps and not any(item.kind == "operate" for item in items):
        # The rules read an app out of a script the call builds from a variable.
        items = [*items, _Item("operate", apps=apps)]
    apps = apps or tuple(dict.fromkeys(n for item in items for n in item.apps))
    built = replace(_built(tool_name, arguments, items), apps=apps)
    if gone:
        return replace(built, warn=gone, warn_params={"said": words, **where})
    if not words:
        return built
    if _risky(review):
        return replace(built, note=words)
    return replace(Gist("model", {"text": words}, by="model"), apps=apps)


def _attach(tool_name: str, arguments: Mapping[str, Any]) -> Gist:
    """The question about attaching a file to a page (#551): the card's own words stay, and the facts ride along.

    The file, its folder and the site are the ones the question names, each as
    one clean line; ``named`` is false for a file nobody vouched for (the
    person did not name it).
    """
    facts = {
        "file": _line(str(arguments.get("name") or "a file"), 80),
        "folder": _line(str(arguments.get("folder") or ""), 120),
        "site": _line(str(arguments.get("site") or "this site"), 80),
        "named": arguments.get("source") != "other",
    }
    return Gist("tool", {"tool": tool_name}, attach=facts)


def _risky(review: ActionReview | None) -> bool:
    """Whether the rules found anything: a step that is not routine, or has a category."""
    return review is not None and (
        review.verdict != "routine" or bool(review.categories)
    )


# ── The model's words ──


def _clean(text: str) -> str:
    """*text* with a secret redacted, format characters dropped and spacing as one line.

    The format characters (right-to-left marks, zero-width spaces: Unicode
    category ``Cf``) can reorder or hide what a line says, so they are
    dropped. Other control characters stay, for a surface to show escaped.
    """
    return " ".join(redact_secrets(_unformatted(text))[0].split())


def _unformatted(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _line(text: str, limit: int) -> str:
    """*text* as one clean line of at most *limit* characters."""
    text = _clean(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _sentence(text: str) -> bool:
    """Whether *text* reads as a description: a few characters with a letter."""
    if not any(ch.isalpha() for ch in text):
        return False
    return len(text) >= 4 or (len(text) >= 2 and _CJK.search(text) is not None)


def _words(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """The model's own words for the call, or ``""``."""
    for name in WORD_FIELDS:
        value = arguments.get(name)
        if isinstance(value, str):
            line = _line(value, WORDS_CHARS)
            if _sentence(line):
                return line
    return _comment(tool_name, arguments) if tool_name in _RUNNERS else ""


def _source(tool_name: str, arguments: Mapping[str, Any]) -> str:
    if tool_name == "python_exec":
        return str(arguments.get("code") or "")
    if tool_name == "shell_exec":
        command = arguments.get("command")
        if isinstance(command, list):
            return " ".join(str(part) for part in command)
        return str(command or "")
    return ""


def _comment(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """The first comment line of a script that starts with comments, as words."""
    for raw in _source(tool_name, arguments).splitlines():
        row = raw.strip()
        if not row:
            continue
        if not row.startswith("#"):
            return ""
        body = row.lstrip("#").strip()
        if _NOT_WORDS.match(body) or _CODE_LIKE.search(body):
            continue
        line = _line(body, WORDS_CHARS)
        return line if _sentence(line) else ""
    return ""


def _mismatch(
    items: Sequence[_Item], review: ActionReview | None, words: str
) -> tuple[str, dict[str, str]]:
    """How the model's *words* disagree with the code, as a warning id and its parameters.

    First the kind of effect: the code deletes or sends and the words do not
    say so. Then, for an effect the words do cover, the place: the words name
    a file, folder or host and none of the places the code acts on is it.
    ``("", {})`` when they agree, or when melo cannot tell.
    """
    found = {item.kind for item in items} & set(_EFFECTS)
    if review is not None:
        found |= set(_EFFECTS) & set(review.categories)
    for effect in _EFFECTS:
        if effect in found and not _agrees(effect, items, review, words):
            return effect, {}
    for effect in _EFFECTS:
        if effect in found:
            elsewhere = _elsewhere_target(effect, items, words)
            if elsewhere:
                return f"{effect}_target", {"target": elsewhere}
    return "", {}


def _agrees(
    effect: str, items: Sequence[_Item], review: ActionReview | None, words: str
) -> bool:
    """Whether the *words* say the step does *effect*."""
    if _AGREES[effect].search(words):
        return True
    # The git words (revert, restore…) cover a step that is nothing but git
    # commands that discard work, and in which the rules found nothing else.
    return (
        effect == "delete"
        and any(item.git for item in items)
        and all(item.git or _harmless(item) for item in items)
        and (review is None or set(review.categories) <= {"delete"})
        and _GIT_AGREES.search(words) is not None
    )


# Commands that cannot delete, which may stand beside a git discard.
_HARMLESS: Final = frozenset(
    ["cd", "pushd", "popd", "echo", "printf", "true", "pwd", "ls"]
)
_HARMLESS_GIT: Final = frozenset(["status", "diff", "log", "show", "add"])


def _harmless(item: _Item) -> bool:
    """Whether *item* is a command that cannot delete (``cd``, ``echo``, ``git status``)."""
    if item.kind != "command" or _SUBSTITUTION.search(item.text):
        return False
    words = item.text.split()
    if words[:1] == ["git"]:
        return _git_sub(words[1:])[0] in _HARMLESS_GIT
    return words[:1] != [] and words[0] in _HARMLESS


def _elsewhere_target(effect: str, items: Sequence[_Item], words: str) -> str:
    """The places the code acts on that the *words* do not name, as text, else ``""``.

    The words must name a place of the same kind (a file or folder, a host).
    Nothing is said when they name none, or when some place the code acts on
    cannot be read (a variable). Every place the code acts on has to be
    named: one that is not is the warning, even when another is.
    """
    mine = [item for item in items if item.kind == effect]
    if not mine or any(item.blind or not item.places for item in mine):
        return ""
    # After a `cd`, a bare `*` or `.` means a folder melo cannot name; before
    # it, it is the folder the step started in.
    moved = next(
        (
            at
            for at, item in enumerate(items)
            if item.kind == "command" and _changes_folder(item.text)
        ),
        len(items),
    )
    places = list(
        dict.fromkeys(
            place
            for at, item in enumerate(items)
            if item.kind == effect
            for place in item.places
            if effect != "delete" or at < moved or not targets.broad(place)
        )
    )
    if effect == "delete":
        named = targets.named_paths(words)
        matches = targets.path_matches
    else:
        named = targets.named_hosts(words)
        matches = targets.host_matches
    if not named or not places:
        return ""
    elsewhere = [
        place
        for place in places
        if not any(matches(place, name) for name in named)
        and not _spelt(effect, place, words)
    ]
    if not elsewhere:
        return ""
    shown = [_short_place(place) for place in elsewhere[:MAX_TARGETS]]
    return ", ".join(shown) + (", …" if len(elsewhere) > MAX_TARGETS else "")


def _changes_folder(text: str) -> bool:
    return text.split(None, 1)[:1] in (["cd"], ["pushd"], ["popd"])


def _spelt(effect: str, place: str, words: str) -> bool:
    """Whether the words write the place's own name (the "build" of "the build folder")."""
    if effect != "delete":
        return False
    base = place.rstrip("/").rsplit("/", 1)[-1]
    return bool(base) and not targets.broad(place) and targets.spells(words, base)


def _short_place(text: str) -> str:
    text = _clean(text)
    return text if len(text) <= TARGET_CHARS else text[: TARGET_CHARS - 1] + "…"


# ── What the code does ──


def _items(tool_name: str, arguments: Mapping[str, Any]) -> list[_Item]:
    """What a ``shell_exec`` or ``python_exec`` call does, in order."""
    if tool_name == "python_exec":
        return _python_items(str(arguments.get("code") or ""))
    found: list[_Item] = []
    for words, text in command_steps(tool_name, arguments):
        item = _classify(words, text)
        if item is not None:
            found.append(item)
    return found


def _classify(words: list[str], text: str = "") -> _Item | None:
    """What one command does: an effect, an app it operates, or just a command to name.

    *text* is the command as written; without it the words are quoted again.
    """
    if not words or words[0].startswith("#") or words[0] in _CLOSERS:
        return None
    readings = [_program(words)]
    found = shell_lexer.unwrap(words)
    if found.unread:  # a wrapper option melo could not read: any program it may run
        readings += [(shell_lexer.program(r[0]), r[1:]) for r in found.readings if r]
    items = [item for reading in readings if (item := _clear_item(*reading))]
    if items:
        return min(items, key=lambda item: _STRONGEST.index(item.kind))
    shown = text or shlex.join(words)
    return _Item("command", text=_unformatted(shown))


def _clear_item(program: str, rest: list[str]) -> _Item | None:
    """What *program* run with *rest* clearly does: an effect or an app it operates."""
    discard = destroys.git_discards(rest) if program == "git" else None
    effect = _command_effect(program, rest, discard)
    if effect:
        places, blind = _places(effect, program, rest, discard)
        return _Item(
            effect,
            places=places,
            blind=blind,
            git=program == "git" and _git_sub(rest)[0] in _GIT_UNREAD,
        )
    named = None
    if program == "osascript":
        named = app_names.osascript_app(rest)
    elif program == "open":
        named = app_names.open_app(rest)
    return _Item("operate", apps=(named,)) if named else None


def _program(words: list[str]) -> tuple[str, list[str]]:
    """The program a command runs, past wrappers (``sudo``, ``xargs``…), and its words."""
    at, wrapper = 0, ""
    while at < len(words):
        word = words[at]
        base = word.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if base == "command" and _LOOKUP.fullmatch(
            words[at + 1] if at + 1 < len(words) else ""
        ):
            return (
                base,
                words[at + 1 :],
            )  # command -v rm looks rm up, it does not run it
        opening = base.lstrip("({")  # a group: (rm x), { rm x; }
        options = shell_lexer.WRAPPERS.get(wrapper)
        if options is not None and word in options.valued:
            at += 2
        elif (
            word.startswith("-")
            or re.fullmatch(r"\d+(?:\.\d+)?[smhd]?", word)
            or shell_lexer.ASSIGNMENT.match(word)
            or base in _WRAPPERS
            or not opening
        ):
            wrapper = base if base in _WRAPPERS else wrapper
            at += 1
        elif opening != base:
            return opening, words[at + 1 :]
        else:
            return base, words[at + 1 :]
    return "", []


def _git_sub(rest: list[str]) -> tuple[str, list[str]]:
    """The subcommand of a ``git`` call, past its options, and the words after it."""
    at = 0
    while at < len(rest):
        word = rest[at]
        if word in _GIT_VALUED:
            at += 2
        elif word.startswith("-"):
            at += 1
        else:
            return word, rest[at + 1 :]
    return "", []


def _command_effect(
    program: str, rest: list[str], discard: destroys.Discard | None = None
) -> str:
    """``delete`` or ``send_out`` when a command clearly does it, else ``""``.

    *discard* is what :func:`~localmelo.melo.checker.review.destroys.git_discards`
    says of a ``git`` call (read once by the caller).
    """
    if program in _DELETE_PROGRAMS:
        return "delete"
    if program == "find" and ("-delete" in rest or _after_exec(rest, _DELETE_PROGRAMS)):
        return "delete"
    if program == "diskutil":
        return "delete" if destroys.diskutil_erases(rest) else ""
    if program == "git":
        sub, args = _git_sub(rest)
        if discard or (
            # as before: every clean not spelt as a dry run, whatever its options
            sub == "clean"
            and not {"-n", "--dry-run"} & set(args)
        ):
            return "delete"
        return "send_out" if sub == "push" else ""
    sends = (
        program == "curl"
        and (_curl_sends(rest) or _in_url(rest))
        or program == "wget"
        and (
            any(
                a.startswith(("--post-", "--body-")) or a.upper() == "--METHOD=POST"
                for a in rest
            )
            or _in_url(rest)
        )
    )
    if sends:
        return "" if _loopback_only(rest) else "send_out"
    if (
        program in ("scp", "rsync")
        and rest
        and _REMOTE.match(rest[-1])
        or program == "rclone"
        and rest[:1]
        and rest[0] in ("copy", "sync", "move", "copyto", "moveto")
        and _REMOTE.match(rest[-1])
        or program in _SEND_PROGRAMS
        or program in _SOCKET_PROGRAMS
        and not any(re.fullmatch(r"-[A-Za-z]*[lz][A-Za-z0-9]*", a) for a in rest)
        or program == "aws"
        and rest[:1] == ["s3"]
        and rest[1:2] in (["cp"], ["mv"], ["sync"])
        and rest[-1].startswith("s3://")
        or program == "gsutil"
        and rest[:1] in (["cp"], ["mv"], ["rsync"])
        and rest[-1].startswith("gs://")
        or program == "gh"
        and rest[:2] in (["gist", "create"], ["release", "upload"])
    ):
        return "send_out"
    return ""


def _path_places(paths: Sequence[str]) -> tuple[tuple[str, ...], bool]:
    """*paths* as places, and whether one cannot be read or none was given.

    A place that holds everything around it (``.``, ``~``, ``./*``) is a
    place: it is known, and it is more than any one file.
    """
    kept = [p for p in paths if targets.parts(p) is not None or targets.broad(p)]
    return tuple(kept), not paths or len(kept) != len(paths)


def _find_places(rest: list[str]) -> tuple[tuple[str, ...], bool]:
    """What a ``find … -delete`` removes: its name patterns, else its folders."""
    roots: list[str] = []
    for word in rest:
        if word.startswith(("-", "(", "!")):
            break
        roots.append(word)
    patterns = [
        rest[at + 1]
        for at, word in enumerate(rest[:-1])
        if word in ("-name", "-iname", "-path", "-ipath")
    ]
    if not patterns:
        return _path_places(roots)
    kept, blind = _path_places(patterns)
    inside, _ = _path_places([r for r in roots if targets.parts(r) is not None])
    return (*kept, *inside), blind


def _hosts_places(hosts: Sequence[str]) -> tuple[tuple[str, ...], bool]:
    named = [h for h in (targets.host_name(h) for h in hosts) if h]
    return tuple(dict.fromkeys(named)), not named or len(named) != len(hosts)


def _places(
    effect: str,
    program: str,
    rest: list[str],
    discard: destroys.Discard | None = None,
) -> tuple[tuple[str, ...], bool]:
    """Where a command's effect lands as written, and whether some place is unreadable.

    Deleted paths for ``delete``, hosts (or a bucket, a remote) for
    ``send_out``. A place melo cannot name (a variable, the current folder,
    a command that gives none) makes the effect *blind*: it is never compared
    with the words.
    """
    plain = [a for a in rest if not a.startswith("-")]
    if effect == "delete":
        if program in _DELETE_PROGRAMS:
            after = rest[rest.index("--") + 1 :] if "--" in rest else plain
            return _path_places(after)
        if program == "find":
            return _find_places(rest)
        if program == "git":
            return _path_places(discard.paths) if discard else ((), True)
        if program == "diskutil":
            disks = [a for a in plain if _DISK.fullmatch(a)]
            return tuple(disks), not disks
        return (), True
    if program in ("curl", "wget"):
        return _hosts_places([a for a in rest if "://" in a])
    if program in ("scp", "rsync"):
        return _hosts_places([a for a in rest if _REMOTE.match(a)])
    if program in _SOCKET_PROGRAMS or program in ("sftp", "ftp", "lftp"):
        first = [a for a in plain if not a.isdigit()][:1]
        return _hosts_places(first)
    if program in _SEND_PROGRAMS and program not in ("sftp", "ftp", "lftp"):
        return _hosts_places([a.rsplit("@", 1)[-1] for a in plain if "@" in a])
    if program in ("aws", "gsutil"):
        buckets = [a.split("://", 1)[1].split("/", 1)[0] for a in rest if "://" in a]
        return tuple(dict.fromkeys(b for b in buckets if b)), not buckets
    if program == "gh":
        return ("github.com",), False
    if program == "rclone" and rest and _REMOTE.match(rest[-1]):
        return (rest[-1].split(":", 1)[0],), False
    return (), True


def _after_exec(rest: list[str], programs: frozenset[str]) -> bool:
    """Whether ``-exec``/``-execdir`` of a ``find`` runs one of *programs*."""
    for at, word in enumerate(rest[:-1]):
        if word in ("-exec", "-execdir", "-ok"):
            base = rest[at + 1].replace("\\", "/").rsplit("/", 1)[-1].lower()
            if base in programs:
                return True
    return False


def _curl_sends(rest: list[str]) -> bool:
    """Whether curl's options send a body or use a method that changes the other end."""
    for at, arg in enumerate(rest):
        following = rest[at + 1] if at + 1 < len(rest) else ""
        if arg.startswith("--"):
            name, _, value = arg.partition("=")
            if _CURL_LONG_SENDS.match(name):
                return True
            if name == "--request" and (value or following).upper() in _METHOD_SENDS:
                return True
        elif arg.startswith("-") and len(arg) > 1:
            for place, ch in enumerate(arg[1:], 1):
                if ch in "dFT":
                    return True
                if ch == "X":
                    method = arg[place + 1 :] or following
                    if method.upper() in _METHOD_SENDS:
                        return True
                    break
                if ch in _CURL_VALUED:
                    break
    return False


def _in_url(rest: list[str]) -> bool:
    """Whether an address carries the output of a command in it (``?d=$(base64 file)``)."""
    return any("://" in word and _SUBSTITUTION.search(word) for word in rest)


def _loopback_only(rest: list[str]) -> bool:
    """Whether every address in a command's words is this computer's own."""
    hosts: list[str] = []
    for word in rest:
        if "://" in word:
            body = word.split("=", 1)[-1] if word.startswith("--") else word
            hosts.append(urlsplit(body).hostname or "")
        elif _BARE_LOCAL.match(word):
            hosts.append("localhost")
    return bool(hosts) and all(loopback(host) for host in hosts)


def _imports(tree: ast.AST) -> dict[str, str]:
    """The names a script gives modules and functions: ``import os as o``, ``from os import remove``."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def _dotted(node: ast.AST, names: Mapping[str, str]) -> str:
    """The dotted name a call's function resolves to; ``?`` stands for what is not a name."""
    if isinstance(node, ast.Name):
        return names.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value, names) or '?'}.{node.attr}"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return f"{_dotted(node.args[0], names) or '?'}.{node.args[1].value}"
    return ""


def _text_of(node: ast.AST | None) -> str:
    return (
        node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else ""
    )


def _python_effect(node: ast.Call, names: Mapping[str, str]) -> str:
    """``delete`` or ``send_out`` when a call in a script clearly does it, else ``""``."""
    dotted = _dotted(node.func, names)
    if not dotted:
        return ""
    module, _, last = dotted.rpartition(".")
    if (
        last == "rmtree"
        or (last in ("remove", "removedirs") and module == "os")
        or (last in ("unlink", "rmdir") and module != "")
    ):
        return "delete"
    urls = [url for url in map(_text_of, node.args) if url.lower().startswith("http")]
    own = bool(urls) and all(loopback(urlsplit(url).hostname or "") for url in urls)
    body = any(
        kw.arg in ("data", "json", "files")
        and not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
        for kw in node.keywords
    )
    method = _text_of(node.args[0] if node.args else None) or next(
        (_text_of(kw.value) for kw in node.keywords if kw.arg == "method"), ""
    )
    sender = bool(_PY_SENDERS & set(module.split(".")))
    if (
        module
        and last in ("post", "put", "patch")
        and (last == "post" or sender or body or bool(urls))
        or last in ("urlopen", "Request")
        and (body or len(node.args) >= 2)
        or module
        and last == "request"
        and method.upper() in _METHOD_SENDS
    ):
        return "" if own else "send_out"
    if module and (last in ("sendmail", "send_message") or last in _PY_SOCKETS):
        return "send_out"
    return ""


def _python_places(
    node: ast.Call, names: Mapping[str, str], effect: str
) -> tuple[tuple[str, ...], bool]:
    """Where a call in a script deletes or sends, when it names the place with a literal."""
    func = node.func
    if effect == "delete":
        target = node.args[0] if node.args else None
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("unlink", "rmdir")
            and isinstance(func.value, ast.Call)
            and _dotted(func.value.func, names).rpartition(".")[2]
            in ("Path", "PurePath")
        ):
            target = func.value.args[0] if func.value.args else None
        text = _text_of(target)
        return _path_places([text]) if text else ((), True)
    url = _text_of(node.args[0] if node.args else None) or next(
        (_text_of(kw.value) for kw in node.keywords if kw.arg == "url"), ""
    )
    return _hosts_places([url]) if "://" in url else ((), True)


def _python_items(code: str) -> list[_Item]:
    """What a script does, in the order of its calls: effects, and the commands it runs."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return []
    names = _imports(tree)
    calls = sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.Call)),
        key=lambda n: (n.lineno, n.col_offset),
    )
    found: list[_Item] = []
    for node in calls[:400]:
        effect = _python_effect(node, names)
        if effect:
            places, blind = _python_places(node, names, effect)
            found.append(_Item(effect, places=places, blind=blind))
            continue
        if _dotted(node.func, names).rpartition(".")[2] not in _PY_RUNS:
            continue
        try:
            source = ast.unparse(node)
        except (ValueError, RecursionError):
            continue
        for words, text in command_steps("python_exec", {"code": source}):
            item = _classify(words, text)
            if item is not None:
                found.append(item)
    return found


# ── Built by code ──


def _short(text: str) -> str:
    text = _clean(text)
    return text if len(text) <= COMMAND_CHARS else text[: COMMAND_CHARS - 1] + "…"


def _apps(review: ActionReview | None) -> tuple[str, ...]:
    """The apps the rules found a script operating, in order."""
    if review is None or "app_control" not in review.categories:
        return ()
    found: list[str] = []
    for reason in review.reasons:
        named = re.match(r"(?:Operates|Opens) (.+?) through (?:a script|macOS)", reason)
        if named and named.group(1) not in found:
            found.append(named.group(1))
    return tuple(found)


def _say(id: str, **params: Any) -> dict[str, Any]:
    return {"id": id, "params": params}


def _fit(pieces: Sequence[str]) -> tuple[str, int]:
    """The commands of *pieces* that fit one line, and how many did not."""
    shown: list[str] = []
    used = 0
    for piece in pieces:
        piece = _clean(piece)
        if shown and used + len(piece) + 2 > COMMAND_CHARS:
            break
        shown.append(piece)
        used += len(piece) + (2 if len(shown) > 1 else 0)
    return _short("; ".join(shown)), len(pieces) - len(shown)


def _runs(items: Sequence[_Item]) -> list[tuple[str, list[str]]]:
    """*items* as parts: neighbours alike become one (commands, apps), a repeat of an effect is dropped."""
    runs: list[tuple[str, list[str]]] = []
    for item in items:
        values = [item.text] if item.kind == "command" else list(item.apps)
        if item.script:
            values = ["script"]
        if runs and runs[-1][0] == item.kind:
            if item.kind == "command":
                runs[-1][1].extend(values)
            else:
                runs[-1][1].extend(v for v in values if v not in runs[-1][1])
        else:
            runs.append((item.kind, values))
    return runs


def _pared(
    runs: list[tuple[str, list[str]]],
) -> tuple[list[tuple[str, list[str]]], int]:
    """*runs* cut to :data:`MAX_STEPS`: plain commands go first, an effect is never hidden."""
    hidden = 0
    runs = list(runs)
    while len(runs) > MAX_STEPS:
        last = next(
            (at for at in range(len(runs) - 1, -1, -1) if runs[at][0] == "command"), -1
        )
        if last < 0:
            break
        hidden += len(runs[last][1])
        del runs[last]
    if len(runs) > MAX_STEPS:  # only effects and apps remain: the first of each kind
        kept: list[tuple[str, list[str]]] = []
        for kind, values in runs:
            same = next((run for run in kept if run[0] == kind), None)
            if same is None:
                kept.append((kind, values))
            elif kind == "operate":
                same[1].extend(v for v in values if v not in same[1])
        runs = kept
    return runs, hidden


def _chain(items: Sequence[_Item], script: bool) -> dict[str, Any]:
    """The sentence for *items*: each part in order, ``then`` between, ``and N more`` for the rest."""
    runs, hidden = _pared(_runs(items))
    parts: list[dict[str, Any]] = []
    for kind, values in runs:
        if kind == "command":
            text, left = _fit(values)
            hidden += left
            parts.append(_say("script_command" if script else "command", command=text))
        elif kind == "operate":
            parts.append(_say("operate", apps=list(values)))
        else:
            parts.append(_say(("script_" if values else "") + f"effect_{kind}"))
    sentence = parts[-1]
    for part in reversed(parts[:-1]):
        sentence = _say("then", first=part, next=sentence)
    if hidden:
        sentence = _say("more", list=sentence, count=str(hidden))
    return sentence


def _built(
    tool_name: str, arguments: Mapping[str, Any], items: Sequence[_Item]
) -> Gist:
    if tool_name in _RUNNERS:
        if not items:
            if tool_name == "python_exec":
                return Gist("script")
            rows = [
                row.strip()
                for row in _source(tool_name, arguments).splitlines()
                if row.strip() and not row.strip().startswith("#")
            ]
            return Gist("command", {"command": _short(" ".join(rows))})
        node = _chain(items, tool_name == "python_exec")
        return Gist(node["id"], node["params"])
    if tool_name in ("file_write", "file_edit"):
        path = str(arguments.get("path") or "").rstrip("/")
        name = path.rsplit("/", 1)[-1] or "a file"
        return Gist("write" if tool_name == "file_write" else "edit", {"name": name})
    return Gist("tool", {"tool": tool_name})
