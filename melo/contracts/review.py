"""The review of one tool call: routine, high-impact or undecided (#279).

The checker's rules (:mod:`localmelo.melo.checker.review`) classify each
``shell_exec``, ``python_exec`` and ``file_write`` call before an approval
policy decides. A session grant covers routine calls only: a high-impact
call asks every time and is never granted. Under a grant, and in auto
mode, the user's model reviews an undecided call (``by="model"``). In auto
mode (#339) nobody is asked: a call that is not routine is blocked, unless
the user allowed that exact step once (:func:`step_key`) or asked for it in
their own words. This module imports only the standard library and
:mod:`~localmelo.melo.contracts.shell_lexer`, which does too.
"""

from __future__ import annotations

import ast
import hashlib
import ipaddress
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal
from urllib.parse import unquote, urlsplit

from localmelo.melo.contracts import shell_lexer

Verdict = Literal["routine", "high", "undecided"]
# The version of the rules a review recorded: 2 added the ``app_control`` kind
# (an allow-list of plain scripts) and the sentence of an approval card (#454);
# 3 reviews what 2 let pass as routine: a script that reaches macOS without
# naming an app, a link of a custom scheme, erasing a disk, discarding
# uncommitted work and stopping other processes.
RULES_VERSION: Final = 3
# The longest command or code (in characters) the rules read, and the card's
# sentence is built from. A longer text is undecided without being read: no
# real step is that long, and reading it would cost more than it is worth.
MAX_TEXT: Final = 256 * 1024

# The card's title per category, strongest first: a call's title and the
# order of its reasons follow this order.
TITLES: Final[dict[str, str]] = {
    "self": "Change localmelo itself?",
    "credentials": "Use your passwords or keys?",
    "money": "Spend money?",
    "account": "Change an account's settings?",
    "system": "Change system settings?",
    "app_control": "Control an app?",
    "open_app": "Open another app?",
    "persistence": "Set up something that runs later?",
    "remote_code": "Run code downloaded from the internet?",
    "install": "Install software?",
    "restore": "Install the project's packages?",
    "env": "Create a programming environment?",
    "download": "Download files to this Mac?",
    "outside_write": "Change files outside {folder}?",
    "delete": "Delete files that cannot be restored?",
    "service": "Open a network service?",
    "send_out": "Send data from this Mac?",
    "opaque": "Run code melo cannot check?",
    "beyond_request": "Do something you did not ask for?",
    "unreviewed": "Run a step melo could not review?",
}
# Each category in a few words, for a blocked step's tag ("blocked: installs
# software") and the reason an auto-mode block records.
CATEGORY_WORDS: Final[dict[str, str]] = {
    "self": "changes localmelo itself",
    "credentials": "uses passwords or keys",
    "money": "spends money",
    "account": "changes an account's sign-in or security settings",
    "system": "changes system settings",
    "app_control": "controls an app",
    "open_app": "opens another app for a page",
    "persistence": "sets up something that runs later",
    "remote_code": "runs code from the internet",
    "install": "installs software",
    "restore": "installs the project's packages",
    "env": "creates a programming environment",
    "download": "downloads files",
    "outside_write": "changes files outside the folder",
    "delete": "deletes files",
    "service": "opens a network service",
    "send_out": "sends data from this Mac",
    "opaque": "runs code melo cannot check",
    "beyond_request": "does something you did not ask for",
    "unreviewed": "could not be reviewed",
}
# Auto mode: the categories the user's own words can allow (they can name
# what the step acts on), and those only an "Allow once" of the exact step
# can. ``self`` is in neither: it is never allowed.
WORDS_CAN_ALLOW: Final = frozenset(
    {
        "install",
        "restore",
        "env",
        "download",
        "outside_write",
        "delete",
        "send_out",
        "service",
        "beyond_request",
    }
)
# ``app_control`` (a script that operates another app) was ``system`` once and
# is as strict: only a click allows it, never words or the model.
CLICK_ONLY: Final = frozenset(
    {
        "credentials",
        "system",
        "app_control",
        "persistence",
        "remote_code",
        "opaque",
        "unreviewed",
    }
)
# Kinds only the user's model names, for a step on a page or in an app: paying
# (a purchase, a payment, a subscription, a transfer) and changing an account's
# sign-in or security settings; and opening another app that a page asked for
# (an address of the app's kind, which the browser in the background does not
# open itself). However well the step serves the goal, only a person's answer
# lets it run: never words or the model. A step reviewed as one of these is
# asked on the card every time (in auto mode a person who watches; #465), and
# no grant of the chat covers it. In ask mode a page's request to open an app
# is a card of its own (``handoff:<scheme>``), where the person may choose to
# let pages open that kind of address again: that grant is their own answer,
# shown on the card, and auto mode never rides it.
CARD_ONLY: Final = frozenset({"money", "account", "open_app"})
# The reason a call of the ``self`` category is denied without asking.
SELF_DENIAL: Final = "melo never lets a call change its own runtime, settings or keys"
# The tools whose arguments the rules read (melo.checker.review); another
# tool is routine to them, and auto mode makes one that is not read-only
# undecided.
RULED_TOOLS: Final = frozenset(["shell_exec", "python_exec", "file_write", "file_edit"])
# A model's category that can let a call run: none, or one words can allow.
MODEL_CAN_ALLOW: Final = frozenset({"none", ""}) | WORDS_CAN_ALLOW


@dataclass(frozen=True)
class ModelAnswer:
    """What the user's model answered about one call (see ``melo.checker.review.model``).

    ``verdict`` is ``routine`` or ``ask``; ``fits_request`` whether the call
    serves what the user asked (``yes``, ``no`` or ``unclear``);
    ``user_asked`` whether the user's own words ask for it; ``reason`` one
    sentence for the user; ``ms`` how long the review took. ``error`` names
    why there is no usable answer (a timeout, a failed request, a reply that
    does not parse, the turn's limit, a call too long to review), and the
    other fields are then empty.
    """

    verdict: str = ""
    category: str = ""
    fits_request: str = ""
    user_asked: str = ""
    reason: str = ""
    ms: int = 0
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        if self.error:
            return {"error": self.error, "ms": self.ms}
        return {
            "verdict": self.verdict,
            "category": self.category,
            "fits_request": self.fits_request,
            "user_asked": self.user_asked,
            "reason": self.reason,
            "ms": self.ms,
        }


@dataclass(frozen=True)
class ActionReview:
    """What the rules found in one call, and the model when it was asked.

    ``categories`` are keys of :data:`TITLES`, strongest first; ``reasons``
    are at most three sentences for the person, each starting with a verb
    ("Installs playwright with pip."); ``paths`` are write paths outside
    the workspace that the call names. A review the model settled has
    ``by="model"`` and its :class:`ModelAnswer`; a high-impact call the
    user's words allowed keeps its categories with the verdict ``routine``.
    ``doubts`` are what the rules could not tell about a high-impact call
    (the reasons an undecided call has in ``reasons``): words never allow
    such a call. ``inputs`` name the local files whose content the rules
    read, each as ``path:digest``; :func:`step_key` includes them, so a
    changed file makes another step. ``changed`` names the files changed
    in this chat that a runner (``pytest``…) runs and melo could not read,
    as they are now: not part of the key, it tells one review of the step
    from the next.
    """

    verdict: Verdict
    categories: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    by: Literal["rules", "model"] = "rules"
    rules_version: int = RULES_VERSION
    model: ModelAnswer | None = None
    doubts: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "verdict": self.verdict,
            "categories": list(self.categories),
            "reasons": list(self.reasons),
            "paths": list(self.paths),
            "by": self.by,
            "rules_version": self.rules_version,
        }
        if self.doubts:
            out["doubts"] = list(self.doubts)
        if self.model is not None:
            out["model"] = self.model.to_json()
        return out


def covered_by_grant(review: ActionReview | None) -> bool:
    """Whether a session grant of the call's tool lets it run unasked."""
    return review is None or review.verdict == "routine"


def offers_grant(review: ActionReview | None) -> bool:
    """Whether the card may offer "allow for the session": never for high impact."""
    return review is None or review.verdict != "high"


def review_title(review: ActionReview | None, folder: str) -> str:
    """The card's title for a high-impact call, or ``""``."""
    if review is None or review.verdict != "high" or not review.categories:
        return ""
    return TITLES.get(review.categories[0], "Run this?").format(folder=folder)


def because(review: ActionReview) -> str:
    """The reasons as one clause: "installs x; creates y" (no final stop)."""
    return "; ".join(r[:1].lower() + r[1:].rstrip(".") for r in review.reasons)


def block_words(review: ActionReview | None) -> str:
    """The first category in a few words ("installs software"), or ``""``."""
    if review is None or not review.categories:
        return ""
    return CATEGORY_WORDS.get(review.categories[0], "")


def _normalized(tool_name: str, arguments: Mapping[str, Any]) -> Any:
    """What a call does: Python as its syntax tree, a command as its exact text."""
    if tool_name == "python_exec":
        code = str(arguments.get("code") or "")
        try:
            return ["python", ast.dump(ast.parse(code))]
        except (SyntaxError, ValueError, RecursionError):
            return ["python-text", " ".join(code.split())]
    if tool_name == "shell_exec":
        command = arguments.get("command")
        # The exact text: a "#" inside a word, or a quote, changes what runs.
        if isinstance(command, list):
            return ["shell-argv", [str(part) for part in command]]
        return ["shell", str(command or "")]
    if tool_name in ("file_write", "file_edit"):
        rest = {k: v for k, v in arguments.items() if k != "path"}
        body = json.dumps(rest, sort_keys=True, ensure_ascii=False, default=str)
        path = os.path.realpath(str(arguments.get("path") or ""))
        return ["file", path, hashlib.sha256(body.encode()).hexdigest()]
    return [
        "call",
        json.loads(json.dumps(dict(arguments), sort_keys=True, default=str)),
    ]


def step_key(
    tool_name: str,
    arguments: Mapping[str, Any],
    workspace_root: str | None,
    inputs: Sequence[str] = (),
) -> str:
    """The hash of one step: the same call in the same folder, on the same files.

    Python is compared as its syntax tree, so a comment or a reflowed line
    is the same step; a shell command by its exact text; a file write by
    its resolved path and a hash of what it writes. *inputs* are the
    review's (:attr:`ActionReview.inputs`): the local files it read, with
    their digests. 32 hex characters.
    """
    root = os.path.realpath(workspace_root) if workspace_root else ""
    shape = [tool_name, root, _normalized(tool_name, arguments), sorted(inputs)]
    text = json.dumps(shape, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:32]


# ── Steps alike (#411): keys that only ever refuse ──

# The calls whose first argument is a command a Python step runs.
_RUNS: Final = frozenset(
    {
        "run",
        "call",
        "check_call",
        "check_output",
        "Popen",
        "system",
        "popen",
        "getoutput",
    }
)
# Flags that only make a command quieter: the same step with or without them.
_QUIET: Final = frozenset({"-q", "-qq", "-qqq", "--quiet", "--silent"})
_PYTHON: Final = re.compile(r"^(?:python(?:\d+(?:\.\d+)*)?|py)$")
_PIP: Final = re.compile(r"^pip(?:\d+(?:\.\d+)*)?$")
_SHELL_SPLIT: Final = re.compile(r"\s*(?:&&|\|\||;|\||\n)\s*")
# What stands for melo's own Python in a step: the same program as "python".
_EXECUTABLE: Final = "sys.executable"


def _program(word: str) -> str:
    name = word.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name == _EXECUTABLE or _PYTHON.match(name):
        return "python"
    return "pip" if _PIP.match(name) else name


def _command(words: Sequence[str]) -> list[str]:
    """*words* as a command alike: the program by name, ``python -m pip`` as
    ``pip``, quoting and spacing gone, flags that only quieten it dropped."""
    tokens = [w.strip() for w in words if w.strip()]
    if not tokens:
        return []
    tokens[0] = _program(tokens[0])
    if tokens[0] == "python" and tokens[1:3] == ["-m", "pip"]:
        tokens = ["pip", *tokens[3:]]
    elif tokens[0] == "uv" and tokens[1:2] == ["pip"]:
        tokens = ["pip", *tokens[2:]]
    return [t for t in tokens if t not in _QUIET]


def _shell_commands(text: str) -> list[list[str]]:
    found = []
    for part in _SHELL_SPLIT.split(text):
        try:
            words = shell_lexer.split(part)
        except ValueError:
            words = part.split()
        if words:
            found.append(words)
    return found


Step = tuple[list[str], str]


def _cut(text: str) -> tuple[list[str], bool]:
    """*text* cut into commands, and whether every quote in it closed.

    The cuts are ``;``, ``&&``, ``||``, ``|``, ``&`` and line breaks, only
    where no quote, ``$(…)`` or backquote is open. A comment (``#`` after a
    space or an operator) runs to its line end and cuts nothing.
    """
    parts: list[str] = []
    stack: list[str] = []  # what is open, innermost last: ' " ` or (
    start = at = 0
    while at < len(text):
        char, top = text[at], (stack[-1] if stack else "")
        if top == "'":
            if char == "'":
                stack.pop()
        elif char == "\\":
            at += 1
        elif top == '"' and char == '"' or top == "`" and char == "`":
            stack.pop()
        elif text.startswith("$(", at):
            stack.append("(")
            at += 1
        elif char == "`" or char in "'\"" and top != '"':
            stack.append(char)
        elif top == "(" and char == "(":
            stack.append("(")
        elif top == "(" and char == ")":
            stack.pop()
        elif top:
            pass
        elif char == "#" and (at == 0 or text[at - 1] in " \t\n;&|("):
            end = text.find("\n", at)
            at = len(text) if end < 0 else end - 1
        elif char in ";\n" or char == "|" or char == "&" and _cuts_here(text, at):
            parts.append(text[start:at])
            if (
                text[at + 1 : at + 2] == "&"
                or char == "|"
                and text[at + 1 : at + 2] == "|"
            ):
                at += 1  # && || and |&
            start = at + 1
        at += 1
    parts.append(text[start:])
    return parts, not stack


def _cuts_here(text: str, at: int) -> bool:
    """Whether the ``&`` at *at* ends a command: ``&&`` or a background ``&``, not ``2>&1`` or ``&>``."""
    before, after = text[at - 1 : at] if at else "", text[at + 1 : at + 2]
    return after == "&" or before not in ("<", ">") and after != ">"


def _shell_parts(text: str) -> list[str]:
    """The commands of a shell *text*, each as written: cut outside quotes only.

    A text that leaves a quote open is read line by line instead.
    """
    text = _joined_lines(text)
    parts, closed = _cut(text)
    if not closed and "\n" in text:
        parts = [part for line in text.split("\n") for part in _cut(line)[0]]
    return [part.strip() for part in parts if part.strip()]


def _shell_steps(text: str) -> list[Step]:
    """The commands of a shell *text* as ``(words, text as written)``."""
    found: list[Step] = []
    for part in _shell_parts(text):
        try:
            words = shell_lexer.split(part)
        except ValueError:
            words = part.split()
        if words:
            found.append((words, part))
    return found


def _word(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        dotted = f"{node.value.id}.{node.attr}"
        return dotted if dotted == _EXECUTABLE else None
    return None


def _python_runs(code: str) -> list[list[str] | str]:
    """What a Python step runs with literal arguments: a word list for
    ``subprocess.run([...])``, a shell text for ``os.system("...")``."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return []
    found: list[list[str] | str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in _RUNS:
            continue
        first = node.args[0]
        if isinstance(first, ast.List | ast.Tuple):
            words = [_word(item) for item in first.elts]
            if words and all(word is not None for word in words):
                found.append([str(word) for word in words])
        elif (text := _word(first)) is not None:
            found.append(text)
    return found


def _python_commands(code: str) -> list[list[str]]:
    """The commands a Python step runs with literal arguments: ``subprocess.run([...])``,
    ``os.system("...")`` and the like."""
    found: list[list[str]] = []
    for run in _python_runs(code):
        found += _shell_commands(run) if isinstance(run, str) else [run]
    return found


def commands_of(tool_name: str, arguments: Mapping[str, Any]) -> list[list[str]]:
    """The commands a ``shell_exec`` or ``python_exec`` call runs, as word lists.

    A shell command is split at ``&&``, ``||``, ``;``, ``|`` and line breaks;
    Python gives the commands it runs with literal arguments
    (``subprocess.run([...])``, ``os.system("...")``). Another tool gives none.
    """
    if tool_name == "shell_exec":
        command = arguments.get("command")
        if isinstance(command, list):
            return [[str(word) for word in command]]
        return _shell_commands(str(command or ""))
    if tool_name == "python_exec":
        return _python_commands(str(arguments.get("code") or ""))
    return []


def command_steps(tool_name: str, arguments: Mapping[str, Any]) -> list[Step]:
    """Like :func:`commands_of`, with each command's text as written.

    A shell text is cut only outside quotes, ``$(…)`` and comments, so
    ``git commit -m "tidy; rm old"`` is one command. The text is ``""`` for a
    command given as a list of words. :func:`commands_of` and the keys built
    on it are unchanged.
    """
    if tool_name == "shell_exec":
        command = arguments.get("command")
        if isinstance(command, list):
            return [([str(word) for word in command], "")]
        return _shell_steps(str(command or ""))
    if tool_name == "python_exec":
        found: list[Step] = []
        for run in _python_runs(str(arguments.get("code") or "")):
            found += _shell_steps(run) if isinstance(run, str) else [(run, "")]
        return found
    return []


def _installs(tokens: Sequence[str]) -> bool:
    return len(tokens) > 2 and tokens[1] == "install"


def loose_keys(
    tool_name: str, arguments: Mapping[str, Any], categories: Sequence[str] = ()
) -> list[str]:
    """Keys that make two calls the same step though their text differs (#411).

    One for the whole list of commands the call runs, alike when only
    spacing, quoting, the program's spelling (``pip3``, ``python -m pip``) or a
    quietening flag differ. A part of it has keys of its own only when it is
    what was blocked: for an install (*categories* has ``install``, or is
    empty: the call is not reviewed yet), each install part, and one key per
    program and what it installs. So a blocked ``pip install numpy && python
    fetch.py`` refuses a later ``pip3 install numpy`` but not ``python
    fetch.py``. They only ever refuse: no allowance is kept by them.
    """
    commands = commands_of(tool_name, arguments)
    if not commands:
        return []
    parts = [tokens for tokens in (_command(words) for words in commands) if tokens]
    if not parts:
        return []

    def key(text: str) -> str:
        return "cmd:" + hashlib.sha256(text.encode()).hexdigest()[:32]

    keys = {key(" && ".join(" ".join(tokens) for tokens in parts))}
    if not categories or "install" in categories:
        for tokens in filter(_installs, parts):
            keys.add(key(" ".join(tokens)))
            targets = sorted({t.lower() for t in tokens[2:] if not t.startswith("-")})
            if targets:
                keys.add(f"install:{tokens[0]}:{','.join(targets)}")
    return sorted(keys)


# ── The sites a step names (#433): keys that only ever refuse ──

# The loose key of a site: ``ask:site:<host>`` (approvals.site_key).
SITE_KEY: Final = "ask:site:"
# A step naming an http(s) address whose encoded host melo cannot read as
# one site (a percent-encoding or an international name that does not
# decode): refused wherever a site was refused, never anywhere else.
UNREADABLE_SITE: Final = "ask:site-unreadable"
# The scheme and authority of each http(s) address in a text. The authority
# stops at "/", so an address nested in another one's path is found too.
_AUTHORITY: Final = re.compile(r"https?://[^\s<>\"'`/?#]*", re.IGNORECASE)
# What a host built at run time has: a variable, a template, a format
# placeholder (``%s``, a ``%`` without two hex digits), an f-string's value.
_BUILT: Final = re.compile(r"[$`{}\x00]|%(?![0-9a-fA-F]{2})")
# A host melo reads as one site, once encoded: labels of letters, digits,
# ``-`` and ``_``.
_HOST_NAME: Final = re.compile(r"[a-z0-9_-]+(?:\.[a-z0-9_-]+)*")
# An IPv4 address in any form a client accepts: 1 to 4 parts, each decimal,
# octal (a leading 0) or hex (0x).
_NUMERIC: Final = re.compile(r"(?:0x[0-9a-f]*|\d+)(?:\.(?:0x[0-9a-f]*|\d+)){0,3}")
# One brace group a shell (or curl) expands into words: ``{a,b}``.
_BRACES: Final = re.compile(r"^([^{}]*)\{([^{}]*,[^{}]*)\}([^{}]*)$")
# Characters after a host that end a sentence or a call, not the host.
_HOST_END: Final = ")],;!'\""
# The longest host name; a longer host text is not one site.
_HOST_CHARS: Final = 253
# A value an f-string fills in: no host is read through it.
_FILLED: Final = "\x00"


def _ipv4(host: str) -> str | None:
    """*host* as a dotted-quad IPv4 address when it is one in another form (``16909060``)."""
    if not _NUMERIC.fullmatch(host):
        return None
    numbers = []
    for part in host.split("."):
        try:
            if part.startswith("0x"):
                numbers.append(int(part[2:] or "0", 16))
            elif len(part) > 1 and part.startswith("0"):
                numbers.append(int(part, 8))
            else:
                numbers.append(int(part))
        except ValueError:
            return None
    *head, last = numbers
    if any(n > 255 for n in head) or last >= 256 ** (4 - len(head)):
        return None
    value = sum(n << (8 * (3 - at)) for at, n in enumerate(head)) + last
    return str(ipaddress.IPv4Address(value))


def site_host(host: str) -> str | None:
    """*host* as one site: in ASCII (IDNA), lower case, without one final dot.

    An IP address is written the one standard way (``::1``, and an IPv4
    address written as one number, in hex or octal as dotted decimal).
    ``None`` when it is empty or cannot be encoded. What
    :func:`~localmelo.melo.contracts.approvals.site_key` uses, so one site
    has one key however its name is spelt.
    """
    try:
        if not host.isascii():
            host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    host = host.lower()
    host = host[:-1] if host.endswith(".") else host
    numeric = _ipv4(host)
    if numeric is not None:
        return numeric
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return host or None


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def url_site(url: str) -> str | None:
    """The site of an http(s) *url* (:func:`site_host`), or ``None``."""
    try:
        parts = urlsplit(url.strip())
        host = site_host(parts.hostname or "")
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https"):
        return None
    return host


def _raw_hosts(address: str) -> set[str]:
    """The host texts *address* (``http(s)://…``) can reach.

    Clients read an address differently: a ``\\`` ends the host for some
    and not for others, and an ``@`` that appears twice is split at the
    first or the last. Each reading is kept, so a refused site cannot hide
    behind one of them.
    """
    rest = address.split("://", 1)[1]
    hosts: set[str] = set()
    for authority in {re.split(r"[/?#]", rest)[0], re.split(r"[/?#\\]", rest)[0]}:
        for hostport in {authority.rpartition("@")[2], authority.partition("@")[2]}:
            if not hostport:
                continue
            if hostport.startswith("["):
                hosts.add(hostport[1:].partition("]")[0])
            else:
                hosts.add(hostport.partition(":")[0])
    return hosts


def _host_key(raw: str) -> str | None:
    """The loose key of one host text, :data:`UNREADABLE_SITE`, or ``None`` (unknown)."""
    raw = raw.rstrip(_HOST_END)
    if not raw or len(raw) > _HOST_CHARS:
        return None
    if _BUILT.search(raw):
        return None  # built at run time: the review judges it
    encoded = "%" in raw
    if encoded:
        try:
            raw = unquote(raw, errors="strict")
        except UnicodeDecodeError:
            return UNREADABLE_SITE
    host = site_host(raw)
    if host is None:
        return UNREADABLE_SITE if encoded or not raw.isascii() else None
    if not (_HOST_NAME.fullmatch(host) or _is_ip(host)):
        # Not a name a client resolves (a pattern, a stray character):
        # unknown, unless an encoding hid what it is.
        return UNREADABLE_SITE if encoded else None
    return SITE_KEY + host


def _expanded(raw: str) -> list[str]:
    """*raw* with one brace group expanded, as a shell or curl does: ``{a,b}.x``."""
    found = _BRACES.match(raw) if len(raw) <= _HOST_CHARS else None
    if found is None:
        return [raw]
    head, choices, tail = found.groups()
    return [head + choice + tail for choice in choices.split(",")]


def _host_sites(raw: str) -> tuple[set[str], bool]:
    """The keys one host text gives, and whether some reading of it gives none."""
    raw = raw.rstrip(_HOST_END)
    if not raw:
        return set(), False
    keys: set[str] = set()
    unread = False
    for host in _expanded(raw):
        key = _host_key(host)
        if key is None:
            unread = True
        else:
            keys.add(key)
    return keys, unread


def _decoded(text: str) -> list[str]:
    """*text*, and it percent-decoded once and twice when that changes it."""
    texts = [text]
    for _ in range(2):
        if "%" not in texts[-1]:
            break
        plain = unquote(texts[-1])
        if plain == texts[-1]:
            break
        texts.append(plain)
    return texts


def _address_sites(text: str) -> tuple[set[str], bool]:
    """The keys of every http(s) address in *text*, nested and encoded ones too.

    The second value: some address in it has a host that gives no key
    (built at run time, a pattern, too long): melo cannot read it.
    """
    keys: set[str] = set()
    unread = False
    for form in _decoded(text):
        for found in _AUTHORITY.finditer(form):
            for raw in _raw_hosts(found.group(0)):
                more, hidden = _host_sites(raw)
                keys |= more
                unread = unread or hidden
    return keys, unread


def _address_keys(text: str) -> set[str]:
    return _address_sites(text)[0]


def address_keys(text: str) -> list[str]:
    """The site keys of every http(s) address in *text*, nested or encoded ones too."""
    return sorted(_address_keys(text))


# ── Network steps: scheme-less curl and wget addresses, hosts melo cannot read ──

# The options of curl and wget whose value is the next word (or joined to
# them): that word is never an address.
_CURL_SHORT: Final = frozenset("AbcCdDeEFHKmoPQrtTuUwxXyYz")
_CURL_LONG_NAMES: Final = """abstract-unix-socket alt-svc aws-sigv4 cacert capath cert cert-type
    ciphers config connect-timeout connect-to continue-at cookie cookie-jar
    create-file-mode crlfile curves data data-ascii data-binary data-raw
    data-urlencode delegation dns-interface dns-ipv4-addr dns-ipv6-addr
    dns-servers doh-url dump-header ech engine etag-compare etag-save
    expect100-timeout form form-string ftp-account ftp-alternative-to-user
    ftp-method ftp-port ftp-ssl-ccc-mode happy-eyeballs-timeout-ms
    haproxy-clientip header hostpubmd5 hostpubsha256 hsts interface ip-tos
    ipfs-gateway json keepalive-cnt keepalive-time key key-type krb libcurl
    limit-rate local-port login-options mail-auth mail-from mail-rcpt
    max-filesize max-redirs max-time netrc-file noproxy oauth2-bearer output
    output-dir parallel-max pass pinnedpubkey preproxy proto proto-default
    proto-redir proxy proxy-cacert proxy-capath proxy-cert proxy-cert-type
    proxy-ciphers proxy-crlfile proxy-header proxy-key proxy-key-type
    proxy-pass proxy-pinnedpubkey proxy-service-name proxy-tls13-ciphers
    proxy-tlsauthtype proxy-tlspassword proxy-tlsuser proxy-user proxy1.0
    pubkey quote random-file range rate referer request request-target
    resolve retry retry-delay retry-max-time sasl-authzid service-name socks4
    socks4a socks5 socks5-gssapi-service socks5-hostname speed-limit
    speed-time stderr telnet-option tftp-blksize time-cond tls-max
    tls13-ciphers tlsauthtype tlspassword tlsuser trace trace-ascii
    trace-config unix-socket upload-file url url-query user user-agent
    variable write-out"""
_CURL_LONG: Final = frozenset(_CURL_LONG_NAMES.split())
_WGET_SHORT: Final = frozenset("aABDeiIloOPQRtTUwX")
_WGET_LONG_NAMES: Final = """accept accept-regex append-output base bind-address bind-dns-address
    body-data body-file ca-certificate ca-directory certificate
    certificate-type config connect-timeout crl-file cut-dirs default-page
    directory-prefix dns-servers dns-timeout domains egd-file exclude-directories
    exclude-domains execute follow-tags ftp-password ftp-user header
    hsts-file http-password http-user ignore-tags include-directories
    input-file level limit-rate load-cookies local-encoding max-redirect
    method output-document output-file password pinnedpubkey post-data
    post-file prefer-family private-key private-key-type progress
    proxy-password proxy-user quota random-file read-timeout referer reject
    reject-regex rejected-log regex-type remote-encoding report-speed
    restrict-file-names save-cookies secure-protocol timeout tries
    use-askpass user user-agent wait waitretry warc-cdx warc-dedup warc-file
    warc-header warc-max-size warc-tempdir"""
_WGET_LONG: Final = frozenset(_WGET_LONG_NAMES.split())
# curl options whose value is a host or an address it connects through.
_CURL_PROXIES: Final = frozenset(
    {"x", "proxy", "preproxy", "socks4", "socks4a", "socks5", "socks5-hostname"}
    | {"doh-url"}
)
# Options that take the addresses from a file: melo cannot read them.
_FROM_FILE: Final = {"curl": ("K", "config"), "wget": ("i", "input-file")}
# Options that only print something: no address is needed.
_INFO: Final = frozenset({"version", "help", "manual", "V", "h", "M"})
# A word curl or wget reads as an address without a scheme: a dotted host,
# perhaps a user before it, a port and a path after it.
_BARE: Final = re.compile(
    r"(?:[^\s/?#@]+@)?([^\s/?#@:\[\]]+\.[^\s/?#@:\[\].]+\.?)(?::\d+)?(?:[/?#]\S*)?"
)
_BARE_CHARS: Final = 2048
# Text a shell or Python fills in when the step runs.
_EXPANDS: Final = re.compile(r"[$`\x00]")
# A proxy setting in any word: ``https_proxy=…``, ``-e http_proxy=…``,
# ``git -c http.proxy=…``.
_PROXY_WORD: Final = re.compile(
    r"(?i)(?:-e)?(?:(?:https?|ftp|all|socks)_proxy|(?:[\w-]+\.)+proxy)=(.*)"
)
# Settings that make curl or wget read their options from a file.
_CONFIG_HOMES: Final = frozenset({"CURL_HOME", "WGETRC", "XDG_CONFIG_HOME", "CURLRC"})
# Programs that reach other computers whose targets melo does not key.
_NETWORK: Final = frozenset(
    {"nc", "ncat", "netcat", "telnet", "socat", "websocat", "ssh", "scp", "sftp"}
    | {"rsync", "ftp", "aria2c", "http", "https", "xh"}
)
_GIT_NETWORK: Final = frozenset({"clone", "fetch", "pull", "push", "ls-remote"})
_SHELLS: Final = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish"})
# Words that run the command after them; the value of each option given.
_WRAPPERS: Final[dict[str, frozenset[str]]] = {
    "command": frozenset(),
    "exec": frozenset(),
    "time": frozenset(),
    "nohup": frozenset(),
    "env": frozenset({"-u", "-C", "-S"}),
    "sudo": frozenset({"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U"}),
    "xargs": frozenset({"-n", "-I", "-P", "-L", "-d", "-E", "-s", "-a", "-J", "-R"}),
    "nice": frozenset({"-n"}),
    "timeout": frozenset({"-s", "-k"}),
    "builtin": frozenset(),
}
# Shell words that open or close a compound command: the command follows.
_KEYWORDS: Final = frozenset(
    {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "!", "{"}
    | {"}", "time", "coproc"}
)
_ASSIGNMENT: Final = re.compile(r"([A-Za-z_]\w*)=(.*)", re.S)
# What separates two commands, or redirects one, in a shell's text.
_OPERATORS: Final = ";&|()<>`\n"
# A command substitution in a word: ``$(…)`` or ``\`…\```.
_SUBSTITUTION: Final = re.compile(r"\$\((.*)\)|`([^`]*)`", re.S)
# Python: calls that fetch an address, and calls that connect to a host.
_PY_URL_CALLS: Final = frozenset({"urlopen", "urlretrieve", "Request"})
_PY_METHODS: Final = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "request", "stream"}
)
_PY_CLIENT_MODULES: Final = frozenset(
    {"requests", "httpx", "urllib3", "aiohttp", "httplib2", "pycurl"}
)
_PY_CLIENTS: Final = frozenset(
    {"Session", "Client", "AsyncClient", "PoolManager", "ClientSession", "Http"}
    | {"Curl"}
)
_PY_HOST_CALLS: Final = frozenset(
    {"HTTPConnection", "HTTPSConnection", "create_connection", "open_connection"}
)
_PY_SPAWNERS: Final = frozenset({"subprocess", "os"})
_DEPTH: Final = 3

Sites = tuple[set[str], bool]


def _joined_lines(text: str) -> str:
    """*text* with each backslash-newline outside single quotes removed, as a shell does."""
    out: list[str] = []
    single = double = False
    index = 0
    while index < len(text):
        char = text[index]
        if single:
            single = char != "'"
        elif char == "\\":
            if text[index + 1 : index + 2] == "\n":
                index += 2
                continue
            out.append(text[index : index + 2])
            index += 2
            continue
        elif char == "'" and not double:
            single = True
        elif char == '"':
            double = not double
        out.append(char)
        index += 1
    return "".join(out)


def _shell_tokens(text: str) -> list[str]:
    """*text* as a shell reads it: words, and operators (``;``, ``&&``, ``(``, ``>``) apart.

    Line continuations are joined first. Quotes are kept together; text
    that does not parse is read line by line, and a line that does not
    parse is split at spaces and operators.
    """
    text = _joined_lines(text)

    def split(part: str) -> list[str]:
        lexer = shell_lexer.Lexer(part, posix=True, punctuation_chars=_OPERATORS)
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)

    try:
        return split(text)
    except ValueError:
        pass
    tokens: list[str] = []
    for line in text.split("\n"):
        try:
            tokens += split(line)
        except ValueError:
            tokens += re.findall(r"[;&|()<>`]+|[^\s;&|()<>`]+", line)
        tokens.append("\n")
    return tokens


Command = tuple[list[str], list[str]]


def _shell_commands_of(tokens: Sequence[str]) -> list[Command]:
    """The simple commands in *tokens*, each with the here-strings (``<<<``) it reads.

    Split at operators; a redirection's target is dropped.
    """
    commands: list[Command] = []
    current: list[str] = []
    here: list[str] = []
    target = ""
    for token in tokens:
        if token and all(c in _OPERATORS for c in token):
            if "(" in token or ")" in token or not ("<" in token or ">" in token):
                if current or here:
                    commands.append((current, here))
                current, here, target = [], [], ""
            else:
                target = "here" if "<<<" in token else "file"
            continue
        if target:
            if target == "here":
                here.append(token)
            target = ""
            continue
        current.append(token)
    if current or here:
        commands.append((current, here))
    return commands


def _shell_words(command: Any) -> list[str]:
    if isinstance(command, list):
        return [str(word) for word in command]
    return _shell_tokens(str(command or ""))


def _unwrapped(
    words: Sequence[str],
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """*words* from the program they run, the assignments before it, and code given to ``env -S``.

    Shell keywords (``if``, ``do``, ``!``, ``{``), assignments and wrappers
    (``sudo``, ``xargs``, ``env``) are dropped.
    """
    rest = list(words)
    assigned: list[tuple[str, str]] = []
    codes: list[str] = []
    while rest:
        if rest[0] in _KEYWORDS:
            rest = rest[1:]
            continue
        found = _ASSIGNMENT.fullmatch(rest[0])
        if found:
            assigned.append((found.group(1), found.group(2)))
            rest = rest[1:]
            continue
        wrapper = _program(rest[0])
        valued = _WRAPPERS.get(wrapper)
        if valued is None:
            break
        rest = rest[1:]
        while rest and rest[0].startswith("-") and rest[0] != "-":
            option = rest[0]
            if wrapper == "env" and option in ("-S", "--split-string") and rest[1:]:
                codes.append(rest[1])  # a command line env splits and runs
            elif wrapper == "env" and option.startswith(("-S", "--split-string=")):
                codes.append(option.split("=", 1)[-1] if "=" in option else option[2:])
            rest = rest[2:] if option in valued else rest[1:]
        if rest and wrapper == "timeout":
            rest = rest[1:]  # its duration
    return rest, assigned, codes


def _proxy_sites(value: str) -> Sites:
    """The keys of a proxy (``host:port``, ``socks5://host``), and whether it is unread."""
    value = value.strip()
    if not value:
        return set(), False
    if _EXPANDS.search(value):
        return set(), True
    plain = re.sub(r"^[A-Za-z][\w+.-]*://", "", value)
    keys, unread = _address_sites("http://" + plain)
    return keys, unread or not keys


def _route_sites(value: str, fields: Sequence[int]) -> Sites:
    """The hosts of a ``--connect-to`` or ``--resolve`` value: some of its ``:`` fields."""
    if _EXPANDS.search(value):
        return set(), True
    parts = value.split(":")
    keys: set[str] = set()
    unread = False
    for at in fields:
        for host in (parts[at].split(",") if at < len(parts) else []):
            more, hidden = _host_sites(host.strip("[]"))
            keys |= more
            unread = unread or hidden
    return keys, unread


def _target_sites(target: str) -> Sites:
    """What one curl or wget address reaches: its keys, and whether melo cannot read it."""
    if _EXPANDS.search(target) or len(target) > _BARE_CHARS:
        return set(), True
    if "://" in target:
        return _address_sites(target)
    if _BARE.fullmatch(target):
        keys, unread = _address_sites("http://" + target)
        return keys, unread or not keys
    if re.search(r"[{}\[\]]", target):
        return set(), True  # a host to expand or glob
    return set(), False


def _fetcher_sites(program: str, args: Sequence[str]) -> Sites:
    """What a ``curl`` or ``wget`` command reaches: keys of its addresses and proxies."""
    curl = program == "curl"
    short, long_ = (_CURL_SHORT, _CURL_LONG) if curl else (_WGET_SHORT, _WGET_LONG)
    file_short, file_long = _FROM_FILE[program]
    targets: list[str] = []
    values: list[tuple[str, str]] = []
    from_file, info_only, ended, index = False, True, False, 0
    while index < len(args):
        arg = args[index]
        index += 1
        if ended or arg == "-" or not arg.startswith("-"):
            targets.append(arg)
        elif arg == "--":
            ended = True
        elif arg.startswith("--"):
            name, joined, value = arg[2:].partition("=")
            info_only = info_only and name in _INFO
            # --expand-<option> (curl 8.3) fills in variables: unread.
            from_file = from_file or name == file_long or name.startswith("expand-")
            if (name in long_ or name.startswith("expand-")) and not joined:
                value = args[index] if index < len(args) else ""
                index += 1
            if curl and name in ("url", "expand-url"):
                targets.append(value)
            elif joined or name in long_:
                values.append((name, value))
        else:
            info_only = info_only and all(c in _INFO for c in arg[1:])
            for at, letter in enumerate(arg[1:], 1):
                from_file = from_file or letter == file_short
                if letter in short:
                    if at < len(arg) - 1:
                        values.append((letter, arg[at + 1 :]))
                    elif index < len(args):
                        values.append((letter, args[index]))
                        index += 1
                    break
    keys: set[str] = set()
    unread = from_file or not targets and not info_only
    for target in targets:
        more, hidden = _target_sites(target)
        keys |= more
        unread = unread or hidden
    for name, value in values:
        if curl and name in _CURL_PROXIES:
            more, hidden = _proxy_sites(value)
        elif curl and name in ("connect-to", "resolve"):
            more, hidden = _route_sites(value, (0, 2))
        else:
            continue
        keys |= more
        unread = unread or hidden
    return keys, unread


def _code_sites(code: str, python: bool, depth: int) -> Sites:
    """What shell (or Python) code given in a word reaches; past :data:`_DEPTH`, unread."""
    if depth >= _DEPTH:
        return set(), bool(code.strip())
    return _python_sites(code, depth + 1) if python else _shell_sites(code, depth + 1)


def _command_sites(words: Sequence[str], depth: int, here: Sequence[str] = ()) -> Sites:
    words, assigned, codes = _unwrapped(words)
    keys: set[str] = set()
    unread = False

    def add(found: Sites) -> None:
        nonlocal unread
        keys.update(found[0])
        unread = unread or found[1]

    for name, value in assigned:
        if _PROXY_WORD.fullmatch(f"{name}={value}"):
            add(_proxy_sites(value))
    for code in codes:
        add(_code_sites(code, False, depth))
    if not words:
        return keys, unread
    program, args = _program(words[0]), list(words[1:])
    # The first word that runs curl or wget, whatever runs it (``stdbuf``,
    # ``watch``, ``script -q /dev/null``…).
    fetcher = next(
        (at for at, word in enumerate(words) if _program(word) in ("curl", "wget")),
        None,
    )
    if program in _SHELLS or program == "python":
        flag = next(
            (
                at
                for at, arg in enumerate(args[:-1])
                if re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", arg)
            ),
            None,
        )
        if flag is not None:
            add(_code_sites(args[flag + 1], program == "python", depth))
        for text in here:  # bash <<< "…"
            add(_code_sites(text, program == "python", depth))
    elif program == "eval":
        add(_code_sites(" ".join(args), False, depth))
    elif fetcher is not None:
        add(_fetcher_sites(_program(words[fetcher]), words[fetcher + 1 :]))
        unread = unread or any(name in _CONFIG_HOMES for name, _ in assigned)
    else:
        plain = [arg for arg in args if not arg.startswith("-")]
        network = program in _NETWORK or (
            program == "git" and bool(set(plain) & _GIT_NETWORK)
        )
        if network:
            add((set(), any(_EXPANDS.search(arg) for arg in args)))
    return keys, unread


def _shell_sites(command: Any, depth: int = 0) -> Sites:
    words = _shell_words(command)
    keys: set[str] = set()
    unread = False
    for word in words:
        keys |= _address_keys(word)
        proxy = _PROXY_WORD.match(word)
        if proxy is not None:
            keys |= _proxy_sites(proxy.group(1))[0]
        # A command substitution inside quotes ("$(…)", "`…`") runs too.
        for inner in _SUBSTITUTION.findall(word):
            more, hidden = _code_sites(inner[0] or inner[1], False, depth)
            keys |= more
            unread = unread or hidden
    commands = [(words, [])] if isinstance(command, list) else _shell_commands_of(words)
    for found, here in commands:
        more, hidden = _command_sites(found, depth, here)
        keys |= more
        unread = unread or hidden
    return keys, unread


def _texts(tree: ast.AST) -> list[str]:
    texts: list[str] = []
    parts: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts.update(id(value) for value in node.values)
            texts.append(_marked(node))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in parts:
            continue
        if isinstance(node.value, str):
            texts.append(node.value)
        elif isinstance(node.value, bytes):
            texts.append(node.value.decode("latin-1"))
    return texts


def _marked(node: ast.JoinedStr) -> str:
    """An f-string's text, each value it fills in as :data:`_FILLED`."""
    return "".join(
        (
            value.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
            else _FILLED
        )
        for value in node.values
    )


def _call_name(node: ast.Call) -> tuple[str, ast.expr | None]:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr, func.value
    return (func.id, None) if isinstance(func, ast.Name) else ("", None)


class _PythonNames:
    """What the names of a snippet hold: imports, one literal each, the clients made."""

    def __init__(self, tree: ast.AST) -> None:
        self.modules: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.modules[alias.asname] = alias.name
                    else:
                        top = alias.name.split(".")[0]
                        self.modules[top] = top
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    self.modules[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )
        bound: dict[str, list[ast.expr | None]] = {}
        # Names that hold a client (``requests.Session()``), by its maker.
        self.clients: dict[str, str] = {}
        self.sockets: set[str] = set()
        for node in ast.walk(tree):
            pairs: list[tuple[ast.expr, ast.expr | None]] = []
            if isinstance(node, ast.Assign):
                pairs = [(target, node.value) for target in node.targets]
            elif isinstance(node, ast.AnnAssign | ast.AugAssign):
                pairs = [(node.target, None)]
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                pairs = [(node.optional_vars, node.context_expr)]
            elif isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
                pairs = [(node.target, None)]
            elif isinstance(node, ast.NamedExpr):
                pairs = [(node.target, node.value)]
            for target, value in pairs:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        single = value if target is name else None
                        bound.setdefault(name.id, []).append(single)
                if isinstance(target, ast.Name) and isinstance(value, ast.Call):
                    maker = self.client_maker(value)
                    if maker:
                        self.clients[target.id] = maker
                    elif _call_name(value)[0] == "socket":
                        self.sockets.add(target.id)
        self.values = {
            name: values[0] for name, values in bound.items() if len(values) == 1
        }

    def dotted(self, node: ast.expr | None) -> str:
        """The dotted name *node* stands for, imports resolved (``r.get`` → ``requests.get``)."""
        if isinstance(node, ast.Name):
            return self.modules.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            head = self.dotted(node.value)
            return f"{head}.{node.attr}" if head else ""
        return ""

    def client_maker(self, node: ast.Call) -> str:
        """``Session``, ``Http``… when *node* makes an HTTP client, else ``""``."""
        dotted = self.dotted(node.func)
        name = dotted.rsplit(".", 1)[-1]
        top = dotted.split(".")[0]
        if name in _PY_CLIENTS or name == "session" and top in _PY_CLIENT_MODULES:
            return "Session" if name == "session" else name
        return ""

    def text(self, node: ast.expr | None, depth: int = 0) -> str | None:
        """The literal text *node* holds (an f-string's values marked), or ``None``."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return _marked(node)
        if isinstance(node, ast.Name) and depth < _DEPTH:
            return self.text(self.values.get(node.id), depth + 1)
        return None

    def client(self, node: ast.expr | None) -> str:
        """The maker of the client *node* is (a module counts as ``module``), or ``""``."""
        if isinstance(node, ast.Name):
            if node.id in self.clients:
                return self.clients[node.id]
            top = self.modules.get(node.id, node.id).split(".")[0]
            return "module" if top in _PY_CLIENT_MODULES else ""
        if isinstance(node, ast.Call):
            return self.client_maker(node)
        return ""


def _argument(node: ast.Call, position: int, *names: str) -> ast.expr | None:
    if len(node.args) > position:
        return node.args[position]
    return next((k.value for k in node.keywords if k.arg in names), None)


def _url_sites(url: str | None) -> Sites:
    """What an address passed to a Python HTTP call reaches."""
    if url is None:
        return set(), True
    if "://" not in url:
        return set(), bool(_EXPANDS.search(url))  # no scheme: it does not run
    return _address_sites(url)


def _python_call_sites(node: ast.Call, names: _PythonNames, depth: int) -> Sites:
    name, receiver = _call_name(node)
    dotted = names.dotted(node.func)
    top = dotted.split(".")[0]
    if name in _RUNS and node.args:
        spawns = top in _PY_SPAWNERS
        first = node.args[0]
        if isinstance(first, ast.List | ast.Tuple):
            words = [names.text(item) for item in first.elts]
            if not words:
                return set(), False
            if words[0] is None:
                return set(), spawns
            return _command_sites(
                [word if word is not None else _FILLED for word in words], depth
            )
        command = names.text(first)
        if command is None:
            return set(), spawns  # a command built at run time
        return _shell_sites(command, depth)
    if name == "setopt" and len(node.args) >= 2:
        option = node.args[0]
        what = option.attr if isinstance(option, ast.Attribute) else ""
        if what in ("URL", "PROXY"):
            text = names.text(node.args[1])
            if text is None:
                return set(), True
            if what == "PROXY":
                return _proxy_sites(text)
            return _target_sites(text)
    maker = names.client(receiver) if receiver is not None else ""
    from_client = isinstance(node.func, ast.Name) and top in _PY_CLIENT_MODULES
    if name in _PY_URL_CALLS or name in _PY_METHODS and (maker or from_client):
        # requests.request(method, url); httplib2's Http().request(url, …).
        position = 1 if name == "request" and maker != "Http" else 0
        return _url_sites(names.text(_argument(node, position, "url", "uri")))
    is_socket = isinstance(receiver, ast.Name) and receiver.id in names.sockets
    target = _argument(node, 0, "host", "address")
    if (
        name in _PY_HOST_CALLS
        or name == "connect"
        and (is_socket or isinstance(target, ast.Tuple))
    ):
        if isinstance(target, ast.Tuple) and target.elts:
            target = target.elts[0]
        host = names.text(target)
        if host is None or _EXPANDS.search(host):
            return set(), True
        return _host_sites(host)
    return set(), False


def _python_sites(code: str, depth: int = 0) -> Sites:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return set(), False  # it does not run
    keys: set[str] = set()
    for text in _texts(tree):
        keys |= _address_keys(text)
    unread = False
    names = _PythonNames(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            more, hidden = _python_call_sites(node, names, depth)
            keys |= more
            unread = unread or hidden
    return keys, unread


def _sites(tool_name: str, arguments: Mapping[str, Any]) -> Sites:
    if tool_name == "shell_exec":
        return _shell_sites(arguments.get("command"))
    if tool_name == "python_exec":
        return _python_sites(str(arguments.get("code") or ""))
    return set(), False


def site_keys(tool_name: str, arguments: Mapping[str, Any]) -> list[str]:
    """The loose keys of the sites a ``shell_exec`` or ``python_exec`` step names (#433).

    One ``ask:site:<host>`` (:func:`site_host`) per address in it: every
    ``http://`` or ``https://`` address in a word of a command or a string
    of Python code, nested in another one's path or percent-encoded in its
    query too (an address passed to ``urllib``, ``requests`` or ``httpx``,
    kept in a variable, or in an f-string whose host is written out); each
    address ``curl`` or ``wget`` is given without a scheme (a word that is
    not an option's value and is a dotted host, perhaps with a port and a
    path), wherever the command stands in the shell's syntax (``if``,
    ``( … )``, ``bash -c``); the proxy it goes through (``-x``,
    ``https_proxy=``, ``-e http_proxy=``, ``git -c http.proxy=``); and the
    literal host of ``http.client``'s connections and of a socket's. A
    percent-encoded host is decoded first; one that does not decode into a
    site gives :data:`UNREADABLE_SITE`. A host built at run time
    (``$HOST``, ``{host}``, ``%s``) gives nothing (see
    :func:`reaches_unread_site`). They only ever refuse a step that names
    a site refused earlier
    (:class:`~localmelo.melo.contracts.approvals.EarlierBlocks`): a blocked
    step never keeps them, since what was refused was the step.
    """
    return sorted(_sites(tool_name, arguments)[0])


def reaches_unread_site(tool_name: str, arguments: Mapping[str, Any]) -> bool:
    """Whether a ``shell_exec`` or ``python_exec`` step reaches a host melo cannot read.

    ``curl`` or ``wget`` with an address or a proxy whose host gives no key
    (one the shell fills in, ``$URL``; brace groups or ranges to expand;
    ``--expand-url``), one read from a file (``-K``, ``-i``, a config home)
    or none at all (``xargs curl``); another network program (``nc``,
    ``ssh``, ``git clone``…) with an argument the shell fills in; Python
    that fetches an address or connects to a host not written out in the
    code (a variable from elsewhere, text joined at run time, an f-string
    that fills in the host), or runs a command built at run time. While a
    site is refused in the task, such a step is reviewed instead of running
    as routine (#433).
    """
    return _sites(tool_name, arguments)[1]
