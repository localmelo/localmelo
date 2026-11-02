"""The few scripts and commands that only operate one app by name (``app_control``).

An allow-list, not a deny-list: AppleScript and JavaScript for Automation can
spell the same effect in too many ways (a shell run as ``do  shell  script``,
a raw ``«event»``, a loaded script, the ObjC bridge, a computed member) for a
list of bad phrases to be complete. So only the plainest forms count, and every
other script stays ``system`` ("Change system settings?"):

- an ``osascript -e`` script that is one ``tell application "<name>"`` block
  (or one-liner) with a bare name, using only plain verbs: ``activate``,
  ``quit``, ``open location`` of a web address, ``close``, ``get`` and ``set``
  of a property, and a ``repeat with … in …`` over what the app lists;
- a JXA one-liner ``Application("<name>")`` with plain property access;
- ``open -a <name>`` (or ``-b com.apple.<Name>``) with nothing else.

The app must not be a terminal, a shell, an editor that runs code, or a
system tool (:data:`SYSTEM_APPS`). Whenever the parse is unsure it returns
``None`` and the caller keeps ``system``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

# Apps that run commands or change settings, so scripting them is a system
# change: terminals, System Events (keys, login items), Settings, script and
# automation tools, editors and IDEs that run code, the login window.
SYSTEM_APPS: Final = frozenset(
    ["terminal", "iterm", "iterm2", "kitty", "alacritty", "warp", "ghostty", "wezterm"]
    + ["hyper", "tabby", "xterm", "rio", "termius", "console", "tmux", "screen"]
    + ["system events", "system settings", "system preferences", "loginwindow"]
    + ["automator", "script editor", "shortcuts", "shortcuts events", "dock"]
    + ["keychain access", "securityagent", "xcode", "python launcher", "idle"]
    + ["visual studio code", "code", "cursor", "zed", "nova", "sublime text"]
    + ["pycharm", "intellij idea", "webstorm", "goland", "clion", "rider"]
    + ["android studio", "atom", "bbedit", "textmate", "emacs", "vim", "macvim"]
    + ["com.apple.terminal", "com.googlecode.iterm2", "com.apple.systemevents"]
    + ["com.apple.systempreferences", "com.apple.scripteditor2", "com.apple.automator"]
    + ["com.apple.keychainaccess", "com.apple.shortcuts.events", "com.apple.shortcuts"]
    + ["com.apple.loginwindow", "com.apple.dock", "com.apple.console"]
)

# A bare name: letters, digits, spaces and a few marks; no dot (so no ".app"),
# no slash (so no path), no quote.
_BARE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _+-]{0,60}")
_BUNDLE: Final = re.compile(r"com\.apple\.([A-Za-z0-9]{1,40})")
_WEB: Final = re.compile(r"https?://[^\s\"'\\]{1,500}")

_TOKEN: Final = re.compile(
    r"""\s*(?:(?P<string>"[^"\\]*")|(?P<word>[A-Za-z][A-Za-z0-9_]*)"""
    r"""|(?P<number>\d+)|(?P<paren>[()]))"""
)
# The words of what an app's items are called and how a few are picked out.
_REF_WORDS: Final = frozenset(
    ["every", "tab", "tabs", "window", "windows", "document", "documents", "front"]
    + ["first", "last", "item", "items", "of", "in", "the", "whose", "url", "name"]
    + ["title", "contains", "contain", "does", "not", "is", "are", "begins", "ends"]
    + ["with", "equals", "equal", "and", "or", "true", "false", "index", "current"]
    + ["visible", "selected"]
)
_PROPERTIES: Final = frozenset(
    ["url", "name", "title", "index", "visible", "selected", "current"]
)


def system_app(name: str) -> bool:
    """Whether *name* (an app's name or bundle id) is one that runs commands or settings."""
    plain = name.strip().casefold().removesuffix(".app")
    return plain in SYSTEM_APPS or plain.replace(" ", "") in {
        known.replace(" ", "") for known in SYSTEM_APPS
    }


def bare_app(name: str) -> bool:
    """Whether *name* is a bare, ordinary app name: no path, no ``.app``, not a system tool."""
    return _BARE.fullmatch(name) is not None and not system_app(name)


def _tokens(line: str) -> list[tuple[str, str]] | None:
    """*line* as ``(kind, text)`` pairs, or ``None`` when anything else is in it."""
    found: list[tuple[str, str]] = []
    at = 0
    while at < len(line):
        if not line[at:].strip():
            break
        m = _TOKEN.match(line, at)
        if m is None:
            return None
        kind = m.lastgroup or ""
        found.append((kind, m.group(kind)))
        at = m.end()
    return found


def _is_ref(tokens: Sequence[tuple[str, str]], declared: set[str]) -> bool:
    """Whether *tokens* are a plain reference: item words, strings, numbers, the loop's names."""
    if not tokens:
        return False
    for kind, text in tokens:
        low = text.lower()
        if kind == "word" and low not in _REF_WORDS and low not in declared:
            return False
    return True


def _statement(
    tokens: list[tuple[str, str]], declared: set[str], opens: list[str]
) -> bool:
    """Whether one line of a ``tell`` body is a plain statement; a ``repeat`` opens a block."""
    words = [(kind, text.lower() if kind == "word" else text) for kind, text in tokens]
    if not words or words[0][0] != "word":
        return False
    verb, rest = words[0][1], words[1:]
    if verb in ("activate", "quit"):
        return not rest
    if verb == "open":
        return (
            len(rest) == 2
            and rest[0] == ("word", "location")
            and rest[1][0] == "string"
            and _WEB.fullmatch(rest[1][1][1:-1]) is not None
        )
    if verb in ("close", "get"):
        return _is_ref(rest, declared)
    if verb == "set":
        if len(rest) < 4 or rest[-2] != ("word", "to"):
            return False
        value = rest[-1]
        literal = value[0] in ("string", "number") or value in (
            ("word", "true"),
            ("word", "false"),
        )
        head = rest[:-2]
        if head[0] == ("word", "url"):  # an address to go to, not a script
            literal = (
                value[0] == "string" and _WEB.fullmatch(value[1][1:-1]) is not None
            )
        return (
            literal
            and head[0][0] == "word"
            and head[0][1] in _PROPERTIES
            and ("word", "of") in head
            and _is_ref(head, declared)
        )
    if verb == "repeat":
        if len(rest) < 4 or rest[0] != ("word", "with") or rest[1][0] != "word":
            return False
        if rest[2] != ("word", "in") or rest[1][1] in _REF_WORDS:
            return False
        if not _is_ref(rest[3:], declared):
            return False
        declared.add(rest[1][1])
        opens.append("repeat")
        return True
    return False


def applescript_app(script: str) -> str | None:
    """The app a plain AppleScript operates, or ``None``.

    The script must be one ``tell application "<name>"`` block or one-liner
    whose lines are all plain statements (see the module doc).
    """
    lines = [line for line in script.replace("\r", "\n").split("\n") if line.strip()]
    if not lines:
        return None
    first = _tokens(lines[0])
    if (
        first is None
        or len(first) < 3
        or [t[1].lower() for t in first[:2]]
        not in (["tell", "application"], ["tell", "app"])
        or first[2][0] != "string"
    ):
        return None
    name = first[2][1][1:-1]
    if not bare_app(name):
        return None
    declared: set[str] = set()
    opens: list[str] = []
    rest = first[3:]
    if rest:  # the one-liner: tell application "X" to <statement>
        if len(lines) != 1 or rest[0][1].lower() != "to":
            return None
        return name if _statement(rest[1:], declared, opens) and not opens else None
    body, closing = lines[1:-1], lines[-1] if len(lines) > 1 else ""
    if [w.lower() for w in closing.split()] != ["end", "tell"]:
        return None
    for line in body:
        tokens = _tokens(line)
        if tokens is None:
            return None
        low = [t[1].lower() for t in tokens]
        if low[:1] == ["end"]:
            if low != ["end", "repeat"] or not opens:
                return None
            opens.pop()
        elif not _statement(tokens, declared, opens):
            return None
    return name if not opens else None


_MEMBER: Final = (
    r"\.\s*(?:(?:activate|quit|close|name|url|title)\s*\(\s*\)"
    r"|(?:windows|tabs|documents)(?:\s*\[\s*\d+\s*\])?"
    r"|openLocation\s*\(\s*(?:\"https?://[^\s\"'\\]+\"|'https?://[^\s\"'\\]+')\s*\))"
)
_JXA: Final = re.compile(
    r"""Application\s*\(\s*(?P<q>["'])(?P<name>[^"'\\]+)(?P=q)\s*\)(?:\s*"""
    + _MEMBER
    + r")*"
)


def jxa_app(script: str) -> str | None:
    """The app a plain JXA one-liner operates, or ``None``: ``Application("Safari").activate()``."""
    text = script.strip().removesuffix(";").strip()
    m = _JXA.fullmatch(text)
    if m is None:
        return None
    name = m.group("name")
    return name if bare_app(name) else None


def osascript_app(args: Sequence[str]) -> str | None:
    """The app an ``osascript`` call operates when it is plain, else ``None``.

    Only ``-e`` pieces and ``-l AppleScript|JavaScript|JXA`` are read; a script
    file, the input, any other option or a second app is not plain.
    """
    pieces: list[str] = []
    language = "applescript"
    at = 0
    while at < len(args):
        arg = args[at]
        if arg == "-e" and at + 1 < len(args):
            pieces.append(args[at + 1])
            at += 2
        elif arg == "-l" and at + 1 < len(args):
            language = args[at + 1].lower()
            at += 2
        else:
            return None
    if not pieces:
        return None
    script = "\n".join(pieces)
    if language == "applescript":
        return applescript_app(script)
    if language in ("javascript", "jxa"):
        return jxa_app(script)
    return None


def open_app(args: Sequence[str]) -> str | None:
    """The app ``open -a <name>`` (or ``-b com.apple.<Name>``) starts, or ``None``.

    Exactly one flag and its value: no path, no arguments for the app, no
    second ``-a`` or ``-b``.
    """
    if len(args) != 2:
        return None
    flag, value = args
    if flag == "-a":
        return value if bare_app(value) else None
    if flag == "-b":
        m = _BUNDLE.fullmatch(value)
        if m is not None and not system_app(value) and not system_app(m.group(1)):
            return m.group(1)
    return None
