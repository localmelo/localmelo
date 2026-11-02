"""The one plain sentence an approval card starts with (#454).

:func:`step_gist` says what a call will do. The sentence is the model's own
words when the call carries them (a ``why``, ``description`` or ``purpose``
argument of a command or script, ``why`` of another tool, or the leading
comment of its code or command) and is built from the tool and its arguments
otherwise: never a model call just for the card.

When the model's words clearly disagree with what the code does (it deletes
files, it sends data out), the sentence is the code's real effect and the
words move into a warning. The checks are a few clear patterns, on a program's
name or a call in the syntax tree, never on text in a string.

A :class:`Gist` holds the ids of catalog entries (``gist.say.<id>``,
``gist.warn.<id>``) and their parameters, so every surface says it in its own
language. Imports only the standard library, the redaction and the contracts.
"""

from __future__ import annotations

import ast
import re
import shlex
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal

from localmelo.melo.checker.redaction import redact_secrets
from localmelo.melo.contracts.review import ActionReview, commands_of

# The arguments that may carry the model's own words about a call. Another
# tool's ``description`` or ``purpose`` is often the content it sends (an
# issue's text, an event's notes), so only ``why`` counts there.
WORD_FIELDS: Final = ("why", "description", "purpose")
OWN_WORD_FIELDS: Final = ("why",)
_RUNNERS: Final = frozenset(["shell_exec", "python_exec"])
# The longest sentence of the model's words, and of a command in a code-built one.
WORDS_CHARS: Final = 160
COMMAND_CHARS: Final = 80
# The ids of the sentences and warnings; the catalog holds one entry for each
# (``gist.say.<id>``, ``gist.warn.<id>``).
SAY_IDS: Final = (
    "model",
    "command",
    "command_apps",
    "script",
    "script_apps",
    "script_command",
    "write",
    "edit",
    "tool",
    "effect_delete",
    "effect_send_out",
)
WARN_IDS: Final = ("delete", "send_out")
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
# The words of a description that cover what the code does, per effect.
_AGREES: Final[dict[str, re.Pattern[str]]] = {
    "delete": re.compile(
        r"(?i)\b(?:delet|remov|eras|clean|clear|wip|purg|prun|trash|discard|destroy"
        r"|drop|uninstall|reset|unlink|overwrit)\w*|\brm\b"
        r"|删|移除|清理|清除|清空|擦除|丢弃|卸载|重置|清掉|覆盖"
    ),
    "send_out": re.compile(
        r"(?i)\b(?:send|upload|post|push|submit|share|transmit|publish|e-?mail|mail"
        r"|sync|back\s?up|export|report|deliver|notify|copy|transfer|deploy)\w*"
        r"|发送|发给|上传|提交|推送|分享|发布|邮件|同步|备份|导出|汇报|通知|复制|拷贝|传输|部署"
    ),
}
# Words that start a command line and run the command after them.
_WRAPPERS: Final = frozenset(
    ["sudo", "doas", "xargs", "nice", "time", "env", "command", "exec", "nohup"]
    + ["timeout", "caffeinate", "stdbuf", "ionice"]
)
_DELETE_PROGRAMS: Final = frozenset(["rm", "rmdir", "unlink", "shred", "srm", "trash"])
_SEND_PROGRAMS: Final = frozenset(["mail", "mailx", "sendmail"])
_CURL_SENDS: Final = re.compile(
    r"^(?:-[A-Za-z]*[dFT]|--data(?:-\w+)?|--form(?:-string)?|--upload-file|--json)$"
)
_METHOD_SENDS: Final = frozenset(["POST", "PUT", "PATCH", "DELETE"])
_REMOTE: Final = re.compile(r"^(?:[\w.-]+@)?[\w.-]+:(?!//)")
_PY_SENDERS: Final = frozenset(["requests", "httpx", "session", "client"])
_PY_FAMILIES: Final = frozenset(["post", "put", "patch"])


@dataclass(frozen=True)
class Gist:
    """What a call does, as catalog ids and parameters.

    ``id`` names the sentence (``gist.say.<id>``) and ``by`` says who wrote
    it: ``model`` (its own words, in ``params["text"]``) or ``code``. A
    ``warn`` id (``gist.warn.<id>``) is set when the model's words clearly
    disagree with the code: ``id`` is then the code's real effect, and
    ``warn_params["said"]`` holds the model's words. ``apps`` names the apps
    a script operates (``Safari``, ``Safari and Notes``), whoever wrote the
    sentence, so a reason line can name them.
    """

    id: str
    params: dict[str, str] = field(default_factory=dict)
    by: Literal["model", "code"] = "code"
    warn: str = ""
    warn_params: dict[str, str] = field(default_factory=dict)
    apps: str = ""

    @property
    def plain(self) -> bool:
        """Whether the card's own question says as much (a write, an edit, a tool)."""
        return self.by == "code" and not self.warn and self.id in PLAIN_IDS

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "say": {"id": self.id, "params": dict(self.params)},
            "by": self.by,
        }
        if self.warn:
            out["warn"] = {"id": self.warn, "params": dict(self.warn_params)}
        if self.apps:
            out["apps"] = self.apps
        return out


def step_gist(
    tool_name: str, arguments: Mapping[str, Any], review: ActionReview | None = None
) -> Gist | None:
    """The sentence for a call of *tool_name*, or ``None`` for a tool whose card has its own.

    The browser tools (``browser:<site>``) ask with a sentence of their own.
    """
    if tool_name.startswith("browser:"):
        return None
    words = _words(tool_name, arguments)
    apps = _apps(review)
    if words:
        gone = _unmentioned(tool_name, arguments, review, words)
        if gone:
            return Gist(
                f"effect_{gone}",
                by="code",
                warn=gone,
                warn_params={"said": words},
                apps=apps,
            )
        return Gist("model", {"text": words}, by="model", apps=apps)
    return replace(_built(tool_name, arguments, apps), apps=apps)


# ── The model's words ──


def _line(text: str, limit: int) -> str:
    """*text* as one line of at most *limit* characters, a secret redacted."""
    text = " ".join(redact_secrets(text)[0].split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _sentence(text: str) -> bool:
    """Whether *text* reads as a description: a few characters with a letter."""
    if not any(ch.isalpha() for ch in text):
        return False
    return len(text) >= 4 or (len(text) >= 2 and _CJK.search(text) is not None)


def _words(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """The model's own words for the call, or ``""``."""
    for name in WORD_FIELDS if tool_name in _RUNNERS else OWN_WORD_FIELDS:
        value = arguments.get(name)
        if isinstance(value, str):
            line = _line(value, WORDS_CHARS)
            if _sentence(line):
                return line
    return _comment(tool_name, arguments)


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


# ── What the code does ──


def _unmentioned(
    tool_name: str,
    arguments: Mapping[str, Any],
    review: ActionReview | None,
    words: str,
) -> str:
    """The effect of the code that the model's *words* do not cover, or ``""``."""
    found = _effects(tool_name, arguments)
    if review is not None:
        found |= {"delete", "send_out"} & set(review.categories)
    for effect in ("delete", "send_out"):
        if effect in found and not _AGREES[effect].search(words):
            return effect
    return ""


def _effects(tool_name: str, arguments: Mapping[str, Any]) -> set[str]:
    """``delete`` and ``send_out``, as far as clear patterns of the call show them."""
    found: set[str] = set()
    for words in commands_of(tool_name, arguments):
        found |= _command_effects(words)
    if tool_name == "python_exec":
        found |= _python_effects(str(arguments.get("code") or ""))
    return found


def _program(words: list[str]) -> tuple[str, list[str]]:
    """The program a command runs, past wrappers (``sudo``, ``xargs``…), and its words."""
    at = 0
    while at < len(words):
        word = words[at]
        base = word.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if (
            word.startswith("-")
            or re.fullmatch(r"\d+(?:\.\d+)?[smhd]?", word)
            or re.match(r"[A-Za-z_]\w*=", word)
            or base in _WRAPPERS
        ):
            at += 1
        else:
            return base, words[at + 1 :]
    return "", []


def _command_effects(words: list[str]) -> set[str]:
    program, rest = _program(words)
    found: set[str] = set()
    if (
        program in _DELETE_PROGRAMS
        or program == "find"
        and ("-delete" in rest or _after_exec(rest, _DELETE_PROGRAMS))
    ):
        found.add("delete")
    elif program == "git" and rest[:1] == ["clean"]:
        if not {"-n", "--dry-run"} & set(rest):
            found.add("delete")
    elif (
        program == "curl"
        and _curl_sends(rest)
        or program == "wget"
        and any(
            a.startswith(("--post-", "--body-")) or a.upper() == "--METHOD=POST"
            for a in rest
        )
        or program in ("scp", "rsync")
        and rest
        and _REMOTE.match(rest[-1])
        or program in _SEND_PROGRAMS
    ):
        found.add("send_out")
    return found


def _after_exec(rest: list[str], programs: frozenset[str]) -> bool:
    """Whether ``-exec``/``-execdir`` of a ``find`` runs one of *programs*."""
    for at, word in enumerate(rest[:-1]):
        if word in ("-exec", "-execdir", "-ok"):
            base = rest[at + 1].replace("\\", "/").rsplit("/", 1)[-1].lower()
            if base in programs:
                return True
    return False


def _curl_sends(rest: list[str]) -> bool:
    for at, arg in enumerate(rest):
        if _CURL_SENDS.match(arg):
            return True
        if arg in ("-X", "--request") and at + 1 < len(rest):
            if rest[at + 1].upper() in _METHOD_SENDS:
                return True
        elif arg.upper().startswith(("-X", "--REQUEST=")):
            method = arg.split("=", 1)[-1] if "=" in arg else arg[2:]
            if method.upper() in _METHOD_SENDS:
                return True
    return False


def _python_effects(code: str) -> set[str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        base = (
            func.value.id
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
            else ""
        )
        if (
            name == "rmtree"
            or (name in ("remove", "unlink", "removedirs", "rmdir") and base == "os")
            or (name in ("unlink", "rmdir") and isinstance(func, ast.Attribute))
        ):
            found.add("delete")
        has_data = any(
            kw.arg == "data"
            and not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
            for kw in node.keywords
        )
        first = node.args[0] if node.args else None
        url = (
            isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and first.value.lower().startswith("http")
        )
        if (
            (
                name in _PY_FAMILIES
                and isinstance(func, ast.Attribute)
                and (base in _PY_SENDERS or url)
            )
            or name in ("urlopen", "Request")
            and has_data
            or name in ("sendmail", "send_message")
            and isinstance(func, ast.Attribute)
        ):
            found.add("send_out")
    return found


# ── Built by code ──


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= COMMAND_CHARS else text[: COMMAND_CHARS - 1] + "…"


def _apps(review: ActionReview | None) -> str:
    """The apps the rules found a script operating, as ``Safari`` or ``Safari and Notes``."""
    if review is None or "app_control" not in review.categories:
        return ""
    for reason in review.reasons:
        found = re.match(r"(?:Operates|Opens) (.+?) through (?:a script|macOS)", reason)
        if found:
            return found.group(1)
    return ""


def _built(tool_name: str, arguments: Mapping[str, Any], apps: str) -> Gist:
    if tool_name == "shell_exec":
        if apps:
            return Gist("command_apps", {"apps": apps})
        rows = [
            row.strip()
            for row in _source(tool_name, arguments).splitlines()
            if row.strip() and not row.strip().startswith("#")
        ]
        return Gist("command", {"command": _short(" ".join(rows))})
    if tool_name == "python_exec":
        if apps:
            return Gist("script_apps", {"apps": apps})
        literal = commands_of(tool_name, arguments)
        if literal:
            return Gist("script_command", {"command": _short(shlex.join(literal[0]))})
        return Gist("script")
    if tool_name in ("file_write", "file_edit"):
        path = str(arguments.get("path") or "").rstrip("/")
        name = path.rsplit("/", 1)[-1] or "a file"
        return Gist("write" if tool_name == "file_write" else "edit", {"name": name})
    return Gist("tool", {"tool": tool_name})
