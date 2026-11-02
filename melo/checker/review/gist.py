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
  sends data out): a warning that quotes them.

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
from localmelo.melo.checker.review.paths import loopback
from localmelo.melo.contracts.review import ActionReview, commands_of

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
    "then",
    "more",
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
# Words that start a command line and run the command after them, with the
# options of each that take a value (so the value is not read as the program).
_WRAPPERS: Final = frozenset(
    ["sudo", "doas", "xargs", "nice", "time", "env", "command", "exec", "nohup"]
    + ["timeout", "caffeinate", "stdbuf", "ionice", "builtin"]
    # shell keywords that put a command after them
    + ["do", "then", "else", "elif", "if", "while", "until", "{", "(", "!"]
)
_WRAPPER_VALUES: Final[dict[str, frozenset[str]]] = {
    "sudo": frozenset(
        ["-u", "-g", "-h", "-p", "-C", "-D", "-R", "-T", "-U", "-r", "-t"]
    ),
    "doas": frozenset(["-u", "-C"]),
    "env": frozenset(["-u", "-C", "-S"]),
    "timeout": frozenset(["-s", "-k", "--signal", "--kill-after"]),
    "nice": frozenset(["-n"]),
    "ionice": frozenset(["-c", "-n", "-p"]),
    "xargs": frozenset(["-I", "-n", "-P", "-L", "-s", "-d", "-E", "-a"]),
}
# The ends of a block, which name no command of their own.
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
# git options before the subcommand that take a value.
_GIT_VALUED: Final = frozenset(["-C", "-c", "--git-dir", "--work-tree", "--namespace"])


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
    whoever wrote the sentence, so a reason line can name them.
    """

    id: str
    params: dict[str, Any] = field(default_factory=dict)
    by: Literal["model", "code"] = "code"
    warn: str = ""
    warn_params: dict[str, str] = field(default_factory=dict)
    note: str = ""
    apps: tuple[str, ...] = ()

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
        return out


@dataclass(frozen=True)
class _Item:
    """One thing a call does: ``operate`` an app, ``delete``, ``send_out`` or run a ``command``."""

    kind: str
    text: str = ""
    apps: tuple[str, ...] = ()


def step_gist(
    tool_name: str, arguments: Mapping[str, Any], review: ActionReview | None = None
) -> Gist | None:
    """The sentence for a call of *tool_name*, or ``None`` for a tool whose card has its own.

    The browser tools (``browser:<site>``) ask with a sentence of their own.
    """
    if tool_name.startswith("browser:"):
        return None
    words = _words(tool_name, arguments)
    items = _items(tool_name, arguments) if tool_name in _RUNNERS else []
    gone = _unmentioned(items, review, words) if words else ""
    if gone and not any(item.kind == gone for item in items):
        items = [*items, _Item(gone)]
    apps = _apps(review)
    if apps and not any(item.kind == "operate" for item in items):
        # The rules read an app out of a script the call builds from a variable.
        items = [*items, _Item("operate", apps=apps)]
    apps = apps or tuple(dict.fromkeys(n for item in items for n in item.apps))
    built = replace(_built(tool_name, arguments, items), apps=apps)
    if gone:
        return replace(built, warn=gone, warn_params={"said": words})
    if not words:
        return built
    if _risky(review):
        return replace(built, note=words)
    return replace(Gist("model", {"text": words}, by="model"), apps=apps)


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


def _unmentioned(
    items: Sequence[_Item], review: ActionReview | None, words: str
) -> str:
    """The effect of the code that the model's *words* do not cover, or ``""``."""
    found = {item.kind for item in items} & set(_EFFECTS)
    if review is not None:
        found |= set(_EFFECTS) & set(review.categories)
    for effect in _EFFECTS:
        if effect in found and not _AGREES[effect].search(words):
            return effect
    return ""


# ── What the code does ──


def _items(tool_name: str, arguments: Mapping[str, Any]) -> list[_Item]:
    """What a ``shell_exec`` or ``python_exec`` call does, in order."""
    if tool_name == "python_exec":
        return _python_items(str(arguments.get("code") or ""))
    found: list[_Item] = []
    for words in commands_of(tool_name, arguments):
        item = _classify(words)
        if item is not None:
            found.append(item)
    return found


def _classify(words: list[str]) -> _Item | None:
    """What one command does: an effect, an app it operates, or just a command to name."""
    if not words or words[0].startswith("#") or words[0] in _CLOSERS:
        return None
    program, rest = _program(words)
    effect = _command_effect(program, rest)
    if effect:
        return _Item(effect)
    named = None
    if program == "osascript":
        named = app_names.osascript_app(rest)
    elif program == "open":
        named = app_names.open_app(rest)
    if named:
        return _Item("operate", apps=(named,))
    return _Item("command", text=shlex.join([_unformatted(word) for word in words]))


def _program(words: list[str]) -> tuple[str, list[str]]:
    """The program a command runs, past wrappers (``sudo``, ``xargs``…), and its words."""
    at, wrapper = 0, ""
    while at < len(words):
        word = words[at]
        base = word.replace("\\", "/").rsplit("/", 1)[-1].lower()
        opening = base.lstrip("({")  # a group: (rm x), { rm x; }
        if word in _WRAPPER_VALUES.get(wrapper, ()):
            at += 2
        elif (
            word.startswith("-")
            or re.fullmatch(r"\d+(?:\.\d+)?[smhd]?", word)
            or re.match(r"[A-Za-z_]\w*=", word)
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


def _command_effect(program: str, rest: list[str]) -> str:
    """``delete`` or ``send_out`` when a command clearly does it, else ``""``."""
    if program in _DELETE_PROGRAMS:
        return "delete"
    if program == "find" and ("-delete" in rest or _after_exec(rest, _DELETE_PROGRAMS)):
        return "delete"
    if program == "git":
        sub, args = _git_sub(rest)
        if sub == "clean" and not {"-n", "--dry-run"} & set(args):
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
            found.append(_Item(effect))
            continue
        if _dotted(node.func, names).rpartition(".")[2] not in _PY_RUNS:
            continue
        try:
            source = ast.unparse(node)
        except (ValueError, RecursionError):
            continue
        for words in commands_of("python_exec", {"code": source}):
            item = _classify(words)
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
            parts.append(_say(f"effect_{kind}"))
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
