"""Shell rules: split a command into simple commands and classify each one."""

from __future__ import annotations

import base64
import os
import re
from collections.abc import Collection
from urllib.parse import urlparse

from localmelo.melo.checker.review import apps, destroys
from localmelo.melo.checker.review import files as file_rules
from localmelo.melo.checker.review import python as py_rules
from localmelo.melo.checker.review.paths import (
    SENDING,
    Context,
    Findings,
    engine_socket,
    loopback,
    own_api,
    own_port,
    reach_own,
    resolve,
    search,
    shown,
    touch,
    where,
)
from localmelo.melo.contracts import shell_lexer

MAX_DEPTH = 3
# Programs that reach a server, and HTTPie's names (method first: http POST url).
_HTTPIE = set(["http", "https", "xh", "xhs"])
_NETWORK = (
    set(["curl", "wget", "nc", "ncat", "netcat", "telnet", "socat", "websocat"])
    | _HTTPIE
)
# What another program runs for melo (see _elsewhere): files macOS opens
# by starting them, AppleScript that drives an app or a shell, launchd's
# and tmux's commands that start something.
_LAUNCHABLE = (
    ".app",
    ".command",
    ".tool",
    ".terminal",
    ".workflow",
    ".action",
    ".scpt",
    ".scptd",
    ".applescript",
    ".sh",
    ".bash",
    ".zsh",
    ".py",
    ".rb",
    ".pl",
    ".pkg",
    ".mpkg",
    ".prefpane",
    ".jar",
)
# What stands for an argument melo cannot see (the words ``xargs`` adds, the
# files a ``find -exec`` hands over). It is not a name a shell variable can
# have, so it is never taken for one.
UNKNOWN_ARG = "$(?)"
# The doubt for a command whose words melo could not read.
UNREAD = "Runs a command melo could not read."
_DRIVES = re.compile(
    r"\btell\s+(application|app)\b|do shell script|run script|Application\(|doShellScript"
    r"|\bdo script\b|«",
    re.IGNORECASE,
)
# JXA's bridge to macOS itself: it can run any program without naming an app.
_BRIDGE = re.compile(r"ObjC\b|\$\.\w")
_LAUNCHD_RUNS = set(["asuser", "submit", "start", "kickstart", "bootstrap", "load"])
_TMUX_RUNS = set(
    ["new", "new-session", "send-keys", "send", "run-shell", "run", "new-window"]
    + ["neww", "split-window", "splitw", "respawn-pane", "respawnp", "respawn-window"]
    + ["respawnw", "pipe-pane", "pipep", "source-file", "source", "if-shell", "if"]
)
# Programs that have another program run what they are given (see _elsewhere).
_ELSEWHERE = set(["open", "osascript", "launchctl", "tmux", "screen", "shortcuts"])
_ELSEWHERE |= set(
    ["automator", "docker", "podman", "nerdctl", "finch", "lima", "limactl"]
)
_ELSEWHERE |= set(["colima", "multipass", "orb", "orbctl"])
_CONTAINERS = set(["docker", "podman", "nerdctl", "finch"])
_VMS = set(["lima", "limactl", "colima", "multipass", "orb", "orbctl"])
# Interpreters melo does not read, with the options that give them code inline.
# Keyed by the lower-case program name: that is what the lookup uses.
_INTERPRETERS: dict[str, set[str]] = {
    "perl": {"-e", "-E"},
    "ruby": {"-e"},
    "node": {"-e", "--eval", "-p", "--print"},
    "nodejs": {"-e", "--eval", "-p", "--print"},
    "php": {"-r", "-B", "-R", "-E", "-F"},
    "lua": {"-e"},
    "luajit": {"-e"},
    "tclsh": set(),
    "wish": set(),
    "expect": {"-c"},
    "swift": {"-e"},
    "deno": {"eval"},
    "bun": {"-e", "--eval", "eval"},
    "rscript": {"-e"},
    "r": {"-e"},
    "julia": {"-e", "--eval", "-E"},
    "pwsh": {"-c", "-Command", "-command"},
    "groovy": {"-e"},
    "jshell": set(),
    "irb": set(),
}
_AWKS = set(["awk", "gawk", "mawk", "nawk"])
# awk code that starts a command: system(), a pipe into or out of one.
_AWK_RUNS = re.compile(r"system\s*\(|\|\s*getline|\|&|\b(print|printf)\b[^;}]*\|")
_EDITORS = set(["vim", "nvim", "vi", "ex", "view", "gvim", "mvim", "vimdiff", "emacs"])
# git settings whose value is a program git starts.
_GIT_RUNS = re.compile(
    r"(?i)^(core\.(pager|editor|sshcommand|fsmonitor|hookspath|askpass|gitproxy)"
    r"|alias\..+|.+\.(command|textconv|program|cmd|tool|helper|editor|pager)"
    r"|filter\..+|credential\..*|sequence\.editor|gpg\..+|include\..+|includeif\..+)$"
)
# Variables that make a program start another one of their naming.
_RUNS_VARS = set(
    ["GIT_PAGER", "GIT_EDITOR", "GIT_SEQUENCE_EDITOR", "GIT_SSH_COMMAND", "GIT_SSH"]
    + ["GIT_EXTERNAL_DIFF", "GIT_ASKPASS", "SSH_ASKPASS", "GIT_PROXY_COMMAND", "PAGER"]
    + ["EDITOR", "VISUAL", "MANPAGER", "LESSOPEN", "LESSCLOSE", "BROWSER", "PERL5OPT"]
    + ["RUBYOPT", "NODE_OPTIONS", "PYTHONSTARTUP", "BASH_ENV", "ENV", "PROMPT_COMMAND"]
    + [
        "DYLD_INSERT_LIBRARIES",
        "LD_PRELOAD",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
    ]
)
# Searches that walk folders on their own, and the options of searches that take a value.
_RECURSIVE = set(["rg", "ag", "ack", "fd", "fdfind"])
_SEARCH_VALUED = set(
    ["-e", "-f", "--regexp", "--file", "--include", "--exclude", "--exclude-dir"]
    + ["-A", "-B", "-C", "-m", "--max-count", "-g", "--glob", "-t", "--type", "-T"]
)
_WRITES = set([">", ">>", "&>", "&>>", ">|", ">&"])
_PY = re.compile(r"python(\d(\.\d+)?)?w?$")
_SHELLS = set(["sh", "bash", "zsh", "dash", "fish"])
_BARE = _SHELLS | set(["node", "perl", "ruby", "php", "deno", "bun"])
# The delimiter of a here-document that is read exactly: a plain word, or one
# in a pair of quotes of its own, and ending where the shell word ends.
_DELIMITER = re.compile(
    r"""[ \t]*(?:([A-Za-z_]\w*)|'([A-Za-z_]\w*)'|"([A-Za-z_]\w*)")(?=[\s;&|()<>]|$)"""
)
LARGE = re.compile(
    r"\.(dmg|pkg|zip|tar|tgz|gz|xz|bz2|7z|whl|gguf|safetensors|bin|iso|pt|onnx)$"
    r"|(^|/)(get-pip\.py|install\.sh|installer\.sh)$"
)
_SIZES = {"playwright": "about 42 MB"}
_PIP_MANIFEST = {"-r", "--requirement", "-e", "--editable"}
_PIP_TARGET = {"-t", "--target", "--prefix", "--root", "-d", "--dest"}
_PIP_VALUED = (
    _PIP_MANIFEST | _PIP_TARGET | set(["-c", "-i", "-f", "--index-url", "--python"])
)
# Programs that fetch a package and run it (see _package_run).
_PACKAGE_RUNNERS = set(["npx", "bunx", "pnpx", "uvx"])
# Writes: the last argument (cp …), every argument, or every one but the first.
_WRITERS = {
    **dict.fromkeys(["cp", "mv", "ln", "link", "rsync", "install", "ditto"], "last"),
    **dict.fromkeys(["touch", "mkdir", "tee", "truncate"], "all"),
    **dict.fromkeys(["chmod", "chown", "chgrp"], "rest"),
}


# GNU tools under the names a package manager gives them (``gln``…).
_GNU_NAMES = {"gln": "ln", "gcp": "cp", "ginstall": "install", "glink": "link"}

# Settings that loosen one of melo's safety rules, as ``melo config`` names
# them. A call does not set one, for a run (``-X``) or for good (``config
# set``): the person does. These are the keys of ``settings.WARNINGS`` in
# ``support``, which this package cannot import; a test keeps the two in step.
LOOSENING = frozenset(["tools.web_fetch_allow_local"])
_MELO = set(["melo", "localmelo"])
_MELO_MODULES = set(["localmelo", "localmelo.__main__"])


def _melo_args(argv: list[str]) -> list[str] | None:
    """The arguments melo's command line gets when *argv* runs melo, else ``None``.

    That is a program named ``melo`` (a path to it, or ``uv run melo`` once
    unwrapped), or Python run as ``-m localmelo``.
    """
    prog = os.path.basename(argv[0])
    if prog in _MELO:
        return argv[1:]
    if _PY.match(prog):
        for i, arg in enumerate(argv[1:-1], 1):
            if arg == "-m" and argv[i + 1] in _MELO_MODULES:
                return argv[i + 2 :]
    return None


def _set_keys(args: list[str]) -> list[str]:
    """The keys a melo command line sets: ``-X K=V``, ``-XK=V``, ``--set K=V``, ``--set=K=V``, ``config set K V``."""
    keys: list[str] = []
    for i, arg in enumerate(args):
        if arg in ("-X", "--set") and i + 1 < len(args):
            keys.append(args[i + 1].partition("=")[0])
        elif arg.startswith("--set="):
            keys.append(arg[6:].partition("=")[0])
        elif arg.startswith("-X") and "=" in arg:
            keys.append(arg[2:].partition("=")[0])
    if "config" in args:
        rest = args[args.index("config") + 1 :]
        if "set" in rest:
            keys.extend(rest[rest.index("set") + 1 : rest.index("set") + 2])
    return [key.strip() for key in keys]


def _settings(f: Findings, argv: list[str]) -> None:
    """A setting that loosens a safety rule is changing localmelo itself.

    A key melo cannot read (built at run time) is a doubt: it may be one.
    """
    args = _melo_args(argv)
    if args is None:
        return
    for key in _set_keys(args):
        if key in LOOSENING:
            f.add(
                "self",
                f"Sets {key}, a setting that loosens one of melo's safety rules.",
            )
        elif re.search(r"[$`*?{}\\[]", key):
            f.doubt("Sets a melo setting that melo could not name.")


def _link_sources(
    ctx: Context, f: Findings, prog: str, args: list[str], cwd: str
) -> None:
    """Record the files a command makes hard links to (``touch(..., "link")``).

    ``link``; ``ln`` without ``-s``; ``cp -l``/``--link``; BSD ``install -l``
    with ``h`` or ``m``; ``pax -rwl``; ``cpio -pl``, whose names come from
    its input (its folder stands for them); ``rsync --link-dest DIR`` (a
    relative DIR is below the destination). A hard link and its source are
    one file, so the source counts as read and written.
    """
    shorts = "".join(a[1:] for a in args if a[:1] == "-" and a[:2] != "--")
    words = [a for a in args if not a.startswith("-")]
    target_dir = _valued(args, "-t", "--target-directory")
    if target_dir is not None and target_dir in words:
        words.remove(target_dir)
    sources: list[str] = []
    named = words[:-1] if len(words) > 1 and target_dir is None else words
    symbolic = "s" in shorts or "--symbolic" in args
    linking_cp = prog == "cp" and ("l" in shorts or "--link" in args)
    if prog == "link" or prog == "ln" and not symbolic or linking_cp:
        sources = named
    elif prog == "install":
        how = _valued(args, "-l", None, joined=True)
        if how is not None and set(how) & set("hm"):
            if how in words:
                words.remove(how)
            sources = words[:-1]
    elif prog == "pax" and {"r", "w", "l"} <= set(shorts):
        sources = words[:-1]
        if not sources:
            _named_by_input(ctx, f, prog, cwd)
    elif prog == "cpio" and {"p", "l"} <= set(shorts):
        _named_by_input(ctx, f, prog, cwd)
    elif prog == "rsync":
        folder = _valued(args, None, "--link-dest")
        if folder is not None and folder in words:
            words.remove(folder)
        if folder is not None and words:
            if not folder.startswith(("/", "~", "$")):
                # rsync reads a relative one below the destination.
                folder = os.path.join(words[-1], folder)
            sources = [folder]
    for source in sources:
        touch(ctx, f, source, "link", cwd)


def _named_by_input(ctx: Context, f: Findings, prog: str, cwd: str) -> None:
    """Hard links to files its input names: melo cannot see them; their folder counts."""
    touch(ctx, f, cwd, "link", cwd)
    f.doubt(f"Makes hard links ({prog}) to files its input names.")


def _valued(
    args: list[str], short: str | None, long: str | None, *, joined: bool = False
) -> str | None:
    """The value of the option *short* (``-t DIR``; with *joined*, ``-tDIR`` too)
    or *long* (``--opt DIR``, ``--opt=DIR``) in *args*, else ``None``."""
    for i, arg in enumerate(args):
        nxt = args[i + 1] if i + 1 < len(args) else None
        if short is not None and arg == short:
            return nxt
        if short is not None and joined and arg.startswith(short) and len(arg) > 2:
            return arg[len(short) :]
        if long is not None and arg == long:
            return nxt
        if long is not None and arg.startswith(long + "="):
            return arg[len(long) + 1 :]
    return None


# Programs that run the project's own files: a test runner, a build tool, a
# package script. With code changed in this chat they are undecided.
_RUNNERS = set(["pytest", "py.test", "tox", "nox", "make", "gmake", "just"])
_RUNNERS |= set(["ninja", "rake", "gradle", "gradlew", "mvn", "npx", "bunx", "pnpx"])
_RUNNERS |= set(["jest", "vitest", "mocha", "pre-commit", "ctest", "hatch"])
# `<tool> run …` runs a script the project's configuration names.
_RUN_WRAPPERS = set(["uv", "poetry", "pdm", "hatch", "pipenv"])
# The options of ``uv run`` and of the package runners that take a value.
_RUN_OPTIONS = shell_lexer.Options(
    frozenset(), frozenset(["--with", "--python", "-p", "-P", "--project", "--from"])
)
# git commands that run the repository's hooks.
_HOOKED = set(["commit", "push", "merge", "rebase", "am", "pull", "checkout"])
_FILE_RUNNERS = set(["node", "deno", "bun", "ruby", "perl", "php", "tsx", "ts-node"])
_PACKAGE_RUNS = set(["test", "t", "run", "run-script", "start", "exec", "x"])
_BUILDS = {
    "cargo": set(["run", "test", "build", "bench", "check"]),
    "go": set(["run", "test", "generate"]),
    "swift": set(["run", "test", "build"]),
    "xcodebuild": set(["test", "build", "build-for-testing", "test-without-building"]),
}
_YARN_OWN = set(["add", "install", "remove", "init", "global", "config", "info"])
# One rule a line: a pattern over "program args…", the category ("?" for
# undecided) and the reason; {prog} is the program.
_RULES = [
    (re.compile(pattern), category, reason)
    for pattern, category, reason in (line.split(" ;; ") for line in r"""
^brew (install|reinstall|upgrade|tap)\b ;; install ;; Installs software with Homebrew.
^brew services (start|run|restart)\b ;; persistence ;; Starts a background service.
^(gem|cargo|go) install\b ;; install ;; Installs software with {prog}.
^(conda|mamba|micromamba) (install|update|remove)\b ;; install ;; Installs packages with {prog}.
^(conda|mamba|micromamba) create\b ;; env ;; Creates a {prog} environment.
^pipx (install|inject|upgrade|run|reinstall)\b ;; install ;; Installs software with pipx.
^(pyenv|nvm|asdf|fnm) install\b ;; env ;; Installs a language version with {prog}.
^virtualenv\b ;; env ;; Creates a Python environment.
^(softwareupdate|installer)\b ;; install ;; Installs system software with {prog}.
^xcode-select .*--install ;; install ;; Installs Apple's command line tools.
^(huggingface-cli|hf) download\b ;; download ;; Downloads a model from Hugging Face.
^(ollama|docker|podman) pull\b ;; download ;; Downloads an image or a model with {prog}.
^playwright install\b ;; download ;; Downloads browsers for Playwright (90 MB or more).
^uv (pip (un)?install|add|remove|tool install)\b ;; install ;; Installs packages with uv.
^uv sync\b ;; restore ;; Installs the project's locked packages with uv sync.
^uv (venv|python install)\b ;; env ;; Creates a Python environment with uv.
^(npm|pnpm|yarn|bun) .*(-g|--global)\b|^yarn global ;; install ;; Installs packages globally with {prog}.
^launchctl (load|bootstrap|enable|submit|kickstart)\b ;; persistence ;; Loads a launchd job that runs on its own.
^(at|batch)\b|^crontab(?! -l$) ;; persistence ;; Changes your scheduled jobs ({prog}).
^security (find-|dump-|add-|delete-|export|unlock-) ;; credentials ;; Reads or changes your keychain.
^defaults (write|delete|import|rename)\b ;; system ;; Changes system settings with defaults.
^(systemsetup|spctl|csrutil|tccutil|chsh|shutdown|reboot|halt|nvram|kmutil|fdesetup)\b ;; system ;; Changes system settings ({prog}).
^killall (Dock|Finder|SystemUIServer|ControlCenter|cfprefsd)\b ;; system ;; Restarts a part of macOS.
^osascript .*(System Events|administrator privileges) ;; system ;; Controls macOS through AppleScript.
^(ngrok|cloudflared|lt)\b|^(nc|ncat|netcat) .*-l|^socat .*LISTEN|^ssh .*-R ;; service ;; Lets other computers reach this Mac.
\s(0\.0\.0\.0|::|\[::\])(\s|$)|=0\.0\.0\.0 ;; service ;; Opens a network service that other computers can reach.
^(scp|sftp|ftp|mail|mailx|sendmail|ssh|nc|ncat|netcat)\b ;; ? ;; Sends data to another computer ({prog}).
^git push .*(\s-f\b|--force|\s\+) ;; delete ;; Force-pushes, which can overwrite commits on the server.
^git push\b ;; ? ;; Pushes commits to a server.
^git reset .*--hard ;; delete ;; Resets with --hard, which discards uncommitted changes.
^git clean .*\s-\w*[xX] ;; delete ;; Deletes untracked and ignored files (git clean -x).
^git config .*--(global|system) ;; persistence ;; Changes your global git settings.
^git config .*core\.hookspath|^git config .*core\.hooksPath ;; persistence ;; Points git at hooks that run on later commits.
""".strip().splitlines())
]


def _uncomment(command: str) -> str:
    """*command* without its ``#`` comments; quotes respected, newlines kept."""
    out, quote, i = [], "", 0
    while i < len(command):
        char = command[i]
        if char == "\\" and quote != "'":
            out.append(command[i : i + 2])
            i += 2
            continue
        if char == "#" and not quote and (i == 0 or command[i - 1] in " \t\n;&|()"):
            end = command.find("\n", i)
            i = end if end >= 0 else len(command)
            continue
        if char in "'\"" and quote in ("", char):
            quote = "" if quote else char
        out.append(char)
        i += 1
    return "".join(out)


def _split_heredocs(command: str) -> tuple[str, list[str], list[str], bool, str]:
    """*command* without its here-document bodies, the bodies, which ones expand, whether all were read exactly, and *command* with the bodies cut but the delimiter lines kept.

    A here-document is read only in its plain forms: ``<<WORD``, ``<<'WORD'``
    or ``<<"WORD"``, one on a line, outside quotes and comments, with its
    delimiter alone on a line later in the text. Anything else with a ``<<``
    in it (``<<-``, a delimiter that is partly quoted, escaped or followed by
    more word, two on a line, a delimiter that never comes, a line continued
    with a backslash) is not read at all: the text is returned as it is (so
    that nothing after it is taken for a body) with the flag false, and the
    caller makes the step undecided. The third list holds the bodies whose
    delimiter is not quoted: the shell runs the ``$(…)`` and backquoted
    commands written in them. The last text still has what a reader of
    ``$(…)`` needs to find where a here-document inside one ends.
    """
    if "<<" not in command:
        return command, [], [], True, command
    unreadable: tuple[str, list[str], list[str], bool, str] = (
        command,
        [],
        [],
        False,
        command,
    )
    kept: list[str] = []
    kept_lines: list[str] = []
    bodies: list[str] = []
    expanded: list[str] = []
    n, at, mark, quote = len(command), 0, 0, ""
    pending: tuple[str, bool] | None = None  # delimiter, quoted
    doubtful = False  # a substitution inside double quotes: the quotes may be off
    while at < n:
        char = command[at]
        if char == "\\" and quote != "'":
            if pending and command[at + 1 : at + 2] == "\n":
                return unreadable
            at += 2
            continue
        if quote:
            if char == quote:
                quote = ""
            elif char == "\n" and pending:
                return unreadable
            elif quote == '"' and (char == "`" or command.startswith("$(", at)):
                doubtful = True
            at += 1
            continue
        if char in "'\"":
            quote = char
        elif command.startswith("$'", at):
            return unreadable
        elif char == "#" and (at == 0 or command[at - 1] in " \t\n;&|()"):
            end = command.find("\n", at)
            at = n if end < 0 else end
            continue
        elif command.startswith("<<<", at):
            at += 3
            continue
        elif command.startswith("<<", at):
            found = _DELIMITER.match(command, at + 2)
            if found is None or command.startswith("<<-", at) or pending or doubtful:
                return unreadable
            word = found.group(1) or found.group(2) or found.group(3)
            pending = (word, found.group(1) is None)
            at = found.end()
            continue
        elif char == "\n" and pending is not None:
            placed = _terminator(command, at + 1, pending[0])
            if placed is None:
                return unreadable
            line, resume = placed
            body = command[at + 1 : line - 1] if line > at + 1 else ""
            kept.append(command[mark : at + 1])
            kept_lines.append(command[mark : at + 1] + command[line:resume])
            bodies.append(body)
            if not pending[1]:
                expanded.append(body)
            at = mark = resume
            pending = None
            continue
        at += 1
    if pending is not None:
        return unreadable  # the line of the operator is the last: no body
    kept.append(command[mark:])
    kept_lines.append(command[mark:])
    return "".join(kept), bodies, expanded, True, "".join(kept_lines)


def _terminator(text: str, at: int, word: str) -> tuple[int, int] | None:
    """Where the line that is exactly *word* starts and where the text goes on after it, else ``None``."""
    n = len(text)
    while at <= n:
        eol = text.find("\n", at)
        stop = n if eol < 0 else eol
        if text[at:stop] == word:
            return at, min(n, stop + 1)
        if eol < 0:
            return None
        at = eol + 1
    return None


# How deep a command substitution may nest to be read exactly.
_MAX_NEST = 6
_BRACES = re.compile(r"[(){}'\"`]")


def _backtick_end(text: str, at: int) -> int | None:
    """The index after the backquote that closes the one at *at*, else ``None``."""
    end, n = at + 1, len(text)
    while end < n and text[end] != "`":
        end += 2 if text[end] == "\\" else 1
    return end + 1 if end < n else None


def _substitution_end(
    text: str, at: int, depth: int = 0, inline: bool = False
) -> int | None:
    """The index of the ``)`` that closes the ``$(`` whose body starts at *at*, else ``None``.

    The body is read exactly or not at all: a quote is followed to its own
    end (a double quote reads a ``$(`` in it as a group of its own), parentheses
    must balance, and anything this reader does not follow (a comment,
    ``${…}`` holding a quote or a bracket, nesting deeper than
    :data:`_MAX_NEST`, a missing end) gives ``None``. In the body of a
    here-document a backquote and a nested here-document give ``None`` too.
    With *inline* (the substitution is in the command text itself) a backquoted
    command is skipped to its own end, and a plain here-document is read to
    its delimiter line, as the shell does, so what it holds is not scanned.
    """
    if depth > _MAX_NEST:
        return None
    n, level = len(text), 1
    pending: str | None = None  # the delimiter of a here-document whose body is next
    while at < n:
        char = text[at]
        if char == "\\":
            if pending is not None and text[at + 1 : at + 2] == "\n":
                return None
            at += 2
        elif char == "'":
            close = text.find("'", at + 1)
            if close < 0 or (pending is not None and "\n" in text[at:close]):
                return None
            at = close + 1
        elif char == '"':
            after = _quote_end(text, at + 1, depth, inline)
            if after is None or (pending is not None and "\n" in text[at:after]):
                return None
            at = after
        elif char == "`":
            end = _backtick_end(text, at) if inline else None
            if end is None:
                return None
            at = end
        elif char == "#" and text[at - 1 : at] in (
            "",
            " ",
            "\t",
            "\n",
            ";",
            "(",
            "&",
            "|",
        ):
            return None
        elif text.startswith("<<<", at) and inline:
            at += 3
        elif text.startswith("<<", at):
            found = _DELIMITER.match(text, at + 2) if inline else None
            if found is None or text.startswith("<<-", at) or pending is not None:
                return None
            pending = found.group(1) or found.group(2) or found.group(3)
            at = found.end()
        elif char == "\n" and pending is not None:
            placed = _terminator(text, at + 1, pending)
            if placed is None:
                return None
            at, pending = placed[1], None
        elif text.startswith("${", at):
            close = text.find("}", at)
            if close < 0 or _BRACES.search(text, at + 2, close):
                return None
            at = close + 1
        elif text.startswith("$(", at) and not text.startswith("$((", at):
            nested = _substitution_end(text, at + 2, depth + 1, inline)
            if nested is None:
                return None
            at = nested + 1
        else:
            level += (char == "(") - (char == ")")
            if level == 0:
                return at if pending is None else None
            at += 1
    return None


def _quote_end(text: str, at: int, depth: int, inline: bool = False) -> int | None:
    """The index after the ``"`` that closes a double-quoted string whose text starts at *at*, else ``None``."""
    n = len(text)
    while at < n:
        char = text[at]
        if char == "\\":
            at += 2
        elif char == '"':
            return at + 1
        elif char == "`":
            end = _backtick_end(text, at) if inline else None
            if end is None:
                return None
            at = end
        elif text.startswith("${", at):
            close = text.find("}", at)
            if close < 0 or _BRACES.search(text, at + 2, close):
                return None
            at = close + 1
        elif text.startswith("$(", at) and not text.startswith("$((", at):
            nested = _substitution_end(text, at + 2, depth + 1, inline)
            if nested is None:
                return None
            at = nested + 1
        else:
            at += 1
    return None


def _expanded_commands(bodies: list[str]) -> tuple[list[str], bool]:
    """The commands written as ``$(…)`` or between backquotes in here-document *bodies*, and whether all were read exactly.

    A backslash keeps a ``$`` or a backquote from starting one. Arithmetic
    (``$((…))``) is not a command. A command that cannot be balanced exactly
    (see :func:`_substitution_end`), or that holds a ``case``, is still
    returned as far as it can be read, and the flag is false.
    """
    found: list[str] = []
    exact = True
    for body in bodies:
        at, n = 0, len(body)
        while at < n:
            char = body[at]
            if char == "\\":
                at += 2
            elif body.startswith("$(", at) and not body.startswith("$((", at):
                end = _substitution_end(body, at + 2)
                inner = body[at + 2 : n if end is None else end]
                exact = exact and end is not None and not _CASE.search(inner)
                found.append(inner)
                at = n if end is None else end + 1
            elif char == "`":
                end = at + 1
                while end < n and body[end] != "`":
                    end += 2 if body[end] == "\\" else 1
                inner = body[at + 1 : min(end, n)]
                exact = exact and end < n and "\\" not in inner
                found.append(inner)
                at = end + 1
            else:
                at += 1
    return found, exact


_CASE = re.compile(r"\b(case|esac)\b")
# What can make the lexer and the shell read a command substitution differently:
# a quote or a backslash in it, or a ``case`` (whose patterns end in ``)``).
_TRICKY = re.compile(r"""['"\\]|\b(?:case|esac)\b""")


# What a comment can follow.
_BLANKS = " \t\n;&|()"


def _lexer_state(text: str, quote: str) -> str:
    """The quote the lexer is in after reading *text* from the quote *quote* (``""``, ``'"'`` or ``"'"``).

    ``"#"`` if the lexer would take a comment to start in it (it drops the
    rest of the line, the end of the substitution with it).
    """
    at, n = 0, len(text)
    while at < n:
        char = text[at]
        if quote == "'":
            quote = "" if char == "'" else quote
        elif char == "\\":
            if quote == "" or text[at + 1 : at + 2] in ('"', "\\"):
                at += 1
        elif char == '"':
            quote = "" if quote else '"'
        elif char == "'" and not quote:
            quote = "'"
        elif char == "#" and not quote and (text[at - 1] if at else "(") in _BLANKS:
            return "#"  # the lexer would drop the rest of the line
        at += 1
    return quote


def _quoted_substitutions(text: str) -> tuple[list[str], bool]:
    """The commands written as ``$(…)`` in *text* that have a quote, a backslash or a ``case`` in them, and whether all were read exactly.

    The lexer that cuts *text* into commands does not know ``$(``: a double
    quote inside a substitution inside double quotes ends the string for it,
    so a command can look like quoted text, and where the quotes of the
    substitution do not pair the way the shell pairs them, the text after it
    is read in the wrong quote. Each such command is therefore read from the
    text itself, to its own end, and reviewed as a command of its own, and
    the flag is false when its end cannot be found, it holds a ``case``, or the
    lexer ends it in another quote than it began (the rest of the text is not
    read right). A substitution in single quotes is not one.
    """
    found: list[str] = []
    exact = True
    at, n, quote = 0, len(text), ""
    while at < n:
        char = text[at]
        if quote == "'":
            close = text.find("'", at)
            if close < 0:
                break
            at, quote = close + 1, ""
        elif char == "\\":
            at += 2
        elif char == "#" and not quote and (text[at - 1] if at else " ") in _BLANKS:
            eol = text.find("\n", at)
            at = n if eol < 0 else eol
        elif char == "'" and not quote:
            quote, at = "'", at + 1
        elif char == '"':
            quote, at = ("" if quote else '"'), at + 1
        elif char == "`":
            end = _backtick_end(text, at)
            if end is None:
                break
            at = end
        elif text.startswith("$(", at) and not text.startswith("$((", at):
            end = _substitution_end(text, at + 2, 0, True)
            if end is None:
                exact, at = False, at + 2
                continue
            body = text[at + 2 : end]
            if _TRICKY.search(body):
                found.append(body)
                if _CASE.search(body) or _lexer_state(body, quote) != quote:
                    exact = False
            at = end + 1
        else:
            at += 1
    return found, exact


def _ansi_c(text: str) -> bool:
    """Whether *text* has an ANSI-C quote (``$'…'``) outside other quotes.

    The lexer ends such a quote at the first ``'``; the shell skips an
    escaped one (``$'\\''``). The two then disagree on where the next command
    starts, so a text with one is not read at all.
    """
    if "$'" not in text:
        return False
    quote, at = "", 0
    while at < len(text):
        char = text[at]
        if char == "\\" and quote != "'":
            at += 2
            continue
        if not quote:
            if text.startswith("$'", at):
                return True
            if char in "'\"":
                quote = char
        elif char == quote:
            quote = ""
        at += 1
    return False


def _backticks(text: str) -> tuple[list[str], bool]:
    """The commands written between backquotes (outside single quotes), as their own text.

    A backquoted command with spaces is several words to the lexer, so the
    pairs are found in the text itself, as ``$(…)`` is a command of its own.
    The flag is false when a backquote is never closed (a shell refuses such
    a text, so the rest is not a command).
    """
    found: list[str] = []
    at, quote = 0, ""
    while at < len(text):
        char = text[at]
        if char == "\\" and quote != "'":
            at += 2
            continue
        if char in "'\"" and quote in ("", char):
            quote = "" if quote else char
        elif char == "`" and quote != "'":
            end = at + 1
            while end < len(text) and text[end] != "`":
                end += 2 if text[end] == "\\" else 1
            if end >= len(text):
                return found, False
            found.append(text[at + 1 : end].replace("\\`", "`"))
            at = end
        at += 1
    return found, True


_ASSIGN = re.compile(r"([A-Za-z_]\w*)(?:\[[^\]]*\])?\+?=(.*)", re.S)
_NAME = re.compile(r"[A-Za-z_]\w*")
_DECLARERS = frozenset(
    ["export", "local", "declare", "typeset", "readonly", "builtin", "command"]
)
_READERS = frozenset(
    ["read", "mapfile", "readarray", "getopts", "select", "printf", "unset", "wait"]
)
# Words in front of a command that do not change which command it is.
_PREFIXES = frozenset(["builtin", "command", "exec", "time"])
# A name given a value inside arithmetic: ``n = 5``, ``n += 1``, ``n++``, ``++n``.
_ARITH = re.compile(
    r"(?<![\w$])([A-Za-z_]\w*)(?:\[[^\]]*\])?\s*(?:[-+*/%&|^]|<<|>>)?=(?!=)"
    r"|(?<![\w$])([A-Za-z_]\w*)\s*(?:\+\+|--)"
    r"|(?:\+\+|--)\s*([A-Za-z_]\w*)"
)
# ``${name:=value}`` and ``${name=value}`` give the name a value as well.
_PARAMETER = re.compile(r"\$\{([A-Za-z_]\w*):?=")
# declare -n, typeset -gn …: the name becomes a reference to another one.
_NAMEREF = re.compile(r"-[A-Za-z]*n[A-Za-z]*")
# declare -i, local -ai …: a value given to the name is evaluated as arithmetic.
_INTEGER = re.compile(r"-[A-Za-z]*i[A-Za-z]*")
# What is between brackets may be arithmetic (``a[n=1]=x``, ``${a[n=1]}``, ``[[ n=1 -eq 1 ]]``).
# Only an expression up to this many characters is scanned; a longer one is not
# skipped, it takes the standing from every name.
_MAX_SUBSCRIPT = 200
_BRACKET = re.compile(r"[\[\]]")
# ``exec {name}<file`` gives the name a descriptor number.
_FD_NAME = re.compile(r"\{([A-Za-z_]\w*)\}\s*[<>]")
# A name, or an element of one (``name[0]``), to the end of a word.
_NAME_AT = re.compile(r"([A-Za-z_]\w*)(?:\[.*\])?$", re.S)
# The builtins that give a value to the name they are given instead of an expansion.
_NAME_TARGETS = frozenset(
    ["read", "mapfile", "readarray", "getopts", "unset", "select"]
)


def _subscripts(text: str) -> list[str] | None:
    """The text between each pair of brackets (a pair inside another counts as well), else ``None``.

    ``None`` when one is longer than :data:`_MAX_SUBSCRIPT`: it cannot be
    scanned for a name given a value, and must not be skipped. A line break
    does not end one (``[[ n=1 -eq\\n 1 ]]`` goes on). The cost stays linear:
    a pair is cut out once it closes, and the first one that is too long ends
    the reading.
    """
    found: list[str] = []
    opened: list[int] = []
    for bracket in _BRACKET.finditer(text):
        at = bracket.start()
        if text[at] == "[":
            opened.append(at)
        elif opened:
            start = opened.pop()
            if at - start - 1 > _MAX_SUBSCRIPT:
                return None
            found.append(text[start + 1 : at])
    return found


def _arith_names(text: str) -> set[str]:
    """The names *text* (arithmetic) gives a value."""
    return {name for found in _ARITH.findall(text) for name in found if name}


def _head(words: list[str]) -> list[str]:
    """*words* from the command word on: past ``do``/``then``/``while``, ``VAR=x`` and ``command``.

    ``IFS= read -r pid`` and ``while read pid`` both run ``read``.
    """
    rest = shell_lexer.past_keywords(words)
    at = 0
    while at < len(rest) and (_ASSIGN.fullmatch(rest[at]) or rest[at] in _PREFIXES):
        at += 1
        while rest[at - 1] in _PREFIXES and at < len(rest) and rest[at][:1] == "-":
            at += 1
    return rest[at:]


def _arithmetic(tokens: list[str]) -> set[str]:
    """The names given a value in ``(( … ))`` / ``$(( … ))`` / ``for (( … ))`` groups.

    A group opens with two ``(`` in a row and ends where the parentheses
    balance again; the tokens between are read as arithmetic. An unclosed
    group runs to the end, and reading goes on after a group, so the cost
    stays linear.
    """
    found: set[str] = set()
    at = 0
    while at < len(tokens) - 1:
        if tokens[at] == "(" and tokens[at + 1] == "(":
            depth, end = 0, at
            while end < len(tokens):
                depth += (tokens[end] == "(") - (tokens[end] == ")")
                end += 1
                if depth == 0:
                    break
            found |= _arith_names(" ".join(tokens[at:end]))
            at = end
        else:
            at += 1
    return found


def _unknowable(head: list[str]) -> bool:
    """Whether the command *head* (from its program on) could give a value to a name it does not spell out.

    That is ``eval``, ``source``, a name reference or an integer variable
    (a value given to one is evaluated as arithmetic), a name built by an
    expansion (``read $name``, ``printf -v "$name"``), and an option word too
    long to be told from a name attached to it.
    """
    if not head:
        return False
    prog, args = head[0], head[1:]
    if prog in ("eval", "source", "."):
        return True
    if prog in _DECLARERS and any(
        _NAMEREF.fullmatch(w) or _INTEGER.fullmatch(w) for w in args
    ):
        return True
    if prog in _READERS and any(w[:1] == "-" and len(w) > 64 for w in args):
        return True
    if prog in _NAME_TARGETS:
        return any("$" in w and w[:1] != "-" for w in args)
    flag = {"printf": "-v", "wait": "-p"}.get(prog)
    if flag:
        for at, word in enumerate(args):
            if word == flag and "$" in "".join(args[at + 1 : at + 2]):
                return True
            if word.startswith(flag) and word != flag and "$" in word:
                return True
    return False


def _written(args: list[str]) -> set[str]:
    """The names the words of ``read``, ``printf -v``, ``wait -p``… may write.

    A word that is a name (or an element, ``name[0]``), and an option with the
    name attached (``-vpid``, ``-apid``, ``-ppid``).
    """
    names: set[str] = set()
    for word in args:
        for start in range(1, len(word)) if word[:1] == "-" else (0,):
            found = _NAME_AT.match(word, start)
            if found:
                names.add(found.group(1))
    return names


def _own_pids(text: str) -> set[str]:
    """The shell variables that only ever hold a process this command started.

    A name counts when every way the text gives it a value is ``name=$!`` in
    the command's own shell (not inside parentheses or a pipeline, where a
    subshell sets its own copy). Any other mention of the name where it may be
    written takes it out: another value (``name=123``, ``name=$(pgrep x)``,
    ``export name=…``), a loop or a ``read`` (``read -a name``, ``printf -v
    name``, ``wait -pname``), arithmetic (``(( name = 1 ))``, ``let``,
    ``a[name=1]``, an integer variable), a descriptor (``{name}<file``), and a
    command that could set one without spelling it out (``eval``, ``source``,
    a name reference, ``read $x``), which takes all names out.
    """
    if "$!" not in text and "${!}" not in text:
        return set()  # no process was started here (and nothing to lex again)
    try:
        tokens = shell_lexer.tokens(text)
    except ValueError:
        return set()
    held: set[str] = set()
    other: set[str] = _arithmetic(tokens) | set(_PARAMETER.findall(text))
    other |= set(_FD_NAME.findall(text))
    subscripts = _subscripts(text)
    if subscripts is None:
        return set()  # a bracket expression too long to scan: trust no name
    for inside in subscripts:
        other |= _arith_names(inside)
    words: list[str] = []
    depth, before = 0, ""
    for token in [*tokens, ";"]:
        if token not in shell_lexer.CUTS:
            if not shell_lexer.OPERATOR.fullmatch(token):
                words.append(token)
            continue
        piped = token in ("|", "|&")
        if words:
            if _unknowable(_head(words)):
                return set()
            if piped and words[-1] == "}":
                return set()  # a group that is piped runs in a subshell
            _bindings(words, depth, held, other, piped or before in ("|", "|&"))
        words, before = [], token
        if token == "(":
            depth += 1
        elif token == ")":
            depth = max(0, depth - 1)
    return held - other


def _bindings(
    words: list[str], depth: int, held: set[str], other: set[str], piped: bool = False
) -> None:
    """Sort the names one simple command gives a value into *held* (``$!``) and *other*.

    *piped* says the command is part of a pipeline, which runs in a subshell
    of its own: what it sets is not the shell's.
    """
    head = _head(words)
    lead = words
    while lead and lead[0] in shell_lexer.KEYWORDS:
        lead = lead[1:]
    if lead[:1] == ["coproc"] and lead[2:3] == ["{"] and _NAME.fullmatch(lead[1]):
        other.add(lead[1])  # coproc NAME { …; } makes NAME an array
    if head[:1] in (["for"], ["select"]) and len(head) > 1:
        other.add(head[1])
    elif head[:1] and head[0] in _READERS:
        other.update(_written(head[1:]))
    elif head[:1] == ["let"]:
        other.update(_arith_names(" ".join(head[1:])))
    core = words[1:] if words[0] in _DECLARERS else words
    mine = [_ASSIGN.fullmatch(w) for w in core]
    for word, found in zip(core, mine, strict=True):
        if found is None:
            continue
        name, value = found.group(1), found.group(2)
        alone = all(m is not None for m in mine) and "[" not in word.split("=")[0]
        if value in ("$!", "${!}") and alone and depth == 0 and not piped:
            held.add(name)
        else:
            other.add(name)
    if words[0] in _DECLARERS:
        other.update(w for w in words[1:] if _NAME.fullmatch(w))


def review_shell(
    ctx: Context, f: Findings, command: str, status_read: bool = False
) -> None:
    """Review *command*; *status_read* says the caller goes by its exit status."""
    if ctx.depth > MAX_DEPTH:
        return f.doubt("Nests commands too deeply for melo to check.")
    site = f.site
    try:
        _review_shell(ctx, f, command, status_read)
    finally:
        f.site = site


def _is_runner(argv: list[str]) -> bool:
    """Whether the command runs tests or a build (a program of ``_RUNNERS``)."""
    return _runs_project(os.path.basename(argv[0]), argv[1:]) or any(
        os.path.basename(word) in _RUNNERS for word in argv
    )


def _pipe_status(
    f: Findings,
    text: str,
    segments: list[shell_lexer.Command],
    status_read: bool,
) -> None:
    """Note a pipeline whose exit status something goes by, without ``pipefail``.

    The status of ``pytest | tail`` is tail's, so ``pytest | tail && git
    commit`` commits after failed tests. That is so when the command goes on
    with ``&&`` or ``||`` after a pipeline that runs tests or a build, or,
    with *status_read* (the code reads the status), when the command ends
    with a pipeline.
    """
    if "pipefail" in text or "PIPESTATUS" in text:
        return
    for i, (op, argv, _) in enumerate(segments):
        after = segments[i + 1][0] if i + 1 < len(segments) else None
        if op not in ("|", "|&") or after in ("|", "|&"):
            continue  # not the last command of a pipeline
        first = i
        while first > 0 and segments[first][0] in ("|", "|&"):
            first -= 1
        if after in ("&&", "||"):
            gated = any(_is_runner(c[1]) for c in segments[first:i] if c[1])
        else:
            gated = status_read and after is None
        if gated:
            tail = os.path.basename(argv[0]) if argv else "the last command"
            f.doubt(
                "Relies on the exit status of a pipeline: it is the last command's "
                f"({tail}), not the earlier one's. Use set -o pipefail."
            )


def _review_shell(
    ctx: Context, f: Findings, command: str, status_read: bool = False
) -> None:
    stripped, bodies, expanded, exact, scan = _split_heredocs(command)
    text = _uncomment(stripped)
    if not exact:
        f.doubt("Has a here-document that melo cannot read exactly.")
    if _ansi_c(text):
        f.doubt("Uses a quote melo cannot read ($'…').")
    try:
        segments = shell_lexer.commands(shell_lexer.tokens(text))
    except ValueError:
        return f.doubt("Has a quote that does not close.")
    _pipe_status(f, text, segments, status_read)
    outer, f.started = f.started, _own_pids(text)
    try:
        ticked, closed = _backticks(text)
        if not closed:
            f.doubt("Has a backquote that does not close.")
        for inner in ticked:
            review_shell(ctx.deeper(), f, inner)
        quoted, sure = _quoted_substitutions(scan)
        if not sure:
            f.doubt("Has a command in $(…) that melo cannot read exactly.")
        for inner in quoted:
            review_shell(ctx.deeper(), f, inner)
        # The shell runs what an unquoted here-document holds in $(…) or
        # backquotes.
        inners, readable = _expanded_commands(expanded)
        if not readable:
            f.doubt("Has a command in a here-document that melo cannot read exactly.")
        for inner in inners:
            review_shell(ctx.deeper(), f, inner)
        _run_segments(ctx, f, segments, bodies, set(ticked))
    finally:
        f.started = outer


def _run_segments(
    ctx: Context,
    f: Findings,
    segments: list[shell_lexer.Command],
    bodies: list[str],
    ticked: Collection[str] = (),
) -> None:
    cwd, before, url, bare_before = ctx.root, "", "", False
    said: str | None = None  # what the last command printed, when literal
    # The last command's site, and the folders it searched when it was a find.
    last_site, searched = "", list[str]()
    for op, argv, redirs in segments:
        # A finding's site: the command as its words (see Findings.hits).
        f.site = "sh:" + " ".join([op, *argv, *(k + t for k, t in redirs)])[:300]
        if op in ("|", "|&"):
            # What a search printed goes on to another command: it reads it.
            for root in searched:
                search(ctx, f, root, cwd)
        body, writes = None, []
        for kind, target in redirs:
            if kind in _WRITES and not target.isdigit():
                touch(ctx, f, target, "write", cwd)
                writes.append((kind, target))
            elif kind == "<":
                touch(ctx, f, target, "read", cwd)
            elif kind == "<<" and bodies:
                body = bodies.pop(0)
        for token in argv:
            for inner in re.findall(r"\$\((.*)\)|`([^`]*)`", token, re.S):
                # A backquoted command that was read from the text is done.
                if (inner[0] or inner[1]) not in ticked:
                    review_shell(ctx.deeper(), f, inner[0] or inner[1])
        if argv[1:2] == ["run"] and os.path.basename(argv[0]) in _RUN_WRAPPERS:
            file_rules.runner(ctx, f, f"{os.path.basename(argv[0])} run")
        readings = _unwrap(ctx, f, argv)
        state = cwd, said, before, url, bare_before, last_site, searched
        # Each reading starts from the commands before it; the first goes
        # last, so what it leaves leads on to the next command.
        for argv in [*readings[1:], *readings[:1]]:
            cwd, said, before, url, bare_before, last_site, searched = state
            if not argv:
                continue
            prog = os.path.basename(argv[0]).lower()
            piped = said if op in ("|", "|&") else None
            said = _literal_output(argv, body, piped)
            _writes_code(ctx, f, argv, said, writes, cwd)
            if prog == "cd":
                cwd = resolve(ctx, (argv + ["~"])[1], cwd) or cwd
                continue
            flags = all(
                a.startswith("-") and a not in ("-c", "-e", "-m") for a in argv[1:]
            )
            bare = (prog in _BARE or bool(_PY.match(prog))) and flags
            links = [a for a in argv if "://" in a]
            if bare and body is not None:
                _code(ctx, f, prog, body)
            elif (
                bare
                and op in ("|", "|&")
                and before == "fetch"
                or (op == "<(" and bare_before and prog in ("curl", "wget"))
            ):
                host = urlparse(url or (links + [""])[0]).netloc or "the internet"
                # The site is the whole pipeline, its URL with it: a new address
                # in an edit is a new finding.
                site = f.site
                f.site = f"{last_site} {site}"[:600]
                f.add("remote_code", f"Runs code downloaded from {host}.")
                if "get-pip" in url + " ".join(links):
                    f.add("install", "Installs pip with get-pip.py.")
                f.site = site
            elif bare and op in ("|", "|&") and before == "decode":
                f.add("opaque", "Runs encoded text as code.")
            elif bare and op in ("|", "|&"):
                f.doubt(f"Runs code that another command prints, with {prog}.")
            given = op in ("|", "|&") or any(
                k in ("<", "<<", "<<-", "<<<") for k, _ in redirs
            )
            run_argv(ctx, f, argv, cwd, given)
            decode = prog in ("base64", "xxd", "openssl") and bool(
                {"-d", "-D", "--decode", "-r"} & set(argv)
            )
            fetch = prog in ("curl", "wget") and not _output(argv)[0]
            before = "fetch" if fetch else "decode" if decode else ""
            url, bare_before = (links + [""])[0], bare
            last_site = f.site
            searched = _find_roots(argv[1:]) if prog == "find" else []


def _unwrap(ctx: Context, f: Findings, argv: list[str]) -> list[list[str]]:
    """Each program *argv* may run, with its words: past keywords, ``VAR=x`` and wrappers.

    The words before the program are read by
    :func:`~localmelo.melo.contracts.shell_lexer.unwrap` (``sudo``, ``env``,
    ``timeout``…); here they are judged, and ``busybox``, ``uv run`` and the
    package runners are read past too. A wrapper option melo does not know
    is a doubt, and the program of each reading of it is reviewed. A command
    line a wrapper has a shell run (``env -S``) is reviewed as a command.
    With ``PATH=.:$PATH`` (a folder named relatively first), a program named
    bare may be a file there: it is named by that path then (``./tool``).
    """
    programs: list[list[str]] = []
    todo = [("", argv, False)]  # the folder PATH= names, words, a program yet
    while todo:
        local, argv, ready = todo.pop()
        if ready:
            programs.append(
                [os.path.join(local, argv[0]), *argv[1:]]
                if local and argv and "/" not in argv[0]
                else argv
            )
            continue
        # xargs adds arguments melo cannot see.
        found = shell_lexer.unwrap(argv, [UNKNOWN_ARG])
        if found.unread:
            f.doubt(UNREAD)
        local = local or _note_assignments(f, found.assignments)
        for wrapper in found.wrappers:
            if wrapper in ("sudo", "doas"):
                f.add("system", "Runs a command as the administrator.")
            elif wrapper in ("nohup", "setsid", "disown"):
                f.add("persistence", "Keeps a program running after this call ends.")
        for code in found.codes:
            review_shell(ctx.deeper(), f, code)
        for argv in reversed(found.readings):
            prog = os.path.basename(argv[0]).lower() if argv else ""
            if prog == "busybox" and argv[1:]:
                todo.append((local, argv[1:], False))  # the applet is the program
                continue
            if prog in _RUN_WRAPPERS and argv[1:2] == ["run"]:
                start = 2
            elif _package_run(f, prog, argv):
                start = 1 if prog in _PACKAGE_RUNNERS else 2  # past dlx / exec / x
            else:
                todo.append((local, argv, True))
                continue
            # The runners have no table of all their options: read as before.
            passed = shell_lexer.past_options(argv[start:], _RUN_OPTIONS)
            here = local or _note_assignments(f, passed.assignments)
            todo.append((here, argv[start + passed.ends[0] :], False))
    return programs


def _note_assignments(f: Findings, assignments: list[tuple[str, str]]) -> str:
    """Note each ``VAR=value`` that has a program start another; the folder ``PATH=`` puts first."""
    local = ""
    for name, value in assignments:
        _runs_var(f, name)
        local = local or _local_path(name, value)
    return local


def _local_path(name: str, value: str) -> str:
    """For ``PATH=…``: the first folder it names relatively (``.``, ``bin``), else ``""``."""
    if name != "PATH":
        return ""
    for part in value.split(":"):
        if part in ("", ".") or not part.startswith(("/", "$", "~")):
            return "./" if part in ("", ".") else part
    return ""


def _package_run(f: Findings, prog: str, argv: list[str]) -> bool:
    """``npx pkg``, ``uvx pkg``, ``pnpm dlx pkg`` and the like: a doubt, then its program.

    They fetch a package when it is not there and run it; melo reads
    neither the package nor what it runs.
    """
    sub = argv[1:2]
    runs = (
        prog in _PACKAGE_RUNNERS
        or prog in ("pnpm", "yarn")
        and sub == ["dlx"]
        or prog == "npm"
        and sub in (["exec"], ["x"])
        or prog == "bun"
        and sub == ["x"]
    )
    if runs:
        words = [
            a
            for a in argv[1:]
            if not a.startswith("-") and a not in ("dlx", "exec", "x")
        ]
        f.doubt(
            f"Runs the package {words[0] if words else 'it names'} with {prog}, which may "
            "download it; melo does not read it."
        )
    return runs


def _runs_var(f: Findings, name: str) -> None:
    """``VAR=value`` before a command: a doubt when *VAR* names a program to start."""
    if name in _RUNS_VARS or name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
        f.doubt(f"Sets {name}, which can make a program start another one.")


def unwrap(ctx: Context, f: Findings, argv: list[str]) -> list[list[str]]:
    """Each program *argv* may run (see _unwrap), for a Python process call."""
    return [found for found in _unwrap(ctx, f, argv) if found] or [argv]


def _find_roots(args: list[str]) -> list[str]:
    """The folders a ``find`` with *args* searches."""
    end = next((i for i, a in enumerate(args) if a[:1] in "-(!"), len(args))
    return args[:end] or ["."]


def _method(argv: list[str]) -> str:
    """The method a ``curl``, ``wget`` or HTTPie command names, upper case, else ``""``.

    ``curl -X POST``, ``-XPOST``, ``--request POST``, ``--request=POST``,
    ``curl --json`` (a POST), ``wget --method=POST`` and ``http POST url``.
    """
    prog, args = os.path.basename(argv[0]), argv[1:]
    for i, arg in enumerate(args):
        nxt = args[i + 1] if i + 1 < len(args) else ""
        if arg in ("-X", "--request", "--method"):
            return nxt.upper()
        if arg.startswith(("--request=", "--method=")):
            return arg.split("=", 1)[1].upper()
        if prog == "curl" and re.fullmatch(r"-X\w+", arg):
            return arg[2:].upper()
        if prog == "curl" and arg in ("--json", "--json-data"):
            return "POST"
    words = [a for a in args if not a.startswith("-")]
    if prog in _HTTPIE and words and words[0].upper() in (*SENDING, "GET", "HEAD"):
        return words[0].upper()
    return ""


def _output(argv: list[str]) -> tuple[str, bool]:
    """The file ``curl``/``wget`` writes (``""``: standard output); whether it sends."""
    target, sends, args = "", False, argv[1:]
    links = [a for a in args if "://" in a]
    remote = os.path.basename(urlparse(links[0]).path) if links else ""
    curl = argv[0].endswith("curl")
    for i, arg in enumerate(args):
        nxt = args[i + 1] if i + 1 < len(args) else ""
        if arg in ("--output", "--output-document") or re.fullmatch(r"-\w*o", arg):
            target = nxt
        elif re.fullmatch(r"-\w*O", arg) or arg == "--remote-name":
            target = remote if curl else nxt
        elif (
            re.match(r"-d|--data|-F|--form|-T|--upload|--post", arg)
            or arg in ("-X", "--request")
            and nxt.upper() in ("POST", "PUT", "PATCH")
        ):
            sends = True
    if not curl and not target and not sends:
        target = remote
    return ("" if target == "-" else target), sends


def run_argv(
    ctx: Context, f: Findings, argv: list[str], cwd: str, stdin: bool = False
) -> None:
    """Classify one simple command, from a shell or a Python process call.

    *stdin* says its input is given (a pipe, a heredoc, ``input=``): an
    interpreter with no program of its own runs that input.
    """
    prog, args = os.path.basename(argv[0]).lower(), argv[1:]
    prog = _GNU_NAMES.get(prog, prog)
    if re.search(r"[$`]", argv[0]):
        f.doubt("Runs a program melo could not name.")
    # Hard links: one file under two names, so the source counts as written.
    _link_sources(ctx, f, prog, args, cwd)
    _settings(f, argv)
    words = [a for a in args if not a.startswith("-")]
    for arg in dict.fromkeys(words):  # a word said again names the same place
        path = None if "://" in arg else resolve(ctx, arg.lstrip("@"), cwd)
        if path and where(ctx, path, write=False) == "credentials":
            touch(ctx, f, arg.lstrip("@"), "read", cwd)
    plain = args[2:] if prog == "git" and args[:1] in (["-C"], ["-c"]) else args
    line = " ".join([prog, *plain])
    for pattern, category, reason in _RULES:
        if pattern.search(line):
            text = reason.format(prog=prog)
            f.doubt(text) if category == "?" else f.add(category, text)
    if _PY.match(prog) or prog == "?":
        _python(ctx, f, argv, cwd)
    elif re.fullmatch(r"pip3?(\.\d+)?", prog):
        _pip(ctx, f, args, argv[0] if "/" in argv[0] else "")
    elif prog in ("npm", "pnpm", "yarn", "bun"):
        if words[:1] in (["add"], ["install"], ["i"]) and len(words) > 1:
            f.add("install", f"Installs {' '.join(words[1:3])} with {prog}.")
        elif words[:1] in (["install"], ["i"], ["ci"]) or prog == "yarn" and not words:
            f.add("restore", f"Installs the project's packages with {prog}.")
    elif prog in ("curl", "wget"):
        target, sends = _output(argv)
        host = urlparse(([a for a in args if "://" in a] + [""])[0])
        method = _method(argv)
        if method in SENDING:
            f.add(
                "send_out",
                f"Sends a {method} request to {host.netloc or 'a server'}.",
            )
        elif sends:
            f.doubt(f"Sends data to {host.netloc or 'a server'}.")
        if target:
            download(ctx, f, host.geturl(), target, cwd)
    elif prog in _HTTPIE and _method(argv) in SENDING:
        host = urlparse(([a for a in args if "://" in a] + [""])[0])
        f.add(
            "send_out",
            f"Sends a {_method(argv)} request to {host.netloc or 'a server'}.",
        )
    elif prog in ("rm", "rmdir", "unlink", "trash"):
        for arg in words:
            touch(ctx, f, arg, "delete", cwd)
    elif prog == "find":
        roots = _find_roots(args)
        if "-delete" in args:
            for root in roots:
                touch(ctx, f, root, "delete", cwd)
        if {"-exec", "-execdir", "-ok", "-okdir"} & set(args):
            for root in roots:
                search(ctx, f, root, cwd)
        for flag in {"-exec", "-execdir", "-ok"} & set(args):
            rest = args[args.index(flag) + 1 :]
            rest = rest[: next((i for i, a in enumerate(rest) if a in ";+"), len(rest))]
            if rest:
                # A launcher gets each file found, which melo cannot name.
                found = (
                    UNKNOWN_ARG if os.path.basename(rest[0]) in _ELSEWHERE else roots[0]
                )
                run_argv(ctx, f, [found if a == "{}" else a for a in rest], cwd)
    elif prog in _WRITERS and not re.search(r"\s[\w.-]+@?[\w.-]*:", line):
        mode = _WRITERS[prog]
        targets = words[-1:] if mode == "last" and len(words) > 1 else words
        for arg in targets[1:] if mode == "rest" else targets:
            touch(ctx, f, arg, "write", cwd)
        for source in words[:-1] if prog == "mv" else []:
            touch(ctx, f, source, "delete", cwd)
    elif prog == "sed" and any(a.startswith("-i") for a in args):
        files = [w for w in words if w]
        for arg in files if "-e" in args else files[1:]:
            touch(ctx, f, arg, "write", cwd)
    elif prog == "dd":
        for arg in (a for a in args if a.startswith("of=")):
            touch(ctx, f, arg[3:], "write", cwd)
    elif prog == "eval":
        if re.search(r"[$`]", " ".join(args)):
            f.add("opaque", "Runs text built at run time (eval).")
        else:
            review_shell(ctx.deeper(), f, " ".join(args))
    elif prog in _SHELLS and (texts := _shell_texts(args)):
        # `bash -c 'cmd'`, `bash -lc 'cmd'`, `bash -c -l 'cmd'`: the text is
        # the command. A cluster such as `-lc` is also taken, as before, for
        # a script file melo cannot read.
        for code in texts:
            if re.search(r"(\$\(|`)\s*(curl|wget)\b", code):
                f.add("remote_code", "Runs a script downloaded from the internet.")
            review_shell(ctx.deeper(), f, code)
        if _cluster_code(args) is not None:
            script(ctx, f, words[0], "shell", cwd)
    elif prog in _SHELLS and words:
        script(ctx, f, words[0], "shell", cwd)
    elif "/" in argv[0]:
        # A program named by its path, not found on PATH: one in the
        # workspace, or one made in this chat, is read (a script is judged
        # by what it says) or, when melo cannot read it (a compiled
        # program), a doubt.
        path = resolve(ctx, argv[0], cwd)
        if path and (where(ctx, path, write=False) == "inside" or _made(ctx, path)):
            script(ctx, f, argv[0], "", cwd)
    elif prog == "java":
        _java(ctx, f, args, cwd)
    if prog in _RECURSIVE or prog in ("grep", "egrep", "fgrep") and _recursive(args):
        for arg in _searched(args):
            search(ctx, f, arg, cwd)
    if prog in _NETWORK or prog == "socat":
        for word in args:
            if engine_socket(f, word):
                break
    if prog in _NETWORK:
        sending = prog not in ("curl", "wget", *_HTTPIE) or _method(argv) in SENDING
        sending = sending or (prog in ("curl", "wget") and _output(argv)[1])
        sending = sending or (prog in _HTTPIE and any("=" in w for w in words[1:]))
        _reaches(f, args, sending)
    if prog in _ELSEWHERE:
        _elsewhere(ctx, f, prog, args, cwd, stdin)
    _destroys(ctx, f, prog, args, cwd)
    _inline(f, prog, args, stdin)
    if _runs_project(prog, words):
        file_rules.runner(ctx, f, prog)
    if prog == "git" and set(plain[:1]) & _HOOKED:
        file_rules.hooks(ctx, f, f"git {plain[0]}")


def _is_cluster(arg: str) -> bool:
    """Whether *arg* is a cluster of letters with a ``c`` in it (``-lc``, ``-ic``, ``-euc``)."""
    return (
        len(arg) > 2
        and arg[0] == "-"
        and arg[1:].isascii()
        and arg[1:].isalpha()
        and "c" in arg
    )


def _cluster_code(args: list[str]) -> str | None:
    """The command text of a shell started with ``-c`` in a cluster of flags (``-lc``, ``-ic``).

    ``-c`` alone is read elsewhere; here is the text after a word such as
    ``-lc``, or ``None`` when there is none.
    """
    for at, arg in enumerate(args[:-1]):
        if arg != "-c" and _is_cluster(arg):
            return args[at + 1]
    return None


def _shell_texts(args: list[str]) -> list[str]:
    """The command texts a shell is given with ``-c``, as the shell reads them.

    A shell reads its options first (``-c``, a cluster such as ``-lc``,
    ``-o name``, ``--``) and takes the first word that is not one of them as
    the command: ``bash -c -l 'cmd'`` and ``bash -c -- 'cmd'`` run ``cmd``.
    The word right after a ``-c`` or a cluster is read too, as before.
    """
    found: list[str] = []
    if "-c" in args[:-1]:
        found.append(args[args.index("-c") + 1])
    inline = _cluster_code(args)
    if inline is not None:
        found.append(inline)
    at, with_c = 0, False
    while at < len(args):
        word = args[at]
        if word == "--":
            at += 1
            break
        if word.startswith("--"):
            at += 2 if word in ("--rcfile", "--init-file") else 1
        elif word[:1] in ("-", "+") and len(word) > 1:
            with_c = with_c or (word[0] == "-" and "c" in word[1:])
            # Each -o and -O in a cluster (-ol, -Ol, -oo) takes a word of its own.
            at += 1 + sum(char in "oO" for char in word[1:])
        else:
            break
    if with_c and at < len(args) and args[at] not in found:
        found.append(args[at])
    return found


def _destroys(ctx: Context, f: Findings, prog: str, args: list[str], cwd: str) -> None:
    """A disk erased, uncommitted work discarded, or another process stopped.

    Rules of :mod:`~localmelo.melo.checker.review.destroys`: a disk or other
    processes are a ``system`` change, discarded work is ``delete``. The
    read-only forms of the same programs (``diskutil list``, ``kill -0``,
    ``git status``) are not findings.
    """
    if prog == "diskutil":
        reason = destroys.diskutil_erases(args)
        if reason:
            f.add("system", reason)
            f.add("delete", reason)
    elif (
        prog.startswith("newfs_")
        or prog == "newfs"
        or (prog == "asr" and {"-erase", "--erase"} & set(args))
    ):
        reason = f"Erases or formats a disk ({prog})."
        f.add("system", reason)
        f.add("delete", reason)
    elif prog == "git":
        sub, rest = destroys.git_sub(args)
        plain = [a for a in rest if not a.startswith("-")]
        there = False
        if sub == "checkout" and len(plain) == 1:
            path = resolve(ctx, plain[0], cwd)
            there = bool(path) and os.path.lexists(path or "")
        found = destroys.git_discards(args, there)
        if found:
            f.add("delete", found.reason)
    elif prog in ("kill", "pkill", "killall", "skill", "killall5"):
        reason = destroys.stops_others(prog, args, f.started)
        if reason:
            f.add("system", reason)
    elif prog == "launchctl" and args[:1] == ["kill"]:
        f.add("system", "Stops a job that launchd runs (launchctl kill).")


def _searched(args: list[str]) -> list[str]:
    """The folders a ``grep -r``, ``rg`` and the like search: its words after the pattern.

    The values of options (``--include x``) are not folders; the pattern is
    the first word unless ``-e`` or ``-f`` gave it.
    """
    out, skip, given = [], False, False
    for arg in args:
        if skip:
            skip = False
        elif arg in _SEARCH_VALUED:
            skip = True
            given = given or arg in ("-e", "-f", "--regexp", "--file")
        elif arg.startswith("-"):
            given = given or arg.startswith(("--regexp=", "--file="))
        else:
            out.append(arg)
    return out if given else out[1:]


def _recursive(args: list[str]) -> bool:
    """Whether ``grep`` *args* search folders: ``-r``, ``-R``, ``--recursive``."""
    return any(
        a in ("--recursive", "--dereference-recursive")
        or re.fullmatch(r"-[A-Za-z]*[rR][A-Za-z]*", a)
        for a in args
    )


def _reaches(f: Findings, args: list[str], sending: bool = True) -> None:
    """A request to melo's own gateway: a URL or ``host:port`` to it, or ``nc host port``."""
    words = [a for a in args if not a.startswith("-") or "://" in a]
    for i, word in enumerate(words):
        if own_api(word, sending=sending):
            return reach_own(f, word)
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if loopback(word) and own_port(word, nxt):
            return reach_own(f, f"{word} {nxt}")


def _elsewhere(
    ctx: Context, f: Findings, prog: str, args: list[str], cwd: str, stdin: bool = False
) -> None:
    """A command another program runs for melo, outside the folder's sandbox.

    macOS opens an app or a script (``open``), AppleScript drives an app or
    a shell (``osascript``), launchd starts a job (``launchctl``), and a
    terminal multiplexer's server runs what it is sent (``tmux``,
    ``screen``): none of these runs under the call's own limits.
    """
    words = [a for a in args if not a.startswith("-")]
    if prog == "open":
        if {"-a", "-b"} & set(args) or any(
            re.fullmatch(r"-\w+", a) and ("a" in a or "b" in a) for a in args
        ):
            return _open_app(f, args)
        for word in words:
            scheme = link_scheme(word)
            if scheme in ("http", "https"):
                if own_api(word, sending=False):
                    reach_own(f, word)
            elif scheme in ("", "file"):
                opened(ctx, f, word, cwd)
            else:
                custom_link(f, word)
    elif prog == "osascript":
        code = " ".join(" ".join(args).split())
        given = "-e" in args
        if not given:
            # A script file, or its input when it names none: melo reads neither.
            f.add("system", "Drives another app or a shell through AppleScript.")
        elif _DRIVES.search(code) or _BRIDGE.search(code) or apps.reaches_out(args):
            _script_apps(f, args)
    elif prog == "shortcuts" and words[:1] == ["run"]:
        f.add("system", "Runs a Shortcut, which can control other apps.")
    elif prog == "automator":
        f.add("system", "Runs an Automator workflow, which can control other apps.")
    elif prog in _CONTAINERS:
        _container(ctx, f, prog, args, cwd)
    elif prog in _VMS:
        f.add(
            "system",
            f"Runs a command in a Linux machine ({prog}) that shares your files.",
        )
    elif prog == "launchctl" and words[:1] and words[0] in _LAUNCHD_RUNS:
        f.add("persistence", f"Starts a job through launchd ({words[0]}).")
    elif prog == "tmux" and words[:1] and words[0] in _TMUX_RUNS:
        f.add("persistence", f"Runs a command in a tmux server ({words[0]}).")
    elif prog == "screen" and (
        "-X" in args
        or any(re.fullmatch(r"-\w*[dD]\w*m\w*|-\w*m\w*[dD]\w*", a) for a in args)
    ):
        f.add("persistence", "Runs a command in a detached screen session.")


def _script_apps(f: Findings, args: list[str]) -> None:
    """A script that drives something: ``app_control`` only when it is plain.

    Only the allow-list of :mod:`~localmelo.melo.checker.review.apps` counts
    (one named app, plain verbs); anything else, whatever it is spelt like, is
    a system change: a shell, a program, keys, login items, the ObjC bridge, a
    terminal, System Events, an app not named outright.
    """
    named = apps.osascript_app(args)
    if named is None:
        f.add("system", "Drives another app or a shell through AppleScript.")
    else:
        f.add("app_control", f"Operates {named} through a script.")


def _open_app(f: Findings, args: list[str]) -> None:
    """``open -a Safari``: ``app_control`` for one app by name, ``system`` otherwise.

    A path (a program melo cannot vet), arguments handed to the app, a second
    ``-a`` or ``-b``, and the apps that run commands stay ``system``.
    """
    named = apps.open_app(args)
    if named is None:
        f.add("system", "Opens a program through macOS, outside melo's limits.")
    else:
        f.add("app_control", f"Opens {named} through macOS, outside melo's limits.")


_SCHEME = re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*):")


def link_scheme(target: str) -> str:
    """The scheme of a link (``shortcuts`` for ``shortcuts://run``), lower case, else ``""``."""
    found = _SCHEME.match(target)
    return found.group(1).lower() if found else ""


def custom_link(f: Findings, target: str) -> bool:
    """Record a link of a custom scheme (anything but http, https and file) as ``system``.

    The app that answers such a link is whichever one the person installed,
    and the link can make it run things (``shortcuts://``, ``obsidian://``,
    ``ssh://``), so it is never ``app_control``, which is only for an app the
    script names and the allow-list knows. A Shortcut gets the sentence
    ``shortcuts run`` has. Returns whether *target* was such a link.
    """
    scheme = link_scheme(target)
    if scheme in ("", "http", "https", "file"):
        return False
    if scheme == "shortcuts":
        f.add("system", "Runs a Shortcut, which can control other apps.")
    else:
        f.add(
            "system",
            f"Opens a {scheme}: link, which another app acts on outside melo's limits.",
        )
    return True


def opened(ctx: Context, f: Findings, target: str, cwd: str) -> None:
    """``open`` (or a browser) of *target*: high when it starts a program, a doubt when unnamed."""
    raw = target[len("file://") :] if target.startswith("file://") else target
    path = resolve(ctx, raw, cwd)
    if path is None:
        f.doubt(f"Opens a file melo cannot name, which may start a program: {target}")
    elif _starts_program(path):
        f.add("system", f"Opens {shown(ctx, path)}, which runs outside melo's limits.")


def _container(ctx: Context, f: Findings, prog: str, args: list[str], cwd: str) -> None:
    """A container run: the host folders it mounts are read and written through it."""
    words = [a for a in args if not a.startswith("-")]
    if not set(words[:2]) & {"run", "create", "exec", "start"}:
        return
    if "--privileged" in args or any(a.startswith("--privileged=true") for a in args):
        f.add("system", "Runs a container with full access to this computer.")
    for i, arg in enumerate(args):
        value = ""
        if arg in ("-v", "--volume", "--mount") and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith(("--volume=", "--mount=", "-v")) and arg not in ("-v",):
            value = arg.split("=", 1)[1] if "=" in arg[:9] else arg[2:]
        if not value:
            continue
        if arg.startswith("--mount"):
            fields = dict(p.split("=", 1) for p in value.split(",") if "=" in p)
            host = fields.get("source") or fields.get("src") or ""
        else:
            host = value.split(":", 1)[0]
        if host.startswith(("/", "~", ".", "$")):
            if "docker.sock" in host:
                f.add("system", "Gives a container control of the container engine.")
            touch(ctx, f, host, "write", cwd)


def _inline(f: Findings, prog: str, args: list[str], stdin: bool) -> None:
    """Code an interpreter runs that melo does not read: given inline, or from its input.

    Python and the shells are read (see run_argv); every other interpreter's
    code is a doubt at least, so a step cannot hand a command to perl, node,
    awk or an editor and look routine.
    """
    words = [a for a in args if not a.startswith("-")]
    if prog in _INTERPRETERS:
        inline = _INTERPRETERS[prog]
        given = any(
            a in inline
            or (prog in ("perl", "ruby") and re.fullmatch(r"-[A-Za-z]*[eE]", a))
            for a in args
        )
        if given:
            return f.doubt(
                f"Runs {prog} code given on the command line, which melo does not read."
            )
        if not words and stdin:
            return f.doubt(
                f"Runs {prog} code from its input, which melo does not read."
            )
        if prog == "swift" and words:
            return f.doubt("Runs a Swift script, which melo does not read.")
    elif prog in _AWKS:
        program = " ".join(words[:1]) if "-f" not in args else ""
        if "-f" in args or _AWK_RUNS.search(program):
            f.doubt(f"Runs commands from {prog} code, which melo does not read.")
    elif prog in _EDITORS:
        if any(
            a
            in (
                "-c",
                "--cmd",
                "-S",
                "-s",
                "-u",
                "--eval",
                "-l",
                "--load",
                "-batch",
                "--batch",
            )
            or a.startswith("+")
            and len(a) > 1
            for a in args
        ):
            f.doubt(f"Runs {prog} commands, which can start other programs.")
    elif prog == "git":
        for i, arg in enumerate(args):
            value = args[i + 1] if arg == "-c" and i + 1 < len(args) else ""
            value = value or (
                arg[len("--config-env=") :] if arg.startswith("--config-env=") else ""
            )
            if value and _GIT_RUNS.match(value.split("=", 1)[0]):
                return f.doubt(
                    f"Runs git with a setting that can start a program ({value.split('=', 1)[0]})."
                )
        if words[:1] == ["config"] and len(words) > 2 and _GIT_RUNS.match(words[1]):
            f.add("persistence", f"Points git at a program it runs later ({words[1]}).")
    elif prog in ("make", "gmake"):
        for i, arg in enumerate(args):
            value = (
                args[i + 1]
                if arg in ("-f", "--file", "--makefile") and i + 1 < len(args)
                else ""
            )
            value = value or (
                arg.split("=", 1)[1]
                if arg.startswith(("--file=", "--makefile="))
                else ""
            )
            if value in ("-", "/dev/stdin") or value.startswith("/dev/fd/"):
                return f.doubt(
                    "Runs a makefile from its input, which melo does not read."
                )


def _made(ctx: Context, path: str) -> bool:
    """Whether the program *path* is new: not there yet, or changed in this chat."""
    try:
        info = os.stat(path)
    except OSError:
        return True  # made by this step (cc -o /tmp/t && /tmp/t)
    return ctx.since is not None and max(info.st_mtime, info.st_ctime) >= ctx.since


def _java(ctx: Context, f: Findings, args: list[str], cwd: str) -> None:
    """``java`` of classes, a jar or a source file in the workspace or made in this chat."""
    places: list[str] = []
    for i, arg in enumerate(args):
        nxt = args[i + 1] if i + 1 < len(args) else ""
        if arg in ("-cp", "-classpath", "--class-path", "-jar"):
            places += nxt.split(":")
        elif arg.startswith("--class-path="):
            places += arg.split("=", 1)[1].split(":")
        elif arg.endswith(".java") and not arg.startswith("-"):
            places.append(arg)
    for place in places:
        path = resolve(ctx, place or ".", cwd)
        if (
            path is None
            or where(ctx, path, write=False) in ("inside", "root")
            or _made(ctx, path)
        ):
            return f.doubt("Runs Java code built here, which melo does not read.")


def _starts_program(path: str) -> bool:
    """Whether opening *path* starts a program: an app, a script or an executable."""
    name = path.lower().rstrip("/")
    if name.endswith(_LAUNCHABLE):
        return True
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _runs_project(prog: str, words: list[str]) -> bool:
    """Whether ``prog words…`` runs the project's files on its own."""
    if prog in _RUNNERS or os.path.basename(prog) in _RUNNERS:
        return True
    if "node_modules/.bin/" in prog:
        return True
    if prog in _FILE_RUNNERS:
        return bool(words)
    if prog in ("npm", "bun"):
        return bool(words) and words[0] in _PACKAGE_RUNS
    if prog in ("yarn", "pnpm"):
        return bool(words) and words[0] not in _YARN_OWN
    if prog == "xcodebuild":
        return bool(set(words) & _BUILDS[prog])
    return prog in _BUILDS and bool(words) and words[0] in _BUILDS[prog]


def _printf(args: list[str]) -> str | None:
    """What ``printf FORMAT ARGS…`` prints, for ``%s``, ``%b``, ``%d`` and escapes."""
    if not args:
        return None
    fmt = args[0].replace("\\n", "\n").replace("\\t", "\t")
    parts, rest, out = re.split(r"%[sbd]", fmt), list(args[1:]), []
    holes = len(parts) - 1
    while True:
        take, rest = rest[:holes], rest[holes:]
        take += [""] * (holes - len(take))
        out.append(
            parts[0] + "".join(v + p for v, p in zip(take, parts[1:], strict=True))
        )
        if not rest or not holes:
            return "".join(out).replace("%%", "%")


def _literal_output(argv: list[str], body: str | None, piped: str | None) -> str | None:
    """What a command prints when melo can tell: ``echo``, ``printf``, a
    here-document, ``cat`` or ``tee`` of a pipe, ``base64 -d`` of one."""
    prog, args = os.path.basename(argv[0]), argv[1:]
    words = [a for a in args if not a.startswith("-")]
    if prog == "echo":
        return " ".join(a for a in args if a not in ("-n", "-e", "-E")) + "\n"
    if prog == "printf":
        return _printf(args)
    if prog in ("cat", "tee") and body is not None:
        return body + "\n"
    if prog == "cat" and not words or prog == "tee":
        return piped
    if prog == "base64" and {"-d", "-D", "--decode"} & set(args) and piped:
        try:
            return base64.b64decode(piped.strip(), validate=False).decode()
        except (ValueError, UnicodeDecodeError):
            return None
    return None


def _writes_code(
    ctx: Context,
    f: Findings,
    argv: list[str],
    text: str | None,
    writes: list[tuple[str, str]],
    cwd: str,
) -> None:
    """``cat > x.py <<EOF``, ``… | tee``, ``echo … > x.sh``, ``cp a.py b.py``.

    Literal text written to a code file is classified as a write of it
    (:func:`~.files.written`), and so is a copy of a local file; ``>>``
    adds to what is there.
    """
    prog, args = os.path.basename(argv[0]), argv[1:]
    words = [a for a in args if not a.startswith("-")]
    if prog == "cp" and len(words) == 2:
        source = resolve(ctx, words[0], cwd)
        target = resolve(ctx, words[1], cwd)
        if source is None or target is None:
            return
        if os.path.isdir(target):
            target = os.path.join(target, os.path.basename(source))
        copied = file_rules._old_text(source)
        if copied:
            old = file_rules._old_text(target) or ""
            file_rules.written(ctx, f, target, copied, old)
        return
    if text is None:
        return
    targets = list(writes)
    if prog == "tee":
        mode = ">>" if "-a" in args else ">"
        targets += [(mode, a) for a in words]
    for kind, target in targets:
        path = resolve(ctx, target, cwd)
        if path is None:
            continue
        old = file_rules._old_text(path) or ""
        new = old + text if kind == ">>" else text
        file_rules.written(ctx, f, path, new, old)


def is_melo_python(ctx: Context, interp: str) -> bool:
    path = os.path.abspath(os.path.join(ctx.root, interp)) if interp else ""
    return bool(path) and (path == ctx.executable or path.startswith(ctx.prefix + "/"))


def _pip(ctx: Context, f: Findings, args: list[str], interp: str) -> None:
    """``pip …`` run by *interp* (``""`` when pip is run by its own name)."""
    sub = next((a for a in args if not a.startswith("-")), "")
    if sub not in ("install", "uninstall", "download"):
        return
    names, manifests, target, rest = [], [], "", args[args.index(sub) + 1 :]
    for before, arg in zip(["", *rest], rest, strict=False):
        if before in _PIP_MANIFEST:
            manifests.append(arg)
        elif before in _PIP_TARGET:
            target = arg
        elif not arg.startswith("-") and before not in _PIP_VALUED:
            names.append(arg)
    what = " ".join(names[:3]) or "packages"
    size = "".join(f" (downloads {_SIZES[n]})" for n in names if n in _SIZES)
    known = interp not in ("", "?") and resolve(ctx, interp)
    env = os.path.dirname(os.path.dirname(known)) if known else ""
    if sub == "download":
        f.add("download", f"Downloads {what} with pip.")
    elif "--user" in rest:
        f.add(
            "install",
            f"Installs {what} with pip into ~/.local, outside "
            f"{ctx.folder}, where every Python of yours finds it.",
        )
    elif is_melo_python(ctx, interp) and not target:
        f.add("self", f"Installs {what} into the Python that runs localmelo.")
    elif not names and manifests and env and where(ctx, env, write=True) == "inside":
        f.add(
            "restore",
            f"Installs the packages of {' '.join(manifests)} into "
            f"{shown(ctx, env)}.",
        )
    else:
        into = f" into {shown(ctx, env)}" if env else ""
        f.add("install", f"Installs {what} with pip{into}{size}.")
    if target:
        touch(ctx, f, target, "write")


def _python(ctx: Context, f: Findings, argv: list[str], cwd: str) -> None:
    """A Python interpreter run with *argv*; ``"?"`` names an unknown program.

    An interpreter in the workspace (``.venv/bin/python``) runs its
    environment's ``.pth`` files at its start: those changed in this chat
    are read.
    """
    interp, i = argv[0], 1
    where_it = resolve(ctx, interp, cwd) if "/" in interp else None
    if where_it is not None and where(ctx, where_it, write=False) == "inside":
        file_rules.site_hooks(ctx, f, os.path.basename(interp))
    while i < len(argv):
        arg, rest = argv[i], argv[i + 1 :]
        if arg in ("-V", "--version", "-h", "--help"):
            return
        if arg == "-c" and rest:
            if re.search(r"\$\(|`", rest[0]):
                f.add("opaque", "Runs Python code built at run time.")
            return py_rules.review_python(ctx.deeper(), f, rest[0])
        if arg == "-m" and rest:
            return _module(ctx, f, interp, rest[0], rest[1:])
        if not arg.startswith("-"):
            if os.path.basename(arg) == "get-pip.py":
                f.add("remote_code", "Runs get-pip.py, pip's installer.")
                return _pip(ctx, f, ["install", "pip", *rest], interp)
            if interp != "?":
                return script(ctx, f, arg, "python", cwd)
            break
        i += 2 if arg in ("-W", "-X") else 1
    if interp == "?":
        f.doubt("Runs a program melo could not name.")


def _module(ctx: Context, f: Findings, interp: str, name: str, args: list[str]) -> None:
    if name == "webbrowser":  # python -m webbrowser <link>
        for word in args:
            custom_link(f, word)
    if name == "pip":
        _pip(ctx, f, args, interp)
    elif name == "ensurepip" and not {"--version", "-h"} & set(args):
        if is_melo_python(ctx, interp) and "--user" not in args:
            f.add("self", "Installs pip into the Python that runs localmelo.")
        else:
            f.add("install", "Installs pip with ensurepip.")
    elif name in ("venv", "virtualenv"):
        dest = next((a for a in args if not a.startswith("-")), ".venv")
        f.add("env", f"Creates a Python environment in {dest}.")
    elif name == "http.server":
        bind = args[args.index("--bind") + 1] if "--bind" in args[:-1] else ""
        if bind not in ("127.0.0.1", "localhost", "::1"):
            f.add("service", "Opens a web server that other computers can reach.")
    elif name in ("playwright", "uv", "pipx", "huggingface_hub", "conda"):
        run_argv(ctx, f, [name.replace("huggingface_hub", "hf"), *args], ctx.root)
    elif name in ("pytest", "unittest", "tox", "nox", "doctest"):
        file_rules.runner(ctx, f, f"python -m {name}")
    else:
        file_rules.module(ctx, f, name, main=True)


def download(ctx: Context, f: Findings, url: str, target: str, cwd: str) -> None:
    """High outside the workspace or for installers and large types; else routine."""
    path = resolve(ctx, target, cwd)
    name = os.path.basename(urlparse(url).path) or "a file"
    kind = where(ctx, path, write=True) if path else "outside"
    if kind not in ("inside", "harmless"):
        f.add(
            "download", f"Downloads {name} to {shown(ctx, path) if path else target}."
        )
        touch(ctx, f, target, "write", cwd)
    elif kind == "inside" and (LARGE.search(path or "") or LARGE.search(url)):
        what = (
            "an installer" if re.search(r"\.(py|sh|pkg|dmg)$", url) else "a large file"
        )
        f.add("download", f"Downloads {name}, {what}.")


def script(ctx: Context, f: Findings, raw: str, kind: str, cwd: str) -> None:
    """Classify a local script the call runs, once (one level deep)."""
    path = resolve(ctx, raw, cwd)
    if not path or ctx.depth > 1:
        return f.doubt(f"Runs {raw}, which melo could not read.")
    text = f.take(path)
    if not text:
        return  # read before, or a doubt
    shebang = text.split("\n", 1)[0]
    python = (
        kind == "python" or not kind and (raw.endswith(".py") or "python" in shebang)
    )
    reader = _shebang(shebang) if not kind and not python else ""
    if reader == "osascript":
        return f.add("system", "Drives another app or a shell through AppleScript.")
    if reader and reader not in _SHELLS | {"ksh", "env", ""}:
        # A program for an interpreter melo does not read: one made in this
        # chat asks, as code given to perl or node inline does.
        if _made(ctx, path):
            f.doubt(f"Runs {os.path.basename(path)}, {reader} code melo does not read.")
        return
    _code(ctx, f, "python" if python else "sh", text, os.path.dirname(path))


def _shebang(line: str) -> str:
    """The interpreter a ``#!`` line names (``env``'s program too), else ``""``."""
    if not line.startswith("#!"):
        return ""
    words = line[2:].split()
    while words and os.path.basename(words[0]) == "env":
        words = [w for w in words[1:] if not w.startswith("-") and "=" not in w]
    return os.path.basename(words[0]) if words else ""


def _code(ctx: Context, f: Findings, prog: str, body: str, here: str = "") -> None:
    if _PY.match(prog):
        py_rules.review_python(ctx.deeper(here or None), f, body)
    elif prog in _SHELLS:
        review_shell(ctx.deeper(here or None), f, body)
