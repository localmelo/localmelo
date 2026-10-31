"""Shell rules: split a command into simple commands and classify each one."""

from __future__ import annotations

import base64
import os
import re
import shlex
from urllib.parse import urlparse

from localmelo.melo.checker.review import files as file_rules
from localmelo.melo.checker.review import python as py_rules
from localmelo.melo.checker.review.paths import (
    SENDING,
    Context,
    Findings,
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
_DRIVES = re.compile(
    r"\btell\s+(application|app)\b|do shell script|run script|Application\(|doShellScript"
    r"|\bdo script\b",
    re.IGNORECASE,
)
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
    "Rscript": {"-e"},
    "R": {"-e"},
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
_OPS = re.compile(r"&&|\|\||;;|\|&|&>>|&>|>>|>&|<&|<<<|<<|<\(|>\(|>\||[;&|()<>\n]")
_SEPARATORS = set([";", "&&", "||", "|", "|&", "&", "(", ")", "<(", ">(", ";;"]) | {
    "\n"
}
_WRITES = set([">", ">>", "&>", "&>>", ">|", ">&"])
_PY = re.compile(r"python(\d(\.\d+)?)?w?$")
_SHELLS = set(["sh", "bash", "zsh", "dash", "fish"])
_BARE = _SHELLS | set(["node", "perl", "ruby", "php", "deno", "bun"])
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")
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
_WRAPPERS = set(
    [
        "env",
        "nice",
        "time",
        "command",
        "builtin",
        "exec",
        "caffeinate",
        "timeout",
        "xargs",
        "npx",
        "stdbuf",
        "unbuffer",
        "chronic",
        "ionice",
        "taskpolicy",
        "arch",
        "xcrun",
        "watch",
        "flock",
    ]
)
# Options of one wrapper that take a value, beyond _VALUED.
_WRAPPER_VALUED: dict[str, set[str]] = {
    "stdbuf": {"-i", "-o", "-e"},
    "taskpolicy": {"-c", "-t", "-l", "-g"},
    "xcrun": {"--sdk", "-sdk", "--toolchain", "-toolchain"},
    "ionice": {"-c", "-n", "-p"},
}
_VALUED = set(
    [
        "-n",
        "-u",
        "-I",
        "-P",
        "-L",
        "-s",
        "-E",
        "-d",
        "-k",
        "--with",
        "--python",
        "-p",
        "--project",
        "--from",
    ]
)
# Writes: the last argument (cp …), every argument, or every one but the first.
_WRITERS = {
    **dict.fromkeys(["cp", "mv", "ln", "rsync", "install", "ditto"], "last"),
    **dict.fromkeys(["touch", "mkdir", "tee", "truncate"], "all"),
    **dict.fromkeys(["chmod", "chown", "chgrp"], "rest"),
}
# Programs that run the project's own files: a test runner, a build tool, a
# package script. With code changed in this chat they are undecided.
_RUNNERS = set(["pytest", "py.test", "tox", "nox", "make", "gmake", "just"])
_RUNNERS |= set(["ninja", "rake", "gradle", "gradlew", "mvn", "npx", "bunx", "pnpx"])
_RUNNERS |= set(["jest", "vitest", "mocha", "pre-commit", "ctest", "hatch"])
# `<tool> run …` runs a script the project's configuration names.
_RUN_WRAPPERS = set(["uv", "poetry", "pdm", "hatch", "pipenv"])
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


def _heredocs(command: str) -> tuple[str, list[str]]:
    """*command* without its here-document bodies, and the bodies in order."""
    lines, out, bodies, i = command.split("\n"), [], [], 0
    while i < len(lines):
        out.append(lines[i])
        i += 1
        for match in _HEREDOC.finditer(out[-1]):
            start = i
            while i < len(lines) and lines[i].strip() != match.group(2):
                i += 1
            bodies.append("\n".join(lines[start:i]))
            i += 1
    return "\n".join(out), bodies


def _segments(command: str) -> list[tuple[str, list[str], list[tuple[str, str]]]]:
    """The simple commands: the operator before each, its argv, its redirections."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars="();<>|&\n")
    lexer.whitespace, lexer.whitespace_split, lexer.commenters = " \t\r", True, ""
    tokens: list[str] = []
    for token in lexer:
        punct = token and all(c in "();<>|&\n" for c in token)
        tokens += _OPS.findall(token) if punct else [token]
    argv: list[str] = []
    redirs: list[tuple[str, str]] = []
    out, op, i = [], "", 0
    while i < len(tokens):
        if tokens[i] in _SEPARATORS:
            out.append((op, argv, redirs))
            op, argv, redirs = tokens[i], [], []
        elif _OPS.fullmatch(tokens[i]):
            if argv and argv[-1].isdigit():
                argv.pop()
            redirs.append((tokens[i], tokens[i + 1] if i + 1 < len(tokens) else ""))
            i += 1
        else:
            argv.append(tokens[i])
        i += 1
    return [s for s in [*out, (op, argv, redirs)] if s[1] or s[2]]


def review_shell(ctx: Context, f: Findings, command: str) -> None:
    if ctx.depth > MAX_DEPTH:
        return f.doubt("Nests commands too deeply for melo to check.")
    site = f.site
    try:
        _review_shell(ctx, f, command)
    finally:
        f.site = site


def _review_shell(ctx: Context, f: Findings, command: str) -> None:
    text, bodies = _heredocs(_uncomment(command))
    try:
        segments = _segments(text)
    except ValueError:
        return f.doubt("Has a quote that does not close.")
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
                review_shell(ctx.deeper(), f, inner[0] or inner[1])
        if argv[1:2] == ["run"] and os.path.basename(argv[0]) in _RUN_WRAPPERS:
            file_rules.runner(ctx, f, f"{os.path.basename(argv[0])} run")
        argv = _unwrap(f, argv)
        if not argv:
            continue
        prog = os.path.basename(argv[0])
        piped = said if op in ("|", "|&") else None
        said = _literal_output(argv, body, piped)
        _writes_code(ctx, f, argv, said, writes, cwd)
        if prog == "cd":
            cwd = resolve(ctx, (argv + ["~"])[1], cwd) or cwd
            continue
        flags = all(a.startswith("-") and a not in ("-c", "-e", "-m") for a in argv[1:])
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


def _unwrap(f: Findings, argv: list[str]) -> list[str]:
    """*argv* without ``VAR=x`` prefixes and wrappers such as ``nice`` or ``uv run``."""
    while argv:
        prog = os.path.basename(argv[0])
        if re.match(r"[A-Za-z_]\w*=", argv[0]):
            _runs_var(f, argv[0])
            argv = argv[1:]
            continue
        if prog in ("sudo", "doas"):
            f.add("system", "Runs a command as the administrator.")
        elif prog in ("nohup", "setsid", "disown"):
            f.add("persistence", "Keeps a program running after this call ends.")
        elif prog in ("uv", "poetry", "pdm", "hatch", "pipenv") and argv[1:2] == [
            "run"
        ]:
            argv = argv[1:]
        elif prog not in _WRAPPERS and prog != "gtimeout":
            return argv
        rest, i = argv[1:], 0
        valued = _VALUED | _WRAPPER_VALUED.get(prog, set())
        while i < len(rest) and (
            rest[i][:1] == "-" or "=" in rest[i] or "timeout" in prog and i == 0
        ):
            if "=" in rest[i] and rest[i][:1] != "-":
                _runs_var(f, rest[i])
            i += 2 if rest[i] in valued else 1
        if prog == "flock" and i < len(rest):
            i += 1  # its lock file, then the command
        # xargs adds arguments melo cannot see.
        argv = rest[i:] + (["$value"] if prog == "xargs" and rest[i:] else [])
    return argv


def _runs_var(f: Findings, assignment: str) -> None:
    """``VAR=value`` before a command: a doubt when *VAR* names a program to start."""
    name = assignment.split("=", 1)[0]
    if name in _RUNS_VARS or name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
        f.doubt(f"Sets {name}, which can make a program start another one.")


def unwrap(f: Findings, argv: list[str]) -> list[str]:
    """*argv* as the program it runs (see _unwrap), for a Python process call."""
    return _unwrap(f, argv) or argv


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
    prog, args = os.path.basename(argv[0]), argv[1:]
    if re.search(r"[$`]", argv[0]):
        f.doubt("Runs a program melo could not name.")
    words = [a for a in args if not a.startswith("-")]
    for arg in words:
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
                    "$value" if os.path.basename(rest[0]) in _ELSEWHERE else roots[0]
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
    elif prog in _SHELLS and "-c" in args[:-1]:
        code = args[args.index("-c") + 1]
        if re.search(r"(\$\(|`)\s*(curl|wget)\b", code):
            f.add("remote_code", "Runs a script downloaded from the internet.")
        review_shell(ctx.deeper(), f, code)
    elif prog in _SHELLS and words:
        script(ctx, f, words[0], "shell", cwd)
    elif argv[0].startswith(("./", "../", "/")):
        path = resolve(ctx, argv[0], cwd)
        if path and where(ctx, path, write=False) == "inside":
            script(ctx, f, argv[0], "", cwd)
    if prog in _RECURSIVE or prog in ("grep", "egrep", "fgrep") and _recursive(args):
        for arg in _searched(args):
            search(ctx, f, arg, cwd)
    if prog in _NETWORK:
        sending = prog not in ("curl", "wget", *_HTTPIE) or _method(argv) in SENDING
        sending = sending or (prog in ("curl", "wget") and _output(argv)[1])
        sending = sending or (prog in _HTTPIE and any("=" in w for w in words[1:]))
        _reaches(f, args, sending)
    if prog in _ELSEWHERE:
        _elsewhere(ctx, f, prog, args, cwd, stdin)
    _inline(f, prog, args, stdin)
    if _runs_project(prog, words):
        file_rules.runner(ctx, f, prog)
    if prog == "git" and set(plain[:1]) & _HOOKED:
        file_rules.hooks(ctx, f, f"git {plain[0]}")


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
            re.fullmatch(r"-\w*[ab]\w*", a) for a in args
        ):
            return f.add(
                "system", "Opens a program through macOS, outside melo's limits."
            )
        for word in words:
            if "://" in word and not word.startswith("file:"):
                if own_api(word, sending=False):
                    reach_own(f, word)
                continue
            opened(ctx, f, word, cwd)
    elif prog == "osascript":
        code = " ".join(args)
        given = "-e" in args
        if not given or _DRIVES.search(code):
            # A script file, or its input when it names none: melo reads neither.
            f.add("system", "Drives another app or a shell through AppleScript.")
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
    _code(ctx, f, "python" if python else "sh", text, os.path.dirname(path))


def _code(ctx: Context, f: Findings, prog: str, body: str, here: str = "") -> None:
    if _PY.match(prog):
        py_rules.review_python(ctx.deeper(here or None), f, body)
    elif prog in _SHELLS:
        review_shell(ctx.deeper(here or None), f, body)
