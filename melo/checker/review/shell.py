"""Shell rules: split a command into simple commands and classify each one."""

from __future__ import annotations

import os
import re
import shlex
from urllib.parse import urlparse

from localmelo.melo.checker.review import python as py_rules
from localmelo.melo.checker.review.paths import (
    Context,
    Findings,
    resolve,
    shown,
    touch,
    where,
)

MAX_DEPTH = 3
SCRIPT_BYTES = 64 * 1024
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
    ]
)
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
    text, bodies = _heredocs(_uncomment(command))
    try:
        segments = _segments(text)
    except ValueError:
        return f.doubt("Has a quote that does not close.")
    cwd, before, url, bare_before = ctx.root, "", "", False
    for op, argv, redirs in segments:
        body = None
        for kind, target in redirs:
            if kind in _WRITES and not target.isdigit():
                touch(ctx, f, target, "write", cwd)
            elif kind == "<":
                touch(ctx, f, target, "read", cwd)
            elif kind == "<<" and bodies:
                body = bodies.pop(0)
        for token in argv:
            for inner in re.findall(r"\$\((.*)\)|`([^`]*)`", token, re.S):
                review_shell(ctx.deeper(), f, inner[0] or inner[1])
        argv = _unwrap(f, argv)
        if not argv:
            continue
        prog = os.path.basename(argv[0])
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
            f.add("remote_code", f"Runs code downloaded from {host}.")
            if "get-pip" in url + " ".join(links):
                f.add("install", "Installs pip with get-pip.py.")
        elif bare and op in ("|", "|&") and before == "decode":
            f.add("opaque", "Runs encoded text as code.")
        elif bare and op in ("|", "|&"):
            f.doubt(f"Runs code that another command prints, with {prog}.")
        run_argv(ctx, f, argv, cwd)
        decode = prog in ("base64", "xxd", "openssl") and bool(
            {"-d", "-D", "--decode", "-r"} & set(argv)
        )
        fetch = prog in ("curl", "wget") and not _output(argv)[0]
        before = "fetch" if fetch else "decode" if decode else ""
        url, bare_before = (links + [""])[0], bare


def _unwrap(f: Findings, argv: list[str]) -> list[str]:
    """*argv* without ``VAR=x`` prefixes and wrappers such as ``nice`` or ``uv run``."""
    while argv:
        prog = os.path.basename(argv[0])
        if re.match(r"[A-Za-z_]\w*=", argv[0]):
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
        while i < len(rest) and (
            rest[i][:1] == "-" or "=" in rest[i] or "timeout" in prog and i == 0
        ):
            i += 2 if rest[i] in _VALUED else 1
        # xargs adds arguments melo cannot see.
        argv = rest[i:] + (["$value"] if prog == "xargs" and rest[i:] else [])
    return argv


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


def run_argv(ctx: Context, f: Findings, argv: list[str], cwd: str) -> None:
    """Classify one simple command, from a shell or a Python process call."""
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
        if sends:
            f.doubt(f"Sends data to {host.netloc or 'a server'}.")
        if target:
            download(ctx, f, host.geturl(), target, cwd)
    elif prog in ("rm", "rmdir", "unlink", "trash"):
        for arg in words:
            touch(ctx, f, arg, "delete", cwd)
    elif prog == "find":
        end = next((i for i, a in enumerate(args) if a[:1] in "-(!"), len(args))
        roots = args[:end] or ["."]
        if "-delete" in args:
            for root in roots:
                touch(ctx, f, root, "delete", cwd)
        for flag in {"-exec", "-execdir", "-ok"} & set(args):
            rest = args[args.index(flag) + 1 :]
            rest = rest[: next((i for i, a in enumerate(rest) if a in ";+"), len(rest))]
            if rest:
                run_argv(ctx, f, [roots[0] if a == "{}" else a for a in rest], cwd)
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
    """A Python interpreter run with *argv*; ``"?"`` names an unknown program."""
    interp, i = argv[0], 1
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
    try:
        if not path or os.path.getsize(path) > SCRIPT_BYTES or ctx.depth > 1:
            raise OSError
        with open(path, encoding="utf-8") as file:
            text = file.read()
    except (OSError, UnicodeDecodeError):
        return f.doubt(f"Runs {raw}, which melo could not read.")
    shebang = text.split("\n", 1)[0]
    python = (
        kind == "python" or not kind and (raw.endswith(".py") or "python" in shebang)
    )
    _code(ctx, f, "python" if python else "sh", text)


def _code(ctx: Context, f: Findings, prog: str, body: str) -> None:
    if _PY.match(prog):
        py_rules.review_python(ctx.deeper(), f, body)
    elif prog in _SHELLS:
        review_shell(ctx.deeper(), f, body)
