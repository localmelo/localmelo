"""Python rules: walk the code's syntax tree and classify what its calls do."""

from __future__ import annotations

import ast
import os
import shlex
from itertools import product
from typing import Any

from localmelo.melo.checker.review import shell
from localmelo.melo.checker.review.paths import (
    Context,
    Findings,
    resolve,
    shown,
    touch,
    under,
)

_RUNS = set(
    [
        "run",
        "call",
        "check_call",
        "check_output",
        "Popen",
        "getoutput",
        "getstatusoutput",
    ]
)
_SHELL_RUNS = set(
    ["system", "popen", "getoutput", "getstatusoutput", "create_subprocess_shell"]
)
_SPAWNS = ("os.exec", "os.spawn", "os.posix_spawn", "asyncio.create_subprocess")
_PATH_OPS = {
    **dict.fromkeys(
        ["write_text", "write_bytes", "touch", "symlink_to", "hardlink_to"], "write"
    ),
    **dict.fromkeys(["unlink", "rmdir"], "delete"),
    **dict.fromkeys(["read_text", "read_bytes"], "read"),
}
_OS_OPS = {
    **dict.fromkeys(
        ["os.remove", "os.unlink", "os.rmdir", "os.removedirs", "shutil.rmtree"],
        "delete",
    ),
    **dict.fromkeys(
        ["os.makedirs", "os.mkdir", "os.chmod", "os.chown", "os.truncate"], "write"
    ),
}
_MOVES = set(["os.rename", "os.replace", "shutil.move"])
_COPIES = set(
    ["shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree", "os.symlink"]
)
_SAVES = set(
    [
        "savefig",
        "to_csv",
        "to_excel",
        "to_parquet",
        "to_json",
        "to_pickle",
        "to_html",
        "save",
    ]
)
_SERVERS = set(
    ["HTTPServer", "ThreadingHTTPServer", "TCPServer", "make_server", "bind"]
)
_DYNAMIC = set(
    ["exec", "eval", "compile", "runpy.run_path", "runpy.run_module", "__import__"]
)
_PASS = set(
    ["str", "os.fspath", "os.path.abspath", "os.path.realpath", "os.path.normpath"]
)


# Commands a local helper may run while the call stays routine: they only
# read (#324). A word or flag of _WRITE_WORDS makes one of them write.
_READ_ONLY = set(
    ["ls", "cat", "head", "tail", "wc", "grep", "egrep", "rg", "pwd", "file"]
    + ["stat", "du", "tree", "which", "find", "echo", "diff"]
)
_GIT_READS = set(
    ["status", "log", "diff", "show", "branch", "rev-parse", "ls-files", "blame"]
    + ["describe", "shortlog", "grep", "ls-tree", "remote"]
)
_WRITE_FLAGS = ("--output", "-delete", "-exec", "-execdir", "-ok", "-fprint", "-fls")
_HELPER_KEYWORDS = set(
    ["capture_output", "text", "check", "cwd", "timeout", "stdout", "stderr"]
    + ["encoding", "errors", "universal_newlines"]
)


def read_only_argv(argv: tuple[str, ...]) -> bool:
    """Whether the literal command *argv* only reads (git status, ls, grep…)."""
    if not argv or any(a.startswith(_WRITE_FLAGS) for a in argv):
        return False
    prog, args = os.path.basename(argv[0]), list(argv[1:])
    if prog != "git":
        return prog in _READ_ONLY
    while args[:1] in (["-C"], ["--no-pager"]):
        args = args[2:] if args[0] == "-C" else args[1:]
    if not args or args[0] not in _GIT_READS:
        return False
    words = [a for a in args[1:] if not a.startswith("-")]
    # `git branch NAME` makes a branch and `git remote add` adds one.
    return not (args[0] in ("branch", "remote") and words)


# The nodes the first pass looks at; the rest are skipped fast.
_BINDS = {ast.Call, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign}
_BINDS |= {ast.NamedExpr, ast.For, ast.AsyncFor, ast.comprehension, ast.withitem}


def _helper(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.Call | None:
    """The process call of a local helper that passes its arguments on as argv.

    ``def run(*a): subprocess.run(list(a), …)`` (also ``a``, ``[*a]`` or
    ``tuple(a)``), or ``def run(cmd): subprocess.run(cmd, …)``, with nothing
    but output and error keywords: no shell, no new session. ``None`` when
    *node* is not such a helper.
    """
    args = node.args
    name = args.vararg.arg if args.vararg else None
    if name is None and len(args.args) - len(args.defaults) == 1:
        name = args.args[0].arg
    found = None
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        func = inner.func
        dotted = func.attr if isinstance(func, ast.Attribute) else ""
        module = getattr(func, "value", None)
        if (
            not isinstance(module, ast.Name)
            or module.id != "subprocess"
            or dotted not in _RUNS - {"getoutput", "getstatusoutput"}
        ):
            continue
        first = inner.args[0] if inner.args else None
        if (
            isinstance(first, ast.Call)
            and isinstance(first.func, ast.Name)
            and first.func.id in ("list", "tuple")
            and len(first.args) == 1
            and not first.keywords
        ):
            first = first.args[0]
        elif (
            isinstance(first, ast.List)
            and len(first.elts) == 1
            and isinstance(first.elts[0], ast.Starred)
        ):
            first = first.elts[0].value
        if (
            found is not None
            or name is None
            or not isinstance(first, ast.Name)
            or first.id != name
            or len(inner.args) != 1
            or any(k.arg not in _HELPER_KEYWORDS for k in inner.keywords)
        ):
            return None
        found = inner
    return found


class _Review:
    def __init__(self, ctx: Context, f: Findings, tree: ast.AST) -> None:
        self.ctx, self.f, self.busy = ctx, f, set[str]()
        self.memo: dict[str | None, list[Any]] = {}
        self.alias: dict[str, str] = {}
        self.bound: dict[str, list[ast.expr]] = {}
        self.later: list[ast.AST] = []  # the calls and imports, once all is bound
        # Local helpers that run their arguments as a command (see _helper),
        # by name: the calls of one are reviewed at each call site instead.
        self.helpers: dict[str, ast.Call] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                inner = _helper(node)
                if inner is not None and node.name not in self.helpers:
                    self.helpers[node.name] = inner
                elif node.name in self.helpers:  # defined twice: not followed
                    self.helpers.pop(node.name)
        self.forwarded = {id(inner) for inner in self.helpers.values()}
        loads = [
            n.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        ]
        called = [
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        for name in list(self.helpers):
            if loads.count(name) != called.count(name):
                # Passed around (map(run, …)): its commands cannot be read.
                self.forwarded.discard(id(self.helpers.pop(name)))
        for node in ast.walk(tree):
            if type(node) not in _BINDS:
                continue
            if isinstance(node, ast.Call | ast.Import | ast.ImportFrom):
                self.later.append(node)
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.alias[a.asname or a.name.split(".")[0]] = (
                        a.name if a.asname else a.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    self.alias[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr):
                targets = getattr(node, "targets", [getattr(node, "target", None)])
                for t in targets:
                    if isinstance(t, ast.Name) and node.value is not None:
                        self.bound.setdefault(t.id, []).append(node.value)
            elif isinstance(node, ast.For | ast.comprehension):
                items = getattr(node.iter, "elts", [ast.Constant(None)])
                if isinstance(node.target, ast.Name):
                    self.bound.setdefault(node.target.id, []).extend(items)
            elif isinstance(node, ast.withitem) and isinstance(
                node.optional_vars, ast.Name
            ):
                self.bound.setdefault(node.optional_vars.id, []).append(
                    node.context_expr
                )

    def name(self, node: ast.AST) -> str:
        """The dotted name of a function or attribute, imports resolved."""
        if isinstance(node, ast.Name):
            return self.alias.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{self.name(node.value)}.{node.attr}"
        return "?"

    def fold(self, node: ast.AST | None) -> list[Any]:
        """The values *node* may have (str, tuple or None), at most eight."""
        name = node.id if isinstance(node, ast.Name) else None
        if name in self.memo:
            return self.memo[name]
        values = [None] if node is None else self._fold(node)
        out = list(dict.fromkeys(values))[:8] or [None]
        if name is not None and not self.busy:
            self.memo[name] = out
        return out

    def _fold(self, node: ast.AST) -> list[Any]:
        ctx, name = self.ctx, self.name(node)
        if isinstance(node, ast.Constant):
            return [None if node.value is None else str(node.value)]
        if (
            isinstance(node, ast.Name)
            and node.id in self.bound
            and node.id not in self.busy
        ):
            self.busy.add(node.id)
            try:
                return [v for n in self.bound[node.id] for v in self.fold(n)]
            finally:
                self.busy.discard(node.id)
        if isinstance(node, ast.List | ast.Tuple):
            return list(product(*(self.fold(e) for e in node.elts)))[:8]
        if (
            isinstance(node, ast.JoinedStr)
            or isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Add | ast.Div)
        ):
            parts = (
                node.values
                if isinstance(node, ast.JoinedStr)
                else [node.left, node.right]
            )
            sep = "/" if isinstance(getattr(node, "op", None), ast.Div) else ""
            folded = [
                self.fold(
                    getattr(p, "value", p) if isinstance(p, ast.FormattedValue) else p
                )
                for p in parts
            ]
            return [_join(sep, combo) for combo in product(*folded)]
        if isinstance(node, ast.IfExp):
            return self.fold(node.body) + self.fold(node.orelse)
        if isinstance(node, ast.Subscript) and self.name(node.value) == "os.environ":
            return [ctx.home] if self.fold(node.slice) == ["HOME"] else [None]
        if name in ("sys.executable", "sys.prefix"):
            return [ctx.executable if name == "sys.executable" else ctx.prefix]
        if not isinstance(node, ast.Call):
            return [None]
        name, args = self.name(node.func), [self.fold(a) for a in node.args]
        attr, base = name.rsplit(".", 1)[-1], getattr(node.func, "value", None)
        if name in ("os.getenv", "os.environ.get"):
            return [ctx.home] if args[:1] == [["HOME"]] else [None]
        if name == "os.path.expanduser" or attr in ("home", "expanduser", "resolve"):
            source = args[0] if args else ["~"] if attr == "home" else self.fold(base)
            return [ctx.home + v[1:] if str(v)[:1] == "~" else v for v in source]
        if name in ("os.getcwd", "pathlib.Path.cwd") or name.startswith("tempfile."):
            return [ctx.root if "cwd" in name else "/tmp"]
        if name in ("os.path.join", "pathlib.Path") or attr == "joinpath":
            folded = [self.fold(base)] if attr == "joinpath" else []
            combos = product(*folded, *args) if folded or args else [(ctx.root,)]
            return [_join("/", c) for c in combos]
        if (name in _PASS or name == "shutil.which") and args:
            return args[0]
        if attr == "absolute":
            return self.fold(base)
        if attr == "split" and not args or name == "shlex.split":
            source = args[0] if args else self.fold(base)
            return [
                tuple(shlex.split(v)) if isinstance(v, str) else None for v in source
            ]
        return [None]

    def run(self) -> None:
        for node in self.later:
            if isinstance(node, ast.Call):
                self.call(node)
            elif isinstance(node, ast.Import | ast.ImportFrom):
                self.imported(node)

    def imported(self, node: ast.Import | ast.ImportFrom) -> None:
        names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
        if any(n.split(".")[0] in ("browser_cookie3", "pycookiecheat") for n in names):
            self.f.add("credentials", "Reads your browser's cookies.")
        if any(n.startswith("pip._internal") for n in names):
            self.f.add("install", "Runs pip's installer from Python.")

    def call(self, node: ast.Call) -> None:
        if id(node) in self.forwarded:
            return  # a helper's own call: its call sites are reviewed
        if isinstance(node.func, ast.Name) and node.func.id in self.helpers:
            return self.helper_call(node, self.helpers[node.func.id])
        ctx, f, name = self.ctx, self.f, self.name(node.func)
        attr, base = name.rsplit(".", 1)[-1], getattr(node.func, "value", None)
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        first = node.args[0] if node.args else None
        module = name.split(".")[0]
        method = base is not None and self.fold(base)[0] is not None
        if (
            module == "subprocess"
            and attr in _RUNS
            or name.startswith(_SPAWNS)
            or name
            in (
                "os.system",
                "os.popen",
                "pty.spawn",
            )
        ):
            self.process(node, name, attr, kw)
        elif attr == "open" and module != "os":
            at = 0 if method else 1
            mode = self.fold(node.args[at] if len(node.args) > at else kw.get("mode"))
            write = any(set(str(m)) & set("wax+") for m in mode if m)
            self.paths(base if method else first, write, "write" if write else "read")
        elif attr in _PATH_OPS and base is not None:
            self.paths(base, _PATH_OPS[attr] != "read", _PATH_OPS[attr])
        elif attr in ("rename", "replace") and len(node.args) == 1 and method:
            self.paths(base, False, "delete")
            self.paths(first, False, "write")
        elif name in _MOVES:
            self.paths(first, True, "delete")
            self.paths(node.args[-1] if node.args else None, True, "write")
        elif name in _OS_OPS:
            self.paths(first, "mkdir" not in name, _OS_OPS[name])
        elif name in _COPIES and len(node.args) > 1:
            self.paths(first, False, "read")
            self.paths(node.args[1], True, "write")
        elif attr in _SAVES and first or "user_data_dir" in kw:
            self.paths(kw.get("user_data_dir", first), False, "write")
        elif attr == "launch_persistent_context" and first:
            self.paths(first, False, "write")
        elif name.endswith("urlretrieve") and first:
            target = node.args[1] if len(node.args) > 1 else kw.get("filename")
            for url, path in product(self.fold(first), self.fold(target)):
                shell.download(ctx, f, str(url or ""), str(path or "/tmp/x"), ctx.root)
        elif name.endswith(("snapshot_download", "hf_hub_download")):
            f.add(
                "download", f"Downloads a model into its cache, outside {ctx.folder}."
            )
        elif module in ("smtplib", "ftplib", "paramiko"):
            f.add("send_out", f"Sends data to another computer ({module}).")
        elif (
            attr in ("post", "put", "patch")
            and (
                module in ("requests", "httpx", "aiohttp")
                or str(self.fold(first)[0]).startswith("http")
            )
            or "data" in kw
            and "url" in name.lower()
        ):
            self.send(node)
        elif module == "keyring":
            f.add("credentials", "Reads or changes a password in your keychain.")
        elif attr in _SERVERS or "host" in kw:
            host = kw.get("host") or getattr(first, "elts", [first])[0]
            if any(h in ("", "0.0.0.0", "::") for h in self.fold(host)):
                f.add("service", "Opens a network service other computers can reach.")
        elif (name in _DYNAMIC or name == "importlib.import_module") and first:
            self.dynamic(name, first)
        elif name in ("venv.create", "venv.EnvBuilder"):
            f.add("env", "Creates a Python environment.")
        elif name in ("ensurepip.bootstrap", "pip.main"):
            f.add("install", "Installs packages with pip from inside Python.")
        elif name in ("os.fork", "os.setsid", "os.daemon"):
            f.add("persistence", "Starts a process that keeps running after this call.")
        elif name in ("print", "json.dumps", "json.dump") and any(
            self.name(a) == "os.environ" for a in node.args
        ):
            f.add(
                "credentials", "Prints every environment variable, which can hold keys."
            )
        if attr in ("get", "urlopen", "stream", "head", "Request") and first:
            for url in self.fold(first):
                if "://" in str(url) and shell.LARGE.search(str(url).split("?")[0]):
                    shell.download(ctx, f, url, url.rsplit("/", 1)[-1], ctx.root)

    def helper_call(self, node: ast.Call, inner: ast.Call) -> None:
        """One call of a local helper: routine only for literal read-only commands."""
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            return self.f.doubt("Runs a command melo could not read.")
        varargs = not isinstance(inner.args[0], ast.Name) or len(node.args) != 1
        if varargs:
            argvs: list[Any] = list(product(*(self.fold(a) for a in node.args)))[:8]
        else:
            argvs = [
                tuple(v) if isinstance(v, list) else v for v in self.fold(node.args[0])
            ]
        for argv in argvs:
            if not isinstance(argv, tuple) or not all(isinstance(a, str) for a in argv):
                self.f.doubt("Runs a command melo could not read.")
                continue
            if not read_only_argv(argv):
                self.f.doubt(
                    f"Runs {os.path.basename(argv[0]) if argv else 'a command'} "
                    "through a helper, which melo allows unasked only for reads."
                )
            if argv:
                shell.run_argv(self.ctx.deeper(), self.f, list(argv), self.ctx.root)

    def paths(self, node: ast.AST | None, doubt: bool, op: str) -> None:
        values = [v for v in self.fold(node) if isinstance(v, str)]
        for value in values:
            touch(self.ctx, self.f, value, op)
        if not values and doubt:
            self.f.doubt(
                f"{'Deletes' if op == 'delete' else 'Writes'} a file melo "
                "could not name."
            )

    def process(self, node: ast.Call, name: str, attr: str, kw: dict[str, Any]) -> None:
        ctx, f = self.ctx, self.f
        if self.fold(kw.get("start_new_session")) == ["True"] or "preexec_fn" in kw:
            f.add("persistence", "Starts a process that keeps running after this call.")
        in_shell = attr in _SHELL_RUNS or self.fold(kw.get("shell")) == ["True"]
        if attr == "create_subprocess_exec" or name.startswith(
            ("os.execl", "os.spawnl")
        ):
            argvs: list[Any] = list(product(*(self.fold(a) for a in node.args)))[:8]
        else:
            many = (
                name.startswith(("os.exec", "os.spawn", "os.posix")) and node.args[1:]
            )
            first = node.args[0] if node.args else kw.get("args")
            argvs = self.fold(node.args[1] if many else first)
        for argv in argvs:
            parts = [argv] if isinstance(argv, str) else list(argv or [])
            if isinstance(argv, str) and in_shell:
                shell.review_shell(ctx.deeper(), f, argv)
            elif not in_shell and any(isinstance(p, str) for p in parts):
                program = parts[0] if isinstance(parts[0], str) else "?"
                rest = [p if isinstance(p, str) else "$value" for p in parts[1:]]
                shell.run_argv(ctx.deeper(), f, [program, *rest], ctx.root)
            else:
                f.doubt("Runs a command melo could not read.")

    def send(self, node: ast.Call) -> None:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and self.name(inner.func) in (
                "open",
                "io.open",
            ):
                for value in self.fold(inner.args[0] if inner.args else None):
                    path = resolve(self.ctx, value) if isinstance(value, str) else None
                    if path and not under(path, self.ctx.root):
                        name = shown(self.ctx, path)
                        return self.f.add(
                            "send_out", f"Sends {name} to another computer."
                        )
        self.f.doubt("Sends data to a server.")

    def dynamic(self, name: str, first: ast.expr) -> None:
        call = getattr(first, "func", None)
        opened = (
            getattr(call, "value", None)
            if getattr(call, "attr", "") == "read"
            else None
        )
        if isinstance(opened, ast.Call) and self.name(opened.func) == "open":
            for path in self.fold(opened.args[0] if opened.args else None):
                shell.script(self.ctx, self.f, str(path or ""), "python", self.ctx.root)
            return
        values = self.fold(first)
        if not all(isinstance(v, str) for v in values):
            self.f.add(
                "opaque", f"Runs code built at run time ({name.split('.')[-1]})."
            )
        elif name in ("exec", "eval", "compile"):
            for code in values:
                review_python(self.ctx.deeper(), self.f, code)
        elif name == "runpy.run_path":
            for path in values:
                shell.script(self.ctx, self.f, path, "python", self.ctx.root)
        elif any(v.split(".")[0] == "pip" for v in values):
            self.f.add("install", "Runs pip's installer from Python.")


def _join(sep: str, parts: tuple[Any, ...]) -> Any:
    if all(isinstance(p, str) for p in parts):
        return sep.join(parts) if sep != "/" else os.path.join(*parts)
    if all(isinstance(p, tuple) for p in parts):
        return tuple(x for p in parts for x in p)
    return None


def review_python(ctx: Context, f: Findings, code: str) -> None:
    if ctx.depth > shell.MAX_DEPTH:
        return f.doubt("Nests code too deeply for melo to check.")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return  # it fails without effect
    _Review(ctx, f, tree).run()
