"""Local files a call writes or runs: their content goes through the same rules.

- A write of code (Python, shell, a Makefile's recipes, package.json's
  scripts) takes the categories of what it writes, so a step split in two
  (write ``conftest.py``, then run ``pytest``) is judged at the write.
- An import of a workspace module, ``python -m`` and ``runpy.run_module``
  classify that module's files (at most :data:`~.paths.MAX_READS` files a
  call, see :meth:`~.paths.Findings.take`).
- A runner that runs the project's files on its own (``pytest``, ``make``,
  ``npm test``…) is undecided when code in the workspace changed in this
  chat, and those files' content is classified too.
"""

from __future__ import annotations

import ast
import json
import os
import re
import textwrap

from localmelo.melo.checker.review.paths import Context, Findings, shown

# Folders whose files a runner does not count as the project's code.
_SKIP_DIRS = frozenset(
    [".git", "node_modules", ".venv", "venv", "__pycache__", ".tox", ".nox"]
    + [".mypy_cache", ".pytest_cache", ".ruff_cache", "build", "dist", "target"]
    + [".next", ".cache", "site-packages", ".idea", ".gradle", ".eggs"]
)
# Files a runner may run: by suffix, and by name.
_CODE_SUFFIXES = (
    *[".py", ".pyw", ".sh", ".bash", ".zsh", ".js", ".mjs", ".cjs", ".ts", ".tsx"],
    *[".jsx", ".rb", ".pl", ".php", ".go", ".rs", ".swift", ".java", ".kt", ".lua"],
    *[".mk", ".toml", ".cfg", ".ini", ".gradle", ".kts"],
)
_CODE_NAMES = frozenset(
    ["Makefile", "makefile", "GNUmakefile", "justfile", "Justfile", "package.json"]
    + ["Rakefile", "Gemfile", "pom.xml", "build.rs", "go.mod", "Package.swift"]
)
# Files a runner's walk looks at, at most; more is a doubt.
MAX_FILES = 5000


def kind_of(path: str, text: str) -> str:
    """``python``, ``shell``, ``make``, ``package`` or ``""`` (not code melo reads)."""
    name = os.path.basename(path)
    lower = name.lower()
    if lower.endswith((".py", ".pyw")):
        return "python"
    if lower.endswith((".sh", ".bash", ".zsh", ".command")):
        return "shell"
    if name in ("Makefile", "makefile", "GNUmakefile") or lower.endswith(".mk"):
        return "make"
    if name == "package.json":
        return "package"
    first = text.split("\n", 1)[0]
    if first.startswith("#!"):
        if "python" in first:
            return "python"
        if re.search(r"\b(sh|bash|zsh|dash)\b", first):
            return "shell"
    return ""


def review_text(ctx: Context, f: Findings, path: str, text: str) -> None:
    """Classify the code *text* of the file *path* with the rules of its kind."""
    from localmelo.melo.checker.review import python, shell

    here = ctx.deeper(here=os.path.dirname(path))
    kind = kind_of(path, text)
    if kind == "python":
        python.review_python(here, f, text)
    elif kind == "shell":
        shell.review_shell(here, f, text)
    elif kind == "make":
        recipes = [
            line[1:].lstrip("@+- ")
            for line in text.split("\n")
            if line.startswith("\t")
        ]
        shell.review_shell(here, f, "\n".join(recipes))
    elif kind == "package":
        try:
            scripts = json.loads(text).get("scripts") or {}
            commands = [str(v) for v in dict(scripts).values()]
        except (ValueError, AttributeError, TypeError):
            return f.doubt(
                f"Holds {os.path.basename(path)}, which melo could not read."
            )
        shell.review_shell(here, f, "\n".join(commands))


def _merged(f: Findings, g: Findings, lead: str) -> None:
    """*g*'s findings in *f*, each reason after *lead* ("Writes x.py, code that")."""

    def said(reason: str) -> str:
        return f"{lead} {reason[:1].lower()}{reason[1:]}"

    for category, reasons in g.high.items():
        for reason in reasons:
            f.add(category, said(reason))
    for reason in g.unsure:
        f.doubt(said(reason))
    f.paths += g.paths
    f.read.update(g.read)


def written(ctx: Context, f: Findings, path: str, text: str) -> None:
    """A write of *text* to *path*: code takes the categories of what it does."""
    if not kind_of(path, text):
        return
    g = Findings(read=dict(f.read))
    review_text(ctx, g, path, text)
    _merged(f, g, f"Writes {shown(ctx, path)}, code that")


def edited(ctx: Context, f: Findings, path: str, old: str, new: str) -> None:
    """An edit of *path* that puts *new* in place of *old*: the file it makes.

    An edit whose *old* text is not in the file fails and changes nothing.
    A file melo cannot read (too large, not text) has *new* classified
    alone, and Python that does not parse alone is a doubt.
    """
    try:
        if os.path.getsize(path) > 64 * 1024:
            raise OSError
        with open(path, encoding="utf-8") as file:
            text: str | None = file.read()
    except (OSError, UnicodeDecodeError):
        text = None
    if text is not None:
        if old in text:
            written(ctx, f, path, text.replace(old, new, 1))
        return
    if kind_of(path, new) == "python":
        try:
            ast.parse(textwrap.dedent(new))
        except (SyntaxError, ValueError):
            return f.doubt(f"Edits {shown(ctx, path)} in a way melo could not read.")
    written(ctx, f, path, textwrap.dedent(new))


def module(ctx: Context, f: Findings, name: str, *, main: bool = False) -> None:
    """Classify the workspace files that importing (or running) *name* runs.

    ``a.b`` runs ``a/__init__.py`` and ``a/b.py`` (or ``a/b/__init__.py``);
    with *main*, ``python -m a`` also runs ``a/__main__.py``. A name that
    is no file in the workspace (the standard library, an installed
    package) is not followed.
    """
    parts = [p for p in name.split(".") if p]
    for base in dict.fromkeys([ctx.here or ctx.root, ctx.root]):
        found = _module_files(base, parts, main)
        if found:
            for path in found:
                _run_file(ctx, f, os.path.realpath(path))
            return


def _module_files(base: str, parts: list[str], main: bool) -> list[str]:
    """The files under *base* that importing ``".".join(parts)`` runs."""
    found: list[str] = []
    folder = base
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        package = os.path.join(folder, part)
        dunder_main = os.path.join(package, "__main__.py")
        if os.path.isfile(os.path.join(package, "__init__.py")):
            found.append(os.path.join(package, "__init__.py"))
        elif os.path.isfile(package + ".py"):
            found.append(package + ".py")  # the rest names what is in it
            break
        elif not os.path.isdir(package):
            break  # not in the workspace (the standard library, a package)
        if last and main and os.path.isfile(dunder_main):
            found.append(dunder_main)
        folder = package
    return found


def _run_file(ctx: Context, f: Findings, path: str) -> None:
    text = f.take(path)
    if text:
        review_text(ctx, f, path, text)


def _code_name(name: str) -> bool:
    return name in _CODE_NAMES or name.lower().endswith(_CODE_SUFFIXES)


def changed_files(ctx: Context) -> tuple[list[str], bool]:
    """Code in the workspace changed since ``ctx.since``; whether the walk was cut."""
    since, count = ctx.since or 0.0, 0
    found: list[str] = []
    for top, dirs, files in os.walk(ctx.root):
        dirs[:] = sorted(
            d for d in dirs if d not in _SKIP_DIRS and not d.endswith(".egg-info")
        )
        for name in sorted(files):
            count += 1
            if count > MAX_FILES:
                return found, True
            if not _code_name(name):
                continue
            path = os.path.join(top, name)
            try:
                info = os.lstat(path)
            except OSError:
                continue
            # The change time too: setting a file's time back changes it.
            if max(info.st_mtime, info.st_ctime) >= since:
                found.append(path)
    return found, False


def runner(ctx: Context, f: Findings, prog: str) -> None:
    """*prog* runs the project's files: undecided when any changed in this chat."""
    if ctx.since is None:
        return
    found, cut = changed_files(ctx)
    if cut:
        f.doubt(f"Runs {prog} in a folder too large for melo to check.")
    if not found:
        return
    names = ", ".join(shown(ctx, p) for p in found[:3])
    more = f" and {len(found) - 3} more" if len(found) > 3 else ""
    f.doubt(f"Runs {prog}, which runs files changed in this chat ({names}{more}).")
    for path in found:
        text = f.take(path) if kind_of(path, "") else ""
        if not text:
            continue
        g = Findings(read=dict(f.read))
        review_text(ctx, g, path, text)
        _merged(f, g, f"Runs {shown(ctx, path)}, changed in this chat, which")
