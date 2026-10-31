"""Local files a call writes or runs: their content goes through the same rules.

- A write of code (Python, shell, a Makefile's recipes, package.json's
  scripts) takes the categories of what it writes, so a step split in two
  (write ``conftest.py``, then run ``pytest``) is judged at the write.
- An import of a workspace module, ``python -m`` and ``runpy.run_module``
  classify that module's files (at most :data:`~.paths.MAX_READS` files a
  call, see :meth:`~.paths.Findings.take`).
- A runner that runs the project's files on its own (``pytest``, ``make``,
  ``npm test``…) classifies the code changed in this chat that melo reads;
  a changed file it cannot read (JavaScript, a config file) makes it
  undecided.

A write or an edit counts only what it adds: what the new text does that
the file's old text did not. What a file's code would do when it runs
later is never ``self``; running it is judged then.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import re
import textwrap
from collections import Counter
from collections.abc import Callable

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
    + [".pre-commit-config.yaml", "CMakeLists.txt"]
)
# Python runs the import lines of a .pth file in site-packages at every start,
# and sitecustomize/usercustomize there.
_PTH_GLOBS = [
    f"{venv}/lib/python*/site-packages/{name}"
    for venv in (".venv", "venv")
    for name in ("*.pth", "sitecustomize.py", "usercustomize.py")
]
# Files a runner's walk looks at, at most; more is a doubt.
MAX_FILES = 5000
# The largest file whose old text an edit or a write is compared with.
EDIT_BYTES = 1024 * 1024


def kind_of(path: str, text: str) -> str:
    """``python``, ``shell``, ``make``, ``package``, ``pth`` or ``""`` (not code melo reads)."""
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
    if lower.endswith(".pth"):
        return "pth"
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
    elif kind == "pth":
        lines = [line for line in text.split("\n") if line.startswith("import")]
        python.review_python(here, f, "\n".join(lines))
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


def _findings(ctx: Context, path: str, text: str) -> Findings:
    """What *text*, as the code of *path*, does: its own findings, not ``self``."""
    g = Findings()
    if text:
        review_text(ctx, g, path, text)
    g.high.pop("self", None)  # running it later is judged then
    return g


def _added(f: Findings, new: Findings, old: Findings, lead: str) -> None:
    """*new*'s findings that *old* lacks, in *f*, each reason after *lead*.

    Findings compare by category, site (the call or command, a download's
    whole pipeline with its URL) and reason, counted: a finding is new when
    its site or its reason is, so a second call like one the file had
    already, or the same call pointed at another address or package, is
    added; a call moved or a file reflowed adds nothing.
    """

    def said(reason: str) -> str:
        return f"{lead} {reason[:1].lower()}{reason[1:]}"

    had = Counter(old.hits)
    for category, site, reason in new.hits:
        if category == "self":
            continue  # running it later is judged then
        if had[(category, site, reason)] > 0:
            had[(category, site, reason)] -= 1
        elif category == "?":
            f.doubt(said(reason))
        else:
            f.add(category, said(reason))
    f.paths += [p for p in new.paths if p not in old.paths]


def _old_text(path: str) -> str | None:
    """*path*'s text now: ``""`` when there is no file, ``None`` when unreadable.

    An edit's file is read up to :data:`EDIT_BYTES`: classifying 300 KB of
    Python takes under 0.1 s, and the review runs off the event loop.
    """
    try:
        if os.path.getsize(path) > EDIT_BYTES:
            return None
        with open(path, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeDecodeError):
        return None


def written(ctx: Context, f: Findings, path: str, text: str, old: str = "") -> None:
    """A write of *text* to *path* over *old*: code takes what it adds."""
    if not kind_of(path, text):
        return
    lead = f"Writes {shown(ctx, path)}, code that"
    _added(f, _findings(ctx, path, text), _findings(ctx, path, old), lead)


def replaced(ctx: Context, f: Findings, path: str, text: str) -> None:
    """A whole-file write of *text* to *path*: what it adds to the file there."""
    old = _old_text(path)
    written(ctx, f, path, text, old or "")


def edited(ctx: Context, f: Findings, path: str, old: str, new: str) -> None:
    """An edit of *path* that puts *new* in place of *old*: what it adds.

    An edit whose *old* text is not in the file fails and changes nothing.
    A file melo cannot read (too large, not text) has *new* classified
    alone, and Python that does not parse alone is a doubt.
    """
    text = _old_text(path)
    if text is not None:
        if old and old in text:
            written(ctx, f, path, text.replace(old, new, 1), text)
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


def _changed(path: str, since: float) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    # The change time too: setting a file's time back changes it.
    return max(info.st_mtime, info.st_ctime) >= since


def changed_files(
    ctx: Context, keep: Callable[[str], bool] = _code_name
) -> tuple[list[str], bool]:
    """Code in the workspace changed since ``ctx.since``; whether the walk was cut.

    The walk skips ``.git``, dependencies and caches, but a virtual
    environment's ``.pth`` files count: Python runs them at every start.
    """
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
            path = os.path.join(top, name)
            if keep(name) and _changed(path, since):
                found.append(path)
    for pattern in _PTH_GLOBS:
        found += sorted(
            p for p in glob.glob(os.path.join(ctx.root, pattern)) if _changed(p, since)
        )
    return found, False


def _marker(path: str) -> str:
    """A changed file melo did not read, as it is now (``path:mtime:size``)."""
    try:
        info = os.lstat(path)
    except OSError:
        return f"{path}:gone"
    return f"{path}:{info.st_mtime_ns}:{info.st_size}"


def runner(ctx: Context, f: Findings, prog: str) -> None:
    """*prog* runs the project's files: what changed in this chat is read.

    A changed file of a kind melo reads is classified (its findings are
    the step's); one melo cannot read makes the step undecided, and only
    those (``Findings.changed``) tell one run of the step from the next:
    the step key is the command, so an allowance holds across edits.
    """
    if ctx.since is None or f.ran_runner:
        return
    f.ran_runner = True
    found, cut = changed_files(ctx)
    if cut:
        f.doubt(f"Runs {prog} in a folder too large for melo to check.")
    unread: list[str] = []
    for path in found:
        text = f.take(path, key=False) if kind_of(path, "") else None
        if text is None:
            unread.append(path)
            f.changed.append(_marker(path))
        elif text:
            g = _findings(ctx, path, text)
            _added(
                f,
                g,
                Findings(),
                f"Runs {shown(ctx, path)}, changed in this chat, which",
            )
    if unread:
        names = ", ".join(shown(ctx, p) for p in unread[:3])
        more = f" and {len(unread) - 3} more" if len(unread) > 3 else ""
        f.doubt(
            f"Runs {prog}, which runs files changed in this chat that melo does "
            f"not read ({names}{more})."
        )


def site_hooks(ctx: Context, f: Findings, prog: str) -> None:
    """An interpreter of the workspace's environment: its start hooks changed
    in this chat (``.pth`` files, ``sitecustomize``) are read."""
    if ctx.since is None:
        return
    for pattern in _PTH_GLOBS:
        for path in sorted(glob.glob(os.path.join(ctx.root, pattern))):
            if not _changed(path, ctx.since):
                continue
            text = f.take(path, key=False)
            if text:
                lead = f"Runs {prog}, which runs {shown(ctx, path)}, changed in this chat, which"
                _added(f, _findings(ctx, path, text), Findings(), lead)


def hooks(ctx: Context, f: Findings, prog: str) -> None:
    """A git command that runs hooks: the hooks changed in this chat are read."""
    if ctx.since is None:
        return
    since = ctx.since
    folder = os.path.join(ctx.root, ".git", "hooks")
    config = os.path.join(ctx.root, ".git", "config")
    if _changed(config, since):
        f.doubt(f"Runs {prog} after .git/config changed, which can name other hooks.")
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return
    for name in names:
        path = os.path.join(folder, name)
        if name.endswith(".sample") or not _changed(path, since):
            continue
        text = f.take(path, key=False)
        if text:
            g = _findings(ctx, path, text) if kind_of(path, text) else Findings()
            lead = f"Runs the git hook {name}, changed in this chat, which"
            _added(f, g, Findings(), lead)
        f.doubt(f"Runs the git hook {name}, changed in this chat.")
