"""The core, ``melo/``, imports nothing from ``support/`` — no network.

The rule is checked on the AST of every module under ``melo/``, so text
that only names ``localmelo.support``, such as a docstring, does not
count, and an import of any form does: plain, relative or through
``importlib.import_module``. Two imports predate the rule and are allowed,
matched by module, name and the scope they sit in rather than by line:
``Config``, imported only for type checking, and ``get_backend``, imported
inside ``_providers_from_config`` when an Agent is built from a Config.

The same walk checks a table of other rules (:data:`FORBIDDEN`): the
terminal UI's libraries stay out of the core, the sessions and the model
listing, and the sessions and the UI stay clear of the gateway.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# The directory that holds pyproject.toml is the localmelo package.
PACKAGE_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = PACKAGE_DIR / "melo"
SUPPORT = "localmelo.support"

# One import of support: the file, the module imported from, the name
# imported ("" for a whole module) and the enclosing scopes, outermost
# first.
Found = tuple[str, str, str, tuple[str, ...]]

ALLOWED: list[Found] = [
    (
        "melo/agent/agent.py",
        "localmelo.support.backends",
        "get_backend",
        ("def _providers_from_config",),
    ),
    (
        "melo/agent/agent.py",
        "localmelo.support.config",
        "Config",
        ("if TYPE_CHECKING",),
    ),
]


def _is_under(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes
    )


def _is_support(module: str) -> bool:
    return _is_under(module, (SUPPORT,))


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


class _ImportsOf(ast.NodeVisitor):
    """Collects the imports of the modules under *prefixes* in one module's AST.

    *package* is the package that the module's relative imports start
    from. Each import found is ``(module, name, scopes)``.
    """

    def __init__(self, package: str, prefixes: tuple[str, ...] = (SUPPORT,)) -> None:
        self.package = package
        self.prefixes = prefixes
        self.scopes: list[str] = []
        self.found: list[tuple[str, str, tuple[str, ...]]] = []

    def _add(self, module: str, name: str) -> None:
        self.found.append((module, name, tuple(self.scopes)))

    def _visit_in(self, scope: str, nodes: list[ast.stmt]) -> None:
        self.scopes.append(scope)
        for node in nodes:
            self.visit(node)
        self.scopes.pop()

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        if _is_type_checking(node.test):
            self._visit_in("if TYPE_CHECKING", node.body)
        else:
            for child in node.body:
                self.visit(child)
        for child in node.orelse:
            self.visit(child)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_in(f"def {node.name}", node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_in(f"def {node.name}", node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_in(f"class {node.name}", node.body)

    def _matches(self, module: str) -> bool:
        return _is_under(module, self.prefixes)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if self._matches(alias.name):
                self._add(alias.name, "")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = self._absolute(node)
        for alias in node.names:
            if self._matches(module) or self._matches(f"{module}.{alias.name}"):
                self._add(module, alias.name)

    def visit_Call(self, node: ast.Call) -> None:
        # importlib.import_module("localmelo.support...") and __import__(...)
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute) else ""
        )
        if name in ("import_module", "__import__") and node.args:
            first = node.args[0]
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and self._matches(first.value)
            ):
                self._add(first.value, "")
        self.generic_visit(node)

    def _absolute(self, node: ast.ImportFrom) -> str:
        """The module an ``from ... import`` reads, with a relative one resolved."""
        if not node.level:
            return node.module or ""
        parts = self.package.split(".")
        base = parts[: len(parts) - (node.level - 1)]
        return ".".join([*base, node.module] if node.module else base)


def imports_of(
    source: str,
    module: str,
    prefixes: tuple[str, ...],
    *,
    is_package: bool = False,
) -> list[tuple[str, str, tuple[str, ...]]]:
    """The imports of the modules under *prefixes* in *source*, the code of *module*."""
    package = module if is_package else module.rpartition(".")[0]
    visitor = _ImportsOf(package, prefixes)
    visitor.visit(ast.parse(source))
    return visitor.found


def support_imports(
    source: str, module: str, *, is_package: bool = False
) -> list[tuple[str, str, tuple[str, ...]]]:
    """The imports of localmelo.support in *source*, the code of *module*."""
    return imports_of(source, module, (SUPPORT,), is_package=is_package)


def _imports_under(path: Path, prefixes: tuple[str, ...]) -> list[Found]:
    """The imports of *prefixes* in every module at *path*, a file or a directory."""
    found: list[Found] = []
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    for file in files:
        relative = file.relative_to(PACKAGE_DIR)
        parts = list(relative.with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        module = ".".join(["localmelo", *parts])
        source = file.read_text(encoding="utf-8")
        found.extend(
            (relative.as_posix(), imported, name, scopes)
            for imported, name, scopes in imports_of(
                source, module, prefixes, is_package=is_package
            )
        )
    return found


def _core_imports() -> list[Found]:
    return _imports_under(CORE_DIR, (SUPPORT,))


def test_the_core_imports_from_support_only_what_predates_the_rule() -> None:
    # Exactly these two: a third fails the rule, and a missing one means
    # this list is out of date (or the walk found nothing).
    assert sorted(_core_imports()) == sorted(ALLOWED)


MODULE = "localmelo.melo.agent.agent"


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("import localmelo.support", [(SUPPORT, "", ())]),
        (
            "import localmelo.support.config as config",
            [("localmelo.support.config", "", ())],
        ),
        ("from localmelo import support", [("localmelo", "support", ())]),
        (
            "from localmelo.support.mcp import McpToolProvider",
            [("localmelo.support.mcp", "McpToolProvider", ())],
        ),
        ("from ...support import config", [(SUPPORT, "config", ())]),
        ("from ... import support", [("localmelo", "support", ())]),
        (
            "def build():\n"
            "    import importlib\n"
            "    return importlib.import_module('localmelo.support.backends')\n",
            [("localmelo.support.backends", "", ("def build",))],
        ),
        (
            "if TYPE_CHECKING:\n"
            "    from localmelo.support.config import Config\n"
            "else:\n"
            "    from localmelo.support.config import load\n",
            [
                ("localmelo.support.config", "Config", ("if TYPE_CHECKING",)),
                ("localmelo.support.config", "load", ()),
            ],
        ),
        (
            "class Agent:\n"
            "    async def run(self):\n"
            "        from localmelo.support.gateway import start_gateway\n",
            [
                (
                    "localmelo.support.gateway",
                    "start_gateway",
                    ("class Agent", "def run"),
                )
            ],
        ),
    ],
    ids=[
        "import",
        "import_as",
        "from_package",
        "from_module",
        "relative_module",
        "relative_package",
        "import_module",
        "type_checking_only_in_its_branch",
        "nested_scopes",
    ],
)
def test_every_form_of_import_is_found(
    source: str, found: list[tuple[str, str, tuple[str, ...]]]
) -> None:
    assert support_imports(source, MODULE) == found


def test_core_imports_and_text_about_support_are_not_imports_of_it() -> None:
    source = (
        '"""Built from a ``localmelo.support.config.Config``."""\n'
        "from localmelo.melo.schema import Message\n"
        "from ..schema import ToolDef\n"
        "from . import options\n"
        'NOTE = "from localmelo.support import config"\n'
    )
    assert support_imports(source, MODULE) == []


# ── The table: what each part of the package never imports ──

# A path under the package (a directory or one module) and the modules it
# must not import. The core's imports of localmelo.support are the test
# above, with its two exceptions.
FORBIDDEN: list[tuple[str, tuple[str, ...]]] = [
    ("melo", ("prompt_toolkit", "rich")),
    (
        "support/sessions",
        (
            "localmelo.support.gateway",
            "localmelo.support.cli",
            "fastapi",
            "prompt_toolkit",
            "rich",
        ),
    ),
    ("support/cli", ("localmelo.support.gateway", "fastapi", "uvicorn")),
    ("support/progress_line.py", ("prompt_toolkit", "rich")),
    ("support/backends", ("prompt_toolkit", "rich")),
]


@pytest.mark.parametrize(
    ("path", "prefixes"), FORBIDDEN, ids=[path for path, _ in FORBIDDEN]
)
def test_no_part_imports_what_the_table_forbids(
    path: str, prefixes: tuple[str, ...]
) -> None:
    where = PACKAGE_DIR / path
    assert where.exists(), f"{path} is not in the package: the table is out of date"
    assert _imports_under(where, prefixes) == []


@pytest.mark.parametrize(
    ("source", "prefixes", "found"),
    [
        ("import rich.console", ("rich",), [("rich.console", "", ())]),
        (
            "from prompt_toolkit import Application",
            ("prompt_toolkit",),
            [("prompt_toolkit", "Application", ())],
        ),
        ("import richer", ("rich",), []),
        (
            "def f():\n    from fastapi import FastAPI\n",
            ("fastapi",),
            [("fastapi", "FastAPI", ("def f",))],
        ),
        (
            "from ..gateway import playground",
            ("localmelo.support.gateway",),
            [("localmelo.support.gateway", "playground", ())],
        ),
    ],
    ids=["submodule", "from_import", "prefix_of_a_name", "in_a_function", "relative"],
)
def test_the_table_finds_third_party_and_relative_imports(
    source: str,
    prefixes: tuple[str, ...],
    found: list[tuple[str, str, tuple[str, ...]]],
) -> None:
    assert imports_of(source, "localmelo.support.sessions.store", prefixes) == found
