"""The core, ``melo/``, imports nothing from ``support/`` — no network.

The rule is checked on the AST of every module under ``melo/``, so text
that only names ``localmelo.support``, such as a docstring, does not
count, and an import of any form does: plain, relative or through
``importlib.import_module``. Two imports predate the rule and are allowed,
matched by module, name and the scope they sit in rather than by line:
``Config``, imported only for type checking, and ``get_backend``, imported
inside ``_providers_from_config`` when an Agent is built from a Config.
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


def _is_support(module: str) -> bool:
    return module == SUPPORT or module.startswith(f"{SUPPORT}.")


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


class _SupportImports(ast.NodeVisitor):
    """Collects the imports of localmelo.support in one module's AST.

    *package* is the package that the module's relative imports start
    from. Each import found is ``(module, name, scopes)``.
    """

    def __init__(self, package: str) -> None:
        self.package = package
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

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if _is_support(alias.name):
                self._add(alias.name, "")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = self._absolute(node)
        for alias in node.names:
            if _is_support(module) or _is_support(f"{module}.{alias.name}"):
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
                and _is_support(first.value)
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


def support_imports(
    source: str, module: str, *, is_package: bool = False
) -> list[tuple[str, str, tuple[str, ...]]]:
    """The imports of localmelo.support in *source*, the code of *module*."""
    visitor = _SupportImports(module if is_package else module.rpartition(".")[0])
    visitor.visit(ast.parse(source))
    return visitor.found


def _core_imports() -> list[Found]:
    found: list[Found] = []
    for path in sorted(CORE_DIR.rglob("*.py")):
        relative = path.relative_to(PACKAGE_DIR)
        parts = list(relative.with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        module = ".".join(["localmelo", *parts])
        source = path.read_text(encoding="utf-8")
        found.extend(
            (relative.as_posix(), imported, name, scopes)
            for imported, name, scopes in support_imports(
                source, module, is_package=is_package
            )
        )
    return found


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
