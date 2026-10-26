"""The scheduler reads no clock, draws no global randomness and imports only the stdlib.

Acceptance (d) and Requirements 2, 8, 23 and 50, checked on the AST of
every module under ``support/scheduler/``, so a docstring that names a
module does not count and an import of any form does.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# The directory that holds pyproject.toml is the localmelo package.
SCHEDULER = Path(__file__).resolve().parents[2] / "support" / "scheduler"
OWN = "localmelo.support.scheduler"

# The policy modules: no I/O, no clock, no randomness (Requirement 8).
POLICY = ("model", "params", "costs", "machine_gates", "levels", "policy", "accounting")
POLICY_FORBIDDEN = {
    "os",
    "pathlib",
    "io",
    "sqlite3",
    "time",
    "datetime",
    "random",
    "uuid",
    "asyncio",
    "threading",
    "subprocess",
    "socket",
}
# The only modules that may import random, to build random.Random instances.
RANDOM_ALLOWED = {"sim/workloads.py", "sim/compare.py"}
CLOCKS = {
    ("time", "time"),
    ("time", "monotonic"),
    ("time", "perf_counter"),
    ("datetime", "now"),
}


def _modules() -> list[tuple[str, ast.Module]]:
    files = sorted(SCHEDULER.rglob("*.py"))
    assert files, SCHEDULER
    return [
        (p.relative_to(SCHEDULER).as_posix(), ast.parse(p.read_text())) for p in files
    ]


def _imports(tree: ast.Module) -> list[str]:
    """Top-level module names imported; relative imports count as the package's own."""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append(OWN if node.level else node.module or "")
    return out


def test_policy_modules_read_no_clock_and_use_no_global_random_or_hash() -> None:
    found: list[str] = []
    for rel, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                pair = (node.value.id, node.attr)
                if pair in CLOCKS or (
                    node.value.id == "random" and node.attr != "Random"
                ):
                    found.append(f"{rel}: {pair[0]}.{pair[1]}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "hash"
            ):
                found.append(f"{rel}: hash(")
            if isinstance(node, ast.ImportFrom) and node.module in (
                "time",
                "random",
                "datetime",
            ):
                found += [
                    f"{rel}: from {node.module} import {a.name}" for a in node.names
                ]
        if any(name.split(".")[0] == "uuid" for name in _imports(tree)):
            found.append(f"{rel}: uuid")
    assert found == []


def test_scheduler_imports_only_the_stdlib_and_itself() -> None:
    stdlib = set(sys.stdlib_module_names)
    for rel, tree in _modules():
        names = _imports(tree)
        for name in names:
            top = name.split(".")[0]
            assert name == OWN or name.startswith(OWN + ".") or top in stdlib, (
                rel,
                name,
            )
            assert top not in {"numpy", "aiosqlite", "httpx", "fastapi"}, (rel, name)
        tops = {name.split(".")[0] for name in names}
        if rel.removesuffix(".py") in POLICY:
            assert not tops & POLICY_FORBIDDEN, (rel, sorted(tops & POLICY_FORBIDDEN))
        if "random" in tops:
            assert rel in RANDOM_ALLOWED, rel
        assert not [
            n for n in names if n.startswith("localmelo.") and not n.startswith(OWN)
        ], rel
