"""Replays are deterministic, and the scheduler reads no clock and imports only the stdlib.

Acceptance (d) and Requirements 2, 7, 8, 23 and 50: a replay gives the
same decision log in one process and in another with a different hash
seed; the AST of every module under ``support/scheduler/`` shows no
clock read, no global randomness and only stdlib imports (a docstring
that names a module does not count, an import of any form does); the
packages export exactly their public names.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localmelo.support.scheduler import ARMS
from localmelo.support.scheduler.sim import compute_metrics, make_workload, run
from localmelo.support.scheduler.sim.engine import Mode

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
# Dotted clock reads at any depth (the lead's decision 3: sim/ too).
DOTTED_CLOCKS = {"datetime.datetime.now", "date.today", "datetime.date.today"}
CLOCK_ATTRS = {"utcnow"}

# The public names of Requirement 7.
PUBLIC = [
    "Task",
    "Call",
    "Dep",
    "Machine",
    "Clock",
    "State",
    "Decision",
    "Dispatch",
    "Params",
    "TaskAdded",
    "TaskUpdated",
    "DepsSet",
    "CallSubmitted",
    "CallEnded",
    "Typing",
    "MachineChanged",
    "Tick",
    "ARMS",
    "REASONS",
    "params_for_arm",
    "decide",
    "admit_lane",
    "dispatch",
    "fallback_fcfs",
    "advance",
    "apply_dispatch",
    "decision_json",
    "evaluate_gates",
]
SIM_PUBLIC = [
    "run",
    "SimResult",
    "Workload",
    "make_workload",
    "SEEDS",
    "ServerParams",
    "import_calls",
    "import_history",
    "ImportReport",
    "TraceCall",
    "TaskTrace",
    "TaskTemplate",
    "read_trace",
    "write_trace",
    "compute_metrics",
    "run_sweep",
    "compare",
]


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


def _dotted(node: ast.AST) -> str:
    """``a.b.c`` for an attribute chain on a name, else ``""``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else ""


def test_policy_modules_read_no_clock_and_use_no_global_random_or_hash() -> None:
    found: list[str] = []
    for rel, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and (
                _dotted(node) in DOTTED_CLOCKS or node.attr in CLOCK_ATTRS
            ):
                found.append(f"{rel}: {_dotted(node) or node.attr}")
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


@pytest.mark.parametrize("arm", ARMS)
def test_a_replay_gives_an_identical_decision_log(arm: Mode) -> None:
    wl = make_workload("W2", 3)
    first, second = run(wl, arm, keep_log=True), run(make_workload("W2", 3), arm)
    assert first.log_sha256 == second.log_sha256 and first.decisions == len(first.log)
    assert compute_metrics(first).to_json() == compute_metrics(second).to_json()


def test_a_replay_in_a_subprocess_with_another_hash_seed_matches() -> None:
    code = (
        "from dataclasses import replace\n"
        "from localmelo.support.scheduler.sim import make_workload, run\n"
        "wl = replace(make_workload('W2c', 2), horizon_us=30 * 60 * 10**6)\n"
        "print(run(wl, 'las2').log_sha256)\n"
    )
    hashes = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        hashes.append(out.stdout.strip())
    assert hashes[0] == hashes[1] and len(hashes[0]) == 64


def test_the_packages_export_exactly_the_public_names() -> None:
    for name, expected in ((OWN, PUBLIC), (OWN + ".sim", SIM_PUBLIC)):
        module = importlib.import_module(name)
        assert sorted(module.__all__) == sorted(expected), name
        assert all(getattr(module, n, None) is not None for n in expected), name
