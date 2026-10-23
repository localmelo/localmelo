"""Executor: registering all, some or none of the built-in tools."""

from __future__ import annotations

from collections.abc import Collection

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.executor import BUILTINS, Executor, builtin_names, register_builtins
from localmelo.melo.memory.coordinator import Hippo

ALL_NAMES = {tool_def.name for tool_def, _ in BUILTINS}


def _registered(names: Collection[str] | None) -> tuple[set[str], set[str]]:
    """Names in the tool registry and in the executor after registering *names*."""
    hippo = Hippo()
    executor = Executor(hippo, Checker())
    register_builtins(executor, hippo, names=names)
    return {t.name for t in hippo.list_tools()}, set(executor._callables)


def test_builtin_names_lists_every_builtin() -> None:
    assert builtin_names() == ALL_NAMES
    assert {"file_read", "file_write", "python_exec", "shell_exec"} <= ALL_NAMES


def test_existing_signature_registers_all() -> None:
    hippo = Hippo()
    executor = Executor(hippo, Checker())
    register_builtins(executor, hippo)
    assert {t.name for t in hippo.list_tools()} == ALL_NAMES
    assert set(executor._callables) == ALL_NAMES


def test_none_registers_all() -> None:
    assert _registered(None) == (ALL_NAMES, ALL_NAMES)


@pytest.mark.parametrize(
    "names",
    [
        {"file_read"},
        frozenset({"python_exec", "shell_exec"}),
        ["file_write", "file_read"],
    ],
)
def test_selection_registers_only_those(names: Collection[str]) -> None:
    assert _registered(names) == (set(names), set(names))


def test_empty_selection_registers_none() -> None:
    assert _registered(()) == (set(), set())


def test_unknown_name_raises_before_registering() -> None:
    hippo = Hippo()
    executor = Executor(hippo, Checker())
    with pytest.raises(ValueError, match="nope") as excinfo:
        register_builtins(executor, hippo, names=["file_read", "nope"])
    for name in ALL_NAMES:
        assert name in str(excinfo.value)
    assert hippo.list_tools() == []
    assert executor._callables == {}
