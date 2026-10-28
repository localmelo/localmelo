"""Executor: registering all, some or none of the built-in tools."""

from __future__ import annotations

from collections.abc import Collection

import pytest

from localmelo.melo.checker import Checker
from localmelo.melo.executor import (
    APPROVAL_BUILTINS,
    BUILTINS,
    Executor,
    builtin_names,
    register_builtins,
)
from localmelo.melo.memory.coordinator import Hippo

ALL_NAMES = {tool_def.name for tool_def, _ in BUILTINS}


def test_the_approval_set_still_names_the_same_three() -> None:
    assert frozenset({"file_write", "python_exec", "shell_exec"}) == APPROVAL_BUILTINS


def _registered(
    names: Collection[str] | None, *, mode: str = "python"
) -> tuple[set[str], set[str]]:
    """Names in the tool registry and in the executor after registering *names*."""
    hippo = Hippo()
    executor = Executor(hippo, Checker())
    register_builtins(executor, hippo, names=names, mode=mode)  # type: ignore[arg-type]
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


def test_none_registers_the_python_mode_defaults() -> None:
    assert _registered(None) == (ALL_NAMES, ALL_NAMES)


def test_the_gateway_default_leaves_shell_exec_out() -> None:
    expected = ALL_NAMES - {"shell_exec"}
    assert _registered(None, mode="gateway") == (expected, expected)


@pytest.mark.parametrize("mode", ["direct", "interactive"])
def test_the_other_modes_keep_shell_exec_in_p13a(mode: str) -> None:
    assert _registered(None, mode=mode) == (ALL_NAMES, ALL_NAMES)


def test_an_explicit_set_wins_over_the_mode() -> None:
    assert _registered({"shell_exec"}, mode="gateway") == (
        {"shell_exec"},
        {"shell_exec"},
    )
    assert _registered((), mode="interactive") == (set(), set())


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
