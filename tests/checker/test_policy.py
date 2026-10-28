"""The mode policy: the default built-ins and whether a mode can ask."""

from __future__ import annotations

from typing import get_args

import pytest

from localmelo.melo.checker.policy import (
    APPROVING_MODES,
    DEFAULT_BUILTINS,
    ToolMode,
    builtin_set_for,
    exec_refusal,
    mode_requires_approval,
)

ALL_BUILTINS = {"file_read", "file_write", "python_exec", "shell_exec"}


def test_every_mode_has_a_default_set() -> None:
    assert set(DEFAULT_BUILTINS) == set(get_args(ToolMode))


@pytest.mark.parametrize(
    ("mode", "shell"),
    [
        ("direct", True),  # P13b flips this after the bench arm
        ("interactive", True),
        ("gateway", False),
        ("python", True),
    ],
)
def test_shell_exec_is_off_only_where_nobody_can_ask(
    mode: ToolMode, shell: bool
) -> None:
    assert ("shell_exec" in DEFAULT_BUILTINS[mode]) is shell


@pytest.mark.parametrize("mode", list(DEFAULT_BUILTINS))
def test_the_other_three_builtins_are_in_every_default_set(mode: ToolMode) -> None:
    assert {"file_read", "file_write", "python_exec"} <= DEFAULT_BUILTINS[mode]


@pytest.mark.parametrize(
    ("mode", "asks"),
    [("direct", False), ("interactive", True), ("gateway", False), ("python", True)],
)
def test_mode_requires_approval(mode: ToolMode, asks: bool) -> None:
    assert mode_requires_approval(mode) is asks


def test_approving_modes_are_the_ones_that_can_ask() -> None:
    assert frozenset({"interactive", "python"}) == APPROVING_MODES


def test_an_explicit_set_wins_over_the_mode() -> None:
    assert builtin_set_for("gateway", {"shell_exec"}) == frozenset({"shell_exec"})
    assert builtin_set_for("interactive", ()) == frozenset()


def test_no_names_resolves_the_mode_default() -> None:
    for mode in DEFAULT_BUILTINS:
        assert builtin_set_for(mode, None) == DEFAULT_BUILTINS[mode]


def test_a_gateway_refuses_shell_exec_but_keeps_python_exec() -> None:
    reason = exec_refusal("shell_exec", "exec", "gateway")
    assert reason is not None
    assert "gateway" in reason
    assert "--builtins all" in reason
    assert exec_refusal("python_exec", "exec", "gateway") is None


@pytest.mark.parametrize("mode", ["direct", "interactive", "python"])
def test_an_approving_or_legacy_mode_refuses_no_exec_tool(mode: ToolMode) -> None:
    assert exec_refusal("shell_exec", "exec", mode) is None


@pytest.mark.parametrize("mode", list(DEFAULT_BUILTINS))
def test_a_read_tool_is_never_refused(mode: ToolMode) -> None:
    assert exec_refusal("file_read", "read", mode) is None
