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
        ("direct", False),
        ("interactive", True),
        ("gateway", False),
        ("python", False),
    ],
)
def test_shell_exec_is_in_the_default_set_per_mode(mode: ToolMode, shell: bool) -> None:
    assert ("shell_exec" in DEFAULT_BUILTINS[mode]) is shell


@pytest.mark.parametrize("mode", list(DEFAULT_BUILTINS))
def test_the_other_three_builtins_are_in_every_default_set(mode: ToolMode) -> None:
    assert {"file_read", "file_write", "python_exec"} <= DEFAULT_BUILTINS[mode]


@pytest.mark.parametrize(
    ("mode", "asks"),
    [("direct", False), ("interactive", True), ("gateway", False), ("python", False)],
)
def test_mode_requires_approval(mode: ToolMode, asks: bool) -> None:
    assert mode_requires_approval(mode) is asks


def test_approving_modes_are_the_ones_that_can_ask() -> None:
    assert frozenset({"interactive"}) == APPROVING_MODES


def test_an_explicit_set_wins_over_the_mode() -> None:
    assert builtin_set_for("gateway", {"shell_exec"}) == frozenset({"shell_exec"})
    assert builtin_set_for("interactive", ()) == frozenset()
    assert builtin_set_for("direct", ALL_BUILTINS) == ALL_BUILTINS


def test_no_names_resolves_the_mode_default() -> None:
    for mode in DEFAULT_BUILTINS:
        assert builtin_set_for(mode, None) == DEFAULT_BUILTINS[mode]


def test_a_bad_checker_mode_raises_value_error() -> None:
    from localmelo.melo.checker import Checker

    with pytest.raises(ValueError, match="mode must be one of"):
        Checker(mode="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["direct", "python", "gateway"])
def test_a_non_approving_mode_refuses_an_unnamed_shell_exec(mode: ToolMode) -> None:
    reason = exec_refusal("shell_exec", "exec", mode)
    assert reason is not None
    assert mode in reason
    assert "--builtins all" in reason


@pytest.mark.parametrize("mode", ["direct", "python"])
def test_a_named_shell_exec_is_the_callers_opt_in(mode: ToolMode) -> None:
    assert exec_refusal("shell_exec", "exec", mode, frozenset({"shell_exec"})) is None


def test_a_named_shell_exec_is_still_refused_in_gateway() -> None:
    reason = exec_refusal("shell_exec", "exec", "gateway", frozenset({"shell_exec"}))
    assert reason is not None
    assert "gateway mode has no approval policy" in reason


def test_an_explicit_empty_set_names_no_tool() -> None:
    reason = exec_refusal("shell_exec", "exec", "direct", frozenset())
    assert reason is not None


@pytest.mark.parametrize("mode", list(DEFAULT_BUILTINS))
def test_python_exec_is_never_refused(mode: ToolMode) -> None:
    assert exec_refusal("python_exec", "exec", mode) is None


def test_the_interactive_mode_refuses_no_exec_tool() -> None:
    assert exec_refusal("shell_exec", "exec", "interactive") is None


@pytest.mark.parametrize("mode", list(DEFAULT_BUILTINS))
def test_a_read_tool_is_never_refused(mode: ToolMode) -> None:
    assert exec_refusal("file_read", "read", mode) is None
