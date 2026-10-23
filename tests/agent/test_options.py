"""AgentOptions: defaults and validation."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from typing import Any, get_args

import pytest

from localmelo.melo.agent import AgentOptions
from localmelo.melo.agent.options import ReflectionGate
from localmelo.melo.executor import builtin_names
from localmelo.melo.schema import MAX_AGENT_STEPS, MAX_ATTEMPTS, STEPS_PER_ATTEMPT


def test_defaults_equal_the_schema_constants() -> None:
    opts = AgentOptions()
    assert opts.max_steps == MAX_AGENT_STEPS
    assert opts.steps_per_attempt == STEPS_PER_ATTEMPT
    assert opts.max_attempts == MAX_ATTEMPTS
    assert opts.attempt_policy == "reflect"
    assert opts.reflection_gate == "flags"
    assert opts.tool_exposure == "retrieve"
    assert opts.builtins is None
    assert opts.memory is True
    assert opts.memory_dir is None
    assert opts.timeout_s is None


def test_is_frozen() -> None:
    opts = AgentOptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        opts.max_steps = 3  # type: ignore[misc]


def test_valid_values_are_kept() -> None:
    opts = AgentOptions(
        max_steps=4,
        attempt_policy="none",
        steps_per_attempt=2,
        max_attempts=1,
        reflection_gate="utility",
        tool_exposure="all",
        builtins=frozenset({"file_read"}),
        memory=False,
        memory_dir="~/memory",
        timeout_s=2.5,
    )
    assert opts.max_steps == 4
    assert opts.attempt_policy == "none"
    assert opts.steps_per_attempt == 2
    assert opts.max_attempts == 1
    assert opts.reflection_gate == "utility"
    assert opts.tool_exposure == "all"
    assert opts.builtins == frozenset({"file_read"})
    assert opts.memory is False
    # Expanded by the agent, not here.
    assert opts.memory_dir == "~/memory"
    assert opts.timeout_s == 2.5


def test_integer_timeout_is_accepted() -> None:
    assert AgentOptions(timeout_s=30).timeout_s == 30


@pytest.mark.parametrize(
    "names",
    [
        ["file_read", "shell_exec"],
        ("shell_exec", "file_read", "shell_exec"),
        {"file_read", "shell_exec"},
        (name for name in ("file_read", "shell_exec")),
    ],
)
def test_builtins_iterable_is_stored_as_frozenset(names: Iterable[str]) -> None:
    opts = AgentOptions(builtins=names)  # type: ignore[arg-type]
    assert isinstance(opts.builtins, frozenset)
    assert opts.builtins == frozenset({"file_read", "shell_exec"})


def test_empty_builtins_selects_none() -> None:
    assert AgentOptions(builtins=frozenset()).builtins == frozenset()


def test_normalized_options_compare_and_hash_equal() -> None:
    from_list = AgentOptions(builtins=["file_read"])  # type: ignore[arg-type]
    from_set = AgentOptions(builtins=frozenset({"file_read"}))
    assert from_list == from_set
    assert hash(from_list) == hash(from_set)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 0),
        ("max_steps", -1),
        ("max_steps", True),
        ("max_steps", 2.0),
        ("max_steps", "3"),
        ("max_steps", None),
        ("steps_per_attempt", 0),
        ("steps_per_attempt", False),
        ("max_attempts", 0),
        ("max_attempts", True),
        ("attempt_policy", "reflection"),
        ("attempt_policy", ""),
        ("attempt_policy", None),
        ("reflection_gate", "legacy"),
        ("reflection_gate", "FLAGS"),
        ("reflection_gate", ""),
        ("reflection_gate", None),
        ("tool_exposure", "some"),
        ("tool_exposure", None),
        ("builtins", "shell_exec"),
        ("builtins", 3),
        ("builtins", ["nope"]),
        ("builtins", [1]),
        ("builtins", [["file_read"]]),
        ("memory", "off"),
        ("memory", 0),
        ("memory", None),
        ("memory_dir", ""),
        ("memory_dir", 3),
        ("timeout_s", 0),
        ("timeout_s", -1.5),
        ("timeout_s", math.inf),
        ("timeout_s", math.nan),
        ("timeout_s", True),
        ("timeout_s", "10"),
    ],
)
def test_invalid_value_raises_naming_the_field(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=rf"^{field} "):
        AgentOptions(**{field: value})


def test_timeout_too_large_for_a_float_raises_naming_the_field() -> None:
    with pytest.raises(ValueError, match=r"^timeout_s "):
        AgentOptions(timeout_s=10**400)


def test_unknown_builtins_lists_unknown_and_known_names() -> None:
    with pytest.raises(ValueError) as excinfo:
        AgentOptions(builtins=frozenset({"file_read", "zap", "nope"}))
    message = str(excinfo.value)
    assert "'nope', 'zap'" in message
    for name in builtin_names():
        assert repr(name) in message


def test_reflection_gate_names_its_values() -> None:
    assert get_args(ReflectionGate) == ("flags", "utility")
    with pytest.raises(ValueError) as excinfo:
        AgentOptions(reflection_gate="legacy")  # type: ignore[arg-type]
    assert str(excinfo.value) == (
        "reflection_gate must be one of 'flags', 'utility', got 'legacy'"
    )
