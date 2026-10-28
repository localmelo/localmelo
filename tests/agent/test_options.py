"""AgentOptions: defaults and validation, and how an Agent takes its root."""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, get_args

import pytest

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.agent.options import ReflectionGate
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.executor import builtin_names
from localmelo.melo.schema import (
    MAX_AGENT_STEPS,
    MAX_ATTEMPTS,
    STEPS_PER_ATTEMPT,
    Message,
    ToolDef,
)


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


# ── workspace_root, record_answers and tool_timeout_s ───────────────────────


class OneAnswerLLM(BaseLLMProvider):
    """Answers every request with "done"."""

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        return Message(role="assistant", content="done")

    async def close(self) -> None:
        pass


@pytest.fixture
def _no_env_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the environment point these tests at a real memory dir."""
    monkeypatch.delenv("LOCALMELO_PERSIST_MEMORY", raising=False)
    monkeypatch.delenv("LOCALMELO_MEMORY_DIR", raising=False)


def test_the_workspace_options_default_to_none_and_off() -> None:
    opts = AgentOptions()
    assert opts.workspace_root is None
    assert opts.record_answers is False
    assert opts.tool_timeout_s is None


def test_workspace_values_are_kept() -> None:
    opts = AgentOptions(
        workspace_root="~/work", record_answers=True, tool_timeout_s=600
    )
    # Resolved by the agent, not here.
    assert opts.workspace_root == "~/work"
    assert opts.record_answers is True
    assert opts.tool_timeout_s == 600


def test_tool_env_defaults_to_none() -> None:
    assert AgentOptions().tool_env is None


def test_tool_env_is_a_copy() -> None:
    given = {"MY_VALUE": "42"}
    opts = AgentOptions(tool_env=given)
    given["MY_VALUE"] = "changed"
    assert opts.tool_env == {"MY_VALUE": "42"}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            ["MY_VALUE=42"],
            "tool_env must be None or a mapping of names to values, "
            "got ['MY_VALUE=42']",
        ),
        (
            "MY_VALUE=42",
            "tool_env must be None or a mapping of names to values, "
            "got 'MY_VALUE=42'",
        ),
        ({"": "1"}, "tool_env has an invalid variable name ''"),
        ({"MY=VALUE": "1"}, "tool_env has an invalid variable name 'MY=VALUE'"),
        ({"A\0B": "1"}, "tool_env has an invalid variable name 'A\\x00B'"),
        ({"MY_VALUE": 42}, "tool_env['MY_VALUE'] must be a str, got 42"),
        ({"MY_VALUE": None}, "tool_env['MY_VALUE'] must be a str, got None"),
    ],
)
def test_tool_env_must_be_a_mapping_of_str_to_str(value: object, message: str) -> None:
    with pytest.raises(ValueError) as caught:
        AgentOptions(tool_env=value)  # type: ignore[arg-type]
    assert str(caught.value) == message


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "workspace_root",
            "",
            "workspace_root must be None or a non-empty str, got ''",
        ),
        ("workspace_root", 3, "workspace_root must be None or a non-empty str, got 3"),
        (
            "workspace_root",
            b"/tmp",
            "workspace_root must be None or a non-empty str, got b'/tmp'",
        ),
        ("record_answers", "yes", "record_answers must be a bool, got 'yes'"),
        ("record_answers", 1, "record_answers must be a bool, got 1"),
        ("record_answers", None, "record_answers must be a bool, got None"),
        (
            "tool_timeout_s",
            0,
            "tool_timeout_s must be None or a finite number > 0, got 0",
        ),
        (
            "tool_timeout_s",
            -1.5,
            "tool_timeout_s must be None or a finite number > 0, got -1.5",
        ),
        (
            "tool_timeout_s",
            math.inf,
            "tool_timeout_s must be None or a finite number > 0, got inf",
        ),
        (
            "tool_timeout_s",
            math.nan,
            "tool_timeout_s must be None or a finite number > 0, got nan",
        ),
        (
            "tool_timeout_s",
            True,
            "tool_timeout_s must be None or a finite number > 0, got True",
        ),
        (
            "tool_timeout_s",
            "10",
            "tool_timeout_s must be None or a finite number > 0, got '10'",
        ),
    ],
)
def test_an_invalid_workspace_option_raises_naming_the_field(
    field: str, value: Any, message: str
) -> None:
    options: dict[str, Any] = {field: value}
    if field == "tool_timeout_s":
        options["workspace_root"] = "/tmp"  # so that only the value is wrong
    with pytest.raises(ValueError, match=rf"^{field} ") as excinfo:
        AgentOptions(**options)
    assert str(excinfo.value) == message


def test_tool_timeout_too_large_for_a_float_raises_naming_the_field() -> None:
    with pytest.raises(ValueError, match=r"^tool_timeout_s must be None"):
        AgentOptions(workspace_root="/tmp", tool_timeout_s=10**400)


def test_tool_timeout_s_needs_a_workspace_root() -> None:
    with pytest.raises(ValueError) as excinfo:
        AgentOptions(tool_timeout_s=600)
    assert str(excinfo.value) == "tool_timeout_s needs a workspace_root"


@pytest.mark.usefixtures("_no_env_persistence")
@pytest.mark.parametrize("kind", ["file", "missing", "nul"])
def test_a_root_that_is_not_a_directory_is_refused(tmp_path: Path, kind: str) -> None:
    if kind == "file":
        path = tmp_path / "notes.txt"
        path.write_text("not a directory")
        root = str(path)
    elif kind == "missing":
        root = str(tmp_path / "nowhere")
    else:
        root = str(tmp_path) + "/a\0b"
    memory_dir = tmp_path / "memory"

    with pytest.raises(ValueError) as excinfo:
        Agent(
            llm=OneAnswerLLM(),
            options=AgentOptions(workspace_root=root, memory_dir=str(memory_dir)),
        )

    assert str(excinfo.value) == f"workspace_root is not a directory: {root!r}"
    # The root is checked before the agent creates anything.
    assert not memory_dir.exists()


@pytest.mark.usefixtures("_no_env_persistence")
def test_the_root_is_the_real_path(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    agent = Agent(llm=OneAnswerLLM(), options=AgentOptions(workspace_root=str(link)))

    assert agent.workspace_root == os.path.realpath(real)
    assert agent.executor.workspace_root == os.path.realpath(real)


@pytest.mark.usefixtures("_no_env_persistence")
def test_a_tilde_in_the_root_is_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "work").mkdir()

    agent = Agent(llm=OneAnswerLLM(), options=AgentOptions(workspace_root="~/work"))

    assert agent.workspace_root == os.path.realpath(tmp_path / "work")


@pytest.mark.usefixtures("_no_env_persistence")
def test_an_agent_without_a_root_has_none() -> None:
    agent = Agent(llm=OneAnswerLLM())
    assert agent.workspace_root is None
    assert agent.executor.workspace_root is None


@pytest.mark.usefixtures("_no_env_persistence")
@pytest.mark.asyncio
async def test_run_result_config_gains_no_key(tmp_path: Path) -> None:
    plain = await Agent(llm=OneAnswerLLM()).run_task("hi")
    with_options = await Agent(
        llm=OneAnswerLLM(),
        options=AgentOptions(
            workspace_root=str(tmp_path), record_answers=True, tool_timeout_s=600
        ),
    ).run_task("hi")

    assert with_options.status == plain.status == "completed"
    assert with_options.config == plain.config
    assert list(with_options.config) == list(plain.config)
