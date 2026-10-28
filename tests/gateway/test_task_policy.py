"""The task boundary: TaskSpec, workspace roots and worker allowlists."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from localmelo.support.config import Config
from localmelo.support.gateway.task_policy import (
    DEFAULT_WORKSPACE,
    TaskSpec,
    TaskSpecError,
    UnknownOriginError,
    WorkspaceError,
    validate_task_workspace,
    worker_tool_allowlist,
)
from localmelo.support.gateway.task_store import TaskStore, TaskStoreError
from localmelo.support.sessions.store import SessionStore


def cfg(workspace: str = "") -> SimpleNamespace:
    return SimpleNamespace(gateway=SimpleNamespace(workspace=workspace))


def test_a_missing_workspace_creates_and_returns_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = validate_task_workspace(None, Config())
    assert path == os.path.realpath(os.path.expanduser(DEFAULT_WORKSPACE))
    assert os.path.isdir(path)


def test_a_user_workspace_is_inside_home_and_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = tmp_path / "projects" / "talk"
    folder.mkdir(parents=True)
    assert validate_task_workspace(str(folder), cfg()) == os.path.realpath(folder)
    assert validate_task_workspace("~/projects/talk", cfg()) == os.path.realpath(folder)


def test_a_user_workspace_outside_home_or_hidden_or_library_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    outside.mkdir()
    (home / ".ssh").mkdir(parents=True)
    (home / "projects" / ".hidden").mkdir(parents=True)
    (home / "Library" / "Caches").mkdir(parents=True)
    cases = [
        (outside, "strictly inside your home"),
        (home, "rooted at"),
        (Path("/"), "rooted at"),
        (home / ".ssh", "hidden folder"),
        (home / "projects" / ".hidden", "hidden folder"),
        (home / "Library" / "Caches", "under ~/Library"),
    ]
    for folder, message in cases:
        with pytest.raises(WorkspaceError, match=message):
            validate_task_workspace(str(folder), cfg())
    # macOS's default filesystem is case-insensitive: ~/library is ~/Library.
    lower = home / "library" / "Caches"
    lower.mkdir(parents=True, exist_ok=True)
    with pytest.raises(WorkspaceError, match="under ~/Library"):
        validate_task_workspace(str(lower), cfg())


def test_a_user_symlink_escaping_home_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    outside = tmp_path / "outside"
    outside.mkdir()
    link = home / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceError, match="strictly inside your home"):
        validate_task_workspace(str(link), cfg())


def test_a_non_user_workspace_must_be_strictly_inside_the_configured_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "workspace"
    inside = root / "task"
    inside.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    assert validate_task_workspace(str(inside), cfg(str(root)), origin="agent") == (
        os.path.realpath(inside)
    )
    for folder in (root, outside):
        with pytest.raises(WorkspaceError, match="strictly inside"):
            validate_task_workspace(str(folder), cfg(str(root)), origin="front")
    for origin in ("agent", "user_rule"):
        child = root / f"task-{origin}"
        assert validate_task_workspace(
            None, cfg(str(root)), origin=origin, task_id=f"task-{origin}"
        ) == os.path.realpath(child)
        assert child.is_dir()
    # A user rule is not the user: it follows the configured root too.
    with pytest.raises(WorkspaceError, match="strictly inside"):
        validate_task_workspace(str(outside), cfg(str(root)), origin="user_rule")


def test_a_configured_workspace_replaces_the_default_for_a_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = validate_task_workspace(None, cfg(str(tmp_path / "custom")))
    assert path == os.path.realpath(tmp_path / "custom")
    assert os.path.isdir(path)


def test_a_relative_or_missing_workspace_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(WorkspaceError, match="absolute path"):
        validate_task_workspace("relative/path", cfg())
    with pytest.raises(WorkspaceError, match="Not a directory"):
        validate_task_workspace(str(tmp_path / "missing"), cfg())
    with pytest.raises(UnknownOriginError):
        validate_task_workspace(None, cfg(), origin="robot")


def test_allowlists_separate_principals_from_derived_tasks() -> None:
    for origin in ("user", "user_rule"):
        policy = worker_tool_allowlist(origin)
        assert policy.builtins == {"file_read", "file_write", "python_exec"}
        assert policy.remember_approvals is True
        assert policy.allow_unmarked_mcp is True
        assert policy.allow_session_grants is True
    for origin in ("front", "harness", "agent", "channel", "sleep"):
        policy = worker_tool_allowlist(origin)
        assert policy.builtins == {"file_read", "file_write", "python_exec"}
        assert policy.remember_approvals is False
        assert policy.allow_unmarked_mcp is False
        assert policy.allow_session_grants is False
    with pytest.raises(UnknownOriginError):
        worker_tool_allowlist("robot")


def test_spec_normalizes_the_prompt_alias_and_defaults() -> None:
    task = TaskSpec.from_payload({"prompt": "  Do the thing  "})
    assert task.objective == "Do the thing"
    assert task.title == "Do the thing"
    assert task.level == 2
    assert task.level_source == "computed"
    assert task.workspace_mode == "shared"
    assert task.deadline is None
    assert task.features == {}


def test_spec_keeps_every_field_and_parses_a_local_deadline() -> None:
    task = TaskSpec.from_payload(
        {
            "objective": "Make slides",
            "title": "Slides",
            "done_when": ["slides.pdf exists"],
            "user_words": "make slides",
            "context": "notes.md",
            "inputs": ["notes.md", {"kind": "file", "ref": "outline.md"}],
            "workspace": "~/work",
            "workspace_mode": "worktree",
            "budget_s": 1200,
            "deadline": "2026-10-26T15:00",
            "features": {"deadline": {"quote": "for 3pm", "trusted": True}},
            "level": 1,
            "reason": "due today",
            "report": "a 3-line summary",
        }
    )
    assert task.done_when == ("slides.pdf exists",)
    assert task.inputs[0] == {"kind": "file", "ref": "notes.md"}
    assert task.workspace == "~/work"
    assert task.workspace_mode == "worktree"
    assert task.budget_s == 1200.0
    assert task.deadline == datetime.fromisoformat("2026-10-26T15:00").timestamp()
    assert task.features["deadline"] == {"quote": "for 3pm", "trusted": True}
    assert task.level == 1
    assert TaskSpec.from_payload({"objective": "x", "deadline": 12.5}).deadline == 12.5
    assert (
        TaskSpec.from_payload(
            {"objective": "x", "deadline": "2026-10-26T15:00 (local)"}
        ).deadline
        == datetime.fromisoformat("2026-10-26T15:00").timestamp()
    )
    assert task.to_dict()["title"] == "Slides"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "objective"),
        ({"objective": "x" * 100_001}, "at most 100000"),
        ({"objective": "x", "context": "x" * 1_501}, "at most 1500"),
        ({"objective": "x", "level": 0}, "P0 is the user"),
        ({"objective": "x", "level": 6}, "level must be"),
        ({"objective": "x", "level": True}, "level must be"),
        ({"objective": "x", "workspace_mode": "readonly"}, "workspace_mode"),
        ({"objective": "x", "workspace_mode": []}, "workspace_mode"),
        ({"objective": "x", "workspace_mode": {}}, "workspace_mode"),
        ({"objective": "x", "budget_s": 0}, "budget_s"),
        ({"objective": "x", "budget_s": True}, "budget_s"),
        ({"objective": "x", "done_when": [""]}, "done_when"),
        ({"objective": "x", "inputs": [3]}, "inputs"),
        ({"objective": "x", "features": {"when": object()}}, "features"),
        ({"objective": "x", "features": {"blob": "x" * 20_000}}, "features"),
        (
            {"objective": "x", "inputs": [{"kind": "file", "ref": "x" * 5_000}]},
            "inputs",
        ),
        ({"objective": "x", "deadline": "tomorrow"}, "deadline"),
    ],
)
def test_a_bad_spec_names_the_field(payload: dict[str, object], message: str) -> None:
    with pytest.raises(TaskSpecError, match=message):
        TaskSpec.from_payload(payload)


@pytest.mark.parametrize("level", range(6))
def test_trusted_user_level_is_pinned(level: int) -> None:
    task = TaskSpec.from_user_payload({"objective": "urgent", "level": level})
    assert (task.level, task.level_source) == (level, "user")


@pytest.mark.parametrize("level", [0, 1])
async def test_trusted_create_stays_pinned_against_computed_updates(
    level: int, tmp_path: Path
) -> None:
    sessions = SessionStore(tmp_path / "sessions.db", writer="gateway")
    await sessions.open()
    try:
        tasks = TaskStore(sessions)
        await tasks.open()
        spec = TaskSpec.from_user_payload({"objective": "urgent", "level": level})
        record, _ = await tasks.create_task(spec, origin="user")
        assert (record.level, record.level_source, record.level_pinned) == (
            level,
            "user",
            1,
        )
        with pytest.raises(TaskStoreError, match="set by the user"):
            await tasks.set_level(record.task_id, 3, source="computed")
    finally:
        await sessions.aclose()


def test_user_default_and_untrusted_levels_are_computed() -> None:
    assert TaskSpec.from_user_payload({"objective": "x"}).level_source == "computed"
    assert (
        TaskSpec.from_payload({"objective": "x", "level": 1}).level_source == "computed"
    )
    for source in ("user", "user_rule", "computed"):
        for constructor in (TaskSpec.from_payload, TaskSpec.from_user_payload):
            with pytest.raises(TaskSpecError, match="reserved"):
                constructor({"objective": "x", "level": 0, "level_source": source})


def test_spec_normalizes_json_for_idempotent_repeats() -> None:
    task = TaskSpec.from_payload(
        {"objective": "x", "features": {1: ("x", "y")}, "inputs": [{2: (3, 4)}]}
    )
    assert task.features == {"1": ["x", "y"]}
    assert task.inputs == ({"2": [3, 4]},)


@pytest.mark.parametrize("root", ["/", "~", "~/", "relative"])
def test_invalid_configured_root_is_rejected_even_with_explicit_child(
    root: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    child = tmp_path / "child"
    child.mkdir()
    with pytest.raises(WorkspaceError, match="configured workspace"):
        validate_task_workspace(str(child), cfg(root), origin="agent")


@pytest.mark.parametrize(
    "task_id", [None, "", ".", "..", "../outside", "/tmp/x", "a/b", "a\\b"]
)
def test_unsafe_task_ids_are_rejected(
    task_id: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    with pytest.raises(WorkspaceError, match="safe task_id"):
        validate_task_workspace(
            None, cfg(str(tmp_path / "root")), origin="agent", task_id=task_id
        )


def test_non_user_symlinks_cannot_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "task").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceError):
        validate_task_workspace(None, cfg(str(root)), origin="agent", task_id="task")
    with pytest.raises(WorkspaceError, match="strictly inside"):
        validate_task_workspace(str(root / "task"), cfg(str(root)), origin="agent")
    (root / "link").symlink_to(root, target_is_directory=True)
    with pytest.raises(WorkspaceError):
        validate_task_workspace(None, cfg(str(root)), origin="agent", task_id="link")


def test_case_variant_home_uses_filesystem_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "HomeCase"
    (home / "Work").mkdir(parents=True)
    variant = tmp_path / "homecase"
    if not variant.exists():
        pytest.skip("requires a case-insensitive filesystem")
    monkeypatch.setenv("HOME", str(home))
    assert validate_task_workspace(str(variant / "Work"), cfg()) == os.path.realpath(
        variant / "Work"
    )
