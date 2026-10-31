"""A task works in its chat's folder, never another; its tools get a HOME of their own.

The worker's Agent and runner are stand-ins (no model): the tests read the
folder each task is created in, the options its Agent is built with, and
what its card says when it ends with only the commit left.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.executor.builtins import _run_python
from localmelo.support.config import Config
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.finishing import commit_command, not_committed
from localmelo.support.gateway.task_policy import WorkspaceError
from localmelo.support.gateway.workers import WORKER_GUIDE, TaskWorkers, task_home
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.gateway.test_workers import FakeRunner, eventually


class OptionsAgent:
    """An Agent stand-in that keeps the options it was built with."""

    def __init__(self, options: Any) -> None:
        self.options = options

    async def close(self) -> None:
        pass


Setup = tuple[TaskWorkers, Config, SessionStore, list[dict[str, Any]], Path]


@pytest.fixture
async def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Setup]:
    home = tmp_path / "person"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in ("UV_CACHE_DIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    cfg = Config()
    cfg.gateway.workspace = "~/localmelo-workspace"
    store = SessionStore(tmp_path / "data" / "sessions.db", writer="gateway")
    await store.open()
    gate = ChatGate(policy="off")
    hub = SessionHub(store)
    built: list[dict[str, Any]] = []

    async def build(**kwargs: Any) -> OptionsAgent:
        built.append(kwargs)
        return OptionsAgent(kwargs["options"])

    monkeypatch.setattr("localmelo.support.gateway.workers.build_session_agent", build)
    monkeypatch.setattr("localmelo.support.gateway.workers.SessionRunner", FakeRunner)
    FakeRunner.instances = []
    workers = TaskWorkers(store, gate, lambda: cfg, hub)
    await workers.open()
    main = await store.main_chat()
    workers.front_id = main.id
    try:
        yield workers, cfg, store, built, home
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await store.aclose()


async def test_a_main_chat_task_works_in_the_default_folder(setup: Setup) -> None:
    workers, cfg, _, _, home = setup
    cfg.gateway.workspace = "~/default-here"
    body, _ = await workers.create({"objective": "tidy", "front_id": workers.front_id})
    assert body["workspace"] == str((home / "default-here").resolve())
    alone, _ = await workers.create({"objective": "tidy again"})
    assert alone["workspace"] == body["workspace"]


async def test_a_side_chat_task_works_in_that_chat_folder(setup: Setup) -> None:
    workers, _, store, _, home = setup
    project = home / "project"
    project.mkdir()
    side = await store.create_session(workspace=str(project.resolve()))
    body, _ = await workers.create({"objective": "fix", "front_id": side.id})
    assert body["workspace"] == str(project.resolve())
    assert body["front_id"] == side.id


async def test_a_folder_the_request_names_wins(setup: Setup) -> None:
    workers, _, store, _, home = setup
    named, project = home / "named", home / "project"
    named.mkdir()
    project.mkdir()
    side = await store.create_session(workspace=str(project))
    body, _ = await workers.create(
        {"objective": "fix", "front_id": side.id, "workspace": str(named)}
    )
    assert body["workspace"] == str(named.resolve())


async def test_a_refused_default_fails_the_task_and_never_runs_elsewhere(
    setup: Setup,
) -> None:
    workers, cfg, _, built, home = setup
    outside = tempfile.mkdtemp(prefix="melo-outside-")
    try:
        cfg.gateway.workspace = outside
        body, _ = await workers.create(
            {"objective": "work", "front_id": workers.front_id}
        )
        await asyncio.sleep(0.05)
        assert (body["state"], body["wait_reason"]) == ("failed", "invalid_workspace")
        assert body["workspace"] == os.path.abspath(outside)
        assert "inside your home folder" in body["folder_problem"]
        assert FakeRunner.instances == [] and built == []
        assert not (home / "localmelo-workspace").exists()
    finally:
        os.rmdir(outside)


async def test_a_refused_side_chat_folder_fails_the_task(setup: Setup) -> None:
    workers, _, store, built, home = setup
    hidden = home / ".secret"
    hidden.mkdir()
    side = await store.create_session(workspace=str(hidden))
    body, _ = await workers.create({"objective": "work", "front_id": side.id})
    assert body["state"] == "failed"
    assert body["workspace"] == str(hidden)
    assert "hidden folder" in body["folder_problem"]
    assert built == []


async def test_a_refused_folder_the_request_names_is_an_error(setup: Setup) -> None:
    workers, _, _, _, home = setup
    (home / ".hidden").mkdir()
    with pytest.raises(WorkspaceError, match="hidden folder"):
        await workers.create({"objective": "work", "workspace": str(home / ".hidden")})


async def test_a_task_s_tools_get_a_home_of_their_own(setup: Setup) -> None:
    workers, _, _, built, home = setup
    (home / ".gitconfig").write_text("[user]\n\tname = Someone\n")
    body, _ = await workers.create({"objective": "test it"})
    await eventually(lambda: len(built) == 1)
    options = built[0]["options"]
    task_home_dir = options.tool_env["HOME"]
    assert os.path.isdir(task_home_dir) and not task_home_dir.startswith(str(home))
    assert options.tool_env["GIT_CONFIG_GLOBAL"] == str(home / ".gitconfig")
    assert options.tool_env["UV_CACHE_DIR"] == str(home / ".cache" / "uv")
    assert options.identity == WORKER_GUIDE and options.task_aids is True
    assert {"git_status", "file_edit", "list_files"} <= options.builtins
    await workers.update(body["task_id"], state="stop")
    await eventually(lambda: not os.path.exists(task_home_dir))


async def test_a_task_home_reaches_the_tools() -> None:
    folder, env = task_home("abc123")
    try:
        printed = await _run_python(
            "import os; print(os.environ['HOME'])", tool_env=env
        )
        assert printed.strip() == folder
    finally:
        os.rmdir(folder)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "a.py").write_text("x = 1\n")
    return root


LEFT = (
    "Done except for the final commit. Remaining step: "
    'git add -A && git commit -m "Fix the parser"'
)


async def test_a_task_with_only_its_commit_left_says_not_committed(repo: Path) -> None:
    assert commit_command(LEFT) == 'git add -A && git commit -m "Fix the parser"'
    line = await not_committed(LEFT, str(repo))
    assert line == 'Not committed. Run: git add -A && git commit -m "Fix the parser"'


@pytest.mark.parametrize(
    "answer",
    [
        "Fixed the parser; committed with `git commit -m 'Fix'`.",
        "Fixed the parser.",
    ],
)
async def test_a_committed_or_plain_answer_says_nothing(
    repo: Path, answer: str
) -> None:
    assert await not_committed(answer, str(repo)) == ""


async def test_a_clean_folder_says_nothing(repo: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "x"],
        cwd=repo,
        check=True,
    )
    assert await not_committed(LEFT, str(repo)) == ""


def test_a_folder_outside_the_roots_says_so_before_whether_it_exists(
    setup: Setup,
) -> None:
    from localmelo.support.gateway.task_policy import folder_problem

    _, cfg, _, _, home = setup
    assert "inside your home folder" in folder_problem("/Volumes/Missing", cfg)
    assert "Not a directory" in folder_problem(str(home / "missing"), cfg)
    (home / "here").mkdir()
    assert folder_problem(str(home / "here"), cfg) == ""


async def test_a_worker_searches_its_task_folder_without_asking(setup: Setup) -> None:
    workers, cfg, _, built, home = setup
    project = home / "project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "parser.py").write_text("def parse_header(line):\n    pass\n")
    await workers.create({"objective": "find it", "workspace": str(project)})
    await eventually(lambda: len(built) == 1)
    # The second provider is ask_owner (#369): the worker asks its chat, never its folder.
    provider, asks = built[0]["tools"]
    assert [t.definition.name for t in await asks.start()] == ["ask_owner"]
    [tool] = await provider.start()
    assert tool.definition.name == "search_files"
    assert tool.requires_approval is False
    assert tool.definition.annotations.get("readOnlyHint") is True
    found = await tool.function(pattern="def parse_header")
    assert "src/parser.py:1" in found
    assert "search_files" in built[0]["options"].identity
