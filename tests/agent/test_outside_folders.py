"""A path outside the workspace in a run (#405): the person is asked, not refused.

The model is scripted. A chat's policy asks the person about a folder
outside the workspace in Ask and in Auto mode; a run the person did not
start, and a policy that cannot ask (a task nobody owns), refuse the path
as before. HOME is a temporary folder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
)
from localmelo.support import config
from localmelo.support.sessions.approvals import ChatPolicy, RememberingPolicy
from localmelo.tests.agent.test_approvals import (
    ENDING,
    Recorder,
    ScriptedLLM,
    _agent,
    _answer,
    _calls,
)

PLAIN = "Path '{path}' is outside allowed workspace root"


class Asker:
    """The person: records each request and gives the next answer."""

    def __init__(self, *answers: ApprovalDecision) -> None:
        self.answers = list(answers)
        self.requests: list[ApprovalRequest] = []

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.answers.pop(0)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("HOME", str(folder))
    return Path(os.path.realpath(folder))


@pytest.fixture
def root(home: Path) -> Path:
    path = home / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def career(home: Path) -> Path:
    path = home / "developer" / "career"
    path.mkdir(parents=True)
    (path / "cv.md").write_text("Engineer, 2020-2026\n")
    return path


def _read(path: Path, call_id: str = "call_r") -> Any:
    return _calls(("file_read", {"path": str(path)}, call_id))


def _files_agent(llm: ScriptedLLM, root: Path) -> Any:
    agent, _ = _agent(
        llm,
        builtins=frozenset({"file_read", "file_write", "list_files"}),
        workspace_root=str(root),
    )
    return agent


async def test_an_outside_read_asks_and_runs_with_no_new_model_turn(
    root: Path, career: Path
) -> None:
    llm = ScriptedLLM([_read(career / "cv.md"), _answer("Your CV says engineer.")])
    agent = _files_agent(llm, root)
    asker = Asker(ApprovalDecision(True))
    recorder = Recorder()

    result = await agent.run_task(
        "look at my career folder", on_event=recorder, approval=ChatPolicy(asker)
    )

    assert result.status == "completed"
    [request] = asker.requests
    assert request.outside is not None
    assert (request.outside.folder, request.outside.access) == (str(career), "read")
    assert request.tier == "read"
    # The model's next request carries the file, not a refusal.
    assert any("Engineer, 2020-2026" in text for text in llm.tool_messages(1))
    [finished] = recorder.of("tool.finished")
    approval = finished.data["approval"]
    assert approval["allowed"] is True and approval["origin"] == "user"
    assert approval["folder"]["scope"] == "once"
    assert approval["folder"]["folder"] == str(career)


async def test_a_denial_gives_the_model_one_line_and_the_usual_denial(
    root: Path, career: Path
) -> None:
    llm = ScriptedLLM([_read(career / "cv.md"), _answer("ok")])
    agent = _files_agent(llm, root)

    await agent.run_task(
        "look", approval=ChatPolicy(Asker(ApprovalDecision(False, "private")))
    )

    [text] = llm.tool_messages(1)
    assert (
        f"Reading {career} outside the workspace needs the user's OK; they were asked."
        in text
    )
    assert f"The user denied this call: private. {ENDING}" in text


async def test_this_chat_lets_a_second_read_run_unasked(
    root: Path, career: Path
) -> None:
    (career / "letter.md").write_text("Dear team")
    llm = ScriptedLLM(
        [
            _read(career / "cv.md", "c1"),
            _read(career / "letter.md", "c2"),
            _answer("done"),
        ]
    )
    agent = _files_agent(llm, root)
    asker = Asker(ApprovalDecision(True, remember=True))
    recorder = Recorder()

    await agent.run_task("read both", on_event=recorder, approval=ChatPolicy(asker))

    assert len(asker.requests) == 1
    scopes = [
        e.data["approval"]["folder"]["scope"] for e in recorder.of("tool.finished")
    ]
    assert scopes == ["chat", "chat"]
    assert recorder.of("tool.finished")[1].data["approval"]["origin"] == "session"


async def test_a_write_after_a_read_grant_still_asks(root: Path, career: Path) -> None:
    llm = ScriptedLLM(
        [
            _read(career / "cv.md", "c1"),
            _calls(
                ("file_write", {"path": str(career / "cv.md"), "content": "x"}, "c2")
            ),
            _answer("done"),
        ]
    )
    agent = _files_agent(llm, root)
    asker = Asker(
        ApprovalDecision(True, remember=True), ApprovalDecision(False, "read only")
    )

    await agent.run_task("edit my cv", approval=ChatPolicy(asker))

    assert [r.outside.access for r in asker.requests if r.outside] == ["read", "write"]
    write = asker.requests[1]
    assert write.review is not None and write.review.verdict == "high"
    assert "outside_write" in write.review.categories
    assert (career / "cv.md").read_text() == "Engineer, 2020-2026\n"


async def test_auto_mode_still_asks_the_person(root: Path, career: Path) -> None:
    llm = ScriptedLLM([_read(career / "cv.md"), _answer("done")])
    agent = _files_agent(llm, root)
    asker = Asker(ApprovalDecision(True))

    await agent.run_task("look", approval=ChatPolicy(asker, "auto"))

    assert len(asker.requests) == 1
    assert any("Engineer" in text for text in llm.tool_messages(1))


async def test_auto_mode_never_allows_on_its_own(root: Path, career: Path) -> None:
    """Auto's own policy has no folder layer: the path is refused, not run."""
    llm = ScriptedLLM([_read(career / "cv.md"), _answer("done")])
    agent = _files_agent(llm, root)

    await agent.run_task("look", approval=AutoPolicy())

    [text] = llm.tool_messages(1)
    assert PLAIN.format(path=career / "cv.md") in text


async def test_a_task_nobody_owns_is_refused_as_before(
    root: Path, career: Path
) -> None:
    llm = ScriptedLLM([_read(career / "cv.md"), _answer("done")])
    agent = _files_agent(llm, root)
    asker = Asker()

    await agent.run_task("look", approval=RememberingPolicy(asker, folders=False))

    assert asker.requests == []
    [text] = llm.tool_messages(1)
    assert PLAIN.format(path=career / "cv.md") in text


async def test_a_turn_the_person_did_not_start_is_refused_as_before(
    root: Path, career: Path
) -> None:
    llm = ScriptedLLM([_read(career / "cv.md"), _answer("done")])
    agent = _files_agent(llm, root)
    asker = Asker()

    await agent.run_task(
        "a task's question", approval=ChatPolicy(asker), from_person=False
    )

    assert asker.requests == []
    [text] = llm.tool_messages(1)
    assert PLAIN.format(path=career / "cv.md") in text


async def test_a_never_offered_place_is_refused_with_the_reason(
    root: Path, home: Path
) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_ed25519").write_text("key")
    llm = ScriptedLLM([_read(home / ".ssh" / "id_ed25519"), _answer("done")])
    agent = _files_agent(llm, root)
    asker = Asker()

    await agent.run_task("read my key", approval=ChatPolicy(asker))

    assert asker.requests == []
    [text] = llm.tool_messages(1)
    assert "holds keys or passwords. melo never opens it" in text


async def test_always_survives_a_new_chat(root: Path, career: Path) -> None:
    first = ScriptedLLM([_read(career / "cv.md"), _answer("done")])
    await _files_agent(first, root).run_task(
        "look",
        approval=ChatPolicy(
            Asker(ApprovalDecision(True, remember=True, always=True)),
            saved=config.SavedFolders(),
        ),
    )
    assert config.load().workspace.extra_folders == ["~/developer/career"]

    second = ScriptedLLM([_read(career / "cv.md"), _answer("done")])
    asker = Asker()
    recorder = Recorder()
    await _files_agent(second, root).run_task(
        "look again",
        on_event=recorder,
        approval=ChatPolicy(asker, saved=config.SavedFolders()),
    )

    assert asker.requests == []
    [finished] = recorder.of("tool.finished")
    assert finished.data["approval"]["folder"]["scope"] == "always"
