"""Folder grants outside the workspace (#405): once, this chat, always, deny.

The policies of the surfaces are asked through ``decide_folder``. A read in
a folder allowed for the chat, or kept in ``[workspace] extra_folders``,
runs unasked; a write always asks and grants nothing; auto mode never
decides. HOME is a temporary folder, so "always" writes a temporary
``config.toml``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from localmelo.melo.contracts.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    AutoPolicy,
    find_folder_asker,
)
from localmelo.melo.contracts.folders import outside_access
from localmelo.support import config
from localmelo.support.sessions.approvals import (
    FOLDER_FOR_CHAT,
    FOLDER_SAVED,
    ApprovalLedger,
    ChatPolicy,
    OwnedTaskPolicy,
    RememberingPolicy,
)


class Asker:
    """An ask function that records each request and gives the next answer."""

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
def letters(home: Path) -> Path:
    path = home / "Documents" / "letters"
    path.mkdir(parents=True)
    (path / "cover-letter.md").write_text("letter")
    (path / "notes").mkdir()
    (path / "notes" / "a.md").write_text("a")
    return path


def _request(tool: str, path: Path) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=f"r-{tool}-{path.name}",
        run_id="run",
        call_id="call",
        tool_name=tool,
        arguments={"path": str(path)},
        preview=f"{tool} {path}",
        workspace_root=None,
        outside=outside_access(tool, str(path)),
    )


def _policy(asker: Asker) -> RememberingPolicy:
    return RememberingPolicy(asker, folders=True, saved=config.SavedFolders())


async def test_allow_once_covers_that_call_only(letters: Path) -> None:
    asker = Asker(ApprovalDecision(True), ApprovalDecision(False, "no"))
    policy = _policy(asker)

    first = await policy.decide_folder(
        _request("file_read", letters / "cover-letter.md")
    )
    second = await policy.decide_folder(
        _request("file_read", letters / "cover-letter.md")
    )

    assert first.allow and not second.allow
    assert len(asker.requests) == 2
    assert policy.allowed_folders == ()


async def test_this_chat_covers_later_reads_below_the_folder(letters: Path) -> None:
    asker = Asker(ApprovalDecision(True, remember=True))
    policy = _policy(asker)

    await policy.decide_folder(_request("file_read", letters / "cover-letter.md"))
    again = await policy.decide_folder(
        _request("file_read", letters / "notes" / "a.md")
    )
    listed = await policy.decide_folder(_request("list_files", letters / "notes"))

    assert (again.allow, again.origin, again.reason) == (
        True,
        "session",
        FOLDER_FOR_CHAT,
    )
    assert listed.allow
    assert len(asker.requests) == 1
    assert policy.allowed_folders == (str(letters),)


async def test_another_chat_asks_again(letters: Path) -> None:
    first = _policy(Asker(ApprovalDecision(True, remember=True)))
    await first.decide_folder(_request("file_read", letters / "cover-letter.md"))
    other_asker = Asker(ApprovalDecision(True))
    other = _policy(other_asker)

    await other.decide_folder(_request("file_read", letters / "cover-letter.md"))

    assert len(other_asker.requests) == 1


async def test_a_grant_does_not_cover_a_sibling_with_a_longer_name(
    letters: Path,
) -> None:
    sibling = letters.parent / "letters-old"
    sibling.mkdir()
    (sibling / "x.md").write_text("x")
    asker = Asker(ApprovalDecision(True, remember=True), ApprovalDecision(True))
    policy = _policy(asker)

    await policy.decide_folder(_request("file_read", letters / "cover-letter.md"))
    await policy.decide_folder(_request("file_read", sibling / "x.md"))

    assert len(asker.requests) == 2


async def test_always_writes_the_setting_and_a_restart_still_allows(
    letters: Path, home: Path
) -> None:
    asker = Asker(ApprovalDecision(True, remember=True, always=True))
    policy = _policy(asker)

    await policy.decide_folder(_request("file_read", letters / "cover-letter.md"))

    saved = config.load().workspace.extra_folders
    assert saved == ["~/Documents/letters"]
    text = Path(config.config_path()).read_text()
    assert '[workspace]\nextra_folders = ["~/Documents/letters"]' in text
    # A new policy, as after a restart, reads the file.
    fresh = _policy(Asker())
    decision = await fresh.decide_folder(
        _request("file_read", letters / "notes" / "a.md")
    )
    assert (decision.allow, decision.origin, decision.reason) == (
        True,
        "policy",
        FOLDER_SAVED,
    )


async def test_removing_the_folder_from_settings_asks_again(letters: Path) -> None:
    await _policy(
        Asker(ApprovalDecision(True, remember=True, always=True))
    ).decide_folder(_request("file_read", letters / "cover-letter.md"))
    cfg = config.load()
    cfg.workspace.extra_folders = []
    config.save(cfg)
    asker = Asker(ApprovalDecision(False, "no"))

    decision = await _policy(asker).decide_folder(
        _request("file_read", letters / "cover-letter.md")
    )

    assert not decision.allow
    assert len(asker.requests) == 1
    assert "[workspace]" not in Path(config.config_path()).read_text()


async def test_deny_reaches_the_caller_and_grants_nothing(letters: Path) -> None:
    policy = _policy(Asker(ApprovalDecision(False, "not that folder")))

    decision = await policy.decide_folder(
        _request("file_read", letters / "cover-letter.md")
    )

    assert (decision.allow, decision.reason) == (False, "not that folder")
    assert policy.allowed_folders == ()


async def test_a_write_after_a_read_grant_still_asks_and_grants_nothing(
    letters: Path,
) -> None:
    asker = Asker(
        ApprovalDecision(True, remember=True, always=True),
        ApprovalDecision(True, remember=True, always=True),
        ApprovalDecision(True),
    )
    policy = _policy(asker)
    await policy.decide_folder(_request("file_read", letters / "cover-letter.md"))

    write = await policy.decide_folder(_request("file_write", letters / "new.md"))
    again = await policy.decide_folder(_request("file_write", letters / "new.md"))

    assert write.allow and not write.remember and not write.always
    assert again.allow
    assert len(asker.requests) == 3, "every write asks"
    assert config.load().workspace.extra_folders == ["~/Documents/letters"]


async def test_a_file_directly_in_home_is_allowed_once_and_never_kept(
    home: Path,
) -> None:
    (home / "todo.txt").write_text("x")
    asker = Asker(
        ApprovalDecision(True, remember=True, always=True), ApprovalDecision(True)
    )
    policy = _policy(asker)

    first = await policy.decide_folder(_request("file_read", home / "todo.txt"))
    await policy.decide_folder(_request("file_read", home / "todo.txt"))

    assert first.allow and not first.remember and not first.always
    assert policy.allowed_folders == ()
    assert config.load().workspace.extra_folders == []
    assert len(asker.requests) == 2


async def test_forget_takes_back_folder_grants(letters: Path) -> None:
    asker = Asker(ApprovalDecision(True, remember=True), ApprovalDecision(True))
    policy = _policy(asker)
    await policy.decide_folder(_request("file_read", letters / "cover-letter.md"))

    assert policy.forget_folder(str(letters))
    assert not policy.forget_folder(str(letters))
    await policy.decide_folder(_request("file_read", letters / "cover-letter.md"))
    assert len(asker.requests) == 2

    policy._folders.append(str(letters))
    policy.forget()
    assert policy.allowed_folders == ()


async def test_a_tool_grant_never_covers_a_folder(letters: Path) -> None:
    asker = Asker(ApprovalDecision(True))
    policy = _policy(asker)
    policy._granted.add("file_write")

    via_decide = await policy.decide(_request("file_write", letters / "x.md"))

    assert not via_decide.allow
    assert asker.requests == []


async def test_auto_mode_asks_the_person_about_a_folder(letters: Path) -> None:
    asker = Asker(ApprovalDecision(True))
    chat = ChatPolicy(asker, "auto", saved=config.SavedFolders())

    assert find_folder_asker(chat) is chat
    decision = await chat.decide_folder(
        _request("file_read", letters / "cover-letter.md")
    )

    assert decision.allow and decision.origin == "user"
    assert len(asker.requests) == 1


async def test_policies_that_cannot_ask_are_never_found(letters: Path) -> None:
    assert find_folder_asker(AutoPolicy()) is None
    assert find_folder_asker(RememberingPolicy(Asker())) is None
    # A task the person did not start keeps folders=False.
    assert find_folder_asker(RememberingPolicy(Asker(), folders=False)) is None


async def test_a_task_uses_its_chats_folder_grants_and_asks_through_its_asker(
    letters: Path,
) -> None:
    ledger = ApprovalLedger()
    chat_grants = {"chat-1": (str(letters),)}
    ledger.folders_of = lambda chat: chat_grants.get(chat, ())
    task_asker = Asker(ApprovalDecision(True))
    asking = RememberingPolicy(task_asker, folders=True)

    async def owner() -> str | None:
        return "chat-1"

    task = OwnedTaskPolicy(
        AutoPolicy(task=True), ledger, "task-1", owner, folders=asking
    )

    assert task.asks_folders and find_folder_asker(task) is task
    read = await task.decide_folder(_request("file_read", letters / "cover-letter.md"))
    write = await task.decide_folder(_request("file_write", letters / "new.md"))

    assert (read.allow, read.origin) == (True, "session")
    assert write.allow and len(task_asker.requests) == 1


async def test_a_task_without_an_asker_cannot_ask() -> None:
    ledger = ApprovalLedger()

    async def owner() -> str | None:
        return None

    task = OwnedTaskPolicy(
        RememberingPolicy(Asker(), folders=False), ledger, "task-1", owner
    )
    assert not task.asks_folders
    assert find_folder_asker(task) is None


async def test_the_ledger_never_keeps_an_outside_call(letters: Path) -> None:
    ledger = ApprovalLedger()
    request = _request("file_write", letters / "x.md")

    assert ledger.record("chat-1", request) is None
    assert ledger.match("chat-1", request) is None


async def test_a_failed_save_still_allows_the_call(
    letters: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(folder: str) -> bool:
        raise config.ConfigError("disk full")

    saved = config.SavedFolders()
    monkeypatch.setattr(saved, "add", broken)
    policy = RememberingPolicy(
        Asker(ApprovalDecision(True, remember=True, always=True)),
        folders=True,
        saved=saved,
    )

    decision = await policy.decide_folder(
        _request("file_read", letters / "cover-letter.md")
    )

    assert decision.allow
    assert policy.allowed_folders == (str(letters),), "kept for the chat at least"


def test_the_decision_rejects_always_on_a_denial() -> None:
    with pytest.raises(ValueError, match="always needs allow=True"):
        ApprovalDecision(False, always=True)
    with pytest.raises(TypeError):
        ApprovalDecision(True, always="yes")  # type: ignore[arg-type]
