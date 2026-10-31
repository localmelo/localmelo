"""Whether bringing in other assistants' notes was offered: one private file in the memory folder.

The first of the terminal, the console and the app to offer it writes the file, whatever the answer; the
offer is open only while memory is on and the file is missing.
"""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path

import pytest

from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.importer import offer

from .test_import import home, memory  # noqa: F401  (fixtures)


def test_the_first_answer_is_kept_in_a_private_file(tmp_path: Path) -> None:
    folder = str(tmp_path / "memory")
    assert not offer.offered(folder)
    offer.mark(folder, "declined")
    offer.mark(folder, "brought")
    assert offer.offered(folder)
    path = Path(folder) / offer.NAME
    assert json.loads(path.read_text())["answer"] == "declined"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(folder).stat().st_mode) == 0o700


def test_only_the_two_answers_are_recorded(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        offer.mark(str(tmp_path / "memory"), "maybe")
    assert not offer.offered(str(tmp_path / "memory"))


def test_two_answers_at_once_leave_one_whole_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record is created exclusively: of eight answers sent together, one is written, whole."""
    folder = str(tmp_path / "memory")
    created = []
    real_open = os.open

    def counting(path: str, flags: int, mode: int = 0o777) -> int:
        fd = real_open(path, flags, mode)
        created.append(path)
        return fd

    monkeypatch.setattr(offer.os, "open", counting)
    gate = threading.Barrier(8)

    def answer(which: str) -> None:
        gate.wait()
        offer.mark(folder, which)

    threads = [
        threading.Thread(target=answer, args=(("brought", "declined")[n % 2],))
        for n in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(created) == 1, "only the first answer created the file"
    record = json.loads((Path(folder) / offer.NAME).read_text())
    assert record["answer"] in offer.ANSWERS and isinstance(record["at"], float)


def test_the_offer_is_open_only_while_memory_is_on_and_it_was_never_made(
    tmp_path: Path,
) -> None:
    on = ChatMemory(lambda: None, directory=str(tmp_path / "a"))
    assert on.import_pending()
    on.import_offered("declined")
    assert not on.import_pending()
    cfg = Config()
    cfg.memory.across_chats = False
    off = ChatMemory(lambda: cfg, directory=str(tmp_path / "b"))
    assert not off.import_pending()


async def test_a_scan_leaves_the_offer_open_and_a_write_closes_it(
    memory: ChatMemory,  # noqa: F811
    home: Path,  # noqa: F811
) -> None:
    plan = await memory.import_scan(home=home)
    assert memory.import_pending(), "looking is not offering"
    gemini = next(s["id"] for s in plan.sources if s["harness"] == "gemini")
    await memory.import_apply(plan, [gemini])
    assert not memory.import_pending()
    assert json.loads((Path(memory.directory) / offer.NAME).read_text())["answer"] == (
        "brought"
    )


async def test_a_refused_write_does_not_close_the_offer(
    memory: ChatMemory,  # noqa: F811
    home: Path,  # noqa: F811
) -> None:
    plan = await memory.import_scan(home=home)
    with pytest.raises(ValueError):
        await memory.import_apply(plan, ["claude-nope"])
    assert memory.import_pending()


def test_a_folder_that_cannot_be_written_is_logged_not_raised(tmp_path: Path) -> None:
    """Notes already written must not turn into an error because the record could not be."""
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    offer.mark(str(blocker / "memory"), "brought")
    assert not offer.offered(str(blocker / "memory"))


def test_a_record_that_cannot_be_finished_leaves_none_so_the_offer_stays_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk after the file was created: no error, and no half-written file that reads as "made"."""
    folder = str(tmp_path / "memory")

    def full(*args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(offer.json, "dump", full)
    offer.mark(folder, "declined")
    assert not offer.offered(folder)
    assert not (Path(folder) / offer.NAME).exists()
    monkeypatch.undo()
    offer.mark(
        folder, "declined"
    )  # the disk has room again: the next answer is the first
    assert offer.offered(folder)


async def test_choosing_no_file_is_no_answer_to_the_offer(
    memory: ChatMemory,  # noqa: F811
    home: Path,  # noqa: F811
) -> None:
    plan = await memory.import_scan(home=home)
    done = await memory.import_apply(plan, [])
    assert done["written_ids"] == [] and memory.import_pending()
