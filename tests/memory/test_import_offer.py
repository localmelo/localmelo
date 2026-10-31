"""Whether bringing in other assistants' notes was offered: one private file in the memory folder.

The first of the terminal, the console and the app to offer it writes the file, whatever the answer; the
offer is open only while memory is on and the file is missing.
"""

from __future__ import annotations

import json
import stat
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
