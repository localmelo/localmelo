"""Records of memory across chats: written from the user's typed words only.

A statement or a remember request becomes one second-person record with its
topic and the user's quote; a secret is refused, not stored redacted; a new
value of the same subject supersedes the old one; a delete leaves no text in the
decision log; the file is a records file (schema 2) that a plain store refuses.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from localmelo.melo.memory.long import LongTermSchemaError
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.support.memory import ChatMemory, records
from localmelo.support.sessions.store import NewSession, SessionStore

HOLDER = "test-host:1:memory"


@pytest.mark.parametrize(
    ("typed", "texts", "topics"),
    [
        ("I prefer short answers.", ["You prefer short answers."], ["preferences"]),
        (
            "I prefer short answers; my group meeting is on Thursdays",
            ["Your group meeting is on Thursdays.", "You prefer short answers."],
            ["about", "preferences"],
        ),
        ("Call me Ray.", ["You go by Ray."], ["about"]),
        ("I'm allergic to peanuts.", ["You are allergic to peanuts."], ["about"]),
        (
            "Remember that Sam has the lab keys.",
            ["Sam has the lab keys."],
            ["people"],
        ),
        (
            "We decided to use Qwen3-Embedding-0.6B for memory search.",
            ["You decided to use Qwen3-Embedding-0.6B for memory search."],
            ["decisions"],
        ),
        ("我习惯用简体中文。", ["你习惯用简体中文。"], ["preferences"]),
        ("我的组会是每周四。", ["你的组会是每周四。"], ["about"]),
        ("叫我小王", ["你希望被叫作小王。"], ["about"]),
        ("记住：我的车停在B2层。", ["你的车停在B2层。"], ["about"]),
    ],
)
def test_statements_in_english_and_chinese_become_second_person_records(
    typed: str, texts: list[str], topics: list[str]
) -> None:
    found, refused = records.candidates(typed)
    assert not refused
    assert sorted(c.text for c in found) == sorted(texts)
    by_text = {c.text: c.topic for c in found}
    assert [by_text[t] for t in texts] == topics
    for cand in found:
        assert cand.quote in typed and len(cand.text) <= records.MAX_RECORD_CHARS


@pytest.mark.parametrize(
    "typed",
    [
        "hello",
        "When is my group meeting?",
        "Do you remember my name?",
        "Remember to buy milk.",
        "My task is to fix the build.",
        "Read notes.md and tell me what my plan is.",
        "x" * 2_100 + " I prefer tea.",
    ],
)
def test_questions_to_dos_meta_and_long_pastes_are_not_records(typed: str) -> None:
    assert records.candidates(typed) == ([], False)


@pytest.mark.parametrize(
    "typed",
    [
        "Remember that my API key is sk-abcdefghijklmnopqrstuvwxyz123456.",
        "My wifi password is hunter2-lighthouse.",
        "我的密码是 abc12345。",
        "My card number is 4111 1111 1111 1111.",
        "My SSN is 123-45-6789.",
        "My verification code is 482913.",
        "My token is bearer abcdefghijklmnopqrstuvwxyz.",
    ],
)
def test_secrets_are_refused_and_never_written(typed: str) -> None:
    found, refused = records.candidates(typed)
    assert found == [] and refused


def test_a_secret_beside_a_statement_refuses_only_the_secret() -> None:
    found, refused = records.candidates("I prefer tea. My PIN is 4821.")
    assert refused and [c.text for c in found] == ["You prefer tea."]
    assert not records.sensitive("My phone number is 412-555-0199.")
    assert not records.sensitive("Order 4111 1111 1111 1112 shipped.")  # fails Luhn


async def _session(store: SessionStore, title: str = "Planning") -> str:
    turn = await store.begin_turn(
        None, title, holder=HOLDER, title=title, new_session=NewSession()
    )
    return turn.session_id


async def _say(memory: ChatMemory, store: SessionStore, session_id: str, text: str):  # type: ignore[no-untyped-def]
    turns = await store.list_turns(session_id)
    return await memory.before_turn(store, turns[-1], text)


@pytest.fixture()
async def store(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture()
async def memory(tmp_path: Path):  # type: ignore[no-untyped-def]
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        yield memory
    finally:
        await memory.aclose()


async def test_a_turn_writes_its_records_with_source_and_says_so(
    memory: ChatMemory, store: SessionStore
) -> None:
    sid = await _session(store)
    found = await _say(
        memory, store, sid, "I prefer short answers; my wifi password is x9"
    )
    assert [w.text for w in found.remembered] == ["You prefer short answers."]
    assert found.refused and found.block == ""
    assert found.event() == {
        "used": [],
        "remembered": [
            {
                "id": found.remembered[0].id,
                "text": "You prefer short answers.",
                "topic": "preferences",
            }
        ],
        "refused": "it looked like a password or key",
    }
    (entry,) = await memory.records()
    assert entry.session_id == sid and entry.source == "user"
    assert entry.admitted_by == "user_statement" and entry.kind == "profile"
    assert entry.provenance["chat"] == "chat “Planning”"
    assert entry.provenance["quote"] == "I prefer short answers"
    assert entry.provenance["turn"] == 1


async def test_saying_it_again_keeps_one_record(
    memory: ChatMemory, store: SessionStore
) -> None:
    sid = await _session(store)
    await _say(memory, store, sid, "I prefer short answers.")
    again = await _say(memory, store, sid, "I prefer short answers!")
    assert again.remembered == () and len(await memory.records()) == 1


@pytest.mark.parametrize(
    ("first", "second", "kept"),
    [
        ("Call me R.", "Call me Ray.", "You go by Ray."),
        (
            "My group meeting is on Thursdays.",
            "My group meeting is now on Fridays.",
            "Your group meeting is now on Fridays.",
        ),
        (
            "I live in Boston.",
            "I live in Pittsburgh now.",
            "You live in Pittsburgh now.",
        ),
    ],
)
async def test_a_new_value_supersedes_the_old_and_undo_brings_it_back(
    memory: ChatMemory, store: SessionStore, first: str, second: str, kept: str
) -> None:
    sid = await _session(store)
    (old,) = (await _say(memory, store, sid, first)).remembered
    (new,) = (await _say(memory, store, sid, second)).remembered
    assert new.replaced == old.id
    assert [e.text for e in await memory.records()] == [kept]
    assert await memory.delete_record(new.id, undo=True)
    assert [e.id for e in await memory.records()] == [old.id]


async def test_different_preferences_are_both_kept(
    memory: ChatMemory, store: SessionStore
) -> None:
    sid = await _session(store)
    await _say(memory, store, sid, "I prefer Python.")
    await _say(memory, store, sid, "I prefer dark mode.")
    assert len(await memory.records()) == 2


async def test_a_delete_blanks_the_decision_text_and_logs_the_delete(
    memory: ChatMemory, store: SessionStore, tmp_path: Path
) -> None:
    sid = await _session(store)
    (note,) = (await _say(memory, store, sid, "My dentist is Dr. Lee.")).remembered
    assert await memory.delete_record(note.id)
    assert not await memory.delete_record(note.id)
    path = tmp_path / "memory" / "records.db"
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM long_term").fetchone() == (0,)
        rows = conn.execute(
            "SELECT action, text, text_sha256 FROM decisions"
        ).fetchall()
    assert rows == [("NEW", None, ""), ("DELETE", None, "")]
    assert b"Dr. Lee" not in path.read_bytes()


async def test_two_writers_at_once_keep_every_record(
    store: SessionStore, tmp_path: Path
) -> None:
    folder = str(tmp_path / "memory")
    first, second = (ChatMemory(lambda: None, directory=folder) for _ in range(2))
    try:
        sid = await _session(store)
        things = ("bike", "car", "kettle", "desk", "scarf", "phone case")
        colors = ("red", "blue", "green", "black", "yellow", "white")
        said = [f"My {t} is {c}." for t, c in zip(things, colors, strict=True)]
        said += [
            f"我的{t}是{c}的。"
            for t, c in zip("书伞杯灯衣盘", "红蓝绿黑黄白", strict=True)
        ]
        results = await asyncio.gather(
            *(
                _say(first if n % 2 else second, store, sid, text)
                for n, text in enumerate(said)
            )
        )
        assert all(len(r.remembered) == 1 for r in results)
        assert len(await first.records()) == len(said)
    finally:
        await first.aclose()
        await second.aclose()


async def test_the_records_file_is_schema_two_and_a_plain_store_refuses_it(
    memory: ChatMemory, store: SessionStore, tmp_path: Path
) -> None:
    sid = await _session(store)
    await _say(memory, store, sid, "I prefer tea.")
    path = tmp_path / "memory" / "records.db"
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (2,)
        row = conn.execute("SELECT topic, quote, edited_at FROM long_term").fetchone()
    assert row == ("preferences", "I prefer tea.", 0)
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert oct((tmp_path / "memory").stat().st_mode & 0o777) == "0o700"
    plain = SqliteLongTerm(path)
    with pytest.raises(LongTermSchemaError, match="version 2"):
        await plain.active_entries()
    await plain.aclose()


async def test_a_version_one_file_becomes_a_records_file(tmp_path: Path) -> None:
    path = tmp_path / "records.db"
    plain = SqliteLongTerm(path)
    await plain.count_entries()
    await plain.aclose()
    upgraded = SqliteLongTerm(path, records=True)
    assert await upgraded.count_entries() == 0
    await upgraded.aclose()
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (2,)
        names = {r[1] for r in conn.execute("PRAGMA table_info(long_term)")}
    assert {"topic", "quote", "edited_at"} <= names


async def test_memory_off_writes_and_recalls_nothing(
    store: SessionStore, tmp_path: Path
) -> None:
    from localmelo.support.config import Config

    cfg = Config()
    cfg.memory.across_chats = False
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    try:
        sid = await _session(store)
        found = await _say(memory, store, sid, "I prefer tea.")
        assert found.event() is None and not (tmp_path / "memory").exists()
        assert await memory.pinned_quotes() == ""
    finally:
        await memory.aclose()
