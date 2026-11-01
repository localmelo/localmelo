"""Records of memory across chats: what the user asks to remember, kept by code.

A remember request ("remember that…", "记住…", "别忘了…", "帮我记一下…") becomes
one second-person record with its topic and the user's quote, without a model
call; a plain statement is not kept by rule (the model keeps it with
``remember``: test_chat_remember.py); a secret is refused, not stored redacted;
a new value of the same subject supersedes the old one; a delete leaves no text
in the decision log; the file is a records file (schema 2) that a plain store
refuses.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from localmelo.melo.memory.long import LongTermSchemaError
from localmelo.melo.memory.long.sqlite import SqliteLongTerm
from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory.recall import BLOCK_HEADER, NOT_KEPT_LINE
from localmelo.support.sessions.store import NewSession, SessionStore

HOLDER = "test-host:1:memory"


# The sentences the rules used to keep as statements: now the model's to keep
# (test_chat_remember.py keeps each with ``remember``).
STATEMENTS = [
    ("I prefer short answers.", "You prefer short answers.", "preferences"),
    ("Call me Ray.", "You go by Ray.", "about"),
    ("I'm allergic to peanuts.", "You are allergic to peanuts.", "about"),
    (
        "We decided to use Qwen3-Embedding-0.6B for memory search.",
        "You decided to use Qwen3-Embedding-0.6B for memory search.",
        "decisions",
    ),
    ("我习惯用简体中文。", "你习惯用简体中文。", "preferences"),
    ("我的组会是每周四。", "你的组会是每周四。", "about"),
    ("叫我小王", "你希望被叫作小王。", "about"),
    (
        "My reading group meets on Tuesdays.",
        "Your reading group meets on Tuesdays.",
        "about",
    ),
    ("Our team lives in Pittsburgh.", "Your team lives in Pittsburgh.", "about"),
    ("我的猫叫团子。", "你的猫叫团子。", "about"),
    ("我的导师姓林。", "你的导师姓林。", "about"),
    ("My piano teacher is Mr. Bellamy.", "Your piano teacher is Mr. Bellamy.", "about"),
    ("My vet is Dr. Yuki Tanaka.", "Your vet is Dr. Yuki Tanaka.", "about"),
    ("My advisor is J. Smith.", "Your advisor is J. Smith.", "about"),
]


@pytest.mark.parametrize(
    ("typed", "texts", "topics"),
    [
        (
            "Remember that Sam has the lab keys.",
            ["Sam has the lab keys."],
            ["people"],
        ),
        (
            "Remember that I prefer short answers; my group meeting is on Thursdays",
            ["You prefer short answers."],
            ["preferences"],
        ),
        (
            "Keep in mind that our team lives in Pittsburgh.",
            ["Your team lives in Pittsburgh."],
            ["about"],
        ),
        (
            "Remember that we decided to use Qwen3-Embedding-0.6B.",
            ["You decided to use Qwen3-Embedding-0.6B."],
            ["decisions"],
        ),
        ("记住：我的车停在B2层。", ["你的车停在B2层。"], ["about"]),
        ("别忘了我的组会是每周四。", ["你的组会是每周四。"], ["about"]),
        ("帮我记一下：我的车停在B2层。", ["你的车停在B2层。"], ["about"]),
        ("记住：叫我小王", ["你希望被叫作小王。"], ["about"]),
        (
            "Remember that my piano teacher is Mr. Bellamy.",
            ["Your piano teacher is Mr. Bellamy."],
            ["about"],
        ),
    ],
)
def test_a_request_to_remember_becomes_a_second_person_record_in_english_and_chinese(
    typed: str, texts: list[str], topics: list[str]
) -> None:
    found, refused = records.candidates(typed)
    assert not refused
    assert [c.text for c in found] == texts
    assert [c.topic for c in found] == topics
    for cand in found:
        assert cand.quote in typed and len(cand.text) <= records.MAX_RECORD_CHARS
        assert cand.admitted_by == "user_request"


@pytest.mark.parametrize(
    ("sentence", "topic"),
    [
        # A rule or a fact about the work is about the user's setup, not a person.
        ("Remember: tests run with -n auto", "about"),
        ("记住：开发的代码一律用英文", "about"),
        ("Remember to use uv", "about"),
        ("Remember that Python's GIL is going away", "about"),
        ("Remember Friday is review day", "about"),
        ("Remember Everything is in ~/Developer", "about"),
        ("Remember Tests run with -n auto", "about"),
        # A fact about a person, with the marker or without it.
        ("Remember that Marco's studio is on Main St", "people"),
        ("Marco's studio is on Main St", "people"),
        ("Remember Marco lives in Berlin", "people"),
        ("Please remember that Dr. Tanaka works at the clinic", "people"),
        ("Sam has the lab keys", "people"),
        ("记住马可住在杭州", "people"),
        ("记住：马可的工作室在中山路88号", "people"),
        ("记住：妈妈的生日是5月3日", "people"),
        ("记住：同事在上海", "people"),
        # Decisions, preferences and first-person facts keep their topic.
        ("Remember that we decided to use uv", "decisions"),
        ("Remember that I prefer tea", "preferences"),
        ("Remember that my sister Anna lives in Berlin", "about"),
        ("记住：我的车停在B2层", "about"),
    ],
)
def test_a_request_is_filed_under_people_only_when_its_fact_is_about_a_person(
    sentence: str, topic: str
) -> None:
    assert records.topic_of(sentence, request=True) == topic


def test_a_statement_is_never_filed_under_people_by_topic_of() -> None:
    assert records.topic_of("Marco lives in Berlin", request=False) == "about"
    assert records.topic_of("Marco's studio is on Main St", request=False) == "about"


@pytest.mark.parametrize(("typed", "note", "topic"), STATEMENTS)
def test_a_plain_statement_is_not_kept_by_rule(
    typed: str, note: str, topic: str
) -> None:
    assert records.candidates(typed) == ([], False)


@pytest.mark.parametrize(
    "typed",
    [
        "hello",
        "When is my group meeting?",
        "Do you remember my name?",
        "Remember to buy milk.",
        "My task is to fix the build.",
        "Does my reading group meet on Tuesdays?",
        "Summarize my reading group notes.",
        "My task meets the spec.",
        "我的问题叫什么名字？",
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
    found, refused = records.candidates("Remember that I prefer tea. My PIN is 4821.")
    assert refused and [c.text for c in found] == ["You prefer tea."]
    assert not records.sensitive("My phone number is 412-555-0199.")
    assert not records.sensitive("Order 4111 1111 1111 1112 shipped.")  # fails Luhn


@pytest.mark.parametrize(
    "text",
    [
        "My PIN is 4821.",
        "my pin is 4821",
        "bank pin: 4821",
        "my pin 4821",
        "Pin code 77",
    ],
)
def test_a_pin_is_a_secret(text: str) -> None:
    assert records.sensitive(text)


@pytest.mark.parametrize(
    "text",
    [
        "Pin the dependency versions in uv.lock.",
        "Pin a note to every chat.",
        "Spinning up",
    ],
)
def test_pin_as_a_verb_is_not_a_secret(text: str) -> None:
    assert not records.sensitive(text)


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
        memory,
        store,
        sid,
        "Remember that I prefer short answers; my wifi password is x9",
    )
    assert [w.text for w in found.remembered] == ["You prefer short answers."]
    assert found.refused
    # The block says what was kept and what was refused, never the secret itself.
    assert found.block == (
        f"\n\n{BLOCK_HEADER}\nmelo kept from this message: “I prefer short "
        f"answers” (saved; do not save it again or in a file)\n{NOT_KEPT_LINE}"
    )
    assert "x9" not in found.block
    data = found.event()
    assert data is not None
    data.pop("recall")  # its numbers are pinned in test_chat_retriever.py
    assert data == {
        "used": [],
        "remembered": [
            {
                "id": found.remembered[0].id,
                "text": "You prefer short answers.",
                "topic": "preferences",
            }
        ],
        "refused": "it looked like a password or key",
        "unsaved_claim": False,
    }
    (entry,) = await memory.records()
    assert entry.session_id == sid and entry.source == "user"
    assert entry.admitted_by == "user_request" and entry.kind == "profile"
    assert entry.provenance["chat"] == "chat “Planning”"
    assert entry.provenance["quote"] == "I prefer short answers"
    assert entry.provenance["turn"] == 1


async def test_saying_it_again_keeps_one_record(
    memory: ChatMemory, store: SessionStore
) -> None:
    sid = await _session(store)
    await _say(memory, store, sid, "Remember that I prefer short answers.")
    again = await _say(memory, store, sid, "Remember that I prefer short answers!")
    assert again.remembered == () and len(await memory.records()) == 1


@pytest.mark.parametrize(
    ("typed", "quote"),
    [
        ("记住：叫我小王", "叫我小王"),
        ("Remember that Sam has the lab keys.", "Sam has the lab keys."),
        ("别忘了我的猫叫团子。", "我的猫叫团子。"),
    ],
)
async def test_the_block_says_a_fact_was_kept_even_when_nothing_is_recalled(
    memory: ChatMemory, store: SessionStore, typed: str, quote: str
) -> None:
    sid = await _session(store)
    said = await _say(memory, store, sid, typed)
    assert said.remembered and said.used == () and said.recall is not None
    assert said.recall.used == 0
    # The user's words, as the recall lines quote them; a repeat says it again.
    line = f"melo kept from this message: “{quote}” (saved; do not save it again or in a file)"
    assert said.block == f"\n\n{BLOCK_HEADER}\n{line}"
    again = await _say(memory, store, sid, typed)
    assert again.remembered == () and line in again.block


async def test_a_question_or_a_lone_secret_carries_no_kept_line(
    memory: ChatMemory, store: SessionStore
) -> None:
    sid = await _session(store)
    asked = await _say(memory, store, sid, "What is my name?")
    assert asked.block == "" and asked.remembered == () and not asked.refused
    assert asked.recall is not None and asked.recall.block_chars == 0
    secret = await _say(memory, store, sid, "My wifi password is hunter2-lighthouse.")
    assert secret.refused and secret.remembered == ()
    assert secret.block == f"\n\n{BLOCK_HEADER}\n{NOT_KEPT_LINE}"
    assert "hunter2" not in secret.block and await memory.records() == []


@pytest.mark.parametrize(
    ("first", "second", "kept"),
    [
        ("记住：叫我小王", "记住：叫我老王", "你希望被叫作老王。"),
        (
            "Remember that my group meeting is on Thursdays.",
            "Remember that my group meeting is now on Fridays.",
            "Your group meeting is now on Fridays.",
        ),
        (
            "Remember that I live in Boston.",
            "Remember that I live in Pittsburgh now.",
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
    await _say(memory, store, sid, "Remember that I prefer Python.")
    await _say(memory, store, sid, "Remember that I prefer dark mode.")
    assert len(await memory.records()) == 2


async def test_a_delete_blanks_the_decision_text_and_logs_the_delete(
    memory: ChatMemory, store: SessionStore, tmp_path: Path
) -> None:
    sid = await _session(store)
    said = "Remember that my dentist is Dr. Lee."
    (note,) = (await _say(memory, store, sid, said)).remembered
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
        said = [
            f"Remember that my {t} is {c}." for t, c in zip(things, colors, strict=True)
        ]
        said += [
            f"记住：我的{t}是{c}的。"
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
    await _say(memory, store, sid, "Remember that I prefer tea.")
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
        found = await _say(memory, store, sid, "Remember that I prefer tea.")
        assert found.event() is None and not (tmp_path / "memory").exists()
        assert await memory.pinned_quotes() == ""
        # The model's tool keeps nothing either.
        said = await memory.remember("You prefer tea.", "preferences")
        assert said == "Not remembered: memory across chats is off in this chat."
        assert not (tmp_path / "memory").exists()
    finally:
        await memory.aclose()
