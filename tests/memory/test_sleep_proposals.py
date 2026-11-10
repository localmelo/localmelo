"""Sleep skips facts already held, with a stub embedding model and no server."""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.support.memory import ChatMemory, records, sleep
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding

from . import test_sleep as rules

memory = rules.memory
chats = rules.chats

TRAIN = "You take the 7:40 train to work."
PARAPHRASE = "Your usual train to the office is the 7:40."


async def test_chat_memory_wires_the_configured_fact_embedding(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await rules.put(memory, TRAIN, vector=[1.0, 0.0])
    await rules.chat(chats, "My usual train to the office is the 7:40.")
    provider = FakeEmbedding("m")

    async def embed(texts: list[str]) -> list[list[float]]:
        provider.calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(provider, "embed", embed)
    monkeypatch.setattr(memory, "embedding", lambda: (provider, "m"))
    report = await memory.sleep("rules", chats=chats)
    assert not report.changes and not report.aborted
    assert provider.calls == [[PARAPHRASE]]
    assert await rules.texts(memory) == {TRAIN}


@pytest.mark.parametrize("why", ["lock", "activity"])
async def test_fact_embedding_respects_lock_and_activity(
    memory: ChatMemory, tmp_path: Path, why: str
) -> None:
    await rules.put(memory, TRAIN, vector=[1.0, 0.0])
    lock = tmp_path / "sleep.lock"
    if why == "lock":
        lock.write_text("held")

    async def active() -> bool:
        return why == "activity"

    async def forbidden(text: str) -> tuple[list[float], str]:
        pytest.fail("a stopped run must not embed a candidate")

    source = sleep.SleepTurn("s", 1, 3, 4, "My usual train to the office is the 7:40.")
    report = await sleep.run(
        *memory.stores(),
        [source],
        variant="rules",
        now=10,
        embed_note=forbidden,
        lock=str(lock),
        is_active=active,
    )
    assert (
        report.aborted
        == {"lock": "stopped: gpu_lock", "activity": "stopped: user_active"}[why]
    )
    assert not report.changes and await memory.stores()[1].meta("sleep_until") == 0


async def test_low_cosine_does_not_fall_back_to_words(memory: ChatMemory) -> None:
    await rules.put(
        memory, "Your dentist is Dr. Lin on Main Street.", vector=[1.0, 0.0]
    )

    async def embed(text: str) -> tuple[list[float], str]:
        return [0.0, 1.0], "m"

    assert not await sleep._FactGuard(embed).held(
        "Your dentist is Dr. Lin.", await memory.records()
    )


async def test_embedding_deadline_uses_word_fallback(
    memory: ChatMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    await rules.put(
        memory, "Your dentist is Dr. Lin on Main Street.", vector=[1.0, 0.0]
    )
    monkeypatch.setattr(sleep, "FACT_TIMEOUT", 0.001)

    async def embed(text: str) -> tuple[list[float], str]:
        await asyncio.Event().wait()
        return [1.0, 0.0], "m"

    assert await sleep._FactGuard(embed).held(
        "Your dentist is Dr. Lin.", await memory.records()
    )


def test_value_terms_reuses_grounding_values() -> None:
    text = (
        'Your appointment with Priya is Monday at 7:40 in room 3. "Main Street". 周四'
    )
    values = records.value_terms(text)
    assert {"priya", "monday", "7", "40", "3", "main", "street"} <= values
    assert any(not term.isascii() for term in values)
    assert "appointment" not in values


async def test_existing_train_fact_is_not_recovered_as_another_note(
    memory: ChatMemory,
) -> None:
    old = await rules.put(memory, TRAIN, vector=[1.0, 0.0])
    source = sleep.SleepTurn(
        "s",
        1,
        rules.LONG_AGO + 10,
        rules.LONG_AGO + 11,
        "My usual train to the office is the 7:40.",
    )
    calls: list[str] = []

    async def embed_note(text: str) -> tuple[list[float], str]:
        calls.append(text)
        return [1.0, 0.0], "m"

    report = await sleep.run(
        *memory.stores(), [source], variant="rules", embed_note=embed_note
    )
    assert calls == [PARAPHRASE]
    assert not report.changes
    assert [(n.id, n.text) for n in await memory.records()] == [(old, TRAIN)]


@pytest.mark.parametrize(
    "old",
    [
        "You take the 8:40 train to work.",
        "Dr. Ada takes the 7:40 train to work.",
        "You take the 7:40 train to the airport.",
    ],
)
async def test_train_guard_does_not_conflate_other_values_owners_or_destinations(
    memory: ChatMemory, old: str
) -> None:
    await rules.put(memory, old)
    source = sleep.SleepTurn(
        "s",
        1,
        rules.LONG_AGO + 10,
        rules.LONG_AGO + 11,
        "My usual train to the office is the 7:40.",
    )
    report = await sleep.run(*memory.stores(), [source], variant="rules")
    assert [c.op for c in report.changes] == ["O3"]
    assert await rules.texts(memory) == {old, PARAPHRASE}


@pytest.mark.parametrize(
    ("quote", "old", "new"),
    [
        (
            "My swim day is Monday.",
            "Your gym day is Monday.",
            "Your swim day is Monday.",
        ),
        (
            "I live on the 3rd floor.",
            "You live in Utrecht.",
            "You live on the 3rd floor.",
        ),
    ],
)
async def test_existing_value_does_not_conflate_other_subjects(
    memory: ChatMemory, quote: str, old: str, new: str
) -> None:
    await rules.put(memory, old, vector=[1.0, 0.0])
    source = sleep.SleepTurn("s", 1, rules.LONG_AGO + 10, rules.LONG_AGO + 11, quote)

    async def embed_note(text: str) -> tuple[list[float], str]:
        return [1.0, 0.0], "m"

    report = await sleep.run(
        *memory.stores(), [source], variant="rules", embed_note=embed_note
    )
    assert await rules.texts(memory) == {old, new}
    assert [c.op for c in report.changes] == ["O3"]


async def test_lexical_restatement_keeps_the_richer_active_note(
    memory: ChatMemory,
) -> None:
    text = "Your dentist is Dr. Lin on Main Street."
    old = await rules.put(memory, text)
    source = sleep.SleepTurn(
        "s", 1, rules.LONG_AGO + 10, rules.LONG_AGO + 11, "My dentist is Dr. Lin."
    )
    report = await sleep.run(*memory.stores(), [source], variant="rules")
    assert not report.changes
    assert [(n.id, n.text) for n in await memory.records()] == [(old, text)]


@pytest.mark.parametrize("failure", ["off", "timeout", "error", "model", "shape"])
async def test_word_guard_and_embedding_fallback_keep_lexical_duplicates_out(
    memory: ChatMemory, failure: str
) -> None:
    await rules.put(
        memory, "Your dentist is Dr. Lin on Main Street.", vector=[1.0, 0.0]
    )

    async def embed_note(text: str) -> tuple[list[float], str]:
        if failure == "timeout":
            raise TimeoutError("stub timeout")
        if failure == "error":
            raise RuntimeError("stub error")
        return (
            [1.0] if failure == "shape" else [1.0, 0.0],
            "other" if failure == "model" else "m",
        )

    guard = sleep._FactGuard(None if failure == "off" else embed_note)
    assert await guard.held("Your dentist is Dr. Lin.", await memory.records())
    # A weaker paraphrase need not be caught by the word-only fallback.
    assert not await guard.held(PARAPHRASE, await memory.records())


async def test_word_guard_does_not_lower_threshold_for_train_pair(
    memory: ChatMemory,
) -> None:
    await rules.put(memory, TRAIN)
    assert not await sleep._FactGuard().held(PARAPHRASE, await memory.records())
