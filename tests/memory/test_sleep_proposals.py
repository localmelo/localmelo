"""Sleep skips facts already held and explains every model operation, without servers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localmelo.support.memory import ChatMemory, records, sleep
from localmelo.support.sessions.store import SessionStore
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding

from . import test_sleep as rules
from .test_sleep_model import Scripted, op

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
    assert "sleep lock is held" in report.aborted or "user is active" in report.aborted
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


@pytest.mark.parametrize("variant", ["rules", "model", "model+tags"])
async def test_existing_train_fact_is_not_recovered_as_another_note(
    memory: ChatMemory, variant: str
) -> None:
    old = await rules.put(memory, TRAIN, vector=[1.0, 0.0])
    source = sleep.SleepTurn(
        "s",
        1,
        rules.LONG_AGO + 10,
        rules.LONG_AGO + 11,
        "My usual train to the office is the 7:40.",
    )
    provider = Scripted([[op("ADD", source.typed, PARAPHRASE)]])
    calls: list[str] = []

    async def embed_note(text: str) -> tuple[list[float], str]:
        calls.append(text)
        return [1.0, 0.0], "m"

    report = await sleep.run(
        *memory.stores(),
        [source],
        variant=variant,
        provider=provider,
        embed_note=embed_note,
    )
    assert calls == [PARAPHRASE]  # cached across rules and the model's ADD
    assert not report.changes
    assert [(n.id, n.text) for n in await memory.records()] == [(old, TRAIN)]
    if variant != "rules":
        assert report.to_json()["proposals"][0]["verdict"] == "refused:decide_mismatch"


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


async def test_selection_is_actual_request_order_including_rewrites(
    memory: ChatMemory,
) -> None:
    old = await rules.put(memory, "Her office is on Pine Street.", session="s", turn=1)
    source = sleep.SleepTurn("s", 1, 3, 4, "Dr. Ada's office is on Pine Street.")
    later = sleep.SleepTurn("s", 2, 5, 6, "Unrelated question?")
    provider = Scripted(
        [
            [op("NOOP", "")],
            [op("NOOP", "")],
            [op("UPDATE", source.typed, source.typed, old)],
        ]
    )
    first = await sleep.run(
        *memory.stores(),
        [source, later],
        variant="model",
        provider=provider,
        now=10,
        budget=2,
    )
    assert first.to_json()["selected"] == [["s", 1], ["s", 2]]
    assert first.selected == first.processed
    second = await sleep.run(
        *memory.stores(),
        [source, later],
        variant="model",
        provider=provider,
        now=10,
        until=6,
        budget=1,
    )
    assert second.selected == [["s", 1]] and not second.processed
    change = second.changes[0]
    assert change.op == "O4"
    assert change.proposal == second.to_json()["proposals"][0]
    assert change.proposal is not None
    assert change.proposal["op"] == "UPDATE" and change.verdict == "applied"


async def test_each_model_op_has_a_bounded_verdict_and_applied_change_label(
    memory: ChatMemory,
) -> None:
    source = sleep.SleepTurn("s", 1, 3, 4, "I enjoy kitesurfing. " + "x" * 240)
    provider = Scripted(
        [
            [
                op("NOOP", source.typed, "z" * 250),
                op("ADD", "not in typed", "You enjoy kitesurfing."),
                op("ADD", "I enjoy kitesurfing.", "You enjoy kitesurfing."),
            ]
        ]
    )
    report = await sleep.run(
        *memory.stores(), [source], variant="model", provider=provider, now=10
    )
    rows = report.to_json()["proposals"]
    assert [r["verdict"] for r in rows] == [
        "noop",
        "refused:quote_not_in_text",
        "applied",
    ]
    assert all(
        set(r)
        == {
            "session_id",
            "turn",
            "op",
            "note_id",
            "target_text",
            "text",
            "quote",
            "verdict",
            "guard",
        }
        for r in rows
    )
    assert all(r["session_id"] == "s" and r["turn"] == 1 for r in rows)
    assert all(len(r["text"]) <= 200 and len(r["quote"]) <= 200 for r in rows)
    # Only an ADD that passed the existing-fact check carries its numbers.
    assert [r["guard"] is None for r in rows] == [True, True, False]
    assert report.refused == 1 and report.selected == [["s", 1]]
    (change,) = report.changes
    assert change.verdict == "applied" and change.proposal == rows[2]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("bad_op", "parse"),
        ("not_object", "parse"),
        ("bad_text", "parse"),
        ("quote", "quote_not_in_text"),
        ("secret", "sensitive"),
        ("unsupported", "unsupported"),
        ("owner", "wrong_owner"),
        ("stale", "stale"),
        ("target", "decide_mismatch"),
        ("newer", "not_newer"),
        ("edited", "edited"),
        ("negation", "no_negation"),
        ("term", "no_shared_term"),
    ],
)
async def test_specific_refusal_checks_are_reported(
    memory: ChatMemory, case: str, expected: str
) -> None:
    old = await rules.put(memory, "You are allergic to peanuts.")
    quote = "I'm allergic to peanuts."
    proposed: Any = op("ADD", quote, "You are allergic to peanuts.")
    turns: list[sleep.SleepTurn] = []
    if case == "bad_op":
        proposed["op"] = "BOGUS"
    elif case == "not_object":
        proposed = 42
    elif case == "bad_text":
        proposed["text"] = 42
    elif case == "quote":
        proposed["quote"] = "invented evidence"
    elif case == "secret":
        quote = "My password is hunter2."
        proposed = op("ADD", quote, "Your password is hunter2.")
    elif case == "unsupported":
        proposed["text"] = "You are allergic to walnuts."
    elif case == "owner":
        quote = "Priya's allergy is peanuts."
        proposed = op("ADD", quote, "Your allergy is peanuts.")
    elif case == "stale":
        turns = [
            sleep.SleepTurn(
                "s",
                2,
                rules.LONG_AGO + 30,
                rules.LONG_AGO + 31,
                "I'm no longer allergic to peanuts.",
            )
        ]
    elif case == "target":
        proposed = op("UPDATE", quote, "You are allergic to peanuts.", 9999)
    elif case in {"newer", "edited"}:
        if case == "edited":
            await memory.edit_record(old, "You are allergic to peanuts.")
        else:
            old = await rules.put(memory, "You are allergic to peanuts.", rules.SOON)
        quote = "I'm no longer allergic to peanuts."
        proposed = op("RETIRE", quote, note=old)
    elif case == "negation":
        proposed = op("RETIRE", quote, note=old)
    elif case == "term":
        quote = "Not relevant."
        proposed = op("RETIRE", quote, note=old)
    source = sleep.SleepTurn("s", 1, rules.LONG_AGO + 10, rules.LONG_AGO + 11, quote)
    # Keep the additional future seed out of O1, which otherwise merges equal subjects.
    engine = sleep._ModelPass(
        memory.stores()[0],
        sleep.SleepReport("test", "model", 0, 0),
        Scripted([[proposed]]),
        None,
        None,
        "",
        memory.stores()[1],
        12,
        turns=turns,
    )
    assert await engine.replay(source, await memory.records())
    report = engine.report
    assert not report.changes and report.refused == 1
    assert report.proposals[0]["verdict"] == f"refused:{expected}"
    assert report.selected == [["s", 1]]
    if case == "secret":
        assert "hunter2" not in str(report.to_json())


@pytest.mark.parametrize("reply", ["not JSON", {"op": "NOOP"}, [None]])
async def test_parse_failure_and_malformed_operations_are_visible(
    memory: ChatMemory, reply: Any
) -> None:
    source = sleep.SleepTurn("s", 1, 3, 4, "A question?")
    report = await sleep.run(
        *memory.stores(), [source], variant="model", provider=Scripted([reply]), now=10
    )
    assert report.selected == [["s", 1]] and report.refused == 1
    assert report.proposals[0]["verdict"] == "refused:parse"
    assert report.processed == ([] if not isinstance(reply, list) else [["s", 1]])


async def test_cap_does_not_hide_returned_operations(
    memory: ChatMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sleep, "MAX_CHANGES", 1)
    source = sleep.SleepTurn(
        "s", 1, 3, 4, "I'm allergic to peanuts. I enjoy kitesurfing."
    )
    report = await sleep.run(
        *memory.stores(),
        [source],
        variant="model",
        now=10,
        provider=Scripted(
            [
                [
                    op(
                        "ADD",
                        "I'm allergic to peanuts.",
                        "You are allergic to peanuts.",
                    ),
                    op("ADD", "I enjoy kitesurfing.", "You enjoy kitesurfing."),
                ]
            ]
        ),
    )
    assert report.capped and len(report.proposals) == 2
    assert [r["verdict"] for r in report.proposals] == ["applied", "noop"]


async def test_noop_diagnostics_do_not_echo_secrets_or_unbounded_model_fields(
    memory: ChatMemory,
) -> None:
    source = sleep.SleepTurn("s", 1, 3, 4, "A question?")
    report = await sleep.run(
        *memory.stores(),
        [source],
        variant="model",
        now=10,
        provider=Scripted(
            [[op("NOOP", "My password is hunter2.", "My password is hunter2.")]]
        ),
    )
    assert report.proposals[0]["verdict"] == "noop"
    assert "hunter2" not in str(report.to_json())


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
        *memory.stores(),
        [source],
        variant="model",
        provider=Scripted([[op("ADD", quote, new)]]),
        embed_note=embed_note,
    )
    assert await rules.texts(memory) == {old, new}
    assert [c.op for c in report.changes] == ["O3"]


@pytest.mark.parametrize("variant", ["rules", "model", "model+tags"])
async def test_lexical_restatement_keeps_the_richer_active_note(
    memory: ChatMemory, variant: str
) -> None:
    text = "Your dentist is Dr. Lin on Main Street."
    old = await rules.put(memory, text)
    source = sleep.SleepTurn(
        "s", 1, rules.LONG_AGO + 10, rules.LONG_AGO + 11, "My dentist is Dr. Lin."
    )
    report = await sleep.run(
        *memory.stores(),
        [source],
        variant=variant,
        provider=Scripted([[op("ADD", source.typed, "Your dentist is Dr. Lin.")]]),
    )
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
