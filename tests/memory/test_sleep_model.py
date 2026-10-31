"""Model sleep is evidence checked, budgeted and undoable; no real providers."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory, sleep
from localmelo.support.sessions.store import SessionStore

from . import test_sleep as rules

memory = rules.memory
chats = rules.chats


class Scripted(BaseLLMProvider):
    def __init__(self, replies: list[Any]) -> None:
        self.replies = iter(replies)
        self.calls: list[list[Message]] = []
        self.bodies: list[dict[str, Any]] = []
        self.after: Callable[[], Awaitable[None]] | None = None

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        assert tools is None
        self.calls.append(messages)
        self.bodies.append(dict(self.extra_body_by_kind.get("retrieve", {})))
        if self.after is not None:
            await self.after()
        reply: Any = next(self.replies, [])
        return Message(
            "assistant", reply if isinstance(reply, str) else json.dumps(reply)
        )

    async def close(self) -> None:
        pass


def op(
    action: str, quote: str, text: str | None = None, note: int | None = None
) -> dict[str, Any]:
    return {"op": action, "note_id": note, "text": text, "quote": quote}


def use(monkeypatch: pytest.MonkeyPatch, replies: list[Any]) -> Scripted:
    provider = Scripted(replies)
    monkeypatch.setattr(ChatMemory, "chat_provider", lambda self: provider)
    return provider


async def test_budget_leftovers_resume_once_and_undo_keeps_processed_markers(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    turns = [await rules.chat(chats, f"I'm allergic to food{i}.") for i in range(5)]
    provider = use(
        monkeypatch,
        [
            [op("ADD", f"I'm allergic to food{i}.", f"You are allergic to food{i}.")]
            for i in range(5)
        ],
    )
    first = await memory.sleep("model", chats=chats, budget=2)
    assert first.processed == [[t.session_id, t.turn] for t in turns[:2]]
    frontier = await memory.stores()[1].meta("sleep_until")
    assert 0 < frontier < (turns[2].finished_at or 0)
    first_rows = rules.rows(memory)
    second = await memory.sleep("model", chats=chats, budget=3)
    assert second.processed == [[t.session_id, t.turn] for t in turns[2:]]
    assert len(provider.calls) == 5 and len(await memory.records()) == 5
    await memory.undo_sleep(second.run_id)
    assert rules.rows(memory) == first_rows
    third = await memory.sleep("model", chats=chats, budget=3)
    assert not third.processed and third.model_calls == 0 and not third.changes
    assert len(provider.calls) == 5


async def test_turns_older_than_fourteen_days_are_dropped(
    memory: ChatMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = use(monkeypatch, [[]])
    now = 20 * 86400.0
    turns = [
        sleep.SleepTurn("s", 1, 1, 2, "old"),
        sleep.SleepTurn("s", 2, now - 2, now - 1, "recent"),
    ]
    report = await sleep.run(
        *memory.stores(), turns, variant="model", provider=provider, now=now
    )
    assert report.processed == [["s", 2]] and report.model_calls == 1


async def test_older_backlog_cannot_revive_a_processed_retraction(
    memory: ChatMemory,
) -> None:
    old = await rules.put(memory, "You are allergic to peanuts.")
    turns = [
        sleep.SleepTurn("s", 1, 1, 2, "I'm allergic to peanuts."),
        sleep.SleepTurn("s", 2, 3, 4, "I'm no longer allergic to peanuts."),
    ]
    provider = Scripted(
        [
            [op("RETIRE", turns[1].typed, note=old)],
            [op("ADD", turns[0].typed, "You are allergic to peanuts.")],
        ]
    )
    # The older turn was pending when a newer retraction completed.
    turns = [
        replace(
            t,
            started_at=t.started_at + rules.LONG_AGO,
            finished_at=t.finished_at + rules.LONG_AGO,
        )
        for t in turns
    ]
    first = await sleep.run(
        *memory.stores(), turns[1:], variant="model", provider=provider, budget=1
    )
    assert [c.op for c in first.changes] == ["RETIRE"]
    second = await sleep.run(
        *memory.stores(), turns, variant="model", provider=provider, budget=1
    )
    assert not second.changes and second.refused == 1
    assert await rules.texts(memory) == set()


async def test_rules_cannot_recover_a_processed_retraction(
    memory: ChatMemory,
) -> None:
    old = await rules.put(memory, "Your manager is Tom Briggs.")
    turns = [
        sleep.SleepTurn(
            "s", 1, rules.LONG_AGO + 1, rules.LONG_AGO + 2, "My manager is Tom Briggs."
        ),
        sleep.SleepTurn(
            "s",
            2,
            rules.LONG_AGO + 3,
            rules.LONG_AGO + 4,
            "Tom Briggs is no longer my manager.",
        ),
    ]
    first = await sleep.run(
        *memory.stores(),
        turns[1:],
        variant="model",
        provider=Scripted([[op("RETIRE", turns[1].typed, note=old)]]),
    )
    assert [c.op for c in first.changes] == ["RETIRE"]
    second = await sleep.run(
        *memory.stores(), turns, variant="model", provider=Scripted([[]])
    )
    assert not second.changes and await rules.texts(memory) == set()


async def test_native_thinking_envelope_keeps_the_valid_answer(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

    quote = "I'm allergic to peanuts."
    await rules.chat(chats, quote)
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "thinking": "Check the lasting fact.",
                    "content": json.dumps(
                        [op("ADD", quote, "You are allergic to peanuts.")]
                    ),
                }
            },
        )

    provider = OllamaNativeChat(
        "http://ollama.test", "test-model", transport=httpx.MockTransport(handler)
    )
    monkeypatch.setattr(ChatMemory, "chat_provider", lambda self: provider)
    try:
        report = await memory.sleep("model", chats=chats)
        assert [c.op for c in report.changes] == ["ADD"]
        assert report.refused == 0 and requests[0]["think"] is True
        assert requests[0]["options"]["temperature"] == 0
        assert await rules.texts(memory) == {"You are allergic to peanuts."}
    finally:
        await provider.close()


async def test_o4_undo_is_not_replayed_but_budget_leftovers_resume(
    memory: ChatMemory,
) -> None:
    old = await rules.put(memory, "Her office is on Pine Street.", session="s", turn=1)
    source = sleep.SleepTurn("s", 1, 3, 4, "Dr. Ada's office is on Pine Street.")
    provider = Scripted(
        [
            [],
            [op("UPDATE", source.typed, "Dr. Ada's office is on Pine Street.", old)],
            [op("UPDATE", source.typed, "Dr. Ada's office is on Pine Street.", old)],
        ]
    )
    first = await sleep.run(
        *memory.stores(), [source], variant="model", provider=provider, budget=1, now=10
    )
    assert not first.changes and first.processed == [["s", 1]]
    second = await sleep.run(
        *memory.stores(), [source], variant="model", provider=provider, until=4, now=10
    )
    assert [c.op for c in second.changes] == ["O4"]
    await memory.undo_sleep(second.run_id)
    third = await sleep.run(
        *memory.stores(), [source], variant="model", provider=provider, until=4, now=10
    )
    assert third.model_calls == 0 and not third.changes
    assert await rules.texts(memory) == {"Her office is on Pine Street."}


async def test_retire_undo_restores_preexisting_edited_conflicts(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = await rules.put(memory, "Your manager is Tom Briggs.", edited=True)
    await rules.put(memory, "Your manager is Ada Smith.", edited=True)
    before = rules.rows(memory)
    quote = "Tom Briggs is no longer my manager."
    await rules.chat(chats, quote)
    use(monkeypatch, [[op("RETIRE", quote, note=old)]])
    report = await memory.sleep("model", chats=chats)
    assert [c.op for c in report.changes] == ["RETIRE"]
    result = await memory.undo_sleep(report.run_id)
    assert result is not None and result["restored"] == 1 and result["kept"] == 0
    assert rules.rows(memory) == before


@pytest.mark.parametrize("failure", ["unreadable", "timeout"])
async def test_failed_reply_retries_once_then_completes_with_parse_refusal(
    memory: ChatMemory,
    chats: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    await rules.chat(chats, "I'm allergic to peanuts.")
    provider = use(monkeypatch, ["not JSON", "<think>unclosed"])
    if failure == "timeout":

        async def timed_out() -> None:
            raise TimeoutError("stub timeout")

        provider.after = timed_out
    first = await memory.sleep("model", chats=chats)
    assert not first.processed
    second = await memory.sleep("model", chats=chats)
    assert second.model_calls == 1 and len(second.processed) == 1
    assert second.to_json()["refusals"][0]["reason"] == "parse"
    third = await memory.sleep("model", chats=chats)
    assert third.model_calls == 0 and len(provider.calls) == 2


@pytest.mark.parametrize("rewrite", [False, True])
async def test_retry_state_survives_reopen_and_allows_a_valid_second_reply(
    memory: ChatMemory, rewrite: bool
) -> None:
    quote = (
        "Dr. Ada's office is on Pine Street." if rewrite else "I'm allergic to peanuts."
    )
    note = (
        await rules.put(memory, "Her office is on Pine Street.", session="s", turn=1)
        if rewrite
        else None
    )
    source = sleep.SleepTurn("s", 1, 3, 4, quote)
    proposal = (
        op("UPDATE", quote, quote, note)
        if rewrite
        else op("ADD", quote, "You are allergic to peanuts.")
    )
    provider = Scripted(["not JSON", [proposal]])
    first = await sleep.run(
        *memory.stores(),
        [source],
        variant="model",
        provider=provider,
        until=4 if rewrite else 0,
        now=10,
    )
    assert not first.processed and not first.changes
    await memory.stores()[1].aclose()
    second = await sleep.run(
        *memory.stores(),
        [source],
        variant="model",
        provider=provider,
        until=4 if rewrite else 0,
        now=10,
    )
    assert [c.op for c in second.changes] == ["O4" if rewrite else "ADD"]
    assert second.refused == 0 and not second.to_json()["refusals"]
    third = await sleep.run(
        *memory.stores(),
        [source],
        variant="model",
        provider=provider,
        until=4 if rewrite else 0,
        now=10,
    )
    assert third.model_calls == 0


async def test_retire_undo_preserves_a_later_edit_to_a_preexisting_conflict(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = await rules.put(memory, "Your manager is Tom Briggs.", edited=True)
    other = await rules.put(memory, "Your manager is Ada Smith.", edited=True)
    quote = "Tom Briggs is no longer my manager."
    await rules.chat(chats, quote)
    use(monkeypatch, [[op("RETIRE", quote, note=old)]])
    report = await memory.sleep("model", chats=chats)
    await memory.edit_record(other, "Your manager is Ada Smith.")
    result = await memory.undo_sleep(report.run_id)
    assert result is not None and result["kept"] == 1 and result["restored"] == 0
    assert await rules.texts(memory) == {"Your manager is Ada Smith."}


async def test_update_cannot_replace_a_different_subject(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = await rules.put(memory, "Your manager is Tom Briggs.")
    quote = "Tom Briggs's office is in room 9C."
    await rules.chat(chats, quote)
    use(monkeypatch, [[op("UPDATE", quote, quote, old)]])
    report = await memory.sleep("model", chats=chats)
    assert not report.changes and report.refused == 1
    assert await rules.texts(memory) == {"Your manager is Tom Briggs."}


async def test_newer_retraction_of_an_edited_note_can_be_undone(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = await rules.put(memory, "You are allergic to peanuts.", edited=True)
    quote = "I'm no longer allergic to peanuts."
    await rules.chat(chats, quote)
    use(monkeypatch, [[op("RETIRE", quote, note=old)]])
    report = await memory.sleep("model", chats=chats)
    assert [c.op for c in report.changes] == ["RETIRE"]
    await memory.undo_sleep(report.run_id)
    assert await rules.texts(memory) == {"You are allergic to peanuts."}


async def test_retire_undo_does_not_revive_a_note_beside_a_later_value(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = await rules.put(memory, "Peanuts cause your allergic reactions.")
    quote = "Peanuts no longer cause my allergic reactions."
    await rules.chat(chats, quote)
    use(monkeypatch, [[op("RETIRE", quote, note=old)]])
    report = await memory.sleep("model", chats=chats)
    assert [c.op for c in report.changes] == ["RETIRE"]
    await rules.put(memory, "Cashews cause your allergic reactions.", at=rules.SOON)
    result = await memory.undo_sleep(report.run_id)
    assert result is not None and result["kept"] == 1 and result["restored"] == 0
    assert await rules.texts(memory) == {"Cashews cause your allergic reactions."}


async def test_all_tag_priorities_use_the_same_budget(
    memory: ChatMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = use(monkeypatch, [[], [], [], [], []])
    tagged = [
        sleep.SleepTurn("s", 1, 1, 2, "unsaved", unsaved_claim=True),
        sleep.SleepTurn("s", 2, 3, 4, "request", remember_request=True),
        sleep.SleepTurn(
            "s",
            3,
            5,
            6,
            "changed",
            remembered=(rules.records.Written(1, "x", "about"),),
        ),
        sleep.SleepTurn("s", 4, 7, 8, "used", used_later=True),
        sleep.SleepTurn("s", 5, 9, 10, "recent"),
    ]
    report = await sleep.run(
        *memory.stores(), tagged, variant="model+tags", provider=provider, now=10
    )
    assert report.model_calls == 5
    assert [
        json.loads(messages[-1].content or "")["typed"] for messages in provider.calls
    ] == [
        "unsaved",
        "request",
        "changed",
        "used",
        "recent",
    ]


async def test_model_change_cap_keeps_the_watermark(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sleep, "MAX_CHANGES", 1)
    typed = "I'm allergic to peanuts. I enjoy kitesurfing."
    await rules.chat(chats, typed)
    use(
        monkeypatch,
        [
            [
                op("ADD", "I'm allergic to peanuts.", "You are allergic to peanuts."),
                op("ADD", "I enjoy kitesurfing.", "You enjoy kitesurfing."),
            ]
        ],
    )
    report = await memory.sleep("model", chats=chats)
    assert len(report.changes) == 1 and report.capped
    assert await memory.stores()[1].meta("sleep_until") == 0


async def test_timeout_preserves_rules_and_the_watermark(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    await rules.chat(chats, "My dentist is Lee Park.")
    provider = use(monkeypatch, [[]])
    monkeypatch.setattr(sleep, "MODEL_TIMEOUT", 0.001)

    async def stalled() -> None:
        await asyncio.Event().wait()

    provider.after = stalled
    report = await memory.sleep("model", chats=chats)
    assert "TimeoutError" in report.aborted and report.model_calls == 1
    assert [c.op for c in report.changes] == ["O3"]
    assert await memory.stores()[1].meta("sleep_until") == 0
    await memory.undo_sleep(report.run_id)
    assert await memory.records() == []


async def test_owner_question_is_not_a_candidate_or_previous_context(
    chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    await rules.chat(chats, "I'm allergic to peanuts.")

    async def owner_turns(*args: Any, **kwargs: Any) -> list[Any]:
        return [SimpleNamespace(status="completed", undone=False, origin="ask_owner")]

    monkeypatch.setattr(chats, "list_turns", owner_turns)
    assert await sleep.sleep_turns(chats, 0) == []


async def test_o4_uses_the_previous_turn_even_before_the_watermark(
    memory: ChatMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = await rules.put(memory, "Her office is on Pine Street.", session="s", turn=2)
    source = sleep.SleepTurn(
        "s", 2, 3, 4, "Her office is on Pine Street.", previous="I met Dr. Ada."
    )
    provider = use(
        monkeypatch,
        [[op("UPDATE", source.typed, "Dr. Ada's office is on Pine Street.", old)]],
    )
    report = await sleep.run(
        *memory.stores(), [source], variant="model", until=5, provider=provider, now=10
    )
    assert [(c.op, c.before_ids) for c in report.changes] == [("O4", [old])]
    assert (
        json.loads(provider.calls[0][-1].content or "")["previous"] == "I met Dr. Ada."
    )


async def test_o4_cannot_turn_a_pronoun_into_the_user(
    memory: ChatMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = await rules.put(memory, "Her office is on Pine Street.", session="s", turn=1)
    source = sleep.SleepTurn(
        "s", 1, 3, 4, "Her office is on Pine Street.", previous="I met Dr. Ada."
    )
    provider = use(
        monkeypatch,
        [[op("UPDATE", source.typed, "Your office is on Pine Street.", old)]],
    )
    report = await sleep.run(
        *memory.stores(), [source], variant="model", until=5, provider=provider, now=10
    )
    assert not report.changes and report.refused == 1


async def test_add_is_checked_reported_and_undone(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    quote = "I'm allergic to peanuts."
    turn = await rules.chat(chats, quote)
    provider = use(monkeypatch, [[op("ADD", quote, "You are allergic to peanuts.")]])
    report = await memory.sleep("model", chats=chats)
    assert not report.aborted
    assert (report.model_calls, report.refused) == (1, 0)
    assert report.ms >= 0
    assert provider.bodies == [{"temperature": 0}]
    assert provider.extra_body_by_kind == {}  # never mutate the shared provider
    (change,) = report.changes
    assert (change.op, change.source_turn) == ("ADD", [turn.session_id, turn.turn])
    (note,) = await memory.records()
    assert (note.text, note.admitted_by, note.provenance["quote"]) == (
        "You are allergic to peanuts.",
        "sleep",
        quote,
    )
    assert (await memory.undo_sleep(report.run_id))["removed"] == 1  # type: ignore[index]
    assert await memory.records() == []


async def test_update_and_retire_are_undoable(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = await rules.put(memory, "Peanuts cause your allergic reactions.")
    quote = "Cashews cause my allergic reactions."
    await rules.chat(chats, quote)
    use(
        monkeypatch,
        [[op("UPDATE", quote, "Cashews cause your allergic reactions.", old)]],
    )
    changed = await memory.sleep("model", chats=chats)
    assert [(c.op, c.before_ids) for c in changed.changes] == [("UPDATE", [old])]
    before_retire = rules.rows(memory)
    (current,) = await memory.records()
    negative = "Cashews no longer cause my allergic reactions."
    await rules.chat(chats, negative)
    use(monkeypatch, [[op("RETIRE", negative, note=current.id)]])
    retired = await memory.sleep("model", chats=chats)
    assert [(c.op, c.after_id) for c in retired.changes] == [("RETIRE", None)]
    assert await memory.records() == []
    assert (await memory.undo_sleep(retired.run_id))["restored"] == 1  # type: ignore[index]
    assert rules.rows(memory) == before_retire
    await memory.undo_sleep(changed.run_id)
    assert await rules.texts(memory) == {"Peanuts cause your allergic reactions."}


@pytest.mark.parametrize(
    "typed,proposal",
    [
        (
            "I'm allergic to peanuts.",
            op("ADD", "made up", "You are allergic to peanuts."),
        ),
        (
            "I'm allergic to peanuts.",
            op("ADD", "I'm allergic to peanuts.", "You are allergic to cashews."),
        ),
        (
            "Marco's studio is on Main Street.",
            op(
                "ADD",
                "Marco's studio is on Main Street.",
                "Your studio is on Main Street.",
            ),
        ),
        (
            "My password is hunter2.",
            op("ADD", "My password is hunter2.", "Your password is hunter2."),
        ),
        ("Nothing lasting.", {"op": "DELETE", "quote": "Nothing lasting."}),
        ("Nothing lasting.", {"op": "ADD", "quote": [], "text": 2}),
    ],
)
async def test_bad_ops_are_dropped_and_counted(
    memory: ChatMemory,
    chats: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
    typed: str,
    proposal: dict[str, Any],
) -> None:
    await rules.chat(chats, typed)
    use(monkeypatch, [[proposal]])
    report = await memory.sleep("model", chats=chats)
    assert not report.changes and report.refused == 1
    assert await memory.records() == []


@pytest.mark.parametrize(
    "quote", ["I'm allergic to peanuts.", "I'm no longer allergic to cashews."]
)
async def test_retire_needs_negation_and_the_retracted_value(
    memory: ChatMemory,
    chats: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
    quote: str,
) -> None:
    old = await rules.put(memory, "You are allergic to peanuts.")
    await rules.chat(chats, quote)
    use(monkeypatch, [[op("RETIRE", quote, note=old)]])
    report = await memory.sleep("model", chats=chats)
    assert not report.changes and report.refused == 1
    assert await rules.texts(memory) == {"You are allergic to peanuts."}


async def test_edited_note_is_untouched_by_an_older_quote(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    quote = "I'm no longer allergic to peanuts."
    await rules.chat(chats, quote)
    old = await rules.put(memory, "You are allergic to peanuts.", edited=True)
    use(monkeypatch, [[op("RETIRE", quote, note=old)]])
    report = await memory.sleep("model", chats=chats)
    assert not report.changes and report.refused == 1


async def test_lock_prevents_even_the_first_model_call(
    tmp_path: Path, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "gpu-lock"
    lock.write_text("busy")
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory.sleep_lock = str(lock)
    memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
    provider = use(monkeypatch, [[]])
    try:
        await rules.chat(chats, "I'm allergic to peanuts.")
        report = await memory.sleep("model", chats=chats)
        assert "lock" in report.aborted and report.model_calls == 0
        assert not provider.calls
        assert await memory.stores()[1].meta("sleep_until") == 0
    finally:
        await memory.aclose()


async def test_new_turn_during_call_aborts_before_writing(
    memory: ChatMemory, chats: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    quote = "I'm allergic to peanuts."
    await rules.chat(chats, quote)
    provider = use(monkeypatch, [[op("ADD", quote, "You are allergic to peanuts.")]])

    async def active() -> None:
        await rules.chat(chats, "Hello again.")

    provider.after = active
    report = await memory.sleep("model", chats=chats)
    assert "active" in report.aborted and report.model_calls == 1
    assert not report.changes and await memory.records() == []
    assert await memory.stores()[1].meta("sleep_until") == 0


@pytest.mark.parametrize("variant,first", [("model", "old"), ("model+tags", "tagged")])
async def test_equal_budget_targets_tags_or_oldest_backlog(
    memory: ChatMemory, monkeypatch: pytest.MonkeyPatch, variant: str, first: str
) -> None:
    monkeypatch.setattr(sleep, "MODEL_BUDGET", 1)
    provider = use(monkeypatch, [[]])
    turns = [
        sleep.SleepTurn("s", 1, 1, 2, "old"),
        sleep.SleepTurn("s", 2, 3, 4, "tagged", unsaved_claim=True),
    ]
    report = await sleep.run(
        *memory.stores(), turns, variant=variant, provider=provider, now=10
    )
    assert report.model_calls == 1
    payload = json.loads(provider.calls[0][-1].content or "")
    assert payload["typed"] == first


@pytest.mark.parametrize("variant", ["model", "model+tags"])
async def test_o4_resolves_a_named_person_from_source_context(
    memory: ChatMemory,
    chats: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    source = await rules.chat(chats, "Dr. Ada's office is on Pine Street.")
    old = await rules.put(
        memory,
        "Her office is on Pine Street.",
        source.started_at,
        session=source.session_id,
        turn=source.turn,
    )
    use(
        monkeypatch,
        [
            [],
            [
                op(
                    "UPDATE",
                    "Dr. Ada's office is on Pine Street.",
                    "Dr. Ada's office is on Pine Street.",
                    old,
                )
            ],
        ],
    )
    report = await memory.sleep(variant, chats=chats)
    assert [(c.op, c.before_ids) for c in report.changes] == [("O4", [old])]
    assert await rules.texts(memory) == {"Dr. Ada's office is on Pine Street."}
    await memory.undo_sleep(report.run_id)
    assert await rules.texts(memory) == {"Her office is on Pine Street."}
