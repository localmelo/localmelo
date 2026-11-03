"""A chat's runner keeps who a request is about: its earlier turns and other chats.

Each turn's run gets the chat's newest finished turns of the person's, from the
session store (``Agent.run_task(earlier_turns=...)``), so a short request keeps
the chat's subject when the window has lost it. A chat's first turn (a side chat
or a terminal chat, never the main chat) gets a digest of what the user's other
chats discussed in the last ``[chat] recent_digest_minutes``, read from memory's
turn index: the main chat discusses a friend, a new chat says "read his résumé
first", and the stub model, which answers about whoever the context names,
answers about the friend. No real model.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo import clock
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.checker.validators import MAX_QUERY_LEN
from localmelo.support.config import Config
from localmelo.support.memory import ChatMemory
from localmelo.support.memory.digest import DIGEST_CHARS, DIGEST_HEAD, Digest
from localmelo.support.memory.index import Document
from localmelo.support.sessions import runner as runner_module
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.agent.test_earlier_turns import SubjectModel

# What the user's own profile says, as pinned notes about the user would.
OWNER = "The user is Sam Rivera, a robotics student."
ABOUT = "What else could Alex do besides UX research?"
ALEX = "Alex could move into product management or service design."


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[SessionStore]:
    store = SessionStore(str(tmp_path / "sessions.db"), writer="gateway")
    await store.open()
    try:
        yield store
    finally:
        await store.aclose()


@pytest.fixture()
async def memory(tmp_path: Path) -> AsyncIterator[ChatMemory]:
    memory = ChatMemory(lambda: None, directory=str(tmp_path / "memory"))
    try:
        yield memory
    finally:
        await memory.aclose()


class Chat:
    """One chat: a runner on an Agent whose model is the subject stub."""

    def __init__(
        self,
        store: SessionStore,
        memory: ChatMemory | None,
        workspace: Path,
        *,
        kind: str = "chat",
        session_id: str | None = None,
    ) -> None:
        workspace.mkdir(exist_ok=True)
        self.model = SubjectModel()
        self.agent = Agent(
            llm=self.model,
            options=AgentOptions(
                builtins=frozenset(),
                record_answers=True,
                memory=False,
                workspace_root=str(workspace),
                identity=OWNER,
            ),
        )
        self.runner = SessionRunner(
            self.agent,
            store,
            session_id,
            new_session=None if session_id else NewSession(kind=kind),
            memory=memory,
        )
        self.kwargs: list[dict[str, Any]] = []
        run_task = self.agent.run_task

        async def recorded(query: str, **kwargs: Any) -> Any:
            self.kwargs.append({"query": query, **kwargs})
            return await run_task(query, **kwargs)

        self.agent.run_task = recorded  # type: ignore[method-assign]

    async def say(self, text: str, answer: str | None = None) -> str:
        if answer is not None:
            self.model.plans.append(_reply(answer))
        result = await self.runner.run_turn(text, title=text)
        assert result.status == "completed"
        return result.result

    def last_message(self) -> str:
        return str(self.model.calls[-1]["messages"][-1].content)

    async def close(self) -> None:
        await self.runner.close()
        await self.agent.close()


def _reply(text: str) -> Any:
    from localmelo.melo.schema import Message

    return Message(role="assistant", content=text)


class TestEarlierTurns:
    async def test_each_turn_gets_the_chats_newest_turns_of_the_person(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        chat = Chat(store, None, tmp_path / "w")
        try:
            await chat.say("What is Alex's background?", "Alex is a UX researcher.")
            assert "earlier_turns" not in chat.kwargs[-1]  # a first turn has none
            # A task's question to the chat is not the person's turn.
            chat.model.plans.append(_reply("The task may go on."))
            await chat.runner.start_turn("[task t1] may I go on?", origin="ask_owner")
            await chat.runner.wait()
            await chat.say("Is Alex looking for work?", "Not that I know.")
            await chat.runner.undo()  # an undone turn is left out
            await chat.say(ABOUT, ALEX)
            await chat.say("List those roles", "Product researcher, service designer.")
            await chat.say("go look it up")
            assert chat.kwargs[-1]["earlier_turns"] == [
                ("What is Alex's background?", "Alex is a UX researcher."),
                (ABOUT, ALEX),
                ("List those roles", "Product researcher, service designer."),
            ]
            assert chat.kwargs[-1]["query"].startswith("go look it up")
        finally:
            await chat.close()

    async def test_at_most_the_newest_three(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        chat = Chat(store, None, tmp_path / "w")
        try:
            for n in range(5):
                await chat.say(f"question {n}", f"answer {n}")
            await chat.say("and now?")
            words = [pair[0] for pair in chat.kwargs[-1]["earlier_turns"]]
            assert words == ["question 2", "question 3", "question 4"]
        finally:
            await chat.close()

    async def test_a_store_that_fails_leaves_the_run_without_them(
        self, store: SessionStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat = Chat(store, None, tmp_path / "w")
        try:
            await chat.say("What is Alex's background?", "Alex is a UX researcher.")

            async def broken(*_: Any, **__: Any) -> Any:
                raise OSError("disk gone")

            monkeypatch.setattr(
                "localmelo.support.sessions.runner.earlier_turns", broken
            )
            assert await chat.say("go on", "Going on.") == "Going on."
            assert "earlier_turns" not in chat.kwargs[-1]
        finally:
            await chat.close()


class TestDigest:
    async def _main_chat_about_alex(
        self, store: SessionStore, memory: ChatMemory, tmp_path: Path
    ) -> str:
        """The main chat's turn about Alex, indexed; returns the main chat's id."""
        main = Chat(store, memory, tmp_path / "main", kind="front")
        try:
            await main.say(ABOUT, ALEX)
            await memory.drain()  # the turn is indexed off the reply path
        finally:
            await main.close()
        assert main.runner.session_id is not None
        return main.runner.session_id

    async def test_a_new_chat_hears_about_the_friend_and_answers_about_him(
        self, store: SessionStore, memory: ChatMemory, tmp_path: Path
    ) -> None:
        await self._main_chat_about_alex(store, memory, tmp_path)
        side = Chat(store, memory, tmp_path / "side")
        try:
            answer = await side.say("read his résumé first")
            message = side.last_message()
            head = DIGEST_HEAD.format(minutes=30)
            assert message.startswith("read his résumé first\n\n")
            digest = message[message.index(head) :]
            assert len(digest) <= DIGEST_CHARS
            (line,) = digest.splitlines()[1:]
            assert line.startswith("- Main chat, ")
            assert f"the user: {ABOUT}" in line and f"melo: {ALEX}" in line
            assert answer == "Roles that fit Alex."
            # Only the typed words are the turn's own.
            session_id = side.runner.session_id
            assert session_id is not None
            turns = await store.list_turns(session_id)
            assert [t.user_text for t in turns] == ["read his résumé first"]
            # The second turn has this chat's own turn: no digest.
            await side.say("and his projects?")
            assert head not in side.last_message()
        finally:
            await side.close()

    async def test_the_turn_memory_event_shows_the_digest_on_that_turn_only(
        self, store: SessionStore, memory: ChatMemory, tmp_path: Path
    ) -> None:
        main_id = await self._main_chat_about_alex(store, memory, tmp_path)
        side = Chat(store, memory, tmp_path / "side")
        try:
            await side.say("read his résumé first")
            await side.say("and his projects?")
            await memory.drain()
            session_id = side.runner.session_id
            assert session_id is not None
            events = [
                e.event
                for e in await store.list_events(session_id)
                if e.event["type"] == "turn.memory"
            ]
        finally:
            await side.close()
        first = [e["data"] for e in events if e["data"]["turn"] == 1]
        second = [e["data"] for e in events if e["data"]["turn"] == 2]
        # The recall's own event comes first; the refreshed one adds the digest.
        assert "digest" not in first[0] and "digest" in first[-1]
        digest = first[-1]["digest"]
        assert digest["minutes"] == 30
        (item,) = digest["items"]
        assert (item["session_id"], item["turn"]) == (main_id, 1)
        message = side.model.calls[0]["messages"][-1].content
        assert f"\n- {item['line']}" in message
        assert not item["line"].startswith("- ")
        assert second and all("digest" not in data for data in second)

    async def test_without_the_digest_the_users_profile_wins(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        # The failure the digest fixes: nothing names the friend.
        side = Chat(store, None, tmp_path / "side")
        try:
            assert await side.say("read his résumé first") == "Roles that fit Sam."
        finally:
            await side.close()

    async def test_the_main_chat_never_gets_one(
        self, store: SessionStore, memory: ChatMemory, tmp_path: Path
    ) -> None:
        side = Chat(store, memory, tmp_path / "side")
        try:
            await side.say(ABOUT, ALEX)
            await memory.drain()
        finally:
            await side.close()
        main = Chat(store, memory, tmp_path / "main", kind="front")
        try:
            await main.say("read his résumé first")
            assert "the user's other chats" not in main.last_message()
        finally:
            await main.close()

    async def test_a_task_question_turn_gets_none(
        self, store: SessionStore, memory: ChatMemory, tmp_path: Path
    ) -> None:
        await self._main_chat_about_alex(store, memory, tmp_path)
        side = Chat(store, memory, tmp_path / "side")
        try:
            await side.runner.start_turn("[task t1] may I go on?", origin="ask_owner")
            await side.runner.wait()
            assert "the user's other chats" not in side.last_message()
        finally:
            await side.close()

    @pytest.mark.parametrize("setting", ["minutes", "across_chats"])
    async def test_a_setting_turns_it_off(
        self, store: SessionStore, tmp_path: Path, setting: str
    ) -> None:
        cfg = Config()
        if setting == "minutes":
            cfg.chat.recent_digest_minutes = 0
        else:
            cfg.memory.across_chats = False
        memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "memory"))
        try:
            cfg.memory.across_chats = True  # the main chat's turn is indexed
            await self._main_chat_about_alex(store, memory, tmp_path)
            if setting == "across_chats":
                cfg.memory.across_chats = False
            side = Chat(store, memory, tmp_path / "side")
            try:
                assert await side.say("read his résumé first") == "Roles that fit Sam."
                assert "the user's other chats" not in side.last_message()
            finally:
                await side.close()
        finally:
            await memory.aclose()

    async def test_a_message_too_long_for_it_goes_without(
        self,
        store: SessionStore,
        memory: ChatMemory,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await self._main_chat_about_alex(store, memory, tmp_path)
        monkeypatch.setattr(runner_module, "MAX_QUERY_LEN", len("read it") + 10)
        side = Chat(store, memory, tmp_path / "side")
        try:
            await side.say("read it")
            assert "the user's other chats" not in side.last_message()
        finally:
            await side.close()
        assert MAX_QUERY_LEN > DIGEST_CHARS

    async def test_a_failed_read_leaves_the_message_as_it_is(
        self,
        store: SessionStore,
        memory: ChatMemory,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def broken(*_: Any, **__: Any) -> Digest:
            raise OSError("index gone")

        monkeypatch.setattr(memory, "recent_digest", broken)
        side = Chat(store, memory, tmp_path / "side")
        try:
            assert await side.say("read his résumé first") == "Roles that fit Sam."
        finally:
            await side.close()


def _block(digest: Digest | None) -> str:
    return "" if digest is None else digest.block


class TestRecentDigest:
    async def _index(self, memory: ChatMemory, docs: list[Document]) -> None:
        await memory.stores()[1].add_many(docs)

    def _doc(
        self,
        session: str,
        turn: int,
        at: float,
        *,
        kind: str = "chat",
        chunk: int = 0,
        text: str = "User: hi\nmelo: hello",
    ) -> Document:
        return Document(
            session_id=session,
            turn=turn,
            surface="gateway",
            kind=kind,
            title=f"chat {session}",
            text=text,
            created_at=at,
            chunk=chunk,
        )

    async def test_only_other_chats_recent_turn_heads(self, memory: ChatMemory) -> None:
        now = time.time()
        await self._index(
            memory,
            [
                self._doc("a", 1, now - 60, text="User: about Alex\nmelo: noted"),
                self._doc("a", 1, now - 60, chunk=1, text="User: about Alex\nmelo: x"),
                self._doc("b", 2, now - 50, text="User: in this chat\nmelo: ok"),
                self._doc("c", 1, now - 40 * 60, text="User: too old\nmelo: ok"),
                self._doc("d", 1, now - 30, kind="task", text="User: a task\nmelo: ok"),
                self._doc("e", -1, now - 20, kind="side_summary", text="summary"),
                self._doc("m", 7, now - 10, kind="front", text="User: main\nmelo: ok"),
            ],
        )
        digest = _block(await memory.recent_digest("b", now=now))
        lines = digest.splitlines()
        assert lines[0] == DIGEST_HEAD.format(minutes=30)
        assert [line.split(" · the user: ")[1] for line in lines[1:]] == [
            "about Alex · melo: noted",
            "main · melo: ok",
        ]
        assert lines[2].startswith("- Main chat, ")

    async def test_the_newest_five_in_600_characters(self, memory: ChatMemory) -> None:
        now = time.time()
        long = "word " * 300
        await self._index(
            memory,
            [
                self._doc(f"s{n}", 1, now - 100 + n, text=f"User: q{n} {long}\nmelo: a")
                for n in range(8)
            ],
        )
        digest = _block(await memory.recent_digest("other", now=now))
        lines = digest.splitlines()[1:]
        assert len(lines) == 5 and len(digest) <= DIGEST_CHARS
        assert [line.split("the user: ")[1].split()[0] for line in lines] == [
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
        ]

    async def test_markdown_and_short_words_read_plainly(
        self, memory: ChatMemory
    ) -> None:
        now = time.time()
        text = "User: go on\nmelo: \n## **Alex** is a *UX researcher*\n| a | b |"
        await self._index(memory, [self._doc("a", 1, now - 5, text=text)])
        digest = _block(await memory.recent_digest("b", now=now))
        assert digest.splitlines()[1].endswith(
            "the user: go on · melo: Alex is a UX researcher"
        )

    async def test_turns_recall_shows_or_a_newer_one_replaced_are_left_out(
        self, memory: ChatMemory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = time.time()
        await self._index(
            memory,
            [
                self._doc(
                    "a", 1, now - 90, text="User: I live in Lisbon.\nmelo: Noted."
                ),
                self._doc("b", 1, now - 60, text="User: I live in Utrecht.\nmelo: ok"),
                self._doc("c", 1, now - 30, text="User: about Alex\nmelo: a friend"),
            ],
        )
        # A newer note replaced the Lisbon turn, as recall reads the corpus.
        monkeypatch.setattr(memory, "_replaced", lambda *_: {("a", 1): None})
        digest = _block(
            await memory.recent_digest("x", exclude_turns={("b", 1)}, now=now)
        )
        assert "Lisbon" not in digest and "Utrecht" not in digest
        assert "about Alex" in digest
        # Nothing left: no digest at all.
        left = await memory.recent_digest(
            "x", exclude_turns={("b", 1), ("c", 1)}, now=now
        )
        assert left is None

    async def test_no_turn_no_digest(self, memory: ChatMemory) -> None:
        assert await memory.recent_digest("b") is None

    async def test_a_pinned_clock_reads_the_turns_near_its_time(
        self, memory: ChatMemory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A memory benchmark asks "on" an earlier day: only the turns stamped
        # within the minutes before that time count, not later ones nor today's.
        pinned = datetime(2023, 5, 20, 13, 40).timestamp()
        monkeypatch.setenv(clock.ENV, "2023-05-20T13:40:00")
        await self._index(
            memory,
            [
                self._doc("a", 1, pinned - 10 * 60, text="User: in time\nmelo: ok"),
                self._doc("b", 1, pinned - 40 * 60, text="User: too early\nmelo: ok"),
                self._doc("c", 1, pinned + 5 * 60, text="User: after it\nmelo: ok"),
                self._doc("d", 1, time.time() - 60, text="User: today\nmelo: ok"),
            ],
        )
        digest = await memory.recent_digest("x")
        assert digest is not None
        assert [item.session_id for item in digest.items] == ["a"]
        assert "the user: in time" in digest.items[0].line
        # Unpinned, today's turn is the one that counts.
        monkeypatch.delenv(clock.ENV)
        digest = await memory.recent_digest("x")
        assert digest is not None and [i.session_id for i in digest.items] == ["d"]

    async def test_its_event_names_each_turn_and_the_line_the_model_read(
        self, memory: ChatMemory
    ) -> None:
        now = time.time()
        await self._index(
            memory,
            [
                self._doc(
                    "m", 3, now - 90, kind="front", text=f"User: {ABOUT}\nmelo: {ALEX}"
                ),
                self._doc("s", 1, now - 30, text="User: go on\nmelo: done"),
            ],
        )
        digest = await memory.recent_digest("x", now=now)
        assert digest is not None
        event = digest.event()
        assert event["minutes"] == 30
        assert [(i["session_id"], i["turn"]) for i in event["items"]] == [
            ("m", 3),
            ("s", 1),
        ]
        lines = digest.block.splitlines()[1:]
        assert [f"- {item['line']}" for item in event["items"]] == lines
        assert event["items"][0]["line"].startswith("Main chat, ")

    @pytest.mark.parametrize("value", [-1, 1441, "30", True, 2.5])
    def test_a_bad_setting_reads_as_the_default(
        self, tmp_path: Path, value: Any
    ) -> None:
        cfg = Config()
        cfg.chat.recent_digest_minutes = value
        memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "m"))
        assert memory.digest_minutes() == 30
        assert any("recent_digest_minutes" in e for e in cfg.setting_errors())

    def test_the_setting_is_read_each_time(self, tmp_path: Path) -> None:
        cfg = Config()
        memory = ChatMemory(lambda: cfg, directory=str(tmp_path / "m"))
        assert memory.digest_minutes() == 30
        cfg.chat.recent_digest_minutes = 5
        assert memory.digest_minutes() == 5
        assert ChatMemory(lambda: None).digest_minutes() == 30
