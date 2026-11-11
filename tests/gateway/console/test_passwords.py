"""A password typed in any entry is cut out where it comes in, once (#547).

Every route that takes the person's text (a first turn, a turn, a note, a
queued message, a routed task, a reply to a task's question, a chore, a chat's
new title) hides a password in it before the text is routed, queued, titled or
stored; a message's files go through the same door (#626). The store, the
model's requests and the events then hold the handle, and the value is in none
of them.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.sign_in import VAULT
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.asks import _restored
from localmelo.support.gateway.handover import words_of
from localmelo.support.gateway.i18n import t
from localmelo.support.sessions.attachments import TextAttachment, compose_user_text
from localmelo.support.sessions.store import typed_words
from localmelo.tests.agent.test_sign_in import LIKE_HEADING

from .test_ask_owner import Script, answer, ask_console, held, tool_call, until
from .test_turn_writes import _post_turn, _wait_turn

SECRETS = {
    "turn": "turnSecret-11",
    "note": "noteSecret-22",
    "create": "createSecret-33",
    "task": "taskSecret-44",
    "queued": "queuedSecret-55",
    "chore": "choreSecret-66",
    "chore_title": "choreTitleSecret-61",
    "chore_edit": "choreEditSecret-62",
    "reply": "replySecret-77",
    "title": "titleSecret-88",
    # Given in a text file attached to the message of the same name (the turn
    # route's own file is the reported-case test's).
    "create_file": "createFileSecret-14",
    "task_file": "taskFileSecret-15",
    "queued_file": "queuedFileSecret-16",
}
STEP = tool_call("task_get", {"task_id": "000000000000"}, "step-1")


def say(kind: str) -> str:
    return f"sign in to forum.example for me, my password is {SECRETS[kind]}"


def files(kind: str) -> list[dict[str, Any]]:
    """A text file whose line gives a password, as the page sends one."""
    text = f"my other login, password: {SECRETS[kind + '_file']}"
    return [{"name": "notes.txt", "text": text, "chars": len(text)}]


def stored(path: Path) -> str:
    """Every row of every table of the store, as text."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return "\n".join(conn.iterdump())


def seen_by_models(models: dict[str, Any]) -> str:
    scripts: list[Script] = [models["main"], *models["workers"]]
    scripts += [
        v
        for k, v in models.items()
        if k not in ("main", "workers", "browsers", "agents")
    ]
    return "\n".join(
        str(message.content)
        for script in scripts
        for request in script.requests
        for message in request
    )


async def test_a_password_typed_in_any_entry_is_in_no_store_title_event_or_model_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main = [held(release, STEP), answer("Done.")]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        # A turn, and a note added to it while it runs.
        started = await _post_turn(client, str(console.main_id), say("turn"))
        assert started.status_code == 202, started.text
        turn_id = started.json()["turn_id"]
        await until(lambda: len(models["main"].requests) == 1)
        noted = await client.post(
            f"/api/turns/{turn_id}/notes", json={"text": say("note")}
        )
        assert noted.status_code == 202, noted.text
        release.set()
        await _wait_turn(client, str(console.main_id))

        # A new chat with its first turn: its title is the typed text.
        created = await client.post(
            "/api/sessions",
            json={"turn": {"text": say("create"), "attachments": files("create")}},
        )
        assert created.status_code == 201, created.text
        side = created.json()["session"]["id"]
        await _wait_turn(client, side)
        assert SECRETS["create"] not in created.text
        assert SECRETS["create_file"] not in created.text

        # A message routed to a task of the chat.
        routed = await _post_turn(
            client,
            str(console.main_id),
            say("task"),
            as_task=True,
            attachments=files("task"),
        )
        assert routed.status_code == 200, routed.text
        assert SECRETS["task"] not in routed.text
        assert SECRETS["task_file"] not in routed.text

        # A message queued while the chat answers a task's question.
        queued: list[tuple[Any, ...]] = []

        async def queue(*args: Any) -> dict[str, Any]:
            queued.append(args)
            return {"echo": "queued"}

        with monkeypatch.context() as spy:
            spy.setattr(console.asks, "holds", lambda session: True)
            spy.setattr(console.asks, "queue", queue)
            await _post_turn(
                client,
                str(console.main_id),
                say("queued"),
                attachments=files("queued"),
            )
        assert "[password-" in repr(queued)
        assert SECRETS["queued"] not in repr(queued)
        assert SECRETS["queued_file"] not in repr(queued)

        # A reply to a task's question, a chore, and a chat's new title.
        replies: list[tuple[Any, ...]] = []

        async def act(*args: Any) -> dict[str, Any]:
            replies.append(args)
            return {}

        with monkeypatch.context() as spy:
            spy.setattr(console.front_for(None), "act", act)
            await client.post(
                "/api/front/echoes/1", json={"do": "reply", "text": say("reply")}
            )
        assert SECRETS["reply"] not in repr(replies) and "[password-" in repr(replies)
        chore = await client.post(
            "/api/chores",
            json={
                "request": say("chore"),
                "title": f"Sign in (password: {SECRETS['chore_title']})",
                "schedule": {"kind": "weekdays", "at": "08:30"},
            },
        )
        assert chore.status_code == 201, chore.text
        edited = await client.patch(
            f"/api/chores/{chore.json()['chore']['id']}",
            json={
                "revision": chore.json()["chore"]["revision"],
                "title": f"Sign in (password: {SECRETS['chore_edit']})",
            },
        )
        assert edited.status_code == 200, edited.text
        renamed = await client.patch(
            f"/api/sessions/{side}", json={"title": say("title")}
        )
        assert renamed.status_code == 200, renamed.text

        await _wait_idle(console)
        everything = (
            stored(tmp_path / "sessions.db")
            + seen_by_models(models)
            + created.text
            + routed.text
            + chore.text
            + edited.text
            + renamed.text
            + (await client.get(f"/api/sessions/{side}")).text
            + (await client.get("/api/chores")).text
        )
        for kind, secret in SECRETS.items():
            assert secret not in everything, f"{kind} password leaked"
        # The handle took its place: the model read it, the store kept it.
        assert "my password is [password-" in stored(tmp_path / "sessions.db")
        assert VAULT.value("[password-1]") == SECRETS["turn"]


async def test_a_password_in_an_attached_file_is_hidden_where_it_is_stored_read_searched_and_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn's rows, its reading routes, the chat search and the model hold the handle; a resend is the same turn (#626)."""
    secret = "quokka-7731"
    # A line like the heading of melo's memory comes before the password line.
    text = f"{LIKE_HEADING}\nmy password is {secret}"
    body = {
        "text": "sign in to forum.example for me",
        "attachments": [{"name": "notes.txt", "text": text, "chars": len(text)}],
        "client_request_id": "file-1",
    }
    async with ask_console(
        tmp_path, monkeypatch, main=[answer("Done.")], run=False
    ) as (console, client, models):
        main = str(console.main_id)
        url = f"/api/sessions/{main}/turns"
        sent = await client.post(url, json=body)
        assert sent.status_code == 202, sent.text
        await _wait_turn(client, main)

        # What is stored: the composed text, the typed text, the file's snapshot.
        [turn] = await console.store.list_turns(main)
        assert turn.user_text.count("[password-1]") == 1
        assert turn.typed_text == "sign in to forum.example for me"
        assert turn.attachments is not None
        assert (
            turn.attachments[0].text == f"{LIKE_HEADING}\nmy password is [password-1]"
        )
        assert secret not in stored(tmp_path / "sessions.db")

        # What the reading routes serve and a search of the chats matches.
        shown = (await client.get(url)).text
        events = (
            await client.get(f"/api/sessions/{main}/events", params={"follow": "false"})
        ).text
        assert secret not in shown + events + sent.text
        assert "my password is [password-1]" in shown + events
        for kind in ("front", "chat"):
            by_value = await client.get(
                "/api/sessions", params={"query": secret, "kind": kind}
            )
            assert by_value.json()["sessions"] == []
        by_handle = await client.get(
            "/api/sessions",
            params={"query": "my password is [password-1]", "kind": "front"},
        )
        assert [row["id"] for row in by_handle.json()["sessions"]] == [main]

        # What the model read: the handle, as the run would have made it.
        assert secret not in seen_by_models(models)
        assert "my password is [password-1]" in seen_by_models(models)
        assert VAULT.value("[password-1]") == secret

        # The same key with the same message is that turn, not a second one;
        # so is a cancel that names the message the same way.
        again = await client.post(url, json=body)
        assert again.status_code == 200, again.text
        assert again.json()["repeated"] and again.json()["turn_id"] == turn.id
        cancel = await client.post(
            f"/api/sessions/{main}/submissions/file-1/cancel",
            json={"text": body["text"], "attachments": body["attachments"]},
        )
        assert cancel.status_code == 200, cancel.text
        assert cancel.json()["submission"]["state"] == "accepted"
        assert len(await console.store.list_turns(main)) == 1
        assert secret not in again.text + cancel.text


def spy_on_words(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """The person's words each run is given (``Agent.run_task``'s ``user_words``)."""
    given: list[Any] = []
    real = Agent.run_task

    async def run_task(self: Agent, query: str, **kwargs: Any) -> Any:
        given.append(kwargs.get("user_words"))
        return await real(self, query, **kwargs)

    monkeypatch.setattr(Agent, "run_task", run_task)
    return given


def a_file(name: str, text: str) -> dict[str, Any]:
    return {"name": name, "text": text, "chars": len(text)}


async def test_the_one_password_of_a_sent_file_is_typed_for_that_run_and_no_retry_or_old_handle_brings_it_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chat turn with a file, from the route to the run: its password is live on the site the words name; a retry carries no set, and a file that only holds the handle's text (#626) gives nothing."""
    secret = "quokka-7741"
    words = "sign in to forum.example for me"
    async with ask_console(
        tmp_path, monkeypatch, main=[answer("Done.")] * 3, run=False
    ) as (console, client, _):
        main = str(console.main_id)
        sent = await _post_turn(
            client,
            main,
            words,
            attachments=[a_file("login.txt", f"password: {secret}")],
        )
        assert sent.status_code == 202, sent.text
        turn = await _wait_turn(client, main)
        entry = console.sessions.get(main)
        assert entry is not None and entry.agent is not None
        review = entry.agent._reviewer
        assert review.sign_in_value("[password-1]", "forum.example") == secret
        assert review.sign_in_value("[password-1]", "other.example") is None
        assert review.holds_sign_in() and review.signs_in("forum.example")

        # Try again sends the stored message again: the set is not stored, so none.
        retried = await client.post(
            f"/api/sessions/{main}/retry", json={"turn_id": turn["id"]}
        )
        assert retried.status_code == 202, retried.text
        await _wait_turn(client, main)
        assert review.sign_in_value("[password-1]", "forum.example") is None
        assert not review.holds_sign_in()

        # A file that holds only the handle's text brings the password back to no one.
        held = a_file("old.txt", "the old one is [password-1]")
        assert (
            await _post_turn(client, main, words, attachments=[held])
        ).status_code == 202
        await _wait_turn(client, main)
        assert VAULT.value("[password-1]") == secret
        assert review.sign_in_value("[password-1]", "forum.example") is None
        assert secret not in stored(tmp_path / "sessions.db")


async def test_a_queued_message_carries_its_files_handles_in_memory_and_a_stored_row_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The set waits with the message in this process; the stored row holds no set, so a row read back after a restart has none (#626)."""
    seen: list[Any] = []
    real = Agent.run_task

    async def run_task(self: Agent, query: str, **kwargs: Any) -> Any:
        seen.append(kwargs.get("file_handles"))
        return await real(self, query, **kwargs)

    monkeypatch.setattr(Agent, "run_task", run_task)
    release = asyncio.Event()
    main = [held(release, answer("First.")), answer("Second.")]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        chat = str(console.main_id)
        assert (await _post_turn(client, chat, "start the job")).status_code == 202
        await until(lambda: len(models["main"].requests) == 1)
        queued = await _post_turn(
            client,
            chat,
            "sign in to forum.example for me",
            attachments=[a_file("login.txt", "password: quokka-5541")],
            client_request_id="q-1",
        )
        assert queued.json()["echo"]["echo"] == "queued", queued.text
        info = await console.store.get_session(chat)
        assert info is not None
        [row] = info.meta["queued"]
        assert "[password-1]" in row["attachments"][0]["text"]
        assert "file_handles" not in row and not _restored(row).file_handles
        release.set()
        await until(lambda: len(seen) == 2)
        await console.asks.settle()
        await _wait_turn(client, chat)
        assert seen == [None, {"[password-1]"}]


async def test_a_queued_message_keeps_its_files_apart_and_its_turn_starts_as_a_sent_one_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Words and files wait apart and start the turn stored apart, with no file text among the person's words; a resend is that turn (#626)."""
    typed = "have a look at the file"
    text = "log in at files-only.example\nmy password is quokka-5521"
    given = spy_on_words(monkeypatch)
    release = asyncio.Event()
    main = [held(release, answer("First.")), answer("Second."), answer("Third.")]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        chat = str(console.main_id)
        assert (await _post_turn(client, chat, "start the job")).status_code == 202
        await until(lambda: len(models["main"].requests) == 1)
        # Words and a file sent while the turn runs wait for the next one.
        with_words = {
            "attachments": [a_file("notes.txt", text)],
            "client_request_id": "q-1",
        }
        queued = await _post_turn(client, chat, typed, **with_words)
        # A file and no words, sent while the chat answers a task's question.
        with monkeypatch.context() as answering:
            answering.setattr(console.asks, "holds", lambda session: True)
            alone = await _post_turn(
                client,
                chat,
                "",
                attachments=[a_file("other.txt", "my password is quokka-5522")],
                client_request_id="q-2",
            )
        for response in (queued, alone):
            assert response.status_code == 200, response.text
            assert response.json()["echo"]["echo"] == "queued"
        info = await console.store.get_session(chat)
        assert info is not None and info.meta["queued"][0]["attachments"]
        release.set()
        await until(lambda: len(given) == 3)
        await console.asks.settle()
        await _wait_turn(client, chat)
        _, first, second = await console.store.list_turns(chat)
        # The turn has its typed words and the file stored apart, the handle in the file.
        assert first.typed_text == typed
        assert first.attachments == (
            TextAttachment(
                "notes.txt", "log in at files-only.example\nmy password is [password-1]"
            ),
        )
        # With no words the stand-in (how many files, no name) is the row's typed text,
        # as for a message that did not wait.
        assert second.typed_text == "[1 file]"
        assert second.attachments == (
            TextAttachment("other.txt", "my password is [password-2]"),
        )
        # The run is given the typed words, never a file's text; the stand-in is none.
        assert given == ["start the job", typed, ""]
        assert "quokka" not in stored(tmp_path / "sessions.db")
        # The same key with the same message is that turn, not a second one.
        again = await _post_turn(client, chat, typed, **with_words)
        assert again.status_code == 200, again.text
        assert again.json()["repeated"] and again.json()["turn_id"] == first.id


async def test_a_turn_tried_again_twice_holds_its_typed_words_and_files_and_runs_on_the_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each retry sends the typed words and the stored files again as the message was sent; no run reads a file's text as words (#626)."""
    typed = "sign in to forum.example for me"
    text = "my other login, password: quokka-5531"
    given = spy_on_words(monkeypatch)
    async with ask_console(
        tmp_path, monkeypatch, main=[answer("Done.")] * 3, run=False
    ) as (console, client, _):
        main = str(console.main_id)
        sent = await _post_turn(
            client,
            main,
            typed,
            attachments=[a_file("notes.txt", text)],
            client_request_id="retry-1",
        )
        assert sent.status_code == 202, sent.text
        turn = await _wait_turn(client, main)
        for _ in range(2):
            retried = await client.post(
                f"/api/sessions/{main}/retry", json={"turn_id": turn["id"]}
            )
            assert retried.status_code == 202, retried.text
            turn = await _wait_turn(client, main)
        original, *retries = [t for t in await console.store.list_turns(main)]
        assert len(retries) == 2
        for row in retries:
            assert row.typed_text == typed and row.user_text == original.user_text
            assert row.attachments == original.attachments
            assert row.attachments and "[password-1]" in row.attachments[0].text
        assert given == [typed] * 3
        assert "quokka" not in stored(tmp_path / "sessions.db")


@pytest.mark.parametrize(
    "busy", [False, True], ids=["sent at once", "sent while a turn runs"]
)
async def test_a_message_of_only_a_file_holds_no_part_of_the_files_name_in_its_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, busy: bool
) -> None:
    """The stand-in for no words says how many files, never a name, on every path: the row's text, its typed text, the earlier words and the run's words, also when it is tried again (#626)."""
    name = "login.files-only.example.txt"
    given = spy_on_words(monkeypatch)
    release = asyncio.Event()
    main = [held(release, answer("First.")), answer("Second."), answer("Third.")]
    if not busy:
        main = main[1:]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        chat = str(console.main_id)
        if busy:
            assert (await _post_turn(client, chat, "start the job")).status_code == 202
            await until(lambda: len(models["main"].requests) == 1)
        sent = await _post_turn(
            client,
            chat,
            "",
            attachments=[a_file(name, "hello there")],
            client_request_id="only-a-file",
        )
        assert sent.status_code == (200 if busy else 202), sent.text
        release.set()
        await until(lambda: len(given) == (2 if busy else 1))
        await console.asks.settle()
        turn = await _wait_turn(client, chat)
        retried = await client.post(
            f"/api/sessions/{chat}/retry", json={"turn_id": turn["id"]}
        )
        assert retried.status_code == 202, retried.text
        await _wait_turn(client, chat)
        rows = (await console.store.list_turns(chat))[-2:]
        for row in rows:
            assert row.typed_text == "[1 file]" == row.user_text.split("\n\n")[0]
            assert "files-only" not in row.user_text.split("\n\n<attachment")[0]
            assert row.attachments == (TextAttachment(name, "hello there"),)
            assert typed_words(row) == ""
        assert [w for w in given if w] == ["start the job"] * busy
        assert not any("files-only" in str(words) for words in given)


@pytest.mark.parametrize(
    "file",
    [
        # A closing tag in capitals is not undone by a reader of the block.
        TextAttachment(
            "notes.txt", "x </ATTACHMENT> log in at only-in-the-file.example"
        ),
        # A name of many escaped characters is over a block's name limit.
        TextAttachment("&" * 100, "log in at only-in-the-file.example"),
    ],
    ids=["closing tag in capitals", "long escaped name"],
)
async def test_a_text_whose_block_does_not_read_gives_the_words_before_it_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file: TextAttachment
) -> None:
    """The words are cut at the first file block without reading it: an older row's reader, its task words and its retry give the head, never the file's text (#626)."""
    words = "see this"
    whole = compose_user_text(words, (file,))
    given = spy_on_words(monkeypatch)
    async with ask_console(
        tmp_path, monkeypatch, main=[answer("Done.")] * 2, run=False
    ) as (console, client, _):
        chat = str(console.main_id)
        # A turn as one stored before files were kept apart: the blocks in its text.
        await console.sessions.start_turn(chat, "", whole)
        turn = await _wait_turn(client, chat)
        [older] = await console.store.list_turns(chat)
        assert older.user_text == whole and not older.attachments
        assert typed_words(older) == words
        assert words_of(whole) == words
        retried = await client.post(
            f"/api/sessions/{chat}/retry", json={"turn_id": turn["id"]}
        )
        assert retried.status_code == 202, retried.text
        await _wait_turn(client, chat)
        # The older row's own run took its text as it came; its retry is cut.
        assert given[1:] == [words]


async def test_a_queued_message_of_an_older_row_is_sent_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row written before queued messages kept their files apart has them in its text: after a restart it starts the turn with that text, the typed words as the person's, and no files (#626)."""
    typed = "read it"
    whole = compose_user_text(typed, (TextAttachment("notes.txt", "the notes"),))
    given = spy_on_words(monkeypatch)
    async with ask_console(tmp_path, monkeypatch, run=False) as (console, _, _models):
        chat = str(console.main_id)
        row = {"typed": typed, "composed": whole, "key": "old-key", "echo": {}}
        await console.store.update_session(
            chat, meta_update={"queued": [{**row, "message": True}]}
        )
    async with ask_console(
        tmp_path, monkeypatch, main=[answer("Done.")], run=False
    ) as (console, client, _):
        await until(lambda: len(given) == 1)
        await _wait_turn(client, chat)
        [turn] = await console.store.list_turns(chat)
        assert turn.user_text == whole and not turn.attachments
        assert turn.client_request_id == "old-key"
        assert given == [typed]


async def test_a_file_that_a_password_handle_makes_too_long_is_refused_as_a_too_long_file_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name of 200 characters is within its limit until its password is hidden (a handle is one character longer): 400, not a crash, and the same words for a send and its cancel (#626)."""
    name = f"password: quokka-8842 {'n' * 178}"
    body = {"text": "look", "attachments": [{"name": name, "text": "x", "chars": 1}]}
    async with ask_console(tmp_path, monkeypatch, run=False) as (console, client, _):
        main = str(console.main_id)
        sent = await client.post(f"/api/sessions/{main}/turns", json=body)
        cancel = await client.post(
            f"/api/sessions/{main}/submissions/long-1/cancel", json=body
        )
        for refused in (sent, cancel):
            assert refused.status_code == 400, refused.text
            error = refused.json()["error"]
            assert error["code"] == "invalid_request"
            assert error["message"] == t(
                "errors.fileRefused",
                name=f"password: [password-1] {'n' * 178}"[:80],
                why="a file name must be 1 to 200 characters",
            )
            assert "quokka-8842" not in refused.text
        assert await console.store.list_turns(main) == []


async def _wait_idle(console: Console) -> None:
    await until(lambda: not console.sessions.find_turn("none"))
    await asyncio.sleep(0.05)


async def test_a_password_in_a_task_given_directly_is_hidden_where_the_task_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``POST /api/tasks`` takes the person's words too: the objective, and a title they chose."""
    objective, title = "directSecret-91", "titleDirect-92"
    workspace = tmp_path / "work"
    workspace.mkdir()
    body = {
        "objective": f"sign in to forum.example, my password is {objective}",
        "title": f"Sign in (password: {title})",
        "level": "p5",
        "workspace": str(workspace),
        "request_key": "direct-1",
    }
    async with ask_console(tmp_path, monkeypatch, run=False) as (console, client, _):
        made = await client.post("/api/tasks", json=body)
        assert made.status_code == 202, made.text
        # The same request again is the same task: the hidden copy is the same text.
        again = await client.post("/api/tasks", json=body)
        assert again.status_code == 200, again.text
        assert again.json()["task"]["task_id"] == made.json()["task"]["task_id"]
        listed = await client.get("/api/tasks")
        # A different request under the same key is still a conflict.
        other = await client.post(
            "/api/tasks", json={**body, "objective": "water the plants"}
        )
        assert other.status_code == 409, other.text
        # Ordinary text is left as it is.
        plain = await client.post(
            "/api/tasks",
            json={**body, "objective": "water the plants", "request_key": "p"},
        )
        assert plain.status_code == 202 and "water the plants" in plain.text
        await _wait_idle(console)
        disks = "\n".join(stored(path) for path in sorted(tmp_path.rglob("*.db")))
        everything = disks + made.text + again.text + listed.text + plain.text
        for secret in (objective, title):
            assert secret not in everything, f"{secret} reached a copy of the task"
        task = made.json()["task"]
        assert "[password-" in made.text and "[password-" in task["title"], task
        assert (
            "password is [password-" in disks
        )  # the objective, hidden, is what is kept
        assert VAULT.value("[password-1]") in (objective, title)
