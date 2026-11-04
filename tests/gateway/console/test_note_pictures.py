"""Pictures that come with the words the person adds to a turn that runs.

Each test runs the console with its main chat on a scripted model. A model reply
that waits for an event keeps a turn running while the test posts notes; nothing
reaches a model server. The rules are those of a message with pictures: the same
kinds and limits, the same refusal for a model that reads none.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from localmelo.support.gateway.console.notes import NOTE_MARK
from localmelo.support.sessions.picture_input import MAX_PICTURES, MAX_UPLOAD_BYTES

from .test_ask_owner import Reply, answer, ask_console, held, until
from .test_pictures_in import FakeVision, _request_pictures, _stage, _upload, png
from .test_turn_notes import STEP, _events, _note, _queued, _seen, _start
from .test_turn_writes import _wait_turn


async def test_a_note_carries_a_picture_to_the_next_step_beside_its_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = png(40, 20)
    release = asyncio.Event()
    main: list[Reply] = [held(release, STEP), answer("A gradient.")]
    async with ask_console(
        tmp_path, monkeypatch, main=main, vision=FakeVision(True), run=False
    ) as (console, client, models):
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "start the job")
        await until(lambda: len(models["main"].requests) == 1)
        picture = await _stage(client, sent)
        response = await _note(
            client, turn_id, "use this one", images=[picture], client_request_id="n1"
        )
        assert response.status_code == 202, response.text
        note = response.json()["note"]
        assert (note["state"], note["text"]) == ("sent", "use this one")
        # The answer lists the picture the way a turn's events do, for the page to show.
        assert [(i["mime_type"], i["size_bytes"]) for i in note["images"]] == [
            ("image/png", len(sent))
        ]
        # Kept before the note was accepted: served from the chat's blobs at once.
        blob = await client.get(
            f"/api/sessions/{main_id}/blobs/{note['images'][0]['sha256']}"
        )
        assert blob.status_code == 200 and blob.content == sent
        release.set()
        finished = await _wait_turn(client, main_id)
        assert finished["status"] == "completed"
        # The second request has the words, marked, and the picture as an image part.
        request = models["main"].requests[1]
        assert _seen(request).endswith(f"{NOTE_MARK}\nuse this one")
        assert _request_pictures(request) == [sent]
        assert _request_pictures(models["main"].requests[0]) == []
        # The turn's events list it, so a reload shows it again.
        events = await _events(console, 1)
        stored = next(e for e in events if e["type"] == "turn.note")
        assert stored["images"] == note["images"]
        assert stored["note"] == note["note"]
        # Read: no next turn, and nothing left in the stored queue.
        await console.asks.settle()
        assert len(await console.store.list_turns(main_id)) == 1
        assert await _queued(console, main_id) == []


async def test_a_note_of_pictures_alone_is_read_as_a_marker_and_a_blank_note_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = png(30, 30)
    release = asyncio.Event()
    main: list[Reply] = [held(release, STEP), answer("Seen.")]
    async with ask_console(
        tmp_path, monkeypatch, main=main, vision=FakeVision(True), run=False
    ) as (console, client, models):
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        blank = await _note(client, turn_id, "   ")
        assert blank.status_code == 400
        assert blank.json()["error"]["code"] == "invalid_request"
        picture = await _stage(client, sent)
        response = await _note(client, turn_id, "  ", images=[picture])
        assert response.status_code == 202, response.text
        assert response.json()["note"]["text"] == ""
        release.set()
        await _wait_turn(client, str(console.main_id))
        request = models["main"].requests[1]
        assert _seen(request).endswith(f"{NOTE_MARK}\n[1 picture]")
        assert _request_pictures(request) == [sent]


async def test_a_note_judges_its_pictures_as_a_message_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    never = asyncio.Event()
    main: list[Reply] = [held(never, answer("Done."))]
    async with ask_console(
        tmp_path, monkeypatch, main=main, vision=FakeVision(True), run=False
    ) as (console, client, models):
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        # A wrong type and a too big one never become pictures to send.
        text = await _upload(client, b"just some words, not a picture")
        assert text.status_code == 415
        assert text.json()["error"]["code"] == "picture_unsupported"
        big = await _upload(client, b"\x89PNG\r\n\x1a\n" + b"0" * MAX_UPLOAD_BYTES)
        assert big.status_code == 413
        ids = [await _stage(client, png(10 + n, 10)) for n in range(MAX_PICTURES + 1)]
        many = await _note(client, turn_id, "all", images=ids)
        assert many.status_code == 400
        assert str(MAX_PICTURES) in many.json()["error"]["message"]
        gone = await _note(client, turn_id, "x", images=["0" * 32])
        assert gone.status_code == 400
        assert gone.json()["error"]["code"] == "picture_gone"
        twice = await _note(client, turn_id, "x", images=[ids[0], ids[0]])
        assert twice.status_code == 400
        # Nothing was kept for any of them.
        assert await _queued(console, main_id) == []
        # Four pictures are a note too.
        four = await _note(client, turn_id, "four", images=ids[:MAX_PICTURES])
        assert four.status_code == 202, four.text
        assert len(four.json()["note"]["images"]) == MAX_PICTURES


async def test_a_note_with_a_picture_is_refused_by_a_model_that_reads_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    never = asyncio.Event()
    main: list[Reply] = [held(never, answer("Done."))]
    async with ask_console(
        tmp_path, monkeypatch, main=main, vision=FakeVision(False), run=False
    ) as (console, client, models):
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        picture = await _stage(client)
        refused = await _note(client, turn_id, "look", images=[picture])
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "model_text_only"
        assert await _queued(console, str(console.main_id)) == []
        # Words alone still join the work.
        assert (await _note(client, turn_id, "just words")).status_code == 202


async def test_an_unknown_model_gets_a_picture_of_a_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = png(24, 24)
    release = asyncio.Event()
    main: list[Reply] = [held(release, STEP), answer("ok")]
    async with ask_console(
        tmp_path, monkeypatch, main=main, vision=FakeVision(None), run=False
    ) as (console, client, models):
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        picture = await _stage(client, sent)
        assert (
            await _note(client, turn_id, "look", images=[picture])
        ).status_code == 202
        release.set()
        await _wait_turn(client, str(console.main_id))
        assert _request_pictures(models["main"].requests[1]) == [sent]


async def test_a_model_found_text_only_before_the_note_is_read_is_told_not_sent_the_picture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [held(release, STEP), answer("Understood.")]
    vision = FakeVision(True)
    async with ask_console(
        tmp_path, monkeypatch, main=main, vision=vision, run=False
    ) as (console, client, models):
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        picture = await _stage(client)
        assert (
            await _note(client, turn_id, "see this", images=[picture])
        ).status_code == 202
        # The model setting changed, or a refusal taught the gateway, before the next step.
        vision.verdict = False
        release.set()
        finished = await _wait_turn(client, str(console.main_id))
        assert finished["status"] == "completed"
        request = models["main"].requests[1]
        assert _request_pictures(request) == []
        words = _seen(request)
        assert "see this" in words
        assert "You cannot see this picture" in words
        assert "the person can" in words


async def test_a_note_with_pictures_after_the_last_step_runs_as_the_next_turn_with_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = png(40, 20)
    release = asyncio.Event()
    main: list[Reply] = [held(release, answer("All done.")), answer("A gradient.")]
    async with ask_console(
        tmp_path, monkeypatch, main=main, vision=FakeVision(True), run=False
    ) as (console, client, models):
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "start the job")
        await until(lambda: len(models["main"].requests) == 1)
        picture = await _stage(client, sent)
        # The only step is under way: nothing will read the note before the turn ends.
        response = await _note(client, turn_id, "and this", images=[picture])
        assert response.status_code == 202, response.text
        release.set()

        async def ran() -> bool:
            turns = await console.store.list_turns(main_id)
            return len(turns) == 2 and turns[1].status != "running"

        await until(ran)
        first, second = await console.store.list_turns(main_id)
        assert first.status == "completed"
        assert (second.user_text, second.typed_text) == ("and this", "and this")
        # The next turn keeps the picture as a message does, and its run was sent it.
        kept = await console.store.turn_images(main_id, 2)
        assert [i.mime_type for i in kept] == ["image/png"]
        assert _request_pictures(models["main"].requests[1]) == [sent]
        await console.asks.settle()
        assert await _queued(console, main_id) == []


async def test_a_picture_note_that_becomes_a_message_has_its_marker_when_it_has_no_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [held(release, answer("All done.")), answer("I see it.")]
    async with ask_console(
        tmp_path, monkeypatch, main=main, vision=FakeVision(True), run=False
    ) as (console, client, models):
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        picture = await _stage(client)
        assert (await _note(client, turn_id, "", images=[picture])).status_code == 202
        release.set()

        async def ran() -> bool:
            turns = await console.store.list_turns(main_id)
            return len(turns) == 2 and turns[1].status != "running"

        await until(ran)
        _, second = await console.store.list_turns(main_id)
        assert second.user_text == "[1 picture]"
        assert len(await console.store.turn_images(main_id, 2)) == 1


async def test_a_picture_note_unread_before_a_restart_runs_after_it_with_its_picture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = png(40, 20)
    never = asyncio.Event()
    async with ask_console(
        tmp_path,
        monkeypatch,
        main=[held(never, answer("stalled"))],
        vision=FakeVision(True),
        run=False,
    ) as (console, client, models):
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        picture = await _stage(client, sent)
        assert (
            await _note(client, turn_id, "after the restart", images=[picture])
        ).status_code == 202
        info = await console.store.get_session(main_id)
        assert info is not None
        [queued] = info.meta["queued"]
        assert queued["typed"] == "after the restart"
        assert [i["mime_type"] for i in queued["images"]] == ["image/png"]
    async with ask_console(
        tmp_path,
        monkeypatch,
        main=[answer("Hi again.")],
        vision=FakeVision(True),
        run=False,
    ) as (console, client, models):

        async def ran() -> bool:
            turns = await console.store.list_turns(main_id)
            return turns[-1].user_text == "after the restart"

        await until(ran)
        await _wait_turn(client, main_id)
        assert _request_pictures(models["main"].requests[0]) == [sent]
        assert len(await console.store.turn_images(main_id, 2)) == 1
        info = await console.store.get_session(main_id)
        assert info is not None and info.meta.get("queued") is None


async def test_stop_hands_the_pictures_of_unread_notes_back_to_be_sent_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = png(40, 20)
    never = asyncio.Event()
    async with ask_console(
        tmp_path,
        monkeypatch,
        main=[held(never, answer("stalled")), answer("Seen again.")],
        vision=FakeVision(True),
        run=False,
    ) as (console, client, models):
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        picture = await _stage(client, sent)
        await _note(client, turn_id, "not that way", images=[picture])
        await _note(client, turn_id, "only words")
        stopped = await client.post(f"/api/turns/{turn_id}/cancel")
        assert stopped.status_code == 202
        back = stopped.json()["unread_notes"]
        assert [n["text"] for n in back] == ["not that way", "only words"]
        assert "images" not in back[1]
        [given] = back[0]["images"]
        # A staged picture again: the composer shows it and sends it with a message.
        assert given["mime_type"] == "image/png" and given["id"] != picture
        shown = await client.get(f"/api/uploads/{given['id']}")
        assert shown.status_code == 200 and shown.content == sent
        await _wait_turn(client, main_id)
        await console.asks.settle()
        assert len(await console.store.list_turns(main_id)) == 1
        assert await _queued(console, main_id) == []
        sent_again = await client.post(
            f"/api/sessions/{main_id}/turns",
            json={"text": "not that way", "images": [given["id"]]},
        )
        assert sent_again.status_code == 202, sent_again.text
        await _wait_turn(client, main_id)
        assert _request_pictures(models["main"].requests[1]) == [sent]


async def test_a_repeated_send_of_a_picture_note_is_one_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main: list[Reply] = [held(release, STEP), answer("Done.")]
    async with ask_console(
        tmp_path, monkeypatch, main=main, vision=FakeVision(True), run=False
    ) as (console, client, models):
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        picture = await _stage(client)
        one = await _note(
            client, turn_id, "once", images=[picture], client_request_id="same"
        )
        again = await _note(
            client, turn_id, "once", images=[picture], client_request_id="same"
        )
        assert one.json() == again.json()
        release.set()
        await _wait_turn(client, str(console.main_id))
        events = await _events(console, 1)
        assert [e["type"] for e in events].count("turn.note") == 1
        assert len(_request_pictures(models["main"].requests[1])) == 1


async def test_notes_with_many_pictures_are_read_over_more_than_one_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step reads the notes whose pictures fit what a request keeps; the others wait for the next step."""
    first, second = asyncio.Event(), asyncio.Event()
    main: list[Reply] = [
        held(first, STEP),
        held(second, STEP),
        answer("Read all three."),
    ]
    async with ask_console(
        tmp_path, monkeypatch, main=main, vision=FakeVision(True), run=False
    ) as (console, client, models):
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        shots = []
        for n in range(3):
            ids = [await _stage(client, png(10 + n * 4 + k, 10)) for k in range(4)]
            shots.append(ids)
            assert (
                await _note(client, turn_id, f"note {n}", images=ids)
            ).status_code == 202
        first.set()
        await until(lambda: len(models["main"].requests) == 2)
        seen = _seen(models["main"].requests[1])
        # Two notes of four pictures fill a step; the third comes with the next one.
        assert "note 0" in seen and "note 1" in seen and "note 2" not in seen
        assert len(_request_pictures(models["main"].requests[1])) == 8
        second.set()
        await _wait_turn(client, str(console.main_id))
        assert "note 2" in _seen(models["main"].requests[2])
        await console.asks.settle()
        assert len(await console.store.list_turns(str(console.main_id))) == 1


async def test_a_note_in_a_side_chat_keeps_its_picture_in_that_chat_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = png(16, 16)
    main_release, side_release = asyncio.Event(), asyncio.Event()
    main: list[Reply] = [held(main_release, STEP), answer("Main done.")]
    side: list[Reply] = [held(side_release, STEP), answer("Side done.")]
    async with ask_console(
        tmp_path,
        monkeypatch,
        main=main,
        sides=[side],
        vision=FakeVision(True),
        run=False,
    ) as (console, client, models):
        main_id = str(console.main_id)
        await _start(client, console, "main work")
        created = await client.post(
            "/api/sessions",
            json={"turn": {"text": "side work"}, "workspace": str(tmp_path / "work")},
        )
        side_id = created.json()["session"]["id"]
        side_turn = str(created.json()["turn_id"])
        await until(
            lambda: len(models["main"].requests) == 1
            and side_id in models
            and models[side_id].requests
        )
        picture = await _stage(client, sent)
        response = await _note(client, side_turn, "for the side", images=[picture])
        assert response.status_code == 202, response.text
        sha = response.json()["note"]["images"][0]["sha256"]
        assert (
            await client.get(f"/api/sessions/{side_id}/blobs/{sha}")
        ).status_code == 200
        assert (
            await client.get(f"/api/sessions/{main_id}/blobs/{sha}")
        ).status_code == 404
        side_release.set()
        await _wait_turn(client, side_id)
        main_release.set()
        await _wait_turn(client, main_id)
        assert _request_pictures(models[side_id].requests[1]) == [sent]
        assert all(_request_pictures(r) == [] for r in models["main"].requests)
