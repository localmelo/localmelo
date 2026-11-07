"""Pictures that come with the words the person adds to a turn that runs.

Each test runs the console with its main chat on a scripted model. A model reply
that waits for an event keeps a turn running while the test posts notes; nothing
reaches a model server. The rules are those of a message with pictures: the same
kinds and limits, the same refusal for a model that reads none.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from localmelo.support.gateway.console.asks import MAX_NOTES
from localmelo.support.gateway.console.notes import NOTE_MARK
from localmelo.support.sessions.picture_input import MAX_PICTURES, MAX_UPLOAD_BYTES

from .test_ask_owner import Reply, answer, ask_console, held, until
from .test_pictures_in import FakeVision, _request_pictures, _stage, _upload, png
from .test_turn_notes import (
    STEP,
    _events,
    _given_back,
    _note,
    _queued,
    _seen,
    _start,
)
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


async def test_a_note_its_turn_ended_without_goes_back_with_its_pictures(
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
        await _note(client, turn_id, "only words")
        release.set()
        await _wait_turn(client, main_id)
        await until(lambda: _given_back(client, main_id))
        await console.asks.settle()
        assert len(await console.store.list_turns(main_id)) == 1
        assert await _queued(console, main_id) == []
        back = await _given_back(client, main_id)
        assert [e["typed"] for e in back] == ["and this", "only words"]
        assert "images" not in back[1]
        [given] = back[0]["images"]
        # A staged picture again: the composer shows it and sends it with a message.
        assert given["mime_type"] == "image/png" and given["id"] != picture
        shown = await client.get(f"/api/uploads/{given['id']}")
        assert shown.status_code == 200 and shown.content == sent
        sent_again = await client.post(
            f"/api/sessions/{main_id}/turns",
            json={"text": "and this", "images": [given["id"]]},
        )
        assert sent_again.status_code == 202, sent_again.text
        await _wait_turn(client, main_id)
        assert _request_pictures(models["main"].requests[1]) == [sent]


async def test_a_note_of_a_picture_alone_goes_back_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    async with ask_console(
        tmp_path, monkeypatch, main=[held(release, answer("All done."))],
        vision=FakeVision(True), run=False,
    ) as (console, client, models):  # fmt: skip
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        picture = await _stage(client)
        assert (await _note(client, turn_id, "", images=[picture])).status_code == 202
        release.set()
        await _wait_turn(client, main_id)
        await until(lambda: _given_back(client, main_id))
        [back] = await _given_back(client, main_id)
        assert back["typed"] == "" and len(back["images"]) == 1


@pytest.mark.parametrize("words", ["after the restart", ""])
async def test_a_picture_note_a_restart_finds_starts_nothing_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, words: str
) -> None:
    never = asyncio.Event()
    async with ask_console(
        tmp_path, monkeypatch, main=[held(never, answer("stalled"))],
        vision=FakeVision(True), run=False,
    ) as (console, client, models):  # fmt: skip
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        picture = await _stage(client, png(40, 20))
        assert (
            await _note(client, turn_id, words, images=[picture])
        ).status_code == 202
    async with ask_console(
        tmp_path, monkeypatch, main=[answer("Hi again.")], vision=FakeVision(True),
        run=False,
    ) as (console, client, models):  # fmt: skip
        await until(lambda: _given_back(client, main_id))
        await console.asks.settle()
        assert models["main"].requests == []
        assert len(await console.store.list_turns(main_id)) == 1
        assert await _queued(console, main_id) == []
        # Its picture is not recovered, and the row does not pretend it is.
        [told] = await _given_back(client, main_id)
        assert told["typed"] == words and "images" not in told
        assert "picture did not come back" in told["line"]


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


def _held_blobs(
    console: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[asyncio.Event, asyncio.Event, list[int]]:
    """The store keeps a picture only once the test lets it: (one is waiting, let them go, how many came)."""
    put, entered, go, came = (
        console.store.put_blobs,
        asyncio.Event(),
        asyncio.Event(),
        [],
    )

    async def slow(*args: Any, **kwargs: Any) -> Any:
        came.append(1)
        entered.set()
        await go.wait()
        return await put(*args, **kwargs)

    monkeypatch.setattr(console.store, "put_blobs", slow)
    return entered, go, came


@pytest.mark.parametrize("later_stop", [False, True])
async def test_a_picture_note_its_turn_ended_without_on_the_way_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, later_stop: bool
) -> None:
    """However long the store takes, and whatever turns come and go meanwhile."""
    never, later = asyncio.Event(), asyncio.Event()
    async with ask_console(
        tmp_path, monkeypatch, vision=FakeVision(True), run=False,
        main=[held(never, answer("x")), held(later, answer("y"))],
    ) as (console, client, models):  # fmt: skip
        main_id = str(console.main_id)
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        picture = await _stage(client, png(4, 4))
        entered, go, _ = _held_blobs(console, monkeypatch)
        sending = asyncio.create_task(
            _note(client, turn_id, "and this", images=[picture])
        )
        await entered.wait()
        assert (await client.post(f"/api/turns/{turn_id}/cancel")).status_code == 202
        await _wait_turn(client, main_id)
        turns = 1
        if later_stop:
            second = await _start(client, console, "something else")
            await until(lambda: len(models["main"].requests) == 2)
            assert (await client.post(f"/api/turns/{second}/cancel")).status_code == 202
            await _wait_turn(client, main_id)
            turns = 2
        go.set()
        answered = await sending
        assert answered.status_code == 409, answered.text
        assert answered.json()["error"]["code"] == "turn_not_running"
        await console.asks.settle()
        await asyncio.sleep(0.1)
        assert len(await console.store.list_turns(main_id)) == turns
        assert await _queued(console, main_id) == []
        # The picture is still staged: the composer sends it again with a message.
        assert console.pictures.resolve([picture])


async def test_pictured_notes_at_once_cannot_pass_the_limit_of_a_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    never = asyncio.Event()
    async with ask_console(
        tmp_path, monkeypatch, main=[held(never, answer("x"))], vision=FakeVision(True),
        run=False,
    ) as (console, client, models):  # fmt: skip
        turn_id = await _start(client, console, "go")
        await until(lambda: len(models["main"].requests) == 1)
        for n in range(MAX_NOTES - 1):
            assert (await _note(client, turn_id, f"note {n}")).status_code == 202
        pictures = [await _stage(client, png(4, 4)) for _ in range(2)]
        entered, go, _ = _held_blobs(console, monkeypatch)
        sending = [
            asyncio.create_task(_note(client, turn_id, "seen", images=[picture]))
            for picture in pictures
        ]
        await entered.wait()
        await asyncio.sleep(0.05)
        go.set()
        codes = sorted([(await send).status_code for send in sending])
        assert codes == [202, 400]
