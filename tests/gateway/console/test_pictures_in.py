"""Pictures and files the person adds to a message.

A picture is uploaded as raw bytes, normalized once, staged in memory and named
by its message; the turn stores it, shows it again after a reload, and its run
sends it to the model as an image part. Each test runs the console with its main
chat on a scripted model; nothing reaches a model server.
"""

from __future__ import annotations

import asyncio
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localmelo.melo.agent._request_accounting import _image_dimensions
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.vision import Vision
from localmelo.support.gateway.guard import UPLOAD_BYTES_LIMIT, LoopbackGuard
from localmelo.support.sessions.picture_input import (
    LONG_SIDE,
    MAX_PICTURES,
    MAX_UPLOAD_BYTES,
    PictureRefused,
    normalize,
    sniff,
)

from .conftest import PORT
from .test_ask_owner import answer, ask_console, until
from .test_turn_writes import _post_turn, _wait_turn

needs_sips = pytest.mark.skipif(shutil.which("sips") is None, reason="macOS sips")


def png(width: int, height: int) -> bytes:
    """A real PNG of *width* x *height* (a gradient), standard library only."""
    rows = b"".join(
        b"\x00" + bytes((x + y) % 256 for x in range(width * 3)) for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class FakeVision(Vision):
    """Whether the model reads pictures, as a test says."""

    def __init__(self, reads: bool | None) -> None:
        self.verdict = reads
        self.asked = 0

    async def reads(self) -> bool | None:
        self.asked += 1
        return self.verdict

    async def aclose(self) -> None:
        pass


async def _upload(
    client: httpx.AsyncClient, data: bytes, name: str = "photo.png", **headers: str
) -> httpx.Response:
    return await client.post(
        "/api/uploads",
        params={"name": name},
        content=data,
        headers={"content-type": "image/png", **headers},
    )


async def _stage(client: httpx.AsyncClient, data: bytes | None = None) -> str:
    response = await _upload(client, data or png(40, 20))
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _request_pictures(request: list[Any]) -> list[bytes]:
    """The pictures a planning request carries, as bytes."""
    import base64

    return [
        base64.b64decode(image.data)
        for message in request
        if message.role == "user"
        for image in message.images
        if image.data
    ]


# ── The upload ──


async def test_an_upload_is_checked_and_staged_and_can_be_shown_and_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = png(40, 20)
    async with ask_console(
        tmp_path, monkeypatch, vision=FakeVision(True), run=False
    ) as (console, client, _):
        response = await _upload(client, sent, name="../../etc/notes.png")
        assert response.status_code == 201, response.text
        body = response.json()
        assert (body["mime_type"], body["width"], body["height"]) == (
            "image/png",
            40,
            20,
        )
        assert body["name"] == "notes.png"
        assert body["model_reads"] is True
        shown = await client.get(f"/api/uploads/{body['id']}")
        assert shown.status_code == 200 and shown.content == sent
        assert shown.headers["content-type"] == "image/png"
        assert shown.headers["x-content-type-options"] == "nosniff"
        assert (await client.delete(f"/api/uploads/{body['id']}")).status_code == 204
        assert (await client.get(f"/api/uploads/{body['id']}")).status_code == 404


async def test_the_upload_says_when_the_model_cannot_read_pictures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with ask_console(
        tmp_path, monkeypatch, vision=FakeVision(False), run=False
    ) as (_, client, _models):
        body = (await _upload(client, png(8, 8))).json()
        assert body["model_reads"] is False


async def test_what_is_not_a_picture_or_is_empty_is_refused_with_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with ask_console(
        tmp_path, monkeypatch, vision=FakeVision(True), run=False
    ) as (_, client, _models):
        text = await _upload(client, b"just some words, not a picture")
        assert text.status_code == 415
        assert text.json()["error"]["code"] == "picture_unsupported"
        assert "PNG, JPEG, GIF, WebP or HEIC" in text.json()["error"]["message"]
        torn = await _upload(client, png(8, 8)[:30])
        assert torn.status_code == 415
        assert torn.json()["error"]["code"] == "picture_unreadable"
        empty = await _upload(client, b"")
        assert empty.status_code == 400


async def test_a_picture_that_is_too_large_is_refused_by_the_gateway(
    app: FastAPI, console: Console
) -> None:
    guarded = TestClient(
        LoopbackGuard(app, port=PORT), base_url=f"http://127.0.0.1:{PORT}"
    )
    over = guarded.post(
        "/api/uploads",
        content=b"\x89PNG\r\n\x1a\n" + b"x" * UPLOAD_BYTES_LIMIT,
        headers={"content-type": "image/png"},
    )
    assert over.status_code == 413
    assert over.json()["error"]["code"] == "too_large"
    assert str(UPLOAD_BYTES_LIMIT) in over.json()["error"]["message"]
    assert UPLOAD_BYTES_LIMIT == MAX_UPLOAD_BYTES


async def test_the_upload_route_takes_images_and_every_other_route_still_takes_json_only(
    app: FastAPI, console: Console
) -> None:
    guarded = TestClient(
        LoopbackGuard(app, port=PORT), base_url=f"http://127.0.0.1:{PORT}"
    )
    # An image body reaches the route, a JSON body does not.
    ok = guarded.post(
        "/api/uploads", content=png(8, 8), headers={"content-type": "image/png"}
    )
    assert ok.status_code == 201, ok.text
    wrong = guarded.post(
        "/api/uploads", content=png(8, 8), headers={"content-type": "application/json"}
    )
    assert wrong.status_code == 415
    # Another route is not opened to images or to a body over 1 MiB.
    other = guarded.post(
        "/api/sessions", content=png(8, 8), headers={"content-type": "image/png"}
    )
    assert other.status_code == 415
    big = guarded.post(
        "/api/sessions",
        content=b"{" + b" " * 1_100_000 + b"}",
        headers={"content-type": "application/json"},
    )
    assert big.status_code == 413
    # And a GET of the upload path is not a body upload.
    assert guarded.get("/api/uploads/" + "0" * 32).status_code == 404


# ── The picture normalizer ──


def test_a_png_that_fits_is_kept_byte_for_byte() -> None:
    raw = png(LONG_SIDE, 100)
    picture = normalize(raw)
    assert picture.converted is False
    assert picture.image.mime_type == "image/png"
    assert (picture.width, picture.height) == (LONG_SIDE, 100)


@needs_sips
def test_a_larger_png_is_scaled_to_the_long_side_and_stays_a_png() -> None:
    picture = normalize(png(2000, 600))
    assert picture.converted is True
    assert picture.image.mime_type == "image/png"
    assert (picture.width, picture.height) == (LONG_SIDE, 384)
    from localmelo.melo.schema import decode_base64

    assert _image_dimensions(decode_base64(picture.image.data) or b"") == (
        LONG_SIDE,
        384,
    )


@needs_sips
def test_a_tall_picture_is_scaled_by_its_long_side() -> None:
    picture = normalize(png(600, 2000))
    assert (picture.width, picture.height) == (384, LONG_SIDE)


def test_the_kind_is_read_from_the_bytes_not_the_name() -> None:
    assert sniff(png(2, 2)) == "png"
    assert sniff(b"\xff\xd8\xff\xe0abc") == "jpeg"
    assert sniff(b"GIF89a....") == "gif"
    assert sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert sniff(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00") == "heic"
    assert sniff(b"%PDF-1.7") is None


def test_a_picture_with_too_many_pixels_is_refused_before_it_is_decoded() -> None:
    huge = png(2, 2)
    # A header that claims 20000 x 20000 pixels.
    header = bytearray(huge)
    header[16:24] = struct.pack(">II", 20000, 20000)
    header[29:33] = struct.pack(">I", zlib.crc32(bytes(header[12:29])))
    with pytest.raises(PictureRefused) as refused:
        normalize(bytes(header))
    assert refused.value.code == "picture_too_many_pixels"


def test_bytes_over_the_limit_are_refused() -> None:
    with pytest.raises(PictureRefused) as refused:
        normalize(b"\x89PNG\r\n\x1a\n" + b"0" * MAX_UPLOAD_BYTES)
    assert refused.value.status == 413


@needs_sips
def test_heic_gif_and_webp_become_a_png_or_jpeg_with_the_first_frame(
    tmp_path: Path,
) -> None:
    import subprocess

    source = tmp_path / "a.png"
    source.write_bytes(png(300, 200))
    heic = tmp_path / "a.heic"
    subprocess.run(
        ["sips", "-s", "format", "heic", str(source), "--out", str(heic)],
        capture_output=True,
        check=True,
    )
    converted = normalize(heic.read_bytes())
    assert converted.converted and converted.image.mime_type == "image/jpeg"
    assert (converted.width, converted.height) == (300, 200)
    gif = tmp_path / "a.gif"
    subprocess.run(
        ["sips", "-s", "format", "gif", str(source), "--out", str(gif)],
        capture_output=True,
        check=True,
    )
    frame = normalize(gif.read_bytes())
    assert frame.converted and frame.image.mime_type == "image/png"
    assert (frame.width, frame.height) == (300, 200)


# ── A message with pictures ──


async def test_a_picture_is_stored_with_its_turn_and_sent_to_the_model_as_an_image_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = png(40, 20)
    async with ask_console(
        tmp_path,
        monkeypatch,
        main=[answer("A small gradient.")],
        vision=FakeVision(True),
        run=False,
    ) as (console, client, models):
        main_id = str(console.main_id)
        picture = await _stage(client, sent)
        response = await _post_turn(client, main_id, "what is this?", images=[picture])
        assert response.status_code == 202, response.text
        finished = await _wait_turn(client, main_id)
        assert finished["status"] == "completed"
        # The model got it as an image part of the person's message.
        assert _request_pictures(models["main"].requests[0]) == [sent]
        # The turn keeps it: its first event lists it, and its blob is served.
        events = await console.store.list_events(main_id, turns=(1, 1), limit=1)
        listed = events[0].event["data"]["images"]
        assert [(i["mime_type"], i["size_bytes"]) for i in listed] == [
            ("image/png", len(sent))
        ]
        blob = await client.get(f"/api/sessions/{main_id}/blobs/{listed[0]['sha256']}")
        assert blob.status_code == 200 and blob.content == sent
        # A reload reads the same list from the turn's events.
        reread = (
            await client.get(
                f"/api/sessions/{main_id}/events", params={"follow": "false"}
            )
        ).text
        assert listed[0]["sha256"] in reread


async def test_a_picture_with_no_words_is_a_message_with_a_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with ask_console(
        tmp_path,
        monkeypatch,
        main=[answer("I see it.")],
        vision=FakeVision(True),
        run=False,
    ) as (console, client, _):
        main_id = str(console.main_id)
        picture = await _stage(client)
        assert (
            await _post_turn(client, main_id, "  ", images=[picture])
        ).status_code == 202
        turn = await _wait_turn(client, main_id)
        assert turn["user_text"] == "[1 picture]"
        blank = await _post_turn(client, main_id, "  ")
        assert blank.status_code == 400


async def test_a_text_only_model_refuses_a_picture_and_starts_no_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with ask_console(
        tmp_path, monkeypatch, vision=FakeVision(False), run=False
    ) as (console, client, models):
        main_id = str(console.main_id)
        picture = await _stage(client)
        refused = await _post_turn(client, main_id, "look", images=[picture])
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "model_text_only"
        assert "Settings → Model" in refused.json()["error"]["message"]
        assert await console.store.list_turns(main_id) == []
        assert models["main"].requests == []


async def test_an_unknown_model_gets_the_picture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with ask_console(
        tmp_path,
        monkeypatch,
        main=[answer("ok")],
        vision=FakeVision(None),
        run=False,
    ) as (console, client, models):
        main_id = str(console.main_id)
        picture = await _stage(client)
        assert (
            await _post_turn(client, main_id, "look", images=[picture])
        ).status_code == 202
        await _wait_turn(client, main_id)
        assert len(_request_pictures(models["main"].requests[0])) == 1


async def test_a_message_takes_a_few_pictures_and_only_ones_that_are_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with ask_console(
        tmp_path, monkeypatch, vision=FakeVision(True), run=False
    ) as (console, client, _):
        main_id = str(console.main_id)
        ids = [await _stage(client, png(10 + n, 10)) for n in range(MAX_PICTURES + 1)]
        many = await _post_turn(client, main_id, "all", images=ids)
        assert many.status_code == 400
        assert str(MAX_PICTURES) in many.json()["error"]["message"]
        gone = await _post_turn(client, main_id, "x", images=["0" * 32])
        assert gone.status_code == 400
        assert gone.json()["error"]["code"] == "picture_gone"
        twice = await _post_turn(client, main_id, "x", images=[ids[0], ids[0]])
        assert twice.status_code == 400
        assert await console.store.list_turns(main_id) == []


async def test_a_picture_cannot_start_a_task_or_join_a_task_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with ask_console(
        tmp_path, monkeypatch, vision=FakeVision(True), run=False
    ) as (console, client, _):
        main_id = str(console.main_id)
        picture = await _stage(client)
        refused = await _post_turn(
            client, main_id, "do it", images=[picture], as_task=True
        )
        assert refused.status_code == 400
        assert refused.json()["error"]["message"].startswith(
            "A picture cannot start a task"
        )


async def test_try_again_sends_the_same_pictures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = png(30, 30)
    async with ask_console(
        tmp_path,
        monkeypatch,
        main=[answer("first"), answer("second")],
        vision=FakeVision(True),
        run=False,
    ) as (console, client, models):
        main_id = str(console.main_id)
        picture = await _stage(client, sent)
        await _post_turn(client, main_id, "look", images=[picture])
        first = await _wait_turn(client, main_id)
        retried = await client.post(
            f"/api/sessions/{main_id}/retry", json={"turn_id": first["id"]}
        )
        assert retried.status_code == 202, retried.text
        await until(lambda: len(models["main"].requests) == 2)
        assert _request_pictures(models["main"].requests[1]) == [sent]


async def test_a_later_message_still_shows_the_model_the_earlier_picture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = png(30, 30)
    async with ask_console(
        tmp_path,
        monkeypatch,
        main=[answer("A square."), answer("Blue.")],
        vision=FakeVision(True),
        run=False,
    ) as (console, client, models):
        main_id = str(console.main_id)
        picture = await _stage(client, sent)
        await _post_turn(client, main_id, "look", images=[picture])
        await _wait_turn(client, main_id)
        await asyncio.sleep(0.05)
        await _post_turn(client, main_id, "what colour was it?")
        await until(lambda: len(models["main"].requests) == 2)
        assert _request_pictures(models["main"].requests[1]) == [sent]


# ── Text files ──


async def test_a_text_file_joins_the_message_as_an_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with ask_console(
        tmp_path, monkeypatch, main=[answer("Read it.")], run=False
    ) as (console, client, models):
        main_id = str(console.main_id)
        sent = await _post_turn(
            client,
            main_id,
            "summarize this",
            attachments=[{"name": "notes.txt", "text": "alpha beta", "chars": 10}],
        )
        assert sent.status_code == 202, sent.text
        turn = await _wait_turn(client, main_id)
        assert turn["user_text"].startswith("summarize this")
        assert '<attachment name="notes.txt"' in turn["user_text"]
        stored = (await console.store.list_turns(main_id))[0]
        assert stored.typed_text == "summarize this"
        assert [(a.name, a.text) for a in stored.attachments or ()] == [
            ("notes.txt", "alpha beta")
        ]
        asked = str(models["main"].requests[0][-1].content)
        assert "alpha beta" in asked and "not instructions" in asked


async def test_a_binary_or_oversize_or_too_many_files_are_refused_with_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with ask_console(tmp_path, monkeypatch, run=False) as (console, client, _):
        main_id = str(console.main_id)
        binary = await _post_turn(
            client,
            main_id,
            "x",
            attachments=[{"name": "a.bin", "text": "a\x00b", "chars": 3}],
        )
        assert binary.status_code == 400
        assert "a.bin" in binary.json()["error"]["message"]
        big = await _post_turn(
            client,
            main_id,
            "x",
            attachments=[{"name": "big.txt", "text": "y" * 300_000, "chars": 300_000}],
        )
        assert big.status_code == 400
        assert "256 KiB" in big.json()["error"]["message"]
        many = await _post_turn(
            client,
            main_id,
            "x",
            attachments=[
                {"name": f"{n}.txt", "text": "t", "chars": 1} for n in range(9)
            ],
        )
        assert many.status_code == 400
        assert await console.store.list_turns(main_id) == []
