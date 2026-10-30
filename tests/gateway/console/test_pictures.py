"""``GET /api/sessions/{id}/pictures``: confinement, content types, the size cap and headers."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localmelo.support.gateway.auth import GatewayAuth
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.guard import LoopbackGuard
from localmelo.support.sessions import pictures

from .conftest import PORT, seed_turn

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"
JPEG = b"\xff\xd8\xff\xe0" + b"pixels"
GIF = b"GIF89a" + b"pixels"
WEBP = b"RIFF\x00\x00\x00\x00WEBPpixels"
SVG = (
    b"<?xml version='1.0'?><svg xmlns='http://www.w3.org/2000/svg'>"
    b"<script>alert(1)</script></svg>"
)
SVG_POLICY = "sandbox; default-src 'none'; style-src 'unsafe-inline'"


async def session_in(console: Console, workspace: Path) -> str:
    return (await seed_turn(console.store, str(workspace))).session_id


def put(path: Path, data: bytes = PNG) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def ask(client: TestClient, session_id: str, path: str | Path) -> httpx.Response:
    resp: httpx.Response = client.get(
        f"/api/sessions/{session_id}/pictures", params={"path": str(path)}
    )
    return resp


def code_of(resp: httpx.Response) -> str:
    return str(resp.json()["error"]["code"])


def guarded(app: FastAPI, auth: GatewayAuth) -> TestClient:
    return TestClient(
        LoopbackGuard(app, port=PORT, auth=auth), base_url=f"http://127.0.0.1:{PORT}"
    )


# ── Confinement ──


async def test_a_picture_inside_the_folder_is_served(
    client: TestClient, console: Console, workspace: str
) -> None:
    root = Path(workspace)
    put(root / "chart.png")
    put(root / "out" / "deep.png", JPEG)
    sid = await session_in(console, root)
    resp = ask(client, sid, "chart.png")
    assert resp.status_code == 200 and resp.content == PNG
    assert resp.headers["content-type"] == "image/png"
    assert ask(client, sid, "out/deep.png").content == JPEG
    assert ask(client, sid, root / "chart.png").status_code == 200


@pytest.mark.parametrize(
    "requested", ["../outside.png", "a/../../outside.png", "x/../chart.png"]
)
async def test_a_dot_dot_path_is_refused(
    client: TestClient, console: Console, workspace: str, requested: str
) -> None:
    root = Path(workspace)
    put(root / "chart.png")
    put(root.parent / "outside.png")
    resp = ask(client, await session_in(console, root), requested)
    assert resp.status_code == 403
    assert code_of(resp) == "outside_workspace"


async def test_an_absolute_path_outside_the_folder_is_refused(
    client: TestClient, console: Console, workspace: str
) -> None:
    root = Path(workspace)
    outside = put(root.parent / "outside.png")
    sid = await session_in(console, root)
    assert ask(client, sid, outside).status_code == 403
    assert ask(client, sid, "/etc/hosts").status_code == 403


async def test_a_link_that_leads_out_is_refused(
    client: TestClient, console: Console, workspace: str
) -> None:
    root = Path(workspace)
    outside = put(root.parent / "elsewhere" / "secret.png")
    (root / "file.png").symlink_to(outside)
    (root / "dir").symlink_to(outside.parent)
    sid = await session_in(console, root)
    assert ask(client, sid, "file.png").status_code == 403
    assert ask(client, sid, "dir/secret.png").status_code == 403


async def test_another_sessions_folder_is_refused(
    client: TestClient, console: Console, workspace: str, tmp_path: Path
) -> None:
    second = tmp_path / "second"
    theirs = put(second / "theirs.png")
    sid = await session_in(console, Path(workspace))
    await session_in(console, second)
    assert ask(client, sid, theirs).status_code == 403
    assert ask(client, sid, "theirs.png").status_code == 404


async def test_an_unknown_session_and_a_session_without_a_folder_are_404(
    client: TestClient, console: Console, workspace: str
) -> None:
    put(Path(workspace) / "a.png")
    assert ask(client, "nosuchsession", "a.png").status_code == 404
    bare = (await console.store.create_session()).id
    resp = ask(client, bare, "a.png")
    assert resp.status_code == 404 and code_of(resp) == "not_found"


async def test_a_missing_file_is_404_and_a_missing_path_is_400(
    client: TestClient, console: Console, workspace: str
) -> None:
    sid = await session_in(console, Path(workspace))
    assert ask(client, sid, "gone.png").status_code == 404
    assert client.get(f"/api/sessions/{sid}/pictures").status_code == 400
    assert (
        client.get(f"/api/sessions/{sid}/pictures", params={"path": ""}).status_code
        == 400
    )


# ── Types and size ──


@pytest.mark.parametrize(
    ("name", "data", "mime"),
    [
        ("a.png", PNG, "image/png"),
        ("a.jpg", JPEG, "image/jpeg"),
        ("a.jpeg", JPEG, "image/jpeg"),
        ("a.gif", GIF, "image/gif"),
        ("a.webp", WEBP, "image/webp"),
        ("a.svg", SVG, "image/svg+xml"),
    ],
)
async def test_the_five_picture_types_are_served_by_their_content(
    client: TestClient,
    console: Console,
    workspace: str,
    name: str,
    data: bytes,
    mime: str,
) -> None:
    put(Path(workspace) / name, data)
    resp = ask(client, await session_in(console, Path(workspace)), name)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == mime


async def test_a_png_that_is_text_is_refused(
    client: TestClient, console: Console, workspace: str
) -> None:
    root = Path(workspace)
    put(root / "fake.png", b"just some text")
    put(root / "page.svg", b"<html><script>alert(1)</script></html>")
    put(root / "notes.txt", PNG)
    (root / "folder.png").mkdir()
    sid = await session_in(console, root)
    for name in ("fake.png", "page.svg", "notes.txt", "folder.png"):
        resp = ask(client, sid, name)
        assert resp.status_code == 415, name
        assert code_of(resp) == "not_a_picture"


async def test_a_picture_over_the_size_cap_is_refused(
    client: TestClient,
    console: Console,
    workspace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    put(Path(workspace) / "big.png", PNG + b"x" * 200)
    sid = await session_in(console, Path(workspace))
    monkeypatch.setattr(pictures, "MAX_PICTURE_BYTES", 100)
    resp = ask(client, sid, "big.png")
    assert resp.status_code == 413 and code_of(resp) == "too_large"


# ── Headers ──


async def test_an_svg_is_sandboxed_and_never_a_page(
    client: TestClient, console: Console, workspace: str
) -> None:
    put(Path(workspace) / "chart.svg", SVG)
    resp = ask(client, await session_in(console, Path(workspace)), "chart.svg")
    assert resp.headers["content-type"] == "image/svg+xml"
    assert resp.headers["content-security-policy"] == SVG_POLICY
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["cache-control"] == "private"
    assert resp.headers["content-disposition"] == "inline"


async def test_a_raster_picture_has_nosniff_private_and_a_sandbox(
    client: TestClient, console: Console, workspace: str
) -> None:
    put(Path(workspace) / "chart.png")
    resp = ask(client, await session_in(console, Path(workspace)), "chart.png")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["cache-control"] == "private"
    assert resp.headers["content-security-policy"] == "default-src 'none'; sandbox"


async def test_through_the_guard_the_svg_policy_gains_only_the_framing_rule(
    app: FastAPI, console: Console, workspace: str, tmp_path: Path
) -> None:
    put(Path(workspace) / "chart.svg", SVG)
    sid = await session_in(console, Path(workspace))
    auth = GatewayAuth(tmp_path / "auth")
    try:
        resp = guarded(app, auth).get(
            f"/api/sessions/{sid}/pictures",
            params={"path": "chart.svg"},
            headers={"Authorization": "Bearer " + auth.read_token().decode()},
        )
        assert resp.status_code == 200
        assert resp.headers.get_list("content-security-policy") == [
            SVG_POLICY + "; frame-ancestors 'none'"
        ]
    finally:
        auth.close()


# ── Sign-in and writes ──


async def test_a_picture_needs_the_console_sign_in(
    app: FastAPI, console: Console, workspace: str, tmp_path: Path
) -> None:
    put(Path(workspace) / "chart.png")
    sid = await session_in(console, Path(workspace))
    auth = GatewayAuth(tmp_path / "auth")
    try:
        client = guarded(app, auth)
        url = f"/api/sessions/{sid}/pictures"
        assert client.get(url, params={"path": "chart.png"}).status_code == 401
        wrong = {"Authorization": "Bearer " + "0" * 64}
        assert (
            client.get(url, params={"path": "chart.png"}, headers=wrong).status_code
            == 401
        )
        right = {"Authorization": "Bearer " + auth.read_token().decode()}
        assert (
            client.get(url, params={"path": "chart.png"}, headers=right).status_code
            == 200
        )
    finally:
        auth.close()


async def test_nothing_in_the_folder_is_written_or_listed(
    client: TestClient, console: Console, workspace: str
) -> None:
    root = Path(workspace)
    put(root / "chart.png")
    sid = await session_in(console, root)
    for method in ("post", "put", "delete", "patch"):
        resp = getattr(client, method)(
            f"/api/sessions/{sid}/pictures", params={"path": "x.png"}
        )
        assert resp.status_code in (404, 405)
    assert ask(client, sid, ".").status_code in (403, 415)
    assert ask(client, sid, root).status_code in (403, 415)
    assert sorted(os.listdir(root)) == ["chart.png"]
