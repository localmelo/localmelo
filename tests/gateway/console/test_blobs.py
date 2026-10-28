"""``GET /api/sessions/{id}/blobs/{sha256}``: images and opaque downloads."""

from __future__ import annotations

from fastapi.testclient import TestClient

from localmelo.melo.schema import ImagePart
from localmelo.support.gateway.console import Console

from .conftest import seed_turn

PNG = b"\x89PNG\r\n\x1a\n" + b"payload"
SVG = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"


async def test_an_image_is_inline(
    client: TestClient, console: Console, workspace: str
) -> None:
    image = ImagePart.from_bytes(PNG, "image/png")
    turn = await seed_turn(console.store, workspace, images=[image])
    resp = client.get(f"/api/sessions/{turn.session_id}/blobs/{image.sha256}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["content-disposition"] == "inline"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["content-security-policy"] == ("default-src 'none'; sandbox")
    assert resp.headers["cache-control"] == "no-store"
    assert resp.content == PNG


async def test_a_non_image_is_an_attachment(
    client: TestClient, console: Console, workspace: str
) -> None:
    image = ImagePart.from_bytes(SVG, "image/svg+xml")
    turn = await seed_turn(console.store, workspace, images=[image])
    resp = client.get(f"/api/sessions/{turn.session_id}/blobs/{image.sha256}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["content-disposition"] == (
        f'attachment; filename="{image.sha256}.bin"'
    )
    assert resp.content == SVG


async def test_an_unknown_blob_is_404(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    resp = client.get(f"/api/sessions/{turn.session_id}/blobs/{'0' * 64}")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


async def test_a_malformed_digest_is_404(
    client: TestClient, console: Console, workspace: str
) -> None:
    turn = await seed_turn(console.store, workspace)
    for digest in ("not-a-digest", "A" * 64, "0" * 63):
        resp = client.get(f"/api/sessions/{turn.session_id}/blobs/{digest}")
        assert resp.status_code == 404


async def test_an_unknown_session_is_404(client: TestClient) -> None:
    resp = client.get(f"/api/sessions/0123456789ab/blobs/{'0' * 64}")
    assert resp.status_code == 404


async def test_a_blob_of_another_session_is_not_found(
    client: TestClient, console: Console, workspace: str, tmp_path: object
) -> None:
    image = ImagePart.from_bytes(PNG, "image/png")
    first = await seed_turn(console.store, workspace, images=[image], title="one")
    second = await seed_turn(console.store, workspace, title="two")
    resp = client.get(f"/api/sessions/{second.session_id}/blobs/{image.sha256}")
    assert resp.status_code == 404
    assert first.session_id != second.session_id
