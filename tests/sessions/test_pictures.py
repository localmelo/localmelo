"""Finding a turn's pictures, sniffing a picture and reading one safely (support/sessions/pictures.py)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from localmelo.support.sessions import pictures

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16
SVG = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"


def make(path: Path, data: bytes = PNG) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ── Finding ──


def test_a_scan_lists_pictures_with_size_and_time(tmp_path: Path) -> None:
    make(tmp_path / "a.png")
    make(tmp_path / "sub" / "b.SVG", SVG)
    make(tmp_path / "notes.txt", b"text")
    found = pictures.scan(str(tmp_path))
    assert set(found) == {"a.png", "sub/b.SVG"}
    assert found["a.png"][0] == len(PNG)


def test_a_scan_skips_hidden_folders_dependencies_links_and_depth(
    tmp_path: Path,
) -> None:
    make(tmp_path / ".git" / "x.png")
    make(tmp_path / ".venv" / "y.png")
    make(tmp_path / "node_modules" / "z.png")
    make(tmp_path / ".hidden.png")
    make(tmp_path / "a" / "b" / "c" / "ok.png")
    make(tmp_path / "a" / "b" / "c" / "d" / "deep.png")
    outside = make(tmp_path.parent / f"{tmp_path.name}-out" / "o.png")
    (tmp_path / "link.png").symlink_to(outside)
    (tmp_path / "linked").symlink_to(outside.parent)
    assert set(pictures.scan(str(tmp_path))) == {"a/b/c/ok.png"}


def test_a_scan_stops_after_its_entry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pictures, "SCAN_MAX_ENTRIES", 5)
    for number in range(20):
        make(tmp_path / f"p{number:02}.png")
    assert len(pictures.scan(str(tmp_path))) <= 5


def test_a_scan_of_a_missing_folder_is_empty(tmp_path: Path) -> None:
    assert pictures.scan(str(tmp_path / "gone")) == {}


def test_new_modified_and_untouched_pictures(tmp_path: Path) -> None:
    make(tmp_path / "kept.png")
    make(tmp_path / "edited.png")
    before = pictures.scan(str(tmp_path))
    time.sleep(0.01)
    make(tmp_path / "edited.png", PNG + b"more")
    make(tmp_path / "made.png")
    listed = pictures.new_pictures(before, pictures.scan(str(tmp_path)))
    assert {(item["path"], item["state"]) for item in listed} == {
        ("edited.png", "modified"),
        ("made.png", "new"),
    }
    assert all(item["size"] > 0 and item["mtime"] > 0 for item in listed)
    assert pictures.new_pictures(before, before) == []


def test_a_deleted_picture_is_not_listed(tmp_path: Path) -> None:
    gone = make(tmp_path / "gone.png")
    before = pictures.scan(str(tmp_path))
    gone.unlink()
    assert pictures.new_pictures(before, pictures.scan(str(tmp_path))) == []


def test_pictures_listed_are_capped(tmp_path: Path) -> None:
    for number in range(pictures.MAX_LISTED + 5):
        make(tmp_path / f"p{number:03}.png")
    listed = pictures.new_pictures({}, pictures.scan(str(tmp_path)))
    assert len(listed) == pictures.MAX_LISTED


def test_an_answer_names_pictures_inside_the_folder(tmp_path: Path) -> None:
    make(tmp_path / "charts" / "fcf.png")
    outside = make(tmp_path.parent / f"{tmp_path.name}-out.png")
    text = (
        f"Saved `{tmp_path}/charts/fcf.png` and ![x](charts/fcf.png), "
        f"not {outside}, not ../x.png, not missing.png, not https://a.example/p.png."
    )
    found = pictures.mentioned(str(tmp_path), text, set())
    assert [(item["path"], item["state"]) for item in found] == [
        ("charts/fcf.png", "mentioned")
    ]


def test_a_name_at_the_end_of_a_sentence_counts_and_a_longer_name_does_not(
    tmp_path: Path,
) -> None:
    make(tmp_path / "chart.png")
    make(tmp_path / "other.png")
    make(tmp_path / "other.png.bak")
    found = pictures.mentioned(
        str(tmp_path), "Saved to chart.png. Not other.png.bak!", set()
    )
    assert [item["path"] for item in found] == ["chart.png"]


def test_a_mentioned_picture_that_is_listed_already_is_not_repeated(
    tmp_path: Path,
) -> None:
    make(tmp_path / "a.png")
    assert pictures.mentioned(str(tmp_path), "a.png", {"a.png"}) == []


def test_a_mention_through_a_link_that_leads_out_is_ignored(tmp_path: Path) -> None:
    outside = make(tmp_path.parent / f"{tmp_path.name}-out" / "o.png")
    (tmp_path / "shortcut.png").symlink_to(outside)
    assert pictures.mentioned(str(tmp_path), "shortcut.png", set()) == []


# ── Sniffing ──


@pytest.mark.parametrize(
    ("head", "mime"),
    [
        (PNG, "image/png"),
        (b"\xff\xd8\xff\xe0" + b"\0" * 8, "image/jpeg"),
        (b"GIF89a" + b"\0" * 8, "image/gif"),
        (b"GIF87a" + b"\0" * 8, "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (SVG, "image/svg+xml"),
        (b'<?xml version="1.0"?>\n<!-- c -->\n<svg width="1"/>', "image/svg+xml"),
        (b"\xef\xbb\xbf  <SVG>", "image/svg+xml"),
        (
            b'<?xml version="1.0"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
            b'"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [ <!ENTITY a "b"> ]>'
            b"<svg/>",
            "image/svg+xml",
        ),
    ],
)
def test_sniff_knows_the_five_types(head: bytes, mime: str) -> None:
    assert pictures.sniff(head) == mime


@pytest.mark.parametrize(
    "head",
    [
        b"",
        b"plain text, not a picture",
        b"<html><svg></svg></html>",
        b"<script>alert(1)</script><svg/>",
        b"<svgx/>",
        b"RIFF\x00\x00\x00\x00WAVEfmt ",
        b"GIF90a",
        b"<!-- never closed <svg/>",
    ],
)
def test_sniff_refuses_what_is_not_a_picture(head: bytes) -> None:
    assert pictures.sniff(head) is None


def test_sniffing_many_comments_is_fast() -> None:
    head = (b"<!---->" * 600) + b"<html>"
    started = time.perf_counter()
    assert pictures.sniff(head) is None
    assert time.perf_counter() - started < 0.5


# ── Reading ──


def refusal(workspace: Path, requested: str) -> pictures.PictureError:
    with pytest.raises(pictures.PictureError) as caught:
        pictures.read(str(workspace), requested)
    return caught.value


def test_read_gives_the_bytes_and_the_type(tmp_path: Path) -> None:
    make(tmp_path / "a.png")
    got = pictures.read(str(tmp_path), "a.png")
    assert (got.mime, got.data) == ("image/png", PNG)
    assert pictures.read(str(tmp_path), str(tmp_path / "a.png")).data == PNG


def test_read_through_a_link_that_stays_inside_works(tmp_path: Path) -> None:
    make(tmp_path / "real" / "a.png")
    (tmp_path / "alias.png").symlink_to(tmp_path / "real" / "a.png")
    assert pictures.read(str(tmp_path), "alias.png").data == PNG


def test_read_refuses_a_path_that_leaves_the_folder(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    make(workspace / "a.png")
    make(tmp_path / "secret.png")
    for requested in ("../secret.png", "a/../../secret.png", "sub/../a.png"):
        assert refusal(workspace, requested).status == 403
    assert refusal(workspace, str(tmp_path / "secret.png")).code == "outside_workspace"
    assert refusal(workspace, "/etc/hosts").status == 403


def test_read_refuses_links_that_lead_out(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = make(tmp_path / "out" / "secret.png")
    (workspace / "file.png").symlink_to(outside)
    (workspace / "dir").symlink_to(outside.parent)
    assert refusal(workspace, "file.png").status == 403
    assert refusal(workspace, "dir/secret.png").status == 403


def test_read_does_not_follow_a_link_swapped_in_after_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    target = make(workspace / "a.png")
    outside = make(tmp_path / "secret.png", PNG + b"secret")
    real_realpath = os.path.realpath

    def swapping(path: str) -> str:
        answer = real_realpath(path)
        if answer == str(target) and target.exists() and not target.is_symlink():
            target.unlink()
            target.symlink_to(outside)
        return answer

    monkeypatch.setattr(pictures.os.path, "realpath", swapping)
    assert refusal(workspace, "a.png").status == 403


def test_read_checks_where_the_open_file_really_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    make(workspace / "a.png")
    outside = make(tmp_path / "secret.png")
    monkeypatch.setattr(pictures, "_opened_path", lambda fd: str(outside))
    assert refusal(workspace, "a.png").status == 403
    monkeypatch.setattr(pictures, "_opened_path", lambda fd: None)
    assert pictures.read(str(workspace), "a.png").mime == "image/png"


def test_read_refuses_a_missing_file_and_bad_requests(tmp_path: Path) -> None:
    assert refusal(tmp_path, "gone.png").status == 404
    assert refusal(tmp_path, "").status == 404
    assert refusal(tmp_path, "a\0.png").status == 404


def test_read_refuses_what_is_not_a_picture(tmp_path: Path) -> None:
    make(tmp_path / "text.png", b"this is text, not a PNG")
    make(tmp_path / "picture.txt")  # a PNG under a name that is not a picture's
    make(tmp_path / "page.svg", b"<html><script>alert(1)</script></html>")
    (tmp_path / "folder.png").mkdir()
    for name in ("text.png", "picture.txt", "page.svg", "folder.png"):
        refused = refusal(tmp_path, name)
        assert (refused.status, refused.code) == (415, "not_a_picture"), name


def test_read_refuses_a_picture_over_the_size_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make(tmp_path / "big.png", PNG + b"x" * 100)
    monkeypatch.setattr(pictures, "MAX_PICTURE_BYTES", 50)
    refused = refusal(tmp_path, "big.png")
    assert (refused.status, refused.code) == (413, "too_large")
    monkeypatch.setattr(pictures, "MAX_PICTURE_BYTES", 10_000)
    assert pictures.read(str(tmp_path), "big.png").mime == "image/png"
