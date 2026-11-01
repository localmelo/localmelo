"""Melo's moves (support/gateway/static/moves, made by scripts/build_moves.py): the committed files are what the
manifest says, on one canvas with its anchor, with a real alpha, within their size budget, with a poster per state and
Melo's head cut from the posters; no picture of Melo from before them is left anywhere; and the build turns a pack into
them with the same durations, playback and posters, the same bytes each time, and refuses a pack that does not hold
what its manifest says. The build's tests make a small pack here, so no art is needed; they and the pixel checks need
Pillow, which is not a dependency of localmelo: `uv run --with pillow pytest tests/test_moves_assets.py`.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOVES = ROOT / "support/gateway/static/moves"
STATES = ["startup", "thinking", "working", "attention", "idle_rest", "idle_play"]
FACES = ["idle", "working", "waiting", "done", "error", "brand"]
# What Melo holds when nothing moves for a reason: a run that ended well, a failure, nothing known or no connection.
HELD = ["done", "error", "offline"]
FAVICONS = [16, 32, 64]
# The moves are one copy, served by the console and read by the app: about 1.6 MB.
MAX_BYTES = 2_000_000


def manifest() -> dict[str, Any]:
    document: dict[str, Any] = json.loads((MOVES / "manifest.json").read_text())
    return document


def webp(path: Path) -> tuple[int, int, bool]:
    """Width, height and whether it has alpha, read from a WebP file's extended header (VP8X)."""
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP", path
    assert data[12:16] == b"VP8X", f"{path.name}: no extended header, so no alpha"
    flags, size = data[20], data[24:30]
    width = 1 + int.from_bytes(size[:3], "little")
    height = 1 + int.from_bytes(size[3:], "little")
    # The alpha flag, and the alpha chunk itself.
    return width, height, bool(flags & 0x10) and b"ALPH" in data[:64]


def expected_files(stills: bool = True) -> set[str]:
    return (
        {"manifest.json"}
        | {f"{state}.webp" for state in STATES}
        | {f"poster-{state}.webp" for state in STATES}
        | ({f"poster-{name}.webp" for name in HELD} if stills else set())
        | {f"face-{name}.png" for name in FACES}
        | {f"favicon-{pixels}.png" for pixels in FAVICONS}
    )


def png(path: Path) -> tuple[int, int, int]:
    """Width, height and colour type (6: RGBA) of a PNG."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    width, height = struct.unpack(">II", data[16:24])
    return width, height, data[25]


def test_the_folder_holds_a_sheet_and_a_poster_per_state_the_faces_and_the_manifest() -> (
    None
):
    assert {path.name for path in MOVES.iterdir()} == expected_files()
    assert sum(path.stat().st_size for path in MOVES.iterdir()) <= MAX_BYTES


def test_the_manifest_gives_each_state_its_timeline_on_one_canvas() -> None:
    document = manifest()
    assert list(document["states"]) == STATES
    size = document["canvas"]["size"]
    # The pack's paws rest at (256, 488) of 512: the whole canvas was scaled, so the anchor scaled with it.
    assert document["canvas"]["anchor"] == [size / 2, size * 488 / 512]
    columns, rows = document["sheet"]["columns"], document["sheet"]["rows"]
    assert document["fade_ms"] == {
        "same_pose": 160,
        "pose_change": 220,
        "attention": 120,
        "reduced_motion": 80,
    }
    for state, entry in document["states"].items():
        assert entry["file"] == f"{state}.webp"
        durations = entry["durations_ms"]
        assert 0 < len(durations) <= columns * rows
        assert all(isinstance(ms, int) and ms > 0 for ms in durations)
        assert 0 <= entry["poster"] < len(durations)
        assert entry["playback"] == ("once" if state == "startup" else "loop")
        assert webp(MOVES / entry["file"]) == (columns * size, rows * size, True)
    # Startup holds its last, sitting frame; the poster is that frame. Every state's poster is a file of its own.
    startup = document["states"]["startup"]
    assert startup["poster"] == len(startup["durations_ms"]) - 1
    for state in STATES + HELD:
        assert webp(MOVES / f"poster-{state}.webp") == (size, size, True)
    # The done and error faces are their stills' heads: done is not the idle face.
    assert (MOVES / "face-done.png").read_bytes() != (
        MOVES / "face-idle.png"
    ).read_bytes()
    # One loop of play is 6 s: the rest-or-play rule plays it once.
    assert sum(document["states"]["idle_play"]["durations_ms"]) == 6000


def test_each_frame_is_drawn_clear_round_the_dog_and_the_paws_rest_on_the_anchor() -> (
    None
):
    image = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    document = manifest()
    size = document["canvas"]["size"]
    columns = document["sheet"]["columns"]
    anchor_y = document["canvas"]["anchor"][1]
    for state, entry in document["states"].items():
        with image.open(MOVES / entry["file"]) as sheet:
            alpha = np.asarray(sheet.convert("RGBA"))[..., 3]
        frames = len(entry["durations_ms"])
        for index in range(columns * document["sheet"]["rows"]):
            x, y = (index % columns) * size, (index // columns) * size
            cell = alpha[y : y + size, x : x + size]
            if index >= frames:
                assert cell.max() == 0, (
                    state,
                    index,
                    "a cell past the last frame is empty",
                )
                continue
            # Clear at the edges (no frame bleeds into its neighbour), solid inside (the art's own fur is 253).
            assert cell[:4].max() == 0 and cell[-4:].max() == 0, (state, index)
            assert cell[:, :4].max() == 0 and cell[:, -4:].max() == 0, (state, index)
            assert (cell >= 250).mean() > 0.05, (state, index)
            # The lowest drawn row is the ground the paws rest on, give or take the art's own shadow.
            lowest = int(np.nonzero(cell.max(axis=1) > 128)[0][-1])
            assert abs(lowest - anchor_y) <= 3, (state, index, lowest)


def test_the_faces_are_the_head_alone_centred_with_a_margin() -> None:
    for name in FACES:
        assert png(MOVES / f"face-{name}.png") == (96, 96, 6), name
    for pixels in FAVICONS:
        assert png(MOVES / f"favicon-{pixels}.png") == (pixels, pixels, 6), pixels
    image = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    for path in [MOVES / f"face-{name}.png" for name in FACES] + [
        MOVES / f"favicon-{pixels}.png" for pixels in FAVICONS
    ]:
        with image.open(path) as picture:
            alpha = np.asarray(picture.convert("RGBA"))[..., 3]
        side = alpha.shape[0]
        rows = np.nonzero(alpha.max(axis=1) > 128)[0]
        columns = np.nonzero(alpha.max(axis=0) > 128)[0]
        # The head fills most of the square, not cut by more than a sliver at any edge, and sits near the middle.
        assert (rows[-1] - rows[0] + 1) / side > 0.6 and (
            columns[-1] - columns[0] + 1
        ) / side > 0.6, path.name
        middle = ((columns[0] + columns[-1]) / 2, (rows[0] + rows[-1]) / 2)
        assert abs(middle[0] - side / 2) < side * 0.12, path.name
        assert abs(middle[1] - side / 2) < side * 0.15, path.name


# What no file may name any more: Melo's pictures from before the moves, their folders, the script that made them,
# and the moves' own old poster.
OLD_ART = [
    "art/akita",
    "static/akita",
    "assets/akita",
    "Resources/akita",
    "build_pet_assets",
    "PetMotion",
    "PetState",
    "startup-poster",
    "sit-idle",
    "sit-working",
    "sit-waiting",
    "sit-done",
    "sit-error",
    "lie-idle",
    "lie-working",
    "lie-waiting",
    "lie-done",
    "lie-error",
    "rest-lazy",
    "sleep-front",
    "sleep-side",
    "sleep-back",
    "face-idle-blink",
]
# Files that tell of the past: the changelog's earlier entries and the site's list of updates.
HISTORY = {"CHANGELOG.md", "docs/en/updates.html", "docs/zh-CN/updates.html"}


def test_no_file_names_a_picture_of_melo_from_before_the_moves() -> None:
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode()
    names = [name for name in listed.split("\0") if name]
    if not names:
        pytest.skip("not a git checkout")
    this = Path(__file__).resolve().relative_to(ROOT).as_posix()
    found = []
    for name in names:
        if name in HISTORY or name == this or not (ROOT / name).is_file():
            continue
        assert not name.startswith(
            ("macos/art/", "support/gateway/static/akita/")
        ), name
        try:
            text = (ROOT / name).read_text()
        except (UnicodeDecodeError, OSError):
            continue
        found += [f"{name}: {word}" for word in OLD_ART if word in text]
    assert not found, found


# ── The build, on a small pack made here ──


def script() -> ModuleType:
    pytest.importorskip("PIL.Image")
    spec = importlib.util.spec_from_file_location(
        "build_moves", ROOT / "scripts/build_moves.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_moves"] = module
    spec.loader.exec_module(module)
    return module


def pack(folder: Path, side: int = 64, frames: int = 3, **changes: Any) -> Path:
    """A pack of flat squares on a `side` px canvas: per state `frames` frames, each a square standing on the
    anchor that moves right by a frame. `changes` replaces parts of the manifest."""
    image = pytest.importorskip("PIL.Image")
    draw = pytest.importorskip("PIL.ImageDraw")
    states: dict[str, Any] = {}
    for number, state in enumerate(STATES):
        names = []
        for index in range(frames):
            picture = image.new("RGBA", (side, side))
            left = side // 4 + index
            draw.Draw(picture).rectangle(
                (left, side // 4, left + side // 3, side - 5),
                fill=(200, 120 + 10 * number, 40, 255),
            )
            name = f"frames/{state}/frame_{index + 1:02}.png"
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            picture.save(folder / name)
            names.append(name)
        states[state] = {
            "frames": names,
            "frame_durations_ms": [100 * (index + 1) for index in range(frames)],
            "poster_frame_index": frames - 1 if state == "startup" else 0,
            **(
                {"playback": "once", "loop_count": 1}
                if state == "startup"
                else {"loop_count": 0}
            ),
            "label": "not copied",
        }
    document = {
        "canvas": {
            "width": side,
            "height": side,
            "anchor": {"x": side / 2, "y": side - 4},
        },
        "states": states,
    }
    document.update(changes)
    (folder / "manifest.json").write_text(json.dumps(document))
    return folder


def test_the_build_keeps_the_timelines_and_scales_the_whole_canvas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    image = pytest.importorskip("PIL.Image")
    build = script()
    monkeypatch.setattr(build, "CELL", 64)  # a quicker build; the same steps
    source = pack(tmp_path / "pack")
    out = tmp_path / "moves"
    total = build.build(source, out)
    assert total == sum(path.stat().st_size for path in out.iterdir())
    document = json.loads((out / "manifest.json").read_text())
    cell = build.CELL
    assert document["canvas"] == {"size": cell, "anchor": [cell / 2, cell * 60 / 64]}
    for state, entry in document["states"].items():
        assert entry == {
            "file": f"{state}.webp",
            "pose": build.POSES[state],
            "playback": "once" if state == "startup" else "loop",
            "poster": 2 if state == "startup" else 0,
            "durations_ms": [100, 200, 300],
        }
        with image.open(out / entry["file"]) as sheet:
            assert sheet.size == (4 * cell, 4 * cell)
            alpha = np.asarray(sheet.convert("RGBA"))[..., 3]
        # Three frames drawn, the rest of the sheet clear; each square sits on the scaled ground line.
        for index in range(16):
            x, y = (index % 4) * cell, (index // 4) * cell
            drawn = alpha[y : y + cell, x : x + cell] > 128
            assert drawn.any() == (index < 3), (state, index)
            if index < 3:
                rows = np.nonzero(drawn.any(axis=1))[0]
                assert abs(int(rows[-1]) + 1 - cell * 60 / 64) <= 2
    # A poster per state at the cell's size; the faces and favicons, cut from the posters (no stills given).
    assert {path.name for path in out.iterdir()} == expected_files(stills=False)
    for state in STATES:
        with image.open(out / f"poster-{state}.webp") as poster:
            assert poster.size == (cell, cell)
    for name in FACES:
        assert png(out / f"face-{name}.png")[:2] == (96, 96)
    for pixels in FAVICONS:
        assert png(out / f"favicon-{pixels}.png")[:2] == (pixels, pixels)
    # The words of the pack are not copied.
    assert "not copied" not in (out / "manifest.json").read_text()
    # The same pack gives the same bytes.
    again = tmp_path / "again"
    build.build(source, again)
    for path in out.iterdir():
        assert (again / path.name).read_bytes() == path.read_bytes(), path.name


def stills(folder: Path, side: int = 64, **changes: Any) -> Path:
    """done.png, error.png and offline.png on the pack's canvas: a square standing on its ground line, in a colour of
    its own. `changes` gives a name another size, or None to leave it out."""
    image = pytest.importorskip("PIL.Image")
    draw = pytest.importorskip("PIL.ImageDraw")
    folder.mkdir(parents=True, exist_ok=True)
    for number, name in enumerate(HELD):
        size = changes.get(name, side)
        if size is None:
            continue
        picture = image.new("RGBA", (size, size))
        draw.Draw(picture).rectangle(
            (size // 4, size // 4, size * 3 // 4, size - 5),
            fill=(60 * number, 90, 200, 255),
        )
        picture.save(folder / f"{name}.png")
    return folder


def test_the_build_writes_the_stills_and_cuts_the_done_and_error_faces_from_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = pytest.importorskip("PIL.Image")
    build = script()
    monkeypatch.setattr(build, "CELL", 32)
    source = pack(tmp_path / "pack")
    plain = tmp_path / "plain"
    build.build(source, plain)
    assert {path.name for path in plain.iterdir()} == expected_files(stills=False)
    out = tmp_path / "moves"
    build.build(source, out, stills(tmp_path / "stills"))
    assert {path.name for path in out.iterdir()} == expected_files()
    for name in HELD:
        with image.open(out / f"poster-{name}.webp") as poster:
            assert poster.size == (32, 32)
    # Only the done and error faces change: they are the stills' heads; the others are the pack's.
    for name in FACES:
        same = (out / f"face-{name}.png").read_bytes() == (
            plain / f"face-{name}.png"
        ).read_bytes()
        assert same == (name not in ("done", "error")), name
    # With the heads drawn for small sizes: the idle face is head-smile, the brand and the tab icons head-neutral.
    folder = tmp_path / "stills"
    for number, name in enumerate(("head-smile", "head-neutral")):
        head = image.new("RGBA", (64, 64))
        head.paste((200, 40 + 100 * number, 60, 255), (12, 8, 52, 61))
        head.save(folder / f"{name}.png")
    heads = tmp_path / "heads"
    build.build(source, heads, folder)
    assert {path.name for path in heads.iterdir()} == expected_files()
    changed = {
        path.name
        for path in heads.iterdir()
        if path.read_bytes() != (out / path.name).read_bytes()
    }
    assert changed == {
        "face-idle.png",
        "face-brand.png",
        *(f"favicon-{pixels}.png" for pixels in FAVICONS),
    }
    assert (heads / "face-idle.png").read_bytes() != (
        heads / "face-brand.png"
    ).read_bytes()
    # A stills folder that is missing one, or has one off the canvas, is refused.
    for broken, words in (
        ({"error": None}, "no error.png"),
        ({"offline": 32}, "offline.png is not on"),
    ):
        with pytest.raises(build.PackError, match=words):
            build.build(
                source,
                tmp_path / "refused",
                stills(tmp_path / f"stills-{words[:3]}", **broken),
            )


@pytest.mark.parametrize(
    "broken, words",
    [
        ({"states": {}}, "has no startup"),
        ({"canvas": {"width": 64, "height": 32, "anchor": {"x": 1, "y": 1}}}, "square"),
        ({"canvas": {"width": 64, "height": 64}}, "no anchor"),
    ],
)
def test_a_pack_without_what_its_manifest_needs_is_refused(
    tmp_path: Path, broken: dict[str, Any], words: str
) -> None:
    build = script()
    source = pack(tmp_path / "pack", **broken)
    with pytest.raises(build.PackError, match=words):
        build.build(source, tmp_path / "moves")
    assert not (tmp_path / "moves").exists()


def test_a_frame_outside_the_pack_off_its_canvas_or_without_a_duration_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = script()
    monkeypatch.setattr(build, "CELL", 32)
    source = pack(tmp_path / "pack")
    document = json.loads((source / "manifest.json").read_text())
    (tmp_path / "outside.png").write_bytes(
        (source / "frames/working/frame_01.png").read_bytes()
    )
    for change, words in [
        (
            {
                "frames": [
                    "../outside.png",
                    *document["states"]["working"]["frames"][1:],
                ]
            },
            "not a file of the pack",
        ),
        ({"frame_durations_ms": [100, 200]}, "one positive duration"),
        ({"frame_durations_ms": [100, 0, 300]}, "one positive duration"),
        ({"poster_frame_index": 3}, "poster"),
    ]:
        edited = json.loads(json.dumps(document))
        edited["states"]["working"].update(change)
        (source / "manifest.json").write_text(json.dumps(edited))
        with pytest.raises(build.PackError, match=words):
            build.build(source, tmp_path / "moves")
    (source / "manifest.json").write_text(json.dumps(document))
    image = pytest.importorskip("PIL.Image")
    image.new("RGBA", (32, 32)).save(source / "frames/working/frame_02.png")
    with pytest.raises(build.PackError, match="64 x 64 canvas"):
        build.build(source, tmp_path / "moves")


def test_moves_heavier_than_the_budget_stop_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = script()
    source = pack(tmp_path / "pack")
    monkeypatch.setattr(build, "CELL", 32)
    monkeypatch.setattr(build, "MAX_BYTES", 1000)
    with pytest.raises(build.PackError, match="more than"):
        build.build(source, tmp_path / "moves")
