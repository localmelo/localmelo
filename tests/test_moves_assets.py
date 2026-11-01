"""Melo's moves (support/gateway/static/moves, made by scripts/build_moves.py): the committed files are what the
manifest says, on one canvas with its anchor, with a real alpha, within their size budget; and the build turns a pack
into them with the same durations, playback and posters, the same bytes each time, and refuses a pack that does not
hold what its manifest says. The build's tests make a small pack here, so no art is needed; they and the pixel checks
need Pillow, which is not a dependency of localmelo: `uv run --with pillow pytest tests/test_moves_assets.py`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOVES = ROOT / "support/gateway/static/moves"
STATES = ["startup", "thinking", "working", "attention", "idle_rest", "idle_play"]
# The moves are one copy, served by the console and copied into the app: about 1.4 MB.
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


def test_the_folder_holds_a_sheet_per_state_the_poster_and_the_manifest() -> None:
    found = {path.name for path in MOVES.iterdir()}
    assert found == {"manifest.json", "startup-poster.webp"} | {
        f"{state}.webp" for state in STATES
    }
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
    # Startup holds its last, sitting frame; the poster is that frame.
    startup = document["states"]["startup"]
    assert startup["poster"] == len(startup["durations_ms"]) - 1
    assert webp(MOVES / "startup-poster.webp") == (size, size, True)
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
    # The words of the pack are not copied.
    assert "not copied" not in (out / "manifest.json").read_text()
    # The same pack gives the same bytes.
    again = tmp_path / "again"
    build.build(source, again)
    for path in out.iterdir():
        assert (again / path.name).read_bytes() == path.read_bytes(), path.name


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
