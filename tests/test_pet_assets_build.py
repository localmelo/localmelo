"""scripts/build_pet_assets.py: the same input gives the same bytes, an edit that does not lie on its base stops the
build with a message, a blink patch changes only the eye band, and the tail and ear splits give the body without the part. The
masters are made here (flat shapes on the sizes of the real ones), so no art is needed. Pillow is not a dependency of
localmelo, so these run where it is installed: `uv run --with pillow pytest tests/test_pet_assets_build.py`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = (
    10,
    10,
)  # where every cut-out lies in its generated picture, as its cut-report.json says
NAVY, FAWN, CREAM = (15, 33, 48, 255), (193, 118, 38, 255), (242, 239, 220, 255)


def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_pet_assets", ROOT / "scripts/build_pet_assets.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_pet_assets"] = module
    spec.loader.exec_module(module)
    return module


def dog(
    size: tuple[int, int],
    eyes: tuple[int, int, int, int] | None = None,
    closed: bool = False,
    tail: tuple[int, int, int, int] | None = None,
    shift: int = 0,
) -> Any:
    """A flat-shape dog on a transparent frame: a body, a head with an outline, two eyes (or the closed lines) in
    the eye band and a tail blob in the tail box. `shift` moves everything sideways, as a model that moved the dog.
    """
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.rounded_rectangle(
        (w * 0.08 + shift, h * 0.2, w * 0.7 + shift, h - 4), 30, FAWN, NAVY, 5
    )
    draw.ellipse((w * 0.1 + shift, h * 0.05, w * 0.45 + shift, h * 0.5), CREAM, NAVY, 5)
    draw.line((w * 0.3 + shift, h * 0.7, w * 0.3 + shift, h - 10), NAVY, 7)
    if eyes:
        x0, y0, x1, y1 = eyes
        for x in (x0 + 25, x1 - 25):
            if closed:
                draw.line(
                    (x - 12 + shift, (y0 + y1) // 2, x + 12 + shift, (y0 + y1) // 2),
                    NAVY,
                    5,
                )
            else:
                draw.ellipse(
                    (
                        x - 10 + shift,
                        (y0 + y1) // 2 - 10,
                        x + 10 + shift,
                        (y0 + y1) // 2 + 10,
                    ),
                    NAVY,
                )
    if tail:
        draw.ellipse(tail, CREAM, NAVY, 5)
    return image


def inset(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """A blob that lies inside a part's box, as the part does."""
    return (box[0] + 20, box[1] + 20, box[2] - 20, box[3] - 20)


def world(tmp_path: Path, shift: int = 0, moved: tuple[str, ...] | None = None) -> Path:
    """A picks.json with cut-outs for the 14 states and the five edits; `shift` moves the dogs of the edits named in
    `moved` (all of them when it is None)."""
    mod = script()
    root = tmp_path / "masters"
    cut = root / "cut"
    cut.mkdir(parents=True)
    sit_eyes = mod.BLINKS["sit-idle"][1]
    lie_eyes = mod.BLINKS["lie-idle"][1]
    rest_eyes = mod.BLINKS["rest-lazy"][1]
    tail_blob, ear_blob = inset(mod.TAIL["lie-done"][1]), inset(mod.EAR["lie-idle"][1])

    def by(name: str) -> int:
        return shift if moved is None or name in moved else 0

    report: dict[str, Any] = {}

    def put(name: str, image: Any) -> str:
        image.save(cut / f"{name}.png")
        report[f"{name}.png"] = {
            "bbox": [
                ORIGIN[0],
                ORIGIN[1],
                ORIGIN[0] + image.width,
                ORIGIN[1] + image.height,
            ]
        }
        return f"cut/{name}.png"

    picks = {}
    for state in (*mod.PET, *mod.PORTRAIT):
        sitting = state.startswith("sit-")
        size = (330, 430) if sitting else (700, 440)
        picks[state] = put(
            state,
            dog(
                size,
                {
                    "sit-idle": sit_eyes,
                    "lie-idle": lie_eyes,
                    "rest-lazy": rest_eyes,
                }.get(state),
                tail=(
                    tail_blob
                    if state == "lie-done"
                    else ear_blob if state == "lie-idle" else None
                ),
            ),
        )
    motion = {
        "sit-idle-blink": put(
            "sit-idle-blink",
            dog((330, 430), sit_eyes, True, shift=by("sit-idle-blink")),
        ),
        "lie-idle-blink": put(
            "lie-idle-blink",
            dog((700, 440), lie_eyes, True, tail=ear_blob, shift=by("lie-idle-blink")),
        ),
        "rest-lazy-blink": put(
            "rest-lazy-blink",
            dog((700, 440), rest_eyes, True, shift=by("rest-lazy-blink")),
        ),
        "lie-done-notail": put(
            "lie-done-notail", dog((700, 440), shift=by("lie-done-notail"))
        ),
        "lie-idle-noear": put(
            "lie-idle-noear", dog((700, 440), lie_eyes, shift=by("lie-idle-noear"))
        ),
    }
    (cut / "cut-report.json").write_text(json.dumps(report))
    picks_file = tmp_path / "picks.json"
    picks_file.write_text(
        json.dumps({"root": str(root), "picks": picks, "motion": motion})
    )
    return picks_file


def extent(mask: Any) -> tuple[int, int, int, int]:
    """Left, top, right and bottom (inclusive) of the true pixels of a mask."""
    xs, ys = np.flatnonzero(mask.any(axis=0)), np.flatnonzero(mask.any(axis=1))
    return int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])


def files(folder: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(folder)): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_the_same_input_gives_the_same_bytes_and_the_moves_are_written(
    tmp_path: Path,
) -> None:
    mod = script()
    picks = world(tmp_path)
    mod.build(picks, tmp_path / "one")
    mod.build(picks, tmp_path / "two")
    first = files(tmp_path / "one")
    assert first == files(tmp_path / "two")
    for piece in (
        "pet/lie-idle-blink",
        "pet/rest-lazy-blink",
        "portrait/sit-idle-blink",
        "face/idle-blink",
        "pet/lie-done-body",
        "pet/lie-done-tail",
        "pet/lie-idle-body",
        "pet/lie-idle-ear",
    ):
        assert f"{piece}.png" in first and f"{piece}@2x.png" in first
    pictures = json.loads(first["anchors.json"])["pictures"]
    assert set(pictures["lie-done"]["tail"]) == {"box", "pivot"}
    assert set(pictures["lie-idle"]["ear"]) == {"box", "pivot"}
    assert "tail" not in pictures["lie-idle"] and "ear" not in pictures["lie-done"]
    assert all(
        pictures[state]["eyes"] for state in ("sit-idle", "lie-idle", "rest-lazy")
    )
    assert pictures["sleep-side"]["z"] == [62, 22]
    # The files the moves add leave the stills as they are: the still of a picture does not depend on its edit.
    assert first["pet/lie-done@2x.png"] != first["pet/lie-done-tail@2x.png"]


def test_an_edit_that_moved_the_dog_stops_the_build_with_a_message(
    tmp_path: Path,
) -> None:
    mod = script()
    with pytest.raises(
        mod.AlignmentError, match=r"sit-idle-blink.*best shift \(-?\d, -?\d\)"
    ):
        mod.build(world(tmp_path, shift=3), tmp_path / "out")
    assert not (tmp_path / "out/anchors.json").exists()  # nothing half-written to ship


def test_a_blink_patch_draws_only_inside_the_eye_band_and_only_where_the_edit_differs() -> (
    None
):
    mod = script()
    box = mod.BLINKS["sit-idle"][1]
    base = dog((330, 430), box)
    edit = dog((330, 430), box, closed=True)
    patch = np.asarray(mod.blink_patch(base, edit, box))
    left, top, right, bottom = extent(patch[..., 3] > 0)
    grow = 3 + 3  # the dilation and the feather
    assert left >= box[0] - grow and right <= box[2] + grow
    assert top >= box[1] - grow and bottom <= box[3] + grow
    # Where it is solid it carries the edit's pixels, and it is solid where the base had an eye and the edit has not.
    solid = patch[..., 3] == 255
    assert solid.any() and (patch[solid][:, :3] == np.asarray(edit)[solid][:, :3]).all()
    mid = (box[1] + box[3]) // 2
    assert patch[mid - 8, box[0] + 25, 3] > 200  # the top of an open eye
    # An edit identical to its base needs no patch at all.
    assert not np.asarray(mod.blink_patch(base, base, box))[..., 3].any()


def test_the_tail_split_leaves_a_body_without_the_tail_and_the_rest_as_it_was() -> None:
    mod = script()
    box = mod.TAIL["lie-done"][1]
    blob = (box[0] + 20, box[1] + 20, box[2] - 20, box[3] - 20)
    base, edit = dog((700, 440), tail=blob), dog((700, 440))
    body, mask = mod.tail_split(base, edit, box)
    body, mask = np.asarray(body), np.asarray(mask)
    middle = ((blob[0] + blob[2]) // 2, (blob[1] + blob[3]) // 2)
    assert (
        mask[middle[1], middle[0]] == 255 and body[middle[1], middle[0], 3] == 0
    )  # the tail is gone
    outside = np.ones(mask.shape, bool)
    outside[box[1] - 10 : box[3] + 10, box[0] - 10 : box[2] + 10] = False
    assert (
        body[outside] == np.asarray(base)[outside]
    ).all()  # away from the tail the body is the picture
    left, top, right, bottom = extent(mask > 0)
    assert left >= box[0] - 8 and right <= box[2] + 8
    assert top >= box[1] - 8 and bottom <= box[3] + 8


@pytest.mark.parametrize("name", ["rest-lazy-blink", "lie-idle-noear"])
def test_each_new_edit_is_checked_on_its_own_and_named_in_the_message(
    tmp_path: Path, name: str
) -> None:
    mod = script()
    with pytest.raises(mod.AlignmentError, match=rf"{name}.*best shift"):
        mod.build(world(tmp_path, shift=3, moved=(name,)), tmp_path / "out")
    assert not (tmp_path / "out/anchors.json").exists()


def test_the_ear_split_leaves_a_body_without_the_ear_and_no_ghost_of_its_outline() -> (
    None
):
    mod = script()
    box, pivot = mod.EAR["lie-idle"][1], mod.EAR["lie-idle"][2]
    blob = inset(box)
    base, edit = dog((700, 440), tail=blob), dog((700, 440))
    body, mask = mod.tail_split(base, edit, box, mod.GROW["lie-idle"])
    body, mask = np.asarray(body), np.asarray(mask)
    middle = ((blob[0] + blob[2]) // 2, (blob[1] + blob[3]) // 2)
    assert mask[middle[1], middle[0]] == 255 and body[middle[1], middle[0], 3] == 0
    left, top, right, bottom = extent(mask > 0)
    assert left >= box[0] - 12 and right <= box[2] + 12 and top >= 0
    assert bottom <= box[3] + 12
    # The ear stands in the clear: where the edit has nothing, the body keeps under 5% of the old outline. The
    # default feather leaves more, which is why the ear grows its mask further.
    clear = np.asarray(edit)[..., 3] == 0
    assert body[clear][:, 3].max() < 0.05 * 255
    plain = np.asarray(mod.tail_split(base, edit, box)[0])
    assert plain[clear][:, 3].max() > 0.05 * 255
    assert box[0] <= pivot[0] <= box[2] and box[1] <= pivot[1] <= box[3]
