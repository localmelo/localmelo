"""The art pack pipeline (scripts/art_pack.py): a folder or a zip of pictures is read safely, checked rule by rule, and
replaces every picture of Melo, the same bytes each time.

The tests make small packs here (a 64 px canvas), so no art is needed, and they never write into the repository: the
pictures go to a temporary root. The zip-safety tests need only the standard library; the rest need Pillow, which is not
a dependency of localmelo: `uv run --with pillow==13.0.0 pytest tests/test_art_pack.py`.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import stat
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
STATES = ["startup", "thinking", "working", "attention", "idle_rest", "idle_play"]
SIDE = 64
ICONUTIL = Path("/usr/bin/iconutil")


@pytest.fixture
def zips(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """scripts/pack_zip.py: it reads names, sizes and bytes with the standard library alone."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("pack_zip")


@pytest.fixture
def art(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """scripts/art_pack.py with the moves made at 64 px, so a build takes a moment."""
    pytest.importorskip("PIL.Image")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = importlib.import_module("art_pack")
    monkeypatch.setattr(module.build_moves, "CELL", SIDE)
    return module


# ── Packs made here ──


def draw(
    path: Path,
    size: tuple[int, int] = (SIDE, SIDE),
    mode: str = "RGBA",
    box: tuple[int, int, int, int] | None = None,
    colour: tuple[int, ...] = (200, 120, 40),
) -> Path:
    """A picture of one filled rectangle, clear round it for RGBA (white for RGB); `box` is where it stands."""
    from PIL import Image, ImageDraw  # type: ignore[import-not-found]

    width, height = size
    box = box or (width // 4, height // 4, width * 3 // 4, height - 5)
    image = Image.new(mode, size, (0, 0, 0, 0) if mode == "RGBA" else (255, 255, 255))
    ImageDraw.Draw(image).rectangle(
        box, fill=(*colour, 255) if mode == "RGBA" else colour
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def make_pack(
    folder: Path,
    frames: int = 3,
    start: int = 1,
    stills: bool = False,
    heads: bool = False,
    brand: tuple[str, ...] = (),
    **changes: Any,
) -> Path:
    """A valid pack of flat rectangles on a 64 px canvas: per state `frames` frames counted from `start`, each a little
    to the right; the stills, heads and brand pictures asked for. `changes` replaces parts of the manifest.
    """
    states: dict[str, Any] = {}
    for number, state in enumerate(STATES):
        names = []
        for index in range(frames):
            name = f"frames/{state}/frame_{start + index:02}.png"
            left = SIDE // 4 + index
            draw(
                folder / name,
                box=(left, SIDE // 4, left + SIDE // 3, SIDE - 5),
                colour=(200, 120 + 10 * number, 40),
            )
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
            "width": SIDE,
            "height": SIDE,
            "anchor": {"x": SIDE / 2, "y": SIDE - 4},
        },
        "states": states,
    }
    document.update(changes)
    (folder / "manifest.json").write_text(json.dumps(document))
    if stills:
        for number, name in enumerate(("done", "error", "offline")):
            draw(folder / "stills" / f"{name}.png", colour=(60 * number, 90, 200))
    if heads:
        for number, name in enumerate(("head-smile", "head-neutral")):
            draw(
                folder / "heads" / f"{name}.png",
                box=(10, 6, 54, SIDE - 4),
                colour=(200, 40 + 100 * number, 60),
            )
    sizes = {"app-icon": (1024, 1024), "menubar": (40, 40), "emblem": (40, 50)}
    for name in brand:
        draw(folder / "brand" / f"{name}.png", sizes[name], "RGB", (6, 6, 30, 30))
    return folder


def edit_manifest(pack: Path, change: Any) -> None:
    """Change the manifest in place: `change` gets the document and edits it."""
    document = json.loads((pack / "manifest.json").read_text())
    change(document)
    (pack / "manifest.json").write_text(json.dumps(document))


def zipped(
    folder: Path, target: Path, top: str = "", extra: dict[str, bytes] | None = None
) -> Path:
    """The files of `folder` as a zip, under the folder `top` if given, as a Unix zip tool writes them."""
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(f"{top}{path.relative_to(folder).as_posix()}")
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
        for name, data in (extra or {}).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, data)
    return target


def tree(folder: Path) -> dict[str, bytes]:
    """Every file under `folder` with its bytes."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def run(
    art: ModuleType, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, str]:
    """The command's exit code and what it printed."""
    code = art.main([*args])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def checked(
    art: ModuleType, capsys: pytest.CaptureFixture[str], pack: Path
) -> tuple[int, str]:
    return run(art, capsys, "check", str(pack))


# ── Zip safety (standard library only) ──


def safe_zip(path: Path, top: str = "") -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{top}manifest.json", "{}")
        archive.writestr(f"{top}frames/startup/frame_01.png", b"\x89PNG one")
        archive.writestr(f"{top}startup.gif", b"GIF89a")
        archive.writestr(f"{top}README.md", "# a pack")
        archive.writestr(f"{top}integration/App.swift", "print()")
        archive.writestr("__MACOSX/._manifest.json", "junk")
        archive.writestr(f"{top}.DS_Store", "junk")
    return path


@pytest.mark.parametrize("top", ["", "Melo_pack/"])
def test_a_zip_unpacks_only_png_and_json_with_the_pack_at_its_root_or_in_one_top_folder(
    zips: ModuleType, tmp_path: Path, top: str
) -> None:
    target = tmp_path / "out"
    target.mkdir()
    written, skipped = zips.unpack(safe_zip(tmp_path / "pack.zip", top), target)
    assert sorted(written) == ["frames/startup/frame_01.png", "manifest.json"]
    assert sorted(
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    ) == sorted(written)
    # What is left out is named, relative to the pack; a Mac's own files are not listed.
    assert sorted(skipped) == ["README.md", "integration/App.swift", "startup.gif"]
    assert (target / "frames/startup/frame_01.png").read_bytes() == b"\x89PNG one"


@pytest.mark.parametrize(
    "name, words",
    [
        ("../evil.png", "climbs out"),
        ("Top/../../evil.png", "climbs out"),
        ("/abs/evil.png", "absolute path"),
        ("C:/evil.png", "drive letter"),
        ("Top\\..\\evil.png", "backslash"),
    ],
)
def test_an_entry_name_that_leaves_the_pack_refuses_the_zip_and_nothing_is_written(
    zips: ModuleType, tmp_path: Path, name: str, words: str
) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Top/manifest.json", "{}")
        handle.writestr(name, b"\x89PNG evil")
    target = tmp_path / "work" / "out"
    target.mkdir(parents=True)
    with pytest.raises(zips.UnsafePackError, match=words):
        zips.unpack(archive, target)
    assert list(target.iterdir()) == []
    assert (
        not (tmp_path / "evil.png").exists()
        and not (tmp_path / "work" / "evil.png").exists()
    )


@pytest.mark.parametrize(
    "mode, words",
    [
        (stat.S_IFLNK | 0o777, "symbolic link"),
        (stat.S_IFIFO | 0o644, "not a regular file"),
        (stat.S_IFCHR | 0o644, "not a regular file"),
    ],
)
def test_a_symlink_or_another_thing_that_is_not_a_regular_file_refuses_the_zip(
    zips: ModuleType, tmp_path: Path, mode: int, words: str
) -> None:
    archive = tmp_path / "links.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Top/manifest.json", "{}")
        link = zipfile.ZipInfo("Top/frames/startup/frame_01.png")
        link.external_attr = mode << 16
        handle.writestr(link, "/etc/passwd")
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(zips.UnsafePackError, match=words):
        zips.unpack(archive, target)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("second", ["Top/a.png", "Top/A.png"])
def test_a_name_used_twice_refuses_the_zip_even_when_only_the_case_differs(
    zips: ModuleType, tmp_path: Path, second: str
) -> None:
    archive = tmp_path / "twice.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Top/manifest.json", "{}")
        handle.writestr("Top/a.png", b"one")
        # The standard library itself warns about a name written twice.
        warned = (
            pytest.warns(UserWarning, match="Duplicate name")
            if second == "Top/a.png"
            else contextlib.nullcontext()
        )
        with warned:
            handle.writestr(second, b"two")
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(zips.UnsafePackError, match="used twice"):
        zips.unpack(archive, target)
    assert list(target.iterdir()) == []


def test_an_encrypted_entry_refuses_the_zip(zips: ModuleType, tmp_path: Path) -> None:
    archive = safe_zip(tmp_path / "pack.zip")
    with zipfile.ZipFile(archive) as handle:
        handle.infolist()[0].flag_bits |= 0x1
        with pytest.raises(zips.UnsafePackError, match="encrypted"):
            zips.check_archive(handle)


@pytest.mark.parametrize(
    "limit, value, words",
    [
        ("MAX_ENTRIES", 3, "entries, more than 3"),
        ("MAX_ENTRY_BYTES", 5, "for one file"),
        ("MAX_TOTAL_BYTES", 20, "in all"),
    ],
)
def test_too_many_entries_one_too_large_or_too_much_in_all_refuses_the_zip(
    zips: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
    value: int,
    words: str,
) -> None:
    monkeypatch.setattr(zips, limit, value)
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(zips.UnsafePackError, match=words):
        zips.unpack(safe_zip(tmp_path / "pack.zip"), target)
    assert list(target.iterdir()) == []


def test_a_file_that_is_not_a_zip_or_a_zip_without_a_manifest_is_refused(
    zips: ModuleType, tmp_path: Path
) -> None:
    text = tmp_path / "notes.zip"
    text.write_text("not a zip")
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(zips.UnsafePackError, match="not a readable zip"):
        zips.unpack(text, target)
    bare = tmp_path / "bare.zip"
    with zipfile.ZipFile(bare, "w") as handle:
        handle.writestr("Top/frames/startup/frame_01.png", b"x")
    with pytest.raises(zips.UnsafePackError, match="no manifest.json"):
        zips.unpack(bare, target)
    two = tmp_path / "two.zip"
    with zipfile.ZipFile(two, "w") as handle:
        handle.writestr("A/manifest.json", "{}")
        handle.writestr("B/manifest.json", "{}")
    with pytest.raises(zips.UnsafePackError, match="more than one folder"):
        zips.unpack(two, target)
    assert list(target.iterdir()) == []


def test_a_top_folder_beside_other_folders_is_the_pack_and_the_others_are_left_out(
    zips: ModuleType, tmp_path: Path
) -> None:
    archive = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Pack/manifest.json", "{}")
        handle.writestr("Other/notes.png", b"x")
    target = tmp_path / "out"
    target.mkdir()
    written, skipped = zips.unpack(archive, target)
    assert written == ["manifest.json"] and skipped == ["Other/notes.png"]


def test_a_folder_pack_is_read_where_it_is_and_a_symlink_or_a_flood_of_files_is_refused(
    zips: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "pack"
    (folder / "frames").mkdir(parents=True)
    (folder / "manifest.json").write_text("{}")
    (folder / "frames" / "a.png").write_bytes(b"x")
    (folder / ".DS_Store").write_bytes(b"junk")
    assert zips.list_folder(folder) == (["frames/a.png", "manifest.json"], "")
    (tmp_path / "outer").mkdir()
    (tmp_path / "outer" / "inner").mkdir()
    (tmp_path / "outer" / "inner" / "manifest.json").write_text("{}")
    assert zips.list_folder(tmp_path / "outer")[1] == "inner/"
    (folder / "link.png").symlink_to(tmp_path / "pack" / "frames" / "a.png")
    with pytest.raises(zips.UnsafePackError, match="symbolic link"):
        zips.list_folder(folder)
    (folder / "link.png").unlink()
    monkeypatch.setattr(zips, "MAX_ENTRIES", 1)
    with pytest.raises(zips.UnsafePackError, match="more than 1 files"):
        zips.list_folder(folder)


# ── check and apply ──


def test_a_valid_pack_passes_check_and_the_report_says_what_it_replaces(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = make_pack(
        tmp_path / "pack",
        stills=True,
        heads=True,
        brand=("app-icon", "menubar", "emblem"),
    )
    before = tree(pack)
    code, out = checked(art, capsys, pack)
    assert code == 0, out
    assert "Problems (0)" in out
    assert (
        "moves/startup.webp  " in out
        and "frames/startup/frame_01.png ... frame_03.png (3 frames)" in out
    )
    assert "moves/poster-done.webp" in out and "stills/done.png" in out
    assert "moves/face-idle.png" in out and "heads/head-smile.png" in out
    assert "docs/assets/melo.png" in out and "brand/emblem.png" in out
    assert "Missing" not in out
    assert "nothing was written" in out
    # Check writes nothing: the pack is as it was.
    assert tree(pack) == before


def test_check_on_a_missing_pack_says_so(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = checked(art, capsys, tmp_path / "nothing.zip")
    assert code == 1 and "no such file or folder" in out


def test_apply_twice_into_two_roots_gives_the_same_bytes(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = make_pack(tmp_path / "pack", stills=True, heads=True)
    roots = [tmp_path / "one", tmp_path / "two"]
    outs = []
    for index, root in enumerate(roots):
        code, out = run(
            art,
            capsys,
            "apply",
            str(pack),
            "-r",
            str(root),
            "-o",
            str(tmp_path / f"preview{index}"),
        )
        assert code == 0, out
        outs.append(out)
    first, second = tree(roots[0]), tree(roots[1])
    assert first == second
    moves = {name for name in first if name.startswith("support/gateway/static/moves/")}
    assert len(moves) == 25
    assert "support/gateway/static/moves/poster-offline.webp" in moves
    assert (
        "docs/assets/img/art-pack-moves.webp" in first
        and "docs/assets/img/art-pack-canvas.webp" in first
    )
    # The preview sheet is left where `-o` says, and the report ends with what changed and what to do.
    sheet = tmp_path / "preview0" / "preview.png"
    assert sheet.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (
        f"Preview: {sheet}" in outs[0] and "Next" in outs[0] and "Changed (" in outs[0]
    )
    # Nothing but the moves, the guide's pictures and (without a checkout's mark.py) no mark was written.
    assert not [
        name
        for name in first
        if not name.startswith(("support/gateway/static/moves/", "docs/assets/img/"))
    ]


def test_apply_without_o_leaves_the_preview_in_a_new_temporary_folder(
    art: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "temp"))
    (tmp_path / "temp").mkdir()
    code, out = run(
        art,
        capsys,
        "apply",
        str(make_pack(tmp_path / "pack")),
        "-r",
        str(tmp_path / "root"),
    )
    assert code == 0, out
    sheets = list((tmp_path / "temp").glob("melo-preview-*/preview.png"))
    assert len(sheets) == 1 and f"Preview: {sheets[0]}" in out


def test_apply_replaces_the_moves_folder_whole_and_rewrites_the_terminal_mark(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    moves = root / "support/gateway/static/moves"
    moves.mkdir(parents=True)
    (moves / "poster-stale.webp").write_bytes(b"old art")
    (moves / "face-idle.png").write_bytes(b"old art")
    (root / "support/cli").mkdir(parents=True)
    mark = root / "support/cli/mark.py"
    mark.write_text((ROOT / "support/cli/mark.py").read_text())
    pack = make_pack(tmp_path / "pack", stills=True, heads=True)
    code, out = run(
        art,
        capsys,
        "apply",
        str(pack),
        "-r",
        str(root),
        "-o",
        str(tmp_path / "preview"),
    )
    assert code == 0, out
    assert not (moves / "poster-stale.webp").exists()
    assert "removed  support/gateway/static/moves/poster-stale.webp" in out
    assert "changed  support/gateway/static/moves/face-idle.png" in out
    # The mark is computed from the new brand face: the same rows the script finds, written in the module.
    scripts = importlib.import_module("make_terminal_mark")
    rows = scripts.pixels(moves / "face-brand.png")
    assert scripts.block(rows) in mark.read_text()
    assert mark.read_text() != (ROOT / "support/cli/mark.py").read_text()
    assert "changed  support/cli/mark.py" in out


def test_a_pack_with_a_problem_writes_nothing_and_apply_refuses(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = make_pack(tmp_path / "pack")
    edit_manifest(pack, lambda document: document["states"].pop("thinking"))
    root = tmp_path / "root"
    code, out = run(
        art,
        capsys,
        "apply",
        str(pack),
        "-r",
        str(root),
        "-o",
        str(tmp_path / "preview"),
    )
    assert code == 1
    assert (
        "six states: the pack has no thinking" in out and "Nothing was written" in out
    )
    assert not root.exists() and not (tmp_path / "preview").exists()


# What each rule says when a pack breaks it: the change to a valid pack and the words of the report.
def corrupt(pack: Path, name: str, data: bytes) -> None:
    (pack / name).write_bytes(data)


def jpeg(pack: Path, name: str) -> None:
    from PIL import Image  # type: ignore[import-not-found]

    Image.new("RGB", (SIDE, SIDE), (200, 100, 50)).save(pack / name, "JPEG")


def case_ids(cases: list[tuple[str, Any, str]]) -> list[str]:
    return [case[0] for case in cases]


def set_state(pack: Path, state: str, **changes: Any) -> Path:
    """Change one state in the manifest; the pack, so that a rule's setup can list several changes."""
    edit_manifest(pack, lambda document: document["states"][state].update(changes))
    return pack


RULES: list[tuple[str, Any, str]] = [
    (
        "no state",
        lambda pack: edit_manifest(pack, lambda d: d["states"].pop("idle_play")),
        "six states: the pack has no idle_play",
    ),
    (
        "too many frames",
        lambda pack: set_state(
            pack,
            "working",
            frames=[f"frames/working/frame_{n:02}.png" for n in range(17)],
        ),
        "frame count: working: 1 to 16 frames, not 17",
    ),
    (
        "no frames",
        lambda pack: set_state(pack, "working", frames=[]),
        "frame count: working: 1 to 16 frames, not 0",
    ),
    (
        "frames not in order",
        lambda pack: set_state(
            pack,
            "working",
            frames=list(
                reversed(
                    json.loads((pack / "manifest.json").read_text())["states"][
                        "working"
                    ]["frames"]
                )
            ),
        ),
        "frame names: working: frame 0 is frames/working/frame_03.png, expected frames/working/frame_01.png",
    ),
    (
        "frames in another folder",
        lambda pack: (
            (pack / "shots").mkdir(),
            (pack / "frames/working/frame_01.png").rename(pack / "shots/a.png"),
            set_state(
                pack,
                "working",
                frames=[
                    "shots/a.png",
                    "frames/working/frame_02.png",
                    "frames/working/frame_03.png",
                ],
            ),
        ),
        "frame names: working: frame 0 is shots/a.png",
    ),
    (
        "a frame is not a file",
        lambda pack: (pack / "frames/working/frame_02.png").unlink(),
        "frame files: working: frames/working/frame_02.png is not a file of the pack",
    ),
    (
        "a frame outside the pack",
        lambda pack: set_state(
            pack,
            "working",
            frames=[
                "../outside.png",
                "frames/working/frame_02.png",
                "frames/working/frame_03.png",
            ],
        ),
        "frame files: working: ../outside.png is not a file of the pack",
    ),
    (
        "too few durations",
        lambda pack: set_state(pack, "working", frame_durations_ms=[100, 200]),
        "durations: working: one positive duration",
    ),
    (
        "a zero duration",
        lambda pack: set_state(pack, "working", frame_durations_ms=[100, 0, 300]),
        "durations: working: one positive duration",
    ),
    (
        "a fraction of a duration",
        lambda pack: set_state(pack, "working", frame_durations_ms=[100, 200.5, 300]),
        "durations: working: one positive duration",
    ),
    (
        "a boolean duration",
        lambda pack: set_state(pack, "working", frame_durations_ms=[100, True, 300]),
        "durations: working: one positive duration",
    ),
    (
        "an unknown playback",
        lambda pack: set_state(pack, "working", playback="twice"),
        'playback: working: playback is "once" or "loop"',
    ),
    (
        "a negative loop count",
        lambda pack: set_state(pack, "working", loop_count=-1),
        'playback: working: playback is "once" or "loop"',
    ),
    (
        "a poster past the end",
        lambda pack: set_state(pack, "working", poster_frame_index=3),
        "poster frame: working: the poster is not one of its frames",
    ),
    (
        "a canvas that is not square",
        lambda pack: edit_manifest(pack, lambda d: d["canvas"].update(height=32)),
        "square canvas: the pack's canvas must be square",
    ),
    (
        "no anchor",
        lambda pack: edit_manifest(pack, lambda d: d["canvas"].pop("anchor")),
        "anchor: the pack's canvas has no anchor",
    ),
    (
        "an anchor outside the canvas",
        lambda pack: edit_manifest(
            pack, lambda d: d["canvas"].update(anchor={"x": 100, "y": 60})
        ),
        "anchor: the anchor (100, 60) is outside the 64 x 64 canvas",
    ),
    (
        "a canvas past the size limit",
        lambda pack: edit_manifest(
            pack, lambda d: d["canvas"].update(width=5000, height=5000)
        ),
        "sane size: the canvas is 5000 px a side, more than 4096",
    ),
    (
        "a manifest that is not JSON",
        lambda pack: (pack / "manifest.json").write_text("{not json"),
        "manifest: manifest.json cannot be read",
    ),
    (
        "a frame without alpha",
        lambda pack: draw(pack / "frames/working/frame_01.png", mode="RGB"),
        "RGBA: frames/working/frame_01.png is RGB, not RGBA",
    ),
    (
        "a frame drawn into its corner",
        lambda pack: draw(pack / "frames/working/frame_01.png", box=(0, 0, 40, 40)),
        "transparent corners: frames/working/frame_01.png has something drawn in a corner",
    ),
    (
        "an empty frame",
        lambda pack: (pack / "frames/working/frame_01.png").write_bytes(_blank(SIDE)),
        "transparent corners: frames/working/frame_01.png has nothing drawn",
    ),
    (
        "a frame off the canvas",
        lambda pack: draw(pack / "frames/working/frame_02.png", size=(32, 32)),
        "on the canvas: frames/working/frame_02.png is 32 x 32 px, not on the 64 x 64 canvas",
    ),
    (
        "a frame that is not a picture",
        lambda pack: corrupt(pack, "frames/working/frame_03.png", b"not a png at all"),
        "readable PNG: frames/working/frame_03.png cannot be read as a PNG",
    ),
    (
        "a frame that is a JPEG",
        lambda pack: jpeg(pack, "frames/working/frame_03.png"),
        "readable PNG: frames/working/frame_03.png is a JPEG file",
    ),
    (
        "a still that is too wide",
        lambda pack: draw(pack / "stills/done.png", size=(4097, 1), box=(10, 0, 20, 0)),
        "sane size: stills/done.png is 4097 x 1 px, more than 4096 px a side",
    ),
    (
        "a head off the canvas",
        lambda pack: draw(pack / "heads/head-smile.png", size=(32, 32)),
        "on the canvas: heads/head-smile.png is 32 x 32 px",
    ),
    (
        "an app icon that is not square",
        lambda pack: draw(pack / "brand/app-icon.png", (1024, 512), "RGB"),
        "app icon: brand/app-icon.png is 1024 x 512 px: it must be square",
    ),
    (
        "an app icon that is too small",
        lambda pack: draw(pack / "brand/app-icon.png", (256, 256), "RGB"),
        "app icon: brand/app-icon.png is 256 px a side: at least 1024 px (512 px is a warning)",
    ),
    (
        "a menu-bar logo that is not square",
        lambda pack: draw(pack / "brand/menubar.png", (40, 20), "RGB"),
        "menu-bar logo: brand/menubar.png is 40 x 20 px: it must be square",
    ),
    (
        "a brand picture that is not a picture",
        lambda pack: corrupt(pack, "brand/emblem.png", b"nope"),
        "readable PNG: brand/emblem.png cannot be read as a PNG",
    ),
]


def _blank(side: int) -> bytes:
    """A fully transparent PNG, as bytes."""
    import io

    from PIL import Image  # type: ignore[import-not-found]

    buffer = io.BytesIO()
    Image.new("RGBA", (side, side)).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("name, change, words", RULES, ids=case_ids(RULES))
def test_each_rule_names_itself_when_a_pack_breaks_it(
    art: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    change: Any,
    words: str,
) -> None:
    pack = make_pack(
        tmp_path / "pack",
        stills=True,
        heads=True,
        brand=("app-icon", "menubar", "emblem"),
    )
    change(pack)
    code, out = checked(art, capsys, pack)
    assert code == 1, out
    assert words in out, out
    assert "the pack has problems; apply would refuse" in out
    # A pack with a problem is not built, so the report lists nothing it would replace.
    assert "Replaces" not in out


def test_a_pack_that_is_over_the_size_budget_is_a_problem(
    art: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(art.build_moves, "MAX_BYTES", 1000)
    code, out = checked(art, capsys, make_pack(tmp_path / "pack"))
    assert code == 1 and "size budget: the moves weigh" in out


def test_one_habit_broken_in_every_frame_is_a_short_entry_not_a_wall(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = make_pack(tmp_path / "pack")
    for path in sorted((pack / "frames").rglob("*.png")):
        draw(path, mode="RGB")
    code, out = checked(art, capsys, pack)
    assert code == 1
    assert out.count("RGBA: frames/") == 4 and "RGBA: and 14 more like these" in out


@pytest.mark.parametrize("start", [0, 1])
def test_frames_may_be_counted_from_zero_or_from_one(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], start: int
) -> None:
    code, out = checked(art, capsys, make_pack(tmp_path / "pack", start=start))
    assert code == 0, out


WARNINGS: list[tuple[str, Any, str]] = [
    (
        "an extra state",
        lambda pack: edit_manifest(
            pack, lambda d: d["states"].update(dance=d["states"]["working"])
        ),
        "six states: the manifest also lists dance; only the six are used",
    ),
    (
        "a startup that loops",
        lambda pack: set_state(pack, "startup", playback="loop", loop_count=0),
        "playback: startup loops",
    ),
    (
        "a working that plays once",
        lambda pack: set_state(pack, "working", playback="once"),
        "playback: working plays once and stops on its last frame",
    ),
    (
        "paws far from the anchor line",
        lambda pack: draw(pack / "frames/working/frame_01.png", box=(16, 10, 40, 39)),
        "paws on the anchor line: working: 1 of 3 picture(s) rest more than 2 px from the anchor line (first: frames/working/frame_01.png, -20 px)",
    ),
    (
        "a head that leaves its box",
        lambda pack: draw(pack / "heads/head-smile.png", box=(0, 20, 60, 40)),
        "head box: heads/head-smile.png draws outside HEAD_BOX",
    ),
    (
        "an app icon under 1024 px",
        lambda pack: draw(pack / "brand/app-icon.png", (600, 600), "RGB"),
        "app icon: brand/app-icon.png is 600 px a side; 1024 px or more keeps the largest size sharp",
    ),
    (
        "a menu-bar logo smaller than the website's",
        lambda pack: draw(pack / "brand/menubar.png", (32, 32), "RGB"),
        "menu-bar logo: brand/menubar.png is 32 px a side; it is scaled up to 64 px",
    ),
]


@pytest.mark.parametrize("name, change, words", WARNINGS, ids=case_ids(WARNINGS))
def test_a_warning_does_not_stop_apply_and_names_its_rule(
    art: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    change: Any,
    words: str,
) -> None:
    pack = make_pack(
        tmp_path / "pack",
        stills=True,
        heads=True,
        brand=("app-icon", "menubar", "emblem"),
    )
    change(pack)
    code, out = checked(art, capsys, pack)
    assert code == 0, out
    assert words in out, out
    assert "Warnings (0)" not in out


def test_a_canvas_smaller_than_the_moves_is_a_warning(
    art: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(art.build_moves, "CELL", 96)
    code, out = checked(art, capsys, make_pack(tmp_path / "pack"))
    assert code == 0, out
    assert "canvas size: the canvas is 64 px a side; the moves are made at 96 px" in out


# ── What is missing, and what is used instead ──


def moves_of(root: Path) -> Path:
    return root / "support/gateway/static/moves"


def applied(
    art: ModuleType, capsys: pytest.CaptureFixture[str], pack: Path, root: Path
) -> tuple[int, str]:
    return run(
        art,
        capsys,
        "apply",
        str(pack),
        "-r",
        str(root),
        "-o",
        str(root.parent / f"preview-{root.name}"),
    )


def test_without_stills_heads_or_brand_the_report_names_what_stands_in(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = make_pack(tmp_path / "pack")
    code, out = checked(art, capsys, pack)
    assert code == 0, out
    for line in (
        "stills/done.png: poster-done.webp is a copy of the startup poster; face-done is cut from the startup poster",
        "stills/error.png: poster-error.webp is a copy of the thinking poster; face-error is cut from the thinking poster",
        "stills/offline.png: poster-offline.webp is a copy of the idle_rest poster",
        "heads/head-smile.png: face-idle is cut from the startup poster",
        "heads/head-neutral.png: face-brand is the idle face; the tab icons are cut from the startup poster",
        "brand/app-icon.png: AppIcon.icns stays as it is",
        "brand/menubar.png: MenuBarLogo.png, MenuBarLogo@2x.png and the website's round logo stay as they are",
        "brand/emblem.png: docs/assets/melo.png (the README emblem) stays as it is",
    ):
        assert line in out, line
    root = tmp_path / "root"
    code, out = applied(art, capsys, pack, root)
    assert code == 0, out
    moves = moves_of(root)
    # The three posters are copies of the states that stand in, so the console never asks for a file that is not there.
    assert (moves / "poster-done.webp").read_bytes() == (
        moves / "poster-startup.webp"
    ).read_bytes()
    assert (moves / "poster-error.webp").read_bytes() == (
        moves / "poster-thinking.webp"
    ).read_bytes()
    assert (moves / "poster-offline.webp").read_bytes() == (
        moves / "poster-idle_rest.webp"
    ).read_bytes()
    # The faces and tab icons are cut from the pack's posters.
    assert (moves / "face-done.png").read_bytes() == (
        moves / "face-idle.png"
    ).read_bytes()
    assert (moves / "face-brand.png").read_bytes() == (
        moves / "face-idle.png"
    ).read_bytes()
    # No brand picture: nothing of the brand is written.
    assert not [
        name
        for name in tree(root)
        if name.startswith(("macos/", "docs/assets/melo", "docs/assets/icon"))
    ]


def test_stills_and_heads_change_only_what_they_feed(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = make_pack(tmp_path / "pack")
    plain = tmp_path / "plain"
    assert applied(art, capsys, pack, plain)[0] == 0
    make_pack(pack, stills=True)
    with_stills = tmp_path / "stills"
    assert applied(art, capsys, pack, with_stills)[0] == 0
    make_pack(pack, stills=True, heads=True)
    with_heads = tmp_path / "heads"
    assert applied(art, capsys, pack, with_heads)[0] == 0
    names = {name for name in tree(moves_of(plain))}

    def changed(left: Path, right: Path) -> set[str]:
        a, b = tree(moves_of(left)), tree(moves_of(right))
        return {name for name in names if a[name] != b[name]}

    assert changed(plain, with_stills) == {
        "poster-done.webp",
        "poster-error.webp",
        "poster-offline.webp",
        "face-done.png",
        "face-error.png",
    }
    assert changed(with_stills, with_heads) == {
        "face-idle.png",
        "face-brand.png",
        "favicon-16.png",
        "favicon-32.png",
        "favicon-64.png",
    }


def test_one_still_given_replaces_only_its_own_stand_in(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = make_pack(tmp_path / "pack")
    draw(pack / "stills/error.png", colour=(10, 20, 220))
    code, out = checked(art, capsys, pack)
    assert code == 0, out
    assert (
        "stills/done.png: poster-done.webp is a copy" in out
        and "stills/error.png: poster-error" not in out
    )
    root = tmp_path / "root"
    assert applied(art, capsys, pack, root)[0] == 0
    moves = moves_of(root)
    assert (moves / "poster-error.webp").read_bytes() != (
        moves / "poster-thinking.webp"
    ).read_bytes()
    assert (moves / "poster-done.webp").read_bytes() == (
        moves / "poster-startup.webp"
    ).read_bytes()


def test_stills_and_heads_at_the_pack_root_are_accepted_and_a_folder_copy_wins(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = make_pack(tmp_path / "pack")
    draw(pack / "done.png", colour=(1, 2, 3))
    draw(pack / "head-smile.png", box=(10, 6, 54, 60), colour=(9, 8, 7))
    code, out = checked(art, capsys, pack)
    assert code == 0, out
    assert "moves/poster-done.webp" in out and "<-  done.png (still)" in out
    assert "<-  head-smile.png" in out and "Ignored" not in out
    draw(pack / "stills/done.png", colour=(200, 0, 0))
    code, out = checked(art, capsys, pack)
    assert code == 0
    assert "stills: done.png is given twice; stills/done.png is used" in out
    assert "<-  stills/done.png (still)" in out


@pytest.mark.skipif(not ICONUTIL.exists(), reason="the app icon needs macOS's iconutil")
def test_brand_pictures_replace_the_icon_the_menu_bar_logo_the_emblem_and_the_websites_logo(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from PIL import Image  # type: ignore[import-not-found]

    pack = make_pack(tmp_path / "pack", brand=("app-icon", "menubar", "emblem"))
    root = tmp_path / "root"
    assets = root / "macos/assets"
    assets.mkdir(parents=True)
    (assets / "AppIcon.icns").write_bytes(b"old icon")
    (assets / "MenuBarLogo.png").write_bytes(b"old logo")
    code, out = applied(art, capsys, pack, root)
    assert code == 0, out
    for line in (
        "macos/assets/AppIcon.icns",
        "macos/assets/MenuBarLogo.png",
        "macos/assets/MenuBarLogo@2x.png",
        "docs/assets/icon_small.png",
    ):
        assert line in out
    assert (
        "changed  macos/assets/AppIcon.icns" in out
        and "new      macos/assets/MenuBarLogo@2x.png" in out
    )
    assert (assets / "AppIcon.icns").read_bytes()[:4] == b"icns"
    with (
        Image.open(assets / "MenuBarLogo.png") as logo,
        Image.open(assets / "MenuBarLogo@2x.png") as big,
    ):
        assert logo.size == (18, 18) and big.size == (36, 36)
        assert logo.mode == "RGBA"
    # The README emblem is the pack's picture as it is; the website's logo is a 64 px square.
    assert (root / "docs/assets/melo.png").read_bytes() == (
        pack / "brand/emblem.png"
    ).read_bytes()
    with Image.open(root / "docs/assets/icon_small.png") as small:
        assert small.size == (64, 64)


@pytest.mark.skipif(not ICONUTIL.exists(), reason="the app icon needs macOS's iconutil")
def test_without_an_app_icon_the_emblem_draws_it_and_what_the_pack_lacks_stays(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = make_pack(tmp_path / "pack", brand=("emblem",))
    root = tmp_path / "root"
    (root / "macos/assets").mkdir(parents=True)
    (root / "macos/assets/MenuBarLogo.png").write_bytes(b"old logo")
    code, out = applied(art, capsys, pack, root)
    assert code == 0, out
    assert "brand/app-icon.png: the app icon is drawn from brand/emblem.png" in out
    assert "app icon: brand/emblem.png is 40 x 50 px and draws the icon" in out
    assert (root / "macos/assets/AppIcon.icns").read_bytes()[:4] == b"icns"
    assert (root / "macos/assets/MenuBarLogo.png").read_bytes() == b"old logo"
    assert not (root / "docs/assets/icon_small.png").exists()


# ── Folders and zips ──


@pytest.mark.parametrize("top", ["", "Melo_pack/"])
def test_a_zip_with_the_pack_at_its_root_or_in_its_top_folder_gives_the_same_pictures(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], top: str
) -> None:
    pack = make_pack(tmp_path / "pack", stills=True, heads=True)
    archive = zipped(
        pack,
        tmp_path / "pack.zip",
        top,
        {"__MACOSX/._manifest.json": b"junk", f"{top}startup.gif": b"GIF"},
    )
    code, out = checked(art, capsys, archive)
    assert code == 0, out
    assert "startup.gif" in out and "__MACOSX" not in out and ".DS_Store" not in out
    from_zip, from_folder = tmp_path / "from-zip", tmp_path / "from-folder"
    assert applied(art, capsys, archive, from_zip)[0] == 0
    assert applied(art, capsys, pack, from_folder)[0] == 0
    assert tree(from_zip) == tree(from_folder)


def test_a_zip_with_an_escaping_name_and_a_symlink_is_refused_and_nothing_is_written(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = make_pack(tmp_path / "pack")
    archive = zipped(
        pack, tmp_path / "evil.zip", "Pack/", {"../evil.png": b"\x89PNG evil"}
    )
    with zipfile.ZipFile(archive, "a") as handle:
        link = zipfile.ZipInfo("Pack/frames/startup/link.png")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        handle.writestr(link, "/etc/passwd")
    root = tmp_path / "root"
    code, out = run(
        art,
        capsys,
        "apply",
        str(archive),
        "-r",
        str(root),
        "-o",
        str(tmp_path / "preview"),
    )
    assert code == 1
    assert "Problems (1)" in out and "zip safety:" in out and "climbs out" in out
    assert "Nothing was written" in out
    assert (
        not root.exists()
        and not (tmp_path / "preview").exists()
        and not (tmp_path / "evil.png").exists()
    )
    code, out = checked(art, capsys, archive)
    assert code == 1 and "zip safety:" in out
    # Each unsafe entry alone is refused as well.
    only_link = tmp_path / "link.zip"
    with zipfile.ZipFile(only_link, "w") as handle:
        handle.writestr("manifest.json", "{}")
        link = zipfile.ZipInfo("frames/startup/frame_01.png")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        handle.writestr(link, "/etc/passwd")
    code, out = checked(art, capsys, only_link)
    assert (
        code == 1
        and "zip safety: 'frames/startup/frame_01.png': a symbolic link" in out
    )


def test_a_folder_with_a_symlink_is_refused(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = make_pack(tmp_path / "pack")
    (pack / "frames/startup/shortcut.png").symlink_to(
        pack / "frames/startup/frame_01.png"
    )
    code, out = checked(art, capsys, pack)
    assert code == 1 and "folder safety:" in out and "symbolic link" in out


def test_a_folder_without_a_manifest_is_refused(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "empty").mkdir()
    code, out = checked(art, capsys, tmp_path / "empty")
    assert code == 1 and "folder safety: no manifest.json" in out


def test_files_the_system_does_not_know_are_ignored_and_never_copied(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = make_pack(tmp_path / "pack")
    for name in (
        "posters/startup.png",
        "posters/thinking.png",
        "README.md",
        "preview.html",
        "notes/a.txt",
        "frames/startup/frame_09.png",
    ):
        (pack / name).parent.mkdir(parents=True, exist_ok=True)
        (pack / name).write_bytes(b"not read")
    code, out = checked(art, capsys, pack)
    assert code == 0, out
    ignored = out.split("Ignored", 1)[1]
    for line in (
        "README.md",
        "preview.html",
        "posters/ (2 files)",
        "notes/ (1 files)",
        "frames/ (1 files)",
    ):
        assert line in ignored, line
    root = tmp_path / "root"
    assert applied(art, capsys, pack, root)[0] == 0
    assert b"not read" not in b"".join(tree(root).values())


# ── The guide's pictures ──


def test_the_guides_pictures_are_drawn_from_the_moves_and_are_the_same_each_time(
    art: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from PIL import Image  # type: ignore[import-not-found]

    code, out = run(art, capsys, "figures", "-o", str(tmp_path / "one"))
    assert code == 0 and "art-pack-moves.webp, art-pack-canvas.webp" in out
    assert run(art, capsys, "figures", "-o", str(tmp_path / "two"))[0] == 0
    assert tree(tmp_path / "one") == tree(tmp_path / "two")
    for name in ("art-pack-moves.webp", "art-pack-canvas.webp"):
        with Image.open(tmp_path / "one" / name) as picture:
            assert picture.format == "WEBP" and picture.width <= 1280
    # They are drawn from the committed moves: with the pinned Pillow the same pictures are in the docs folder.
    if art.Image.__version__ != art.PINNED_PILLOW:
        pytest.skip(f"the committed pictures are drawn with Pillow {art.PINNED_PILLOW}")
    for name in ("art-pack-moves.webp", "art-pack-canvas.webp"):
        committed = ROOT / "docs/assets/img" / name
        assert committed.read_bytes() == (tmp_path / "one" / name).read_bytes()
