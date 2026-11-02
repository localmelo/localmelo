#!/usr/bin/env python3
"""Check an art pack and replace every picture of Melo with it, in one run. The format and the art rules are in
docs/guides/art-pack.md.

    uv run --with pillow==13.0.0 python scripts/art_pack.py check <pack.zip or pack folder>
    uv run --with pillow==13.0.0 python scripts/art_pack.py apply <pack.zip or pack folder> [-o DIR]
    uv run --with pillow==13.0.0 python scripts/art_pack.py figures

Pillow is not a dependency of localmelo: `--with` fetches it for this one run. The committed pictures were made with
Pillow 13.0.0, whose wheel bundles libwebp 1.6.0; another Pillow or libwebp may encode other bytes, so keep the pin
unless the pictures are made again on purpose.

`check` writes nothing to the checkout. It reads the pack, tests every rule, builds the pictures in a temporary
folder that it removes (the size budget needs them), and prints a report: what the pack replaces, what is missing and
what is used instead, the problems that make `apply` refuse, and the warnings. The exit code is 0 without problems.

`apply` checks, then replaces in one run: the moves folder (support/gateway/static/moves, the same bytes for the same
pack), the terminal welcome's mark (support/cli/mark.py), the brand pictures when the pack has them (the app icon, the
menu-bar logo, the README emblem and the website's round logo) and the two pictures of the guide. It writes a preview
sheet of everything to a temporary folder (or `-o DIR`) and prints it with the changed files and the next steps.
`-r ROOT` writes under ROOT instead of this checkout.

The pack is a folder, or a zip whose root is the pack or holds it as its one top folder:

    manifest.json                          the states, frames, durations, playback, poster frames and the canvas
    frames/<state>/frame_NN.png            the six states: startup, thinking, working, attention, idle_rest, idle_play
    stills/done.png error.png offline.png  optional: what Melo holds when nothing moves (also accepted at the root)
    heads/head-smile.png head-neutral.png  optional: the head drawn for small sizes (also accepted at the root)
    brand/app-icon.png menubar.png emblem.png   optional

Anything else in the pack is listed as ignored. The pack is untrusted data: only PNG and JSON files are read, nothing
in it is run, and a zip is checked before anything is written (scripts/pack_zip.py).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import build_moves  # type: ignore[import-not-found]
import make_app_icons  # type: ignore[import-not-found]
import make_terminal_mark  # type: ignore[import-not-found]
import pack_sheets  # type: ignore[import-not-found]
import pack_zip  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
# Where the pictures go, from a checkout's root.
MOVES = "support/gateway/static/moves"
ICONS = "macos/assets"
EMBLEM = "docs/assets/melo.png"
LOGO = "docs/assets/icon_small.png"
MARK = "support/cli/mark.py"
FIGURES = "docs/assets/img"
LOGO_PIXELS = 64  # the website's round logo is a 64 px square, as it was
PINNED_PILLOW = "13.0.0"
BRAND = ("app-icon", "menubar", "emblem")
# The ground line is 2% of the canvas wide: the art's own shadow sits a little under the paws.
GROUND_SHARE = 0.02
# The picture size the guide asks of the app icon and the least it takes.
ICON_SIDE, ICON_LEAST = 1024, 512
EMBLEM_BYTES = 2_000_000
# The room a clear corner needs: a block this many pixels square (less on a tiny canvas) with nothing drawn.
CORNER = 8
SOLID = 128  # an alpha above this is "drawn"
SHOWN = 4  # problems of one rule shown before "and N more"


@dataclass
class Report:
    """What `check` found, grouped as it is printed."""

    pack: str
    replaces: list[tuple[str, str]] = field(default_factory=list)
    instead: list[tuple[str, str]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)

    def problem(self, rule: str, text: str) -> None:
        self.problems.append(f"{rule}: {text}")

    def warn(self, rule: str, text: str) -> None:
        self.warnings.append(f"{rule}: {text}")

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self) -> str:
        lines = [f"Art pack: {self.pack}", ""]
        if self.replaces:
            lines.append("Replaces (moves/ is support/gateway/static/moves/)")
            width = max(len(name) for name, _ in self.replaces)
            lines += [
                f"  {name:<{width}}  <-  {source}" for name, source in self.replaces
            ]
            lines.append("")
        if self.instead:
            lines.append("Missing -> uses instead")
            lines += [f"  {name}: {text}" for name, text in self.instead]
            lines.append("")
        lines.append(f"Problems ({len(self.problems)})")
        lines += [f"  {line}" for line in squeeze(self.problems)] or ["  none"]
        lines.append("")
        lines.append(f"Warnings ({len(self.warnings)})")
        lines += [f"  {line}" for line in squeeze(self.warnings)] or ["  none"]
        if self.ignored:
            lines += ["", f"Ignored ({len(self.ignored)})"]
            lines += [f"  {line}" for line in self.ignored]
        return "\n".join(lines)


def squeeze(lines: list[str]) -> list[str]:
    """Findings of one rule, the first few in full and the rest as a count, so one broken habit is one short entry."""
    shown: dict[str, int] = {}
    more: dict[str, int] = {}
    out: list[str] = []
    order: list[str] = []
    for line in lines:
        rule = line.split(":", 1)[0]
        if rule not in order:
            order.append(rule)
        shown[rule] = shown.get(rule, 0) + 1
        if shown[rule] <= SHOWN:
            out.append(line)
        else:
            more[rule] = more.get(rule, 0) + 1
    for rule, count in more.items():
        # Put each count after the last line of its rule.
        last = max(
            index for index, line in enumerate(out) if line.startswith(rule + ":")
        )
        out.insert(last + 1, f"{rule}: and {count} more like these")
    return out


@dataclass
class Pack:
    """A pack that can be read: the folder that holds its files, the names of those files, and the names left out."""

    root: Path
    files: set[str]
    skipped: list[str]
    label: str


@contextmanager
def open_pack(source: Path) -> Iterator[Pack]:
    """The pack at `source`, a zip (unpacked into a temporary folder that is removed after) or a folder (read where it
    is). Raises UnsafePackError for what must not be read."""
    if not source.exists():
        raise pack_zip.UnsafePackError(f"{source}: no such file or folder")
    if source.is_dir():
        names, prefix = pack_zip.list_folder(source)
        inside = {name[len(prefix) :] for name in names if name.startswith(prefix)}
        outside = [name for name in names if not name.startswith(prefix)]
        yield Pack(source / prefix.rstrip("/"), inside, outside, str(source))
        return
    with tempfile.TemporaryDirectory(prefix="art-pack-") as temp:
        written, skipped = pack_zip.unpack(source, Path(temp))
        yield Pack(Path(temp), set(written), skipped, str(source))


@dataclass
class Look:
    """What the checks learn about one picture."""

    size: tuple[int, int]
    solid: tuple[int, int, int, int] | None = None
    transparent: bool = False
    profile: bool = False


def examine(
    path: Path,
    label: str,
    report: Report,
    canvas: int | None = None,
    art: bool = True,
) -> Look | None:
    """Read one picture and test the rules that need its pixels: a readable PNG of a sane size; for the art (frames,
    stills and heads) on the pack's canvas, RGBA, with clear corners and something drawn. None when it cannot go on.
    """
    try:
        if path.stat().st_size > pack_zip.MAX_ENTRY_BYTES:
            report.problem(
                "sane size",
                f"{label} is larger than {pack_zip.MAX_ENTRY_BYTES // 2**20} MB",
            )
            return None
        with Image.open(path) as image:
            if image.format != "PNG":
                report.problem(
                    "readable PNG", f"{label} is a {image.format} file, not a PNG"
                )
                return None
            width, height = image.size
            if max(width, height) > build_moves.MAX_SIDE:
                report.problem(
                    "sane size",
                    f"{label} is {width} x {height} px, more than {build_moves.MAX_SIDE} px a side",
                )
                return None
            image.load()
            mode = image.mode
            alpha = image.getchannel("A") if mode in ("RGBA", "LA") else None
            profile = image.info.get("icc_profile")
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        report.problem("readable PNG", f"{label} cannot be read as a PNG ({error})")
        return None
    look = Look(
        (width, height),
        transparent=alpha is not None and alpha.getextrema()[0] < 255,
        profile=bool(profile) and b"sRGB" not in bytes(profile),
    )
    if not art:
        return look
    if canvas is not None and (width, height) != (canvas, canvas):
        report.problem(
            "on the canvas",
            f"{label} is {width} x {height} px, not on the {canvas} x {canvas} canvas",
        )
    if mode != "RGBA" or alpha is None:
        report.problem(
            "RGBA", f"{label} is {mode}, not RGBA: export it with an alpha channel"
        )
        return look
    block = max(1, min(CORNER, width // 8, height // 8))
    corners = [
        (0, 0),
        (width - block, 0),
        (0, height - block),
        (width - block, height - block),
    ]
    loudest = max(
        alpha.crop((x, y, x + block, y + block)).getextrema()[1] for x, y in corners
    )
    solid = alpha.point(lambda value: 255 if value > SOLID else 0).getbbox()
    look.solid = solid
    if loudest > 0:
        report.problem(
            "transparent corners",
            f"{label} has something drawn in a corner (alpha up to {loudest}): the background must be clear",
        )
    elif solid is None:
        report.problem(
            "transparent corners", f"{label} has nothing drawn: it is fully transparent"
        )
    return look


def named_frames(state: str, names: list[str]) -> str | None:
    """None when `names` are frames/<state>/frame_NN.png counted without a gap from 00 (or from 01), in play order;
    else what is wrong."""
    start = 0 if f"frames/{state}/frame_00.png" in names else 1
    for number, name in enumerate(names):
        wanted = f"frames/{state}/frame_{start + number:02}.png"
        if name != wanted:
            return (
                f"{state}: frame {number} is {name}, expected {wanted} "
                "(frames are counted without a gap from frame_00 or frame_01, in play order)"
            )
    return None


def rel(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def source_text(paths: list[Path], note: str, root: Path) -> str:
    """The pack files a picture is made of: one name, or the first and the last of a folder's frames."""
    names = [rel(path, root) for path in paths]
    if len(names) == 1:
        where = names[0]
    elif len({name.rsplit("/", 1)[0] for name in names}) == 1:
        where = f"{names[0]} ... {names[-1].rsplit('/', 1)[-1]}"
    else:
        where = f"{names[0]} ... {names[-1]}"
    return f"{where} ({note})" if note else where


def pictures_of(
    pack: Pack, report: Report, used: set[str], side: int | None
) -> tuple[dict[str, Path], dict[str, Look]]:
    """The stills and heads the pack gives, by name, each looked at; the same file at the root and in its folder
    counts once (the folder's)."""
    found: dict[str, Path] = {}
    looks: dict[str, Look] = {}
    for name in (*build_moves.STILLS, *build_moves.HEADS.values()):
        folder = "heads" if name in build_moves.HEADS.values() else "stills"
        here = [c for c in (f"{folder}/{name}.png", f"{name}.png") if c in pack.files]
        used.update(here)
        if not here:
            continue
        if len(here) == 2:
            report.warn(folder, f"{name}.png is given twice; {here[0]} is used")
        look = examine(pack.root / here[0], here[0], report, side)
        if look is not None:
            found[name] = pack.root / here[0]
            looks[name] = look
            if name in build_moves.HEADS.values() and side and look.solid:
                check_head(here[0], look.solid, side, report)
    return found, looks


def check_head(
    label: str, solid: tuple[int, int, int, int], side: int, report: Report
) -> None:
    """A head must lie inside HEAD_BOX: what is outside it is cut off from the face."""
    unit, slack = side / 512, side / 256
    box = [edge * unit for edge in build_moves.HEAD_BOX]
    if (
        solid[0] < box[0] - slack
        or solid[1] < box[1] - slack
        or solid[2] > box[2] + slack
        or solid[3] > box[3] + slack
    ):
        report.warn(
            "head box",
            f"{label} draws outside HEAD_BOX {build_moves.HEAD_BOX} (of a 512 px canvas); that part is cut off the face",
        )


def brand_of(pack: Pack, report: Report, used: set[str]) -> dict[str, Path]:
    """The brand pictures the pack gives, looked at: the app icon and the menu-bar logo square, the icon big enough."""
    found: dict[str, Path] = {}
    for name in BRAND:
        label = f"brand/{name}.png"
        if label not in pack.files:
            continue
        used.add(label)
        look = examine(pack.root / label, label, report, art=False)
        if look is None:
            continue
        width, height = look.size
        found[name] = pack.root / label
        if look.profile:
            report.warn(
                "colour profile",
                f"{label} has a colour profile that is not sRGB; its colours are read as sRGB",
            )
        if name == "emblem":
            weight = (pack.root / label).stat().st_size
            if weight > EMBLEM_BYTES:
                report.warn(
                    "emblem",
                    f"{label} weighs {weight / 1e6:.1f} MB; the README and the website load it",
                )
            continue
        rule = "app icon" if name == "app-icon" else "menu-bar logo"
        if width != height:
            report.problem(rule, f"{label} is {width} x {height} px: it must be square")
        elif name == "app-icon":
            if width < ICON_LEAST:
                report.problem(
                    rule,
                    f"{label} is {width} px a side: at least {ICON_SIDE} px (512 px is a warning)",
                )
            elif width < ICON_SIDE:
                report.warn(
                    rule,
                    f"{label} is {width} px a side; {ICON_SIDE} px or more keeps the largest size sharp",
                )
        elif width < LOGO_PIXELS:
            report.warn(
                rule,
                f"{label} is {width} px a side; it is scaled up to {LOGO_PIXELS} px for the website",
            )
        if name == "app-icon" and look.transparent:
            report.warn(rule, f"{label} has transparent pixels; they are laid on white")
    return found


def grouped(names: list[str]) -> list[str]:
    """The ignored files: a folder as one line with its count, and three or more loose files of one kind likewise."""
    folders: dict[str, int] = {}
    kinds: dict[str, list[str]] = {}
    for name in sorted(names):
        if "/" in name:
            top = name.split("/", 1)[0] + "/"
            folders[top] = folders.get(top, 0) + 1
        else:
            kinds.setdefault(Path(name).suffix, []).append(name)
    lines = []
    for suffix, loose in kinds.items():
        if suffix and len(loose) >= 3:
            lines.append(f"*{suffix} ({len(loose)} files)")
        else:
            lines += loose
    return [*lines, *(f"{top} ({count} files)" for top, count in folders.items())]


def instead_for(
    pictures: dict[str, Path], brand: dict[str, Path]
) -> list[tuple[str, str]]:
    """What stands in for each optional part the pack does not have."""
    lines: list[tuple[str, str]] = []
    standins = build_moves.STILL_STAND_INS
    faces = {"done": "face-done", "error": "face-error"}
    for name in build_moves.STILLS:
        if name not in pictures:
            text = f"poster-{name}.webp is a copy of the {standins[name]} poster"
            if name in faces:
                text += f"; {faces[name]} is cut from the {build_moves.FACES[name][0]} poster by FACES"
            lines.append((f"stills/{name}.png", text))
    smile, neutral = build_moves.HEADS["idle"], build_moves.HEADS["brand"]
    if smile not in pictures:
        lines.append(
            (f"heads/{smile}.png", "face-idle is cut from the startup poster by FACES")
        )
    if neutral not in pictures:
        lines.append(
            (
                f"heads/{neutral}.png",
                "face-brand is the idle face; the tab icons are cut from the startup poster by FAVICON_BOX",
            )
        )
    if "app-icon" not in brand:
        lines.append(
            (
                "brand/app-icon.png",
                (
                    "the app icon is drawn from brand/emblem.png"
                    if "emblem" in brand
                    else "AppIcon.icns stays as it is"
                ),
            )
        )
    if "menubar" not in brand:
        lines.append(
            (
                "brand/menubar.png",
                "MenuBarLogo.png, MenuBarLogo@2x.png and the website's round logo stay as they are",
            )
        )
    if "emblem" not in brand:
        lines.append(
            ("brand/emblem.png", f"{EMBLEM} (the README emblem) stays as it is")
        )
    return lines


def inspect(pack: Pack, stage: Path | None = None) -> Report:
    """Test every rule on a pack. With `stage` (a folder to build in) and no problems, also build the pictures there,
    laid out as in a checkout, and test the size budget."""
    report = Report(pack.label)
    used: set[str] = {"manifest.json"}
    if Image.__version__ != PINNED_PILLOW:
        report.warn(
            "encoder",
            f"Pillow {Image.__version__} is not the pinned {PINNED_PILLOW}: the pictures may differ in bytes "
            "from the committed ones",
        )
    manifest: Any = None
    frames: dict[str, list[tuple[str, Look]]] = {}
    try:
        manifest = json.loads((pack.root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as error:
        report.problem("manifest", f"manifest.json cannot be read: {error}")
    side: int | None = None
    if manifest is not None:
        found, notes = build_moves.manifest_findings(pack.root, manifest)
        report.problems += found
        report.warnings += notes
        canvas = manifest.get("canvas") if isinstance(manifest, dict) else None
        canvas = canvas if isinstance(canvas, dict) else {}
        width = canvas.get("width")
        if (
            isinstance(width, int)
            and not isinstance(width, bool)
            and width == canvas.get("height")
            and 0 < width <= build_moves.MAX_SIDE
        ):
            side = width
        frames = check_frames(pack, manifest, side, report, used)
    pictures, looks = pictures_of(pack, report, used, side)
    if side and isinstance(manifest, dict):
        check_ground(manifest, frames, looks, side, report)
    brand = brand_of(pack, report, used)
    report.instead = instead_for(pictures, brand)
    report.ignored = grouped([*(pack.files - used), *pack.skipped])
    if stage is not None and report.ok:
        build_stage(pack, pictures, brand, stage, report)
    return report


def check_frames(
    pack: Pack, manifest: Any, side: int | None, report: Report, used: set[str]
) -> dict[str, list[tuple[str, Look]]]:
    """The frame names, and each frame's pixels; the looks at the frames, per state."""
    looks: dict[str, list[tuple[str, Look]]] = {}
    states = manifest.get("states") if isinstance(manifest, dict) else None
    if not isinstance(states, dict):
        return looks
    root = pack.root.resolve()
    for state in build_moves.POSES:
        entry = states.get(state)
        names = entry.get("frames") if isinstance(entry, dict) else None
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            continue
        wrong = named_frames(state, names)
        if wrong:
            report.problem("frame names", wrong)
        used.update(names)
        for name in names:
            path = (pack.root / name).resolve()
            if path.is_relative_to(root) and path.is_file():
                look = examine(path, name, report, side)
                if look is not None:
                    looks.setdefault(state, []).append((name, look))
    return looks


def check_ground(
    manifest: dict[str, Any],
    frames: dict[str, list[tuple[str, Look]]],
    stills: dict[str, Look],
    side: int,
    report: Report,
) -> None:
    """A warning for the pictures whose lowest drawn row is far from the anchor line: the whole canvas is scaled, so
    the paws rest where the artist drew them."""
    anchor = (manifest.get("canvas") or {}).get("anchor") or {}
    y = anchor.get("y")
    if not isinstance(y, int | float) or isinstance(y, bool):
        return
    slack = max(2.0, side * GROUND_SHARE)
    groups = {state: items for state, items in frames.items()}
    groups.update(
        {
            f"still {name}": [(f"stills/{name}.png", stills[name])]
            for name in build_moves.STILLS
            if name in stills
        }
    )
    for group, items in groups.items():
        off = [
            (name, round(look.solid[3] - y))
            for name, look in items
            if look.solid is not None and abs(look.solid[3] - y) > slack
        ]
        if off:
            report.warn(
                "paws on the anchor line",
                f"{group}: {len(off)} of {len(items)} picture(s) rest more than {slack:.0f} px from the anchor "
                f"line (first: {off[0][0]}, {off[0][1]:+d} px)",
            )


def build_stage(
    pack: Pack,
    pictures: dict[str, Path],
    brand: dict[str, Path],
    stage: Path,
    report: Report,
) -> None:
    """Build everything a pack makes into `stage`, laid out as in a checkout, and fill the report's list of what it
    replaces."""
    moves = stage / MOVES
    notes: build_moves.Trace = {}
    print("Building the moves to measure them (about 20 seconds)...", file=sys.stderr)
    try:
        build_moves.build(pack.root, moves, pictures, notes, fill=True)
    except build_moves.PackError as error:
        text = str(error)
        rule, _, rest = text.partition(": ")
        report.problem(
            rule if rule == "size budget" else "build",
            rest if rule == "size budget" else text,
        )
        return
    report.replaces = [
        (f"moves/{name}", source_text(paths, note, pack.root))
        for name, (paths, note) in notes.items()
    ]
    try:
        make_terminal_mark.pixels(moves / "face-brand.png")
        report.replaces.append(
            (MARK, "moves/face-brand.png (the terminal welcome's mark, computed)")
        )
    except (ValueError, OSError) as error:
        report.warn(
            "terminal mark",
            f"cannot be made from face-brand.png ({error}); {MARK} stays as it is",
        )
    make_brand(pack, brand, stage, report)
    pack_sheets.figures(moves, stage / FIGURES)
    report.replaces += [
        (f"{FIGURES}/{name}", "the posters in moves/ (the guide's pictures, drawn)")
        for name in pack_sheets.FIGURES
    ]


def make_brand(pack: Pack, brand: dict[str, Path], stage: Path, report: Report) -> None:
    """The app icon, the menu-bar logo, the README emblem and the website's logo, from the brand pictures."""
    if "emblem" in brand:
        (stage / EMBLEM).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(brand["emblem"], stage / EMBLEM)
        report.replaces.append((EMBLEM, rel(brand["emblem"], pack.root)))
    icon_source = brand.get("app-icon") or brand.get("emblem")
    art = make_app_icons.load(icon_source) if icon_source else None
    if (
        art is not None
        and "app-icon" not in brand
        and art.shape[0] < make_app_icons.ICON_TILE
    ):
        report.warn(
            "app icon",
            f"brand/emblem.png is {art.shape[1]} x {art.shape[0]} px and draws the icon (there is no "
            "brand/app-icon.png): its largest size is scaled up",
        )
    logo = make_app_icons.load(brand["menubar"]) if "menubar" in brand else None
    if art is not None and not make_app_icons.ICONUTIL.exists():
        report.warn(
            "app icon",
            f"{make_app_icons.ICONUTIL} is part of macOS and is missing: AppIcon.icns is not made",
        )
        art = None
    written = make_app_icons.build_icons(stage / ICONS, art, logo)
    for name in written:
        source = icon_source if name == "AppIcon.icns" else brand["menubar"]
        assert source is not None
        report.replaces.append((f"{ICONS}/{name}", rel(source, pack.root)))
    if logo is not None:
        small = Image.fromarray(logo).resize(
            (LOGO_PIXELS, LOGO_PIXELS), Image.Resampling.LANCZOS
        )
        (stage / LOGO).parent.mkdir(parents=True, exist_ok=True)
        small.save(stage / LOGO, optimize=True)
        report.replaces.append(
            (LOGO, f"{rel(brand['menubar'], pack.root)} ({LOGO_PIXELS} px)")
        )


@dataclass
class Change:
    path: str
    kind: str  # "new", "changed", "removed"


def promote(stage: Path, root: Path) -> tuple[list[Change], int]:
    """Copy what was built into the checkout at `root`: the moves folder is replaced whole (a file that is not in the
    new set is removed), the other pictures are written over the old ones. The changes, and the files left the same.
    """
    changes: list[Change] = []
    same = 0
    staged_moves = stage / MOVES
    target_moves = root / MOVES
    if target_moves.is_dir():
        kept = {path.name for path in staged_moves.iterdir()}
        for path in sorted(target_moves.iterdir()):
            if path.is_file() and path.name not in kept:
                path.unlink()
                changes.append(Change(f"{MOVES}/{path.name}", "removed"))
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(stage).as_posix()
        target = root / relative
        data = path.read_bytes()
        if target.is_file() and target.read_bytes() == data:
            same += 1
            continue
        changes.append(Change(relative, "changed" if target.is_file() else "new"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return changes, same


def icon_pictures(stage: Path) -> tuple[Any, Any, Any]:
    """The brand pictures a stage holds, as pictures for the preview: the icon, the menu-bar logo at 18 and 36 px,
    and the emblem (each None when the pack gave none)."""

    def opened(relative: str) -> Any:
        path = stage / relative
        if not path.is_file():
            return None
        with Image.open(path) as image:
            return image.convert("RGBA")

    icon = None
    icns = stage / ICONS / "AppIcon.icns"
    if icns.is_file():
        with Image.open(icns) as image:
            icon = image.convert("RGBA")
    one, two = opened(f"{ICONS}/MenuBarLogo.png"), opened(f"{ICONS}/MenuBarLogo@2x.png")
    return (
        icon,
        (one, two) if one is not None and two is not None else None,
        opened(EMBLEM),
    )


def refuse(error: Exception, source: Path) -> Report:
    """A report that says why a pack was not read."""
    report = Report(str(source))
    rule = "folder safety" if source.is_dir() else "zip safety"
    report.problem(rule, str(error))
    return report


def command_check(source: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="art-pack-check-") as stage:
        try:
            with open_pack(source) as pack:
                report = inspect(pack, Path(stage))
        except pack_zip.UnsafePackError as error:
            report = refuse(error, source)
    print(report.render())
    print()
    print(
        "Result: no problems, nothing was written."
        if report.ok
        else "Result: the pack has problems; apply would refuse."
    )
    return 0 if report.ok else 1


def command_apply(source: Path, preview_dir: Path | None, root: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="art-pack-stage-") as temp:
        stage = Path(temp)
        try:
            with open_pack(source) as pack:
                report = inspect(pack, stage)
                if report.ok:
                    return finish(report, stage, preview_dir, root)
        except pack_zip.UnsafePackError as error:
            report = refuse(error, source)
    print(report.render())
    print()
    print("Result: the pack has problems. Nothing was written.")
    return 1


def finish(report: Report, stage: Path, preview_dir: Path | None, root: Path) -> int:
    """Write a checked pack's pictures into the checkout, draw the preview, print what changed and what to do next."""
    print(report.render())
    # The preview first: if it cannot be drawn, nothing has been written yet.
    folder = (
        preview_dir
        if preview_dir is not None
        else Path(tempfile.mkdtemp(prefix="melo-preview-"))
    )
    sheet = folder / "preview.png"
    pack_sheets.preview(stage / MOVES, sheet, *icon_pictures(stage))
    changes, same = promote(stage, root)
    mark = root / MARK
    if not mark.is_file():
        print(
            f"\nThe terminal mark was not written: there is no {MARK} under {root}.",
            file=sys.stderr,
        )
    else:
        before = mark.read_text(encoding="utf-8")
        try:
            make_terminal_mark.rewrite(mark, root / MOVES / "face-brand.png")
        except (ValueError, OSError) as error:
            print(f"\nThe terminal mark was not changed: {error}", file=sys.stderr)
        if mark.read_text(encoding="utf-8") != before:
            changes.append(Change(MARK, "changed"))
        else:
            same += 1
    print()
    if changes:
        print(f"Changed ({len(changes)}), {same} more are the same as before")
        print(*(f"  {change.kind:<8} {change.path}" for change in changes), sep="\n")
    else:
        print(f"Changed (0): all {same} files are the same as before")
    print()
    print(f"Preview: {sheet}")
    print()
    print("Next")
    print("  1. Look at the preview sheet (a light page and a dark page)")
    print(
        "  2. Run the picture tests: uv run --with pillow==13.0.0 pytest tests/test_moves_assets.py tests/cli/test_mark.py -q"
    )
    print("  3. Commit the changed files (git status lists them)")
    print("  4. Rebuild the app: uv run python scripts/build_macos_app.py --install")
    print(
        "  The screenshots in docs/assets/img still show the console and the app as they were: retake them."
    )
    return 0


def command_figures(out: Path | None) -> int:
    folder = out if out is not None else ROOT / FIGURES
    names = pack_sheets.figures(ROOT / MOVES, folder)
    print(f"Wrote {', '.join(names)} in {folder}")
    return 0


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog="art_pack.py",
        description="Check an art pack and replace every picture of Melo with it.",
        epilog="The pack is read only and never run. The format is in docs/guides/art-pack.md.",
    )
    commands = cli.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="read a pack and report; writes nothing")
    check.add_argument("pack", type=Path, help="a pack folder or a zip")
    apply = commands.add_parser(
        "apply", help="check, then replace every picture of Melo"
    )
    apply.add_argument("pack", type=Path, help="a pack folder or a zip")
    apply.add_argument(
        "-o",
        type=Path,
        metavar="DIR",
        help="where to leave the preview sheet (default: a new temporary folder)",
    )
    apply.add_argument(
        "-r",
        type=Path,
        metavar="ROOT",
        default=ROOT,
        help="write under ROOT, not this checkout",
    )
    figures = commands.add_parser(
        "figures", help="redraw the guide's two pictures from the committed moves"
    )
    figures.add_argument(
        "-o", type=Path, metavar="DIR", help=f"where to write (default {FIGURES})"
    )
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "check":
            return command_check(args.pack)
        if args.command == "apply":
            return command_apply(args.pack, args.o, args.r)
        return command_figures(args.o)
    except OSError as error:
        print(f"art_pack: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
