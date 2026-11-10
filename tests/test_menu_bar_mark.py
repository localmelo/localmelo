"""The menu-bar mark: computed from Melo's head picture, coloured by the app for the bar it is on."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from localmelo.tests import macos_swift

ROOT = Path(__file__).resolve().parents[1]


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_menu_bar_mark", ROOT / "scripts/make_menu_bar_mark.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass looks its module up
    spec.loader.exec_module(module)
    return module


def test_the_committed_masks_are_the_head_picture_drawn_as_lines() -> None:
    # Run scripts/make_menu_bar_mark.py when the picture changes.
    script = _script()
    assert script.files() == {
        name: (ROOT / "macos/assets" / name).read_bytes() for name in script.NAMES
    }


def test_the_fur_colour_is_the_pictures_and_keeps_three_to_one_on_either_bar() -> None:
    script = _script()
    palette = script.palette_fur()
    assert palette == script.fur_colours()
    for bar, colour in palette.items():
        for end in script.BARS[bar]:
            assert script.contrast(colour, end) >= 3, (bar, end)


# The palette needs the console window's files, as the pill's tests compile them.
FILES = (
    "GatewayPolicy.swift",
    "L10n.swift",
    "MenuWords.swift",
    "PetPolicy.swift",
    "PetMoves.swift",
    "MotionGate.swift",
    "StatusView.swift",
    "MenuBarMark.swift",
)

DRAW = r"""
import AppKit
_ = NSApplication.shared
NSApp.setActivationPolicy(.prohibited)
let mark = MenuBarMark(folder: URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true))!
for name in [NSAppearance.Name.aqua, .darkAqua] {
    let appearance = NSAppearance(named: name)!
    for badge in [false, true] {
        let image = mark.image(for: appearance, badge: badge)
        let rep = image.representations.compactMap { $0 as? NSBitmapImageRep }.first { $0.pixelsHigh == 36 }!
        // A pixel the line mask covers and one the fur mask covers, as the bytes of the sRGB bitmap hold them
        // (premultiplied; given back unpremultiplied).
        precondition(rep.colorSpace.cgColorSpace?.name == CGColorSpace.sRGB)
        func at(_ x: Int, _ y: Int) -> String {
            let bytes = rep.bitmapData! + y * rep.bytesPerRow + x * 4
            let alpha = Int(bytes[3])
            let rgb = (0..<3).map { alpha == 0 ? 0 : min(255, (Int(bytes[$0]) * 255 + alpha / 2) / alpha) }
            return String(format: "%02x%02x%02x%02x", rgb[0], rgb[1], rgb[2], alpha)
        }
        let args = CommandLine.arguments
        print(name.rawValue, badge, Int(image.size.width), rep.pixelsWide, at(Int(args[2])!, Int(args[3])!),
              at(Int(args[4])!, Int(args[5])!))
    }
}
"""


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_the_mark_takes_the_bars_colours_in_light_and_dark_and_the_dot_makes_room() -> (
    None
):
    script = _script()
    line, fur = script.mark(2)
    # A pixel each mask covers fully.
    ly, lx = (int(v[0]) for v in np.nonzero(line == 255))
    fy, fx = (int(v[0]) for v in np.nonzero(fur == 255))
    out = macos_swift.run_swift(
        FILES, DRAW, str(ROOT / "macos/assets"), str(lx), str(ly), str(fx), str(fy)
    )
    rows = [row.split() for row in out.splitlines()]
    palette = script.palette_fur()
    hexes = {bar: bytes(palette[bar]).hex() for bar in palette}
    seen = {}
    for name, badge, width, pixels, line_pixel, fur_pixel in rows:
        bar = "dark" if "Dark" in name else "light"
        # 18 points, or 23 with the waiting dot beside the head; drawn at 2x for Retina.
        assert (width, pixels) == (("23", "46") if badge == "true" else ("18", "36"))
        # The lines in the bar's label colour (black or white at 85%), the ears and crown in the fur pair.
        assert line_pixel[:6] == ("000000" if bar == "light" else "ffffff"), name
        assert 0xD0 <= int(line_pixel[6:], 16) <= 0xDA, name
        assert fur_pixel == hexes[bar] + "ff", name
        seen[name] = line_pixel
    assert len(set(seen.values())) == 2
