"""The desktop pet's pure rules (PetPolicy.swift): when it shows, where its panel goes and what is remembered of it.
Its moves are tests/test_macos_pet_moves.py's. The Swift is compiled with a test main, as tests/test_macos_pet_policy.py
does; no window is opened here (tests/test_macos_webview_integration.py has those).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "macos/LocalMeloApp"
darwin = pytest.mark.skipif(
    sys.platform != "darwin", reason="Swift AppKit host is macOS-only"
)


def run_swift(tmp_path: Path, files: list[str], source: str) -> None:
    """Compile `files` of the app with a test main (a raw string) and run it."""
    main = tmp_path / "main.swift"
    main.write_text("import Foundation\nimport QuartzCore\n" + source)
    binary = tmp_path / "test-pet-desktop"
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            *[str(APP / name) for name in files],
            str(main),
            "-o",
            str(binary),
        ],
        env={
            **os.environ,
            "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
        },
        check=True,
        timeout=120,
    )
    subprocess.run([str(binary)], check=True, timeout=10)


def test_the_panel_holds_the_moves_canvas_with_the_paws_on_its_ground_line() -> None:
    manifest = json.loads(
        (ROOT / "support/gateway/static/moves/manifest.json").read_text()
    )
    pixels = manifest["canvas"]["size"]
    anchor_x, anchor_y = manifest["canvas"]["anchor"]
    source = (APP / "PetPolicy.swift").read_text()
    assert "let petPanelSize = CGSize(width: 132, height: 97)" in source
    assert "let petBelowPaws: CGFloat = 3" in source
    assert "let petMoveSide: CGFloat = 100" in source
    # The canvas, 100 pt square with its paws 3 pt above the panel's bottom, fits the panel's width; what it leaves
    # outside the panel's height (above and below) is clear in the art: 22 of every 512 px at each edge.
    unit = 100 / pixels
    left = 132 / 2 - anchor_x * unit
    bottom = 3 - (pixels - anchor_y) * unit
    clear = 100 * 22 / 512
    assert left >= 0 and left + 100 <= 132
    assert -bottom < clear and bottom + 100 - 97 < clear


def test_the_pet_is_silent_when_a_wait_starts_because_the_menu_bar_item_announces_it() -> (
    None
):
    # Two announcements of one wait would be read twice by VoiceOver: only the menu-bar item posts one, and the pet
    # is read through its label when it is focused.
    assert "announcementRequested" in (APP / "MenuBar.swift").read_text()
    pet = (APP / "PetPanel.swift").read_text()
    assert "announcementRequested" not in pet and "announcement:" not in pet
    assert "setAccessibilityLabel(mood.label)" in pet


@darwin
def test_the_pet_shows_only_while_the_window_is_put_away_and_a_click_goes_where_it_is_needed(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        ["GatewayPolicy.swift", "L10n.swift", "PetPolicy.swift"],
        r"""
// The window is "in front" when it is ordered in, not minimized and the app is not hidden.
precondition(petWindowShown(visible: true, minimized: false, appHidden: false))
precondition(!petWindowShown(visible: false, minimized: false, appHidden: false))  // closed
precondition(!petWindowShown(visible: true, minimized: true, appHidden: false))  // minimized
precondition(!petWindowShown(visible: true, minimized: false, appHidden: true))  // Command-H
// The pet shows when it is not, unless the person hid Melo.
precondition(petShouldShow(windowShown: false, hiddenByPerson: false))
precondition(!petShouldShow(windowShown: true, hiddenByPerson: false))
precondition(!petShouldShow(windowShown: false, hiddenByPerson: true))
precondition(!petShouldShow(windowShown: true, hiddenByPerson: true))
// A click goes to what waits, in every other mood to the window.
precondition(petClickTarget(.waiting) == .waiting)
for mood in [PetMood.unknown, .resting, .sleeping, .working(2), .done, .failed, .stopped] {
    precondition(petClickTarget(mood) == .window)
}
// A press that moves 3 points or less is a click; farther, a drag.
precondition(petIsClick(moved: 0) && petIsClick(moved: 3) && !petIsClick(moved: 3.1))
""",
    )


@darwin
def test_what_is_remembered_survives_a_damaged_file_and_a_changed_screen(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        ["GatewayPolicy.swift", "L10n.swift", "PetPolicy.swift"],
        r"""
func settings(_ text: String?) -> PetSettings { petSettings(text.map { Data($0.utf8) }) }

// Nothing, or a file that is not ours: the defaults. Partly damaged: what is still good.
precondition(settings(nil) == PetSettings())
precondition(settings("") == PetSettings() && settings("not json") == PetSettings() && settings("[1,2]") == PetSettings())
precondition(settings("{}") == PetSettings())
precondition(settings(#"{"hidden":true}"#).hidden)
precondition(!settings(#"{"hidden":"yes"}"#).hidden)
let partly = settings(#"{"hidden":true,"screen":7,"spots":{"a":{"dx":-10,"dy":5},"b":"x","c":{"dx":1}}}"#)
precondition(partly.hidden && partly.screen == nil && partly.spots.isEmpty)  // spots damaged as a whole: none kept
let good = settings(#"{"hidden":false,"screen":"A","spots":{"A":{"dx":-10,"dy":5}}}"#)
precondition(good.screen == "A" && good.spots["A"] == PetSpot(dx: -10, dy: 5))
precondition(settings(#"{"spots":{"A":{"dx":1e999,"dy":0}}}"#).spots.isEmpty)  // not a finite number

// Written and read back, the same; the file is stable text (sorted keys).
var saved = PetSettings()
saved.hidden = true
saved.remember(screen: "B", spot: PetSpot(dx: -300, dy: 120))
saved.remember(screen: "A", spot: PetSpot(dx: -16, dy: -3))
precondition(petSettings(petSettingsData(saved)) == saved)
precondition(saved.screen == "A" && saved.spots.count == 2)
precondition(petSettingsData(saved) == petSettingsData(petSettings(petSettingsData(saved))))
// A person with many screens over the years keeps the newest spot, and at most 16.
var many = PetSettings()
for index in 0..<40 { many.remember(screen: "S\(index)", spot: PetSpot(dx: 0, dy: CGFloat(index))) }
precondition(many.spots.count == 16 && many.spots["S39"] == PetSpot(dx: 0, dy: 39) && many.screen == "S39")

// Placement. The main screen has the Dock at the bottom (usable area from y 70); a second sits to its left.
let main = PetScreen(key: "main", visible: CGRect(x: 0, y: 70, width: 1512, height: 900))
let side = PetScreen(key: "side", visible: CGRect(x: -1920, y: 0, width: 1920, height: 1055))
precondition(petPlacement(PetSettings(), screens: []) == nil)
// First time: the bottom right of the first screen, 16 pt in, the paws on the usable area's bottom edge.
let first = petPlacement(PetSettings(), screens: [main, side])!
precondition(first.screen == main && first.frame == CGRect(x: 1512 - 16 - 132, y: 67, width: 132, height: 97))
// A saved spot on a screen is used when that is the screen it was last on.
var dragged = PetSettings()
dragged.remember(screen: "side", spot: petSpot(of: CGRect(x: -900, y: 400, width: 132, height: 97), on: side.visible))
let moved = petPlacement(dragged, screens: [main, side])!
precondition(moved.screen == side && moved.frame == CGRect(x: -900, y: 400, width: 132, height: 97))
// That screen is unplugged: the first screen, at its own spot (none: the default), never off screen.
let back = petPlacement(dragged, screens: [main])!
precondition(back.screen == main && back.frame == first.frame)
dragged.remember(screen: "main", spot: PetSpot(dx: -500, dy: 200))
precondition(petPlacement(dragged, screens: [main, side])!.screen == main)
dragged.screen = "side"
precondition(petPlacement(dragged, screens: [main, side])!.screen == side)
// A smaller resolution or a taller Dock: the spot is the offset from the bottom right corner, pulled onto the screen.
let small = PetScreen(key: "main", visible: CGRect(x: 0, y: 0, width: 800, height: 500))
var far = PetSettings()
far.remember(screen: "main", spot: PetSpot(dx: -100, dy: 900))
precondition(petPlacement(far, screens: [small])!.frame == CGRect(x: 800 - 100 - 132, y: 500 - 97, width: 132, height: 97))
var corner = PetSettings()
corner.remember(screen: "main", spot: PetSpot(dx: -16, dy: -3))
let taller = PetScreen(key: "main", visible: CGRect(x: 0, y: 120, width: 1512, height: 850))
precondition(petPlacement(corner, screens: [taller])!.frame == CGRect(x: 1364, y: 117, width: 132, height: 97))
""",
    )


@darwin
def test_the_console_settings_may_read_and_choose_what_is_on_the_desktop(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        ["GatewayPolicy.swift", "L10n.swift", "PetPolicy.swift"],
        r"""
precondition(desktopRequest(["action": "get"]) == .get)
for mode in DesktopMode.allCases {
    precondition(desktopRequest(["action": "set", "mode": mode.rawValue]) == .set(mode))
}
let refused: [Any] = [["action": "set"], ["action": "set", "mode": "dog"], ["action": "set", "mode": 1],
                      ["action": "bogus"], ["mode": "pet"], "get", 7, [String: Any]()]
precondition(refused.allSatisfy { desktopRequest($0) == nil })
""",
    )


def test_the_desktop_bridge_is_registered_with_the_folder_bridge_and_removed_with_it() -> (
    None
):
    main = (APP / "main.swift").read_text()
    bridge = (APP / "FolderBridge.swift").read_text()
    assert 'static let name = "localmeloDesktop"' in bridge
    assert "acceptsBridgeMessage(" in bridge.split("final class DesktopBridge")[1]
    assert main.count("name: DesktopBridge.name)") == 1
    assert main.count("removeScriptMessageHandler(forName: DesktopBridge.name") == 1
    # The answer is the choice in force: the same one the menus show.
    assert "self?.pet?.setDesktop(mode)" in main
