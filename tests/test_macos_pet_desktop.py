"""The desktop pet's pure rules (PetPolicy.swift) and its motion (PetMotion.swift): when it shows, where its panel
goes and what is remembered of it, and the animations each mood gets. The Swift is compiled with a test main, as
tests/test_macos_pet_policy.py does; no window is opened here (tests/test_macos_webview_integration.py has those).
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


def test_the_panel_is_the_canvas_the_lying_pictures_share() -> None:
    pictures = json.loads((ROOT / "macos/art/akita/anchors.json").read_text())[
        "pictures"
    ]
    lying = {
        name: entry
        for name, entry in pictures.items()
        if name.startswith(("lie-", "rest-", "sleep-"))
    }
    assert len(lying) == 9
    assert {tuple(entry["canvas"]) for entry in lying.values()} == {(132, 97)}
    assert {entry["ground"] for entry in lying.values()} == {94}
    source = (APP / "PetPolicy.swift").read_text()
    assert "let petPanelSize = CGSize(width: 132, height: 97)" in source
    assert "let petBelowPaws: CGFloat = 3" in source  # 97 - 94


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
        ["GatewayPolicy.swift", "PetPolicy.swift"],
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
for mood in [PetMood.unknown, .resting, .sleeping(.sleepSide), .working(2), .done, .failed, .stopped] {
    precondition(petClickTarget(mood) == .window)
}
// A press that moves 3 points or less is a click; farther, a drag.
precondition(petIsClick(moved: 0) && petIsClick(moved: 3) && !petIsClick(moved: 3.1))
// Cross-fades: none for the first picture, a swap under Reduce Motion, 0.9 s from pose to pose, else 0.15 s.
var env = PetEnvironment()
precondition(petFade(from: nil, to: .restLazy, environment: env) == 0)
precondition(petFade(from: .restLazy, to: .restLazy, environment: env) == 0)
precondition(petFade(from: .restLazy, to: .lieWorking, environment: env) == 0.15)
precondition(petFade(from: .lieWorking, to: .sleepSide, environment: env) == 0.15)
precondition(petFade(from: .sleepFront, to: .sleepSide, environment: env) == 0.9)
env.lowPower = true  // loops stop, but the change still fades, shortly
precondition(petFade(from: .sleepFront, to: .sleepSide, environment: env) == 0.15)
precondition(petFade(from: .restLazy, to: .lieDone, environment: env) == 0.15)
env = PetEnvironment(); env.reduceMotion = true
precondition(petFade(from: .sleepFront, to: .sleepSide, environment: env) == 0)
precondition(petFade(from: .restLazy, to: .lieDone, environment: env) == 0)
""",
    )


@darwin
def test_what_is_remembered_survives_a_damaged_file_and_a_changed_screen(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        ["GatewayPolicy.swift", "PetPolicy.swift"],
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
def test_each_mood_gets_its_breath_and_its_beat_and_nothing_moves_when_it_may_not(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        ["GatewayPolicy.swift", "PetPolicy.swift", "PetMotion.swift"],
        r"""
// The plan: unknown and stopped lie still; sleeping breathes slow and deep; the rest breathe awake.
let still = PetMotionPlan(breath: nil, beat: nil)
precondition(petMotionPlan(.unknown) == still && petMotionPlan(.stopped) == still)
let asleep = PetBreath(amount: 1.03, seconds: 4.4), awake = PetBreath(amount: 1.02, seconds: 3.2)
for pose in PetState.sleepPoses { precondition(petMotionPlan(.sleeping(pose)) == PetMotionPlan(breath: asleep, beat: nil)) }
precondition(petMotionPlan(.resting) == PetMotionPlan(breath: awake, beat: nil))
precondition(petMotionPlan(.failed) == PetMotionPlan(breath: awake, beat: nil))
precondition(petMotionPlan(.working(1)) == PetMotionPlan(breath: awake, beat: .bob))
precondition(petMotionPlan(.working(1)) == petMotionPlan(.working(7)))  // a changed count is not a new beat
precondition(petMotionPlan(.waiting) == PetMotionPlan(breath: awake, beat: .hop))
precondition(petMotionPlan(.done) == PetMotionPlan(breath: awake, beat: .wiggle))

// The layers: both turn about the middle of the ground line, 3 pt above the bottom of the 132 x 97 canvas.
let motion = PetMotion(size: petPanelSize, anchor: CGPoint(x: 66, y: 3))
for layer in [motion.moving, motion.picture] {
    precondition(layer.anchorPoint == CGPoint(x: 0.5, y: 3.0 / 97.0))
    precondition(abs(layer.frame.minX) < 1e-9 && abs(layer.frame.minY) < 1e-9)
    precondition(layer.frame.size == CGSize(width: 132, height: 97))
}
precondition(motion.picture.superlayer === motion.moving)

// A picture: a swap has no transition; a fade has one of its length; the layer reads the bitmap at its scale.
func bitmap(_ width: Int, _ height: Int) -> CGImage {
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    return context.makeImage()!
}
motion.show(bitmap(264, 194), fade: 0)
precondition(motion.picture.animation(forKey: kCATransition) == nil && motion.picture.contentsScale == 2)
precondition(motion.picture.contents != nil)
motion.show(bitmap(132, 97), fade: 0.9)
let fade = motion.picture.animation(forKey: kCATransition) as? CATransition
precondition(fade?.type == .fade && fade?.duration == 0.9 && motion.picture.contentsScale == 1)

func number(_ value: Any?) -> Double? { (value as? NSNumber)?.doubleValue }

// Playing: a breath that goes on for ever, in and out, about the ground line; the beat only on entering.
motion.play(petMotionPlan(.done), entering: true, animate: true)
let breath = motion.picture.animation(forKey: "breath") as? CABasicAnimation
precondition(breath?.keyPath == "transform.scale.y" && number(breath?.toValue) == 1.02 && number(breath?.fromValue) == 1)
precondition(breath?.duration == 1.6 && breath?.autoreverses == true && breath?.repeatCount == .infinity)  // 3.2 s a breath
let wag = motion.moving.animation(forKey: "beat") as? CAKeyframeAnimation
precondition(wag?.keyPath == "transform.rotation.z" && wag?.duration == 1.6 && wag?.repeatCount == 0)
precondition(wag?.values?.count == 11 && wag?.keyTimes?.count == 11 && wag?.timingFunctions?.count == 10)
precondition(number(wag?.values?.first) == 0 && number(wag?.values?.last) == 0)  // from rest, back to rest
motion.play(petMotionPlan(.done), entering: false, animate: true)
precondition(motion.moving.animation(forKey: "beat") == nil && motion.picture.animation(forKey: "breath") != nil)
motion.play(petMotionPlan(.working(1)), entering: true, animate: true)
let bob = motion.moving.animation(forKey: "beat") as? CAKeyframeAnimation
precondition(bob?.keyPath == "transform.translation.y" && bob?.duration == 0.35 && bob?.values?.count == 3)
precondition(number(bob?.values?[1]) == 3)
motion.play(petMotionPlan(.waiting), entering: true, animate: true)
precondition(number((motion.moving.animation(forKey: "beat") as? CAKeyframeAnimation)?.values?[1]) == 6)
let slow = motion.picture.animation(forKey: "breath") as? CABasicAnimation
precondition(slow?.duration == 1.6)
motion.play(petMotionPlan(.sleeping(.sleepBack)), entering: true, animate: true)
let deep = motion.picture.animation(forKey: "breath") as? CABasicAnimation
precondition(deep?.duration == 2.2 && number(deep?.toValue) == 1.03 && motion.moving.animation(forKey: "beat") == nil)

// When the pet may not move, nothing runs and the still picture stays (Reduce Motion, Low Power, covered).
motion.play(petMotionPlan(.done), entering: true, animate: false)
precondition(motion.picture.animation(forKey: "breath") == nil && motion.moving.animationKeys() == nil)
precondition(motion.picture.contents != nil)
motion.play(petMotionPlan(.done), entering: true, animate: true)
motion.stop()
precondition(motion.picture.animation(forKey: "breath") == nil && motion.moving.animationKeys() == nil)
// A fade in progress is not a loop: stopping the loops leaves it alone.
motion.show(bitmap(264, 194), fade: 0.15)
motion.stop()
precondition(motion.picture.animation(forKey: kCATransition) != nil)
""",
    )
