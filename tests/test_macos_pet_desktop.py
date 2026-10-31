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
for mood in [PetMood.unknown, .resting, .sleeping(.sleepSide), .working(2), .done, .failed, .stopped] {
    precondition(petClickTarget(mood) == .window)
}
// A press that moves 3 points or less is a click; farther, a drag.
precondition(petIsClick(moved: 0) && petIsClick(moved: 3) && !petIsClick(moved: 3.1))
// Cross-fades: none for the first picture or the same one; 0.25 s from pose to pose and 0.9 s between sleeping poses
// while the pet may move (a squash and rise hides the first, not the turn); else the plain 0.2 s cross-fade.
var env = PetEnvironment()
precondition(petFade(from: nil, to: .restLazy, environment: env) == 0)
precondition(petFade(from: .restLazy, to: .restLazy, environment: env) == 0)
precondition(petFade(from: .restLazy, to: .lieWorking, environment: env) == 0.25)
precondition(petFade(from: .lieWorking, to: .sleepSide, environment: env) == 0.25)
precondition(petFade(from: .sleepFront, to: .sleepSide, environment: env) == 0.9)
precondition(petDips(from: .restLazy, to: .lieWorking, environment: env))
precondition(petDips(from: .sleepFront, to: .restLazy, environment: env))  // waking up dips
precondition(!petDips(from: nil, to: .restLazy, environment: env) && !petDips(from: .lieDone, to: .lieDone, environment: env))
precondition(!petDips(from: .sleepFront, to: .sleepSide, environment: env))  // the turn is its slow fade alone
env.lowPower = true  // loops stop, but the change still fades, plainly: no squash
precondition(petFade(from: .sleepFront, to: .sleepSide, environment: env) == 0.2)
precondition(petFade(from: .restLazy, to: .lieDone, environment: env) == 0.2)
precondition(!petDips(from: .restLazy, to: .lieDone, environment: env))
env = PetEnvironment(); env.reduceMotion = true  // no motion at all: a plain 200 ms cross-fade, no squash
precondition(petFade(from: .sleepFront, to: .sleepSide, environment: env) == 0.2)
precondition(petFade(from: .restLazy, to: .lieDone, environment: env) == 0.2)
precondition(petFade(from: nil, to: .restLazy, environment: env) == 0)
precondition(!petDips(from: .restLazy, to: .lieDone, environment: env))
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
def test_each_mood_gets_its_breath_and_its_beat_and_nothing_moves_when_it_may_not(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        ["GatewayPolicy.swift", "L10n.swift", "PetPolicy.swift", "PetMotion.swift"],
        r"""
// The plan: stopped lies still; unknown does not breathe but blinks and twitches its ear (its picture has the patch
// and the ear pieces); sleeping breathes slow and deep and has its "z"; the resting dog breathes awake and blinks;
// the rest breathe awake; only a finished run wags the tail.
let still = PetMotionPlan(breath: nil, beat: nil)
precondition(petMotionPlan(.stopped) == still)
precondition(petMotionPlan(.unknown) == PetMotionPlan(breath: nil, beat: nil, blinks: true, twitches: true))
let asleep = PetBreath(amount: 1.03, seconds: 4.4), awake = PetBreath(amount: 1.02, seconds: 3.2)
for pose in PetState.sleepPoses {
    precondition(petMotionPlan(.sleeping(pose)) == PetMotionPlan(breath: asleep, beat: nil, blinks: false, drifts: true))
}
precondition(petMotionPlan(.resting) == PetMotionPlan(breath: awake, beat: nil, blinks: true))  // rest-lazy blinks too
precondition(petMotionPlan(.failed) == PetMotionPlan(breath: awake, beat: nil))
precondition(petMotionPlan(.working(1)) == PetMotionPlan(breath: awake, beat: .bob))
precondition(petMotionPlan(.working(1)) == petMotionPlan(.working(7)))  // a changed count is not a new beat
precondition(petMotionPlan(.waiting) == PetMotionPlan(breath: awake, beat: .hop))
precondition(petMotionPlan(.done) == PetMotionPlan(breath: awake, beat: .wag))
// Which moods get which loop: the blink and the ear follow the picture, the "z" the sleeping poses, the wag the
// finished run.
let moods: [PetMood] = [.unknown, .resting, .working(1), .waiting, .done, .failed, .stopped, .sleeping(.sleepSide)]
precondition(moods.filter { petMotionPlan($0).blinks } == [.unknown, .resting])
precondition(moods.filter { petMotionPlan($0).twitches } == [.unknown])
precondition(moods.filter { petMotionPlan($0).drifts } == [.sleeping(.sleepSide)])
precondition(moods.filter { petMotionPlan($0).beat == .wag } == [.done])

// The layers: both turn about the middle of the ground line, 3 pt above the bottom of the 132 x 97 canvas.
let motion = PetMotion(size: petPanelSize, anchor: CGPoint(x: 66, y: 3))
for layer in [motion.moving, motion.picture] {
    precondition(layer.anchorPoint == CGPoint(x: 0.5, y: 3.0 / 97.0))
    precondition(abs(layer.frame.minX) < 1e-9 && abs(layer.frame.minY) < 1e-9)
    precondition(layer.frame.size == CGSize(width: 132, height: 97))
}
precondition(motion.picture.superlayer === motion.moving)
// The blink patch, the tail and the ear lie over the picture, canvas-sized; the tail turns about its base, 101.8 pt
// from the left and 53 pt from the top, so 44 pt up from the bottom of the layer; the ear about 64.98 pt and 19.82 pt.
precondition(motion.blink.superlayer === motion.picture && motion.tail.superlayer === motion.picture)
precondition(motion.ear.superlayer === motion.picture && motion.ear.opacity == 0)
for layer in [motion.blink, motion.tail, motion.ear] { precondition(layer.bounds.size == CGSize(width: 132, height: 97)) }
precondition(abs(motion.ear.anchorPoint.x - 64.98 / 132) < 1e-9 && abs(motion.ear.anchorPoint.y - 77.18 / 97.0) < 1e-9)
precondition(abs(motion.ear.position.x - 64.98) < 1e-9 && abs(motion.ear.position.y - 77.18) < 1e-9)
precondition(abs(motion.tail.anchorPoint.x - 101.8 / 132) < 1e-9 && abs(motion.tail.anchorPoint.y - 44.0 / 97.0) < 1e-9)
precondition(motion.tail.position == CGPoint(x: 101.8, y: 44) && motion.blink.opacity == 0 && motion.tail.isHidden)
precondition(motion.drift.superlayer === motion.moving && motion.drift.opacity == 0)  // the "z" is not on a still picture

// A picture: a swap has no transition; a fade has one of its length; the layer reads the bitmap at its scale.
func bitmap(_ width: Int, _ height: Int) -> CGImage {
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    return context.makeImage()!
}
let picture = bitmap(264, 194), patch = bitmap(264, 194), body = bitmap(264, 194), tailPiece = bitmap(264, 194)
func same(_ contents: Any?, _ image: CGImage) -> Bool { (contents as AnyObject?) === (image as AnyObject) }
motion.show(picture, fade: 0)
precondition(motion.picture.animation(forKey: kCATransition) == nil && motion.picture.contentsScale == 2)
precondition(same(motion.picture.contents, picture))
motion.show(bitmap(132, 97), fade: 0.9)
let fade = motion.picture.animation(forKey: kCATransition) as? CATransition
precondition(fade?.type == .fade && fade?.duration == 0.9 && motion.picture.contentsScale == 1)

func number(_ value: Any?) -> Double? { (value as? NSNumber)?.doubleValue }

// Playing: a breath that goes on for ever, in and out, about the ground line; the beat only on entering.
let finished = PetParts(blink: nil, body: body, tail: tailPiece, zStart: nil)
motion.show(picture, parts: finished, fade: 0)
motion.play(petMotionPlan(.done), entering: true, animate: true)
let breath = motion.picture.animation(forKey: "breath") as? CABasicAnimation
precondition(breath?.keyPath == "transform.scale.y" && number(breath?.toValue) == 1.02 && number(breath?.fromValue) == 1)
precondition(breath?.duration == 1.6 && breath?.autoreverses == true && breath?.repeatCount == .infinity)  // 3.2 s a breath
// The wag: the tail layer (not the whole picture) turns 16 degrees both ways, four cycles, and settles: 1.6 s, once.
// The picture is the body without its tail while it plays, and the tail lies over it.
let wag = motion.tail.animation(forKey: "beat") as? CAKeyframeAnimation
precondition(wag?.keyPath == "transform.rotation.z" && wag?.duration == 1.6 && wag?.repeatCount == 0)
precondition(wag?.values?.count == 11 && wag?.keyTimes?.count == 11 && wag?.timingFunctions?.count == 10)
precondition(number(wag?.values?.first) == 0 && number(wag?.values?.last) == 0)  // from rest, back to rest
precondition(abs(number(wag?.values?[1])! - 16 * Double.pi / 180) < 1e-9 && abs(number(wag?.values?[2])! + 16 * Double.pi / 180) < 1e-9)
precondition(motion.moving.animation(forKey: "beat") == nil)
precondition(!motion.tail.isHidden && same(motion.picture.contents, body) && same(motion.tail.contents, tailPiece))
// Without the pieces there is nothing to wag; the still picture shows.
motion.show(picture, fade: 0)
motion.play(petMotionPlan(.done), entering: true, animate: true)
precondition(motion.tail.animation(forKey: "beat") == nil && motion.tail.isHidden && same(motion.picture.contents, picture))
motion.show(picture, parts: finished, fade: 0)
motion.play(petMotionPlan(.done), entering: false, animate: true)
precondition(motion.moving.animation(forKey: "beat") == nil && motion.picture.animation(forKey: "breath") != nil)
precondition(motion.tail.animation(forKey: "beat") == nil && motion.tail.isHidden && same(motion.picture.contents, picture))
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
precondition(deep?.duration == 2.2 && number(deep?.toValue) == 1.03)
precondition(motion.moving.animation(forKey: "beat") != nil)  // the hop in the air is not cut: it lands (a sleeping dog adds no beat)

// The blink: two in an 11 s loop, at 4.2 s and 9.6 s, each 60 ms in, 30 ms held and 70 ms out; only with a patch.
let idle = PetParts(blink: patch, body: nil, tail: nil, zStart: nil)
motion.show(picture, parts: idle, fade: 0)
precondition(same(motion.blink.contents, patch) && motion.blink.opacity == 0)
motion.play(petMotionPlan(.unknown), entering: true, animate: true)
let blinking = motion.blink.animation(forKey: "blink") as? CAKeyframeAnimation
precondition(blinking?.keyPath == "opacity" && blinking?.duration == 11 && blinking?.repeatCount == .infinity)
let at = (blinking?.keyTimes ?? []).map { number($0)! * 11 }, level = (blinking?.values ?? []).map { number($0)! }
precondition(at.count == 10 && level == [0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
for (start, begins) in [(1, 4.2), (5, 9.6)] {
    precondition(abs(at[start] - begins) < 1e-9)
    for (step, expected) in [60.0, 30, 70].enumerated() {
        precondition(abs((at[start + step + 1] - at[start + step]) * 1000 - expected) < 1e-6)
    }
}
motion.play(petMotionPlan(.working(1)), entering: true, animate: true)  // a mood whose picture has no patch does not blink
precondition(motion.blink.animation(forKey: "blink") == nil)
motion.play(petMotionPlan(.resting), entering: true, animate: true)  // the resting dog's picture has its own patch
precondition(motion.blink.animation(forKey: "blink") != nil)
motion.show(picture, fade: 0)  // a picture with no patch has nothing to fade in
motion.play(petMotionPlan(.unknown), entering: true, animate: true)
precondition(motion.blink.contents == nil && motion.blink.animation(forKey: "blink") == nil)

// The ear twitch: only with the ear pieces, every 13 s, 420 ms: the ear turns -12 degrees in 0.12 s, +4 in 0.16 s and
// back to rest in 0.14 s, about its base, and shows only then; the picture is its ear-less body for that window and
// the whole still picture the rest of the time, so the still (the model value) is never cut.
let calm = PetParts(blink: patch, body: nil, tail: nil, zStart: nil, earBody: body, ear: tailPiece)
motion.show(picture, parts: calm, fade: 0)
motion.play(petMotionPlan(.unknown), entering: true, animate: true)
let twitch = motion.ear.animation(forKey: "twitch") as? CAAnimationGroup
precondition(twitch?.duration == 13 && twitch?.repeatCount == .infinity && twitch?.animations?.count == 2)
let turn = twitch?.animations?.first as? CAKeyframeAnimation, shows = twitch?.animations?.last as? CAKeyframeAnimation
precondition(turn?.keyPath == "transform.rotation.z" && turn?.duration == 13 && turn?.values?.count == 6)
let turns = (turn?.values ?? []).map { number($0)! * 180 / Double.pi }, when = (turn?.keyTimes ?? []).map { number($0)! * 13 }
for (got, want) in zip(turns, [0.0, 0, -12, 4, 0, 0]) { precondition(abs(got - want) < 1e-9) }
precondition(abs(when[1] - 7) < 1e-9 && abs((when[2] - when[1]) * 1000 - 120) < 1e-6)
precondition(abs((when[3] - when[2]) * 1000 - 160) < 1e-6 && abs((when[4] - when[3]) * 1000 - 140) < 1e-6)
precondition(abs((when[4] - when[1]) * 1000 - 420) < 1e-6 && turn?.timingFunctions?.count == 5)
precondition(shows?.keyPath == "opacity" && shows?.calculationMode == .discrete && shows?.values?.count == 3)
precondition((shows?.values ?? []).map { number($0)! } == [0, 1, 0])
precondition((shows?.keyTimes ?? []).map { number($0)! * 13 }.enumerated().allSatisfy { abs($0.element - [0.0, 7, 7.42][$0.offset]) < 1e-9 })
let swap = motion.picture.animation(forKey: "twitch") as? CAKeyframeAnimation
precondition(swap?.keyPath == "contents" && swap?.calculationMode == .discrete && swap?.duration == 13)
precondition(swap?.repeatCount == .infinity && swap?.values?.count == 3 && swap?.keyTimes == shows?.keyTimes)
precondition(same(swap?.values?[0], picture) && same(swap?.values?[1], body) && same(swap?.values?[2], picture))
precondition(same(motion.picture.contents, picture) && motion.ear.opacity == 0 && motion.tail.isHidden)
motion.stop()
precondition(motion.ear.animation(forKey: "twitch") == nil && motion.picture.animation(forKey: "twitch") == nil)
// No pieces, no twitch; a mood whose picture has none does not twitch even if the layers still hold pieces.
motion.show(picture, parts: idle, fade: 0)
motion.play(petMotionPlan(.unknown), entering: true, animate: true)
precondition(motion.ear.animation(forKey: "twitch") == nil && motion.picture.animation(forKey: "twitch") == nil)
motion.show(picture, parts: calm, fade: 0)
motion.play(petMotionPlan(.resting), entering: true, animate: true)
precondition(motion.ear.animation(forKey: "twitch") == nil && motion.picture.animation(forKey: "twitch") == nil)

// The "z": only on a sleeping pose, 3 s, up 4 pt while fading in and out, starting where the pose says (y-up layers).
let asleepParts = PetParts(blink: nil, body: nil, tail: nil, zStart: CGPoint(x: 86, y: 24))
motion.show(picture, parts: asleepParts, fade: 0)
motion.play(petMotionPlan(.sleeping(.sleepFront)), entering: true, animate: true)
let zed = motion.drift.animation(forKey: "drift") as? CAAnimationGroup
precondition(zed?.duration == 3 && zed?.repeatCount == .infinity && zed?.animations?.count == 2)
let rise = zed?.animations?.first as? CABasicAnimation, zfade = zed?.animations?.last as? CAKeyframeAnimation
precondition(rise?.keyPath == "transform.translation.y" && number(rise?.toValue) == 4 && rise?.duration == 3)
precondition(zfade?.keyPath == "opacity" && zfade?.duration == 3 && number(zfade?.values?.first) == 0 && number(zfade?.values?.last) == 0)
precondition(motion.drift.position == CGPoint(x: 86, y: 73) && motion.drift.opacity == 0)
motion.play(petMotionPlan(.resting), entering: true, animate: true)
precondition(motion.drift.animation(forKey: "drift") == nil)

// When the pet may not move, no loop runs and the still picture stays (Reduce Motion, Low Power, heat, covered, asleep).
for (mood, parts) in [(PetMood.done, finished), (.unknown, idle), (.unknown, calm), (.sleeping(.sleepFront), asleepParts)] {
    motion.show(picture, parts: parts, fade: 0)
    motion.play(petMotionPlan(mood), entering: true, animate: true)
    precondition(motion.picture.animation(forKey: "breath") != nil || mood == .unknown)
    precondition([motion.moving, motion.blink, motion.tail, motion.ear, motion.drift].contains { $0.animationKeys() != nil })
    motion.play(petMotionPlan(mood), entering: true, animate: false)
    precondition(motion.picture.animation(forKey: "breath") == nil)
    for layer in [motion.moving, motion.blink, motion.tail, motion.ear, motion.drift] { precondition(layer.animationKeys() == nil) }
    precondition(motion.tail.isHidden && motion.blink.opacity == 0 && motion.ear.opacity == 0 && motion.drift.opacity == 0)
    precondition(motion.picture.animation(forKey: "twitch") == nil)
    precondition(same(motion.picture.contents, picture))  // the whole still picture, tail and all
    motion.play(petMotionPlan(mood), entering: true, animate: true)
    motion.stop()
    for layer in [motion.moving, motion.blink, motion.tail, motion.ear, motion.drift] { precondition(layer.animationKeys() == nil) }
    precondition(motion.picture.animation(forKey: "breath") == nil && same(motion.picture.contents, picture))
}
// A fade in progress is not a loop: stopping the loops leaves it alone.
motion.show(picture, fade: 0.15)
motion.stop()
precondition(motion.picture.animation(forKey: kCATransition) != nil)
""",
    )


@darwin
def test_a_change_waits_for_a_rest_point_and_a_change_of_pose_hides_in_a_squash(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        ["GatewayPolicy.swift", "L10n.swift", "PetPolicy.swift", "PetMotion.swift"],
        r"""
func number(_ value: Any?) -> Double? { (value as? NSNumber)?.doubleValue }
func bitmap(_ width: Int, _ height: Int) -> CGImage {
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    return context.makeImage()!
}
func near(_ a: Double, _ b: Double, _ tolerance: Double = 1e-9) -> Bool { abs(a - b) < tolerance }

// The rest wait: nothing at rest; a breath that ends within 0.4 s is waited for, a later end is not (it is eased to
// rest instead); a squash or lift in the air has to land; an eye that is shut has to open; never more than 0.4 s.
let calm = PetBreath(amount: 1.02, seconds: 3.2)
precondition(petWaitMost == 0.4)
precondition(petRestWait(landsIn: 0, breath: nil, breathAge: 0, opensIn: 0) == 0)
precondition(petRestWait(landsIn: 0, breath: calm, breathAge: 0, opensIn: 0) == 0)  // a breath just begun: at rest now
precondition(petRestWait(landsIn: 0, breath: calm, breathAge: 1.0, opensIn: 0) == 0)  // 2.2 s to its end: do not wait
precondition(near(petRestWait(landsIn: 0, breath: calm, breathAge: 3.0, opensIn: 0), 0.2))
precondition(near(petRestWait(landsIn: 0, breath: calm, breathAge: 3.2 * 5 + 3.0, opensIn: 0), 0.2))  // later breaths
precondition(petRestWait(landsIn: 0, breath: calm, breathAge: -0.2, opensIn: 0) == 0)  // the breath has not begun
precondition(near(petRestWait(landsIn: 0.3, breath: calm, breathAge: 1.0, opensIn: 0), 0.3))
precondition(near(petRestWait(landsIn: 0.1, breath: calm, breathAge: 3.0, opensIn: 0.15), 0.2))  // the longest wins
precondition(petRestWait(landsIn: 0.9, breath: calm, breathAge: 3.0, opensIn: 0) == 0.4)
precondition(petRestWait(landsIn: -1, breath: nil, breathAge: 0, opensIn: -1) == 0)
precondition(PetMotion.opensIn(blinkAge: 4.0) == 0 && PetMotion.opensIn(blinkAge: 4.36) == 0)
precondition(near(PetMotion.opensIn(blinkAge: 4.25), 0.11) && near(PetMotion.opensIn(blinkAge: 11 + 9.7), 0.06))

// A first picture or no change: no wait, no squash. Resting -> working keeps the breath (the same one) and squashes.
let motion = PetMotion(size: petPanelSize, anchor: CGPoint(x: 66, y: 3))
let picture = bitmap(264, 194)
motion.show(picture, fade: 0)
precondition(motion.moving.animation(forKey: "dip") == nil && motion.picture.animation(forKey: kCATransition) == nil)
motion.play(petMotionPlan(.resting), entering: true, animate: true)
let breathBegan = motion.breathAt
precondition(motion.picture.animation(forKey: "breath") != nil && motion.restWait() <= petWaitMost)
motion.shown = { _, path in path == "transform.scale.y" ? 1.012 : nil }  // the eye sees the dog mid-breath
let dips = PetBegin(wait: 0, dip: true)
let before = CACurrentMediaTime()
motion.show(bitmap(264, 194), fade: 0.25, begin: dips)
motion.play(petMotionPlan(.working(1)), entering: true, animate: true, begin: dips)
precondition(motion.breathAt == breathBegan && motion.picture.animation(forKey: "breath") != nil)  // not cut, not restarted
precondition(motion.picture.animation(forKey: "release.transform.scale.y") == nil)  // nothing was removed, so nothing eased
// The squash: the whole pet, 0.35 s, lower (0.95) and wider (1.025) at the low point, half way; the fade (0.25 s) is
// centred on it, so the pose changes at its lowest, and the bob springs up from it.
let dip = motion.moving.animation(forKey: "dip") as? CAAnimationGroup
let down = dip?.animations?.first as? CAKeyframeAnimation, wide = dip?.animations?.last as? CAKeyframeAnimation
precondition(dip?.duration == 0.35 && dip?.animations?.count == 2 && dip!.beginTime >= before)
precondition(down?.keyPath == "transform.scale.y" && down?.values?.map { number($0)! } == [1, 0.95, 1])
precondition(wide?.keyPath == "transform.scale.x" && wide?.values?.map { number($0)! } == [1, 1.025, 1])
precondition(down?.keyTimes?.map { number($0)! } == [0, 0.5, 1] && down?.timingFunctions?.count == 2)
let fade = motion.picture.animation(forKey: kCATransition) as? CATransition
precondition(fade?.type == .fade && fade?.duration == 0.25 && fade?.fillMode == .backwards)  // the old picture is held
precondition(near(fade!.beginTime + 0.125, dip!.beginTime + 0.175))
let bob = motion.moving.animation(forKey: "beat") as? CAKeyframeAnimation
precondition(bob?.keyPath == "transform.translation.y" && near(bob!.beginTime, dip!.beginTime + 0.175, 1e-3))  // read a moment later
// While it is in the air a second change has to wait for it to land (at most 0.4 s).
let landing = motion.restWait()
precondition(landing > 0.2 && landing <= petWaitMost)

// A wait moves the whole change later and holds the old picture until then.
motion.stop()
precondition(motion.restWait() == 0 && motion.moving.animation(forKey: "dip") == nil)
let waits = PetBegin(wait: 0.3, dip: true)
let asked = CACurrentMediaTime()
motion.show(picture, fade: 0.25, begin: waits)
motion.play(petMotionPlan(.waiting), entering: true, animate: true, begin: waits)
let later = motion.moving.animation(forKey: "dip") as? CAAnimationGroup
precondition(later!.beginTime >= asked + 0.3 && later!.beginTime < asked + 0.4)
let held = motion.picture.animation(forKey: kCATransition) as? CATransition
precondition(held!.beginTime >= asked + 0.35 && held?.fillMode == .backwards)
let hop = motion.moving.animation(forKey: "beat") as? CAKeyframeAnimation
precondition(number(hop?.values?[1]) == 6 && near(hop!.beginTime, later!.beginTime + 0.175, 1e-3))

// A breath the plan changes is not cut: the old one is taken off and the eye's value eases to rest (0.25 s) before
// the new one begins.
motion.stop()
motion.shown = { _, path in path == "transform.scale.y" ? 1.012 : nil }
motion.play(petMotionPlan(.resting), entering: false, animate: true)
let at = CACurrentMediaTime()
motion.play(petMotionPlan(.sleeping(.sleepFront)), entering: true, animate: true)
let ease = motion.picture.animation(forKey: "release.transform.scale.y") as? CABasicAnimation
precondition(ease?.fromValue as? Double == 1.012 && number(ease?.toValue) == 1 && ease?.duration == 0.25)
let slow = motion.picture.animation(forKey: "breath") as? CABasicAnimation
precondition(number(slow?.toValue) == 1.03 && slow!.beginTime >= at + 0.25 && motion.breathAt == slow!.beginTime)
precondition(motion.restWait() > 0.2)  // a change that follows waits for the ease to end
// ... at rest already (or off screen, with nothing to read) there is nothing to ease from.
motion.stop()
motion.shown = { _, _ in 1.0 }
motion.play(petMotionPlan(.resting), entering: false, animate: true)
motion.play(petMotionPlan(.sleeping(.sleepFront)), entering: false, animate: true)
precondition(motion.picture.animation(forKey: "release.transform.scale.y") == nil)
precondition((motion.picture.animation(forKey: "breath") as? CABasicAnimation)!.beginTime < CACurrentMediaTime() + 0.01)

// The sleeping dog's "z" fades out over 0.18 s from where it is when the dog wakes, and is not restarted while it stays.
let zed = PetParts(blink: nil, body: nil, tail: nil, zStart: CGPoint(x: 86, y: 24))
motion.show(picture, parts: zed, fade: 0)
motion.play(petMotionPlan(.sleeping(.sleepFront)), entering: true, animate: true)
precondition(motion.drift.animation(forKey: "drift") != nil)
motion.shown = { layer, path in layer === motion.drift ? (path == "opacity" ? 0.8 : 2) : nil }
motion.play(petMotionPlan(.sleeping(.sleepSide)), entering: false, animate: true)
precondition(motion.drift.animation(forKey: "drift") != nil && motion.drift.animation(forKey: "release.opacity") == nil)
motion.play(petMotionPlan(.resting), entering: true, animate: true)
let gone = motion.drift.animation(forKey: "release.opacity") as? CABasicAnimation
precondition(motion.drift.animation(forKey: "drift") == nil && gone?.fromValue as? Double == 0.8 && number(gone?.toValue) == 0)
precondition(gone?.duration == 0.18 && number((motion.drift.animation(forKey: "release.transform.translation.y") as? CABasicAnimation)?.fromValue) == 2)

// Reduce Motion (and the other stops): the picture changes by the plain cross-fade, no squash, and nothing plays.
var env = PetEnvironment(); env.reduceMotion = true
let plain = PetBegin(wait: 0, dip: petDips(from: .sleepFront, to: .restLazy, environment: env))
precondition(plain == PetBegin())
motion.play(petMotionPlan(.working(1)), entering: true, animate: true)
motion.show(picture, fade: petFade(from: .sleepFront, to: .restLazy, environment: env), begin: plain)
motion.play(petMotionPlan(.resting), entering: true, animate: shouldAnimate(env), begin: plain)
let still = motion.picture.animation(forKey: kCATransition) as? CATransition
precondition(still?.duration == 0.2 && still?.beginTime == 0 && still?.fillMode == .removed)
for layer in [motion.moving, motion.picture, motion.blink, motion.tail, motion.ear, motion.drift] {
    precondition((layer.animationKeys() ?? []).filter { $0 != kCATransition }.isEmpty)
}
""",
    )


@darwin
def test_a_change_of_mood_does_not_jump_the_picture_where_the_eye_sees_it(
    tmp_path: Path,
) -> None:
    # The layers are rendered by a CARenderer (no window, no screen), which gives them presentation layers: what the
    # eye sees now. A sleeping breath is read mid-way; the change to a resting one must start from that value (the old
    # code snapped it back to 1.0, 0.009 away).
    run_swift(
        tmp_path,
        ["GatewayPolicy.swift", "L10n.swift", "PetPolicy.swift", "PetMotion.swift"],
        r"""
import Metal
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    print("no Metal device: live check skipped")
    exit(0)
}
let description = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 264, height: 194, mipmapped: false)
description.usage = [.renderTarget, .shaderRead]
description.storageMode = .shared
let renderer = CARenderer(mtlTexture: device.makeTexture(descriptor: description)!, options: [kCARendererMetalCommandQueue: queue])
let root = CALayer()
root.bounds = CGRect(origin: .zero, size: petPanelSize)
root.anchorPoint = .zero
let motion = PetMotion(size: petPanelSize, anchor: CGPoint(x: 66, y: 3))
root.addSublayer(motion.moving)
renderer.layer = root
renderer.bounds = root.bounds
let context = CGContext(data: nil, width: 264, height: 194, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
let picture = context.makeImage()!
func scale() -> Double {
    CATransaction.flush()  // the presentation layer follows the clock at each commit
    return (motion.picture.presentation()?.value(forKeyPath: "transform.scale.y") as? NSNumber)?.doubleValue ?? -1
}

motion.show(picture, fade: 0)
motion.play(petMotionPlan(.sleeping(.sleepFront)), entering: true, animate: true)
Thread.sleep(forTimeInterval: 0.9)
let seen = scale()
precondition(seen > 1.005 && seen < 1.03)  // mid-breath, read from the presentation layer
let begin = PetBegin(wait: 0, dip: true)
motion.show(picture, fade: 0.25, begin: begin)
motion.play(petMotionPlan(.resting), entering: true, animate: true, begin: begin)
let after = scale()
precondition(abs(after - seen) < 0.003)  // it goes on from where it was
Thread.sleep(forTimeInterval: 0.2)
let easing = scale()
precondition(easing < after && easing > 1)  // ... and eases down to rest
Thread.sleep(forTimeInterval: 0.2)
precondition(abs(scale() - 1) < 0.002)  // rest, before the new breath begins
""",
    )


@darwin
def test_the_calm_portrait_blinks_only_while_loops_may_run(tmp_path: Path) -> None:
    # No window is shown: the portrait is built off screen, so it is not visible and the gate keeps it still; the
    # test then gives it each environment by hand, as the gate would.
    pictures = ROOT / "macos/art/akita"
    run_swift(
        tmp_path,
        [
            "GatewayPolicy.swift",
            "L10n.swift",
            "PetPolicy.swift",
            "PetMotion.swift",
            "MotionGate.swift",
            "StatusView.swift",
        ],
        rf"""
import AppKit
_ = NSApplication.shared
let portrait = PortraitView(pictures: URL(fileURLWithPath: "{pictures}"))
precondition(portrait.state == .sitWorking && !portrait.isBlinking)  // the working puppy has no patch
portrait.show(.sitIdle)
precondition(!portrait.isBlinking)  // built off screen: not visible, so it stays still
portrait.loops(PetEnvironment())
precondition(portrait.isBlinking)  // the calm puppy blinks when loops may run
for stop in [{{ (e: inout PetEnvironment) in e.reduceMotion = true }}, {{ e in e.lowPower = true }},
             {{ e in e.thermal = .serious }}, {{ e in e.thermal = .critical }}, {{ e in e.visible = false }},
             {{ e in e.screenAwake = false }}] {{
    var environment = PetEnvironment()
    stop(&environment)
    portrait.loops(environment)
    precondition(!portrait.isBlinking)  // each stop leaves the still picture
}}
portrait.loops(PetEnvironment())
portrait.show(.sitError)
precondition(!portrait.isBlinking)  // a picture with no patch does not blink
portrait.loops(PetEnvironment())
precondition(!portrait.isBlinking)
portrait.show(.sitIdle)
portrait.loops(PetEnvironment())
precondition(portrait.isBlinking)
""",
    )
