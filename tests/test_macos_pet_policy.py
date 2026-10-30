"""PetPolicy.swift: Melo's states, the mood the health reply and the quiet clock give, its words, when it may move and
where its panel goes. The Swift is compiled with a test main, as tests/test_macos_gateway_policy.py does.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "macos/LocalMeloApp"
darwin = pytest.mark.skipif(
    sys.platform != "darwin", reason="Swift AppKit host is macOS-only"
)


def run_swift(tmp_path: Path, source: str) -> None:
    """Compile the two policy files with a test main (a raw string: it holds `\\#(` interpolations) and run it."""
    main = tmp_path / "main.swift"
    main.write_text("import Foundation\n" + source)
    binary = tmp_path / "test-pet-policy"
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            str(APP / "GatewayPolicy.swift"),
            str(APP / "PetPolicy.swift"),
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
    subprocess.run([str(binary)], check=True, timeout=5)


def test_the_states_are_the_pictures_the_build_ships() -> None:
    source = (APP / "PetPolicy.swift").read_text()
    enum = source[source.index("enum PetState") : source.index("static let sleepPoses")]
    states = re.findall(r'= "([a-z-]+)"', enum)
    pictures = json.loads((ROOT / "macos/art/akita/anchors.json").read_text())
    assert sorted(states) == sorted(pictures["pictures"]) and len(states) == 14
    for state in states:  # the folder the Swift names is the folder the file is in
        folder = "portrait" if state.startswith("sit-") else "pet"
        assert (ROOT / f"macos/art/akita/{folder}/{state}@2x.png").is_file()


def test_the_first_window_shows_the_portrait_at_the_canvas_of_the_sitting_pictures() -> (
    None
):
    pictures = json.loads((ROOT / "macos/art/akita/anchors.json").read_text())
    canvases = {
        tuple(pictures["pictures"][name]["canvas"])
        for name in ("sit-idle", "sit-working", "sit-waiting", "sit-done", "sit-error")
    }
    assert canvases == {(108, 140)}
    assert (
        "let portraitSize = CGSize(width: 108, height: 140)"
        in (APP / "PetPolicy.swift").read_text()
    )
    # The picture is decoration beside the title that says the same in words, and no longer a framed print.
    status = (APP / "StatusView.swift").read_text()
    assert "setAccessibilityElement(false)" in status
    assert "melo.png" not in status and "NSBezierPath" not in status


@darwin
def test_the_mood_follows_the_health_reply_and_the_quiet_clock(tmp_path: Path) -> None:
    run_swift(
        tmp_path,
        r"""
func health(running: Int = 0, waiting: String = "[]", recent: String = "null") -> Data {
    Data(#"{"api":2,"status":{"running":{"chats":\#(running),"tasks":0,"runs":0,"terminal":0},"waiting":\#(waiting),"recent":\#(recent)}}"#.utf8)
}
func ended(_ word: String, _ ago: String, kind: String = "chat") -> String {
    #"{"kind":"\#(kind)","id":"s1","outcome":"\#(word)","ended_s_ago":\#(ago)}"#
}
func mood(_ data: Data?, stopped: Bool = false, touch: Double = 0, seed: Int = 0) -> PetMood {
    petMood(health: data, stopped: stopped, sinceTouch: touch, poseSeed: seed)
}
let waits = #"[{"kind":"chat","id":"s1"}]"#

// Before the first answer, or with a reply that is not a health body, nothing is known: lie-idle.
precondition(mood(nil) == .unknown && mood(Data("not json".utf8)) == .unknown)
precondition(mood(Data(#"{"api":2}"#.utf8)) == .unknown)
// Nothing runs: the lazy rest, the default.
precondition(mood(health()) == .resting)
// The app's own failure beats everything; a wait beats work; work beats a run that just finished.
precondition(mood(health(running: 2, waiting: waits), stopped: true) == .stopped)
precondition(mood(health(running: 2, waiting: waits)) == .waiting)
precondition(mood(health(waiting: waits, recent: ended("done", "1.0"))) == .waiting)
precondition(mood(health(running: 3)) == .working(3))
precondition(mood(health(running: 1, recent: ended("done", "0.5"))) == .working(1))
// A finished run shows for a short time: done for 6 s, an error for 10 s, then rest.
precondition(mood(health(recent: ended("done", "0.0"))) == .done)
precondition(mood(health(recent: ended("done", "5.9"))) == .done)
precondition(mood(health(recent: ended("done", "6.0"))) == .resting)
precondition(mood(health(recent: ended("failed", "9.9", kind: "task"))) == .failed)
precondition(mood(health(recent: ended("failed", "10.0"))) == .resting)
// A run the person stopped, or an outcome this app does not know, changes nothing.
precondition(mood(health(recent: ended("stopped", "1.0"))) == .resting)
precondition(mood(health(recent: ended("exploded", "1.0"))) == .resting)
precondition(mood(health(recent: ended("done", "-1"))) == .resting)
precondition(recentOutcome(health(recent: ended("done", "2.5", kind: "bogus"))) == nil)
precondition(recentOutcome(health(recent: #"{"kind":"run","id":"","outcome":"done","ended_s_ago":1}"#)) == nil)
precondition(recentOutcome(health(recent: #"{"kind":"run","id":"r1","outcome":"done"}"#)) == nil)
precondition(recentOutcome(health()) == nil && recentOutcome(Data(#"{"api":2}"#.utf8)) == nil)
let found = recentOutcome(health(recent: ended("failed", "7", kind: "terminal")))
precondition(found == RecentOutcome(kind: "terminal", id: "s1", outcome: .failed, endedAgo: 7))

// Quiet: a minute after the last touch, with no run ever, the pet sleeps in the pose its seed gives.
precondition(mood(health(), touch: 59.9) == .resting)
precondition(mood(health(), touch: 60) == .sleeping(.sleepFront))
precondition(mood(health(), touch: 60, seed: 1) == .sleeping(.sleepSide))
precondition(mood(health(), touch: 60, seed: 2) == .sleeping(.sleepBack))
precondition(mood(health(), touch: 60, seed: -1) == .sleeping(.sleepBack))
// The quiet is the shorter of the touch and the end of the newest run: a run that ended 30 s ago keeps the pet
// up however long ago it was touched, and a touch 10 s ago wakes a pet that finished long ago.
precondition(mood(health(recent: ended("done", "30")), touch: 5000) == .resting)
precondition(mood(health(recent: ended("done", "61")), touch: 5000) == .sleeping(.sleepFront))
precondition(mood(health(recent: ended("done", "7000")), touch: 10) == .resting)
precondition(mood(health(recent: ended("failed", "61")), touch: 5000) == .sleeping(.sleepFront))
// The sleeping pose turns every 3 minutes through side-lying, belly-down and on the back, and starts over.
precondition(mood(health(), touch: 239.9) == .sleeping(.sleepFront))
precondition(mood(health(), touch: 240) == .sleeping(.sleepSide))
precondition(mood(health(), touch: 419.9) == .sleeping(.sleepSide))
precondition(mood(health(), touch: 420) == .sleeping(.sleepBack))
precondition(mood(health(), touch: 600) == .sleeping(.sleepFront))
precondition(mood(health(), touch: 600, seed: 1) == .sleeping(.sleepSide))
// A touch ends the sleep at once, and the pose it left is not the one it falls asleep in next: the clock restarts.
precondition(mood(health(), touch: 0.0, seed: 2) == .resting)
// Odd clocks never trap: a huge or infinite quiet sleeps, a negative or NaN one rests.
precondition(mood(health(), touch: 1e300).picture.isAsleep && mood(health(), touch: .infinity).picture.isAsleep)
precondition(mood(health(), touch: -5) == .resting && mood(health(), touch: .nan) == .resting)

// Each mood has one picture; the two that share lie-error say different things.
precondition(PetMood.unknown.picture == .lieIdle && PetMood.resting.picture == .restLazy)
precondition(PetMood.working(2).picture == .lieWorking && PetMood.waiting.picture == .lieWaiting)
precondition(PetMood.done.picture == .lieDone && PetMood.failed.picture == .lieError)
precondition(PetMood.stopped.picture == .lieError)
precondition(PetMood.sleeping(.sleepSide).picture == .sleepSide)
precondition(PetMood.unknown.label == "Melo")
precondition(PetMood.resting.label == "Melo, resting. Nothing running.")
precondition(PetMood.sleeping(.sleepBack).label == "Melo, sleeping. Nothing running.")
precondition(PetMood.working(1).label == "Melo, working. 1 running.")
precondition(PetMood.working(2).label == "Melo, working. 2 running.")
precondition(PetMood.waiting.label == "Melo, waiting for your approval.")
precondition(PetMood.done.label == "Melo, done.")
precondition(PetMood.failed.label == "Melo, the last run did not finish.")
precondition(PetMood.stopped.label == "Melo, localmelo stopped.")
precondition(PetMood.waiting.help == "Opens what waits for your approval")
precondition(PetMood.resting.help == "Opens localmelo" && PetMood.stopped.help == "Opens localmelo")

// The 14 pictures: names, folders and files for each screen scale.
precondition(PetState.allCases.count == 14 && Set(PetState.allCases.map { $0.rawValue }).count == 14)
precondition(PetState.sitDone.folder == "portrait" && PetState.restLazy.folder == "pet")
precondition(PetState.lieIdle.fileName(scale: 1) == "lie-idle.png")
precondition(PetState.sleepBack.fileName(scale: 2) == "sleep-back@2x.png")
precondition(PetState.sleepBack.fileName(scale: 3) == "sleep-back@2x.png")
precondition(PetState.allCases.filter { $0.isAsleep } == [.sleepFront, .sleepSide, .sleepBack])

// The first window's portrait: working while the server starts and signs in, calm once it is ready, the
// drooping ear when it cannot go on (a failure before or after the console had loaded alike).
precondition(startPortrait(failed: false, ready: false) == .sitWorking)
precondition(startPortrait(failed: false, ready: true) == .sitIdle)
precondition(startPortrait(failed: true, ready: false) == .sitError)
precondition(startPortrait(failed: true, ready: true) == .sitError)
precondition(startPortrait(failed: false, ready: true).fileName(scale: 2) == "sit-idle@2x.png")
""",
    )


@darwin
def test_the_pet_moves_only_when_it_may_and_its_panel_stays_on_screen(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        r"""
var env = PetEnvironment()
precondition(shouldAnimate(env) && petCrossfade == 0.2)
env.reduceMotion = true
precondition(!shouldAnimate(env))  // no motion; a change is the plain 0.2 s cross-fade
env = PetEnvironment(); env.lowPower = true
precondition(!shouldAnimate(env))  // loops stop, a change still fades
env = PetEnvironment(); env.thermal = .fair
precondition(shouldAnimate(env))
env.thermal = .serious
precondition(!shouldAnimate(env))
env.thermal = .critical
precondition(!shouldAnimate(env))
env = PetEnvironment(); env.visible = false
precondition(!shouldAnimate(env))
env = PetEnvironment(); env.screenAwake = false
precondition(!shouldAnimate(env))

// The panel: 131 x 97 pt with 3 pt clear under the paws, on a screen whose Dock takes the bottom 70 pt.
let size = CGSize(width: 131, height: 97)
let below: CGFloat = 3
let screen = CGRect(x: 0, y: 70, width: 1512, height: 900)
let home = petDefaultFrame(size: size, belowPaws: below, in: screen)
// Bottom right, 16 pt from the edge, the paws (3 pt above the panel's bottom) on the Dock's top edge.
precondition(home == CGRect(x: 1365, y: 67, width: 131, height: 97))
precondition(home.minY + below == screen.minY && screen.maxX - home.maxX == petEdgeInset)
// Another screen to the left of this one: the same maths from its own origin.
let left = CGRect(x: -1920, y: 0, width: 1920, height: 1055)
precondition(petDefaultFrame(size: size, belowPaws: below, in: left) == CGRect(x: -131 - 16, y: -3, width: 131, height: 97))

// Clamping: back inside on every side; the strip under the paws may hang below the bottom edge.
func clamp(_ x: CGFloat, _ y: CGFloat) -> CGRect {
    petClamped(CGRect(x: x, y: y, width: 131, height: 97), belowPaws: below, in: screen)
}
precondition(clamp(-50, 500) == CGRect(x: 0, y: 500, width: 131, height: 97))
precondition(clamp(1500, 500) == CGRect(x: 1512 - 131, y: 500, width: 131, height: 97))
precondition(clamp(100, 0) == CGRect(x: 100, y: 67, width: 131, height: 97))
precondition(clamp(100, 5000) == CGRect(x: 100, y: 970 - 97, width: 131, height: 97))
precondition(clamp(100, 300) == CGRect(x: 100, y: 300, width: 131, height: 97))
let tiny = CGRect(x: 10, y: 20, width: 100, height: 50)
precondition(petClamped(CGRect(x: 500, y: 500, width: 131, height: 97), belowPaws: below, in: tiny).origin == CGPoint(x: 10, y: 17))

// A drop: paws within 12 pt of the bottom edge snap onto it, above or below; farther away it stays.
func drop(_ pawsAbove: CGFloat) -> CGRect {
    petSnapped(CGRect(x: 400, y: screen.minY - below + pawsAbove, width: 131, height: 97), belowPaws: below, in: screen)
}
precondition(drop(0).minY == 67 && drop(5).minY == 67 && drop(12).minY == 67 && drop(-8).minY == 67)
precondition(drop(12.5).minY == 67 + 12.5 && drop(200).minY == 67 + 200)
precondition(drop(5).minX == 400 && drop(5).size == size)

// The saved spot is the offset of the bottom-right corner, so the pet follows its corner.
let spot = petSpot(of: home, on: screen)
precondition(spot == PetSpot(dx: -16, dy: -3))
precondition(petFrame(size: size, spot: spot, belowPaws: below, on: screen) == home)
let dragged = CGRect(x: 600, y: 400, width: 131, height: 97)
let kept = petSpot(of: dragged, on: screen)
precondition(petFrame(size: size, spot: kept, belowPaws: below, on: screen) == dragged)
// A taller Dock: the pet rides up with the usable area's bottom edge.
let dock = CGRect(x: 0, y: 100, width: 1512, height: 870)
precondition(petFrame(size: size, spot: spot, belowPaws: below, on: dock) == CGRect(x: 1365, y: 97, width: 131, height: 97))
// A smaller screen: a spot that is off it is pulled back on, a spot that fits stays where it is.
let small = CGRect(x: 0, y: 0, width: 1024, height: 700)
precondition(petFrame(size: size, spot: PetSpot(dx: -1400, dy: 800), belowPaws: below, on: small)
    == CGRect(x: 0, y: 700 - 97, width: 131, height: 97))
precondition(petFrame(size: size, spot: kept, belowPaws: below, on: small)
    == CGRect(x: 1024 - 781 - 131, y: 330, width: 131, height: 97))
""",
    )


@darwin
def test_each_state_gets_its_own_moves_and_only_those(tmp_path: Path) -> None:
    run_swift(
        tmp_path,
        r"""
// The blink: the three calm pictures have an eyes-closed patch (the sleeping dogs' eyes are shut, the working,
// waiting, done and error faces have none).
let blinking = PetState.allCases.filter { $0.blinkPatch != nil }
precondition(Set(blinking) == [.sitIdle, .lieIdle, .restLazy])
precondition(PetState.sitIdle.blinkPatch == "sit-idle-blink" && PetState.lieIdle.blinkPatch == "lie-idle-blink")
precondition(PetState.restLazy.blinkPatch == "rest-lazy-blink")
// The ear twitch: only the idle dog, as a body without its right ear and the ear.
let twitching = PetState.allCases.filter { $0.earPieces != nil }
precondition(twitching == [.lieIdle])
precondition(PetState.lieIdle.earPieces?.body == "lie-idle-body" && PetState.lieIdle.earPieces?.ear == "lie-idle-ear")
// The tail wag: only the finished picture, as a body without its tail and the tail.
let wagging = PetState.allCases.filter { $0.tailPieces != nil }
precondition(wagging == [.lieDone])
precondition(PetState.lieDone.tailPieces?.body == "lie-done-body" && PetState.lieDone.tailPieces?.tail == "lie-done-tail")
// The "z": only the three sleeping poses, each from its own spot above the head.
precondition(Set(PetState.allCases.filter { $0.sleepZ != nil }) == Set(PetState.sleepPoses))
precondition(PetState.sleepPoses.allSatisfy { $0.isAsleep && $0.sleepZ != nil })
precondition(Set(PetState.sleepPoses.map { "\($0.sleepZ!)" }).count == 3)
// A piece's file is named like its picture's, in the picture's folder.
precondition(PetState.lieIdle.pieceFile("lie-idle-blink", scale: 2) == "lie-idle-blink@2x.png")
precondition(PetState.sitIdle.pieceFile("sit-idle-blink", scale: 1) == "sit-idle-blink.png")
precondition(PetState.lieDone.fileName(scale: 2) == "lie-done@2x.png" && PetState.lieDone.folder == "pet")
""",
    )


def test_the_moves_use_the_pieces_and_anchors_the_build_ships() -> None:
    pictures = json.loads((ROOT / "macos/art/akita/anchors.json").read_text())[
        "pictures"
    ]
    source = (APP / "PetPolicy.swift").read_text()
    # Every piece the Swift names is a file in both scales, on its picture's canvas.
    for state, folder, pieces in (
        ("sit-idle", "portrait", ["sit-idle-blink"]),
        ("lie-idle", "pet", ["lie-idle-blink", "lie-idle-body", "lie-idle-ear"]),
        ("rest-lazy", "pet", ["rest-lazy-blink"]),
        ("lie-done", "pet", ["lie-done-body", "lie-done-tail"]),
    ):
        for name in pieces:
            assert f'"{name}"' in source
            for suffix, scale in (("", 1), ("@2x", 2)):
                data = (
                    ROOT / f"macos/art/akita/{folder}/{name}{suffix}.png"
                ).read_bytes()
                width, height = struct.unpack(">II", data[16:24])
                assert [width, height] == [c * scale for c in pictures[state]["canvas"]]
    # The "z" starts where anchors.json says, and the tail turns about the pivot it says.
    for state, point in (
        ("sleep-front", (86, 24)),
        ("sleep-side", (62, 22)),
        ("sleep-back", (30, 40)),
    ):
        assert pictures[state]["z"] == list(point)
        camel = re.sub(r"-(\w)", lambda m: m.group(1).upper(), state)
        assert f"case .{camel}: return CGPoint(x: {point[0]}, y: {point[1]})" in source
    pivot = pictures["lie-done"]["tail"]["pivot"]
    assert f"petTailPivot = CGPoint(x: {pivot[0]}, y: {pivot[1]:g})" in source
    ear = pictures["lie-idle"]["ear"]
    assert (
        f"petEarPivot = CGPoint(x: {ear['pivot'][0]}, y: {ear['pivot'][1]})" in source
    )
    # The marks lie on their pictures: the eye bands inside the dog's box, the tail's pivot inside the tail's box.
    for state in ("sit-idle", "lie-idle", "rest-lazy"):
        x, y, w, h = pictures[state]["eyes"]
        bx, by, bw, bh = pictures[state]["box"]
        assert bx < x and by < y and x + w < bx + bw and y + h < by + bh
    x, y, w, h = pictures["lie-done"]["tail"]["box"]
    assert x <= pivot[0] <= x + w and y <= pivot[1] <= y + h
    x, y, w, h = ear["box"]
    assert x <= ear["pivot"][0] <= x + w and y <= ear["pivot"][1] <= y + h
