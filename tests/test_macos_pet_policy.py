"""PetPolicy.swift: Melo's states, the mood the health reply and the quiet clock give, its words, when it may move and
where its panel goes. The Swift is compiled with a test main, as tests/test_macos_gateway_policy.py does.
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
            str(APP / "L10n.swift"),
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


def test_the_first_window_shows_a_still_of_the_moves_at_their_canvas() -> None:
    manifest = json.loads(
        (ROOT / "support/gateway/static/moves/manifest.json").read_text()
    )
    side = manifest["canvas"]["size"] // 2  # the 2x poster at its points
    assert (
        f"let portraitSize = CGSize(width: {side}, height: {side})"
        in (APP / "PetPolicy.swift").read_text()
    )
    for state in ("startup", "thinking"):  # the two stills the window shows
        assert (ROOT / f"support/gateway/static/moves/poster-{state}.webp").is_file()
    # The wake-up's poster is its last frame: Melo sitting up, smiling.
    startup = manifest["states"]["startup"]
    assert startup["poster"] == len(startup["durations_ms"]) - 1
    # The picture is decoration beside the title that says the same in words, and no longer a framed print or a
    # blinking one.
    status = (APP / "StatusView.swift").read_text()
    assert "setAccessibilityElement(false)" in status
    assert "poster-\\(next.rawValue).webp" in status
    assert (
        "melo.png" not in status
        and "NSBezierPath" not in status
        and "blink" not in status
    )


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
func mood(_ data: Data?, stopped: Bool = false, touch: Double = 0) -> PetMood {
    petMood(health: data, stopped: stopped, sinceTouch: touch)
}
let waits = #"[{"kind":"chat","id":"s1"}]"#

// Before the first answer, or with a reply that is not a health body, nothing is known.
precondition(mood(nil) == .unknown && mood(Data("not json".utf8)) == .unknown)
precondition(mood(Data(#"{"api":2}"#.utf8)) == .unknown)
// Nothing runs: the rest, the default.
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

// Quiet: a minute after the last touch, with no run ever, the pet sleeps.
precondition(mood(health(), touch: 59.9) == .resting)
precondition(mood(health(), touch: 60) == .sleeping && mood(health(), touch: 600) == .sleeping)
// The quiet is the shorter of the touch and the end of the newest run: a run that ended 30 s ago keeps the pet
// up however long ago it was touched, and a touch 10 s ago wakes a pet that finished long ago.
precondition(mood(health(recent: ended("done", "30")), touch: 5000) == .resting)
precondition(mood(health(recent: ended("done", "61")), touch: 5000) == .sleeping)
precondition(mood(health(recent: ended("done", "7000")), touch: 10) == .resting)
precondition(mood(health(recent: ended("failed", "61")), touch: 5000) == .sleeping)
// A touch ends the sleep at once: the clock restarts.
precondition(mood(health(), touch: 0.0) == .resting)
// Odd clocks never trap: a huge or infinite quiet sleeps, a negative or NaN one rests.
precondition(mood(health(), touch: 1e300) == .sleeping && mood(health(), touch: .infinity) == .sleeping)
precondition(mood(health(), touch: -5) == .resting && mood(health(), touch: .nan) == .resting)

// The words: the moods that look the same (a failure and a stopped app, rest and sleep) say different things.
precondition(PetMood.unknown.label == "Melo")
precondition(PetMood.resting.label == "Melo, resting. Nothing running.")
precondition(PetMood.sleeping.label == "Melo, sleeping. Nothing running.")
precondition(PetMood.working(1).label == "Melo, working. 1 running.")
precondition(PetMood.working(2).label == "Melo, working. 2 running.")
precondition(PetMood.waiting.label == "Melo, waiting for your approval.")
precondition(PetMood.done.label == "Melo, done.")
precondition(PetMood.failed.label == "Melo, the last run did not finish.")
precondition(PetMood.stopped.label == "Melo, localmelo stopped.")
precondition(PetMood.waiting.help == "Opens what waits for your approval")
precondition(PetMood.resting.help == "Opens localmelo" && PetMood.stopped.help == "Opens localmelo")

// The first window's portrait: the end of the wake-up while the server starts and once it is ready, the thinking
// pose when it cannot go on, so a failure never reads as a healthy start.
precondition(startPortrait(failed: false) == .startup && startPortrait(failed: true) == .thinking)
""",
    )


@darwin
def test_the_pet_moves_only_when_it_may_and_its_panel_stays_on_screen(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        r"""
// The activity pill's loops stop for Reduce Motion, Low Power Mode, heat, a hidden panel and a sleeping screen.
var env = PetEnvironment()
precondition(shouldAnimate(env))
env.reduceMotion = true
precondition(!shouldAnimate(env))
env = PetEnvironment(); env.lowPower = true
precondition(!shouldAnimate(env))
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
// The pet's moves stop being moves only under Reduce Motion: Low Power Mode and heat leave them playing; a hidden
// panel or a sleeping screen stops their clock where it is.
env = PetEnvironment()
precondition(petMovesMayMove(env) && petMovesRun(env))
env.reduceMotion = true
precondition(!petMovesMayMove(env))
env = PetEnvironment(); env.lowPower = true; env.thermal = .critical
precondition(petMovesMayMove(env) && petMovesRun(env))
env = PetEnvironment(); env.visible = false
precondition(petMovesMayMove(env) && !petMovesRun(env))
env = PetEnvironment(); env.screenAwake = false
precondition(!petMovesRun(env))

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
def test_each_mood_shows_moves_or_a_still_never_both(tmp_path: Path) -> None:
    run_swift(
        tmp_path,
        r"""
let moods: [PetMood] = [.unknown, .resting, .sleeping, .working(1), .waiting, .done, .failed, .stopped]
// Exactly one of the two for every mood, thinking or not.
for mood in moods {
    for thinking in [false, true] {
        precondition((petMove(mood, thinking: thinking) == nil) != (petStill(mood) == nil))
    }
}
// The stills: nothing known lies at rest; a finished run is the end of the wake-up (sitting up, smiling); a failure
// and a stopped app are the thinking pose, never a healthy rest.
precondition(petStill(.unknown) == .idleRest && petStill(.done) == .startup)
precondition(petStill(.failed) == .thinking && petStill(.stopped) == .thinking)
// Every other mood plays: work types, the model alone thinks, a wait raises a paw, rest and sleep lie down.
precondition(petMove(.working(2), thinking: false) == .working && petMove(.working(2), thinking: true) == .thinking)
precondition(petMove(.waiting, thinking: true) == .attention)
precondition(petMove(.resting, thinking: false) == .idleRest && petMove(.sleeping, thinking: false) == .idleRest)
""",
    )
