"""PillPolicy.swift: what the activity pill shows from the health reply, when it is hidden and when the pet comes back,
and where its panel goes. The Swift is compiled with a test main, as tests/test_macos_pet_policy.py does; no window is
opened here (the pill is drawn offscreen by tests/test_macos_pill_panel.py).
"""

from __future__ import annotations

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

# The health reply the fixtures build: items that run, the focus, waits, the newest finished run and the model server.
FIXTURES = r"""
func item(_ kind: String, _ id: String, _ activity: String, _ icon: String, title: String = "") -> String {
    #"{"kind":"\#(kind)","id":"\#(id)","title":"\#(title)","activity":"\#(activity)","icon":"\#(icon)"}"#
}
func health(items: [String] = [], focus: String? = nil, waiting: [String] = [], recent: String = "null",
            server: String = "null") -> Data {
    let list = items.joined(separator: ",")
    let wait = waiting.map { #"{"kind":"chat","id":"\#($0)"}"# }.joined(separator: ",")
    let first = focus ?? items.first ?? "null"
    return Data(#"{"api":2,"model_server":\#(server),"status":{"running":{"chats":\#(items.count),"tasks":0,"runs":0,"terminal":0},"items":[\#(list)],"waiting":[\#(wait)],"recent":\#(recent),"focus":\#(first)}}"#.utf8)
}
func ended(_ outcome: String, _ ago: String, title: String = "Slides", kind: String = "chat", id: String = "c9") -> String {
    #"{"kind":"\#(kind)","id":"\#(id)","title":"\#(title)","outcome":"\#(outcome)","ended_s_ago":\#(ago)}"#
}
let reading = item("chat", "c1", "Reading example.com", "web")
let tests = item("task", "t1", "Running the tests", "run")
"""


def run_swift(tmp_path: Path, source: str) -> None:
    """Compile the policy files with a test main (a raw string: it holds `\\#(` interpolations) and run it."""
    main = tmp_path / "main.swift"
    main.write_text("import Foundation\n" + FIXTURES + source)
    binary = tmp_path / "test-pill-policy"
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            *[
                str(APP / name)
                for name in (
                    "GatewayPolicy.swift",
                    "PetPolicy.swift",
                    "PillPolicy.swift",
                )
            ],
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


@darwin
def test_the_pill_says_what_the_focus_does_and_how_many_more_run(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        r"""
func snap(_ data: Data, age: Double = 0) -> PillSnapshot? { pillSnapshot(health: data, age: age) }

// Nothing known, or nothing running and nothing just ended: no pill.
precondition(snap(Data("not json".utf8)) == nil && snap(Data(#"{"api":2}"#.utf8)) == nil)
precondition(snap(health()) == nil)
// Work: the focus's own words and kind, the mark of that kind, the working pose for it, no chip for one.
let one = snap(health(items: [reading]))!.model
precondition(one.state == .working && one.text == "Reading example.com" && one.icon == .web && one.more == 0)
precondition(one.glyph == "\u{2295}" && one.portrait == .workRead && one.click == .route("#/s/c1"))
// Several run: the newest one is named and the rest are counted ("+1").
let two = snap(health(items: [tests, reading], focus: tests))!.model
precondition(two.text == "Running the tests" && two.more == 1 && two.click == .route("#/tasks/t1") && two.portrait == .workLaptop)
precondition(snap(health(items: [reading, tests, reading, tests]))!.model.more == 3)
// Each kind of work has its pose and mark: reading and the web, thinking and memory, writing, commands and tools.
precondition(PillPortrait.working(.read) == .workRead && PillPortrait.working(.web) == .workRead)
precondition(PillPortrait.working(.think) == .workThink && PillPortrait.working(.memory) == .workThink)
precondition(PillPortrait.working(.write) == .workLaptop && PillPortrait.working(.run) == .workLaptop)
precondition(PillPortrait.working(.tool) == .workLaptop && PillIcon(name: "nonsense") == .tool && PillIcon(name: nil) == .tool)
// A run or terminal chat has no page: a press only opens the window. A focus that is missing still reads "Working".
let run = snap(health(items: [item("run", "r1", "Writing notes.md", "write")]))!.model
precondition(run.click == .window && run.glyph == "\u{270E}")
let blind = snap(health(items: [reading], focus: "null"))!.model
precondition(blind.text == "Working" && blind.click == .window && blind.state == .working)
// What the gateway says is cleaned: one line, cut at 200 characters, and an empty line is no focus.
let messy = snap(health(items: [item("chat", "c1", "Reading\\n  a   page", "read")]))!.model
precondition(messy.text == "Reading a page")
precondition(pillFocus(health(items: [item("chat", "c1", "", "read")])) == nil)
precondition(pillFocus(health(items: [item("chat", "", "Thinking", "think")])) == nil)
precondition(pillFocus(health(items: [item("alien", "c1", "Thinking", "think")])) == nil)
precondition(pillFocus(health(items: [item("chat", "c1", String(repeating: "x", count: 500), "think")]))!.activity.count == 200)

// A wait beats work and the model server: amber, "Waiting for your approval", a press goes to what waits.
let waits = snap(health(items: [reading, tests], waiting: ["c1"], server: #"{"reachable":false}"#))!.model
precondition(waits.state == .waiting && waits.text == "Waiting for your approval" && waits.click == .waiting)
precondition(waits.glyph == "!" && waits.portrait == .waitAlert && waits.more == 1)
// A held task is not running, so a wait can come with a count of 0 and no chip.
precondition(snap(health(waiting: ["c1"]))!.model.more == 0)
// The model server down shows while something runs; nothing running shows nothing (the pet lies there as before).
let down = snap(health(items: [reading], server: #"{"reachable":false}"#))!.model
precondition(down.state == .serverDown && down.text == "Model server not reachable" && down.portrait == .error)
precondition(snap(health(server: #"{"reachable":false}"#)) == nil)
precondition(snap(health(items: [reading], server: #"{"reachable":true}"#))!.model.state == .working)
precondition(snap(health(items: [reading], server: #"{"reachable":null}"#))!.model.state == .working)

// A finished run: "Done: <title>" for 6 s, "Stopped: <title>" for 10 s, counted from when it ended and the age of the reply.
let done = snap(health(recent: ended("done", "1.5")), age: 0.5)!
precondition(done.model.state == .done && done.model.text == "Done: Slides" && done.model.glyph == "\u{2713}")
precondition(done.model.portrait == .done && done.model.click == .route("#/s/c9") && abs(done.expiresIn! - 4) < 1e-9)
precondition(snap(health(recent: ended("done", "5.9")))!.expiresIn! > 0 && snap(health(recent: ended("done", "6.0"))) == nil)
precondition(snap(health(recent: ended("done", "5.0")), age: 1.5) == nil)  // the reply is old: the 6 s are used up
let failed = snap(health(recent: ended("failed", "9.9", title: "Fix CI", kind: "task", id: "t4")))!
precondition(failed.model.state == .failed && failed.model.text == "Stopped: Fix CI" && failed.model.glyph == "\u{2717}")
precondition(failed.model.portrait == .error && failed.model.click == .route("#/tasks/t4"))
precondition(snap(health(recent: ended("failed", "10.0"))) == nil)
// A run the person stopped shows nothing; a title that is missing leaves the bare word; a run's end has no page.
precondition(snap(health(recent: ended("stopped", "1"))) == nil)
precondition(snap(health(recent: ended("done", "1", title: "")))!.model.text == "Done")
precondition(snap(health(recent: ended("done", "1", kind: "run", id: "r2")))!.model.click == .window)
// Work beats a finished run.
precondition(snap(health(items: [reading], recent: ended("done", "1")))!.model.state == .working)

// VoiceOver: one sentence for the pill, and where a press goes.
precondition(one.label == "Melo is reading example.com. Opens the chat.")
precondition(two.label == "Melo is running the tests. Opens the task.")
precondition(waits.label == "Melo is waiting for your approval. Opens the chat.")
precondition(done.model.label == "Melo is done: Slides. Opens the chat.")
precondition(failed.model.label == "Melo stopped: Fix CI. Opens the task.")
precondition(down.label == "Melo: model server not reachable. Opens the chat.")

// The picture: the new art's file when it is there, else today's face, at the screen's scale.
precondition(pillPictureFile(.workRead, scale: 2, exists: { _ in true }) == "pill/work-read@2x.png")
precondition(pillPictureFile(.workRead, scale: 1, exists: { _ in true }) == "pill/work-read.png")
precondition(pillPictureFile(.workRead, scale: 2, exists: { _ in false }) == "face/working@2x.png")
precondition(pillPictureFile(.workLaptop, scale: 1, exists: { _ in false }) == "face/working.png")
precondition(pillPictureFile(.waitAlert, scale: 2, exists: { _ in false }) == "face/waiting@2x.png")
precondition(pillPictureFile(.done, scale: 2, exists: { _ in true }) == "face/done@2x.png")  // no new art for it
precondition(pillPictureFile(.error, scale: 1, exists: { _ in true }) == "face/error.png")
precondition(pillPictureFile(.workThink, scale: 2, exists: { $0 == "pill/work-think@2x.png" }) == "pill/work-think@2x.png")
precondition(pillPictureFile(.workRead, scale: 2, exists: { $0 == "pill/work-think@2x.png" }) == "face/working@2x.png")
""",
    )


@darwin
def test_the_pill_follows_the_clock_the_window_and_the_choice(tmp_path: Path) -> None:
    run_swift(
        tmp_path,
        r"""
var policy = PillPolicy()
func ask(_ data: Data?, reply: Double, now: Double, window: Bool = false, mode: DesktopMode = .pill,
         stopped: Bool = false) -> PillDecision {
    policy.update(health: data, healthAt: reply, stopped: stopped, windowShown: window, mode: mode, now: now)
}
let idle = health()
let writing = item("chat", "c1", "Writing slides.md", "write")

// Quiet: nothing to show, and nothing to wag about.
var d = ask(idle, reply: 0, now: 0)
precondition(d.pill == nil && !d.petWags && d.recheckIn == nil)
// Work starts: the pill appears with the line, and asks to be asked again before the reply goes stale.
d = ask(health(items: [reading]), reply: 10, now: 10)
precondition(d.pill?.text == "Reading example.com" && d.pill?.state == .working && !d.petWags)
precondition(d.recheckIn == 12)
// The line changes at most once per 1.5 s: a newer one waits until then, and the pill asks to be asked again then.
d = ask(health(items: [tests]), reply: 10.5, now: 10.5)
precondition(d.pill?.text == "Reading example.com" && d.pill?.icon == .web && d.pill?.portrait == .workRead)
precondition(d.pill?.click == .route("#/tasks/t1"))  // a press already goes to the new focus
precondition(abs(d.recheckIn! - 1.0) < 1e-9)
d = ask(health(items: [tests]), reply: 10.5, now: 11.49)
precondition(d.pill?.text == "Reading example.com")
d = ask(health(items: [tests]), reply: 10.5, now: 11.5)
precondition(d.pill?.text == "Running the tests" && d.pill?.icon == .run && d.pill?.portrait == .workLaptop)
// The count is not the line: "+1" shows at once.
d = ask(health(items: [tests, reading], focus: tests), reply: 12, now: 12)
precondition(d.pill?.text == "Running the tests" && d.pill?.more == 1)
// A new state is not held: the wait shows at once, though the line changed a moment ago.
d = ask(health(items: [tests], waiting: ["t1"]), reply: 12.2, now: 12.2)
precondition(d.pill?.state == .waiting && d.pill?.text == "Waiting for your approval")
d = ask(health(items: [tests]), reply: 12.4, now: 12.4)
precondition(d.pill?.state == .working && d.pill?.text == "Running the tests")

// The end: "Done: <title>" for 6 s counted from when the run ended; then the pet comes back wagging, once.
d = ask(health(recent: ended("done", "0.5")), reply: 20, now: 20)
precondition(d.pill?.state == .done && d.pill?.text == "Done: Slides" && d.pill?.more == 0)
precondition(d.recheckIn == 5.5)
d = ask(health(recent: ended("done", "0.5")), reply: 20, now: 25.25)
precondition(d.pill?.state == .done && !d.petWags && d.recheckIn == 0.25)
d = ask(health(recent: ended("done", "0.5")), reply: 20, now: 25.5)
precondition(d.pill == nil && d.petWags && d.recheckIn == nil)
d = ask(health(recent: ended("done", "2.4")), reply: 22, now: 26)
precondition(d.pill == nil && !d.petWags)  // asked again: the wag was told once
// A failed end shows 10 s and the pet comes back without a wag.
d = ask(health(recent: ended("failed", "1", title: "Fix CI")), reply: 30, now: 30)
precondition(d.pill?.state == .failed && d.pill?.text == "Stopped: Fix CI")
d = ask(health(recent: ended("failed", "1", title: "Fix CI")), reply: 30, now: 38.5)
precondition(d.pill?.state == .failed)
d = ask(health(recent: ended("failed", "1", title: "Fix CI")), reply: 30, now: 39)
precondition(d.pill == nil && !d.petWags)
// Work that starts during the "Done" takes the pill over at once.
d = ask(health(recent: ended("done", "1")), reply: 40, now: 40)
precondition(d.pill?.state == .done)
d = ask(health(items: [reading], recent: ended("done", "3")), reply: 42, now: 42)
precondition(d.pill?.state == .working && !d.petWags)

// A reply that is not renewed is not believed for ever: after 12 s the pill goes (the gateway may have hung).
d = ask(health(items: [reading]), reply: 50, now: 50)
precondition(d.pill != nil && d.recheckIn == 12)
d = ask(health(items: [reading]), reply: 50, now: 61.9)
precondition(d.pill != nil && abs(d.recheckIn! - 0.1) < 1e-9)
d = ask(health(items: [reading]), reply: 50, now: 62)
precondition(d.pill == nil)
d = ask(nil, reply: 62, now: 62)
precondition(d.pill == nil)  // no reply yet

// The pill is there only while the window is put away, with the pill chosen, and the app has not failed.
d = ask(health(items: [reading]), reply: 70, now: 70)
precondition(d.pill != nil)
d = ask(health(items: [reading]), reply: 70, now: 70, window: true)
precondition(d.pill == nil && !d.petWags)
d = ask(health(items: [reading]), reply: 70, now: 70, mode: .pet)
precondition(d.pill == nil)
d = ask(health(items: [reading]), reply: 70, now: 70, mode: .nothing)
precondition(d.pill == nil)
d = ask(health(items: [reading]), reply: 70, now: 70, stopped: true)
precondition(d.pill == nil)
// A "Done" cut short by the window opening, or by a new choice, is no wag.
d = ask(health(recent: ended("done", "1")), reply: 80, now: 80)
precondition(d.pill?.state == .done)
d = ask(health(recent: ended("done", "1")), reply: 80, now: 81, window: true)
precondition(d.pill == nil && !d.petWags)
d = ask(health(recent: ended("done", "1")), reply: 80, now: 82)
precondition(d.pill?.state == .done)
d = ask(health(recent: ended("done", "1")), reply: 80, now: 82.5, mode: .pet)
precondition(d.pill == nil && !d.petWags)

// "×" hides the pill until the next task starts. The same work stays hidden, whatever it does next.
policy = PillPolicy()
d = ask(health(items: [reading, tests], focus: reading), reply: 100, now: 100)
precondition(d.pill != nil)
policy.dismiss()
d = ask(health(items: [reading, tests], focus: reading), reply: 100, now: 100.1)
precondition(d.pill == nil && !d.petWags)
d = ask(health(items: [reading, tests], focus: tests), reply: 102, now: 102)
precondition(d.pill == nil)
d = ask(health(items: [tests], waiting: ["t1"]), reply: 104, now: 104)
precondition(d.pill == nil)  // the same work asking for an answer is the same work: the menu bar says it
// Its end is hidden too, and the pet is there: the pill does not come back for what the person sent away.
d = ask(health(recent: ended("done", "1", id: "c1")), reply: 106, now: 106)
precondition(d.pill == nil)
// Something that was not running starts: the pill comes back.
d = ask(health(items: [item("chat", "c2", "Thinking", "think")]), reply: 108, now: 108)
precondition(d.pill?.text == "Thinking")
// After a quiet moment, even the same chat's next turn is a new task.
policy.dismiss()
d = ask(health(items: [item("chat", "c2", "Thinking", "think")]), reply: 110, now: 110)
precondition(d.pill == nil)
d = ask(idle, reply: 112, now: 112)
precondition(d.pill == nil)
d = ask(health(items: [item("chat", "c2", "Reading a page", "read")]), reply: 114, now: 114)
precondition(d.pill?.text == "Reading a page")
// A run that started and ended between two replies is a new task as well.
policy.dismiss()
d = ask(health(recent: ended("done", "1", id: "c7")), reply: 116, now: 116)
precondition(d.pill?.state == .done && d.pill?.text == "Done: Slides")
// "×" on the pill that says "Done" hides it at once, without a wag.
policy.dismiss()
d = ask(health(recent: ended("done", "2", id: "c7")), reply: 118, now: 118)
precondition(d.pill == nil && !d.petWags)
// Dismissing before anything was seen hides nothing for long: what comes next is new.
policy = PillPolicy()
policy.dismiss()
d = ask(health(items: [reading]), reply: 120, now: 120)
precondition(d.pill != nil)
""",
    )


@darwin
def test_the_pill_sits_at_the_pets_spot_and_stays_on_the_screen(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        r"""
// The width: the portrait and mark, the line (cut at 320 pt) and the tail; the "+1" chip adds its own room.
precondition(pillWidth(text: 152, more: 0) == 244)  // "Reading example.com" in the design
precondition(pillWidth(text: 118, more: 1) == 74 + 118 + 8 + 30 + 14)
precondition(pillWidth(text: 900, more: 0) == 74 + 320 + 18 && pillWidth(text: -5, more: 0) == 92)
precondition(pillPanelSize(width: 244) == CGSize(width: 254, height: 58))  // 5 pt around the 48 pt capsule

let main = PetScreen(key: "main", visible: CGRect(x: 0, y: 70, width: 1512, height: 880))  // Dock below, menu bar above
let side = PetScreen(key: "side", visible: CGRect(x: -1920, y: 0, width: 1920, height: 1055))
let size = pillPanelSize(width: 244)
precondition(pillPlacement(PetSettings(), size: size, screens: []) == nil)
// First time: the bottom right of the first screen, the pet's 16 pt in, the clear strip on the Dock's edge.
let first = pillPlacement(PetSettings(), size: size, screens: [main, side])!
precondition(first.screen == main && first.frame == CGRect(x: 1512 - 16 - 254, y: 70, width: 254, height: 58))
// The pet's saved spot is the pill's: the right edges meet, whatever the width, and the bottoms are level.
var settings = PetSettings()
let petAt = CGRect(x: 900, y: 300, width: 132, height: 97)
settings.remember(screen: "main", spot: petSpot(of: petAt, on: main.visible))
let there = pillPlacement(settings, size: size, screens: [main])!.frame
precondition(there.maxX == petAt.maxX && there.minY == petAt.minY)
let wider = pillPlacement(settings, size: pillPanelSize(width: 400), screens: [main])!.frame
precondition(wider.maxX == petAt.maxX && wider.width == 410)
// The screen it was left on, when it is still there; else the first one.
settings.remember(screen: "side", spot: PetSpot(dx: -100, dy: 200))
precondition(pillPlacement(settings, size: size, screens: [main, side])!.screen == side)
precondition(pillPlacement(settings, size: size, screens: [main])!.screen == main)
// Never over the menu bar or the Dock: a spot from a taller screen is pulled in, on every side.
var low = PetSettings()
low.remember(screen: "main", spot: PetSpot(dx: -16, dy: -50))
precondition(pillPlacement(low, size: size, screens: [main])!.frame.minY == 70)
var high = PetSettings()
high.remember(screen: "main", spot: PetSpot(dx: -16, dy: 5000))
precondition(pillPlacement(high, size: size, screens: [main])!.frame.maxY == 950)
var left = PetSettings()
left.remember(screen: "main", spot: PetSpot(dx: -5000, dy: 100))
precondition(pillPlacement(left, size: size, screens: [main])!.frame.minX == 0)
var right = PetSettings()
right.remember(screen: "main", spot: PetSpot(dx: 300, dy: 100))
precondition(pillPlacement(right, size: size, screens: [main])!.frame.maxX == 1512)
// A smaller screen than the pill is wide keeps its left corner rather than leave the screen's other side.
let narrow = PetScreen(key: "narrow", visible: CGRect(x: 0, y: 0, width: 200, height: 300))
precondition(pillPlacement(PetSettings(), size: size, screens: [narrow])!.frame.minX == 0)

// A longer line: the right edge stays and the pill grows to the left; at the screen's edge it is pulled back in.
let grown = pillResized(CGRect(x: 1000, y: 200, width: 254, height: 58), size: pillPanelSize(width: 400), in: main.visible)
precondition(grown == CGRect(x: 1254 - 410, y: 200, width: 410, height: 58))
let edge = pillResized(CGRect(x: 10, y: 200, width: 254, height: 58), size: pillPanelSize(width: 400), in: main.visible)
precondition(edge.minX == 0 && edge.width == 410)

// A drop: the pet's snapping (within 12 pt of the Dock it lands on the usable area's edge), kept on screen, and the
// spot is the pet's kind, so the pet comes back where the pill was left.
let near = pillDropped(CGRect(x: 600, y: 80, width: 254, height: 58), in: main.visible)
precondition(near.frame == CGRect(x: 600, y: 70, width: 254, height: 58))
let off = pillDropped(CGRect(x: 600, y: 20, width: 254, height: 58), in: main.visible)
precondition(off.frame.minY == 70)
let free = pillDropped(CGRect(x: 600, y: 400, width: 254, height: 58), in: main.visible)
precondition(free.frame.minY == 400 && free.spot == PetSpot(dx: 854 - 1512, dy: 330))
let top = pillDropped(CGRect(x: 600, y: 940, width: 254, height: 58), in: main.visible)
precondition(top.frame.maxY == 950)
var kept = PetSettings()
kept.remember(screen: "main", spot: free.spot)
precondition(petPlacement(kept, screens: [main])!.frame.maxX == free.frame.maxX)  // the pet's right edge is the pill's
""",
    )


@darwin
def test_the_desktop_choice_is_saved_with_the_pets_settings(tmp_path: Path) -> None:
    run_swift(
        tmp_path,
        r"""
func settings(_ text: String?) -> PetSettings { petSettings(text.map { Data($0.utf8) }) }

// The default is Melo and the pill; a file from before the pill is the same, and a hidden Melo stays hidden.
precondition(PetSettings().mode == .pill && settings(nil).mode == .pill && settings("{}").mode == .pill)
precondition(settings(#"{"hidden":true}"#).mode == .nothing && settings(#"{"hidden":false}"#).mode == .pill)
precondition(settings(#"{"pill":false}"#).mode == .pet && settings(#"{"pill":true}"#).mode == .pill)
precondition(settings(#"{"pill":"no"}"#).mode == .pill && settings(#"{"pill":0}"#).mode == .pill)  // damaged: the default
precondition(settings(#"{"hidden":true,"pill":false}"#).mode == .nothing)
// Choosing: each of the three is one setting; Nothing keeps the pill's flag, so Show Melo brings back what was there.
var saved = PetSettings()
saved.mode = .pet
precondition(!saved.hidden && !saved.pill && saved.mode == .pet)
saved.mode = .nothing
precondition(saved.hidden && !saved.pill && saved.mode == .nothing)
saved.hidden = false  // what "Show Melo" does
precondition(saved.mode == .pet)
saved.mode = .pill
precondition(!saved.hidden && saved.pill && saved.mode == .pill)
saved.mode = .nothing
precondition(saved.hidden && saved.pill)
// Written and read back, the same, with the spots.
saved.remember(screen: "A", spot: PetSpot(dx: -16, dy: 0))
precondition(petSettings(petSettingsData(saved)) == saved && petSettings(petSettingsData(saved)).mode == .nothing)
saved.mode = .pet
precondition(petSettings(petSettingsData(saved)).mode == .pet)
// The menu's words.
precondition(DesktopMode.allCases.map { $0.title } == ["Melo and the activity pill", "Melo only", "Nothing"])
precondition(DesktopMode.allCases.map { $0.rawValue } == ["pill", "pet", "nothing"])
// A finished run's title reaches the pet's reading of the reply: cleaned, and empty when missing.
let titled = #"{"api":2,"status":{"recent":{"kind":"task","id":"t1","title":"  Fix\nthe   build ","outcome":"done","ended_s_ago":1}}}"#
precondition(recentOutcome(Data(titled.utf8))?.title == "Fix the build")
let bare = #"{"api":2,"status":{"recent":{"kind":"task","id":"t1","outcome":"done","ended_s_ago":1}}}"#
precondition(recentOutcome(Data(bare.utf8))?.title == "")
precondition(consoleRoute(kind: "chat", id: "a1") == "#/s/a1" && consoleRoute(kind: "task", id: "t-2") == "#/tasks/t-2")
precondition(consoleRoute(kind: "run", id: "r1") == nil && consoleRoute(kind: "chat", id: "a/b") == nil)
precondition(consoleRoute(kind: "chat", id: "") == nil && consoleRoute(kind: "chat", id: String(repeating: "a", count: 129)) == nil)
precondition(modelServerReachable(Data(#"{"model_server":{"reachable":false}}"#.utf8)) == false)
precondition(modelServerReachable(Data(#"{"model_server":{"reachable":true}}"#.utf8)) == true)
precondition(modelServerReachable(Data(#"{"model_server":null}"#.utf8)) == nil && modelServerReachable(Data("{}".utf8)) == nil)
""",
    )
