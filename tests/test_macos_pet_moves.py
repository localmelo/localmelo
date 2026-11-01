"""The desktop pet's moves (PetMoves.swift, and their rules in PetPolicy.swift): which state or still each mood shows,
the model's own mark for thinking, the wake-up and the plays now and then, the manifest the app reads, and the player
on layers built off screen: a key-frame animation of the frames, attention's quick fade, the same state never
restarting, held posters for the stills, posters when nothing may move, and a clock that stops while the pet cannot be
seen. No window is opened.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "macos/LocalMeloApp"
MOVES = ROOT / "support/gateway/static/moves"
darwin = pytest.mark.skipif(
    sys.platform != "darwin", reason="Swift AppKit host is macOS-only"
)


def run_swift(tmp_path: Path, files: list[str], source: str) -> None:
    """Compile `files` of the app with a test main (a raw string) and run it."""
    main = tmp_path / "main.swift"
    main.write_text("import AppKit\nimport QuartzCore\n" + source)
    binary = tmp_path / "test-pet-moves"
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
        timeout=180,
    )
    subprocess.run([str(binary)], check=True, timeout=20)


POLICY = ["GatewayPolicy.swift", "L10n.swift", "PetPolicy.swift"]


@darwin
def test_each_mood_has_its_move_or_its_still(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        POLICY,
        r"""
precondition(petMove(.waiting, thinking: false) == .attention)
precondition(petMove(.waiting, thinking: true) == .attention)  // a wait wins over the model answering
precondition(petMove(.working(2), thinking: false) == .working)
precondition(petMove(.working(1), thinking: true) == .thinking)
precondition(petMove(.resting, thinking: false) == .idleRest)
precondition(petMove(.sleeping, thinking: false) == .idleRest)
// Nothing known, a run that just ended, a failure and a stopped app show a still: a poster of the moves, held.
for mood in [PetMood.unknown, .done, .failed, .stopped] {
    precondition(petMove(mood, thinking: true) == nil && petStill(mood) != nil)
}
// Their words stay the mood's.
precondition(PetMood.working(1).label == "Melo, working. 1 running.")
precondition(PetMood.waiting.label == "Melo, waiting for your approval.")
precondition(PetMove.allCases.map { $0.rawValue } == ["startup", "thinking", "working", "attention", "idle_rest", "idle_play"])

// The model thinks only on the gateway's own mark, for everything that runs.
func health(_ running: Int, _ items: String) -> Data {
    Data(#"{"status": {"running": {"chats": \#(running), "tasks": 0, "runs": 0, "terminal": 0}, "items": [\#(items)], "waiting": []}}"#.utf8)
}
let answering = #"{"kind": "chat", "id": "a", "icon": "think", "answering": true}"#
let words = #"{"kind": "chat", "id": "b", "icon": "think", "activity": "Thinking"}"#
let tool = #"{"kind": "task", "id": "c", "icon": "run", "answering": false}"#
precondition(petThinking(health(1, answering)))
precondition(!petThinking(health(1, words)))  // the word alone: an older gateway, or no events to read
precondition(!petThinking(health(2, answering + "," + tool)))
precondition(!petThinking(health(2, answering)))  // one of the two is not listed
precondition(!petThinking(health(0, "")) && !petThinking(Data("{}".utf8)))

// Where the canvas lies on the panel: its anchor (the paws) on the ground line, 3 pt up, at the middle; nothing
// drawn is cut (the art leaves 22 of 512 px clear above the ears and 22 below the paws).
let frame = petMoveFrame(side: 100, anchor: CGPoint(x: 128, y: 244), pixels: 256, in: CGSize(width: 132, height: 97), belowPaws: 3)
precondition(frame == CGRect(x: 16, y: 3 - 100.0 * 12 / 256, width: 100, height: 100))
precondition(frame.maxY - 97 < 100.0 * 22 / 512 && -frame.minY < 100.0 * 22 / 512)
precondition(petMoveSide == 100)

// What may move: the moves show posters under Reduce Motion only, keep playing in Low Power Mode and heat, and stop
// while unseen.
var env = PetEnvironment()
precondition(petMovesMayMove(env) && petMovesRun(env))
env.reduceMotion = true; precondition(!petMovesMayMove(env) && petMovesRun(env))
env = PetEnvironment(); env.lowPower = true; precondition(petMovesMayMove(env))
env = PetEnvironment(); env.thermal = .serious; precondition(petMovesMayMove(env))
env = PetEnvironment(); env.visible = false; precondition(petMovesMayMove(env) && !petMovesRun(env))
env = PetEnvironment(); env.screenAwake = false; precondition(!petMovesRun(env))
""",
    )


@darwin
def test_the_pet_wakes_up_once_and_plays_now_and_then_while_it_rests(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        POLICY,
        r"""
precondition(petPlayAfter == 20 && petPlayEvery == 300 && petSitFor == 8)
let start = Date(timeIntervalSince1970: 1_000_000)
func at(_ seconds: Double) -> Date { start.addingTimeInterval(seconds) }
var picker = PetMovePicker()
// The first move at rest is the wake-up, held until it ends; then Melo sits on its last frame for 8 s, and rests.
precondition(picker.pick(.resting, thinking: false, still: false, now: at(0))! == (.startup, false))
precondition(picker.pick(.resting, thinking: false, still: false, now: at(30))! == (.startup, false))
picker.ended(.startup, now: at(31))
precondition(picker.pick(.resting, thinking: false, still: false, now: at(31))! == (.startup, false))
precondition(picker.pick(.resting, thinking: false, still: false, now: at(38.9))! == (.startup, false))
precondition(picker.pick(.resting, thinking: false, still: false, now: at(39))! == (.idleRest, false))
picker.ended(.startup, now: at(40))  // a late end of the wake-up never sits again
precondition(picker.pick(.resting, thinking: false, still: false, now: at(41))! == (.idleRest, false))
// 20 s into the rest: one play, to its end, then rest; not again for 5 minutes.
precondition(picker.pick(.resting, thinking: false, still: false, now: at(58))! == (.idleRest, false))
precondition(picker.pick(.resting, thinking: false, still: false, now: at(59))! == (.idlePlay, true))
precondition(picker.pick(.resting, thinking: false, still: false, now: at(60))! == (.idlePlay, true))
picker.ended(.idlePlay, now: at(65))
precondition(picker.pick(.resting, thinking: false, still: false, now: at(66))! == (.idleRest, false))
precondition(picker.pick(.resting, thinking: false, still: false, now: at(358))! == (.idleRest, false))
precondition(picker.pick(.resting, thinking: false, still: false, now: at(359))! == (.idlePlay, true))
// Asleep it only rests, and a play under way gives way.
precondition(picker.pick(.sleeping, thinking: false, still: false, now: at(360))! == (.idleRest, false))
// Work and a wait take over at once; the rest after them starts again from its beginning.
precondition(picker.pick(.working(1), thinking: false, still: false, now: at(400))! == (.working, false))
precondition(picker.pick(.waiting, thinking: false, still: false, now: at(401))! == (.attention, false))
precondition(picker.pick(.resting, thinking: false, still: false, now: at(402))! == (.idleRest, false))
precondition(picker.pick(.resting, thinking: false, still: false, now: at(700))! == (.idlePlay, true))
// No moves for an ending (a still shows): the picker says nothing, and the rest after it starts over.
precondition(picker.pick(.done, thinking: false, still: false, now: at(701)) == nil)
precondition(picker.restSince == nil && !picker.playing)
// A still ends the sitting after the wake-up at once: back at rest the pet lies down.
var sitting = PetMovePicker()
_ = sitting.pick(.resting, thinking: false, still: false, now: at(0))
sitting.ended(.startup, now: at(3))
precondition(sitting.pick(.failed, thinking: false, still: false, now: at(4)) == nil)
precondition(sitting.pick(.resting, thinking: false, still: false, now: at(5))! == (.idleRest, false))
// So does work: no sitting is left over after it.
var busy2 = PetMovePicker()
_ = busy2.pick(.resting, thinking: false, still: false, now: at(0))
busy2.ended(.startup, now: at(3))
precondition(busy2.pick(.working(1), thinking: false, still: false, now: at(4))! == (.working, false))
precondition(busy2.pick(.resting, thinking: false, still: false, now: at(5))! == (.idleRest, false))

// A pet that is first seen while work runs never plays the wake-up later.
var busy = PetMovePicker()
precondition(busy.pick(.working(1), thinking: true, still: false, now: at(0))! == (.thinking, false))
precondition(busy.pick(.resting, thinking: false, still: false, now: at(1))! == (.idleRest, false))
// When nothing may move there is no play.
var still = PetMovePicker()
still.ended(.startup, now: at(-100))
precondition(still.pick(.resting, thinking: false, still: true, now: at(0))! == (.idleRest, false))
precondition(still.pick(.resting, thinking: false, still: true, now: at(1000))! == (.idleRest, false))
""",
    )


@darwin
def test_the_app_reads_the_shipped_manifest_and_plays_it_on_layers_off_screen(
    tmp_path: Path,
) -> None:
    run_swift(
        tmp_path,
        [*POLICY, "PetMoves.swift"],
        rf"""
let folder = URL(fileURLWithPath: "{MOVES}")
let manifest = petMovesManifest(try! Data(contentsOf: folder.appendingPathComponent("manifest.json")))!
precondition(manifest.pixels == 256 && manifest.anchor == CGPoint(x: 128, y: 244))
precondition(manifest.columns == 4 && manifest.rows == 4 && manifest.states.count == 6)
precondition(manifest.states[.startup]!.once && manifest.states[.startup]!.poster == 15)
precondition(PetMove.allCases.filter {{ $0 != .startup }}.allSatisfy {{ !manifest.states[$0]!.once }})
precondition(abs(manifest.states[.idlePlay]!.length - 6) < 1e-9)
// The fades: none at first, 220 ms between poses, 160 ms within one, 120 ms into attention, 80 ms when still.
precondition(manifest.fade(from: nil, to: .startup, still: false) == 0)
precondition(manifest.fade(from: .working, to: .idleRest, still: false) == 0.22)
precondition(manifest.fade(from: .startup, to: .thinking, still: false) == 0.16)
precondition(manifest.fade(from: .idlePlay, to: .attention, still: false) == 0.12)
precondition(manifest.fade(from: .working, to: .attention, still: true) == 0.08)
// A manifest that is not one is refused.
precondition(petMovesManifest(Data("{{}}".utf8)) == nil)
var broken = try! JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("manifest.json"))) as! [String: Any]
var states = broken["states"] as! [String: [String: Any]]
states["working"]!["file"] = "../elsewhere.webp"
broken["states"] = states
precondition(petMovesManifest(try! JSONSerialization.data(withJSONObject: broken)) == nil)

// The frames: discrete key frames on contentsRect, each frame from the start of its duration.
let working = manifest.states[.working]!
let frames = petFramesAnimation(working, manifest: manifest, once: false)
precondition(frames.keyPath == "contentsRect" && frames.calculationMode == .discrete)
precondition(frames.values!.count == 16 && frames.keyTimes!.count == 17 && frames.keyTimes!.last == 1)
precondition(abs(frames.keyTimes![1].doubleValue - 1.0 / 16) < 1e-9 && frames.repeatCount == .infinity)
precondition(abs(frames.duration - 1.28) < 1e-9)
precondition(petFramesAnimation(working, manifest: manifest, once: true).repeatCount == 1)

// The cells: the sheet's first row is its top; Core Animation reads contentsRect y up on the Mac, which a
// two-by-two sheet drawn here shows (its top-left quarter red).
precondition(manifest.cell(0) == CGRect(x: 0, y: 0.75, width: 0.25, height: 0.25))
precondition(manifest.cell(6) == CGRect(x: 0.5, y: 0.5, width: 0.25, height: 0.25))
let grid = PetMovesManifest(pixels: 2, anchor: .zero, columns: 2, rows: 2, samePose: 0, poseChange: 0, attention: 0,
                            still: 0, states: [:])
let sheet = CGContext(data: nil, width: 2, height: 2, bitsPerComponent: 8, bytesPerRow: 8,
                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
sheet.setFillColor(CGColor(srgbRed: 0, green: 0, blue: 1, alpha: 1)); sheet.fill(CGRect(x: 0, y: 0, width: 2, height: 2))
sheet.setFillColor(CGColor(srgbRed: 1, green: 0, blue: 0, alpha: 1)); sheet.fill(CGRect(x: 0, y: 1, width: 1, height: 1))  // y up: top left
let probe = CALayer()
probe.frame = CGRect(x: 0, y: 0, width: 4, height: 4)
probe.contents = sheet.makeImage()
probe.contentsRect = grid.cell(0)
probe.magnificationFilter = .nearest
let out = CGContext(data: nil, width: 4, height: 4, bitsPerComponent: 8, bytesPerRow: 16,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
probe.render(in: out)
let pixel = out.data!.load(fromByteOffset: (2 * 4 + 2) * 4, as: UInt32.self)
precondition(pixel & 0xff > 200 && (pixel >> 16) & 0xff < 50, "frame 1 is the sheet's top-left cell")

// The player: the sheets decode (WebP with alpha), one layer per role, attention ready before it is asked for.
_ = NSApplication.shared
let player = PetMovePlayer(folder: folder, side: 100)!
precondition(player.layer.sublayers!.count == 2 && player.layer.bounds.size == CGSize(width: 100, height: 100))
let (outgoing, incoming) = (player.layer.sublayers![0], player.layer.sublayers![1])
var ended: [PetMove] = []
player.onEnded = {{ ended.append($0) }}
precondition(player.set(.startup))
let wake = incoming.animation(forKey: "frames") as! CAKeyframeAnimation
precondition(wake.repeatCount == 1 && wake.fillMode == .forwards && !wake.isRemovedOnCompletion)
precondition(incoming.contentsRect == manifest.cell(15))  // where it holds once it ends
precondition(incoming.animation(forKey: "fade") == nil)  // the first state does not fade in
// The same state again does not restart it.
player.set(.startup)
precondition(incoming.animation(forKey: "frames") === wake)
// Attention interrupts at once, from the frame on screen, in 120 ms.
player.set(.attention)
precondition(player.wanted == .attention)
let urgent = incoming.animation(forKey: "fade") as! CABasicAnimation
precondition(urgent.duration == 0.12 && outgoing.animation(forKey: "fade")?.duration == 0.12)
precondition(outgoing.contents != nil && incoming.contentsRect == manifest.cell(0))
precondition((incoming.animation(forKey: "frames") as! CAKeyframeAnimation).repeatCount == .infinity)
// A late end of the cut wake-up says nothing: the latest request won.
let late = wake
player.animationDidStop(late, finished: true)
precondition(ended.isEmpty)
// Between poses 220 ms; a looping state asked to play once holds and says when it ended.
player.set(.idlePlay, once: true)
precondition(incoming.animation(forKey: "fade")?.duration == 0.22)
let play = incoming.animation(forKey: "frames") as! CAKeyframeAnimation
precondition(play.repeatCount == 1 && incoming.contentsRect == manifest.cell(15))
player.animationDidStop(play, finished: false)  // taken off, not ended
precondition(ended.isEmpty)
player.animationDidStop(play, finished: true)
precondition(ended == [.idlePlay])
// Nothing may move: posters, an 80 ms fade, no frames.
player.setStill(true)
precondition(player.still && incoming.animation(forKey: "frames") == nil)
precondition(incoming.contentsRect == manifest.cell(0) && incoming.animation(forKey: "fade")?.duration == 0.08)
player.set(.working)
precondition(incoming.animation(forKey: "frames") == nil && incoming.animation(forKey: "fade")?.duration == 0.08)
player.setStill(false)
precondition(incoming.animation(forKey: "frames") != nil)
// A still of the pet's: the state's poster, held, with no frames and the fade of any other change; Reduce Motion
// coming or going leaves it as it is, and asking for the state's moves starts them.
player.set(.thinking, poster: true)
precondition(player.wanted == .thinking && player.poster && incoming.animation(forKey: "frames") == nil)
precondition(incoming.contentsRect == manifest.cell(0) && incoming.animation(forKey: "fade")?.duration == 0.22)
player.set(.startup, poster: true)
precondition(incoming.contentsRect == manifest.cell(15) && incoming.animation(forKey: "frames") == nil)
let held = incoming.animation(forKey: "fade")
player.setStill(true); player.setStill(false)
precondition(incoming.animation(forKey: "fade") === held && player.poster)
player.set(.startup, poster: true)
precondition(incoming.animation(forKey: "fade") === held)  // the same still again: nothing
player.set(.startup)
precondition(!player.poster && incoming.animation(forKey: "frames") != nil)
// Unseen, the clock stops where it is; seen again, it goes on.
precondition(player.running)
player.run(false)
precondition(!player.running && player.layer.speed == 0)
player.run(false)
player.run(true)
precondition(player.running && player.layer.speed == 1)
// A folder without the files gives no player, and no pet.
precondition(PetMovePlayer(folder: URL(fileURLWithPath: "/nonexistent"), side: 100) == nil)
// A sheet that cannot be read: its poster file stands in, whole and held; without that either, nothing changes.
let unreadable = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("moves-unreadable-\(getpid())")
try? FileManager.default.removeItem(at: unreadable)
try! FileManager.default.copyItem(at: folder, to: unreadable)
try! Data("not a picture".utf8).write(to: unreadable.appendingPathComponent("working.webp"))
let fallback = PetMovePlayer(folder: unreadable, side: 100)!
let standIn = fallback.layer.sublayers![1]
precondition(fallback.set(.working) && fallback.wanted == .working)
precondition(standIn.animation(forKey: "frames") == nil && standIn.contents != nil)
precondition(standIn.contentsRect == CGRect(x: 0, y: 0, width: 1, height: 1))
try! FileManager.default.removeItem(at: unreadable.appendingPathComponent("poster-working.webp"))
let bare = PetMovePlayer(folder: unreadable, side: 100)!
precondition(bare.set(.idleRest) && !bare.set(.working) && bare.wanted == .idleRest)
try? FileManager.default.removeItem(at: unreadable)
// The other five sheets decode off the main thread, soon after the player is made.
let early = PetMovePlayer(folder: folder, side: 100)!
precondition(early.isReady(.attention))
let deadline = Date().addingTimeInterval(10)
while !PetMove.allCases.allSatisfy({{ early.isReady($0) }}) && Date() < deadline {{
    RunLoop.main.run(until: Date().addingTimeInterval(0.05))
}}
precondition(PetMove.allCases.allSatisfy {{ early.isReady($0) }})
// The app reads the moves where the bundled gateway serves them.
let resources = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("resources-\(getpid())")
let served = resources.appendingPathComponent("runtime/bin/python/lib/python3.11/site-packages/localmelo/support/gateway/static/moves")
precondition(runtimeMovesFolder(resources: resources) == nil)
try! FileManager.default.createDirectory(at: served, withIntermediateDirectories: true)
precondition(runtimeMovesFolder(resources: resources) == nil)  // no manifest: not the moves
try! Data("{{}}".utf8).write(to: served.appendingPathComponent("manifest.json"))
precondition(runtimeMovesFolder(resources: resources)?.path == served.path)
try? FileManager.default.removeItem(at: resources)
""",
    )


def test_one_player_draws_the_whole_pet_and_the_stills_keep_their_words() -> None:
    panel = (APP / "PetPanel.swift").read_text()
    # The words are the mood's, written when the mood or the language changes; then its still, held, or its move.
    refresh = panel.split("private func refresh()")[1].split("private func applyMoves")[
        0
    ]
    assert refresh.index("view.describe(mood)") < refresh.index("showMoves()")
    assert "if !described || mood != old { view.describe(mood) }" in refresh
    assert "described = false\n        if isShown { refresh() }" in panel
    # The picker hears every mood (a still ends a rest, a play and the sitting), then the still wins.
    assert refresh.index("picker.pick(") < refresh.index("petStill(mood)")
    assert "moves.set(poster, poster: true)" in refresh
    # The moves' layer is the pet: added once, never hidden; no other picture is drawn on the panel.
    assert panel.count("view.layer?.addSublayer(moves.layer)") == 1
    assert "isHidden" not in panel and "CGImageSource" not in panel
    # Without the moves there is no pet.
    assert "guard moves != nil, place() else { return }" in panel
    # The app hands the pet, the start window and the pill one moves folder: the one the bundled gateway serves.
    main = (APP / "main.swift").read_text()
    runtime = "Bundle.main.resourceURL.flatMap(runtimeMovesFolder(resources:))"
    assert (
        f"let moves = {runtime}" in main and f"StatusView(pictures: {runtime})" in main
    )
    assert "movesFolder: moves" in main and "PillController(pictures: moves," in main
    assert 'appendingPathComponent("moves")' not in main
    # No timer drives a frame: Core Animation runs the key frames.
    moves = (APP / "PetMoves.swift").read_text()
    assert "Timer" not in moves and "CADisplayLink" not in moves
