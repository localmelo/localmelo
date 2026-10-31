"""PillPanel.swift: the pill's view drawn offscreen, light and dark, and what it does with the pointer, VoiceOver and
motion. The Swift is compiled with a test main; no window is shown (a view is drawn into a bitmap), so this is safe
beside the person's work. The pictures land in `.cache/shots/` of the checkout (the folder is ignored by git).
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "macos/LocalMeloApp"
SHOTS = ROOT / ".cache/shots"
darwin = pytest.mark.skipif(
    sys.platform != "darwin", reason="Swift AppKit host is macOS-only"
)
FILES = (
    "GatewayPolicy.swift",
    "PetPolicy.swift",
    "PetMotion.swift",
    "MotionGate.swift",
    "PetPanel.swift",
    "StatusView.swift",
    "PillPolicy.swift",
    "PillPanel.swift",
)

RENDER = r"""
import AppKit
import ImageIO

// AppKit needs an application object to draw; nothing is shown (no window is made) and it has no Dock icon.
_ = NSApplication.shared
NSApp.setActivationPolicy(.prohibited)
let pictures = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let out = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
try! FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

func face(_ portrait: PillPortrait) -> CGImage? {
    let file = pillPictureFile(portrait, scale: 2) { FileManager.default.fileExists(atPath: pictures.appendingPathComponent($0).path) }
    guard let source = CGImageSourceCreateWithURL(pictures.appendingPathComponent(file) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
}
func model(_ state: PillState, _ text: String, _ icon: PillIcon, more: Int = 0, _ portrait: PillPortrait,
           _ click: PillClick = .window) -> PillModel {
    PillModel(state: state, text: text, icon: icon, more: more, portrait: portrait, click: click)
}
let long = "Writing a very long file name that does not fit in the pill and keeps going and going.md"
let rows: [(String, PillModel, Bool)] = [
    ("Working: the newest running item", model(.working, "Reading example.com", .web, .workRead), false),
    ("Several run: the newest and how many more", model(.working, "Running the tests", .run, more: 1, .workLaptop), false),
    ("Waiting: amber rim, three pulses then still", model(.waiting, "Waiting for your approval", .wait, .waitAlert, .waiting), false),
    ("Done: 6 s, then the pet comes back", model(.done, "Done: Slides for the 3 pm meeting", .tool, .done), false),
    ("Failed: 10 s", model(.failed, "Stopped: Fix CI", .tool, .error), false),
    ("The model server is down", model(.serverDown, "Model server not reachable", .tool, .error), false),
    ("Hover: the x hides it until the next task", model(.working, "Cancelling the subscription", .web, .workRead), true),
    ("A long line is cut at 320 pt", model(.working, long, .write, .workLaptop), false),
]

func bitmap(_ size: CGSize) -> NSBitmapImageRep {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size.width * 2), pixelsHigh: Int(size.height * 2),
                               bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                               colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    rep.size = size
    return rep
}

func view(_ model: PillModel, dark: Bool, hover: Bool) -> PillView {
    let view = PillView()
    view.blur.isHidden = true  // the system's blur shows only on a screen; here the capsule's own tint is drawn
    view.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
    view.show(model, picture: face(model.portrait), fade: false)
    view.frame = NSRect(origin: .zero, size: pillPanelSize(width: view.capsuleWidth))
    if hover {
        view.mouseEntered(with: NSEvent.enterExitEvent(with: .mouseEntered, location: .zero, modifierFlags: [], timestamp: 0,
                                                       windowNumber: 0, context: nil, eventNumber: 0, trackingNumber: 0,
                                                       userData: nil)!)
    }
    view.layoutSubtreeIfNeeded()
    return view
}

/// The view drawn at 2x into a bitmap. The portrait's layers are hosted in a layer of their own, which a bitmap drawing of
/// the views leaves out, so they are drawn on top at the portrait's place.
func snapshot(_ view: PillView) -> NSBitmapImageRep {
    let rep = bitmap(view.frame.size)
    view.cacheDisplay(in: view.bounds, to: rep)
    NSGraphicsContext.saveGraphicsState()
    let context = NSGraphicsContext(bitmapImageRep: rep)!
    NSGraphicsContext.current = context
    let scale = CGFloat(rep.pixelsWide) / view.frame.width / context.cgContext.ctm.a
    context.cgContext.scaleBy(x: scale, y: scale)
    context.cgContext.translateBy(x: view.host.frame.minX, y: view.host.frame.minY)
    view.disc.render(in: context.cgContext)
    NSGraphicsContext.restoreGraphicsState()
    return rep
}

for dark in [false, true] {
    let sheet = CGSize(width: 800, height: 40 + CGFloat(rows.count) * 70)
    let canvas = bitmap(sheet)
    let desktop = dark ? NSColor(srgbRed: 0.18, green: 0.23, blue: 0.17, alpha: 1)
                       : NSColor(srgbRed: 0.62, green: 0.72, blue: 0.58, alpha: 1)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: canvas)
    desktop.setFill()
    NSRect(origin: .zero, size: sheet).fill()
    for (index, row) in rows.enumerated() {
        let shown = view(row.1, dark: dark, hover: row.2)
        let size = shown.frame.size
        let rep = snapshot(shown)
        let top = sheet.height - 20 - CGFloat(index + 1) * 70 + 6
        rep.draw(in: NSRect(x: 40, y: top, width: size.width, height: size.height), from: .zero, operation: .sourceOver,
                 fraction: 1, respectFlipped: true, hints: nil)
        let caption = NSAttributedString(string: row.0, attributes: [
            .font: NSFont.systemFont(ofSize: 12), .foregroundColor: dark ? NSColor(white: 0.93, alpha: 1) : NSColor(white: 0.1, alpha: 1)])
        caption.draw(at: NSPoint(x: 40 + 420 + 30, y: top + size.height / 2 - 7))
    }
    NSGraphicsContext.restoreGraphicsState()
    var colours = Set<String>()
    for y in stride(from: 0, to: canvas.pixelsHigh, by: 7) {
        for x in stride(from: 0, to: canvas.pixelsWide, by: 11) {
            if let c = canvas.colorAt(x: x, y: y)?.usingColorSpace(.sRGB) { colours.insert("\(Int(c.redComponent * 255)),\(Int(c.greenComponent * 255)),\(Int(c.blueComponent * 255))") }
        }
    }
    precondition(colours.count > 12, "the sheet is flat: \(colours.count) colours")  // a pill, its ring, a face and text
    // The pill's own pixel: the capsule is drawn light on the light desktop and dark on the dark one.
    let inside = canvas.colorAt(x: 2 * 120, y: Int(canvas.size.height * 2) - 2 * 38)!.usingColorSpace(.sRGB)!
    print(dark ? "dark" : "light", "capsule brightness", inside.brightnessComponent)
    if dark { precondition(inside.brightnessComponent < 0.5) } else { precondition(inside.brightnessComponent > 0.8) }
    let data = canvas.representation(using: .png, properties: [:])!
    try! data.write(to: out.appendingPathComponent(dark ? "pill-dark.png" : "pill-light.png"))
}
// One pill alone, 2x, for each state (the test reads their size).
for (index, row) in rows.prefix(6).enumerated() {
    let shown = view(row.1, dark: false, hover: false)
    let rep = snapshot(shown)
    try! rep.representation(using: .png, properties: [:])!.write(to: out.appendingPathComponent("pill-\(index).png"))
}
print("rendered")
"""


def compile_swift(tmp_path: Path, source: str) -> Path:
    main = tmp_path / "main.swift"
    main.write_text(source)
    binary = tmp_path / "test-pill-panel"
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            *[str(APP / name) for name in FILES],
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
    return binary


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


@darwin
def test_the_pill_is_drawn_in_light_and_dark_for_every_state(tmp_path: Path) -> None:
    binary = compile_swift(tmp_path, RENDER)
    shots = SHOTS / "pill"
    subprocess.run(
        [str(binary), str(ROOT / "macos/art/akita"), str(shots)],
        check=True,
        timeout=60,
    )
    light, dark = shots / "pill-light.png", shots / "pill-dark.png"
    # The sheets hold the eight rows at 2x (the Swift checked they are not flat and that the two appearances differ).
    assert png_size(light) == png_size(dark) == (1600, 2 * (40 + 8 * 70))
    # One pill alone per state, at 2x: the 48 pt capsule and its 5 pt margin high, wider for a longer line.
    sizes = [png_size(shots / f"pill-{index}.png") for index in range(6)]
    assert {height for _, height in sizes} == {116}
    assert (
        sizes[3][0] > sizes[2][0] > sizes[0][0]
    )  # the line "Done: Slides for the 3 pm meeting" is the longest
    assert (
        sizes[1][0] > sizes[0][0] - 60
    )  # the "+1" chip adds room to a line that is shorter
    print("shots:", light, dark)


@darwin
def test_the_view_reads_as_one_button_and_its_parts_follow_the_state(
    tmp_path: Path,
) -> None:
    binary = compile_swift(
        tmp_path,
        r"""
import AppKit
_ = NSApplication.shared
NSApp.setActivationPolicy(.prohibited)
func model(_ state: PillState, _ text: String, _ icon: PillIcon, more: Int = 0, _ portrait: PillPortrait,
           _ click: PillClick = .window) -> PillModel {
    PillModel(state: state, text: text, icon: icon, more: more, portrait: portrait, click: click)
}
let reading = model(.working, "Reading example.com", .web, .workRead, .route("#/s/c1"))
let view = PillView()
view.show(reading, picture: nil, fade: false)
view.frame = NSRect(origin: .zero, size: pillPanelSize(width: view.capsuleWidth))
view.layoutSubtreeIfNeeded()

// One accessibility element: the pill, a button, with the one sentence; its parts are not read on their own.
precondition(view.isAccessibilityElement() && view.accessibilityRole() == .button)
precondition(view.accessibilityLabel() == "Melo is reading example.com. Opens the chat.")
precondition(view.subviews.allSatisfy { !$0.isAccessibilityElement() })
precondition(view.accessibilityCustomActions()?.map { $0.name } == ["Hide until the next task"])
var clicks = 0, hides = 0
view.onClick = { clicks += 1 }
view.onDismiss = { hides += 1 }
precondition(view.accessibilityPerformPress() && clicks == 1)
view.accessibilityCustomActions()?.first?.handler?()
precondition(hides == 1)
// The words and the width: the line is 14 pt, one line, and the capsule is as wide as the policy says.
precondition(view.line.stringValue == "Reading example.com" && view.line.font?.pointSize == 14)
precondition(view.line.lineBreakMode == .byTruncatingTail && view.mark.stringValue == "\u{2295}")
precondition(view.capsule.height == 48 && view.capsule.width == view.capsuleWidth)
precondition(view.line.frame.width <= 320 && view.line.frame.minX == view.capsule.minX + 74)
precondition(view.more.isHidden && view.toolTip == "Reading example.com")
// A longer line than 320 pt is cut: the capsule stops growing, the field is 320 wide and ends with an ellipsis.
let long = model(.working, String(repeating: "A long line of words ", count: 12), .write, .workLaptop)
view.show(long, picture: nil, fade: false)
view.frame = NSRect(origin: .zero, size: pillPanelSize(width: view.capsuleWidth))
view.layoutSubtreeIfNeeded()
precondition(view.capsuleWidth == 74 + 320 + 18 && view.line.frame.width == 320)
// The "+1" chip leaves its room and shows its count.
view.show(model(.working, "Running the tests", .run, more: 2, .workLaptop), picture: nil, fade: false)
view.layoutSubtreeIfNeeded()
precondition(!view.more.isHidden && view.more.stringValue == "+2")
precondition(view.more.frame.maxX == view.capsule.maxX - 14 && view.line.frame.maxX <= view.more.frame.minX)

// The pointer: the "x" shows while it is over the pill and goes when it leaves; the margin is not the pill.
view.frame = NSRect(origin: .zero, size: pillPanelSize(width: view.capsuleWidth))
view.layoutSubtreeIfNeeded()
func crossing(_ type: NSEvent.EventType) -> NSEvent {
    NSEvent.enterExitEvent(with: type, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0, context: nil,
                           eventNumber: 0, trackingNumber: 0, userData: nil)!
}
let middle = NSPoint(x: view.bounds.midX, y: view.bounds.midY)
precondition(view.close.isHidden && view.hitTest(middle) === view)
precondition(view.hitTest(NSPoint(x: 1, y: view.bounds.midY)) == nil)  // the clear strip: no press
view.mouseEntered(with: crossing(.mouseEntered))
precondition(!view.close.isHidden)
let corner = NSPoint(x: view.close.frame.midX, y: view.close.frame.midY)
precondition(view.hitTest(corner) === view.close)
var pressed = 0
view.close.onPress = { pressed += 1 }
view.close.mouseDown(with: NSEvent.mouseEvent(with: .leftMouseDown, location: .zero, modifierFlags: [], timestamp: 0,
                                              windowNumber: 0, context: nil, eventNumber: 0, clickCount: 1, pressure: 1)!)
precondition(pressed == 1)
view.mouseExited(with: crossing(.mouseExited))
precondition(view.close.isHidden && view.hitTest(corner) !== view.close)

// Motion: it breathes only while it works and only if it may move; a wait pulses its ring three times, once.
view.play(.working, entering: true, animate: true)
let breath = view.disc.animation(forKey: "breath") as? CABasicAnimation
precondition(breath?.keyPath == "transform.scale" && breath?.repeatCount == .infinity && breath?.autoreverses == true)
precondition(view.disc.animation(forKey: "pulse") == nil)
view.play(.working, entering: false, animate: false)
precondition(view.disc.animation(forKey: "breath") == nil)  // Reduce Motion, Low Power, heat, hidden: still
view.play(.waiting, entering: true, animate: true)
let pulse = view.disc.animation(forKey: "pulse") as? CABasicAnimation
precondition(pulse?.keyPath == "borderWidth" && pulse?.repeatCount == 3 && view.disc.animation(forKey: "breath") == nil)
view.play(.waiting, entering: false, animate: true)
precondition(view.disc.animation(forKey: "pulse") != nil)  // not cut short by the next reply
view.play(.waiting, entering: false, animate: false)
precondition(view.disc.animation(forKey: "pulse") == nil)
view.play(.working, entering: true, animate: true)
view.play(.done, entering: true, animate: true)
precondition(view.disc.animation(forKey: "breath") == nil && view.disc.animation(forKey: "pulse") == nil)

// The "On the desktop" choice: three entries in each menu that offers it, one check mark across all of them.
let desktop = DesktopMenu()
var chosen: [DesktopMode] = []
desktop.choose = { chosen.append($0) }
let first = desktop.makeItem(), second = desktop.makeItem()
precondition(first.title == "On the desktop" && first.submenu?.items.map { $0.title } == ["Melo and the activity pill", "Melo only", "Nothing"])
func marks(_ item: NSMenuItem) -> [Bool] { item.submenu!.items.map { $0.state == .on } }
precondition(marks(first) == [true, false, false] && marks(second) == [true, false, false])
desktop.show(.pet)
precondition(marks(first) == [false, true, false] && marks(second) == [false, true, false])
let nothing = second.submenu!.items[2]
_ = NSApp.sendAction(nothing.action!, to: nothing.target, from: nothing)
precondition(chosen == [.nothing])
let late = desktop.makeItem()  // a menu made later starts at the current choice
precondition(marks(late) == [false, true, false])

// The pet's menu has the choice, and a hidden pet is "Nothing".
let pet = PetController(settingsURL: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("pill-test-\(getpid())/pet.json"),
                        pictures: nil)
precondition(pet.view.menu?.items.map { $0.title } == ["Open localmelo", "Hide Melo", "On the desktop", "", "Quit localmelo"])
var told: [DesktopMode] = [], hidden: [Bool] = []
pet.desktopChanged = { told.append($0) }
pet.hiddenChanged = { hidden.append($0) }
pet.setDesktop(.pet)
pet.setDesktop(.nothing)
pet.setHidden(false)  // Show Melo: what was there before Nothing
pet.setDesktop(.pill)
precondition(told == [.pet, .nothing, .pet, .pill] && hidden == [true, false])
precondition(pet.settings.mode == .pill && !pet.hiddenByPerson)
pet.setHidden(true)
precondition(pet.settings.mode == .nothing && pet.hiddenByPerson && hidden == [true, false, true])
pet.setHidden(true)  // no change: no news
precondition(told.count == 5 && hidden.count == 3)
try? FileManager.default.removeItem(at: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("pill-test-\(getpid())"))
print("ok")
""",
    )
    subprocess.run([str(binary)], check=True, timeout=30)


def test_the_app_wires_the_pill_beside_the_pet_and_never_launches_it_here() -> None:
    main = (APP / "main.swift").read_text()
    # Each reply, each window change, the stop and the restart reach the pill after the pet.
    assert main.count("pill.update(health") + main.count("pill?.update(health") == 3
    # The pet hears first when the window comes back and last when it goes away, so neither flashes.
    assert "(shown ? hear : hear.reversed()).forEach" in main
    assert main.index("{ self.pet?.windowShown($0) }") < main.index(
        "{ self.pill?.windowShown($0) }"
    )
    assert "pill.openRoute" in main and "open(route: pendingRoute)" in main
    assert "menuBar.offerDesktop(pet.desktop.makeItem())" in main
    # The pill is a panel that is never key and does not hide with the app, as the pet's is.
    panel = (APP / "PillPanel.swift").read_text()
    assert "override var canBecomeKey: Bool { false }" in panel
    assert "panel.canHide = false" in panel and ".canJoinAllSpaces" in panel
    assert panel.count("Timer.scheduledTimer") == 1  # the one the policy asks for
    assert "announcementRequested" not in panel  # the menu-bar item announces a wait
