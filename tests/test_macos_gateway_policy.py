from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from localmelo.tests import macos_swift

ROOT = Path(__file__).resolve().parents[1]


def run_policy(source: str, *more: str) -> None:
    """Run `source` as the test main of the policy file (and any other of the app's files named)."""
    macos_swift.run_swift(
        ["GatewayPolicy.swift", "L10n.swift", *more], "import Foundation\n" + source
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_gateway_instance_and_listener_decisions() -> None:
    run_policy("""let expected = "0123456789abcdef0123456789abcdef"
precondition(acceptsInstance(expected, expected: expected, childRunning: true))
precondition(!acceptsInstance(nil, expected: expected, childRunning: true))
precondition(!acceptsInstance("other", expected: expected, childRunning: true))
precondition(!acceptsInstance(expected, expected: expected, childRunning: false))
// A gateway the terminal started has no instance: its pages send no header.
precondition(acceptsInstance(nil, expected: "", childRunning: true))
precondition(!acceptsInstance(expected, expected: "", childRunning: true))
// The endpoint file names a running gateway; the app attaches to it rather than start a second.
let gatewayId = "fedcba9876543210fedcba9876543210"
let file = Data(#"{"pid": 4242, "host": "127.0.0.1", "port": 8123, "instance": "", "id": "fedcba9876543210fedcba9876543210", "started": 1.5}"#.utf8)
let endpoint = gatewayEndpoint(file)
precondition(endpoint == GatewayEndpoint(pid: 4242, port: 8123, instance: "", id: gatewayId))
precondition(gatewayEndpoint(nil) == nil)
precondition(gatewayEndpoint(Data("not json".utf8)) == nil)
precondition(gatewayEndpoint(Data(#"{"pid": 4242, "host": "0.0.0.0", "port": 8123, "id": "x"}"#.utf8)) == nil)
precondition(gatewayEndpoint(Data(#"{"pid": 0, "host": "127.0.0.1", "port": 8123, "id": "x"}"#.utf8)) == nil)
precondition(gatewayEndpoint(Data(#"{"pid": 4242, "host": "127.0.0.1", "port": 70000, "id": "x"}"#.utf8)) == nil)
precondition(gatewayEndpoint(Data(#"{"pid": 4242, "host": "127.0.0.1", "port": 8123, "id": ""}"#.utf8)) == nil)
precondition(gatewayStart(endpoint, alive: { $0 == 4242 }) == .attach(endpoint!))
precondition(gatewayStart(endpoint, alive: { _ in false }) == .start)
precondition(gatewayStart(nil, alive: { _ in true }) == .start)
// The port must answer with the file's id; another process there, or a newer gateway, is not it.
precondition(attachConfirmed(status: 200, gatewayHeader: gatewayId, endpoint: endpoint!))
precondition(!attachConfirmed(status: 200, gatewayHeader: nil, endpoint: endpoint!))
precondition(!attachConfirmed(status: 200, gatewayHeader: "other", endpoint: endpoint!))
precondition(!attachConfirmed(status: nil, gatewayHeader: gatewayId, endpoint: endpoint!))
// Quit leaves an attached gateway running, and its menu item says so.
precondition(quitTitle(attached: false) == "Quit localmelo")
precondition(quitTitle(attached: true) == "Quit localmelo (the gateway keeps running)")
precondition(attachedStoppedMessage.contains("Try Again starts a new one"))
precondition(!AttachedGateway(pid: 4242).owned)
precondition(ownsListener([42], child: 42))
precondition(!ownsListener([42, 43], child: 42))
precondition(!ownsListener(nil, child: 42))
precondition(isGatewayURL(URL(string: "http://127.0.0.1:9999/")!, port: 9999))
precondition(!isGatewayURL(URL(string: "http://localhost:9999/")!, port: 9999))
precondition(isGatewayPage(URL(string: "http://127.0.0.1:9999/#/workers")!, port: 9999))
precondition(!isGatewayPage(URL(string: "http://127.0.0.1:9999/api/health")!, port: 9999))
// A file dropped on the window that the page did not take is an attempt to open it: the file, a blob or a
// data address, a link to another port, an address with a user name, and an upload are never the gateway page.
for dropped in ["file:///Users/someone/Pictures/shot.png", "file:///tmp/notes.txt", "blob:http://127.0.0.1:9999/6b1c",
                "data:text/html,hello", "about:blank", "http://127.0.0.1:9998/", "http://user@127.0.0.1:9999/"] {
    precondition(!isGatewayURL(URL(string: dropped)!, port: 9999), dropped)
    precondition(!isGatewayPage(URL(string: dropped)!, port: 9999), dropped)
}
precondition(!isGatewayPage(URL(string: "http://127.0.0.1:9999/api/uploads")!, port: 9999))
let owned = GatewayHealth(store: "unavailable", error: "owned", reason: "owned", running: 0)
let failed = GatewayHealth(store: "unavailable", error: "corrupt", reason: "open-error", running: 0)
precondition(storeFailureMessage(owned).contains("Stop the other gateway"))
precondition(!storeFailureMessage(failed).contains("other gateway"))
// The window says it plainly, with what its buttons do; the signal or the status goes to the log.
let said = gatewayExitMessage()
precondition(said == "Its background service quit unexpectedly. Try Again starts it again; Open Log shows what happened.")
precondition(!said.contains("signal") && !said.contains("status") && !said.contains("gateway"))
precondition(gatewayExitMessage(detail: " Port 9 is held by process 4.").contains("unexpectedly. Port 9 is held by process 4. Try Again"))
precondition(gatewayExitLogLine(status: 9, signalled: true) == "The gateway was terminated by signal 9.")
precondition(gatewayExitLogLine(status: 2, signalled: false) == "The gateway exited (status 2).")
""")


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_running_count_title_and_quit_text() -> None:
    run_policy("""func health(_ running: String) -> Data {
    Data((#"{"api":2,"status":{"running":"# + running + #","tasks":{}}}"#).utf8)
}
let none = runningCounts(health(#"{"chats":0,"tasks":0,"runs":0,"terminal":0}"#))
let busy = runningCounts(health(#"{"chats":1,"tasks":2,"runs":1,"terminal":3}"#))
let chat = runningCounts(health(#"{"chats":1,"tasks":0,"runs":0,"terminal":0}"#))
let runsOnly = runningCounts(health(#"{"chats":0,"tasks":0,"runs":1,"terminal":0}"#))
precondition(none?.total == 0 && busy?.total == 7)
precondition(runningCounts(Data(#"{"api":2}"#.utf8)) == nil)
precondition(runningCounts(health(#"{"chats":-1,"tasks":0,"runs":0,"terminal":0}"#)) == nil)
precondition(runningCounts(health(#"{"chats":"1","tasks":0,"runs":0,"terminal":0}"#)) == nil)
precondition(runningCounts(Data("not json".utf8)) == nil)
precondition(menuTitle(.idle) == "" && menuTitle(.unknown) == "" && menuTitle(.running(7)) == "7")
// What quitting does, in plain words: chats stop, tasks continue when it opens again; nothing says "gateway".
let chats = runningCounts(health(#"{"chats":3,"tasks":0,"runs":0,"terminal":0}"#))
let tasks = runningCounts(health(#"{"chats":0,"tasks":2,"runs":0,"terminal":0}"#))
precondition(quitMessage(busy, statusKnown: true)
    == "Quitting stops 1 chat. 2 tasks will continue when it opens again. Not affected: 1 run and 3 terminal chats.")
precondition(quitMessage(tasks, statusKnown: true) == "2 tasks will continue when it opens again.")
precondition(quitMessage(chat, statusKnown: true) == "Quitting stops 1 chat.")
precondition(quitMessage(chats, statusKnown: true) == "Quitting stops 3 chats.")
precondition(quitMessage(runsOnly, statusKnown: true)
    == "Quitting stops anything that is running. Not affected: 1 run.")
precondition(quitMessage(none, statusKnown: true) == "Quitting stops anything that is running.")
precondition(quitMessage(nil, statusKnown: true) == "Quitting stops anything that is running.")
precondition(quitMessage(nil, statusKnown: false) == "Could not check what is running. Quitting stops anything that is running.")
for counts in [none, busy, chat, chats, tasks, runsOnly, nil] {
    precondition(!quitMessage(counts, statusKnown: false).lowercased().contains("gateway"))
}
""")


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_quitting_asks_only_when_something_would_stop() -> None:
    run_policy(
        r"""
func health(_ chats: Int, _ tasks: Int, _ runs: Int, waiting: String = "[]") -> Data {
    let running = #"{"chats":\#(chats),"tasks":\#(tasks),"runs":\#(runs),"terminal":1}"#
    return Data(#"{"api":2,"status":{"running":\#(running),"waiting":\#(waiting)}}"#.utf8)
}
precondition(!quitNeedsConfirmation(health(0, 0, 0)) && !quitNeedsConfirmation(health(0, 0, 2)))
precondition(quitNeedsConfirmation(health(1, 0, 0)) && quitNeedsConfirmation(health(0, 1, 0)))
precondition(quitNeedsConfirmation(health(0, 0, 0, waiting: #"[{"kind":"task","id":"t1"}]"#)))
precondition(quitNeedsConfirmation(nil) && quitNeedsConfirmation(Data(#"{"api":2}"#.utf8)))
""",
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_gateway_port_comes_from_the_startup_line() -> None:
    run_policy(
        r"""
func port(_ line: String) -> Int? { gatewayPort(fromLine: line) }
precondition(port("  Gateway: http://127.0.0.1:51234") == 51234)
precondition(port("Gateway: http://[::1]:51234\n") == 51234)
precondition(port("  Gateway: http://[fe80::1]:65535") == 65535)
for other in ["", "garbage", "Starting localmelo gateway ...", "  Web UI:  http://127.0.0.1:51234/",
              "  Sign in: run melo --gateway-login --port 51234 in another terminal",
              "  Gateway: http://127.0.0.1:", "  Gateway: http://127.0.0.1:0",
              "  Gateway: http://127.0.0.1:65536", "  Gateway: http://127.0.0.1:12ab",
              "  Gateway: http://[::1:51234", "  Gateway: http://:51234",
              "  Gateway: https://127.0.0.1:51234", "log: Gateway: http://127.0.0.1:51234"] {
    precondition(port(other) == nil, other)
}

// The line can arrive in pieces, after other output, and is reported once.
var output = StartupOutput()
precondition(output.append(Data("\n  Starting localmelo gateway ...\n  Chat: mlc\n".utf8)) == nil)
precondition(output.append(Data("  Gateway: http://127.0.".utf8)) == nil)
precondition(output.append(Data("0.1:51234\n  Web UI:  http://127.0.0.1:51234/\n".utf8)) == 51234)
precondition(output.append(Data("  Gateway: http://127.0.0.1:1\n".utf8)) == nil)

// No startup line: nothing is reported, however much other output comes.
var silent = StartupOutput()
precondition(silent.append(Data("warning: something else\r\n".utf8)) == nil)
precondition(silent.append(Data(repeating: 65, count: 100_000)) == nil)
precondition(silent.append(Data("\n  Gateway: http://127.0.0.1:777\n".utf8)) == 777)
""",
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_window_texts_and_actions() -> None:
    run_policy(
        r"""
// The step line follows what the app knows: no port yet, a port, a signed-in console.
precondition(startStepText(portKnown: false, ready: false) == "Starting the local server")
precondition(startStepText(portKnown: true, ready: false) == "Signing in")
precondition(startStepText(portKnown: true, ready: true) == "Opening your workspace")

// A failure keeps the reason it is given; the title says whether the console had loaded.
let reason = gatewayExitMessage()
let early = failureNotice(reason: reason, ready: false)
let late = failureNotice(reason: reason, ready: true)
precondition(early.title == "localmelo could not start" && late.title == "localmelo stopped")
precondition(early.reason == reason && late.reason == reason)
// Every failure offers the same three actions, in the order of the buttons.
precondition(early.actions.map { $0.title } == ["Try Again", "Open Log", "Quit"])
precondition(late.actions == [.tryAgain, .openLog, .quit])
""",
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_sign_in_again_is_offered_after_failed_sign_ins_a_locked_page_or_developer_mode() -> (
    None
):
    run_policy(
        r"""
let idle = PageState()
precondition(!offersSignIn(failures: 0, page: idle), "a working app does not offer it")
precondition(!offersSignIn(failures: signInFailureLimit - 1, page: idle), "a few retries while the gateway starts are not a failure")
precondition(offersSignIn(failures: signInFailureLimit, page: idle))
precondition(offersSignIn(failures: 0, page: PageState(developer: true, locked: false)), "a developer always has it")
precondition(offersSignIn(failures: 0, page: PageState(developer: false, locked: true)), "a locked console needs it")

// The page says its mode and whether it is locked; anything else reads as the default.
precondition(pageState(#"{"mode":"developer","locked":false}"#) == PageState(developer: true, locked: false))
precondition(pageState(#"{"mode":"default","locked":true}"#) == PageState(developer: false, locked: true))
precondition(pageState(#"{"mode":"","locked":false}"#) == idle)
for damaged in [nil, "", "not json", #"{"mode":7,"locked":"yes"}"#, "[]"] as [String?] {
    precondition(pageState(damaged) == idle, damaged ?? "nil")
}
""",
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_menu_bar_state_words_and_route_come_from_health() -> None:
    run_policy(
        r"""
func health(_ running: String, waiting: String = "[]") -> Data {
    Data((#"{"api":2,"status":{"running":"# + running + #","tasks":{},"items":[],"waiting":"# + waiting + #"}}"#).utf8)
}
let none = #"{"chats":0,"tasks":0,"runs":0,"terminal":0}"#
let busy = #"{"chats":1,"tasks":2,"runs":1,"terminal":3}"#
let chat = #"[{"kind":"chat","id":"3f9c0d2a"}]"#
let both = #"[{"kind":"task","id":"t-1"},{"kind":"chat","id":"3f9c0d2a"}]"#

// State: idle, running with the console chip's count, or waiting; a held task is not running.
precondition(menuBarState(health(none)) == .idle)
precondition(menuBarState(health(busy)) == .running(7))
precondition(menuBarState(health(busy, waiting: chat)) == .waiting(running: 7))
precondition(menuBarState(health(none, waiting: both)) == .waiting(running: 0))
precondition(menuBarState(Data(#"{"api":2}"#.utf8)) == .unknown)
precondition(menuBarState(Data("not json".utf8)) == .unknown)
// A list that is missing or damaged never makes the item claim a wait.
precondition(menuBarState(Data((#"{"status":{"running":"# + none + #"}}"#).utf8)) == .idle)
precondition(menuBarState(health(none, waiting: #"{"kind":"chat"}"#)) == .idle)

// The words: the count beside the mark, VoiceOver's label and the menu's first line.
let states: [MenuBarState] = [.unknown, .idle, .running(1), .running(3), .waiting(running: 0), .waiting(running: 2)]
precondition(states.map(menuTitle) == ["", "", "1", "3", "", "2"])
precondition(states.map(menuBarLabel) == [
    "localmelo", "localmelo, nothing running", "localmelo, 1 running", "localmelo, 3 running",
    "localmelo, waiting for your approval", "localmelo, waiting for your approval"])
precondition(states.map(menuStatusLine) == [nil, "Nothing running", "1 running", "3 running", "Nothing running", "2 running"])

// The route opens the first chat or task that waits.
precondition(waitingRoute(health(busy, waiting: chat)) == "#/s/3f9c0d2a")
precondition(waitingRoute(health(busy, waiting: both)) == "#/tasks/t-1")
precondition(waitingRoute(health(busy)) == nil && waitingRoute(Data("{}".utf8)) == nil)
// An id or kind the console would not open is left out, so a route is safe to hand to the page.
let long = String(repeating: "a", count: 129)
let odd = #"[{"kind":"chat","id":"a b"},{"kind":"chat","id":"../x"},{"kind":"chat","id":""},"#
    + #"{"kind":"chat","id":""# + long + #""},{"kind":"run","id":"r1"},{"kind":"task"},{"kind":"chat","id":"ok.1:2-3_x"}]"#
precondition(waitingItems(health(none, waiting: odd)) == [WaitingItem(kind: "chat", id: "ok.1:2-3_x")])
""",
    )


def contrast(one: str, other: str) -> float:
    """The WCAG contrast ratio of two #rrggbb colours, given without the #."""

    def luminance(color: str) -> float:
        light = [int(color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        r, g, b = (
            c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in light
        )
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    high, low = sorted((luminance(one), luminance(other)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_the_menu_bar_dot_has_three_to_one_against_its_ring_and_either_bar() -> None:
    # The dot is not a template, so its colours are ours: light, then dark, in the palette.
    palette = (ROOT / "macos/LocalMeloApp/StatusView.swift").read_text()
    dot, ring = (
        re.findall(rf"static let {name} = pair\(0x(\w{{6}}), 0x(\w{{6}})\)", palette)[0]
        for name in ("badge", "badgeRing")
    )
    bars = (
        ("ffffff", "ececee"),
        ("000000", "2c2c2e"),
    )  # the ends of a light, a dark bar
    for appearance, ends in enumerate(bars):
        for behind in (ring[appearance], *ends):
            assert contrast(dot[appearance], behind) >= 3, (appearance, behind)


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_the_quit_bridge_takes_one_message_from_the_gateway_page() -> None:
    run_policy(
        r"""
// The console's /exit asks the app to quit with exactly this message.
precondition(quitRequest(["action": "quit"]))
let refused: [Any] = [["action": "quit", "force": true], ["action": "exit"], ["action": 1], [String: Any](), "quit", ["quit"], 7]
precondition(refused.allSatisfy { !quitRequest($0) })
precondition(acceptsBridgeMessage(scheme: "http", host: "127.0.0.1", port: 9, isMainFrame: true, gatewayPort: 9))
precondition(!acceptsBridgeMessage(scheme: "http", host: "127.0.0.1", port: 10, isMainFrame: true, gatewayPort: 9))
""",
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_the_window_has_no_title_band_and_its_strip_moves_it_from_the_gateway_page() -> (
    None
):
    run_policy(
        r"""
import AppKit
_ = NSApplication.shared
// The window keeps its title for the Window menu and VoiceOver, but shows no band: the page reaches the top edge.
let window = consoleWindow()
precondition(window.styleMask.isSuperset(of: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView]))
precondition(window.titleVisibility == .hidden && window.titlebarAppearsTransparent && window.title == "localmelo")
precondition(window.contentView!.frame.height == window.frame.height)
// The strip's size comes from the system's title bar, not from a number in the page.
let bar = WindowBar(titleBarOf: window)
precondition(bar.height > 0 && bar.buttons > 0 && bar.buttons < 200, "\(bar)")
let script = windowBarScript(WindowBar(height: 31.5, buttons: 69))
precondition(script.contains("'--window-bar', '32px'") && script.contains("'--window-buttons', '69px'"), script)
precondition(windowBarScript(WindowBar()).contains("'--window-bar', '0px'"))
precondition(windowBarScript(WindowBar(height: .nan, buttons: -4)).contains("'--window-buttons', '0px'"))
// The page asks with exactly these messages.
precondition(windowRequest(["action": "drag"]) == .drag && windowRequest(["action": "double-click"]) == .doubleClick)
let refused: [Any] = [["action": "drag", "x": 1], ["action": "zoom"], ["action": 1], [String: Any](), "drag", ["drag"], 7]
precondition(refused.allSatisfy { windowRequest($0) == nil })
// Only the gateway's own page is heard: its main frame, not a frame inside it.
precondition(acceptsBridgeMessage(scheme: "http", host: "127.0.0.1", port: 9, isMainFrame: true, gatewayPort: 9))
precondition(!acceptsBridgeMessage(scheme: "http", host: "127.0.0.1", port: 9, isMainFrame: false, gatewayPort: 9))
// A double click does what the system setting says.
precondition(doubleClickAction(setting: nil, minimizes: false) == .zoom)
precondition(doubleClickAction(setting: nil, minimizes: true) == .minimize)
precondition(doubleClickAction(setting: "Maximize", minimizes: true) == .zoom)
precondition(doubleClickAction(setting: "Fill", minimizes: false) == .zoom)
precondition(doubleClickAction(setting: "Minimize", minimizes: false) == .minimize)
precondition(doubleClickAction(setting: "None", minimizes: true) == .none)
""",
        "ConsoleWindow.swift",
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_the_browser_bridge_takes_one_message_from_the_gateway_page_and_the_pop_out_stays_on_the_gateway() -> (
    None
):
    run_policy(
        r"""
// The Browser panel asks for its own window with exactly this message.
func source(_ session: String) -> BrowserRequest? { BrowserSource(session: session).map { .popOut($0) } }
precondition(browserRequest(["action": "popOut", "session": "abc"]) == source("abc"))
precondition(browserRequest(["action": "popOut", "session": "a.b:c-d_9"]) == source("a.b:c-d_9"))
precondition(browserRequest(["action": "popOut", "session": String(repeating: "a", count: 128)]) != nil)
// A running task's browser names its exact source with the same fields the gateway's routes take: all three, or none.
let task = BrowserSource(session: "t9", owner: "c1", gen: "a1b2c3d4", run: "run-1")!
precondition(browserRequest(["action": "popOut", "session": "t9", "owner": "c1", "gen": "a1b2c3d4", "run": "run-1"]) == .popOut(task))
let refused: [Any] = [
    ["action": "close", "session": "abc"], ["action": 1, "session": "abc"], ["session": "abc"], ["action": "popOut"],
    ["action": "popOut", "session": "abc", "force": true],
    ["action": "popOut", "session": "a/b"], ["action": "popOut", "session": "a#b"], ["action": "popOut", "session": "a b"],
    ["action": "popOut", "session": "../x"], ["action": "popOut", "session": "a\u{0}b"], ["action": "popOut", "session": ""],
    ["action": "popOut", "session": String(repeating: "a", count: 129)],
    ["action": "popOut", "session": 7], ["action": "popOut", "session": ["abc"]], ["action": "popOut", "session": NSNull()],
    [String: Any](), "popOut", ["popOut", "abc"], 7,
    ["action": "popOut", "session": "t9", "owner": "c1"], ["action": "popOut", "session": "t9", "owner": "c1", "gen": "a1b2c3d4"],
    ["action": "popOut", "session": "t9", "owner": "c/1", "gen": "a1b2c3d4", "run": "r"],
    ["action": "popOut", "session": "t9", "owner": "c1", "gen": "A1B2C3D4", "run": "r"],
    ["action": "popOut", "session": "t9", "owner": "c1", "gen": "a1b2c3d4", "run": "a b"],
    ["action": "popOut", "session": "t9", "owner": 7, "gen": "a1b2c3d4", "run": "r"],
    ["action": "popOut", "session": "t9", "owner": "c1", "gen": "a1b2c3d4", "run": "r", "url": "http://x"],
]
precondition(refused.allSatisfy { browserRequest($0) == nil })
// Only the gateway's own page is heard, as for the other bridges.
precondition(acceptsBridgeMessage(scheme: "http", host: "127.0.0.1", port: 9, isMainFrame: true, gatewayPort: 9))
precondition(!acceptsBridgeMessage(scheme: "http", host: "127.0.0.1", port: 9, isMainFrame: false, gatewayPort: 9))
precondition(!acceptsBridgeMessage(scheme: "http", host: "127.0.0.1", port: 10, isMainFrame: true, gatewayPort: 9))

// The window shows the source's live page; a source the console would not open, or no port, gives none.
func live(_ session: String, _ port: Int = 9) -> URL? { BrowserSource(session: session).flatMap { browserLiveURL($0, port: port) } }
precondition(live("abc")?.absoluteString == "http://127.0.0.1:9/#/live/abc")
precondition(live("a.b:c-d_9", 51234)?.absoluteString == "http://127.0.0.1:51234/#/live/a.b:c-d_9")
for session in ["", "a/b", "a#b", "a b", "a\u{0}b", "a%2Fb", String(repeating: "a", count: 129)] {
    precondition(BrowserSource(session: session) == nil, session)
}
let abc = BrowserSource(session: "abc")!
precondition(browserLiveURL(abc, port: 0) == nil && browserLiveURL(abc, port: -1) == nil)
precondition(browserLiveURL(abc, port: 65536) == nil)
// A task's window names the run it shows: the page reads `gen`, `run` and `owner` from the address.
precondition(browserLiveURL(task, port: 9)?.absoluteString == "http://127.0.0.1:9/#/live/t9?gen=a1b2c3d4&run=run-1&owner=c1")
// That address is a gateway page, so the pop-out's own navigation rule lets it in.
precondition(isGatewayPage(live("abc")!, port: 9) && isGatewayPage(browserLiveURL(task, port: 9)!, port: 9))

// The pop-out's main frame goes to the gateway's page alone, its subframes to the gateway; nothing else.
func allows(_ text: String, main: Bool = true, port: Int = 9) -> Bool {
    browserWindowAllows(URL(string: text)!, port: port, isMainFrame: main)
}
precondition(allows("http://127.0.0.1:9/") && allows("http://127.0.0.1:9/#/live/abc") && allows("http://127.0.0.1:9/?x=1#/live/abc"))
for other in ["http://127.0.0.1:10/", "http://localhost:9/", "http://127.0.0.1/", "https://127.0.0.1:9/",
              "http://example.com/", "http://127.0.0.1.example.com:9/", "http://u:p@127.0.0.1:9/", "ftp://127.0.0.1:9/",
              "file:///etc/hosts", "about:blank", "javascript:alert(1)", "data:text/html,x",
              "http://127.0.0.1:9/api/health", "http://127.0.0.1:9/static/app.js"] {
    precondition(!allows(other), other)
}
// A subframe may load the gateway's files and API but nothing from another origin.
precondition(allows("http://127.0.0.1:9/static/app.js", main: false) && allows("http://127.0.0.1:9/api/health", main: false))
precondition(allows("http://127.0.0.1:9/", main: false))
for other in ["http://127.0.0.1:10/", "http://localhost:9/", "https://127.0.0.1:9/", "http://example.com/", "about:blank",
              "http://u:p@127.0.0.1:9/"] {
    precondition(!allows(other, main: false), other)
}
// Before the gateway has a port, nothing is allowed.
precondition(!allows("http://127.0.0.1:0/", port: 0) && !allows("http://127.0.0.1:0/", main: false, port: 0))
""",
        "BrowserPolicy.swift",
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_the_folder_bridge_takes_known_requests_from_the_gateway_page_and_shows_only_chat_folders() -> (
    None
):
    run_policy(
        r"""
precondition(folderRequest(["action": "choose"]) == .choose)
precondition(folderRequest(["action": "reveal", "path": "~/b"]) == .reveal("~/b"))
let refused: [Any] = [["action": "bogus"], ["action": "reveal"], ["action": "reveal", "path": "b"], ["action": 1], "choose", 7,
                      ["action": "reveal", "path": "/a\u{0}b"], ["action": "reveal", "path": "/" + String(repeating: "a", count: 4096)]]
precondition(refused.allSatisfy { folderRequest($0) == nil })
func accepts(scheme: String = "http", host: String = "127.0.0.1", port: Int = 9, main: Bool = true, gateway: Int = 9) -> Bool {
    acceptsBridgeMessage(scheme: scheme, host: host, port: port, isMainFrame: main, gatewayPort: gateway)
}
precondition(accepts())
precondition(!accepts(main: false) && !accepts(scheme: "https") && !accepts(host: "localhost") && !accepts(port: 10))
precondition(!accepts(gateway: 0) && !accepts(scheme: "", host: "", port: 0, gateway: 0))
let fm = FileManager.default
let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
let mine = root.appendingPathComponent("mine"), other = root.appendingPathComponent("other")
let file = root.appendingPathComponent("file").path, gone = root.appendingPathComponent("gone").path
try! fm.createDirectory(at: mine, withIntermediateDirectories: true)
try! fm.createDirectory(at: other, withIntermediateDirectories: true)
fm.createFile(atPath: file, contents: nil)
let known = chatFolders(Data(#"{"sessions":[{"workspace":"\#(mine.path)"},{"workspace":null},{"title":"x"}]}"#.utf8))
precondition(known == [mine.path] && chatFolders(Data("nope".utf8)).isEmpty)
precondition(revealTarget(mine.path, allowed: known)?.resolvingSymlinksInPath().path == mine.resolvingSymlinksInPath().path)
precondition(revealTarget(other.path, allowed: known) == nil, "a folder that is no chat's")
precondition(revealTarget(mine.path + "/../other", allowed: known) == nil)
precondition(revealTarget(file, allowed: [file]) == nil && revealTarget(gone, allowed: [gone]) == nil)
""",
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_real_host_typechecks() -> None:
    # The files the build compiles, as it finds them.
    macos_swift.typecheck_swift(macos_swift.app_sources())


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_the_model_submenu_rows_follow_the_picks_and_the_switch() -> None:
    run_policy(
        r"""let health = Data(#"{"model": {"backend": "melo_serve", "name": "Qwen3.8-27B"}, "switch": {"id": 3, "pick": "8b", "state": "waiting", "title": "Switching to Qwen3 8B…", "detail": "Waiting for 1 chat to finish"}}"#.utf8)
precondition(servesModels(health) && servedModel(health) == "Qwen3.8-27B")
precondition(!servesModels(Data(#"{"model": {"backend": "ollama"}}"#.utf8)))
let running = serveSwitch(health)!
precondition(running.active && running.pick == "8b")
let list = serveList(Data(#"{"foot": "Every chat on this Mac follows the switch.", "picks": [{"id": "27b", "name": "Qwen3.8 27B", "words": "Splash · needs 21 GB · fits", "current": true, "loadable": true}, {"id": "8b", "name": "Qwen3 8B", "words": "oMLX · needs 5 GB · fits", "current": false, "loadable": true}, {"id": "70b", "name": "Qwen3 70B", "words": "Splash · needs 48 GB, 31 GB free", "current": false, "loadable": false}, {"name": "no id"}]}"#.utf8))!
precondition(list.picks.count == 3)
let idle = modelMenuRows(list, switching: nil)
precondition(idle[0] == .pick(id: "27b", title: "Qwen3.8 27B", subtitle: "Splash · needs 21 GB · fits", checked: true, enabled: true))
precondition(idle[2] == .pick(id: "70b", title: "Qwen3 70B", subtitle: "Splash · needs 48 GB, 31 GB free", checked: false, enabled: false))
precondition(idle.last == .foot("Every chat on this Mac follows the switch."))
precondition(modelMenuSubtitle(list, switching: nil) == "Qwen3.8 27B")
let busy = modelMenuRows(list, switching: running)
precondition(busy[0] == .status(title: "Switching to Qwen3 8B…", subtitle: "Waiting for 1 chat to finish"))
precondition(busy[1] == .cancel && busy[2] == .separator)
precondition(!busy.contains { if case .pick(_, _, _, _, true) = $0 { return true } else { return false } })
precondition(modelMenuSubtitle(list, switching: running) == "Switching to Qwen3 8B…")
let failed = serveSwitch(Data(#"{"switch": {"id": 4, "pick": "8b", "state": "failed", "title": "Did not switch to Qwen3 8B", "detail": "1 chat is still working; try again when it finishes"}}"#.utf8))!
let told = modelMenuRows(list, switching: failed)
precondition(told[0] == .status(title: "Did not switch to Qwen3 8B", subtitle: "1 chat is still working; try again when it finishes"))
precondition(told[1] == .separator && told[2] == idle[0])
precondition(modelMenuSubtitle(list, switching: failed) == "Qwen3.8 27B")
precondition(!modelMenuNeedsDrawing(idle, shown: idle, open: false))  // a poll with nothing new
precondition(!modelMenuNeedsDrawing(told, shown: idle, open: true))  // never under the pointer
precondition(modelMenuNeedsDrawing(told, shown: idle, open: false))
precondition(modelMenuNeedsDrawing(idle, shown: [], open: true))  // an open one still empty
""",
        "ModelMenu.swift",
    )
