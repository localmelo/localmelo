from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_gateway_instance_and_listener_decisions(tmp_path: Path) -> None:
    main = tmp_path / "main.swift"
    main.write_text("""import Foundation
let expected = "0123456789abcdef0123456789abcdef"
precondition(acceptsInstance(expected, expected: expected, childRunning: true))
precondition(!acceptsInstance(nil, expected: expected, childRunning: true))
precondition(!acceptsInstance("other", expected: expected, childRunning: true))
precondition(!acceptsInstance(expected, expected: expected, childRunning: false))
precondition(ownsListener([42], child: 42))
precondition(!ownsListener([42, 43], child: 42))
precondition(!ownsListener(nil, child: 42))
precondition(isGatewayURL(URL(string: "http://127.0.0.1:9999/")!, port: 9999))
precondition(!isGatewayURL(URL(string: "http://localhost:9999/")!, port: 9999))
precondition(isGatewayPage(URL(string: "http://127.0.0.1:9999/#/workers")!, port: 9999))
precondition(!isGatewayPage(URL(string: "http://127.0.0.1:9999/api/health")!, port: 9999))
let owned = GatewayHealth(store: "unavailable", error: "owned", reason: "owned", running: 0)
let failed = GatewayHealth(store: "unavailable", error: "corrupt", reason: "open-error", running: 0)
precondition(storeFailureMessage(owned).contains("Stop the other gateway"))
precondition(!storeFailureMessage(failed).contains("other gateway"))
precondition(gatewayExitMessage(status: 9, signalled: true).contains("signal 9"))
precondition(gatewayExitMessage(status: 2, signalled: false).contains("status 2"))
""")
    binary = tmp_path / "test-policy"
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            str(ROOT / "macos/LocalMeloApp/GatewayPolicy.swift"),
            str(main),
            "-o",
            str(binary),
        ],
        env={
            **os.environ,
            "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
        },
        check=True,
        timeout=60,
    )
    subprocess.run([str(binary)], check=True, timeout=5)


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_running_count_title_and_quit_text(tmp_path: Path) -> None:
    main = tmp_path / "main.swift"
    main.write_text("""import Foundation
func health(_ running: String) -> Data {
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
precondition(menuTitle(none) == "localmelo")
precondition(menuTitle(nil) == "localmelo")
precondition(menuTitle(busy) == "localmelo · 7")
precondition(quitMessage(busy, statusKnown: true)
    == "2 tasks and 1 chat are running. Quitting stops the gateway and everything running in it. "
    + "Not affected: 1 run and 3 terminal chats.")
precondition(quitMessage(chat, statusKnown: true)
    == "1 chat is running. Quitting stops the gateway and everything running in it.")
precondition(quitMessage(runsOnly, statusKnown: true)
    == "Quitting stops the gateway and any running tasks. Not affected: 1 run.")
precondition(quitMessage(none, statusKnown: true) == "Quitting stops the gateway and any running tasks.")
precondition(quitMessage(nil, statusKnown: true) == "Quitting stops the gateway and any running tasks.")
precondition(quitMessage(nil, statusKnown: false).hasPrefix("The task status is unavailable."))
""")
    binary = tmp_path / "test-running"
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            str(ROOT / "macos/LocalMeloApp/GatewayPolicy.swift"),
            str(main),
            "-o",
            str(binary),
        ],
        env={
            **os.environ,
            "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
        },
        check=True,
        timeout=60,
    )
    subprocess.run([str(binary)], check=True, timeout=5)


def run_policy(tmp_path: Path, source: str) -> None:
    """Compile the policy file with a test main and run it."""
    main = tmp_path / "main.swift"
    main.write_text("import Foundation\n" + source)
    binary = tmp_path / "test-policy"
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            str(ROOT / "macos/LocalMeloApp/GatewayPolicy.swift"),
            str(main),
            "-o",
            str(binary),
        ],
        env={
            **os.environ,
            "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
        },
        check=True,
        timeout=60,
    )
    subprocess.run([str(binary)], check=True, timeout=5)


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift AppKit host is macOS-only")
def test_gateway_port_comes_from_the_startup_line(tmp_path: Path) -> None:
    run_policy(
        tmp_path,
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
def test_window_texts_and_actions(tmp_path: Path) -> None:
    run_policy(
        tmp_path,
        r"""
// The step line follows what the app knows: no port yet, a port, a signed-in console.
precondition(startStepText(portKnown: false, ready: false) == "Starting the local server")
precondition(startStepText(portKnown: true, ready: false) == "Signing in")
precondition(startStepText(portKnown: true, ready: true) == "Opening your workspace")

// A failure keeps the reason it is given; the title says whether the console had loaded.
let reason = gatewayExitMessage(status: 9, signalled: true)
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
def test_real_host_typechecks() -> None:
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            "-typecheck",
            str(ROOT / "macos/LocalMeloApp/GatewayPolicy.swift"),
            str(ROOT / "macos/LocalMeloApp/StatusView.swift"),
            str(ROOT / "macos/LocalMeloApp/main.swift"),
        ],
        env={
            **os.environ,
            "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
        },
        check=True,
        timeout=60,
    )
