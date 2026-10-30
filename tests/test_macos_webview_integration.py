from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS WebKit"),
    pytest.mark.skipif(
        os.environ.get("LOCALMELO_MACOS_UI_TESTS") != "1",
        reason="requires an opted-in macOS GUI session",
    ),
]

# Instrument a temporary copy of the real host at observable lifecycle points.
# The tests do not replace its navigation, shutdown, or alert decisions.
OBSERVER = r"""
let events = URL(fileURLWithPath: ProcessInfo.processInfo.environment["TEST_EVENTS"]!)
func record(_ text: String) {
    let data = (text + "\n").data(using: .utf8)!
    if !FileManager.default.fileExists(atPath: events.path) {
        FileManager.default.createFile(atPath: events.path, contents: nil)
    }
    let handle = try! FileHandle(forWritingTo: events)
    handle.seekToEndOfFile()
    handle.write(data)
    handle.closeFile()
}
var quitPhase = 0
var didSignInAgain = false
var didRetry = false
"""

FAKE_GATEWAY = r"""import http.server, json, os, pathlib, sys, threading, time
from http.cookies import SimpleCookie
from localmelo.support.gateway.auth import GatewayAuth
port = int(sys.argv[sys.argv.index("--port") + 1])
home = pathlib.Path(os.environ["HOME"])
mode = os.environ["TEST_MODE"]
instance = os.environ["LOCALMELO_APP_INSTANCE"]
if mode == "wrong-quit":
    mode = "wrong"
if mode == "retry":
    starts = home / "starts"
    mode = "correct" if starts.exists() else "wrong"
    with starts.open("a") as stream:
        stream.write("start\n")
pidfile = home / "gateway-pid"
pidfile.write_text(str(os.getpid()))
auth = GatewayAuth()
requests = home / "requests"
logins = home / "logins"
closing = False

def record(text):
    with requests.open("a") as stream:
        stream.write(text + "\n")

def note_login():
    with logins.open("a") as stream:
        stream.write("login\n")

# Let a plain listener take the port, then answer from the accepted connection
# only once it holds the port. The app must notice the new owner.
def handoff():
    global closing
    if closing:
        return
    closing = True
    def close():
        server.shutdown()
        server.server_close()
        (home / "port-free").write_text("yes")
    threading.Thread(target=close, daemon=True).start()
    deadline = time.monotonic() + 5
    while not (home / "foreign-ready").exists() and time.monotonic() < deadline:
        time.sleep(.02)

def cookie_ok(header):
    cookie = SimpleCookie()
    try:
        cookie.load(header)
    except Exception:
        return False
    name = f"melo_session_{port}"
    return name in cookie and auth.valid_cookie(cookie[name].value, port)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/auth/login":
            self.send_error(404)
            return
        record("POST " + self.path)
        if self.headers.get("X-LocalMelo-Instance") != instance:
            self.send_error(421)
            return
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        cookie = auth.consume_login(data.get("code", ""), port, instance=instance)
        if self.headers.get("Origin") != f"http://127.0.0.1:{port}" or not cookie:
            self.send_error(403)
            return
        note_login()
        if self.headers.get("X-Test-Browser") == "1":
            (home / "browser-login-ok").write_text("yes")
        if mode == "no-cookie":
            self.send_response(204)
            self.end_headers()
            return
        if mode == "rotate" and (home / "rotate-now").exists():
            (home / "rotate-now").unlink()
        self.send_response(204)
        self.send_header("Set-Cookie", f"melo_session_{port}={cookie}; HttpOnly; Path=/")
        self.end_headers()
    def do_GET(self):
        record("GET " + self.path)
        if self.path == "/api/health":
            if not cookie_ok(self.headers.get("Cookie", "")):
                body = b"{}"
                status = 401
            elif mode == "rotate" and (home / "rotate-now").exists():
                body = b"{}"
                status = 401
            elif mode == "store":
                body = b'{"api":2,"sessions":{"store":"unavailable","store_error":"Another melo gateway owns the store","store_reason":"owned","running":0}}'
                status = 200
            elif mode == "store-open-error":
                body = b'{"api":2,"sessions":{"store":"unavailable","store_error":"permission denied","store_reason":"open-error","running":0}}'
                status = 200
            elif mode == "quit":
                body = b'{"api":2,"sessions":{"store":"ok","store_error":"","running":1}}'
                status = 200
            else:
                body = b'{"api":2,"sessions":{"store":"ok","store_error":"","running":0}}'
                status = 200
            if mode == "handoff-health" and status == 200:
                handoff()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if status == 200 and mode in ("correct", "quit", "idlequit", "keys", "rotate", "signin"):
                self.send_header("X-LocalMelo-Instance", instance)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.headers.get("X-LocalMelo-Instance") is None:
            (home / "requested-page").write_text("yes")
        page = '<html><body id="test-page">test page</body></html>'
        body = page.encode()
        if mode == "handoff-probe" and self.path == "/":
            handoff()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        if mode == "wrong":
            self.send_header("X-LocalMelo-Instance", "wrong")
        elif mode != "absent":
            self.send_header("X-LocalMelo-Instance", instance)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args): pass
server = None
if mode == "foreign":
    # Report a port that another process holds.
    port = int(os.environ["TEST_FOREIGN_PORT"])
elif mode != "silent":
    # --port 0: bind first, then say which port came out.
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    (home / "port").write_text(f"{port}\n")
if mode != "silent":
    print(f"  Gateway: http://127.0.0.1:{port}", flush=True)
while not (home / "exit-child").exists(): time.sleep(.05)
if server:
    server.shutdown()
pidfile.unlink()
"""


@pytest.fixture(scope="module", autouse=True)
def serialize_webkit(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    # xdist workers share the base's parent. Do not make several AppKit hosts
    # compete for focus during the same test run.
    import fcntl

    path = tmp_path_factory.getbasetemp().parent / "localmelo-webkit.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("webview-app")
    bundle = root / "Test.app/Contents"
    executable = bundle / "MacOS/Test"
    executable.parent.mkdir(parents=True)
    runtime = bundle / "Resources/runtime/bin/melo"
    runtime.parent.mkdir(parents=True)
    runtime.write_text(f"#!{sys.executable}\n" + FAKE_GATEWAY)
    runtime.chmod(0o755)
    python = runtime.parent / "python/bin/python3"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    source = (ROOT / "macos/LocalMeloApp/main.swift").read_text()
    injections = {
        "        webView = nil": '        webView = nil\n        record("removed")',
        "        tearDownWebView()\n        if let failed = child, failed.isRunning {": r"""        tearDownWebView()
        record("alert " + text)
        if ProcessInfo.processInfo.environment["TEST_MODE"] == "retry" && !didRetry {
            didRetry = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                let event = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [],
                                            timestamp: 0, windowNumber: self.window.windowNumber,
                                            context: nil, characters: "\r", charactersIgnoringModifiers: "\r",
                                            isARepeat: false, keyCode: 36)!
                record("return " + String(self.window.performKeyEquivalent(with: event)))
            }
        }
        if ProcessInfo.processInfo.environment["TEST_MODE"] == "wrong-quit" {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.statusView.buttons[.quit]?.performClick(nil)
            }
        }
        if let failed = child, failed.isRunning {""",
        "    private func showStep() {": '    private func showStep() {\n        record("step " + startStepText(portKnown: port != 0, ready: ready))',
        "        window.makeFirstResponder(webView)": '        window.makeFirstResponder(webView)\n        record("shown")',
        "    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {": r"""    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        record("loaded " + (webView.url?.absoluteString ?? ""))
        let items = NSApp.mainMenu?.items.first?.submenu?.items.map { $0.title }.filter { !$0.isEmpty } ?? []
        record("names window=" + window.title + " status=" + (statusItem.button?.title ?? "")
               + " menu=" + items.joined(separator: ","))
        if let menu = NSApp.mainMenu?.items.last?.submenu {
            record("edit " + menu.items.map { $0.title + ":" + $0.keyEquivalent }.joined(separator: ","))
        }
        if let item = NSApp.mainMenu?.items.first?.submenu?.items.first(where: { $0.title == "Sign In Again" }) {
            let name = item.action.map { NSStringFromSelector($0) } ?? "none"
            let resolved = item.action.flatMap { NSApp.target(forAction: $0, to: item.target, from: item) } != nil
            record("signin-menu " + name + (resolved ? " resolved" : " unresolved"))
        } else {
            record("signin-menu missing")
        }
        if let hash = webView.url?.fragment {
            let fields = hash.split(separator: "&")
            let code = fields.first { $0.hasPrefix("login=") }?.dropFirst("login=".count)
            let launch = fields.first { $0.hasPrefix("instance=") }?.dropFirst("instance=".count)
            if let code = code, let launch = launch {
                var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/auth/login")!)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.setValue("http://127.0.0.1:\(port)", forHTTPHeaderField: "Origin")
                request.setValue(String(launch), forHTTPHeaderField: "X-LocalMelo-Instance")
                request.setValue("1", forHTTPHeaderField: "X-Test-Browser")
                request.httpBody = try? JSONSerialization.data(withJSONObject: ["code": String(code)])
                URLSession.shared.dataTask(with: request).resume()
            }
        }
        if ProcessInfo.processInfo.environment["TEST_MODE"] == "keys" {
            webView.evaluateJavaScript("document.body.innerHTML='<input id=editor value=hello>'; document.getElementById('editor').focus()") { _, _ in
                let event = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: .command,
                                            timestamp: 0, windowNumber: self.window.windowNumber,
                                            context: nil, characters: "a", charactersIgnoringModifiers: "a",
                                            isARepeat: false, keyCode: 0)!
                _ = webView.performKeyEquivalent(with: event)
                webView.evaluateJavaScript("document.getElementById('editor').selectionEnd - document.getElementById('editor').selectionStart") { value, _ in
                    record("selection " + String(describing: value ?? "nil"))
                }
            }
        }
        if ["quit", "idlequit"].contains(ProcessInfo.processInfo.environment["TEST_MODE"] ?? "") {
            DispatchQueue.main.async { NSApp.terminate(nil) }
        }
        if ProcessInfo.processInfo.environment["TEST_MODE"] == "signin" && !didSignInAgain {
            didSignInAgain = true
            record("sign-in-action")
            let home = ProcessInfo.processInfo.environment["HOME"]!
            func performSignIn(_ remaining: Int) {
                if FileManager.default.fileExists(atPath: home + "/browser-login-ok") || remaining == 0 {
                    _ = (NSApp.delegate as? NSObject)?.perform(Selector(("signInAgain")))
                } else {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { performSignIn(remaining - 1) }
                }
            }
            performSignIn(100)
        }""",
        "    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {": '    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {\n        record("quit-request")',
        '                    alert.addButton(withTitle: "Cancel")': """                    alert.addButton(withTitle: "Cancel")
                    record("quit-prompt")
                    record("quit-title " + alert.messageText)
                    if ["quit", "idlequit"].contains(ProcessInfo.processInfo.environment["TEST_MODE"] ?? "") {
                        let timer = Timer(timeInterval: 0.1, repeats: false) { _ in
                            if quitPhase == 0 {
                                quitPhase = 1
                                record("quit-cancel")
                                alert.buttons[1].performClick(nil)
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                    guard child.isRunning, self.webView != nil else {
                                        record("quit-not-resumed")
                                        return
                                    }
                                    record("quit-resumed")
                                    NSApp.terminate(nil)
                                }
                            } else {
                                record("quit-confirm")
                                alert.buttons[0].performClick(nil)
                            }
                        }
                        RunLoop.main.add(timer, forMode: .modalPanel)
                    }""",
    }
    for needle, replacement in injections.items():
        assert source.count(needle) == 1
        source = source.replace(needle, replacement)
    quit_at = source.index("    func applicationShouldTerminate(")
    prefix, quit_source = source[:quit_at], source[quit_at:]
    callback = (
        "healthSession.dataTask(with: request) { [weak self] data, response, _ in"
    )
    assert quit_source.count(callback) == 1
    quit_source = quit_source.replace(
        callback, callback + '\n            record("quit-health-returned")'
    )
    source = prefix + quit_source
    assert source.count("application.run()") == 1
    main = root / "main.swift"
    main.write_text(
        source.replace("application.run()", OBSERVER + "\napplication.run()")
    )
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            str(ROOT / "macos/LocalMeloApp/GatewayPolicy.swift"),
            str(ROOT / "macos/LocalMeloApp/StatusView.swift"),
            str(main),
            "-o",
            str(executable),
        ],
        env={
            **os.environ,
            "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
        },
        check=True,
        timeout=90,
    )
    return executable


def wait_for(
    path: Path,
    predicate: str,
    process: subprocess.Popen[bytes],
    timeout: float = 15,
    count: int = 1,
) -> str:
    deadline = time.monotonic() + timeout
    text = ""
    while time.monotonic() < deadline:
        if path.exists():
            text = path.read_text()
            if text.count(predicate) >= count:
                return text
        assert (
            process.poll() is None
        ), f"host exited early: {process.returncode}; events={text!r}"
        time.sleep(0.1)
    pytest.fail(f"timed out waiting for {predicate!r}; events={text!r}")


def launch(app: Path, home: Path, mode: str, **extra: str) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        "HOME": str(home),
        "TEST_EVENTS": str(home / "events"),
        "TEST_MODE": mode,
        **extra,
    }
    return subprocess.Popen(
        [str(app)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop(process: subprocess.Popen[bytes], home: Path) -> None:
    # Foundation Process starts the gateway in a separate process group.

    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)
    pidfile = home / "gateway-pid"
    if pidfile.exists():
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(pidfile.read_text()), signal.SIGKILL)


@contextlib.contextmanager
def running(
    app: Path, home: Path, mode: str
) -> Iterator[tuple[subprocess.Popen[bytes], int]]:
    process = launch(app, home, mode)
    try:
        # The app asked for port 0; the fake gateway says which port it got.
        yield process, int(wait_for(home / "port", "\n", process))
    finally:
        stop(process, home)


@pytest.mark.parametrize("mode", ["wrong", "absent"])
def test_wrong_or_absent_instance_header_sends_no_login_code(
    app: Path, tmp_path: Path, mode: str
) -> None:
    with running(app, tmp_path, mode) as (process, port):
        events = wait_for(tmp_path / "events", "alert ", process)
        requests = (tmp_path / "requests").read_text()
        assert "GET /" in requests, "the instance check did not reach the listener"
        assert "POST /api/auth/login" not in requests
        assert not (
            tmp_path / "logins"
        ).exists(), "a login code was sent to a foreign listener"
        assert not (tmp_path / "requested-page").exists()
        assert "removed" in events
        assert "loaded" not in events
        assert f"Port {port} is held by another process" in events
        time.sleep(1)  # the failure is shown in the window: the app stays open
        assert process.poll() is None


def test_a_login_that_stores_no_cookie_is_retried(app: Path, tmp_path: Path) -> None:
    with running(app, tmp_path, "no-cookie") as (process, _):
        wait_for(tmp_path / "logins", "login", process, timeout=20, count=2)
        assert not (tmp_path / "requested-page").exists()
        assert not (tmp_path / "browser-login-ok").exists()


@pytest.mark.parametrize("mode", ["rotate", "signin"])
def test_reauthentication_gets_a_new_session(
    app: Path, tmp_path: Path, mode: str
) -> None:
    with running(app, tmp_path, mode) as (process, _):
        if mode == "rotate":
            wait_for(tmp_path / "browser-login-ok", "yes", process)
            (tmp_path / "rotate-now").write_text("yes")
        else:
            wait_for(tmp_path / "events", "sign-in-action", process)
        wait_for(tmp_path / "logins", "login", process, timeout=20, count=4)
        events = wait_for(tmp_path / "events", "loaded", process, count=2)
        assert "alert " not in events


def test_command_select_all_in_webview(app: Path, tmp_path: Path) -> None:
    with running(app, tmp_path, "keys") as (process, _):
        wait_for(tmp_path / "events", "selection 5", process)


def test_the_window_and_menus_call_the_app_localmelo(app: Path, tmp_path: Path) -> None:
    with running(app, tmp_path, "correct") as (process, _):
        events = wait_for(tmp_path / "events", "names ", process)
        line = next(x for x in events.splitlines() if x.startswith("names "))
        assert line == (
            "names window=localmelo status=localmelo"
            " menu=Open localmelo,Sign In Again,Quit localmelo"
        )


def test_sign_in_again_menu_item_is_wired(app: Path, tmp_path: Path) -> None:
    with running(app, tmp_path, "correct") as (process, _):
        events = wait_for(tmp_path / "events", "signin-menu ", process)
        assert "signin-menu signInAgain resolved" in events


@pytest.mark.parametrize("mode", ["handoff-probe", "handoff-health"])
def test_listener_that_takes_the_port_after_the_probe_receives_no_code(
    app: Path, tmp_path: Path, mode: str
) -> None:
    received: list[str] = []
    servers: list[HTTPServer] = []

    class ForeignHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received.append("POST " + self.path)
            self.send_response(204)
            self.end_headers()

        def do_GET(self) -> None:
            received.append("GET " + self.path)
            self.send_response(401)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    def take_port() -> None:
        # The fake child holds the port until its handoff closes the listener.
        deadline = time.monotonic() + 10
        while not (tmp_path / "port-free").exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        port = int((tmp_path / "port").read_text())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            candidate = socket.socket()
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                candidate.close()
                time.sleep(0.005)
                continue
            server = HTTPServer(
                ("127.0.0.1", port), ForeignHandler, bind_and_activate=False
            )
            server.socket.close()
            server.socket = candidate
            server.server_address = candidate.getsockname()
            server.server_activate()
            servers.append(server)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            (tmp_path / "foreign-ready").write_text("yes")
            return
        (tmp_path / "foreign-ready").write_text("failed")

    threading.Thread(target=take_port, daemon=True).start()
    process = launch(app, tmp_path, mode)
    try:
        events = wait_for(tmp_path / "events", "alert ", process)
        port = int((tmp_path / "port").read_text())
        assert f"Port {port} is held by another process" in events
        assert not (tmp_path / "browser-login-ok").exists()
        time.sleep(0.5)
        assert not received, f"the foreign listener received {received}"
    finally:
        stop(process, tmp_path)
        for server in servers:
            server.shutdown()
            server.server_close()


def test_child_exit_removes_loaded_webview(app: Path, tmp_path: Path) -> None:
    with running(app, tmp_path, "correct") as (process, port):
        wait_for(tmp_path / "browser-login-ok", "yes", process)
        logins = (tmp_path / "logins").read_text().splitlines()
        assert len(logins) == 2, "health and page must consume separate one-use codes"
        events = wait_for(tmp_path / "events", "edit ", process)
        for item in ("Undo:z", "Redo:Z", "Cut:x", "Copy:c", "Paste:v", "Select All:a"):
            assert item in events
        log = tmp_path / "Library/Logs/LocalMelo/gateway.log"
        assert f"Gateway: http://127.0.0.1:{port}" in log.read_text()
        child_pid = int((tmp_path / "gateway-pid").read_text())
        os.kill(child_pid, signal.SIGKILL)
        events = wait_for(tmp_path / "events", "alert ", process)
        assert "removed" in events
        assert "terminated by signal 9" in events
        time.sleep(1)  # the failure is shown in the window: the app stays open
        assert process.poll() is None


def test_each_step_shows_until_the_console_has_loaded(
    app: Path, tmp_path: Path
) -> None:
    with running(app, tmp_path, "correct") as (process, _):
        events = wait_for(tmp_path / "events", "shown", process)
        order = [
            events.index(text)
            for text in (
                "step Starting the local server",
                "step Signing in",
                "step Opening your workspace",
                "loaded ",
                "shown",
            )
        ]
        assert order == sorted(order), events


def test_try_again_starts_a_new_gateway_after_a_collision(
    app: Path, tmp_path: Path
) -> None:
    process = launch(app, tmp_path, "retry")  # the first start meets a wrong owner
    try:
        events = wait_for(tmp_path / "events", "alert ", process)
        port = int((tmp_path / "port").read_text())
        assert f"Port {port} is held by another process" in events
        # The host presses Return, the default button's key; the second start is right.
        wait_for(tmp_path / "events", "loaded ", process)
        assert "return true" in (tmp_path / "events").read_text()
        assert (tmp_path / "starts").read_text().count("start") == 2
        assert process.poll() is None
    finally:
        stop(process, tmp_path)


def test_quit_in_the_failure_view_ends_the_app_without_a_dialog(
    app: Path, tmp_path: Path
) -> None:
    process = launch(app, tmp_path, "wrong-quit")
    try:
        wait_for(tmp_path / "events", "alert ", process)
        assert process.wait(timeout=15) == 0  # the host clicks Quit by itself
        assert "quit-prompt" not in (tmp_path / "events").read_text()
    finally:
        stop(process, tmp_path)


def test_owned_store_stops_host_before_page_load(app: Path, tmp_path: Path) -> None:
    with running(app, tmp_path, "store") as (process, _):
        wait_for(tmp_path / "events", "removed", process)
        events = (tmp_path / "events").read_text()
        assert "session store is unavailable" in events
        assert "Another melo gateway" in events
        assert not (tmp_path / "requested-page").exists()


def test_non_owner_store_error_gives_relevant_hint(app: Path, tmp_path: Path) -> None:
    with running(app, tmp_path, "store-open-error") as (process, _):
        events = wait_for(tmp_path / "events", "alert ", process)
        assert "permission denied" in events
        assert "store permissions" in events
        assert "Stop the other gateway" not in events
        assert not (tmp_path / "requested-page").exists()


@pytest.mark.parametrize("mode", ["quit", "idlequit"])
def test_quit_can_cancel_then_stop(app: Path, tmp_path: Path, mode: str) -> None:
    with running(app, tmp_path, mode) as (process, _):
        events = wait_for(tmp_path / "events", "quit-resumed", process)
        assert "quit-cancel" in events
        assert "quit-title Quit localmelo?" in events
        child_pid = int((tmp_path / "gateway-pid").read_text())
        process.wait(timeout=15)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        events = (tmp_path / "events").read_text()
        assert "quit-confirm" in events
        assert process.returncode == 0


def test_a_port_the_child_does_not_own_receives_no_request(
    app: Path, tmp_path: Path
) -> None:
    requests: list[str] = []

    class ForeignHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"foreign")

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), ForeignHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    process = launch(
        app, tmp_path, "foreign", TEST_FOREIGN_PORT=str(server.server_address[1])
    )
    try:
        time.sleep(3)  # many polls: the app checks the owner before any request
        assert process.poll() is None
        assert not requests
        events = tmp_path / "events"  # not written until something happens
        assert not events.exists() or "loaded" not in events.read_text()
    finally:
        stop(process, tmp_path)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_a_gateway_that_never_reports_a_port_is_stopped_at_the_deadline(
    app: Path, tmp_path: Path
) -> None:
    process = launch(app, tmp_path, "silent")
    try:
        events = wait_for(tmp_path / "events", "alert ", process, timeout=40)
        assert "did not report its port" in events
        assert "loaded" not in events
    finally:
        stop(process, tmp_path)
