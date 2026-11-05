"""A file dropped on the real app window: the console takes it, and the shell never leaves the page.

This builds a copy of the app from its own sources and runs it against a real gateway (started from this
checkout, with a temporary home folder and a model server that answers the same thing to everything). The
app's window is invisible and click-through, the app does not take the focus, and its menu-bar item and pet
stay hidden, so the check does not disturb whoever is using the machine. The drop goes through WebKit's own
drag-and-drop path: a dragging session whose pasteboard holds a file URL is handed to the web view, as the
system hands it a drag from Finder.

Opt in with ``LOCALMELO_MACOS_UI_TESTS=1``. The build takes about a minute.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
import zlib
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from localmelo.tests.fixtures.http_stub import HttpStub

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS WebKit"),
    pytest.mark.skipif(
        os.environ.get("LOCALMELO_MACOS_UI_TESTS") != "1",
        reason="requires an opted-in macOS GUI session",
    ),
]

# A dragging session as the system builds one for a drag from Finder, and the steps of the check, run inside
# a temporary copy of the real host once the console is up. Every result is written to the events file.
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
var dropDriven = false

final class FileDrag: NSObject, NSDraggingInfo {
    let board: NSPasteboard
    let point: NSPoint
    weak var target: NSWindow?
    init(board: NSPasteboard, point: NSPoint, window: NSWindow) {
        self.board = board
        self.point = point
        self.target = window
    }
    var draggingDestinationWindow: NSWindow? { target }
    var draggingSourceOperationMask: NSDragOperation { .copy }
    var draggingLocation: NSPoint { point }
    var draggedImageLocation: NSPoint { point }
    var draggedImage: NSImage? { nil }
    var draggingPasteboard: NSPasteboard { board }
    var draggingSource: Any? { nil }
    var draggingSequenceNumber: Int { 4242 }
    var draggingFormation: NSDraggingFormation = .default
    var animatesToDestination: Bool = false
    var numberOfValidItemsForDrop: Int = 1
    var springLoadingHighlight: NSSpringLoadingHighlight { .none }
    func slideDraggedImage(to screenPoint: NSPoint) {}
    func enumerateDraggingItems(options enumOpts: NSDraggingItemEnumerationOptions = [], for view: NSView?,
                                classes classArray: [AnyClass], searchOptions: [NSPasteboard.ReadingOptionKey: Any] = [:],
                                using block: @escaping (NSDraggingItem, Int, UnsafeMutablePointer<ObjCBool>) -> Void) {}
    func resetSpringLoading() {}
}

extension App {
    func driveDrop() {
        guard !dropDriven else { return }
        dropDriven = true
        let env = ProcessInfo.processInfo.environment

        func after(_ seconds: Double, _ step: @escaping () -> Void) {
            DispatchQueue.main.asyncAfter(deadline: .now() + seconds, execute: step)
        }
        func ask(_ script: String, _ done: @escaping (String) -> Void) {
            webView.evaluateJavaScript(script) { value, error in
                if let error = error { record("js-error " + error.localizedDescription) }
                done(value.map { String(describing: $0) } ?? "nil")
            }
        }
        // Poll the page until *script* is true, up to *seconds*.
        func until(_ script: String, seconds: Double = 15, _ done: @escaping (Bool) -> Void) {
            let deadline = Date().addingTimeInterval(seconds)
            func poll() {
                ask(script) { value in
                    if value == "1" || value == "true" { done(true) }
                    else if Date() > deadline { done(false) }
                    else { after(0.2, poll) }
                }
            }
            poll()
        }
        // Hand the web view a dragging session over its middle, with the file on the dragging pasteboard.
        func drag(_ path: String, then: @escaping () -> Void) {
            let board = NSPasteboard(name: NSPasteboard.Name("com.localmelo.test.drag"))
            board.clearContents()
            board.writeObjects([URL(fileURLWithPath: path) as NSURL])
            let point = webView.convert(NSPoint(x: webView.bounds.midX, y: webView.bounds.midY), to: nil)
            let info = FileDrag(board: board, point: point, window: window)
            let destination = webView as NSDraggingDestination
            let entered = destination.draggingEntered?(info) ?? []
            record("drag-entered \((path as NSString).lastPathComponent) op=\(entered.rawValue)")
            after(0.4) {
                let updated = destination.draggingUpdated?(info) ?? []
                record("drag-updated op=\(updated.rawValue)")
                after(0.4) {
                    let accepted = destination.performDragOperation?(info) ?? false
                    destination.concludeDragOperation?(info)
                    record("drag-performed accepted=\(accepted)")
                    then()
                }
            }
        }
        func address() -> String { webView.url?.absoluteString ?? "none" }

        record("drop-driver start")
        until("document.documentElement.dataset.ready === '1' && !document.getElementById('input').disabled") { ready in
            record("page-ready \(ready) url=\(address())")
            guard ready else { NSApp.terminate(nil); return }
            // 1. A picture: it is uploaded and shown as a thumbnail in the message box.
            drag(env["TEST_DROP_PNG"]!) {
                until("document.querySelectorAll('#media-tray .media-thumb').length === 1", seconds: 15) { shown in
                    ask("(document.querySelector('#media-tray .media-thumb')||{}).src || 'none'") { source in
                        record("picture-shown \(shown) src=\(source) url=\(address())")
                        // 2. A file the console cannot take: it says so, and stays.
                        drag(env["TEST_DROP_PDF"]!) {
                            until("document.getElementById('notice').textContent.includes('not supported yet')", seconds: 10) { told in
                                ask("document.getElementById('notice').textContent") { notice in
                                    record("other-refused \(told) notice=\(notice) url=\(address())")
                                    // 3. A drop the page does not take at all: WebKit's own answer to a file is to open it,
                                    //    and the shell must refuse that.
                                    ask("for (const type of ['dragenter', 'dragover', 'dragleave', 'drop']) { document.addEventListener(type, (event) => event.stopImmediatePropagation(), true); } 'blocked'") { _ in
                                        drag(env["TEST_DROP_PDF"]!) {
                                            after(2.5) {
                                                ask("document.getElementById('input') ? 'composer' : 'gone'") { page in
                                                    record("missed-drop url=\(address()) page=\(page)")
                                                    record("drop-driver done")
                                                    NSApp.terminate(nil)
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
"""


def png(width: int, height: int) -> bytes:
    """A real PNG of *width* x *height*, standard library only."""
    rows = b"".join(
        b"\x00" + bytes((x + y) % 256 for x in range(width * 3)) for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class _Model(http.server.BaseHTTPRequestHandler):
    """A model server that answers every request the same way."""

    def _send(self, status: int, body: dict[str, object]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._send(
                200, {"object": "list", "data": [{"id": "stub", "object": "model"}]}
            )
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self._send(
            200,
            {
                "id": "x",
                "object": "chat.completion",
                "model": "stub",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )


@contextlib.contextmanager
def model_server() -> Iterator[int]:
    with HttpStub(_Model) as stub:
        yield stub.port


def build_app(root: Path) -> Path:
    """The app's sources with the observer added, built into a bundle that starts this checkout's gateway."""
    bundle = root / "Test.app/Contents"
    executable = bundle / "MacOS/Test"
    executable.parent.mkdir(parents=True)
    melo = Path(sys.executable).parent / "melo"
    assert melo.exists(), "run the test from the project's own environment"
    runtime = bundle / "Resources/runtime/bin/melo"
    runtime.parent.mkdir(parents=True)
    runtime.write_text(f'#!/bin/sh\nexec "{melo}" "$@"\n')
    runtime.chmod(0o755)
    # The app asks this interpreter for its one-use sign-in codes.
    python = runtime.parent / "python/bin/python3"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    for logo in ("MenuBarLogo.png", "MenuBarLogo@2x.png"):
        shutil.copy(ROOT / "macos/assets" / logo, bundle / "Resources" / logo)
    shutil.copytree(
        ROOT / "support/gateway/static/moves",
        python.parent.parent
        / "lib/python3.11/site-packages/localmelo/support/gateway/static/moves",
    )
    source = (ROOT / "macos/LocalMeloApp/main.swift").read_text()
    injections = {
        # No Dock icon, and no taking the focus from whoever is working.
        "        NSApp.setActivationPolicy(.regular)\n": "        NSApp.setActivationPolicy(.accessory)\n",
        "        window.center()\n        window.makeKeyAndOrderFront(nil)\n        NSApp.activate(ignoringOtherApps: true)\n": (
            "        window.center()\n        window.alphaValue = 0\n        window.ignoresMouseEvents = true\n"
            "        window.makeKeyAndOrderFront(nil)\n"
        ),
        "                          showPet: #selector(showPet))\n": (
            "                          showPet: #selector(showPet))\n        menuBar.statusItem.isVisible = false\n"
        ),
        "        showConsole()\n    }\n": "        showConsole()\n        driveDrop()\n    }\n",
        # Every navigation the shell turns away.
        "            decisionHandler(.cancel)\n            if action.navigationType == .linkActivated,": (
            '            decisionHandler(.cancel)\n            record("cancelled " + url.absoluteString)\n'
            "            if action.navigationType == .linkActivated,"
        ),
    }
    for needle, replacement in injections.items():
        assert source.count(needle) == 1, needle
        source = source.replace(needle, replacement)
    assert source.count("application.run()") == 1
    main = root / "main.swift"
    main.write_text(
        source.replace("application.run()", OBSERVER + "\napplication.run()")
    )
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            *(
                str(path)
                for path in sorted((ROOT / "macos/LocalMeloApp").glob("*.swift"))
                if path.name != "main.swift"
            ),
            str(main),
            "-o",
            str(executable),
        ],
        env={
            **os.environ,
            "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
        },
        check=True,
        timeout=300,
    )
    return executable


def wait_for(
    path: Path, predicate: str, process: subprocess.Popen[bytes], timeout: float
) -> str:
    deadline = time.monotonic() + timeout
    text = ""
    while time.monotonic() < deadline:
        if path.exists():
            text = path.read_text()
            if predicate in text:
                return text
        if process.poll() is not None:
            break
        time.sleep(0.2)
    return path.read_text() if path.exists() else text


def test_a_dropped_picture_is_taken_and_a_missed_drop_never_leaves_the_page(
    tmp_path: Path,
) -> None:
    app = build_app(tmp_path)
    home = tmp_path / "home"
    (home / ".cache/localmelo").mkdir(parents=True)
    picture = tmp_path / "shot.png"
    picture.write_bytes(png(48, 32))
    other = tmp_path / "report.pdf"
    other.write_bytes(b"%PDF-1.4\n% not a picture or text\n")
    events = tmp_path / "events"
    with model_server() as port:
        (home / ".cache/localmelo/config.toml").write_text(
            'chat_backend = "vllm"\nembedding_backend = "none"\n\n'
            f'[vllm]\nchat_url = "http://127.0.0.1:{port}/v1"\nchat_model = "stub"\n'
        )
        process = subprocess.Popen(
            [str(app)],
            env={
                **os.environ,
                "HOME": str(home),
                "TEST_EVENTS": str(events),
                "TEST_DROP_PNG": str(picture),
                "TEST_DROP_PDF": str(other),
                # Model servers are found only where this test says.
                "LOCALMELO_DETECT_URLS": "",
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            text = wait_for(events, "drop-driver done", process, timeout=120)
            process.wait(timeout=30)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            endpoint = home / ".cache/localmelo/gateway/endpoint.json"
            if endpoint.exists():
                with contextlib.suppress(Exception, ProcessLookupError):
                    os.kill(
                        int(json.loads(endpoint.read_text())["pid"]), signal.SIGTERM
                    )
    lines = text.splitlines()

    def field(line: str, name: str) -> str:
        """The value after ``name=`` up to the next space."""
        return line.split(f"{name}=", 1)[1].split(" ", 1)[0]

    def origin(address: str) -> str:
        """``http://127.0.0.1:PORT``: the console's address without its path, query and route."""
        parts = urlsplit(address)
        return f"{parts.scheme}://{parts.netloc}"

    assert any(line.startswith("page-ready true") for line in lines), text
    gateway = origin(
        field(next(line for line in lines if line.startswith("page-ready")), "url")
    )
    assert gateway.startswith("http://127.0.0.1:"), gateway
    # The picture was taken: uploaded, and shown as the thumbnail of the staged copy.
    shown = next(line for line in lines if line.startswith("picture-shown"))
    assert shown.startswith("picture-shown true "), shown
    assert field(shown, "src").startswith(f"{gateway}/api/uploads/"), shown
    assert origin(field(shown, "url")) == gateway, shown
    # A file that is neither a picture nor text is told "not yet", and the window stays on the console.
    refused = next(line for line in lines if line.startswith("other-refused"))
    assert refused.startswith("other-refused true"), refused
    assert origin(field(refused, "url")) == gateway, refused
    # The page took no part in the last drop: WebKit tried to open the file, the shell turned that away,
    # and the window is still the console.
    missed = next(line for line in lines if line.startswith("missed-drop"))
    assert origin(field(missed, "url")) == gateway and missed.endswith(
        "page=composer"
    ), missed
    assert any(line.startswith("cancelled file:") for line in lines), (
        "WebKit never tried to open the dropped file, so the missed drop proved nothing\n"
        + text
    )
    assert not any(line.startswith("cancelled http") for line in lines), text
