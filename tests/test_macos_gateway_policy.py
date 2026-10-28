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
