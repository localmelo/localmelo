from __future__ import annotations

import os
import subprocess
import sys
import textwrap

LABEL = "ai.localmelo.gateway"
PLIST_DIR = os.path.expanduser("~/Library/LaunchAgents")
PLIST_PATH = os.path.join(PLIST_DIR, f"{LABEL}.plist")


def _plist_xml(
    python: str,
    module_args: list[str],
    log_dir: str,
) -> str:
    args_xml = "\n".join(
        f"        <string>{a}</string>" for a in [python, "-m"] + module_args
    )
    template = f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{LABEL}</string>
            <key>ProgramArguments</key>
            <array>
        {args_xml}
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>ThrottleInterval</key>
            <integer>3</integer>
            <key>StandardOutPath</key>
            <string>{log_dir}/gateway.stdout.log</string>
            <key>StandardErrorPath</key>
            <string>{log_dir}/gateway.stderr.log</string>
        </dict>
        </plist>
    """
    return textwrap.dedent(template)


def install(port: int = 8401) -> str:
    if sys.platform != "darwin":
        raise RuntimeError("daemon install only supported on macOS (launchd)")

    log_dir = os.path.expanduser("~/.localmelo/logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(PLIST_DIR, exist_ok=True)

    python = sys.executable
    module_args = ["localmelo", "--serve", "--port", str(port)]

    plist = _plist_xml(python, module_args, log_dir)
    with open(PLIST_PATH, "w") as f:
        f.write(plist)

    # unload if already loaded, ignore errors
    subprocess.run(
        ["launchctl", "bootout", f"gui/{os.getuid()}", PLIST_PATH],
        capture_output=True,
    )
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", PLIST_PATH],
        check=True,
    )

    return PLIST_PATH


def uninstall() -> bool:
    if not os.path.exists(PLIST_PATH):
        return False
    subprocess.run(
        ["launchctl", "bootout", f"gui/{os.getuid()}", PLIST_PATH],
        capture_output=True,
    )
    os.remove(PLIST_PATH)
    return True


def status() -> dict[str, object]:
    result = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"installed": False, "running": False}

    running = "state = running" in result.stdout
    pid = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("pid = "):
            pid = int(line.split("=")[1].strip())
            break

    return {"installed": True, "running": running, "pid": pid}
