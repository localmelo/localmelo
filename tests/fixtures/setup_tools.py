"""Fake ``uv`` and ``melo-serve`` for the tests of the first-run setup job.

Nothing here installs, downloads or starts anything: both commands are one script
in a temporary folder that logs its argv and answers from files there. The
terminal's job tests (``tests/cli/test_server_setup_run.py``) and the console's
setup card (``tests/gateway/console/test_setup_flow.py``) run the real job over it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from localmelo.support import melo_serve

FIXTURES = Path(__file__).parent
# The plan of a 32 GB Mac (an oMLX chat pick, the embedding model on oMLX too), and
# of a 128 GB Mac whose Splash model is already there (a chat pick on one backend,
# the embedding on the other).
PLAN = FIXTURES / "melo_serve_plan.v1.json"
PLAN_SPLASH = FIXTURES / "melo_serve_plan_splash.v1.json"
CHAT, EMBED, SPLASH = "qwen3-8b-omlx", "qwen3-embedding-0.6b-omlx", "qwen3.8-27b-splash"

# One script plays both tools: ``uv`` (named so) and ``melo-serve``. What it
# answers is read from files in $FAKE_DIR, and each call is logged there. Its
# ``get`` refuses a --skip that is not an item of its run, as the real one does
# (exit 2), and says what it was stopped by in the real words.
FAKE = f"#!{sys.executable}\n" + r"""
import json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

d = Path(os.environ["FAKE_DIR"])
name, args = os.path.basename(sys.argv[0]), sys.argv[1:]
with open(d / "log", "a") as log:
    log.write(json.dumps([name, *args]) + "\n")
flag = lambda key: (d / key).exists()
text = lambda key, default="": (d / key).read_text().strip() if flag(key) else default
say = lambda **event: print(json.dumps(event), flush=True)

if name == "uv":
    if flag("uv_fails"):
        if text("uv_fails") == "flood":  # more than a pipe holds
            sys.stderr.write(("y" * 99 + "\n") * 3000)
        print("error: network is down", file=sys.stderr)
        sys.exit(1)
    if args[:2] == ["tool", "install"] or args[:3] == ["tool", "install", "--force"]:
        (d / "version").write_text(text("next_version", "0.4.0"))
        tool = d / "tools" / "melo-serve" / "bin"
        tool.mkdir(parents=True, exist_ok=True)
        for folder in (tool, tool.parent, tool.parent.parent):
            folder.chmod(0o755)
        shutil.copy(__file__, tool / "melo-serve")
        (tool / "melo-serve").chmod(0o755)
        link = Path(os.environ["UV_TOOL_BIN_DIR"]) / "melo-serve"
        link.unlink(missing_ok=True)
        link.symlink_to(tool / "melo-serve")
    elif args[:2] == ["tool", "run"]:
        print(text("plan.json"))
else:
    if args == ["--version"]:
        print("melo-serve " + text("version"))
    elif args[0] == "plan":
        print(text("plan.json"))
    elif args[0] == "get":
        plan = json.loads(text("plan.json"))
        backend = {p["id"]: p["backend"] for p in [plan["pick"], *plan["alternatives"]]}
        backend.update({i["id"][6:]: i["backend"] for i in plan["items"] if i["kind"] == "model"})
        picks, skips, rest = [], [], iter(args[1:])
        for arg in rest:
            if arg == "--skip":
                skips.append(next(rest))
            elif arg != "--json":
                picks.append(arg)
        known = {f"model:{p}" for p in picks} | {f"backend:{backend[p]}" for p in picks}
        for item in skips:
            if item not in known:
                say(item="get", state="failed", line=f"failed: {item} is not an item of this run; nothing was downloaded")
                sys.exit(2)
        for item in skips:
            say(item=item, state="skipped", line="skipped (you chose to skip it)")
        stopped = lambda *_: (say(item="get", state="failed", line="Stopped between steps. Run the same command again to continue."), sys.exit(130))
        if flag("get_hangs"):  # until a TERM stops it
            signal.signal(signal.SIGTERM, stopped)
        if flag("get_deaf"):  # until a KILL does
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for line in text("get.jsonl").splitlines():
            if json.loads(line)["item"] not in skips:
                print(line, flush=True)
        if flag("get_hangs") or flag("get_deaf"):
            time.sleep(60)
        if flag("get_leaves"):  # a process of its own, which keeps the pipe and speaks later
            late = json.dumps({"item": "late", "state": "done"})
            subprocess.Popen([sys.executable, "-c", f"import time; time.sleep(1); print({late!r}, flush=True)"])
        if flag("get_sigint"):  # Ctrl-C reaches the job; this goes on a moment
            time.sleep(1)
            os.kill(os.getppid(), signal.SIGINT)
            time.sleep(0.5)
            say(item="model:kept", state="done", line="finished what it was doing")
        sys.exit(int(text("get_exit", "0")))
    elif args[0] == "install":
        if flag("install_fails"):
            print("The login item is installed (~/Library/LaunchAgents/x.plist), but melo_serve did not start; see melo-serve doctor", file=sys.stderr)
            sys.exit(1)
        print("Registering...\ninstalled ai.localmelo.melo_serve (starts at login)")
    elif args[0] == "use":
        if text("use_fails") == args[1]:
            print("model did not load", file=sys.stderr)
            sys.exit(1)
        print("Switching to " + args[1] + "...\nNow serving " + args[1] + " for chat")
"""


def _done(item: str, line: str, size: int = 0) -> dict[str, Any]:
    return {
        "item": item,
        "state": "done",
        "line": line,
        "done_bytes": size,
        "total_bytes": size,
    }


# What ``melo-serve get`` prints for each plan: backends first, then the models it
# leaves, then the downloads, with the closing lines it has.
OMLX_RUNNING = {
    "item": "backend:omlx",
    "state": "running",
    "done_bytes": 310,
    "total_bytes": 881,
    "line": "downloading oMLX 0.7.3",
}
OMLX_DONE = {
    "item": "backend:omlx",
    "state": "done",
    "line": "installed 0.7.3 into ~/Applications/oMLX.app",
}
GET: list[dict[str, Any]] = [
    OMLX_RUNNING,
    OMLX_DONE,
    _done(f"model:{CHAT}", "downloaded Qwen3 8B into ~/models/huggingface", 4939212390),
    _done(
        f"model:{EMBED}",
        "downloaded Qwen3 Embedding 0.6B into ~/models/huggingface",
        1288490189,
    ),
]
GET_SPLASH: list[dict[str, Any]] = [
    {
        "item": "backend:splash",
        "state": "skipped",
        "line": "already installed (1.5.0), skipped",
    },
    OMLX_RUNNING,
    OMLX_DONE,
    {
        "item": f"model:{SPLASH}",
        "state": "skipped",
        "line": "already downloaded, skipped",
    },
    _done(
        f"model:{EMBED}",
        "downloaded Qwen3 Embedding 0.6B into ~/models/huggingface",
        1288490189,
    ),
]


class Fakes:
    """The folder of fake tools: ``bin`` (on PATH) with ``uv``, and the uv tool folder."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.dir = root
        self.tool_bin = root / "uv-bin"
        self.bin = root / "bin"
        for folder in (root, self.bin, self.tool_bin):
            folder.mkdir(parents=True, exist_ok=True)
            folder.chmod(0o755)
        self.script(self.bin / "uv")
        monkeypatch.setenv("FAKE_DIR", str(root))
        monkeypatch.setenv("UV_TOOL_BIN_DIR", str(self.tool_bin))
        monkeypatch.setenv("PATH", f"{self.bin}:/usr/bin:/bin")
        monkeypatch.setattr(melo_serve, "command", melo_serve._find_command)
        self.plan(PLAN, GET)

    def plan(self, plan: Path, get: list[dict[str, Any]]) -> None:
        """The plan ``melo-serve plan`` prints, and what ``melo-serve get`` prints."""
        self.put("plan.json", plan.read_text())
        self.put("get.jsonl", "\n".join(json.dumps(event) for event in get))

    @staticmethod
    def script(path: Path) -> None:
        path.write_text(FAKE)
        path.chmod(0o755)

    def put(self, key: str, text: str) -> None:
        (self.dir / key).write_text(text)

    def installed(self, version: str) -> None:
        """A melo-serve of *version* that ``uv tool install`` left (as the fake uv would)."""
        self.put("next_version", version)
        subprocess.run([self.bin / "uv", "tool", "install", "x"], check=True)
        (self.dir / "log").unlink()

    def calls(self, tool: str, *first: str) -> list[list[str]]:
        log = self.dir / "log"
        entries = (
            [json.loads(line) for line in log.read_text().splitlines()]
            if log.exists()
            else []
        )
        return [
            e[1:]
            for e in entries
            if e[0] == tool and e[1 : 1 + len(first)] == list(first)
        ]
