"""The first-run setup job (support/server_setup/run.py), with fake ``uv`` and ``melo-serve``.

Nothing here installs, downloads or starts anything: both commands are scripts
in a temporary folder that log their argv, and the login items are not touched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from localmelo.support import config, melo_serve
from localmelo.support.server_setup import run
from localmelo.tests.fixtures.melo_serve import role, write_state

PLAN = Path(__file__).parents[1] / "fixtures" / "melo_serve_plan.v1.json"
CHAT, EMBED = "qwen3.8-27b-splash", "qwen3-embedding-0.6b-omlx"

# One script plays both tools: ``uv`` (named so) and ``melo-serve``. What it
# answers is read from files in $FAKE_DIR, and each call is logged there.
FAKE = f"""#!{sys.executable}
import json, os, shutil, signal, sys, time
from pathlib import Path
from typing import Any

d = Path(os.environ["FAKE_DIR"])
name, args = os.path.basename(sys.argv[0]), sys.argv[1:]
with open(d / "log", "a") as log:
    log.write(json.dumps([name, *args]) + "\\n")
flag = lambda key: (d / key).exists()
text = lambda key, default="": (d / key).read_text().strip() if flag(key) else default

if name == "uv":
    if flag("uv_fails"):
        print("error: network is down", file=sys.stderr)
        sys.exit(1)
    if args[:2] == ["tool", "install"] or args[:3] == ["tool", "install", "--force"]:
        (d / "version").write_text(text("next_version", "0.3.0"))
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
        skips = [args[i + 1] for i, arg in enumerate(args) if arg == "--skip"]
        for item in skips:
            print(json.dumps({{"item": item, "state": "skipped", "line": "skipped (you chose to skip it)"}}), flush=True)
        for line in text("get.jsonl").splitlines():
            if json.loads(line)["item"] not in skips:
                print(line, flush=True)
        if flag("get_hangs"):
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(130))
            time.sleep(60)
        sys.exit(int(text("get_exit", "0")))
    elif args[0] == "install":
        print("Registering...\\ninstalled ai.localmelo.melo_serve (starts at login)")
    elif args[0] == "use":
        print("Switching to " + args[1] + "...\\nNow serving " + args[1] + " for chat")
"""

GET: list[dict[str, Any]] = [
    {
        "item": "backend:splash",
        "state": "skipped",
        "line": "already installed (1.5.0), skipped",
    },
    {
        "item": "backend:omlx",
        "state": "running",
        "done_bytes": 310,
        "total_bytes": 881,
        "line": "downloading oMLX 0.7.3",
    },
    {
        "item": "backend:omlx",
        "state": "done",
        "line": "installed 0.7.3 into ~/Applications/oMLX.app",
    },
    {
        "item": "model:qwen3.8-27b-splash",
        "state": "done",
        "line": "installed Qwen3.8 27B into the cache",
    },
    {
        "item": "model:qwen3-embedding-0.6b-omlx",
        "state": "done",
        "line": "already downloaded, skipped",
    },
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
        self.put("plan.json", PLAN.read_text())
        self.put("get.jsonl", "\n".join(json.dumps(event) for event in GET))

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


@pytest.fixture
def fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, melo_serve_home: Path
) -> Fakes:
    fake = Fakes(tmp_path / "fake", monkeypatch)
    # melo_serve runs, as the real one does once it is up: this is what ``use`` leaves.
    write_state(
        melo_serve_home,
        chat=role("Qwen3.8-27B"),
        embedding=role("Qwen3-Embedding-0.6B", "omlx", "oMLX"),
    )
    return fake


@pytest.fixture(autouse=True)
def _socket(melo_serve_socket: Path) -> None:
    """A melo_serve that answers on its control socket."""


def _job(
    skip: tuple[str, ...] = (), *, gateway: bool = False, cancel: bool = False
) -> tuple[run.SetupJob, list[run.Event]]:
    """A job for the fixture's plan; with *cancel*, it cancels itself at the first download progress."""
    events: list[run.Event] = []

    def collect(event: run.Event) -> None:
        events.append(event)
        if cancel and isinstance(event, run.ItemEvent) and event.done_bytes:
            job.cancel()

    found = run.plan()
    assert found.available, found.reason
    job = run.SetupJob(found, CHAT, skip, gateway_service=gateway, on_event=collect)
    return job, events


def _lines(events: list[run.Event]) -> list[tuple[str, str, str]]:
    """``(row, state, line)`` of every event that has a line, then the end."""
    rows = [
        (e.item, e.state, e.line)
        for e in events
        if isinstance(e, run.ItemEvent) and e.line
    ]
    return [*rows, ("end", events[-1].state, events[-1].error)]  # type: ignore[union-attr]


GATEWAY = "localmelo.support.gateway.daemon.install"


def test_a_full_run_skips_nothing_and_says_what_each_step_did(fakes: Fakes) -> None:
    # No melo-serve yet: the plan comes from uv, then uv installs it, as asked.
    job, events = _job(gateway=True)
    with mock.patch(GATEWAY, return_value="/lm/ai.localmelo.gateway.plist"):
        assert job.run() == "done"
    tool = (fakes.dir / "tools" / "melo-serve").resolve()
    url = melo_serve.ARCHIVE_URL
    assert fakes.calls("uv", "tool", "run")[0] == [
        "tool",
        "run",
        "--python",
        sys.executable,
        "--from",
        url,
        "melo-serve",
        "plan",
        "--json",
    ]
    assert fakes.calls("uv", "tool", "install") == [
        ["tool", "install", "--managed-python", url]
    ]
    assert fakes.calls("melo-serve", "get") == [["get", CHAT, EMBED, "--json"]]
    assert fakes.calls("melo-serve", "use") == [["use", CHAT], ["use", EMBED]]
    assert _lines(events) == [
        ("melo_serve", "done", f"installed melo_serve 0.3.0 into {tool}"),
        *((e["item"], e["state"], e["line"]) for e in GET),
        (
            "service:melo_serve",
            "done",
            "installed ai.localmelo.melo_serve (starts at login)",
        ),
        (
            "service:gateway",
            "done",
            "installed gateway service into /lm/ai.localmelo.gateway.plist",
        ),
        ("load:chat", "done", f"Now serving {CHAT} for chat"),
        ("load:embedding", "done", f"Now serving {EMBED} for chat"),
        ("save", "done", "saved melo_serve as the chat server"),
        ("end", "done", ""),
    ]
    download = [e for e in events if isinstance(e, run.ItemEvent) and e.done_bytes]
    assert [(e.item, e.done_bytes, e.total_bytes) for e in download] == [
        ("backend:omlx", 310, 881)
    ]
    cfg = config.load()
    assert cfg.chat_backend == cfg.embedding_backend == "melo_serve"
    assert cfg.melo_serve.chat_model == cfg.melo_serve.embedding_model == ""


SKIPS = [
    # (skipped id, picks loaded, melo-serve's service, gateway's, saved, rows that say skipped)
    ("backend:splash", [EMBED], True, True, False, {"backend:splash", "save"}),
    ("model:" + CHAT, [EMBED], True, True, False, {"model:" + CHAT, "save"}),
    ("backend:omlx", [CHAT], True, True, True, {"backend:omlx"}),
    ("model:" + EMBED, [CHAT], True, True, True, {"model:" + EMBED}),
    ("service:melo_serve", [CHAT, EMBED], False, True, True, {"service:melo_serve"}),
    ("service:gateway", [CHAT, EMBED], True, False, True, {"service:gateway"}),
    ("melo_serve", [CHAT, EMBED], True, True, True, {"melo_serve"}),
]


@pytest.mark.parametrize(
    ("skip", "picks", "service", "gateway", "saved", "rows"), SKIPS
)
def test_each_skip_leaves_out_exactly_its_step(
    fakes: Fakes,
    skip: str,
    picks: list[str],
    service: bool,
    gateway: bool,
    saved: bool,
    rows: set[str],
) -> None:
    fakes.installed("0.3.0")  # at the minimum: melo_serve itself is left as it is
    job, events = _job((skip,), gateway=True)
    with mock.patch(GATEWAY, return_value="/lm/g.plist") as install:
        assert job.run() == "done"
    assert fakes.calls("uv", "tool", "install") == []
    asked = ["--skip", skip] if ":" in skip and "service" not in skip else []
    assert fakes.calls("melo-serve", "get") == [["get", CHAT, EMBED, *asked, "--json"]]
    assert [c[1] for c in fakes.calls("melo-serve", "use")] == picks
    assert bool(fakes.calls("melo-serve", "install")) is service
    assert install.called is gateway
    chosen = {i for i, e in job.items.items() if e.line == run.SKIPPED}
    assert chosen == rows
    if skip != "melo_serve":
        assert job.items["melo_serve"].line == "already installed (0.3.0), skipped"
    path = Path(config.config_path())
    assert path.exists() is saved
    if saved:  # without the embedding model, memory goes by words only
        assert config.load().embedding_backend == (
            "melo_serve" if EMBED in picks else "none"
        )


def test_a_melo_serve_below_the_minimum_is_updated_with_force(fakes: Fakes) -> None:
    fakes.installed("0.1.0")
    fakes.put("next_version", "0.3.0")
    job, events = _job()
    assert job.run() == "done"
    assert fakes.calls("uv", "tool", "run")  # 0.1.0 has no `plan`: asked through uv
    assert fakes.calls("uv", "tool", "install") == [
        ["tool", "install", "--force", "--managed-python", melo_serve.ARCHIVE_URL]
    ]
    assert _lines(events)[0] == (
        "melo_serve",
        "done",
        "updated melo_serve 0.1.0 → 0.3.0",
    )


def test_cancel_during_get_ends_the_job_cancelled_before_anything_else_runs(
    fakes: Fakes,
) -> None:
    fakes.installed("0.3.0")
    fakes.put("get_hangs", "1")
    job, events = _job(cancel=True)
    assert job.run() == job.state == "cancelled"
    assert events[-1] == run.Finished("cancelled")
    assert not [
        e for e in events if isinstance(e, run.ItemEvent) and e.state == "failed"
    ]
    assert (
        fakes.calls("melo-serve", "install") == fakes.calls("melo-serve", "use") == []
    )
    assert not Path(config.config_path()).exists()


def test_a_failed_get_stops_the_job_with_the_reason(fakes: Fakes) -> None:
    fakes.installed("0.3.0")
    fakes.put("get_exit", "2")
    fakes.put(
        "get.jsonl",
        json.dumps(
            {
                "item": "model:" + CHAT,
                "state": "failed",
                "line": "not enough space: needs 22.4 GB, 18.0 GB free",
            }
        ),
    )
    job, events = _job()
    assert job.run() == "failed"
    assert _lines(events)[-1] == (
        "end",
        "failed",
        "not enough space: needs 22.4 GB, 18.0 GB free",
    )
    assert fakes.calls("melo-serve", "install") == []


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        (
            lambda f: f.put("plan.json", "not json"),
            "melo-serve plan did not answer with JSON",
        ),
        (
            lambda f: f.put("plan.json", "[" + " " * 300_000 + "]"),
            "uv printed more than 256 KB",
        ),
        (lambda f: f.put("uv_fails", ""), "error: network is down"),
        (
            lambda f: f.put("plan.json", json.dumps({"schema": "other/1"})),
            "melo-serve plan did not answer with a plan",
        ),
    ],
)
def test_a_plan_that_cannot_be_read_is_unavailable_with_one_line(
    fakes: Fakes, setup: Callable[[Fakes], None], reason: str
) -> None:
    setup(fakes)
    found = run.plan()
    assert (found.available, found.reason, found.pick) == (False, reason, None)


def test_a_plan_without_uv_is_unavailable(
    fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "find_uv", lambda: None)
    assert run.plan() == run.Plan(False, run.NO_UV)


def test_the_plan_is_data_each_field_is_checked(fakes: Fakes) -> None:
    data = json.loads(PLAN.read_text())
    data["items"][0]["size_bytes"] = "big"
    data["items"][1][
        "id"
    ] = "--hf-home"  # an id goes into an argv: not one that is an option
    data["alternatives"][0]["backend"] = 7
    data["machine"]["free_gb"] = True
    fakes.put("plan.json", json.dumps(data))
    found = run.plan()
    assert found.available and found.pick == run.Pick(CHAT, "Qwen3.8 27B", "splash")
    assert [i.id for i in found.items] == [
        "backend:splash",
        "model:" + CHAT,
        "model:" + EMBED,
    ]
    assert (
        found.items[0].size_bytes == 0
        and found.alternatives == ()
        and found.machine.free_gb == 0.0
    )
    data["pick"]["id"] = "--evil"
    fakes.put("plan.json", json.dumps(data))
    assert not run.plan().available


def test_uv_is_found_next_to_the_interpreter_then_on_path_then_in_the_usual_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def uv(folder: Path) -> str:
        folder.mkdir(parents=True, mode=0o755, exist_ok=True)
        folder.chmod(0o755)
        Fakes.script(folder / "uv")
        return str((folder / "uv").resolve())

    home, on_path, beside = (
        tmp_path / "home",
        tmp_path / "path",
        tmp_path / "venv" / "bin",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(on_path))
    monkeypatch.setattr(sys, "executable", str(beside / "python"))
    # The real /opt/homebrew/bin (the user's, group admin, mode 775) is not under test.
    monkeypatch.setattr(melo_serve, "_admin_gid", lambda: None)
    assert run.find_uv() is None
    local = uv(home / ".local" / "bin")
    assert run.find_uv() == local
    found = uv(on_path)
    assert run.find_uv() == found
    assert run.find_uv() != local
    here = uv(beside)
    assert run.find_uv() == here
    (beside / "uv").chmod(0o777)  # others could change it: passed over
    assert run.find_uv() == found
