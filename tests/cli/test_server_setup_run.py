"""The first-run setup job (support/server_setup/run.py), with fake ``uv`` and ``melo-serve``.

Nothing here installs, downloads or starts anything: both commands are scripts
in a temporary folder that log their argv, and the login items are not touched.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from localmelo.support import config, melo_serve
from localmelo.support.gateway import daemon
from localmelo.support.server_setup import run
from localmelo.tests.fixtures.melo_serve import role, write_state

FIXTURES = Path(__file__).parents[1] / "fixtures"
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


@pytest.fixture
def fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, melo_serve_home: Path
) -> Fakes:
    fake = Fakes(tmp_path / "fake", monkeypatch)
    # melo_serve runs, as the real one does once it is up: this is what ``use`` leaves.
    write_state(
        melo_serve_home,
        chat=role("Qwen3 8B", "omlx", "oMLX"),
        embedding=role("Qwen3-Embedding-0.6B", "omlx", "oMLX"),
    )
    return fake


@pytest.fixture(autouse=True)
def _socket(melo_serve_socket: Path) -> None:
    """A melo_serve that answers on its control socket."""


def _job(
    skip: tuple[str, ...] = (),
    *,
    pick: str = CHAT,
    gateway: bool = False,
    cancel: bool = False,
    on_event: Callable[[run.Event], None] | None = None,
) -> tuple[run.SetupJob, list[run.Event]]:
    """A job for the fixture's plan; with *cancel*, it cancels itself at the first download progress."""
    events: list[run.Event] = []

    def collect(event: run.Event) -> None:
        events.append(event)
        if cancel and isinstance(event, run.ItemEvent) and event.done_bytes:
            job.cancel()
        if on_event:
            on_event(event)

    found = run.plan()
    assert found.available, found.reason
    job = run.SetupJob(found, pick, skip, gateway_service=gateway, on_event=collect)
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
INSTALLED = daemon.Installed(
    "installed",
    "/lm/ai.localmelo.gateway.plist",
    "installed ai.localmelo.gateway (starts at login)",
)


def test_a_full_run_skips_nothing_and_says_what_each_step_did(
    fakes: Fakes, capsys: pytest.CaptureFixture[str]
) -> None:
    # No melo-serve yet: the plan comes from uv, then uv installs it, as asked.
    job, events = _job(gateway=True)
    with mock.patch(GATEWAY, return_value=INSTALLED):
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
            "installed ai.localmelo.gateway (starts at login)",
        ),
        ("load:chat", "done", f"Now serving {CHAT} for chat"),
        ("load:embedding", "done", f"Now serving {EMBED} for chat"),
        ("save", "done", "saved melo_serve as the chat server"),
        ("end", "done", ""),
    ]
    download = [e for e in events if isinstance(e, run.ItemEvent) and e.done_bytes]
    assert [(e.item, e.done_bytes, e.total_bytes) for e in download] == [
        ("backend:omlx", 310, 881),
        (f"model:{CHAT}", 4939212390, 4939212390),
        (f"model:{EMBED}", 1288490189, 1288490189),
    ]
    cfg = config.load()
    assert cfg.chat_backend == cfg.embedding_backend == "melo_serve"
    assert cfg.melo_serve.chat_model == cfg.melo_serve.embedding_model == ""
    assert capsys.readouterr().out == ""  # a job has no screen to print on


SKIPS = [
    # (skipped id, picks loaded, melo-serve's service, gateway's, saved, rows that say skipped)
    ("backend:splash", [EMBED], True, True, False, {"backend:splash", "save"}),
    ("model:" + SPLASH, [EMBED], True, True, False, {"model:" + SPLASH, "save"}),
    ("backend:omlx", [SPLASH], True, True, True, {"backend:omlx"}),
    ("model:" + EMBED, [SPLASH], True, True, True, {"model:" + EMBED}),
    ("service:melo_serve", [SPLASH, EMBED], False, True, True, {"service:melo_serve"}),
    ("service:gateway", [SPLASH, EMBED], True, False, True, {"service:gateway"}),
    ("melo_serve", [SPLASH, EMBED], True, True, True, {"melo_serve"}),
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
    fakes.plan(
        PLAN_SPLASH, GET_SPLASH
    )  # a chat pick on one backend, the embedding on the other
    job, events = _job((skip,), pick=SPLASH, gateway=True)
    with mock.patch(GATEWAY, return_value=INSTALLED) as install:
        assert job.run() == "done"
    assert fakes.calls("uv", "tool", "install") == []
    asked = ["--skip", skip] if ":" in skip and "service" not in skip else []
    assert fakes.calls("melo-serve", "get") == [
        ["get", SPLASH, EMBED, *asked, "--json"]
    ]
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


ALTERNATIVE = [
    # (skipped id, what get is told to leave out, picks loaded, saved): the plan's
    # items describe the Splash pick, but the job is for an oMLX alternative
    ("backend:splash", [], [CHAT, EMBED], True),  # not part of this run: not passed
    ("model:" + SPLASH, [], [CHAT, EMBED], True),
    ("model:" + CHAT, ["model:" + CHAT], [EMBED], False),
    ("backend:omlx", ["backend:omlx"], [], False),  # the embedding is on it too
]


@pytest.mark.parametrize(("skip", "asked", "picks", "saved"), ALTERNATIVE)
def test_a_job_for_an_alternative_pick_skips_what_belongs_to_its_own_run(
    fakes: Fakes, skip: str, asked: list[str], picks: list[str], saved: bool
) -> None:
    fakes.installed("0.3.0")
    fakes.plan(PLAN_SPLASH, GET_SPLASH)
    job, _ = _job((skip,), pick=CHAT)
    assert job.run() == "done"  # the fake get refuses an item that is not of its run
    assert fakes.calls("melo-serve", "get") == [
        ["get", CHAT, EMBED, *(a for item in asked for a in ("--skip", item)), "--json"]
    ]
    assert [c[1] for c in fakes.calls("melo-serve", "use")] == picks
    assert Path(config.config_path()).exists() is saved


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


@pytest.mark.parametrize("skip", [(), ("melo_serve",)])
def test_a_melo_serve_that_stays_below_the_minimum_fails_the_job(
    fakes: Fakes, skip: tuple[str, ...]
) -> None:
    fakes.installed("0.1.0")  # uv's new one is not the one that is found first
    job, events = _job(skip)
    assert job.run() == "failed"
    error = events[-1].error  # type: ignore[union-attr]
    assert "0.1.0" in error and "0.3.0 or later is needed" in error
    assert fakes.calls("melo-serve", "get") == []


def test_cancel_during_get_ends_the_job_cancelled_before_anything_else_runs(
    fakes: Fakes,
) -> None:
    fakes.installed("0.3.0")
    fakes.put("get_hangs", "1")
    job, events = _job(cancel=True)
    assert job.run() == job.state == "cancelled"
    assert events[-1] == run.Finished("cancelled")
    # get reports a stop as a failure; the row is not painted as one
    assert [e for e in events if isinstance(e, run.ItemEvent) and e.item == "get"] == [
        run.ItemEvent(
            "get",
            "skipped",
            "Stopped between steps. Run the same command again to continue.",
        )
    ]
    assert not [
        e for e in events if isinstance(e, run.ItemEvent) and e.state == "failed"
    ]
    assert (
        fakes.calls("melo-serve", "install") == fakes.calls("melo-serve", "use") == []
    )
    assert not Path(config.config_path()).exists()


def test_a_cancel_kills_a_get_that_does_not_stop(
    fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "CANCEL_GRACE_S", 0.5)
    fakes.installed("0.3.0")
    fakes.put("get_deaf", "1")
    job, events = _job(cancel=True)
    started = time.monotonic()
    assert job.run() == "cancelled"
    assert time.monotonic() - started < 20
    assert events[-1] == run.Finished("cancelled")
    assert job.items["download"].state == "skipped"  # get printed nothing about it


def test_ctrl_c_waits_for_get_to_finish_and_ends_the_job_cancelled(
    fakes: Fakes,
) -> None:
    fakes.installed("0.3.0")
    fakes.put(
        "get_sigint", "1"
    )  # the fake sends SIGINT to this process, as a terminal does
    job, events = _job()
    assert job.run() == "cancelled"
    assert job.items["model:kept"].line == "finished what it was doing"
    assert events[-1] == run.Finished("cancelled")
    assert fakes.calls("melo-serve", "use") == []


def test_a_process_get_leaves_behind_neither_holds_the_job_nor_speaks_after_it(
    fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "PIPE_GRACE_S", 0.2)
    fakes.installed("0.3.0")
    fakes.put("get_leaves", "1")  # it keeps the pipe open for a second and then prints
    job, events = _job()
    started = time.monotonic()
    assert job.run() == "done"
    assert time.monotonic() - started < 1.0
    time.sleep(1.5)
    assert events[-1] == run.Finished("done")
    assert "late" not in job.items


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


def _bad_config(fakes: Fakes) -> None:
    path = Path(config.config_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("chat_backend = = oops")


FAILURES = [
    # (setup, what the job's last line says: the reason, then what is true now)
    (
        lambda f: f.put("install_fails", ""),
        "failed: The login item is installed (~/Library/LaunchAgents/x.plist), but "
        "melo_serve did not start; see melo-serve doctor",
    ),
    (
        lambda f: f.put("use_fails", EMBED),
        "failed: model did not load; the chat model is serving; nothing was saved",
    ),
    (
        _bad_config,
        "failed: could not save the config:",  # then the parser's words, then:
    ),
]


@pytest.mark.parametrize(("setup", "line"), FAILURES)
def test_a_failure_after_the_first_steps_says_what_is_left_not_that_nothing_changed(
    fakes: Fakes, setup: Callable[[Fakes], None], line: str
) -> None:
    fakes.installed("0.3.0")
    setup(fakes)
    job, events = _job()
    assert job.run() == job.state == "failed"
    error = events[-1].error  # type: ignore[union-attr]
    assert error.startswith(line) and "nothing was changed" not in error
    if "config" in line:
        assert error.endswith("; the models are loaded; nothing was saved")


def test_a_job_runs_once_and_a_callback_that_raises_stops_nothing(fakes: Fakes) -> None:
    def gone(event: run.Event) -> None:
        raise BrokenPipeError("the web socket closed")

    fakes.installed("0.3.0")
    job, _ = _job(on_event=gone)
    assert job.run() == "done"
    with pytest.raises(RuntimeError, match="runs once"):
        job.run()


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
        (  # more than a pipe holds, which nobody reads to the end
            lambda f: f.put("uv_fails", "flood"),
            "error: network is down",
        ),
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


def test_a_version_that_is_not_one_is_not_trusted_and_the_plan_still_never_raises(
    fakes: Fakes,
) -> None:
    fakes.installed("0.3.0")
    fakes.put("version", "9" * 5000 + ".0.0")  # int() would refuse it
    found = run.plan()  # the installed one is not at the minimum: asked through uv
    assert found.available and fakes.calls("uv", "tool", "run")


def test_text_from_a_child_is_one_line_a_terminal_can_print() -> None:
    assert (
        run._clip("done\nfailed: on fire\t\x1b[2Jx\ud800 ") == "done failed: on fire x?"
    )
    assert len(run._clip("y" * 1000)) == run.MAX_TEXT_CHARS


def test_the_plan_is_data_each_field_is_checked(fakes: Fakes) -> None:
    data = json.loads(PLAN.read_text())
    data["items"][0]["size_bytes"] = "big"
    data["items"][1]["backend"] = 7
    # an id goes into an argv: not one that is an option, with or without its prefix
    data["items"][2]["id"] = "model:--hf-home"
    data["alternatives"][0]["backend"] = 7
    data["pick"]["name"] = "Qwen\n\nInstall this: run.sh"
    data["machine"]["free_gb"] = True
    fakes.put("plan.json", json.dumps(data))
    found = run.plan()
    assert found.available
    assert found.pick == run.Pick(CHAT, "Qwen Install this: run.sh", "omlx")
    assert [(i.id, i.backend) for i in found.items] == [
        ("backend:omlx", ""),
        ("model:" + CHAT, ""),
        ("model:--hf-home", "omlx"),
    ]
    assert (
        found.items[0].size_bytes == 0
        and found.alternatives == ()
        and found.machine.free_gb == 0.0
    )
    job = run.SetupJob(found, CHAT)  # the embedding model is dropped, not passed on
    assert job._picks == [CHAT]
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

    home, on_path, beside, usual = (
        tmp_path / "home",
        tmp_path / "path",
        tmp_path / "venv" / "bin",
        tmp_path / "usual",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(on_path))
    monkeypatch.setattr(sys, "executable", str(beside / "python"))
    # The usual folders of this Mac are not under test: a uv of the user's there would be found.
    monkeypatch.setattr(run, "UV_FOLDERS", ("~/.local/bin", str(usual)))
    assert run.find_uv() is None
    last = uv(usual)
    assert run.find_uv() == last
    local = uv(home / ".local" / "bin")
    assert run.find_uv() == local
    found = uv(on_path)
    assert run.find_uv() == found
    here = uv(beside)
    assert run.find_uv() == here
    (beside / "uv").chmod(0o777)  # others could change it: passed over
    assert run.find_uv() == found
