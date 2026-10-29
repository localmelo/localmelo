"""Direct runs are recorded as worker sessions by default, without changing the run."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from localmelo.tests.fixtures import stub_llm

MELO = [sys.executable, "-m", "localmelo"]


@pytest.fixture
def stub() -> Iterator[stub_llm.StubServer]:
    server = stub_llm.StubServer().start()
    try:
        yield server
    finally:
        server.close()


def _run(
    home: Path,
    stub: stub_llm.StubServer,
    *args: str,
    task: str = "say hello",
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    home.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALMELO_")}
    env["HOME"] = str(home)
    base = stub.base_url.rsplit("/v1", 1)[0]
    return subprocess.run(
        [
            *MELO,
            "--base-url",
            f"{base}/s/answer/v1",
            "--chat-model",
            "stub-model",
            *args,
            "--",
            task,
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd or home,
        timeout=120,
    )


def _store(home: Path) -> Path:
    return home / ".cache/localmelo/cli/sessions.db"


def _workers(home: Path) -> list[tuple[str, str, str]]:
    if not _store(home).exists():
        return []
    with sqlite3.connect(_store(home)) as conn:
        return conn.execute(
            "SELECT s.kind, s.title, t.status FROM sessions s JOIN turns t "
            "ON t.session_id = s.id ORDER BY s.created_at"
        ).fetchall()


def test_a_plain_task_is_recorded_without_changing_the_run(
    tmp_path: Path, stub: stub_llm.StubServer
) -> None:
    recorded = _run(tmp_path / "a", stub, "--output-format", "json")
    first = [dict(r) for r in stub.requests]
    stub.requests.clear()
    plain = _run(tmp_path / "b", stub, "--output-format", "json", "--no-save-session")
    assert first == stub.requests, "the model sees the same requests"
    assert recorded.returncode == plain.returncode == 0

    def record(out: str) -> dict[str, object]:
        data = json.loads(out)
        for key in ("task_id", "duration_s", "localmelo_version"):
            data.pop(key, None)
        return data  # type: ignore[no-any-return]

    assert record(recorded.stdout) == record(plain.stdout)
    assert recorded.stderr == plain.stderr == ""
    assert _workers(tmp_path / "a") == [("worker", "say hello", "completed")]
    assert not _store(tmp_path / "b").exists()


def test_a_brief_run_is_recorded(tmp_path: Path, stub: stub_llm.StubServer) -> None:
    brief = tmp_path / "brief.md"
    brief.write_text("Say hello from a brief\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALMELO_")}
    env["HOME"] = str(tmp_path)
    base = stub.base_url.rsplit("/v1", 1)[0]
    done = subprocess.run(
        [*MELO, "--base-url", f"{base}/s/answer/v1", "--chat-model", "m"]
        + ["--brief", str(brief)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert _workers(tmp_path) == [("worker", "Say hello from a brief", "completed")]


def test_the_config_switch_and_the_flag_keep_a_run_out_of_the_store(
    tmp_path: Path, stub: stub_llm.StubServer
) -> None:
    flagged = _run(tmp_path / "flag", stub, "--no-save-session")
    assert flagged.returncode == 0
    assert not _store(tmp_path / "flag").exists()

    home = tmp_path / "config"
    (home / ".cache/localmelo").mkdir(parents=True)
    (home / ".cache/localmelo/config.toml").write_text(
        "[sessions]\nrecord_direct = false\n", encoding="utf-8"
    )
    assert _run(home, stub).returncode == 0
    assert not _store(home).exists()
    # An explicit title or parent run records anyway.
    assert _run(home, stub, "--save-session", "Named").returncode == 0
    assert _workers(home) == [("worker", "Named", "completed")]


def test_a_store_that_cannot_be_written_warns_and_leaves_the_run_unchanged(
    tmp_path: Path, stub: stub_llm.StubServer
) -> None:
    ok = _run(tmp_path / "ok", stub, "--no-save-session")
    home = tmp_path / "locked"
    cache = home / ".cache/localmelo"
    cache.mkdir(parents=True)
    cache.chmod(0o500)
    try:
        if os.access(cache, os.W_OK):
            pytest.skip("directory permissions are not enforced here")
        blocked = _run(home, stub)
    finally:
        cache.chmod(0o700)
    assert blocked.returncode == ok.returncode == 0
    assert blocked.stdout == ok.stdout
    lines = blocked.stderr.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("melo: could not record this run (")
    assert lines[0].endswith("); the run is unchanged")


def test_concurrent_runs_all_record(tmp_path: Path, stub: stub_llm.StubServer) -> None:
    procs = []
    base = stub.base_url.rsplit("/v1", 1)[0]
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALMELO_")}
    env["HOME"] = str(tmp_path)
    for n in range(5):
        procs.append(
            subprocess.Popen(
                [*MELO, "--base-url", f"{base}/s/answer/v1", "--chat-model", "m"]
                + ["--", f"task number {n}"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=tmp_path,
            )
        )
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err
        assert err == ""
    titles = sorted(title for _, title, _ in _workers(tmp_path))
    assert titles == [f"task number {n}" for n in range(5)]


@pytest.mark.parametrize(
    "argv",
    [
        ["--no-save-session", "--save-session", "T", "--", "task"],
        ["--no-save-session", "--parent-run", "p", "--", "task"],
        ["--parent-run", "p"],
    ],
)
def test_conflicting_recording_flags_are_usage_errors(argv: list[str]) -> None:
    done = subprocess.run(
        [*MELO, *argv], capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    assert done.returncode == 2


def test_the_config_round_trips_record_direct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localmelo.support import config

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = config.load()
    assert cfg.sessions.record_direct is True
    cfg.sessions.record_direct = False
    config.save(cfg)
    assert "record_direct = false" in Path(config.config_path()).read_text("utf-8")
    assert config.load().sessions.record_direct is False
