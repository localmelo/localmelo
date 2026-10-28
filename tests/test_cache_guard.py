"""Tests for the real-cache guards (``tests/_cache_guard.py``)."""

from __future__ import annotations

import builtins
import os
from pathlib import Path

import pytest

from localmelo.tests import _cache_guard


def _fake_home(tmp_path: Path) -> Path:
    home = tmp_path / "fake-home"
    (home / ".cache" / "localmelo" / "scheduler").mkdir(parents=True, exist_ok=True)
    return home


def _guard(tmp_path: Path) -> _cache_guard.RealCacheGuard:
    home = _fake_home(tmp_path)
    return _cache_guard.RealCacheGuard(_cache_guard.real_cache_dir(home), home=home)


def _run_dir(tmp_path: Path) -> Path:
    return _fake_home(tmp_path) / ".cache" / "localmelo" / "scheduler" / "runs" / "r1"


def test_a_write_under_the_real_cache_is_refused(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    with pytest.raises(_cache_guard.RealCacheError):
        guard.check(_run_dir(tmp_path), "create a directory")
    with pytest.raises(_cache_guard.RealCacheError):
        guard.check(guard.root, "write")  # the cache root itself
    guard.check(tmp_path / "elsewhere", "write")  # a temporary path is fine


def test_every_write_operation_is_guarded(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    target = _run_dir(tmp_path)
    operations = [
        lambda: guard.mkdir(target),
        lambda: guard.os_open(target, os.O_CREAT | os.O_WRONLY),
        lambda: guard.rename(target, target.with_suffix(".moved")),
        lambda: guard.replace(target, target.with_suffix(".replaced")),
        lambda: guard.remove(target),
        lambda: guard.unlink(target),
        lambda: guard.rmdir(target),
        lambda: guard.symlink(target, target.with_suffix(".link")),
        lambda: guard.utime(target),
        lambda: guard.chmod(target),
        lambda: guard.connect(target / "sessions.db"),
    ]
    for operation in operations:
        with pytest.raises(_cache_guard.RealCacheError):
            operation()


def test_an_opt_out_never_allows_the_cache(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    guard.allow_real_home = True
    with pytest.raises(_cache_guard.RealCacheError):
        guard.check(_run_dir(tmp_path), "write")


def test_reads_and_memory_databases_pass(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    with pytest.raises(FileNotFoundError):
        guard.open(guard.root / "missing.txt")
    connection = guard.connect(":memory:")
    connection.close()
    out = tmp_path / "out.txt"
    with guard.open(out, "w", encoding="utf-8") as handle:
        handle.write("x")
    assert out.read_text(encoding="utf-8") == "x"


def test_resolving_the_cache_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = _guard(tmp_path)
    monkeypatch.setenv("HOME", str(guard.home))
    with pytest.raises(_cache_guard.RealCacheError):
        guard.expanduser("~/.cache/localmelo/scheduler")
    assert guard.expanduser("~/notes") == str(guard.home / "notes")


def test_the_real_home_is_refused_only_while_watched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = _guard(tmp_path)
    monkeypatch.setenv("HOME", str(guard.home))

    assert guard.expanduser("~") == str(guard.home)  # not watching: allowed
    guard.watch_home = True
    with pytest.raises(_cache_guard.RealCacheError):
        guard.expanduser("~")
    guard.allow_real_home = True  # the marker opt-out
    assert guard.expanduser("~") == str(guard.home)


def test_install_refuses_writes_and_uninstall_restores(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    original_open = builtins.open
    original_expanduser = os.path.expanduser
    guard.install()
    try:
        assert builtins.open is not original_open
        assert os.path.expanduser is not original_expanduser
        with pytest.raises(_cache_guard.RealCacheError):
            open(_run_dir(tmp_path) / "calls.jsonl", "w")  # noqa: SIM115
        with pytest.raises(_cache_guard.RealCacheError):
            (guard.root / "new.txt").write_text("x", encoding="utf-8")
    finally:
        guard.uninstall()
    assert builtins.open is original_open
    assert os.path.expanduser is original_expanduser
    assert not (guard.root / "new.txt").exists()
