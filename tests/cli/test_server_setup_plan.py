"""Setup planning reads only metadata and never grants execution permission."""

from __future__ import annotations

import builtins
import hashlib
import json
import shlex
import socket
import subprocess
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.support.backends import discovery
from localmelo.support.server_setup import plan as setup

_REAL_DISK = setup._disk
_REAL_RAM = setup.ram_bytes


@pytest.fixture(autouse=True)
def machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(setup.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(setup.platform, "mac_ver", lambda: ("27.2", (), ""))
    monkeypatch.setattr(setup, "ram_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(setup, "_disk", lambda path: ("/volume", 10_000_000_000))


def test_recipe_and_plan_are_pinned_frozen_and_digest_is_canonical() -> None:
    detected = setup.detect(existing_servers=["http://example.invalid/v1"])
    planned = setup.plan(detected)
    r = planned.recipe
    assert planned.version == 1 and not planned.keep_running
    assert planned.allowed_operations == ()
    assert sum(n for _, n in r.files) == 2278972236
    assert len(r.files) == 13
    assert len(r.runtime_commit) == len(r.model_revision) == 40
    assert all(len(asset.sha256) == 64 for asset in r.assets)
    assert detected.existing_servers == ("http://example.invalid/v1",)
    manifest = json.dumps(asdict(planned), sort_keys=True, separators=(",", ":"))
    assert planned.digest == hashlib.sha256(manifest.encode()).hexdigest()
    assert planned == setup.plan(
        setup.detect(existing_servers=detected.existing_servers)
    )
    assert planned.digest != replace(planned, keep_running=True).digest
    assert planned.digest != replace(planned, version=2).digest
    assert (
        planned.digest
        != replace(planned, recipe=replace(r, runtime_commit="a" * 40)).digest
    )
    for instance, field, value in (
        (planned, "keep_running", True),
        (r, "runtime_version", "next"),
        (r.assets[0], "bytes", 0),
        (detected, "support", "yes"),
        (detected.disks[0], "free_bytes", 0),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(instance, field, value)
    assert isinstance(r.files, tuple) and isinstance(r.files[0], tuple)
    assert isinstance(r.assets, tuple) and isinstance(r.assets[0].macos, tuple)
    assert isinstance(detected.disks[0].paths, tuple)


@pytest.mark.parametrize(
    ("env", "suffix"),
    [
        ({}, "home/.cache/huggingface/hub"),
        ({"XDG_CACHE_HOME": "xdg"}, "xdg/huggingface/hub"),
        ({"HF_HOME": "hf", "XDG_CACHE_HOME": "xdg"}, "hf/hub"),
        ({"HF_HUB_CACHE": "hub", "HF_HOME": "hf"}, "hub"),
    ],
)
def test_hf_cache_precedence(tmp_path: Path, env: dict[str, str], suffix: str) -> None:
    absolute = {name: str(tmp_path / path) for name, path in env.items()}
    assert setup.cache_root(absolute, tmp_path / "home") == tmp_path / suffix


@pytest.mark.parametrize(
    ("version", "asset_index"), [("15.7", 0), ("26.1", 1), ("27.2", 1)]
)
def test_only_verified_macos_assets_are_selected(
    monkeypatch: pytest.MonkeyPatch, version: str, asset_index: int
) -> None:
    monkeypatch.setattr(setup.platform, "mac_ver", lambda: (version, (), ""))
    detected = setup.detect()
    assert detected.asset == setup.PINNED_RECIPE.assets[asset_index]
    assert "Hardware matches" in detected.support
    assert "not measured" in detected.support


@pytest.mark.parametrize(
    ("system", "arch", "version"),
    [
        ("Linux", "aarch64", ""),
        ("Windows", "AMD64", ""),
        ("Darwin", "x86_64", "15"),
        ("Darwin", "arm64", "14"),
        ("Darwin", "arm64", "28"),
        ("Darwin", "arm64", ""),
    ],
)
def test_unsupported_platform_uses_existing_guide_without_inventing_an_asset(
    monkeypatch: pytest.MonkeyPatch, system: str, arch: str, version: str
) -> None:
    monkeypatch.setattr(setup.platform, "system", lambda: system)
    monkeypatch.setattr(setup.platform, "machine", lambda: arch)
    monkeypatch.setattr(setup.platform, "mac_ver", lambda: (version, (), ""))
    monkeypatch.setattr(
        discovery.sys,
        "platform",
        {"Darwin": "darwin", "Linux": "linux", "Windows": "win32"}[system],
    )
    planned = setup.plan(setup.detect())
    steps = setup.manual_steps(planned)
    text = "\n".join((*setup.preview(planned), *steps))
    how = discovery.how_to_start()
    assert planned.detection.asset is None
    assert "Manual only" in text and "Manual fallback" in text
    assert "copy only; check compatibility on this machine" in text
    assert all(command in steps for command in how.commands)
    assert how.note in steps and discovery.GUIDE_URL in steps
    assert (
        "hf download" not in text
        and "OMLX_HF_CACHE_ENABLED=0" not in text
        and ".dmg" not in text
    )


@pytest.mark.parametrize("probe", ["system", "machine", "mac_ver"])
def test_platform_probe_errors_produce_manual_only_plan(
    monkeypatch: pytest.MonkeyPatch, probe: str
) -> None:
    def unavailable() -> Any:
        raise OSError("platform metadata unavailable")

    monkeypatch.setattr(setup.platform, probe, unavailable)
    detected = setup.detect()
    assert detected.asset is None and "Manual only" in detected.support


@pytest.mark.parametrize("memory", [None, 8 * 1024**3])
def test_unknown_or_insufficient_ram_is_not_claimed_supported(
    monkeypatch: pytest.MonkeyPatch, memory: int | None
) -> None:
    monkeypatch.setattr(setup, "ram_bytes", lambda: memory)
    assert "Manual only" in setup.detect().support


def test_partial_snapshot_counts_only_complete_expected_size_files_and_shared_blobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    detected = setup.detect()
    snapshot = Path(detected.snapshot)
    snapshot.mkdir(parents=True)
    store = snapshot.parent.parent
    blobs = store / "blobs"
    blobs.mkdir()
    sizes = dict(setup.PINNED_RECIPE.files)
    (snapshot / ".gitattributes").write_bytes(b"a" * sizes[".gitattributes"])
    (blobs / "complete").write_bytes(b"b" * sizes["config.json"])
    (snapshot / "config.json").symlink_to("../../blobs/complete")
    (snapshot / "README.md").write_bytes(b"c" * (sizes["README.md"] + 1))
    (snapshot / "added_tokens.json").write_bytes(b"partial")
    (blobs / "unfinished.incomplete").write_bytes(
        b"d" * sizes["generation_config.json"]
    )
    (snapshot / "generation_config.json").symlink_to(
        "../../blobs/unfinished.incomplete"
    )
    outside = tmp_path / "outside"
    outside.write_bytes(b"e" * sizes["special_tokens_map.json"])
    (snapshot / "special_tokens_map.json").symlink_to(outside)
    (snapshot / "tokenizer.json").symlink_to("missing-blob")
    (snapshot / "vocab.json").mkdir()  # Not a regular file.
    unreadable = snapshot / "tokenizer_config.json"
    unreadable.write_bytes(b"f" * sizes["tokenizer_config.json"])
    monkeypatch.setattr(setup.os, "access", lambda path, mode: path != unreadable)
    before = sorted(str(path) for path in cache.rglob("*"))
    detected = setup.detect()
    assert detected.completed_files == (".gitattributes", "config.json")
    assert detected.missing_model_bytes == 2278972236 - 1570 - 938
    assert sorted(str(path) for path in cache.rglob("*")) == before
    assert "not integrity-verified" in "\n".join(setup.preview(setup.plan(detected)))


@pytest.mark.parametrize("escape", ["store", "snapshot"])
def test_symlinked_store_or_snapshot_outside_store_is_not_counted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, escape: str
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    snapshot = Path(setup.detect().snapshot)
    outside = tmp_path / "outside"
    outside.mkdir()
    if escape == "store":
        snapshot.parent.parent.parent.mkdir(parents=True)
        snapshot.parent.parent.symlink_to(outside, target_is_directory=True)
    else:
        snapshot.parent.mkdir(parents=True)
        snapshot.symlink_to(outside, target_is_directory=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_bytes(b"a" * 938)
    detected = setup.detect()
    assert detected.completed_files == () and detected.missing_model_bytes == 2278972236


@pytest.mark.parametrize("alias", ["another-repo", "another-revision"])
def test_repo_or_revision_alias_cannot_count_another_snapshots_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, alias: str
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    snapshot = Path(setup.detect().snapshot)
    store = snapshot.parent.parent
    if alias == "another-repo":
        target = cache / "models--other-repo"
        target.mkdir(parents=True)
        store.symlink_to(target, target_is_directory=True)
        snapshot.mkdir(parents=True)
    else:
        target = store / "snapshots" / ("a" * 40)
        target.mkdir(parents=True)
        snapshot.symlink_to(target, target_is_directory=True)
    (snapshot / "config.json").write_bytes(b"a" * 938)
    assert setup.detect().completed_files == ()


def test_configured_cache_root_symlink_keeps_normal_shared_blob_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    cache = tmp_path / "cache"
    cache.symlink_to(storage, target_is_directory=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    snapshot = Path(setup.detect().snapshot)
    snapshot.mkdir(parents=True)
    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir()
    (blobs / "complete").write_bytes(b"a" * 938)
    (snapshot / "config.json").symlink_to("../../blobs/complete")
    assert setup.detect().completed_files == ("config.json",)


def test_probe_errors_conservatively_leave_cache_missing_and_disk_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def denied(*args: object, **kwargs: object) -> Any:
        raise PermissionError("metadata unavailable")

    monkeypatch.setattr(Path, "resolve", denied)
    assert _REAL_DISK(tmp_path / "not-created") == (str(tmp_path / "not-created"), None)
    monkeypatch.setattr(setup.ctypes, "CDLL", denied)
    assert _REAL_RAM() is None
    planned = setup.plan(setup.detect())
    assert planned.detection.completed_files == ()
    monkeypatch.setattr(setup, "_disk", lambda path: (str(path), None))
    text = "\n".join(setup.preview(setup.plan(setup.detect())))
    assert "unknown free" in text and "not a guaranteed fit" in text


@pytest.mark.parametrize("same_volume", [True, False])
def test_disk_payload_is_grouped_by_volume_and_never_guarantees_fit(
    monkeypatch: pytest.MonkeyPatch, same_volume: bool
) -> None:
    def disk(path: Path) -> tuple[str, int]:
        return ("/same" if same_volume else str(path), 100)

    monkeypatch.setattr(setup, "_disk", disk)
    planned = setup.plan(setup.detect())
    disks = planned.detection.disks
    assert len(disks) == (1 if same_volume else 2)
    assert sum(d.payload_bytes for d in disks) == 2278972236 + 881239063
    assert all(d.free_bytes == 100 and d.overhead_bytes is None for d in disks)
    text = "\n".join(setup.preview(planned))
    assert text.count("Not enough space for payload alone") == len(disks)
    assert "unknown (unmeasured)" in text and "lower bound" in text


def test_disk_probe_stops_when_even_the_filesystem_root_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path("/unavailable-volume/hf")
    expected = [path, *path.parents]
    seen: list[Path] = []

    def missing(at: Path) -> bool:
        seen.append(at)
        assert len(seen) <= len(
            expected
        ), "disk probe kept checking an unavailable root"
        return False

    with monkeypatch.context() as probes:
        probes.setattr(Path, "resolve", lambda at: at)
        probes.setattr(Path, "exists", missing)
        assert _REAL_DISK(path) == (str(path), None)
    assert seen == expected


def test_real_disk_probe_uses_existing_ancestor_without_mkdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    called: list[Path] = []

    def usage(path: Path) -> Any:
        called.append(path)
        return SimpleNamespace(free=12345)

    monkeypatch.setattr(setup.shutil, "disk_usage", usage)
    path = tmp_path / "not-created" / "yet"
    volume, free = _REAL_DISK(path)
    assert called == [tmp_path.resolve()] and free == 12345
    assert Path(volume).is_absolute() and not path.exists()


def test_all_reserved_ports_and_out_of_range_values_are_denied_without_binding() -> (
    None
):
    assert all(
        not setup.port_allowed(p)
        for p in (*range(8400, 8800), 8955, 11434, 11500, 0, -1, 49151, 65536)
    )
    assert setup.port_allowed(49152) and setup.port_allowed(65535)
    assert setup.choose_port() == 49152
    assert setup.choose_port([49152, 49153]) == 49154


def test_manual_commands_are_pinned_shell_quoted_and_do_not_copy_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home with 'quotes'"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "HF cache 'one'"))
    planned = setup.plan(setup.detect())
    commands = setup.manual_steps(planned)
    hf = shlex.split(next(line for line in commands if " hf download " in line))
    assert hf[1:] == [
        "hf",
        "download",
        planned.recipe.model_repo,
        "--revision",
        planned.recipe.model_revision,
    ]
    assert hf[0] == f"HF_HUB_CACHE={planned.detection.cache_root}"
    serve = shlex.split(next(line for line in commands if "omlx serve" in line))
    assert serve == [
        "OMLX_HF_CACHE_ENABLED=0",
        "omlx",
        "serve",
        "--base-path",
        planned.detection.state_path,
        "--model-dir",
        planned.detection.snapshot,
        "--host",
        "127.0.0.1",
        "--port",
        "49152",
    ]
    text = "\n".join(commands)
    assert planned.recipe.download_base + planned.detection.asset.name in text  # type: ignore[union-attr]
    assert "served ID may be its revision hash" in text and "/v1/models" in text
    assert "use the server-returned ID" in text
    assert all(
        bad not in text for bad in ("--local-dir", "pip install", "~/models", "sudo")
    )


def test_detect_plan_preview_and_manual_have_no_network_writes_processes_or_binds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError("preview attempted a side effect")

    original_open = builtins.open

    def read_only(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        assert not any(flag in mode for flag in "wax+")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read_only)
    for name in ("mkdir", "unlink", "rename", "replace", "write_bytes", "write_text"):
        monkeypatch.setattr(Path, name, forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(httpx.Client, "request", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "request", forbidden)
    planned = setup.plan(setup.detect())
    assert setup.preview(planned) and setup.manual_steps(planned)
    assert planned.allowed_operations == ()
