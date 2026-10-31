"""Setup planning reads only metadata and never grants execution permission."""

from __future__ import annotations

import builtins
import hashlib
import json
import shlex
import socket
import subprocess
import sys
import urllib.request
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from threading import Event, Thread
from time import monotonic
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
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
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
        ({"HUGGINGFACE_HUB_CACHE": "legacy", "HF_HOME": "hf"}, "legacy"),
        (
            {"HF_HUB_CACHE": "hub", "HUGGINGFACE_HUB_CACHE": "legacy", "HF_HOME": "hf"},
            "hub",
        ),
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
def test_unknown_or_insufficient_ram_warns_without_claiming_manual_only(
    monkeypatch: pytest.MonkeyPatch, memory: int | None
) -> None:
    monkeypatch.setattr(setup, "ram_bytes", lambda: memory)
    detected = setup.detect()
    assert detected.asset is not None
    assert (
        "Manual only" not in detected.support
        and "Hardware matches" not in detected.support
    )
    assert ("unknown" if memory is None else "may be slow") in detected.support


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin physical RAM probe")
def test_real_darwin_ram_is_positive_and_a_multiple_of_one_mib() -> None:
    memory = _REAL_RAM()
    assert memory is not None and memory > 0 and memory % 2**20 == 0


def test_ram_details_use_gib_labeled_gb_and_retain_exact_bytes() -> None:
    text = "\n".join(setup.details(setup.plan(setup.detect())))
    assert "RAM: 16 GB (17,179,869,184 bytes)" in text
    assert "RAM: 17.18 GB" not in text


def test_state_path_uses_melos_existing_cache_root() -> None:
    detected = setup.detect()
    expected = Path.home() / ".cache" / "localmelo" / "server-setup" / "omlx-0.7.3"
    assert detected.state_path == str(expected)
    assert not expected.exists()


def test_bounded_detect_returns_completed_metadata_and_reraises_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = setup.detect()
    monkeypatch.setattr(setup, "detect", lambda: expected)
    assert setup.detect_bounded(timeout=0.5) == expected

    def failed() -> Any:
        raise RuntimeError("metadata failed")

    monkeypatch.setattr(setup, "detect", failed)
    with pytest.raises(RuntimeError, match="metadata failed"):
        setup.detect_bounded(timeout=0.5)


def test_bounded_detect_returns_unknown_without_waiting_for_stuck_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = setup.detect()
    release = Event()
    workers: list[Thread] = []

    def tracked_thread(**kwargs: Any) -> Thread:
        worker = Thread(**kwargs)
        workers.append(worker)
        return worker

    def blocked() -> setup.Detection:
        assert release.wait(1), "test fake was not released"
        return expected

    monkeypatch.setattr(setup, "Thread", tracked_thread, raising=False)
    monkeypatch.setattr(setup, "detect", blocked)
    try:
        started = monotonic()
        detected = setup.detect_bounded(timeout=0.02)
        assert monotonic() - started < 0.75
        assert workers and workers[0].daemon and workers[0].is_alive()
        assert detected.system == detected.architecture == "unknown"
        assert (
            detected.ram_bytes is None
            and detected.asset is None
            and detected.disks == ()
        )
        assert (
            detected.cache_root == detected.snapshot == detected.state_path == "unknown"
        )
        assert detected.completed_files == () and "timed out" in detected.support
        assert detected != expected
    finally:
        release.set()
        for worker in workers:
            worker.join(timeout=1)
            assert not worker.is_alive()


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
    assert "not integrity-verified" in "\n".join(setup.details(setup.plan(detected)))
    assert (
        "Download: up to 3.2 GB · Cached: <0.1 GB"
        in setup.preview(setup.plan(detected))[2]
    )


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
    text = "\n".join(setup.details(setup.plan(setup.detect())))
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
    text = "\n".join(setup.details(planned))
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


def test_default_preview_is_short_human_readable_and_counts_runtime_download() -> None:
    planned = setup.plan(setup.detect())
    lines = setup.preview(planned)
    assert len(lines) == 5 and all(len(line) < 80 for line in lines)
    assert lines[0] == "No model server is running yet."
    assert lines[2] == "Download: up to 3.2 GB · Cached: 0 GB · Free: 10.0 GB"
    assert lines[3] == "Planned lifetime: stop when melo closes."
    assert "Ctrl+C" in "\n".join(setup.manual_steps(planned))
    assert lines[4] == "Automatic setup is coming soon: Show manual steps to do it now."
    text = "\n".join(lines)
    for internal in (
        planned.digest,
        planned.recipe.runtime_commit,
        planned.recipe.model_revision,
        planned.detection.cache_root,
        planned.detection.state_path,
        str(planned.port),
        "SHA256",
        "bytes",
        "staging",
        "Platform:",
    ):
        assert internal not in text
    detailed = "\n".join(setup.details(planned))
    assert planned.digest in detailed and planned.detection.snapshot in detailed
    assert "2,278,972,236 bytes" in detailed and "881,239,063 bytes" in detailed
    assert "unknown (unmeasured)" in detailed and "not a guaranteed fit" in detailed


@pytest.mark.parametrize(
    ("missing", "download", "cached"),
    [(2278972236 - 2508, "3.2 GB", "<0.1 GB"), (0, "0.9 GB", "2.3 GB")],
)
def test_concise_download_includes_runtime_even_when_model_is_cached(
    missing: int, download: str, cached: str
) -> None:
    detection = replace(setup.detect(), missing_model_bytes=missing)
    line = setup.preview(setup.plan(detection))[2]
    assert f"Download: up to {download}" in line and f"Cached: {cached}" in line


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        ("ram-low", "may be slow"),
        ("ram-unknown", "RAM unknown"),
        ("disk-low", "not enough disk space"),
        ("disk-unknown", "free space unknown"),
    ],
)
@pytest.mark.parametrize("interactive", [False, True])
def test_preview_adds_only_one_warning_line(
    problem: str, expected: str, interactive: bool
) -> None:
    detected = setup.detect()
    if problem.startswith("ram"):
        detected = replace(
            detected, ram_bytes=8 * 1024**3 if problem == "ram-low" else None
        )
    else:
        detected = replace(
            detected,
            disks=(
                replace(
                    detected.disks[0], free_bytes=1 if problem == "disk-low" else None
                ),
            ),
        )
    lines = setup.preview(setup.plan(detected), interactive=interactive)
    assert len(lines) == 6 and expected.lower() in lines[-1].lower()
    assert ("Details" if interactive else "the manual guide") in lines[-1]
    # The piped guide line is "Guide: " plus the URL, which may be wider.
    assert all(len(line) < 80 or discovery.GUIDE_URL in line for line in lines)


@pytest.mark.parametrize("interactive", [False, True])
def test_preview_combines_warnings_and_shows_multiple_volume_free_space(
    interactive: bool,
) -> None:
    detected = setup.detect()
    detected = replace(
        detected,
        ram_bytes=None,
        disks=(
            replace(detected.disks[0], free_bytes=1),
            setup.DiskSpace("/other", ("/other/cache",), 10, None),
        ),
    )
    lines = setup.preview(
        setup.plan(detected, keep_running=True), interactive=interactive
    )
    assert len(lines) == 6
    assert "Free/volume: <0.1 GB / unknown" in lines[2]
    assert "keep running after melo closes" in lines[3]
    assert "Several setup checks need attention" in lines[-1]
    assert ("Details" if interactive else "the manual guide") in lines[-1]
    assert all(len(line) < 80 or discovery.GUIDE_URL in line for line in lines)


def test_manual_only_summary_hides_cache_lifetime_and_irrelevant_disk_warning() -> None:
    detected = setup.detect()
    detected = replace(
        detected,
        asset=None,
        ram_bytes=None,
        disks=(replace(detected.disks[0], free_bytes=1),),
    )
    lines = setup.preview(setup.plan(detected, keep_running=True))
    text = "\n".join(lines)
    assert len(lines) <= 6 and all(len(line) < 80 for line in lines)
    assert "Manual only" in text and discovery.GUIDE_URL in text
    for irrelevant in (
        "Cached",
        "lifetime",
        "Free",
        "disk",
        "Several setup",
        "Show manual steps",
    ):
        assert irrelevant not in text


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
        "$HOME/.cache/localmelo/server-setup/omlx-0.7.3",  # Shell expands $HOME.
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


def test_shell_path_uses_home_variable_inside_home_and_escapes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    assert setup._shell_path(str(home), home) == '"$HOME"'
    assert setup._shell_path(str(home), None) == '"$HOME"'
    assert setup._shell_path(str(home / ".cache" / "hub"), home) == (
        '"$HOME/.cache/hub"'
    )
    assert setup._shell_path(str(home / "with space"), None) == '"$HOME/with space"'
    spaced = home / "my $dir" / 'a"b`c\\d'
    assert setup._shell_path(str(spaced), home) == '"$HOME/my \\$dir/a\\"b\\`c\\\\d"'


def test_shell_path_quotes_paths_outside_home_and_resolves_symlinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    outside = tmp_path / "home2" / "$cache x"  # Sibling sharing a name prefix.
    assert setup._shell_path(str(outside), home) == shlex.quote(str(outside))
    assert setup._shell_path("relative/$x", home) == shlex.quote("relative/$x")
    real = tmp_path / "real-home"
    link = tmp_path / "linked-home"
    real.mkdir()
    link.symlink_to(real, target_is_directory=True)
    assert setup._shell_path(str(real / "c d"), link) == '"$HOME/c d"'
    monkeypatch.setenv("HOME", str(home))
    assert setup._shell_path(str(home / ".cache")) == '"$HOME/.cache"'


def test_manual_steps_never_print_the_home_folder_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "someone"
    monkeypatch.setenv("HOME", str(home))
    text = "\n".join(setup.manual_steps(setup.plan(setup.detect())))
    for form in {str(home), str(home.resolve())}:
        assert form not in text
    assert 'HF_HUB_CACHE="$HOME/.cache/huggingface/hub" hf download ' in text
    assert (
        '--base-path "$HOME/.cache/localmelo/server-setup/omlx-0.7.3" '
        '--model-dir "$HOME/.cache/huggingface/hub/models--mlx-community--'
        "Qwen3-4B-Instruct-2507-4bit/snapshots/"
        '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b" '
    ) in text


def test_piped_preview_labels_the_guide_link() -> None:
    planned = setup.plan(setup.detect())
    assert setup.preview(planned, interactive=False)[4] == (
        f"Guide: {discovery.GUIDE_URL}"
    )
    assert setup.preview(planned)[4] == (
        "Automatic setup is coming soon: Show manual steps to do it now."
    )


def test_recommended_line_only_appears_with_a_runtime_asset() -> None:
    supported = setup.details(setup.plan(setup.detect()))
    assert "Recommended: oMLX + Qwen3 4B Instruct." in supported
    unsupported = setup.details(setup.plan(replace(setup.detect(), asset=None)))
    assert not any("Recommended:" in line for line in unsupported)


@pytest.mark.parametrize("bounded", [False, True])
def test_detect_plan_preview_and_manual_have_no_network_writes_processes_or_binds(
    monkeypatch: pytest.MonkeyPatch,
    bounded: bool,
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
    for name in (
        "mkdir",
        "makedirs",
        "rename",
        "replace",
        "unlink",
        "remove",
        "rmdir",
        "system",
    ):
        monkeypatch.setattr(setup.os, name, forbidden)
    for name in (
        "copy",
        "copy2",
        "copyfile",
        "copytree",
        "move",
        "rmtree",
        "unpack_archive",
    ):
        monkeypatch.setattr(setup.shutil, name, forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request, "urlretrieve", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(httpx.Client, "request", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "request", forbidden)
    planned = setup.plan(setup.detect_bounded() if bounded else setup.detect())
    assert (
        setup.preview(planned)
        and setup.details(planned)
        and setup.manual_steps(planned)
    )
    assert planned.allowed_operations == ()
