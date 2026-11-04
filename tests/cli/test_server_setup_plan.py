"""The no-server screen reads only the platform and never runs or writes anything."""

from __future__ import annotations

import builtins
import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.support.backends import discovery
from localmelo.support.server_setup import plan as setup

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


def test_the_recipe_is_pinned() -> None:
    r = setup.PINNED_RECIPE
    assert r.model_bytes == 2278972236 and len(r.files) == 13
    assert len(r.model_revision) == 40
    assert all(len(asset.sha256) == 64 for asset in r.assets)


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


def test_state_path_is_under_melos_cache_root_and_not_created() -> None:
    expected = Path.home() / ".cache" / "localmelo" / "server-setup" / "omlx-0.7.3"
    assert setup.detect().state_path == str(expected)
    assert not expected.exists()


@pytest.mark.parametrize(
    ("version", "asset_index"), [("15.7", 0), ("26.1", 1), ("27.2", 1)]
)
def test_only_verified_macos_assets_are_selected(
    monkeypatch: pytest.MonkeyPatch, version: str, asset_index: int
) -> None:
    monkeypatch.setattr(setup.platform, "mac_ver", lambda: (version, (), ""))
    assert setup.detect().asset == setup.PINNED_RECIPE.assets[asset_index]


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
def test_an_unsupported_platform_gets_the_platform_guide_not_an_invented_asset(
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
    detected = setup.detect()
    how = discovery.how_to_start()
    lines = setup.preview(detected)
    steps = setup.numbered_steps(detected)
    assert detected.asset is None
    assert len(lines) <= 6 and all(len(line) < 80 for line in lines)
    assert "Manual only" in "\n".join(lines) and discovery.GUIDE_URL in lines
    assert [
        line.strip() for line in steps if line.startswith(setup.COMMAND_INDENT)
    ] == (list(how.commands))
    assert (
        f"   {how.note}" in steps and steps[-1] == f"More help: {discovery.GUIDE_URL}"
    )
    text = "\n".join((*lines, *steps))
    assert how.name in text and "Enter an address" in text
    for absent in ("Download:", "hf download", "OMLX_HF_CACHE_ENABLED=0", ".dmg"):
        assert absent not in text


@pytest.mark.parametrize("probe", ["system", "machine", "mac_ver"])
def test_a_platform_probe_error_gives_the_manual_screen(
    monkeypatch: pytest.MonkeyPatch, probe: str
) -> None:
    def unavailable() -> Any:
        raise OSError("platform metadata unavailable")

    monkeypatch.setattr(setup.platform, probe, unavailable)
    detected = setup.detect()
    assert detected.asset is None
    assert "Setup checks are unavailable" in "\n".join(setup.preview(detected))


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin physical RAM probe")
def test_real_darwin_ram_is_positive_and_a_multiple_of_one_mib() -> None:
    memory = _REAL_RAM()
    assert memory is not None and memory > 0 and memory % 2**20 == 0


def test_the_default_preview_is_short_and_counts_the_runtime_download() -> None:
    lines = setup.preview(setup.detect())
    assert lines == (
        "No model server is running yet.",
        "Recommended: oMLX with Qwen3 4B Instruct.",
        "Download: up to 3.2 GB",
        "Automatic setup is coming soon; melo can show you how to do it now.",
    )
    assert all(len(line) < 80 for line in lines)
    assert setup.preview(setup.detect(), interactive=False)[3] == (
        f"Guide: {discovery.GUIDE_URL}"
    )


@pytest.mark.parametrize(
    ("memory", "expected"),
    [(None, "RAM unknown"), (8 * 1024**3, "may be slow")],
)
def test_unknown_or_low_ram_adds_one_warning_line(
    monkeypatch: pytest.MonkeyPatch, memory: int | None, expected: str
) -> None:
    monkeypatch.setattr(setup, "ram_bytes", lambda: memory)
    lines = setup.preview(setup.detect())
    assert len(lines) == 5 and expected in lines[-1] and "the manual guide" in lines[-1]
    assert all(len(line) < 80 for line in lines)


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


def _numbered(lines: tuple[str, ...]) -> list[str]:
    return [line for line in lines if re.match(r"\d\. ", line)]


def test_the_steps_are_six_plain_numbered_steps_with_indented_commands() -> None:
    detected = setup.detect()
    steps = setup.numbered_steps(detected)
    numbered = _numbered(steps)
    assert [line[:2] for line in numbered] == ["1.", "2.", "3.", "4.", "5.", "6."]
    assert steps[-1] == f"More help: {discovery.GUIDE_URL}"
    assert detected.asset is not None
    assert numbered[0] == (
        f"1. Download oMLX (0.9 GB): {detected.recipe.download_base}"
        f"{detected.asset.name}"
    )
    assert "up to 2.3 GB" in numbered[3]
    commands = [line for line in steps if line.startswith(setup.COMMAND_INDENT)]
    assert len(commands) == 3
    assert commands[0].strip().startswith("shasum -a 256 ")
    assert commands[1].strip().startswith("HF_HUB_CACHE=") and " hf download " in (
        commands[1]
    )
    assert commands[2].strip().startswith("OMLX_HF_CACHE_ENABLED=0 omlx serve ")
    assert f"--port {setup.PORT}" in commands[2]
    assert f"http://127.0.0.1:{setup.PORT}/v1" in numbered[5]
    assert "Enter an address" in numbered[5]
    text = "\n".join(steps)
    assert "Gatekeeper" in text and "Ctrl+C" in text
    for bad in ("--local-dir", "pip install", "sudo", "bytes", "approval"):
        assert bad not in text


def test_the_steps_never_print_the_home_folder_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "someone"
    monkeypatch.setenv("HOME", str(home))
    text = "\n".join(setup.numbered_steps(setup.detect()))
    for form in {str(home), str(home.resolve())}:
        assert form not in text
    assert '"$HOME/Downloads/oMLX-0.7.3-macos26-27.dmg"' in text
    assert 'HF_HUB_CACHE="$HOME/.cache/huggingface/hub" hf download ' in text
    assert '--base-path "$HOME/.cache/localmelo/server-setup/omlx-0.7.3" ' in text
    assert (
        '--model-dir "$HOME/.cache/huggingface/hub/models--mlx-community--'
        "Qwen3-4B-Instruct-2507-4bit/snapshots/"
        '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b" '
    ) in text


def test_the_screen_never_writes_opens_a_connection_or_starts_a_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError("the setup screen attempted a side effect")

    original_open = builtins.open

    def read_only(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        assert not any(flag in mode for flag in "wax+")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read_only)
    for name in ("mkdir", "unlink", "rename", "replace", "write_bytes", "write_text"):
        monkeypatch.setattr(Path, name, forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(httpx.Client, "request", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "request", forbidden)
    detected = setup.detect()
    assert setup.preview(detected) and setup.numbered_steps(detected)
