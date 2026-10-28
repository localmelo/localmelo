from __future__ import annotations

import importlib.util
import os
import plistlib
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_macos_app.py"
spec = importlib.util.spec_from_file_location("build_macos_app", SCRIPT)
assert spec and spec.loader
build_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_app)


def test_relocated_launcher_via_symlinks(tmp_path: Path) -> None:
    runtime = tmp_path / "moved app/Contents/Resources/runtime/bin"
    python = runtime / "python/bin/python3"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$PYTHONHOME|$PYTHONPATH|$PYTHONNOUSERSITE" "$@"\n'
    )
    python.chmod(0o755)
    launcher = runtime / "melo"
    launcher.write_text(build_app.LAUNCHER)
    launcher.chmod(0o755)
    link = tmp_path / "melo"
    link.symlink_to(launcher)
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PYTHONHOME": "/bad",
        "PYTHONPATH": "/bad",
    }
    result = subprocess.run(
        [str(link), "--help", "two words"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        "||1",
        "-m",
        "localmelo",
        "--help",
        "two words",
    ]


def test_verify_links_rejects_escape_and_broken(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "file").touch()
    link = root / "link"
    link.symlink_to("file")
    build_app.verify_links(root)
    link.unlink()
    link.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="unsafe symlink"):
        build_app.verify_links(root)
    link.unlink()
    link.symlink_to("missing")
    with pytest.raises(ValueError, match="unsafe symlink"):
        build_app.verify_links(root)


def test_build_assembles_locked_runtime_and_swift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    (root / "macos/LocalMeloApp").mkdir(parents=True)
    (root / "macos/LocalMeloApp/App.swift").touch()
    for file in (
        "__init__.py",
        "__main__.py",
        "melo/__init__.py",
        "support/__init__.py",
        "tests/smoke/data/scenarios.json",
    ):
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(build_app, "ROOT", root)
    monkeypatch.setattr(build_app.sys, "platform", "darwin")
    monkeypatch.setattr(
        build_app.os, "uname", lambda: type("Uname", (), {"machine": "arm64"})()
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    commands = []

    def fake_run(args: list[str], *, env: dict[str, str] | None = None) -> None:
        commands.append(args)
        if args[:3] == ["uv", "python", "install"]:
            assert env is not None
            python = (
                Path(env["UV_PYTHON_INSTALL_DIR"])
                / f"cpython-{build_app.PYTHON_VERSION}-macos-aarch64-none/bin/python3"
            )
            python.parent.mkdir(parents=True)
            python.touch()
        elif args[:2] == ["uv", "export"]:
            Path(args[-1]).touch()
        elif args[0] == "/usr/bin/xcrun":
            assert (
                env
                and env["DEVELOPER_DIR"] == "/Applications/Xcode.app/Contents/Developer"
            )
            Path(args[-1]).touch()

    monkeypatch.setattr(build_app, "run", fake_run)
    output = tmp_path / "app with spaces/LocalMelo.app"
    build_app.build(output)
    runtime = output / "Contents/Resources/runtime/bin"
    assert (runtime / "melo").is_file()
    assert (
        runtime / "python/lib/python3.11/site-packages/localmelo/support/__init__.py"
    ).is_file()
    assert "--locked" in commands[1] and "gateway" in commands[1]
    assert "--require-hashes" in commands[2] and "--only-binary" in commands[2]
    assert str(root / "macos/LocalMeloApp/App.swift") in commands[3]
    with (output / "Contents/Info.plist").open("rb") as f:
        plist = plistlib.load(f)
    assert plist["NSAppTransportSecurity"] == {"NSAllowsLocalNetworking": True}
    assert plist["CFBundleExecutable"] == "LocalMelo"
    with pytest.raises(FileExistsError):
        build_app.build(output)
    shutil.rmtree(output)
