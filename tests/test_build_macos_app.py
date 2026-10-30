from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import io
import os
import plistlib
import shutil
import subprocess
import tarfile
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
        "-P",
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


def test_verified_archive_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        content = b"fake python"
        member = tarfile.TarInfo("python/bin/python3")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    payload = data.getvalue()
    monkeypatch.setattr(
        build_app.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload)
    )
    monkeypatch.setattr(build_app, "PYTHON_SHA256", "0" * 64)
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_app.fetch_python(bad)
    assert not (bad / "extracted").exists()
    monkeypatch.setattr(build_app, "PYTHON_SHA256", hashlib.sha256(payload).hexdigest())
    good = tmp_path / "good"
    good.mkdir()
    assert (build_app.fetch_python(good) / "bin/python3").read_bytes() == b"fake python"


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
        "support/gateway/static/melo.png",
        "tests/smoke/data/scenarios.json",
    ):
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (root / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
    monkeypatch.setattr(build_app, "ROOT", root)
    monkeypatch.setattr(build_app.sys, "platform", "darwin")
    monkeypatch.setattr(
        build_app.os, "uname", lambda: type("Uname", (), {"machine": "arm64"})()
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    commands = []

    def fake_fetch(stage: Path) -> Path:
        python = stage / "extracted/python"
        (python / "bin").mkdir(parents=True)
        (python / "bin/python3").touch()
        return python

    monkeypatch.setattr(build_app, "fetch_python", fake_fetch)

    def fake_run(args: list[str], *, env: dict[str, str] | None = None) -> None:
        commands.append(args)
        if args[:2] == ["uv", "export"]:
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
    assert (output / "Contents/Resources/melo.png").is_file()  # the starting view
    assert (
        runtime / "python/lib/python3.11/site-packages/localmelo/support/__init__.py"
    ).is_file()
    assert "--locked" in commands[0] and "gateway" in commands[0]
    assert "--require-hashes" in commands[1] and "--only-binary" in commands[1]
    assert str(root / "macos/LocalMeloApp/App.swift") in commands[2]
    site = runtime / "python/lib/python3.11/site-packages"
    distributions = list(importlib.metadata.distributions(path=[str(site)]))
    assert [(d.metadata["Name"], d.version) for d in distributions] == [
        ("localmelo", "0.1.0")
    ]
    with (output / "Contents/Info.plist").open("rb") as f:
        plist = plistlib.load(f)
    assert plist["NSAppTransportSecurity"] == {"NSAllowsLocalNetworking": True}
    assert plist["CFBundleExecutable"] == "LocalMelo"
    with pytest.raises(FileExistsError):
        build_app.build(output)
    shutil.rmtree(output)
