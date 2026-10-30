from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import io
import os
import plistlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

MP = pytest.MonkeyPatch
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
    assert commands[-2][:4] == ["/usr/bin/codesign", "--force", "--sign", "-"]  # ad hoc
    assert commands[-1][1:4] == ["--verify", "--deep", "--strict"]
    with (output / "Contents/Info.plist").open("rb") as f:
        plist = plistlib.load(f)
    assert plist["NSAppTransportSecurity"] == {"NSAllowsLocalNetworking": True}
    assert plist["CFBundleExecutable"] == "LocalMelo"
    with pytest.raises(FileExistsError):
        build_app.build(output)
    shutil.rmtree(output)


def test_build_tools_are_quiet_unless_they_fail(
    capfd: pytest.CaptureFixture[str],
) -> None:
    build_app.run([sys.executable, "-c", "print('requirements dump')"])
    assert capfd.readouterr() == ("", "")
    with pytest.raises(subprocess.CalledProcessError):
        build_app.run([sys.executable, "-c", "import sys; print('why'); sys.exit(3)"])
    assert "why" in capfd.readouterr().err


def test_cache_hit_checks_the_digest(tmp_path: Path, monkeypatch: MP) -> None:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        member = tarfile.TarInfo("python/bin/python3")
        member.size = 11
        archive.addfile(member, io.BytesIO(b"fake python"))
    payload = data.getvalue()
    monkeypatch.setattr(build_app, "PYTHON_SHA256", hashlib.sha256(payload).hexdigest())
    downloads: list[str] = []

    def urlopen(url: str, **kw: object) -> io.BytesIO:
        downloads.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(build_app.urllib.request, "urlopen", urlopen)
    for name in ("first", "second", "third"):
        (tmp_path / name).mkdir()
        python = build_app.fetch_python(tmp_path / name)
        assert (python / "bin/python3").read_bytes() == b"fake python"
        (cached,) = (Path.home() / "Library/Caches/LocalMelo").glob("python-*")
        if name == "second":
            assert len(downloads) == 1  # the cache was used
            cached.write_bytes(b"not the pinned archive")  # a damaged cache is not
    assert len(downloads) == 2 and cached.read_bytes() == payload


@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS codesign")
def test_signing_is_ad_hoc_nested_code_first_and_verifies(tmp_path: Path) -> None:
    app = tmp_path / "LocalMelo.app"
    tool = app / "Contents/Resources/runtime/bin/tool"
    for path in (app / "Contents/MacOS/LocalMelo", tool):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            "/usr/bin/true", path
        )  # a universal Mach-O file with Apple's signature
    (tool.parent / "link").symlink_to("tool")  # relative, as in the payload
    info = {
        "CFBundleExecutable": "LocalMelo",
        "CFBundleIdentifier": "org.localmelo.app",
    }
    (app / "Contents/Info.plist").write_bytes(plistlib.dumps(info))
    build_app.sign(app)  # signs for real, then runs codesign --verify --deep --strict
    for path in (app, tool):
        shown = subprocess.run(
            ["/usr/bin/codesign", "-dv", str(path)], text=True, capture_output=True
        ).stderr
        assert "Signature=adhoc" in shown and "TeamIdentifier=not set" in shown
        assert "Authority" not in shown


BUILDS: list[Path] = []  # what the fake build was asked for


@pytest.fixture
def home(tmp_path: Path, monkeypatch: MP) -> Path:
    """A temporary HOME whose build() makes a tiny bundle: no network, Xcode or signing."""
    BUILDS.clear()
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:/usr/bin:/bin")

    def fake_build(output: Path) -> None:
        BUILDS.append(output)
        launcher = output / build_app.LAUNCHER_PATH
        launcher.parent.mkdir(parents=True)
        launcher.write_text(f'#!/bin/sh\necho "melo build {len(BUILDS)}"\n')
        launcher.chmod(0o755)
        info = {"CFBundleIdentifier": "org.localmelo.app"}
        (output / "Contents/Info.plist").write_bytes(plistlib.dumps(info))

    monkeypatch.setattr(build_app, "build", fake_build)
    return Path.home()


def melo_says(home: Path) -> str:
    out = subprocess.run([home / ".local/bin/melo"], capture_output=True, text=True)
    return out.stdout.strip()


@pytest.mark.parametrize("on_path", [True, False])
def test_install_gives_the_app_and_melo(
    home: Path, monkeypatch: MP, on_path: bool
) -> None:
    if on_path:
        monkeypatch.setenv("PATH", f"{home}/.local/bin:/usr/bin:/bin")
    lines = build_app.install().splitlines()
    launcher = home / "Applications/LocalMelo.app" / build_app.LAUNCHER_PATH
    assert (home / ".local/bin/melo").resolve() == launcher.resolve()
    assert melo_says(home) == "melo build 1"
    assert [p.name for p in (home / "Applications").iterdir()] == ["LocalMelo.app"]
    assert len(lines) == 3 and "installed in ~/Applications" in lines[0]
    assert "Spotlight" in lines[1] and str(home) not in "\n".join(lines)
    assert ("melo --version" in lines[2]) == on_path
    assert ("export PATH" in lines[2]) != on_path


def test_second_install_upgrades_and_a_failed_one_keeps(
    home: Path, monkeypatch: MP
) -> None:
    build_app.install()
    assert build_app.install().startswith("LocalMelo is updated")
    assert melo_says(home) == "melo build 2"

    def broken(output: Path) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(build_app, "build", broken)
    with pytest.raises(ValueError, match="boom"):
        build_app.install()
    assert melo_says(home) == "melo build 2"
    assert [p.name for p in (home / "Applications").iterdir()] == ["LocalMelo.app"]


@pytest.mark.parametrize("plist", [{"CFBundleIdentifier": "com.example.other"}, None])
def test_a_foreign_app_is_refused(
    home: Path,
    plist: dict[str, str] | None,
    monkeypatch: MP,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = home / "Applications/LocalMelo.app"
    (app / "Contents").mkdir(parents=True)
    (app / "keep.txt").write_text("mine")
    if plist:
        (app / "Contents/Info.plist").write_bytes(plistlib.dumps(plist))
    for flag in ("--install", "--uninstall"):
        monkeypatch.setattr(sys, "argv", ["build_macos_app.py", flag])
        with pytest.raises(SystemExit) as stopped:
            build_app.main()
        assert stopped.value.code == 1
        assert "Move or rename it" in capsys.readouterr().err
    assert (app / "keep.txt").read_text() == "mine"
    assert not BUILDS and not (home / ".local").exists()  # refused before the build


@pytest.mark.parametrize("where", ["PATH", "~/.local/bin"])
def test_a_foreign_melo_is_never_replaced(
    home: Path, tmp_path: Path, where: str
) -> None:
    other = tmp_path / "bin/melo" if where == "PATH" else home / ".local/bin/melo"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("#!/bin/sh\necho mine\n")
    other.chmod(0o755)
    out = build_app.install()
    assert other.read_text().endswith("echo mine\n") and not other.is_symlink()
    assert (home / "Applications/LocalMelo.app").is_dir()
    assert "left alone" in out and "remove it and run this command again" in out
    assert (home / ".local/bin/melo").exists() == (where == "~/.local/bin")


def test_uninstall_removes_exactly_what_install_made(home: Path) -> None:
    others = [home / "Applications/Other.app/keep", home / ".local/bin/other"]
    for other in others:
        other.parent.mkdir(parents=True, exist_ok=True)
        other.touch()
    build_app.install()
    first, second = build_app.uninstall().splitlines()
    assert (
        first
        == "Removed LocalMelo from ~/Applications and the melo command from ~/.local/bin."
    )
    assert "settings and chats" in second and all(other.exists() for other in others)
    assert not (home / "Applications/LocalMelo.app").exists()
    assert not (home / ".local/bin/melo").exists()
    assert "nothing to remove" in build_app.uninstall()
