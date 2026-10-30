from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import io
import os
import plistlib
import shutil
import struct
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


STD = "lib/python3.11"
# One file in each place the trim clears, laid out as the standalone archive has them.
UNUSED_FILES = [
    f"{STD}/idlelib/__init__.py",
    f"{STD}/idlelib/idle_test/test_config.py",
    f"{STD}/tkinter/__init__.py",
    f"{STD}/turtle.py",
    f"{STD}/turtledemo/two_canvases.py",
    f"{STD}/ensurepip/_bundled/pip-24.0-py3-none-any.whl",
    f"{STD}/lib2to3/main.py",
    f"{STD}/lib-dynload/_tkinter.cpython-311-darwin.so",
    "lib/libtcl9.0.dylib",
    "lib/libtcl9tk9.0.dylib",
    "lib/tcl9.0/init.tcl",
    "lib/tcl9/9.0/tcltest-2.5.11.tm",
    "lib/tk9.0/tk.tcl",
    "lib/itcl4.3.8/pkgIndex.tcl",
    "lib/thread3.0.6/pkgIndex.tcl",
    f"{STD}/site-packages/pip/__init__.py",
    f"{STD}/site-packages/pip-26.2.1.dist-info/METADATA",
    f"{STD}/site-packages/setuptools/__init__.py",
    f"{STD}/site-packages/setuptools-84.0.0.dist-info/METADATA",
    f"{STD}/site-packages/_distutils_hack/__init__.py",
    f"{STD}/site-packages/distutils-precedence.pth",
    "bin/pip",
    "bin/pip3",
    "bin/pip3.11",
    "bin/idle3.11",
    "bin/2to3-3.11",
    # The standard library's own tests: the stripped archive has none, another may.
    f"{STD}/test/support/__init__.py",
    f"{STD}/ctypes/test/test_wintypes.py",
    f"{STD}/distutils/tests/test_build.py",
    f"{STD}/sqlite3/test/dbapi.py",
    f"{STD}/unittest/test/test_case.py",
]
# What stays: the rest of the standard library, and every dependency, test folders and
# names that start like a removed one included.
USED_FILES = [
    "bin/python3.11",
    "bin/pydoc3.11",
    "lib/libpython3.11.dylib",
    f"{STD}/os.py",
    f"{STD}/pydoc.py",
    f"{STD}/unittest/__init__.py",
    f"{STD}/distutils/__init__.py",
    f"{STD}/venv/__init__.py",
    f"{STD}/ctypes/__init__.py",
    f"{STD}/sqlite3/__init__.py",
    f"{STD}/email/test/not_the_stdlib_tests.py",
    f"{STD}/lib-dynload/_dbm.cpython-311-darwin.so",
    f"{STD}/site-packages/README.txt",
    f"{STD}/site-packages/localmelo/__init__.py",
    f"{STD}/site-packages/dep/__init__.py",
    f"{STD}/site-packages/dep/test/test_dep.py",
    f"{STD}/site-packages/dep/tests/test_dep.py",
    f"{STD}/site-packages/test/__init__.py",
    f"{STD}/site-packages/pipdeptree/__init__.py",
    f"{STD}/site-packages/setuptools_scm/__init__.py",
]


def make_tree(root: Path, names: list[str]) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)


def test_the_trim_list_holds_only_paths_below_the_runtime() -> None:
    assert build_app.UNUSED
    for pattern in build_app.UNUSED:
        path = Path(pattern)
        assert not path.is_absolute() and ".." not in path.parts
        assert path.parts[0] in ("bin", "lib")


def test_trim_removes_exactly_the_unused_parts(tmp_path: Path) -> None:
    python = tmp_path / "python"
    make_tree(python, UNUSED_FILES + USED_FILES)
    links = {"2to3": "2to3-3.11", "idle3": "idle3.11", "python3": "python3.11"}
    for name, target in links.items():
        (python / "bin" / name).symlink_to(target)
    build_app.trim_python(python)
    left = {
        p.relative_to(python).as_posix() for p in python.rglob("*") if not p.is_dir()
    }
    assert left == {*USED_FILES, "bin/python3"}
    build_app.verify_links(python)  # no link was left without its file
    build_app.trim_python(python)  # nothing is left to remove, and that is fine


def test_trim_removes_a_link_and_never_follows_it(tmp_path: Path) -> None:
    python = tmp_path / "python"
    make_tree(python, [f"{STD}/site-packages/dep/__init__.py"])
    (python / STD / "tkinter").symlink_to("site-packages/dep")  # inside, so allowed
    build_app.trim_python(python)
    assert not (python / STD / "tkinter").is_symlink()
    assert (python / STD / "site-packages/dep/__init__.py").is_file()


def test_trim_leaves_lookalike_test_folders_alone(tmp_path: Path) -> None:
    # Folders named like the removed parts, above the runtime and inside a dependency.
    above = tmp_path / "test/tests/tkinter/pip"
    python = above / "python"
    make_tree(python, UNUSED_FILES + USED_FILES)
    make_tree(tmp_path / "test", ["keep.txt", "tests/keep.txt"])
    make_tree(above, ["keep.txt", "idlelib/keep.txt"])
    build_app.trim_python(python)
    assert not (python / STD / "idlelib").exists()
    assert not (python / STD / "unittest/test").exists()
    for kept in (
        tmp_path / "test/keep.txt",
        tmp_path / "test/tests/keep.txt",
        above / "keep.txt",
        above / "idlelib/keep.txt",
        python / STD / "site-packages/dep/test/test_dep.py",
        python / STD / "site-packages/dep/tests/test_dep.py",
        python / STD / "site-packages/test/__init__.py",
    ):
        assert kept.is_file(), kept


@pytest.mark.parametrize(
    "link, target",
    [
        (f"{STD}/idlelib", "idlelib"),  # a listed folder that is a link out
        (STD, ""),  # a link out above the listed folders
        ("bin/pip", "keep.txt"),  # a listed file that is a link out
    ],
)
def test_trim_refuses_a_listed_path_that_leads_out_of_the_runtime(
    tmp_path: Path, link: str, target: str
) -> None:
    outside = tmp_path / "outside"
    make_tree(outside, ["idlelib/__init__.py", "tkinter/__init__.py", "keep.txt"])
    python = tmp_path / "python"
    make_tree(python, UNUSED_FILES + USED_FILES)
    path = python / link
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    path.symlink_to(outside / target)
    with pytest.raises(ValueError, match="unsafe symlink"):
        build_app.trim_python(python)
    assert sorted(p.name for p in outside.iterdir()) == [
        "idlelib",
        "keep.txt",
        "tkinter",
    ]
    assert (outside / "idlelib/__init__.py").is_file()
    assert (python / "lib/libtcl9.0.dylib").is_file()  # the refusal removed nothing


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


def use_developer_tools(tmp_path: Path, monkeypatch: MP, installed: str) -> str:
    """Make the full Xcode, only the Command Line Tools, or neither look installed.

    Returns the developer folder the build should choose.
    """
    monkeypatch.delenv("DEVELOPER_DIR", raising=False)
    xcode = tmp_path / "Xcode.app/Contents/Developer"
    clt = tmp_path / "CommandLineTools"
    if installed == "xcode":
        xcode.mkdir(parents=True)
    if installed in ("xcode", "clt"):
        (clt / "usr/bin").mkdir(parents=True)
        (clt / "usr/bin/swiftc").touch()
    monkeypatch.setattr(build_app, "XCODE", str(xcode))
    monkeypatch.setattr(build_app, "xcode_select_path", lambda: str(clt))
    return str(xcode if installed == "xcode" else clt)


@pytest.mark.parametrize(
    "installed, summary",
    [("xcode", "Built with Xcode."), ("clt", "Built with the Command Line Tools.")],
)
def test_build_assembles_locked_runtime_and_swift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    installed: str,
    summary: str,
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
    (root / "macos/assets").mkdir(parents=True)
    # Each asset holds its own name, so a copy under the wrong name shows.
    for name in ("AppIcon.icns", "MenuBarLogo.png", "MenuBarLogo@2x.png"):
        (root / "macos/assets" / name).write_bytes(name.encode())
    for name in ("anchors.json", "pet/rest-lazy.png", "portrait/sit-idle@2x.png"):
        (root / "macos/art/akita" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "macos/art/akita" / name).write_bytes(name.encode())
    monkeypatch.setattr(build_app, "ROOT", root)
    monkeypatch.setattr(build_app.sys, "platform", "darwin")
    monkeypatch.setattr(
        build_app.os, "uname", lambda: type("Uname", (), {"machine": "arm64"})()
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    developer = use_developer_tools(tmp_path, monkeypatch, installed)
    commands = []
    unused_at_install: list[bool] = []

    def fake_fetch(stage: Path) -> Path:
        python = stage / "extracted/python"
        (python / "bin").mkdir(parents=True)
        (python / "bin/python3").touch()
        make_tree(python, [f"{STD}/os.py", f"{STD}/idlelib/__init__.py"])
        make_tree(python, [f"{STD}/site-packages/pip/__init__.py"])
        return python

    monkeypatch.setattr(build_app, "fetch_python", fake_fetch)

    def fake_run(args: list[str], *, env: dict[str, str] | None = None) -> None:
        commands.append(args)
        if args[:2] == ["uv", "export"]:
            Path(args[-1]).touch()
        elif args[:3] == ["uv", "pip", "install"]:
            python = Path(args[args.index("--python") + 1]).parent.parent
            unused_at_install.append((python / STD / "idlelib").exists())
        elif args[0] == "/usr/bin/xcrun":
            assert env and env["DEVELOPER_DIR"] == developer
            Path(args[-1]).touch()

    monkeypatch.setattr(build_app, "run", fake_run)
    output = tmp_path / "app with spaces/LocalMelo.app"
    build_app.build(output)
    runtime = output / "Contents/Resources/runtime/bin"
    assert (runtime / "melo").is_file()
    assert (output / "Contents/Resources/melo.png").is_file()  # the starting view
    # The committed icon and menu-bar logo are copied, the icon under the name the plist gives.
    resources = output / "Contents/Resources"
    assert (resources / "LocalMelo.icns").read_bytes() == b"AppIcon.icns"
    assert (resources / "MenuBarLogo.png").read_bytes() == b"MenuBarLogo.png"
    assert (resources / "MenuBarLogo@2x.png").read_bytes() == b"MenuBarLogo@2x.png"
    # Melo's pictures keep their folders: the app finds them under Resources/akita.
    akita = resources / "akita"
    for name in ("anchors.json", "pet/rest-lazy.png", "portrait/sit-idle@2x.png"):
        assert (akita / name).read_bytes() == name.encode()
    assert (
        runtime / "python/lib/python3.11/site-packages/localmelo/support/__init__.py"
    ).is_file()
    assert "--locked" in commands[0] and "gateway" in commands[0]
    assert "--require-hashes" in commands[1] and "--only-binary" in commands[1]
    # The parts the app never runs are gone before the dependencies go in, so a
    # dependency can never be mistaken for one of them, and before the compile.
    assert unused_at_install == [False]
    assert (runtime / "python" / STD / "os.py").is_file()
    assert not (runtime / "python" / STD / "idlelib").exists()
    assert not (runtime / "python" / STD / "site-packages/pip").exists()
    assert str(root / "macos/LocalMeloApp/App.swift") in commands[2]
    site = runtime / "python/lib/python3.11/site-packages"
    distributions = list(importlib.metadata.distributions(path=[str(site)]))
    assert [(d.metadata["Name"], d.version) for d in distributions] == [
        ("localmelo", "0.1.0")
    ]
    assert "compileall" in commands[3] and commands[3][0].endswith("bin/python3")
    assert commands[-2][:4] == ["/usr/bin/codesign", "--force", "--sign", "-"]  # ad hoc
    assert commands[-1][1:4] == ["--verify", "--deep", "--strict"]
    with (output / "Contents/Info.plist").open("rb") as f:
        plist = plistlib.load(f)
    assert plist["NSAppTransportSecurity"] == {"NSAllowsLocalNetworking": True}
    assert plist["CFBundleExecutable"] == "LocalMelo"
    assert plist["CFBundleIdentifier"] == "org.localmelo.app"
    # The Dock and the app menu say localmelo; only the file and the executable keep LocalMelo.
    assert plist["CFBundleName"] == plist["CFBundleDisplayName"] == "localmelo"
    # Alerts and the Dock show the icon the plist names, which is in the bundle when it is signed.
    assert (output / "Contents/Resources" / plist["CFBundleIconFile"]).is_file()
    assert summary in capsys.readouterr().out.splitlines()
    with pytest.raises(FileExistsError):
        build_app.build(output)
    shutil.rmtree(output)


def test_the_callers_developer_dir_wins(tmp_path: Path, monkeypatch: MP) -> None:
    use_developer_tools(tmp_path, monkeypatch, "xcode")  # Xcode is there too
    beta = str(tmp_path / "Beta.app/Contents/Developer")
    monkeypatch.setenv("DEVELOPER_DIR", beta)
    assert build_app.swift_developer_dir() == beta
    assert build_app.tools_name(beta) == "Beta"


@pytest.mark.parametrize("mode", ["--output", "--install"])
def test_no_swift_compiler_refuses_in_one_line_before_any_work(
    tmp_path: Path, monkeypatch: MP, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    use_developer_tools(tmp_path, monkeypatch, "none")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(build_app.sys, "platform", "darwin")
    monkeypatch.setattr(
        build_app.os, "uname", lambda: type("Uname", (), {"machine": "arm64"})()
    )

    def started(*args: object, **kwargs: object) -> None:
        raise AssertionError("the build started")

    monkeypatch.setattr(build_app, "fetch_python", started)
    monkeypatch.setattr(build_app, "run", started)
    argv = ["build_macos_app.py", mode]
    if mode == "--output":
        argv.append(str(tmp_path / "out/LocalMelo.app"))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as stopped:
        build_app.main()
    assert stopped.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # no "Building localmelo" line either
    assert captured.err == (
        "The app needs Apple's Swift compiler. Install it with: xcode-select --install\n"
    )
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "home/Applications").exists()  # nothing was made


ASSETS = Path(__file__).resolve().parents[1] / "macos/assets"


def png_header(data: bytes) -> tuple[int, int, int]:
    """The width, height and colour type of a PNG, read from its header (colour type 6 is RGBA)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, _, color = struct.unpack(">IIBB", data[16:26])
    return width, height, color


def test_the_committed_icon_holds_every_size() -> None:
    data = (ASSETS / "AppIcon.icns").read_bytes()
    assert data[:4] == b"icns" and int.from_bytes(data[4:8], "big") == len(data)
    entries, offset = {}, 8
    while offset < len(data):
        size = int.from_bytes(data[offset + 4 : offset + 8], "big")
        entries[data[offset : offset + 4]] = data[offset + 8 : offset + size]
        offset += size
    # 16 to 1024 px: ic04 and ic05 are the two small sizes (not PNG), ic07 to ic14 the others.
    pixels = {b"ic07": 128, b"ic08": 256, b"ic09": 512, b"ic10": 1024}
    pixels |= {b"ic11": 32, b"ic12": 64, b"ic13": 256, b"ic14": 512}
    assert {b"ic04", b"ic05", *pixels} <= set(entries)
    for kind, size in pixels.items():
        assert png_header(entries[kind]) == (size, size, 6), kind


def test_the_committed_menu_bar_logo_has_a_png_for_each_scale() -> None:
    for name, size in (("MenuBarLogo.png", 18), ("MenuBarLogo@2x.png", 36)):
        assert png_header((ASSETS / name).read_bytes()) == (size, size, 6), name


def test_bundled_python_is_precompiled_and_nothing_writes_bytecode(
    tmp_path: Path,
) -> None:
    python = tmp_path / "python"
    site = python / "lib/python3.11/site-packages"
    (site / "demo").mkdir(parents=True)
    (site / "demo/__init__.py").write_text("VALUE = 1\n")
    build_app.precompile(Path(sys.executable), python, site)
    (pyc,) = (site / "demo/__pycache__").glob("__init__.*.pyc")
    data = pyc.read_bytes()
    assert int.from_bytes(data[4:8], "little") == 0b01  # hash-based, source not checked
    assert str(tmp_path).encode() not in data  # no build path
    assert b"lib/python3.11/site-packages/demo/__init__.py" in data
    assert "dont_write_bytecode = True" in (site / "localmelo-bundle.pth").read_text()
    assert "export PYTHONDONTWRITEBYTECODE=1" in build_app.LAUNCHER


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
    monkeypatch.setattr(build_app, "swift_developer_dir", lambda: build_app.XCODE)
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
    assert build_app.install().startswith("localmelo is updated")
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


@pytest.mark.parametrize("on_path", [True, False])
def test_the_checkouts_own_melo_is_not_the_persons_command(
    home: Path, tmp_path: Path, monkeypatch: MP, on_path: bool
) -> None:
    # `uv run` puts the checkout's .venv/bin first on PATH; its melo exists only inside it.
    clone = tmp_path / "clone"
    monkeypatch.setattr(build_app, "ROOT", clone)
    own = clone / ".venv/bin/melo"
    own.parent.mkdir(parents=True)
    own.write_text("#!/bin/sh\necho dev\n")
    own.chmod(0o755)
    folders = [str(own.parent), "/usr/bin", "/bin"]
    monkeypatch.setenv("PATH", ":".join(folders + ([f"{home}/.local/bin"] * on_path)))
    last = build_app.install().splitlines()[-1]
    launcher = home / "Applications/LocalMelo.app" / build_app.LAUNCHER_PATH
    assert (home / ".local/bin/melo").resolve() == launcher.resolve()
    assert ("melo --version" in last) == on_path and ("export PATH" in last) != on_path
    assert own.read_text().endswith("echo dev\n")


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
        == "Removed localmelo from ~/Applications and the melo command from ~/.local/bin."
    )
    assert "settings and chats" in second and all(other.exists() for other in others)
    assert not (home / "Applications/LocalMelo.app").exists()
    assert not (home / ".local/bin/melo").exists()
    assert "nothing to remove" in build_app.uninstall()


def test_install_and_uninstall_print_localmelo(
    home: Path, tmp_path: Path, monkeypatch: MP, capsys: pytest.CaptureFixture[str]
) -> None:
    # The home fixture puts tmp_path/bin on PATH: a melo that is not ours, so the
    # install leaves it alone and prints the refusal line.
    other = tmp_path / "bin/melo"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("#!/bin/sh\necho mine\n")
    other.chmod(0o755)
    printed = ""
    for flag in ("--install", "--uninstall", "--uninstall"):
        monkeypatch.setattr(sys, "argv", ["build_macos_app.py", flag])
        build_app.main()
        printed += capsys.readouterr().out
    assert "Building localmelo. This can take a minute." in printed
    assert "localmelo is installed in ~/Applications" in printed
    assert "To use localmelo's instead" in printed
    assert "Removed localmelo from ~/Applications" in printed
    assert "localmelo is not installed in ~/Applications" in printed
    assert "LocalMelo" not in printed  # a file or folder name only, never printed
