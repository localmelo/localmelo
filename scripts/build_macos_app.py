#!/usr/bin/env python3
"""Build the localmelo macOS app; with --install, also put it on this Mac.

The app is signed ad hoc, so no Apple developer account is needed. It is made for the
Mac that builds it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import plistlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
from pathlib import Path
from xml.parsers.expat import ExpatError

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.11.16"
# Digest from the official python-build-standalone GitHub release asset metadata.
PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260924/"
    "cpython-3.11.16%2B20260924-aarch64-apple-darwin-install_only_stripped.tar.gz"
)
PYTHON_SHA256 = "e1d745b07b6acc0641dbb3237d3c5953deeeed182141bab2242684076fd86547"
BUNDLE_ID = "org.localmelo.app"
XCODE = "/Applications/Xcode.app/Contents/Developer"
LAUNCHER_PATH = "Contents/Resources/runtime/bin/melo"

# Resolve every link, including a Homebrew-style link to this launcher. No build
# paths or shell-interpreted user arguments are embedded in the bundle.
LAUNCHER = r"""#!/bin/sh
set -eu
SOURCE=$0
while [ -L "$SOURCE" ]; do
    DIR=$(CDPATH= cd -P -- "$(dirname -- "$SOURCE")" && pwd)
    TARGET=$(readlink "$SOURCE")
    case "$TARGET" in /*) SOURCE=$TARGET ;; *) SOURCE=$DIR/$TARGET ;; esac
done
DIR=$(CDPATH= cd -P -- "$(dirname -- "$SOURCE")" && pwd)
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
exec "$DIR/python/bin/python3" -P -m localmelo "$@"
"""


def run(args: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run a build tool quietly; what it printed is shown only when it fails."""
    done = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
    if done.returncode:
        print(done.stdout + done.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(done.returncode, args[:1])  # not every file


def verify_links(root: Path) -> None:
    """Reject broken links and links escaping the relocatable Python payload."""
    root = root.resolve()
    for path in root.rglob("*"):
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
                target.relative_to(root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError(f"Python payload has unsafe symlink: {path}") from exc


def download_python(archive: Path) -> None:
    """Download the pinned archive and reject it unless it has the pinned digest."""
    digest = hashlib.sha256()
    with (
        urllib.request.urlopen(PYTHON_URL, timeout=120) as response,
        archive.open("wb") as out,
    ):
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != PYTHON_SHA256:
        raise ValueError("Standalone Python archive SHA-256 mismatch")


def fetch_python(stage: Path) -> Path:
    """Verify the exact official archive before opening or extracting it.

    A verified archive is cached by digest in ~/Library/Caches/LocalMelo and hashed on every use.
    """
    cached = Path.home() / f"Library/Caches/LocalMelo/python-{PYTHON_SHA256}.tar.gz"
    archive = cached
    try:
        with cached.open("rb") as f:
            hit = hashlib.file_digest(f, "sha256").hexdigest() == PYTHON_SHA256
    except OSError:
        hit = False
    if not hit:
        (stage / "download").mkdir()
        archive = stage / "download/python.tar.gz"
        download_python(archive)
        with contextlib.suppress(OSError):  # the cache only saves time
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive, cached.with_suffix(".part"))
            os.replace(cached.with_suffix(".part"), cached)
    extracted = stage / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extracted, filter="data")
    python = extracted / "python"
    if not (python / "bin/python3").is_file():
        raise ValueError("Standalone Python archive has an unexpected layout")
    verify_links(python)
    return python


def is_macho(path: Path) -> bool:
    """Mach-O (thin, either width) or universal, told by the first four bytes."""
    with path.open("rb") as f:
        return f.read(4) in (
            b"\xcf\xfa\xed\xfe",
            b"\xce\xfa\xed\xfe",
            b"\xca\xfe\xba\xbe",
        )


def precompile(interpreter: Path, python: Path, site: Path) -> None:
    """Compile the bundled Python to bytecode, before the bundle is signed.

    The standalone archive has none, and the launcher forbids writing it, so every start
    compiled all it imported. These files are hash-based and never check the source, so a
    copy that changes file times cannot make them stale, and the build path is stripped.
    The .pth file keeps a direct run of the interpreter, which skips the launcher, from
    writing bytecode into the signed bundle.
    """
    run(
        [str(interpreter), "-I", "-B", "-m", "compileall", "-q", "-f", "-j", "0"]
        + ["--invalidation-mode", "unchecked-hash", "-s", str(python), str(site.parent)]
    )
    (site / "localmelo-bundle.pth").write_text(
        "import sys; sys.dont_write_bytecode = True\n"
    )


def sign(app: Path) -> None:
    """Sign ad hoc (no certificate, account or name): Mach-O files first, the bundle last."""
    machos = sorted(
        str(p)
        for p in app.rglob("*")
        if p.is_file() and not p.is_symlink() and is_macho(p)
    )
    codesign = ["/usr/bin/codesign", "--force", "--sign", "-"]
    if machos:
        run([*codesign, *machos])
    run([*codesign, str(app)])
    run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)])


def build(output: Path) -> None:
    if sys.platform != "darwin" or os.uname().machine != "arm64":
        raise RuntimeError("Building requires an arm64 macOS host")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    swift_files = sorted((ROOT / "macos/LocalMeloApp").glob("*.swift"))
    if not swift_files:
        raise RuntimeError("No Swift host files found in macos/LocalMeloApp")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build away from the destination, so failure never leaves a partial app.
    with tempfile.TemporaryDirectory(
        prefix="localmelo-build-", dir=output.parent
    ) as temp:
        stage = Path(temp)
        app = stage / "LocalMelo.app"
        contents = app / "Contents"
        binary = contents / "MacOS"
        runtime = contents / "Resources/runtime/bin"
        binary.mkdir(parents=True)
        runtime.mkdir(parents=True)
        python = runtime / "python"
        shutil.copytree(fetch_python(stage), python, symlinks=True)
        verify_links(python)
        interpreter = python / "bin/python3"
        requirements = stage / "requirements.txt"
        run(
            [
                "uv",
                "export",
                "--locked",
                "--extra",
                "gateway",
                "--no-emit-project",
                "--output-file",
                str(requirements),
            ]
        )
        run(
            [
                "uv",
                "pip",
                "install",
                "--system",
                "--break-system-packages",
                "--python",
                str(interpreter),
                "--require-hashes",
                "--only-binary",
                ":all:",
                "--no-deps",
                "-r",
                str(requirements),
            ]
        )
        # Copy the project package without an editable install or build-time paths.
        site = (
            python
            / "lib"
            / f"python{'.'.join(PYTHON_VERSION.split('.')[:2])}"
            / "site-packages"
        )
        package = site / "localmelo"
        package.mkdir(parents=True, exist_ok=True)
        for name in ("__init__.py", "__main__.py"):
            shutil.copy2(ROOT / name, package / name)
        for name in ("melo", "support"):
            shutil.copytree(
                ROOT / name,
                package / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        scenario = package / "tests/smoke/data/scenarios.json"
        scenario.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "tests/smoke/data/scenarios.json", scenario)
        with (ROOT / "pyproject.toml").open("rb") as f:
            version = tomllib.load(f)["project"]["version"]
        metadata = site / f"localmelo-{version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: localmelo\nVersion: {version}\n"
        )
        launcher = runtime / "melo"
        launcher.write_text(LAUNCHER)
        launcher.chmod(0o755)
        # The window's starting view shows Melo's portrait.
        shutil.copy2(
            ROOT / "support/gateway/static/melo.png", contents / "Resources/melo.png"
        )
        verify_links(python)
        run(
            [
                "/usr/bin/xcrun",
                "swiftc",
                *map(str, swift_files),
                "-o",
                str(binary / "LocalMelo"),
            ],
            env={
                **os.environ,
                "DEVELOPER_DIR": XCODE,
            },
        )
        precompile(interpreter, python, site)
        with (contents / "Info.plist").open("wb") as f:
            plistlib.dump(
                {
                    "CFBundleExecutable": "LocalMelo",
                    "CFBundleIdentifier": BUNDLE_ID,
                    # What the Dock and the app menu call it; LocalMelo.app is only the file name.
                    "CFBundleName": "localmelo",
                    "CFBundleDisplayName": "localmelo",
                    "CFBundlePackageType": "APPL",
                    "CFBundleShortVersionString": version,
                    "CFBundleVersion": "1",
                    "NSHighResolutionCapable": True,
                    "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
                },
                f,
            )
        sign(app)
        shutil.move(str(app), str(output))


class InstallError(RuntimeError):
    """A refusal, worded for the person who ran the command."""


def check_target(target: Path) -> bool:
    """Refuse an app there that is not ours (bundle id org.localmelo.app). True if it exists."""
    if not (target.exists() or target.is_symlink()):
        return False
    try:
        with (target / "Contents/Info.plist").open("rb") as f:
            found: object = plistlib.load(f).get("CFBundleIdentifier")
    except (OSError, ValueError, ExpatError):
        found = None
    if found != BUNDLE_ID or target.is_symlink():
        raise InstallError(
            "~/Applications/LocalMelo.app is a different app, so it was left as it is.\n"
            "Move or rename it, then run this command again."
        )
    return True


def melo_on_path() -> str | None:
    """The melo command a new Terminal would find, not counting this checkout's own.

    `uv run` puts the checkout's .venv/bin first on PATH, and the melo there exists only
    inside `uv run`: it is not a command the person has.
    """
    own = ROOT.resolve()
    folders = [
        folder
        for folder in os.environ.get("PATH", "").split(os.pathsep)
        if not Path(folder).resolve().is_relative_to(own)
    ]
    return shutil.which("melo", path=os.pathsep.join(folders))


def links_to(link: Path, launcher: Path) -> bool:
    return link.is_symlink() and os.path.realpath(link) == os.path.realpath(launcher)


def install() -> str:
    """Build the app, put it in ~/Applications and link the melo command. No admin."""
    target = Path.home() / "Applications/LocalMelo.app"
    check_target(target)  # before the long build
    target.parent.mkdir(parents=True, exist_ok=True)
    # Build beside the target, so the last step is a rename and a failure changes nothing.
    scratch = Path(tempfile.mkdtemp(prefix=".LocalMelo-", dir=target.parent))
    try:
        print("Building localmelo. This can take a minute.", flush=True)
        build(scratch / "LocalMelo.app")
        upgraded = check_target(target)  # it may have been opened meanwhile
        if upgraded:
            os.rename(target, scratch / "previous.app")
        try:
            os.rename(scratch / "LocalMelo.app", target)
        except OSError:
            if upgraded:
                os.rename(scratch / "previous.app", target)
            raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    link, launcher = Path.home() / ".local/bin/melo", target / LAUNCHER_PATH
    other = None  # a melo command that is not ours, left alone
    if not links_to(link, launcher):
        other = link if os.path.lexists(link) else melo_on_path()
        if not other:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(launcher)
    if other:
        command = (
            f"Your own melo command ({str(other).replace(str(Path.home()), '~')}) was left "
            "alone. To use localmelo's instead, remove it and run this command again."
        )
    elif melo_on_path():
        command = "The melo command is ready in Terminal: melo --version"
    else:
        command = (
            "To use melo in Terminal, run this once and open a new window: "
            "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.zshrc"
        )
    return (
        f"localmelo is {'updated' if upgraded else 'installed'} in ~/Applications, "
        "the Applications folder in your home folder.\n"
        "Next: open localmelo from Spotlight (Command-Space, then type localmelo) or from that folder.\n"
        + command
    )


def uninstall() -> str:
    """Remove the app and the melo link that install made, and nothing else."""
    target = Path.home() / "Applications/LocalMelo.app"
    link = Path.home() / ".local/bin/melo"
    present = check_target(target)
    linked = links_to(link, target / LAUNCHER_PATH)
    if linked:
        link.unlink()
    if present:
        shutil.rmtree(target)
    removed = [
        *(["localmelo from ~/Applications"] if present else []),
        *(["the melo command from ~/.local/bin"] if linked else []),
    ]
    if not removed:
        return "localmelo is not installed in ~/Applications, so there is nothing to remove."
    return (
        f"Removed {' and '.join(removed)}.\n"
        "Your settings and chats were not touched; they stay in ~/.cache/localmelo."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, default=ROOT / "build/LocalMelo.app")
    mode.add_argument(
        "--install",
        action="store_true",
        help="build the app, put it in ~/Applications and link the melo command",
    )
    mode.add_argument(
        "--uninstall", action="store_true", help="remove what --install made"
    )
    args = parser.parse_args()
    try:
        if args.install:
            print(install())
        elif args.uninstall:
            print(uninstall())
        else:
            build(args.output.absolute())
            print(f"Built development app (signed ad hoc): {args.output}")
    except InstallError as exc:
        parser.exit(1, f"{exc}\n")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Build failed: {exc}\n")


if __name__ == "__main__":
    main()
