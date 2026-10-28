#!/usr/bin/env python3
"""Build an unsigned, local-development macOS application bundle."""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.11.16"

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
exec "$DIR/python/bin/python3" -m localmelo "$@"
"""


def run(args: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=ROOT, env=env, check=True)


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
        managed = stage / "managed"
        env = os.environ.copy()
        env["UV_PYTHON_INSTALL_DIR"] = str(managed)
        run(["uv", "python", "install", "--no-bin", PYTHON_VERSION], env=env)
        installs = list(managed.glob(f"cpython-{PYTHON_VERSION}-macos-aarch64-*"))
        if len(installs) != 1:
            raise RuntimeError("uv did not install the expected standalone Python")
        python = runtime / "python"
        shutil.copytree(installs[0], python, symlinks=True)
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
        launcher = runtime / "melo"
        launcher.write_text(LAUNCHER)
        launcher.chmod(0o755)
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
                "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
            },
        )
        with (contents / "Info.plist").open("wb") as f:
            plistlib.dump(
                {
                    "CFBundleExecutable": "LocalMelo",
                    "CFBundleIdentifier": "org.localmelo.app",
                    "CFBundleName": "LocalMelo",
                    "CFBundleDisplayName": "LocalMelo",
                    "CFBundlePackageType": "APPL",
                    "CFBundleShortVersionString": "0.1.0",
                    "CFBundleVersion": "1",
                    "NSHighResolutionCapable": True,
                    "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
                },
                f,
            )
        shutil.move(str(app), str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/LocalMelo.app")
    args = parser.parse_args()
    try:
        build(args.output.absolute())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Build failed: {exc}\n")
    print(f"Built unsigned development app: {args.output}")


if __name__ == "__main__":
    main()
