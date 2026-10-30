#!/usr/bin/env python3
"""Build the localmelo macOS app; with --install, also put it on this Mac.

The app is signed ad hoc, so no Apple developer account is needed. It is made for the
Mac that builds it.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import os
import plistlib
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zlib
from pathlib import Path
from xml.parsers.expat import ExpatError

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.11.16"
# Digest from the official python-build-standalone GitHub release asset metadata.
PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260924/"
    "cpython-3.11.16%2B20260924-aarch64-apple-darwin-install_only_stripped.tar.gz"
)
PYTHON_SHA256 = "e1d745b07b6acc0641dbb3237d3c5953deeeed182141bab2242684076fd86547"
# The standard library folder, below the runtime folder (python/ in the bundle).
STDLIB = f"lib/python{'.'.join(PYTHON_VERSION.split('.')[:2])}"
# What the app never runs in the standalone Python, as patterns below the runtime folder.
# They are matched there and never against an absolute path, so the folder the app is
# built in cannot match one. Everything else stays: `unittest`, `distutils`, `venv` and
# `pydoc` too, and every dependency, which goes in after the trim.
UNUSED = (
    # IDLE, tkinter and turtle, with the Tcl/Tk libraries they load
    f"{STDLIB}/idlelib",
    f"{STDLIB}/tkinter",
    f"{STDLIB}/turtle.py",
    f"{STDLIB}/turtledemo",
    f"{STDLIB}/lib-dynload/_tkinter*.so",
    "lib/libtcl*.dylib",
    "lib/tcl[0-9]*",
    "lib/tk[0-9]*",
    "lib/itcl[0-9]*",
    "lib/thread[0-9]*",
    # 2to3, pip and setuptools (ensurepip carries a wheel of each), and their commands
    f"{STDLIB}/lib2to3",
    f"{STDLIB}/ensurepip",
    f"{STDLIB}/site-packages/pip",
    f"{STDLIB}/site-packages/pip-*.dist-info",
    f"{STDLIB}/site-packages/setuptools",
    f"{STDLIB}/site-packages/setuptools-*.dist-info",
    f"{STDLIB}/site-packages/_distutils_hack",
    f"{STDLIB}/site-packages/distutils-precedence.pth",
    "bin/2to3*",
    "bin/idle3*",
    "bin/pip*",
    # The standard library's own tests. The stripped archive pinned above has none, so
    # these match nothing today; they keep a different archive from growing the app.
    f"{STDLIB}/test",
    f"{STDLIB}/ctypes/test",
    f"{STDLIB}/distutils/tests",
    f"{STDLIB}/sqlite3/test",
    f"{STDLIB}/unittest/test",
)
BUNDLE_ID = "org.localmelo.app"
XCODE = "/Applications/Xcode.app/Contents/Developer"
NO_SWIFT = (
    "The app needs Apple's Swift compiler. Install it with: xcode-select --install"
)
LAUNCHER_PATH = "Contents/Resources/runtime/bin/melo"
# The app icon is Melo, the localmelo mascot, as the README shows the emblem (docs/assets/melo.png),
# on a rounded tile of the macOS icon grid: 824 of 1024 px, with 185 px corners. The emblem is scaled
# to the tile's height and fades into the tile over ICON_FEATHER px at its edges. The source is 534 px
# tall, so the 1024-px size shows it about 1.5 times larger than it is: those pixels are upscaled until
# larger artwork arrives, and every other size is made from the source by shrinking it.
ICON_GRID, ICON_TILE, ICON_RADIUS, ICON_FEATHER = 1024, 824, 185, 40
# The iconset names iconutil expects, with their pixel sizes.
ICONSET = {
    f"icon_{n}x{n}{suffix}.png": n * scale
    for n in (16, 32, 128, 256, 512)
    for suffix, scale in (("", 1), ("@2x", 2))
}

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


def trim_python(python: Path) -> None:
    """Leave out the parts of the standalone Python that the app never runs (UNUSED).

    Every match is checked before the first is removed: one that leads outside the
    payload through a link is refused like in verify_links, and nothing is removed then.
    A path that is not there is fine, and a link is removed, never followed.
    """
    root = python.resolve()
    found = sorted({m for p in UNUSED for m in glob.glob(p, root_dir=python)})
    for name in found:
        if not (python / name).resolve().is_relative_to(root):
            raise ValueError(f"Python payload has unsafe symlink: {python / name}")
    for name in found:
        path = python / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif os.path.lexists(path):  # not inside a folder that went before it
            path.unlink()


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


def read_png(path: Path) -> np.ndarray:
    """The pixels of an 8-bit PNG that is not interlaced: (height, width, 3), or 4 with alpha.

    Palette, RGB and RGBA files are read and the filters of their rows undone here, so the
    build needs no imaging library.
    """
    data, offset = path.read_bytes(), 8
    chunks: dict[bytes, bytes] = {}
    while offset < len(data):
        size, kind = struct.unpack(">I4s", data[offset : offset + 8])
        chunks[kind] = chunks.get(kind, b"") + data[offset + 8 : offset + 8 + size]
        offset += size + 12
    width, height, depth, color, _, _, interlace = struct.unpack(
        ">IIBBBBB", chunks[b"IHDR"]
    )
    if depth != 8 or interlace or color not in (2, 3, 6):
        raise ValueError(f"{path.name} must be an 8-bit RGB, RGBA or palette PNG")
    step = {2: 3, 3: 1, 6: 4}[color]  # bytes per pixel
    stride, raw = width * step, zlib.decompress(chunks[b"IDAT"])
    rows, above = [], bytes(stride)
    for y in range(height):
        mode = raw[y * (stride + 1)]
        row = bytearray(raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)])
        for i in range(stride if mode else 0):
            left = row[i - step] if i >= step else 0
            corner = above[i - step] if i >= step else 0
            if mode == 1:
                guess = left
            elif mode == 2:
                guess = above[i]
            elif mode == 3:
                guess = (left + above[i]) // 2
            else:  # Paeth: the neighbour nearest to left + above - corner, ties in this order
                near = left + above[i] - corner
                guess = min((left, above[i], corner), key=lambda c: abs(near - c))
            row[i] = (row[i] + guess) & 255
        rows.append(row)
        above = bytes(row)
    pixels = np.frombuffer(b"".join(rows), np.uint8).reshape(height, width, step)
    if color == 3:
        return np.frombuffer(chunks[b"PLTE"], np.uint8).reshape(-1, 3)[pixels[..., 0]]
    return pixels


def write_png(path: Path, pixels: np.ndarray) -> None:
    """Write (height, width, 3 or 4) pixels as a PNG that says it is sRGB."""
    height, width, channels = pixels.shape

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body)
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    color = 2 if channels == 3 else 6
    header = struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)
    rows = b"".join(
        b"\0" + row.tobytes() for row in pixels
    )  # filter 0: rows as they are
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"sRGB", b"\0")
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def resize(pixels: np.ndarray, height: int, width: int) -> np.ndarray:
    """(h, w, 3) pixels resampled to (height, width, 3) with a Lanczos filter, wider when shrinking."""

    def weights(n_in: int, n_out: int) -> np.ndarray:
        scale = n_in / n_out
        centers = (np.arange(n_out)[:, None] + 0.5) * scale
        x = (np.arange(n_in) + 0.5 - centers) / max(scale, 1)
        w = np.where(abs(x) < 3, np.sinc(x) * np.sinc(x / 3), 0)
        return w / w.sum(axis=1, keepdims=True)

    rows = np.tensordot(weights(len(pixels), height), pixels.astype(float), axes=1)
    return np.tensordot(rows, weights(pixels.shape[1], width), (1, 1)).transpose(
        0, 2, 1
    )


def icon_pixels(emblem: np.ndarray, size: int) -> np.ndarray:
    """The app icon at one size, (size, size, 4): the emblem on a rounded tile of its own paper.

    The tile's colour is the mean of the emblem's four corners. The emblem's paper differs a
    little along each of its edges, so a flat tile would show a rectangle: the emblem fades into
    the tile instead. Every size is resampled from the source itself, never from a larger size.
    """
    scale = size / ICON_GRID
    side, radius = ICON_TILE * scale, ICON_RADIUS * scale
    # The tile's coverage of each pixel, from its distance to a rounded square: smooth at any size.
    inset = np.abs(np.mgrid[:size, :size] + 0.5 - size / 2) - (side / 2 - radius)
    outside = (
        np.hypot(*np.maximum(inset, 0)) + np.minimum(inset.max(axis=0), 0) - radius
    )
    coverage = np.clip(0.5 - outside, 0, 1)
    corners = [emblem[:8, :8], emblem[:8, -8:], emblem[-8:, :8], emblem[-8:, -8:]]
    paper = np.mean(corners, axis=(0, 1, 2))
    height = round(side)
    width = round(height * emblem.shape[1] / emblem.shape[0])
    # The fade: 0 at the emblem's edges, rising smoothly to 1 ICON_FEATHER px inside them.
    ys, xs = np.arange(height) + 0.5, np.arange(width) + 0.5
    near = np.minimum(np.minimum(ys, height - ys)[:, None], np.minimum(xs, width - xs))
    fade = np.clip(near / (ICON_FEATHER * scale), 0, 1)
    fade = (fade * fade * (3 - 2 * fade))[..., None]
    top, left = (size - height) // 2, (size - width) // 2
    canvas = np.tile(paper, (size, size, 1))
    canvas[top : top + height, left : left + width] += (
        resize(emblem, height, width) - paper
    ) * fade
    return np.rint(np.dstack([canvas, coverage * 255]).clip(0, 255)).astype(np.uint8)


def make_iconset(iconset: Path) -> None:
    """Write the PNGs that iconutil reads, each made from docs/assets/melo.png."""
    emblem = read_png(ROOT / "docs/assets/melo.png")
    if emblem.shape[2] != 3:
        raise ValueError("docs/assets/melo.png must be a PNG without an alpha channel")
    iconset.mkdir()
    for name, size in ICONSET.items():
        write_png(iconset / name, icon_pixels(emblem, size))


def make_icon(resources: Path, stage: Path) -> None:
    """Render LocalMelo.icns from docs/assets/melo.png: the iconset, then iconutil, which ships with macOS."""
    iconset = stage / "LocalMelo.iconset"
    make_iconset(iconset)
    icns = resources / "LocalMelo.icns"
    run(["/usr/bin/iconutil", "-c", "icns", str(iconset), "-o", str(icns)])


def xcode_select_path() -> str:
    """The developer folder that `xcode-select -p` names, or "" when none is set."""
    done = subprocess.run(
        ["/usr/bin/xcode-select", "-p"], capture_output=True, text=True
    )
    return done.stdout.strip() if done.returncode == 0 else ""


def swift_developer_dir() -> str:
    """The developer folder whose swiftc builds the app, chosen in this order.

    DEVELOPER_DIR when the caller set it; else the full Xcode; else the folder that
    `xcode-select -p` names, when it has a Swift compiler (the Command Line Tools).
    """
    if os.environ.get("DEVELOPER_DIR"):
        return os.environ["DEVELOPER_DIR"]
    if Path(XCODE).is_dir():
        return XCODE
    selected = xcode_select_path()
    if selected and (Path(selected) / "usr/bin/swiftc").is_file():
        return selected
    raise RefusalError(NO_SWIFT)


def tools_name(developer: str) -> str:
    """The developer folder in words, for the build's summary."""
    if Path(developer).name == "CommandLineTools":
        return "the Command Line Tools"
    app = Path(developer).parent.parent.name  # the Xcode.app above Contents/Developer
    return app.removesuffix(".app") if app.endswith(".app") else developer


def build(output: Path) -> None:
    if sys.platform != "darwin" or os.uname().machine != "arm64":
        raise RuntimeError("Building requires an arm64 macOS host")
    developer = swift_developer_dir()  # refused here, before any work
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
        # Before the dependencies go in: a pattern can only meet what the archive has.
        trim_python(python)
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
        site = python / STDLIB / "site-packages"
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
        # The menu-bar item shows this logo, in colour.
        shutil.copy2(
            ROOT / "docs/assets/icon_small.png", contents / "Resources/icon_small.png"
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
                "DEVELOPER_DIR": developer,
            },
        )
        precompile(interpreter, python, site)
        make_icon(contents / "Resources", stage)
        with (contents / "Info.plist").open("wb") as f:
            plistlib.dump(
                {
                    "CFBundleExecutable": "LocalMelo",
                    "CFBundleIconFile": "LocalMelo.icns",
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
    print(f"Built with {tools_name(developer)}.", flush=True)


class RefusalError(RuntimeError):
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
        raise RefusalError(
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
    swift_developer_dir()  # refused before anything is made or printed
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
    except RefusalError as exc:
        parser.exit(1, f"{exc}\n")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Build failed: {exc}\n")


if __name__ == "__main__":
    main()
