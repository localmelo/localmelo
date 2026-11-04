"""The no-server screen: a pinned recommendation and the steps to do it by hand.

Nothing here runs, downloads, writes or opens a port; it only reads the platform.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from localmelo.support.backends.discovery import GUIDE_URL, how_to_start


@dataclass(frozen=True)
class Asset:
    name: str
    bytes: int
    sha256: str
    macos: tuple[int, ...]


@dataclass(frozen=True)
class Recipe:
    runtime_version: str
    download_base: str
    assets: tuple[Asset, ...]
    model_repo: str
    model_revision: str
    files: tuple[tuple[str, int], ...]

    @property
    def model_bytes(self) -> int:
        return sum(size for _, size in self.files)


PINNED_RECIPE = Recipe(
    "0.7.3",
    "https://github.com/jundot/omlx/releases/download/v0.7.3/",
    (
        Asset(
            "oMLX-0.7.3-macos15-sequoia.dmg",
            838158761,
            "cd5e3b454798a002f32ef2f3bf06ae033ea2f4e5b02aa039a2c68445498605f1",
            (15,),
        ),
        Asset(
            "oMLX-0.7.3-macos26-27.dmg",
            881239063,
            "2acfde41aeac0b022dbca803ca109437f3ccaf0c33f3415881526b985bee86f2",
            (26, 27),
        ),
    ),
    "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
    (
        (".gitattributes", 1570),
        ("README.md", 969),
        ("added_tokens.json", 707),
        ("chat_template.jinja", 4040),
        ("config.json", 938),
        ("generation_config.json", 238),
        ("merges.txt", 1671853),
        ("model.safetensors", 2263022417),
        ("model.safetensors.index.json", 63964),
        ("special_tokens_map.json", 613),
        ("tokenizer.json", 11422654),
        ("tokenizer_config.json", 5440),
        ("vocab.json", 2776833),
    ),
)

# The loopback port the steps start the server on: in the dynamic range, away
# from the ports melo and the common model servers use. Never bound by melo.
PORT = 49152


@dataclass(frozen=True)
class Detection:
    """What this computer is, and where the manual steps put things."""

    recipe: Recipe
    system: str
    ram_bytes: int | None
    asset: Asset | None  # None: no validated installer for this computer
    cache_root: str
    snapshot: str
    state_path: str


def cache_root(env: Mapping[str, str], home: Path) -> Path:
    """Use the Hub's cache precedence without importing or contacting the Hub."""
    # Replacements take precedence over still-functional deprecated variables:
    # https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#deprecated-environment-variables
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(name):
            return Path(env[name]).expanduser().absolute()
    hf_home = env.get("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser() / "hub").absolute()
    return (
        Path(env.get("XDG_CACHE_HOME") or home / ".cache").expanduser()
        / "huggingface"
        / "hub"
    ).absolute()


def ram_bytes() -> int | None:
    """Read physical RAM without starting a process (unknown is not supported)."""
    try:
        if platform.system() == "Darwin":
            value, size = ctypes.c_uint64(), ctypes.c_size_t(8)
            libc = ctypes.CDLL(None)
            if (
                libc.sysctlbyname(
                    b"hw.memsize", ctypes.byref(value), ctypes.byref(size), None, 0
                )
                == 0
            ):
                return int(value.value)
            return None
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def detect(recipe: Recipe = PINNED_RECIPE) -> Detection:
    """Read the platform and work out the paths; nothing is probed or contacted."""
    try:
        system, arch, version = (
            platform.system(),
            platform.machine(),
            platform.mac_ver()[0],
        )
    except (OSError, ValueError):
        system, arch, version = "unknown", "unknown", ""
    home = Path.home()
    cache = cache_root(os.environ, home)
    snapshot = (
        cache
        / ("models--" + recipe.model_repo.replace("/", "--"))
        / "snapshots"
        / recipe.model_revision
    )
    state = (
        home
        / ".cache"
        / "localmelo"
        / "server-setup"
        / f"omlx-{recipe.runtime_version}"
    )
    major = int(version.split(".")[0]) if version.split(".")[0].isdigit() else 0
    asset = (
        next((a for a in recipe.assets if major in a.macos), None)
        if system == "Darwin" and arch == "arm64"
        else None
    )
    return Detection(
        recipe, system, ram_bytes(), asset, str(cache), str(snapshot), str(state)
    )


def _size(value: int) -> str:
    return f"{value / 1_000_000_000:.1f} GB"


def _shell_path(path: str, home: Path | None = None) -> str:
    """Quote a path for a copy-ready command without printing the user name.

    A path inside the home folder (the home folder itself included) becomes
    "$HOME/<rest>" in double quotes, so the shell expands it; copy-ready
    commands never print the person's user name. Anything else is shlex-quoted.
    """
    if os.path.isabs(path):
        try:
            root = (Path.home() if home is None else home).resolve()
            target = Path(path).resolve()
            if target.is_relative_to(root):
                rest = target.relative_to(root).as_posix()
                if rest == ".":
                    return '"$HOME"'
                escaped = "".join(
                    "\\" + char if char in '\\"$`' else char for char in rest
                )
                return f'"$HOME/{escaped}"'
        except (OSError, RuntimeError, ValueError):
            pass
    return shlex.quote(path)


AUTOMATIC_SOON = "Automatic setup is coming soon; melo can show you how to do it now."


def preview(d: Detection, *, interactive: bool = True) -> tuple[str, ...]:
    """The short summary the no-server screen prints."""
    if d.asset is None:
        return (
            "No model server is running yet.",
            "Manual only: use a compatible model server.",
            (
                "Setup checks are unavailable; use the manual guide."
                if d.system == "unknown"
                else "No supported oMLX installer for this computer."
            ),
            "Manual guide:",
            GUIDE_URL,
        )
    lines = [
        "No model server is running yet.",
        "Recommended: oMLX with Qwen3 4B Instruct.",
        f"Download: up to {_size(d.recipe.model_bytes + d.asset.bytes)}",
        (AUTOMATIC_SOON if interactive else f"Guide: {GUIDE_URL}"),
    ]
    if d.ram_bytes is None:
        lines.append("RAM unknown; check the 16 GB recommendation in the manual guide.")
    elif d.ram_bytes < 16 * 1024**3:
        lines.append("Under 16 GB RAM: this model may be slow; check the manual guide.")
    return tuple(lines)


# Numbered-step commands are indented by this much, so a terminal prints them
# whole (never word-wrapped) and a reader can tell them from the sentences.
COMMAND_INDENT = " " * 5


def numbered_steps(d: Detection) -> tuple[str, ...]:
    """At most six plain numbered steps to set a server up by hand, then the guide.

    Copy-only: melo runs none of this. Commands are indented by
    ``COMMAND_INDENT``; paths inside the home folder are written as ``$HOME``.
    """
    r = d.recipe
    if d.asset is None:
        how = how_to_start()
        return (
            f"1. Install and start a model server, for example {how.name}:",
            *(f"{COMMAND_INDENT}{command}" for command in how.commands),
            f"   {how.note}",
            "2. Back in melo, choose Enter an address and type the server's address.",
            f"More help: {GUIDE_URL}",
        )
    download = f'"$HOME/Downloads/{d.asset.name}"'
    return (
        f"1. Download oMLX ({_size(d.asset.bytes)}): {r.download_base}{d.asset.name}",
        "2. Optional: check the download by running:",
        f"{COMMAND_INDENT}shasum -a 256 {download}",
        f"   It should print this checksum: {d.asset.sha256}",
        "3. Open the file and move oMLX to Applications. Use the normal macOS "
        "prompts; do not bypass Gatekeeper. Then make Contents/MacOS/omlx-cli "
        "inside the app available as the command omlx.",
        f"4. Download the model (up to {_size(r.model_bytes)}; files already cached "
        "are skipped):",
        f"{COMMAND_INDENT}HF_HUB_CACHE={_shell_path(d.cache_root)} hf download "
        f"{shlex.quote(r.model_repo)} --revision {shlex.quote(r.model_revision)}",
        "5. Start the server. Leave that window open; Ctrl+C stops it:",
        f"{COMMAND_INDENT}OMLX_HF_CACHE_ENABLED=0 omlx serve "
        f"--base-path {_shell_path(d.state_path)} "
        f"--model-dir {_shell_path(d.snapshot)} "
        f"--host 127.0.0.1 --port {PORT}",
        f"6. Back in melo, choose Enter an address and type http://127.0.0.1:{PORT}/v1",
        f"More help: {GUIDE_URL}",
    )
