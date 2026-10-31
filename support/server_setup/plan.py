"""Pinned, immutable setup previews. Probes read metadata; nothing executes or writes."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shlex
import shutil
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

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
    runtime_commit: str
    download_base: str
    release_source: str
    assets: tuple[Asset, ...]
    model_repo: str
    model_revision: str
    files: tuple[tuple[str, int], ...]
    model_source: str


PINNED_RECIPE = Recipe(
    "0.7.3",
    "9de79bc39a01c1f3fec1e849903aaf0dd5478d7a",
    "https://github.com/jundot/omlx/releases/download/v0.7.3/",
    "https://api.github.com/repos/jundot/omlx/releases/tags/v0.7.3",
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
    "https://huggingface.co/api/models/mlx-community/Qwen3-4B-Instruct-2507-4bit/tree/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b?recursive=true",
)


@dataclass(frozen=True)
class DiskSpace:
    volume: str
    paths: tuple[str, ...]
    payload_bytes: int
    free_bytes: int | None
    # Expansion/staging/native cache overhead has not been measured.
    overhead_bytes: int | None = None


@dataclass(frozen=True)
class Detection:
    system: str
    architecture: str
    os_version: str
    ram_bytes: int | None
    existing_servers: tuple[str, ...]
    cache_root: str
    snapshot: str
    state_path: str
    completed_files: tuple[str, ...]
    missing_model_bytes: int
    asset: Asset | None
    support: str
    disks: tuple[DiskSpace, ...]


@dataclass(frozen=True)
class SetupPlan:
    version: int
    recipe: Recipe
    detection: Detection
    port: int
    keep_running: bool = False
    allowed_operations: tuple[str, ...] = ()  # Preview only, never consent.

    @property
    def digest(self) -> str:
        manifest = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(manifest.encode()).hexdigest()


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


def _completed(snapshot: Path, store: Path, cache: Path, name: str, size: int) -> bool:
    try:
        repo = cache.resolve() / store.name
        if (
            store.resolve() != repo
            or snapshot.resolve() != repo / "snapshots" / snapshot.name
        ):
            return False
        target = (snapshot / name).resolve(strict=True)
        if not target.is_relative_to(repo) or target.name.endswith(".incomplete"):
            return False
        info = target.stat()
        return (
            stat.S_ISREG(info.st_mode)
            and info.st_size == size
            and os.access(target, os.R_OK)
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _disk(path: Path) -> tuple[str, int | None]:
    """Find the existing ancestor and mount for a prospective directory."""
    try:
        at = path.resolve()
        while not at.exists():
            if at.parent == at:
                return str(path), None
            at = at.parent
        device = at.stat().st_dev
        mount = at
        while mount.parent != mount and mount.parent.stat().st_dev == device:
            mount = mount.parent
        return str(mount), shutil.disk_usage(at).free
    except (OSError, RuntimeError, ValueError):
        return str(path), None


def detect(
    *, existing_servers: Sequence[str] = (), recipe: Recipe = PINNED_RECIPE
) -> Detection:
    """Reuse discovery's result; never probe endpoints or read server settings."""
    try:
        system, arch, version = (
            platform.system(),
            platform.machine(),
            platform.mac_ver()[0],
        )
    except (OSError, ValueError):
        system, arch, version = "unknown", "unknown", ""
    memory = ram_bytes()
    home = Path.home()
    cache = cache_root(os.environ, home)
    store = cache / ("models--" + recipe.model_repo.replace("/", "--"))
    snapshot = store / "snapshots" / recipe.model_revision
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
    if asset is None:
        support = "Manual only: no validated oMLX installer for this platform/architecture/version."
    elif memory is None:
        support = "RAM unknown; at least 16 GB RAM recommended."
    elif memory < 16 * 1024**3:
        support = "Under 16 GB RAM: this model may be slow."
    else:
        support = "Hardware matches recommendation (16 GB RAM recommended); memory/quality not measured."
    completed = tuple(
        name
        for name, size in recipe.files
        if _completed(snapshot, store, cache, name, size)
    )
    missing = sum(size for name, size in recipe.files if name not in completed)
    volumes: dict[str, DiskSpace] = {}
    for path, needed in ((cache, missing), (state, asset.bytes if asset else 0)):
        volume, free = _disk(path)
        old = volumes.get(volume)
        if old:
            needed += old.payload_bytes
            free = (
                min(old.free_bytes, free)
                if old.free_bytes is not None and free is not None
                else None
            )
        volumes[volume] = DiskSpace(
            volume, (*old.paths, str(path)) if old else (str(path),), needed, free
        )
    return Detection(
        system,
        arch,
        version,
        memory,
        tuple(existing_servers),
        str(cache),
        str(snapshot),
        str(state),
        completed,
        missing,
        asset,
        support,
        tuple(volumes.values()),
    )


def detect_bounded(*, timeout: float = 2.0) -> Detection:
    """Bound slow metadata I/O; daemon threads never delay interpreter shutdown."""
    results: Queue[Detection | Exception] = Queue(maxsize=1)

    def probe() -> None:
        try:
            results.put(detect())
        except Exception as exc:
            results.put(exc)

    Thread(target=probe, name="melo-setup-detect", daemon=True).start()
    try:
        result = results.get(timeout=timeout)
    except Empty:
        return Detection(
            "unknown",
            "unknown",
            "",
            None,
            (),
            "unknown",
            "unknown",
            "unknown",
            (),
            sum(size for _, size in PINNED_RECIPE.files),
            None,
            "Setup checks timed out; use the manual guide.",
            (),
        )
    if isinstance(result, Exception):
        raise result
    return result


def port_allowed(port: int) -> bool:
    return (
        port not in {8955, 11434, 11500}
        and not 8400 <= port <= 8799
        and 49152 <= port <= 65535
    )


def choose_port(occupied: Sequence[int] = ()) -> int:
    """Prospective choice only, excluding known occupiers; never bind/reserve."""
    return next(
        port
        for port in range(49152, 65536)
        if port_allowed(port) and port not in occupied
    )


def plan(
    detection: Detection, *, recipe: Recipe = PINNED_RECIPE, keep_running: bool = False
) -> SetupPlan:
    return SetupPlan(1, recipe, detection, choose_port(), keep_running)


def _bytes(value: int | None) -> str:
    return (
        "unknown"
        if value is None
        else f"{value / 1_000_000_000:.2f} GB ({value:,} bytes)"
    )


def _size(value: int | None) -> str:
    if value is None:
        return "unknown"
    if 0 < value < 100_000_000:
        return "<0.1 GB"
    return f"{value / 1_000_000_000:.1f} GB" if value else "0 GB"


def _ram(value: int | None) -> str:
    return "unknown" if value is None else f"{value / 1024**3:g} GB ({value:,} bytes)"


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


def preview(setup: SetupPlan, *, interactive: bool = True) -> tuple[str, ...]:
    """A human summary; technical metadata requires Details or developer mode."""
    d = setup.detection
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
    total = sum(size for _, size in setup.recipe.files)
    download = d.missing_model_bytes + d.asset.bytes
    free = " / ".join(_size(disk.free_bytes) for disk in d.disks) or "unknown"
    space = "Free/volume" if len(d.disks) > 1 else "Free"
    lines = [
        "No model server is running yet.",
        "Recommended: oMLX with Qwen3 4B Instruct.",
        f"Download: up to {_size(download)} · Cached: {_size(total - d.missing_model_bytes)} · {space}: {free}",
        (
            "Planned lifetime: keep running after melo closes."
            if setup.keep_running
            else "Planned lifetime: stop when melo closes."
        ),
        (
            "Automatic setup is coming soon: Show manual steps to do it now."
            if interactive
            else f"Guide: {GUIDE_URL}"
        ),
    ]
    warnings = []
    help_target = "Details" if interactive else "the manual guide"
    if d.ram_bytes is None:
        warnings.append(
            f"RAM unknown; check the 16 GB recommendation in {help_target}."
        )
    elif d.ram_bytes < 16 * 1024**3:
        warnings.append(
            f"Under 16 GB RAM: this model may be slow; check {help_target}."
        )
    if any(disk.free_bytes is None for disk in d.disks):
        warnings.append(f"Free space unknown; check {help_target} before manual setup.")
    if any(
        disk.free_bytes is not None and disk.free_bytes < disk.payload_bytes
        for disk in d.disks
    ):
        warnings.append(f"Not enough disk space for the download; check {help_target}.")
    if warnings:
        lines.append(
            warnings[0]
            if len(warnings) == 1
            else f"Several setup checks need attention; see {help_target} before manual setup."
        )
    return tuple(lines)


def details(setup: SetupPlan) -> tuple[str, ...]:
    """The full read-only manifest, including unmeasured overhead and sources."""
    d, r = setup.detection, setup.recipe
    lines = [
        "No model server is running yet. Setup preview only; nothing installed or started.",
        *(["Recommended: oMLX + Qwen3 4B Instruct."] if d.asset else []),
        f"Platform: {d.system} {d.os_version} / {d.architecture}; RAM: {_ram(d.ram_bytes)}.",
        d.support,
        f"SetupPlan v{setup.version} SHA256: {setup.digest}",
        f"Pinned: oMLX {r.runtime_version} ({r.runtime_commit}); {r.model_repo} @ {r.model_revision}.",
        f"Model download: up to {_bytes(d.missing_model_bytes)} missing of {_bytes(sum(n for _, n in r.files))}; {len(d.completed_files)}/{len(r.files)} exact-size cached files (not integrity-verified).",
        f"HF cache: {d.cache_root}",
        f"Snapshot (read-only): {d.snapshot}",
        f"New melo-owned state: {d.state_path}",
    ]
    if d.asset:
        lines += [
            f"Runtime download: {_bytes(d.asset.bytes)}; {r.download_base}{d.asset.name}",
            f"Runtime SHA256: {d.asset.sha256}",
        ]
    for disk in d.disks:
        lines += [
            f"Disk at {disk.volume}: {_bytes(disk.payload_bytes)} payload needed; {_bytes(disk.free_bytes)} free. Paths: {', '.join(disk.paths)}."
        ]
        if disk.free_bytes is not None and disk.free_bytes < disk.payload_bytes:
            lines += ["Not enough space for payload alone. Nothing installed."]
    lines += [
        "Disk expansion/staging overhead: unknown (unmeasured); payload is a lower bound, not a guaranteed fit.",
        f"Keep available after melo closes: {'On' if setup.keep_running else 'Off'} (preview only).",
        (
            "Planned lifetime: keep running until explicitly stopped."
            if setup.keep_running
            else "Planned lifetime: stop when this melo session closes."
        ),
        "Manual commands run independently; stop them yourself with Ctrl+C.",
        f"Prospective loopback port: {setup.port} (availability unverified; not reserved).",
        "Automatic setup is not available yet. No approval is recorded.",
    ]
    return tuple(lines)


def manual_steps(setup: SetupPlan) -> tuple[str, ...]:
    d, r = setup.detection, setup.recipe
    lifetime = "Manual commands run independently; stop them yourself with Ctrl+C."
    if d.asset is None:
        how = how_to_start()
        return (
            "Manual fallback: use an existing compatible model server, then melo setup.",
            lifetime,
            f"Existing platform guide for {how.name} (copy only; check compatibility on this machine):",
            *how.commands,
            how.note,
            f"Official release information: {r.release_source}",
            GUIDE_URL,
        )
    return (
        "Manual steps (copy only; melo does not run these):",
        lifetime,
        f"Download: {r.download_base}{d.asset.name}",
        f"Verify SHA256 {d.asset.sha256}; install through normal macOS controls. Make the bundled Contents/MacOS/omlx-cli available as omlx; do not bypass Gatekeeper.",
        f"HF_HUB_CACHE={_shell_path(d.cache_root)} hf download {shlex.quote(r.model_repo)} --revision {shlex.quote(r.model_revision)}",
        f"OMLX_HF_CACHE_ENABLED=0 omlx serve --base-path {_shell_path(d.state_path)} --model-dir {_shell_path(d.snapshot)} --host 127.0.0.1 --port {setup.port}",
        f"Check the port is free before starting; enumerate http://127.0.0.1:{setup.port}/v1/models. The direct snapshot's served ID may be its revision hash rather than the repository ID; use the server-returned ID.",
        "Run melo setup and enter that server address (the prospective port is outside default discovery).",
    )
