"""Read an art pack's files safely, from a folder or from a zip. Only the standard library is used.

An art pack comes from outside the repository, so it is untrusted data: this module reads names, sizes and bytes and
nothing else. Nothing in a pack is imported, run or opened by another program. `scripts/art_pack.py` calls it.

A zip is checked whole before one byte is written. It is refused when:

- it holds more than MAX_ENTRIES entries, an entry of more than MAX_ENTRY_BYTES, or more than MAX_TOTAL_BYTES in all
  (by the sizes it declares, and again by the bytes actually read);
- an entry name is an absolute path, climbs out with `..`, has a backslash, a NUL byte or a drive letter;
- an entry is a symbolic link or any other thing that is not a regular file or a folder (read from the entry's
  `external_attr`), or is encrypted;
- two entries have the same name, or names that differ only in case (a Mac would write both to one file).

Only the PNG and JSON files are extracted, into a folder the caller made; every other file is listed as skipped. The
zip's root is the pack, or the pack is the one top folder that holds `manifest.json`. A folder is read the same way
(its symbolic links refused, its counts and sizes capped), without copying anything.
"""

from __future__ import annotations

import os
import re
import stat
import zipfile
from collections.abc import Iterable
from pathlib import Path

MAX_ENTRIES = 2000
MAX_ENTRY_BYTES = 50 * 1024 * 1024
MAX_TOTAL_BYTES = 300 * 1024 * 1024
# The only kinds of file that are read; the rest of a pack is left where it is.
READ_SUFFIXES = (".png", ".json")
# What a Mac adds to a zip or a folder: not the pack's, never listed.
JUNK_PARTS = ("__MACOSX",)
JUNK_NAMES = (".DS_Store",)


class UnsafePackError(ValueError):
    """A zip or a folder that must not be read, worded for the person who ran the command."""


def is_junk(parts: tuple[str, ...]) -> bool:
    """A Mac's own files (`__MACOSX/`, `.DS_Store`, `._name`): not part of the pack."""
    return (
        any(part in JUNK_PARTS for part in parts)
        or parts[-1] in JUNK_NAMES
        or parts[-1].startswith("._")
    )


def safe_parts(name: str) -> tuple[str, ...]:
    """The folders and file of an entry name, or UnsafePackError when the name could leave the folder it is written in."""
    if not name or "\0" in name:
        raise UnsafePackError(f"{name!r}: an empty name or a NUL byte")
    if "\\" in name:
        raise UnsafePackError(f"{name!r}: a backslash in the name")
    if name.startswith("/"):
        raise UnsafePackError(f"{name!r}: an absolute path")
    if re.match(r"^[A-Za-z]:", name):
        raise UnsafePackError(f"{name!r}: a drive letter")
    parts = tuple(part for part in name.split("/") if part not in ("", "."))
    if ".." in parts:
        raise UnsafePackError(f"{name!r}: a path that climbs out with '..'")
    if not parts:
        raise UnsafePackError(f"{name!r}: no name")
    return parts


def check_archive(
    archive: zipfile.ZipFile,
) -> list[tuple[zipfile.ZipInfo, tuple[str, ...]]]:
    """Every entry of the zip checked; the files among them with their path parts. Nothing is read or written."""
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES:
        raise UnsafePackError(
            f"the zip has {len(infos)} entries, more than {MAX_ENTRIES}"
        )
    files: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
    seen: set[str] = set()
    total = 0
    for info in infos:
        parts = safe_parts(info.filename)
        if info.flag_bits & 0x1:
            raise UnsafePackError(f"{info.filename!r}: an encrypted entry")
        # The entry's Unix file type, from the high bits of `external_attr`; a zip made without one (by Python's own
        # zipfile, or on Windows) has 0 there and is a plain file.
        kind = stat.S_IFMT(info.external_attr >> 16)
        if kind == stat.S_IFLNK:
            raise UnsafePackError(f"{info.filename!r}: a symbolic link")
        if kind and kind not in (stat.S_IFREG, stat.S_IFDIR):
            raise UnsafePackError(f"{info.filename!r}: not a regular file or a folder")
        if kind and (kind == stat.S_IFDIR) != info.is_dir():
            raise UnsafePackError(f"{info.filename!r}: the name and the type disagree")
        if info.file_size > MAX_ENTRY_BYTES:
            raise UnsafePackError(
                f"{info.filename!r}: {info.file_size / 2**20:.0f} MB, "
                f"more than {MAX_ENTRY_BYTES // 2**20} MB for one file"
            )
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise UnsafePackError(
                f"the zip holds more than {MAX_TOTAL_BYTES // 2**20} MB in all"
            )
        if info.is_dir():
            continue
        key = "/".join(parts).casefold()
        if key in seen:
            raise UnsafePackError(
                f"{info.filename!r}: the name is used twice (names that differ only in case count)"
            )
        seen.add(key)
        files.append((info, parts))
    return files


def find_prefix(names: Iterable[str]) -> str:
    """Where the pack is among the files' relative paths: `""` when `manifest.json` is at the root, else `"Top/"`
    for the one top folder that holds it."""
    listed = set(names)
    if "manifest.json" in listed:
        return ""
    tops = sorted({name.split("/")[0] for name in listed if "/" in name})
    holders = [top for top in tops if f"{top}/manifest.json" in listed]
    if not holders:
        raise UnsafePackError(
            "no manifest.json at the root or in the one top folder (the pack must hold it)"
        )
    if len(holders) > 1:
        raise UnsafePackError(
            f"more than one folder holds a manifest.json: {', '.join(holders)}"
        )
    return f"{holders[0]}/"


def unpack(archive_path: Path, into: Path) -> tuple[list[str], list[str]]:
    """Extract the pack's PNG and JSON files from a zip into `into` (a new, empty folder the caller removes after).

    Returns the names written and the names left out (relative to the pack's root; files outside the pack's top folder
    keep their full name). Raises UnsafePackError, with nothing written, when the zip must be refused.
    """
    try:
        archive = zipfile.ZipFile(archive_path)
    except (zipfile.BadZipFile, OSError) as error:
        raise UnsafePackError(f"not a readable zip: {error}") from error
    with archive:
        members = [
            (info, parts)
            for info, parts in check_archive(archive)
            if not is_junk(parts)
        ]
        prefix = find_prefix("/".join(parts) for _, parts in members)
        written: list[str] = []
        skipped: list[str] = []
        total = 0
        base = into.resolve()
        for info, parts in members:
            name = "/".join(parts)
            if not name.startswith(prefix):
                skipped.append(name)
                continue
            inner = name[len(prefix) :]
            if not inner.lower().endswith(READ_SUFFIXES):
                skipped.append(inner)
                continue
            target = into.joinpath(*inner.split("/"))
            if not target.resolve().is_relative_to(base):
                raise UnsafePackError(f"{info.filename!r}: a path outside the pack")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, open(target, "xb") as sink:
                    left = info.file_size
                    while chunk := source.read(1 << 16):
                        left -= len(chunk)
                        total += len(chunk)
                        if left < 0 or total > MAX_TOTAL_BYTES:
                            raise UnsafePackError(
                                f"{info.filename!r}: more bytes than the zip says it holds"
                            )
                        sink.write(chunk)
            except (zipfile.BadZipFile, OSError) as error:
                raise UnsafePackError(f"{info.filename!r}: {error}") from error
            written.append(inner)
    return written, skipped


def list_folder(folder: Path) -> tuple[list[str], str]:
    """The files of a pack folder, relative and sorted, with where the pack is among them (`find_prefix`). Symbolic
    links and other things that are not regular files are refused, and the counts and sizes are capped.
    """
    if folder.is_symlink():
        raise UnsafePackError(f"{folder}: a symbolic link")
    names: list[str] = []
    total = 0
    for current, dirs, files in os.walk(folder, followlinks=False):
        for entry in [*dirs, *files]:
            path = Path(current, entry)
            relative = path.relative_to(folder).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise UnsafePackError(f"{relative!r}: a symbolic link")
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise UnsafePackError(f"{relative!r}: not a regular file or a folder")
            if is_junk(tuple(relative.split("/"))):
                continue
            size = path.lstat().st_size
            if size > MAX_ENTRY_BYTES:
                raise UnsafePackError(
                    f"{relative!r}: {size / 2**20:.0f} MB, "
                    f"more than {MAX_ENTRY_BYTES // 2**20} MB for one file"
                )
            total += size
            names.append(relative)
            if len(names) > MAX_ENTRIES:
                raise UnsafePackError(f"the folder holds more than {MAX_ENTRIES} files")
    if total > MAX_TOTAL_BYTES:
        raise UnsafePackError(
            f"the folder holds more than {MAX_TOTAL_BYTES // 2**20} MB"
        )
    names.sort()
    return names, find_prefix(names)
