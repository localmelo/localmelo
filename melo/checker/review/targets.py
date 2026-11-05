"""What a sentence names, and what a command acts on, compared by name.

The mismatch warning of :mod:`~localmelo.melo.checker.review.gist` asks two
questions about a step the model described in words: which files, folders or
hosts do the words name, and which does the code delete or send to. This
module answers both from text alone and says whether they match.

It is deliberately conservative. A word counts as a name only when it is
written as one: a path (``build/``, ``/tmp/x``, ``notes.tmp``, ``.cache``), a
quoted or backticked word, a web address or a domain. A bare word such as
"build" is not taken for a name (it is also an ordinary word), but it may
rescue a match when it spells one of the code's targets. The code's target
that melo cannot read (a variable, a command substitution, ``.``) is never
compared at all.
"""

from __future__ import annotations

import fnmatch
import re
import unicodedata
from typing import Final
from urllib.parse import urlsplit

# A domain or a path chunk: ASCII only, so words beside it in another
# script (a Chinese sentence) do not stick to it.
_CHUNK: Final = re.compile(r"[A-Za-z0-9_.~/@+\-]+")
_URL: Final = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]{0,31}://[^\s\"'“”‘’<>`]+")
# A disk written as a device name (``disk3``, ``/dev/rdisk3s1``).
_DISK: Final = re.compile(
    r"(?<![\w/.\-])(?:/dev/)?r?disk\d+(?:s\d+)*(?![\w\-])(?!\.\w)", re.I
)
_QUOTED: Final = re.compile(
    r"'([\w.\-/~*]{1,80})'|\"([\w.\-/~*]{1,80})\"|`([\w.\-/~*]{1,80})`"
    r"|“([\w.\-/~*]{1,80})”|‘([\w.\-/~*]{1,80})’"
)
_EXTENSION: Final = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,7}$")
_DOMAIN: Final = re.compile(r"[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9][a-z0-9\-]*)+")
_IPV4: Final = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
# Abbreviations that read like a file name or a domain.
_NOT_NAMES: Final = frozenset(
    ["e.g", "i.e", "etc", "vs", "approx", "a.k.a", "cf", "p.m", "a.m", "no"]
)
# The ends of file names, which a domain's last label is not taken for.
_FILE_ENDINGS: Final = frozenset(
    re.split(
        r"\s+",
        "md py sh txt json log pdf zip csv png jpg jpeg gif yaml yml toml html htm js ts"
        " tsx jsx cfg ini tar gz tgz bz2 xz rar doc docx xls xlsx ppt pptx rtf svg mp3"
        " mp4 mov wav avi sql db sqlite bak tmp swp o a so dylib pyc pyo class jar war"
        " rb go rs c h cpp hpp java kt swift php pl lua r xml lock env conf plist dmg"
        " pkg iso img bin out key pem crt cer sig asc gpg csr ipynb npy npz pt gguf"
        " safetensors onnx parquet tsv jsonl epub ics vcf eml scpt command app",
    )
)
# What cannot be read as one name: a variable, a substitution, a template.
_BUILT: Final = re.compile(r"[$`{}]|\$\(")
# A target that is the place the command runs in, not a thing it names.
_BROAD: Final = frozenset([".", "..", "/", "~", "*", "**", ""])


def _norm(text: str) -> str:
    """*text* in compatibility form (NFKC), so a full-width spelling is the plain one."""
    return unicodedata.normalize("NFKC", text)


def built(text: str) -> bool:
    """Whether *text* is made at run time (``$DIR``, ``$(…)``, a backquote)."""
    return _BUILT.search(text) is not None


def parts(path: str) -> list[str] | None:
    """The folder and file names of *path* in lower case, or ``None`` when it cannot be read.

    ``./build/``, ``build`` and ``/tmp/x/build`` end in ``build``; ``src/*``
    ends in a glob inside ``src``. Variables in the folders are allowed
    (``$TMP/build`` reads as ``build``); a name that is itself built is not
    readable, and neither is a place that holds everything around it (see
    :func:`broad`).
    """
    text = _norm(path).strip().strip("'\"")
    found = [p for p in text.split("/") if p not in ("", ".", "~")]
    if not found or built(found[-1]) or broad(path):
        return None
    if _glob_only(found[-1]) and any(built(p) for p in found[:-1]):
        return None  # everything in a folder melo cannot name
    return [p.casefold() for p in found]


def broad(path: str) -> bool:
    """Whether *path* is a place that holds everything around it (``.``, ``~``, ``/``, ``*``, ``..``).

    Deleting one is known to be much more than any one file the words name,
    so it is compared (and never matches), unlike a path built at run time.
    """
    text = _norm(path).strip().strip("'\"")
    if not text or built(text):
        return False
    found = [p for p in text.split("/") if p not in ("", ".", "~")]
    return not found or found[-1] == ".." or (len(found) == 1 and found[0] in _BROAD)


def _wild(name: str) -> bool:
    return any(char in name for char in "*?[")


def _glob_only(name: str) -> bool:
    return bool(name) and set(name) <= set("*?[]")


def path_matches(target: str, named: str) -> bool:
    """Whether the code's *target* is what the words' *named* path names.

    Equal names match, and so does a target inside the named folder
    (``build/out.o`` for ``build``). A target that is the folder holding the
    named file does not: the code deletes more than the words say. Nor does
    a glob match one named file it covers (``src/*`` for ``src/a.py``): it
    matches only an equal glob, the extension it ends in (``*.log`` for
    ``.log``) or the named folder it lists.
    """
    have, want = parts(target), parts(named)
    if have is None or want is None:
        return False
    if _wild(have[-1]):
        if want[-1] in have[:-1]:
            return True  # the folder the glob lists (or one above it)
        last = have[-1]
        return want[-1] == last or (
            last.startswith("*.") and not _wild(last[1:]) and want[-1] == last[1:]
        )
    # A named glob covers the target it matches (``*.log`` for ``debug.log``).
    return (
        want[-1] == have[-1]
        or want[-1] in have[:-1]
        or (_wild(want[-1]) and fnmatch.fnmatchcase(have[-1], want[-1]))
    )


def host_name(text: str) -> str:
    """The host of an address, ``user@host:path`` or a bare host, lower case, else ``""``."""
    text = _norm(text).strip().strip("'\"")
    if "://" in text:
        try:
            host = urlsplit(text).hostname or ""
        except ValueError:
            return ""
    else:
        host = text.rsplit("@", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    host = host.casefold().removeprefix("www.")
    return "" if built(host) else host


def host_matches(target: str, named: str) -> bool:
    """Whether the code's *target* host is the host the words name (or one of its subdomains).

    A name with no dot (a bucket, a remote) matches only the same name
    written the same way: it is not a part of a domain.
    """
    have, want = host_name(target), host_name(named)
    if not have or not want:
        return False
    if "." not in have or "." not in want:
        return have == want
    return have == want or have.endswith("." + want) or want.endswith("." + have)


def _strip(chunk: str) -> str:
    """*chunk* without the punctuation that ends a sentence around it."""
    chunk = chunk.strip(",;:!?()[]<>\"'")
    chunk = chunk.rstrip(".")
    return chunk


def _explicit_path(chunk: str) -> bool:
    """Whether *chunk* is written as a path: it has a slash at an end or start, an ending, or a dot first."""
    last = chunk.rstrip("/").rsplit("/", 1)[-1]
    if not last or chunk.casefold() in _NOT_NAMES or chunk.isdigit():
        return False
    if chunk.startswith(("/", "~", "./", "../")) or chunk.endswith("/"):
        return True
    return bool(_EXTENSION.search(last)) or (
        last.startswith(".") and len(last) > 1 and last[1].isalnum()
    )


def named_paths(words: str) -> list[str]:
    """The files and folders the *words* name by writing them as such."""
    text = _URL.sub(" ", _norm(words))
    found: list[str] = []
    for quoted in _QUOTED.findall(text):
        name = next(q for q in quoted if q)
        if name not in found and not _glob_only(name):
            found.append(name)
    for chunk in _CHUNK.findall(text):
        chunk = _strip(chunk)
        if (
            chunk
            and _explicit_path(chunk)
            and chunk not in found
            and "@" not in chunk  # an address, not a path
        ):
            found.append(chunk)
    for disk in _DISK.findall(text):
        if disk.casefold() not in found:
            found.append(disk.casefold())
    return found


def named_hosts(words: str) -> list[str]:
    """The hosts the *words* name: web addresses, domains, addresses of mail, IPv4."""
    found: list[str] = []
    words = _norm(words)

    def add(host: str) -> None:
        if host and host not in found:
            found.append(host)

    for url in _URL.findall(words):
        add(host_name(url.rstrip(".,;)")))
    text = _URL.sub(" ", words)
    for chunk in _CHUNK.findall(text):
        chunk = _strip(chunk)
        if "@" in chunk:
            chunk = chunk.rsplit("@", 1)[-1]
        low = chunk.casefold().removeprefix("www.").split("/", 1)[0]
        if _IPV4.fullmatch(low):
            add(low)
        elif _DOMAIN.fullmatch(low):
            last = low.rsplit(".", 1)[-1]
            if (
                last not in _FILE_ENDINGS
                and not last.isdigit()
                and low not in _NOT_NAMES
            ):
                add(low)
    return found


# A word that says where a thing is, not what it is: "old.txt in Documents".
_PLACE_BEFORE: Final = (
    r"(?:in|inside|within|from|under|beneath|below|of|at|on|into)\s+"
    r"(?:(?:the|my|our|your|this|that|its)\s+)?(?:[\w.\-]+\s+)?"
)
_PLACE_AFTER: Final = r"\s*(?:里|中|下|内|目录下|文件夹里|文件夹中|目录里|目录中)"
_PLACE_ZH: Final = "(?:在|于|从)\\s*"


def spells(words: str, name: str) -> bool:
    """Whether *words* write *name* as the thing itself, not as the place of another.

    The "build" of "the build folder" or "build and dist" spells the name; the
    "Documents" of "old.txt in Documents" or "src里的old.tmp" says where the
    named file is, so deleting that folder is not what the words say.
    """
    low = _norm(words).casefold()
    word = re.escape(name.casefold())
    for found in re.finditer(rf"(?<![\w.\-/]){word}(?![\w\-/]|\.\w)", low):
        before, after = low[: found.start()], low[found.end() :]
        located = (
            re.search(rf"\b{_PLACE_BEFORE}$", before)
            or re.match(_PLACE_AFTER, after)
            or re.search(rf"{_PLACE_ZH}$", before)
        )
        if not located:
            return True
    return False
