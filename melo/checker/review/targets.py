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
from typing import Final
from urllib.parse import urlsplit

# A domain or a path chunk: ASCII only, so words beside it in another
# script (a Chinese sentence) do not stick to it.
_CHUNK: Final = re.compile(r"[A-Za-z0-9_.~/@+\-]+")
_URL: Final = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s\"'“”‘’<>`]+")
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


def built(text: str) -> bool:
    """Whether *text* is made at run time (``$DIR``, ``$(…)``, a backquote)."""
    return _BUILT.search(text) is not None


def parts(path: str) -> list[str] | None:
    """The folder and file names of *path* in lower case, or ``None`` when it cannot be read.

    ``./build/``, ``build`` and ``/tmp/x/build`` end in ``build``. Variables
    in the folders are allowed (``$TMP/build`` reads as ``build``); a name
    that is itself built, or a place such as ``.`` or ``~``, is not readable.
    """
    text = path.strip().strip("'\"")
    if not text:
        return None
    found = [p for p in text.split("/") if p not in ("", ".", "~")]
    if not found or found[-1] in _BROAD or built(found[-1]):
        return None
    return [p.casefold() for p in found]


def _glob_only(name: str) -> bool:
    return bool(name) and set(name) <= set("*?[]")


def path_matches(target: str, named: str) -> bool:
    """Whether the code's *target* is what the words' *named* path names.

    Equal names match (a glob in the target matches the names it covers), and
    so does a target inside the named folder (``build/out.o`` for ``build``).
    A target that is the folder holding the named file does not: the code
    deletes more than the words say.
    """
    have, want = parts(target), parts(named)
    if have is None or want is None:
        return False
    if _glob_only(have[-1]):
        # build/* : everything in the folder before it
        return bool(set(want) & set(have[:-1]))
    return fnmatch.fnmatchcase(want[-1], have[-1]) or want[-1] in have[:-1]


def host_name(text: str) -> str:
    """The host of an address, ``user@host:path`` or a bare host, lower case, else ``""``."""
    text = text.strip().strip("'\"")
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

    A name with no dot (a bucket, a remote) matches a label of the named host.
    """
    have, want = host_name(target), host_name(named)
    if not have or not want:
        return False
    if "." not in have:
        return have in want.split(".")
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
    text = _URL.sub(" ", words)
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
    return found


def named_hosts(words: str) -> list[str]:
    """The hosts the *words* name: web addresses, domains, addresses of mail, IPv4."""
    found: list[str] = []

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


def bare_words(words: str) -> set[str]:
    """The words of *words* as written, lower case, without the punctuation around them."""
    return {w.casefold() for w in (_strip(w) for w in words.split()) if w}
