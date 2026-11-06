"""What the person's own words name: a key file, a script's address, a local server (#547).

Auto mode shows a card, instead of stopping a step without one, when the step
uses a key the person wrote, runs a script whose address they wrote, or reads
a server on this computer at an address they wrote; and it asks nothing about
a file in a folder they wrote. These functions read the
texts they are given, which are the person's own messages and never a page's
or a tool's, and say whether the text names the subject *exactly*: a key by
its path (``~/.ssh/id_ed25519``, ``$HOME/...`` or absolute) or by a file name
that is specific (``id_ed25519``), a script by its whole address, a server by
``localhost:3000``, ``127.0.0.1:3000`` or ``[::1]:3000``, a folder by its path
(it names every file under it). Naming a folder does not name a key file in
it, a longer name does not name a shorter one, and a host without its port or
a script's host without its path names nothing. They import only the standard
library and :mod:`localmelo.melo.contracts.folders`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from localmelo.melo.contracts.folders import above_home

__all__ = ["folder_named", "key_named", "local_sites", "loopback_site", "script_named"]

# A file name is specific when it holds a digit or one of _ . - : "id_ed25519"
# and ".npmrc", not "credentials" or "config".
_SPECIFIC = re.compile(r"[0-9_.-]")
MIN_NAME = 5
_LOOPBACK = {"localhost": "127.0.0.1", "127.0.0.1": "127.0.0.1", "[::1]": "::1"}
_SERVER = re.compile(
    r"(?<![\w.-])(?:https?://)?(localhost|127\.0\.0\.1|\[::1\]):([0-9]{1,5})(?![\w:.-]*\w)",
    re.IGNORECASE,
)


def _token(
    text: str, needle: str, *, before: str = r"\w./~$-", folder: bool = False
) -> bool:
    """Whether *needle* stands in *text* as a whole word of a path or address.

    Not inside a longer one: nothing that can continue a name stands right
    before it, and after it comes no name character, no ``/`` and no ``.``
    followed by one (``id_ed25519.pub`` is not ``id_ed25519``). A *folder* may
    be written with a ``/`` after it.
    """
    pattern = (
        rf"(?<![{before}]){re.escape(needle)}{'/?' if folder else ''}(?![\w/%-]|\.\w)"
    )
    return re.search(pattern, text) is not None


def _forms(path: str, home: str) -> set[str]:
    """The ways a person writes *path*: absolute, ``~/…``, ``$HOME/…``, ``${HOME}/…``."""
    homes = {home.rstrip("/"), os.path.realpath(home).rstrip("/")} - {""}
    forms = {path}
    for base in homes:
        if path.startswith(base + "/"):
            rest = path[len(base) + 1 :]
            forms |= {
                f"~/{rest}",
                f"$HOME/{rest}",
                f"${{HOME}}/{rest}",
                f"{base}/{rest}",
            }
    return forms


def key_named(path: str, texts: Iterable[str], home: str) -> bool:
    """Whether a text names the key file *path* (an absolute, canonical path)."""
    forms, name = _forms(path, home), os.path.basename(path)
    for text in texts:
        if any(_token(text, form, before=r"\w.~$-") for form in forms):
            return True
        if len(name) >= MIN_NAME and _SPECIFIC.search(name) and _token(text, name):
            return True
    return False


def folder_named(path: str, texts: Iterable[str], home: str) -> bool:
    """Whether a text names *path* or a folder above it (an absolute, canonical path).

    The home folder and the folders above it never count: naming ``~`` names
    nothing in it. A path written deeper than *path* names nothing of it.
    """
    texts = list(texts)
    here = path.rstrip("/")
    while here and here != "/" and not above_home(here, home=home):
        forms = _forms(here, home)
        if any(
            _token(t, f, before=r"\w.~$-", folder=True) for t in texts for f in forms
        ):
            return True
        here = os.path.dirname(here)
    return False


def script_named(url: str, texts: Iterable[str]) -> bool:
    """Whether a text holds the whole address *url* of a script (its scheme may differ)."""
    needle = url.strip().split("://", 1)[-1].rstrip("/")
    if "/" not in needle or len(needle) < MIN_NAME or any(c.isspace() for c in needle):
        return False
    return any(_token(text, needle, before=r"\w.-") for text in texts)


def local_sites(texts: Iterable[str]) -> frozenset[tuple[str, int]]:
    """The servers on this computer the texts write, as ``(address, port)``.

    ``localhost`` is read as ``127.0.0.1``. Only an explicit port counts; a
    bare ``localhost`` or an address on the network names nothing.
    """
    found: set[tuple[str, int]] = set()
    for text in texts:
        for host, port in _SERVER.findall(text):
            if 0 < int(port) < 65536:
                found.add((_LOOPBACK[host.lower()], int(port)))
    return frozenset(found)


def loopback_site(host: str) -> tuple[str, int] | None:
    """*host* (``localhost:3000``, ``127.0.0.1:5173``, ``[::1]:80``) as ``(address, port)``, else ``None``.

    Only a host of this computer with a port in it; ``localhost`` is read as
    ``127.0.0.1``.
    """
    found = _SERVER.fullmatch(host.strip())
    if found is None or not 0 < int(found.group(2)) < 65536:
        return None
    return _LOOPBACK[found.group(1).lower()], int(found.group(2))
