"""``[tools] web_fetch_allow_local``: the test servers on this computer ``web_fetch`` may read.

This module is outside :mod:`localmelo.support.web` on purpose: the config reads
the rules below to validate the key, and importing that package loads the whole
tool. Only :func:`fetcher` imports the fetcher, when it is called.

A developer setting, empty (off) by default, for a test that needs
``web_fetch`` to read a fixture page served on this computer. It loosens the
rule that ``web_fetch`` never reaches this computer (see
:mod:`localmelo.support.web.guard`), so the rules around it are strict:

- An entry is exactly ``127.0.0.1:PORT`` or ``[::1]:PORT``: one of those two
  addresses and one port from 1 to 65535. No host name (``localhost``), no
  other address (private or local network ranges, ``127.0.0.2``, ``0.0.0.0``),
  no range of ports and no wildcard. Any other entry is a config error
  (:func:`problems`, reported by ``Config.validate`` and ``-X``) and is never
  used (:func:`sites_of`).
- The port of melo's own gateway and console, of a gateway that is running,
  and of every model server (the configured URLs and the ``-u`` URL of the
  run) is refused even when it is listed (:func:`own_ports`). A session
  passes its own live config: ``melo serve --port N`` sets the gateway port
  in memory only, so the saved file does not know it. A model server that is
  in neither the config nor the run's URLs is not protected.
- The list is read when a session's ``web_fetch`` is made. A chat with a
  person asks about a site on it (see ``WebFetchProvider._admit``; in auto
  mode each URL is reviewed by itself), an unattended run reads it, and a call
  of the shell or Python cannot set it (the shell review keeps :data:`KEY`'s full
  name, ``tools.web_fetch_allow_local``, among the settings it guards).
- The list reaches only ``web_fetch``'s fetcher, as its ``allow`` hook
  (:class:`~localmelo.support.web.guard.LocalAllow`); the browser tools use
  the guard's own checks with no hook. A redirect meets the same list.
- It is config only: the file, or ``-X`` for one run. No tool argument,
  console setting or message sets it.
- A fetch it allows is marked: ``allowed_local`` in the call's
  ``tool.finished`` event, and :data:`NOTE` in what the call shows.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from localmelo.support import config
from localmelo.support.local_ports import GATEWAY, MODEL, own_ports, urls_of

if TYPE_CHECKING:
    from localmelo.support.config import Config
    from localmelo.support.web.fetch import WebFetcher

KEY = "web_fetch_allow_local"
# The only hosts an entry may have, as written in the entry and as the guard reads them.
HOSTS = {"127.0.0.1": "127.0.0.1", "[::1]": "::1"}
FORM = "127.0.0.1:PORT or [::1]:PORT"
NOTE = (
    "[This address is a test server on this computer, which "
    "[tools] web_fetch_allow_local lists.]"
)
# One more sentence for the model's note about web_fetch while the list has an entry.
MODEL_NOTE = (
    " One exception: a developer setting lets web_fetch read the test servers on "
    "this computer that the developer listed (an address like http://127.0.0.1:PORT/); "
    "every other address on this computer stays refused."
)

_ENTRY = re.compile(r"(127\.0\.0\.1|\[::1\]):([1-9][0-9]{0,4})")
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9.-]*")
_RANGE = re.compile(r"[0-9]+-[0-9]+")


def parse(entry: object) -> tuple[str, int] | str:
    """The site ``(address, port)`` of *entry*, or why it is not an entry (a sentence)."""
    if not isinstance(entry, str):
        return f"{entry!r} must be text such as {FORM}"
    if not entry:
        return f"an entry is empty; write {FORM}"
    if any(ch.isspace() or ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F for ch in entry):
        return f"{entry!r} has spaces or hidden characters in it; write {FORM}"
    if "*" in entry:
        return f"{entry!r} has a wildcard; list each address and port as {FORM}"
    found = _ENTRY.fullmatch(entry)
    if found and int(found.group(2)) <= 65535:
        return HOSTS[found.group(1)], int(found.group(2))
    host, port = _split(entry)
    if port is None or port == "":
        return f"{entry!r} has no port; write {FORM}"
    if _NAME.fullmatch(host):
        return (
            f"{entry!r} is a name; names such as localhost are not allowed, "
            f"only the addresses in {FORM}"
        )
    if host not in HOSTS:
        return (
            f"{entry!r} is not allowed: web_fetch reads only {FORM}, "
            "with a port from 1 to 65535"
        )
    if _RANGE.fullmatch(port):
        return f"{entry!r} is a range of ports; list each port as {FORM}"
    return (
        f"{entry!r} has a port that is not a number from 1 to 65535 "
        "(digits only, no leading zero)"
    )


def _split(entry: str) -> tuple[str, str | None]:
    """The host and the port text of *entry* as written; no port is ``None``."""
    if entry.startswith("["):
        end = entry.find("]")
        if end == -1:
            return entry, None
        host, rest = entry[: end + 1], entry[end + 1 :]
        return host, rest[1:] if rest.startswith(":") else None
    host, colon, port = entry.rpartition(":")
    return (host, port) if colon else (entry, None)


def sites_of(entries: object) -> frozenset[tuple[str, int]]:
    """The valid sites of *entries*; anything that is not a valid entry is left out."""
    if not isinstance(entries, list):
        return frozenset()
    sites = (parse(entry) for entry in entries)
    return frozenset(site for site in sites if not isinstance(site, str))


def problems(cfg: Config) -> list[str]:
    """The config errors of ``[tools] web_fetch_allow_local``, one sentence each."""
    entries: Any = cfg.tools.web_fetch_allow_local
    if not isinstance(entries, list):
        return [
            f"[tools] {KEY} must be a list of entries such as {FORM}, got {entries!r}"
        ]
    errors: list[str] = []
    ports = own_ports(cfg)
    for entry in entries:
        site = parse(entry)
        if isinstance(site, str):
            errors.append(f"[tools] {KEY}: {site}")
        elif site[1] in ports:
            errors.append(
                f"[tools] {KEY}: {entry!r} is {ports[site[1]]}, which web_fetch "
                "never reads, even when it is listed"
            )
    return errors


def fetcher(
    cfg: Config | None = None, *, protect: Iterable[str | None] = ()
) -> WebFetcher:
    """The :class:`WebFetcher` for *cfg* (the saved config with this run's ``-X``, if None).

    It has a :class:`~localmelo.support.web.guard.LocalAllow` when the list
    holds a valid entry, and no hook at all otherwise. *protect* are the
    model server URLs of this run that are not in the config (``-u``); their
    ports are refused too, as are the config's and a running gateway's, each
    time an address is checked.
    """
    from localmelo.support.web.fetch import WebFetcher
    from localmelo.support.web.guard import LocalAllow

    if cfg is None:
        try:
            cfg = config.load()
        except Exception:  # an unreadable config gives no list, never a wider one
            return WebFetcher()
    sites = sites_of(cfg.tools.web_fetch_allow_local)
    if not sites:
        return WebFetcher()
    urls = tuple(protect)
    loaded = cfg
    return WebFetcher(
        allow=LocalAllow(sites, lambda: own_ports(loaded, urls, running=True))
    )


__all__ = [
    "FORM",
    "GATEWAY",
    "HOSTS",
    "KEY",
    "MODEL",
    "MODEL_NOTE",
    "NOTE",
    "fetcher",
    "own_ports",
    "parse",
    "problems",
    "sites_of",
    "urls_of",
]
