"""melo's own ports: the gateway's and the model servers' the config names.

No page, fetch or relay may reach these, whatever a list allows: web_fetch's
list of test servers (:mod:`localmelo.support.web_local`) and the background
browser's sign-in helpers (:mod:`localmelo.support.browser.local_apps`) both
refuse them. This module reads only the config and the running gateway's
endpoint file, so the browser can use it without reading web_fetch's setting.
It also reads which servers on this computer the person wrote the address of
(:func:`written_sites`, #547), which both tools let through, and the browser
in the background (Chrome) too.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from urllib.parse import SplitResult, urlsplit

from localmelo.melo.contracts.approvals import TOOL_ASK
from localmelo.support import gateway_endpoint
from localmelo.support.backends.openai_family import CLOUD_BACKENDS, LOCAL_BACKENDS

if TYPE_CHECKING:
    from localmelo.support.config import Config

GATEWAY = "the port of melo's own gateway and console"
MODEL = "the port of a model server"
_SCHEME_PORTS = {"http": 80, "https": 443}
# The addresses of this computer a written server can have (``localhost`` is read as the first).
LOOPBACK = ("127.0.0.1", "::1")


def _url_port(url: object) -> int | None:
    """The port a model server *url* names, or its scheme's own; ``None`` if unreadable."""
    if not isinstance(url, str) or not url.strip():
        return None
    text = url.strip()
    try:
        parts = urlsplit(text if "://" in text else f"http://{text}")
        port = parts.port
    except ValueError:
        return None
    return port or _SCHEME_PORTS.get(parts.scheme.lower())


def urls_of(cfg: Config) -> list[str]:
    """Every model server URL *cfg* sets: the local backends' two and the cloud ones' ``base_url``."""
    urls: list[object] = []
    for name in LOCAL_BACKENDS:
        section = getattr(cfg, name)
        urls.extend((section.chat_url, section.embedding_url))
    urls.extend(getattr(cfg, name).base_url for name in CLOUD_BACKENDS)
    return [url for url in urls if isinstance(url, str) and url.strip()]


def own_ports(
    cfg: Config, urls: Iterable[str | None] = (), *, running: bool = False
) -> dict[int, str]:
    """The ports that are melo's own, each with what it is; none can be listed.

    The ``[gateway]`` port of *cfg*, which is also the console's (the live
    config of a running session has the port it really bound), and every
    port of the model servers the config names (``chat_url`` and
    ``embedding_url`` of the local backends, ``base_url`` of the cloud ones).
    A port is refused whatever its host: a server on this computer is
    reachable at ``127.0.0.1`` from any of its addresses. *urls* are more
    model server URLs (the run's ``-u``). With *running*, the port of a
    gateway that runs (``--port 0`` binds one that is in no config) is
    added.
    """
    ports: dict[int, str] = {}
    gateway = cfg.gateway.port
    if (
        isinstance(gateway, int)
        and not isinstance(gateway, bool)
        and 0 < gateway < 65536
    ):
        ports.setdefault(gateway, GATEWAY)
    if running:
        try:
            found = gateway_endpoint.read()
        except Exception:  # an unreadable file leaves the other ports
            found = None
        if found is not None and 0 < found.port < 65536:
            ports.setdefault(found.port, GATEWAY)
    for url in [*urls, *urls_of(cfg)]:
        port = _url_port(url)
        if port is not None:
            ports.setdefault(port, MODEL)
    return ports


def written_sites() -> frozenset[tuple[str, int]]:
    """The servers on this computer the person's own words write in this run, as ``(address, port)`` (#547).

    The run's question policy says it from the person's messages
    (``written_sites``): the address is ``127.0.0.1`` or ``::1`` (``localhost``
    is read as the first) and the port is the one they wrote. A page's or a
    model's address never counts, and with no policy (or one that cannot say)
    the answer is none. The ports of melo's own gateway and model servers are
    the callers' to refuse: none is ever let through, written or listed
    (:func:`own_ports`).
    """
    found = getattr(TOOL_ASK.get(), "written_sites", None)
    return frozenset(found()) if callable(found) else frozenset()


def written_site(parts: SplitResult) -> tuple[str, int] | None:
    """The server on this computer *parts* names, when the person's own words wrote it (#547).

    The host is ``localhost``, ``127.0.0.1`` or ``[::1]``, the port is in the
    address, and :func:`written_sites` has that host and port: ``(address,
    port)`` with ``localhost`` read as ``127.0.0.1``. Anything else is ``None``.
    """
    try:
        host, port = (parts.hostname or "").lower(), parts.port
    except ValueError:
        return None
    host = "127.0.0.1" if host == "localhost" else host
    if host not in LOOPBACK or port is None or (host, port) not in written_sites():
        return None
    return host, port


__all__ = [
    "GATEWAY",
    "LOOPBACK",
    "MODEL",
    "own_ports",
    "urls_of",
    "written_site",
    "written_sites",
]
