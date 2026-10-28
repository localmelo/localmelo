"""Loopback address validation shared by the gateway and local clients."""

from __future__ import annotations

import ipaddress
import socket

_LOOPBACK_NAMES = frozenset({"localhost", "localhost."})


def is_loopback(host: str) -> bool:
    """Accept loopback names and numeric addresses without DNS lookup.

    Numeric forms include those accepted by binding, such as 127.1,
    2130706433, bracketed IPv6 and IPv4-mapped loopback addresses.
    """
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if host.lower() in _LOOPBACK_NAMES:
        return True
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(
            host, None, type=socket.SOCK_STREAM, flags=socket.AI_NUMERICHOST
        )
    except (OSError, UnicodeError, ValueError):
        return False
    return bool(infos) and all(_is_loopback_ip(str(info[4][0])) for info in infos)


def _is_loopback_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address.partition("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback
