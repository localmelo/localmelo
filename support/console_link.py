"""Open the web console in the browser, signed in with a one-use link."""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import webbrowser

# Set over SSH, where a browser would open on the other computer, and in the
# macOS app's gateway, which signs its own window in.
_NOT_AT_THE_SCREEN = (
    "SSH_CONNECTION",
    "SSH_CLIENT",
    "SSH_TTY",
    "LOCALMELO_APP_INSTANCE",
)


def can_open() -> bool:
    """A person at this computer's terminal: not over SSH, a daemon, a pipe or the app."""
    return (
        sys.stdin is not None
        and sys.stdout is not None
        and sys.stdin.isatty()
        and sys.stdout.isatty()
        and not any(os.environ.get(name) for name in _NOT_AT_THE_SCREEN)
    )


def listening(host: str, port: int) -> bool:
    """Whether something accepts connections at *host*:*port*: a gateway to sign in to."""
    try:
        socket.create_connection((host.strip("[]"), port), timeout=1).close()
    except OSError:
        return False
    return True


def show_sign_in(base: str, code: str, *, opened: str) -> bool:
    """Open *base* signed in with *code* and print *opened*, else print the link.

    The link is a credential for a minute, so it goes to a terminal only.
    ``$BROWSER`` picks the browser, as for any Python program.
    """
    link = f"{base}#login={code}"
    # A browser that will not start is a reason to print the link, not to fail.
    with contextlib.suppress(Exception):
        if can_open() and webbrowser.open(link):
            print(opened)
            return True
    from localmelo.support.gateway.auth import LOGIN_TTL

    print(f"Open this link within {LOGIN_TTL} seconds. Do not share it:\n{link}")
    return False


def open_console(host: str, port: int) -> None:
    """What ``melo serve`` runs once the gateway listens: open the console, signed in."""
    from localmelo.support.gateway.auth import GatewayAuth

    auth = GatewayAuth()
    try:
        code = auth.issue_login(port)
    finally:
        auth.close()
    show_sign_in(
        f"http://{host}:{port}/",
        code,
        opened="  Web console: opened in your browser (melo console opens it again).",
    )
