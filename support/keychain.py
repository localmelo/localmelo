"""The macOS keychain, where a hosted service's API key is kept.

An item has the service ``localmelo.<name>`` and the account ``localmelo``. The
config holds only a reference to it (``keychain:localmelo.<name>``, see
:func:`parse_ref`); the key itself is read when a request needs it and is never
logged, saved to a file or returned by a route.

The key is written through ``security -i`` with the command on stdin, so it never
sits on a command line where ``ps`` could show it. It is read through
``security find-generic-password -w``, whose output is a pipe.

Every function that talks to ``security`` goes through :func:`_security`, which
tests replace. :func:`status` is cached for a few seconds, because config
validation asks about the same item on every health check.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time

from localmelo.support.words import t_in

ACCOUNT = "localmelo"
SERVICE_PREFIX = "localmelo."
REF_PREFIX = "keychain:"
SECURITY = "/usr/bin/security"
TIMEOUT_S = 15.0
# How long a status answer is reused.
STATUS_TTL_S = 5.0

_SERVICE_RE = re.compile(r"^localmelo\.[a-z0-9][a-z0-9_-]{0,39}$")
# A key is one run of printable ASCII, so a paste mistake is caught before
# anything is written.
_KEY_RE = re.compile(r"^[\x21-\x7e]{8,512}$")
# ``security`` exit codes: the item is not there; the keychain cannot be used now.
_NOT_FOUND = 44
_NOT_ALLOWED = {36, 45, 128}

SAVED = "saved"
MISSING = "missing"
LOCKED = "locked"
UNAVAILABLE = "unavailable"


class KeychainError(Exception):
    """The keychain could not do what was asked; the message says what to do.

    *kind* is ``missing``, ``locked``, ``unavailable``, ``bad`` or ``failed``.
    The message is the catalog's English words for it (``hosted.key.<kind>``);
    the console says the same in the person's language through :func:`words_key`.
    """

    def __init__(self, kind: str) -> None:
        super().__init__(t_in("en", words_key(kind)))
        self.kind = kind


def words_key(kind: str) -> str:
    """The catalog key of the words for a :class:`KeychainError` *kind*."""
    return f"hosted.key.{kind}"


def available() -> bool:
    """Whether this computer has the macOS keychain."""
    return sys.platform == "darwin" and os.path.exists(SECURITY)


def service_for(name: str) -> str:
    """The keychain service of the preset or key *name*."""
    return f"{SERVICE_PREFIX}{name}"


def ref_for(service: str) -> str:
    """The config reference to the item of *service*."""
    return f"{REF_PREFIX}{service}"


def valid_service(service: str) -> bool:
    """Whether *service* is one of melo's own item names."""
    return _SERVICE_RE.fullmatch(service) is not None


def parse_ref(text: str) -> str | None:
    """The service a config reference names; ``None`` for an empty reference.

    Raises :class:`ValueError` for anything else that is not a reference to one
    of melo's own items, so a hand-edited config cannot make melo read another
    program's password.
    """
    if not text:
        return None
    if text.startswith(REF_PREFIX) and valid_service(text[len(REF_PREFIX) :]):
        return text[len(REF_PREFIX) :]
    raise ValueError(
        f"{text!r} is not a keychain reference like keychain:localmelo.name"
    )


def clean_key(text: str) -> str:
    """*text* as an API key: trimmed, without a leading ``Bearer``.

    Raises :class:`KeychainError` (``bad``) when what is left is not one run
    of printable characters of a believable length. The message never contains
    the text.
    """
    key = text.strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    if not _KEY_RE.fullmatch(key):
        raise KeychainError("bad")
    return key


def _security(args: list[str], stdin: str | None = None) -> tuple[int, str, str]:
    """Run ``security`` with *args* (and *stdin*): exit code, stdout, stderr."""
    try:
        done = subprocess.run(  # noqa: S603 - a fixed program, no shell
            [SECURITY, *args],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        raise KeychainError(UNAVAILABLE) from exc
    except subprocess.TimeoutExpired as exc:
        raise KeychainError(LOCKED) from exc
    return done.returncode, done.stdout, done.stderr


def _blocked(code: int, err: str) -> KeychainError:
    lowered = err.lower()
    if (
        code in _NOT_ALLOWED
        or "interaction is not allowed" in lowered
        or "locked" in lowered
    ):
        return KeychainError(LOCKED)
    return KeychainError("failed")


_cache: dict[str, tuple[float, str]] = {}
_cache_lock = threading.Lock()


def _forget(service: str) -> None:
    with _cache_lock:
        _cache.pop(service, None)


def status(service: str) -> str:
    """Whether the item of *service* is there: ``saved``, ``missing``, ``locked`` or ``unavailable``.

    Reads no secret, and is reused for :data:`STATUS_TTL_S` seconds.
    """
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(service)
        if hit is not None and now - hit[0] < STATUS_TTL_S:
            return hit[1]
    if not available():
        answer = UNAVAILABLE
    else:
        try:
            code, _out, err = _security(
                ["find-generic-password", "-s", service, "-a", ACCOUNT]
            )
        except KeychainError as exc:
            answer = exc.kind
        else:
            if code == 0:
                answer = SAVED
            elif code == _NOT_FOUND:
                answer = MISSING
            else:
                answer = _blocked(code, err).kind
    with _cache_lock:
        _cache[service] = (now, answer)
    return answer


def read(service: str) -> str:
    """The key in the item of *service*.

    Raises :class:`KeychainError` with a plain fix when it is missing, locked or
    the keychain is not available.
    """
    if not available():
        raise KeychainError(UNAVAILABLE)
    code, out, err = _security(
        ["find-generic-password", "-s", service, "-a", ACCOUNT, "-w"]
    )
    if code == _NOT_FOUND:
        raise KeychainError(MISSING)
    if code != 0:
        raise _blocked(code, err)
    key = out.rstrip("\n")
    if not key:
        raise KeychainError(MISSING)
    return key


def write(service: str, key: str) -> None:
    """Save *key* in the item of *service*, replacing an earlier one.

    The command goes to ``security -i`` on stdin; the key is quoted there and is
    on no command line. Raises :class:`KeychainError` for a key that does not
    look like one, or a keychain that cannot be written.
    """
    if not valid_service(service):
        raise KeychainError("failed")
    key = clean_key(key)
    if not available():
        raise KeychainError(UNAVAILABLE)
    quoted = key.replace("\\", "\\\\").replace('"', '\\"')
    command = f'add-generic-password -U -s {service} -a {ACCOUNT} -w "{quoted}"\n'
    try:
        code, _out, err = _security(["-i"], command)
    finally:
        _forget(service)
    if code != 0:
        raise _blocked(code, err)


def remove(service: str) -> None:
    """Delete the item of *service*; nothing happens when it is not there."""
    if not available():
        raise KeychainError(UNAVAILABLE)
    try:
        code, _out, err = _security(
            ["delete-generic-password", "-s", service, "-a", ACCOUNT]
        )
    finally:
        _forget(service)
    if code not in (0, _NOT_FOUND):
        raise _blocked(code, err)
