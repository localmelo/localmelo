"""Local gateway credentials and short-lived, single-use browser login codes."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import os
import secrets
import stat
import time
from pathlib import Path

TOKEN_FILE = "token"
LOGIN_TTL = 60
COOKIE_TTL = 30 * 24 * 60 * 60


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _mac(key: bytes, purpose: str, payload: str) -> str:
    return _encode(hmac.digest(key, f"{purpose}:{payload}".encode(), "sha256"))


class GatewayAuth:
    """A private directory holding the bearer and consumed login-code markers."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path.home() / ".cache" / "localmelo" / "gateway" / "auth"
        self._directory = self._open_directory(self.path)
        try:
            self._create_token()
        except BaseException:
            self.close()
            raise
        self.rotation = asyncio.Event()

    @staticmethod
    def _open_directory(path: Path) -> int:
        # Walk from / with O_NOFOLLOW: a symlink at any component is refused.
        path = Path(os.path.abspath(path))
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            for component in path.parts[1:]:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(component, 0o700, dir_fd=fd)
                child = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                os.close(fd)
                fd = child
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise PermissionError(
                    "Gateway auth directory must be owned by you and mode 0700"
                )
            return fd
        except BaseException:
            os.close(fd)
            raise

    def close(self) -> None:
        if getattr(self, "_directory", -1) >= 0:
            os.close(self._directory)
            self._directory = -1

    def __del__(self) -> None:
        self.close()

    def _create_token(self) -> None:
        try:
            fd = os.open(
                TOKEN_FILE,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._directory,
            )
        except FileExistsError:
            self.read_token()
            return
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(secrets.token_hex(32).encode("ascii"))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(TOKEN_FILE, dir_fd=self._directory)
            raise

    def read_token(self) -> bytes:
        fd = os.open(TOKEN_FILE, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._directory)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
            ):
                raise PermissionError(
                    "Gateway token must be a private 0600 regular file"
                )
            data = os.read(fd, 129)
            if len(data) != 64 or any(c not in b"0123456789abcdef" for c in data):
                raise ValueError("Gateway token is invalid")
            return data
        finally:
            os.close(fd)

    def rotate(self) -> None:
        name = ".token-" + secrets.token_hex(16)
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._directory,
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(secrets.token_hex(32).encode("ascii"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                name, TOKEN_FILE, src_dir_fd=self._directory, dst_dir_fd=self._directory
            )
            self.invalidate_streams()
            os.fsync(self._directory)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(name, dir_fd=self._directory)

    def issue_login(self, port: int, *, now: int | None = None) -> str:
        timestamp = int(time.time()) if now is None else now
        payload = f"{timestamp}.{port}.{secrets.token_hex(16)}"
        return payload + "." + _mac(self.read_token(), "login", payload)

    def consume_login(
        self, code: str, port: int, *, now: int | None = None
    ) -> str | None:
        if len(code) > 256 or not code.isascii():
            return None
        parts = code.split(".")
        if (
            len(parts) != 4
            or not parts[0].isdigit()
            or parts[1] != str(port)
            or len(parts[2]) != 32
        ):
            return None
        timestamp = int(parts[0])
        current = int(time.time()) if now is None else now
        if timestamp > current or current - timestamp >= LOGIN_TTL:
            return None
        key = self.read_token()
        payload = ".".join(parts[:3])
        if not hmac.compare_digest(parts[3], _mac(key, "login", payload)):
            return None
        marker = "used-" + hashlib.sha256(code.encode()).hexdigest()
        try:
            fd = os.open(
                marker,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._directory,
            )
        except FileExistsError:
            return None
        os.close(fd)
        cookie = secrets.token_urlsafe(32)
        name = "session-" + hashlib.sha256(cookie.encode()).hexdigest()
        payload = f"{current}.{port}"
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._directory,
        )
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            stream.write(payload + "." + _mac(key, "session:" + name, payload))
        return cookie

    def valid_cookie(self, cookie: str, port: int, *, now: int | None = None) -> bool:
        try:
            name = "session-" + hashlib.sha256(cookie.encode("ascii")).hexdigest()
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._directory)
        except (UnicodeEncodeError, FileNotFoundError):
            return False
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
            ):
                return False
            parts = os.read(fd, 256).decode("ascii").split(".")
        except (UnicodeDecodeError, OSError):
            return False
        finally:
            os.close(fd)
        if len(parts) != 3 or not parts[0].isdigit() or parts[1] != str(port):
            return False
        current = int(time.time()) if now is None else now
        timestamp = int(parts[0])
        payload = ".".join(parts[:2])
        return (
            timestamp <= current
            and current - timestamp < COOKIE_TTL
            and hmac.compare_digest(
                parts[2], _mac(self.read_token(), "session:" + name, payload)
            )
        )

    def invalidate_streams(self) -> None:
        previous, self.rotation = self.rotation, asyncio.Event()
        previous.set()

    def revoke_cookie(self, cookie: str) -> None:
        try:
            name = "session-" + hashlib.sha256(cookie.encode("ascii")).hexdigest()
            os.unlink(name, dir_fd=self._directory)
        except (UnicodeEncodeError, FileNotFoundError):
            pass

    def valid_bearer(self, token: str) -> bool:
        try:
            return hmac.compare_digest(token.encode("ascii"), self.read_token())
        except UnicodeEncodeError:
            return False
