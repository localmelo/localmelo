"""Local gateway credentials and short-lived, single-use browser login codes."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import fcntl
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

    def __init__(
        self, path: Path | None = None, *, allow_corrupt: bool = False
    ) -> None:
        self.path = path or Path.home() / ".cache" / "localmelo" / "gateway" / "auth"
        self._directory = self._open_directory(self.path)
        self.rotation = asyncio.Event()
        try:
            with self._locked():
                self._create_token(allow_corrupt=allow_corrupt)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _open_directory(path: Path) -> int:
        path = Path(os.path.abspath(path))
        boundary = (
            path.parent
            if path.name == "auth" and path.parent.name == "gateway"
            else path
        )
        prefix = boundary.parent
        prefix.mkdir(mode=0o700, parents=True, exist_ok=True)
        prefix = prefix.resolve(strict=True)
        fd = os.open(prefix, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for component in path.relative_to(boundary.parent).parts:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(component, 0o700, dir_fd=fd)
                try:
                    child = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=fd,
                    )
                except OSError as exc:
                    raise OSError(
                        exc.errno,
                        f"Unsafe gateway auth path (symlinks are not allowed at or below {boundary}): {path}",
                    ) from exc
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

    @contextlib.contextmanager
    def _locked(self):  # type: ignore[no-untyped-def]
        fcntl.flock(self._directory, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._directory, fcntl.LOCK_UN)

    def _create_token(self, *, allow_corrupt: bool) -> None:
        try:
            data = self._read_token_file()
        except FileNotFoundError:
            self._write_token(replace=False)
            return
        if not data:
            self._write_token(replace=True)
        elif not allow_corrupt:
            self._validate_token(data)

    def _read_token_file(self) -> bytes:
        fd = os.open(
            TOKEN_FILE,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=self._directory,
        )
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
            return os.read(fd, 129)
        finally:
            os.close(fd)

    def _validate_token(self, data: bytes) -> bytes:
        if len(data) != 64 or any(c not in b"0123456789abcdef" for c in data):
            raise ValueError(
                f"Gateway token at {self.path / TOKEN_FILE} is invalid; "
                "run melo console --reset to replace it"
            )
        return data

    def read_token(self) -> bytes:
        return self._validate_token(self._read_token_file())

    def rotate(self) -> None:
        with self._locked():
            # Even a corrupt token must be a safe file before replacing it.
            with contextlib.suppress(FileNotFoundError):
                self._read_token_file()
            self._write_token(replace=True)

    def _write_token(self, *, replace: bool) -> None:
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
            if not replace:
                placeholder = os.open(
                    TOKEN_FILE,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=self._directory,
                )
                try:
                    os.fchmod(placeholder, 0o600)
                finally:
                    os.close(placeholder)
            os.replace(
                name, TOKEN_FILE, src_dir_fd=self._directory, dst_dir_fd=self._directory
            )
            self.invalidate_streams()
            os.fsync(self._directory)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(name, dir_fd=self._directory)

    # A code minted for the macOS app carries the app's per-launch instance.
    # The signature covers it, so only a gateway started with that instance
    # accepts the code: a process that takes the app's port receives a code it
    # cannot use. Other gateways use an empty instance.
    def issue_login(
        self, port: int, *, instance: str = "", now: int | None = None
    ) -> str:
        timestamp = int(time.time()) if now is None else now
        payload = f"{timestamp}.{port}.{instance}.{secrets.token_hex(16)}"
        return payload + "." + _mac(self.read_token(), "login", payload)

    def consume_login(
        self, code: str, port: int, *, instance: str = "", now: int | None = None
    ) -> str | None:
        if len(code) > 256 or not code.isascii():
            return None
        parts = code.split(".")
        if (
            len(parts) != 5
            or not parts[0].isdigit()
            or parts[1] != str(port)
            or parts[2] != instance
            or len(parts[3]) != 32
        ):
            return None
        timestamp = int(parts[0])
        current = int(time.time()) if now is None else now
        if timestamp > current or current - timestamp >= LOGIN_TTL:
            return None
        key = self.read_token()
        payload = ".".join(parts[:4])
        if not hmac.compare_digest(parts[4], _mac(key, "login", payload)):
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
