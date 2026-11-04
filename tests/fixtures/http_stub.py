"""One HTTP server for tests: a handler in a thread on 127.0.0.1, on a port the system picks."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, Self

# ``serve_forever`` looks for a stop this often. Its default is 0.5 s, which
# ``close()`` waits out on average, so every test with a server paid that.
POLL_S = 0.01


def _no_log(self: BaseHTTPRequestHandler, format: str, *args: Any) -> None:
    """Write no request log into the test output."""


class HttpStub:
    """*handler* served in a thread, without its request log: ``base_url`` is the address plus *path*.

    It starts at once unless ``start=False``; ``start()`` then does it, and
    ``close()`` stops it. Use it as ``HttpStub(Handler)`` in a ``with``, or as
    the base of a class that builds its handler and records what it saw.
    """

    def __init__(
        self,
        handler: type[BaseHTTPRequestHandler],
        *,
        port: int = 0,
        path: str = "",
        start: bool = True,
    ) -> None:
        quiet = type(handler.__name__, (handler,), {"log_message": _no_log})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), quiet)
        self.httpd.daemon_threads = True
        self._path = path
        self._thread = threading.Thread(
            target=self.httpd.serve_forever,
            kwargs={"poll_interval": POLL_S},
            daemon=True,
        )
        if start:
            self.start()

    @property
    def port(self) -> int:
        return int(self.httpd.server_port)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}{self._path}"

    def start(self) -> Self:
        self._thread.start()
        return self

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()
