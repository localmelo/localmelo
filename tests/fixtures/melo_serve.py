"""A fake melo_serve: its ``melo_serve.state/1`` state file, and a control socket.

The socket accepts connections (what makes a state running) and, given a
:class:`FakeServe`, answers ``GET /picks`` and ``POST /switch`` as melo_serve's
control socket does, with the four chat picks the mockups show.
"""

from __future__ import annotations

import json
import shutil
import socketserver
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

URL = "http://127.0.0.1:8321/v1"
# The Mac of the picks below: 64 GB, and 31 GB there for a model that replaces the 27B.
TOTAL_GB = 64
FREE_FOR_A_PICK_GB = 31


def role(model: str, backend: str = "splash", name: str = "Splash") -> dict[str, Any]:
    return {
        "pick": model.lower(),
        "short": "m",
        "name": model,
        "model": model,
        "backend": backend,
        "backend_name": name,
        "status": "ready",
    }


def write_state(
    home: Path,
    *,
    chat: dict[str, Any] | None = None,
    embedding: dict[str, Any] | None = None,
    status: str = "ready",
    login: bool = True,
    generation: int = 1,
) -> None:
    """Write ``state.json`` in *home*. It says nothing is running until a socket listens."""
    state = {
        "schema": "melo_serve.state/1",
        "generation": generation,
        "status": status,
        "front_door": {"url": URL, "pid": 1, "auth": "none"},
        "service": {"login": login},
        "roles": {"chat": chat, "embedding": embedding},
    }
    path = home / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    path.chmod(0o600)  # as melo_serve writes it: the umask cannot matter


def pick_row(
    pick_id: str,
    name: str,
    backend: str,
    model: str,
    mem_gb: float,
    *,
    verdict: str = "fits",
    load_s: float = 20,
    role_name: str = "chat",
) -> dict[str, Any]:
    """One row of ``GET /picks``, its fit worked out as melo_serve does on the Mac above."""
    after = (FREE_FOR_A_PICK_GB - mem_gb) / TOTAL_GB * 100
    return {
        "id": pick_id,
        "short": pick_id.split("-")[1],
        "name": name,
        "role": role_name,
        "backend": backend,
        "backend_name": {"splash": "Splash", "omlx": "oMLX"}[backend],
        "model": model,
        "summary": "",
        "disk_gb": round(mem_gb * 0.85, 1),
        "mem_gb": mem_gb,
        "load_s": load_s,
        "decode_tps": None,
        "caps": [],
        "current": False,
        "problem": "",
        "fit": {
            "verdict": verdict,
            "need_gb": mem_gb,
            "after_pct": round(after, 1),
            "measured": True,
            "reason": "",
        },
    }


def picks_rows() -> list[dict[str, Any]]:
    """The mockups' picks: the 27B is served; the 70B does not fit; one embedding pick."""
    return [
        pick_row(
            "qwen3.8-27b-splash", "Qwen3.8 27B", "splash", "Qwen3.8-27B", 21, load_s=40
        ),
        pick_row("qwen3-8b-omlx", "Qwen3 8B", "omlx", "Qwen3-8B-4bit", 5),
        pick_row("qwen3-4b-omlx", "Qwen3 4B", "omlx", "Qwen3-4B-4bit", 2.5, load_s=12),
        pick_row(
            "qwen3-70b-splash",
            "Qwen3 70B",
            "splash",
            "Qwen3-70B",
            48,
            verdict="too_big",
        ),
        pick_row("bge-m3-omlx", "BGE M3", "omlx", "bge-m3", 1.2, role_name="embedding"),
    ]


@dataclass
class FakeServe:
    """What the fake control socket answers, and what it was asked.

    ``GET /picks`` answers *reply* when it is set (to send something
    malformed), else *rows* with the current one marked. ``POST /switch``
    waits for *gate* when there is one, refuses with *refuse* (a status and an
    ``error`` object) when set, and refuses a stale ``expected_generation`` as
    melo_serve does; otherwise it serves the pick from then on, one
    *generation* later, and writes the state file in *home*.
    """

    home: Path
    rows: list[dict[str, Any]] = field(default_factory=picks_rows)
    current: str = "qwen3.8-27b-splash"
    generation: int = 1
    reply: Any = None
    refuse: tuple[int, dict[str, Any]] | None = None
    gate: threading.Event | None = None
    switches: list[dict[str, Any]] = field(default_factory=list)

    def write(self) -> None:
        """The state file as melo_serve writes it for what it serves now."""
        row = next(row for row in self.rows if row["id"] == self.current)
        chat = role(row["model"], row["backend"], row["backend_name"])
        chat.update(pick=row["id"], short=row["short"], name=row["name"])
        write_state(self.home, chat=chat, generation=self.generation)

    def picks(self) -> Any:
        if self.reply is not None:
            return self.reply
        rows = [{**row, "current": row["id"] == self.current} for row in self.rows]
        return {"memory": {"total_gb": TOTAL_GB, "free_pct": 15.0}, "picks": rows}

    def switch(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        self.switches.append(body)
        if self.gate is not None:
            self.gate.wait(timeout=10)
        if self.refuse is not None:
            return self.refuse
        expected = body.get("expected_generation")
        if expected is not None and expected != self.generation:
            error = {"type": "stale_generation", "message": "the state moved on"}
            return 409, {"error": {**error, "generation": self.generation}}
        changed = body["pick"] != self.current
        if changed:
            self.current, self.generation = body["pick"], self.generation + 1
            self.write()
        model = next(row["model"] for row in self.rows if row["id"] == self.current)
        reply = {"changed": changed, "generation": self.generation, "role": "chat"}
        return 200, {**reply, "pick": self.current, "model": model, "fit": None}


class _Control(BaseHTTPRequestHandler):
    """A connection and nothing else is all a probe asks for; a request gets its answer."""

    server: _Server

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet: a Unix socket's client has no address to log."""

    def _answer(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        fake = self.server.fake
        if fake is None or self.path != "/picks":
            self._answer(404, {"error": {"type": "not_here", "message": "no"}})
            return
        self._answer(200, fake.picks())

    def do_POST(self) -> None:
        fake = self.server.fake
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if fake is None or self.path != "/switch":
            self._answer(404, {"error": {"type": "not_here", "message": "no"}})
            return
        self._answer(*fake.switch(body))


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    fake: FakeServe | None = None


@contextmanager
def listening(fake: FakeServe | None = None) -> Iterator[Path]:
    """A control socket that accepts connections: what makes a state running.

    With *fake*, it also answers the control requests. It lives in a short
    folder under ``/tmp``, because macOS allows 104 bytes for a socket path,
    which pytest's ``tmp_path`` exceeds. Both are removed on exit.
    """
    folder = Path(tempfile.mkdtemp(dir="/tmp"))
    server = _Server(str(folder / "control.sock"), _Control)
    server.fake = fake
    # As melo_serve does: the umask cannot matter.
    (folder / "control.sock").chmod(0o600)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield folder / "control.sock"
    finally:
        if fake is not None and fake.gate is not None:
            fake.gate.set()
        server.shutdown()
        server.server_close()
        shutil.rmtree(folder, ignore_errors=True)
