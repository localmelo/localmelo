"""``melo memory``: list, forget, status and reindex, and the setup question.

The notes are the console's, read from the memory folder. Forgetting asks as a
list on a terminal (keep first) and by number when input is piped. The setup
asks "Remember across chats with <model>?" when the server lists an embedding
model, checked against a fake server that lists one and serves embeddings.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.support import command_words, config, memory_cli, onboard
from localmelo.support.backends import discovery
from localmelo.support.memory import ChatMemory
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.tests.cli.test_choice_lists import DOWN, ENTER, ESC, seen, typed
from localmelo.tests.memory.chat_memory_helpers import teach, vector

HOLDER = "test-host:1:command"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "cfg" / "config.toml"))
    yield tmp_path


async def _seed(home: Path) -> None:
    store = SessionStore(str(home / "sessions.db"), writer="gateway")
    await store.open()
    memory = ChatMemory(lambda: None)
    try:
        for text in ("I prefer short answers.", "My dentist is Lee Park."):
            turn = await store.begin_turn(
                None, text, holder=HOLDER, new_session=NewSession()
            )
            await teach(memory, store, turn, text)
    finally:
        await memory.aclose()
        await store.aclose()


@pytest.fixture()
def notes(home: Path) -> Path:
    asyncio.run(_seed(home))
    return home


def _count() -> int:
    async def count() -> int:
        memory = ChatMemory(lambda: None)
        try:
            return len(await memory.records())
        finally:
            await memory.aclose()

    return asyncio.run(count())


def test_list_prints_the_notes_newest_first_and_status_counts_them(
    notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert memory_cli.main(["list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].endswith("About you    Your dentist is Lee Park.")
    assert lines[1].strip().startswith("from chat “My dentist is Lee Park.”, ")
    assert "You prefer short answers." in lines[2]
    assert memory_cli.main(["list", "-n", "1"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 2
    assert memory_cli.main(["status"]) == 0
    status = capsys.readouterr().out
    assert "on, across chats" in status and "notes       2" in status
    assert "by words only" in status and "~/.cache/localmelo/memory" in status


def test_an_empty_list_says_how_memory_starts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert memory_cli.main(["list"]) == 0
    assert capsys.readouterr().out.startswith(
        "Nothing remembered yet. melo keeps a note"
    )


def test_forget_one_note_by_number(
    notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    async def first_id() -> int:
        memory = ChatMemory(lambda: None)
        try:
            return (await memory.records())[0].id
        finally:
            await memory.aclose()

    wanted = asyncio.run(first_id())
    assert memory_cli.main(["forget", "999", "-y"]) == 1
    assert "there is no note 999" in capsys.readouterr().err
    assert memory_cli.main(["forget", str(wanted), "-y"]) == 0
    assert f"Forgot note {wanted}." in capsys.readouterr().out
    assert _count() == 1


@pytest.mark.parametrize(("answer", "left"), [("1", 2), ("", 2), ("2", 0)])
def test_forget_everything_asks_by_number_when_input_is_piped(
    notes: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
    left: int,
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(answer + "\n"))
    code = memory_cli.main(["forget"])
    out = capsys.readouterr().out
    assert (
        "Forget 2 notes and melo's search index of 0 chat turns? Your chats stay."
        in out
    )
    assert "[1] Keep everything" in out and "[2] Forget everything" in out
    assert (code, _count()) == ((0 if left == 0 else 1), left)


def test_forget_everything_is_a_list_on_a_terminal_and_keeps_first(
    notes: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for keys, left in ((ENTER, 2), (ESC, 2), (DOWN + ENTER, 0)):
        asyncio.run(_seed(notes)) if _count() == 0 else None
        with typed(monkeypatch, keys) as drawn:
            memory_cli.main(["forget"])
        text = seen(drawn)
        assert "❯ 1. Keep everything" in text and "2. Forget everything" in text
        assert _count() == left, keys


def test_yes_forgets_everything_without_asking(
    notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert memory_cli.main(["forget", "-y"]) == 0
    assert "Forgot everything melo remembered." in capsys.readouterr().out
    assert _count() == 0


def test_reindex_without_chats_or_a_model_says_what_it_did(
    notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert memory_cli.main(["reindex"]) == 0
    assert (
        "Indexed 0 chat turns and gave 0 notes a vector, by words only"
        in capsys.readouterr().out
    )


def test_the_command_word_has_help_and_refuses_a_stray_word(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as helped:
        command_words.expand(["memory", "-h"])
    assert helped.value.code == 0
    assert "melo memory [list" in capsys.readouterr().out
    with pytest.raises(SystemExit) as refused:
        command_words.expand(["memory", "what", "do", "you", "know"])
    assert refused.value.code == 2
    assert 'melo: "memory" is a melo command' in capsys.readouterr().err


# ── Setup: "Remember across chats with <model>?" against a server that lists an embedding model ──


class _Server:
    """``GET /v1/models`` lists a chat and an embedding model; ``POST /v1/embeddings`` serves vectors."""

    def __init__(self) -> None:
        self.embedded: list[list[str]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def _send(self, code: int, body: dict[str, Any]) -> None:
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.endswith("/models"):
                    self._send(
                        200,
                        {"data": [{"id": "chat-8b"}, {"id": "Qwen3-Embedding-0.6B"}]},
                    )
                else:
                    self._send(404, {})

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.embedded.append(list(body["input"]))
                data = [
                    {"index": i, "embedding": vector(t)}
                    for i, t in enumerate(body["input"])
                ]
                self._send(200, {"data": data, "model": body["model"]})

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._httpd.server_port}/v1"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.mark.parametrize(
    ("keys", "offered"), [(ENTER, "Qwen3-Embedding-0.6B"), (DOWN + ENTER, None)]
)
def test_setup_offers_the_embedding_model_a_server_lists_and_probes_it(
    monkeypatch: pytest.MonkeyPatch, keys: str, offered: str | None
) -> None:
    server = _Server()
    try:
        # A real transport: the folder's guard refuses a listing without one, and this server is ours.
        models = asyncio.run(
            discovery.list_models(
                "vllm", server.url, transport=httpx.AsyncHTTPTransport()
            )
        )
        assert discovery.embedding_models(models) == ["Qwen3-Embedding-0.6B"]
        with typed(monkeypatch, keys) as drawn:
            chosen = onboard._offer_embedding(
                server.url, "chat-8b", models, ollama=False
            )
        assert chosen == (offered, False)
        text = seen(drawn)
        assert "Remember across chats with Qwen3-Embedding-0.6B?" in text
        assert "❯ 1. Yes, with it" in text and "2. No, by words only" in text
        # Yes sends one real embeddings request to the server that lists the model.
        assert bool(server.embedded) is (offered is not None)
    finally:
        server.close()
