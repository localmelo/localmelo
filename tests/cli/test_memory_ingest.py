"""``melo memory ingest FILE``: past chats from a file, a developer verb.

Only in developer mode, like ``sleep``. It prints one line, or JSON with ``-j``;
``-X notes=none`` keeps no notes; a file over the size bound, a broken file and a
console or app that owns the chats are refused with one line and no write.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from localmelo.support import config, memory_cli
from localmelo.support.memory import ChatMemory, ingest
from localmelo.support.sessions.store import SessionStore, default_path
from localmelo.tests.memory.chat_memory_helpers import FakeEmbedding
from localmelo.tests.memory.hippo_helpers import StubEncoder, item, says

FILE = {
    "sessions": [
        {
            "id": "s1",
            "date": "2023-05-20T13:40:00",
            "turns": [
                {"role": "user", "text": "I live in Pittsburgh."},
                {"role": "assistant", "text": "Pittsburgh is a lovely city."},
            ],
        },
        {
            "id": "s2",
            "title": "猫",
            "date": "2023-06-01T09:00:00",
            "turns": [
                {"role": "user", "text": "我的猫是一只橘猫。"},
                {"role": "assistant", "text": "橘猫很可爱。"},
                {"role": "user", "text": "Do you know a good vet?"},
            ],
        },
    ]
}


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "cfg" / "config.toml"))
    yield tmp_path


@pytest.fixture()
def developer(home: Path) -> None:
    (home / "cfg").mkdir(parents=True, exist_ok=True)
    (home / "cfg" / "config.toml").write_text('[ui]\nmode = "developer"\n')


@pytest.fixture()
def path(home: Path) -> str:
    target = home / "chats.json"
    target.write_text(json.dumps(FILE))
    return str(target)


def _held() -> tuple[int, int]:
    async def held() -> tuple[int, int]:
        memory = ChatMemory(config.load)
        try:
            return len(await memory.records()), (await memory.status())["documents"]
        finally:
            await memory.aclose()

    return asyncio.run(held())


def test_the_verb_is_for_developer_mode_and_writes_nothing_without_it(
    path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert memory_cli.main(["ingest", path]) == 2
    assert 'needs [ui] mode = "developer"' in capsys.readouterr().err
    assert _held() == (0, 0)


def test_it_prints_a_summary_line(
    developer: None, path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert memory_cli.main(["ingest", path]) == 0
    line = capsys.readouterr().out
    assert line.startswith(
        "Ingested 2 sessions, 3 turns, 3 documents (by words only) and 2 notes in "
    )
    assert _held() == (2, 3)


def test_j_prints_json_with_the_store_ids_of_the_labels(
    developer: None, path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert memory_cli.main(["ingest", path, "-j"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {
        "sessions",
        "turns",
        "docs",
        "notes",
        "embedded_chars",
        "cut",
        "seconds",
        "ids",
    } <= set(out)
    assert (out["sessions"], out["turns"], out["docs"], out["notes"]) == (2, 3, 3, 2)
    assert set(out["ids"]) == {"s1", "s2"}

    async def titles() -> dict[str, str]:
        chats = SessionStore(writer="gateway")
        await chats.open()
        try:
            return {
                s.id: s.title for s in await chats.list_sessions(limit=9, kinds=None)
            }
        finally:
            await chats.aclose()

    assert asyncio.run(titles())[out["ids"]["s2"]] == "猫"


def test_x_notes_none_keeps_no_notes_and_other_x_values_are_refused(
    developer: None, path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert memory_cli.main(["ingest", path, "-X", "notes=sometimes"]) == 2
    assert "-X notes takes rules or none" in capsys.readouterr().err
    assert memory_cli.main(["ingest", path, "-X", "speed=fast"]) == 2
    assert "-X takes notes=rules or notes=none" in capsys.readouterr().err
    assert _held() == (0, 0)
    assert memory_cli.main(["ingest", path, "-j", "-X", "notes=none"]) == 0
    assert json.loads(capsys.readouterr().out)["notes"] == 0
    assert _held() == (0, 3)


def test_a_file_over_the_size_bound_is_refused_before_it_is_read(
    developer: None, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    big = home / "big.json"
    with big.open("wb") as handle:  # a sparse file: no disk is used
        handle.truncate(ingest.MAX_FILE_BYTES + 1)
    assert memory_cli.main(["ingest", str(big)]) == 2
    err = capsys.readouterr().err
    assert "big.json is 500 MB; the limit is 500 MB" in err and "Split it" in err
    assert _held() == (0, 0)


def test_a_missing_or_broken_file_is_one_line_and_exit_2(
    developer: None, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert memory_cli.main(["ingest", str(home / "nope.json")]) == 2
    assert "melo: cannot read " in capsys.readouterr().err
    broken = home / "broken.json"
    broken.write_text('{"sessions": [{"id": "a", "turns": []}]}')
    assert memory_cli.main(["ingest", str(broken)]) == 2
    assert "melo: session a: has no date" in capsys.readouterr().err
    assert _held() == (0, 0)


def test_it_refuses_while_the_console_or_the_app_owns_the_chats(
    developer: None, path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    async def own_then_ingest() -> int:
        owner = SessionStore(default_path("gateway"), writer="gateway")
        await owner.open()
        try:
            return await memory_cli._ingest(path, [], False)
        finally:
            await owner.aclose()

    assert asyncio.run(own_then_ingest()) == 1
    assert "the console or the app is running and owns the chats" in (
        capsys.readouterr().err
    )
    assert _held() == (0, 0)


def test_with_an_embedding_model_it_says_so_and_embeds_the_turns(
    developer: None,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    embedding = FakeEmbedding(model="fake-embed")
    monkeypatch.setattr(ChatMemory, "embedding", lambda self: (embedding, "fake-embed"))
    assert memory_cli.main(["ingest", path, "-j"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"] == "fake-embed" and out["embedded_chars"] > 0
    assert out["error"] == ""


def test_x_ingest_takes_turns_or_compress_and_refuses_any_other_value(
    developer: None, path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert memory_cli.main(["ingest", path, "-X", "ingest=zip"]) == 2
    err = capsys.readouterr().err
    assert "melo: -X ingest takes turns or compress, got 'zip'" in err
    assert memory_cli.main(["ingest", path, "-X", "ingest"]) == 2
    assert "ingest=turns or ingest=compress" in capsys.readouterr().err
    assert _held() == (0, 0)
    assert memory_cli.main(["ingest", path, "-X", "ingest=turns"]) == 0
    assert capsys.readouterr().out.startswith("Ingested 2 sessions, 3 turns")
    assert _held() == (2, 3)


def test_x_ingest_compress_without_a_chat_model_is_one_line_exit_2_and_no_write(
    developer: None,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(ChatMemory, "chat_provider", lambda self: None)
    assert memory_cli.main(["ingest", path, "-X", "ingest=compress"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("melo: ingest=compress needs a chat model: ")
    assert err.count("\n") == 1
    assert _held() == (0, 0)


def test_x_ingest_compress_encodes_each_chat_and_j_prints_the_report(
    developer: None,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub = StubEncoder(
        lambda prompt: (
            says(
                [item("Home city", "The user said they live in Pittsburgh.", "profile")]
            )
            if "Pittsburgh" in prompt
            else says()
        )
    )
    monkeypatch.setattr(ChatMemory, "chat_provider", lambda self: stub)
    assert memory_cli.main(["ingest", path, "-j", "-X", "ingest=compress"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ingest"] == "compress" and out["model_calls"] == 2
    assert (out["sessions"], out["turns"], out["docs"]) == (2, 3, 3)
    assert set(out["compress"]) == {"s1", "s2"}
    (stored,) = out["compress"]["s1"]["items"]
    assert (stored["title"], stored["kind"]) == ("Home city", "profile")
    assert stored["sources"] == [["s1", 0]]
    # The model kept nothing of the second chat; its one lasting statement is made
    # into an item by the rules (forced), and the policy ignores it as too short.
    second = out["compress"]["s2"]
    assert second["gate"]["forced"] == [0] and second["items"] == []
    assert [(r["action"], r["reason"]) for r in second["routes"]] == [
        ("IGNORE", "short fact")
    ]
    assert len(stub.prompts) == 2
    assert _held()[1] == 3

    async def items() -> int:
        memory = ChatMemory(config.load)
        try:
            return await memory.stores()[1].count_items()
        finally:
            await memory.aclose()

    assert asyncio.run(items()) == 1


def test_the_summary_line_of_a_compress_run_says_how_many_items_and_calls(
    developer: None,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub = StubEncoder(says())
    monkeypatch.setattr(ChatMemory, "chat_provider", lambda self: stub)
    assert memory_cli.main(["ingest", path, "-X", "ingest=compress"]) == 0
    line = capsys.readouterr().out
    assert line.startswith("Ingested 2 sessions, 3 turns, 3 documents")
    assert " Encoded " in line and "(2 model calls, 0 from the cache)." in line
