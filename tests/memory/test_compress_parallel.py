"""Memory ingest asks the encoder for several windows at once (developer setting, #428).

``[memory] compress_parallel`` (1-8, default 1) lets ``melo memory ingest`` send the
windows of all the chats of a run to the chat model together, at most that many
requests in flight, where it sent them one by one. Only the time changes: the
encoding of each chat, the routed items, the reports and the stats totals are the
ones of the serial run. Synthetic chats and a stub chat model that sleeps a little:
no model server.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

import localmelo.__main__ as cli
from localmelo.melo.schema import Message, ToolDef
from localmelo.support import config
from localmelo.support.config import Config, MemoryConfig, save
from localmelo.support.memory import hippo_encode as hippo
from localmelo.support.memory.hippo_encode import Cache, Stats, Turn
from localmelo.tests.memory.hippo_helpers import (
    StubEncoder,
    item,
    numbers_in,
    says,
    turn,
    window_sizes,
)
from localmelo.tests.memory.test_ingest_compress import close, compress, settings

DELAY_S = 0.02
# Narrow enough that each exchange of ``chat()`` is a window of its own.
NARROW = 70


class Slow(StubEncoder):
    """A stub chat model that takes a moment to answer and records when each request
    started and ended and how many were in flight at once."""

    def __init__(self, reply: Any, *, usage: tuple[int, int] | None = None) -> None:
        super().__init__(reply, usage=usage)
        # Shared with the copy the encoder makes of the provider (a dict, not ints).
        self.spans: list[tuple[float, float]] = []
        self.count = {"running": 0, "most": 0}

    @property
    def running(self) -> int:
        return self.count["running"]

    @property
    def most(self) -> int:
        return self.count["most"]

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.count["running"] += 1
        self.count["most"] = max(self.count["most"], self.count["running"])
        began = time.monotonic()
        try:
            await asyncio.sleep(DELAY_S)
            return await super().chat(messages, tools)
        finally:
            self.count["running"] -= 1
            self.spans.append((began, time.monotonic()))

    def overlap(self) -> int:
        """The most requests that were running at one moment, from the recorded times."""
        edges = sorted(
            [(a, 1) for a, _ in self.spans] + [(b, -1) for _, b in self.spans],
            key=lambda edge: (edge[0], edge[1]),
        )
        now = best = 0
        for _, step in edges:
            now += step
            best = max(best, now)
        return best


def chat(count: int, label: str = "") -> list[Turn]:
    return [turn(n, f"{label}Question {n}?", f"Answer {n}.") for n in range(count)]


def by_number(prompt: str) -> str:
    """One item for each exchange of the window, named by the chat and the number."""
    head = prompt.split("\n", 1)[0]
    return says(
        [
            item(
                f"{head} #{n}", f"The user asked {head} question {n}.", "episodic", [n]
            )
            for n in numbers_in(prompt)
        ]
    )


def shape(encoding: hippo.Encoding) -> dict[str, Any]:
    return {
        "pieces": encoding.pieces,
        "windows": [w.to_json() for w in encoding.windows],
        "kept": encoding.kept,
        "ignored": encoding.ignored,
        "unmentioned": encoding.unmentioned,
        "forced": encoding.forced,
    }


# ── encode ──


async def test_parallel_windows_overlap_within_the_limit_and_give_the_serial_encoding() -> (
    None
):
    turns = chat(12)
    assert window_sizes(turns, NARROW) == [1] * 12
    serial_model, parallel_model = Slow(by_number), Slow(by_number)
    serial = await hippo.encode(serial_model, "T", turns, window_chars=NARROW)
    parallel = await hippo.encode(
        parallel_model,
        "T",
        turns,
        window_chars=NARROW,
        limit=asyncio.Semaphore(4),
    )
    assert shape(parallel) == shape(serial)
    assert [p.sources for p in parallel.pieces] == [(n,) for n in range(12)]
    assert serial_model.most == 1 and serial_model.overlap() == 1
    assert 2 <= parallel_model.most <= 4
    assert 2 <= parallel_model.overlap() <= 4
    assert sorted(parallel_model.prompts) == sorted(serial_model.prompts)


async def test_a_limit_of_one_asks_in_the_serial_order() -> None:
    turns = chat(6)
    serial_model, one_model = Slow(by_number), Slow(by_number)
    serial = await hippo.encode(serial_model, "T", turns, window_chars=NARROW)
    one = await hippo.encode(
        one_model, "T", turns, window_chars=NARROW, limit=asyncio.Semaphore(1)
    )
    assert shape(one) == shape(serial)
    assert one_model.prompts == serial_model.prompts
    assert one_model.most == 1


async def test_a_reply_that_comes_late_does_not_change_the_order_of_the_result() -> (
    None
):
    turns = chat(5)

    class Reversed(Slow):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            number = numbers_in(str(messages[-1].content))[0]
            await asyncio.sleep(0.01 * (5 - number))  # the first window answers last
            return await super().chat(messages, tools)

    serial = await hippo.encode(Slow(by_number), "T", turns, window_chars=NARROW)
    parallel = await hippo.encode(
        Reversed(by_number),
        "T",
        turns,
        window_chars=NARROW,
        limit=asyncio.Semaphore(8),
    )
    assert shape(parallel) == shape(serial)


async def test_stats_totals_are_equal_between_modes(tmp_path: Path) -> None:
    turns = chat(7)
    totals = []
    for limit in (None, asyncio.Semaphore(3)):
        stats = Stats()
        await hippo.encode(
            Slow(by_number, usage=(11, 5)),
            "T",
            turns,
            window_chars=NARROW,
            stats=stats,
            limit=limit,
            cache=Cache(str(tmp_path / f"cache-{limit is None}.jsonl")),
        )
        totals.append(stats)
    assert totals[0] == totals[1]
    assert totals[0].calls == 7 and totals[0].cache_hits == 0
    assert (totals[0].prompt_tokens, totals[0].completion_tokens) == (77, 35)
    assert totals[0].reported


async def test_a_cache_hit_makes_no_request_in_either_mode(tmp_path: Path) -> None:
    turns = chat(6)
    cache = Cache(str(tmp_path / "cache.jsonl"))
    for limit in (None, asyncio.Semaphore(4)):
        model = Slow(by_number)
        stats = Stats()
        encoding = await hippo.encode(
            model,
            "T",
            turns,
            window_chars=NARROW,
            cache=cache,
            stats=stats,
            limit=limit,
        )
        if limit is None:  # the first run asks and fills the cache
            assert (stats.calls, stats.cache_hits) == (6, 0)
            first = encoding
            continue
        assert model.prompts == [] and (stats.calls, stats.cache_hits) == (0, 6)
        assert all(w.cached for w in encoding.windows)
        assert [p for p in encoding.pieces] == [p for p in first.pieces]


async def test_a_cache_filled_by_a_parallel_run_serves_a_serial_run(
    tmp_path: Path,
) -> None:
    turns = chat(8)
    path = str(tmp_path / "cache.jsonl")
    filled = await hippo.encode(
        Slow(by_number),
        "T",
        turns,
        window_chars=NARROW,
        cache=Cache(path),
        limit=asyncio.Semaphore(4),
    )
    assert len(Path(path).read_text().splitlines()) == 8
    model = Slow(by_number)
    stats = Stats()
    again = await hippo.encode(
        model, "T", turns, window_chars=NARROW, cache=Cache(path), stats=stats
    )
    assert model.prompts == [] and (stats.calls, stats.cache_hits) == (0, 8)
    assert again.pieces == filled.pieces


async def test_a_failing_window_fails_alone_and_the_others_are_kept(
    tmp_path: Path,
) -> None:
    turns = chat(6)

    def reply(prompt: str) -> str:
        return "RAISE" if 2 in numbers_in(prompt) else by_number(prompt)

    results = []
    for limit in (None, asyncio.Semaphore(4)):
        stats = Stats()
        cache = Cache(str(tmp_path / f"cache-{limit is None}.jsonl"))
        encoding = await hippo.encode(
            Slow(reply),
            "T",
            turns,
            window_chars=NARROW,
            cache=cache,
            stats=stats,
            limit=limit,
        )
        results.append((encoding, stats, cache))
    (serial, serial_stats, _), (parallel, parallel_stats, cache) = results
    assert [w.failed for w in parallel.windows] == [False, False, True] + [False] * 3
    assert [p.sources for p in parallel.pieces if not p.forced] == [
        (n,) for n in (0, 1, 3, 4, 5)
    ]
    assert shape(parallel) == shape(serial)
    assert parallel_stats == serial_stats and parallel_stats.calls == 7  # 2 tries
    assert len(Path(cache.path).read_text().splitlines()) == 5  # not the failed one


async def test_the_same_window_in_two_chats_is_asked_once_as_a_serial_run_would(
    tmp_path: Path,
) -> None:
    turns = chat(3)
    serial_stats, parallel_stats = Stats(), Stats()
    serial_model, parallel_model = Slow(by_number), Slow(by_number)
    cache = Cache(str(tmp_path / "serial.jsonl"))
    for _ in range(2):
        await hippo.encode(
            serial_model,
            "T",
            turns,
            window_chars=NARROW,
            cache=cache,
            stats=serial_stats,
        )
    limit = asyncio.Semaphore(4)
    cache = Cache(str(tmp_path / "parallel.jsonl"))
    both = await hippo.gather(
        [
            hippo.encode(
                parallel_model,
                "T",
                turns,
                window_chars=NARROW,
                cache=cache,
                stats=parallel_stats,
                limit=limit,
            )
            for _ in range(2)
        ]
    )
    assert parallel_stats == serial_stats
    assert (parallel_stats.calls, parallel_stats.cache_hits) == (3, 3)
    assert len(parallel_model.prompts) == 3
    assert shape(both[0]) == shape(both[1]) | {"windows": shape(both[0])["windows"]}
    assert [w.cached for w in both[0].windows] == [False] * 3
    assert [w.cached for w in both[1].windows] == [True] * 3
    assert cache.inflight == {}


async def test_stopping_the_run_cancels_the_pending_asks_and_keeps_the_answered(
    tmp_path: Path,
) -> None:
    turns = chat(6)

    class Stuck(Slow):
        async def chat(
            self, messages: list[Message], tools: list[ToolDef] | None = None
        ) -> Message:
            if numbers_in(str(messages[-1].content))[0] >= 3:
                self.count["running"] += 1
                try:
                    await asyncio.Event().wait()  # never answers
                finally:
                    self.count["running"] -= 1
            return await super().chat(messages, tools)

    model = Stuck(by_number)
    cache = Cache(str(tmp_path / "cache.jsonl"))
    task = asyncio.ensure_future(
        hippo.encode(
            model,
            "T",
            turns,
            window_chars=NARROW,
            cache=cache,
            limit=asyncio.Semaphore(4),
        )
    )
    for _ in range(100):
        await asyncio.sleep(0.01)
        if (
            len(Path(cache.path).read_text().splitlines()) >= 3
            if Path(cache.path).exists()
            else False
        ):
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert model.running == 0
    assert cache.inflight == {}
    kept = [json.loads(line) for line in Path(cache.path).read_text().splitlines()]
    assert len(kept) == 3  # the windows answered before the stop stay answered
    assert len(model.prompts) == 3  # nothing was asked after it


# ── ingest ──

LONG = "x" * 2800


def data() -> dict[str, Any]:
    """Three chats of three exchanges, each too long to share a window of 4000."""
    sessions = []
    for day, name in enumerate(("alpha", "beta", "gamma"), start=1):
        turns: list[dict[str, str]] = []
        for n in range(3):
            turns.append({"role": "user", "text": f"{name} {n} {LONG}"})
            turns.append({"role": "assistant", "text": f"Answer {n}."})
        sessions.append(
            {
                "id": name,
                "title": name.title(),
                "date": f"2023-05-{10 + day}T09:00:00",
                "turns": turns,
            }
        )
    return {"sessions": sessions}


def facts(prompt: str) -> str:
    """Items that replace one another across chats, so routing order matters."""
    title = re.search(r"Chat title: (\S+)", prompt)
    name = title.group(1) if title else "?"
    return says(
        [
            item("Home", f"The user said {name} lives at number {n}.", "profile", [n])
            for n in numbers_in(prompt)
        ]
    )


def report_shape(report: Any) -> dict[str, Any]:
    out: dict[str, Any] = report.to_json()
    for volatile in ("seconds", "ids"):
        out.pop(volatile)
    return out


async def test_ingest_in_parallel_gives_the_serial_reports_items_and_routes(
    tmp_path: Path,
) -> None:
    made: dict[str, dict[str, Any]] = {}
    for mode, parallel in (("serial", 1), ("parallel", 4)):
        model = Slow(facts)
        cfg = settings(tmp_path, compress_at_chars=4000, compress_parallel=parallel)
        cfg.memory.compress_cache = str(tmp_path / f"cache-{mode}.jsonl")
        report, chats, memory = await compress(
            tmp_path, data(), model, sub=mode, cfg=cfg
        )
        try:
            _, index = memory.stores()
            items = await index.fetch_items(
                list(range(1, await index.count_items() + 1))
            )
            made[mode] = {
                "report": report_shape(report),
                "stats": (report.stats.calls, report.stats.cache_hits),
                "items": [
                    (x.title, x.text, x.kind, x.status, x.replaced_by, x.encoder)
                    for x in items
                ],
                "model": model,
            }
        finally:
            await close(chats, memory)
    serial, many = made["serial"], made["parallel"]
    assert many["report"] == serial["report"]
    assert many["stats"] == serial["stats"] == (9, 0)
    assert many["items"] == serial["items"]
    assert len(serial["items"]) == 9
    assert serial["model"].most == 1
    assert 2 <= many["model"].overlap() <= 4
    routes = [r["action"] for r in many["report"]["compress"]["beta"]["routes"]]
    assert routes == [
        r["action"] for r in serial["report"]["compress"]["beta"]["routes"]
    ]


async def test_ingest_cache_hits_make_no_request_and_a_parallel_cache_serves_a_serial_run(
    tmp_path: Path,
) -> None:
    filler = Slow(facts)
    report, chats, memory = await compress(
        tmp_path,
        data(),
        filler,
        sub="fill",
        cfg=settings(tmp_path, compress_at_chars=4000, compress_parallel=4),
    )
    await close(chats, memory)
    assert (report.stats.calls, report.stats.cache_hits) == (9, 0)
    for sub, parallel in (("serial", 1), ("parallel", 4)):
        model = Slow(facts)
        again, chats, memory = await compress(
            tmp_path,
            data(),
            model,
            sub=sub,
            cfg=settings(tmp_path, compress_at_chars=4000, compress_parallel=parallel),
        )
        await close(chats, memory)
        assert model.prompts == []
        assert (again.stats.calls, again.stats.cache_hits) == (0, 9)


async def test_compress_chat_with_an_encoding_does_not_ask_again(
    tmp_path: Path,
) -> None:
    from localmelo.support.memory import ingest

    model = Slow(facts)
    cfg = settings(tmp_path)
    _, chats, memory = await compress(tmp_path, data(), model, sub="x", cfg=cfg)
    try:
        asked = len(model.prompts)
        turns = chat(2)
        encoding = await memory.encode_chat(
            "T", turns, cache=Cache(str(tmp_path / "c"))
        )
        assert len(model.prompts) == asked + 1  # one window of two short exchanges
        before = len(model.prompts)
        done = await memory.compress_chat("s", "T", turns, encoding=encoding)
        assert len(model.prompts) == before
        assert done.encoding is encoding
        assert ingest.MODES  # the module the ingest drives is the same one
    finally:
        await close(chats, memory)


async def test_encode_chat_without_a_chat_model_says_so(tmp_path: Path) -> None:
    _, chats, memory = await compress(tmp_path, data(), Slow(facts), sub="n")
    try:
        memory.chat_provider = lambda: None  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            await memory.encode_chat("T", chat(1))
    finally:
        await close(chats, memory)


# ── The setting ──


def errors(**memory: object) -> list[str]:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.memory = MemoryConfig(**memory)  # type: ignore[arg-type]
    return [error for error in cfg.validate() if "[memory]" in error]


def test_the_default_is_one_and_the_bounds_are_one_and_eight() -> None:
    assert MemoryConfig().compress_parallel == 1
    assert config.MEMORY_COMPRESS_PARALLEL == (1, 8)
    assert errors(compress_parallel=1) == [] and errors(compress_parallel=8) == []
    assert errors(compress_parallel=0) != [] and errors(compress_parallel=9) != []
    assert errors(compress_parallel=True) != [] and errors(compress_parallel=2.0) != []


def test_it_is_saved_only_when_not_one_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    with (
        mock.patch.object(config, "CONFIG_PATH", str(path)),
        mock.patch.object(config, "CONFIG_DIR", str(tmp_path)),
    ):
        save(cfg)
        assert "compress_parallel" not in path.read_text()
        cfg.memory = MemoryConfig(compress_parallel=1)
        save(cfg)
        assert "compress_parallel" not in path.read_text()
        cfg.memory = MemoryConfig(compress_parallel=4)
        save(cfg)
        assert "compress_parallel = 4\n" in path.read_text()
        assert config.load().memory.compress_parallel == 4


def test_x_reads_it_for_one_run_and_refuses_a_bad_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    try:
        cli._parse(
            [
                "-X",
                "memory.compress_parallel=4",
                "--base-url",
                "http://127.0.0.1:1/v1",
                "q",
            ]
        )
        assert config.overrides() == {"memory.compress_parallel": 4}
        assert config.load().memory.compress_parallel == 4
        for bad in ("0", "9", "many"):
            with pytest.raises(SystemExit) as exit_:
                cli._parse(
                    [
                        "-X",
                        f"memory.compress_parallel={bad}",
                        "--base-url",
                        "http://127.0.0.1:1/v1",
                        "q",
                    ]
                )
            assert exit_.value.code == 2
            assert "memory.compress_parallel" in capsys.readouterr().err
    finally:
        config.set_overrides({})


def test_the_service_reads_it_and_falls_back_to_one() -> None:
    from localmelo.support.memory import ChatMemory

    def read(value: object) -> int:
        cfg = Config(chat_backend="vllm", embedding_backend="none")
        cfg.memory = MemoryConfig(compress_parallel=value)  # type: ignore[arg-type]
        return ChatMemory(lambda: cfg).compress_parallel()

    for value, wanted in ((4, 4), (1, 1), (8, 8), (0, 1), (9, 1), ("4", 1)):
        assert read(value) == wanted
