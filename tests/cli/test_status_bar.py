"""The status bar's usage and work zones (#282): figures by backend kind, limits,
prices, narrow terminals, and the tasks beside the chat with their actions."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from localmelo.melo.schema import Message
from localmelo.support import config, task_cli
from localmelo.support.cli.app import StatusBar, status_fragments
from localmelo.support.cli.text import cells
from localmelo.support.cli.work import (
    Action,
    GatewayTasks,
    WorkBeside,
    WorkSummary,
    WorkTask,
    actions,
    summarize,
    task_rows,
    work_parts,
)
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM, rate_limits
from localmelo.support.usage_snapshot import (
    BarPart,
    UsageSnapshot,
    bar_usage,
    meter,
    session_cost,
)
from localmelo.tests.cli.ui_helpers import ENTER, FakeController, running_app


def _snap(*requests: tuple[int, int | None, int], model: str = "m") -> UsageSnapshot:
    snap = UsageSnapshot(model=model)
    for prompt, cached, completion in requests:
        usage: dict[str, int] = {"prompt_tokens": prompt}
        usage["completion_tokens"] = completion
        if cached is not None:
            usage["cached_tokens"] = cached
        snap.observe(
            {"type": "assistant.message", "ts": 100.0, "data": {"usage": usage}}
        )
    return snap


def _text(parts: list[BarPart]) -> str:
    return "".join(
        (" " if p.glued else " · ") * bool(n) + p.text for n, p in enumerate(parts)
    )


# ── The usage zone ──


def test_a_meter_shows_only_a_real_share_and_any_use() -> None:
    assert [meter(p) for p in (0, 1, 42, 100)] == ["▯▯▯▯▯", "▮▯▯▯▯", "▮▮▯▯▯", "▮▮▮▮▮"]
    assert meter(42, ascii=True) == "##---"


def test_a_filling_context_turns_warn_then_error() -> None:
    snap = _snap((1000, None, 5))
    for measured, role in ((60, ""), (75, "warn"), (95, "error")):
        snap.observe(
            {
                "type": "context_budget",
                "data": {"measured_prompt_tokens": measured, "input_limit": 100},
            }
        )
        ctx = [p for p in bar_usage(snap, local=True, now=100.0) if "ctx" in p.text]
        assert ctx[0].role == role, measured


def test_a_poor_cache_warns_only_after_the_first_request() -> None:
    first = bar_usage(_snap((1000, 0, 5)), local=True, now=100.0)
    assert [p for p in first if p.text == "cache 0%"][0].role == ""  # cold start
    later = bar_usage(_snap((1000, 0, 5), (1000, 100, 5)), local=True, now=100.0)
    assert [p.role for p in later if p.text.startswith("cache")] == ["warn"]
    good = bar_usage(_snap((1000, 900, 5)), local=True, now=100.0)
    assert [p.role for p in good if p.text.startswith("cache")] == ["ok"]


def test_a_server_that_reports_no_cache_shows_no_cache_figure() -> None:
    snap = _snap((1000, None, 5))
    assert _text(bar_usage(snap, local=True, now=100.0, last_prompt=1000)) == (
        "↑ 1.0k ↓ 5 · ctx 1.0k"
    )


def test_ascii_says_in_and_out_and_reset() -> None:
    snap = _snap((1000, None, 5))
    assert _text(bar_usage(snap, local=True, now=100.0, ascii=True)).startswith(
        "in 1.0k out 5"
    )
    cloud = bar_usage(snap, local=False, now=100.0, limits=(42, 160.0), ascii=True)
    assert _text(cloud).endswith("limit 42% ##--- reset 1m")


def test_the_cost_sums_each_model_at_its_own_price() -> None:
    snap = _snap((1_000_000, 0, 0), model="a")
    snap.model = "b"
    snap.observe(
        {
            "type": "assistant.message",
            "data": {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}},
        }
    )
    prices = {"a": {"input": 1.0, "output": 2.0}, "b": {"input": 3.0, "output": 4.0}}
    assert session_cost(snap, prices) == pytest.approx(4.0)
    # A model used without a price makes the cost unknown: tokens instead.
    assert session_cost(snap, {"a": prices["a"]}) is None
    assert _text(bar_usage(snap, local=False, now=0.0, prices=prices)) == "$4.00"


def test_a_tiny_cost_is_not_drawn_as_zero_and_local_is_never_priced() -> None:
    snap = _snap((100, 0, 5))
    prices = {"m": {"input": 1.0, "output": 1.0}}
    assert bar_usage(snap, local=False, now=0.0, prices=prices)[0].text == "<$0.01"
    assert bar_usage(snap, local=True, now=0.0, prices=prices)[0].text == "↑ 100 ↓ 5"


def test_a_limit_without_a_reset_has_no_clock() -> None:
    parts = bar_usage(_snap((10, None, 1)), local=False, now=0.0, limits=(95, None))
    assert _text(parts).endswith("limit 95% ▮▮▮▮▮") and parts[-2].role == "error"


# ── Rate limits from response headers ──


def test_rate_limits_take_the_tightest_window_and_its_reset() -> None:
    headers = {
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-remaining-requests": "59",
        "x-ratelimit-reset-requests": "1s",
        "x-ratelimit-limit-tokens": "1000",
        "X-RateLimit-Remaining-Tokens": "580",
        "x-ratelimit-reset-tokens": "6m0s",
    }
    assert rate_limits(headers, now=1000.0) == (42, 1360.0)
    anthropic = {
        "anthropic-ratelimit-input-tokens-limit": "100",
        "anthropic-ratelimit-input-tokens-remaining": "10",
        "anthropic-ratelimit-input-tokens-reset": "1970-01-01T00:10:00Z",
    }
    assert rate_limits(anthropic, now=0.0) == (90, 600.0)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"content-type": "application/json"},
        {"x-ratelimit-limit-requests": "0", "x-ratelimit-remaining-requests": "0"},
        {"x-ratelimit-limit-requests": "a", "x-ratelimit-remaining-requests": "1"},
        {"x-ratelimit-limit-requests": "60"},
    ],
)
def test_rate_limits_need_a_limit_and_what_remains(headers: dict[str, str]) -> None:
    assert rate_limits(headers) is None


def test_an_unreadable_reset_keeps_the_share() -> None:
    headers = {
        "x-ratelimit-limit-tokens": "10",
        "x-ratelimit-remaining-tokens": "5",
        "x-ratelimit-reset-tokens": "soon",
    }
    assert rate_limits(headers) == (50, None)


async def test_the_provider_keeps_the_limits_of_its_last_response() -> None:
    body = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    limited = {"x-ratelimit-limit-requests": "4", "x-ratelimit-remaining-requests": "1"}
    replies = [httpx.Response(200, json=body, headers=limited)]
    replies.append(httpx.Response(200, json=body))
    llm = OpenAICompatLLM(
        "http://127.0.0.1:9/v1",
        "m",
        transport=httpx.MockTransport(lambda request: replies.pop(0)),
    )
    try:
        await llm.chat([Message("user", "hi")])
        assert llm.last_limits == (75, None)
        await llm.chat([Message("user", "hi")])
        assert llm.last_limits is None  # this response reported none
    finally:
        await llm.close()


# ── [pricing] ──


def test_pricing_is_configuration_kept_on_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = Path(config.config_path())
    path.parent.mkdir(parents=True)
    path.write_text(
        'chat_backend = "openai"\n[openai]\nchat_model = "gpt-x"\n'
        'api_key_env = "K"\n[pricing."gpt-x"]\ninput = 2.5\noutput = 10\n'
        "cache_read = 1.25\n",
        encoding="utf-8",
    )
    cfg = config.load()
    assert cfg.pricing == {"gpt-x": {"input": 2.5, "output": 10, "cache_read": 1.25}}
    assert cfg.pricing_errors() == [] and config.Config().pricing == {}
    config.save(cfg)
    assert config.load().pricing == cfg.pricing


@pytest.mark.parametrize(
    ("table", "error"),
    [
        ({"m": {"input": 1}}, "needs input and output"),
        ({"m": {"input": -1, "output": 1}}, "input must be 0 or more"),
        ({"m": {"input": True, "output": 1}}, "input must be a number"),
        ({"m": {"input": 1, "output": 1, "cache_write": 1}}, "unknown key"),
        ([1], "must be a table"),
    ],
)
def test_a_bad_price_is_reported(table: Any, error: str) -> None:
    cfg = config.Config()
    cfg.pricing = table
    assert error in " ".join(cfg.pricing_errors())


# ── Narrow terminals ──

USAGE = (
    BarPart("↑ 1.2M ↓ 38k"),
    BarPart("ctx 12k/128k"),
    BarPart("▮▯▯▯▯", detail=True, glued=True),
    BarPart("cache 83%", "ok"),
    BarPart("(avg 71%)", detail=True, glued=True),
    BarPart("idle 3m", detail=True),
)
WORK = tuple(work_parts(WorkSummary(running=2, queued=1, top=1)))


def _line(width: int, *, work: bool = True, mode: str = "") -> str:
    bar = StatusBar("Fix the parser", mode, usage=USAGE, work=WORK if work else ())
    line = "".join(part[1] for part in status_fragments(bar, width, "…"))
    assert cells(line) == width, (width, line)
    return line.rstrip()


def test_a_wide_bar_shows_every_zone_and_the_title() -> None:
    assert _line(140) == (
        "  ↑ 1.2M ↓ 38k · ctx 12k/128k ▮▯▯▯▯ · cache 83% (avg 71%) · idle 3m"
        " │ 2 running · 1 queued · P1 · ctrl+t tasks │ Fix the parser"
    )


def test_a_narrow_bar_drops_the_title_then_work_details_then_usage_details() -> None:
    usage = "↑ 1.2M ↓ 38k · ctx 12k/128k ▮▯▯▯▯ · cache 83% (avg 71%) · idle 3m"
    assert _line(115) == f"  {usage} │ 2 running · 1 queued · P1 · ctrl+t tasks"
    # The priority goes before the key that opens the list.
    assert _line(105) == f"  {usage} │ 2 running · 1 queued · ctrl+t tasks"
    assert _line(90) == f"  {usage} │ 2 running · 1 queued"
    assert _line(80) == (
        "  ↑ 1.2M ↓ 38k · ctx 12k/128k ▮▯▯▯▯ · cache 83% (avg 71%)"
        " │ 2 running · 1 queued"
    )
    assert (
        _line(65) == "  ↑ 1.2M ↓ 38k · ctx 12k/128k · cache 83% │ 2 running · 1 queued"
    )
    # Then the figures go right to left: the work's, then the usage's.
    assert _line(60) == "  ↑ 1.2M ↓ 38k · ctx 12k/128k · cache 83% │ 2 running"
    assert _line(50) == "  ↑ 1.2M ↓ 38k · ctx 12k/128k · cache 83%"
    assert _line(30) == "  ↑ 1.2M ↓ 38k · ctx 12k/128k"


@pytest.mark.parametrize("width", range(6, 150, 7))
def test_a_full_bar_never_overflows_and_keeps_the_mode(width: int) -> None:
    for mode in ("", "auto", "⚠ YOLO: no approvals"):
        line = _line(width, mode=mode)
        assert " … " not in line and not line.strip().startswith("…")
        if mode and width > len(mode) + 3:
            assert line.endswith(mode)


def test_the_work_zone_draws_running_and_held_by_kind_and_its_key() -> None:
    parts = work_parts(WorkSummary(running=1, held=1, top=0))
    bar = StatusBar("", "", work=tuple(parts))
    drawn = status_fragments(bar, 80, "…")
    assert ("class:status_live", "1 running") in drawn
    assert ("class:status_warn", "1 held") in drawn
    assert ("class:status_figure", "P0") in drawn
    assert ("class:key", "ctrl+t") in drawn and ("class:faint", " tasks") in drawn


# ── The tasks beside the chat ──

TASKS = [
    WorkTask("t1", "Bench the cache", "running", 2),
    WorkTask("t2", "Write the changelog", "queued", 1),
    WorkTask("t3", "Old idea", "parked", 5),
    WorkTask("t4", "Waits for an answer", "held", 0),
]


def test_tasks_are_counted_by_state_with_the_top_priority_not_paused() -> None:
    assert summarize(TASKS) == WorkSummary(running=1, queued=1, held=1, paused=1, top=0)
    assert summarize([TASKS[2]]) == WorkSummary(paused=1)
    assert work_parts(WorkSummary()) == []


def test_each_state_offers_only_what_it_allows() -> None:
    def labels(task: WorkTask) -> list[str]:
        return [action.label for action in actions(task)]

    assert labels(TASKS[0]) == ["Pause", "Raise to P1", "Lower to P3", "Stop"]
    assert labels(TASKS[2]) == ["Resume", "Stop"]  # paused: P5 until resumed
    assert labels(TASKS[3]) == ["Pause", "Lower to P1", "Stop"]  # P0 is the top
    assert labels(WorkTask("t", "x", "queued", 4)) == ["Pause", "Raise to P3", "Stop"]
    assert labels(WorkTask("t", "x", "done", 2)) == []


def test_the_rows_line_up_priority_state_and_title() -> None:
    assert task_rows(TASKS[:3]) == [
        "P2  running  Bench the cache",
        "P1  queued   Write the changelog",
        "P5  paused   Old idea",
    ]


def _gateway(tasks: list[dict[str, Any]], seen: list[httpx.Request]) -> GatewayTasks:
    def reply(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"tasks": tasks})
        body = json.loads(request.content)
        state = {"pause": "parked", "stop": "cancelled"}.get(
            body.get("state"), "queued"
        )
        return httpx.Response(200, json={"task": {**tasks[0], "state": state}})

    def connect(timeout: float = 10) -> httpx.Client:
        return httpx.Client(
            base_url="http://127.0.0.1:9", transport=httpx.MockTransport(reply)
        )

    return GatewayTasks(connect=connect, token=lambda: Path(__file__))


def test_the_gateway_is_asked_as_melo_task_asks_it() -> None:
    raw: list[dict[str, Any]] = [
        {"task_id": "t1", "title": "Bench", "state": "running", "level": 2},
        {"task_id": "t9", "title": "Done", "state": "done", "level": 2},
        {"bad": True},
    ]
    seen: list[httpx.Request] = []
    gateway = _gateway(raw, seen)
    assert gateway.available()
    [task] = gateway.list()
    assert task == WorkTask("t1", "Bench", "running", 2)
    after = gateway.act(task, Action("Pause", "pause"))
    assert after is not None and after.state == "parked"
    gateway.act(task, Action("Raise to P1", "level", "P1"))
    gateway.act(task, Action("Resume", "resume"))
    # The requests `melo task pause|level|resume` sends, byte for byte.
    expected: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        expected.append(request)
        return httpx.Response(200, json={"task": {}})

    client = httpx.Client(
        base_url="http://127.0.0.1:9", transport=httpx.MockTransport(record)
    )
    for argv in (["pause", "t1"], ["level", "t1", "P1"], ["resume", "t1"]):
        args = task_cli.parser().parse_args([*argv, "--json"])
        task_cli.execute(args, client, out=io.StringIO(), stdin=io.StringIO())
    assert [(r.method, r.url.path, r.content) for r in seen[1:]] == [
        (r.method, r.url.path, r.content) for r in expected
    ]


def test_no_token_means_no_gateway_and_no_request(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    gateway = replace_token(_gateway([], seen), tmp_path / "missing")
    assert not gateway.available()


def replace_token(gateway: GatewayTasks, path: Path) -> GatewayTasks:
    return GatewayTasks(connect=gateway._connect, token=lambda: path)


def test_a_refusal_comes_back_as_its_words() -> None:
    def connect(timeout: float = 10) -> httpx.Client:
        return httpx.Client(
            base_url="http://127.0.0.1:9",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(409, json={"error": {"code": "not_paused"}})
            ),
        )

    gateway = GatewayTasks(connect=connect, token=lambda: Path(__file__))
    with pytest.raises(Exception, match=r"HTTP 409 \(not_paused\)"):
        gateway.act(TASKS[0], Action("Resume", "resume"))


async def test_a_gateway_that_fails_any_way_leaves_no_zone_and_no_error() -> None:
    def connect(timeout: float = 10) -> httpx.Client:
        raise RuntimeError("the config file cannot be read")

    work = WorkBeside(GatewayTasks(connect=connect, token=lambda: Path(__file__)))
    await work.refresh()
    assert work.tasks == [] and work.summary == WorkSummary()


class FakeTasks:
    """A gateway with these tasks that records each action it is asked for."""

    def __init__(self, tasks: list[WorkTask], *, refuse: bool = False) -> None:
        self.tasks, self.refuse = list(tasks), refuse
        self.acted: list[tuple[str, Action]] = []

    def available(self) -> bool:
        return True

    def list(self) -> list[WorkTask]:
        return list(self.tasks)

    def act(self, task: WorkTask, action: Action) -> WorkTask | None:
        from localmelo.support.cli.work import WorkError

        if self.refuse:
            raise WorkError("gateway returned HTTP 409 (invalid_transition)")
        self.acted.append((task.task_id, action))
        state = {"pause": "parked", "stop": "cancelled"}.get(action.command, task.state)
        return replace(task, state=state)


class ScriptedApp:
    """Answers each list in turn and keeps what it was asked and the hints."""

    def __init__(self, answers: list[int | None]) -> None:
        self.answers = answers
        self.asked: list[tuple[str, list[str], int]] = []
        self.hints: list[str] = []

    def hint(self, text: str, seconds: float = 2.0) -> None:
        self.hints.append(text)

    async def choose_from(
        self, title: str, options: Any, *, default: int = 0
    ) -> int | None:
        self.asked.append((title, list(options), default))
        return self.answers.pop(0)


async def test_each_action_goes_to_the_gateway_once_chosen() -> None:
    for index, (label, command, level) in enumerate(
        [
            ("Pause", "pause", None),
            ("Raise to P1", "level", "P1"),
            ("Lower to P3", "level", "P3"),
        ]
    ):
        tasks = FakeTasks(TASKS)
        app = ScriptedApp([0, index])
        await WorkBeside(tasks).open(app)  # type: ignore[arg-type]
        assert app.asked[0][0] == "Tasks beside this chat"
        assert app.asked[1][0] == "Bench the cache · P2 · running"
        assert tasks.acted == [("t1", Action(label, command, level))]
    assert app.hints[-1] == "Bench the cache is now P3"


def test_the_bar_and_the_list_read_levels_as_p_codes() -> None:
    parts = work_parts(WorkSummary(running=2, queued=1, top=1))
    assert [part.text for part in parts] == [
        "2 running",
        "1 queued",
        "P1",
        "ctrl+t tasks",
    ]
    labels = [action.label for action in actions(TASKS[0])]
    assert labels == ["Pause", "Raise to P1", "Lower to P3", "Stop"]
    assert task_rows(TASKS) == [
        "P2  running  Bench the cache",
        "P1  queued   Write the changelog",
        "P5  paused   Old idea",
        "P0  held     Waits for an answer",
    ]
    words = ("Now", "Today", "Normal", "Whenever", "If idle", "Later")
    assert not any(word in " ".join(task_rows(TASKS)) for word in words)


async def test_ctrl_t_reads_levels_as_p_codes_in_every_mode() -> None:
    tasks = FakeTasks(TASKS)
    app = ScriptedApp([0, 1])
    await WorkBeside(tasks).open(app)  # type: ignore[arg-type]
    assert app.asked[0][1][0] == "P2  running  Bench the cache"
    assert app.asked[1][0] == "Bench the cache · P2 · running"
    assert app.asked[1][1] == ["Pause", "Raise to P1", "Lower to P3", "Stop"]
    assert tasks.acted == [("t1", Action("Raise to P1", "level", "P1"))]
    assert app.hints == ["Bench the cache is now P1"]


async def test_resume_goes_to_the_gateway_for_a_paused_task() -> None:
    tasks = FakeTasks(TASKS)
    app = ScriptedApp([2, 0])
    await WorkBeside(tasks).open(app)  # type: ignore[arg-type]
    assert app.asked[1][1] == ["Resume", "Stop"]
    assert tasks.acted == [("t3", Action("Resume", "resume"))]
    assert app.hints == ["Resumed: Old idea"]


async def test_stop_asks_first_and_keep_it_does_nothing() -> None:
    tasks = FakeTasks(TASKS)
    app = ScriptedApp([0, 3, 1])
    await WorkBeside(tasks).open(app)  # type: ignore[arg-type]
    title, options, default = app.asked[2]
    assert title.startswith("Stop Bench the cache?")
    assert options == ["Stop it", "Keep it"] and default == 1  # the safe one
    assert tasks.acted == []
    app = ScriptedApp([0, 3, 0])
    await WorkBeside(tasks).open(app)  # type: ignore[arg-type]
    assert tasks.acted == [("t1", Action("Stop", "stop"))]
    assert app.hints == ["Stopped: Bench the cache"]


async def test_esc_closes_each_list_and_a_refusal_is_a_hint() -> None:
    for answers in ([None], [0, None]):
        tasks = FakeTasks(TASKS)
        await WorkBeside(tasks).open(ScriptedApp(answers))  # type: ignore[arg-type]
        assert tasks.acted == []
    app = ScriptedApp([0, 0])
    await WorkBeside(FakeTasks(TASKS, refuse=True)).open(app)  # type: ignore[arg-type]
    assert app.hints == [
        "Could not pause: gateway returned HTTP 409 (invalid_transition)"
    ]
    app = ScriptedApp([])
    await WorkBeside(FakeTasks([])).open(app)  # type: ignore[arg-type]
    assert app.hints[0].startswith("No tasks beside this chat")


async def test_ctrl_t_opens_the_list_and_enter_acts_on_a_task() -> None:
    tasks = FakeTasks(TASKS)
    work = WorkBeside(tasks)

    class WorkController(FakeController):
        app: Any = None

        async def open_work(self) -> None:
            await work.open(self.app)

    controller = WorkController()
    async with running_app(controller, columns=80) as app:
        controller.app = app.ui
        app.type("\x14")  # ctrl+t
        await app.wait(lambda: "Tasks beside this chat" in app.text())
        assert "P2  running  Bench the cache" in app.text()
        app.type(ENTER)
        await app.wait(lambda: "1. Pause" in app.text())
        assert "Bench the cache · P2 · running" in app.text()
        app.type(ENTER)
        await app.wait(lambda: tasks.acted == [("t1", Action("Pause", "pause"))])
        await app.wait(lambda: "Paused: Bench the cache" in app.text())


def test_an_argv_namespace_is_what_the_terminal_sends() -> None:
    # The terminal builds the parsed form of `melo task <cmd> <id> --json`.
    parsed = task_cli.parser().parse_args(["level", "t1", "P1", "--json"])
    assert vars(parsed) == vars(
        argparse.Namespace(command="level", id="t1", level="P1", json=True)
    )


def test_a_task_title_falls_back_to_its_id_and_a_bad_level_is_none() -> None:
    task = WorkTask.from_json({"task_id": "t", "state": "queued", "level": 9})
    assert task == WorkTask("t", "t", "queued", None)


async def test_an_idle_bar_redraws_once_a_second_only_while_a_clock_shows() -> None:
    class Clocked(FakeController):
        clock = False

        def clock_visible(self) -> bool:
            return self.clock

    controller = Clocked()
    async with running_app(controller) as app:
        await app.settle()
        drawn = [0]
        invalidate = app.ui.app.invalidate

        def counted() -> None:
            drawn[0] += 1
            invalidate()

        app.ui.app.invalidate = counted  # type: ignore[method-assign]
        await asyncio.sleep(0.6)
        assert drawn[0] == 0  # idle, and no clock: nothing redraws
        controller.clock = True
        started = time.monotonic()
        await asyncio.sleep(1.5)
        assert 1 <= drawn[0] <= time.monotonic() - started + 1
