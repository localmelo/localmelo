"""The welcome card's first-run setup (setup_flow.py): the plan, the job and its progress.

The real plan and the real job run over a fake ``uv`` and ``melo-serve``
(tests/fixtures/setup_tools.py) and a fake melo_serve on a Unix socket: nothing
is installed, downloaded or started.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from localmelo.support import config
from localmelo.support.gateway import daemon
from localmelo.support.gateway.console import Console, setup_flow
from localmelo.support.server_setup import run, terminal
from localmelo.tests.fixtures.setup_tools import CHAT, EMBED, OMLX_RUNNING, PLAN, Fakes

from .conftest import PORT

CHAT_ID, EMBED_ID = f"model:{CHAT}", f"model:{EMBED}"


@pytest.fixture(autouse=True)
def _on_a_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_flow.sys, "platform", "darwin")


@pytest.fixture(autouse=True)
def _no_gateway_service(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The gateway's own login item is never registered from inside the gateway."""
    asked: list[str] = []

    def install() -> daemon.Installed:
        asked.append("install")
        raise AssertionError("the gateway's login item was registered")

    monkeypatch.setattr(daemon, "install", install)
    return asked


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=f"http://127.0.0.1:{PORT}"
    ) as client:
        yield client


async def until(check: Callable[[], Awaitable[Any] | Any]) -> Any:
    for _ in range(600):
        found = check()
        found = await found if isinstance(found, Awaitable) else found
        if found:
            return found
        await asyncio.sleep(0.05)
    raise AssertionError("never happened")


async def job(client: httpx.AsyncClient) -> dict[str, Any] | None:
    found: dict[str, Any] | None = (await client.get("/api/setup/detect")).json()["job"]
    return found


async def ended(client: httpx.AsyncClient) -> dict[str, Any]:
    """The job, once it is over."""

    async def over() -> dict[str, Any] | None:
        found = await job(client)
        return found if found and found["state"] != "running" else None

    found: dict[str, Any] = await until(over)
    return found


def states(found: dict[str, Any]) -> dict[str, str]:
    return {row["id"]: row["state"] for row in found["rows"]}


async def start(
    client: httpx.AsyncClient, *skip: str, pick: str = CHAT
) -> httpx.Response:
    return await client.post("/api/setup/start", json={"pick": pick, "skip": skip})


# ── The plan ──


async def test_the_plan_is_the_terminals_list_in_the_words_of_the_request(
    setup_fakes: Fakes, client: httpx.AsyncClient
) -> None:
    found = (await client.get("/api/setup/plan")).json()
    lines = terminal.consent_lines(run.plan(), set(), "")
    assert (found["mac"], found["intro"]) == (lines[0], lines[1])
    assert f"{found['download']} {found['free']}" == lines[-1]
    # The terminal's rows, but the gateway's login item: a job inside the gateway
    # does not register it.
    assert [(r["id"], r["state"], r["off"]) for r in found["rows"]] == [
        ("melo_serve", "todo", False),
        ("backend:omlx", "todo", True),
        (CHAT_ID, "todo", True),
        (EMBED_ID, "partial", True),
        ("service:melo_serve", "todo", True),
    ]
    assert (found["rows"][1]["name"], found["rows"][1]["where"]) == (
        "oMLX 0.7.3",
        "pinned download → ~/Applications/oMLX.app",
    )
    assert found["rows"][3]["where"] == "0.5 of 1.3 GB already here"
    assert found["pick"]["backend_name"] == "oMLX" and found["can_set_up"]
    assert found["action"] == "Continue (6.6 GB left)"  # part of it is here already
    zh = {"X-LocalMelo-Language": "zh-Hans"}
    chinese = (await client.get("/api/setup/plan", headers=zh)).json()
    assert (
        chinese["intro"].startswith("推荐：")
        and chinese["rows"][0]["kind"] == "模型服务"
    )
    assert [r["id"] for r in chinese["rows"]] == [r["id"] for r in found["rows"]]


async def test_nothing_here_yet_says_set_up_with_the_size(
    setup_fakes: Fakes, client: httpx.AsyncClient
) -> None:
    data = json.loads(PLAN.read_text())
    data["items"][2].update(state="todo", cached_bytes=0)
    data["download_bytes"] += 500_000_000
    setup_fakes.put("plan.json", json.dumps(data))
    found = (await client.get("/api/setup/plan")).json()
    assert (
        found["action"]
        == "Set up Qwen3 8B (7.1 GB)"
        == terminal.setup_row(run.plan(), ())
    )


async def test_the_list_is_worded_again_for_what_was_left_out(
    setup_fakes: Fakes, client: httpx.AsyncClient
) -> None:
    found = (await client.get("/api/setup/plan", params={"skip": EMBED_ID})).json()
    assert found["notes"] == [
        "Without Qwen3 Embedding 0.6B, memory goes by words only."
    ]
    assert (
        found["rows"][3]["state"] == "skipped"
        and found["download"] == "Download about 5.8 GB."
    )
    both = {"skip": f"{CHAT_ID},backend:omlx"}
    nothing = (await client.get("/api/setup/plan", params=both)).json()
    assert nothing["notes"][0].startswith(
        "Without oMLX and Qwen3 8B, melo_serve cannot chat yet. "
    )
    bad = await client.get("/api/setup/plan", params={"skip": "model:nope"})
    assert (bad.status_code, bad.json()["error"]["code"]) == (400, "invalid_request")


class Counted:
    """A ``run.plan`` that counts its calls and waits for a gate."""

    def __init__(self, answer: run.Plan) -> None:
        self.answer, self.calls = answer, 0

    def __call__(self, timeout: float = 0.0, pick: str = "") -> run.Plan:
        self.calls += 1
        return self.answer


async def test_one_asker_serves_every_caller_and_an_answer_is_kept_for_a_minute(
    setup_fakes: Fakes,
    client: httpx.AsyncClient,
    console: Console,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counted = Counted(run.plan())
    monkeypatch.setattr(run, "plan", counted)
    now = [100.0]
    console.setup.clock = lambda: now[0]
    answers = await asyncio.gather(*(client.get("/api/setup/plan") for _ in range(5)))
    assert counted.calls == 1 and {r.status_code for r in answers} == {200}
    now[0] += 59
    await client.get("/api/setup/plan")
    assert counted.calls == 1
    now[0] += 2
    await client.get("/api/setup/plan")
    assert counted.calls == 2


async def test_a_plan_that_cannot_run_says_why_and_is_asked_again_soon(
    setup_fakes: Fakes,
    client: httpx.AsyncClient,
    console: Console,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counted = Counted(
        run.Plan(False, "no model can run on this Mac", hint="Add memory.")
    )
    monkeypatch.setattr(run, "plan", counted)
    now = [0.0]
    console.setup.clock = lambda: now[0]
    found = (await client.get("/api/setup/plan")).json()
    assert found == {
        "available": False,
        "reason": "no model can run on this Mac",
        "hint": "Add memory.",
    }
    await client.get("/api/setup/plan")
    assert counted.calls == 1
    now[0] += 6  # the network may be back: not kept for a minute
    await client.get("/api/setup/plan")
    assert counted.calls == 2
    monkeypatch.setattr(setup_flow.sys, "platform", "linux")
    other = (await client.get("/api/setup/plan")).json()
    assert other["available"] is False and other["reason"] == "this is not a Mac"
    assert counted.calls == 2  # nothing was asked of uv


# ── Starting and stopping ──


async def test_a_start_is_checked_against_the_last_plan_that_was_served(
    setup_fakes: Fakes, client: httpx.AsyncClient
) -> None:
    first = await start(client)
    assert (first.status_code, first.json()["error"]["code"]) == (409, "no_plan")
    await client.get("/api/setup/plan")
    for body in (
        {"pick": "nope-omlx", "skip": []},
        {"pick": CHAT, "skip": ["model:nope"]},
        {"pick": CHAT, "skip": ["--hf-home"]},
        {"pick": CHAT, "extra": 1},
    ):
        refused = await client.post("/api/setup/start", json=body)
        assert (refused.status_code, refused.json()["error"]["code"]) == (
            400,
            "invalid_request",
        )
    assert await job(client) is None and setup_fakes.calls("melo-serve", "get") == []
    # An alternative of the plan is a pick too, and so are the rows the job adds.
    ok = await start(client, "service:gateway", "melo_serve", pick="qwen3-4b-omlx")
    assert ok.status_code == 202
    await ended(client)


async def test_a_job_runs_in_the_gateway_and_its_progress_rides_on_detect(
    setup_fakes: Fakes,
    client: httpx.AsyncClient,
    _no_gateway_service: list[str],
) -> None:
    assert await job(client) is None
    await client.get("/api/setup/plan")
    started = await start(client, EMBED_ID, "service:melo_serve")
    assert started.status_code == 202
    body = started.json()["job"]
    assert (body["state"], body["pick"], body["name"]) == ("running", CHAT, "Qwen3 8B")
    assert states(body)[EMBED_ID] == "skipped"  # what was left out says so at once
    done = await ended(client)
    assert (done["state"], done["error"], done["saved"]) == ("done", "", True)
    assert [row["id"] for row in done["rows"]] == [
        "melo_serve",
        "backend:omlx",
        CHAT_ID,
        EMBED_ID,
        "service:melo_serve",
        "load:chat",
        "save",
    ]
    assert states(done) == {
        "melo_serve": "done",
        "backend:omlx": "done",
        CHAT_ID: "done",
        EMBED_ID: "skipped",
        "service:melo_serve": "skipped",
        "load:chat": "done",
        "save": "done",
    }
    names = {row["id"]: row["name"] for row in done["rows"]}
    assert names["load:chat"] == "Loading Qwen3 8B" and names[CHAT_ID] == "Qwen3 8B"
    # The job's own sentences, as the terminal shows them.
    lines = {row["id"]: row["line"] for row in done["rows"]}
    assert lines["backend:omlx"] == "installed 0.7.3 into ~/Applications/oMLX.app"
    assert lines["save"] == "saved melo_serve as the chat server"
    assert setup_fakes.calls("melo-serve", "get") == [
        ["get", CHAT, EMBED, "--skip", EMBED_ID, "--json"]
    ]
    assert config.load().chat_backend == "melo_serve"
    assert _no_gateway_service == []
    # One more may follow when the first is over.
    assert (await start(client)).status_code == 202
    await ended(client)


async def test_a_download_shows_how_far_it_is_and_stop_ends_the_job_keeping_the_files(
    setup_fakes: Fakes, client: httpx.AsyncClient
) -> None:
    setup_fakes.put("get_hangs", "")
    megabytes = {**OMLX_RUNNING, "done_bytes": 310_000_000, "total_bytes": 881_000_000}
    setup_fakes.put("get.jsonl", json.dumps(megabytes))
    assert (await client.post("/api/setup/cancel")).status_code == 409
    await client.get("/api/setup/plan")
    assert (await start(client)).status_code == 202

    async def downloading() -> dict[str, Any] | None:
        found = await job(client)
        row = (
            next(r for r in found["rows"] if r["id"] == "backend:omlx") if found else {}
        )
        return row if row.get("state") == "running" else None

    row = await until(downloading)
    assert (row["progress"], row["done_bytes"], row["total_bytes"]) == (
        "310 of 881 MB",
        310_000_000,
        881_000_000,
    )
    again = await start(client)
    assert (again.status_code, again.json()["error"]["code"]) == (409, "setup_running")
    stopping = await client.post("/api/setup/cancel")
    assert stopping.status_code == 200 and stopping.json()["job"]["stopping"] is True
    stopped = await ended(client)
    assert (stopped["state"], stopped["stopping"], stopped["error"]) == (
        "cancelled",
        False,
        "",
    )
    # The row that was downloading was stopped; what never started waits.
    assert states(stopped)["backend:omlx"] == "skipped"
    assert states(stopped)["save"] == "waiting" and stopped["saved"] is False
    assert setup_fakes.calls("melo-serve", "use") == []  # nothing after the stop ran
    gone = await client.post("/api/setup/cancel")
    assert (gone.status_code, gone.json()["error"]["code"]) == (409, "no_setup")


async def test_a_failed_job_says_why_and_the_next_look_at_the_plan_asks_again(
    setup_fakes: Fakes, client: httpx.AsyncClient
) -> None:
    asked = lambda: len(setup_fakes.calls("uv", "tool", "run"))  # noqa: E731
    await client.get("/api/setup/plan")
    await client.get("/api/setup/plan")
    assert asked() == 1
    setup_fakes.put("uv_fails", "")
    assert (await start(client)).status_code == 202
    failed = await ended(client)
    assert failed["state"] == "failed"
    assert failed["error"] == "failed: error: network is down; nothing was changed"
    assert states(failed)["melo_serve"] == "failed"
    after = (
        await client.get("/api/setup/plan")
    ).json()  # what is here may have changed
    assert asked() == 2 and after["reason"] == "error: network is down"


async def test_a_gateway_that_shuts_down_ends_the_job_cancelled(
    setup_fakes: Fakes, client: httpx.AsyncClient, console: Console
) -> None:
    setup_fakes.put("get_hangs", "")
    await client.get("/api/setup/plan")
    assert (await start(client)).status_code == 202
    await until(lambda: setup_fakes.calls("melo-serve", "get"))
    await console.aclose()
    flow = console.setup.job
    assert flow is not None and flow.finished is not None
    assert flow.finished.state == "cancelled"
    assert not flow.thread.is_alive()
    assert Path(config.config_path()).exists() is False  # nothing was saved
