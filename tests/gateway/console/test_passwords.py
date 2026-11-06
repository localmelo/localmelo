"""A password typed in any entry is cut out where it comes in, once (#547).

Every route that takes the person's text (a first turn, a turn, a note, a
queued message, a routed task, a reply to a task's question, a chore, a chat's
new title) hides a password in it before the text is routed, queued, titled or
stored. The store, the model's requests and the events then hold the handle, and
the value is in none of them.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.sign_in import VAULT
from localmelo.support.gateway.console import Console

from .test_ask_owner import Script, answer, ask_console, held, tool_call, until
from .test_turn_writes import _post_turn, _wait_turn

SECRETS = {
    "turn": "turnSecret-11",
    "note": "noteSecret-22",
    "create": "createSecret-33",
    "task": "taskSecret-44",
    "queued": "queuedSecret-55",
    "chore": "choreSecret-66",
    "reply": "replySecret-77",
    "title": "titleSecret-88",
}
STEP = tool_call("task_get", {"task_id": "000000000000"}, "step-1")


def say(kind: str) -> str:
    return f"sign in to forum.example for me, my password is {SECRETS[kind]}"


def stored(path: Path) -> str:
    """Every row of every table of the store, as text."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return "\n".join(conn.iterdump())


def seen_by_models(models: dict[str, Any]) -> str:
    scripts: list[Script] = [models["main"], *models["workers"]]
    scripts += [
        v
        for k, v in models.items()
        if k not in ("main", "workers", "browsers", "agents")
    ]
    return "\n".join(
        str(message.content)
        for script in scripts
        for request in script.requests
        for message in request
    )


async def test_a_password_typed_in_any_entry_is_in_no_store_title_event_or_model_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    main = [held(release, STEP), answer("Done.")]
    async with ask_console(tmp_path, monkeypatch, main=main, run=False) as (
        console,
        client,
        models,
    ):
        # A turn, and a note added to it while it runs.
        started = await _post_turn(client, str(console.main_id), say("turn"))
        assert started.status_code == 202, started.text
        turn_id = started.json()["turn_id"]
        await until(lambda: len(models["main"].requests) == 1)
        noted = await client.post(
            f"/api/turns/{turn_id}/notes", json={"text": say("note")}
        )
        assert noted.status_code == 202, noted.text
        release.set()
        await _wait_turn(client, str(console.main_id))

        # A new chat with its first turn: its title is the typed text.
        created = await client.post(
            "/api/sessions", json={"turn": {"text": say("create")}}
        )
        assert created.status_code == 201, created.text
        side = created.json()["session"]["id"]
        await _wait_turn(client, side)
        assert SECRETS["create"] not in created.text

        # A message routed to a task of the chat.
        routed = await _post_turn(
            client, str(console.main_id), say("task"), as_task=True
        )
        assert routed.status_code == 200, routed.text
        assert SECRETS["task"] not in routed.text

        # A message queued while the chat answers a task's question.
        queued: list[tuple[Any, ...]] = []

        async def queue(*args: Any) -> dict[str, Any]:
            queued.append(args)
            return {"echo": "queued"}

        with monkeypatch.context() as spy:
            spy.setattr(console.asks, "holds", lambda session: True)
            spy.setattr(console.asks, "queue", queue)
            await _post_turn(client, str(console.main_id), say("queued"))
        assert SECRETS["queued"] not in repr(queued) and "[password-" in repr(queued)

        # A reply to a task's question, a chore, and a chat's new title.
        replies: list[tuple[Any, ...]] = []

        async def act(*args: Any) -> dict[str, Any]:
            replies.append(args)
            return {}

        with monkeypatch.context() as spy:
            spy.setattr(console.front_for(None), "act", act)
            await client.post(
                "/api/front/echoes/1", json={"do": "reply", "text": say("reply")}
            )
        assert SECRETS["reply"] not in repr(replies) and "[password-" in repr(replies)
        chore = await client.post(
            "/api/chores",
            json={
                "request": say("chore"),
                "schedule": {"kind": "weekdays", "at": "08:30"},
            },
        )
        assert chore.status_code == 201, chore.text
        renamed = await client.patch(
            f"/api/sessions/{side}", json={"title": say("title")}
        )
        assert renamed.status_code == 200, renamed.text

        await _wait_idle(console)
        everything = (
            stored(tmp_path / "sessions.db")
            + seen_by_models(models)
            + created.text
            + routed.text
            + chore.text
            + renamed.text
            + (await client.get(f"/api/sessions/{side}")).text
            + (await client.get("/api/chores")).text
        )
        for kind, secret in SECRETS.items():
            assert secret not in everything, f"{kind} password leaked"
        # The handle took its place: the model read it, the store kept it.
        assert "my password is [password-" in stored(tmp_path / "sessions.db")
        assert VAULT.value("[password-1]") == SECRETS["turn"]


async def _wait_idle(console: Console) -> None:
    await until(lambda: not console.sessions.find_turn("none"))
    await asyncio.sleep(0.05)


async def test_a_password_in_a_task_given_directly_is_hidden_where_the_task_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``POST /api/tasks`` takes the person's words too: the objective, and a title they chose."""
    objective, title = "directSecret-91", "titleDirect-92"
    workspace = tmp_path / "work"
    workspace.mkdir()
    body = {
        "objective": f"sign in to forum.example, my password is {objective}",
        "title": f"Sign in (password: {title})",
        "level": "p5",
        "workspace": str(workspace),
        "request_key": "direct-1",
    }
    async with ask_console(tmp_path, monkeypatch, run=False) as (console, client, _):
        made = await client.post("/api/tasks", json=body)
        assert made.status_code == 202, made.text
        # The same request again is the same task: the hidden copy is the same text.
        again = await client.post("/api/tasks", json=body)
        assert again.status_code == 200, again.text
        assert again.json()["task"]["task_id"] == made.json()["task"]["task_id"]
        listed = await client.get("/api/tasks")
        # A different request under the same key is still a conflict.
        other = await client.post(
            "/api/tasks", json={**body, "objective": "water the plants"}
        )
        assert other.status_code == 409, other.text
        # Ordinary text is left as it is.
        plain = await client.post(
            "/api/tasks",
            json={**body, "objective": "water the plants", "request_key": "p"},
        )
        assert plain.status_code == 202 and "water the plants" in plain.text
        await _wait_idle(console)
        disks = "\n".join(stored(path) for path in sorted(tmp_path.rglob("*.db")))
        everything = disks + made.text + again.text + listed.text + plain.text
        for secret in (objective, title):
            assert secret not in everything, f"{secret} reached a copy of the task"
        task = made.json()["task"]
        assert "[password-" in made.text and "[password-" in task["title"], task
        assert (
            "password is [password-" in disks
        )  # the objective, hidden, is what is kept
        assert VAULT.value("[password-1]") in (objective, title)
