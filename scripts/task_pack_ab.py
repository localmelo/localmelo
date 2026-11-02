"""Task-pack A/B: does a chat-started task stay on the right subject with the handover?

A task the chat starts runs in a session of its own. Before, its first request
held the request and the notes the code matched to it; after, it also holds
the chat's newest turns and what the chat's turn recalled or searched (and the
person's words when the request does not hold them). This script builds both
first requests with the real code (a real session store, real words-only
memory, real task workers on a recording model), then replays each one to a
model and judges whether the task's first step is about the person or thing
the request means. It is not packaged and changes nothing outside a temporary
directory.

The cases
---------
Ten chat-started tasks whose answer depends on an earlier turn or a recalled
item: "read his résumé first", "去查一下", "go look it up", "write it up for
her", "check what he said about the deadline", "the format we agreed on", "the
trip", "看看他的作品集", "compare their battery tests", "draft the reply to her
email". The person's own notes say who they are (Sam, a data engineer in
Rotterdam), so a task that loses the subject tends to work on the person.

Each case has the words that show the right subject (``right``) and the words
that show the person's own profile (``wrong``). A reply (its text and any tool
call's arguments) is **right** when it holds a right word, **wrong** when it
holds a wrong word and no right one, else **unclear**.

Decision rule, fixed before any run
-----------------------------------
The handover stays as built when, over all reps, the after arm is right on at
least 2 more case-reps than the before arm per rep on average, and no case
that before got right in most reps is unclear or wrong after in most reps.
Otherwise the per-case replies go to review before anything changes.

Also reported: each arm's first-request characters (whole request and the
task's own message), and how much a typical first request grows.

Usage::

    uv run python scripts/task_pack_ab.py --stub --out stub.json
    uv run python scripts/task_pack_ab.py --base-url http://127.0.0.1:8000/v1 \\
        --model my-model --reps 2 --out results.json --dump requests

``--stub`` answers with a stand-in model that names the right subject only when
the request holds it: it proves the plumbing, not the model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx

from localmelo.melo.agent import Agent
from localmelo.melo.contracts.providers import BaseLLMProvider
from localmelo.melo.schema import Message, ToolDef
from localmelo.support.config import Config
from localmelo.support.gateway import workers as workers_module
from localmelo.support.gateway.chat_gate import ChatGate
from localmelo.support.gateway.handover import handover_of
from localmelo.support.gateway.task_policy import TaskSpec
from localmelo.support.gateway.workers import TaskWorkers
from localmelo.support.memory import ChatMemory, records
from localmelo.support.memory.service import CHAT_TURN
from localmelo.support.sessions.agents import session_options
from localmelo.support.sessions.hub import SessionHub
from localmelo.support.sessions.store import NewSession, SessionStore, TurnInfo

HOLDER = "task-pack-ab:1"
OWNER_NOTES = (
    ("Your name is Sam.", "My name is Sam.", "about"),
    ("You are a data engineer.", "I am a data engineer.", "about"),
    ("You live in Rotterdam.", "I live in Rotterdam.", "about"),
)
OWNER_WORDS = ("Sam", "data engineer", "Rotterdam")


@dataclass(frozen=True)
class Case:
    """One chat-started task: the chat's turns, its request, and how to judge it."""

    name: str
    request: str
    right: tuple[str, ...]
    turns: tuple[tuple[str, str], ...] = ()
    other_chat: tuple[tuple[str, str], ...] = ()
    search: str = ""  # the chat's own memory search, as its model would make it
    wrong: tuple[str, ...] = OWNER_WORDS


CASES: tuple[Case, ...] = (
    Case(
        "resume",
        "read his résumé first and tell me which roles fit",
        ("Alex",),
        turns=(
            (
                "My friend Alex is looking for a new job.",
                "Got it, Alex is job hunting.",
            ),
            ("He has eight years of backend work in Go.", "Eight years of Go, noted."),
        ),
    ),
    Case(
        "look-up-zh",
        "去查一下",
        ("小林", "乌得勒支", "Utrecht", "1200"),
        turns=(
            ("我朋友小林在找乌得勒支的公寓。", "好的，小林在乌得勒支找房子。"),
            ("她的预算是每月1200欧。", "记下了：每月1200欧。"),
        ),
    ),
    Case(
        "look-it-up",
        "go look it up",
        ("parsley", "2.3", "empty input"),
        turns=(
            ("The parsley parser crashes on empty input.", "Probably a missing guard."),
            ("It started after version 2.3.", "So 2.3 introduced it."),
        ),
    ),
    Case(
        "write-it-up",
        "write it up for her",
        ("Maya", "coral"),
        turns=(
            ("My friend Maya finished her thesis on coral reefs.", "Congratulations!"),
            (
                "She wants a short summary for her committee.",
                "A short summary, got it.",
            ),
        ),
    ),
    Case(
        "deadline",
        "check what he said about the deadline and plan my week",
        ("March 3", "Ben"),
        other_chat=(("Ben said the grant deadline moved to March 3.", "Noted."),),
        search="Ben grant deadline",
        wrong=(),
    ),
    Case(
        "format",
        "make the expense report in the format we agreed on",
        ("amount",),
        other_chat=(
            (
                "For expense reports use a table with columns date, item, amount.",
                "Will do.",
            ),
        ),
        search="expense report format",
        wrong=(),
    ),
    Case(
        "trip",
        "find hotels for the trip",
        ("Lisbon",),
        other_chat=(("Alex and I fly to Lisbon on May 12 for the trip.", "Noted."),),
        search="trip",
        wrong=(),
    ),
    Case(
        "portfolio-zh",
        "看看他的作品集，帮他准备面试",
        ("阿杰",),
        turns=(
            ("我朋友阿杰是做UI设计的。", "好的，阿杰是UI设计师。"),
            ("他想转去做产品经理。", "明白，阿杰想转产品经理。"),
        ),
    ),
    Case(
        "compare",
        "compare their battery tests in reviews",
        ("X1", "MacBook"),
        turns=(
            (
                "I'm choosing between the ThinkPad X1 and a MacBook Air.",
                "Both are light.",
            ),
            ("Battery life matters most.", "Then battery decides it."),
        ),
        wrong=(),
    ),
    Case(
        "reply-email",
        "draft the reply to her email",
        ("Chen", "lease", "rent"),
        turns=(
            ("My landlord Ms. Chen emailed about renewing the lease.", "Renew?"),
            ("Yes, but I want to ask for a lower rent.", "Renew, lower rent."),
        ),
    ),
)


class Recording(BaseLLMProvider):
    """The task's model while the request is built: keeps the first request, ends the run."""

    def __init__(self) -> None:
        self.requests: list[tuple[list[Message], list[ToolDef]]] = []

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.requests.append((list(messages), list(tools or [])))
        return Message(role="assistant", content="Done.\nEnding: ready")

    async def close(self) -> None:
        return None


class Stub(BaseLLMProvider):
    """Names the case's right subject only when the request holds it."""

    def __init__(self, case: Case) -> None:
        self.case = case

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        text = "\n".join(str(m.content) for m in messages if m.role == "user")
        found = [word for word in self.case.right if word in text]
        about = found[0] if found else "your own profile"
        return Message(role="assistant", content=f"First I look into {about}.")

    async def close(self) -> None:
        return None


def config(root: Path) -> Config:
    cfg = Config(chat_backend="vllm", embedding_backend="none")
    cfg.gateway.workspace = str(root / "workspace")
    return cfg


async def say(
    store: SessionStore,
    chat: str | None,
    typed: str,
    answer: str | None,
    *,
    kind: str = "front",
) -> TurnInfo:
    """A turn of *typed* in *chat* (a new chat of *kind* when None); running when no *answer*."""
    info = await store.begin_turn(
        chat,
        typed,
        holder=HOLDER,
        typed_text=typed,
        new_session=None if chat else NewSession(kind=kind),
    )
    if answer is None:
        return info
    return await store.finish_turn(
        info.session_id,
        info.turn,
        holder=HOLDER,
        status="completed",
        cause="run",
        result=None,
        answer=answer,
    )


async def first_request(case: Case, root: Path, *, after: bool) -> dict[str, Any]:
    """The task's first request for *case*, built by the real code, before or after."""
    (root / "workspace").mkdir(parents=True, exist_ok=True)
    cfg = config(root)
    store = SessionStore(root / "sessions.db", writer="gateway")
    await store.open()
    memory = ChatMemory(lambda: cfg, directory=str(root / "memory"))
    gate = ChatGate(policy="off")
    hub = SessionHub(store)
    recording = Recording()

    async def build(**kwargs: Any) -> Agent:
        agent = Agent(
            llm=kwargs["wrap_llm"](recording),
            options=session_options(kwargs["options"], kwargs["workspace"]),
        )
        await agent.start()
        return agent

    workers_module.build_session_agent = build  # type: ignore[assignment]
    workers = TaskWorkers(store, gate, lambda: cfg, hub, memory=memory)
    await workers.open()
    try:
        for text, quote, topic in OWNER_NOTES:
            cand = records.Candidate(text, quote, topic, "user_statement")
            source = {"session_id": "owner", "turn": 1, "chat": "Main chat"}
            await records.write(
                memory.stores()[0], [cand], [None], source=source, embed_model=""
            )
        other: str | None = None
        for typed, answer in case.other_chat:
            done = await say(store, other, typed, answer, kind="chat")
            other = done.session_id
            memory.after_turn(store, done, typed, answer)
        chat: str | None = None
        for typed, answer in case.turns:
            done = await say(store, chat, typed, answer)
            chat = done.session_id
            memory.after_turn(store, done, typed, answer)
        await memory.drain()
        now = await say(store, chat, case.request, None)
        # The chat's turn: its recall, then its own memory search.
        await memory.before_turn(store, now, case.request)
        turn = CHAT_TURN.get()
        if case.search and turn is not None:
            turn.add_searched(await memory.search(case.search))
        handover = (
            await handover_of(store, now.session_id, now, turn) if after else None
        )
        CHAT_TURN.set(None)
        spec = TaskSpec.from_payload(
            {"objective": case.request, "workspace": str(root / "workspace")}
        )
        # Before: no own words either (the request is the person's message here).
        spec = replace(spec, user_words=case.request if after else "")
        task, _ = await workers.create_spec(
            spec,
            front_id=now.session_id,
            chat_started=True,
            handover=handover,
        )
        async with asyncio.timeout(30):
            while not recording.requests:
                await asyncio.sleep(0.02)
        messages, tools = recording.requests[0]
        own = next(m for m in messages if m.role == "user")
        stored = (await workers.get(task["task_id"]))["spec"]
        return {
            "messages": messages,
            "tools": tools,
            "request_chars": sum(len(str(m.content)) for m in messages),
            "message_chars": len(str(own.content)),
            "context_chars": len(str(own.content)) - len(case.request),
            "spec": {
                k: stored.get(k, "")
                for k in ("own_words", "chat_turns", "chat_picked", "context_pack")
            },
            "holds_right": any(word in str(own.content) for word in case.right),
        }
    finally:
        await workers.aclose()
        await hub.aclose()
        await gate.close()
        await memory.aclose()
        await store.aclose()


def verdict(case: Case, reply: Message) -> str:
    calls = getattr(reply, "tool_calls", None) or []
    text = (
        str(reply.content or "")
        + " "
        + json.dumps(
            [getattr(call, "arguments", call) for call in calls],
            ensure_ascii=False,
            default=str,
        )
    )
    if any(word.casefold() in text.casefold() for word in case.right):
        return "right"
    if any(word.casefold() in text.casefold() for word in case.wrong):
        return "wrong"
    return "unclear"


def environment(args: argparse.Namespace) -> dict[str, Any]:
    def command(*argv: str) -> str:
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"unavailable: {exc}"
        return (done.stdout or done.stderr).strip() or f"exit {done.returncode}"

    models: Any = "stub"
    if not args.stub:
        try:
            models = httpx.get(f"{args.base_url.rstrip('/')}/models", timeout=10).json()
        except (httpx.HTTPError, ValueError) as exc:
            models = f"unavailable: {exc}"
    return {
        "model": "stub" if args.stub else args.model,
        "server": "" if args.stub else args.base_url,
        "server_models": models,
        "platform": platform.platform(),
        "sw_vers_build": command("sw_vers", "-buildVersion"),
        "xcodebuild_version": command("xcodebuild", "-version"),
        "python": sys.version,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def judge_model(args: argparse.Namespace, case: Case) -> BaseLLMProvider:
    if args.stub:
        return Stub(case)
    from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

    return OpenAICompatLLM(
        args.base_url,
        args.model,
        api_key=os.environ.get("TASK_PACK_AB_API_KEY"),
        timeout=args.request_timeout,
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = [case for case in CASES if not args.cases or case.name in args.cases]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="task-pack-ab-") as work:
        # Nothing reads or writes the real home: memory, sessions and tasks live here.
        os.environ["HOME"] = work
        for case in cases:
            row: dict[str, Any] = {"case": case.name, "arms": {}}
            for arm in ("before", "after"):
                built = await first_request(
                    case, Path(work) / case.name / arm, after=arm == "after"
                )
                if args.dump:
                    out = Path(args.dump) / f"{case.name}-{arm}.txt"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(
                        "\n\n".join(
                            f"## {m.role}\n{m.content}" for m in built["messages"]
                        ),
                        encoding="utf-8",
                    )
                verdicts, replies = [], []
                for _ in range(args.reps):
                    model = judge_model(args, case)
                    try:
                        reply = await model.chat(built["messages"], built["tools"])
                    finally:
                        await model.close()
                    verdicts.append(verdict(case, reply))
                    replies.append(str(reply.content or "")[:300])
                row["arms"][arm] = {
                    "request_chars": built["request_chars"],
                    "message_chars": built["message_chars"],
                    "context_chars": built["context_chars"],
                    "holds_right": built["holds_right"],
                    "spec": built["spec"],
                    "verdicts": verdicts,
                    "replies": replies,
                }
            rows.append(row)
            print(
                f"{case.name:14} before {row['arms']['before']['verdicts']} "
                f"after {row['arms']['after']['verdicts']} "
                f"+{row['arms']['after']['message_chars'] - row['arms']['before']['message_chars']} chars",
                flush=True,
            )
    return {"env": environment(args), "cases": rows, "summary": summary(rows, args)}


def summary(rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    def right(arm: str) -> int:
        return sum(row["arms"][arm]["verdicts"].count("right") for row in rows)

    def mostly_right(row: dict[str, Any], arm: str) -> bool:
        return bool(row["arms"][arm]["verdicts"].count("right") * 2 > args.reps)

    lost = [
        row["case"]
        for row in rows
        if mostly_right(row, "before") and not mostly_right(row, "after")
    ]
    growth = [
        row["arms"]["after"]["message_chars"] - row["arms"]["before"]["message_chars"]
        for row in rows
    ]
    gain = (right("after") - right("before")) / max(args.reps, 1)
    return {
        "right_before": right("before"),
        "right_after": right("after"),
        "case_reps": len(rows) * args.reps,
        "request_holds_right_before": sum(
            r["arms"]["before"]["holds_right"] for r in rows
        ),
        "request_holds_right_after": sum(
            r["arms"]["after"]["holds_right"] for r in rows
        ),
        "message_growth_chars": {
            "median": statistics.median(growth) if growth else 0,
            "max": max(growth, default=0),
            "min": min(growth, default=0),
        },
        "lost_cases": lost,
        "keeps": gain >= 2 and not lost,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stub", action="store_true", help="a stand-in model")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible /v1 URL")
    parser.add_argument("--model", default="", help="the model's name on the server")
    parser.add_argument("--reps", type=int, default=1, help="replies per case and arm")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--cases", nargs="*", default=[], help="case names to run")
    parser.add_argument("--out", default="", help="write the results here (JSON)")
    parser.add_argument("--dump", default="", help="write each first request here")
    args = parser.parse_args(argv)
    if not args.stub and not (args.base_url and args.model):
        parser.error("pass --stub, or --base-url and --model")
    os.umask(0o077)  # the stores hold transcripts
    results = asyncio.run(run(args))
    print(json.dumps(results["summary"], indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
