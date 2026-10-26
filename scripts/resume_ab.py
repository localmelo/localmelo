"""Resume A/B: after an interruption, does restoring the window beat pairs?

A session store keeps a checkpoint of the working window at every turn end,
so a fresh Agent can resume a session two ways (SessionRunner's
``resume_mode``): ``pairs`` rebuilds the window from each turn's user text
and answer, ``window`` restores the checkpoint, tool steps included. This
script compares the two on the same stored states, against a real
OpenAI-compatible model server. It is not packaged and changes nothing: it
works in a directory of its own (``--work``).

Decision rule, fixed before any run
-----------------------------------
``window`` becomes the default only if all three hold; otherwise ``pairs``
stays:

1. its next-step success beats ``pairs`` over the 90 paired states by an
   exact two-sided McNemar test at p < 0.05, with ``window`` winning more of
   the discordant pairs;
2. its answer recall (the fact that appears only in an earlier answer) is at
   most 5 points below ``pairs``';
3. its median probe prompt tokens are at most 1.5 times ``pairs``'.

With 90 pairs, the smallest split that passes rule 1 is 6 to 0 discordant
pairs (p = 0.031; 8 to 1 and 10 to 2 give 0.039), so the smallest gain the
test can confirm is about 7 points of next-step success. When
0.05 <= p < 0.2, ``pairs`` stays, and the lead may order one independent run
on 30 new tasks (``--task-seed``), judged alone by the same rule; the two
runs are never pooled.

The design
----------
- 30 scripted tasks of 3 to 5 turns, each in a fresh workspace with
  ``file_read``, ``file_write`` and ``shell_exec`` under an allow-all
  policy. Each task holds a code word that appears only in a tool output of
  its first turn, and a ticket number that appears only in an answer.
- Each task is interrupted once, before its last turn (the probe), in three
  ways: (a) a graceful restart between turns, in a child process; (b) a
  process death (``os._exit(73)``) during a tool call of the probe turn;
  (c) an Agent eviction and a reopen with no in-memory copy. That gives 90
  stored states.
- Each state is resumed by both arms, three times each (``--seeds`` with
  ``--seed-param``, else three repetitions), so the arms are paired on task,
  interruption and state; a pair's outcome is the majority of its three.
- The probe turn measures next-step success (the task's check command
  passes in the workspace and no side-effecting call repeats one that ran
  before the interruption), recall of each fact, repeated work, and cost:
  the probe's first planning request's prompt tokens and latency (the
  provider does not stream, so this is the time to the whole reply), the
  resumed window's characters, prompt-cap drops and ``load_conversation``
  warnings.
- The results name the model, the server and its version, ``sw_vers``'s
  build and ``xcodebuild -version``.

Usage::

    uv run python scripts/resume_ab.py --base-url http://127.0.0.1:8000/v1 \\
        --model qwen3-8b --work /tmp/resume-ab --out results.json
    uv run python scripts/resume_ab.py --self-test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from localmelo.melo.agent import Agent, AgentOptions
from localmelo.melo.contracts.approvals import AllowAllPolicy
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import (
    NewSession,
    SessionStore,
    StoredEvent,
)

ARMS = ("pairs", "window")
INTERRUPTS = ("a", "b", "c")
TOOLS = frozenset({"file_read", "file_write", "shell_exec"})
SIDE_EFFECTS = frozenset({"file_write", "shell_exec"})
TICKET = re.compile(r"TICKET-\d{4}")
NO_TOOL_CALL = "the probe turn made no tool call"
CHECK_PY = """\
import pathlib
import sys

lines = pathlib.Path("summary.txt").read_text().split()
sys.exit(0 if sys.argv[1] in lines and sys.argv[2] in lines else 1)
"""
WORDS = (
    "amber", "basil", "cedar", "delta", "ember", "fjord", "garnet", "harbor",
    "indigo", "juniper", "kelp", "lumen", "maple", "nimbus", "onyx", "pepper",
    "quartz", "raven", "saffron", "tundra",
)  # fmt: skip


# ── Tasks ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Task:
    """One scripted task: the turns before the interruption, then the probe."""

    index: int
    data_file: str
    code_word: str
    lines: int
    turns: tuple[str, ...]
    probe: str


def make_tasks(count: int = 30, seed: int = 20261026) -> list[Task]:
    """*count* tasks of 3 to 5 turns, the probe included; the same for a seed."""
    rng = random.Random(seed)
    tasks = []
    for index in range(count):
        name = f"data/{rng.choice(WORDS)}_{index:02d}.txt"
        code = f"{rng.choice(WORDS).upper()}-{rng.randint(1000, 9999)}"
        lines = rng.randint(8, 40)
        turns = [
            f"Read {name} with file_read and tell me only how many lines it has. "
            "Do not repeat anything else from it.",
            "Invent a ticket number for this task in the form TICKET-<four digits> "
            "and tell me only that number.",
        ]
        extra = index % 3  # 0, 1 or 2 more turns: 3 to 5 in all
        if extra >= 1:
            turns.append(
                "Write the line count you found into report.txt with file_write."
            )
        if extra >= 2:
            turns.append(
                "Run `ls` with shell_exec and tell me how many entries it lists."
            )
        probe = (
            "Write summary.txt with file_write: one line with the code word that "
            f"{name} holds, as you read it earlier (do not read the file again), and "
            "one line with the ticket number you gave me. Then run "
            "`python check.py <code word> <ticket>` with shell_exec and tell me "
            "whether it passed."
        )
        tasks.append(Task(index, name, code, lines, tuple(turns), probe))
    return tasks


def private_dir(path: Path) -> None:
    """*path* and its missing parents, readable by this user only, as the store wants."""
    missing = [p for p in (path, *path.parents) if not p.exists()]
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
        directory.chmod(0o700)


def prepare_workspace(task: Task, root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    (root / "data").mkdir(parents=True)
    rng = random.Random(task.index)
    rows = [
        f"row {n}: {rng.choice(WORDS)} {rng.randint(1, 99)}" for n in range(task.lines)
    ]
    rows[rng.randrange(task.lines)] = f"the code word is {task.code_word}"
    (root / task.data_file).write_text("\n".join(rows) + "\n")
    (root / "check.py").write_text(CHECK_PY)


# ── Agents and stores ───────────────────────────────────────────────────────


async def build_agent(args: argparse.Namespace, root: Path, seed: int | None) -> Agent:
    extra = {"seed": seed} if args.seed_param and seed is not None else None
    llm = OpenAICompatLLM(
        args.base_url,
        args.model,
        api_key=os.environ.get("RESUME_AB_API_KEY"),
        timeout=args.request_timeout,
        extra_body=extra,
    )
    agent = Agent(
        llm=llm,
        options=AgentOptions(
            workspace_root=str(root),
            record_answers=True,
            tool_exposure="all",
            builtins=TOOLS,
            timeout_s=args.turn_timeout,
        ),
    )
    await agent.start()
    return agent


def paths(work: Path, task: Task, interrupt: str) -> dict[str, Path]:
    state = work / "states" / f"task-{task.index:02d}-{interrupt}"
    return {
        "state": state,
        "live": work / "live" / f"task-{task.index:02d}",  # one workspace path per task
        "db": state / "sessions.db",
        "snapshot": state / "workspace",
        "meta": state / "meta.json",
    }


async def run_turns(
    args: argparse.Namespace, task: Task, live: Path, db: Path, interrupt: str
) -> str:
    """The turns before the interruption; for (b), the probe that dies in a tool."""
    store = SessionStore(db, writer="gateway")
    agent = await build_agent(args, live, None)

    def die_in_a_tool(event: StoredEvent) -> None:
        if interrupt == "b" and event.event["type"] == "tool.started":
            os._exit(73)

    runner = SessionRunner(
        agent, store, None, new_session=NewSession(), on_event=die_in_a_tool
    )
    policy = AllowAllPolicy()
    for text in task.turns:
        await runner.run_turn(text, approval=policy)
    assert runner.session_id is not None
    if interrupt == "b":
        await runner.run_turn(task.probe, approval=policy)
        # No tool call to die in: die after the turn, and say so.
        print(NO_TOOL_CALL, file=sys.stderr, flush=True)
        os._exit(73)
    # (a) and (c): the gateway stops, or the Agent is evicted, between turns.
    await runner.close(cause="shutdown" if interrupt == "a" else "closed")
    await agent.close()
    await store.aclose()
    return runner.session_id


async def make_state(
    args: argparse.Namespace, task: Task, interrupt: str
) -> dict[str, Any]:
    """One stored state: the file and the workspace after the interruption."""
    where = paths(Path(args.work), task, interrupt)
    if where["meta"].exists():
        meta: dict[str, Any] = json.loads(where["meta"].read_text())
        return meta
    if where["state"].exists():
        shutil.rmtree(where["state"])
    private_dir(where["state"])
    died_in_tool = interrupt == "b"
    prepare_workspace(task, where["live"])
    if interrupt == "c":
        session_id = await run_turns(args, task, where["live"], where["db"], interrupt)
    else:
        child = subprocess.run(
            [
                sys.executable,
                __file__,
                "--child",
                f"{task.index}:{interrupt}",
                *child_args(args),
            ],
            capture_output=True,
            text=True,
            timeout=args.turn_timeout * (len(task.turns) + 2),
        )
        died_in_tool = interrupt == "b" and NO_TOOL_CALL not in child.stderr
        expected = 73 if interrupt == "b" else 0
        if child.returncode != expected:
            raise RuntimeError(
                f"state {task.index}{interrupt}: the child exited {child.returncode}: "
                f"{child.stderr[-2000:]}"
            )
        store = SessionStore(where["db"], writer="gateway")
        sessions = await store.list_sessions(limit=1)
        await store.aclose()  # its open recovered the turn that died
        session_id = sessions[0].id
    shutil.copytree(where["live"], where["snapshot"])
    meta = await describe_state(where["db"], session_id)
    meta["died_in_tool"] = died_in_tool
    where["meta"].write_text(json.dumps(meta, indent=2))
    return meta


def child_args(args: argparse.Namespace) -> list[str]:
    return [
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--work",
        args.work,
        "--turn-timeout",
        str(args.turn_timeout),
        "--request-timeout",
        str(args.request_timeout),
        "--task-seed",
        str(args.task_seed),
    ]


async def describe_state(db: Path, session_id: str) -> dict[str, Any]:
    """The calls made before the interruption, and the facts to recall."""
    store = SessionStore(db, writer="gateway")
    calls = [
        [e.event["data"]["name"], e.event["data"]["arguments"]]
        for e in await all_events(store, session_id)
        if e.event["type"] == "tool.started"
    ]
    tickets = [
        match.group(0)
        for turn in await store.list_turns(session_id)
        for match in TICKET.finditer(turn.answer)
    ]
    await store.aclose()
    return {
        "session_id": session_id,
        "calls": calls,
        "ticket": tickets[0] if tickets else None,
    }


async def all_events(store: SessionStore, session_id: str) -> list[StoredEvent]:
    events: list[StoredEvent] = []
    while True:
        after = (events[-1].id or 0) if events else 0
        page = await store.list_events(session_id, after_id=after)
        events += page
        if len(page) < 1000:
            return events


# ── One probe ───────────────────────────────────────────────────────────────


class _Counter(logging.Handler):
    def __init__(self, needle: str) -> None:
        super().__init__(logging.DEBUG)
        self.needle = needle
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if self.needle in record.getMessage():
            self.count += 1


@dataclass
class Probe:
    task: int
    interrupt: str
    arm: str
    repetition: int
    success: bool = False
    check_passed: bool = False
    repeated_side_effects: int = 0
    repeated_calls: int = 0
    recall_tool_fact: bool = False
    recall_answer_fact: bool | None = None
    reread: bool = False
    prompt_tokens: int | None = None
    first_request_s: float | None = None
    resumed_turns: int = 0
    resumed_checkpoint_turns: int = 0
    fell_back: bool = False
    resumed_chars: int = 0
    prompt_cap_drops: int = 0
    load_warnings: int = 0
    status: str = ""
    error: str = ""
    calls: list[list[Any]] = field(default_factory=list)


async def run_probe(
    args: argparse.Namespace,
    task: Task,
    interrupt: str,
    meta: dict[str, Any],
    arm: str,
    repetition: int,
    seed: int | None,
) -> Probe:
    where = paths(Path(args.work), task, interrupt)
    run_dir = (
        Path(args.work)
        / "runs"
        / f"task-{task.index:02d}-{interrupt}-{arm}-{repetition}"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)
    private_dir(run_dir)
    shutil.copy(where["db"], run_dir / "sessions.db")
    if where["live"].exists():
        shutil.rmtree(where["live"])
    shutil.copytree(where["snapshot"], where["live"])
    probe = Probe(task.index, interrupt, arm, repetition)
    drops = _Counter("oldest working-memory messages")
    warnings = _Counter("from the loaded conversation")
    agent_log = logging.getLogger("localmelo.melo.agent.agent")
    agent_log.addHandler(drops)
    agent_log.addHandler(warnings)
    agent_log.setLevel(logging.INFO)
    events: list[StoredEvent] = []
    store = SessionStore(run_dir / "sessions.db", writer="gateway")
    agent = await build_agent(args, where["live"], seed)
    answer = ""
    try:
        runner = SessionRunner(
            agent,
            store,
            meta["session_id"],
            on_event=events.append,
            resume_mode=arm,  # type: ignore[arg-type]
        )
        await runner.resume()
        info = runner.last_resume
        assert info is not None
        probe.resumed_turns, probe.resumed_checkpoint_turns = (
            info.turns,
            info.checkpoint_turns,
        )
        probe.fell_back, probe.resumed_chars = info.fell_back, info.chars
        try:
            result = await runner.run_turn(task.probe, approval=AllowAllPolicy())
        except Exception as exc:  # recorded, so one broken probe does not end the run
            probe.status, probe.error = "harness_error", repr(exc)
        else:
            probe.status, probe.error, answer = (
                result.status,
                result.error,
                result.result,
            )
        await runner.close()
    finally:
        agent_log.removeHandler(drops)
        agent_log.removeHandler(warnings)
        await agent.close()
        await store.aclose()
    probe.prompt_cap_drops, probe.load_warnings = drops.count, warnings.count
    messages = [
        e.event["data"] for e in events if e.event["type"] == "assistant.message"
    ]
    if messages:
        probe.prompt_tokens = messages[0]["usage"].get("prompt_tokens")
        probe.first_request_s = messages[0]["duration_s"]
    probe.calls = [
        [e.event["data"]["name"], e.event["data"]["arguments"]]
        for e in events
        if e.event["type"] == "tool.started"
    ]
    before = [json.dumps(call, sort_keys=True) for call in meta["calls"]]
    for name, arguments in probe.calls:
        if json.dumps([name, arguments], sort_keys=True) in before:
            probe.repeated_calls += 1
            probe.repeated_side_effects += name in SIDE_EFFECTS
        if name == "file_read" and arguments.get("path", "").endswith(task.data_file):
            probe.reread = True
    ticket = meta["ticket"]
    probe.recall_tool_fact = task.code_word in answer or _summary_has(
        where["live"], task.code_word
    )
    if ticket is not None:
        probe.recall_answer_fact = ticket in answer or _summary_has(
            where["live"], ticket
        )
    probe.check_passed = ticket is not None and _check(
        where["live"], task.code_word, ticket
    )
    probe.success = probe.check_passed and probe.repeated_side_effects == 0
    return probe


def _summary_has(root: Path, text: str) -> bool:
    summary = root / "summary.txt"
    return summary.exists() and text in summary.read_text(errors="replace")


def _check(root: Path, code_word: str, ticket: str) -> bool:
    done = subprocess.run(
        [sys.executable, "check.py", code_word, ticket],
        cwd=root,
        capture_output=True,
        timeout=60,
    )
    return done.returncode == 0


# ── The statistics and the rule ─────────────────────────────────────────────


def mcnemar_exact(window_only: int, pairs_only: int) -> float:
    """The exact two-sided McNemar p-value of a discordant split."""
    n = window_only + pairs_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(window_only, pairs_only) + 1))
    return min(1.0, 2.0 * tail / float(2**n))


def majority(values: Sequence[bool]) -> bool:
    return sum(values) * 2 > len(values)


def decide(probes: Sequence[Probe]) -> dict[str, Any]:
    """The paired outcomes, the three rules and the verdict."""
    by_state: dict[tuple[int, str], dict[str, list[Probe]]] = {}
    for probe in probes:
        by_state.setdefault((probe.task, probe.interrupt), {}).setdefault(
            probe.arm, []
        ).append(probe)
    window_only = pairs_only = both = neither = 0
    recall: dict[str, list[bool]] = {arm: [] for arm in ARMS}
    tool_recall: dict[str, list[bool]] = {arm: [] for arm in ARMS}
    for arms in by_state.values():
        if set(arms) != set(ARMS):
            continue
        won = {arm: majority([p.success for p in arms[arm]]) for arm in ARMS}
        if won["window"] and not won["pairs"]:
            window_only += 1
        elif won["pairs"] and not won["window"]:
            pairs_only += 1
        elif won["window"]:
            both += 1
        else:
            neither += 1
        for arm in ARMS:
            answers = [
                p.recall_answer_fact
                for p in arms[arm]
                if p.recall_answer_fact is not None
            ]
            if answers:
                recall[arm].append(majority(answers))
            tool_recall[arm].append(majority([p.recall_tool_fact for p in arms[arm]]))
    p = mcnemar_exact(window_only, pairs_only)
    rate = {arm: 100 * sum(v) / len(v) if v else 0.0 for arm, v in recall.items()}
    tokens: dict[str, float | None] = {}
    for arm in ARMS:
        counts = [p.prompt_tokens for p in probes if p.arm == arm and p.prompt_tokens]
        tokens[arm] = float(statistics.median(counts)) if counts else None
    rule_1 = p < 0.05 and window_only > pairs_only
    rule_2 = rate["window"] >= rate["pairs"] - 5
    window_tokens, pairs_tokens = tokens["window"], tokens["pairs"]
    rule_3 = (
        window_tokens is not None
        and pairs_tokens is not None
        and window_tokens <= 1.5 * pairs_tokens
    )
    verdict = "window" if rule_1 and rule_2 and rule_3 else "pairs"
    note = ""
    if verdict == "pairs" and 0.05 <= p < 0.2:
        note = "0.05 <= p < 0.2: the lead may order one independent run on 30 new tasks"
    return {
        "pairs": len(by_state),
        "discordant": {"window_only": window_only, "pairs_only": pairs_only},
        "concordant": {"both": both, "neither": neither},
        "mcnemar_p": p,
        "answer_recall_pct": rate,
        "tool_recall_pct": {
            arm: 100 * sum(v) / len(v) if v else 0.0 for arm, v in tool_recall.items()
        },
        "median_prompt_tokens": tokens,
        "rules": {"1_success": rule_1, "2_recall": rule_2, "3_tokens": rule_3},
        "default": verdict,
        "note": note,
    }


def self_test() -> None:
    assert round(mcnemar_exact(6, 0), 3) == 0.031
    assert round(mcnemar_exact(8, 1), 3) == 0.039
    assert round(mcnemar_exact(10, 2), 3) == 0.039
    assert mcnemar_exact(5, 0) > 0.05  # 5 to 0 does not pass
    assert mcnemar_exact(0, 0) == 1.0
    probes = []
    for task in range(10):
        for arm in ARMS:
            for rep in range(3):
                win = arm == "window" or task >= 7
                probes.append(
                    Probe(
                        task,
                        "a",
                        arm,
                        rep,
                        success=win,
                        recall_answer_fact=True,
                        prompt_tokens=100,
                    )
                )
    verdict = decide(probes)
    assert verdict["discordant"] == {"window_only": 7, "pairs_only": 0}
    assert verdict["default"] == "window", verdict
    print("self-test passed")


# ── The run ─────────────────────────────────────────────────────────────────


def environment(args: argparse.Namespace) -> dict[str, Any]:
    def command(*argv: str) -> str:
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"unavailable: {exc}"
        return (done.stdout or done.stderr).strip() or f"exit {done.returncode}"

    try:
        models = httpx.get(f"{args.base_url.rstrip('/')}/models", timeout=10).json()
    except (httpx.HTTPError, ValueError) as exc:
        models = f"unavailable: {exc}"
    return {
        "model": args.model,
        "server": args.base_url,
        "server_version": args.server_version,
        "server_models": models,
        "platform": platform.platform(),
        "sw_vers_build": command("sw_vers", "-buildVersion"),
        "xcodebuild_version": command("xcodebuild", "-version"),
        "python": sys.version,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = make_tasks(30, args.task_seed)[: args.tasks]
    interrupts = [i for i in args.interrupts.split(",") if i]
    seeds: list[int | None] = (
        [int(s) for s in args.seeds.split(",")]
        if args.seed_param
        else [None] * args.repeats
    )
    record: dict[str, Any] = {"environment": environment(args), "args": vars(args)}
    probes: list[Probe] = []
    for task in tasks:
        for interrupt in interrupts:
            meta = await make_state(args, task, interrupt)
            for arm in ARMS:
                for repetition, seed in enumerate(seeds):
                    probe = await run_probe(
                        args, task, interrupt, meta, arm, repetition, seed
                    )
                    probes.append(probe)
                    print(
                        f"task {task.index:02d}{interrupt} {arm:6} #{repetition}: "
                        f"{'ok' if probe.success else 'no'} ({probe.status}, "
                        f"{probe.prompt_tokens} prompt tokens)",
                        flush=True,
                    )
    record["probes"] = [asdict(probe) for probe in probes]
    record["decision"] = decide(probes)
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else ""
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--server-version", default="", help="as the server reports it")
    parser.add_argument(
        "--work", default="resume-ab-work", help="a directory of its own"
    )
    parser.add_argument("--out", default="resume_ab_results.json")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--task-seed", type=int, default=20261026)
    parser.add_argument("--interrupts", default=",".join(INTERRUPTS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--seed-param", action="store_true", help="send each seed as 'seed'"
    )
    parser.add_argument("--turn-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--child", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if not args.model:
        parser.error("--model is required")
    if args.child:
        index, interrupt = args.child.split(":")
        task = make_tasks(30, args.task_seed)[int(index)]
        where = paths(Path(args.work), task, interrupt)
        asyncio.run(run_turns(args, task, where["live"], where["db"], interrupt))
        return
    args.work = str(Path(args.work).resolve())
    record = asyncio.run(run(args))
    Path(args.out).write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps(record["decision"], indent=2))


if __name__ == "__main__":
    main()
