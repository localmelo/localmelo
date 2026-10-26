"""The simulator's event loop: a workload replayed through the policy and a server.

``run()`` drives the same ``advance()``, ``dispatch()`` and
``apply_dispatch()`` the live gate will run, over a simulated clock in
integer µs from 0 (design 8.1; spec Requirement 47). Events at one
instant run in a fixed rank order and events pushed at the current
instant run in a further step, so a run is a pure function of its
inputs: the engine draws no random number and reads no clock, and the
sha256 of every decision line identifies a replay (acceptance d). Mode
``off`` sends every call straight to the server's queue and still books
the accounting, which is the baseline with no gate at all.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from localmelo.support.scheduler.accounting import advance, apply_dispatch
from localmelo.support.scheduler.costs import Rates, alone_service_s
from localmelo.support.scheduler.model import (
    Call,
    CallEnded,
    CallSubmitted,
    Clock,
    DepsSet,
    Dispatch,
    Event,
    Machine,
    MachineChanged,
    Outcome,
    SessionKind,
    State,
    Task,
    TaskAdded,
    TaskUpdated,
    Tick,
    Typing,
    is_holdable,
    is_lane,
    us,
)
from localmelo.support.scheduler.params import Params, params_for_arm
from localmelo.support.scheduler.policy import decide, decision_json, dispatch
from localmelo.support.scheduler.sim.server import Job, Server, ServerParams
from localmelo.support.scheduler.sim.trace import dumps
from localmelo.support.scheduler.sim.workloads import (
    FRONT_COMPLETION_TOKENS,
    FRONT_NEW_TOKENS,
    FRONT_PROMPT_TOKENS,
    TYPING_LEAD_S,
    Workload,
    WorkloadTask,
    alone_s,
)

# Ranks of the events at one instant (spec Requirement 47). Parks and
# expiries that a decision causes run as task events, in a further step.
END, MACHINE, USER, FRONT, TASK, SUBMIT, DEADLINE, TICK = range(8)

# Reasons that mean waiting only for capacity or order (decision D5).
_CAPACITY = (("Priority", ""), ("Resources", ""))

# Seconds between typing heartbeats before a front message.
_TYPING_EVERY_S = 10.0

Mode = Literal["off", "fifo", "class", "las2"]


@dataclass(frozen=True)
class TaskRecord:
    """One task's outcome; times in µs from mono 0, ``None`` when it never happened."""

    task_id: str
    gold_level: int
    initial_level: int
    origin: str
    kind: str
    arrival_us: int
    first_admit_us: int | None
    finish_us: int | None
    status: str
    deadline_wall_us: int | None
    service_s: float
    alone_s: float
    wait_own_s: float
    wait_total_s: float
    parks: int
    prompt_tokens: int
    cached_tokens: int
    reprefill_tokens: int


@dataclass(frozen=True)
class ReplyRecord:
    """One front turn: sent, first visible event and whole reply (``None``: not yet)."""

    send_us: int
    first_event_us: int | None
    end_us: int | None
    kind: str
    by_code: bool


@dataclass(frozen=True)
class SimResult:
    """A run's records and identity (spec Requirement 48); never validated in S1."""

    workload: str
    seed: int
    mode: str
    horizon_us: int
    wall_start_us: int
    tasks: tuple[TaskRecord, ...]
    replies: tuple[ReplyRecord, ...]
    decisions: int
    log_sha256: str
    log: tuple[str, ...]
    alarms: tuple[tuple[str, int], ...]
    busy_us: int
    yields: int
    foreign_lock_task_calls: int
    params_json: str
    validated: bool = False


@dataclass
class _TaskRun:
    spec: WorkloadTask
    next: int = 0  # the next template call to submit
    done: bool = False
    first_admit: int | None = None
    finish: int | None = None
    parks: int = 0
    prompt: int = 0
    cached: int = 0
    reprefill: int = 0
    wait_own_us: int = 0
    queue_us: int = 0  # mode off: time in the server's queue


@dataclass
class _Turn:
    send: int
    kind: str
    left: int  # lane calls still to submit
    first: int | None = None
    end: int | None = None
    by_code: bool = False


@dataclass
class _Live:
    """A call the engine still has to end: its job and whom it serves."""

    job: Job
    call: Call
    task: int | None = None
    turn: int | None = None
    failed: bool = False
    admitted: bool = False


def params_json(p: Params) -> str:
    """Canonical JSON of *p*; an infinite float is the string ``"inf"``."""
    values = {
        k: ("inf" if isinstance(v, float) and math.isinf(v) else v)
        for k, v in asdict(p).items()
    }
    return dumps(values)


class _Engine:
    def __init__(
        self, wl: Workload, mode: Mode, p: Params, sp: ServerParams, keep_log: bool
    ) -> None:
        self.wl, self.mode, self.p, self.keep_log = wl, mode, p, keep_log
        rates = sp.rates or Rates(p.prefill_tok_s, p.decode_tok_s)
        self.rates = rates
        self.server = Server(sp.n_srv or p.n_srv, rates, sp)
        self.heap: list[tuple[int, int, int, tuple[Any, ...]]] = []
        self.seq = 0
        self.state = State(clock=Clock(0, wl.wall_start_us))
        self.machine = Machine()
        self.away: int | None = None
        self.live: dict[str, _Live] = {}
        self.tasks = [_TaskRun(t) for t in wl.tasks]
        self.index = {t.task_id: i for i, t in enumerate(wl.tasks)}
        self.turns: list[_Turn] = []
        self.fronts_left = len(wl.front_messages)
        self.sha = hashlib.sha256()
        self.log: list[str] = []
        self.decisions = self.yields = self.foreign = 0
        self.submitted: list[_Live] = []  # calls submitted in the current step
        self.alarms: Counter[str] = Counter()

    def push(self, t: int, rank: int, payload: tuple[Any, ...]) -> None:
        heapq.heappush(self.heap, (t, rank, self.seq, payload))
        self.seq += 1

    def push_ends(self, ends: list[tuple[int, str, int]]) -> None:
        for t, cid, gen in ends:
            self.push(t, END, ("end", cid, gen))

    # ── Calls ──

    def send(self, now: int, lives: list[_Live]) -> None:
        """Start admitted calls on the server, in order."""
        for lv in lives:
            lv.admitted = True
            if lv.task is not None and self.machine.lock == "foreign":
                self.foreign += 1
        ends, started = self.server.submit(now, [lv.job for lv in lives])
        self.push_ends(ends)
        self.started(now, started)

    def started(self, now: int, jobs: list[Job]) -> None:
        """Mode off: book server-queue time, the only wait, and the first start."""
        if self.mode != "off":
            return
        for j in jobs:
            lv = self.live[j.call_id]
            if lv.task is not None:
                tr = self.tasks[lv.task]
                tr.queue_us += now - lv.call.submit_mono_us
                tr.first_admit = now if tr.first_admit is None else tr.first_admit

    def submit(self, now: int, lv: _Live, events: list[Event]) -> None:
        events.append(CallSubmitted(lv.call))
        self.live[lv.call.call_id] = lv
        self.submitted.append(lv)
        if self.mode == "off":
            self.send(now, [lv])

    def task_call(self, now: int, i: int, events: list[Event]) -> None:
        tr = self.tasks[i]
        calls = tr.spec.template.calls
        tc = calls[tr.next]
        prev = calls[tr.next - 1].prompt_tokens if tr.next else 0
        warm = tc.warm_cached_tokens
        warm = min(tc.prompt_tokens, prev) if warm is None else warm
        cid = f"{tr.spec.task_id}/{tr.next}"
        tid = tr.spec.task_id
        session: SessionKind = "chat" if tr.spec.kind == "chat" else "task"
        call = Call(cid, tid, session, tc.kind, tc.attempt, now, tc.prompt_tokens)
        job = Job(cid, tid, tc.prompt_tokens, warm, tc.completion_tokens)
        tr.next += 1
        self.submit(now, _Live(job, call, task=i, failed=tc.status == "failed"), events)

    def lane_call(self, now: int, ti: int, events: list[Event]) -> None:
        turn = self.turns[ti]
        cid = f"front-{ti}-{(2 if turn.kind == 'create' else 1) - turn.left}"
        turn.left -= 1
        p = FRONT_PROMPT_TOKENS
        call = Call(cid, None, "front", "plan", 1, now, p)
        job = Job(cid, "front", p, p - FRONT_NEW_TOKENS, FRONT_COMPLETION_TOKENS)
        self.push(now + us(self.p.lane_deadline_s), DEADLINE, ("deadline", cid))
        self.submit(now, _Live(job, call, turn=ti), events)

    def end(self, now: int, ends: list[tuple[str, int]], events: list[Event]) -> None:
        """End every call whose completion event is current (stale ones are skipped)."""
        jobs, pushed, started = self.server.end(now, ends)
        self.push_ends(pushed)
        self.started(now, started)
        for job in jobs:
            self.ended(now, job, events)

    def ended(self, now: int, job: Job, events: list[Event]) -> None:
        cid = job.call_id
        lv = self.live.pop(cid)
        charge = us(alone_service_s(job.prompt, job.h, job.completion, self.rates))
        outcome: Outcome = "failed" if lv.failed else "ok"
        events.append(CallEnded(cid, outcome, charge, job.prompt, job.completion))
        if lv.task is not None:
            tr = self.tasks[lv.task]
            tr.prompt += job.prompt
            tr.cached += job.h
            tr.reprefill += max(0, job.warm - job.h)
            calls = tr.spec.template.calls
            if not tr.done and tr.next < len(calls):
                gap = max(0.0, calls[tr.next].gap_before_s)
                self.push(now + us(gap), SUBMIT, ("submit", lv.task))
            elif not tr.done:
                self.finish(now, lv.task, "done", events)
        elif lv.turn is not None:
            turn = self.turns[lv.turn]
            if turn.end is None:
                turn.first = now if turn.first is None else turn.first
                if turn.left > 0:
                    self.push(now, SUBMIT, ("lane", lv.turn))
                else:
                    turn.end = now

    def finish(self, now: int, i: int, phase: str, events: list[Event]) -> None:
        tr = self.tasks[i]
        tr.done = True
        tr.finish = now if phase == "done" else None
        events.append(TaskUpdated(tr.spec.task_id, (("phase", phase),)))

    def answer(self, now: int, turn: _Turn) -> None:
        """End *turn* by code at *now* (G4: an answer by code counts as admitted)."""
        if turn.end is None:
            turn.first = now if turn.first is None else turn.first
            turn.end, turn.by_code = now, True

    # ── One step ──

    def step(self, now: int, batch: list[tuple[Any, ...]]) -> None:
        p, wl = self.p, self.wl
        events: list[Event] = []
        self.submitted = []
        if self.decisions == 0 and wl.deps:
            events.append(DepsSet(wl.deps))
        ends = [(pl[1], pl[2]) for pl in batch if pl[0] == "end"]
        if ends:  # rank 0: first, and together
            self.end(now, ends, events)
        for payload in batch:
            kind = payload[0]
            if kind == "end":
                continue
            elif kind in ("machine", "away"):
                if kind == "away":
                    self.away = now if payload[1] else None
                    m = self.machine
                else:
                    m = payload[1]
                self.machine = replace(m, away_since_mono_us=self.away)
                events.append(MachineChanged(self.machine))
            elif kind == "typing":
                events.append(Typing())
            elif kind == "front":
                self.fronts_left -= 1
                self.turns.append(
                    _Turn(now, payload[1], 2 if payload[1] == "create" else 1)
                )
                self.lane_call(now, len(self.turns) - 1, events)
            elif kind == "arrive":
                self.arrive(now, payload[1], events)
            elif kind == "submit":
                if not self.tasks[payload[1]].done:
                    self.task_call(now, payload[1], events)
            elif kind == "lane":
                if self.turns[payload[1]].end is None:
                    self.lane_call(now, payload[1], events)
            elif kind == "deadline":
                self.deadline(now, payload[1], events)
            elif kind == "park":
                self.park(now, payload[1], events)
            elif kind == "expire":
                self.expire(now, payload[1], events)
            elif kind == "tick":
                events.append(Tick())
                self.push(now + us(p.tick_s), TICK, ("tick",))
        clock = Clock(now, wl.wall_start_us + now)
        if self.mode != "off":
            self.own_wait(now)
        self.state = advance(self.state, clock, events, p)
        if self.mode == "off":
            d = decide(self.state, clock, p)
            sub = [lv.call for lv in self.submitted]
            decision = replace(
                d,
                admit=tuple(c.call_id for c in sub if is_holdable(c)),
                admit_unheld=tuple(
                    c.call_id for c in sub if not is_lane(c) and not is_holdable(c)
                ),
                floor=(),
                reasons=(),
                yields=(),
                parks=(),
                expire=(),
            )
            lane = tuple(c.call_id for c in sub if is_lane(c))
            disp = Dispatch(lane, (), decision)
        else:
            disp = dispatch(self.state, clock, p)
        self.state = apply_dispatch(self.state, clock, disp, p)
        if self.mode != "off":
            self.act(now, disp)
        line = decision_json(disp)
        self.sha.update(line.encode() + b"\n")
        if self.keep_log:
            self.log.append(line)
        self.decisions += 1
        self.yields += len(disp.decision.yields)
        self.alarms.update(a.split(":", 1)[0] for a in disp.decision.alarms)

    def arrive(self, now: int, i: int, events: list[Event]) -> None:
        spec = self.tasks[i].spec
        task = Task(
            spec.task_id,
            kind=spec.kind,
            origin=spec.origin,
            level=spec.level,
            level_source="user" if spec.level == 0 else "computed",
            deadline_wall_us=spec.deadline_wall_us,
            blocked_now=spec.blocked_now,
            size=spec.size,
            created_mono_us=now,
            created_wall_us=self.wl.wall_start_us + now,
        )
        events.append(TaskAdded(task))  # before any other event of the task
        if spec.template.calls:
            self.task_call(now, i, events)
        else:
            self.finish(now, i, "done", events)

    def deadline(self, now: int, cid: str, events: list[Event]) -> None:
        lv = self.live.get(cid)
        if lv is None or lv.turn is None:
            return  # the call ended
        self.answer(now, self.turns[lv.turn])
        if not lv.admitted:  # a held lane call is withdrawn; one in flight runs on
            del self.live[cid]
            events.append(CallEnded(cid, "aborted", 0))

    def park(self, now: int, i: int, events: list[Event]) -> None:
        tr = self.tasks[i]
        if not tr.done:
            tr.parks += 1
            self.server.evict(tr.spec.task_id)
            events.append(TaskUpdated(tr.spec.task_id, (("phase", "parked"),)))

    def expire(self, now: int, i: int, events: list[Event]) -> None:
        if self.tasks[i].done:
            return
        for cid, lv in sorted(self.live.items()):
            if lv.task == i and not lv.admitted:
                del self.live[cid]
                events.append(CallEnded(cid, "aborted", 0))
        self.finish(now, i, "expired", events)

    def act(self, now: int, disp: Dispatch) -> None:
        """Start admissions, answer the lane by code, and act on marks."""
        d = disp.decision
        self.send(
            now,
            [self.live[c] for c in disp.lane_admit + d.admit_unheld + d.admit],
        )
        for cid in d.admit_unheld + d.admit:
            i = self.live[cid].task
            if i is not None and self.tasks[i].first_admit is None:
                self.tasks[i].first_admit = now
        for cid, _ in disp.lane_answer:
            lv = self.live.pop(cid)
            if lv.turn is not None:
                self.answer(now, self.turns[lv.turn])
        busy = {c.task_id for c in self.state.calls if c.admitted_mono_us is not None}
        phase = {t.task_id: t.phase for t in self.state.tasks}
        for tid, _cause in d.parks:
            if phase.get(tid) == "held" and tid not in busy and tid in self.index:
                self.push(now, TASK, ("park", self.index[tid]))
        for tid in d.expire:
            if tid in self.index:
                self.push(now, TASK, ("expire", self.index[tid]))

    def own_wait(self, now: int) -> None:
        """Book own-class waits for the starvation metric (decision D30)."""
        st = self.state
        dt = min(max(0, now - st.clock.mono_us), 2 * us(self.p.tick_s))
        if dt == 0:
            return
        pending = {
            c.task_id for c in st.calls if c.admitted_mono_us is None and is_holdable(c)
        }
        inflight = {c.task_id for c in st.calls if c.admitted_mono_us is not None}
        level = {
            t.task_id: t.level if t.level_eff is None else t.level_eff for t in st.tasks
        }
        waiting = [
            t.task_id
            for t in st.tasks
            if t.task_id in pending and (t.last_reason, t.last_detail) in _CAPACITY
        ]
        busy = [level[t] for t in waiting] + [level[t] for t in inflight if t in level]
        for tid in waiting:
            if not any(b < level[tid] for b in busy) and tid in self.index:
                self.tasks[self.index[tid]].wait_own_us += dt

    # ── The loop ──

    def done(self) -> bool:
        return (
            all(tr.done for tr in self.tasks)
            and not self.live
            and self.fronts_left == 0
            and all(t.end is not None for t in self.turns)
        )

    def run(self) -> SimResult:
        wl = self.wl
        for i, t in enumerate(wl.tasks):
            self.push(t.arrival_us, TASK, ("arrive", i))
        for at, kind in wl.front_messages:
            self.push(at, FRONT, ("front", kind))
            k = 0
            while (t_us := at - us(TYPING_LEAD_S) + k * us(_TYPING_EVERY_S)) < at:
                if t_us >= 0:
                    self.push(t_us, USER, ("typing",))
                k += 1
        for start, stop in wl.presence:
            self.push(start, USER, ("away", True))
            self.push(stop, USER, ("away", False))
        for at, m in wl.machine_events:
            self.push(at, MACHINE, ("machine", m))
        self.push(0, TICK, ("tick",))
        last = 0
        while self.heap and self.heap[0][0] < wl.horizon_us and not self.done():
            now = last = self.heap[0][0]
            batch = []
            while self.heap and self.heap[0][0] == now:
                batch.append(heapq.heappop(self.heap)[3])
            self.step(now, batch)
        self.server.progress(wl.horizon_us if not self.done() else last)
        return self.result()

    def result(self) -> SimResult:
        by_id = {t.task_id: t for t in self.state.tasks}
        records = []
        for tr in self.tasks:
            s = tr.spec
            t = by_id.get(s.task_id)
            gaps = sum(max(0.0, c.gap_before_s) for c in s.template.calls[1:])
            own, total = tr.wait_own_us, t.wait_total_us if t else 0
            if self.mode == "off":
                own = total = tr.queue_us
            records.append(
                TaskRecord(
                    s.task_id,
                    s.gold_level,
                    s.level,
                    s.origin,
                    s.kind,
                    s.arrival_us,
                    tr.first_admit,
                    tr.finish,
                    t.phase if t else "absent",
                    s.deadline_wall_us,
                    t.service_us / 1e6 if t else 0.0,
                    alone_s(s.template, self.rates) + gaps,
                    own / 1e6,
                    total / 1e6,
                    tr.parks,
                    tr.prompt,
                    tr.cached,
                    tr.reprefill,
                )
            )
        return SimResult(
            self.wl.name,
            self.wl.seed,
            self.mode,
            self.wl.horizon_us,
            self.wl.wall_start_us,
            tuple(records),
            tuple(
                ReplyRecord(t.send, t.first, t.end, t.kind, t.by_code)
                for t in self.turns
            ),
            self.decisions,
            self.sha.hexdigest(),
            tuple(self.log),
            tuple(sorted(self.alarms.items())),
            self.server.busy_us,
            self.yields,
            self.foreign,
            params_json(self.p),
        )


def run(
    workload: Workload,
    mode: Mode,
    params: Params | None = None,
    server: ServerParams | None = None,
    *,
    keep_log: bool = False,
) -> SimResult:
    """Replay *workload* under *mode* (spec Requirement 47).

    *params* defaults to the arm's preset (``fifo`` for ``off``); *server*
    to ``ServerParams()``. Deterministic: equal inputs give an equal
    result, ``log_sha256`` included.
    """
    if mode not in ("off", "fifo", "class", "las2"):
        raise ValueError(f"mode: {mode!r} is not off, fifo, class or las2")
    p = params or params_for_arm("fifo" if mode == "off" else mode)
    return _Engine(workload, mode, p, server or ServerParams(), keep_log).run()
