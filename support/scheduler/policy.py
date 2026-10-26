"""The scheduling decision: which pending calls go to the server, and why not.

``decide()`` is the procedure of design 3.3 as a pure, total function of
the state: gates, levels, the lane's reserve, the background cap, the
memory check, a strict class fill in the arm's order with credit floors,
then yield and park marks and one reason per task. ``dispatch()`` is the
only entry the caller uses: it admits the lane before and independently
of ``decide()`` (design 3.3, the lane's fast path), and falls back to
first-come order, then to a last-resort decision, when code raises, so a
bug can neither close the lane nor open the machine gates (decision D10).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, replace

from localmelo.support.scheduler.costs import (
    Rates,
    chi_hat_s,
    delta_s,
    others_since,
    predict_cached,
)
from localmelo.support.scheduler.levels import Levels, compute_levels, unmet_start_after
from localmelo.support.scheduler.machine_gates import Gates, evaluate_gates, hard_gate
from localmelo.support.scheduler.model import (
    CHI_HAT_FLOOR_US,
    MAX_INT,
    STARTING_PHASES,
    TERMINAL_PHASES,
    Call,
    Clock,
    Decision,
    Dispatch,
    Machine,
    State,
    Task,
    clean_state,
    is_holdable,
    is_lane,
    is_planning,
    is_retry,
    is_terminal,
    root_of,
    us,
)
from localmelo.support.scheduler.params import Params

# Estimates in seconds are clamped to ±this, so every product stays finite.
_MAX_EST_S = 1e15

# Phases of a started, resident task.
_STARTED = ("running", "held")

# A key, as returned by _order_keys(): ends in a unique id; never NaN.
Key = tuple[object, ...]


def _us_capped(seconds: float) -> int:
    """``us()`` of *seconds* clamped into 0..MAX_INT µs; inf and NaN never raise."""
    if not seconds >= 0:
        return 0
    return us(min(seconds, MAX_INT / 1e6))


def _inf(v: float | None) -> float:
    return math.inf if v is None else v


def _est(v: float) -> float:
    return min(max(v, -_MAX_EST_S), _MAX_EST_S)


def _limit(p: Params) -> tuple[int, bool]:
    """The non-lane limit and whether the reserve was clamped (design 3.3, step 3)."""
    r_f_eff = 0 if p.n_srv <= 2 else p.r_f
    return max(1, p.n_srv - 1 - r_f_eff), p.n_srv - 1 - r_f_eff < 1


def _stored_levels(st: State) -> Levels:
    """Levels without inheritance or aging, for the fallback: out of range as 2, a non-user 0 as 1."""
    base = {
        t.task_id: (1 if t.level == 0 and t.level_source != "user" else t.level)
        for t in st.tasks
        if not is_terminal(t.phase)
    }
    none: dict[str, int | None] = {k: None for k in base}
    return Levels(
        base,
        base,
        base,
        none,
        frozenset(),
        (),
        frozenset(),
        unmet_start_after(st),
        (),
        (),
    )


def _kv(c: Call, by_id: dict[str, Task], p: Params) -> float:
    """KV bytes a call may hold: kv_bytes_per_token · (P + D99) (design 3.3, step 5)."""
    t = by_id.get(c.task_id or "")
    if c.prompt_tokens is not None:
        prompt = c.prompt_tokens
    elif t is not None and t.last_prompt_tokens is not None:
        prompt = t.last_prompt_tokens
    else:
        prompt = p.prompt_prior_tokens
    d99 = (
        p.completion_p99_tokens
        if c.completion_p99_tokens is None
        else c.completion_p99_tokens
    )
    return p.kv_bytes_per_token * (prompt + d99)


def _mem_cap(m: Machine, p: Params) -> float:
    """M_kv minus the memory a lock declares when ``lock_lines`` is on."""
    declared = max(0.0, m.lock_mem_gib) * 2**30 if p.lock_lines else 0.0
    return p.m_kv_bytes - declared


# γ per arm (design 3.4): LAS-2's two queues; S6b adds gittins and cmu here.
_GAMMA: dict[str, Callable[[Task, Params], float]] = {
    "fifo": lambda root, p: 1.0,
    "class": lambda root, p: 1.0,
    "las2": lambda root, p: 2.0 if root.promo_service_us < us(p.las_q_s) else 1.0,
}


# ── One decision ──


class _Pass:
    """The working values of one decision over a cleaned copy of the state."""

    def __init__(self, state: State, now: Clock, p: Params, fallback: bool) -> None:
        self.p, self.now, self.fallback = p, now, fallback
        self.arm = "fifo" if fallback else p.arm
        self.st, cleaned = clean_state(state)
        self.alarms: list[str] = list(cleaned)
        self.by_id = {t.task_id: t for t in self.st.tasks}
        self.sched = [t for t in self.st.tasks if not is_terminal(t.phase)]
        self.lv = (
            _stored_levels(self.st) if fallback else compute_levels(self.st, now, p)
        )
        self.eff = self.lv.eff
        self.asks: list[tuple[str, str]] = list(self.lv.asks)
        self.alarms += self.lv.alarms
        self.rates = Rates(p.prefill_tok_s, p.decode_tok_s)
        self.limit, clamped = _limit(p)
        if clamped:
            self.alarms.append("reserve_clamped")
        calls = self.st.calls
        self.inflight = [c for c in calls if c.admitted_mono_us is not None]
        self.pending = [c for c in calls if c.admitted_mono_us is None]
        self.lane_tasks = {c.task_id for c in calls if is_lane(c) and c.task_id}
        busy = {c.task_id for c in self.inflight if c.task_id}
        self.planning: dict[str, list[Call]] = {}
        for c in self.pending:
            if is_planning(c) and c.task_id:
                self.planning.setdefault(c.task_id, []).append(c)
        for c in calls:  # a taskless never-held call is legitimate, not an orphan
            unknown = c.task_id not in self.by_id
            if not is_lane(c) and unknown and (c.task_id or is_planning(c)):
                self.alarms.append(f"orphan_call:{c.call_id}")
        self.active = {
            t.task_id
            for t in self.sched
            if t.task_id in busy
            or t.task_id in self.lane_tasks
            or (t.phase == "running" and t.task_id not in self.planning)
        }
        self.mem_cap = _mem_cap(self.st.machine, p)
        self.mem_used = 0.0
        for c in self.inflight:
            self.mem_used += self.kv(c)
        self.keys: dict[str, Key] = {}
        self.scores: dict[str, float] = {}

    # ── Per-task values (Requirement 37) ──

    def prompt(self, t: Task, c: Call | None) -> int:
        """P: the call's prompt, else the task's last prompt, else the prior."""
        if c is None:
            c = next((x for x in self.inflight if x.task_id == t.task_id), None)
        if c is not None and c.prompt_tokens is not None:
            return c.prompt_tokens
        if t.last_prompt_tokens is not None:
            return t.last_prompt_tokens
        return self.p.prompt_prior_tokens

    def kv(self, c: Call) -> float:
        return _kv(c, self.by_id, self.p)

    def fits(self, c: Call) -> bool:
        return self.mem_used + self.kv(c) <= self.mem_cap

    def c_hat(self, t: Task) -> float:
        """Ĉ: the estimate, else the size prior minus the task's own service."""
        if t.remaining_s_est is not None:
            return _est(t.remaining_s_est)
        idx = {"small": 0, "large": 2}.get(t.size, 1)
        return max(self.p.s0_s, self.p.remaining_prior_s[idx] - t.service_us / 1e6)

    def cached_hat(self, t: Task, prompt: int) -> int:
        """Ĥ from the task's own last call: the cache is per session (decision D50)."""
        others = others_since(self.st.recent_admits, t.task_id)
        return predict_cached(
            prompt,
            t.last_prompt_tokens,
            t.last_call_end_mono_us,
            self.now.mono_us,
            others,
            self.p,
        )

    def sigma(self, t: Task) -> float | None:
        """Slack in seconds, for a task with a deadline that is not late."""
        d = self.lv.d_eff.get(t.task_id)
        if d is None or t.task_id in self.lv.late:
            return None
        p, lvl = self.p, self.eff[t.task_id]
        share = (
            float(self.limit)
            if lvl <= 1
            else (p.phi2 if lvl == 2 else p.phi3) * self.limit
        )
        share = share if lvl <= 1 else max(share, 1.0)
        rate = max(0.01, min(1.0, share / max(1, self.n_class.get(lvl, 0))))
        need = self.c_hat(t) / rate + _est(t.tools_left_s_est or 0.0)
        return (d - self.now.wall_us) / 1e6 - p.kappa_s * need

    def score(self, t: Task, c: Call | None) -> float:
        """u · b · g (design 3.4)."""
        if t.task_id in self.scores:
            return self.scores[t.task_id]
        p, now = self.p, self.now
        sig = self.sigma(t)
        if sig is None or sig >= p.h_u_s:
            u = 1.0
        elif sig <= 0:
            u = p.u_max
        else:
            u = 1 + (p.u_max - 1) * (1 - sig / p.h_u_s)
        b = 1.0
        if t.mention_mono_us is not None and t.mention_mono_us <= now.mono_us:
            age = (now.mono_us - t.mention_mono_us) / max(1, us(p.tau_f_s))
            b = 1 + p.focus_boost_beta * math.exp(-age)
        root = root_of(t, self.by_id)
        s_bar = p.s0_s
        if root.calls_done > 0:
            s_bar = max(p.s0_s, root.service_us / 1e6 / root.calls_done)
        prompt = self.prompt(t, c)
        dlt = delta_s(prompt, self.cached_hat(t, prompt), self.rates)
        g = _GAMMA[self.arm](root, p) * s_bar / (s_bar + dlt)
        self.scores[t.task_id] = u * b * g
        return self.scores[t.task_id]

    def key(self, t: Task, c: Call | None) -> Key:
        """The arm's key (design 3.4; spec Requirement 44); smallest first."""
        if t.task_id in self.keys:
            return self.keys[t.task_id]
        tid, lvl = t.task_id, self.eff[t.task_id]
        rank, start = _inf(t.user_rank), _inf(t.started_mono_us)
        key: Key
        if self.arm == "fifo":
            key = (c.submit_mono_us, c.call_id, tid) if c else (math.inf, "", tid)
        elif self.arm == "class" or lvl == 0:
            key = (rank, start, t.created_mono_us, tid)
        elif lvl == 1:
            d = _inf(self.lv.d_eff.get(tid))
            key = (rank, tid in self.lv.late, d, start, t.created_mono_us, tid)
        elif lvl in (2, 3):
            key = (rank, 1, -self.score(t, c), start, t.created_mono_us, tid)
        else:
            key = (rank, t.created_mono_us, tid)
        self.keys[tid] = key
        return key

    # ── Blocks (Requirement 27) ──

    def own_block(self, t: Task, record: bool) -> tuple[str, str] | None:
        """Items 1-6: the task's own state, which no machine change can clear."""
        if self.lv.base[t.task_id] == 5:
            return ("Paused", "later")
        if t.paused:
            return ("Paused", "")
        if t.phase == "interrupted":
            return ("Input", "interrupted")
        if t.phase == "waiting":
            return {"input": ("Input", ""), "approval": ("Approval", "")}.get(
                t.wait_reason, ("Dependency", "")
            )
        holders = self.lv.unmet.get(t.task_id, ())
        if holders:
            for h in holders if record else ():
                ht = self.by_id.get(h)
                if ht is None:
                    self.alarms.append(f"unknown_holder:{h}")
                elif ht.phase in TERMINAL_PHASES:
                    self.asks.append((t.task_id, "dep_failed"))
                elif self.lv.base.get(h) == 5:
                    self.asks.append((t.task_id, "dep_later"))
            return ("Dependency", "")
        if t.task_id in self.lv.cycle:  # waits on itself (decision D16)
            return ("Dependency", "")
        if t.budget_exhausted:
            return ("Budget", "")
        return None

    def hot(self, t: Task) -> float:
        """The KV bytes a resident task keeps hot: its last prompt, else the prior."""
        tokens = t.last_prompt_tokens
        prompt = self.p.prompt_prior_tokens if tokens is None else tokens
        return self.p.kv_bytes_per_token * prompt

    def large(self, t: Task) -> bool:
        return t.size == "large" or self.c_hat(t) > self.p.battery_large_s

    # ── The procedure ──

    def prepare(self, g: Gates) -> None:
        """Never-held calls and retries, blocks, candidates and the arm's order."""
        p, eff = self.p, self.eff
        self.unheld: list[str] = []
        self.done_now: set[str] = set()  # tasks with a call admitted now
        self.retry_hold: dict[str, tuple[str, str]] = {}
        self.inflight_nonlane = sum(1 for c in self.inflight if not is_lane(c))
        self.bg = sum(
            1
            for c in self.inflight
            if not is_lane(c) and (c.task_id not in eff or eff[c.task_id] >= 2)
        )
        hard = hard_gate(self.st.machine)
        for c in self.pending:
            t = self.by_id.get(c.task_id or "")
            if is_lane(c) or is_holdable(c):
                continue
            if t is None and c.task_id is not None:
                continue  # an orphan: its task is unknown
            if t is not None and is_terminal(t.phase):
                continue
            if is_retry(c):
                hold = hard or (None if self.fits(c) else ("Memory", "kv"))
                if hold is not None:
                    if t is not None:
                        self.retry_hold.setdefault(t.task_id, hold)
                    continue
            self.unheld.append(c.call_id)
            self.take(t, c)
        self.blocks: dict[str, tuple[str, str]] = {}
        self.cands: list[tuple[Task, Call]] = []
        for t in self.sched:
            if t.task_id in self.active:
                continue
            starting = t.phase in STARTING_PHASES
            blk = self.own_block(t, True) or g.block(
                eff[t.task_id], starting, self.large(t)
            )
            if blk is not None:
                self.blocks[t.task_id] = blk
                continue
            mine = sorted(
                (c for c in self.planning.get(t.task_id, ()) if is_holdable(c)),
                key=lambda c: (c.submit_mono_us, c.call_id),
            )
            if len(mine) > 1:
                self.alarms.append(f"extra_pending_call:{t.task_id}")
            if mine:
                self.cands.append((t, mine[0]))
        counted = {t.task_id for t, _ in self.cands} | self.active | self.done_now
        self.n_class: dict[int, int] = {}
        for t in self.sched:
            if t.task_id in counted:
                lvl = eff[t.task_id]
                self.n_class[lvl] = self.n_class.get(lvl, 0) + 1
        if self.arm == "fifo":
            self.cands.sort(key=lambda tc: self.key(*tc))
        else:
            self.cands.sort(key=lambda tc: (eff[tc[0].task_id], self.key(*tc)))
        self.running = [
            t
            for t in self.sched
            if t.phase in _STARTED
            and t.task_id not in self.lane_tasks
            and self.own_block(t, False) is None
        ]
        lane_busy = any(is_lane(c) for c in self.inflight)
        recent = [
            x
            for x in (self.st.last_typing_mono_us, self.st.last_front_end_mono_us)
            if x is not None and self.now.mono_us - x <= us(p.tau_focus_s)
        ]
        self.focus = lane_busy or bool(recent)

    def take(self, t: Task | None, c: Call) -> None:
        """Count an admission: slots, the background cap and memory."""
        self.inflight_nonlane += 1
        if t is None or self.eff.get(t.task_id, 2) >= 2:
            self.bg += 1
        self.mem_used += self.kv(c)
        if t is not None:
            self.done_now.add(t.task_id)

    def floors(self, g: Gates) -> list[tuple[Task, Call, int]]:
        """χ̂, runnable classes and floor admissions (design 3.3, floors)."""
        p, eff = self.p, self.eff
        best: dict[int, tuple[Task, Call]] = {}
        for t, c in self.cands:
            best.setdefault(eff[t.task_id], (t, c))
        self.chi = [CHI_HAT_FLOOR_US, CHI_HAT_FLOOR_US]
        self.runnable = [2 in best, 3 in best]
        self.open = [False, False]
        self.best = best
        moves: list[tuple[Task, Call, int]] = []
        for i, cls in enumerate((2, 3)):
            if cls not in best:
                continue
            t, c = best[cls]
            root = root_of(t, self.by_id)
            d_hat = float(p.completion_prior_tokens)
            if root.calls_done > 0:
                d_hat = root.completion_tokens_done / root.calls_done
            prompt = self.prompt(t, c)
            est = chi_hat_s(prompt, self.cached_hat(t, prompt), d_hat, self.rates)
            self.chi[i] = max(CHI_HAT_FLOOR_US, _us_capped(est))
        if not p.floors or self.fallback:
            return moves
        urgent = any(
            eff[t.task_id] == 1 and (s := self.sigma(t)) is not None and s < p.h_u_s
            for t, _ in self.cands
        )
        p0_cand = any(eff[t.task_id] == 0 for t, _ in self.cands)
        for i, cls in enumerate((2, 3)):
            if cls not in best:
                continue
            t, c = best[cls]
            ok = g.block(cls, False, False) is None and self.fits(c) and not urgent
            waited = any(
                eff[x.task_id] == cls and x.wait_held_us >= us(p.w_floor_s)
                for x, _ in self.cands
            )
            self.open[i] = ok and waited
            busy = any(x.floor_class == cls for x in self.inflight)
            credit = self.st.credit_us[i]
            if ok and credit >= self.chi[i] and not busy and not p0_cand:
                moves.append((t, c, cls))
        return moves

    def worst(self, pool: list[Task], held: set[str]) -> Task | None:
        """The task of *pool* not in *held* first, then the highest level, then the largest key."""

        def rank(x: Task) -> tuple[bool, int, Key]:
            cand = next((c for t, c in self.cands if t.task_id == x.task_id), None)
            return (x.task_id not in held, self.eff[x.task_id], self.key(x, cand))

        return max(pool, key=rank, default=None)

    def run(self, g: Gates) -> Decision:
        p, eff = self.p, self.eff
        self.prepare(g)
        moves = self.floors(g)
        moved = {t.task_id for t, _, _ in moves}
        order = [(t, c, cls) for t, c, cls in moves]
        order += [(t, c, 0) for t, c in self.cands if t.task_id not in moved]
        marks = p.yield_marks and not self.fallback
        ws_mark = ""
        room = p.m_hot_bytes - p.m_front_bytes
        hot = sum(self.hot(t) for t in self.running)
        if not self.fallback and self.running and hot > room:
            top = self.worst(self.running, set())
            ws_mark = top.task_id if top else ""
        cap_bg = p.n_fg if self.focus else p.n_a
        n_running = len(self.running)
        admit: list[str] = []
        floor: list[tuple[str, int]] = []
        why: dict[str, tuple[str, str]] = {}
        claimed: list[str] = []
        at_limit: list[Task] = []
        for t, c, cls in order:
            tid, lvl = t.task_id, eff[t.task_id]
            starting = t.phase in STARTING_PHASES
            cause: tuple[str, str] | None = None
            if self.inflight_nonlane >= self.limit:
                cause = ("capacity", "")
                if lvl <= 1:
                    at_limit.append(t)
            elif lvl >= 2 and self.bg >= cap_bg:
                cause = ("capacity", "")
            elif starting and n_running >= p.w_max:
                cause = ("capacity", "")
                if lvl <= 1 and marks:
                    pool = [
                        x
                        for x in self.running
                        if x.task_id not in claimed and eff[x.task_id] > lvl
                    ]
                    top = self.worst(pool, self.active | self.done_now)
                    if top is not None:
                        claimed.append(top.task_id)
            elif (
                starting
                and n_running > 0
                and not self.fallback
                and hot + self.hot(t) > room
            ):  # a start would push the working set over, then park (step 8)
                cause = ("Memory", "working_set")
            elif not self.fits(c):
                cause = ("Memory", "kv")
            elif tid in claimed:
                cause = ("Priority", "")
            elif tid == ws_mark:
                cause = ("Memory", "kv")
            if cause is None:
                admit.append(c.call_id)
                self.take(t, c)
                if starting:
                    n_running += 1
                    hot += self.hot(t)
                if cls:
                    floor.append((c.call_id, cls))
                continue
            if cause[0] == "capacity":
                cause = ("Priority", "") if why else ("Resources", "")
            why[tid] = cause
        active_all = self.active | self.done_now
        yields: list[str] = []
        for w in at_limit if marks else []:
            pool = [
                x
                for x in self.sched
                if x.task_id in active_all
                and x.task_id not in yields
                and eff[x.task_id] > eff[w.task_id]
            ]
            pool.sort(key=lambda x: (eff[x.task_id], self.key(x, None)), reverse=True)
            for x in pool:
                xl = eff[x.task_id]
                top_c = self.best.get(xl)
                if (
                    self.arm == "las2"
                    and xl in (2, 3)
                    and top_c is not None
                    and self.score(x, None) * p.h_y > self.score(*top_c)
                ):
                    continue
                yields.append(x.task_id)
                break
        parks: dict[str, str] = {}
        expire: list[str] = []
        if not self.fallback:
            pressure = self.st.machine.memory_pressure
            for t in self.sched if pressure != "normal" else []:
                if t.phase in _STARTED and eff[t.task_id] >= (
                    3 if pressure == "warning" else 2
                ):
                    parks[t.task_id] = "memory_pressure"
            for tid in claimed:
                parks.setdefault(tid, "resident_slot")
            if ws_mark:
                parks.setdefault(ws_mark, "working_set")
            for t, _ in self.cands:
                if (
                    t.phase in _STARTED
                    and t.task_id not in self.done_now
                    and t.wait_held_us >= us(p.t_park_s)
                ):
                    parks.setdefault(t.task_id, "t_park")
            for t in self.sched:
                if t.task_id in active_all or eff[t.task_id] != 4:
                    continue  # running now, or needed by a waiter (inherited)
                last = t.last_activity_wall_us
                idle_us = self.now.wall_us - (
                    t.created_wall_us if last is None else last
                )
                if self.lv.base[t.task_id] == 4 and idle_us >= us(p.p4_expiry_s):
                    expire.append(t.task_id)
        reasons: list[tuple[str, str, str]] = []
        for t in self.sched:
            tid = t.task_id
            if tid in active_all:
                continue
            r = self.blocks.get(tid) or why.get(tid) or self.retry_hold.get(tid)
            if r is None:
                r = ("Priority", "")
                self.alarms.append(f"no_pending_call:{tid}")
            reasons.append((tid, r[0], r[1]))
        return Decision(
            admit=tuple(admit),
            admit_unheld=tuple(self.unheld),
            floor=tuple(floor),
            reasons=tuple(reasons),
            levels=tuple((t.task_id, eff[t.task_id]) for t in self.sched),
            aged=self.lv.aged,
            yields=tuple(yields),
            parks=tuple(sorted(parks.items())),
            expire=tuple(expire),
            asks=tuple(sorted(dict.fromkeys(self.asks))),
            floor_open=(self.open[0], self.open[1]),
            class_runnable=(self.runnable[0], self.runnable[1]),
            chi_hat_us=(self.chi[0], self.chi[1]),
            battery_band=g.battery_band,
            idle_window=g.idle,
            focus=self.focus,
            limit_nonlane=self.limit,
            alarms=tuple(sorted(dict.fromkeys(self.alarms))),
            fallback=self.fallback,
        )

    def closed(self, error: str) -> Decision:
        """The fallback when the gates raise: never-held calls only (fail closed)."""
        unheld: list[str] = []
        for c in self.pending:
            t = self.by_id.get(c.task_id or "")
            if is_planning(c) or is_lane(c):
                continue
            if t is None and c.task_id is None:
                unheld.append(c.call_id)
            elif t is not None and not is_terminal(t.phase):
                unheld.append(c.call_id)
                self.active.add(t.task_id)
        reasons = tuple(
            (t.task_id, "Resources", "fallback")
            for t in self.sched
            if t.task_id not in self.active
        )
        alarms = [*self.alarms, f"decide_failed:{error}", "gates_failed"]
        return Decision(
            admit_unheld=tuple(unheld),
            reasons=reasons,
            levels=tuple((t.task_id, self.eff[t.task_id]) for t in self.sched),
            battery_band=self.st.battery_band,
            limit_nonlane=self.limit,
            alarms=tuple(sorted(dict.fromkeys(alarms))),
            fallback=True,
        )


# ── Public entries ──


def decide(state: State, now: Clock, params: Params) -> Decision:
    """The decision for every task call; pure, total and deterministic (spec Requirements 22-37)."""
    ps = _Pass(state, now, params, fallback=False)
    st = ps.st
    return ps.run(
        evaluate_gates(
            st.machine, st.battery_band, st.busy_above_p4_mono_us, now, params
        )
    )


def _order_keys(
    state: State, now: Clock, params: Params
) -> tuple[tuple[str, Key], ...]:
    """Each candidate's task id and key, in the arm's order before floors."""
    ps = _Pass(state, now, params, fallback=False)
    st = ps.st
    ps.prepare(
        evaluate_gates(
            st.machine, st.battery_band, st.busy_above_p4_mono_us, now, params
        )
    )
    return tuple((t.task_id, ps.key(t, c)) for t, c in ps.cands)


def fallback_fcfs(state: State, now: Clock, params: Params, error: str) -> Decision:
    """First-come order with every gate, cap and block but no inheritance (decision D10).

    *error* is the class name of what ``decide()`` raised, never its
    message. If the gates raise too, task calls fail closed.
    """
    ps = _Pass(state, now, params, fallback=True)
    st = ps.st
    try:
        gates = evaluate_gates(
            st.machine, st.battery_band, st.busy_above_p4_mono_us, now, params
        )
    except Exception:
        return ps.closed(error)
    d = ps.run(gates)
    return replace(d, alarms=tuple(sorted({*d.alarms, f"decide_failed:{error}"})))


def admit_lane(
    state: State, now: Clock, params: Params
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """The lane's oldest pending call: admitted, answered by code, or left waiting.

    Only the slot limit, the foreign lock under ``reservation_policy``, a
    down server or unloaded model, and the memory check apply (design
    3.10, G2 and G4); battery, thermal and pressure never gate the lane.
    """
    st, _ = clean_state(state)
    lane = [c for c in st.calls if is_lane(c)]
    if any(c.admitted_mono_us is not None for c in lane):
        return (), ()
    waiting = sorted(
        (c for c in lane if c.admitted_mono_us is None),
        key=lambda c: (c.submit_mono_us, c.call_id),
    )
    if not waiting:
        return (), ()
    c, m = waiting[0], st.machine
    if m.lock == "foreign":
        if params.reservation_policy == "ask":
            return (), ((c.call_id, "reservation"),)
        if params.reservation_policy == "respect":
            return (), ()
    if not m.server_up:
        return (), ((c.call_id, "server_down"),)
    if m.model != "loaded":
        return (), ((c.call_id, "model_not_loaded"),)
    if sum(1 for x in st.calls if x.admitted_mono_us is not None) >= params.n_srv:
        return (), ()
    by_id = {t.task_id: t for t in st.tasks}
    used = 0.0
    for x in st.calls:
        if x.admitted_mono_us is not None:
            used += _kv(x, by_id, params)
    if used + _kv(c, by_id, params) > _mem_cap(m, params):
        return (), ((c.call_id, "memory"),)
    return (c.call_id,), ()


def _lane_by_count(state: State, params: Params) -> tuple[str, ...]:
    """The oldest pending lane call when a slot is free, from counts and ids only."""
    lane = [c for c in state.calls if c.session_kind == "front"]
    if any(c.admitted_mono_us is not None for c in lane):
        return ()
    inflight = sum(1 for c in state.calls if c.admitted_mono_us is not None)
    waiting = sorted(
        (c.submit_mono_us if isinstance(c.submit_mono_us, int) else 0, c.call_id)
        for c in lane
        if c.admitted_mono_us is None
    )
    return (waiting[0][1],) if waiting and inflight < params.n_srv else ()


def _with_lane(
    state: State,
    now: Clock,
    admit: tuple[str, ...],
    answer: tuple[tuple[str, str], ...],
) -> State:
    """*state* with the admitted lane call in flight and answered ones removed."""
    gone = {cid for cid, _ in answer}
    calls = tuple(
        replace(c, admitted_mono_us=now.mono_us) if c.call_id in admit else c
        for c in state.calls
        if c.call_id not in gone
    )
    return replace(state, calls=calls)


def _last_resort(
    state: State, p: Params, exc: BaseException, exc2: BaseException
) -> Decision:
    """The decision when the fallback raises too: nothing but the lane is admitted."""
    tasks: dict[str, Task] = {}
    for t in state.tasks:
        tasks.setdefault(t.task_id, t)
    busy = {c.task_id for c in state.calls if c.admitted_mono_us is not None}
    lane = {c.task_id for c in state.calls if c.session_kind == "front"}
    planning = {
        c.task_id
        for c in state.calls
        if c.admitted_mono_us is None
        and c.session_kind != "front"
        and c.kind in ("plan", "reflect")
    }
    levels, reasons = [], []
    for tid in sorted(tasks):
        t = tasks[tid]
        if t.phase in TERMINAL_PHASES:
            continue
        lvl = t.level if isinstance(t.level, int) and 0 <= t.level <= 5 else 2
        levels.append((tid, 1 if lvl == 0 and t.level_source != "user" else lvl))
        if not (
            tid in busy or tid in lane or (t.phase == "running" and tid not in planning)
        ):
            reasons.append((tid, "Resources", "fallback"))
    alarms = sorted(
        {
            f"decide_failed:{type(exc).__name__}",
            f"fallback_failed:{type(exc2).__name__}",
        }
    )
    return Decision(
        levels=tuple(levels),
        reasons=tuple(reasons),
        battery_band=state.battery_band,
        limit_nonlane=_limit(p)[0],
        alarms=tuple(alarms),
        fallback=True,
    )


def dispatch(
    state: State,
    now: Clock,
    params: Params,
    *,
    decide_fn: Callable[[State, Clock, Params], Decision] = decide,
) -> Dispatch:
    """The lane first, then the decision for task calls (spec Requirement 38).

    The lane is admitted whatever ``decide_fn`` does, raising included.
    """
    by_count = False
    try:
        lane_admit, lane_answer = admit_lane(state, now, params)
    except Exception:  # the lane must never depend on other code
        lane_admit, lane_answer, by_count = _lane_by_count(state, params), (), True
    view = _with_lane(state, now, lane_admit, lane_answer)
    try:
        decision = decide_fn(view, now, params)
    except Exception as exc:
        try:
            decision = fallback_fcfs(view, now, params, type(exc).__name__)
        except Exception as exc2:
            try:
                decision = _last_resort(view, params, exc, exc2)
            except Exception:  # nothing may keep the lane's answer from the caller
                decision = Decision(alarms=("last_resort_failed",), fallback=True)
    if by_count:
        decision = replace(
            decision, alarms=tuple(sorted({*decision.alarms, "lane_check_failed"}))
        )
    return Dispatch(lane_admit, lane_answer, decision)


def decision_json(dispatch: Dispatch) -> str:
    """One canonical JSON line for *dispatch*; equal inputs give equal lines."""
    return json.dumps(
        asdict(dispatch),
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
        ensure_ascii=False,
    )
