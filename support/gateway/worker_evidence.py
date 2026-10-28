"""Read durable coordinates for a worker-owned safe point.

This does not establish a safe point or fence a provider/tool. The caller must
hold and revalidate its generation-checked coordinator and exact safe-point
identity throughout the read and before acting on its result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import SessionStore, TurnInfo

Site = Literal["plan", "reflect", "approval"]
_PAGE_SIZE = 1000


class EvidenceError(RuntimeError):
    """The durable evidence cannot safely identify the requested site."""


@dataclass(frozen=True)
class SafePointEvidence:
    turn_id: str
    site: Site
    step: int
    attempt: int
    event_id: int
    request_id: str | None = None


def _positive(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check(runner: SessionRunner, store: SessionStore, turn: TurnInfo) -> None:
    current = runner.current_turn
    if (
        store.failed is not None
        or current is None
        or (current.id, current.session_id, current.turn)
        != (turn.id, turn.session_id, turn.turn)
        or runner.session_id != turn.session_id
        or runner.run_ended
    ):
        raise EvidenceError("store failed or active turn changed")


async def committed_evidence(
    runner: SessionRunner,
    store: SessionStore,
    *,
    turn_id: str,
    site: Site,
    request_id: str | None = None,
) -> SafePointEvidence:
    """Read only committed events for the active turn, failing closed.

    A missing receipt after a failed writer is never treated as proof that the
    writer rolled back. The caller must also prove the requested site is the
    still-current fenced safe point; events alone cannot prove that boundary.
    """
    if site not in ("plan", "reflect", "approval"):
        raise ValueError("invalid safe-point site")
    if not isinstance(turn_id, str) or not turn_id:
        raise ValueError("turn_id must be non-empty")
    if (site == "approval" and (not isinstance(request_id, str) or not request_id)) or (
        site != "approval" and request_id is not None
    ):
        raise ValueError("approval requires a request ID; other sites must omit it")
    turn = runner.current_turn
    if turn is None or turn.id != turn_id:
        raise EvidenceError("active turn changed")
    _check(runner, store, turn)
    await runner.flush()
    _check(runner, store, turn)
    high = await store.latest_event_id(turn.session_id)
    _check(runner, store, turn)

    after = 0
    started = False
    run_id: str | None = None
    step: int | None = None
    attempt: int | None = None
    last_site: str | None = None
    site_id: int | None = None
    approval_id: int | None = None
    seen_steps: set[int] = set()
    seen_reflections: set[int] = set()
    while after < high:
        page = await store.list_events(
            turn.session_id,
            after_id=after,
            limit=_PAGE_SIZE,
            turns=(turn.turn, turn.turn),
        )
        _check(runner, store, turn)
        if not page:
            break
        for stored in page:
            if (
                stored.id is None
                or stored.id <= after
                or stored.turn != turn.turn
                or stored.session_id != turn.session_id
            ):
                raise EvidenceError("invalid persisted event ordering")
            after = stored.id
            if after > high:
                break
            event = stored.event
            if not isinstance(event, dict) or not isinstance(event.get("data"), dict):
                raise EvidenceError("malformed persisted event")
            kind, data = event.get("type"), event["data"]
            event_run = event.get("run_id")
            if event_run and (
                not isinstance(event_run, str)
                or (run_id is not None and event_run != run_id)
            ):
                raise EvidenceError("event belongs to another run")
            if kind == "turn.started":
                if (
                    started
                    or data.get("turn_id") != turn.id
                    or data.get("turn") != turn.turn
                ):
                    raise EvidenceError("ambiguous turn identity")
                started = True
            elif kind == "turn.finished":
                raise EvidenceError("turn already finished")
            elif kind == "run.started":
                value = event.get("run_id")
                if run_id is not None or not isinstance(value, str) or not value:
                    raise EvidenceError("ambiguous run identity")
                run_id = value
            elif kind in ("step.started", "reflection.started"):
                if not run_id or event.get("run_id") != run_id:
                    raise EvidenceError("event belongs to another run")
                value = data.get("attempt")
                if not _positive(value):
                    raise EvidenceError("invalid agent attempt")
                if kind == "step.started":
                    number = data.get("step")
                    if (
                        not _positive(number)
                        or number in seen_steps
                        or (step is not None and number <= step)
                    ):
                        raise EvidenceError("ambiguous agent step")
                    step, attempt = number, value
                    seen_steps.add(number)
                    last_site = "plan"
                else:
                    if step is None or attempt != value or value in seen_reflections:
                        raise EvidenceError("ambiguous reflection coordinates")
                    seen_reflections.add(value)
                    last_site = "reflect"
                site_id = after
            elif kind == "approval.requested":
                req = data.get("request_id")
                if not isinstance(req, str) or not req:
                    raise EvidenceError("invalid approval request")
                if req == request_id:
                    if (
                        approval_id is not None
                        or not run_id
                        or step is None
                        or last_site != "plan"
                    ):
                        raise EvidenceError("ambiguous approval coordinates")
                    approval_id = after
            elif kind == "run.finished":
                raise EvidenceError("run already finished")
        if after >= high or len(page) < _PAGE_SIZE:
            break
    _check(runner, store, turn)
    if not started or not run_id or step is None or attempt is None:
        raise EvidenceError("missing committed turn or agent coordinates")
    if site == "approval":
        if (
            approval_id is None
            or last_site != "plan"
            or site_id is None
            or approval_id < site_id
        ):
            raise EvidenceError("missing or stale approval request")
        site_id = approval_id
    elif last_site != site or site_id is None:
        raise EvidenceError("missing or stale safe-point event")
    return SafePointEvidence(turn.id, site, step, attempt, site_id, request_id)
