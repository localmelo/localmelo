"""The steps auto mode blocked in a chat or a task, and their Allow once (#339).

A block is kept in the ``tool.finished`` record of a call auto mode did not
run: its ``approval``, or one of its ``tool_asks`` (a question the call's tool
asked itself, such as typing on a page, ``tool`` naming the question), with
origin ``auto``, ``allowed`` false, the ``review`` and, when Allow once may run
it, its ``step_key``. :func:`blocks_in` reads both alike. Allow once finds the
block by its ``request_id`` in that session's records (a client never sends a
key), allows that key once in the session's :class:`AutoPolicy`, and sends
the person's message that asks melo to go ahead. A block whose step ran on
an allowance, or has one waiting, is not allowed again.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final

from localmelo.melo.contracts.approvals import keep_holds
from localmelo.melo.contracts.review import TITLES, ActionReview, ModelAnswer
from localmelo.support.gateway.handover import fenced
from localmelo.support.gateway.i18n import t
from localmelo.support.sessions.store import SessionStore


class BlockError(LookupError):
    """No block to allow: *code* is ``not_found``, ``not_allowable``, ``already_allowed``,
    ``not_keepable`` (nothing may be kept for it, #619) or ``request_changed``
    (its scheduled task's request changed since its run)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code

    @property
    def status(self) -> int:
        """The HTTP status the API answers with."""
        return 404 if self.code == "not_found" else 409

    @property
    def message(self) -> str:
        """What the API says, in the console's language."""
        if self.code == "not_found":
            return t("errors.blockNotFound")
        if self.code == "already_allowed":
            return t("errors.blockAllowedAlready")
        if self.code == "not_keepable":
            return t("errors.blockNotKeepable")
        if self.code == "request_changed":
            return t("errors.keepStale")
        return t("errors.blockNotAllowable")


def _approval(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    approval = data.get("approval") if isinstance(data, dict) else None
    return approval if isinstance(approval, dict) else {}


def blocks_in(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Auto's blocks and the allowances used in the ``tool.finished`` *event*, in order.

    The call's own ``approval`` first, then its ``tool_asks``: each one whose
    origin is ``auto`` and that is not allowed, or that ran on an Allow once
    (``allowance`` ``once``) or on a scheduled task's kept answer
    (``allowance`` ``kept``, #619). Anything else in the record is none.
    """
    data = event.get("data")
    asks = data.get("tool_asks") if isinstance(data, dict) else None
    return [
        one
        for one in [_approval(event), *(asks if isinstance(asks, list) else [])]
        if isinstance(one, dict)
        and (
            (one.get("origin") == "auto" and one.get("allowed") is False)
            or one.get("allowance") in ("once", "kept")
        )
    ]


def _categories(approval: dict[str, Any]) -> list[str]:
    review = approval.get("review")
    found = review.get("categories") if isinstance(review, dict) else None
    return [str(c) for c in found] if isinstance(found, list) else []


def allowable(approval: dict[str, Any]) -> tuple[str, list[str]] | None:
    """A block's step key and categories when Allow once may run it, else ``None``.

    Never for a step that changes localmelo itself, never for a call a
    stopped turn did not review (it has no categories of its own), and never
    for a question its tool asked itself (``tool``) without its arguments:
    the person sees all of it first, or nothing is allowed.
    """
    key = approval.get("step_key")
    categories = [c for c in _categories(approval) if c in TITLES]
    if (
        ("tool" in approval and not isinstance(approval.get("arguments"), dict))
        or approval.get("origin") != "auto"
        or approval.get("allowed") is not False
        or not isinstance(key, str)
        or not key
        or not categories
        or "self" in categories
    ):
        return None
    return key, categories


async def block_to_allow(
    store: SessionStore,
    session_id: str,
    request_id: str,
    live: Callable[[str], bool] = lambda key: False,
) -> tuple[str, list[str], str]:
    """The step key, categories and tool of the block *request_id* Allow once may run.

    :class:`BlockError` ``not_found`` for no such block in *session_id*,
    ``not_allowable`` for one that can never be allowed (:func:`allowable`),
    and ``already_allowed`` once the step ran on an allowance after it, or
    while *live* says its key has one not used yet.
    """
    found = None
    for stored in reversed(await store.auto_records(session_id, limit=200)):
        found = next(
            (
                (stored, one)
                for one in blocks_in(stored.event)
                if one.get("request_id") == request_id and one.get("allowed") is False
            ),
            None,
        )
        if found is not None:
            break
    if found is None:
        raise BlockError("not_found")
    stored, approval = found
    allowed = allowable(approval)
    if allowed is None:
        raise BlockError("not_allowable")
    key, categories = allowed
    block = next(
        (
            b
            for b in await blocked_steps(store, session_id)
            if b["request_id"] == request_id
        ),
        None,
    )
    if live(key) or (block is not None and block["used_at"] is not None):
        raise BlockError("already_allowed")
    raw = stored.event.get("data")
    tool = str(raw.get("name") or "") if isinstance(raw, dict) else ""
    return key, categories, tool


def step_words(categories: list[str]) -> str:
    """What the step does, in the page's language: "install software"."""
    title = t(f"auto.title.{categories[0]}") if categories else ""
    return title[:1].lower() + title[1:]


def go_ahead_text(categories: list[str]) -> str:
    """The person's message Allow once sends (decision 7)."""
    return t("auto.goAhead", what=step_words(categories))


async def blocked_steps(store: SessionStore, session_id: str) -> list[dict[str, Any]]:
    """The steps blocked in *session_id*, newest last, each with its record.

    ``used_at`` is when the allowed step ran (a later call of the same key
    allowed once), else ``None``. A question the call's tool asked itself
    has its name in ``question`` and its arguments (the whole text typed,
    what a form sends) in ``arguments``; a call's own block has ``None`` in both.
    """
    records = await store.auto_records(session_id, limit=200)
    out: list[dict[str, Any]] = []
    for stored in records:
        event = stored.event
        raw = event.get("data")
        data: dict[str, Any] = raw if isinstance(raw, dict) else {}
        own = _approval(event)
        for approval in blocks_in(event):
            if approval.get("allowance") == "kept":
                continue  # a step that ran on a kept answer was never blocked
            if approval.get("allowance") == "once":
                for block in out:
                    if block["step_key"] == approval.get("step_key") and not block.get(
                        "used_at"
                    ):
                        block["used_at"] = event.get("ts")
                continue
            review = approval.get("review")
            question = approval is not own
            asked = approval.get("arguments") if question else None
            out.append(
                {
                    "request_id": approval.get("request_id"),
                    "call_id": data.get("call_id"),
                    "name": data.get("name"),
                    "step_key": approval.get("step_key"),
                    "reason": approval.get("reason", ""),
                    "review": review if isinstance(review, dict) else None,
                    "ts": event.get("ts"),
                    "turn": stored.turn,
                    "can_allow": allowable(approval) is not None,
                    "used_at": None,
                    "preview": str(approval.get("preview") or "") if question else "",
                    "question": str(approval.get("tool") or "") if question else None,
                    "arguments": asked if isinstance(asked, dict) else None,
                    # Whether the text holds the person's details (#619): only
                    # an explicit ``False`` lets a kept answer be made from it.
                    "personal": approval.get("personal") if question else None,
                }
            )
    # The step as the chat shows it: its tool.started preview.
    for turn in sorted({b["turn"] for b in out if b["turn"] is not None}):
        events = await store.list_events(session_id, turns=(turn, turn))
        previews = {
            e.event["data"].get("call_id"): str(e.event["data"].get("preview") or "")
            for e in events
            if e.event.get("type") == "tool.started"
            and isinstance(e.event.get("data"), dict)
        }
        for block in out:
            if block["turn"] == turn and not block["preview"]:
                block["preview"] = previews.get(block["call_id"], "")
    return out


# Who blocked a step, as a follow-up and the refusal of a repeat name them (#411).
BLOCKERS: dict[str, str] = {
    "auto": "Auto mode",
    "user": "the user",
    "policy": "melo's rules",
}
# An auto record of a call a stopped turn never reviewed: not blocked on its own.
_STOPPED = "the turn stopped"


async def denied_steps(
    store: SessionStore, session_id: str, *, limit: int = 100
) -> list[dict[str, Any]]:
    """The steps blocked in *session_id* by the person, auto mode or melo's rules (#411).

    Newest last, each with its ``call_key`` (what makes a later call the same
    step), its ``loose_keys`` (what makes one alike), ``who`` blocked it, its
    ``reason``, the ``step_key`` of an auto block, its tool ``name`` and its
    ``preview``. Only real denials carry a call key. A call a stopped turn never
    reviewed, a refusal of a repeat, and a record from before call keys were
    kept are left out. A question the call's tool asked itself and had refused
    (``tool_asks``: a site web_fetch or the browser would open, #431) is a
    blocked step too, with the question's own preview ("read <address>").
    """
    records = await store.denied_records(session_id, limit=limit)
    out: list[dict[str, Any]] = []
    for stored in records:
        raw = stored.event.get("data")
        data: dict[str, Any] = raw if isinstance(raw, dict) else {}
        asks = data.get("tool_asks")
        denials = [_approval(stored.event)] + [
            ask
            for ask in (asks if isinstance(asks, list) else [])
            if isinstance(ask, dict)
            and ask.get("allowed") is False
            and ask.get("origin") in BLOCKERS
        ]
        for approval in denials:
            key = approval.get("call_key")
            if (
                not isinstance(key, str)
                or not key
                or approval.get("earlier")
                or approval.get("reason") == _STOPPED
            ):
                continue
            loose = approval.get("loose_keys")
            out.append(
                {
                    "call_key": key,
                    "loose_keys": (
                        [str(k) for k in loose] if isinstance(loose, list) else []
                    ),
                    "who": BLOCKERS.get(str(approval.get("origin")), "melo's rules"),
                    "reason": str(approval.get("reason") or ""),
                    "step_key": str(approval.get("step_key") or ""),
                    "name": str(data.get("name") or ""),
                    "call_id": data.get("call_id"),
                    "turn": stored.turn,
                    "preview": str(approval.get("preview") or ""),
                }
            )
    for turn in sorted({b["turn"] for b in out if b["turn"] is not None}):
        events = await store.list_events(session_id, turns=(turn, turn))
        previews = {
            e.event["data"].get("call_id"): str(e.event["data"].get("preview") or "")
            for e in events
            if e.event.get("type") == "tool.started"
            and isinstance(e.event.get("data"), dict)
        }
        for block in out:
            if block["turn"] == turn and not block["preview"]:
                block["preview"] = previews.get(block["call_id"], "")
    return out


def keep_site(block: dict[str, Any]) -> str | None:
    """The site a scheduled task's kept answer made from the stored *block* holds (#619), or ``None``.

    Only a block Allow once may run as it was written (``can_allow`` from
    :func:`allowable`: the whole text typed or form sent kept, no secret in
    it, its site not refused) and that a kept answer would have let run, by
    the gate's own predicate (:func:`keep_holds`) on what the block stored:
    its review with the model's answer, and ``personal`` (``False`` only when
    melo's detector found none of the person's details in the text). Whether
    its task is a scheduled task's run is the caller's to check.
    """
    question, arguments = block.get("question"), block.get("arguments")
    if not (block.get("can_allow") and question and isinstance(arguments, dict)):
        return None
    found = keep_holds(
        str(question),
        arguments,
        _review(block.get("review")),
        personal=block.get("personal") is not False,
    )
    return None if found is None else found[0]


def _review(stored: object) -> ActionReview:
    """The fields of a stored review :func:`keep_holds` reads; anything missing reads as not reviewed."""
    review = stored if isinstance(stored, dict) else {}
    categories = review.get("categories")
    model = review.get("model")
    return ActionReview(
        "high" if review.get("verdict") == "high" else "undecided",
        tuple(str(c) for c in categories) if isinstance(categories, list) else (),
        by="model" if review.get("by") == "model" else "rules",
        model=(
            ModelAnswer(fits_request=str(model.get("fits_request") or ""))
            if isinstance(model, dict) and not model.get("error")
            else None
        ),
    )


async def kept_uses(store: SessionStore, session_id: str) -> list[dict[str, Any]]:
    """The steps of *session_id* that ran on its scheduled task's kept answer (#619), oldest first.

    Each as its record has it: the question (``browser-type:<origin>`` or
    ``browser-submit:<origin>``), what was typed or sent (``arguments``) and
    when (``ts``), for the task's page to say so.
    """
    out: list[dict[str, Any]] = []
    for stored in await store.auto_records(session_id, limit=200):
        for one in blocks_in(stored.event):
            if one.get("allowance") == "kept" and isinstance(
                one.get("arguments"), dict
            ):
                out.append(
                    {"question": str(one.get("tool") or ""),
                     "arguments": one["arguments"], "ts": stored.event.get("ts")}
                )  # fmt: skip
    return out


# What a task's run reads about a step the person allowed once (#617), in melo's
# words: the step as it was asked, fenced as data, so a retry can repeat it.
ALLOWED_HEAD: Final = (
    "[melo's note: the person allowed one blocked step once]\n"
    "It runs only exactly as below: the same site, the same box and every "
    "character of the text; anything else is reviewed again. The step:"
)
ALLOWED_END: Final = "[end of the allowed step]"
# The order a step's facts are told in; the text, all of it, comes last.
_TOLD: Final = ("site", "on", "field", "to", "how", "browser")


def _step_told(question: str, arguments: dict[str, Any]) -> str:
    """One allowed question as data lines: its kind and site, then its facts, then its text."""
    kind = "typing" if question.startswith("browser-type:") else "sending a form"
    lines = [f"step: {kind}"]
    lines += [f"{name}: {arguments[name]}" for name in _TOLD if arguments.get(name)]
    fields = arguments.get("fields")
    if isinstance(fields, list) and fields:
        lines.append("fields:")
        lines += [
            f"- {row[0]}: {row[1]}"
            for row in fields
            if isinstance(row, list) and len(row) == 2
        ]
    if "text" in arguments:
        lines += ["text:", str(arguments["text"])]
    return "\n".join(lines)


async def allowed_note(
    store: SessionStore, session_ids: Sequence[str], live: Callable[[str], bool]
) -> str:
    """The note of each step blocked in *session_ids* that holds a live allowance, or ``""``.

    Only a question its tool asked itself (a text typed, a form sent), written
    from its stored record, never from anyone's words about it; *live* says
    whether its key is allowed once and not used.
    """
    told = [
        _step_told(str(block["question"]), block["arguments"])
        for session in session_ids
        for block in await blocked_steps(store, session)
        if block["question"]
        and isinstance(block["arguments"], dict)
        and block["used_at"] is None
        and isinstance(block["step_key"], str)
        and live(block["step_key"])
    ]
    return fenced("allowed", ALLOWED_HEAD, "\n\n".join(told), ALLOWED_END)
