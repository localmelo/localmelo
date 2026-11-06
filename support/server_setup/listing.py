"""The first-run list as data, in the words of the console's catalogs.

One list for every surface: the terminal prints it as a table
(:mod:`~localmelo.support.server_setup.terminal`) and the console's welcome card
draws it. :func:`build` turns a :class:`~localmelo.support.server_setup.run.Plan`
and what the person left out into a :class:`Listing`: the rows (what, how big,
where), the total, and one line for each thing that leaving a row out leads to.
The words are the ``setup.list.*`` keys of ``static/i18n``, read through *say*
(:func:`english` for the terminal, the gateway's ``t`` for a request's language),
so a surface that words a row differently has its own key, never its own copy.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any, Final

from localmelo.support.server_setup import run
from localmelo.support.words import t_in

# Reads the words of a catalog key, with its parameters filled in.
Say = Callable[..., str]

# A backend's name when the plan has no row for it.
BACKENDS: Final = {"omlx": "oMLX", "splash": "Splash"}
# The words of a plan item's kind (backend or model; whether the chat or the memory needs it).
KINDS: Final = {
    ("backend", True): "setup.list.kind.chatBackend",
    ("backend", False): "setup.list.kind.memoryBackend",
    ("model", True): "setup.list.kind.chatModel",
    ("model", False): "setup.list.kind.memoryModel",
}
# The words of the ways an item reaches this Mac; another way shows as it is.
VIA: Final = {
    "dmg": "setup.list.via.dmg",
    "homebrew": "setup.list.via.homebrew",
    "huggingface": "setup.list.via.huggingface",
}
# What leaving out the chat model or the login item leads to: the terminal's
# words name its commands, the card's do not.
NO_CHAT: Final = {
    True: "setup.list.effect.noChat",
    False: "setup.list.effect.card.noChat",
}
NO_LOGIN: Final = {
    True: "setup.list.effect.noLogin",
    False: "setup.list.effect.card.noLogin",
}
THE_MODEL: Final = {
    "chat": "setup.row.the.chat",
    "embedding": "setup.row.the.embedding",
}
_VERSION_TAIL: Final = re.compile(r"\s+\d[\d.]*$")


def english(key: str, /, **params: Any) -> str:
    """The words of *key* in English: what a surface without a request's language says."""
    return t_in("en", key, **params)


def bare(name: str) -> str:
    """*name* without its version: ``oMLX 0.7.3`` is ``oMLX``."""
    return _VERSION_TAIL.sub("", name)


def backend_name(plan: run.Plan, backend: str) -> str:
    """The name of *backend* (an id such as ``omlx``) as a person says it."""
    item = next((i for i in plan.items if i.id == f"backend:{backend}"), None)
    return bare(item.name) if item and item.name else BACKENDS.get(backend, backend)


def _gb(count: float) -> str:
    return f"{count / 1e9:.1f}"


def human(count: int) -> str:
    """A size as a person reads it: ``881 MB``, ``4.6 GB``."""
    if count >= 1_000_000_000:
        return f"{_gb(count)} GB"
    return f"{max(1, round(count / 1e6))} MB"


def amounts(part: int, whole: int) -> tuple[str, str]:
    """*part* in the unit of *whole*, without it, and *whole* with it: ``("14.1", "21.2 GB")``."""
    shown = _gb(part) if whole >= 1_000_000_000 else str(round(part / 1e6))
    return shown, human(whole)


def left(item: run.Item) -> int:
    """What is still to download for *item*."""
    if item.state == "done":
        return 0
    return max(item.size_bytes - item.cached_bytes, 0)


def size_left(plan: run.Plan, skip: Collection[str]) -> int:
    """Bytes still to download for *plan* when the items in *skip* are left out."""
    return max(
        plan.download_bytes - sum(left(item) for item in plan.items if item.id in skip),
        0,
    )


def skippable_ids(plan: run.Plan, *, gateway: bool = True) -> list[str]:
    """The ids a person may leave out: what is not here yet, and the login items.

    *gateway* is whether the gateway's own login item is among them (a job that
    runs inside the gateway does not register it).
    """
    return [
        *(
            item.id
            for item in plan.items
            if item.state != "done" and item.kind in ("backend", "model")
        ),
        run.SERVICE_MELO_SERVE,
        *([run.SERVICE_GATEWAY] if gateway else []),
    ]


def can_set_up(plan: run.Plan, skip: Collection[str]) -> bool:
    """Whether the list may say yes: the pick fits the disk, or the person left something out."""
    return plan.enough_disk or bool(skip)


def setup_row(
    plan: run.Plan, skip: Collection[str], say: Say, *, resume: bool = False
) -> str:
    """The row that says yes: ``Set up Qwen3 8B (5.8 GB)``, or with *resume* ``Continue (1.6 GB left)``."""
    assert plan.pick is not None
    remaining = size_left(plan, skip)
    if resume and remaining:
        return say("setup.list.resume", size=human(remaining))
    return say(
        "setup.list.setUp",
        model=plan.pick.name,
        size=human(remaining) if remaining else say("setup.list.noDownload"),
    )


def _free(plan: run.Plan, say: Say) -> str:
    free = plan.machine.free_gb
    return say("setup.list.free", gb=f"{free:.1f}" if 0 < free < 10 else f"{free:.0f}")


def this_mac(plan: run.Plan, say: Say) -> str:
    """``This Mac: Apple M5 Max · 128 GB · macOS 27.2 · 83 GB free``."""
    mac = plan.machine
    parts = [
        mac.chip,
        f"{mac.ram_gb} GB" if mac.ram_gb else "",
        say("setup.list.macos", version=mac.macos) if mac.macos else "",
        _free(plan, say) if mac.free_gb else "",
    ]
    return say("setup.list.mac", specs=" · ".join(part for part in parts if part))


@dataclass(frozen=True)
class Row:
    """One row of the list.

    ``id`` is what the job reports the row by (and what a person may leave out
    when ``off`` allows it). ``state`` is ``todo``, ``here`` (nothing to do),
    ``partial`` (some of it downloaded) or ``skipped``; ``size`` is what is left
    to download, or ``""``; ``where`` says where it goes, or why nothing happens.
    """

    id: str
    name: str
    kind: str
    size: str
    where: str
    state: str
    off: bool = False


def _row(
    item: run.Item,
    skip: Collection[str],
    chat: run.Pick,
    seen: set[tuple[str, str]],
    say: Say,
) -> Row:
    """The row of one plan item."""
    chat_side = (
        item.id == f"backend:{chat.backend}"
        if item.kind == "backend"
        else item.role == "chat"
    )
    kind = say(KINDS[("backend" if item.kind == "backend" else "model", chat_side)])
    todo = item.state != "done" and item.kind in ("backend", "model")
    if item.state == "done":
        return Row(item.id, item.name, kind, "", say("setup.list.here"), "here")
    if item.id in skip:
        return Row(
            item.id, item.name, kind, "", say("setup.list.skipped"), "skipped", todo
        )
    size = human(left(item))
    if item.state == "partial":
        part, whole = amounts(item.cached_bytes, item.size_bytes)
        where = say("setup.list.partial", part=part, whole=whole)
        return Row(item.id, item.name, kind, size, where, "partial", todo)
    place = (item.via, item.dest)  # a place that is shown once is "same place" after
    where = (say(VIA[item.via]) if item.via in VIA else item.via) + (
        f" → {item.dest}" if item.dest else ""
    )
    if item.kind == "model" and place in seen:
        where = say("setup.list.samePlace")
    seen.add(place)
    return Row(item.id, item.name, kind, size, where, "todo", todo)


def _names(
    plan: run.Plan, ids: Collection[str], wanted: Collection[str], say: Say
) -> str:
    """The names of the items in *ids* that are in *wanted*, such as ``oMLX and Qwen3 8B``."""
    names = [bare(i.name) for i in plan.items if i.id in ids and i.id in wanted]
    return say("setup.list.and").join(names)


def effects(
    plan: run.Plan, skip: Collection[str], say: Say, *, terminal: bool
) -> list[str]:
    """One line for each thing that leaving out *skip* leads to.

    The terminal's lines name its commands (``melo-serve get`` finishes the
    download later, ``melo serve`` starts what is not running); a card has its
    own words for the same facts.
    """
    if plan.pick is None:
        return []
    chat, embedding = run.role_items(plan, plan.pick)
    lines = []
    if names := _names(plan, skip, chat, say):
        lines.append(say(NO_CHAT[terminal], names=names, pick=plan.pick.id))
    if names := _names(plan, skip, embedding, say):
        lines.append(say("setup.list.effect.noMemory", names=names))
    if run.SERVICE_MELO_SERVE in skip:
        lines.append(say(NO_LOGIN[terminal]))
    if terminal and run.SERVICE_GATEWAY in skip:
        lines.append(say("setup.list.effect.noGateway"))
    return lines


@dataclass(frozen=True)
class Listing:
    """What a person says yes to: the Mac, the pick, the rows, the total and what leaving rows out leads to."""

    this_mac: str
    recommended: str
    rows: list[Row]
    download: str
    free: str
    notes: list[str] = field(default_factory=list)
    left_bytes: int = 0


def build(
    plan: run.Plan,
    skip: Collection[str],
    melo_serve_there: str,
    say: Say,
    *,
    chosen: bool = False,
    terminal: bool = True,
) -> Listing:
    """The list for *plan* (which must be available) with *skip* left out.

    *melo_serve_there* is the version of the melo_serve already on this Mac, or
    ``""``; *chosen* whether the pick is the person's choice, not the
    recommendation. A *terminal* list has the gateway's login item as a row; a
    card runs inside the gateway and has none.
    """
    assert plan.pick is not None
    pick = plan.pick
    if melo_serve_there:
        service = Row(
            run.MELO_SERVE,
            "melo_serve",
            say("setup.list.kind.service"),
            "",
            say("setup.list.here"),
            "here",
        )
    elif run.MELO_SERVE in skip:
        service = Row(
            run.MELO_SERVE,
            "melo_serve",
            say("setup.list.kind.service"),
            "",
            say("setup.list.skipped"),
            "skipped",
        )
    else:
        service = Row(
            run.MELO_SERVE,
            "melo_serve",
            say("setup.list.kind.service"),
            say("setup.list.serveSize"),
            say("setup.list.viaUv"),
            "todo",
        )
    seen: set[tuple[str, str]] = set()
    login = say("setup.list.login")
    rows = [
        service,
        *(_row(item, skip, pick, seen, say) for item in plan.items),
        Row(
            run.SERVICE_MELO_SERVE,
            login,
            "melo_serve",
            "",
            (
                say("setup.list.skipped")
                if run.SERVICE_MELO_SERVE in skip
                else say("setup.list.loginNote")
            ),
            "skipped" if run.SERVICE_MELO_SERVE in skip else "todo",
            True,
        ),
    ]
    if terminal:
        rows.append(
            Row(
                run.SERVICE_GATEWAY,
                login,
                say("setup.list.kind.gateway"),
                "",
                (
                    say("setup.list.skipped")
                    if run.SERVICE_GATEWAY in skip
                    else say("setup.list.sameNotice")
                ),
                "skipped" if run.SERVICE_GATEWAY in skip else "todo",
                True,
            )
        )
    remaining = size_left(plan, skip)
    notes = []
    if not can_set_up(plan, skip):
        notes.append(say("setup.list.noRoom"))
    notes += effects(plan, skip, say, terminal=terminal)
    return Listing(
        this_mac(plan, say),
        say(
            "setup.list.chosen" if chosen else "setup.list.recommended",
            model=pick.name,
            backend=backend_name(plan, pick.backend),
        ),
        rows,
        (
            say("setup.list.download", size=human(remaining))
            if remaining
            else say("setup.list.nothing")
        ),
        (
            say("setup.list.freeSentence", free=_free(plan, say))
            if plan.machine.free_gb
            else ""
        ),
        notes,
        remaining,
    )


def row_names(plan: run.Plan, say: Say) -> dict[str, str]:
    """The plain name of each row the job can report, by id."""
    names = {
        run.MELO_SERVE: "melo_serve",
        run.SERVICE_MELO_SERVE: say("setup.list.login"),
        run.SERVICE_GATEWAY: say("setup.row.gateway"),
        run.DOWNLOAD: say("setup.row.download"),
        run.SAVE: say("setup.row.save"),
        "config:hf_home": say("setup.row.folder"),
        "cache:hf_home": say("setup.row.folder"),
        "disk": say("setup.row.disk"),
        "get": say("setup.row.download"),
        **{i.id: bare(i.name) for i in plan.items if i.name},
    }
    for role, row in (("chat", run.LOAD_CHAT), ("embedding", run.LOAD_EMBEDDING)):
        item = next((i for i in plan.items if i.role == role), None)
        names[row] = say(
            "setup.row.loading",
            name=bare(item.name) if item else say(THE_MODEL[role]),
        )
    return names
