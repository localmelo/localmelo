"""Step templates: episodes, statistics, the skeleton, the gate and the level.

A template version is computed at a run's end from the template's recent
episodes (:func:`build_version`): step statistics over the *statistics
window* (:func:`statistics_window`) and a skeleton, the named steps of the
medoid episode (:func:`medoid`, :func:`collapse`). At a run's start,
:func:`decide_level` says what a display may show from it: level 0 facts
only, 1 past facts and named steps, 2 a bar. Everything here is pure.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Final

from localmelo.melo.memory.procedures.classify import PASS_ANCHOR, TEST, WRITE
from localmelo.melo.memory.procedures.normalize import known_model
from localmelo.melo.memory.procedures.trace import Segment, first_steps

COMPLETED: Final = "completed"
# The statuses of a run that ended through the normal path.
NORMAL_END: Final = frozenset(
    {"completed", "budget_exhausted", "stuck", "gate_stopped"}
)

DENOMINATORS: Final = ("median", "dynamic", "milestone", "hybrid")
DROPS: Final = ("off", "conditional", "unconditional")
STATISTICS: Final = ("completed", "normal")

# How a run's routine was found.
MATCHES: Final = ("new", "exact", "routine", "similar")

# The share of a window's episodes that must reach a milestone for the
# milestone bar to use it, and the fraction the hybrid bar jumps to at the
# first pass.
MILESTONE_SHARE: Final = 0.6
HYBRID_JUMP: Final = 0.9


@dataclass(frozen=True)
class Params:
    """The parameters of the learner and the tracker (``params_json``).

    ``window`` episodes at most enter a version; ``stats`` says which:
    ``completed`` runs only, or every ``normal`` end (completed,
    budget_exhausted, stuck, gate_stopped); ``drop`` is the exploration
    drop (section 5.6); the gate needs ``min_episodes`` and a spread of at
    most ``spread``; the bar never passes ``cap`` while running, with the
    ``denominator`` rule; a model whose mean shadow error over its last
    ``demotion_window`` scores (at least ``demotion_min``) exceeds
    ``demotion`` is demoted to level 1.
    """

    window: int = 8
    stats: str = "completed"
    drop: str = "conditional"
    min_episodes: int = 2
    spread: float = 0.5
    cap: float = 0.95
    denominator: str = "dynamic"
    demotion: float = 0.18
    demotion_window: int = 10
    demotion_min: int = 5

    def __post_init__(self) -> None:
        if self.stats not in STATISTICS:
            raise ValueError(f"stats must be one of {STATISTICS}, not {self.stats!r}")
        if self.drop not in DROPS:
            raise ValueError(f"drop must be one of {DROPS}, not {self.drop!r}")
        if self.denominator not in DENOMINATORS:
            raise ValueError(
                f"denominator must be one of {DENOMINATORS}, not {self.denominator!r}"
            )
        if self.window < 1 or self.min_episodes < 1 or self.demotion_window < 1:
            raise ValueError("window, min_episodes and demotion_window must be >= 1")
        for name in ("spread", "cap", "demotion"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a number greater than 0")
        if self.cap >= 1:
            raise ValueError("cap must be below 1")

    @property
    def statistics_statuses(self) -> frozenset[str]:
        return frozenset({COMPLETED}) if self.stats == "completed" else NORMAL_END

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Params:
        """Params from ``params_json``; unknown keys are ignored."""
        names = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})


DEFAULT_PARAMS: Final = Params()


@dataclass(frozen=True)
class Episode:
    """One finished run of a routine: a row of ``episodes``."""

    task_id: str
    routine_id: str
    template_id: str
    model: str
    status: str
    steps: int
    created_at: float
    harness_version: str = "unknown"
    match: str = "new"
    level: int = 0
    template_version: int | None = None
    attempts: int = 1
    n_tool: int = 0
    duration_s: float = 0.0
    first_pass_step: int | None = None
    cycles: int = 0
    segments: tuple[Segment, ...] = ()
    shadow_bar_error: float | None = None

    @property
    def completed(self) -> bool:
        return self.status == COMPLETED

    def first_step_of(self, anchor: str) -> int | None:
        """The step of the first W, T or pass (``P``), from the episode's record."""
        if anchor == PASS_ANCHOR and self.first_pass_step is not None:
            return self.first_pass_step
        return first_steps(self.segments).get(anchor)


@dataclass(frozen=True)
class SkeletonItem:
    """A named step of a template: one segment, or a repeated pattern.

    A pattern item keeps its pattern's classes, no target, and how many
    times the pattern repeated (``repeat``); a single item has one class,
    its target and ``repeat`` ``None``.
    """

    cls: tuple[str, ...]
    target: str | None
    label: str
    repeat: int | None = None

    def matches(self, cls: str, target: str | None) -> bool:
        """Whether a call of *cls* on *target* matches this item."""
        if cls not in self.cls:
            return False
        return self.target is None or self.target == target

    def as_dict(self) -> dict[str, Any]:
        return {
            "cls": list(self.cls),
            "target": self.target,
            "label": self.label,
            "repeat": self.repeat,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SkeletonItem:
        classes = data.get("cls") or []
        if isinstance(classes, str):
            classes = [classes]
        repeat = data.get("repeat")
        target = data.get("target")
        return cls(
            cls=tuple(str(c) for c in classes),
            target=None if target is None else str(target),
            label=str(data.get("label", "")),
            repeat=None if repeat is None else int(repeat),
        )


@dataclass(frozen=True)
class Stats:
    """The step statistics of a template version (``stats_json``).

    ``steps`` are the statistics window's step counts, oldest first;
    ``spread`` is (hi − lo) ÷ median; ``wall_s`` the low, median and high
    durations; ``attempts_share`` the share of episodes with two or more
    attempts; ``tail_after_pass`` the median steps after the first pass;
    ``anchors`` the median step and fraction of each milestone (``W``,
    ``T``, ``P``) that enough episodes reached; ``last_steps`` and
    ``last_wall_s`` describe the latest completed episode.
    """

    n: int = 0
    steps: tuple[int, ...] = ()
    median: float = 0.0
    lo: int = 0
    hi: int = 0
    spread: float = 0.0
    cv: float = 0.0
    wall_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    attempts_share: float = 0.0
    tail_after_pass: float | None = None
    gate: bool = False
    gate_reason: str = ""
    anchors: tuple[tuple[str, float, float], ...] = ()
    last_steps: int | None = None
    last_wall_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "steps": list(self.steps),
            "median": self.median,
            "lo": self.lo,
            "hi": self.hi,
            "spread": round(self.spread, 6),
            "cv": round(self.cv, 6),
            "wall_s": [round(value, 3) for value in self.wall_s],
            "attempts_share": round(self.attempts_share, 6),
            "tail_after_pass": self.tail_after_pass,
            "gate": self.gate,
            "gate_reason": self.gate_reason,
            "anchors": [list(anchor) for anchor in self.anchors],
            "last_steps": self.last_steps,
            "last_wall_s": self.last_wall_s,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Stats:
        wall = data.get("wall_s") or [0.0, 0.0, 0.0]
        tail = data.get("tail_after_pass")
        last_wall = data.get("last_wall_s")
        last_steps = data.get("last_steps")
        return cls(
            n=int(data.get("n", 0)),
            steps=tuple(int(s) for s in data.get("steps", ())),
            median=float(data.get("median", 0.0)),
            lo=int(data.get("lo", 0)),
            hi=int(data.get("hi", 0)),
            spread=float(data.get("spread", 0.0)),
            cv=float(data.get("cv", 0.0)),
            wall_s=(float(wall[0]), float(wall[1]), float(wall[2])),
            attempts_share=float(data.get("attempts_share", 0.0)),
            tail_after_pass=None if tail is None else float(tail),
            gate=bool(data.get("gate", False)),
            gate_reason=str(data.get("gate_reason", "")),
            anchors=tuple(
                (str(a[0]), float(a[1]), float(a[2])) for a in data.get("anchors", ())
            ),
            last_steps=None if last_steps is None else int(last_steps),
            last_wall_s=None if last_wall is None else float(last_wall),
        )


@dataclass(frozen=True)
class Template:
    """One version of a template: a row of ``templates``."""

    template_id: str
    version: int
    routine_id: str
    model: str
    options: Mapping[str, Any]
    status: str
    source: str
    admitted_by: str
    source_ids: tuple[str, ...]
    skeleton: tuple[SkeletonItem, ...]
    medoid: str | None
    stats: Stats
    params: Params
    created_at: float
    parent_version: int | None = None
    eval: Mapping[str, Any] | None = None
    harness_version: str = "unknown"
    # "sleep" when the skeleton came from a sleep job: online versions
    # then copy it and refresh only the statistics.
    skeleton_source: str = "online"


TEMPLATE_STATUSES: Final = ("active", "superseded", "candidate", "rejected")
TEMPLATE_SOURCES: Final = ("online", "sleep", "user")


def statistics_window(
    window: Sequence[Episode], first_task_id: str | None, params: Params
) -> tuple[list[Episode], bool]:
    """The episodes the statistics use, and whether the first one was dropped.

    *window* holds the template's latest episodes whose status the
    statistics use, oldest first, at most ``params.window`` of them;
    *first_task_id* is the template's first episode, whatever its
    status. That episode is exploration: once two later completed
    episodes exist, it leaves the statistics under the ``unconditional``
    drop, and under the ``conditional`` drop only when it is completed
    and has more steps than every later completed episode.
    """
    episodes = list(window)
    if params.drop == "off" or first_task_id is None:
        return episodes, False
    first = next((e for e in episodes if e.task_id == first_task_id), None)
    if first is None:
        return episodes, False
    later = [e.steps for e in episodes if e is not first and e.completed]
    if len(later) < 2:
        return episodes, False
    if params.drop == "conditional" and not (
        first.completed and first.steps > max(later)
    ):
        return episodes, False
    return [e for e in episodes if e is not first], True


def compute_stats(
    window: Sequence[Episode], first_task_id: str | None, params: Params
) -> Stats:
    """The statistics of *window* (see :func:`statistics_window`) and the gate.

    The gate passes when the statistics hold at least
    ``params.min_episodes`` episodes, one of them completed, with a spread
    of at most ``params.spread``; ``gate_reason`` says why it does not.
    """
    usable = [e for e in window if e.steps > 0]
    episodes, dropped = statistics_window(usable, first_task_id, params)
    last_completed = next((e for e in reversed(usable) if e.completed), None)
    last_steps = None if last_completed is None else last_completed.steps
    last_wall = None if last_completed is None else last_completed.duration_s
    if not episodes:
        return Stats(
            gate_reason="no runs", last_steps=last_steps, last_wall_s=last_wall
        )
    steps = [e.steps for e in episodes]
    median = float(statistics.median(steps))
    lo, hi = min(steps), max(steps)
    spread = (hi - lo) / median if median > 0 else math.inf
    mean = statistics.fmean(steps)
    cv = statistics.pstdev(steps) / mean if mean > 0 else 0.0
    walls = [e.duration_s for e in episodes]
    tails = [e.steps - e.first_pass_step for e in episodes if e.first_pass_step]
    reason = ""
    if len(episodes) < params.min_episodes:
        reason = f"{len(episodes)} of {params.min_episodes} runs"
        if dropped:
            reason += " after the first run left as exploration"
    elif not any(e.completed for e in episodes):
        reason = "no completed run"
    elif spread > params.spread:
        reason = f"spread {spread:.2f} above {params.spread:g}"
    return Stats(
        n=len(episodes),
        steps=tuple(steps),
        median=median,
        lo=lo,
        hi=hi,
        spread=spread if math.isfinite(spread) else 1e9,
        cv=cv,
        wall_s=(min(walls), float(statistics.median(walls)), max(walls)),
        attempts_share=sum(e.attempts >= 2 for e in episodes) / len(episodes),
        tail_after_pass=float(statistics.median(tails)) if tails else None,
        gate=not reason,
        gate_reason=reason,
        anchors=_anchors(episodes),
        last_steps=last_steps,
        last_wall_s=last_wall,
    )


def _anchors(episodes: Sequence[Episode]) -> tuple[tuple[str, float, float], ...]:
    """Each milestone that enough *episodes* reached: (name, step, fraction)."""
    found: list[tuple[str, float, float]] = []
    for anchor in (WRITE, TEST, PASS_ANCHOR):
        reached = [
            (step, e.steps)
            for e in episodes
            if (step := e.first_step_of(anchor)) is not None and e.steps > 0
        ]
        if reached and len(reached) >= MILESTONE_SHARE * len(episodes):
            found.append(
                (
                    anchor,
                    float(statistics.median(s for s, _ in reached)),
                    float(statistics.median(s / n for s, n in reached)),
                )
            )
    return tuple(sorted(found, key=lambda anchor: anchor[1]))


# The medoid compares at most an episode's first MEDOID_MAX_SEGMENTS
# segments, and at most MEDOID_MAX_CELLS token pairs in all (the products
# of two sequences' lengths, over each pair once): about 50 ms of pure
# Python at 200 ns a pair. A run's end computes it inside asyncio.timeout,
# which cannot interrupt CPU, so past the budget a version keeps its
# parent's skeleton instead (see build_version).
MEDOID_MAX_SEGMENTS: Final = 200
MEDOID_MAX_CELLS: Final = 250_000


def _tokens(episode: Episode) -> list[tuple[str, str | None]]:
    return [(s.cls, s.target) for s in episode.segments[:MEDOID_MAX_SEGMENTS]]


def medoid_cells(episodes: Sequence[Episode]) -> int:
    """The token pairs :func:`medoid` compares for *episodes*."""
    lengths = [min(len(e.segments), MEDOID_MAX_SEGMENTS) for e in episodes]
    return sum(
        lengths[i] * lengths[j]
        for i in range(len(lengths))
        for j in range(i + 1, len(lengths))
    )


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """The Levenshtein distance between two sequences of tokens."""
    previous = list(range(len(b) + 1))
    for i, token in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (token != other),
                )
            )
        previous = current
    return previous[-1]


def medoid(episodes: Sequence[Episode]) -> Episode | None:
    """The episode with the least summed edit distance to the others.

    The sequences compared are each episode's first
    :data:`MEDOID_MAX_SEGMENTS` segments as (class, target) tokens, each
    pair once. Ties go to the latest episode, so a change of habit enters
    as soon as it is as central as the old one.
    """
    if not episodes:
        return None
    tokens = [_tokens(e) for e in episodes]
    totals = [0] * len(tokens)
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            distance = edit_distance(tokens[i], tokens[j])
            totals[i] += distance
            totals[j] += distance
    least = min(totals)
    best = max(i for i, total in enumerate(totals) if total == least)
    return episodes[best]


def collapse(segments: Sequence[Segment], longest: int = 4) -> tuple[SkeletonItem, ...]:
    """*segments* as skeleton items, a repeated adjacent pattern as one item.

    At each position, the pattern of 2 to *longest* segments (by class and
    target) that repeats right after itself over the most segments becomes
    one item with its classes and ``repeat``; otherwise the segment is an
    item of its own.
    """
    keys = [(s.cls, s.target) for s in segments]
    items: list[SkeletonItem] = []
    i = 0
    while i < len(segments):
        best_length, best_repeat = 1, 1
        for length in range(2, longest + 1):
            pattern = keys[i : i + length]
            if len(pattern) < length:
                break
            repeat = 1
            while keys[i + repeat * length : i + (repeat + 1) * length] == pattern:
                repeat += 1
            if repeat >= 2 and repeat * length > best_repeat * best_length:
                best_length, best_repeat = length, repeat
        if best_repeat >= 2:
            pattern_segments = segments[i : i + best_length]
            classes: list[str] = []
            for segment in pattern_segments:
                if segment.cls not in classes:
                    classes.append(segment.cls)
            label = " + ".join(segment.label for segment in pattern_segments)
            items.append(
                SkeletonItem(
                    cls=tuple(classes),
                    target=None,
                    label=label[:40],
                    repeat=best_repeat,
                )
            )
            i += best_length * best_repeat
        else:
            segment = segments[i]
            items.append(
                SkeletonItem(
                    cls=(segment.cls,), target=segment.target, label=segment.label
                )
            )
            i += 1
    return tuple(items)


def build_version(
    *,
    template_id: str,
    routine_id: str,
    model: str,
    options: Mapping[str, Any],
    window: Sequence[Episode],
    first_task_id: str | None,
    parent: Template | None,
    params: Params,
    harness_version: str,
    created_at: float,
) -> Template:
    """The next online version of a template, active, from *window*.

    *window* is the template's latest episodes whose status the statistics
    use (this run's included), oldest first. The statistics come from it
    (:func:`compute_stats`); the skeleton is its completed episodes'
    medoid, or the parent's skeleton unchanged when it came from a sleep
    job. When the medoid would compare more than :data:`MEDOID_MAX_CELLS`
    token pairs, the parent's skeleton is kept, or a first version takes
    its latest completed episode's.
    """
    stats = compute_stats(window, first_task_id, params)
    skeleton_source = "online"
    completed = [e for e in window if e.completed]
    if parent is not None and "sleep" in (parent.source, parent.skeleton_source):
        skeleton, medoid_id = parent.skeleton, parent.medoid
        skeleton_source = "sleep"
    elif parent is not None and medoid_cells(completed) > MEDOID_MAX_CELLS:
        skeleton, medoid_id = parent.skeleton, parent.medoid
    else:
        if medoid_cells(completed) > MEDOID_MAX_CELLS:
            chosen: Episode | None = completed[-1]
        else:
            chosen = medoid(completed)
        skeleton = collapse(chosen.segments) if chosen is not None else ()
        medoid_id = chosen.task_id if chosen is not None else None
    return Template(
        template_id=template_id,
        version=1 if parent is None else parent.version + 1,
        routine_id=routine_id,
        model=model,
        options=dict(options),
        status="active",
        source="online",
        admitted_by="harness",
        source_ids=tuple(e.task_id for e in window),
        skeleton=skeleton,
        medoid=medoid_id,
        stats=stats,
        params=params,
        created_at=created_at,
        parent_version=None if parent is None else parent.version,
        harness_version=harness_version,
        skeleton_source=skeleton_source,
    )


def model_demoted(scores: Sequence[float], params: Params) -> bool:
    """Whether a model's latest shadow scores (newest last) demote it."""
    latest = list(scores)[-params.demotion_window :]
    if len(latest) < params.demotion_min:
        return False
    return statistics.fmean(latest) > params.demotion


@dataclass(frozen=True)
class TemplateView:
    """What a run's lookup found: the template a display rests on.

    ``level`` is what may be shown (section 6.1) and ``reason`` why not
    more; ``match`` is how the routine was found (``exact``, ``routine``
    or ``similar``); ``title`` names the routine a similar match borrowed
    from.
    """

    level: int
    reason: str
    match: str
    routine_id: str
    template_id: str
    version: int
    model: str
    stats: Stats
    skeleton: tuple[SkeletonItem, ...]
    params: Params = field(default=DEFAULT_PARAMS)
    title: str | None = None

    def basis(self) -> dict[str, Any]:
        """Which memory the display rests on; ``title`` names a similar
        match's routine (``None`` otherwise)."""
        return {
            "routine_id": self.routine_id,
            "template_id": self.template_id,
            "version": self.version,
            "match": self.match,
            "model": self.model,
            "title": self.title,
        }


def decide_level(
    template: Template,
    *,
    match: str,
    model: str,
    model_scores: Sequence[float],
) -> tuple[int, str]:
    """The level a run may show from *template*, and why not more.

    Level 2 needs an exact or named match, a known model id, the
    version's gate (its statistics) and a model that is not demoted.
    Every other case with a template is level 1.
    """
    if match == "similar":
        return 1, "similar task"
    if not known_model(model):
        return 1, "no model id"
    if not template.stats.gate:
        return 1, template.stats.gate_reason or "gate"
    if model_demoted(model_scores, template.params):
        return 1, "model demoted"
    return 2, ""


def view_of(
    template: Template,
    *,
    match: str,
    model: str,
    model_scores: Sequence[float],
    title: str | None = None,
) -> TemplateView:
    """The :class:`TemplateView` of *template* for a run of *model*."""
    level, reason = decide_level(
        template, match=match, model=model, model_scores=model_scores
    )
    return TemplateView(
        level=level,
        reason=reason,
        match=match,
        routine_id=template.routine_id,
        template_id=template.template_id,
        version=template.version,
        model=template.model,
        stats=template.stats,
        skeleton=template.skeleton,
        params=template.params,
        title=title,
    )
