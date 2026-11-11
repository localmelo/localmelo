"""Compare the words rule of the note gate on a labelled set of notes, offline.

Run by hand, never by the tests::

    uv run python scripts/eval_note_gate.py

Each case in ``tests/memory/data/note_gate_restate.json`` and ``note_gate_heldout.json``
(the first set shaped the rule; the held-out one was written before the second round of
changes and not edited after) is a note, the texts the user typed, and whether the gate
should admit it: ``restate`` (the note says what the user typed in other words: admit),
``swap`` (the same with one detail changed), ``added`` (a detail the user never said)
and ``inferred`` (a conclusion the user did not state), all refuse. ``crosslang`` cases
(a note in the other language) are reported on their own. A chat model is called
only when both ``--judge-url`` and ``--judge-model`` are supplied.

The arms, as the chat's ``remember`` tool applies them (``T0``, ``T1`` and ``T1b`` are
the words rule that checks a note against what the user typed; ``J`` needs
``--judge-url`` and ``--judge-model``):

``T0``   every Chinese or Japanese character or pair must be found, no slack (the rule
         before CJK wording was told from details).
``T1``   a CJK term is a value only when quoted, a number or an untyped character that
         is no common wording; the rest counts toward the share of words found, a pair
         as much as a word.
``T1b``  ``T1`` with the share counted over words and single CJK characters only
         (:func:`records.unsupported` is this one while it is the shipped rule).
``J``    ``T1b``, then a chat judge only for refusals due solely to untyped CJK
         terms, never an order, a missing quoted span, number, weekday or relative
         date, Latin value or Korean word. Reports how many notes reached the
         judge. The prompt is frozen before measurement.

Ship rule for ``J`` on the held-out set: admit at least 85% of zh restatements
and refuse zh swaps no less often than ``T1b``'s 85%. ``ChatMemory.remember`` applies
this arm with the chat's own model as the judge.

Nothing is downloaded or written, and no model server is called unless the judge is
explicitly enabled.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localmelo.support.memory import note_judge, records
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

DATA = [
    Path(__file__).resolve().parent.parent / f"tests/memory/data/note_gate_{name}.json"
    for name in ("restate", "heldout")
]
ARMS = ("T0", "T1", "T1b")
# What each kind should get: admitted (a restatement) or refused.
KINDS = ("restate", "swap", "added", "inferred", "crosslang")


@dataclass(frozen=True)
class Verdict:
    """One arm's decision on one case, and why."""

    admitted: bool
    why: str


def typed(
    note: str, said: Sequence[str], *, old_values: bool = False, chars: bool = False
) -> list[str]:
    """The typed rule (:func:`records.unsupported`) as a comparison runs it: with
    *old_values* every CJK term is a value; with *chars* the share of words found
    counts words and single CJK characters, not pairs."""
    gone = records.missing_terms(note, said)
    if not gone:
        return []
    valued = (
        records.value_terms(note) if old_values else records.value_terms(note, said)
    )
    values = [term for term in gone if term in valued]
    others = [term for term in records.note_terms(note) if term not in valued]
    lost = [term for term in gone if term not in values]

    def count(terms: Sequence[str]) -> int:
        return sum(1 for t in terms if not chars or t.isascii() or len(t) == 1)

    if not values and count(others) - count(lost) >= records.WORDS_FOUND * count(
        others
    ):
        return []
    return values + lost


def typed_t0(note: str, said: Sequence[str]) -> list[str]:
    return typed(note, said, old_values=True)


def typed_t1(note: str, said: Sequence[str]) -> list[str]:
    return typed(note, said)


def typed_t1b(note: str, said: Sequence[str]) -> list[str]:
    return typed(note, said, chars=True)


def run(cases: list[dict[str, Any]]) -> dict[str, dict[str, Verdict]]:
    """``{arm: {case id: verdict}}`` for the words arms."""
    out: dict[str, dict[str, Verdict]] = {arm: {} for arm in ARMS}
    for case in cases:
        note, said = case["note"], case["said"]
        for arm, rule in (("T0", typed_t0), ("T1", typed_t1), ("T1b", typed_t1b)):
            missing = rule(note, said)
            out[arm][case["id"]] = Verdict(not missing, ", ".join(missing[:6]))
    return out


async def run_j(
    cases: list[dict[str, Any]],
    baseline: dict[str, Verdict],
    judge: Callable[[str, Sequence[str]], Awaitable[bool]],
) -> tuple[dict[str, Verdict], int]:
    """Arm J: keep T1b's decisions unless a CJK-only refusal gets a clear yes.

    Return the verdicts and the number of notes sent to the judge, not the
    number it admitted. The cases run serially.
    """
    out = dict(baseline)
    reached = 0
    for case in cases:
        note, said = case["note"], case["said"]
        if baseline[case["id"]].admitted or not note_judge.cjk_only_refusal(note, said):
            continue
        reached += 1
        admitted = await judge(note, said)
        out[case["id"]] = Verdict(admitted, "judge: yes" if admitted else "judge: no")
    return out, reached


def right(case: dict[str, Any], verdict: Verdict) -> bool:
    return bool(verdict.admitted == (case["label"] == "admit"))


def share(hits: int, total: int) -> str:
    return f"{hits}/{total} ({100 * hits / total:.0f}%)" if total else "-"


def table(
    cases: list[dict[str, Any]], results: dict[str, dict[str, Verdict]]
) -> list[str]:
    """Rates per kind and language: the admit rate where the label is admit, the
    refuse rate where it is refuse."""
    arms = list(results)
    langs = sorted({c["lang"] for c in cases})
    lines = ["| kind | lang | n | rate of | " + " | ".join(arms) + " |"]
    lines.append("|" + "---|" * (4 + len(arms)))
    for kind in KINDS:
        for lang in [*langs, "all"]:
            if kind == "crosslang":
                parts = {
                    label: [
                        c
                        for c in cases
                        if c["kind"] == kind
                        and c["label"] == label
                        and lang in (c["lang"], "all")
                    ]
                    for label in ("admit", "refuse")
                }
            else:
                label = "admit" if kind == "restate" else "refuse"
                parts = {
                    label: [
                        c
                        for c in cases
                        if c["kind"] == kind and lang in (c["lang"], "all")
                    ]
                }
            for label, group in parts.items():
                if not group:
                    continue
                rate_of = "admit" if label == "admit" else "refuse"
                row = [kind, lang, str(len(group)), rate_of]
                for arm in arms:
                    hits = sum(right(c, results[arm][c["id"]]) for c in group)
                    row.append(share(hits, len(group)))
                lines.append("| " + " | ".join(row) + " |")
            if kind == "swap":
                # A swap says little when the arm refuses its base restatement
                # already: count the swaps whose base the arm admits.
                group = [c for c in parts["refuse"] if c.get("base")]
                if group:
                    row = ["swap, base admitted", lang, str(len(group)), "refuse"]
                    for arm in arms:
                        eligible = [
                            c for c in group if results[arm][c["base"]].admitted
                        ]
                        hits = sum(right(c, results[arm][c["id"]]) for c in eligible)
                        row.append(share(hits, len(eligible)))
                    lines.append("| " + " | ".join(row) + " |")
    return lines


def mistakes(
    cases: list[dict[str, Any]], results: dict[str, dict[str, Verdict]]
) -> list[str]:
    lines: list[str] = []
    for arm, verdicts in results.items():
        refused = [
            c for c in cases if c["label"] == "admit" and not verdicts[c["id"]].admitted
        ]
        admitted = [
            c for c in cases if c["label"] == "refuse" and verdicts[c["id"]].admitted
        ]
        lines.append(f"\n## {arm}: false refusals ({len(refused)})")
        for c in refused:
            lines.append(
                f"- {c['id']} [{c['kind']}] {c['note']}\n    {verdicts[c['id']].why}"
            )
        lines.append(f"\n## {arm}: false admissions ({len(admitted)})")
        for c in admitted:
            lines.append(
                f"- {c['id']} [{c['kind']}] {c['note']}\n    {verdicts[c['id']].why}"
            )
    return lines


async def evaluate(args: argparse.Namespace) -> None:
    provider = (
        OpenAICompatLLM(args.judge_url, args.judge_model) if args.judge_url else None
    )
    try:
        if provider is None:
            print("No --judge-url/--judge-model: arm J skipped.\n")

        async def judge(note: str, said: Sequence[str]) -> bool:
            assert provider is not None
            return await note_judge.judge(provider, note, said)

        everything: dict[str, dict[str, dict[str, list[Any]]]] = {}
        for path in args.data:
            cases = json.loads(path.read_text(encoding="utf-8"))
            results = run(cases)
            if provider is not None:
                results["J"], reached = await run_j(cases, results["T1b"], judge)
            print(f"## {path.name}: {len(cases)} cases, arms {', '.join(results)}\n")
            if provider is not None:
                print(f"J: {reached}/{len(cases)} notes reached the judge.\n")
            print("\n".join(table(cases, results)))
            if not args.summary:
                print("\n".join(mistakes(cases, results)))
            print()
            everything[path.name] = {
                arm: {k: [v.admitted, v.why] for k, v in verdicts.items()}
                for arm, verdicts in results.items()
            }
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(everything, ensure_ascii=False, indent=1), encoding="utf-8"
            )
    finally:
        if provider is not None:
            await provider.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data", type=Path, nargs="+", default=DATA, help="one table per file"
    )
    parser.add_argument("--judge-url", help="chat server URL (arm J)")
    parser.add_argument("--judge-model", help="chat model identifier (arm J)")
    parser.add_argument("--summary", action="store_true", help="the tables only")
    parser.add_argument("--json", type=Path, help="write every verdict here")
    args = parser.parse_args()

    if bool(args.judge_url) != bool(args.judge_model):
        parser.error("--judge-url and --judge-model must be supplied together")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
