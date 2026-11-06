"""Compare the note gate's rules on a labelled set of notes, offline.

Run by hand, never by the tests (the meaning arms load an embedding model)::

    uv run python scripts/eval_note_gate.py
    uv run --with torch --with transformers python scripts/eval_note_gate.py \\
        --model /path/to/embedding-model-dir

Each case in ``tests/memory/data/note_gate_restate.json`` and ``note_gate_heldout.json``
(the first set shaped the rule; the held-out one was written before the second round of
changes and not edited after) is a note, the texts the user typed, and whether the gate
should admit it: ``restate`` (the note says what the user typed in other words: admit),
``swap`` (the same with one detail changed), ``added`` (a detail the user never said)
and ``inferred`` (a conclusion the user did not state), all refuse. ``crosslang`` cases
(a note in the other language) are reported on their own. A chat model is called
only when both ``--judge-url`` and ``--judge-model`` are supplied.

The arms, as the chat's ``remember`` tool applies them (``T0``, ``T1`` and ``T1b`` are
the typed rule, the default; the meaning arms need ``--model``, and ``J`` needs
``--judge-url`` and ``--judge-model``):

``T0``   every Chinese or Japanese character or pair must be found, no slack (the rule
         before CJK wording was told from details).
``T1``   a CJK term is a value only when quoted, a number or an untyped character that
         is no common wording; the rest counts toward the share of words found, a pair
         as much as a word.
``T1b``  ``T1`` with the share counted over words and single CJK characters only
         (:func:`records.unsupported` is this one while it is the shipped rule).
``M``    ``MemoryService._admit`` and ``_by_meaning`` as they were: the words check
         (``T1``), then the key tokens (numbers, dates, addresses, Latin names), then
         the embedding's cosine to the typed windows at least ``--cosine``.
``M2``   ``M`` with key tokens that also read number words and a name after a title
         ("Dr. Osei"), and a check that the note's Chinese or Japanese values
         (``unheld_cjk_values``) are found, before the cosine. A candidate: it is
         measured here and is not part of the shipped ``meaning`` level.
``M2b``  ``M2`` with ``T1b`` as its words check.
``J``    ``T1b``, then a chat judge only for refusals due solely to untyped CJK
         terms, never a missing quoted span, number or Latin value. Reports how
         many notes reached the judge. The prompt is frozen before measurement.

Ship rule for ``J`` on the held-out set: admit at least 85% of zh restatements
and refuse zh swaps no less often than ``T1b``'s 85%. ``ChatMemory.remember`` applies
this arm at the default level, with the chat's own model as the judge.

The embedding model is read from a local directory, nothing is downloaded or
written, and no model server is called unless the judge is explicitly enabled.
Texts are embedded as the chat embeds a note and its windows (``query=False``:
the text as it is, no instruction).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import re
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localmelo.support.memory import note_gate, note_judge, records
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

DATA = [
    Path(__file__).resolve().parent.parent / f"tests/memory/data/note_gate_{name}.json"
    for name in ("restate", "heldout")
]
ARMS = ("T0", "T1", "T1b", "M", "M2", "M2b")
# What ``M2`` changes about the key tokens: a sentence does not end after a title
# ("Dr. Osei"), and number words count as the digits do ("six" is 6).
TITLE_SENTENCE = re.compile(
    r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bProf\.)"
    r"(?<=[.!?])\s+|\n|[。！？]"
)
UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
)
NUMBER_WORDS = {word: str(n) for n, word in enumerate(UNITS.split())} | {
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
}
WORD_OF = {digits: word for word, digits in NUMBER_WORDS.items()}
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
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


@contextlib.contextmanager
def title_sentences() -> Iterator[None]:
    """The key tokens with a sentence that does not end after a title."""
    saved = note_gate._SENTENCE
    note_gate._SENTENCE = TITLE_SENTENCE
    try:
        yield
    finally:
        note_gate._SENTENCE = saved


def unheld_keys(note: str, said: Sequence[str]) -> list[str]:
    """The key tokens of *note* that *said* lacks, as ``M2`` reads them: the usual
    ones (names after a title included), and the number words, each held by the
    word or its digits; a digit key is held by its word too."""
    folded = [note_gate.fold(text) for text in said]

    def digits(norm: str) -> bool:
        pattern = rf"(?<!\d)(?<!\d\.){re.escape(norm)}(?!\d)(?!\.\d)"
        return any(re.search(pattern, text) for text in folded)

    def spelled(norm: str) -> bool:
        word = WORD_OF.get(norm)
        pattern = rf"(?<![^\W\d_]){word}(?![^\W\d_])"
        return word is not None and any(re.search(pattern, t) for t in folded)

    with title_sentences():
        usual = [k for k in note_gate.unsupported_keys(note, said) if not spelled(k)]
    for word in re.findall(r"[^\W\d_]+", note):
        norm = NUMBER_WORDS.get(word.lower())
        if norm and not (digits(norm) or spelled(norm)) and word not in usual:
            usual.append(word)
    return usual


def unheld_cjk_values(note: str, said: Sequence[str]) -> list[str]:
    """The Chinese or Japanese values of *note* (quoted, a number character, or an
    untyped character that is no common wording) that *said* lacks; none when
    nothing typed is in a CJK script, as there are no characters to compare."""
    if not any(CJK.search(text) for text in said):
        return []
    valued = records.value_terms(note, said)
    return [
        term
        for term in records.missing_terms(note, said)
        if term in valued and not term.isascii()
    ]


class Embedder:
    """The embedding model read from a local directory: last-token pooling over
    left-padded batches, L2-normalised, on the CPU."""

    def __init__(self, model_dir: Path, threads: int) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        self.torch.set_num_threads(threads)
        self.tok = transformers.AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, padding_side="left"
        )
        self.model = (
            transformers.AutoModel.from_pretrained(
                str(model_dir), local_files_only=True, dtype=self.torch.float32
            )
            .to("cpu")
            .eval()
        )
        self.end = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.cache: dict[str, list[float]] = {}

    def embed(self, texts: Sequence[str], batch: int = 1) -> None:
        """Embed every text not yet cached (one at a time by default, so no
        padding bends a vector)."""
        todo = sorted({t for t in texts if t not in self.cache}, key=len)
        for start in range(0, len(todo), batch):
            part = todo[start : start + batch]
            # The tokenizer ends each text with the end-of-text token the model
            # pools on; padding goes on the left.
            enc = self.tok(part, padding=True, truncation=True, max_length=512)
            if any(row[-1] != self.end for row in enc["input_ids"]):
                raise RuntimeError("a text does not end with the end-of-text token")
            tensors = {k: self.torch.tensor(v) for k, v in enc.items()}
            with self.torch.no_grad():
                hidden = self.model(**tensors).last_hidden_state
            pooled = hidden[:, -1]  # left padding: the last position is the last token
            pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            for text, vector in zip(part, pooled.tolist(), strict=True):
                self.cache[text] = vector

    def vector(self, text: str) -> list[float]:
        return self.cache[text]


def typed_windows(said: Sequence[str]) -> list[str]:
    return note_gate.recent_windows(list(said), note_gate.TYPED_WINDOWS)


def by_meaning(
    case: dict[str, Any],
    words: Callable[[str, Sequence[str]], list[str]],
    embedder: Embedder,
    min_cosine: float,
    *,
    detailed: bool,
) -> Verdict:
    """``MemoryService._admit`` at the ``meaning`` level, for typed sources only;
    *detailed* is the key tokens with number words and titles and the CJK value
    check, off for the level as it was."""
    note, said = case["note"], case["said"]
    if note_gate.is_instruction(note):
        return Verdict(False, "instruction")
    missing = words(note, said)
    if not missing:
        return Verdict(True, "words")
    keys = (
        unheld_keys(note, said) if detailed else note_gate.unsupported_keys(note, said)
    )
    if keys:
        return Verdict(False, "key: " + ", ".join(keys[:3]))
    if detailed and (details := unheld_cjk_values(note, said)):
        return Verdict(False, "detail: " + ", ".join(details[:3]))
    windows = typed_windows(said)
    if not windows:
        return Verdict(False, "no window; words missing: " + ", ".join(missing[:3]))
    best = max(
        note_gate.cosine(embedder.vector(note), embedder.vector(w)) for w in windows
    )
    if best >= min_cosine:
        return Verdict(True, f"meaning {best:.2f}")
    return Verdict(False, f"far {best:.2f}; words missing: {', '.join(missing[:3])}")


def run(
    cases: list[dict[str, Any]], embedder: Embedder | None, min_cosine: float
) -> dict[str, dict[str, Verdict]]:
    """``{arm: {case id: verdict}}`` for every arm that can run."""
    out: dict[str, dict[str, Verdict]] = {arm: {} for arm in ARMS}
    if embedder is None:
        for arm in ("M", "M2", "M2b"):
            del out[arm]
    else:
        embedder.embed(
            [
                text
                for case in cases
                for text in [case["note"], *typed_windows(case["said"])]
            ]
        )
    for case in cases:
        note, said = case["note"], case["said"]
        for arm, rule in (("T0", typed_t0), ("T1", typed_t1), ("T1b", typed_t1b)):
            missing = rule(note, said)
            out[arm][case["id"]] = Verdict(not missing, ", ".join(missing[:6]))
        if embedder is not None:
            for arm, rule, detailed in (
                ("M", typed_t1, False),
                ("M2", typed_t1, True),
                ("M2b", typed_t1b, True),
            ):
                out[arm][case["id"]] = by_meaning(
                    case, rule, embedder, min_cosine, detailed=detailed
                )
    return out


async def run_j(
    cases: list[dict[str, Any]],
    baseline: dict[str, Verdict],
    judge: Callable[[str, Sequence[str]], Awaitable[bool]],
) -> tuple[dict[str, Verdict], int]:
    """Arm J: keep T1b's decisions unless a CJK-only refusal gets a clear yes.

    Return the verdicts and the number of notes sent to the judge, not the
    number it admitted. No embedding is needed and the cases run serially.
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
    embedder = Embedder(args.model, args.threads) if args.model else None
    provider = (
        OpenAICompatLLM(args.judge_url, args.judge_model) if args.judge_url else None
    )
    try:
        if embedder is None:
            print("No --model: the meaning arms (M, M2, M2b) skipped.\n")
        if provider is None:
            print("No --judge-url/--judge-model: arm J skipped.\n")

        async def judge(note: str, said: Sequence[str]) -> bool:
            assert provider is not None
            return await note_judge.judge(provider, note, said)

        everything: dict[str, dict[str, dict[str, list[Any]]]] = {}
        for path in args.data:
            cases = json.loads(path.read_text(encoding="utf-8"))
            results = run(cases, embedder, args.cosine)
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
    parser.add_argument(
        "--model", type=Path, help="embedding model directory (arms M, M2, M2b)"
    )
    parser.add_argument("--judge-url", help="chat server URL (arm J)")
    parser.add_argument("--judge-model", help="chat model identifier (arm J)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cosine", type=float, default=note_gate.MIN_COSINE)
    parser.add_argument("--summary", action="store_true", help="the tables only")
    parser.add_argument("--json", type=Path, help="write every verdict here")
    args = parser.parse_args()

    if bool(args.judge_url) != bool(args.judge_model):
        parser.error("--judge-url and --judge-model must be supplied together")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
