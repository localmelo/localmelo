"""Compare the note gate's rules on a labelled set of notes, offline.

Run by hand, never by the tests (the meaning arms load an embedding model)::

    uv run python scripts/eval_note_gate.py
    uv run --with torch --with transformers python scripts/eval_note_gate.py \\
        --model /path/to/embedding-model-dir

Each case in ``tests/memory/data/note_gate_restate.json`` is a note, the texts the
user typed, and whether the gate should admit it: ``restate`` (the note says what
the user typed in other words: admit), ``swap`` (the same with one detail changed),
``added`` (a detail the user never said) and ``inferred`` (a conclusion the user
did not state), all refuse. ``crosslang`` cases (a note in the other language) are
reported on their own. No chat model is called.

The arms, as the chat's ``remember`` tool applies them:

``T0``  the typed rule before CJK wording was told from details: every Chinese or
        Japanese character or pair must be found, no slack.
``T1``  the typed rule now (:func:`records.unsupported`).
``M``   the ``meaning`` level (``MemoryService._admit`` and ``_by_meaning``): the
        words check, then the key tokens, then the embedding's cosine to the
        typed windows at least ``--cosine``. Needs ``--model``.
``M1``  ``M`` with ``T1`` as its words check.

The embedding model is read from a local directory, nothing is downloaded or
written, and no model server is called. Texts are embedded as the chat embeds a
note and its windows (``query=False``: the text as it is, no instruction).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localmelo.support.memory import note_gate, records

DATA = (
    Path(__file__).resolve().parent.parent / "tests/memory/data/note_gate_restate.json"
)
ARMS = ("T0", "T1", "M", "M1")
# What each kind should get: admitted (a restatement) or refused.
KINDS = ("restate", "swap", "added", "inferred", "crosslang")


@dataclass(frozen=True)
class Verdict:
    """One arm's decision on one case, and why."""

    admitted: bool
    why: str


@contextlib.contextmanager
def old_values() -> Iterator[None]:
    """:func:`records.value_terms` as it was before CJK wording: with no evidence
    texts to look characters up in, every CJK term is a value."""
    original = records.value_terms
    records.value_terms = lambda note, evidence=None: original(note)  # type: ignore[assignment]
    try:
        yield
    finally:
        records.value_terms = original  # type: ignore[assignment]


def typed_old(note: str, said: Sequence[str]) -> list[str]:
    with old_values():
        return records.unsupported(note, said)


def typed_new(note: str, said: Sequence[str]) -> list[str]:
    return records.unsupported(note, said)


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
) -> Verdict:
    """``MemoryService._admit`` at the ``meaning`` level, for typed sources only."""
    note, said = case["note"], case["said"]
    if note_gate.is_instruction(note):
        return Verdict(False, "instruction")
    missing = words(note, said)
    if not missing:
        return Verdict(True, "words")
    keys = note_gate.unsupported_keys(note, said)
    if keys:
        return Verdict(False, "key: " + ", ".join(keys[:3]))
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
        del out["M"], out["M1"]
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
        for arm, rule in (("T0", typed_old), ("T1", typed_new)):
            missing = rule(note, said)
            out[arm][case["id"]] = Verdict(not missing, ", ".join(missing[:6]))
        if embedder is not None:
            out["M"][case["id"]] = by_meaning(case, typed_old, embedder, min_cosine)
            out["M1"][case["id"]] = by_meaning(case, typed_new, embedder, min_cosine)
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument(
        "--model", type=Path, help="embedding model directory (arms M, M1)"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cosine", type=float, default=note_gate.MIN_COSINE)
    parser.add_argument("--summary", action="store_true", help="the table only")
    parser.add_argument("--json", type=Path, help="write every verdict here")
    args = parser.parse_args()

    cases = json.loads(args.data.read_text(encoding="utf-8"))
    embedder = Embedder(args.model, args.threads) if args.model else None
    if embedder is None:
        print("No --model: arms M and M1 skipped.\n")
    results = run(cases, embedder, args.cosine)

    print(f"{len(cases)} cases, arms {', '.join(results)}\n")
    print("\n".join(table(cases, results)))
    if not args.summary:
        print("\n".join(mistakes(cases, results)))
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    arm: {k: [v.admitted, v.why] for k, v in verdicts.items()}
                    for arm, verdicts in results.items()
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
