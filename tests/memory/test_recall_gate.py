from typing import Any

import pytest

from localmelo.melo.agent.options import AgentOptions
from localmelo.melo.memory.recall_gate import (
    digest_has_evidence,
    digest_task_key,
    gate_long_docs,
    is_task_digest,
    same_task,
)
from localmelo.melo.memory.retrieval.base import Doc


def doc(name: str, score: float, key: str = "", backend: str = "bm25") -> Doc:
    return Doc(
        f"long:{name}",
        "long",
        name,
        meta={
            "score": score,
            "retrieval": backend,
            "source": "digest",
            "kind": "episodic",
            "provenance": {"task_key": key},
        },
    )


def test_task_key_matches_normalized_digest_task_line() -> None:
    assert digest_task_key("# Task\n\nSolve this problem.") == digest_task_key(
        "solve   THIS problem."
    )
    assert digest_task_key("") == ""
    assert digest_task_key("Solve this problem.") != digest_task_key("Another task.")


def test_task_key_includes_the_workspace_realpath(tmp_path: Any) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    assert digest_task_key("the same task", str(first)) == digest_task_key(
        "the same task", str(first / ".")
    )
    assert digest_task_key("the same task", str(first)) != digest_task_key(
        "the same task", str(second)
    )
    assert digest_task_key("the same task") != digest_task_key(
        "the same task", str(first)
    )


def test_task_key_redacts_secrets_before_hashing() -> None:
    first = "deploy with sk-" + "a" * 24
    second = "deploy with sk-" + "b" * 24
    assert digest_task_key(first) == digest_task_key(second)
    assert digest_task_key(first) != digest_task_key("deploy with no secret")


def test_warm_gate_prioritizes_same_and_requires_comparable_stronger_cross() -> None:
    docs = [
        doc("cross", 11.0),
        doc("same", 10.0, "key"),
        doc("weaker", 10.9),
        doc("unmatched-backend", 0.9, backend="embedding"),
    ]
    kept = gate_long_docs(docs, "key", ratio=1.1, cap=1, cold=1)
    assert [d.doc_id for d in kept] == ["long:same", "long:cross"]


def test_a_cross_task_digest_below_the_ratio_is_dropped() -> None:
    # Cap 2 leaves room, so only the ratio can drop the cross digest.
    docs = [doc("same", 10.0, "key"), doc("just-under", 10.999)]
    assert [
        d.doc_id for d in gate_long_docs(docs, "key", ratio=1.1, cap=2, cold=1)
    ] == ["long:same"]
    docs.append(doc("at-threshold", 11.0))
    assert [
        d.doc_id for d in gate_long_docs(docs, "key", ratio=1.1, cap=2, cold=1)
    ] == [
        "long:same",
        "long:at-threshold",
    ]


def test_cold_gate_and_unknown_keys() -> None:
    docs = [doc("first", 5), doc("second", 4)]
    assert [
        d.doc_id for d in gate_long_docs(docs, "key", ratio=1.1, cap=1, cold=1)
    ] == ["long:first"]
    assert gate_long_docs(docs, "key", ratio=1.1, cap=1, cold=0) == []
    assert gate_long_docs(docs, "", ratio=1.1, cap=0, cold=1) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("task: x\nfiles: a.py", True),
        ("task: x\nverified: pytest passed", True),
        ("task: x\nfiles:   ", False),
        ("task: x\nverified:", False),
        ("task: x\nfound: files: a.py", False),
        ("task: x\noutcome: completed", False),
    ],
)
def test_digest_evidence_needs_a_nonempty_files_or_verified_line(
    text: str, expected: bool
) -> None:
    assert digest_has_evidence(text) is expected


def test_general_memories_bypass_digest_gate() -> None:
    fact = Doc(
        "long:fact",
        "long",
        "a remembered fact",
        meta={"source": "user", "kind": "semantic", "score": 2.0},
    )
    assert gate_long_docs(
        [fact, doc("old-digest", 4)], "key", ratio=1.1, cap=1, cold=0
    ) == [fact]


def test_a_reasoning_note_is_not_a_task_digest() -> None:
    # Notes, drafts and reflections bypass the gate even when their
    # provenance happens to carry the same task key.
    note = Doc(
        "long:note",
        "long",
        "a task-local reasoning note",
        meta={
            "source": "reasoning_note",
            "kind": "episodic",
            "score": 2.0,
            "provenance": {"task_key": "key"},
        },
    )
    assert is_task_digest(note) is False
    assert same_task(note, "key") is False
    kept = gate_long_docs(
        [note, doc("same", 3.0, "key"), doc("cross", 2.5)],
        "key",
        ratio=1.1,
        cap=1,
        cold=1,
    )
    assert [item.doc_id for item in kept] == ["long:same", "long:note"]


def test_the_cap_counts_only_the_qualifying_cross_digests() -> None:
    docs = [
        doc("same", 10.0, "key"),
        doc("cross-low", 12.0),
        doc("cross-high", 13.0),
    ]
    kept = gate_long_docs(docs, "key", ratio=1.1, cap=1, cold=1)
    assert [item.doc_id for item in kept] == ["long:same", "long:cross-high"]


def test_the_ratio_is_compared_within_one_backend() -> None:
    docs = [
        doc("same-bm25", 0.5, "key", backend="bm25"),
        doc("same-embedding", 0.9, "key", backend="embedding"),
        doc("cross-bm25", 0.7, backend="bm25"),
    ]
    kept = gate_long_docs(docs, "key", ratio=1.1, cap=1, cold=1)
    # A global maximum would compare 0.7 against 0.9 and drop it.
    assert [item.doc_id for item in kept] == [
        "long:same-embedding",
        "long:same-bm25",
        "long:cross-bm25",
    ]


def test_an_exact_ratio_boundary_survives_binary_floating_point() -> None:
    docs = [doc("same", 0.8, "key"), doc("at-boundary", 0.88)]
    kept = gate_long_docs(docs, "key", ratio=1.1, cap=1, cold=1)
    assert [item.doc_id for item in kept] == ["long:same", "long:at-boundary"]
    docs.append(doc("just-under", 0.8799))
    kept = gate_long_docs(docs, "key", ratio=1.1, cap=1, cold=1)
    assert [item.doc_id for item in kept] == ["long:same", "long:at-boundary"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cross_task_ratio": float("nan")},
        {"cross_task_ratio": 0.9},
        {"cross_task_cap": 3},
        {"cross_task_cold": True},
    ],
)
def test_gate_options_validate(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AgentOptions(**kwargs)
