import pytest

from localmelo.melo.agent.options import AgentOptions
from localmelo.melo.memory.recall_gate import digest_task_key, gate_long_docs
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


def test_warm_gate_prioritizes_same_and_requires_comparable_stronger_cross() -> None:
    docs = [
        doc("cross", 11.0),
        doc("same", 10.0, "key"),
        doc("weaker", 10.9),
        doc("unmatched-backend", 0.9, backend="embedding"),
    ]
    kept = gate_long_docs(docs, "key", ratio=1.1, cap=1, cold=1)
    assert [d.doc_id for d in kept] == ["long:same", "long:cross"]


def test_cold_gate_and_unknown_keys() -> None:
    docs = [doc("first", 5), doc("second", 4)]
    assert [
        d.doc_id for d in gate_long_docs(docs, "key", ratio=1.1, cap=1, cold=1)
    ] == ["long:first"]
    assert gate_long_docs(docs, "key", ratio=1.1, cap=1, cold=0) == []
    assert gate_long_docs(docs, "", ratio=1.1, cap=0, cold=1) == []


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
