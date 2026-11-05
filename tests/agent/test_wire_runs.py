"""The document of a scripted run (``wire_runs``): ids and long texts are normalized."""

from __future__ import annotations

import hashlib

from localmelo.tests.agent.wire_runs import LONG_TEXT_CHARS, LONG_TEXT_KEPT, Normalizer


def test_ids_become_ordinals_in_order_of_first_appearance() -> None:
    task, step_1, step_2 = "a" * 32, "b" * 32, "c" * 32
    call = "call_" + "d" * 24
    normalize = Normalizer(tasks=[task], steps=[step_2, step_1])
    assert normalize(
        {
            "calls": [call, f"answers {call}"],
            "task": task,
            "steps": [step_1, step_2, step_1],
            "other": "e" * 32,
            # A digest is not an id.
            "sha256": "f" * 64,
        }
    ) == {
        "calls": ["<call:1>", "answers <call:1>"],
        "task": "<task:1>",
        "steps": ["<step:1>", "<step:2>", "<step:1>"],
        "other": "<id:1>",
        "sha256": "f" * 64,
    }


def test_a_long_text_keeps_its_start_length_and_digest() -> None:
    text = "x" * (LONG_TEXT_CHARS + 1)
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert Normalizer()([text, "x" * LONG_TEXT_CHARS]) == [
        f"{'x' * LONG_TEXT_KEPT}… [{LONG_TEXT_CHARS + 1} chars, sha256 {digest}]",
        "x" * LONG_TEXT_CHARS,
    ]
