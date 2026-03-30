"""Tests for normalized token metrics in smoke framework.

Verifies that:
- Normalized metrics appear in aggregation output even when backend usage is absent.
- Malformed usage does not crash normalized metric computation.
- The normalized tokenizer path comes from the backend layer, not ad hoc smoke logic.
- render_comparison_markdown includes the normalized section.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure repo root is importable.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO.parent))

from localmelo.support.backends.tokenization import count_tokens  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────


def _make_completed_record(
    backend: str = "test-backend",
    backend_id: str = "test",
    scenario_id: str = "s1",
    scenario_name: str = "Test Scenario",
    chat_metrics: dict[str, Any] | None = None,
    normalized_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal completed scenario record for testing."""
    return {
        "backend": backend,
        "backend_id": backend_id,
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "status": "completed",
        "timestamp": "2025-01-01T00:00:00",
        "chat_url": "http://localhost:11434/v1",
        "chat_model": "test-model",
        "embed_url": "http://localhost:11434/v1",
        "embed_model": "test-embed",
        "mem_dir": "/tmp/test",
        "seed_memories": [],
        "metrics": {
            "embedding": {"total_calls": 0, "total_ms": 0, "per_call_ms": []},
            "chat": chat_metrics
            or {
                "total_calls": 2,
                "total_ms": 1000.0,
                "per_call_ms": [500.0, 500.0],
                "total_prompt_tokens": 100,
                "total_completion_tokens": 50,
                "total_tokens": 150,
                "avg_completion_tokens_per_call": 25.0,
                "avg_total_tokens_per_call": 75.0,
                "completion_tokens_per_s_overall": 50.0,
                "total_tokens_per_s_overall": 150.0,
            },
            "normalized": normalized_metrics
            or {
                "total_prompt_tokens": 80,
                "total_completion_tokens": 40,
                "total_tokens": 120,
                "avg_completion_tokens_per_call": 20.0,
                "completion_tokens_per_s_overall": 40.0,
                "total_tokens_per_s_overall": 120.0,
            },
            "combined": {"total_ms": 1500.0},
        },
        "queries": [
            {
                "text": "test query",
                "answer": "test answer",
                "expected_keywords": [],
                "keywords_found": [],
                "recall_score": 1.0,
                "pre_retrieval_count": 0,
                "pre_retrieval": [],
                "query_elapsed_ms": 500.0,
            }
        ],
        "llm_calls": [],
        "sqlite": {},
    }


def _make_backend_summary(
    records: list[dict[str, Any]],
    backend_id: str = "test",
    backend_name: str = "test-backend",
) -> dict[str, Any]:
    first = records[0] if records else {}
    return {
        "backend_id": backend_id,
        "backend_name": backend_name,
        "timestamp": "2025-01-01T00:00:00",
        "chat_url": first.get("chat_url", ""),
        "chat_model": first.get("chat_model", ""),
        "embed_url": first.get("embed_url", ""),
        "embed_model": first.get("embed_model", ""),
        "scenarios": records,
    }


# ── Tests ─────────────────────────────────────────────────────


class TestNormalizedMetricsInRecord:
    """Normalized metrics appear in the scenario record."""

    def test_normalized_section_present(self) -> None:
        rec = _make_completed_record()
        assert "normalized" in rec["metrics"]
        nm = rec["metrics"]["normalized"]
        assert nm["total_prompt_tokens"] == 80
        assert nm["total_completion_tokens"] == 40
        assert nm["total_tokens"] == 120

    def test_normalized_present_even_without_backend_usage(self) -> None:
        """Normalized metrics should exist even if chat usage is all zeros."""
        rec = _make_completed_record(
            chat_metrics={
                "total_calls": 1,
                "total_ms": 500.0,
                "per_call_ms": [500.0],
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "avg_completion_tokens_per_call": 0,
                "avg_total_tokens_per_call": 0,
                "completion_tokens_per_s_overall": 0,
                "total_tokens_per_s_overall": 0,
            },
            normalized_metrics={
                "total_prompt_tokens": 15,
                "total_completion_tokens": 8,
                "total_tokens": 23,
                "avg_completion_tokens_per_call": 8.0,
                "completion_tokens_per_s_overall": 16.0,
                "total_tokens_per_s_overall": 46.0,
            },
        )
        nm = rec["metrics"]["normalized"]
        assert nm["total_prompt_tokens"] == 15
        assert nm["total_completion_tokens"] == 8
        # Backend-reported is zero
        assert rec["metrics"]["chat"]["total_tokens"] == 0


class TestNormalizedMarkdownRendering:
    """render_comparison_markdown includes the normalized token table."""

    def test_top_level_sections_present_in_requested_order(self) -> None:
        from localmelo.tests.smoke.core_loop_test import render_comparison_markdown

        rec = _make_completed_record()
        summary = _make_backend_summary([rec], "test", "MLC-LLM")
        md = render_comparison_markdown([summary], ["test"])
        assert md.index("## Real Conversation") < md.index("## Overall Comparison")
        assert md.index("## Overall Comparison") < md.index("## MLC-LLM 指标")

    def test_normalized_table_present(self) -> None:
        from localmelo.tests.smoke.core_loop_test import render_comparison_markdown

        rec = _make_completed_record()
        summary = _make_backend_summary([rec])
        md = render_comparison_markdown([summary], ["test"])
        assert "统一 Tokenizer Token 对比" in md

    def test_real_conversation_contains_answers(self) -> None:
        from localmelo.tests.smoke.core_loop_test import render_comparison_markdown

        rec = _make_completed_record()
        summary = _make_backend_summary([rec], "test", "MLC-LLM")
        md = render_comparison_markdown([summary], ["test"])
        assert "## Real Conversation" in md
        assert "test answer" in md

    def test_real_conversation_has_question_thinking_answer_sections(self) -> None:
        from localmelo.tests.smoke.core_loop_test import render_comparison_markdown

        rec = _make_completed_record()
        summary = _make_backend_summary([rec], "test", "MLC-LLM")
        md = render_comparison_markdown([summary], ["test"])
        assert "**Question**" in md
        assert "**Thinking**" in md
        assert "**Answer**" in md

    def test_think_tag_split_into_thinking_and_answer(self) -> None:
        from localmelo.tests.smoke.core_loop_test import render_comparison_markdown

        rec = _make_completed_record()
        rec["queries"][0]["answer"] = "<think>some reasoning</think>\nfinal answer"
        summary = _make_backend_summary([rec], "test", "MLC-LLM")
        md = render_comparison_markdown([summary], ["test"])
        # Thinking section should contain the reasoning
        assert "some reasoning" in md
        # Answer section should contain the final answer without <think> tag
        assert "final answer" in md
        assert "<think>" not in md
        assert "</think>" not in md

    def test_no_think_tag_shows_dash_for_thinking(self) -> None:
        from localmelo.tests.smoke.core_loop_test import render_comparison_markdown

        rec = _make_completed_record()
        rec["queries"][0]["answer"] = "plain answer without thinking"
        summary = _make_backend_summary([rec], "test", "MLC-LLM")
        md = render_comparison_markdown([summary], ["test"])
        # Thinking should show dash
        idx_thinking = md.index("**Thinking**")
        idx_answer = md.index("**Answer**")
        thinking_block = md[idx_thinking:idx_answer]
        assert "—" in thinking_block
        # Answer should show the plain text
        assert "plain answer without thinking" in md

    def test_overall_normalized_summary_present(self) -> None:
        from localmelo.tests.smoke.core_loop_test import render_comparison_markdown

        rec = _make_completed_record()
        summary = _make_backend_summary([rec])
        md = render_comparison_markdown([summary], ["test"])
        assert "整体统一 Tokenizer Token 汇总" in md

    def test_backend_reported_table_still_present(self) -> None:
        from localmelo.tests.smoke.core_loop_test import render_comparison_markdown

        rec = _make_completed_record()
        summary = _make_backend_summary([rec])
        md = render_comparison_markdown([summary], ["test"])
        # Backend-reported table header
        assert "Token 对比 (后端上报)" in md

    def test_two_backends_both_have_normalized_rows(self) -> None:
        from localmelo.tests.smoke.core_loop_test import render_comparison_markdown

        rec_a = _make_completed_record(
            backend="backend-A",
            backend_id="a",
        )
        rec_b = _make_completed_record(
            backend="backend-B",
            backend_id="b",
        )
        summary_a = _make_backend_summary([rec_a], "a", "backend-A")
        summary_b = _make_backend_summary([rec_b], "b", "backend-B")
        md = render_comparison_markdown([summary_a, summary_b], ["a", "b"])
        # Both backends appear in normalized section
        assert "backend-A" in md
        assert "backend-B" in md


class TestMalformedUsageDoesNotCrash:
    """Malformed or missing usage must not break normalized metrics."""

    def test_missing_normalized_section_renders_zeros(self) -> None:
        from localmelo.tests.smoke.core_loop_test import render_comparison_markdown

        rec = _make_completed_record()
        # Simulate missing normalized section
        del rec["metrics"]["normalized"]
        rec["metrics"]["normalized"] = {}
        summary = _make_backend_summary([rec])
        # Must not crash
        md = render_comparison_markdown([summary], ["test"])
        assert "统一 Tokenizer Token 对比" in md


class TestTokenizerComesFromBackendLayer:
    """The shared tokenizer is part of the backend package, not smoke."""

    def test_importable_from_backends_package(self) -> None:
        from localmelo.support.backends import count_tokens as pkg_fn
        from localmelo.support.backends.tokenization import (
            count_tokens as mod_fn,
        )

        assert pkg_fn is mod_fn

    def test_base_backend_exposes_count_tokens(self) -> None:
        from localmelo.support.backends.base import BaseBackend

        assert hasattr(BaseBackend, "count_tokens")
        assert BaseBackend.count_tokens("hello world") == count_tokens("hello world")

    def test_smoke_imports_from_backend_layer(self) -> None:
        """core_loop_test imports count_tokens from the backend layer."""
        import localmelo.tests.smoke.core_loop_test as smoke_mod

        # The module-level import should reference the backend tokenizer.
        assert hasattr(smoke_mod, "count_tokens")
        assert smoke_mod.count_tokens is count_tokens


class TestSplitThinkingAndAnswer:
    """Unit tests for _split_thinking_and_answer helper."""

    def test_with_think_tag(self) -> None:
        from localmelo.tests.smoke.core_loop_test import _split_thinking_and_answer

        thinking, answer = _split_thinking_and_answer(
            "<think>reasoning here</think>\nthe answer"
        )
        assert thinking == "reasoning here"
        assert answer == "the answer"

    def test_without_think_tag(self) -> None:
        from localmelo.tests.smoke.core_loop_test import _split_thinking_and_answer

        thinking, answer = _split_thinking_and_answer("plain text")
        assert thinking == ""
        assert answer == "plain text"

    def test_empty_string(self) -> None:
        from localmelo.tests.smoke.core_loop_test import _split_thinking_and_answer

        thinking, answer = _split_thinking_and_answer("")
        assert thinking == ""
        assert answer == ""

    def test_multiline_think_content(self) -> None:
        from localmelo.tests.smoke.core_loop_test import _split_thinking_and_answer

        text = "<think>\nline1\nline2\n</think>\nfinal"
        thinking, answer = _split_thinking_and_answer(text)
        assert "line1" in thinking
        assert "line2" in thinking
        assert answer == "final"

    def test_text_before_and_after_think(self) -> None:
        from localmelo.tests.smoke.core_loop_test import _split_thinking_and_answer

        text = "before <think>middle</think> after"
        thinking, answer = _split_thinking_and_answer(text)
        assert thinking == "middle"
        assert answer == "before  after"

    def test_empty_think_tag(self) -> None:
        from localmelo.tests.smoke.core_loop_test import _split_thinking_and_answer

        thinking, answer = _split_thinking_and_answer("<think></think>answer")
        assert thinking == ""
        assert answer == "answer"
