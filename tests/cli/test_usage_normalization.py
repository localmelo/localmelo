"""Regression tests for provider usage normalization."""

from __future__ import annotations

from localmelo.support.providers.llm.openai_compat import (
    _coerce_token_count,
    _normalize_usage,
)


class TestCoerceTokenCount:
    """_coerce_token_count handles arbitrary input safely."""

    def test_none(self) -> None:
        assert _coerce_token_count(None) == 0

    def test_int(self) -> None:
        assert _coerce_token_count(42) == 42

    def test_zero(self) -> None:
        assert _coerce_token_count(0) == 0

    def test_negative_clamped(self) -> None:
        assert _coerce_token_count(-5) == 0

    def test_string_int(self) -> None:
        assert _coerce_token_count("7") == 7

    def test_string_garbage(self) -> None:
        assert _coerce_token_count("abc") == 0

    def test_float(self) -> None:
        assert _coerce_token_count(3.9) == 3

    def test_bool_true(self) -> None:
        # bool is a subclass of int; True == 1
        assert _coerce_token_count(True) == 1

    def test_empty_string(self) -> None:
        assert _coerce_token_count("") == 0

    def test_list(self) -> None:
        assert _coerce_token_count([1, 2]) == 0


class TestNormalizeUsage:
    """_normalize_usage produces clean dict or None."""

    def test_none_input(self) -> None:
        assert _normalize_usage(None) is None

    def test_string_input(self) -> None:
        assert _normalize_usage("not a dict") is None

    def test_list_input(self) -> None:
        assert _normalize_usage([1, 2, 3]) is None

    def test_normal_usage(self) -> None:
        raw = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        result = _normalize_usage(raw)
        assert result == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }

    def test_total_backfill(self) -> None:
        """total_tokens missing -> back-filled from prompt + completion."""
        raw = {"prompt_tokens": 10, "completion_tokens": 20}
        result = _normalize_usage(raw)
        assert result is not None
        assert result["total_tokens"] == 30

    def test_total_null_backfill(self) -> None:
        """total_tokens=None -> back-filled."""
        raw = {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": None}
        result = _normalize_usage(raw)
        assert result is not None
        assert result["prompt_tokens"] == 5
        assert result["completion_tokens"] == 12
        assert result["total_tokens"] == 17

    def test_string_values_coerced(self) -> None:
        raw = {"prompt_tokens": "7", "completion_tokens": "3"}
        result = _normalize_usage(raw)
        assert result is not None
        assert result["prompt_tokens"] == 7
        assert result["completion_tokens"] == 3
        assert result["total_tokens"] == 10

    def test_negative_values_clamped(self) -> None:
        raw = {"prompt_tokens": -5, "completion_tokens": 2, "total_tokens": -1}
        result = _normalize_usage(raw)
        assert result is not None
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 2
        assert result["total_tokens"] == 2

    def test_empty_dict(self) -> None:
        """Empty dict -> all zeros, no crash."""
        result = _normalize_usage({})
        assert result == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_all_none_fields(self) -> None:
        raw = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        result = _normalize_usage(raw)
        assert result == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_total_present_and_correct(self) -> None:
        """When total_tokens is present and non-zero, it is kept as-is."""
        raw = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 50}
        result = _normalize_usage(raw)
        assert result is not None
        assert result["total_tokens"] == 50
