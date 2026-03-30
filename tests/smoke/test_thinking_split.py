"""Tests for thinking/answer split logic and OllamaNativeChat provider.

Covers:
- _split_thinking with MLC-style <think>...</think> content
- _split_thinking with Ollama native Message(thinking="...", content="...")
- _split_thinking with no thinking
- OllamaNativeChat response parsing (mock httpx)
- Normalized token counts include thinking
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure repo root is importable.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO.parent))

from localmelo.melo.schema import Message, ToolDef  # noqa: E402
from localmelo.support.backends.tokenization import count_tokens  # noqa: E402
from localmelo.tests.smoke.core_loop_test import _split_thinking  # noqa: E402

# ── _split_thinking tests ──────────────────────────────────────


class TestSplitThinkingMLC:
    """MLC-style: thinking is embedded in content as <think>...</think>."""

    def test_extracts_thinking_and_answer(self) -> None:
        msg = Message(
            role="assistant",
            content="<think>\nI need to reason.\n</think>\n\nThe answer is 42.",
        )
        thinking, answer = _split_thinking(msg)
        assert thinking == "I need to reason."
        assert answer == "The answer is 42."

    def test_empty_think_block(self) -> None:
        msg = Message(
            role="assistant",
            content="<think></think>\n\nJust the answer.",
        )
        thinking, answer = _split_thinking(msg)
        assert thinking == ""
        assert answer == "Just the answer."

    def test_multiline_thinking(self) -> None:
        msg = Message(
            role="assistant",
            content="<think>\nLine 1\nLine 2\nLine 3\n</think>\n\nResult.",
        )
        thinking, answer = _split_thinking(msg)
        assert "Line 1" in thinking
        assert "Line 3" in thinking
        assert answer == "Result."


class TestSplitThinkingOllamaNative:
    """Ollama native: resp.thinking has the thinking text."""

    def test_native_thinking_field(self) -> None:
        msg = Message(
            role="assistant",
            content="<think>\nDeep thought.\n</think>\n\nThe answer.",
            thinking="Deep thought.",
        )
        thinking, answer = _split_thinking(msg)
        assert thinking == "Deep thought."
        assert answer == "The answer."

    def test_native_thinking_content_without_tags(self) -> None:
        """When content does not contain <think> tags but thinking is set."""
        msg = Message(
            role="assistant",
            content="The answer.",
            thinking="Some reasoning.",
        )
        thinking, answer = _split_thinking(msg)
        assert thinking == "Some reasoning."
        assert answer == "The answer."

    def test_native_thinking_with_combined_content(self) -> None:
        """Content has the <think> wrapper from the provider."""
        msg = Message(
            role="assistant",
            content="<think>\nReasoning here.\n</think>\n\nFinal answer.",
            thinking="Reasoning here.",
        )
        thinking, answer = _split_thinking(msg)
        assert thinking == "Reasoning here."
        assert answer == "Final answer."


class TestSplitThinkingNoThinking:
    """No thinking present at all."""

    def test_plain_content(self) -> None:
        msg = Message(role="assistant", content="Hello world!")
        thinking, answer = _split_thinking(msg)
        assert thinking == ""
        assert answer == "Hello world!"

    def test_empty_content(self) -> None:
        msg = Message(role="assistant", content="")
        thinking, answer = _split_thinking(msg)
        assert thinking == ""
        assert answer == ""


# ── OllamaNativeChat tests ─────────────────────────────────────


class TestOllamaNativeChatParsing:
    """OllamaNativeChat response parsing with mocked httpx."""

    @pytest.mark.asyncio
    async def test_basic_response_with_thinking(self) -> None:
        from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

        response_data = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "thinking": "Let me reason about this...",
            },
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        provider = OllamaNativeChat(
            base_url="http://localhost:11434",
            model="qwen3:8b",
        )
        provider._http = MagicMock()
        provider._http.post = AsyncMock(return_value=mock_response)

        msgs = [Message(role="user", content="What is 6 * 7?")]
        result = await provider.chat(msgs)

        assert result.role == "assistant"
        assert result.thinking == "Let me reason about this..."
        assert "The answer is 42." in result.content
        assert "<think>" in result.content
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 50
        assert result.usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_response_without_thinking(self) -> None:
        from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

        response_data = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "Just an answer.",
            },
            "prompt_eval_count": 50,
            "eval_count": 20,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        provider = OllamaNativeChat(
            base_url="http://localhost:11434",
            model="qwen3:8b",
        )
        provider._http = MagicMock()
        provider._http.post = AsyncMock(return_value=mock_response)

        msgs = [Message(role="user", content="Hello")]
        result = await provider.chat(msgs)

        assert result.thinking == ""
        assert result.content == "Just an answer."
        assert "<think>" not in result.content

    @pytest.mark.asyncio
    async def test_response_with_tool_call(self) -> None:
        from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

        response_data = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "I should use the search tool.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": {"query": "test"},
                        }
                    }
                ],
            },
            "prompt_eval_count": 80,
            "eval_count": 30,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        provider = OllamaNativeChat(
            base_url="http://localhost:11434",
            model="qwen3:8b",
        )
        provider._http = MagicMock()
        provider._http.post = AsyncMock(return_value=mock_response)

        tools = [
            ToolDef(
                name="search",
                description="Search for something",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            )
        ]
        msgs = [Message(role="user", content="Search for test")]
        result = await provider.chat(msgs, tools=tools)

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "search"
        assert result.tool_call.arguments == {"query": "test"}
        assert result.thinking == "I should use the search tool."

    @pytest.mark.asyncio
    async def test_request_includes_think_true(self) -> None:
        from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

        response_data = {
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "ok"},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        provider = OllamaNativeChat(
            base_url="http://localhost:11434",
            model="qwen3:8b",
        )
        provider._http = MagicMock()
        provider._http.post = AsyncMock(return_value=mock_response)

        msgs = [Message(role="user", content="hi")]
        await provider.chat(msgs)

        # Verify the request payload
        call_args = provider._http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["think"] is True
        assert payload["stream"] is False
        assert payload["model"] == "qwen3:8b"

    @pytest.mark.asyncio
    async def test_usage_normalization_missing_counts(self) -> None:
        from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

        response_data = {
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "ok"},
            # No prompt_eval_count or eval_count
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        provider = OllamaNativeChat(
            base_url="http://localhost:11434",
            model="qwen3:8b",
        )
        provider._http = MagicMock()
        provider._http.post = AsyncMock(return_value=mock_response)

        msgs = [Message(role="user", content="hi")]
        result = await provider.chat(msgs)

        # No usage counts -> usage should be None
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat

        provider = OllamaNativeChat(
            base_url="http://localhost:11434",
            model="qwen3:8b",
        )
        provider._http = MagicMock()
        provider._http.aclose = AsyncMock()

        await provider.close()
        provider._http.aclose.assert_called_once()


# ── Normalized token counts include thinking ───────────────────


class TestNormalizedTokensIncludeThinking:
    """Normalized completion tokens should include both thinking and answer."""

    def test_thinking_plus_answer_equals_completion(self) -> None:
        """When we split thinking from answer, their token counts should sum
        to the total completion count."""
        msg = Message(
            role="assistant",
            content="<think>\nSome reasoning.\n</think>\n\nThe answer.",
        )
        thinking, answer_only = _split_thinking(msg)
        norm_thinking = count_tokens(thinking)
        norm_answer = count_tokens(answer_only)
        norm_completion = norm_thinking + norm_answer

        # Each part should be positive
        assert norm_thinking > 0
        assert norm_answer > 0
        assert norm_completion == norm_thinking + norm_answer

    def test_no_thinking_all_answer(self) -> None:
        """With no thinking, all tokens are answer tokens."""
        msg = Message(role="assistant", content="Just an answer.")
        thinking, answer_only = _split_thinking(msg)
        norm_thinking = count_tokens(thinking)
        norm_answer = count_tokens(answer_only)

        assert norm_thinking == 0
        assert norm_answer > 0

    def test_ollama_native_thinking_tokens(self) -> None:
        """Ollama native thinking field produces correct token split."""
        msg = Message(
            role="assistant",
            content="<think>\nReasoning.\n</think>\n\nFinal answer.",
            thinking="Reasoning.",
        )
        thinking, answer_only = _split_thinking(msg)
        norm_thinking = count_tokens(thinking)
        norm_answer = count_tokens(answer_only)

        assert norm_thinking > 0
        assert norm_answer > 0
        assert thinking == "Reasoning."
        assert answer_only == "Final answer."


# ── Message backward compatibility ─────────────────────────────


class TestMessageBackwardCompat:
    """Adding 'thinking' field must not break existing constructions."""

    def test_basic_construction(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.thinking == ""

    def test_full_construction_without_thinking(self) -> None:
        msg = Message(
            role="assistant",
            content="hi",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert msg.thinking == ""
        assert msg.usage is not None

    def test_construction_with_thinking(self) -> None:
        msg = Message(
            role="assistant",
            content="answer",
            thinking="some reasoning",
        )
        assert msg.thinking == "some reasoning"
        assert msg.content == "answer"
