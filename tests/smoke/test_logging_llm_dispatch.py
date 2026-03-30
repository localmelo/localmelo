"""Tests that smoke LoggingLLM correctly dispatches to the right provider.

Covers:
- Ollama backend -> OllamaNativeChat inner provider
- MLC/other backend -> OpenAICompatLLM inner provider
- LoggingLLM records all expected fields for both paths
- LoggingLLM delegates chat() to the inner provider (not OpenAI-compat)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO.parent))

from localmelo.melo.schema import Message  # noqa: E402
from localmelo.support.providers.llm.ollama_chat import OllamaNativeChat  # noqa: E402
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM  # noqa: E402
from localmelo.tests.smoke.core_loop_test import LoggingLLM  # noqa: E402

# ── Provider selection in LoggingLLM ──────────────────────────


class TestLoggingLLMWrapsOllamaNative:
    """LoggingLLM wrapping OllamaNativeChat delegates to native /api/chat."""

    @pytest.mark.asyncio
    async def test_ollama_inner_provider_type(self) -> None:
        inner = OllamaNativeChat(
            base_url="http://localhost:11434",
            model="qwen3:8b",
        )
        llm = LoggingLLM(inner=inner)
        assert isinstance(llm._inner, OllamaNativeChat)

    @pytest.mark.asyncio
    async def test_ollama_chat_delegates_to_native(self) -> None:
        """chat() should call the inner OllamaNativeChat, not OpenAI-compat."""
        response_data = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "The answer.",
                "thinking": "Let me think...",
            },
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        inner = OllamaNativeChat(
            base_url="http://localhost:11434",
            model="qwen3:8b",
        )
        inner._http = MagicMock()
        inner._http.post = AsyncMock(return_value=mock_response)

        llm = LoggingLLM(inner=inner)
        msgs = [Message(role="user", content="What is 6 * 7?")]
        result = await llm.chat(msgs)

        # Verify inner provider was called on native endpoint
        inner._http.post.assert_called_once()
        call_args = inner._http.post.call_args
        assert call_args[0][0] == "/api/chat"  # native endpoint, not /chat/completions
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["think"] is True

        # Verify response has thinking
        assert result.thinking == "Let me think..."
        assert "The answer." in result.content

    @pytest.mark.asyncio
    async def test_ollama_logging_records_all_fields(self) -> None:
        """Logging captures thinking/answer split from Ollama native response."""
        response_data = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "Final answer.",
                "thinking": "Deep reasoning here.",
            },
            "prompt_eval_count": 80,
            "eval_count": 40,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        inner = OllamaNativeChat(
            base_url="http://localhost:11434",
            model="qwen3:8b",
        )
        inner._http = MagicMock()
        inner._http.post = AsyncMock(return_value=mock_response)

        llm = LoggingLLM(inner=inner)
        msgs = [Message(role="user", content="Test")]
        await llm.chat(msgs)

        assert len(llm.call_log) == 1
        entry = llm.call_log[0]

        # All required logging fields must be present
        assert "response_content" in entry
        assert "thinking" in entry
        assert "answer_only" in entry
        assert "normalized_thinking_tokens" in entry
        assert "normalized_answer_tokens" in entry
        assert "normalized_completion_tokens" in entry

        # Thinking should be captured from the native field
        assert entry["thinking"] == "Deep reasoning here."
        assert entry["normalized_thinking_tokens"] > 0
        assert entry["normalized_answer_tokens"] > 0


class TestLoggingLLMWrapsOpenAICompat:
    """LoggingLLM wrapping OpenAICompatLLM delegates to /v1/chat/completions."""

    @pytest.mark.asyncio
    async def test_mlc_inner_provider_type(self) -> None:
        inner = OpenAICompatLLM(
            base_url="http://localhost:8400/v1",
            model="qwen3-0.6b",
        )
        llm = LoggingLLM(inner=inner)
        assert isinstance(llm._inner, OpenAICompatLLM)

    @pytest.mark.asyncio
    async def test_mlc_chat_delegates_to_openai_compat(self) -> None:
        """chat() should call the inner OpenAICompatLLM on /chat/completions."""
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>\nReasoning.\n</think>\n\nThe answer.",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "total_tokens": 80,
            },
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        inner = OpenAICompatLLM(
            base_url="http://localhost:8400/v1",
            model="qwen3-0.6b",
        )
        inner._http = MagicMock()
        inner._http.post = AsyncMock(return_value=mock_response)

        llm = LoggingLLM(inner=inner)
        msgs = [Message(role="user", content="Hello")]
        result = await llm.chat(msgs)

        # Verify inner provider was called on OpenAI-compat endpoint
        inner._http.post.assert_called_once()
        call_args = inner._http.post.call_args
        assert call_args[0][0] == "/chat/completions"

        # Verify response
        assert "<think>" in result.content
        assert "The answer." in result.content

    @pytest.mark.asyncio
    async def test_mlc_logging_records_all_fields(self) -> None:
        """Logging captures thinking/answer split from MLC <think> tags."""
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>\nMLC reasoning.\n</think>\n\nMLC answer.",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "total_tokens": 80,
            },
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        inner = OpenAICompatLLM(
            base_url="http://localhost:8400/v1",
            model="qwen3-0.6b",
        )
        inner._http = MagicMock()
        inner._http.post = AsyncMock(return_value=mock_response)

        llm = LoggingLLM(inner=inner)
        msgs = [Message(role="user", content="Test")]
        await llm.chat(msgs)

        assert len(llm.call_log) == 1
        entry = llm.call_log[0]

        # All required logging fields
        assert "response_content" in entry
        assert "thinking" in entry
        assert "answer_only" in entry
        assert "normalized_thinking_tokens" in entry
        assert "normalized_answer_tokens" in entry
        assert "normalized_completion_tokens" in entry

        # MLC thinking comes from <think> tag parsing
        assert entry["thinking"] == "MLC reasoning."
        assert entry["answer_only"] == "MLC answer."
        assert entry["normalized_thinking_tokens"] > 0
        assert entry["normalized_answer_tokens"] > 0


class TestLoggingLLMModelProperty:
    """LoggingLLM.model proxies the inner provider's model attribute."""

    def test_model_from_ollama(self) -> None:
        inner = OllamaNativeChat(
            base_url="http://localhost:11434",
            model="qwen3:8b",
        )
        llm = LoggingLLM(inner=inner)
        assert llm.model == "qwen3:8b"

    def test_model_from_openai_compat(self) -> None:
        inner = OpenAICompatLLM(
            base_url="http://localhost:8400/v1",
            model="qwen3-0.6b",
        )
        llm = LoggingLLM(inner=inner)
        assert llm.model == "qwen3-0.6b"


class TestLoggingLLMClose:
    """LoggingLLM.close() delegates to the inner provider."""

    @pytest.mark.asyncio
    async def test_close_delegates(self) -> None:
        inner = MagicMock()
        inner.close = AsyncMock()
        llm = LoggingLLM(inner=inner)
        await llm.close()
        inner.close.assert_called_once()
