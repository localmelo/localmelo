"""Shared deterministic tokenizer for cross-backend comparison metrics.

Provides a single, dependency-free token counter that every backend
inherits through :class:`BaseBackend`.  Because the same function is
applied to raw text regardless of which backend produced it, the
resulting counts form a uniform ruler suitable for apples-to-apples
comparison across MLC, Ollama, vLLM, cloud APIs, etc.

The counter is a lightweight regex approximation — *not* a model-specific
BPE/SentencePiece tokenizer.  Absolute counts will differ from any
particular model's tokenizer, but relative comparisons across backends
remain valid because the same deterministic rule is always applied.

Design constraints (see pyproject.toml):
- Zero additional dependencies
- No runtime network fetches
- No per-model tokenizer assets
- Deterministic: same input → same count, always
"""

from __future__ import annotations

import re

# Pre-split CJK ideographs so mixed-script text like "Hello你好" becomes
# "Hello 你 好" before the generic word matcher runs.
_CJK_CHAR_PATTERN = re.compile(
    r"([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])",
    re.UNICODE,
)

# Matches token-like pieces after CJK pre-splitting:
#   - Unicode letter sequences from non-CJK scripts
#   - Digit sequences
#   - Individual punctuation / symbols
_TOKEN_PATTERN = re.compile(r"[^\W\d_]+|[0-9]+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Count tokens in *text* using a deterministic regex approximation.

    Returns 0 for empty or whitespace-only input.
    """
    if not text or not text.strip():
        return 0
    text = _CJK_CHAR_PATTERN.sub(r" \1 ", text)
    return len(_TOKEN_PATTERN.findall(text))
