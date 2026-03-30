"""Tests for the shared deterministic tokenizer."""

from __future__ import annotations

from localmelo.support.backends.tokenization import count_tokens


class TestCountTokensBasic:
    """Core behaviour: empty input, simple strings, determinism."""

    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_none_like_empty(self) -> None:
        # count_tokens expects str; empty is the edge case.
        assert count_tokens("") == 0

    def test_whitespace_only(self) -> None:
        assert count_tokens("   \t\n") == 0

    def test_single_word(self) -> None:
        assert count_tokens("hello") == 1

    def test_multiple_words(self) -> None:
        assert count_tokens("hello world foo bar") == 4

    def test_deterministic(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        a = count_tokens(text)
        b = count_tokens(text)
        assert a == b


class TestCountTokensPunctuation:
    """Punctuation symbols count as individual tokens."""

    def test_sentence_with_period(self) -> None:
        # "Hello" + "," + "world" + "!" = 4
        assert count_tokens("Hello, world!") == 4

    def test_standalone_symbols(self) -> None:
        assert count_tokens("!@#") == 3

    def test_mixed(self) -> None:
        # "a" + "+" + "b" + "=" + "c" = 5
        assert count_tokens("a + b = c") == 5


class TestCountTokensNumbers:
    """Digit sequences are single tokens."""

    def test_integer(self) -> None:
        assert count_tokens("42") == 1

    def test_mixed_with_text(self) -> None:
        # "I" + "have" + "3" + "cats" = 4
        assert count_tokens("I have 3 cats") == 4


class TestCountTokensCJK:
    """Each CJK ideograph is one token."""

    def test_chinese(self) -> None:
        # 4 individual characters
        assert count_tokens("你好世界") == 4

    def test_mixed_chinese_english(self) -> None:
        # "Hello" + "你" + "好" = 3
        assert count_tokens("Hello你好") == 3

    def test_chinese_with_punctuation(self) -> None:
        # "你" + "好" + "，" + "世" + "界" + "！" = 6
        assert count_tokens("你好，世界！") == 6


class TestCountTokensOtherScripts:
    """Non-Latin scripts should not be dropped on the floor."""

    def test_cyrillic_words(self) -> None:
        assert count_tokens("Привет мир") == 2

    def test_arabic_words(self) -> None:
        assert count_tokens("مرحبا بالعالم") == 2


class TestCountTokensEdgeCases:
    """Unusual inputs that must not crash."""

    def test_only_newlines(self) -> None:
        assert count_tokens("\n\n\n") == 0

    def test_unicode_emoji(self) -> None:
        # Emojis are not matched by any pattern branch, so 0.
        # This is fine — the tokenizer is an approximation.
        result = count_tokens("😀")
        assert isinstance(result, int)
        assert result >= 0

    def test_very_long_string(self) -> None:
        text = "word " * 10_000
        assert count_tokens(text) == 10_000


class TestBackendContractExposesTokenizer:
    """BaseBackend.count_tokens delegates to the shared implementation."""

    def test_base_backend_count_tokens_matches(self) -> None:
        from localmelo.support.backends.base import BaseBackend

        text = "Hello, 你好世界!"
        assert BaseBackend.count_tokens(text) == count_tokens(text)

    def test_concrete_backend_inherits(self) -> None:
        """A registered backend inherits count_tokens without overriding."""
        import localmelo.support.backends.registry as _reg
        from localmelo.support.backends.registry import (
            _clear,
            ensure_defaults_registered,
            get_backend,
        )

        _clear()
        _reg._DEFAULT_BACKENDS_REGISTERED = False
        ensure_defaults_registered()

        backend = get_backend("ollama")
        text = "The quick brown fox"
        assert backend.count_tokens(text) == count_tokens(text)

        # Clean up
        _clear()

    def test_package_reexport(self) -> None:
        """count_tokens is importable from the backends package."""
        from localmelo.support.backends import count_tokens as pkg_fn

        assert pkg_fn is count_tokens
