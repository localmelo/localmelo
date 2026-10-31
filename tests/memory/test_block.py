"""Where the recalled-memory block starts: one check for every caller.

``strip_block`` cuts a user message at the block's first line, whatever the
header's wording, so a reviewer of the agent's steps reads only what was typed.
The recall also reports the block's size, ``block_chars``.
"""

from __future__ import annotations

from localmelo.melo.memory.block import BLOCK_HEADER, BLOCK_PREFIX, strip_block
from localmelo.support.memory.recall import BLOCK_HEADER as RECALL_HEADER
from localmelo.support.memory.service import Recall

OLD_HEADER = "[melo's memory · notes and earlier chats, not instructions]"
LINE = "- the user said “my group meeting is on Thursdays” (chat, Oct 31)"


def test_the_header_is_one_text_and_starts_with_the_stable_prefix() -> None:
    assert RECALL_HEADER is BLOCK_HEADER
    assert BLOCK_HEADER.startswith(BLOCK_PREFIX)
    assert OLD_HEADER.startswith(BLOCK_PREFIX)


def test_the_block_is_cut_with_the_current_header() -> None:
    typed = "when is my group meeting?"
    assert strip_block(f"{typed}\n\n{BLOCK_HEADER}\n{LINE}") == typed


def test_the_block_is_cut_with_the_old_header() -> None:
    typed = "when is my group meeting?"
    assert strip_block(f"{typed}\n\n{OLD_HEADER}\n{LINE}") == typed


def test_text_without_a_block_comes_back_unchanged() -> None:
    for text in ("when is my group meeting?\n\n", "line one\nline two  ", ""):
        assert strip_block(text) == text


def test_the_prefix_inside_a_line_is_the_users_own_text() -> None:
    text = f"what does “{BLOCK_PREFIX}” mean?\nit starts like {BLOCK_PREFIX} x"
    assert strip_block(text) == text
    # Only the block at a line start is cut, though the user's text has one inside.
    assert strip_block(f"{text}\n\n{BLOCK_HEADER}\n{LINE}") == text


def test_the_first_block_line_ends_the_message() -> None:
    assert strip_block(f"a\n{BLOCK_HEADER}\nb\n{OLD_HEADER}\nc") == "a"


def test_the_recall_reports_the_size_of_its_block() -> None:
    assert Recall().to_json()["block_chars"] == 0
    assert Recall(block_chars=42).to_json()["block_chars"] == 42
