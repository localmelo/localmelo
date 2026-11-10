"""The recalled-memory block's heading and the size the recall reports for it.

Only the code that appends the block writes the heading, and nothing parses it:
which part of a message is melo's notes comes from the caller that composed the
message, never from searching the text.
"""

from __future__ import annotations

from localmelo.melo.memory.block import BLOCK_HEADER
from localmelo.support.memory.recall import BLOCK_HEADER as RECALL_HEADER
from localmelo.support.memory.service import Recall


def test_the_header_is_one_text() -> None:
    assert RECALL_HEADER is BLOCK_HEADER


def test_the_recall_reports_the_size_of_its_block() -> None:
    assert Recall().to_json()["block_chars"] == 0
    assert Recall(block_chars=42).to_json()["block_chars"] == 42
