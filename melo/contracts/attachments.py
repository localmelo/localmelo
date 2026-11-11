"""Where a file's text starts in a message, and what is left as the person's words (#626).

A text file sent with a message travels in the message's text, after the words the
person typed, as a block that starts with :data:`FILE_OPENER`. The file's text is
data: it never counts as the person's words, which decide which password
handles can be typed and which sites, key files and scripts the person named.
:func:`words_before_files` is the one cut: the run's step review takes in every
message of the person's through it, so a path nobody thought of is closed too.

This module imports only the standard library.
"""

from __future__ import annotations

from typing import Final

# Where a file's block starts in a composed text; the note line comes after it.
FILE_OPENER: Final = "\n\n<attachment name="
# The line that ends a block.
FILE_CLOSER: Final = "</attachment>"
# The first run (no blank in it) of the block's opening tag.
_OPENING_TAG: Final = FILE_OPENER.split()[0]


def words_before_files(text: str) -> str:
    """What comes before the first place a file's block can start in *text*.

    It never reads a block: it cuts at the opener, so a block that is cut short,
    mis-sized or escaped in a way a reader would not undo cannot leave a file's
    text in the words. Typed text that holds an opener is cut short too, which
    is the safe direction. A text that begins with a block, as a message with no
    words does once the line breaks before its first block are taken off, has no
    words. A text with no opener comes back as it is.
    """
    cut = text.find(FILE_OPENER)
    if cut < 0:
        return "" if text.lstrip().startswith(FILE_OPENER.lstrip()) else text
    return text[:cut]


def is_file_tag(text: str, start: int, stop: int) -> bool:
    """Whether ``text[start:stop]``, a run with no blank in it, is a block's tag: the
    first run of an opening tag, or a run that begins with a closing tag.

    A tag is never a value a password word takes (a line that ends in one, then
    the block). It reads the text in place: a caller asks this of runs of any length.
    """
    return text.startswith(FILE_CLOSER, start, stop) or (
        stop - start == len(_OPENING_TAG) and text.startswith(_OPENING_TAG, start)
    )
