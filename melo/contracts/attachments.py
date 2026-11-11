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


def words_before_files(text: str) -> str:
    """What comes before the first place a file's block can start in *text*.

    It never reads a block: it cuts at the opener, so a block that is cut short,
    mis-sized or escaped in a way a reader would not undo cannot leave a file's
    text in the words. Typed text that holds an opener is cut short too, which
    is the safe direction. A text with no opener comes back as it is.
    """
    cut = text.find(FILE_OPENER)
    return text if cut < 0 else text[:cut]
