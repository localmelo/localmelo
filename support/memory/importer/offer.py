"""Whether bringing in other assistants' notes was ever offered on this computer.

One small file in the memory folder, shared by the terminal, the console and the
app: whichever offers it first writes the file, whatever the person answers, and
none of them offers again by itself.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time

from localmelo.melo.memory._sqlite import make_private_dir

NAME = "import_offer.json"
ANSWERS = ("brought", "declined")
log = logging.getLogger(__name__)


def offered(directory: str) -> bool:
    """Whether the offer was made (answered, or the notes brought in some other way)."""
    return os.path.isfile(os.path.join(directory, NAME))


def mark(directory: str, answer: str) -> None:
    """Record the offer as made, with the *answer* (``brought`` or ``declined``); the first stays.

    The file is created exclusively, so two answers at once leave one whole record, the first. A folder that
    cannot be written (or fills up half way, which leaves no record) is logged, not raised: the notes just
    written must not turn into an error.
    """
    if answer not in ANSWERS:
        raise ValueError(f"The answer is one of {ANSWERS}, not {answer!r}.")
    path = os.path.join(directory, NAME)
    try:
        make_private_dir(directory)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return  # the first answer stays
    except OSError:
        log.warning(
            "Could not record the import offer in %s.", directory, exc_info=True
        )
        return
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump({"answer": answer, "at": time.time()}, file)
    except OSError:
        # A half-written record would read as an offer made: leave none, so the offer stays open.
        log.warning(
            "Could not record the import offer in %s.", directory, exc_info=True
        )
        with contextlib.suppress(OSError):
            os.unlink(path)
