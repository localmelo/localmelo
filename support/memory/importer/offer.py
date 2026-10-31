"""Whether bringing in other assistants' notes was ever offered on this computer.

One small file in the memory folder, shared by the terminal, the console and the
app: whichever offers it first writes the file, whatever the person answers, and
none of them offers again by itself.
"""

from __future__ import annotations

import json
import os
import time

from localmelo.melo.memory._sqlite import make_private_dir

NAME = "import_offer.json"


def offered(directory: str) -> bool:
    """Whether the offer was made (answered, or the notes brought in some other way)."""
    return os.path.isfile(os.path.join(directory, NAME))


def mark(directory: str, answer: str) -> None:
    """Record the offer as made, with the *answer* (``brought`` or ``declined``); the first stays."""
    if offered(directory):
        return
    make_private_dir(directory)
    fd = os.open(os.path.join(directory, NAME), os.O_WRONLY | os.O_CREAT, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump({"answer": answer, "at": time.time()}, file)
