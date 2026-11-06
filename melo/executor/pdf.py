"""Read a PDF: its text page by page, and a picture of each page that has none.

``file_read`` and ``web_fetch`` both read a PDF through this module, so a PDF
means the same in either. A file or a download is a PDF by its first bytes
(:data:`MAGIC`), whatever its name or type says.

The bytes go to a child process (:mod:`._pdf_child`) on its standard input,
and nothing else goes with them: the library that parses them is not
thread-safe, and a broken file must not take the gateway down. The child is a
fresh interpreter started by path (``python -I``), so melo's own modules are
not loaded into it and pypdfium2 is imported only there. It runs in a group of
its own that the wall clock (:data:`TIMEOUT_S`) or a cancellation kills whole,
and it ends by itself after :data:`TIMEOUT_S` plus a few seconds of CPU. What
it answers is read up to :data:`MAX_ANSWER_BYTES`. A file over :data:`MAX_BYTES`
is never sent. Of the pages, :data:`MAX_PAGES` are read, and of the text,
:data:`MAX_CHARS` characters. Nothing leaves the child but one JSON object, and
what goes wrong reaches the model as one plain line (:class:`PdfError`),
never a traceback.

The text of a page under :data:`NO_TEXT_CHARS` characters is a page with no
text (a scan, a drawing): its text stays empty, a line says so, and
:func:`with_pictures` sends a picture of it, as many as one result carries
(:data:`MAX_PICTURES`). A picture is a PNG no larger than a screenshot (the
width :data:`PICTURE_WIDTH` the agent scales tool pictures to). An empty page
never reads as an empty document: its line is there whether or not its picture
reaches the model.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from localmelo.melo.executor.env import child_env
from localmelo.melo.executor.process_gate import kill_process_group
from localmelo.melo.schema import ImagePart

# A PDF starts with this, and nothing else does.
MAGIC = b"%PDF-"

MAX_BYTES = 50 * 1024 * 1024
MAX_PAGES = 300
MAX_CHARS = 1_000_000
TIMEOUT_S = 20.0
# Bytes the child may answer: the text at three bytes a character at most, and
# the pictures of one result as base64.
MAX_ANSWER_BYTES = 24 * 1024 * 1024
# A page with fewer characters than this, spaces left out, has no text.
NO_TEXT_CHARS = 20

# Pictures one result carries: the default of ``AgentOptions.tool_images_keep``,
# so none of them loses its pixels when the next arrives.
MAX_PICTURES = 3
PICTURE_DPI = 150
# The width the agent scales a tool's picture to (``MAX_IMAGE_WIDTH`` in
# ``melo.agent._tool_images``; a test holds the two equal). A page picture is
# made no wider, and no taller than a screenshot of a tall page.
PICTURE_WIDTH = 1280
PICTURE_HEIGHT = 2048

# Room a caller leaves in MAX_OUTPUT_LEN for the note :func:`with_pictures` adds.
NOTE_ROOM = 800

_CHILD = Path(__file__).with_name("_pdf_child.py")
_MARKER = re.compile(
    r"^\[page (\d+): no text, picture (?:not sent|attached|failed)\]$", re.M
)

_MB = 1024 * 1024
_BROKEN = "melo could not read this PDF: the file is damaged or is not a PDF."
_PROBLEMS = {
    "password": "This PDF is password-protected, so melo cannot read it.",
    "unavailable": "melo cannot read PDFs here: its PDF reader is not installed.",
}


class PdfError(ValueError):
    """A PDF that could not be read; the message is one plain line for the model."""


def is_pdf(head: bytes) -> bool:
    """Whether bytes that start with *head* are a PDF."""
    return head.startswith(MAGIC)


def read_file(source: BinaryIO) -> bytes:
    """All of the PDF file *source*, from its start; :class:`PdfError` when it is over :data:`MAX_BYTES`."""
    data = source.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise PdfError(
            f"This PDF is larger than {MAX_BYTES // _MB} MB; melo reads PDFs up to that size."
        )
    return data


def marker(page: int, state: str = "not sent") -> str:
    """The line that says page *page* has no text, and what became of its picture."""
    return f"[page {page}: no text, picture {state}]"


def page_list(pages: Sequence[int]) -> str:
    """*pages* (ascending) as ``3, 5-7, 9``."""
    runs: list[tuple[int, int]] = []
    for page in pages:
        if runs and page == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], page)
        else:
            runs.append((page, page))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


@dataclass
class PdfText:
    """What the child read: the document's page count and the text of its first pages."""

    pages: int
    texts: tuple[str, ...]
    cut: bool = False

    @functools.cached_property
    def blank(self) -> frozenset[int]:
        """The pages (from 1) with no text."""
        return frozenset(
            n
            for n, text in enumerate(self.texts, 1)
            if len("".join(text.split())) < NO_TEXT_CHARS
        )

    @functools.cached_property
    def text(self) -> str:
        """The pages in order, each under its page line; a page with no text says so."""
        lines: list[str] = []
        for n, page in enumerate(self.texts, 1):
            lines.append(f"--- page {n} of {self.pages} ---")
            if page:
                lines.append(page)
            if n in self.blank:
                lines.append(marker(n, "not sent"))
        if len(self.texts) < self.pages or self.cut:
            unread = len(self.texts) < self.pages
            lines.append(
                f"[the text is cut: melo reads at most {MAX_PAGES} pages and {MAX_CHARS:,} "
                "characters of a PDF"
                + (
                    f"; pages {len(self.texts) + 1}-{self.pages} are not read"
                    if unread
                    else ""
                )
                + "]"
            )
        return "\n".join(lines) + "\n"

    def line_of(self, page: int) -> int:
        """The line (from 1) of the line that says page *page* has no text."""
        return self.text.split("\n").index(marker(page)) + 1


class _AnswerTooLargeError(Exception):
    """The child answered more than :data:`MAX_ANSWER_BYTES`."""


async def _exchange(proc: asyncio.subprocess.Process, data: bytes) -> bytes:
    """Send *data* to the child and collect its answer, never more than the limit."""
    assert proc.stdin is not None and proc.stdout is not None
    stdin, stdout = proc.stdin, proc.stdout

    async def send() -> None:
        # A child that ended early closes its end: not an error of ours.
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            stdin.write(data)
            await stdin.drain()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            stdin.close()
            await stdin.wait_closed()

    async def receive() -> bytes:
        chunks: list[bytes] = []
        size = 0
        while chunk := await stdout.read(1 << 16):
            size += len(chunk)
            if size > MAX_ANSWER_BYTES:
                raise _AnswerTooLargeError
            chunks.append(chunk)
        return b"".join(chunks)

    _, answer = await asyncio.gather(send(), receive())
    await proc.wait()
    return answer


async def _ask(mode: str, limits: dict[str, Any], data: bytes) -> dict[str, Any]:
    """The child's answer for *mode*; :class:`PdfError` for every way it can fail."""
    if not sys.executable:
        raise PdfError(_PROBLEMS["unavailable"])
    limits = {**limits, "cpu": int(TIMEOUT_S) + 5}
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(_CHILD),
            mode,
            json.dumps(limits),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=child_env(),
            start_new_session=True,
        )
    except OSError:
        raise PdfError(_PROBLEMS["unavailable"]) from None
    try:
        async with asyncio.timeout(TIMEOUT_S):
            answer = await _exchange(proc, data)
    except TimeoutError:
        await kill_process_group(proc)
        raise PdfError(
            f"Reading this PDF took longer than {TIMEOUT_S:g} seconds; melo stopped."
        ) from None
    except _AnswerTooLargeError:
        await kill_process_group(proc)
        raise PdfError(_BROKEN) from None
    except BaseException:
        await kill_process_group(proc)
        raise
    try:
        found = json.loads(answer) if proc.returncode == 0 else None
    except ValueError:
        found = None
    if not isinstance(found, dict):
        raise PdfError("melo could not read this PDF: the PDF reader stopped on it.")
    status = found.get("status")
    if status != "ok":
        raise PdfError(_PROBLEMS.get(str(status), _BROKEN))
    return found


async def read_text(data: bytes) -> PdfText:
    """The text of the PDF in *data*; :class:`PdfError` when it cannot be read."""
    answer = await _ask("text", {"pages": MAX_PAGES, "chars": MAX_CHARS}, data)
    pages, texts = answer.get("pages"), answer.get("texts")
    if (
        not isinstance(pages, int)
        or not isinstance(texts, list)
        or not all(isinstance(text, str) for text in texts)
        or not 0 < len(texts) <= pages
    ):
        raise PdfError(_BROKEN)
    return PdfText(pages, tuple(texts), answer.get("cut") is True)


async def render_pages(data: bytes, pages: Sequence[int]) -> dict[int, ImagePart]:
    """A picture of each of *pages* (from 1) that could be drawn, by page."""
    limits = {
        "pages": list(pages),
        "dpi": PICTURE_DPI,
        "width": PICTURE_WIDTH,
        "height": PICTURE_HEIGHT,
    }
    answer = await _ask("render", limits, data)
    made: dict[int, ImagePart] = {}
    for item in answer.get("pictures") or []:
        page, png = item.get("page"), item.get("png")
        if page in pages and isinstance(png, str) and page not in made:
            made[page] = ImagePart(mime_type="image/png", data=png)
    return made


def _plural(pages: Sequence[int], one: str, many: str) -> str:
    return one if len(pages) == 1 else many


async def with_pictures(
    data: bytes,
    doc: PdfText,
    shown: str,
    locate: Callable[[int], str],
) -> tuple[str, str, list[ImagePart]]:
    """The pictures that go with *shown*, the part of ``doc.text`` a result shows.

    Returns *shown* with the line of each page that has no text brought up to
    date (``picture attached``, ``picture failed``, or left as ``picture not
    sent`` for the pages past :data:`MAX_PICTURES`), a note for after it (empty
    when *shown* holds no such page) and the pictures, in page order. Only a
    page *shown* has one: a read that skips it sends nothing for it. *locate*
    gives the words that read on from a page the result left out (``start_line=40``,
    ``part=2``). A picture that cannot be drawn is a failed page, never an error
    of the read.
    """
    wanted: list[int] = []
    for found in _MARKER.finditer(shown):
        page = int(found.group(1))
        if page in doc.blank and page not in wanted:
            wanted.append(page)
    if not wanted:
        return shown, "", []
    chosen = wanted[:MAX_PICTURES]
    try:
        made = await render_pages(data, chosen)
    except PdfError:
        made = {}
    state = {page: "attached" if page in made else "failed" for page in chosen}
    shown = _MARKER.sub(
        lambda m: (
            marker(int(m.group(1)), state[int(m.group(1))])
            if int(m.group(1)) in state
            else m.group(0)
        ),
        shown,
    )
    attached = [page for page in chosen if page in made]
    failed = [page for page in chosen if page not in made]
    left = wanted[MAX_PICTURES:]
    parts: list[str] = []
    if attached:
        parts.append(
            f"{_plural(attached, 'Page', 'Pages')} {page_list(attached)} "
            f"{_plural(attached, 'has', 'have')} no text; "
            f"{_plural(attached, 'its picture is', 'their pictures are')} attached to "
            "this result, in page order. If pictures do not reach you, say that "
            f"{_plural(attached, 'this page', 'these pages')} could not be read."
        )
    if failed:
        parts.append(
            f"{_plural(failed, 'Page', 'Pages')} {page_list(failed)} "
            f"{_plural(failed, 'has', 'have')} no text and no picture could be made: "
            f"{_plural(failed, 'it', 'they')} could not be read."
        )
    if left:
        parts.append(
            f"{_plural(left, 'Page', 'Pages')} {page_list(left)} "
            f"{_plural(left, 'has', 'have')} no text and no picture came with this "
            f"result (at most {MAX_PICTURES} do); read on from page {left[0]} with "
            f"{locate(left[0])} to get {_plural(left, 'it', 'them')}."
        )
    images = [made[page] for page in attached]
    return shown, "[" + " ".join(parts) + "]", images
