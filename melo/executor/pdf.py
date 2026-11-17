"""Read a PDF: its text page by page, and a picture of each page that has none.

``file_read`` and ``web_fetch`` both read a PDF through this module, so a PDF
means the same in either. A file or a download is a PDF by its first bytes
(:data:`MAGIC`), whatever its name or type says.

The bytes go to a child process (:mod:`._pdf_child`) on its standard input,
and nothing else goes with them: the library that parses them is not
thread-safe, and a crash of it must end the child, not the gateway. The child
is a fresh interpreter started by path (``python -I``), so melo's own modules
are not loaded into it and pypdfium2 is imported only there. It runs in a group
of its own, and ends by itself: after :data:`TIMEOUT_S` plus a few seconds of
CPU or of wall clock, whichever comes first, so it is bounded even if this
process is gone. While it runs this process watches it, and kills the group
whole on the wall clock (:data:`TIMEOUT_S`), a cancellation, an answer over its
limit, or a resident memory over :data:`MAX_RSS_BYTES`. That last limit is a
sample, not a sandbox: it looks every :data:`WATCH_S`, so a burst between two
looks can pass it unseen; it stops ordinary runaway memory only. When the
memory cannot be read :data:`BLIND_LOOKS` looks in a row while the child still
runs, the child is stopped too. One read runs at a time in a process, so the
limit holds for all of them together. A file over :data:`MAX_BYTES` is never
sent. Of the pages, :data:`MAX_PAGES` are read, and of the text,
:data:`MAX_CHARS` characters (a page's text is whole in the child before that
limit cuts it). Nothing leaves the child but one JSON object, and the child is
not trusted: its answer is read up to a limit that depends on what was asked,
and checked (types, counts, sizes, every byte of a picture's PNG structure)
before any of it is used. What goes wrong reaches the model as one plain line
(:class:`PdfError`), never a traceback.

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
import base64
import binascii
import contextlib
import functools
import json
import re
import struct
import subprocess
import sys
import threading
import zlib
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TypeGuard

from localmelo.melo.executor import _pdf_child
from localmelo.melo.executor.env import child_env
from localmelo.melo.executor.process_gate import kill_process_group
from localmelo.melo.schema import ImagePart

# A PDF starts with this, and nothing else does.
MAGIC = b"%PDF-"

_MB = 1024 * 1024

MAX_BYTES = 50 * _MB
MAX_PAGES = 300
MAX_CHARS = 1_000_000
TIMEOUT_S = 20.0
# The child ends itself this many seconds after TIMEOUT_S (CPU and wall clock).
_GRACE_S = 5
# Resident memory the child may use, and how often it is looked at. Resident
# memory is what the kernel gives it at the moment: native allocations that
# are not touched do not count, and a burst between two looks is not seen.
MAX_RSS_BYTES = 1024 * _MB
WATCH_S = 0.25
# A look at the memory that does not answer in this long is a look that failed;
# that many failed looks in a row stop a child that still runs.
PS_TIMEOUT_S = 1.0
BLIND_LOOKS = 3

# A page with fewer characters than this, spaces left out, has no text.
NO_TEXT_CHARS = 20

# Characters of base64 one picture may take; the child draws a page smaller to fit.
PICTURE_BYTES = 4 * _MB

# Pictures one result carries: ``KEEP_IMAGES`` of ``melo.agent._tool_images``,
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

_BROKEN = "melo could not read this PDF: the file is damaged or is not a PDF."
_PROBLEMS = {
    "password": "This PDF is password-protected, so melo cannot read it.",
    "unavailable": "melo cannot read PDFs here: its PDF reader is not installed.",
}
_PNG = b"\x89PNG\r\n\x1a\n"


class PdfError(ValueError):
    """A PDF that could not be read; the message is one plain line for the model."""


def is_pdf(head: bytes) -> bool:
    """Whether bytes that start with *head* are a PDF."""
    return head.startswith(MAGIC)


def read_file(source: BinaryIO, head: bytes = b"") -> bytes:
    """The PDF file *source*: *head* (bytes already read from its start) and the rest of it.

    :class:`PdfError` when it is over :data:`MAX_BYTES`.
    """
    data = head + source.read(MAX_BYTES + 1 - len(head))
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

    def _blocks(self) -> list[str]:
        """Each page as one block: its page line, its text, and the line that says it has none."""
        blocks: list[str] = []
        for n, page in enumerate(self.texts, 1):
            lines = [f"--- page {n} of {self.pages} ---"]
            if page:
                lines.append(page)
            if n in self.blank:
                lines.append(marker(n, "not sent"))
            blocks.append("\n".join(lines))
        return blocks

    def _cut_note(self) -> list[str]:
        """The line that says the text stops short of the document, when it does."""
        unread = len(self.texts) < self.pages
        if not unread and not self.cut:
            return []
        return [
            f"[the text is cut: melo reads at most {MAX_PAGES} pages and {MAX_CHARS:,} "
            "characters of a PDF"
            + (
                f"; pages {len(self.texts) + 1}-{self.pages} are not read"
                if unread
                else ""
            )
            + "]"
        ]

    @functools.cached_property
    def text(self) -> str:
        """The pages in order, each under its page line; a page with no text says so."""
        return "\n".join([*self._blocks(), *self._cut_note()]) + "\n"

    def pieces(self, per: int = MAX_PICTURES) -> list[str]:
        """The text in pieces that each hold at most *per* pages with no text.

        A reader that shows one piece at a time (``web_fetch``'s parts) thus
        has every picture the piece needs to give, and the next piece starts
        at the first page left out.
        """
        pieces: list[list[str]] = [[]]
        blanks = 0
        for n, block in enumerate(self._blocks(), 1):
            if n in self.blank:
                if blanks == per:
                    pieces.append([])
                    blanks = 0
                blanks += 1
            pieces[-1].append(block)
        pieces[-1].extend(self._cut_note())
        return ["\n".join(piece) + "\n" for piece in pieces]

    def line_of(self, page: int) -> int:
        """The line (from 1) of the line that says page *page* has no text."""
        return self.text.split("\n").index(marker(page)) + 1


class _AnswerTooLargeError(Exception):
    """The child answered more than it was allowed to."""


async def _exchange(proc: asyncio.subprocess.Process, data: bytes, limit: int) -> bytes:
    """Send *data* to the child and collect its answer, never more than *limit* bytes."""
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
            if size > limit:
                raise _AnswerTooLargeError
            chunks.append(chunk)
        return b"".join(chunks)

    _, answer = await asyncio.gather(send(), receive())
    await proc.wait()
    return answer


async def _resident(pid: int) -> int | None:
    """Resident memory of process *pid*, in bytes; None when the look failed (it is gone, no ``ps``, no answer in time)."""
    try:
        ps = await asyncio.create_subprocess_exec(
            "ps",
            "-o",
            "rss=",
            "-p",
            str(pid),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    try:
        out, _ = await asyncio.wait_for(ps.communicate(), PS_TIMEOUT_S)
        return int(out.split()[0]) * 1024  # ps gives kilobytes
    except (TimeoutError, ValueError, IndexError):
        return None
    finally:
        # A look that timed out or was cancelled leaves no process behind.
        if ps.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                ps.kill()
            await ps.wait()


async def _watch(proc: asyncio.subprocess.Process, stopped: list[str]) -> None:
    """Stop the child *proc* leads, saying why in *stopped* first, when it is over its memory or cannot be watched.

    Over :data:`MAX_RSS_BYTES`, or :data:`BLIND_LOOKS` failed looks in a row
    while it still runs. A child that has ended is a normal end, whatever the
    last look found.
    """
    blind = 0
    while proc.returncode is None:
        await asyncio.sleep(WATCH_S)
        used = await _resident(proc.pid)
        if proc.returncode is not None:
            return
        blind = blind + 1 if used is None else 0
        if used is not None and used > MAX_RSS_BYTES:
            reason = (
                f"Reading this PDF needed more than {MAX_RSS_BYTES // _MB} MB of "
                "memory; melo stopped."
            )
        elif blind >= BLIND_LOOKS:
            reason = (
                "melo stopped reading this PDF: it could not check how much memory "
                "the PDF reader used."
            )
        else:
            continue
        stopped.append(reason)
        await kill_process_group(proc)
        return


# One PDF read at a time in this process (a lock of threads, polled, so it holds
# across event loops and a cancelled wait leaves nothing held).
_ONE_READ = threading.Lock()
_POLL_S = 0.02


@contextlib.asynccontextmanager
async def _one_at_a_time() -> AsyncIterator[None]:
    """Hold the place of the one PDF read that runs: the memory limit is for all of them."""
    while not _ONE_READ.acquire(blocking=False):
        await asyncio.sleep(_POLL_S)
    try:
        yield
    finally:
        _ONE_READ.release()


async def _ask(
    mode: str, limits: dict[str, Any], data: bytes, answer_limit: int
) -> dict[str, Any]:
    """The child's answer for *mode*, at most *answer_limit* bytes; :class:`PdfError` for every way it can fail."""
    if not sys.executable:
        raise PdfError(_PROBLEMS["unavailable"])
    async with _one_at_a_time():
        return await _run_child(mode, limits, data, answer_limit)


async def _run_child(
    mode: str, limits: dict[str, Any], data: bytes, answer_limit: int
) -> dict[str, Any]:
    ends = int(TIMEOUT_S) + _GRACE_S
    limits = {**limits, "cpu": ends, "wall": ends}
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
    stopped: list[str] = []
    watcher = asyncio.create_task(_watch(proc, stopped))
    try:
        async with asyncio.timeout(TIMEOUT_S):
            answer = await _exchange(proc, data, answer_limit)
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
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
    if stopped:
        raise PdfError(stopped[0])
    try:
        found = json.loads(answer) if proc.returncode == 0 else None
    except (
        ValueError,
        RecursionError,
    ):  # not JSON, or nested past what the decoder takes
        found = None
    if not isinstance(found, dict):
        raise PdfError("melo could not read this PDF: the PDF reader stopped on it.")
    status = found.get("status")
    if status != "ok":
        raise PdfError(_PROBLEMS.get(str(status), _BROKEN))
    return found


async def read_text(data: bytes) -> PdfText:
    """The text of the PDF in *data*; :class:`PdfError` when it cannot be read."""
    # Four bytes a character at most, and room for the quotes and page list.
    answer = await _ask(
        "text", {"pages": MAX_PAGES, "chars": MAX_CHARS}, data, MAX_CHARS * 4 + 65_536
    )
    pages, texts = answer.get("pages"), answer.get("texts")
    if (
        type(pages) is not int
        or not isinstance(texts, list)
        or not 0 < len(texts) <= min(pages, MAX_PAGES)
        or not all(type(text) is str for text in texts)
        or sum(map(len, texts)) > MAX_CHARS
    ):
        raise PdfError(_BROKEN)
    # What the child says is cleaned again here: it is not taken on its word.
    return PdfText(
        pages, tuple(map(_pdf_child.clean, texts)), answer.get("cut") is True
    )


def _is_png(raw: bytes) -> bool:
    """Whether *raw* is a whole PNG within the size of a screenshot, by its structure alone.

    The signature, then chunks whose lengths stay in *raw* and whose CRCs hold,
    ``IHDR`` first (with a width and height within the limits), some ``IDAT``,
    ``IEND`` last and nothing after it. Nothing is decoded here: the pixels are
    never opened by this process.
    """
    if raw[:8] != _PNG:
        return False
    view = memoryview(raw)
    pos = 8
    kinds: list[bytes] = []
    while pos < len(raw):
        if (kinds and kinds[-1] == b"IEND") or pos + 12 > len(raw):
            return False
        (size,) = struct.unpack(">I", raw[pos : pos + 4])
        end = pos + 8 + size
        if end + 4 > len(raw):
            return False
        (crc,) = struct.unpack(">I", raw[end : end + 4])
        if zlib.crc32(view[pos + 4 : end]) != crc:
            return False
        kind = bytes(view[pos + 4 : pos + 8])
        if not kinds:
            if kind != b"IHDR" or size != 13:
                return False
            width, height = struct.unpack(">II", raw[pos + 8 : pos + 16])
            if not (0 < width <= PICTURE_WIDTH and 0 < height <= PICTURE_HEIGHT):
                return False
        elif kind == b"IEND" and size != 0:
            return False
        kinds.append(kind)
        pos = end + 4
    return kinds[-1:] == [b"IEND"] and b"IDAT" in kinds


def _is_picture(png: object) -> TypeGuard[str]:
    """Whether *png* is base64 of a whole PNG (see :func:`_is_png`) within :data:`PICTURE_BYTES`."""
    if type(png) is not str or len(png) > PICTURE_BYTES:
        return False
    try:
        return _is_png(base64.b64decode(png, validate=True))
    except (binascii.Error, ValueError):
        return False


async def render_pages(data: bytes, pages: Sequence[int]) -> dict[int, ImagePart]:
    """A picture of each of *pages* (from 1) that could be drawn, by page.

    Only a page asked for, once, and a picture that passes :func:`_is_picture`.
    """
    limits = {
        "pages": list(pages),
        "dpi": PICTURE_DPI,
        "width": PICTURE_WIDTH,
        "height": PICTURE_HEIGHT,
        "bytes": PICTURE_BYTES,
    }
    answer = await _ask("render", limits, data, MAX_PICTURES * PICTURE_BYTES + 65_536)
    pictures = answer.get("pictures")
    made: dict[int, ImagePart] = {}
    for item in pictures[: len(pages)] if isinstance(pictures, list) else []:
        page = item.get("page") if isinstance(item, dict) else None
        png = item.get("png") if isinstance(item, dict) else None
        if (
            type(page) is int
            and page in pages
            and page not in made
            and _is_picture(png)
        ):
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
