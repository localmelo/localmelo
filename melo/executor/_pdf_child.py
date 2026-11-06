"""The child process that opens one PDF for :mod:`localmelo.melo.executor.pdf`.

Run by path, as ``python -I _pdf_child.py MODE JSON`` with the file's bytes on
standard input and nothing else: no path, no descriptor, no environment of
melo's. It imports only the standard library and, once the bytes are in,
pypdfium2, so melo itself never loads the library, and a crash of it
takes down this process and not the gateway. It writes one JSON object to
standard output and nothing to standard error. It ends itself: the kernel stops
it after ``cpu`` seconds of CPU, and an alarm after ``wall`` seconds of any
kind, so its life is bounded even if the process that started it is gone.

Modes (``JSON`` holds the limits the parent chose):

- ``text``: ``{"status": "ok", "pages": N, "texts": [...], "cut": bool}`` with
  the text of the first pages, in order, within the page and character limits
  (``cut``: the last page's text was shortened to fit).
- ``render``: ``{"status": "ok", "pictures": [{"page": n, "png": base64}]}``
  for the pages asked for, as PNG files at most ``width`` by ``height`` pixels
  and at most ``bytes`` characters of base64 each: a page whose picture is
  larger is drawn smaller until it fits (never below a width of
  ``_MIN_WIDTH`` pixels, and then it is left out).

``status`` is ``password`` for a document that needs one, ``broken`` for bytes
pdfium cannot open or a page it cannot read, and ``unavailable`` when
pypdfium2 is not installed. The module stays free of imports from melo on
purpose: the interpreter finds nothing next to it, so it runs by path.
"""

from __future__ import annotations

import base64
import json
import re
import signal
import struct
import sys
import zlib
from typing import Any

# pdfium's FPDF_ERR_PASSWORD.
PASSWORD_ERROR = 4

# Control characters (and lone surrogates) that stay out of the text; tab and
# line break stay.
_UNSAFE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\ud800-\udfff]")


def clean(text: str) -> str:
    """One page's text with plain line breaks, no control characters, no trailing spaces.

    The parent applies it again to what it is answered: it does not take the
    child's word for it.
    """
    text = _UNSAFE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def _png(width: int, height: int, stride: int, pixels: Any) -> bytes:
    """An RGB PNG of a bitmap whose rows are *stride* bytes apart."""
    view = memoryview(pixels).cast("B")
    row = width * 3
    raw = bytearray()
    for y in range(height):
        raw += b"\x00"  # filter: none
        raw += view[y * stride : y * stride + row]

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body)
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    head = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", head)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 4))
        + chunk(b"IEND", b"")
    )


def _limit(cpu: int, wall: int) -> None:
    """Bound this process: *cpu* seconds of CPU (the kernel ends it), *wall* seconds of any kind (an alarm ends it).

    The alarm is what ends a child that waits on a parent that is gone, or on
    a call that never returns: neither uses CPU, and a native call cannot
    ignore the signal.
    """
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError):
        pass
    if hasattr(signal, "alarm"):
        signal.alarm(wall)


def _texts(document: Any, limits: dict[str, Any]) -> dict[str, Any]:
    total = len(document)
    texts: list[str] = []
    room = int(limits["chars"])
    cut = False
    for index in range(min(total, int(limits["pages"]))):
        page = document[index]
        try:
            textpage = page.get_textpage()
            try:
                # Rows kept whole: a place and a date stay on the line of their school.
                text = clean(textpage.get_text_bounded())
            finally:
                textpage.close()
        finally:
            page.close()
        texts.append(text[: max(room, 0)])
        cut = len(text) > room
        room -= len(text)
        if room <= 0:
            break
    return {"status": "ok", "pages": total, "texts": texts, "cut": cut}


# A picture is never drawn narrower than this: below it a page is no longer legible.
_MIN_WIDTH = 480
# Draws before a page is given up on, each smaller than the one before.
_TRIES = 5


def _picture(page: Any, limits: dict[str, Any]) -> str | None:
    """Page *page* as base64 of a PNG within the limits, or None when it cannot be made so."""
    across, down = page.get_size()
    if not (across > 0 and down > 0):
        return None
    # 150 dpi, never wider or taller than the limits: a picture of a page is
    # the size of a screenshot.
    scale = min(limits["dpi"] / 72, limits["width"] / across, limits["height"] / down)
    for _ in range(_TRIES):
        if across * scale < _MIN_WIDTH:
            return None
        bitmap = page.render(scale=scale, rev_byteorder=True)
        try:
            png = base64.b64encode(
                _png(bitmap.width, bitmap.height, bitmap.stride, bitmap.buffer)
            ).decode("ascii")
        finally:
            bitmap.close()
        if len(png) <= limits["bytes"]:
            return png
        # The size follows the area, so the side shrinks by the root of the
        # excess, a little more to land under it.
        scale *= 0.9 * (limits["bytes"] / len(png)) ** 0.5
    return None


def _pictures(document: Any, limits: dict[str, Any]) -> dict[str, Any]:
    pictures: list[dict[str, Any]] = []
    total = len(document)
    for number in limits["pages"]:
        if not 1 <= number <= total:
            continue
        page = document[number - 1]
        try:
            png = _picture(page, limits)
        except Exception:  # one page that will not draw does not end the others
            continue
        finally:
            page.close()
        if png is not None:
            pictures.append({"page": number, "png": png})
    return {"status": "ok", "pictures": pictures}


def run(mode: str, limits: dict[str, Any], data: bytes) -> dict[str, Any]:
    """The answer for *mode* about the PDF in *data*."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return {"status": "unavailable"}
    try:
        document = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        code = getattr(exc, "err_code", None)
        return {"status": "password" if code == PASSWORD_ERROR else "broken"}
    except Exception:
        return {"status": "broken"}
    try:
        if len(document) == 0:
            return {"status": "broken"}
        return (
            _pictures(document, limits)
            if mode == "render"
            else _texts(document, limits)
        )
    except Exception:
        return {"status": "broken"}
    finally:
        document.close()


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("text", "render"):
        return 2
    limits = json.loads(argv[2])
    _limit(int(limits["cpu"]), int(limits["wall"]))
    answer = run(argv[1], limits, sys.stdin.buffer.read())
    sys.stdout.buffer.write(json.dumps(answer, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
