"""The child process that opens one PDF for :mod:`localmelo.melo.executor.pdf`.

Run by path, as ``python -I _pdf_child.py MODE JSON`` with the file's bytes on
standard input and nothing else: no path, no descriptor, no environment of
melo's. It imports only the standard library and, once the bytes are in,
pypdfium2, so melo itself never loads the library, and a PDF that breaks it
takes down this process and not the gateway. It writes one JSON object to
standard output and nothing to standard error.

Modes (``JSON`` holds the limits the parent chose):

- ``text``: ``{"status": "ok", "pages": N, "texts": [...], "cut": bool}`` with
  the text of the first pages, in order, within the page and character limits
  (``cut``: the last page's text was shortened to fit).
- ``render``: ``{"status": "ok", "pictures": [{"page": n, "png": base64}]}``
  for the pages asked for, as PNG files at most ``width`` by ``height`` pixels.

``status`` is ``password`` for a document that needs one, ``broken`` for bytes
pdfium cannot open or a page it cannot read, and ``unavailable`` when
pypdfium2 is not installed. The module stays free of imports from melo on
purpose: the interpreter finds nothing next to it, so it runs by path.
"""

from __future__ import annotations

import base64
import json
import re
import struct
import sys
import zlib
from typing import Any

# pdfium's FPDF_ERR_PASSWORD.
PASSWORD_ERROR = 4

# Control characters (and lone surrogates) that stay out of the text; tab and
# line break stay.
_UNSAFE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\ud800-\udfff]")


def _clean(text: str) -> str:
    """One page's text with plain line breaks, no control characters, no trailing spaces."""
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


def _limit_cpu(seconds: int) -> None:
    """A backstop for the parent's wall clock: the kernel ends this process after *seconds* of CPU."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError):
        pass


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
                text = _clean(textpage.get_text_bounded())
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


def _pictures(document: Any, limits: dict[str, Any]) -> dict[str, Any]:
    pictures: list[dict[str, Any]] = []
    total = len(document)
    for number in limits["pages"]:
        if not 1 <= number <= total:
            continue
        page = document[number - 1]
        try:
            across, down = page.get_size()
            if not (across > 0 and down > 0):
                continue
            # 150 dpi, never wider or taller than the limits: a picture of a
            # page is the size of a screenshot.
            scale = min(
                limits["dpi"] / 72,
                limits["width"] / across,
                limits["height"] / down,
            )
            bitmap = page.render(scale=scale, rev_byteorder=True)
            try:
                png = _png(bitmap.width, bitmap.height, bitmap.stride, bitmap.buffer)
            finally:
                bitmap.close()
        except Exception:  # one page that will not draw does not end the others
            continue
        finally:
            page.close()
        pictures.append({"page": number, "png": base64.b64encode(png).decode("ascii")})
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
    _limit_cpu(int(limits["cpu"]))
    answer = run(argv[1], limits, sys.stdin.buffer.read())
    sys.stdout.buffer.write(json.dumps(answer, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
