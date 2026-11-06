"""Small PDFs built in memory for the tests of the PDF reader.

:func:`make_pdf` writes one page per entry: a string is the page's text (one
line per line of the string; any character, since a font with a ``ToUnicode``
map carries what is not ASCII), ``None`` is a page that only has a drawing, as
a scan has, with no text at all. No file in the repository is a PDF.
"""

from __future__ import annotations

import base64
import zlib
from collections.abc import Sequence

# A one-page PDF with a user password ("secret"), made once with the standard RC4
# security handler: pdfium refuses to open it without the password.
LOCKED = zlib.decompress(
    base64.b64decode(
        "eNqNVGtsFFUUDpooTIJRRLAlxktpMdHSmXvnLetG6LJAEFm3DwJI5c7M3bLNdqfOzlYKPlApoStg"
        "0KbUGlNLjRECPkEJjzUIBgQJz9IaMbQ1xYanhVoeS4p3tlA3rCTs7I97v3vO9517zrknx+fxToB5"
        "ApPT+eehYwwEHDC1MsblYlifZRpRnVjAZWhYVwnPycjNuN0MCRuODUqxLayqIID14VISYdh8Mxq2"
        "AWTYGUEjAuYBgRr6wfwUVz7NNR/bOGSWMgMcwOH2pzgI/6vFsDOJEcSTzUVUhaOfBBGQVUS1WD+J"
        "mFFLp1yOj9ekESUXEIhJbtaLgHRbhf5p1GGbhO0IUAbOfdii27RIxLRIHG6GLYhqdnLrgPTuk3GE"
        "JFXZaSRUSeygjlNYpHth4VJZCswwzc6UsG4awXApYKcbNLqgXTVhGsN6SESnxDhsO7ZOyu/Kmz/d"
        "4yB3oaenBVURm5RPDwfMJImflAYjtlUFXIjIqgR1kddosmZZBrGcOFxIorCmqwhzAlaUgNsRq6gI"
        "kXIne1wytZ7ZAHKcs3YqU2gWhYP0GgTId+RWTsnKCyRcai8EvIocAxoCweVM6LvNNds+HN4ycfzV"
        "4q5I9VtxnB27er2u8q+Sz1aMOpzf8+PcTnS9a+zIvVey1h59+OXsV5tAyWNbdpW6evsaUWX2urq6"
        "L+ZEX1zv6ds4pIFtTvyUWdzRenHv+babl4+O8A1/7dyh6NpVDZ7s4/pXsVr/kh+mVuzvX5a7zv17"
        "nJz85+MNAsiIrar7/Ma5DQmp97mz32inh8efjC/2tZVdOfDpjJ5o1Y59c8uvjcsdleX1dDctXv1r"
        "2XiYcaW/81TbL0/t7Jt4aujbC+RY9FK8um3Ort3Fvu43+uMHt5aMXH75Zu2NhubR69fc/33eR/4b"
        "sUi+nHiv2V9UM1+c9dKjK3qat7ZnbL6w/9omNM1/X8kHMxfyyx544vLOenO5bxJJnBl2tvZke9f5"
        "xqVdI57ecez9i789vy+Sp6Pd/c9Wtx3OrGzxFrtcz5QVfftOjf/gppWPb9sSqz9QMXpZb//fX8Ym"
        "nVgae+SkPz67tb3l3I68rxdUrsxPkHnZuzb+3AGLpCVHx5wY5/7jkzFn5lUPe7eVdKCtD26fGqx3"
        "CnarJLdKp6SXTkytXFmg+005T1ndJGS+fmTP9sQQz/Y9x832icMWtZzOymgYmXvhkj3UW+ga685s"
        "PPLQAXlNyPVKuoqaolIMEG1PwA/qQaTQVwsEpAqqJCPaOKw3GLLpAGMLbEyfh2XQxgUujoiInoqY"
        "JwJRZEHn9QDWNRIQIcQyj3WB6EgzFEk0dIM3BAnJEhRUKGMtYMhEoA1dBFwSMnhoECJr1AnJUIEY"
        "qoIgGLIODZ3qK1pAICKVkEUFC1ASOI7upUAggDnIKamTdJFFAgxH3wjDDf6AJIq8CAJgEIN09CRP"
        "wv9hopqGQaikY5KchvFcOh8vp2OSyN+JQU6V0jCopujaFg6GiJUsVEFwMXHuxvpN0wb8wHRNDhh4"
        "a+1xJpbIC5CyaLrIYZEoBtIpqYKhhmmikUDc92IyPzkiraoKG6i3h0vExpadTDEUIM/k5EyZ5WX+"
        "BXHfUic="
    )
)


def _ascii(line: str) -> bool:
    return line.isascii()


def _escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _cmap(chars: Sequence[str]) -> bytes:
    """A ``ToUnicode`` map that sends the two-byte code of each character to itself."""
    blocks = []
    for start in range(0, len(chars), 100):
        group = chars[start : start + 100]
        entries = "\n".join(f"<{ord(c):04X}> <{ord(c):04X}>" for c in group)
        blocks.append(f"{len(group)} beginbfchar\n{entries}\nendbfchar")
    return (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        + "\n".join(blocks)
        + "\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    ).encode("ascii")


def _content(text: str | None) -> bytes:
    if text is None:
        return b"0.6 g 72 400 300 200 re f"  # a grey block, no text
    out = ["BT", "14 TL", "72 720 Td"]
    for line in text.split("\n"):
        if _ascii(line):
            out.append(f"/F1 11 Tf ({_escape(line)}) Tj T*")
        else:
            codes = "".join(f"{ord(c):04X}" for c in line)
            out.append(f"/F2 11 Tf <{codes}> Tj T*")
    out.append("ET")
    return "\n".join(out).encode("latin-1")


def make_pdf(pages: Sequence[str | None]) -> bytes:
    """A PDF with one page for each of *pages*: its text, or ``None`` for a page with no text."""
    texts = "".join(t for t in pages if t)
    chars = sorted(set(texts) - {"\n"})
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",  # the page tree, filled in below
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            b"<< /Type /Font /Subtype /Type0 /BaseFont /Song /Encoding /Identity-H "
            b"/DescendantFonts [<< /Type /Font /Subtype /CIDFontType0 /BaseFont /Song "
            b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
            b"/DW 1000 >>] /ToUnicode 5 0 R >>"
        ),
    ]
    cmap = _cmap(chars)
    objects.append(b"<< /Length %d >>\nstream\n" % len(cmap) + cmap + b"endstream")
    kids = []
    for text in pages:
        content = _content(text)
        base = len(objects) + 1
        kids.append(f"{base} 0 R")
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                f"/Contents {base + 1} 0 R >>"
            ).encode("ascii")
        )
        objects.append(
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"
        )
    objects[1] = (
        f"<< /Type /Pages /Count {len(pages)} /Kids [{' '.join(kids)}] >>".encode()
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)
