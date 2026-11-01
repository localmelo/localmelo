"""Draw the pictures that show an art pack: the preview sheet `art_pack.py apply` leaves for you to look at, and the
two pictures of the guide (docs/guides/art-pack.md). Everything is drawn from the moves folder (and the brand pictures)
that a pack made, never by hand. Pillow draws the text with its built-in font, so the same files give the same bytes.

    uv run --with pillow==13.0.0 python scripts/art_pack.py figures
"""

from __future__ import annotations

from pathlib import Path

import build_moves  # type: ignore[import-not-found]
from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]

# The two pages the preview shows Melo on: a light and a dark one.
LIGHT, DARK = (251, 248, 243, 255), (28, 27, 26, 255)
INK = {LIGHT: (59, 49, 40, 255), DARK: (232, 226, 218, 255)}
# Okabe-Ito colours: three outlines and a line that stay apart for colour-blind eyes.
BLUE, GREEN, VERMILLION, PURPLE = (
    (0, 114, 178, 255),
    (0, 158, 115, 255),
    (213, 94, 0, 255),
    (204, 121, 167, 255),
)
MOVE_NAMES = tuple(build_moves.POSES)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size)


def opened(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGBA")


def shrink(image: Image.Image, side: int) -> Image.Image:
    """`image` scaled to `side` px square on premultiplied colour (no dark halo)."""
    return (
        image.convert("RGBa")
        .resize((side, side), Image.Resampling.LANCZOS)
        .convert("RGBA")
    )


def centred(
    draw: ImageDraw.ImageDraw, text: str, middle: int, top: int, size: int, fill: tuple
) -> None:
    box = draw.textbbox((0, 0), text, font=font(size))
    draw.text((middle - (box[2] - box[0]) / 2, top), text, font=font(size), fill=fill)


GAP, MARGIN = 24, 24


def cell_width(picture: Image.Image, caption: str) -> int:
    """The room a picture and its caption take side by side."""
    return int(max(picture.width, round(font(14).getlength(caption))))


def row_width(items: list[tuple[Image.Image, str]]) -> int:
    return MARGIN * 2 + sum(cell_width(*item) + GAP for item in items) - GAP


def band(
    page: tuple[int, int, int, int],
    rows: list[tuple[str, list[tuple[Image.Image, str]]]],
    width: int,
) -> Image.Image:
    """One page of the preview: each row is a label and its pictures with a caption under each."""
    ink = INK[page]
    heights = [
        24 + max(picture.height for picture, _ in items) + 48 for _, items in rows
    ]
    sheet = Image.new("RGBA", (width, MARGIN * 2 + sum(heights)), page)
    draw = ImageDraw.Draw(sheet)
    top = MARGIN
    for (label, items), height in zip(rows, heights, strict=True):
        draw.text((MARGIN, top), label, font=font(16), fill=ink)
        x = MARGIN
        for picture, caption in items:
            room = cell_width(picture, caption)
            sheet.alpha_composite(picture, (x + (room - picture.width) // 2, top + 24))
            centred(
                draw, caption, x + room // 2, top + 24 + picture.height + 6, 14, ink
            )
            x += room + GAP
        top += height
    return sheet


def preview(
    moves: Path,
    out: Path,
    icon: Image.Image | None = None,
    menubar: tuple[Image.Image, Image.Image] | None = None,
    emblem: Image.Image | None = None,
) -> None:
    """The preview sheet, a PNG, of what a pack made in `moves`: every poster and still, the faces at the sizes they
    are shown, the tab icons at their real size, and the brand pictures when the pack gave some; on a light and a
    dark page."""
    posters = [
        (shrink(opened(moves / f"poster-{name}.webp"), 128), name)
        for name in (*MOVE_NAMES, *build_moves.STILLS)
    ]
    faces: list[tuple[Image.Image, str]] = []
    for name in (*build_moves.FACES, "brand"):
        face = opened(moves / f"face-{name}.png")
        group = Image.new("RGBA", (96 + 48 + 28 + 24 + 3 * 10, 96))
        x = 0
        for side in (96, 48, 28, 24):
            small = face if side == 96 else shrink(face, side)
            group.alpha_composite(small, (x, (96 - side) // 2))
            x += side + 10
        faces.append((group, name))
    tabs: list[tuple[Image.Image, str]] = [
        (opened(moves / f"favicon-{side}.png"), f"{side} px")
        for side in build_moves.FAVICONS
    ]
    tabs.append(
        (
            opened(moves / "favicon-16.png").resize((64, 64), Image.Resampling.NEAREST),
            "16 px, enlarged",
        )
    )
    brand: list[tuple[Image.Image, str]] = []
    if icon is not None:
        brand += [
            (shrink(icon, 128), "app icon"),
            (shrink(icon, 32), "app icon, 32 px"),
        ]
    if menubar is not None:
        brand += [(menubar[1], "menu bar, 36 px"), (menubar[0], "menu bar, 18 px")]
    if emblem is not None:
        height = 128
        brand.append(
            (
                emblem.resize(
                    (round(emblem.width * height / emblem.height), height),
                    Image.Resampling.LANCZOS,
                ),
                "emblem",
            )
        )
    rows = [
        ("Moves: the poster of each state, as the empty chat shows it", posters[:6]),
        ("Stills: a run that ended well, a failure, nothing known", posters[6:]),
        ("Faces, at 96, 48, 28 and 24 px", faces),
        ("Tab icons, at their real size", tabs),
    ]
    if brand:
        rows.append(("Brand", brand))
    width = max(row_width(items) for _, items in rows)
    pages = [band(page, rows, width) for page in (LIGHT, DARK)]
    sheet = Image.new("RGBA", (width, sum(p.height for p in pages)))
    sheet.paste(pages[0], (0, 0))
    sheet.paste(pages[1], (0, pages[0].height))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.convert("RGB").save(out, "PNG", optimize=True)


def checker(side: int, cell: int = 16) -> Image.Image:
    """A grey-and-white board, so a transparent canvas shows."""
    board = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    draw = ImageDraw.Draw(board)
    for y in range(0, side, cell):
        for x in range(0, side, cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle(
                    (x, y, x + cell - 1, y + cell - 1), fill=(236, 232, 226, 255)
                )
    return board


def dashed(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    colour: tuple[int, int, int, int],
    width: int = 3,
    dash: int = 10,
) -> None:
    """A dashed rectangle."""
    x0, y0, x1, y1 = box
    for start in range(0, int(x1 - x0), dash * 2):
        end = min(start + dash, x1 - x0)
        draw.line((x0 + start, y0, x0 + end, y0), fill=colour, width=width)
        draw.line((x0 + start, y1, x0 + end, y1), fill=colour, width=width)
    for start in range(0, int(y1 - y0), dash * 2):
        end = min(start + dash, y1 - y0)
        draw.line((x0, y0 + start, x0, y0 + end), fill=colour, width=width)
        draw.line((x1, y0 + start, x1, y0 + end), fill=colour, width=width)


def moves_figure(moves: Path) -> Image.Image:
    """The six moves' posters with their state names, three across."""
    cell, margin = 256, 28
    columns = 3
    rows = -(-len(MOVE_NAMES) // columns)
    width = margin + columns * (cell + margin)
    height = margin + rows * (cell + 40 + margin // 2)
    sheet = Image.new("RGBA", (width, height), LIGHT)
    draw = ImageDraw.Draw(sheet)
    for index, name in enumerate(MOVE_NAMES):
        x = margin + (index % columns) * (cell + margin)
        y = margin + (index // columns) * (cell + 40 + margin // 2)
        poster = opened(moves / f"poster-{name}.webp")
        if poster.width != cell:
            poster = shrink(poster, cell)
        sheet.alpha_composite(poster, (x, y))
        centred(draw, name, x + cell // 2, y + cell + 8, 22, INK[LIGHT])
    return sheet


def canvas_figure(moves: Path) -> Image.Image:
    """The canvas, with the anchor line and the three boxes the pictures are cut by."""
    side, margin, legend = 420, 24, 330
    sheet = Image.new("RGBA", (margin * 3 + side + legend, margin * 2 + side), LIGHT)
    draw = ImageDraw.Draw(sheet)
    board = checker(side)
    poster = shrink(opened(moves / "poster-startup.webp"), side)
    board.alpha_composite(poster)
    sheet.paste(board, (margin, margin))
    draw.rectangle(
        (margin - 1, margin - 1, margin + side, margin + side),
        outline=INK[LIGHT],
        width=2,
    )
    unit = side / 512

    def at(box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = (margin + edge * unit for edge in box)
        return x0, y0, x1, y1

    # The anchor line first and wider, so it shows round the head box's bottom edge, which lies on it.
    anchor = margin + 488 * unit
    draw.line((margin, anchor, margin + side, anchor), fill=PURPLE, width=7)
    middle = margin + 256 * unit
    draw.line((middle, anchor - 12, middle, anchor + 12), fill=PURPLE, width=3)
    draw.rectangle(at(build_moves.HEAD_BOX), outline=BLUE, width=3)
    dashed(draw, at(build_moves.HEAD_FAVICON_BOX), GREEN)
    dashed(draw, at(build_moves.FAVICON_BOX), VERMILLION)
    x = margin * 2 + side
    top = margin + 6
    entries = [
        (PURPLE, "line", "Anchor line", "Where the paws rest: (256, 488)"),
        (BLUE, "box", "Head box", "A head is fitted in this box"),
        (GREEN, "dash", "Tab icon from a head", "The head, cut closer"),
        (VERMILLION, "dash", "Tab icon, no head", "Cut from the wake-up's end"),
    ]
    for colour, kind, title, note in entries:
        if kind == "line":
            draw.line((x, top + 11, x + 34, top + 11), fill=colour, width=3)
        elif kind == "box":
            draw.rectangle((x, top + 2, x + 34, top + 20), outline=colour, width=3)
        else:
            dashed(draw, (x, top + 2, x + 34, top + 20), colour, width=3, dash=6)
        draw.text((x + 50, top - 2), title, font=font(20), fill=INK[LIGHT])
        draw.text((x + 50, top + 24), note, font=font(15), fill=INK[LIGHT])
        top += 74
    draw.text(
        (x, margin + side - 56),
        "Sizes are for a 512 px canvas;",
        font=font(15),
        fill=INK[LIGHT],
    )
    draw.text(
        (x, margin + side - 34),
        "a larger or smaller one scales them.",
        font=font(15),
        fill=INK[LIGHT],
    )
    return sheet


def save_figure(image: Image.Image, path: Path) -> None:
    """A lossless WebP, so the lines and the text stay sharp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, "WEBP", lossless=True, method=6)


FIGURES = ("art-pack-moves.webp", "art-pack-canvas.webp")


def figures(moves: Path, out: Path) -> list[str]:
    """Draw the guide's two pictures from the posters in `moves` into `out`; their names."""
    save_figure(moves_figure(moves), out / FIGURES[0])
    save_figure(canvas_figure(moves), out / FIGURES[1])
    return list(FIGURES)
