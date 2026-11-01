"""Coloured terminal blocks for the documentation.

One parser reads a shell snippet or a console transcript and splits it into
parts: the prompt, the command, a subcommand, flags, values, quoted strings,
comments and output. Three writers draw the same parts: HTML for the website
and the Markdown guides, SVG pictures for the READMEs (GitHub cannot style
HTML) and plain text for the Copy button. The colours are the terminal's own
(`support/cli/theme.py`), and `contrast` checks them. No runtime library.

    uv run python scripts/usage_blocks.py sync [--check]   rewrite the generated regions
    uv run python scripts/usage_blocks.py contrast         print the contrast report

The blocks live in `usage_blocks.toml`. A page holds a region
`<!-- usage:ID -->...<!-- /usage -->` that `sync` fills in.
"""

from __future__ import annotations

import argparse
import html
import itertools
import math
import re
import sys
import tomllib
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("usage_blocks.toml")
PICTURES = ROOT / "docs" / "assets" / "usage"

Token = tuple[str, str]  # (role, text); the role "" is plain text

# Roles: p prompt, c command, s subcommand, f flag, a argument or value, q quoted
# string, k comment, o output, y check mark, e cross, x punctuation.
BOLD, ITALIC, TEXT_ROLES = "cs", "k", "pcsfaqkoyex"
CHAT_PROMPT = "›"  # melo's own chat marks what you typed with this (support/cli)
HUE_ROLES = "csfaqo"  # told apart by colour, weight and shape together
# The terminal's roles (support/cli/theme.py) on a light and on a dark ground.
_COLOURS = {
    "light": "bg=#fbfbf9 bar=#eef0e7 raised=#f0f0ee line=#cbd0c1 rule=#82877a "
    "ink=#1d1f1b accent=#3f5a36 p=#666a60 c=#9a560c s=#3f5a36 f=#015b77 a=#80602d "
    "q=#8252a9 k=#666a60 o=#4d5149 y=#026f5e e=#a21829 x=#5f6359",
    "dark": "bg=#161915 bar=#1e211c raised=#262925 line=#33372f rule=#767d6c "
    "ink=#eef0e9 accent=#b5cf9f p=#91978a c=#d98b3b s=#b5cf9f f=#61b1cf a=#c9a774 "
    "q=#ceacf7 k=#91978a o=#c2c7ba y=#81e7d3 e=#f6a296 x=#a7ad9f",
}
PALETTE = {n: dict(i.split("=") for i in c.split()) for n, c in _COLOURS.items()}
LABELS = {  # (the Copy button, after a copy, after a failed copy, the picture's alt)
    "en": ("Copy", "Copied", "Copy failed", "Terminal"),
    "zh": ("复制", "已复制", "复制失败", "终端"),
}

# Commands whose next plain words are subcommands (`melo memory list`).
SUBCOMMANDS = frozenset(
    {
        "melo",
        "uv",
        "git",
        "brew",
        "omlx",
        "ollama",
        "hf",
        "pip",
        "pip3",
        "ruff",
        "black",
    }
)
_SUBCOMMAND = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_ASSIGN = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)=(.*)", re.S)
_CHAIN = frozenset({"|", "||", "&&", ";", "{", "("})
# The `uv run` options that take a value in the next word (`--with pillow==13.0.0`).
UV_VALUE_OPTIONS = frozenset(
    {"--with", "--extra", "--group", "--project", "--directory", "--python", "-p"}
)
_LEX = re.compile(
    r"""(?P<sp>[ \t]+)
    |(?P<com>\#.*)
    |(?P<q>"(?:\\.|[^"\\])*(?:"|$)|'[^']*(?:'|$))
    |(?P<op>\|\||&&|>>|[|;&<>(){}]|\\$)
    |(?P<w>(?:\\.|[^\s'"|;&<>(){}\\\#])(?:\\.|[^\s'"|;&<>(){}\\])*)""",
    re.X,
)


@dataclass(frozen=True)
class Line:
    """One displayed row: a command (`cmd`), a comment (`note`) or output (`out`)."""

    kind: str
    tokens: tuple[Token, ...]
    prompt: bool = False
    glyph: str = "$"  # the prompt: `$` for a shell, `›` for melo's own chat


@dataclass(frozen=True)
class Block:
    id: str
    text: str
    mode: str = "shell"
    svg: bool = False
    title: Sequence[str] = ("", "")  # English, Chinese
    caption: Sequence[str] = ("", "")


@dataclass
class _Command:
    """Where a command line is: the next word may be a command or a subcommand."""

    expect_command: bool = True
    name: str = ""
    subcommands: bool = False
    # After `uv run`: the options that come before the command it runs, and whether the next word is an option's value.
    options: bool = False
    value_next: bool = False

    def chain(self) -> None:
        self.expect_command, self.name, self.subcommands = True, "", False
        self.options = self.value_next = False


def _assignment(word: str) -> list[Token]:
    found = _ASSIGN.fullmatch(word)
    assert found is not None
    name, value = found.groups()
    return [("f", name), ("x", "="), *([("a", value)] if value else [])]


def _run_option(word: str, state: _Command) -> list[Token] | None:
    """A word between `uv run` and the command it runs: an option, or the value of one that takes a value; None
    when the command itself begins."""
    if state.value_next:
        state.value_next = False
        return [("a", word)]
    if word.startswith("-") and len(word) > 1 and not word[1].isdigit():
        name, equals, value = word.partition("=")
        state.value_next = not equals and name in UV_VALUE_OPTIONS
        return [
            ("f", name),
            *([("x", "=")] if equals else []),
            *([("a", value)] if value else []),
        ]
    state.options = False
    return None


def _word(word: str, state: _Command) -> list[Token]:
    if state.options and (option := _run_option(word, state)) is not None:
        return option
    if state.expect_command:
        if _ASSIGN.fullmatch(word):
            return _assignment(word)
        state.expect_command, state.name = False, word
        state.subcommands = word in SUBCOMMANDS
        return [("c", word)]
    state.subcommands &= not (word.startswith("-") or _ASSIGN.fullmatch(word))
    if word.startswith("-") and len(word) > 1 and not word[1].isdigit():
        name, equals, value = word.partition("=")
        return [
            ("f", name),
            *([("x", "=")] if equals else []),
            *([("a", value)] if value else []),
        ]
    if _ASSIGN.fullmatch(word):
        return _assignment(word)
    if state.subcommands and _SUBCOMMAND.fullmatch(word):
        if state.name == "uv" and word == "run":  # `uv run melo ...` runs a command
            state.chain()
            state.options = True
        return [("s", word)]
    state.subcommands = False
    return [("a", word)]


def _command_tokens(body: str, state: _Command) -> tuple[list[Token], bool]:
    """Tokens of one command line, and whether a trailing backslash continues it."""
    tokens: list[Token] = []
    for match in _LEX.finditer(body):
        kind, text = match.lastgroup, match.group()
        if kind == "w":
            tokens.extend(_word(text, state))
            continue
        tokens.append(({"sp": "", "com": "k", "q": "q", "op": "x"}[str(kind)], text))
        if kind == "op" and text in _CHAIN:
            state.chain()
    return tokens, body.rstrip().endswith("\\")


def _chat_line(row: str) -> Line:
    """A message typed into melo's chat: a `/command` with its words, or plain words."""
    if row.startswith("/"):
        tokens, _ = _command_tokens(row, _Command())
    else:
        tokens = [("q", row)]
    return Line("cmd", tuple(tokens), prompt=True, glyph=CHAT_PROMPT)


def parse(text: str, mode: str = "shell") -> list[Line]:
    """Split *text* into rows.

    In `console` mode only `$ ` rows are commands; in `chat` mode every row is a
    message typed into melo's own chat (`/approvals auto`, or a sentence).
    """
    lines: list[Line] = []
    state, continued = _Command(), False
    for row in text.strip("\n").split("\n"):
        if mode == "chat":
            lines.append(_chat_line(row) if row else Line("out", ()))
            continue
        if mode == "console" and not continued and not row.startswith("$ "):
            mark = {"✓": "y", "✗": "e"}.get(row[:1])
            tokens = ((mark, row[0]), ("o", row[1:])) if mark else (("o", row),)
            lines.append(Line("out", tokens if row else ()))
            continue
        body = row[2:] if mode == "console" and not continued else row
        if not continued:
            state = _Command()
            if body.lstrip().startswith("#"):
                lines.append(Line("note", (("k", body),)))
                continue
        tokens_, next_continued = _command_tokens(body, state)
        lines.append(Line("cmd", tuple(tokens_), prompt=not continued))
        continued = next_continued
    return lines


def copy_text(lines: Sequence[Line]) -> str:
    """The commands alone: no prompt, no output, no comments."""
    rows = (
        "".join(t for r, t in ln.tokens if r != "k").rstrip()
        for ln in lines
        if ln.kind == "cmd"
    )
    return "\n".join(row for row in rows if row)


def cells(text: str) -> int:
    """Width of *text* in terminal cells: Chinese characters take two."""
    wide = unicodedata.east_asian_width
    return sum(
        0 if unicodedata.combining(c) else 2 if wide(c) in "WF" else 1 for c in text
    )


def _pick(values: Sequence[str], lang: str) -> str:
    return values[lang == "zh"] or values[0]


def _span(role: str, text: str) -> str:
    escaped = html.escape(text, quote=False)
    if role == "a" and len(text) > 24:  # a long path or URL may wrap after a slash
        escaped = re.sub(r"(?<!/)/(?!/)", "/<wbr>", escaped)
    return f'<span class="{role}">{escaped}</span>' if role else escaped


def render_html(
    block: Block,
    lang: str = "en",
    *,
    text: str | None = None,
    mode: str | None = None,
    show_title: bool = True,
) -> str:
    """The block as a window with a title, a Copy button and coloured rows."""
    lines = parse(block.text if text is None else text, mode or block.mode)
    copy = copy_text(lines)
    copy_label, done, failed, _ = LABELS[lang]
    title, caption = (html.escape(_pick(v, lang)) for v in (block.title, block.caption))
    bar = f'<div class="usage-bar"><span class="usage-title">{title if show_title else ""}</span>'
    if copy:
        bar += (
            f'<button class="usage-copy" type="button" hidden aria-live="polite" '
            f'data-done="{done}" data-fail="{failed}">{copy_label}</button>'
        )
    rows = "\n".join(
        '<span class="ln {}">{}{}</span>'.format(
            ln.kind,
            f'<span class="p">{ln.glyph}</span> ' if ln.prompt else "",
            "".join(_span(r, t) for r, t in ln.tokens),
        )
        for ln in lines
    )
    attrs = (f' aria-label="{title}"' if title else "") + (
        f' data-copy="{html.escape(copy).replace(chr(10), "&#10;")}"' if copy else ""
    )
    note = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (
        f'<figure class="usage"{attrs}>{bar}</div>'
        f'<pre class="usage-body"><code>{rows}</code></pre>{note}</figure>'
    )


def render_fence(code: str, info: str, lang: str = "en") -> str:
    """A Markdown fence tagged `console` or `shell`, as the same block."""
    return render_html(Block("fence", code, mode=info), lang, text=code.rstrip("\n"))


# SVG: the same rows drawn at fixed cell positions, for the READMEs.
CELL_W, ROW_H, FONT, PAD_X, PAD_Y, BAR_H = 8.4, 22, 14, 20, 14, 36


def _num(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def render_svg(block: Block, scheme: str, lang: str = "en") -> str:
    c = PALETTE[scheme]
    lines = parse(block.text, block.mode)
    cols = max(
        (cells("".join(t for _, t in ln.tokens)) + 2 * ln.prompt for ln in lines),
        default=0,
    )
    w, h = max(
        380, math.ceil(cols * CELL_W + 2 * PAD_X)
    ), BAR_H + 2 * PAD_Y + ROW_H * len(lines)
    css = "".join(
        f".{r}{{fill:{c[r]}{';font-weight:700' * (r in BOLD)}{';font-style:italic' * (r in ITALIC)}}}"
        for r in TEXT_ROLES
    )
    rows = lambda i: BAR_H + PAD_Y + ROW_H * i  # noqa: E731
    bands = "".join(
        f'<rect y="{rows(i)}" width="{w}" height="{ROW_H}" fill="{c["raised"]}"/>'
        f'<rect y="{rows(i)}" width="3" height="{ROW_H}" fill="{c["accent"]}"/>'
        for i, ln in enumerate(lines)
        if ln.kind == "cmd"
    )
    dots = "".join(
        f'<circle cx="{18 + 17 * i}" cy="{BAR_H // 2}" r="4.5" fill="{c["rule"]}"/>'
        for i in range(3)
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'role="img"><title>{html.escape(_pick(block.title, lang))}</title>',
        f"<style>text{{font:{FONT}px ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',"
        f"monospace;fill:{c['ink']};white-space:pre}}{css}</style>",
        f'<clipPath id="w"><rect width="{w}" height="{h}" rx="10"/></clipPath><g clip-path="url(#w)">'
        f'<rect width="{w}" height="{h}" fill="{c["bg"]}"/><rect width="{w}" height="{BAR_H}" '
        f'fill="{c["bar"]}"/><rect y="{BAR_H - 1}" width="{w}" height="1" fill="{c["line"]}"/>{dots}{bands}',
    ]
    for i, line in enumerate(lines):
        column, spans = 0, []
        for role, text in (
            (("p", line.glyph), ("", " "), *line.tokens) if line.prompt else line.tokens
        ):
            if text.strip():  # textLength keeps the width the same in any font
                cls = f' class="{role}"' if role else ""
                spans.append(
                    f'<tspan x="{_num(PAD_X + column * CELL_W)}" '
                    f'textLength="{_num(cells(text) * CELL_W)}"{cls}>{html.escape(text)}</tspan>'
                )
            column += cells(text)
        if spans:
            out.append(
                f'<text y="{_num(rows(i) + 15.5)}" xml:space="preserve">{"".join(spans)}</text>'
            )
    out.append(
        f'<rect x=".5" y=".5" width="{w - 1}" height="{h - 1}" rx="9.5" fill="none" '
        f'stroke="{c["line"]}"/></g></svg>\n'
    )
    return "\n".join(out)


def render_markdown(block: Block, lang: str) -> str:
    """For a README: the light and dark pictures, then the commands as text."""
    base = f"docs/assets/usage/{block.id}"
    alt = html.escape(f"{LABELS[lang][3]}: {_pick(block.title, lang)}")
    return (
        f'<picture><source media="(prefers-color-scheme: dark)" srcset="{base}.dark.svg">'
        f'<img alt="{alt}" src="{base}.light.svg"></picture>'
        f"\n\n```bash\n{copy_text(parse(block.text, block.mode))}\n```"
    )


def render_steps(
    steps: Sequence[Mapping[str, Any]], blocks: Mapping[str, Block], lang: str
) -> str:
    """The numbered cards: a heading, one line, then a terminal block or a picture."""
    items = []
    for step in steps:
        body = [
            f'<li class="step"><h3><a href="{step["href"]}">{html.escape(_pick(step["title"], lang))}</a></h3>',
            f'<p>{html.escape(_pick(step["line"], lang))}</p>',
        ]
        names = step.get("blocks", [])
        # A card with one block already names it in its heading.
        body += [render_html(blocks[n], lang, show_title=len(names) > 1) for n in names]
        if "image" in step:
            body.append(
                f'<figure class="step-shot"><img src="../assets/img/{step["image"]}" '
                f'alt="{html.escape(_pick(step["alt"], lang), quote=True)}" width="1280" height="800" '
                'loading="lazy"></figure>'
            )
        items.append("\n".join(body) + "</li>")
    return '<ol class="steps">\n' + "\n".join(items) + "\n</ol>"


def load(source: Path = SOURCE) -> tuple[dict[str, Block], list[dict[str, Any]]]:
    data = tomllib.loads(source.read_text(encoding="utf-8"))
    blocks = {
        name: Block(
            name,
            entry["text"],
            entry.get("mode", "shell"),
            entry.get("svg", False),
            entry["title"],
            entry.get("caption", ("", "")),
        )
        for name, entry in data["block"].items()
    }
    return blocks, data["step"]


_REGION = re.compile(r"<!-- usage:(?P<id>[\w-]+) -->(?P<body>.*?)<!-- /usage -->", re.S)


def targets(root: Path = ROOT) -> Iterator[tuple[Path, str]]:
    """The files that hold regions, with their language."""
    yield root / "README.md", "en"
    yield root / "README.zh-CN.md", "zh"
    for folder, lang in (("en", "en"), ("zh-CN", "zh")):
        yield from ((p, lang) for p in sorted((root / "docs" / folder).glob("*.html")))


def sync_text(text: str, path: Path, lang: str) -> str:
    blocks, steps = load()

    def fill(match: re.Match[str]) -> str:
        name = match.group("id")
        if path.suffix == ".md":
            body = "\n" + render_markdown(blocks[name], lang) + "\n"
        elif name == "steps":
            body = render_steps(steps, blocks, lang)
        else:
            body = render_html(blocks[name], lang)
        return f"<!-- usage:{name} -->{body}<!-- /usage -->"

    return _REGION.sub(fill, text)


def sync(check: bool = False, root: Path = ROOT) -> list[str]:
    """Rewrite every region and picture; with *check*, only report what is stale."""
    blocks, _ = load()
    wanted = {
        PICTURES / f"{b.id}.{scheme}.svg": render_svg(b, scheme)
        for b in blocks.values()
        if b.svg
        for scheme in PALETTE
    }
    for path, lang in targets(root):
        if path.is_file():
            wanted[path] = sync_text(path.read_text(encoding="utf-8"), path, lang)
    stale = []
    for path, text in wanted.items():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            stale.append(str(path.relative_to(root)))
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8", newline="\n")
    return stale


# Contrast: WCAG ratios, and how far apart the hues stay for colour-blind readers.
def _linear(colour: str) -> list[float]:
    return [
        ((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92
        for v in (int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    ]


def ratio(a: str, b: str) -> float:
    lum = [
        0.2126 * (x := _linear(c))[0] + 0.7152 * x[1] + 0.0722 * x[2] for c in (a, b)
    ]
    return (max(lum) + 0.05) / (min(lum) + 0.05)


_CVD = {  # Machado et al. 2009, full severity, on linear RGB
    "protanopia": "0.152286 1.052583 -0.204868 0.114503 0.786281 0.099216 -0.003882 -0.048116 1.051998",
    "deuteranopia": "0.367322 0.860646 -0.227968 0.280085 0.672501 0.047413 -0.011820 0.042940 0.968881",
    "tritanopia": "1.255528 -0.076749 -0.178779 -0.078411 0.930809 0.147602 0.004733 0.691367 0.303900",
}


def hue_distance(a: str, b: str, kind: str) -> float:
    """Lab distance between two colours as seen with a kind of colour blindness."""
    m = [float(v) for v in _CVD[kind].split()]
    labs = []
    for colour in (a, b):
        lin = _linear(colour)
        r, g, bl = (
            min(1.0, max(0.0, sum(m[3 * i + j] * lin[j] for j in range(3))))
            for i in range(3)
        )
        x, y, z = (
            (0.4124 * r + 0.3576 * g + 0.1805 * bl) / 0.95047,
            0.2126 * r + 0.7152 * g + 0.0722 * bl,
            (0.0193 * r + 0.1192 * g + 0.9505 * bl) / 1.08883,
        )
        fx, fy, fz = (
            t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116 for t in (x, y, z)
        )
        labs.append((116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)))
    return math.dist(*labs)


MIN_TEXT, MIN_MARK, MIN_HUE = 4.5, 3.0, 7.0


def contrast_report() -> tuple[list[str], list[str]]:
    """The report lines and the problems (an empty list means every check passes)."""
    rows, problems = [], []

    def check(scheme: str, what: str, got: float, need: float) -> None:
        rows.append(f"  {what:<28} {got:5.2f}  (needs {need})")
        if got < need:
            problems.append(f"{scheme} {what}: {got:.2f} < {need}")

    for scheme, c in PALETTE.items():
        rows.append(f"{scheme}: window ground {c['bg']}, command row {c['raised']}")
        for role in TEXT_ROLES:
            check(
                scheme, f"{role} {c[role]} on ground", ratio(c[role], c["bg"]), MIN_TEXT
            )
            check(
                scheme,
                f"{role} {c[role]} on command row",
                ratio(c[role], c["raised"]),
                MIN_TEXT,
            )
        check(scheme, "title bar text", ratio(c["k"], c["bar"]), MIN_TEXT)
        check(scheme, "window dots", ratio(c["rule"], c["bar"]), MIN_MARK)
        check(scheme, "command rule", ratio(c["accent"], c["raised"]), MIN_MARK)
        for kind in _CVD:
            gap, pair = min(
                (hue_distance(c[a], c[b], kind), a + b)
                for a, b in itertools.combinations(HUE_ROLES, 2)
            )
            check(scheme, f"closest hues {pair} ({kind[:4]})", gap, MIN_HUE)
    return rows, problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("command", choices=("sync", "contrast"))
    parser.add_argument("--check", action="store_true", help="only report stale files")
    args = parser.parse_args(argv)
    if args.command == "contrast":
        rows, problems = contrast_report()
        print("\n".join(rows + (problems or ["All contrast checks pass."])))
        return 1 if problems else 0
    stale = sync(check=args.check)
    for name in stale:
        print(("stale: " if args.check else "wrote: ") + name)
    if args.check and stale:
        print("Run: uv run python scripts/usage_blocks.py sync", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
