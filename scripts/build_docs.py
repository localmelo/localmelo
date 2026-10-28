"""Render the documentation manifest into an isolated static preview.

This foundation does not write generated pages into the repository. The
manifest currently inventories old pages; missing Chinese sources are not
fabricated. Markdown is untrusted: raw HTML is escaped and unsafe URLs are
rendered as text, not links. Python 3.11+, markdown-it-py 4.2.0 (MIT).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import posixpath
import shutil
import tempfile
import unicodedata
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import unquote, urlsplit, urlunsplit

from markdown_it import MarkdownIt


class DocumentationError(ValueError):
    """The manifest or its paths cannot be built safely."""


def _safe_url(value: str) -> bool:
    """Allow ordinary local links, HTTP(S) and mailto, not scripts or files."""
    decoded = unquote(html.unescape(value)).strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in decoded):
        return False
    if "\\" in decoded or decoded.startswith("//"):
        return False
    try:
        parsed = urlsplit(decoded)
    except ValueError:
        return False
    return not parsed.scheme or parsed.scheme.lower() in ("http", "https", "mailto")


def _slug(text: str) -> str:
    """A deterministic, GitHub-like heading anchor; repeated names get suffixes."""
    text = text.strip().lower()
    return "".join(
        "-" if char.isspace() else char
        for char in text
        if char.isspace() or char in "-_" or unicodedata.category(char)[0] not in "PCS"
    )


def _parser(rewrite: Callable[[str], str] | None = None) -> MarkdownIt:
    parser = (
        MarkdownIt("commonmark", {"html": False})
        .enable("table")
        .enable("strikethrough")
    )
    # Markdown-it's default permits some data:image URLs; docs need none.
    parser.validateLink = _safe_url  # type: ignore[assignment]

    def headings_and_links(state: Any) -> None:
        used: set[str] = set()
        for index, token in enumerate(state.tokens):
            if token.type == "heading_open":
                inline = state.tokens[index + 1]
                text = "".join(
                    child.content
                    for child in inline.children or []
                    if child.type in ("text", "code_inline")
                )
                base = _slug(text)
                candidate = base
                count = 0
                while candidate in used:
                    count += 1
                    candidate = f"{base}-{count}"
                used.add(candidate)
                token.attrSet("id", candidate)
            if token.children:
                for child in token.children:
                    attr = (
                        "href"
                        if child.type == "link_open"
                        else "src" if child.type == "image" else ""
                    )
                    if not attr:
                        continue
                    value = child.attrGet(attr)
                    if value and rewrite:
                        child.attrSet(attr, rewrite(value))

    parser.core.ruler.after("inline", "docs_headings_and_links", headings_and_links)
    return parser


def render_markdown(markdown: str) -> str:
    """Render tables, code and headings with safe links and escaped HTML."""
    return cast(str, _parser().render(markdown))


def _inside(root: Path, path: str, kind: str) -> Path:
    pure = PurePosixPath(path)
    if not path or pure.is_absolute() or ".." in pure.parts or "\\" in path:
        raise DocumentationError(f"unsafe {kind} path: {path!r}")
    result = (root / pure).resolve()
    if not result.is_relative_to(root.resolve()):
        raise DocumentationError(f"unsafe {kind} path: {path!r}")
    return result


def _relative(source: str, target: str) -> str:
    return posixpath.relpath(target, posixpath.dirname(source) or ".")


def _rewriter(
    source: str, output: str, targets: dict[str, str]
) -> Callable[[str], str]:
    def rewrite(value: str) -> str:
        if not _safe_url(value):
            return "#"
        parts = urlsplit(value)
        if parts.scheme or not parts.path:
            return value
        normalized = posixpath.normpath(
            posixpath.join(posixpath.dirname(source), unquote(parts.path))
        )
        if normalized in targets:
            return urlunsplit(
                (
                    "",
                    "",
                    _relative(output, targets[normalized]),
                    parts.query,
                    parts.fragment,
                )
            )
        return value

    return rewrite


def _page(title: str, content: str, *, lang: str, nav: str, alternate: str) -> str:
    language = html.escape(lang, quote=True)
    label = html.escape(title)
    switch = (
        f'<a hreflang="{html.escape("zh-CN" if lang == "en" else "en")}" href="{html.escape(alternate, quote=True)}">{"中文" if lang == "en" else "English"}</a>'
        if alternate
        else ""
    )
    return (
        '<!doctype html>\n<html lang="' + language + '">\n<head>\n'
        '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{label} · localmelo</title>\n</head>\n<body>\n"
        '<a href="#main">Skip to content</a>\n'
        f'<nav aria-label="Documentation">{nav}</nav>\n{switch}\n'
        f'<main id="main"><article>\n{content}</article></main>\n</body>\n</html>\n'
    )


def build(manifest_path: Path, output_dir: Path) -> list[Path]:
    """Build a preview into *output_dir*; never create missing translations.

    Manifest source paths are relative to the parent of ``docs/``. Existing
    hand-authored HTML is copied verbatim in this foundation; only Markdown
    is rendered. Destination must be outside the source tree so a preview
    cannot overwrite authored or committed site files.
    """
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent.parent
    output_dir = output_dir.resolve()
    if output_dir == root or output_dir.is_relative_to(root):
        raise DocumentationError("output directory must be outside the source tree")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise DocumentationError("output directory must be empty")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise DocumentationError("unsupported manifest version")
    pages = data["pages"]
    targets: dict[str, str] = {}
    seen: set[str] = set()
    for page in pages:
        for lang in ("en", "zh"):
            source, output = page.get(f"{lang}_source"), page.get(f"{lang}_output")
            if not source:
                continue
            _inside(root, source, "source")
            _inside(output_dir, output, "output")
            if output in seen:
                raise DocumentationError(f"duplicate output path: {output}")
            seen.add(output)
            targets[source] = output
    written: list[Path] = []
    sections = sorted(data["sections"], key=lambda section: section["order"])
    for page in pages:
        for lang in ("en", "zh"):
            source, output = page.get(f"{lang}_source"), page.get(f"{lang}_output")
            if not source:
                continue
            source_file = _inside(root, source, "source")
            dest = _inside(output_dir, output, "output")
            if source_file.suffix.lower() == ".html":
                rendered = source_file.read_text(encoding="utf-8")
            else:
                text = source_file.read_text(encoding="utf-8")
                content = _parser(_rewriter(source, output, targets)).render(text)
                nav = "".join(
                    f'<span>{html.escape(section["title"])}</span>'
                    + "".join(
                        f'<a href="{html.escape(_relative(output, item[f"{lang}_output"]), quote=True)}">{html.escape(item["id"])}</a>'
                        for item in sorted(pages, key=lambda entry: entry["order"])
                        if item["section"] == section["id"]
                        and item.get(f"{lang}_source")
                    )
                    for section in sections
                )
                other = "zh" if lang == "en" else "en"
                alternate = (
                    _relative(output, page[f"{other}_output"])
                    if page.get(f"{other}_source")
                    else ""
                )
                rendered = _page(
                    page["id"],
                    content,
                    lang="zh-CN" if lang == "zh" else "en",
                    nav=nav,
                    alternate=alternate,
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(rendered, encoding="utf-8", newline="\n")
            written.append(dest)
    assets = root / "docs" / "assets"
    if assets.is_dir():
        shutil.copytree(assets, output_dir / "assets", dirs_exist_ok=True)
    return written


def is_current(manifest_path: Path, output_dir: Path) -> bool:
    """Whether every generated file matches an existing output tree byte for byte."""
    with tempfile.TemporaryDirectory(prefix="melo-docs-check-") as temporary:
        generated = Path(temporary) / "site"
        build(manifest_path, generated)
        if not output_dir.is_dir():
            return False
        expected = {
            path.relative_to(generated)
            for path in generated.rglob("*")
            if path.is_file()
        }
        actual = {
            path.relative_to(output_dir)
            for path in output_dir.rglob("*")
            if path.is_file()
        }
        return expected == actual and all(
            (output_dir / path).read_bytes() == (generated / path).read_bytes()
            for path in expected
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="preview directory outside the checkout",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare an existing output tree without writing it",
    )
    args = parser.parse_args()
    try:
        if args.check:
            if not is_current(args.manifest, args.output):
                parser.exit(1, "build_docs: generated output is stale\n")
            print("Generated output is current")
            return
        pages = build(args.manifest, args.output)
    except (DocumentationError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"build_docs: {exc}\n")
    print(f"Built {len(pages)} pages in {os.fspath(args.output)}")


if __name__ == "__main__":
    main()
