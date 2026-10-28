"""The docs builder is deterministic, navigable and does not trust Markdown HTML."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localmelo.scripts import build_docs


def test_markdown_supports_headings_tables_fences_and_duplicate_ids() -> None:
    html = build_docs.render_markdown(
        "# Start here\n\n## Repeat\n\n## Repeat\n\n"
        "| A | B |\n|---|---|\n| x | y |\n\n```python\nprint(1)\n```\n"
    )
    assert '<h1 id="start-here">' in html
    assert '<h2 id="repeat">' in html
    assert '<h2 id="repeat-1">' in html
    assert "<table>" in html and "<td>x</td>" in html
    assert '<code class="language-python">print(1)' in html
    collisions = build_docs.render_markdown("# X\n\n# X-1\n\n# X\n")
    assert '<h1 id="x-2">' in collisions


def test_markdown_never_executes_raw_html_or_dangerous_links() -> None:
    html = build_docs.render_markdown(
        "<script>alert(1)</script>\n\n<img src=x onerror=alert(1)>\n\n"
        "[bad](javascript:alert(1)) [encoded](jav&#x61;script:alert(1)) "
        "[safe](https://example.org/) [local](../other.md#part)\n"
    )
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html
    assert 'href="javascript:' not in html and 'href="jav' not in html
    assert 'href="https://example.org/"' in html
    assert 'href="../other.md#part"' in html


def test_standalone_explicit_anchors_are_safe() -> None:
    html = build_docs.render_markdown(
        '<a id="before-you-begin"></a>\n\n## 开始之前\n\n'
        '<a id="bad" onclick="alert(1)"></a>\n\n'
        'Text <a id="inline"></a> text\n\n'
        '```html\n<a id="code"></a>\n```\n\n'
        "<script>alert(1)</script>\n"
    )
    assert '<a id="before-you-begin"></a>' in html
    assert '<h2 id="开始之前">' in html
    assert '<a id="bad" onclick=' not in html
    assert '<a id="inline"></a>' not in html
    assert '<a id="code"></a>' not in html
    assert "<script>" not in html


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "jav%61script:alert(1)",
        "data:text/html,hello",
        "file:///etc/passwd",
        "//example.org/path",
        "%2f%2fexample.org/path",
        "java\nscript:alert(1)",
        "https://[invalid",
        r"https:\\example.org",
    ],
)
def test_unsafe_urls_are_rejected(url: str) -> None:
    assert not build_docs._safe_url(url)


def manifest(root: Path) -> Path:
    docs = root / "docs"
    docs.mkdir()
    (docs / "start.md").write_text("# Start\n\n[Next](next.md#details)\n")
    (docs / "next.md").write_text("# Next\n\n## Details\n\nHello.\n")
    path = docs / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "sections": [{"id": "start", "title": "Start", "order": 1}],
                "pages": [
                    {
                        "id": "start",
                        "section": "start",
                        "order": 1,
                        "en_source": "docs/start.md",
                        "zh_source": None,
                        "en_output": "en/start/start.html",
                        "zh_output": "zh-CN/start/start.html",
                        "aliases": ["docs/start.md"],
                        "fragments": {},
                    },
                    {
                        "id": "next",
                        "section": "start",
                        "order": 2,
                        "en_source": "docs/next.md",
                        "zh_source": None,
                        "en_output": "en/start/next.html",
                        "zh_output": "zh-CN/start/next.html",
                        "aliases": ["docs/next.md"],
                        "fragments": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_is_reproducible_and_rewrites_relative_links(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = manifest(root)
    first, second = tmp_path / "one", tmp_path / "two"
    build_docs.build(path, first)
    build_docs.build(path, second)
    files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert files == sorted(
        p.relative_to(second) for p in second.rglob("*") if p.is_file()
    )
    assert files == [Path("en/start/next.html"), Path("en/start/start.html")]
    for name in files:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    html = (first / "en/start/start.html").read_text()
    assert 'href="next.html#details"' in html
    assert '<html lang="en">' in html
    assert 'aria-label="Documentation"' in html
    assert not (first / "zh-CN/start/start.html").exists()


def test_real_manifest_builds_identically_twice(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    first, second = tmp_path / "first", tmp_path / "second"
    manifest_path = root / "docs/manifest.json"
    pages = build_docs.build(manifest_path, first)
    expected = sum(
        bool(page.get(f"{language}_source"))
        for page in json.loads(manifest_path.read_text())["pages"]
        for language in ("en", "zh")
    )
    assert len(pages) == expected
    build_docs.build(manifest_path, second)
    files = {p.relative_to(first) for p in first.rglob("*") if p.is_file()}
    assert files == {p.relative_to(second) for p in second.rglob("*") if p.is_file()}
    assert all((first / p).read_bytes() == (second / p).read_bytes() for p in files)
    assert build_docs.is_current(manifest_path, first)
    assert (first / "en/project/style.html").is_file()
    assert (first / "zh-CN/project/style.html").is_file()


def test_stale_committed_output_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = manifest(root)
    output = tmp_path / "committed"
    build_docs.build(path, output)
    assert build_docs.is_current(path, output)
    extra = output / "en/start/obsolete.html"
    extra.write_text("obsolete")
    assert not build_docs.is_current(path, output)
    extra.unlink()
    (output / "en/start/start.html").write_text("stale")
    assert not build_docs.is_current(path, output)
    (output / "en/start/start.html").unlink()
    assert not build_docs.is_current(path, output)


def test_builder_does_not_overwrite_a_nonempty_directory(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = manifest(root)
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        build_docs.build(path, output)
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize(
    "bad", ["../escape.html", "/absolute.html", "en/../../bad.html"]
)
def test_build_rejects_unsafe_output_paths(tmp_path: Path, bad: str) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = manifest(root)
    data = json.loads(path.read_text())
    data["pages"][0]["en_output"] = bad
    path.write_text(json.dumps(data))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="output"):
        build_docs.build(path, out)
    assert not (tmp_path / "escape.html").exists()
    assert not (tmp_path / "bad.html").exists()
