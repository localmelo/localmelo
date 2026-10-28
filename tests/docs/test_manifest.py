"""Inventory and offline link checks for the pre-migration documentation tree."""

from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from collections import Counter
from functools import cache
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import pytest
from markdown_it import MarkdownIt

from localmelo.scripts import build_docs

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/manifest.json"
MD = MarkdownIt("commonmark", {"html": True}).enable("table")
# A specific broken link may be recorded here as (source, literal URL, reason).
# Entries must disappear when the underlying issue is fixed.
LINK_DEBT: dict[tuple[str, str], str] = {}
# Headings removed upstream may already have inbound links; keep their stubs.
RETIRED_FRAGMENTS = {
    "docs/scheduler.md": {"tasks-workers-and-the-scheduler-store-s3a1"},
}


class HTMLInventory(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(attributes["id"] or "")
        if tag == "a" and attributes.get("name"):
            self.ids.append(attributes["name"] or "")
        for key in ("href", "src"):
            if attributes.get(key):
                self.links.append(attributes[key] or "")

    handle_startendtag = handle_starttag


def slug(text: str) -> str:
    """GitHub-style heading ID, including Unicode and repeated hyphens."""
    text = text.lower().strip()
    return "".join(
        "-" if ch.isspace() else ch
        for ch in text
        if ch.isspace() or ch in "-_" or unicodedata.category(ch)[0] not in "PCS"
    )


def inline_text(children: list) -> str:
    return "".join(
        (
            token.content
            if token.type in {"text", "code_inline"}
            else (
                inline_text(token.children or [])
                if token.type == "image"
                else " " if token.type in {"softbreak", "hardbreak"} else ""
            )
        )
        for token in children
    )


def inventory(text: str, suffix: str) -> tuple[list[str], list[str]]:
    html = HTMLInventory()
    if suffix in {".html", ".htm"}:
        html.feed(text)
        return html.ids, html.links
    tokens = MD.parse(text)
    headings: list[str] = []
    counts: Counter[str] = Counter()
    used: set[str] = set()
    links: list[str] = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            base = slug(inline_text(tokens[i + 1].children or []))
            count = counts[base]
            candidate = f"{base}-{count}" if count else base
            while candidate in used:
                count += 1
                candidate = f"{base}-{count}"
            counts[base] = count + 1
            used.add(candidate)
            headings.append(candidate)
        if token.type in {"html_block", "html_inline"}:
            html.feed(token.content)
        for child in token.children or []:
            if child.type in {"link_open", "image"}:
                target = child.attrGet("href" if child.type == "link_open" else "src")
                if isinstance(target, str) and target:
                    links.append(target)
            elif child.type == "html_inline":
                html.feed(child.content)
    return headings + html.ids, links + html.links


STYLE = {"docs/STYLE.md", "docs/zh-CN/STYLE.md"}


@cache
def tracked_docs() -> set[str]:
    output = subprocess.check_output(["git", "ls-files", "docs"], cwd=ROOT, text=True)
    # The style pair is new in this worktree and may not be staged yet.
    return {
        p for p in output.splitlines() if p.endswith((".md", ".html", ".txt"))
    } | STYLE


def safe_path(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        bool(path)
        and not p.is_absolute()
        and ".." not in p.parts
        and str(p) == path
        and "\\" not in path
    )


def check_fragment_mapping(
    alias: str, mapping: str | dict[str, str], anchors: list[str], output: str
) -> None:
    """Check default same-ID routing or require explicit split destinations."""
    prefix = "zh-CN/" if alias.startswith("docs/zh-CN/") else "en/"
    if isinstance(mapping, str):
        target = output if mapping == "same" else mapping
        assert (
            safe_path(target) and target.startswith(prefix) and target.endswith(".html")
        ), alias
        return
    assert isinstance(mapping, dict), alias
    required = set(anchors) | RETIRED_FRAGMENTS.get(alias, set())
    missing = sorted(required - set(mapping))
    assert (
        not missing
    ), f"{alias}: unmapped anchors {missing}; assign their destinations in docs/manifest.json"
    extra = set(mapping) - set(anchors) - RETIRED_FRAGMENTS.get(alias, set())
    assert not extra, f"{alias}: unexpected historical anchors {sorted(extra)}"
    for fragment, destination in mapping.items():
        target, sep, anchor = destination.partition("#")
        assert (
            sep
            and fragment
            and anchor
            and (anchor == fragment or fragment in RETIRED_FRAGMENTS.get(alias, set()))
        ), (alias, fragment, destination)
        assert (
            safe_path(target) and target.startswith(prefix) and target.endswith(".html")
        ), alias


def test_fragment_rules_accept_unsplit_new_anchors() -> None:
    check_fragment_mapping(
        "docs/en/updates.html", "same", ["old", "u-worker-history"], "en/updates.html"
    )
    check_fragment_mapping(
        "docs/en/quickstart.html",
        "same",
        ["json-output", "new"],
        "en/quickstart.html",
    )


def test_fragment_rules_reject_unsafe_or_cross_language_targets() -> None:
    for target in ("../escape.html", "zh-CN/updates.html", "en/updates.html#fragment"):
        with pytest.raises(AssertionError):
            check_fragment_mapping(
                "docs/en/updates.html", target, ["anchor"], "en/updates.html"
            )


def test_fragment_rules_require_a_destination_for_new_split_anchors() -> None:
    with pytest.raises(
        AssertionError, match=r"docs/cli\.md.*worker-histories.*manifest"
    ):
        check_fragment_mapping(
            "docs/cli.md",
            {"old": "en/reference/cli.html#old"},
            ["old", "worker-histories"],
            "en/reference/cli.html",
        )


def test_historical_fragments_survive_page_moves() -> None:
    baseline = json.loads((ROOT / "tests/docs/legacy_fragments.json").read_text())
    pages = json.loads(MANIFEST.read_text())["pages"]
    for source, mapping in baseline.items():
        page = next(page for page in pages if source in page["aliases"])
        path = ROOT / source
        anchors, _ = inventory(path.read_text(), path.suffix)
        if page["en_source"] == source:
            assert set(mapping) - RETIRED_FRAGMENTS.get(source, set()) <= set(
                anchors
            ), source
            continue
        assert (
            "<script" not in path.read_text().lower()
        ), f"{source}: stub needs JavaScript"
        assert set(mapping) <= set(
            anchors
        ), f"{source}: missing historical stub anchors"
        tokens = MD.parse(path.read_text())
        heading_links = {}
        heading_ids = iter(anchors)
        for index, token in enumerate(tokens):
            if token.type != "heading_open":
                continue
            anchor = next(heading_ids)
            children = tokens[index + 1].children or []
            links = [
                child.attrGet("href") for child in children if child.type == "link_open"
            ]
            if links:
                heading_links[anchor] = links[0]
        for anchor, destination in mapping.items():
            href = heading_links.get(anchor)
            assert isinstance(href, str), f"{source}#{anchor}: no visible link"
            parsed = urlsplit(href)
            target, _, target_anchor = destination.partition("#")
            assert (path.parent / unquote(parsed.path)).resolve() == (
                ROOT / target
            ).resolve(), (source, anchor, href)
            assert unquote(parsed.fragment) == target_anchor, (source, anchor, href)
            target_path = ROOT / target
            assert target_path.is_file(), f"{source}#{anchor}: missing {target}"
            target_ids, _ = inventory(target_path.read_text(), target_path.suffix)
            assert (
                target_anchor in target_ids
            ), f"{source}#{anchor}: missing {destination}"


def test_historical_website_paths_and_ids_remain() -> None:
    baseline = json.loads((ROOT / "tests/docs/legacy_site_ids.json").read_text())
    for source, expected in baseline.items():
        path = ROOT / source
        assert path.is_file(), f"missing historical website path {source}"
        actual, _ = inventory(path.read_text(), ".html")
        missing = Counter(expected) - Counter(actual)
        assert not missing, f"{source}: missing historical IDs {dict(missing)}"


def test_entry_point_document_links() -> None:
    prefixes = (
        ("https://localmelo.github.io/localmelo/", ROOT / "docs"),
        ("https://github.com/localmelo/localmelo/blob/main/", ROOT),
    )
    names = [
        "README.md",
        "README.zh-CN.md",
        "docs/README.md",
        "docs/llms.txt",
        "docs/for-agents.md",
    ]
    names.extend(
        path.relative_to(ROOT).as_posix()
        for language in ("en", "zh-CN")
        for path in sorted((ROOT / "docs" / language).glob("*.html"))
    )
    for name in names:
        source = ROOT / name
        _, links = inventory(source.read_text(), source.suffix)
        for link in links:
            base = source.parent
            value = link
            for prefix, root in prefixes:
                if link.startswith(prefix):
                    base, value = root, link.removeprefix(prefix)
                    break
            else:
                if urlsplit(link).scheme or link.startswith("//"):
                    continue
            parsed = urlsplit(value)
            target = (base / unquote(parsed.path)).resolve() if parsed.path else source
            if not target.is_relative_to(ROOT / "docs") or target.suffix not in {
                ".md",
                ".txt",
            }:
                continue
            assert target.is_file(), f"{name}: missing {link}"
            if parsed.fragment:
                anchors, _ = inventory(target.read_text(), target.suffix)
                assert (
                    unquote(parsed.fragment) in anchors
                ), f"{name}: missing anchor {link}"


def test_manifest_schema_and_inventory() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert set(manifest) == {"version", "sections", "pages"}
    assert manifest["version"] == 1
    sections = manifest["sections"]
    assert all(set(s) == {"id", "title", "order"} for s in sections)
    section_ids = [s["id"] for s in sections]
    assert len(section_ids) == len(set(section_ids))
    assert len({s["order"] for s in sections}) == len(sections)
    assert all(isinstance(s["order"], int) and s["title"] for s in sections)
    ids, outputs, aliases, orders = [], [], [], []
    for page in manifest["pages"]:
        assert set(page) == {
            "id",
            "section",
            "order",
            "en_source",
            "zh_source",
            "en_output",
            "zh_output",
            "aliases",
            "fragments",
        }
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", page["id"])
        assert page["section"] in section_ids
        assert isinstance(page["order"], int)
        ids.append(page["id"])
        orders.append((page["section"], page["order"]))
        assert page["en_source"] or page["zh_source"]
        for lang in ("en", "zh"):
            source, output = page[f"{lang}_source"], page[f"{lang}_output"]
            if source is not None:
                assert safe_path(source) and source in tracked_docs()
                assert source in page["aliases"]
                assert output is not None
            if output is not None:
                assert (
                    safe_path(output)
                    and output.startswith("en/" if lang == "en" else "zh-CN/")
                    and output.endswith(".html")
                )
                outputs.append(output)
        assert isinstance(page["aliases"], list)
        assert set(page["fragments"]) == set(page["aliases"])
        for alias in page["aliases"]:
            assert safe_path(alias) and alias in tracked_docs()
            aliases.append(alias)
            expected, _ = inventory((ROOT / alias).read_text(), Path(alias).suffix)
            lang = "zh" if alias.startswith("docs/zh-CN/") else "en"
            check_fragment_mapping(
                alias, page["fragments"][alias], expected, page[f"{lang}_output"]
            )
    assert len(ids) == len(set(ids))
    assert len(orders) == len(set(orders))
    assert len(outputs) == len(set(outputs))
    assert len(aliases) == len(set(aliases))
    assert set(aliases) == tracked_docs()


def test_split_destinations_and_existing_language_pairs() -> None:
    pages = {p["id"]: p for p in json.loads(MANIFEST.read_text())["pages"]}
    assert (
        pages["cli"]["fragments"]["docs/cli.md"]["the-interactive-ui"]
        == "en/guides/terminal.html#the-interactive-ui"
    )
    assert (
        pages["cli"]["fragments"]["docs/cli.md"]["options"]
        == "en/reference/cli.html#options"
    )
    assert (
        pages["quickstart"]["fragments"]["docs/quickstart.md"][
            "serve-a-model-with-omlx"
        ]
        == "en/guides/model-servers.html#serve-a-model-with-omlx"
    )
    for name in (
        "index",
        "quickstart",
        "architecture",
        "memory-design",
        "updates",
        "developers",
    ):
        page = pages[f"site-{name}"]
        assert page["en_source"] == f"docs/en/{name}.html"
        assert page["zh_source"] == f"docs/zh-CN/{name}.html"
        assert set(page["fragments"]) == {page["en_source"], page["zh_source"]}
    style = pages["style"]
    assert (style["en_source"], style["zh_source"]) == (
        "docs/STYLE.md",
        "docs/zh-CN/STYLE.md",
    )
    quickstart = pages["site-quickstart"]
    for lang, source in (
        ("en", "docs/en/quickstart.html"),
        ("zh", "docs/zh-CN/quickstart.html"),
    ):
        # Keep the real HTML and every ID at its existing URL until both
        # language destinations and per-fragment mappings are implemented.
        assert quickstart["fragments"][source] == "same"
        assert quickstart[f"{lang}_output"] == source.removeprefix("docs/")


def test_manifest_fragment_destinations_exist() -> None:
    pages = json.loads(MANIFEST.read_text())["pages"]
    outputs = {
        page[f"{lang}_output"]: page[f"{lang}_source"]
        for page in pages
        for lang in ("en", "zh")
        if page[f"{lang}_output"] and page[f"{lang}_source"]
    }

    @cache
    def output_ids(path: Path) -> set[str]:
        ids, _ = inventory(path.read_text(), path.suffix)
        return set(ids)

    for page in pages:
        for alias, rule in page["fragments"].items():
            if isinstance(rule, str):
                lang = "zh" if alias.startswith("docs/zh-CN/") else "en"
                target = page[f"{lang}_output"] if rule == "same" else rule
                anchors, _ = inventory((ROOT / alias).read_text(), Path(alias).suffix)
                destinations = {anchor: f"{target}#{anchor}" for anchor in anchors}
            else:
                destinations = rule
            for old_anchor, destination in destinations.items():
                output, _, anchor = destination.partition("#")
                assert (
                    output in outputs
                ), f"{alias}#{old_anchor}: missing output {output}"
                source = ROOT / outputs[output]
                assert (
                    source.is_file()
                ), f"{alias}#{old_anchor}: missing source {source}"
                assert anchor in output_ids(
                    source
                ), f"{alias}#{old_anchor}: missing destination {destination}"


def test_inventory_slug_and_code_fences() -> None:
    anchors, links = inventory(
        '# Hello, 世界!\n\n## Same `code`\n\n## Same `code`\n\n```md\n# Not a heading\n```\n\n<a id="explicit"></a>\n[ok](./x.md#same-code)\n',
        ".md",
    )
    assert anchors == ["hello-世界", "same-code", "same-code-1", "explicit"]
    assert links == ["./x.md#same-code"]


def test_local_links_and_anchors() -> None:
    broken = {}
    for source in sorted(tracked_docs()):
        source_path = ROOT / source
        _, links = inventory(source_path.read_text(), source_path.suffix)
        for link in links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or link.startswith("//"):
                continue
            path = unquote(parsed.path)
            # Root-relative links denote a deployed web root, not a repository path.
            if path.startswith("/"):
                target = ROOT / "docs" / path.lstrip("/")
            else:
                target = source_path.parent / path if path else source_path
            target = target.resolve()
            if not target.is_relative_to(ROOT):
                broken[(source, link)] = "outside repository"
                continue
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                broken[(source, link)] = "missing file"
            elif parsed.fragment and target.suffix in {".md", ".html", ".htm", ".txt"}:
                anchors, _ = inventory(target.read_text(), target.suffix)
                if unquote(parsed.fragment) not in anchors:
                    broken[(source, link)] = "missing anchor"
    assert broken == LINK_DEBT, "Link debt changed:\n" + "\n".join(
        f"{k!r}: {v!r}," for k, v in sorted(broken.items())
    )


@pytest.mark.parametrize(
    "source",
    [
        "docs/start/quickstart.md",
        "docs/guides/configure.md",
        "docs/guides/model-servers.md",
        "docs/guides/embedding.md",
        "docs/guides/api-client.md",
        "docs/guides/python.md",
        "docs/guides/tools-mcp.md",
        "docs/guides/memory.md",
        "docs/guides/offline-retrieval.md",
        "docs/guides/terminal.md",
        "docs/guides/sessions.md",
        "docs/guides/gateway.md",
        "docs/guides/web-console.md",
        "docs/concepts/how-it-works.md",
        "docs/concepts/memory.md",
        "docs/concepts/procedures.md",
        "docs/concepts/retrieval.md",
    ],
)
def test_chinese_twin_keeps_anchors_and_examples(source: str) -> None:
    pages = json.loads(MANIFEST.read_text())["pages"]
    page = next(p for p in pages if p["en_source"] == source)
    expected = source.replace("docs/", "docs/zh-CN/", 1)
    assert page["zh_source"] == expected
    en = (ROOT / source).read_text()
    zh = (ROOT / expected).read_text()
    en_ids, _ = inventory(en, ".md")
    zh_ids, _ = inventory(zh, ".md")
    assert set(en_ids) <= set(zh_ids)
    assert len(zh_ids) == len(set(zh_ids)), expected
    rendered_ids, _ = inventory(build_docs.render_markdown(zh), ".html")
    assert set(en_ids) <= set(rendered_ids), expected
    assert len(rendered_ids) == len(set(rendered_ids)), expected

    def fences(text: str) -> list[tuple[str, str]]:
        return [(t.info, t.content) for t in MD.parse(text) if t.type == "fence"]

    assert fences(en) == fences(zh), expected

    def inline_code(text: str) -> Counter[str]:
        return Counter(
            " ".join(child.content.split())
            for token in MD.parse(text)
            for child in (token.children or [])
            if child.type == "code_inline"
        )

    assert inline_code(en) == inline_code(zh), expected
