"""``extract``: a page as text, without what the page hides from people.

The rule for a style that hides text (:func:`hidden_style`) is the one the note gate
reads a file with, so a page and a file are judged alike. Synthetic HTML only.
"""

from __future__ import annotations

import pytest

from localmelo.support.web.extract import extract, hidden_style, hides
from localmelo.tests.web.pages import PAGE

BASE = "https://pages.test/"


def text(html: str) -> str:
    return extract(html, BASE).text


# ── Text a style or an attribute hides ──


@pytest.mark.parametrize(
    "hiding",
    [
        'style="display:none"',
        'style="DISPLAY : NONE !important"',
        'style="visibility:hidden"',
        'style="font-size:0"',
        'style="font-size: 1px"',
        'style="FONT-SIZE:2px;margin:0"',
        'style="font-size:0.1em"',
        'style="opacity:0"',
        'style="opacity: 0.0"',
        'style="color:#fff"',
        'style="COLOR:#FFFFFF"',
        'style="color:white"',
        'style="color: rgb(255, 255, 255)"',
        'style="color:rgb(255 255 255)"',
        'style="margin:0; color:white; padding:0"',
        'style="display/**/:none"',
        'style="display&#58;none"',
        "hidden",
        'aria-hidden="true"',
    ],
)
@pytest.mark.parametrize("tag", ["span", "div", "b"])
def test_text_a_page_hides_is_not_extracted(hiding: str, tag: str) -> None:
    html = f"<p>Before.</p><{tag} {hiding}>Secret words.</{tag}><p>After.</p>"
    assert text(html) == "Before.\n\nAfter."


def test_white_text_set_with_the_font_tag_is_not_extracted() -> None:
    for color in ("white", "#FFF", "#ffffff", "rgb(255,255,255)"):
        html = f'<p>Before.</p><font color="{color}">Secret words.</font><p>After.</p>'
        assert text(html) == "Before.\n\nAfter."
    assert "Seen" in text('<font color="red">Seen</font>')
    assert "Seen" in text('<span color="white">Seen</span>')


@pytest.mark.parametrize(
    "shown",
    [
        'style="display:block"',
        'style="visibility:visible"',
        'style="font-size:12px"',
        'style="font-size:3px"',
        'style="font-size:1em"',
        'style="color:#000"',
        'style="color:#fffffe"',
        'style="background-color:#fff"',
        'style="opacity:0.5"',
        'style="opacity:1"',
        'aria-hidden="false"',
        'class="hidden"',
    ],
)
def test_text_a_page_shows_is_extracted(shown: str) -> None:
    html = f"<p>One.</p><span {shown}>Seen words.</span><p>Two.</p>"
    assert text(html) == "One.\n\nSeen words.\n\nTwo."


def test_a_hidden_element_takes_the_elements_inside_it_and_nothing_after() -> None:
    html = (
        "<p>Before.</p><div hidden>one <div>two</div> three <b>four</b></div>"
        "<p>After.</p>"
    )
    assert text(html) == "Before.\n\nAfter."


# ── A hidden element a browser closes by itself ──


def test_a_hidden_paragraph_and_a_hidden_list_item_are_skipped() -> None:
    assert text("<p hidden>Secret.</p><p>Seen.</p>") == "Seen."
    assert text('<p style="display:none">Secret.</p><p>Seen.</p>') == "Seen."
    html = '<ul><li>One.</li><li style="display:none">Secret.</li><li>Two.</li></ul>'
    assert text(html) == "- One.\n- Two."
    html = '<p>A.</p><p style="color:white">Secret.</p><p>B.</p>'
    assert text(html) == "A.\n\nB."
    html = '<table><tr><td>a<td style="font-size:0">secret<td>c</table>'
    assert text(html) == "a | c"


def test_a_hidden_element_never_closed_ends_where_a_browser_ends_it() -> None:
    # The next paragraph, list item or block, and the end of its parent.
    assert text("<p hidden>Secret.<p>One.<p>Two.") == "One.\n\nTwo."
    assert text("<ul><li hidden>Secret.<li>One.<li>Two.</ul>") == "- One.\n- Two."
    assert text("<p hidden>Secret.<div>A block.</div><p>Seen.") == "A block.\n\nSeen."
    assert text("<div><p hidden>Secret.</div><p>After.") == "After."
    assert text("<div><li hidden>Secret.<b>bold</b></div>After.") == "After."
    # ... and one that nothing ends runs to the end of the page.
    assert text("<p>Seen.</p><p hidden>Secret.") == "Seen."
    # A sub-list inside a hidden item is hidden with it; the next item is not.
    html = "<ul><li hidden>Secret.<ul><li>Inner.</li><li>Inner two.</ul><li>Seen.</ul>"
    assert text(html) == "- Seen."
    # A hidden paragraph that is closed, followed by paragraphs that are not.
    assert text("<p hidden>Secret.</p>Loose text.<p>One.<p>Two.") == (
        "Loose text.\n\nOne.\n\nTwo."
    )


def test_a_self_closing_hidden_element_is_only_itself() -> None:
    assert text("<p>One.</p><p hidden/><p>Two.</p>") == "One.\n\nTwo."
    assert text("<img hidden alt='x'><p>Seen.</p>") == "Seen."


# ── The rule itself ──


@pytest.mark.parametrize(
    ("style", "hidden"),
    [
        ("", False),
        ("display:none", True),
        ("display: block", False),
        ("visibility: hidden", True),
        ("visibility:collapse", True),
        ("font-size: 0", True),
        ("font-size: 0px", True),
        ("font-size: 2px", True),
        ("font-size: 2.1px", False),
        ("font-size: 1pt", True),
        ("font-size: 2pt", False),
        ("font-size: 0.125em", True),
        ("font-size: 0.2em", False),
        ("font-size: 10%", True),
        ("font-size: 1", False),
        ("font-size: calc(1px)", False),
        ("opacity: 0", True),
        ("opacity: 0%", True),
        ("opacity: .0", True),
        ("opacity: 0.01", False),
        ("color: white", True),
        ("color: #FFF", True),
        ("color: #ffff", True),
        ("color: #ffffffff", True),
        ("color: rgba(255,255,255,0.4)", True),
        ("color: rgb(255 255 255 / 50%)", True),
        ("color: #fffffe", False),
        ("color: rgb(255, 255, 254)", False),
        ("background-color: white", False),
        ("border-color: white", False),
        ("color: black; background: white", False),
        ("margin: 0; Color : White ;", True),
        ("display:none !IMPORTANT", True),
        ("/* display:none */ color: black", False),
        ("a;;b:;:c;display", False),
    ],
)
def test_hidden_style(style: str, hidden: bool) -> None:
    assert hidden_style(style) is hidden


def test_hides_reads_the_attributes_of_an_element() -> None:
    assert hides("div", {"hidden": None})
    assert hides("div", {"hidden": ""})
    assert hides("div", {"aria-hidden": " True "})
    assert not hides("div", {"aria-hidden": "false"})
    assert hides("div", {"style": "display:none"})
    assert hides("font", {"color": "WHITE"})
    assert not hides("font", {"color": "red"})
    assert not hides("div", {"color": "white"})
    assert not hides("div", {"class": "hidden", "title": "hidden"})
    assert not hides("div", {})


# ── A page with none of that reads as before ──

PAGE_TEXT = (
    "# Tomatoes\n\nTomatoes like sun and water. Plant them after the last frost, "
    "about sixty centimetres apart, and tie the stems to a stake as they grow.\n\n"
    "## Watering\n\n- Water in the morning.\n- Keep the leaves dry.\n\n"
    "Month | Task\nMay | Plant\n\nSee the beans page and another site. Some more "
    "text so that the main part is long enough to be kept on its own, as a reader "
    "expects."
)
ORDINARY = (
    "<ul><li>a<li>b<li>c</ul><p>one<p>two<p>three<table><tr><td>a<td>b<tr><td>c"
    "<td>d</table><dl><dt>x<dd>y<dt>z<dd>w</dl>"
    '<p style="color:#333;font-size:14px;opacity:1;display:block">styled</p>'
    '<p class="hidden" title="hidden">classed</p>'
)
ORDINARY_TEXT = (
    "- a\n- b\n- c\n\none\n\ntwo\n\nthree\n\na | b\nc | d\n\nx\ny\nz\nw\n\n"
    "styled\n\nclassed"
)


def test_the_page_the_fetch_tests_serve_extracts_as_it_always_did() -> None:
    found = extract(PAGE, BASE)
    assert found.title == "Garden notes" and found.scripts == 1
    assert found.description == "Notes about a small garden."
    assert found.text == PAGE_TEXT
    assert [(link.text, link.href) for link in found.links] == [
        ("the beans page", "https://pages.test/beans"),
        ("another site", "https://other.example/x"),
    ]


def test_a_page_of_auto_closed_elements_and_plain_styles_extracts_as_it_always_did() -> (
    None
):
    assert text(ORDINARY) == ORDINARY_TEXT


def test_markup_the_parser_cannot_finish_never_raises() -> None:
    for html in [
        "<p hidden",
        "<li hidden><li hidden><li hidden>",
        "</p></li><p hidden></ul>",
        "<p hidden><b><i></b></i></p>",
        "<td hidden></tr><tr hidden>",
        "<p hidden/>",
        '<p style="font-size:">x',
    ]:
        assert isinstance(text(html), str)
