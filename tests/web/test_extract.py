"""``extract``: a page as text, without what the page hides from people.

The page reader drops only what no person sees: an element that is not drawn at all,
and a piece of text whose size is tiny or whose colour is white with no background
behind it, following the inline styles down the page. ``opacity: 0`` hides nothing
there (a script fades such text in). The strict rule for a raw file
(:func:`hidden_style`) is the one the note gate uses. Synthetic HTML only.
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
        'style="opacity:0"',
        'style="opacity: 0.0"',
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


# ── Text that is white, tiny or faded is dropped only when no person sees it ──


def shown(html: str) -> str:
    return text(html)


@pytest.mark.parametrize(
    "dark",
    [
        '<div style="background:#111;color:#fff">Seen words.</div>',
        '<div style="background-color: rgb(10, 10, 10)"><p style="color:white">Seen words.</p></div>',
        '<body style="background:#000"><h1 style="color:#fff">Seen words.</h1></body>',
        '<section style="background: url(hero.jpg) center"><p style="color:#fff">Seen words.</p></section>',
        '<div style="background-image:linear-gradient(#000,#333)"><b style="color:#FFF">Seen words.</b></div>',
        '<div style="background:#111"><div><div><span style="color:#fff">Seen words.</span></div></div></div>',
        '<table bgcolor="#000"><tr><td><font color="white">Seen words.</font></td></tr></table>',
        '<div style="background:#111"><font color=white>Seen words.</font></div>',
        '<div style="color:#fff;background:#111 !important">Seen words.</div>',
    ],
)
def test_white_text_on_a_dark_background_is_kept(dark: str) -> None:
    assert "Seen words." in text(f"<p>Before.</p>{dark}<p>After.</p>")


@pytest.mark.parametrize(
    "bare",
    [
        '<div style="color:#fff">Secret words.</div>',
        '<div style="background:#fff;color:#fff">Secret words.</div>',
        '<div style="background-color:white;color:white">Secret words.</div>',
        '<div style="background:transparent;color:#fff">Secret words.</div>',
        '<div style="background:none;color:#fff">Secret words.</div>',
        '<div style="color:#fff"><p>Secret words.</p></div>',
        "<font color=white>Secret words.</font>",
        '<div style="background:#111"><div style="background:#fff"><p style="color:#fff">Secret words.</p></div></div>',
        '<div style="background:#111"></div><p style="color:#fff">Secret words.</p>',
    ],
)
def test_white_text_with_no_background_or_a_white_one_is_dropped(bare: str) -> None:
    found = text(f"<p>Before.</p>{bare}<p>After.</p>")
    assert "Secret" not in found
    assert "Before." in found and "After." in found


def test_an_opacity_zero_section_is_kept_for_a_script_fades_it_in() -> None:
    html = (
        '<section style="opacity:0;transform:translateY(24px)">'
        "<h2>Seen heading</h2><p>Seen words.</p><ul><li>Seen item</li></ul></section>"
        '<div style="opacity: 0 !important">Seen too.</div>'
    )
    found = text(html)
    for words in ("## Seen heading", "Seen words.", "- Seen item", "Seen too."):
        assert words in found


def test_a_font_size_zero_parent_keeps_the_children_that_set_a_size() -> None:
    html = (
        '<div style="font-size:0">'
        '<span style="font-size:16px">One</span>'
        '<span style="font-size:1rem">Two</span>'
        '<span style="font: 14px Arial">Three</span>'
        '<span style="font-size:1em">Four</span>'
        "<span>Five</span>"
        "</div>"
    )
    found = text(html)
    assert "One" in found and "Two" in found and "Three" in found
    assert "Four" not in found and "Five" not in found  # 1em of 0 is still 0


def test_a_font_size_zero_span_with_no_reset_is_dropped() -> None:
    for html in (
        '<p>Seen.</p><span style="font-size:0">Secret.</span><p>Also seen.</p>',
        '<p>Seen.</p><div style="font-size:0"><p>Secret.</p><b>Secret too.</b></div><p>Also seen.</p>',
        '<p>Seen.</p><p style="font: 0/0 serif">Secret.</p><p>Also seen.</p>',
    ):
        assert text(html) == "Seen.\n\nAlso seen."


def test_the_nearest_size_and_colour_win() -> None:
    html = (
        '<div style="color:#fff"><span style="color:#000">Seen</span> Secret</div>'
        '<p>Mid. <span style="color:#fff">Secret</span> After.</p>'
        '<div style="font-size:1px"><span style="font-size:inherit">Secret</span>'
        '<span style="font-size:12px">Seen too</span></div>'
    )
    found = text(html)
    assert "Seen" in found and "Seen too" in found
    assert "Secret" not in found
    assert "Mid." in found and "After." in found


def test_text_after_a_white_element_is_shown_again() -> None:
    assert text('<p>Mid <span style="color:#fff">Secret</span> after.</p>') == (
        "Mid after."
    )
    # A paragraph or item never closed does not carry its colour to the next one.
    assert text('<p style="color:white">Secret.<p>Seen.') == "Seen."
    assert text('<ul><li style="color:white">Secret.<li>Seen.</ul>') == "- Seen."


def test_what_is_dropped_leaves_no_marker_and_no_link_behind() -> None:
    assert text('<ul><li style="color:#fff">Secret</li><li>Seen</li></ul>') == "- Seen"
    assert text('<h2 style="color:#fff">Secret</h2><p>Seen</p>') == "Seen"
    assert text('<table><tr><td style="color:#fff">x<td>a<td>b</table>') == "a | b"
    found = extract(
        '<a href="/evil" style="color:#fff">evil</a> <a href="/ok">ok</a>', BASE
    )
    assert [link.href for link in found.links] == ["https://pages.test/ok"]


def test_an_element_that_is_not_drawn_takes_what_it_holds_whatever_the_children_say() -> (
    None
):
    html = (
        '<div style="visibility:hidden"><span style="visibility:visible">a</span></div>'
        '<div style="display:none"><div style="background:#111"><p style="color:#fff">'
        "b</p></div></div>"
        '<div hidden><p style="font-size:16px">c</p></div>'
        '<div aria-hidden="true"><b style="color:#000">d</b></div>'
        "<p>Seen.</p>"
    )
    assert text(html) == "Seen."


def test_stray_end_tags_and_unclosed_elements_are_read_in_linear_time() -> None:
    page = "<div style='color:#fff'>" * 30000 + "</span>" * 30000 + "<p>x</p>"
    assert isinstance(text(page), str)
    page = "<div>" * 30000 + "</div></b>" * 30000
    assert text(page) == ""


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
        ("font: 0/0 serif", True),
        ("font: bold 1px a", True),
        ("font: 12px/1.5 Arial", False),
        ("font: bold 700 serif", False),
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
