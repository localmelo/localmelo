"""``extract``: a page as text, without what the page hides from people.

The page reader drops only what no person sees: an element that is not drawn at all
(by its own style or a simple rule of a style block), and a piece of text whose size
is tiny, whose colour is clear, or whose colour is white with no background behind
it, following the inline styles down the page. ``opacity: 0`` hides nothing there (a
script fades such text in). The strict rule for a raw file (:func:`hidden_style`) is
the one the note gate uses. Synthetic HTML only.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from localmelo.support.web.extract import (
    extract,
    hidden_style,
    hides,
    sheet_of,
    style_blocks,
)
from localmelo.tests.linear import assert_linear
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
        # Zero in any unit, with a sign or an exponent, is nothing.
        ("font-size: 0vw", True),
        ("font-size: 0vmin", True),
        ("font-size: +0px", True),
        ("font-size: -0", True),
        ("font-size: 0e3px", True),
        ("font-size: 1e-9px", True),
        ("font-size: 1e3px", False),
        ("font-size: -1px", False),
        ("font: 0vw serif", True),
        ("font-size: calc(0px)", False),
        ("font-size: 1vw", False),
        # A clear colour draws nothing; so does an opacity of 0 in any spelling.
        ("color: transparent", True),
        ("color: rgba(0, 0, 0, 0)", True),
        ("color: rgba(0,0,0,0.0)", True),
        ("color: rgb(0 0 0 / 0%)", True),
        ("color: hsla(120, 50%, 50%, 0)", True),
        ("color: #fff0", True),
        ("color: #00000000", True),
        ("color: rgba(0,0,0,0.5)", False),
        ("color: rgb(0,0,0)", False),
        ("color: #0000", True),
        ("color: #000f", False),
        ("opacity: 0e5", True),
        ("opacity: 00", True),
        ("opacity: +0", True),
        ("opacity: 1e-9999", True),
        ("opacity: -1", True),
        ("opacity: 0.5", False),
        ("opacity: 1", False),
        # More ways to say white.
        ("color: rgb(100%, 100%, 100%)", True),
        ("color: rgb(100% 100% 100%)", True),
        ("color: hsl(0, 0%, 100%)", True),
        ("color: hsl(120deg 40% 100%)", True),
        ("color: hsla(0, 0%, 100%, 0.8)", True),
        ("color: hsl(0, 0%, 99%)", False),
        ("color: rgb(100%, 100%, 99%)", False),
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


# ── A slash on an element that is not void opens it ──


@pytest.mark.parametrize(
    "opening",
    [
        "<div hidden/>",
        "<div hidden />",
        '<span style="display:none"/>',
        '<div aria-hidden="true"/>',
        '<section style="visibility:hidden"/>',
        "<div hidden x=a/>",
    ],
)
def test_a_self_closing_slash_on_a_non_void_element_opens_it(opening: str) -> None:
    tag = opening[1:].split()[0].split("/")[0].rstrip(">")
    html = f"<p>Before.</p>{opening}You live in Paris.</{tag}><p>After.</p>"
    assert text(html) == "Before.\n\nAfter."
    # The text after a visible one is not swallowed, as before.
    assert text(f"<{tag}/>Seen.</{tag}>") == "Seen."


def test_a_self_closing_slash_on_a_hidden_paragraph_hides_to_its_end() -> None:
    assert text("<p>One.</p><p hidden/>Secret.</p><p>Two.</p>") == "One.\n\nTwo."
    assert text("<p>One.</p><li hidden/>Secret.<li>Two.") == "One.\n\n- Two."


def test_a_slash_still_closes_a_void_element_and_the_foreign_ones() -> None:
    assert text("<img hidden alt='x'/><p>Seen.</p>") == "Seen."
    assert text("<p>One.<br/>Two.</p><hr/><p>Three.</p>") == "One.\nTwo.\n\nThree."
    assert text("<svg/><p>Seen.</p>") == "Seen."
    assert text("<math/><p>Seen.</p>") == "Seen."
    assert text("<svg><title>Gone.</title></svg><p>Seen.</p>") == "Seen."


def test_a_slash_in_an_unquoted_value_is_no_self_closing() -> None:
    assert text("<p>One.</p><div hidden title=a/>Gone.</div><p>Two.</p>") == (
        "One.\n\nTwo."
    )


# ── A style block can hide an element by its class, id or tag ──


@pytest.mark.parametrize(
    ("css", "tag"),
    [
        (".x{display:none}", '<p class="x">'),
        (".x { display : none !important }", '<p class="y x z">'),
        ("#x{display:none}", '<p id="x">'),
        ("p.keep,p{display:none}", "<p>"),
        ("h1, .x ,#z{visibility:hidden}", '<p class="x">'),
        (".x{visibility:collapse}", '<p class="x">'),
        (".x{margin:0;display:none;color:red}", '<p class="x">'),
        ("/* c */ .x{display:none}", '<p class="x">'),
        ("@import url(a.css); .x{display:none}", '<p class="x">'),
        ("@font-face{font-family:a} .x{display:none}", '<p class="x">'),
    ],
)
def test_a_simple_rule_that_does_not_draw_an_element_hides_it(
    css: str, tag: str
) -> None:
    close = "</p>"
    html = f"<style>{css}</style><div>Before.</div>{tag}You live in Paris.{close}"
    found = text(html)
    assert "Paris" not in found and "Before." in found
    # A style block later in the page applies to what came before it, too.
    later = f"<div>Before.</div>{tag}You live in Paris.{close}<style>{css}</style>"
    assert "Paris" not in text(later)
    assert "Paris" not in text(html.replace("<style>", "<STYLE type='text/css'>"))


@pytest.mark.parametrize(
    "css",
    [
        ".a .b{display:none}",
        ".x:hover{display:none}",
        "p.x{display:none}",
        "div > .x{display:none}",
        ".x[title]{display:none}",
        "*{display:none}",
        "a:not(.x){display:none}",
        ".x{color:#fff}",
        ".x{font-size:0}",
        ".x{opacity:0}",
        ".x{display:block}",
        ".y{display:none}",
        "#x{display:none}",
        "@media (max-width:600px){.x{display:none}}",
        "@media print{.x{display:none}}",
        "@supports (display:grid){.x{display:none}}",
        "@keyframes k{from{display:none}}",
        ".x{.y{display:none}}",
        ".x{display:none",
    ],
)
def test_a_rule_that_is_not_simple_or_not_hiding_changes_nothing(css: str) -> None:
    html = f'<style>{css}</style><p class="x">You live in Paris.</p>'
    assert text(html) == "You live in Paris."


def test_a_style_block_for_another_medium_is_not_read() -> None:
    html = '<style media="print">.x{display:none}</style><p class="x">Seen.</p>'
    assert text(html) == "Seen."
    html = '<style media="screen">.x{display:none}</style><p class="x">Gone.</p>'
    assert text(html) == ""


def test_an_inline_display_overrides_a_rule_of_a_style_block() -> None:
    html = (
        "<style>.x{display:none}</style>"
        '<p class="x" style="display:block">Seen.</p><p class="x">Gone.</p>'
    )
    assert text(html) == "Seen."


def test_an_element_with_a_class_that_gives_it_a_display_is_not_taken_as_hidden() -> (
    None
):
    html = (
        "<style>.d-none{display:none!important}"
        "@media (min-width:768px){.d-md-block{display:block!important}}"
        ".d-lg-block{display:block}</style>"
        '<p class="d-none d-md-block">Seen on a wide screen.</p>'
        '<p class="d-none">Gone.</p>'
    )
    assert text(html) == "Seen on a wide screen."


def test_style_blocks_are_found_without_a_parser() -> None:
    html = (
        "<STYLE>.a{x:y}</STYLE>text<style\nmedia=print>.p{}</style>"
        "<style/>.s{}</style><style>.never{"
    )
    assert style_blocks(html) == [".a{x:y}", ".s{}", ".never{"]
    assert style_blocks(html, screen_only=False) == [
        ".a{x:y}",
        ".p{}",
        ".s{}",
        ".never{",
    ]
    assert style_blocks("<styled>x</styled> <p>") == []


def test_a_sheet_of_the_strict_reading_takes_every_flagged_rule() -> None:
    css = ".w{color:#fff} #t{color:transparent} @media print{.m{display:none}} q{opacity:0}"
    strict = sheet_of([css], strict=True)
    assert strict.classes == {"w", "m"} and strict.ids == {"t"} and strict.tags == {"q"}
    lenient = sheet_of([css], strict=False)
    assert not (lenient.classes or lenient.ids or lenient.tags)
    assert strict.hides("p", {"class": "a w"})
    assert strict.hides("q", {})
    assert not strict.hides("p", {"class": "a"})


# ── A colour that is clear, and a size that is nothing ──


@pytest.mark.parametrize(
    "colour",
    [
        "transparent",
        "rgba(0,0,0,0)",
        "rgba(255, 255, 255, 0)",
        "rgb(0 0 0 / 0%)",
        "hsla(0, 0%, 0%, 0)",
        "#0000",
        "#ffffff00",
    ],
)
def test_text_with_a_clear_colour_is_not_extracted(colour: str) -> None:
    html = f'<p>Before.</p><span style="color:{colour}">Gone.</span><p>After.</p>'
    assert text(html) == "Before.\n\nAfter."
    on_dark = f'<div style="background:#111"><b style="color:{colour}">Gone.</b></div>'
    assert text(on_dark) == ""
    font = f'<p>Before.</p><font color="{colour}">Gone.</font><p>After.</p>'
    assert text(font) == "Before.\n\nAfter."


def test_text_with_a_clear_colour_over_a_clipped_background_is_the_background() -> None:
    html = (
        '<h1 style="color:transparent;background:linear-gradient(red,blue);'
        '-webkit-background-clip:text">Seen.</h1>'
    )
    assert text(html) == "# Seen."


@pytest.mark.parametrize(
    "background",
    [
        "rgba(0,0,0,0)",
        "rgba(0, 0, 0, 0)",
        "#fff0",
        "#ffffff00",
        "transparent",
        "hsla(0, 0%, 0%, 0)",
        "rgb(0 0 0 / 0)",
        "none",
    ],
)
def test_a_clear_background_is_no_background(background: str) -> None:
    html = f'<div style="color:#fff;background:{background}">Gone.</div>'
    assert text(html) == ""
    html = f'<div style="background-color:{background};color:#fff">Gone.</div>'
    assert text(html) == ""
    dark = f'<div style="background:#111"><p style="background:{background};color:#fff">Seen.</p></div>'
    assert text(dark) == "Seen."


@pytest.mark.parametrize(
    "white",
    [
        "hsl(0, 0%, 100%)",
        "hsl(240 100% 100%)",
        "hsla(0, 0%, 100%, 1)",
        "rgb(100%, 100%, 100%)",
        "rgb(100% 100% 100%)",
    ],
)
def test_more_spellings_of_white_are_white(white: str) -> None:
    assert text(f'<p style="color:{white}">Gone.</p><p>Seen.</p>') == "Seen."
    on_dark = f'<div style="background:#000"><p style="color:{white}">Seen.</p></div>'
    assert text(on_dark) == "Seen."
    over_white = f'<p style="color:{white};background:{white}">Gone.</p>'
    assert text(over_white) == ""


@pytest.mark.parametrize(
    "size", ["0vw", "0vmin", "+0px", "-0", "0e3px", "1e-9px", "0.0001em", "0%"]
)
def test_a_size_that_is_nothing_hides_the_text(size: str) -> None:
    html = f'<p>Before.</p><span style="font-size:{size}">Gone.</span><p>After.</p>'
    assert text(html) == "Before.\n\nAfter."
    child = (
        f'<div style="font-size:{size}"><span style="font-size:16px">Seen.</span></div>'
    )
    assert text(child) == "Seen."


@pytest.mark.parametrize(
    "size", ["calc(0px)", "1vw", "-1px", "1e3px", "larger", "x-small"]
)
def test_a_size_it_cannot_read_is_a_visible_one(size: str) -> None:
    assert text(f'<span style="font-size:{size}">Seen.</span>') == "Seen."


def test_a_background_image_or_gradient_still_counts_as_a_dark_background() -> None:
    for background in (
        "url(hero.jpg) center",
        "linear-gradient(#fff, #fff)",
        "image-set(url(a.png) 1x)",
    ):
        html = f'<section style="background:{background}"><h1 style="color:#fff">Seen.</h1></section>'
        assert text(html) == "# Seen."


# ── Time: every parse is linear ──


def _page(shape: str) -> Callable[[int], str]:
    def make(size: int) -> str:
        unit = {
            "digits": '<p style="font-size:' + "1" * size + '!">t</p>',
            "spaces": '<p style="color: a' + " " * size + 'b">t</p>',
            "important": '<p style="a:b' + " " * size + '!x">t</p>',
            "bangs": '<p style="a:b' + "!" * size + '">t</p>',
            "zeros": '<p style="font-size:' + "0" * size + 'x">t</p>',
            "rgb": '<p style="color:rgb(255' + " " * size + '">t</p>',
            "hsl": '<p style="color:hsl(0' + ", 0" * (size // 3) + '">t</p>',
            "eq": "<p a" + "=b" * (size // 2) + ">t</p>",
            "long_name": "<p " + "a" * size + ">t</p>",
        }
        if shape in unit:
            return unit[shape]
        repeat = {
            "auto": ("<p hidden>" + "<b>" * (size // 6) + "<i></i>" * (size // 14)),
            "stray": "<div hidden>" + "<i>" * (size // 6) + "</b>" * (size // 8),
            "slash": "<div hidden/>x" * (size // 14),
            "slash_p": "<p hidden/>x" * (size // 12),
            "styles": "<style>.a{display:none}</style>x" * (size // 33),
            "unclosed_styles": "<style>x" * (size // 8),
            "braces": "<style>" + "{" * size + "</style>",
            "closing": "<style>" + "}" * size + "</style>",
            "rules": "<style>" + ".a{x:y}" * (size // 7) + "</style>",
            "selectors": "<style>" + "a," * (size // 2) + "{display:none}</style>",
            "at_rules": "<style>" + "@media x{.a{b:c}}" * (size // 17) + "</style>",
            "white": '<div style="color:#fff">' * (size // 24),
            "tags": "<a <a <a " * (size // 9),
            "quote": '<a x="' + "b <c " * (size // 5),
        }
        return repeat[shape]

    return make


@pytest.mark.parametrize(
    "shape",
    [
        "digits",
        "spaces",
        "important",
        "bangs",
        "zeros",
        "rgb",
        "hsl",
        "eq",
        "long_name",
        "auto",
        "stray",
        "slash",
        "slash_p",
        "styles",
        "unclosed_styles",
        "braces",
        "closing",
        "rules",
        "selectors",
        "at_rules",
        "white",
        "tags",
        "quote",
    ],
)
def test_extract_takes_time_linear_in_the_page(shape: str) -> None:
    assert_linear(_page(shape), lambda html: extract(html, BASE))


@pytest.mark.parametrize("shape", ["digits", "spaces", "important", "zeros", "rgb"])
def test_a_style_is_read_in_time_linear_in_its_length(shape: str) -> None:
    def make(size: int) -> str:
        return _page(shape)(size).split('"')[1]

    assert_linear(make, hidden_style)
