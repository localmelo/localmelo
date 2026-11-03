"""The result-page readers, on saved pages: no network."""

from __future__ import annotations

from localmelo.support.web import search_pages as pages
from localmelo.tests.web import search_fixtures as fx


def test_bing_rss_gives_titles_addresses_and_snippets_in_order() -> None:
    hits = pages.parse_bing_rss(fx.BING_RSS)
    assert [h.url for h in hits] == [
        "https://www.example.com/garden/tomato-blight",
        "https://extension.example.org/blight",
        "https://shop.example.com/sprays?id=7&ref=a",
    ]
    assert hits[0].title == "Tomato blight: what it is & how to stop it"
    assert hits[0].snippet.startswith("Blight is a fungus.")
    # The feed's own title and image links are not results.
    assert all("bing.com" not in h.url for h in hits)


def test_bing_rss_leaves_out_items_without_a_title_or_a_web_address() -> None:
    titles = [h.title for h in pages.parse_bing_rss(fx.BING_RSS)]
    assert "Not a web address" not in titles and "" not in titles


def test_bing_rss_refuses_a_feed_that_declares_entities() -> None:
    assert pages.parse_bing_rss(fx.BING_RSS_ENTITY) == []


def test_bing_rss_reads_nothing_from_what_is_not_a_feed() -> None:
    assert pages.parse_bing_rss(fx.BING_RSS_EMPTY) == []
    assert pages.parse_bing_rss("<html><body>Please verify</body></html>") == []
    assert pages.parse_bing_rss("") == []
    assert pages.parse_bing_rss("\x00\x01 not xml") == []


def test_baidu_gives_the_cards_own_address_not_the_redirect() -> None:
    hits = pages.parse_baidu(fx.BAIDU)
    assert [h.url for h in hits] == [
        "https://www.example.com/zh/tomato",
        "https://garden.example.org/zh/blight",
    ]
    assert hits[0].title == "番茄 晚疫病 怎么防治"
    assert "晚疫病在阴雨天" in hits[0].snippet
    # The title is not repeated in the snippet and scripts are not read.
    assert "怎么防治" not in hits[0].snippet and "never read" not in hits[0].snippet


def test_baidu_leaves_out_engine_cards_repeats_and_cards_without_a_web_address() -> (
    None
):
    titles = [h.title for h in pages.parse_baidu(fx.BAIDU)]
    assert titles == ["番茄 晚疫病 怎么防治", "番茄病害图谱"]


def test_baidu_reads_nothing_from_a_check_page() -> None:
    assert pages.parse_baidu("<html><body>百度安全验证 请完成验证</body></html>") == []
    assert pages.parse_baidu("") == []


def test_links_and_text_make_hits_with_the_text_between_titles_as_snippet() -> None:
    hits = pages.read_links(fx.BROWSER_TEXT, fx.BROWSER_LINKS, fx.BROWSER_OWN)
    assert [h.url for h in hits] == [
        "https://www.example.com/garden/tomato-blight",
        "https://extension.example.org/blight",
        "https://shop.example.com/sprays?id=7",
        "http://127.0.0.1:8080/admin",  # the searcher drops it, see test_search
    ]
    assert hits[0].title == "Tomato blight: what it is and how to stop it"
    assert "Water the soil" in hits[0].snippet
    assert "Late blight" not in hits[0].snippet
    assert "cool, wet weather" in hits[1].snippet


def test_links_skip_the_engines_own_pages_short_labels_and_address_labels() -> None:
    hits = pages.read_links(fx.BROWSER_TEXT, fx.BROWSER_LINKS, fx.BROWSER_OWN)
    titles = [h.title for h in hits]
    assert "Images" not in titles
    assert not any("›" in t for t in titles)
    assert not any("privacy" in h.url for h in hits)
    assert len({h.url for h in hits}) == len(hits)


def test_links_without_the_title_in_the_text_still_give_a_hit() -> None:
    hits = pages.read_links(
        "nothing here",
        [("A long enough title here", "https://www.example.com/x")],
        frozenset(),
    )
    assert hits == [
        pages.Hit("A long enough title here", "https://www.example.com/x", "")
    ]


def test_a_title_the_page_wraps_over_lines_still_finds_its_snippet() -> None:
    hits = pages.read_links(
        "A long enough title\nthat wraps over two lines\nand the snippet after it",
        [
            (
                "A long enough title that wraps over two lines",
                "https://www.example.com/x",
            )
        ],
        frozenset(),
    )
    assert [h.snippet for h in hits] == ["and the snippet after it"]


def test_a_huge_or_deep_page_is_read_within_limits() -> None:
    deep = "<div>" * 5000 + "text" + "</div>" * 5000
    wide = "<p>x</p>" * (pages.MAX_NODES + 10)
    assert pages.parse_baidu(deep) == []
    assert pages.parse_baidu(wide) == []
    many = "".join(
        f'<div class="result c-container" mu="https://e{i}.example.com/">'
        f"<h3>Title {i}</h3><p>Body {i}</p></div>"
        for i in range(pages.MAX_HITS + 20)
    )
    assert len(pages.parse_baidu(many)) == pages.MAX_HITS
