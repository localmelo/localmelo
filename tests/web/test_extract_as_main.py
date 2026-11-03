"""The page reader is main's: ``extract`` gives the same title, text, links and description
on a set of ordinary pages as it did before the note gate cleaned raw material (#442).

The expected values below were taken from main's reader. They include the cases the note
gate's strict rule would read differently (white text, a style sheet that hides a block, a
self-closed element): the reader keeps its own rules, and the gate does its own cleaning.
"""

from __future__ import annotations

import pytest

from localmelo.support.web.extract import extract

BASE = "https://example.com/"

PAGES: list[tuple[str, str, str, str, list[tuple[str, str]], str]] = [
    (
        "hero: inline dark bg on section + white h1",
        '\n<section style="background:#0b1020;padding:80px"><h1 style="color:#fff">Build faster</h1>\n<p style="color:#ffffff">Ship your app in days.</p></section><p>Body text.</p>',
        "",
        "# Build faster\n\nShip your app in days.\n\nBody text.",
        [],
        "",
    ),
    (
        "hero: dark bg from class, inline white text (site builder)",
        '\n<style>.hero{background:#0b1020}</style>\n<section class="hero"><h1 style="color:#fff">Build faster</h1>\n<p style="color:#ffffff">Ship your app in days.</p></section><p>Body text.</p>',
        "",
        "# Build faster\n\nShip your app in days.\n\nBody text.",
        [],
        "",
    ),
    (
        "Wix-like strip: bg on sibling layer, inline white span",
        '\n<section class="strip"><div class="bgLayers"><div class="colorUnderlay" style="background-color:#121212"></div></div>\n<div class="inlineContent"><p class="font_8" style="font-size:18px;line-height:1.6em"><span style="color:#FFFFFF;">We cater weddings in Lisbon.</span></p></div></section>\n<p>Contact us.</p>',
        "",
        "We cater weddings in Lisbon.\n\nContact us.",
        [],
        "",
    ),
    (
        "Gutenberg cover: overlay span sibling, inline white text",
        '\n<div class="wp-block-cover"><span aria-hidden="true" class="wp-block-cover__background has-background-dim" style="background-color:#000000"></span>\n<img class="wp-block-cover__image-background" src="/hero.jpg" alt="">\n<div class="wp-block-cover__inner-container"><p class="has-text-align-center has-text-color" style="color:#ffffff;font-size:48px">Open every day 8 to 6</p></div></div>\n<p>More below.</p>',
        "",
        "Open every day 8 to 6\n\nMore below.",
        [],
        "",
    ),
    (
        "Elementor: background overlay div sibling + white heading",
        '\n<section class="elementor-section"><div class="elementor-background-overlay" style="background-image:url(/bg.jpg)"></div>\n<div class="elementor-container"><h2 class="elementor-heading-title" style="color:#FFFFFF">Our services</h2></div></section>',
        "",
        "## Our services",
        [],
        "",
    ),
    (
        "CTA button: inline white text, bg via class",
        '\n<style>.btn{background:#2563eb;padding:8px}</style>\n<p>Ready?</p><a class="btn" href="/signup" style="color:#fff">Sign up free</a>',
        "",
        "Ready?\n\nSign up free",
        [("Sign up free", "https://example.com/signup")],
        "",
    ),
    (
        "footer: dark class bg, inline white links",
        '\n<style>.footer{background:#111}</style>\n<footer class="footer"><a href="/privacy" style="color:#ffffff">Privacy</a> <a href="/terms" style="color:#ffffff">Terms</a></footer>',
        "",
        "Privacy Terms",
        [
            ("Privacy", "https://example.com/privacy"),
            ("Terms", "https://example.com/terms"),
        ],
        "",
    ),
    (
        "email archive: bgcolor + font white",
        '\n<table><tr><td bgcolor="#1a1a1a"><font color="#ffffff">Weekly digest: 3 new posts</font></td></tr></table>',
        "",
        "Weekly digest: 3 new posts",
        [],
        "",
    ),
    (
        "bootstrap classes only (bg-dark text-white)",
        '\n<div class="bg-dark text-white"><h2>Pricing</h2><p>From 9 dollars.</p></div>',
        "",
        "## Pricing\n\nFrom 9 dollars.",
        [],
        "",
    ),
    (
        "inline-block grid: inline font-size:0 parent, class reset",
        '\n<style>.col{display:inline-block;width:50%;font-size:16px}</style>\n<div class="row" style="font-size:0"><div class="col">Left column text.</div><div class="col">Right column text.</div></div>',
        "",
        "Left column text.\nRight column text.",
        [],
        "",
    ),
    (
        "inline-block grid: inline font-size:0 parent, inline reset",
        '\n<div style="font-size:0"><div style="display:inline-block;font-size:16px">Left.</div><div style="display:inline-block;font-size:1rem">Right.</div></div>',
        "",
        "Left.\nRight.",
        [],
        "",
    ),
    (
        "site builder: inline opacity:0 (animate on scroll)",
        '\n<div data-aos="fade-up" style="opacity:0;transform:translateY(20px)"><h2>Our story</h2><p>Since 1998.</p></div>',
        "",
        "## Our story\n\nSince 1998.",
        [],
        "",
    ),
    (
        "nav menu with display:none submenu",
        '\n<nav><ul class="menu"><li><a href="/a">About</a><ul class="sub-menu" style="display:none"><li><a href="/team">Team</a></li></ul></li><li><a href="/b">Blog</a></li></ul></nav><main><p>Hello.</p></main>',
        "",
        "- About\n- Blog\n\nHello.",
        [("About", "https://example.com/a"), ("Blog", "https://example.com/b")],
        "",
    ),
    (
        "table",
        "<table><tr><th>Name</th><th>Role</th></tr><tr><td>Ana</td><td>Chef</td></tr></table>",
        "",
        "Name | Role\nAna | Chef",
        [],
        "",
    ),
    (
        "details",
        "<details><summary>Shipping</summary><p>Ships in 2 days.</p></details>",
        "",
        "Shipping\n\nShips in 2 days.",
        [],
        "",
    ),
    (
        "pre",
        "<pre>def f():\n    return 1\n</pre><p>After.</p>",
        "",
        "def f():\nreturn 1\n\nAfter.",
        [],
        "",
    ),
    (
        "Tailwind inlined: hidden md:block",
        '\n<style>.hidden{display:none}@media (min-width:768px){.md\\:block{display:block}}</style>\n<nav class="hidden md:block"><a href="/docs">Docs</a> <a href="/pricing">Pricing</a></nav><p>Welcome.</p>',
        "",
        "Docs Pricing\n\nWelcome.",
        [
            ("Docs", "https://example.com/docs"),
            ("Pricing", "https://example.com/pricing"),
        ],
        "",
    ),
    (
        "Tailwind inlined: hidden lg:flex sidebar",
        '\n<style>.hidden{display:none}@media (min-width:1024px){.lg\\:flex{display:flex}}</style>\n<aside class="hidden lg:flex"><p>Table of contents: Install, Configure, Deploy.</p></aside><p>Main text.</p>',
        "",
        "Table of contents: Install, Configure, Deploy.\n\nMain text.",
        [],
        "",
    ),
    (
        "IE conditional comment with a style",
        '\n<!--[if lt IE 9]><style>.content{display:none}</style><![endif]-->\n<div class="content"><p>Our opening hours are 9 to 5.</p></div>',
        "",
        "Our opening hours are 9 to 5.",
        [],
        "",
    ),
    (
        "noscript style hiding a JS widget",
        '\n<noscript><style>.js-only{display:none}</style></noscript>\n<div class="js-only"><p>Search our catalogue of 2,000 books.</p></div>',
        "",
        "Search our catalogue of 2,000 books.",
        [],
        "",
    ),
    (
        "FOUC guard: body visibility hidden",
        "\n<html><head><style>body{visibility:hidden}</style><script>addEventListener('load',()=>document.body.style.visibility='visible')</script></head>\n<body><h1>Acme</h1><p>We make anvils.</p></body></html>",
        "",
        "# Acme\n\nWe make anvils.",
        [],
        "",
    ),
    (
        "XHTML script self-closed, then more scripts",
        '\n<html><head><script src="/a.js"/><script>var x=1;</script></head><body><h1>Title</h1><p>Body text.</p></body></html>',
        "",
        "# Title\n\nBody text.",
        [],
        "",
    ),
    (
        "self-closed anchor",
        '<p><a name="top"/>Intro text <a href="/more">more</a>.</p>',
        "",
        "Intro text more.",
        [("more", "https://example.com/more")],
        "",
    ),
    (
        "self-closed div spacer",
        '<div class="clear"/><p>After the spacer.</p><div>Box</div>',
        "",
        "After the spacer.\n\nBox",
        [],
        "",
    ),
    (
        "style in a script string",
        '\n<script>var css = "<style>.main{display:none}</style>";</script><div class="main"><p>Article body.</p></div>',
        "",
        "Article body.",
        [],
        "",
    ),
    (
        "media=print style block",
        '<style media="print">.screen-only{display:none}</style><div class="screen-only">Visible on screen.</div>',
        "",
        "Visible on screen.",
        [],
        "",
    ),
    (
        "dark mode: prefers-color-scheme body bg, inline white",
        '\n<style>@media (prefers-color-scheme: dark){body{background:#000}}</style><p style="color:#fff">Night text.</p>',
        "",
        "Night text.",
        [],
        "",
    ),
]


@pytest.mark.parametrize(
    ("label", "html", "title", "text", "links", "description"),
    PAGES,
    ids=[p[0] for p in PAGES],
)
def test_the_page_reader_gives_the_same_as_main(
    label: str,
    html: str,
    title: str,
    text: str,
    links: list[tuple[str, str]],
    description: str,
) -> None:
    page = extract(html, BASE)
    assert page.title == title
    assert page.text == text
    assert [(link.text, link.href) for link in page.links] == links
    assert page.description == description
