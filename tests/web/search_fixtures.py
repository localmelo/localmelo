"""Saved result pages for the web_search tests: the shapes measured from the engines, with made-up results.

Nothing here is fetched; every address is on ``example.com`` or ``example.org``.
"""

from __future__ import annotations

# Bing's RSS answer (``search?format=rss``): one item without a title, one with a
# link that is not an address, one with markup and entities in its description.
BING_RSS = """<?xml version="1.0" encoding="utf-8" ?><rss version="2.0"><channel>
<title>Bing: garden tomato blight</title>
<link>http://www.bing.com:80/search?q=garden+tomato+blight</link>
<description>Search results</description>
<image><url>http://www.bing.com:80/s/a/rsslogo.gif</url><title>garden tomato blight</title>
<link>http://www.bing.com:80/search?q=garden+tomato+blight</link></image>
<item><title>Tomato blight: what it is &amp; how to stop it</title>
<link>https://www.example.com/garden/tomato-blight</link>
<description>Blight is a fungus. Water the soil, not the leaves, and remove sick plants.</description>
<pubDate>Sun, 01 Nov 2026 20:53:00 GMT</pubDate></item>
<item><title>Late blight in tomato and potato</title>
<link>https://extension.example.org/blight</link>
<description>Late blight spreads in cool, wet weather. Tomato plants show dark patches.</description></item>
<item><title></title><link>https://www.example.com/untitled</link><description>No title.</description></item>
<item><title>Not a web address</title><link>javascript:alert(1)</link><description>Skipped.</description></item>
<item><title>Tomato blight sprays compared</title>
<link>https://shop.example.com/sprays?id=7&amp;ref=a</link>
<description>&lt;b&gt;Sprays&lt;/b&gt; for tomato blight, ranked.</description></item>
</channel></rss>"""

# A feed that declares an entity: refused whole, as an entity bomb would be.
BING_RSS_ENTITY = """<?xml version="1.0"?><!DOCTYPE rss [<!ENTITY a "aaaa">]>
<rss><channel><item><title>&a;</title><link>https://www.example.com/a</link></item></channel></rss>"""

BING_RSS_EMPTY = (
    '<?xml version="1.0" encoding="utf-8" ?><rss version="2.0"><channel>'
    "<title>Bing: nothing</title></channel></rss>"
)

# Baidu's page: result cards are div.c-container whose ``mu`` is the real address.
BAIDU = """<html><body><div id="content_left">
<div class="result c-container new-pmd" id="1" mu="https://www.example.com/zh/tomato">
  <h3 class="c-title"><a href="http://www.baidu.com/link?url=abc">番茄 <em>晚疫病</em> 怎么防治</a></h3>
  <div class="c-abstract">番茄晚疫病在阴雨天容易发生，要及时清理病叶并保持通风。</div>
  <script>var secret = "never read";</script>
</div>
<div class="result-op c-container new-pmd" id="2" mu="http://nourl.ubs.baidu.com/61344">
  <h3>An answer card from the engine itself</h3><p>Not a page.</p>
</div>
<div class="result c-container" id="3" mu="https://www.example.com/zh/tomato">
  <h3>Same address again</h3><p>Repeated.</p>
</div>
<div class="result c-container" id="4">
  <h3>No address on this card</h3><p>Left out.</p>
</div>
<div class="result c-container" id="5" mu="https://garden.example.org/zh/blight">
  <h3>番茄病害图谱</h3><p>常见番茄病害：晚疫病、早疫病、叶霉病。</p>
</div>
<div class="result c-container" id="6" mu="ftp://files.example.com/blight">
  <h3>Not a web address</h3><p>Left out.</p>
</div>
</div></body></html>"""

# What a browser window reads from a result page: the text, and the links in order.
BROWSER_TEXT = (
    "garden tomato blight All Images News "
    "Tomato blight: what it is and how to stop it www.example.com › garden › tomato-blight "
    "Blight is a fungus. Water the soil, not the leaves, and remove sick plants. "
    "Late blight in tomato and potato extension.example.org "
    "Late blight spreads in cool, wet weather on tomato plants. "
    "Tomato blight sprays compared shop.example.com "
    "Sprays for tomato blight, ranked by price. "
    "Settings Privacy Feedback"
)
BROWSER_LINKS = [
    ("Images", "https://duckduckgo.com/?q=x&iax=images"),
    (
        "Tomato blight: what it is and how to stop it",
        "https://www.example.com/garden/tomato-blight",
    ),
    (
        "www.example.com › garden › tomato-blight",
        "https://www.example.com/garden/tomato-blight",
    ),
    ("Late blight in tomato and potato", "https://extension.example.org/blight"),
    ("extension.example.org", "https://extension.example.org/blight"),
    ("Tomato blight sprays compared", "https://shop.example.com/sprays?id=7"),
    ("Tomato blight (again, same address)", "https://shop.example.com/sprays?id=7"),
    ("Privacy policy of this search engine", "https://duckduckgo.com/privacy"),
    ("A link to this computer that must go", "http://127.0.0.1:8080/admin"),
]
BROWSER_OWN = frozenset({"duckduckgo.com"})

# What a block page reads as.
CHALLENGE_TEXT = (
    "Unfortunately, bots use DuckDuckGo too. Please complete the following challenge."
)
