"""Measure the keyless ways a public web search can be read.

Run by hand, never by the tests (it reaches real search engines)::

    uv run python scripts/search_probe.py --out DIR --http
    LOCALMELO_PROBE_BROWSER=1 uv run python scripts/search_probe.py --out DIR --browser

``--http`` asks each engine's result page with ``web_fetch``'s fetcher (a plain
GET, no cookies, no login). ``--browser`` loads the same pages in a Safari
automation window, which needs Safari's remote automation switched on and opens
a window, so it also needs ``LOCALMELO_PROBE_BROWSER=1``.

It is polite on purpose: at most two requests per query per way (the second only
after a failed first), at least ``GAP_S`` seconds between two requests to one
engine, no login, and a way that is blocked on its first few queries is not asked
again. Each request is one line in ``DIR/requests.jsonl``; ``DIR/summary.md`` is
the table: how many queries answered with five or more complete results, how many
were blocked, the median and 90th-percentile time, and the parse rate (of the
pages that were not blocks, the share the reader found results in).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

from localmelo.support.web import search as searching
from localmelo.support.web import search_pages as pages
from localmelo.support.web.fetch import FetchError, WebFetcher
from localmelo.support.web.guard import AddressError
from localmelo.support.web.tool import readable

GAP_S = 2.2
# A way blocked on this many queries in a row is not asked again.
GIVE_UP = 4
# Results with all three parts needed to count as an answer.
NEEDED = 5

QUERIES: list[tuple[str, str, str]] = [
    # (kind, language, words)
    ("product", "en", "iPhone 17 Pro battery life review"),
    ("product", "en", "Nintendo Switch 2 price"),
    ("company", "en", "Toyota Camry hybrid mpg"),
    ("person", "en", "Satya Nadella"),
    ("news", "en", "federal reserve interest rate decision november"),
    ("news", "en", "Tesla Cybertruck recall"),
    ("error", "en", "ModuleNotFoundError: No module named 'numpy' conda"),
    ("error", "en", "error: externally-managed-environment pip install macOS"),
    ("date", "en", "what day is it in Los Angeles"),
    ("howto", "en", "how to reset SMC on a MacBook"),
    ("tech", "en", "python asyncio TaskGroup exception handling"),
    ("weather", "en", "weather in Pittsburgh tomorrow"),
    ("product", "zh", "iPhone 17 Pro 电池续航 评测"),
    ("company", "zh", "比亚迪 2026 销量"),
    ("person", "zh", "雷军 小米 发布会"),
    ("news", "zh", "央行 降息 最新消息"),
    ("news", "zh", "今天 国际新闻 头条"),
    ("error", "zh", "报错 ModuleNotFoundError: No module named 'torch'"),
    ("date", "zh", "洛杉矶现在几点"),
    ("slang", "zh", "绝绝子 是什么意思"),
    ("slang", "zh", "city不city 什么梗"),
    ("howto", "zh", "MacBook 怎么重置 SMC"),
    ("tech", "zh", "梯度下降 原理 通俗解释"),
    ("weather", "zh", "上海 明天 天气"),
]

# More signs of a page that asks for a check, beyond the ones melo itself looks for.
BLOCK_MARKERS = (
    *searching.BLOCK_MARKERS,
    "anomaly",
    "anubis",
    "unusual activity",
    "to continue, please",
    "cookie consent",
    "before you continue",
    "consent.google",
    "please enable javascript",
)
relevant = searching.relevant


def looks_blocked(text: str) -> str:
    """The block marker *text* holds, or an empty string."""
    low = text[:200_000].lower()
    return next((m for m in BLOCK_MARKERS if m in low), "")


# ── readers for pages that are measured but not shipped ──


def _bing_address(href: str) -> str:
    """The address behind a Bing result link (``…/ck/a?…&u=a1<base64>``)."""
    import base64

    parts = urlsplit(href)
    if (parts.hostname or "").lower().endswith("bing.com"):
        body = parse_qs(parts.query).get("u", [""])[0]
        if not body.startswith("a1"):
            return ""
        try:
            raw = base64.urlsafe_b64decode(body[2:] + "=" * (-len(body[2:]) % 4))
            href = raw.decode("utf-8")
        except ValueError:
            return ""
    return href if href.startswith(("http://", "https://")) else ""


def parse_bing(body: str) -> list[pages.Hit]:
    """Bing's result page: ``li.b_algo`` with a title link and a caption."""
    hits: list[pages.Hit] = []
    for node in pages.tree(body).walk():
        if node.tag != "li" or "b_algo" not in node.classes():
            continue
        head = node.find(lambda n: n.tag == "h2")
        link = head.find(lambda n: n.tag == "a" and "href" in n.attrs) if head else None
        if link is None:
            continue
        para = node.find(lambda n: n.tag == "p")
        url = _bing_address(link.attrs["href"])
        if link.text() and url:
            hits.append(pages.Hit(link.text(), url, para.text() if para else ""))
    return hits


def parse_duck(body: str) -> list[pages.Hit]:
    """DuckDuckGo's page without scripts: ``a.result__a`` with ``.result__snippet``."""
    hits: list[pages.Hit] = []
    for node in pages.tree(body).walk():
        if node.tag != "div" or "result" not in node.classes():
            continue
        link = node.find(lambda n: n.tag == "a" and "result__a" in n.classes())
        if link is None:
            continue
        href = link.attrs.get("href", "")
        href = "https:" + href if href.startswith("//") else href
        if "duckduckgo.com/l/" in href:
            href = unquote(parse_qs(urlsplit(href).query).get("uddg", [""])[0])
        snippet = node.find(lambda n: "result__snippet" in n.classes())
        if link.text() and href.startswith(("http://", "https://")):
            hits.append(pages.Hit(link.text(), href, snippet.text() if snippet else ""))
    return hits


@dataclass(frozen=True)
class Way:
    key: str
    engine: str  # ways of one engine are asked one at a time, GAP_S apart
    url: str  # with {q}
    read: Callable[[str], list[pages.Hit]]


# The domains of each engine, whose links are not results.
OWN: dict[str, tuple[str, ...]] = {
    "bing": ("bing.com", "microsoft.com", "msn.com"),
    "duck": ("duckduckgo.com",),
    "google": ("google.com", "gstatic.com", "googleusercontent.com", "youtube.com"),
    "brave": ("brave.com",),
    "startpage": ("startpage.com",),
    "baidu": ("baidu.com",),
    "mojeek": ("mojeek.com",),
    "yahoo": ("yahoo.com", "bing.com"),
}


def _unread(_: str) -> list[pages.Hit]:
    return []


# Every way that ships, then pages that are only measured (with no reader, they
# can only be told apart as blocked or unread).
WAYS: dict[str, Way] = {
    **{
        key: Way(key, way.engine.lower(), way.url, way.read)
        for key, way in searching.WAYS.items()
    },
    **{
        way.key: way
        for way in (
            Way("bing", "bing", "https://www.bing.com/search?q={q}", parse_bing),
            Way(
                "duck-html",
                "duck",
                "https://html.duckduckgo.com/html/?q={q}",
                parse_duck,
            ),
            Way(
                "duck-lite", "duck", "https://lite.duckduckgo.com/lite/?q={q}", _unread
            ),
            Way("brave", "brave", "https://search.brave.com/search?q={q}", _unread),
            Way("mojeek", "mojeek", "https://www.mojeek.com/search?q={q}", _unread),
            Way("google", "google", "https://www.google.com/search?q={q}", _unread),
            Way(
                "startpage",
                "startpage",
                "https://www.startpage.com/do/search?q={q}",
                _unread,
            ),
            Way("yahoo", "yahoo", "https://search.yahoo.com/search?p={q}", _unread),
        )
    },
}


def complete(hits: list[pages.Hit]) -> int:
    """How many of *hits* have a title, an address and a snippet."""
    return sum(1 for h in hits if h.title and h.url and h.snippet)


def classify(
    status: int, hits: list[pages.Hit], text: str, error: str, words: str = ""
) -> str:
    """``ok`` (five complete results on topic), ``offtopic``, ``partial``, ``blocked``,
    ``unread`` or ``error``."""
    if error:
        return "error"
    if status in (202, 401, 403, 407, 429, 451, 503) or status >= 500:
        return "blocked"
    if complete(hits) >= NEEDED:
        return "ok" if relevant(words, hits) else "offtopic"
    if hits:
        return "partial"
    return "blocked" if looks_blocked(text) else "unread"


async def http_request(
    fetcher: WebFetcher, way: Way, words: str
) -> tuple[int, str, str, float, str]:
    """One GET: status, text, final address, seconds and an error sentence."""
    started = time.monotonic()
    try:
        page = await fetcher.fetch(way.url.format(q=quote(words)), accept=readable)
    except (AddressError, FetchError) as exc:
        return 0, "", "", time.monotonic() - started, str(exc)[:160]
    text = page.body.decode(page.charset or "utf-8", "replace")
    return page.status, text, page.url, time.monotonic() - started, ""


async def run_engine(
    ways: list[Way], out: Path, raw: Path | None, log: list[dict[str, Any]]
) -> None:
    """Every way of one engine, one request at a time and GAP_S apart."""
    fetcher = WebFetcher()
    last = 0.0
    for way in ways:
        misses = 0
        saved = False
        for kind, lang, words in QUERIES:
            if misses >= GIVE_UP:
                log.append(
                    {"way": way.key, "words": words, "lang": lang, "kind": kind,
                     "result": "skipped", "attempt": 0}
                )  # fmt: skip
                continue
            for attempt in (1, 2):
                wait = GAP_S - (time.monotonic() - last)
                if wait > 0:
                    await asyncio.sleep(wait)
                status, text, final, seconds, error = await http_request(
                    fetcher, way, words
                )
                last = time.monotonic()
                hits = way.read(text) if text and status == 200 else []
                result = classify(status, hits, text, error, words)
                log.append(
                    {"way": way.key, "words": words, "lang": lang, "kind": kind,
                     "attempt": attempt, "status": status, "seconds": round(seconds, 2),
                     "hits": len(hits), "complete": complete(hits), "result": result,
                     "final": final[:100], "error": error,
                     "marker": looks_blocked(text) if result == "blocked" else ""}
                )  # fmt: skip
                if (
                    raw is not None
                    and result in ("ok", "offtopic", "partial")
                    and not saved
                ):
                    (raw / f"{way.key}.html").write_text(text, encoding="utf-8")
                    saved = True
                if result in ("ok", "offtopic", "partial", "unread"):
                    misses = 0
                    break
                misses += 1
                if result == "blocked":
                    break  # a block is not asked again for this query
        print(f"  {way.key}: done", flush=True)


async def run_http(
    keys: list[str], out: Path, raw: Path | None
) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    engines: dict[str, list[Way]] = {}
    for key in keys:
        engines.setdefault(WAYS[key].engine, []).append(WAYS[key])
    await asyncio.gather(*(run_engine(w, out, raw, log) for w in engines.values()))
    return log


# ── the browser ──


async def run_browser(
    keys: list[str], out: Path, raw: Path | None
) -> list[dict[str, Any]]:
    from localmelo.support.browser.safari import SafariAutomationDriver

    log: list[dict[str, Any]] = []
    driver = SafariAutomationDriver()
    await driver.start()
    tab: str | None = None
    last = 0.0
    try:
        for key in keys:
            way = WAYS[key]
            misses = 0
            saved = False
            for kind, lang, words in QUERIES:
                label = f"browser:{key}"
                if misses >= GIVE_UP:
                    log.append(
                        {"way": label, "words": words, "lang": lang, "kind": kind,
                         "result": "skipped", "attempt": 0}
                    )  # fmt: skip
                    continue
                wait = GAP_S - (time.monotonic() - last)
                if wait > 0:
                    await asyncio.sleep(wait)
                started = time.monotonic()
                error = ""
                text = title = html = ""
                links = 0
                generic: list[pages.Hit] = []
                try:
                    url = way.url.format(q=quote(words))
                    if tab is None:
                        tab = (await driver.open(url, new_tab=True)).id
                    else:
                        await driver.open(url, new_tab=False, tab=tab)
                    raw_page = await driver.page(tab)
                    text, title = raw_page.text, raw_page.title
                    pairs = [
                        (e.label, e.href) for e in raw_page.elements if e.kind == "link"
                    ]
                    links = len(pairs)
                    own = frozenset(OWN.get(way.engine, ()))
                    generic = pages.read_links(raw_page.text, pairs, own)
                    shown = await driver._eval("document.documentElement.outerHTML")
                    html = shown if isinstance(shown, str) else ""
                except Exception as exc:  # noqa: BLE001 - one failed load is one row
                    error = f"{type(exc).__name__}: {exc}"[:160]
                seconds = time.monotonic() - started
                last = time.monotonic()
                # The reader that ships reads the page as text and links; the engine's
                # own reader on the markup is counted for comparison.
                specific = len(way.read(html)) if html else 0
                hits = generic
                result = classify(200, hits, f"{title}\n{text}", error, words)
                log.append(
                    {"way": label, "words": words, "lang": lang, "kind": kind,
                     "attempt": 1, "status": 200 if not error else 0,
                     "seconds": round(seconds, 2), "hits": len(hits), "specific": specific,
                     "generic": len(generic), "complete": complete(hits),
                     "result": result, "links": links,
                     "html_chars": len(html), "title": title[:60], "error": error,
                     "marker": looks_blocked(f"{title}\n{text}") if result == "blocked" else ""}
                )  # fmt: skip
                if raw is not None and html and not saved:
                    # The first page of each way, whatever it was: to read what it holds.
                    (raw / f"browser-{key}.html").write_text(html, encoding="utf-8")
                    saved = True
                misses = (
                    0
                    if result in ("ok", "offtopic", "partial", "unread")
                    else misses + 1
                )
            print(f"  browser:{key}: done", flush=True)
    finally:
        if tab is not None:
            with contextlib.suppress(
                Exception
            ):  # leaving the window is not worth a trace
                await driver.close_tab(tab)
        await driver.close()
    return log


# ── the table ──


def percentile(values: list[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(share * (len(ordered) - 1)))]


def summary(log: list[dict[str, Any]]) -> str:
    rows = [
        "| way | asked | ok (5+ complete, on topic) | off topic | partial | blocked | "
        "unread | error | skipped | median s | p90 s | parse rate | ok en | ok zh |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for way in dict.fromkeys(row["way"] for row in log):
        mine = [r for r in log if r["way"] == way]
        # The last attempt of each query is the query's result.
        final: dict[str, dict[str, Any]] = {}
        for row in mine:
            final[row["words"]] = row
        counts = {
            name: sum(1 for r in final.values() if r["result"] == name)
            for name in (
                "ok",
                "offtopic",
                "partial",
                "blocked",
                "unread",
                "error",
                "skipped",
            )
        }
        asked = len(final) - counts["skipped"]
        times = [r["seconds"] for r in mine if r.get("seconds")]
        answered = (
            counts["ok"] + counts["offtopic"] + counts["partial"] + counts["unread"]
        )
        parsed = counts["ok"] + counts["offtopic"] + counts["partial"]
        parse = parsed / answered if answered else 0.0
        by_lang = {
            lang: sum(
                1 for r in final.values() if r["result"] == "ok" and r["lang"] == lang
            )
            for lang in ("en", "zh")
        }
        rows.append(
            f"| {way} | {asked} | {counts['ok']} | {counts['offtopic']} | {counts['partial']} | "
            f"{counts['blocked']} | {counts['unread']} | {counts['error']} | "
            f"{counts['skipped']} | {statistics.median(times) if times else 0:.2f} | "
            f"{percentile(times, 0.9):.2f} | {parse:.0%} | {by_lang['en']}/12 | "
            f"{by_lang['zh']}/12 |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", required=True, type=Path, help="folder for the results"
    )
    parser.add_argument(
        "--http", action="store_true", help="measure the plain GET ways"
    )
    parser.add_argument(
        "--browser", action="store_true", help="measure the browser ways"
    )
    parser.add_argument(
        "--ways", default="", help="comma list of ways (default: all for the mode)"
    )
    parser.add_argument("--save-raw", action="store_true", help="keep one page per way")
    parser.add_argument("--limit", type=int, default=0, help="only the first N queries")
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="only every Nth query (a sample of both languages)",
    )
    args = parser.parse_args()
    if args.every > 1:
        QUERIES[:] = QUERIES[:: args.every]
    if args.limit:
        del QUERIES[args.limit :]
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw" if args.save_raw else None
    if raw is not None:
        raw.mkdir(exist_ok=True)
    keys = [k for k in args.ways.split(",") if k] or list(WAYS)
    unknown = [k for k in keys if k not in WAYS]
    if unknown:
        raise SystemExit(
            f"unknown way(s): {', '.join(unknown)}; known: {', '.join(WAYS)}"
        )
    log: list[dict[str, Any]] = []
    if args.http:
        print("plain GET ways", flush=True)
        log += asyncio.run(run_http(keys, out, raw))
    if args.browser:
        if os.environ.get("LOCALMELO_PROBE_BROWSER") != "1":
            raise SystemExit(
                "--browser opens a Safari window: set LOCALMELO_PROBE_BROWSER=1"
            )
        print("browser ways", flush=True)
        log += asyncio.run(run_browser(keys, out, raw))
    name = "requests" + ("-browser" if args.browser and not args.http else "")
    with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
        for row in log:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    table = summary(log)
    (out / f"summary{name[8:]}.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
