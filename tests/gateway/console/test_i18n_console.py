"""The console's language end to end: the page's first guess, the setting, and the server's own words.

``GET /`` writes its guess on ``<html>`` from ``[console] language``, the page's cookie and
``Accept-Language``; the page (``i18n.mjs``) then decides with the browser's or the Mac's
languages and keeps its choice in the ``localmelo_language`` cookie, which every ``/api`` call
carries. A client that sends neither the cookie nor ``X-LocalMelo-Language`` (the terminal, a
script) reads English, whatever the setting says.
"""

from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from localmelo.support.config import ConsoleConfig
from localmelo.support.gateway import webapp
from localmelo.support.gateway.console import Console
from localmelo.support.gateway.console.activity import _row, result_line
from localmelo.support.gateway.console.cards import card_json
from localmelo.support.gateway.console.routes_memory import _from_words
from localmelo.support.gateway.i18n import using
from localmelo.support.sessions.store import StoredEvent

from .conftest import PORT


def html_tag(client: TestClient, **headers: str) -> str:
    page = client.get("/", headers=headers).text
    found = re.search(r"<html[^>]*>", page)
    assert found is not None
    return found.group(0)


def with_page(app: FastAPI, console: Console) -> TestClient:
    webapp.mount(app, console.cfg)
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}")


def set_language(console: Console, language: str) -> None:
    """What saving Settings does: the console holds a new config."""
    console.cfg = replace(console.cfg, console=ConsoleConfig(language=language))


def test_the_page_guesses_from_the_browser_while_the_setting_is_auto(
    app: FastAPI, console: Console
) -> None:
    client = with_page(app, console)
    assert html_tag(client, **{"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}) == (
        '<html lang="zh-Hans" data-language="auto" data-i18n-pending>'
    )
    assert html_tag(client, **{"Accept-Language": "en-US,en;q=0.9"}) == (
        '<html lang="en" data-language="auto">'
    )
    assert html_tag(client, **{"Accept-Language": "zh-TW"}) == (
        '<html lang="en" data-language="auto">'
    )
    # The page's own choice, from the Mac's languages in the app, wins over the browser's header.
    client.cookies.set("localmelo_language", "zh-Hans")
    assert html_tag(client, **{"Accept-Language": "en-US"}) == (
        '<html lang="zh-Hans" data-language="auto" data-i18n-pending>'
    )


def test_the_setting_overrides_the_system_and_follows_settings(
    app: FastAPI, console: Console
) -> None:
    client = with_page(app, console)
    set_language(console, "en")
    assert html_tag(client, **{"Accept-Language": "zh-CN"}) == (
        '<html lang="en" data-language="en">'
    )
    set_language(console, "zh-Hans")
    assert html_tag(client, **{"Accept-Language": "en-US"}) == (
        '<html lang="zh-Hans" data-language="zh-Hans" data-i18n-pending>'
    )
    # A value the file should not hold reads as auto; it never reaches the page as written.
    set_language(console, '"><script>')
    assert html_tag(client, **{"Accept-Language": "en"}) == (
        '<html lang="en" data-language="auto">'
    )


def test_the_server_writes_its_words_in_the_language_the_page_asks_for(
    client: TestClient, console: Console
) -> None:
    gone = "/api/sessions/no-such-chat"
    english = client.get(gone).json()["error"]
    assert english["code"] == "not_found"
    assert english["message"] == "Session not found."
    client.cookies.set("localmelo_language", "zh-Hans")
    assert client.get(gone).json()["error"]["message"] == "找不到这个会话。"
    client.cookies.clear()
    asked = client.get(gone, headers={"X-LocalMelo-Language": "zh-Hans"})
    assert asked.json()["error"]["message"] == "找不到这个会话。"
    # The code stays the same in every language: a client reads it, not the words.
    assert asked.json()["error"]["code"] == "not_found"
    unknown = client.get(gone, headers={"X-LocalMelo-Language": "fr"})
    assert unknown.json()["error"]["message"] == "Session not found."


def test_the_terminal_reads_english_whatever_the_setting(
    client: TestClient, console: Console
) -> None:
    set_language(console, "zh-Hans")
    found = client.get(
        "/api/sessions/no-such-chat", headers={"Accept-Language": "zh-CN"}
    )
    assert found.json()["error"]["message"] == "Session not found."


def test_activity_rows_name_an_untitled_chat_in_the_reader_s_language(
    client: TestClient,
) -> None:
    assert _row("task", "t1", "", "done", 0.0)["title"] == "Task"
    with using("zh-Hans"):
        assert _row("task", "t1", "", "done", 0.0)["title"] == "任务"
        assert _row("run", "r1", "Fix CI", "done", 0.0)["title"] == "Fix CI"
        assert result_line("", "[interrupt: stopped]") == "回答前已停止"
    zh = {"X-LocalMelo-Language": "zh-Hans"}
    future = client.get("/api/activity?since=4102444800", headers=zh)
    assert future.json()["error"]["message"] == "since 必须是过去的时间。"


def card(data: dict[str, Any]) -> StoredEvent:
    return StoredEvent(
        7, "main", 0, {"type": "attention.card", "ts": 1.0, "data": data}
    )


def test_a_card_is_stored_in_english_and_read_in_the_reader_s_language() -> None:
    answered = {
        "state": "answered",
        "line": "Allowed for this task",
        "words": {"key": "cards.answer.allowedForTask", "params": {}},
    }
    assert card_json(card(answered))["line"] == "Allowed for this task"
    folder = {
        "state": "failed",
        "line": "Could not use its folder: ~/gone does not exist",
        "words": {
            "key": "cards.folderRefused",
            "params": {"problem": "~/gone does not exist"},
        },
    }
    with using("zh-Hans"):
        assert card_json(card(answered))["line"] == "这个任务已允许"
        assert (
            card_json(card(folder))["line"]
            == "无法使用它的文件夹：~/gone does not exist"
        )
        # A card with the model's own words keeps them.
        done = {"state": "done", "line": "Fixed the tests."}
        assert card_json(card(done))["line"] == "Fixed the tests."


class Chat:
    def __init__(self, kind: str, title: str) -> None:
        self.kind, self.title = kind, title


def test_a_note_s_source_reads_in_the_reader_s_language() -> None:
    at = time.mktime((2026, 10, 30, 12, 0, 0, 0, 0, -1))
    assert _from_words(Chat("chat", "Plans"), "", False, at, "web") == (
        "from chat “Plans”, Oct 30"
    )
    with using("zh-Hans"):
        assert _from_words(Chat("chat", "Plans"), "", False, at, "web") == (
            "来自对话“Plans”，10月30日"
        )
        assert (
            _from_words(Chat("front", ""), "", False, at, "web")
            == "来自主对话，10月30日"
        )
        assert _from_words(None, "", True, at, "web") == "来自已删除的对话，10月30日"
