"""Chinese website wording that carries a security or run/turn claim.

These pages are hand-synchronized with their English sources rather than
checked by a pair test, so the statements most likely to drift silently are
pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "page",
    ["docs/zh-CN/architecture.html", "docs/zh-CN/index.html"],
)
def test_security_statement_keeps_the_bearer_qualifier(page: str) -> None:
    text = (ROOT / page).read_text()
    assert "网关要求 bearer 令牌或浏览器会话" in text
    assert "持有令牌者能以你的权限操作" in text
    assert "同一用户的进程之间没有隔离" in text
    assert "可以你的权限操作" not in text
    assert "令牌无法隔离同一用户的进程" not in text


@pytest.mark.parametrize(
    "page",
    [
        "docs/zh-CN/architecture.html",
        "docs/zh-CN/quickstart.html",
        "docs/zh-CN/updates.html",
    ],
)
def test_live_area_reports_the_runs_progress_facts(page: str) -> None:
    text = (ROOT / page).read_text()
    assert "本次运行的进度事实" in text
    assert "本轮的进度事实" not in text


def test_web_console_twin_follows_the_playground_move() -> None:
    en = (ROOT / "docs/guides/web-console.md").read_text()
    zh = (ROOT / "docs/zh-CN/guides/web-console.md").read_text()
    assert "disabled-by-default [legacy playground routes]" in en
    assert "旧版 playground 路由" in zh
    assert "../../reference/http-v1.md#legacy-playground" in zh
    assert "security-1b" not in zh
    assert "_test_webapp" not in zh
