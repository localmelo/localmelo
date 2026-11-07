"""What a waiting card and an activity row say of an approval: plain words (#576)."""

from __future__ import annotations

from typing import Any, cast

import pytest

from localmelo.melo.contracts.approvals import ApprovalRequest
from localmelo.support.gateway.console.activity import approvals_of
from localmelo.support.gateway.console.cards import MainChatCards, _waiting_words
from localmelo.support.sessions.store import StoredEvent
from localmelo.support.words import said_in

SITE = "https://www.example.com"


@pytest.mark.parametrize(
    "tool, arguments, line, chinese",
    [
        (
            f"browser-act:{SITE}",
            {"site": "www.example.com", "how": "click"},
            "Wants to click or choose on example.com",
            "想在 example.com 上点击或选择",
        ),
        (
            f"browser:{SITE}",
            {"site": "www.example.com", "via": "web_fetch"},
            "Wants to read example.com",
            "想读取 example.com",
        ),
        (
            f"takeover:{SITE}",
            {"site": "www.example.com", "need": "sign-in"},
            "Needs you on example.com",
            "需要你在 example.com 上操作",
        ),
        (
            f"browser-type:{SITE}",
            {"site": "www.example.com"},
            "Wants to type into example.com",
            "想在 example.com 中输入",
        ),
    ],
)
def test_a_waiting_card_for_a_site_says_what_melo_wants_to_do_there(
    tool: str, arguments: dict[str, Any], line: str, chinese: str
) -> None:
    """The card names the site and the act, never ``browser-act:`` or ``takeover:``."""
    request = ApprovalRequest(
        request_id="r1",
        run_id="",
        call_id="c1",
        tool_name=tool,
        arguments=arguments,
        preview="",
        workspace_root=None,
    )
    # What the card keeps of an approval's arguments: where, never what.
    kept = {k: v for k, v in arguments.items() if k in ("via", "app")}
    found = _waiting_words({"tool_name": tool, "arguments": kept}, request)
    assert found["line"] == line
    assert said_in("zh-Hans", found["said"]) == chinese
    assert tool not in found["line"]


def _requested(n: int, tool: str) -> StoredEvent:
    data = {
        "request_id": f"r{n}",
        "call_id": f"c{n}",
        "tool_name": tool,
        "arguments": {"site": "www.example.com"},
        "preview": "",
        "workspace_root": "/w",
    }
    return StoredEvent(n, "s", 1, {"type": "approval.requested", "ts": n, "data": data})


def test_an_approval_row_for_a_site_says_what_melo_wanted_to_do_there() -> None:
    """The activity line names the act and the site, never ``browser-act:``."""
    events = [
        _requested(n, tool)
        for n, tool in enumerate(
            (f"browser:{SITE}", f"browser-act:{SITE}", f"takeover:{SITE}"), 1
        )
    ]
    [first, second, third] = approvals_of(events)["s"]
    assert first["what"] == "Open example.com"
    assert second["what"] == "Click or choose on example.com"
    assert third["what"] == "Take over example.com"


def _projected(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """What the main chat's cards keep of an approval another chat asked."""
    cards = MainChatCards(cast(Any, None))
    cards._main_id = "main"
    data = {"request_id": "r1", "call_id": "c1", "tool_name": tool}
    cards._heard(
        StoredEvent(
            1,
            "task",
            1,
            {
                "type": "approval.requested",
                "ts": 1,
                "data": {**data, "arguments": arguments, "preview": ""},
            },
        )
    )
    return cast(dict[str, Any], cards._heard_events[-1].event["data"])


@pytest.mark.parametrize(
    "how, line, chinese",
    [
        ("type into", "Wants to type into example.com", "想在 example.com 中输入"),
        (
            "click",
            "Wants to click or choose on example.com",
            "想在 example.com 上点击或选择",
        ),
        # Only the built-in action is kept; a word of a page's own never is.
        (
            "delete everything",
            "Wants to click or choose on example.com",
            "想在 example.com 上点击或选择",
        ),
    ],
)
def test_a_point_card_says_whether_melo_types_or_clicks(
    how: str, line: str, chinese: str
) -> None:
    """The card says typing for a point asked for typing, from what it keeps of it."""
    tool = f"browser-point:{SITE}"
    arguments = {"site": "www.example.com", "access": "point", "how": how, "on": "x"}
    kept = _projected(tool, arguments)
    assert kept["tool_name"] == tool
    assert set(kept["arguments"]) <= {"path", "url", "via", "app", "how"}
    request = ApprovalRequest(
        request_id="r1",
        run_id="",
        call_id="c1",
        tool_name=tool,
        arguments=arguments,
        preview="",
        workspace_root=None,
    )
    found = _waiting_words(kept, request)
    assert found["line"] == line
    assert said_in("zh-Hans", found["said"]) == chinese
