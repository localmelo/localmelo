"""A window of an app the page did not ask for: the person decides, and nothing is told before.

Only the app an address opens is read on its own. Any other window that
appears inside the admission deadline of melo's act raises a card from the
provider itself ("<App> opened a window right after melo's click: let melo see
it?"); neither the model nor a ``window_read`` call asks it, and a window the
person does not allow (or does not answer) is never named, read or told.

The person is ``Person``: the chat's real ``RememberingPolicy`` behind scripted
answers, so "Allow <App> in this chat" is the policy's own grant. The screen is
the fake desktop helper; nothing here touches the real screen.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts.approvals import ApprovalDecision, ApprovalRequest
from localmelo.support.browser import provider as provider_mod
from localmelo.support.browser.desktop import Front
from localmelo.support.browser.driver import Element
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.support.sessions.approvals import RememberingPolicy
from localmelo.tests.browser.fakes import button
from localmelo.tests.browser.test_acting import HOME, Asks, asking, world
from localmelo.tests.browser.test_late_windows import SSO, opened, ref_of, window_calls
from localmelo.tests.browser.test_other_windows import SIGN_IN, Screen, run, watch

CHAT = "window-read:com.example.chat"
SECOND_CHAT = {**SIGN_IN, "window_id": 9, "title": "Pick an account", "z_index": 14}
SECRET = ("Sign in to example.com", "Approve sign-in", "Pick an account")

ALLOW_ONCE = ApprovalDecision(True)
ALLOW_APP = ApprovalDecision(True, remember=True)
DENY = ApprovalDecision(False, "not this one")


@pytest.fixture
def screen(tmp_path: Path) -> Screen:
    return Screen(tmp_path)


class Person(Asks):
    """The person at the chat's card, and the browser's other questions answered yes.

    *answers* are what they say to each window card in turn (``None``: no
    answer, ever). The cards go through a real ``RememberingPolicy``, so an
    answer that allows with ``remember`` is the chat's grant of that app, and
    its next window is allowed without a card (``cards`` keeps the ones shown).
    *during* runs when a question of a given name is asked: what happens on the
    screen while the person is deciding.
    """

    def __init__(
        self,
        *answers: ApprovalDecision | None,
        during: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.answers = list(answers)
        self.cards: list[ApprovalRequest] = []
        self.policy = RememberingPolicy(self._card)
        self._during = during

    async def _card(self, request: ApprovalRequest) -> ApprovalDecision:
        self.cards.append(request)
        answer = self.answers.pop(0)
        if answer is None:
            await asyncio.Event().wait()
        assert answer is not None
        return answer

    async def __call__(
        self, name: str, arguments: Mapping[str, Any], preview: str
    ) -> ApprovalDecision:
        self.asked.append((name, arguments, preview))
        if self._during is not None:
            self._during(name)
        if not name.startswith("window-read:"):
            return ApprovalDecision(True, remember=True)
        request = ApprovalRequest(
            request_id=f"r{len(self.asked)}",
            run_id="run",
            call_id=f"c{len(self.asked)}",
            tool_name=name,
            arguments=arguments,
            preview=preview,
            workspace_root="/ws",
        )
        return await self.policy.decide(request)

    @property
    def windows(self) -> list[str]:
        """The window questions the provider asked, by name."""
        return [name for name in self.names if name.startswith("window-read:")]


async def settled(provider: BrowserToolProvider) -> None:
    """Let the cards the provider raised run to their end."""
    end = time.monotonic() + 5.0
    while provider._cards and time.monotonic() < end:
        await asyncio.sleep(0.01)
    assert not provider._cards


def told(*outputs: Any) -> str:
    return "\n".join(out.text for out in outputs)


def hidden(
    person: Person, screen: Screen, provider: BrowserToolProvider, text: str
) -> None:
    """Nothing about the window reached a model, or was read from the screen."""
    for title in SECRET:
        assert title not in text
    assert "opened a window" not in text and "Another app" not in text
    assert provider._held == []
    assert window_calls(screen) == []


# ── The reviewer's cases: an unrelated app is no longer disclosed ──


async def test_an_unrelated_app_four_seconds_after_an_ordinary_click_asks_and_tells_nothing(
    screen: Screen,
) -> None:
    person = Person(DENY)
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, strict=True)
    )
    try:
        with asking(person):
            click = await run(tools, "browser_click", ref="e1")
            screen.show(
                SIGN_IN, front=False
            )  # 4 s after the click, beyond its late wait
            screen.clock.now += 1
            read = await run(tools, "browser_read")
            await settled(provider)
            again = await run(tools, "browser_read")
            latest = await run(tools, "window_read")
            named = await run(tools, "window_read", window="w1")
    finally:
        await provider.close()
    # The provider asked the person, once, in the question's own words.
    [asked] = [item for item in person.asked if item[0].startswith("window-read:")]
    name, arguments, preview = asked
    assert name == CHAT
    assert (
        arguments["app"] == "Chat" and arguments["window"] == "Sign in to example.com"
    )
    assert preview == (
        'Chat opened a window "Sign in to example.com" right after melo\'s click: '
        "let melo see it?"
    )
    [card] = person.cards
    assert card.tool_name == CHAT
    # Denied: not named, not read, not told by any step, not readable by id.
    hidden(person, screen, provider, told(click, read, again))
    assert latest.is_error and named.is_error


async def test_a_window_that_opened_while_a_submit_waited_for_its_card_is_never_taken_for_its_act(
    screen: Screen,
) -> None:
    """The baseline is renewed right before the act is dispatched, after the approval."""
    for strict in (True, False):
        folder = screen.path.parent / f"strict-{strict}"
        folder.mkdir()
        here = Screen(folder)

        def appears(name: str, here: Screen = here) -> None:
            if name.startswith("browser-submit:"):
                here.show(SIGN_IN, front=False)  # while the person decides on the form

        person = Person(during=appears)
        provider, tools = await opened(
            world(Element("button", "Send", submit=HOME)), watch(here, strict=strict)
        )
        try:
            with asking(person):
                click = await run(tools, "browser_click", ref="e1")
                read = await run(tools, "browser_read")
                await settled(provider)
        finally:
            await provider.close()
        assert any(name.startswith("browser-submit:") for name in person.names)
        assert person.windows == [] and person.cards == [], strict
        hidden(person, here, provider, told(click, read))


# ── The card ──


async def test_allow_once_lets_melo_see_that_window_only(screen: Screen) -> None:
    person = Person(ALLOW_ONCE, DENY)
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, strict=True)
    )
    try:
        with asking(person):
            await run(tools, "browser_click", ref="e1")
            screen.show(SIGN_IN, front=False)
            screen.clock.now += 1
            read = await run(tools, "browser_read")
            await settled(provider)
            again = await run(tools, "browser_read")
            window = await run(tools, "window_read", window="w1")
            third = await run(tools, "browser_read")
            # The next act: another window of the same app asks again.
            screen.clock.at(1.0, lambda: screen.show(SECOND_CHAT, front=False))
            await run(tools, "browser_click", ref=ref_of(third))
            await settled(provider)
            last = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert "opened a window" not in read.text  # asked, not yet allowed
    assert "Chat opened a window 'Sign in to example.com' (w1)" in again.text
    assert "Allow" in again.text and not window.is_error
    assert "opened a window" not in third.text  # told once
    assert len(person.cards) == 2, "the answer was for that one window"
    assert not person.policy.allowed_tools
    assert "Pick an account" not in last.text  # denied


async def test_allow_the_app_in_this_chat_lets_its_later_windows_through_unasked(
    screen: Screen,
) -> None:
    person = Person(ALLOW_APP, DENY)
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, strict=True)
    )
    try:
        with asking(person):
            await run(tools, "browser_click", ref="e1")
            screen.show(SIGN_IN, front=False)
            screen.clock.now += 1
            await run(tools, "browser_read")
            await settled(provider)
            first = await run(tools, "browser_read")
            fresh = await run(tools, "browser_read")

            def both() -> None:
                screen.show(SECOND_CHAT, front=False)
                screen.show(SSO, front=False)

            screen.clock.at(1.0, both)
            await run(tools, "browser_click", ref=ref_of(fresh))
            await settled(provider)
            second = await run(tools, "browser_read")
            window = await run(tools, "window_read", window="w2")
    finally:
        await provider.close()
    assert "Chat opened a window 'Sign in to example.com' (w1)" in first.text
    # The grant is the chat's own, by the app's bundle id; the provider still asked,
    # and the policy answered without a card for the same app.
    assert person.policy.allowed_tools == {CHAT}
    assert person.windows.count(CHAT) == 2
    assert [card.tool_name for card in person.cards] == [
        CHAT,
        "window-read:com.example.sso",
    ]
    assert "Chat opened a window 'Pick an account' (w2)" in second.text
    assert not window.is_error
    # Another app is still its own question, and was denied: not told at all.
    assert "SSO" not in second.text and "Approve sign-in" not in second.text


async def test_dont_allow_tells_nothing_not_even_the_apps_name(screen: Screen) -> None:
    person = Person(DENY)
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, strict=True)
    )
    try:
        with asking(person):
            screen.clock.at(
                1.0, lambda: screen.show(SIGN_IN)
            )  # in front, inside the late wait
            click = await run(tools, "browser_click", ref="e1")
            await settled(provider)
            read = await run(tools, "browser_read")
            later = await run(tools, "browser_click", ref=ref_of(read))
    finally:
        await provider.close()
    assert len(person.cards) == 1, "it asks once: the window is in the baseline after"
    hidden(person, screen, provider, told(click, read, later))
    assert "Chat" not in told(click, read, later)


async def test_no_answer_within_the_limit_tells_nothing(
    screen: Screen, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provider_mod, "CARD_S", 0.05)
    person = Person(None)
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, strict=True)
    )
    try:
        with asking(person):
            screen.clock.at(1.0, lambda: screen.show(SIGN_IN))
            click = await run(tools, "browser_click", ref="e1")
            await settled(provider)  # the card ends at its limit
            read = await run(tools, "browser_read")
            named = await run(tools, "window_read", window="w1")
    finally:
        await provider.close()
    assert len(person.cards) == 1
    hidden(person, screen, provider, told(click, read))
    assert named.is_error


async def test_a_card_is_raised_while_the_model_is_still_planning(
    screen: Screen,
) -> None:
    """The sampler finds the window; the person is asked at once, with no step running."""
    person = Person(ALLOW_ONCE)
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, strict=True)
    )
    try:
        with asking(person):
            await run(tools, "browser_click", ref="e1")
            screen.clock.now += 2
            screen.show(SIGN_IN, front=False)
            await screen.pacer.tick(provider)
            await settled(provider)
            assert len(person.cards) == 1  # asked, though no step has run since
            read = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert "Chat opened a window 'Sign in to example.com' (w1)" in read.text


async def test_without_a_way_to_ask_nothing_is_told(screen: Screen) -> None:
    person = Person()
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, strict=True)
    )
    try:
        with asking(person):
            await run(tools, "browser_click", ref="e1")
        provider._ask = None  # the run that acted has ended: nobody to ask
        screen.clock.now += 1
        screen.show(SIGN_IN, front=False)
        with asking(None):
            read = await run(tools, "browser_read")
            again = await run(tools, "browser_read")
    finally:
        await provider.close()
    assert person.cards == [] and person.windows == []
    hidden(person, screen, provider, told(read, again))


# ── The app the page asked for is still read on its own ──


async def test_the_app_the_page_asked_for_is_read_on_its_own_and_any_other_asks(
    screen: Screen,
) -> None:
    person = Person(DENY)
    driver = world(button("Quick sign-in"))
    driver.lands["Quick sign-in"] = "chat://authorize?site=example.com"
    desk = watch(screen, strict=True, asked=("chat", "com.example.chat"))
    provider, tools = await opened(driver, desk)
    try:
        with asking(person):

            def both() -> None:
                screen.show(SSO)  # on top, but not the app the address opened
                screen.show(SIGN_IN)

            screen.clock.at(1.0, both)
            click = await run(tools, "browser_click", ref="e1")
            await settled(provider)
            read = await run(tools, "browser_read")
            window = await run(tools, "window_read", window="w1")
    finally:
        await provider.close()
    lines = [line for line in click.text.splitlines() if "opened a window" in line]
    assert len(lines) == 1
    assert lines[0].startswith(
        "- Chat (the app the page asked for) opened a window 'Sign in to example.com' (w1)"
    )
    assert not window.is_error  # read on its own: no card
    # The other app asked, was denied, and is nowhere in what melo was told.
    assert [card.tool_name for card in person.cards] == ["window-read:com.example.sso"]
    assert "SSO" not in told(click, read) and "Approve sign-in" not in told(click, read)
    assert (
        window_calls(screen) == ["get_window_state"] * 2
    )  # Chat's buttons, then the read


# ── Nobody keeps waiting ──


async def test_cards_end_when_the_chat_is_forgotten_or_closed(screen: Screen) -> None:
    for how in ("forget", "close"):
        folder = screen.path.parent / how
        folder.mkdir()
        here = Screen(folder)
        person = Person(None)
        provider, tools = await opened(
            world(button("Quick sign-in")), watch(here, strict=True)
        )
        try:
            with asking(person):
                here.clock.at(1.0, lambda here=here: here.show(SIGN_IN))  # type: ignore[misc]
                await run(tools, "browser_click", ref="e1")
                await asyncio.sleep(0.05)
                assert len(person.cards) == 1 and provider._cards
                cards = list(provider._cards)
                if how == "forget":
                    provider.forget()
                else:
                    await provider.close()
                await asyncio.sleep(0)
                assert not provider._cards and all(card.done() for card in cards), how
        finally:
            await provider.close()


async def test_an_app_the_person_did_not_allow_is_not_named_when_it_comes_back_to_the_front(
    screen: Screen,
) -> None:
    # Denied once: when the same app later comes to the front with no new window,
    # the name-only line must not name it either.
    person = Person(DENY)
    provider, tools = await opened(
        world(button("Quick sign-in")), watch(screen, strict=True)
    )
    try:
        with asking(person):
            screen.clock.at(1.0, lambda: screen.show(SIGN_IN))
            first = await run(tools, "browser_click", ref="e1")
            await settled(provider)
            screen.front = Front("Safari", "com.apple.Safari", 10)
            read = await run(tools, "browser_read")
            chat = Front(
                str(SIGN_IN["app_name"]),
                str(SIGN_IN["bundle"]),
                int(str(SIGN_IN["pid"])),
            )
            screen.clock.at(1.0, lambda: setattr(screen, "front", chat))
            again = await run(tools, "browser_click", ref=ref_of(read))
            await settled(provider)
    finally:
        await provider.close()
    assert "Chat" not in told(first, read, again)
