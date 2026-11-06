"""browser_attach: a file goes to a page's upload field, and only a file melo may send (#551).

The tool gives a click that opens a file chooser the file, with no dialog. What
it will attach is decided before anything is asked, and fails closed: a place
that holds keys or passwords, a link or ``..`` into one, a file named like a
secret, and any file the run does not vouch for (the person named it, or it is
in the chat's folder) are refused with nobody asked. The chat's work folder is
the browser's own. Then the person is asked every time, naming the file, its
folder and the site.
"""

from __future__ import annotations

import asyncio
import functools
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.contracts import work_folder
from localmelo.melo.contracts.approvals import ApprovalDecision
from localmelo.support.browser import chrome
from localmelo.support.browser.driver import (
    Act,
    ActResult,
    BrowserError,
    Changes,
    Element,
)
from localmelo.support.browser.provider import BrowserToolProvider
from localmelo.tests.browser.fakes import FakeDriver, button, link, page
from localmelo.tests.browser.test_downloads import _driver
from localmelo.tests.browser.test_local_apps import Calls
from localmelo.tests.browser.test_provider import asking, run

APPLY = "https://jobs.example/apply"
BUTTON = "Upload resume"


class Run:
    """The run's question handle: it answers *vouches* (path to why) and records every question."""

    def __init__(self, vouches: Mapping[str, str] | None = None, *, allow: bool = True):
        self.vouches = dict(vouches or {})
        self.allow = allow
        self.asked: list[tuple[str, Mapping[str, Any], str]] = []
        self.tried: list[str] = []
        self.during: Callable[[], None] | None = None  # run while the person decides

    def attachable(self, path: str) -> str | None:
        self.tried.append(path)
        return self.vouches.get(path)

    async def __call__(
        self, name: str, arguments: Mapping[str, Any], preview: str
    ) -> ApprovalDecision:
        self.asked.append((name, arguments, preview))
        if self.during is not None:
            self.during()
        return ApprovalDecision(self.allow, "" if self.allow else "not this file")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "home"
    (folder / "Documents").mkdir(parents=True)
    (folder / ".ssh").mkdir()
    (folder / ".ssh" / "id_rsa").write_text("PRIVATE KEY")
    monkeypatch.setenv("HOME", str(folder))
    return Path(os.path.realpath(folder))


@pytest.fixture
def resume(home: Path) -> Path:
    path = home / "Documents" / "resume.pdf"
    path.write_bytes(b"%PDF-1.4 " + b"x" * 2048)
    return path


async def setup(
    *elements: Element, driver: FakeDriver | None = None
) -> tuple[FakeDriver, Any]:
    driver = driver or FakeDriver(
        page(APPLY, "Apply here.", *(elements or (button(BUTTON),)), title="Apply")
    )
    provider = BrowserToolProvider(driver, chat_id="abc123")
    tools = {t.definition.name: t for t in await provider.start()}
    with asking(None):
        await run(tools, "browser_open", url=APPLY)
    return driver, tools


async def attach(
    tools: Any, asks: Run | None, file: str, ref: str = "e1"
) -> tuple[Any, list[Any]]:
    with asking(asks):  # type: ignore[arg-type]
        return await run(tools, "browser_attach", ref=ref, file=file), []


async def test_a_vouched_file_is_attached_and_the_person_is_asked_with_its_name_folder_and_site(
    resume: Path,
) -> None:
    driver, tools = await setup()
    asks = Run({str(resume): "named"})
    out, _ = await attach(tools, asks, str(resume))
    assert not out.is_error, out.text
    assert ("attach", str(resume)) in driver.calls
    for words in (
        'Attached "resume.pdf" (2 KB)',
        "jobs.example",
        "upload a file as soon",
    ):
        assert words in out.text, out.text
    # The send first, then the click on the field's button (the site's own question).
    [(name, arguments, preview), (clicks, _, _)] = asks.asked
    assert clicks == "browser-act:https://jobs.example"
    assert name == "browser-attach:https://jobs.example"
    assert arguments["access"] == "attach" and arguments["name"] == "resume.pdf"
    assert arguments["folder"] == "~/Documents" and arguments["source"] == "named"
    assert preview.startswith(
        "attach resume.pdf from ~/Documents to a field on jobs.example"
    )
    assert str(resume) not in repr(
        arguments
    ), "the question shows no more than it needs"


async def test_the_chats_work_folder_needs_no_voucher_but_still_asks() -> None:
    driver, tools = await setup()
    folder = work_folder.ensure("abc123")
    file = Path(folder) / "form.pdf"
    file.write_bytes(b"%PDF")
    asks = Run()
    out, _ = await attach(tools, asks, str(file))
    assert not out.is_error and ("attach", str(file)) in driver.calls
    assert asks.asked[0][1]["source"] == "work" and asks.tried == []


@pytest.mark.parametrize(
    ("how", "words"),
    [
        ("unvouched", "attaches only a file the user named"),
        ("relative", "full path"),
        ("missing", "not a file melo can read"),
        ("folder", "is not a file"),
        ("key", "holds keys or passwords"),
        ("link", "holds keys or passwords"),
        ("dotdot", "holds keys or passwords"),
        ("env", "looks like a key, a password or a secret"),
    ],
)
async def test_a_file_that_is_not_allowed_is_refused_and_nobody_is_asked(
    home: Path, resume: Path, how: str, words: str
) -> None:
    (home / "Documents" / "key.pdf").symlink_to(home / ".ssh" / "id_rsa")
    (home / "Documents" / ".env").write_text("TOKEN=1")
    file = {
        "unvouched": str(resume),
        "relative": "Documents/resume.pdf",
        "missing": str(home / "Documents" / "gone.pdf"),
        "folder": str(home / "Documents"),
        "key": str(home / ".ssh" / "id_rsa"),
        "link": str(home / "Documents" / "key.pdf"),
        "dotdot": str(home / "Documents" / ".." / ".ssh" / "id_rsa"),
        "env": str(home / "Documents" / ".env"),
    }[how]
    driver, tools = await setup()
    # The run vouches for everything it is asked about, as when the person named it all.
    everything = {
        file: "named",
        **{str(p): "named" for p in (home / "Documents").iterdir()},
    }
    asks = Run({} if how == "unvouched" else everything)
    out, _ = await attach(tools, asks, file)
    assert out.is_error and words in out.text, out.text
    assert asks.asked == [] and not [
        c for c in driver.calls if c[0] in ("attach", "act")
    ]


async def test_without_a_run_that_vouches_only_the_work_folder_is_attached(
    resume: Path,
) -> None:
    driver, tools = await setup()
    out, _ = await attach(tools, None, str(resume))
    assert out.is_error and "attaches only a file the user named" in out.text
    assert ("attach", str(resume)) not in driver.calls


async def test_a_refusal_by_the_person_attaches_nothing_and_says_so(
    resume: Path,
) -> None:
    driver, tools = await setup()
    asks = Run({str(resume): "named"}, allow=False)
    out, _ = await attach(tools, asks, str(resume))
    assert out.is_error and "melo did not attach resume.pdf to jobs.example" in out.text
    assert "not this file" in out.text
    assert not [c for c in driver.calls if c[0] in ("attach", "act")]


async def test_a_file_swapped_for_a_link_while_the_person_decides_is_not_attached(
    home: Path, resume: Path
) -> None:
    driver, tools = await setup()
    asks = Run({str(resume): "named"})

    def swap() -> None:
        resume.unlink()
        resume.symlink_to(home / ".ssh" / "id_rsa")

    asks.during = swap
    out, _ = await attach(tools, asks, str(resume))
    assert out.is_error and "holds keys or passwords" in out.text
    assert not [c for c in driver.calls if c[0] in ("attach", "act")]


@pytest.mark.parametrize(
    "element",
    [
        Element("textbox", "Name"),
        link("Home", "https://jobs.example/"),
        button("Send application", submit="https://jobs.example/send"),
        Element("button", "Upload", site="https://other.example/frame"),
        Element("textbox", "", sensitive="password"),
    ],
    ids=[
        "a text box",
        "a link",
        "a send button",
        "a box from another site",
        "a secret",
    ],
)
async def test_only_a_file_field_or_an_upload_button_of_the_page_is_clicked_to_attach(
    resume: Path, element: Element
) -> None:
    driver, tools = await setup(element)
    asks = Run({str(resume): "named"})
    out, _ = await attach(tools, asks, str(resume))
    assert out.is_error and asks.asked == []
    assert not [c for c in driver.calls if c[0] in ("attach", "act")]


async def test_a_browser_that_cannot_answer_a_chooser_says_so_and_asks_nobody(
    resume: Path,
) -> None:
    driver = FakeDriver(page(APPLY, "Apply.", button(BUTTON)))
    driver.attaches = False
    driver, tools = await setup(driver=driver)
    asks = Run({str(resume): "named"})
    out, _ = await attach(tools, asks, str(resume))
    assert out.is_error and "cannot attach a file" in out.text and asks.asked == []


# ── Chrome: the chooser a click opens is answered with the file ──


class Clicking:
    """A stand-in for the click of ``act``: it may open a file chooser, now or a moment later."""

    def __init__(
        self, browser: chrome._Chrome, *, opens: bool, later: float = 0.0
    ) -> None:
        self.browser, self.opens, self.later = browser, opens, later

    async def __call__(self, tab: str | None, act: Act) -> ActResult:
        if self.opens:
            params = {"frameId": "F", "mode": "selectSingle", "backendNodeId": 77}
            opened = functools.partial(
                self.browser._dispatch,
                {
                    "method": "Page.fileChooserOpened",
                    "params": params,
                    "sessionId": "S1",
                },
            )
            if self.later:
                asyncio.get_running_loop().call_later(self.later, opened)
            else:
                opened()
        return ActResult("", "browser", "", Changes())


async def _chrome_attach(
    monkeypatch: pytest.MonkeyPatch, *, opens: bool, later: float = 0.0
) -> tuple[Calls, Exception | None]:
    calls = Calls()
    driver, browser = _driver(calls)

    async def focus(tab: str | None) -> str:
        return "S1"

    monkeypatch.setattr(driver, "_focus", focus)
    monkeypatch.setattr(driver, "act", Clicking(browser, opens=opens, later=later))
    monkeypatch.setattr(chrome, "CHOOSER_S", 0.2)
    act = Act("click", "https://jobs.example/apply", Element("button", "Upload"))
    try:
        await driver.attach("t1", act, "/h/resume.pdf")
    except BrowserError as error:
        return calls, error
    return calls, None


@pytest.mark.parametrize("later", [0.0, 0.05], ids=["at once", "a moment later"])
async def test_chrome_gives_the_chooser_the_click_opens_the_file_and_stops_intercepting(
    monkeypatch: pytest.MonkeyPatch, later: float
) -> None:
    calls, error = await _chrome_attach(monkeypatch, opens=True, later=later)
    assert error is None
    assert [(m, p) for m, p, _ in calls.made] == [
        ("Page.setInterceptFileChooserDialog", {"enabled": True}),
        ("DOM.setFileInputFiles", {"files": ["/h/resume.pdf"], "backendNodeId": 77}),
        ("Page.setInterceptFileChooserDialog", {"enabled": False}),
    ]


async def test_a_click_that_opens_no_chooser_attaches_nothing_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, error = await _chrome_attach(monkeypatch, opens=False)
    assert error is not None and "opened no file chooser" in str(error)
    assert [m for m, _, _ in calls.made] == [
        "Page.setInterceptFileChooserDialog",
        "Page.setInterceptFileChooserDialog",
    ]
    assert [p for _, p, _ in calls.made] == [{"enabled": True}, {"enabled": False}]
