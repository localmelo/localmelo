"""browser_attach: a file goes to a page's upload field, and only a file melo may send (#551).

The tool gives a click that opens a file chooser the file, with no dialog. What
it will attach is decided before anything is asked, and fails closed: a place
that holds keys or passwords, a link or ``..`` into one, a file named like a
secret are refused with nobody asked. A file the run vouches for (the person
named it, or it is in the chat's folder) and the chat's work folder, which is
the browser's own, come next; any other file is asked about by name, whoever
suggested it. Then the person is asked every time, naming the file, its folder
and the site, and the file is checked again right before the browser gets it.
"""

from __future__ import annotations

import asyncio
import functools
import os
import shutil
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
    Upload,
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


@pytest.mark.parametrize("how", ["loose", "linked", "localmelo linked"])
async def test_a_file_in_a_work_folder_that_is_no_longer_sound_is_no_work_file(
    tmp_path: Path, how: str
) -> None:
    """The folder is checked when a file is classified: a link or an opened-up folder is not the browser's own, so its file is asked about as any other."""
    driver, tools = await setup()
    folder = work_folder.ensure("abc123")
    file = Path(folder) / "form.pdf"
    file.write_bytes(b"%PDF")
    base = os.path.dirname(folder)
    if how == "loose":
        os.chmod(folder, 0o755)
    elif how == "linked":
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "form.pdf").write_bytes(b"%PDF")
        shutil.rmtree(folder)
        os.symlink(elsewhere, folder)
    else:
        os.rename(base, tmp_path / "moved")
        os.symlink(tmp_path / "moved", base)
    asks = Run()
    out, _ = await attach(tools, asks, str(file))
    assert asks.asked[0][1]["source"] == "other", out.text


@pytest.mark.parametrize(
    ("how", "words"),
    [
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
    asks = Run(everything)
    out, _ = await attach(tools, asks, file)
    assert out.is_error and words in out.text, out.text
    assert asks.asked == [] and not [
        c for c in driver.calls if c[0] in ("attach", "act")
    ]


async def test_a_file_nobody_vouched_for_is_asked_about_by_name_and_goes_only_on_yes(
    resume: Path,
) -> None:
    """The page or the model suggested it, or the person named only its folder: the card names exactly this file."""
    driver, tools = await setup()
    yes = Run()
    out, _ = await attach(tools, yes, str(resume))
    assert not out.is_error, out.text
    name, arguments, preview = yes.asked[0]
    assert name.startswith("browser-attach:")
    assert arguments["source"] == "other" and arguments["name"] == "resume.pdf"
    assert preview.startswith("attach resume.pdf from ~/Documents to a field on")
    assert ("attach", str(resume)) in driver.calls
    driver, tools = await setup()
    no = Run(allow=False)
    out, _ = await attach(tools, no, str(resume))
    assert out.is_error and "did not attach resume.pdf" in out.text
    assert not [c for c in driver.calls if c[0] in ("attach", "act")]


async def test_with_nobody_to_ask_a_file_nobody_vouched_for_is_not_attached(
    resume: Path,
) -> None:
    driver, tools = await setup()
    out, _ = await attach(tools, None, str(resume))
    assert out.is_error and "nobody can be asked here" in out.text
    assert not [c for c in driver.calls if c[0] in ("attach", "act")]


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


async def test_a_file_swapped_while_the_click_is_asked_about_is_not_given_to_the_browser(
    home: Path, resume: Path
) -> None:
    """The file card was answered; the click has a question of its own, and the file changes during that one."""
    driver, tools = await setup()
    asks = Run({str(resume): "named"})

    def swap() -> None:
        if asks.asked[-1][0].startswith("browser-act:"):
            resume.unlink()
            resume.symlink_to(home / ".ssh" / "id_rsa")

    asks.during = swap
    out, _ = await attach(tools, asks, str(resume))
    assert [n.split(":")[0] for n, _, _ in asks.asked] == [
        "browser-attach",
        "browser-act",
    ]
    assert out.is_error and "holds keys or passwords" in out.text
    assert ("attach", str(resume)) not in driver.calls


async def test_another_file_at_the_same_path_is_not_the_file_that_was_asked_about(
    resume: Path,
) -> None:
    driver, tools = await setup()
    asks = Run({str(resume): "named"})

    def replace() -> None:
        if asks.asked[-1][0].startswith("browser-act:"):
            resume.unlink()
            resume.write_bytes(b"%PDF-1.4 another file")

    asks.during = replace
    out, _ = await attach(tools, asks, str(resume))
    assert out.is_error and "changed while melo waited" in out.text
    assert ("attach", str(resume)) not in driver.calls


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


def _ok() -> None:
    pass


async def _chrome_attach(
    monkeypatch: pytest.MonkeyPatch,
    *,
    opens: bool,
    later: float = 0.0,
    check: Callable[[], None] = _ok,
    calls: Calls | None = None,
) -> tuple[Calls, Exception | None]:
    calls = calls or Calls()
    driver, browser = _driver(calls)

    async def focus(tab: str | None) -> str:
        driver._current = "MINE"  # the tab the click is on, one of this driver's
        return "S1"

    monkeypatch.setattr(driver, "_focus", focus)
    monkeypatch.setattr(driver, "act", Clicking(browser, opens=opens, later=later))
    monkeypatch.setattr(chrome, "CHOOSER_S", 0.2)
    act = Act("click", "https://jobs.example/apply", Element("button", "Upload"))
    try:
        await driver.attach("t1", act, Upload("/h/resume.pdf", check))
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


async def test_a_file_that_changed_while_the_chooser_waited_is_not_given_to_chrome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check runs after the chooser opened, right before the file goes: the last wait is behind it."""

    def changed() -> None:
        raise BrowserError(
            "resume.pdf changed while melo waited, so nothing was attached."
        )

    calls, error = await _chrome_attach(
        monkeypatch, opens=True, later=0.05, check=changed
    )
    assert error is not None and "changed while melo waited" in str(error)
    assert [p for m, p, _ in calls.made if m == "DOM.setFileInputFiles"] == []
    assert [p for m, p, _ in calls.made if "Intercept" in m] == [
        {"enabled": True},
        {"enabled": False},
    ]


class Refusing(Calls):
    """Chrome's pipe where some commands fail: *fail* is called with the method and its parameters."""

    def __init__(self, fail: Callable[[str, dict[str, Any]], bool]) -> None:
        super().__init__()
        self.fail = fail

    def __call__(  # type: ignore[override]
        self,
        method: str,
        params: dict[str, Any] | None = None,
        session: str | None = None,
    ) -> Any:
        future = super().__call__(method, params, session)
        if self.fail(method, params or {}):
            future = asyncio.get_running_loop().create_future()
            future.set_exception(BrowserError(f"{method} failed"))
        return future


def _turn_off_fails(method: str, params: dict[str, Any]) -> bool:
    return method == "Page.setInterceptFileChooserDialog" and not params["enabled"]


async def test_an_interception_that_cannot_be_turned_off_is_told_and_the_tab_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The file went; the page would keep handing its choosers to melo, so the step says it, and the tab is retired."""
    calls = Refusing(_turn_off_fails)
    _, error = await _chrome_attach(monkeypatch, opens=True, calls=calls)
    assert error is not None and "file was attached, but" in str(error)
    assert "closed the tab" in str(error)
    made = [m for m, _, _ in calls.made]
    assert made.count("Page.setInterceptFileChooserDialog") == 3  # on, off, off again
    assert made[-1] == "Target.closeTarget"


async def test_when_even_the_tab_cannot_be_closed_the_browser_is_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = Refusing(lambda m, p: _turn_off_fails(m, p) or m == "Target.closeTarget")
    stopped: list[bool] = []

    async def stop(self: chrome.ChromeBackgroundDriver) -> None:
        stopped.append(True)

    monkeypatch.setattr(chrome.ChromeBackgroundDriver, "close", stop)
    _, error = await _chrome_attach(monkeypatch, opens=True, calls=calls)
    assert error is not None and "stopped this browser" in str(error)
    assert stopped == [True]


async def test_a_failed_attach_that_also_cannot_turn_off_says_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = Refusing(_turn_off_fails)
    _, error = await _chrome_attach(monkeypatch, opens=False, calls=calls)
    assert error is not None
    assert "opened no file chooser" in str(error) and "closed the tab" in str(error)


async def test_an_interception_whose_start_failed_is_turned_off_anyway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command may have reached Chrome before it failed to answer: the cleanup does not assume it did not."""
    calls = Refusing(
        lambda m, p: m == "Page.setInterceptFileChooserDialog" and p["enabled"]
    )
    _, error = await _chrome_attach(monkeypatch, opens=True, calls=calls)
    assert error is not None
    assert [p for m, p, _ in calls.made if "Intercept" in m] == [
        {"enabled": True},
        {"enabled": False},
    ]
