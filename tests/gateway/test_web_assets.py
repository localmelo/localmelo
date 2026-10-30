"""Verify the workspace shell and its packaged runtime resources."""

from __future__ import annotations

import re
import struct
import tomllib
from collections.abc import AsyncIterator
from html.parser import HTMLParser
from importlib import resources
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi import FastAPI  # noqa: E402

from localmelo.support.gateway import webapp  # noqa: E402
from localmelo.support.gateway.playground import (  # noqa: E402
    _SCENARIOS_PATH,
    SmokePlayground,
)


class ShellAssets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script" and attributes.get("src"):
            self.paths.append(str(attributes["src"]))
        if tag == "link" and attributes.get("rel") in {
            "stylesheet",
            "icon",
            "apple-touch-icon",
        }:
            self.paths.append(str(attributes["href"]))
        if tag == "img" and attributes.get("src"):
            self.paths.append(str(attributes["src"]))


class IconLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.icons: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "link" and attributes.get("rel") in {"icon", "apple-touch-icon"}:
            self.icons.append(attributes)


class InputPlaceholders(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.placeholders: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "input" and attributes.get("id") and attributes.get("placeholder"):
            self.placeholders[str(attributes["id"])] = str(attributes["placeholder"])


@pytest.fixture()
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    manager = SmokePlayground(memory_dir=tmp_path)
    app.state.smoke_playground = manager
    webapp.mount(app, enable_playground=True)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        await manager.close_all()


async def test_instance_header_only_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALMELO_APP_INSTANCE", "0123456789abcdef" * 2)
    app = FastAPI()
    manager = SmokePlayground(memory_dir=tmp_path)
    app.state.smoke_playground = manager
    webapp.mount(app)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            page = await client.get("/")
            assert page.status_code == 200
            assert page.headers["x-localmelo-instance"] == "0123456789abcdef" * 2
            assert (
                "x-localmelo-instance"
                not in (await client.get("/assets/workspace.js")).headers
            )
    finally:
        await manager.close_all()


@pytest.mark.parametrize("instance", [None, "wrong-value", "a" * 31])
async def test_bad_or_absent_instance_has_no_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, instance: str | None
) -> None:
    if instance is None:
        monkeypatch.delenv("LOCALMELO_APP_INSTANCE", raising=False)
    else:
        monkeypatch.setenv("LOCALMELO_APP_INSTANCE", instance)
    app = FastAPI()
    manager = SmokePlayground(memory_dir=tmp_path)
    app.state.smoke_playground = manager
    webapp.mount(app)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert "x-localmelo-instance" not in (await client.get("/")).headers
    finally:
        await manager.close_all()


async def test_shell_assets_and_module_imports_are_served(
    client: httpx.AsyncClient,
) -> None:
    page = await client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    parser = ShellAssets()
    parser.feed(page.text)
    assert any(path.endswith(".css") for path in parser.paths)
    assert any(path.endswith(".js") for path in parser.paths)

    for path in set(parser.paths):
        assert path.startswith("/assets/"), "The workspace must load its assets locally"
        response = await client.get(path)
        assert response.status_code == 200, path
        assert response.content, path
        media_type = response.headers["content-type"].split(";")[0]
        if path.endswith(".png"):
            assert media_type == "image/png"
            assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
        elif path.endswith(".css"):
            assert media_type == "text/css"
        else:
            assert media_type in {"text/javascript", "application/javascript"}
            imports = re.findall(r"from\s+['\"]([^'\"]+)['\"]", response.text)
            for module in imports:
                module_response = await client.get(str(response.url.join(module)))
                assert module_response.status_code == 200, module
                assert module_response.headers["content-type"].split(";")[0] in {
                    "text/javascript",
                    "application/javascript",
                }


async def test_icon_links_point_at_images_large_enough_for_their_use(
    client: httpx.AsyncClient,
) -> None:
    page = await client.get("/")
    parser = IconLinks()
    parser.feed(page.text)
    assert parser.icons
    for icon in parser.icons:
        image = (await client.get(str(icon["href"]))).content
        assert image[12:16] == b"IHDR", icon
        width, height = struct.unpack(">II", image[16:24])
        if icon["rel"] == "icon":
            assert icon.get("sizes") == f"{width}x{height}", icon
        else:
            # Touch icons are drawn at up to 180 px; a smaller image is scaled up and blurs.
            assert min(width, height) >= 180, icon


async def test_chat_url_placeholder_matches_the_omlx_hint(
    client: httpx.AsyncClient,
) -> None:
    page = await client.get("/")
    parser = InputPlaceholders()
    parser.feed(page.text)
    # The hint beside the field names oMLX, whose server listens on port 8000 by default.
    assert "oMLX" in page.text
    assert parser.placeholders["s-chat-url"] == "http://127.0.0.1:8000/v1"


@pytest.mark.parametrize(
    "path", ["/assets/missing.js", "/assets/../webapp.py", "/assets/%2e%2e/webapp.py"]
)
async def test_asset_routes_do_not_return_the_shell_or_python_source(
    client: httpx.AsyncClient, path: str
) -> None:
    response = await client.get(path)
    assert response.status_code == 404
    assert "SmokePlayground" not in response.text
    assert "<html" not in response.text


async def test_scenarios_load_with_workspace_resources(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/v1/playground/scenarios")
    assert response.status_code == 200
    scenarios = response.json()["scenarios"]
    assert scenarios
    assert all(scenario["id"] and scenario["name"] for scenario in scenarios)


def test_package_data_includes_all_workspace_runtime_resources() -> None:
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    declared: set[Path] = set()
    for package, patterns in config["tool"]["setuptools"]["package-data"].items():
        package_root = Path(str(resources.files(package)))
        for pattern in patterns:
            declared.update(path.resolve() for path in package_root.glob(pattern))

    required = {path.resolve() for path in webapp._STATIC.iterdir() if path.is_file()}
    required.add(_SCENARIOS_PATH.resolve())
    missing = sorted(str(path.relative_to(root)) for path in required - declared)
    assert not missing, f"Runtime resources missing from package-data: {missing}"


def _shell_attributes(html: str, tag: str, element_id: str) -> dict[str, str | None]:
    """The attributes of the first *tag* with *element_id* in *html*."""
    pattern = rf'<{tag}\b[^>]*\bid="{re.escape(element_id)}"[^>]*>'
    match = re.search(pattern, html, re.S)
    assert match is not None, f"missing <{tag} id={element_id}>"
    attributes = dict(re.findall(r'([A-Za-z-]+)="([^"]*)"', match.group(0)))
    return attributes


async def test_the_shell_has_a_level_chip_with_aria_attributes(
    client: httpx.AsyncClient,
) -> None:
    page = await client.get("/")
    chip = _shell_attributes(page.text, "button", "level-btn")
    assert chip.get("aria-haspopup") == "menu"
    assert chip.get("aria-expanded") == "false"
    assert chip.get("aria-label") == "Chat level"
    body = re.search(
        r'<button\b[^>]*\bid="level-btn"[^>]*>(.*?)</button>', page.text, re.S
    )
    assert body is not None
    assert "P2" in body.group(1)
    menu = _shell_attributes(page.text, "div", "level-menu")
    assert menu.get("role") == "menu"
    assert re.search(r'<div\b[^>]*\bid="level-menu"[^>]*\bhidden\b', page.text)


async def test_the_shell_loads_the_same_assets_as_before(
    client: httpx.AsyncClient,
) -> None:
    page = await client.get("/")
    parser = ShellAssets()
    parser.feed(page.text)
    assert set(parser.paths) == {
        "/assets/icon_small.png",
        "/assets/melo.png",
        "/assets/theme-init.js",
        "/assets/workspace.css",
        "/assets/workspace.js",
    }


async def test_settings_module_is_served_as_javascript(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/assets/settings.mjs")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "setupSettings" in response.text


def test_quickstart_describes_available_workspace_controls() -> None:
    quickstart = Path(__file__).parents[2] / "docs" / "start" / "quickstart.md"
    text = quickstart.read_text(encoding="utf-8")
    assert (
        "Approval cards, connection settings and the session details drawer are" in text
    )
    assert (
        "The approval cards and the settings dialog arrive in a later update"
        not in text
    )


async def test_theme_init_is_served_as_javascript(client: httpx.AsyncClient) -> None:
    response = await client.get("/assets/theme-init.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "localmelo-console.theme" in response.text


async def test_drawer_module_is_served_as_javascript(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/assets/drawer.mjs")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "setupDrawer" in response.text


async def test_the_shell_has_a_machine_state_span_with_an_accessible_name(
    client: httpx.AsyncClient,
) -> None:
    page = await client.get("/")
    span = _shell_attributes(page.text, "span", "machine-state")
    assert span.get("role") == "status"
    assert span.get("aria-label")
    assert re.search(r'<span\b[^>]*\bid="machine-state"[^>]*\bhidden\b', page.text)
    assert re.search(
        r'<span\b[^>]*\bid="machine-state"[^>]*\baria-live="polite"', page.text
    )


async def test_the_shell_keeps_the_machine_line_between_the_title_and_the_level(
    client: httpx.AsyncClient,
) -> None:
    page = await client.get("/")
    header = re.search(r"<header\b.*?</header>", page.text, re.S)
    assert header is not None
    text = header.group(0)
    assert text.index('id="machine-state"') > text.index('id="header-model"')
    assert text.index('id="machine-state"') < text.index('id="level-btn"')


async def test_tasks_module_is_served_and_the_shell_links_the_task_view(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/assets/tasks.mjs")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "setupTasks" in response.text
    page = await client.get("/")
    assert 'href="#/tasks" id="tasks-link"' in page.text
    assert re.search(r'<section\b[^>]*\bid="task-panel"[^>]*\bhidden\b', page.text)


async def test_task_live_module_is_served(client: httpx.AsyncClient) -> None:
    response = await client.get("/assets/task-live.mjs")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "taskLive" in response.text


async def test_status_row_module_is_served_and_the_shell_has_its_row(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/assets/status-row.mjs")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "setupStatusRow" in response.text
    page = await client.get("/")
    assert re.search(r'<div\b[^>]*\bid="status-row"[^>]*\bhidden\b', page.text)


async def test_running_module_is_served_and_the_shell_has_its_chip(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/assets/running.mjs")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "setupRunning" in response.text
    page = await client.get("/")
    assert re.search(r'<div\b[^>]*\bid="running-root"[^>]*\bhidden\b', page.text)
    assert re.search(r'<div\b[^>]*\bid="running-menu"[^>]*\bhidden\b', page.text)
    assert 'id="running-btn"' in page.text


def _stylesheet() -> str:
    css = (webapp._STATIC / "workspace.css").read_text(encoding="utf-8")
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _outside_tokens() -> str:
    """The stylesheet without its custom-property definitions: the token block."""
    return re.sub(r"--[a-z0-9-]+\s*:[^;]+;", "", _stylesheet())


def _tokens(scale: str) -> dict[str, str]:
    """The px tokens of a scale ("text" or "radius"): name to value."""
    return dict(re.findall(rf"--({scale}-[a-z]+):\s*(\d+px)", _stylesheet()))


def _off_scale(scale: str, values: list[str]) -> set[str]:
    """The values that are neither a token of *scale* nor its px value."""
    tokens = _tokens(scale)
    allowed = {f"var(--{name})" for name in tokens} | set(tokens.values())
    return {value.strip() for value in values} - allowed - {"inherit"}


def test_font_sizes_come_from_the_scale_and_none_is_under_12px() -> None:
    css = _outside_tokens()
    sizes = re.findall(r"font-size:\s*([^;}]+)", css)
    for font in re.findall(r"\bfont:\s*([^;}]+)", css):
        sizes.append(font.split()[0].split("/")[0])
    assert sizes and not _off_scale("text", sizes), _off_scale("text", sizes)
    assert min(int(px[:-2]) for px in _tokens("text").values()) >= 12


def test_radii_come_from_the_four_tokens() -> None:
    radii = re.findall(r"border-radius:\s*([^;}]+)", _outside_tokens())
    assert len(_tokens("radius")) == 4
    assert radii and not _off_scale("radius", radii), _off_scale("radius", radii)


def test_no_colour_literal_sits_outside_the_token_block() -> None:
    literal = r"(?<![\w-])#[0-9a-fA-F]{3,8}(?![\w-])|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch)\("
    found = re.findall(literal, _outside_tokens())
    assert not found, f"colour literals outside the token block: {found}"


def test_every_variable_the_stylesheet_uses_is_defined() -> None:
    css = _stylesheet()
    used = set(re.findall(r"var\((--[a-z0-9-]+)", css))
    undefined = used - set(re.findall(r"(--[a-z0-9-]+)\s*:", css))
    assert used and not undefined, f"undefined variables: {sorted(undefined)}"


def test_muted_text_has_a_rule_and_undone_turns_do_not_fade() -> None:
    css = _stylesheet()
    assert re.search(r"(?m)^\.muted\s*\{[^}]*\bcolor:\s*var\(--ink-3\)", css)
    undone = re.search(r"\.turn\.undone\s*\{([^}]*)\}", css)
    assert undone and "opacity" not in undone.group(1)
    ring = r"\.composer:focus-within\s*\{[^}]*outline:\s*2px solid var\(--accent\)"
    assert re.search(ring, css)


def test_shell_buttons_have_no_native_title_tooltips() -> None:
    page = (webapp._STATIC / "index.html").read_text(encoding="utf-8")
    assert not re.findall(r"<button\b[^>]*\btitle=", page)


async def test_setup_card_module_is_served_and_the_shell_has_its_card(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/assets/setup-card.mjs")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "setupModelCard" in response.text
    page = await client.get("/")
    assert re.search(r'<section\b[^>]*\bid="setup-card"[^>]*\bhidden\b', page.text)
    assert "setup-btn" not in page.text and "POSSIBILITY" not in page.text


def test_a_done_status_line_draws_in_the_success_colour() -> None:
    css = _stylesheet()
    assert re.search(
        r'\.turn-status\[data-severity="ok"\]\s*\{[^}]*color:\s*var\(--ok\)', css
    )


def test_the_selected_chat_row_has_a_cue_that_is_not_only_a_colour() -> None:
    css = _stylesheet()
    active = re.search(r"\.session-item\.active\s*\{([^}]*)\}", css)
    assert active and "box-shadow: inset 3px 0 0 var(--accent)" in active.group(1)
    assert re.search(
        r"\.session-item\.active \.session-title\s*\{[^}]*font-weight:\s*600", css
    )


def test_red_marks_an_error_and_nothing_milder_in_a_status_card() -> None:
    css = _stylesheet()
    for name in ("failure-error", "failure-hint", "approval-warning"):
        rule = re.search(rf"(?m)^\.{name}\s*\{{([^}}]*)\}}", css)
        assert rule and "--error" not in rule.group(1), name
    assert re.search(r'\.failure-card\[data-severity="error"\] \.failure-error', css)
    assert re.search(r'\.status-mark\[data-severity="error"\]\s*\{[^}]*--error', css)


def test_the_task_view_buttons_skip_the_answer_classes() -> None:
    # `.task-panel button` (0,1,1) would out-rank `.primary-button`, `.outline-button` and
    # `.quiet-button` (0,1,0), so every rule that paints a task button must leave those out.
    answers = ("primary-button", "outline-button", "quiet-button")
    paint = re.compile(r"(?:^|;)\s*(?:color|background[\w-]*|border[\w-]*)\s*:")
    for group, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _stylesheet()):
        if not paint.search(body):
            continue
        for selector in re.split(r",\s*(?![^()]*\))", group):
            selector = selector.strip()
            if re.match(r"\.task-panel button(?![\w-])", selector):
                missing = [name for name in answers if f".{name}" not in selector]
                assert not missing, f"{selector} restyles {missing} in the Tasks view"


async def test_the_shell_words_the_approval_bar(client: httpx.AsyncClient) -> None:
    page = (await client.get("/")).text
    bar = re.search(r'<div id="approval-bar".*?</div>', page, re.S)
    assert bar is not None
    assert "melo is waiting for your answer" in bar.group(0)
    go = r'<button\b[^>]*\bid="approval-go"[^>]*>\s*Review\s*</button>'
    assert re.search(go, bar.group(0))


async def test_the_shell_offers_try_again_when_the_gateway_is_unreachable(
    client: httpx.AsyncClient,
) -> None:
    page = (await client.get("/")).text
    banner = re.search(r'<div\b[^>]*\bid="offline-banner"[^>]*>.*?</div>', page, re.S)
    assert banner is not None and "hidden" in banner.group(0).split(">")[0]
    assert "Could not reach the gateway" in banner.group(0)
    retry = r'<button\b[^>]*\bid="offline-retry"[^>]*>\s*Try again\s*</button>'
    assert re.search(retry, banner.group(0))


async def test_the_shell_has_every_id_default_mode_hides(
    client: httpx.AsyncClient,
) -> None:
    # Default mode hides these from script, so each must be in the page to be hidden.
    page = (await client.get("/")).text
    for element_id in (
        "level-btn",
        "details-trigger",
        "machine-state",
        "workers-link",
        "tasks-link",
        "section-nav",
        "composer-note",
        "lock-banner",
        "settings-label",
        "settings-tabs",
        "settings-head-model",
        "settings-meta",
        "settings-theme-legend",
    ):
        assert f'id="{element_id}"' in page, element_id
    assert re.search(
        r'<div\b[^>]*\bid="lock-banner"[^>]*\brole="status"[^>]*\bhidden\b', page
    )


def test_touch_screens_and_phones_get_44px_targets() -> None:
    css = _stylesheet()
    block = re.search(
        r"@media \(pointer: coarse\), \(max-width: 700px\)\s*\{(.*?)\n\}", css, re.S
    )
    assert block is not None
    for name in ("send-button", "stop-button", "icon-button"):
        assert re.search(rf"\.{name}\b[^{{}}]*\{{[^}}]*\b44px", block.group(1)), name
    assert re.search(r'\.level-item\[aria-checked="true"\]', css)
