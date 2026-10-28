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
    webapp.mount(app)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
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
    assert chip.get("aria-haspopup") == "listbox"
    assert chip.get("aria-expanded") == "false"
    assert chip.get("aria-label") == "Chat level"
    body = re.search(
        r'<button\b[^>]*\bid="level-btn"[^>]*>(.*?)</button>', page.text, re.S
    )
    assert body is not None
    assert "P2" in body.group(1)
    menu = _shell_attributes(page.text, "div", "level-menu")
    assert menu.get("role") == "listbox"
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
    quickstart = Path(__file__).parents[2] / "docs" / "quickstart.md"
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
