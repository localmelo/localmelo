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
from localmelo.support.providers._http import UNREACHABLE_PREFIX  # noqa: E402


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


class ImageTags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img":
            self.images.append(dict(attrs))


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
        elif path.endswith(".webp"):
            assert media_type == "image/webp"
            assert response.content[8:12] == b"WEBP"
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


async def test_melo_s_pictures_in_the_shell_are_sharp_and_say_only_what_words_do_not(
    client: httpx.AsyncClient,
) -> None:
    page = await client.get("/")
    tags = ImageTags()
    tags.feed(page.text)
    shown = {
        str(image["src"]): image
        for image in tags.images
        if str(image["src"]).startswith("/assets/akita/")
    }
    assert set(shown) == {
        "/assets/akita/face-idle.png",  # the brand, in the sidebar and on the lock screen
        "/assets/akita/face-waiting.png",  # the approval bar
    }
    for path, image in shown.items():
        data = (await client.get(path)).content
        wide, tall = struct.unpack(">II", data[16:24])
        width, height = int(str(image["width"])), int(str(image["height"]))
        # A 2x file shown at half its size or less is sharp on a Retina screen, and in the right shape.
        assert wide >= 2 * width and wide * height == tall * width, path
        # Words already say it where the face sits.
        assert image["alt"] == "", path
    assert shown["/assets/akita/face-waiting.png"]["width"] == "24"
    # The empty chat's Melo is the sitting dog the moves wake up into, at 2x, and names itself.
    (mascot,) = [i for i in tags.images if i.get("class") == "welcome-mascot"]
    assert mascot["src"] == "/assets/moves/startup-poster.webp"
    assert (mascot["width"], mascot["height"]) == ("128", "128")
    assert mascot["alt"]
    poster = await client.get(str(mascot["src"]))
    assert poster.headers["content-type"] == "image/webp"
    assert webp_size(poster.content) == (256, 256)
    assert "melo is waiting for your answer" in page.text
    # The tab icon is the idle face at the three sizes a browser asks for.
    icons = IconLinks()
    icons.feed(page.text)
    assert [(i["sizes"], i["href"]) for i in icons.icons] == [
        ("16x16", "/assets/akita/favicon-16.png"),
        ("32x32", "/assets/akita/favicon-32.png"),
        ("64x64", "/assets/akita/favicon-64.png"),
    ]


def test_the_faces_and_the_portrait_are_not_clipped_to_a_rounded_tile() -> None:
    # The pictures are cut-outs with ears and a tail at their corners: a radius would trim them.
    for group, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _stylesheet()):
        if re.search(r"\.(brand-icon|welcome-mascot|approval-note)\b", group):
            assert "border-radius" not in body, group.strip()


def _blink_timeline() -> list[tuple[float, float]]:
    """The melo-blink keyframes as (seconds into the 11 s loop, opacity), in order."""
    body = re.search(r"@keyframes melo-blink\s*\{(.*?)\n\}", _stylesheet(), re.S)
    assert body, "the blink keyframes"
    points = []
    for stops, opacity in re.findall(
        r"([\d.%, ]+)\{\s*opacity:\s*([\d.]+);\s*\}", body.group(1)
    ):
        for stop in stops.split(","):
            points.append((float(stop.strip().rstrip("%")) * 11 / 100, float(opacity)))
    return sorted(points)


def test_the_blink_is_css_only_two_blinks_in_eleven_seconds_and_still_under_reduced_motion() -> (
    None
):
    css = _stylesheet()
    # The patch is the background of the wrapper's ::after: clear at rest, with one animation over 11 s, for ever.
    rule = re.search(r"\.blinks::after\s*\{([^}]*)\}", css)
    assert rule and "opacity: 0;" in rule.group(1)
    assert re.search(r"animation:\s*melo-blink 11s linear infinite", rule.group(1))
    for picture in ("sit-idle-blink", "face-idle-blink"):
        assert f"url(/assets/akita/{picture}.png)" in css
        assert (webapp._STATIC / "akita" / f"{picture}.png").is_file()
    # Two blinks, at 4.2 s and 9.6 s: 60 ms in, 30 ms held, 70 ms out (the pet's own timing); clear in between.
    timeline = _blink_timeline()
    assert timeline[0] == (0.0, 0.0) and timeline[-1] == (11.0, 0.0)
    for start in (4.2, 9.6):
        edges = [
            (round(s - start, 2), o)
            for s, o in timeline
            if start - 0.01 <= s <= start + 0.17
        ]
        assert edges == [(0.0, 0.0), (0.06, 1.0), (0.09, 1.0), (0.16, 0.0)], start
    assert sum(1 for _, o in timeline if o == 1.0) == 4  # no third blink
    # Reduce Motion removes every animation: the patch stays clear, which is the still picture.
    reduced = re.search(
        r"@media \(prefers-reduced-motion: reduce\)\s*\{(.*?)\n\}", css, re.S
    )
    assert reduced and "animation: none !important" in reduced.group(1)
    assert "*::after" in reduced.group(1)
    # And no script runs the blink: the lines that mention it set up markup and nothing else.
    script = (webapp._STATIC / "workspace.js").read_text(encoding="utf-8")
    for line in (line for line in script.splitlines() if "blink" in line):
        assert not re.search(
            r"setTimeout|setInterval|requestAnimationFrame|animate\(", line
        )


async def test_the_empty_chat_shows_the_new_sitting_dog_without_the_old_blink_patch(
    client: httpx.AsyncClient,
) -> None:
    page = (await client.get("/")).text
    # The eyes-closed patch was cut from the old sitting picture: it would not lie on the new one.
    assert re.search(
        r'<img\s+class="welcome-mascot"\s+src="/assets/moves/startup-poster.webp"', page
    )
    assert "blinks portrait" not in page
    assert ".blinks.portrait" not in _stylesheet()


async def test_the_moves_are_served_with_their_player(
    client: httpx.AsyncClient,
) -> None:
    script = await client.get("/assets/moves.mjs")
    assert script.status_code == 200 and "javascript" in script.headers["content-type"]
    manifest = (await client.get("/assets/moves/manifest.json")).json()
    for entry in manifest["states"].values():
        sheet = await client.get(f"/assets/moves/{entry['file']}")
        assert sheet.headers["content-type"] == "image/webp", entry["file"]
    # The panel draws them: its portrait holds the canvas.
    assert "startMoves" in (await client.get("/assets/panel.mjs")).text


async def test_chat_url_placeholder_matches_the_omlx_hint(
    client: httpx.AsyncClient,
) -> None:
    page = await client.get("/")
    parser = InputPlaceholders()
    parser.feed(page.text)
    # oMLX's server listens on port 8000 by default. The plain hint beside the field shows that address
    # as its example, and the developer's wording (the catalog's, for settings.mjs) still names oMLX.
    placeholder = parser.placeholders["s-chat-url"]
    assert placeholder == "http://127.0.0.1:8000/v1"
    assert f"such as {placeholder}." in page.text
    # A screen reader reads that instruction with the field.
    assert (
        _shell_attributes(page.text, "input", "s-chat-url")["aria-describedby"]
        == "s-chat-url-hint"
    )
    settings = await client.get("/assets/settings.mjs")
    assert "settings.${set}.${name}" in settings.text
    words = (await client.get("/assets/i18n/en.json")).json()
    assert (
        "oMLX supports the OpenAI-compatible adapter."
        in words["settings.developer.urlHint"]
    )


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
        "/assets/akita/face-idle.png",
        "/assets/akita/face-waiting.png",
        "/assets/akita/favicon-16.png",
        "/assets/akita/favicon-32.png",
        "/assets/akita/favicon-64.png",
        "/assets/moves/startup-poster.webp",
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


def test_the_console_reads_an_unreachable_server_as_the_provider_words_it() -> None:
    turns = (webapp._STATIC / "turns.mjs").read_text(encoding="utf-8")
    assert f'const UNREACHABLE_PREFIX = "{UNREACHABLE_PREFIX}";' in turns


def test_quickstart_describes_available_workspace_controls() -> None:
    quickstart = Path(__file__).parents[2] / "docs" / "start" / "quickstart.md"
    text = quickstart.read_text(encoding="utf-8")
    assert "Approval cards, settings and the session details drawer are" in text
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


async def test_tasks_module_is_served_and_the_panel_holds_the_task_view(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/assets/tasks.mjs")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "setupTasks" in response.text
    page = await client.get("/")
    # A task opens in the panel beside the chat; the sidebar has no Tasks link of its own.
    assert "tasks-link" not in page.text
    assert 'href="#/tasks"' not in page.text
    panel = re.search(r'<aside\b[^>]*\bid="panel"[^>]*>.*?</aside>', page.text, re.S)
    assert panel is not None
    assert re.search(r'<section\b[^>]*\bid="task-panel"[^>]*\bhidden\b', panel.group(0))


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


async def test_pictures_module_is_served_and_the_shell_has_its_light_box(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/assets/pictures.mjs")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "picturesNode" in response.text
    page = await client.get("/")
    assert re.search(r'<dialog\b[^>]*\bid="picture-box"', page.text)


def webp_size(data: bytes) -> tuple[int, int]:
    """Width and height of a WebP file, from its first chunk (VP8X, VP8L or VP8)."""
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    kind = data[12:16]
    if kind == b"VP8X":
        size = data[24:30]
        return (
            1 + int.from_bytes(size[:3], "little"),
            1 + int.from_bytes(size[3:], "little"),
        )
    if kind == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return 1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF)
    assert kind == b"VP8 "
    return (
        int.from_bytes(data[26:28], "little") & 0x3FFF,
        int.from_bytes(data[28:30], "little") & 0x3FFF,
    )


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


async def test_import_card_module_is_served_and_the_shell_has_its_hidden_card(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/assets/import-card.mjs")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "setupImportCard" in response.text and "importSection" in response.text
    page = (await client.get("/")).text
    card = re.search(r'<section\b[^>]*\bid="import-card"[^>]*>', page)
    assert card and " hidden" in card.group(
        0
    ), "the card is hidden until the page shows it"
    # In the conversation area, after the welcome and before the messages.
    assert (
        page.index('id="welcome"')
        < page.index('id="import-card"')
        < page.index('id="messages"')
    )


async def test_the_shell_has_the_folder_chip_and_the_starters_in_place_of_the_cards(
    client: httpx.AsyncClient,
) -> None:
    assert "setupFolderPicker" in (await client.get("/assets/folder.mjs")).text
    page = (await client.get("/")).text
    hidden = set(
        re.findall(r'<(?:div|button)\b[^>]*\bid="([\w-]+)"[^>]*\bhidden\b', page)
    )
    assert {"folder-picker", "folder-menu", "folder-path-row", "starters"} <= hidden
    starters = re.findall(r'data-prompt="([^"]+)"[^>]*>([^<]+)<', page)
    assert [label for _, label in starters] == [
        "Summarize the files in this folder",
        "Plan a small project with me",
        "Explain a piece of code",
    ]
    assert all(prompt.startswith(label) for prompt, label in starters)
    assert "suggestions" not in page and "suggestions" not in _stylesheet()


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


async def test_the_lock_screen_names_the_one_step(client: httpx.AsyncClient) -> None:
    page = (await client.get("/")).text
    lock = re.search(r'<main class="lock-screen".*?</main>', page, re.S)
    assert lock is not None
    assert ">Sign in to localmelo</h1>" in lock.group(0)
    assert (
        "In a terminal, run <code>melo console</code>. It opens this page"
        in lock.group(0)
    )


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


async def test_the_shell_pins_the_main_chat_and_drops_the_chats_switch(
    client: httpx.AsyncClient,
) -> None:
    page = (await client.get("/")).text
    assert "chats-link" not in page
    assert re.search(r'<div\b[^>]*\bid="main-chat-row"[^>]*\bhidden\b', page)
    parent = re.search(r'<button\b[^>]*\bid="header-parent"[^>]*>([^<]*)<', page)
    assert parent is not None and parent.group(1) == "Main chat"
    assert "hidden" in parent.group(0)
    workers = re.search(r'id="welcome-workers"[^>]*hidden><a href="#/workers"', page)
    assert workers is not None
    module = (await client.get("/assets/main-chat.mjs")).text
    assert "export function timeline" in module
    css = _stylesheet()
    # Phones and touch screens: the way back to the main chat is a 44 px target.
    block = re.search(
        r"@media \(max-width: 700px\), \(pointer: coarse\)\s*\{\s*\.header-parent[^}]*44px",
        css,
    )
    assert block is not None
