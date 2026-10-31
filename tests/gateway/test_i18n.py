"""The console's words in English and Simplified Chinese: the catalogs and the shared rules.

``support/gateway/static/i18n/<language>.json`` is read by the page (``i18n.mjs``), the server
(``support/gateway/i18n.py``) and the macOS app (``L10n.swift``, through the strings the build
writes). These tests keep the two catalogs in step, keep every key the code names in them,
keep the Python rules equal to the shared cases that ``test_i18n.mjs`` runs in JavaScript, and
fail on a new English literal in a file whose words come from the catalog.
"""

from __future__ import annotations

import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette")

from localmelo.support.gateway.i18n import (  # noqa: E402
    accept_languages,
    catalog,
    format_entry,
    page_language,
    pick_language,
    request_language,
    t,
    t_in,
    using,
)

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "support/gateway/static"
CONSOLE = ROOT / "support/gateway/console"
# Words written by modules the terminal runs too, which need no gateway.
USAGE = ROOT / "support/usage_snapshot.py"
DISPLAY = ROOT / "support/sessions/display.py"
APP = ROOT / "macos/LocalMeloApp"
CASES = json.loads((ROOT / "tests/fixtures/i18n_cases.json").read_text("utf-8"))
_PARAM = re.compile(r"\{(\w+)\}")

# The files whose words come from the catalog: a new English literal in one fails the scan.
CONVERTED_JS = (
    "blocked.mjs",
    "drawer.mjs",
    "folder.mjs",
    "levels.mjs",
    "main-chat.mjs",
    "memory-tab.mjs",
    "memory.mjs",
    "mode.mjs",
    "panel.mjs",
    "pictures.mjs",
    "running.mjs",
    "scheduled-tab.mjs",
    "settings.mjs",
    "setup-card.mjs",
    "status-row.mjs",
    "task-live.mjs",
    "tasks.mjs",
    "turns.mjs",
    "worker-history.mjs",
    "workspace.js",
)
CONVERTED_PY = (
    "activity.py",
    "cards.py",
    "chores.py",
    "front.py",
    "registry.py",
    "routes_chores.py",
    "routes_details.py",
    "routes_memory.py",
    "routes_page.py",
    "routes_pictures.py",
    "routes_sessions.py",
    "routes_turns.py",
)

# What stays English on purpose. JavaScript: key names, a code the scheduler sends, and the
# message of an internal error that is never shown.
JS_ENGLISH = {
    "Escape",
    "Enter",
    "Tab",
    "Reservation",
    "melo did not finish an answer",
    "cannot reach the model server at",
    "noopener noreferrer",
    # Class lists of one word each.
    "msg user",
    "msg assistant",
    "blinks face",
    # How the gateway answers an approval, and how a card stored before it kept its words began.
    "allowed once",
    "allowed for this chat",
    "allowed automatically",
    "no answer",
    "Allowed",
    "Denied",
    "Waiting for your approval",
    # The reason auto mode records for a later call of a turn it stopped (#339).
    "the turn stopped",
}
# Names read the same in every language: a literal of names alone is not English.
NAMES = {"localmelo", "melo", "Melo", "Python"}
# Units beside a number stay as the terminal writes them: `9s`, `2-3 min`, `3.4k`.
UNITS = {"s", "m", "h", "k", "M", "min", "KB", "MB", "GB", "TB"}
# Python: API validation messages for clients that send a malformed request; the page
# never sends one.
PY_ENGLISH = {
    "f'{do} does not apply to this line.'",
    "level: P0 to P5",
    "level: P1 to P4",
    "front_id must name a console chat",
    "Choose what to change.",
    "Worker filters require kind=worker or kind=terminal.",
    "as_task: tasks start from a chat",
    "reason: only a denial takes a reason",
}
# Swift: the app's log lines, the gateway's own command line, and names.
SWIFT_ENGLISH = {
    "The gateway was terminated by signal ",
    "The gateway exited (status ",
}


def forms(entry: Any) -> list[str]:
    return [entry] if isinstance(entry, str) else list(entry.values())


def params(entry: Any) -> set[str]:
    return {name for text in forms(entry) for name in _PARAM.findall(text)}


# ── The catalogs ──


def test_both_catalogs_have_every_key_with_the_same_parameters() -> None:
    en, zh = catalog("en"), catalog("zh-Hans")
    assert sorted(en) == sorted(zh)
    for key in en:
        assert params(en[key]) == params(zh[key]), key
        # Chinese joins its sentences with nothing; every other entry has words.
        if key != "common.sentenceGap":
            assert all(text.strip() for text in forms(zh[key])), key


def test_a_count_has_one_and_other_in_english_and_one_form_in_chinese() -> None:
    for key, entry in catalog("en").items():
        if isinstance(entry, dict):
            assert set(entry) == {"one", "other"}, key
            assert "n" in params(entry), key
            zh = catalog("zh-Hans")[key]
            assert isinstance(zh, str) or set(zh) == {"other"}, key


def test_the_catalog_files_are_sorted_json_with_a_last_newline() -> None:
    for name in ("en", "zh-Hans"):
        path = STATIC / "i18n" / f"{name}.json"
        data = json.loads(path.read_text("utf-8"))
        written = json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=2)
        assert path.read_text("utf-8") == written + "\n", name


# ── The rules both languages share (tests/fixtures/i18n_cases.json) ──


@pytest.mark.parametrize("case", CASES["format"], ids=lambda case: case["text"])
def test_python_fills_words_as_the_shared_cases_say(case: dict[str, Any]) -> None:
    assert format_entry(case["entry"], case["params"], case["language"]) == case["text"]


@pytest.mark.parametrize(
    "case", CASES["pick"], ids=lambda case: f"{case['setting']}-{case['preferred']}"
)
def test_python_picks_the_language_as_the_shared_cases_say(
    case: dict[str, Any],
) -> None:
    assert pick_language(case["setting"], case["preferred"]) == case["language"]


def test_a_key_falls_back_to_english_then_to_itself() -> None:
    assert t_in("zh-Hans", "running.more", n=2) == "还有 2 个"
    assert t_in("fr", "running.more", n=2) == "and 2 more"
    assert t_in("zh-Hans", "no.such.key") == "no.such.key"
    assert t("running.more", n=2) == "and 2 more"
    with using("zh-Hans"):
        assert t("running.more", n=2) == "还有 2 个"
    assert t("running.more", n=2) == "and 2 more"


def test_accept_language_is_read_by_quality() -> None:
    assert accept_languages("en-US;q=0.5, zh-CN, *;q=0.1, fr;q=0") == ["zh-CN", "en-US"]
    assert accept_languages("de;q=oops, en") == ["en"]
    assert accept_languages("") == []


def test_a_request_asks_by_header_or_cookie_and_anything_else_is_english() -> None:
    cookie = (b"cookie", b"melo_session_1=x; localmelo_language=zh-Hans")
    assert request_language([cookie]) == "zh-Hans"
    assert request_language([(b"x-localmelo-language", b"zh-Hans")]) == "zh-Hans"
    assert request_language([(b"X-LocalMelo-Language", b"en"), cookie]) == "en"
    assert request_language([(b"x-localmelo-language", b"fr")]) == "en"
    assert request_language([(b"accept-language", b"zh-CN")]) == "en"
    assert request_language([(b"cookie", b"bad=;;=\x7f")]) == "en"
    assert request_language([]) == "en"


def test_the_page_guess_takes_the_setting_then_the_cookie_then_the_browser() -> None:
    chinese = [(b"accept-language", b"zh-CN,zh;q=0.9")]
    assert page_language("auto", chinese) == "zh-Hans"
    assert page_language("en", chinese) == "en"
    assert (
        page_language("auto", [*chinese, (b"cookie", b"localmelo_language=en")]) == "en"
    )
    assert page_language("zh-Hans", [(b"accept-language", b"en")]) == "zh-Hans"
    assert page_language("auto", []) == "en"


# ── Every key the code names is in the catalog, and every key is named ──

_JS_KEY = re.compile(r"""\bt\(\s*(["'`])([A-Za-z0-9_.${}]+?)\1""")
_PY_KEY = re.compile(r"""\b(?:t\(|said\(|t_in\("[\w-]+",\s*)f?"([A-Za-z0-9_.{}]+)\"""")
# A key built in a template or an f-string: the part before the first `${` or `{` is its stem.
_JS_STEM = re.compile(r"`([a-z][A-Za-z0-9_]*\.[A-Za-z0-9_.]*)\$\{")
_PY_STEM = re.compile(r"""\bf"([a-z][A-Za-z]*(?:\.[A-Za-z]+)*\.)\{\w+\}[A-Za-z]*\"""")
_QUOTED_KEY = re.compile(r"""["']([a-z][A-Za-z]*(?:\.[A-Za-z]+)+)["']""")
_WORDS_KEY = re.compile(r"""_words\(\s*f?"([A-Za-z0-9_.{}]+)\"""")
_HTML_KEY = re.compile(r"""data-i18n(?:-[a-z-]+)?="([^"]+)\"""")
_SWIFT_T = re.compile(r'L10n\.t\("([^"]+)",\s*"((?:[^"\\]|\\.)*)"')
_SWIFT_PLURAL = re.compile(
    r'L10n\.plural\("([^"]+)",[^,]+,\s*one:\s*"((?:[^"\\]|\\.)*)",\s*other:\s*"((?:[^"\\]|\\.)*)"'
)


def named_keys() -> tuple[set[str], set[str], set[str]]:
    """The keys the code names, the stems of the keys it builds (``turns.status.``), and every key-like string."""
    found: set[str] = set()
    templates: set[str] = set()
    for path in [*STATIC.glob("*.mjs"), STATIC / "workspace.js"]:
        found |= {key for _, key in _JS_KEY.findall(path.read_text("utf-8"))}
        templates |= set(_JS_STEM.findall(path.read_text("utf-8")))
    found |= set(_HTML_KEY.findall((STATIC / "index.html").read_text("utf-8")))
    for path in [
        *CONSOLE.glob("*.py"),
        *ROOT.glob("support/gateway/*.py"),
        USAGE,
        DISPLAY,
    ]:
        text = path.read_text("utf-8")
        found |= set(_PY_KEY.findall(text)) | set(_WORDS_KEY.findall(text))
        templates |= set(_PY_STEM.findall(text))
    for path in APP.glob("*.swift"):
        text = path.read_text("utf-8")
        found |= {key for key, _ in _SWIFT_T.findall(text)}
        found |= {key for key, _, _ in _SWIFT_PLURAL.findall(text)}
    quoted: set[str] = set()
    for path in [
        *STATIC.glob("*.mjs"),
        STATIC / "workspace.js",
        *CONSOLE.glob("*.py"),
        USAGE,
        DISPLAY,
    ]:
        quoted |= set(_QUOTED_KEY.findall(path.read_text("utf-8")))
    built = {key for key in found if "{" in key or "$" in key}
    stems = {re.split(r"\$\{|\{", key)[0] for key in built} | templates
    return found - built, stems, quoted


def test_every_key_the_code_names_is_in_the_catalog() -> None:
    keys, stems, _ = named_keys()
    english = catalog("en")
    assert sorted(key for key in keys if key not in english) == []
    for stem in stems:
        assert any(key.startswith(stem) for key in english), stem


def test_every_catalog_key_is_named_by_the_code() -> None:
    keys, stems, quoted = named_keys()
    # Keys named through a table of names the code passes on (`_ANSWERS`, `_MARKER_WORDS`).
    tabled = {"activity.marker.", "cards.answer."}
    unused = [
        key
        for key in catalog("en")
        if key not in keys | quoted and not key.startswith((*stems, *tabled))
    ]
    assert unused == []


def test_the_app_says_in_english_what_the_catalog_says() -> None:
    english = catalog("en")
    named = set()
    for path in APP.glob("*.swift"):
        text = path.read_text("utf-8")
        for key, words in _SWIFT_T.findall(text):
            named.add(key)
            assert english.get(key) == words.replace('\\"', '"'), key
        for key, one, other in _SWIFT_PLURAL.findall(text):
            named.add(key)
            assert english.get(key) == {"one": one, "other": other}, key
    assert sorted(key for key in english if key.startswith("app.")) == sorted(named)


# ── No new English in a converted file ──


def js_literals(source: str) -> list[str]:
    """The string and template literals of a module, outside comments; ``${…}`` reads as ``X``."""
    found: list[str] = []
    i, n, last = 0, len(source), ""
    while i < n:
        ch = source[i]
        if source.startswith("//", i):
            i = source.find("\n", i)
            i = n if i < 0 else i
        elif source.startswith("/*", i):
            i = source.index("*/", i) + 2
        elif ch in "'\"":
            j = i + 1
            while source[j] != ch:
                j += 2 if source[j] == "\\" else 1
            found.append(source[i + 1 : j])
            i, last = j + 1, "x"
        elif ch == "`":
            j, depth, text = i + 1, 0, []
            while depth or source[j] != "`":
                if source[j] == "\\":
                    j += 2
                    continue
                if not depth and source.startswith("${", j):
                    depth, j = 1, j + 2
                    text.append("X")
                    continue
                if depth:
                    depth += {"{": 1, "}": -1}.get(source[j], 0)
                    if source[j] in "'\"":  # a string inside ${…}
                        quote, j = source[j], j + 1
                        while source[j] != quote:
                            j += 2 if source[j] == "\\" else 1
                else:
                    text.append(source[j])
                j += 1
            found.append("".join(text))
            i, last = j + 1, "x"
        elif ch == "/" and (
            not last
            or last in "(,=:[!&|?{};+-*%<>~^"
            or source[max(0, i - 7) : i].rstrip().endswith("return")
        ):
            j = i + 1
            klass = False
            while klass or source[j] != "/":
                if source[j] == "\\":
                    j += 1
                elif source[j] == "[":
                    klass = True
                elif source[j] == "]":
                    klass = False
                j += 1
            i, last = j + 1, "x"
        else:
            if not ch.isspace():
                last = ch
            i += 1
    return found


def english(text: str) -> bool:
    """Whether a literal reads as words a person sees, not code, a key, a path or a class list."""
    words = text.strip()
    if not re.search(r"[A-Za-z]{2}", words) or words in JS_ENGLISH:
        return False
    if set(re.findall(r"[A-Za-z]+", words.replace("X", " "))) <= UNITS:
        return False
    if re.search(r"[/#=\[\]<>]", words) or re.fullmatch(
        r"[a-z][\w-]*(\.[\w-]*)+", words
    ):
        return False  # a path, a selector, markup or a catalog key
    tokens = [token for token in words.split() if token not in NAMES]
    if not tokens:
        return False
    words = " ".join(tokens)
    if all(re.fullmatch(r"[a-z0-9_-]+", token) for token in tokens) and any(
        "-" in token for token in tokens
    ):
        return False  # a list of classes
    if len(tokens) == 1:
        return bool(re.fullmatch(r"[A-Z][a-z]+", words))
    return any(re.fullmatch(r"[A-Za-z][a-z']+", token) for token in tokens)


@pytest.mark.parametrize("name", CONVERTED_JS)
def test_a_converted_module_shows_no_english_of_its_own(name: str) -> None:
    source = (STATIC / name).read_text("utf-8")
    assert [text for text in js_literals(source) if english(text)] == []


def test_the_scan_finds_english_and_passes_code() -> None:
    source = r"""
    // A comment with Words in it.
    const a = 'Forget it', b = `${n} running`, c = "Default";
    const d = 'outline-button danger', e = `/sessions/${id}/turns`, f = t('memory.used');
    const g = /["'](Words here)/g, h = "button, [href]", k = 'click';
    """
    assert [text for text in js_literals(source) if english(text)] == [
        "Forget it",
        "X running",
        "Default",
    ]


def py_messages(path: Path) -> list[str]:
    """The English literals a converted file passes as an API error's message."""
    found = []
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if not (
            isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ApiError"
        ):
            continue
        message = node.args[2] if len(node.args) > 2 else None
        if isinstance(message, ast.Constant) and isinstance(message.value, str):
            found.append(message.value)
        elif isinstance(message, ast.JoinedStr):
            found.append(ast.unparse(message))
    return found


@pytest.mark.parametrize("name", CONVERTED_PY)
def test_a_converted_route_says_its_errors_through_the_catalog(name: str) -> None:
    left = [text for text in py_messages(CONSOLE / name) if text not in PY_ENGLISH]
    assert [
        text
        for text in left
        if not text.startswith(("f'{name} must", 'f"kind must', "f'reason: at most"))
    ] == []


def test_the_app_shows_no_english_outside_the_catalog() -> None:
    literal = re.compile(r'"((?:[^"\\]|\\.)*)"')
    left = []
    for path in sorted(APP.glob("*.swift")):
        for line in path.read_text("utf-8").splitlines():
            code = line.split("//", 1)[0] if not line.strip().startswith("//") else ""
            code = _SWIFT_T.sub("", _SWIFT_PLURAL.sub("", code))
            for text in literal.findall(code):
                plain = re.sub(r"\\\(.*?\)", " ", text)
                if any(plain.startswith(item) for item in SWIFT_ENGLISH):
                    continue
                if re.search(r"[A-Za-z]{2,} [A-Za-z]{2,}", plain) and not re.search(
                    r"[=;/]|import |\.[a-z]", plain
                ):
                    left.append(f"{path.name}: {text}")
    assert left == []


def test_the_level_meanings_in_english_are_the_gateway_s() -> None:
    from localmelo.support.gateway.front_rules import LEVEL_MEANINGS

    english = catalog("en")
    assert [english[f"levels.p{n}"] for n in range(6)] == list(LEVEL_MEANINGS)


# ── The page's markup ──

# Words that read the same in every language: names, codes and each language's own name.
HTML_KEPT = {"localmelo", "Melo", "LOCAL", "P2", "English", "简体中文"}


class _PageWords(HTMLParser):
    """The words of the page that a person reads, without a catalog key."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, dict[str, str | None]]] = []
        self.left: list[str] = []

    def _attributes(self, attrs: dict[str, str | None]) -> None:
        for name in ("aria-label", "placeholder", "title", "data-prompt"):
            value = attrs.get(name) or ""
            words = re.search(r"[A-Za-z]{2}", value) and value not in HTML_KEPT
            # A path or an address in a placeholder is an example, the same in every language.
            if (
                words
                and f"data-i18n-{name}" not in attrs
                and not re.match(r"(~|https?:)/", value)
            ):
                self.left.append(f"{name}={value}")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        found = dict(attrs)
        self._attributes(found)
        if tag not in ("img", "input", "meta", "link", "br"):
            self.stack.append((tag, found))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._attributes(dict(attrs))

    def handle_endtag(self, tag: str) -> None:
        while self.stack and self.stack.pop()[0] != tag:
            pass

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not re.search(r"[A-Za-z]{2}", text) or text in HTML_KEPT:
            return
        tags = {tag for tag, _ in self.stack}
        keyed = any("data-i18n" in attrs for _, attrs in self.stack)
        # Code, a key and a setting's name stay as they are written.
        code = tags & {"title", "script", "style", "code", "kbd"} or any(
            "st-key" in (attrs.get("class") or "") for _, attrs in self.stack
        )
        if not keyed and not code:
            self.left.append(text)


def test_every_word_of_the_page_has_a_key() -> None:
    page = _PageWords()
    page.feed((STATIC / "index.html").read_text("utf-8"))
    assert page.left == []
