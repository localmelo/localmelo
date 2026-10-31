"""L10n.swift: the app's words in the console's language, chosen as the page and the gateway choose it.

The Swift is compiled with a test main, as tests/test_macos_pet_policy.py does; no window is opened.
The language rule runs the shared cases of tests/fixtures/i18n_cases.json, and the Chinese words come
from a bundle holding the strings file the build writes (scripts/build_macos_app.py).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "macos/LocalMeloApp"
CASES = json.loads((ROOT / "tests/fixtures/i18n_cases.json").read_text("utf-8"))
spec = importlib.util.spec_from_file_location(
    "build_macos_app", ROOT / "scripts/build_macos_app.py"
)
assert spec and spec.loader
build_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_app)
darwin = pytest.mark.skipif(
    sys.platform != "darwin", reason="Swift AppKit host is macOS-only"
)


def swift(value: object) -> str:
    """A Swift literal of a JSON string, list or string dictionary."""
    if isinstance(value, dict):
        pairs = ", ".join(f"{swift(k)}: {swift(v)}" for k, v in value.items())
        return f"[{pairs}]" if pairs else "[:]"
    if isinstance(value, list):
        return "[" + ", ".join(swift(item) for item in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def run_swift(tmp_path: Path, source: str) -> str:
    main = tmp_path / "main.swift"
    main.write_text("import Foundation\n" + source, encoding="utf-8")
    binary = tmp_path / "test-l10n"
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "swiftc",
            str(APP / "L10n.swift"),
            str(APP / "GatewayPolicy.swift"),
            str(main),
            "-o",
            str(binary),
        ],
        env={
            **os.environ,
            "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
        },
        check=True,
        timeout=120,
    )
    done = subprocess.run([str(binary)], check=True, timeout=5, capture_output=True)
    return done.stdout.decode()


def test_the_strings_file_holds_the_app_keys_escaped() -> None:
    text = build_app.strings_file(
        {
            "app.a": 'Say "hi"\\now',
            "app.b": {"one": "{n} task", "other": "{n} tasks"},
            "running.more": "and {n} more",
        }
    )
    assert text == '"app.a" = "Say \\"hi\\"\\\\now";\n"app.b" = "{n} tasks";\n'


@darwin
def test_the_app_picks_its_language_and_reads_the_built_strings(tmp_path: Path) -> None:
    resources = tmp_path / "Bundle"
    assert build_app.write_localizations(resources) == ["en", "zh-Hans"]
    checks = [
        f"precondition(pickLanguage(setting: {swift(case['setting'])}, preferred: "
        f"{swift(case['preferred'])}) == {swift(case['language'])})"
        for case in CASES["pick"]
    ]
    checks += [
        f"precondition(L10n.fill({swift(case['entry'])}, {swift(case['params'])}) == "
        f"{swift(case['text'])})"
        for case in CASES["format"]
        if isinstance(case["entry"], str)
    ]
    out = run_swift(
        tmp_path,
        "\n".join(checks) + f"""
precondition(consoleLanguageSetting(nil) == "auto")
precondition(consoleLanguageSetting("[ui]\\nlanguage = \\"en\\"\\n") == "auto")
precondition(consoleLanguageSetting("[console]\\nlanguage = \\"zh-Hans\\"  # mine\\n[ui]\\nmode = \\"default\\"\\n") == "zh-Hans")
precondition(consoleLanguageSetting("[console]\\n  language='en'\\n") == "en")
precondition(languagesScript(["zh-Hans-CN", "en-US"]) == #"window.localmeloLanguages = ["zh-Hans-CN","en-US"];"#)

// English needs no strings: the words are the code's.
precondition(L10n.language == "en")
precondition(quitTitle(attached: false) == "Quit localmelo")
precondition(quitMessage(RunningCounts(chats: 1, tasks: 2, runs: 1, terminal: 0), statusKnown: true)
    == "Quitting stops what is running: 2 tasks and 1 chat. Not affected: 1 run.")
let bundle = Bundle(path: {swift(str(resources))})!
L10n.use("zh-Hans", bundle: bundle)
precondition(L10n.language == "zh-Hans")
precondition(quitTitle(attached: true) == "退出 localmelo（网关继续运行）")
precondition(L10n.plural("app.quit.tasks", 1, one: "{{n}} task", other: "{{n}} tasks") == "1 个任务")
precondition(L10n.t("app.no-such-key", "Kept {{x}}", ["x": "as is"]) == "Kept as is")
print(quitMessage(RunningCounts(chats: 1, tasks: 2, runs: 1, terminal: 0), statusKnown: true))
print(startStepText(portKnown: true, ready: false))
// A language the bundle lacks reads English.
L10n.use("fr", bundle: bundle)
precondition(L10n.language == "en" && quitTitle(attached: false) == "Quit localmelo")
""",
    )
    assert out.splitlines() == [
        "退出会停止正在运行的 2 个任务和 1 个对话。不受影响：1 次运行。",
        "正在登录",
    ]
