"""The coloured terminal blocks: what the parser colours, and that the pages are in step."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest

from localmelo.scripts import build_docs
from localmelo.scripts import usage_blocks as ub

ROOT = Path(__file__).resolve().parents[2]
PAGES = ("index", "quickstart", "developers")


def roles(text: str, mode: str = "shell") -> list[tuple[str, str]]:
    """The coloured parts of the first command row, as (role, text)."""
    return [(r, t) for r, t in ub.parse(text, mode)[0].tokens if r]


def test_a_command_splits_into_name_subcommand_flags_values_and_strings() -> None:
    row = 'melo -u http://127.0.0.1:8000/v1 -m Qwen3-8B-4bit "What is 6*7?"'
    assert roles(row) == [
        ("c", "melo"),
        ("f", "-u"),
        ("a", "http://127.0.0.1:8000/v1"),
        ("f", "-m"),
        ("a", "Qwen3-8B-4bit"),
        ("q", '"What is 6*7?"'),
    ]
    assert roles("melo memory forget 3 -y") == [
        ("c", "melo"),
        ("s", "memory"),
        ("s", "forget"),
        ("a", "3"),
        ("f", "-y"),
    ]


def test_uv_run_options_come_before_the_command_it_runs() -> None:
    assert roles(
        "uv run --with pillow==13.0.0 python scripts/art_pack.py check x.zip"
    ) == [
        ("c", "uv"),
        ("s", "run"),
        ("f", "--with"),
        ("a", "pillow==13.0.0"),
        ("c", "python"),
        ("a", "scripts/art_pack.py"),
        ("a", "check"),
        ("a", "x.zip"),
    ]
    assert roles("uv run --locked pytest -q") == [
        ("c", "uv"),
        ("s", "run"),
        ("f", "--locked"),
        ("c", "pytest"),
        ("f", "-q"),
    ]
    assert roles("uv run melo --serve")[2] == ("c", "melo")


def test_a_flag_with_equals_and_a_setting_split_at_the_equals() -> None:
    assert roles("curl --max-time=5 -X request_timeout_s=600") == [
        ("c", "curl"),
        ("f", "--max-time"),
        ("x", "="),
        ("a", "5"),
        ("f", "-X"),
        ("f", "request_timeout_s"),
        ("x", "="),
        ("a", "600"),
    ]
    assert roles("LOCALMELO_SMOKE=1 uv run pytest -q")[:5] == [
        ("f", "LOCALMELO_SMOKE"),
        ("x", "="),
        ("a", "1"),
        ("c", "uv"),
        ("s", "run"),
    ]
    assert ("c", "pytest") in roles("LOCALMELO_SMOKE=1 uv run pytest -q")


def test_quotes_pipes_and_comments_keep_their_own_roles() -> None:
    parts = roles("{ printf 'a b'; } | curl -H @- URL  # note, not a command")
    assert ("q", "'a b'") in parts and ("c", "curl") in parts
    assert parts[-1] == ("k", "# note, not a command")
    assert ub.parse("# only a comment")[0].kind == "note"
    assert ("a", "http://x/y#frag") in roles("curl http://x/y#frag")


def test_a_console_transcript_marks_prompts_output_and_check_marks() -> None:
    lines = ub.parse("$ melo\n✓ ready\nFound 你好 at ~/demo\n$ melo setup", "console")
    assert [(ln.kind, ln.prompt) for ln in lines] == [
        ("cmd", True),
        ("out", False),
        ("out", False),
        ("cmd", True),
    ]
    assert lines[1].tokens == (("y", "✓"), ("o", " ready"))
    assert lines[2].tokens == (("o", "Found 你好 at ~/demo"),)


def test_a_chat_block_marks_each_message_and_colours_slash_commands() -> None:
    text = "/approvals auto\nremind me to send the report on Friday at 9\n提醒我明天九点喝水"
    lines = ub.parse(text, "chat")
    assert [(ln.kind, ln.prompt, ln.glyph) for ln in lines] == [("cmd", True, "›")] * 3
    assert roles(text, "chat") == [("c", "/approvals"), ("a", "auto")]
    assert lines[1].tokens == (("q", "remind me to send the report on Friday at 9"),)
    assert ub.copy_text(lines) == text
    block = ub.Block("x", text, mode="chat")
    assert 'class="p">›</span>' in ub.render_html(block) and "$" not in ub.render_html(
        block
    )
    assert "›" in ub.render_svg(block, "light")
    assert 'class="p">$</span>' in ub.render_html(
        ub.Block("y", "melo")
    )  # a shell keeps `$`


def test_copy_has_the_commands_only() -> None:
    text = "$ melo  # open the chat\nsome output\n$ curl -sS URL \\\n  -d '{}'\nmore"
    lines = ub.parse(text, "console")
    assert ub.copy_text(lines) == "melo\ncurl -sS URL \\\n  -d '{}'"
    shell = ub.parse("# setup\ncd localmelo  # keep this name\nuv sync", "shell")
    assert ub.copy_text(shell) == "cd localmelo\nuv sync"


def test_html_escapes_text_and_keeps_the_command_in_data_copy() -> None:
    block = ub.Block("x", 'echo "<b>&</b>"', title=("A & B", ""))
    out = ub.render_html(block)
    assert "&lt;b&gt;&amp;&lt;/b&gt;" in out and "<b>" not in out
    assert 'data-copy="echo &quot;&lt;b&gt;&amp;&lt;/b&gt;&quot;"' in out
    assert 'class="usage-title">A &amp; B<' in out
    assert 'class="p">$</span>' in out and 'class="c">echo' in out


def test_a_block_without_commands_has_no_copy_button_and_labels_follow_the_language() -> (
    None
):
    block = ub.Block("x", "plain output", mode="console", title=("", "输出"))
    assert "usage-copy" not in ub.render_html(block, "zh")
    out = ub.render_html(ub.Block("x", "melo"), "zh")
    assert ">复制</button>" in out and 'data-done="已复制"' in out


def test_svg_is_stable_themed_and_wide_enough_for_wide_characters() -> None:
    block = ub.Block("x", '$ melo "a<b"\n你好你好你好你好你好你好', mode="console")
    assert ub.cells("abc") == 3 and ub.cells("你好") == 4 and ub.cells("e\u0301") == 1
    light, dark = ub.render_svg(block, "light"), ub.render_svg(block, "dark")
    assert light == ub.render_svg(block, "light") and light != dark
    assert ub.PALETTE["dark"]["bg"] in dark and ub.PALETTE["light"]["bg"] in light
    assert "a&lt;b" in light and "<tspan" in light
    narrow = ub.render_svg(ub.Block("y", "你好"), "light")
    assert 'width="380"' in narrow  # the window never shrinks below its minimum
    wide = ub.render_svg(ub.Block("y", "你" * 60, mode="console"), "light")
    assert 'width="1048"' in wide  # 60 characters of two cells at 8.4 plus the padding


def test_the_palette_passes_contrast_and_the_css_uses_it() -> None:
    _, problems = ub.contrast_report()
    assert problems == []
    css = (ROOT / "docs/assets/site.css").read_text()
    light, dark = css.split("@media (prefers-color-scheme: dark) {\n  .usage {")
    dark = dark.split("}", 1)[0]
    for scheme, source in (
        ("light", light.split("/* Terminal blocks")[1]),
        ("dark", dark),
    ):
        found = dict(re.findall(r"--u-(\w+): (#[0-9a-f]{6});", source))
        assert found == dict(ub.PALETTE[scheme]), scheme


def test_the_builder_draws_console_and_shell_fences_only() -> None:
    page = build_docs.render_markdown(
        "```shell\nmelo serve\n```\n\n```console\n$ melo\nhi\n```\n\n```python\nx = 1\n```\n"
    )
    assert page.count('<figure class="usage"') == 2
    assert 'data-copy="melo serve"' in page and 'data-copy="melo"' in page
    assert '<code class="language-python">x = 1' in page


def test_every_generated_region_is_current() -> None:
    assert ub.sync(check=True) == []


@pytest.mark.parametrize("lang", ["en", "zh-CN"])
def test_pages_keep_their_commands_and_no_loose_command_blocks(lang: str) -> None:
    for page in PAGES:
        text = (ROOT / f"docs/{lang}/{page}.html").read_text()
        loose = re.findall(r"<pre><code>(.*?)</code></pre>", text, re.S)
        assert all("__main__.py" in code for code in loose), (lang, page)
    quickstart = (ROOT / f"docs/{lang}/quickstart.html").read_text()
    copies = [html.unescape(c) for c in re.findall(r'data-copy="([^"]*)"', quickstart)]
    for command in (
        'melo -u http://127.0.0.1:11434/v1 -m qwen3:8b "What is 6*7?"',
        'melo -u http://127.0.0.1:8000/v1 -m Qwen3-8B-4bit "What is 6*7?"',
        "omlx serve --model-dir ~/models --port 8000",
        "uv sync --extra gateway",
    ):
        assert any(command in copy for copy in copies), command


def test_the_twins_hold_the_same_blocks_in_the_same_order() -> None:
    for page in PAGES:
        found = [
            re.findall(
                r'data-copy="([^"]*)"', (ROOT / f"docs/{lang}/{page}.html").read_text()
            )
            for lang in ("en", "zh-CN")
        ]
        assert found[0] == found[1] and found[0], page


def test_the_five_steps_each_show_a_block_or_a_picture() -> None:
    blocks, steps = ub.load()
    assert len(steps) == 5
    for step in steps:
        assert bool(step.get("blocks")) != bool(step.get("image")), step["id"]
        for name in step.get("blocks", ()):  # type: ignore[attr-defined]
            assert name in blocks
        if step.get("image"):
            assert (ROOT / "docs/assets/img" / str(step["image"])).is_file()
    out = ub.render_steps(steps, blocks, "en")
    assert out.count('<li class="step">') == 5 and out.startswith('<ol class="steps">')
    for name in ("index", "quickstart"):
        page = (ROOT / f"docs/en/{name}.html").read_text()
        assert "<!-- usage:steps -->" in page


def test_the_readmes_show_a_picture_and_the_same_commands_as_text() -> None:
    blocks, _ = ub.load()
    for name in ("README.md", "README.zh-CN.md"):
        text = (ROOT / name).read_text()
        regions = re.findall(r"<!-- usage:([\w-]+) -->(.*?)<!-- /usage -->", text, re.S)
        assert len(regions) == 5, name
        for block_id, body in regions:
            commands = ub.copy_text(
                ub.parse(blocks[block_id].text, blocks[block_id].mode)
            )
            assert f"```bash\n{commands}\n```" in body
            for scheme in ("light", "dark"):
                assert (ROOT / f"docs/assets/usage/{block_id}.{scheme}.svg").is_file()
