"""How import reads the words of a file: wrapped lines, who "I" and "the user" are, key details.

The files are small invented ones under *tmp_path*; no real file is read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localmelo.support.memory import records
from localmelo.support.memory.importer import build_plan
from localmelo.support.memory.importer.rules import reject
from localmelo.support.memory.importer.sources import markdown
from localmelo.support.memory.importer.voice import to_you

from .import_home import OWN, put
from .test_import_budget import folder, home, write_note  # noqa: F401  (fixtures)

# ── Wrapped lines ──


def statements(md: str) -> list[tuple[int, str, str]]:
    return [(s.line, s.heading, s.text) for s in markdown(md, "p")]


def test_a_wrapped_bullet_or_paragraph_is_one_statement_numbered_by_its_first_line() -> (
    None
):
    md = (
        "# Rules\n"
        "- Hand design decisions over (the continuation gate,\n"
        "  the review step) and wait for the answer.\n"
        "- A short one here.\n"
        "\n"
        "A paragraph that wraps\n"
        "over three lines\n"
        "and then ends here.\n"
    )
    assert statements(md) == [
        (
            2,
            "Rules",
            "Hand design decisions over (the continuation gate, the review step) and wait for the answer.",
        ),
        (4, "Rules", "A short one here."),
        (6, "Rules", "A paragraph that wraps over three lines and then ends here."),
    ]


def test_a_line_that_starts_something_else_does_not_continue_the_one_above() -> None:
    md = (
        "- first point wraps\n"
        "  here and stops.\n"
        "- second point wraps\n"
        "```\ncode\n```\n"
        "third point wraps\n"
        "| a | b |\n"
        "fourth point wraps\n"
        "<!-- note -->\n"
        "fifth point wraps\n"
        "> quoted point\n"
        "sixth point wraps\n"
        "# Heading\n"
        "seventh point wraps\n"
        "https://example.test/page\n"
    )
    assert [text for _, _, text in statements(md)] == [
        "first point wraps here and stops.",
        "second point wraps",
        "third point wraps",
        "fourth point wraps",
        "fifth point wraps",
        "quoted point sixth point wraps",
        "seventh point wraps",
    ]


def test_lines_that_each_end_a_sentence_stay_apart_and_an_abbreviation_does_not_end_one() -> (
    None
):
    md = (
        "Keep answers short and plain.\n"
        "Use uv for new Python projects.\n"
        "- Uses tools, e.g.\n"
        "  git and make for builds.\n"
    )
    assert [text for _, _, text in statements(md)] == [
        "Keep answers short and plain.",
        "Use uv for new Python projects.",
        "Uses tools, e.g. git and make for builds.",
    ]


def test_a_why_line_and_a_link_alone_are_not_joined_to_the_fact_above() -> None:
    md = "A fact without an end mark\n**Why:** told directly\nand more of why.\n[A link](docs/x.md)\n"
    assert [text for _, _, text in statements(md)] == ["A fact without an end mark"]


def test_a_wrapped_note_file_line_reaches_the_plan_whole(
    home: Path,  # noqa: F811
) -> None:
    put(
        folder(home) / "wrapped.md",
        "---\nname: wrapped\ndescription: x\ntype: feedback\n---\n"
        "- Hand design decisions over (the continuation gate,\n  the review step) and wait.\n",
    )
    plan = build_plan(home, [], timeout_s=30)
    (notes,) = plan.notes.values()
    assert [n.text for n in notes] == [
        "Hand design decisions over (the continuation gate, the review step) and wait."
    ]
    assert notes[0].provenance["line"] == 6


# ── Who "I" and "the user" are ──


@pytest.mark.parametrize(
    ("said", "put_to_you"),
    [
        ("The user is a robotics researcher.", "You are a robotics researcher."),
        ("The user hands design decisions to me.", "You hand design decisions to me."),
        ("The user has a PhD in robotics.", "You have a PhD in robotics."),
        ("The user doesn't like jargon.", "You don't like jargon."),
        ("The user's repo lives under ~/work.", "Your repo lives under ~/work."),
        (
            "The user always prefers short answers and asks me to stop early.",
            "You always prefer short answers and ask me to stop early.",
        ),
        (
            "When the user asks for a summary, keep it short.",
            "When you ask for a summary, keep it short.",
        ),
        ("用户偏好简洁的回复。", "你偏好简洁的回复。"),
        ("用户的邮箱不要写进代码。", "你的邮箱不要写进代码。"),
        # "I" and "me" are the assistant's: no claim about you.
        (
            "After any PR I open is merged, I update the changelog.",
            "After any PR I open is merged, I update the changelog.",
        ),
        ("I prefer plain words.", "I prefer plain words."),
        ("I should ask the user first.", "Ask the user first."),
        ("I always run tests before a push.", "Always run tests before a push."),
        ("I should not push to main.", "Do not push to main."),
        ("I'll keep notes short.", "Keep notes short."),
        # "the user" that is no subject or owner keeps its words.
        ("Tell the user what changed.", "Tell the user what changed."),
        ("The user interface is slow.", "The user interface is slow."),
        (
            "The user settings are stored locally.",
            "The user settings are stored locally.",
        ),
        ("用户界面要简单。", "用户界面要简单。"),
    ],
)
def test_an_assistants_note_puts_the_user_in_the_second_person_and_keeps_its_own_i(
    said: str, put_to_you: str
) -> None:
    assert records.finished(to_you(said)) == put_to_you


def test_notes_an_assistant_wrote_say_you_for_the_user_and_files_the_user_wrote_keep_their_swap(
    home: Path,  # noqa: F811
) -> None:
    write_note(
        home,
        "role",
        "The user is a robotics researcher",
        ["The user hands design decisions to me."],
        "user",
    )
    put(
        home / ".claude" / "CLAUDE.md",
        "- I live in Osaka and my sister Mei visits me.\n",
    )
    put(home / ".hermes" / "memories" / "USER.md", "I prefer plain words in replies.")
    put(home / ".hermes" / "memories" / "MEMORY.md", "I keep zebra quartz notes short.")
    plan = build_plan(home, [], timeout_s=30)
    by_harness = {
        (s["harness"], Path(s["path"]).name): [n.text for n in plan.notes[s["id"]]]
        for s in plan.sources
    }
    assert by_harness[("claude", "memory")] == [
        "You are a robotics researcher.",
        "You hand design decisions to me.",
    ]
    # What the user wrote keeps today's output.
    line = "I live in Osaka and my sister Mei visits me."
    assert by_harness[("claude", "CLAUDE.md")] == [records.second_person(line)]
    assert (
        records.second_person(line)
        == "You live in Osaka and your sister Mei visits you."
    )
    assert by_harness[("hermes", "USER.md")] == ["You prefer plain words in replies."]
    assert by_harness[("hermes", "MEMORY.md")] == ["I keep zebra quartz notes short."]


# ── Details of a key and of how another tool picks a setting ──

FINGERPRINT = "SHA256:AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcdEFG"


@pytest.mark.parametrize(
    "line",
    [
        f"Key: `~/.ssh/id_example` (ED25519, {FINGERPRINT}, no passphrase). Authenticates as someuser.",
        f"The host key fingerprint is {FINGERPRINT}.",
        "MD5 fingerprint de:ad:be:ef:00:11:22:33:44:55:66:77:88:99:aa:bb for the old host.",
        "Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleExampleExampleExample1234 someuser@host",
        "Public key: ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQExampleExampleExample1234 someuser@host",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "PGP key 4F9A1C2B7D3E8F0A1B2C3D4E5F60718293A4B5C6 signs the releases.",
        "Fingerprint 4F9A 1C2B 7D3E 8F0A 1B2C 3D4E 5F60 7182 93A4 B5C6.",
        "The private key is ~/.ssh/id_ed25519 and has a passphrase.",
        "Deploy key at /Users/someuser/.ssh/deploy_example, fingerprint noted elsewhere.",
    ],
)
def test_key_details_are_secrets(line: str) -> None:
    assert reject(line, OWN) == "secret"


@pytest.mark.parametrize(
    "line",
    [
        "Host aliases live in ~/.ssh/config.",
        "Add the public key to ~/.ssh/authorized_keys on the server.",
        "Commit 4f9a1c2b7d3e8f0a1b2c3d4e5f60718293a4b5c6 fixed the build.",
        "Use ssh keys for GitHub.",
        "Set HF_HOME to ~/Developer/ml_source/huggingface.",
        "Put DEVELOPER_DIR in the project's .envrc.",
    ],
)
def test_a_how_to_that_only_mentions_ssh_files_or_a_commit_is_not_a_key_detail(
    line: str,
) -> None:
    assert reject(line, OWN) == ""


@pytest.mark.parametrize(
    "line",
    [
        "Precedence (docs): env SOME_TOOL_COMPACT_WINDOW > settings > default.",
        "SOME_TOOL_HOME takes precedence over OTHER_TOOL_DIR.",
    ],
)
def test_how_another_tool_picks_a_setting_is_mechanics(line: str) -> None:
    assert reject(line, OWN) == "mechanics"
