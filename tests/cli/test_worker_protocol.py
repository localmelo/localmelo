"""The brief path's worker protocol: the paragraph appended to the query, the
workspace probe it hands the agent, and the digest that probe returns."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

import localmelo.__main__ as cli
from localmelo.melo.agent import Agent, AgentOptions
from localmelo.support.cli.worker_contract import (
    WORKER_PROTOCOL,
    workspace_fingerprint,
)
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

TASK = "Explain the example."


def _run_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> tuple[list[dict[str, Any]], list[AgentOptions]]:
    """The request bodies of a CLI run and the options it built the agent with."""
    bodies: list[dict[str, Any]] = []
    built: list[AgentOptions] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
        )

    def build(*args: Any, options: AgentOptions, **kwargs: Any) -> Agent:
        built.append(options)
        return Agent(
            llm=OpenAICompatLLM(
                base_url="http://llm.test/v1",
                model="stub",
                transport=httpx.MockTransport(handler),
            ),
            options=AgentOptions(memory=False, brief_probe=options.brief_probe),
        )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_build_direct_mode_agent", build)
    monkeypatch.setattr(sys, "argv", argv)
    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code in (0, None)
    return bodies, built


def _user_texts(body: dict[str, Any]) -> list[str]:
    return [m["content"] for m in body["messages"] if m["role"] == "user"]


def test_the_protocol_is_appended_to_a_brief_query_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief = tmp_path / "brief.md"
    brief.write_text(TASK + "\n", encoding="utf-8")
    with_brief, built = _run_main(
        tmp_path, monkeypatch, ["melo", "--brief", str(brief)]
    )
    without, built_plain = _run_main(tmp_path, monkeypatch, ["melo", TASK])
    assert _user_texts(with_brief[0]) == [f"{TASK}\n\n{WORKER_PROTOCOL}"]
    assert _user_texts(without[0]) == [TASK]
    assert callable(built[0].brief_probe)
    assert built_plain[0].brief_probe is None


def test_the_paragraph_is_the_same_on_every_call() -> None:
    assert WORKER_PROTOCOL.startswith("Make the smallest change that satisfies")
    assert WORKER_PROTOCOL.endswith("which checks you ran and their result.")
    assert "\n" not in WORKER_PROTOCOL


def _git(path: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
    )


def test_fingerprint_follows_every_kind_of_change(tmp_path: Path) -> None:
    assert workspace_fingerprint(str(tmp_path)) is None  # not a git tree
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tracked.txt").write_text("one\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "Initial")
    clean = workspace_fingerprint(str(tmp_path))
    assert clean is not None and workspace_fingerprint(str(tmp_path)) == clean
    (tmp_path / "tracked.txt").write_text("two\n")
    edited = workspace_fingerprint(str(tmp_path))
    assert edited != clean
    # Another edit of the file that was dirty already still counts.
    (tmp_path / "tracked.txt").write_text("three\n")
    assert workspace_fingerprint(str(tmp_path)) != edited
    (tmp_path / "tracked.txt").write_text("one\n")
    assert workspace_fingerprint(str(tmp_path)) == clean
    (tmp_path / "new.txt").write_text("a")
    created = workspace_fingerprint(str(tmp_path))
    assert created != clean
    (tmp_path / "new.txt").write_text("a longer text")
    assert workspace_fingerprint(str(tmp_path)) != created


def test_fingerprint_in_a_tree_without_a_commit(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    empty = workspace_fingerprint(str(tmp_path))
    assert empty is not None
    (tmp_path / "a.txt").write_text("x")
    assert workspace_fingerprint(str(tmp_path)) != empty
