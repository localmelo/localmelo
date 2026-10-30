"""``AgentOptions.brief_items``: a brief run names the numbered items of the
brief that no edit has touched yet, in each reflection request and in the
deadline check."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from localmelo.melo.agent._brief_aids import (
    CHANGED,
    ITEM_LIMIT,
    ITEMS_HEADER,
    MAX_ITEMS,
    NO_FILE,
    NOT_YET,
    REFLECTION_ADDENDUM,
    edited_files,
    file_tokens,
    mark_items,
    parse_items,
    render_items,
    workspace_stamps,
)
from localmelo.melo.agent._brief_gate import CHECK_FAILED_LINE, BriefCheckResult
from localmelo.melo.contracts.events import AgentEvent
from localmelo.melo.schema import Message
from localmelo.tests.agent.test_reflection_gate import (
    FakeLLM,
    _agent,
    _answer,
    _echo_call,
    _is_reflection,
    _reflection,
)

BRIEF = """Fix three things.

1. Handle the empty list in `src/stats.py`
2. Add a test in tests/test_stats.py
3) Update the changelog
Check: pytest -q
"""


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "stats.py").write_text("x = 1\n")
    (root / "tests" / "test_stats.py").write_text("def test(): pass\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


class TestParsing:
    def test_numbered_lines_and_checkboxes(self) -> None:
        brief = "Do:\n1. first `a.py`\n2) second\n- [ ] third\n- [x] fourth\n- plain\n"
        assert parse_items(brief) == [
            "1. first `a.py`",
            "2. second",
            "3. third",
            "4. fourth",
        ]

    def test_code_blocks_and_nested_lines_are_ignored(self) -> None:
        brief = "```\n1. in code\n2. in code\n```\n1. one\n   2. nested\n2. two\n"
        assert parse_items(brief) == ["1. one", "2. two"]

    def test_fewer_than_two_items_give_none(self) -> None:
        assert parse_items("1. only one\nCheck: true\n") == []
        assert parse_items("no items at all") == []

    def test_items_are_shortened_and_capped(self) -> None:
        long = "1. " + "word " * 60
        [first, *_] = parse_items(f"{long}\n2. b\n")
        assert len(first) == ITEM_LIMIT and first.endswith("...")
        many = "\n".join(f"{n}. item {n}" for n in range(1, 30))
        assert len(parse_items(many)) == MAX_ITEMS

    def test_path_like_words_only(self) -> None:
        item = "1. In `src/stats.py` and utils.py, see e.g. 1.5 and docs/"
        assert file_tokens(item) == ["src/stats.py", "utils.py", "docs/"]
        assert file_tokens("2. Update the changelog") == []


class TestMarks:
    def test_changed_not_yet_and_no_file_named(self) -> None:
        items = parse_items(BRIEF)
        assert mark_items(items, {"src/stats.py"}) == [CHANGED, NOT_YET, NO_FILE]
        assert mark_items(items, {"pkg/tests/test_stats.py"}) == [
            NOT_YET,
            CHANGED,
            NO_FILE,
        ]
        assert mark_items(items, set()) == [NOT_YET, NOT_YET, NO_FILE]

    def test_a_directory_token_matches_the_files_below_it(self) -> None:
        assert mark_items(["1. x docs/", "2. y"], {"docs/a.md"}) == [CHANGED, NO_FILE]

    def test_the_block_lists_the_untouched_items_and_never_says_done(self) -> None:
        items = parse_items(BRIEF)
        marks = mark_items(items, {"src/stats.py"})
        block = render_items(items, marks)
        assert block == f"{ITEMS_HEADER} 2\n2. Add a test in tests/test_stats.py"
        assert "done" not in block
        assert render_items(items, [CHANGED, CHANGED, NO_FILE]) == ""


class TestEvidence:
    def test_stamps_follow_git_status(self, repo: Path) -> None:
        before = workspace_stamps(str(repo))
        assert before == {}
        (repo / "src" / "stats.py").write_text("x = 2\n")
        (repo / "new").mkdir()
        (repo / "new" / "mod.py").write_text("")
        (repo / "tests" / "test_stats.py").unlink()
        after = workspace_stamps(str(repo))
        assert after is not None and before is not None
        assert edited_files(before, after) == {
            "src/stats.py",
            "new/mod.py",
            "tests/test_stats.py",
        }
        assert after["tests/test_stats.py"] == (-1, -1)

    def test_a_file_dirty_before_the_run_is_not_an_edit(self, repo: Path) -> None:
        (repo / "src" / "stats.py").write_text("x = 3\n")
        before = workspace_stamps(str(repo))
        assert before is not None
        assert edited_files(before, workspace_stamps(str(repo)) or {}) == set()
        (repo / "src" / "stats.py").write_text("x = 4 # longer\n")
        assert edited_files(before, workspace_stamps(str(repo)) or {}) == {
            "src/stats.py"
        }

    def test_outside_a_git_tree_there_is_no_evidence(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        assert workspace_stamps(str(plain)) is None


class _WritingLLM(FakeLLM):
    """Writes *files* into the workspace as its *call*-th request arrives."""

    def __init__(self, replies: list[Message], writes: dict[int, Path]) -> None:
        super().__init__(replies)
        self.writes = writes

    async def chat(self, messages: list[Message], tools: Any = None) -> Message:
        path = self.writes.get(len(self.calls))
        if path is not None:
            path.write_text("edited by the scripted run\n")
        return await super().chat(messages, tools)


def _notices(events: list[AgentEvent]) -> list[Mapping[str, Any]]:
    return [
        e.data for e in events if e.type == "notice" and e.data["kind"] == "brief_items"
    ]


def _users(llm: FakeLLM, request: int) -> list[str]:
    return [m.content for m in llm.calls[request]["messages"] if m.role == "user"]


async def _run(
    repo: Path, replies: list[Message], **options: Any
) -> tuple[Any, _WritingLLM, list[AgentEvent]]:
    events: list[AgentEvent] = []
    llm = _WritingLLM(replies, {1: repo / "src" / "stats.py"})
    agent = _agent(
        llm,
        brief_probe=lambda workspace: f"after {len(llm.calls)} requests",
        workspace_root=str(repo),
        **options,
    )
    result = await agent.run_task(BRIEF, on_event=events.append)
    return result, llm, events


class TestInARun:
    @pytest.mark.asyncio
    async def test_the_reflection_request_names_the_untouched_item(
        self, repo: Path
    ) -> None:
        _result, llm, events = await _run(
            repo,
            [
                _echo_call("a"),
                _echo_call("b"),
                _reflection("done", best_effort_result="x"),
            ],
            brief_items=True,
            steps_per_attempt=2,
        )
        [call] = [c for c in llm.calls if _is_reflection(c)]
        last = call["messages"][-1].content
        assert f"{REFLECTION_ADDENDUM}\n{ITEMS_HEADER} 2\n2. Add a test" in last
        [notice] = _notices(events)
        assert notice["detail"] == {
            "scope": "reflection",
            "marks": {"1": CHANGED, "2": NOT_YET, "3": NO_FILE},
        }

    @pytest.mark.asyncio
    async def test_the_deadline_check_names_it_two_steps_before_the_end(
        self, repo: Path
    ) -> None:
        _result, llm, events = await _run(
            repo,
            [_echo_call("a"), _echo_call("b"), _echo_call("c"), _answer("done")],
            brief_items=True,
            attempt_policy="none",
            max_steps=4,
        )
        block = f"{ITEMS_HEADER} 2\n2. Add a test in tests/test_stats.py"
        assert [block in _users(llm, n)[-1] for n in range(4)] == [
            False,
            False,
            True,
            True,
        ]
        [notice] = _notices(events)
        assert notice["detail"]["scope"] == "deadline"

    @pytest.mark.asyncio
    async def test_a_failing_check_joins_the_block_in_one_user_line(
        self, repo: Path
    ) -> None:
        async def failing(executor: Any, command: str) -> BriefCheckResult:
            return BriefCheckResult(command, 1, 0.1, "boom")

        _result, llm, _events = await _run(
            repo,
            [_echo_call("a"), _echo_call("b"), _echo_call("c"), _answer("done")],
            brief_items=True,
            brief_check=failing,
            attempt_policy="none",
            max_steps=4,
        )
        last = _users(llm, 2)[-1]
        assert last.startswith(ITEMS_HEADER)
        assert last.endswith(CHECK_FAILED_LINE) and "Exit code: 1" in last

    @pytest.mark.asyncio
    async def test_planning_requests_before_the_deadline_never_carry_it(
        self, repo: Path
    ) -> None:
        _result, llm, _events = await _run(
            repo,
            [_echo_call("a"), _echo_call("b"), _echo_call("c"), _answer("done")],
            brief_items=True,
            attempt_policy="none",
            max_steps=4,
        )
        for n in range(2):
            assert not any(ITEMS_HEADER in text for text in _users(llm, n))

    @pytest.mark.asyncio
    async def test_off_by_default_changes_nothing(self, repo: Path) -> None:
        replies = lambda: [  # noqa: E731
            _echo_call("a"),
            _echo_call("b"),
            _echo_call("c"),
            _answer("done"),
        ]
        off = await _run(repo, replies(), attempt_policy="none", max_steps=4)
        (repo / "src" / "stats.py").write_text("x = 1\n")
        plain = await _run(
            repo, replies(), brief_items=False, attempt_policy="none", max_steps=4
        )
        assert not _notices(off[2]) and not _notices(plain[2])
        bodies = [
            [[(m.role, m.content) for m in c["messages"]] for c in run[1].calls]
            for run in (off, plain)
        ]
        assert bodies[0] == bodies[1]
        assert not any(ITEMS_HEADER in str(b) for b in bodies)

    @pytest.mark.asyncio
    async def test_a_brief_with_one_item_gets_no_reminder(self, repo: Path) -> None:
        events: list[AgentEvent] = []
        llm = FakeLLM([_echo_call("a"), _echo_call("b"), _echo_call("c"), _answer("d")])
        agent = _agent(
            llm,
            brief_probe=lambda workspace: "same",
            brief_items=True,
            workspace_root=str(repo),
            attempt_policy="none",
            max_steps=4,
        )
        await agent.run_task("1. only `a.py`\nmore text", on_event=events.append)
        assert not _notices(events)
        assert not any(ITEMS_HEADER in t for n in range(4) for t in _users(llm, n))

    @pytest.mark.asyncio
    async def test_outside_a_brief_run_the_option_is_inert(self, repo: Path) -> None:
        events: list[AgentEvent] = []
        llm = FakeLLM([_echo_call("a"), _echo_call("b"), _echo_call("c"), _answer("d")])
        agent = _agent(
            llm,
            brief_items=True,
            workspace_root=str(repo),
            attempt_policy="none",
            max_steps=4,
        )
        await agent.run_task(BRIEF, on_event=events.append)
        assert not _notices(events)


def test_the_option_must_be_a_bool() -> None:
    from localmelo.melo.agent.options import AgentOptions

    assert AgentOptions().brief_items is False
    with pytest.raises(ValueError, match="brief_items"):
        AgentOptions(brief_items="on")  # type: ignore[arg-type]
