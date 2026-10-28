"""Reasoning notes: the goldens, note blocks, the heuristic, drafts and chunks.

Every expected value here and in ``fixtures/notes`` was derived by hand
from the rules of the R3 spec (Requirements 2 to 6) before it was run.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from localmelo.melo.memory import notes
from localmelo.melo.memory.notes import (
    DRAFT_WRITTEN_SHARE,
    TASK_SOURCE_RECORD_LIMIT,
    Draft,
    ReasoningNote,
    SourceRecordLog,
    ThoughtChunk,
    candidate_items,
    digest_note,
    draft_coverage,
    draft_title,
    extract_draft,
    mark_drafts_written,
    note_from_reply,
    note_title,
    parse_note_block,
    path_spans,
    render_digest,
    render_draft_block,
    render_note,
    render_note_block,
    same_path,
    short_call,
    split_note,
    split_sentences,
    thought_chunks,
)
from localmelo.melo.schema import ToolCall, ToolResult

FIXTURES = Path(__file__).parent / "fixtures" / "notes"
GOLDEN_NAMES = (
    "en_tool_turn",
    "en_model_note",
    "en_answer",
    "en_empty",
    "en_unclosed_note",
    "en_errors_and_records",
    "en_overflow",
    "zh_tool_turn",
    "zh_model_note",
    "mixed_zh_en",
    "en_dotted_names",
)


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _note(
    calls: list[ToolCall], results: list[ToolResult], content: str = ""
) -> ReasoningNote:
    return note_from_reply(content, "", calls, results)[0]


# ── Goldens ──────────────────────────────────────────────────────────────────


def test_the_goldens_are_the_ones_the_spec_lists() -> None:
    assert sorted(path.stem for path in FIXTURES.glob("*.json")) == sorted(GOLDEN_NAMES)


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_golden(name: str) -> None:
    case = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    assert case["name"] == name
    reply = case["reply"]
    calls = [ToolCall(**item) for item in reply["tool_calls"]]
    results = [ToolResult(**item) for item in reply["tool_results"]]

    note, rest = note_from_reply(reply["content"], reply["thinking"], calls, results)
    actual = dataclasses.asdict(note)
    # R6-lite adds the record texts beside the ids; the golden keeps the
    # fields it was written for.
    source_records = actual.pop("source_records")
    assert actual == case["expected_note"]
    assert [record_id for record_id, _text in source_records] == note.source_ids
    assert all(len(text) <= 500 for _record_id, text in source_records)
    assert rest == case["expected_rest"]

    draft = extract_draft(
        reply["thinking"] + "\n\n" + rest, recent_path=case["recent_path"]
    )
    assert (dataclasses.asdict(draft) if draft else None) == case["expected_draft"]

    chunks = thought_chunks(reply["thinking"])
    assert [dataclasses.asdict(chunk) for chunk in chunks] == case["expected_chunks"]

    assert render_note(note) == case["expected_render"]


# ── Caps and types ───────────────────────────────────────────────────────────


def test_caps() -> None:
    assert notes.NOTE_CHARS == 1_200
    assert notes.NOTE_PLAN_CHARS == 300
    assert notes.NOTE_ITEM_CHARS == 200
    assert notes.NOTE_OBSERVED_CHARS == 160
    assert (
        notes.NOTE_MAX_FOUND,
        notes.NOTE_MAX_DECIDED,
        notes.NOTE_MAX_HYPOTHESES,
        notes.NOTE_MAX_OPEN,
        notes.NOTE_MAX_OBSERVED,
    ) == (4, 3, 3, 3, 5)
    assert notes.FOUND_WINDOW_CHARS == 4_000
    assert notes.DECIDED_TAIL_SHARE == 0.4
    assert notes.TITLE_CHARS == 120
    assert notes.DRAFT_CHARS == 8_000
    assert notes.DRAFT_SHOWN_CHARS == 6_000
    assert notes.DRAFT_WRITTEN_SHARE == 0.6
    assert notes.THOUGHT_CHUNK_CHARS == 1_500


def test_file_extensions() -> None:
    assert len(notes.FILE_EXTENSIONS) == 50
    assert {"py", "md", "json", "ts", "sh", "zsh", "ipynb"} <= notes.FILE_EXTENSIONS
    for left_out in ("cfg", "conf", "env", "lock", "log", "patch"):
        assert left_out not in notes.FILE_EXTENSIONS


def test_reasoning_note_has_the_sketch_fields_in_order() -> None:
    assert [field.name for field in dataclasses.fields(ReasoningNote)] == [
        "note_id",
        "task_id",
        "step",
        "attempt",
        "call_seq",
        "kind",
        "source",
        "plan",
        "found",
        "decided",
        "hypotheses",
        "open_questions",
        "observed",
        "files",
        "verified",
        "outcome",
        "drafts",
        "reasoning_chars",
        "source_ids",
        "source_records",
        "remember",
        "remember_by",
        "stage",
        "entry_id",
        "uses",
        "last_used_at",
        "created_at",
    ]
    note = ReasoningNote()
    assert (note.note_id, note.task_id, note.step, note.attempt, note.call_seq) == (
        "",
        "",
        0,
        0,
        0,
    )
    assert (note.kind, note.source, note.stage) == ("step", "heuristic", "fresh")
    assert (note.remember, note.remember_by, note.entry_id) == (False, "", "")


def test_draft_and_thought_chunk() -> None:
    assert dataclasses.asdict(Draft()) == {
        "draft_id": "",
        "task_id": "",
        "step": 0,
        "call_seq": 0,
        "text": "",
        "target": "",
        "status": "pending",
        "lines": 0,
        "blocks": 0,
        "cut": False,
        "created_at": 0.0,
    }
    chunk = ThoughtChunk(title="t", text="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.title = "u"  # type: ignore[misc]


def test_note_from_reply_is_deterministic() -> None:
    path = FIXTURES / "en_tool_turn.json"
    reply = json.loads(path.read_text(encoding="utf-8"))["reply"]
    calls = [ToolCall(**item) for item in reply["tool_calls"]]
    results = [ToolResult(**item) for item in reply["tool_results"]]
    first = note_from_reply(reply["content"], reply["thinking"], calls, results)
    second = note_from_reply(reply["content"], reply["thinking"], calls, results)
    assert first == second
    assert first[0].note_id == ""


# ── split_note and parse_note_block ──────────────────────────────────────────


def test_split_note_takes_the_last_block_and_removes_every_block() -> None:
    content = "A <note>plan: one</note> B <note>plan: two</note> C"
    assert split_note(content) == ("plan: two", "A  B  C")


def test_split_note_ignores_case() -> None:
    assert split_note("<NOTE> x </Note>Answer.") == ("x", "Answer.")


def test_split_note_unclosed_block_runs_to_the_end() -> None:
    assert split_note("Done.\n<note>\nplan: finish\n") == ("plan: finish", "Done.")


def test_split_note_without_a_block_returns_the_content_unchanged() -> None:
    content = "  An answer with </note> but no opening tag.\n"
    assert split_note(content) == (None, content)


def test_split_note_leaves_a_closing_tag_without_an_opening_one() -> None:
    assert split_note("x </note> y <note>z</note>") == ("z", "x </note> y")


def test_parse_note_block_fields_markers_and_continuations() -> None:
    block = (
        "Run the folds first\n"
        "- Plan: then read the tests\n"
        "* found：foldl passes; foldr fails；reverse is missing\n"
        "  because the stub raises\n"
        "\n"
        "• Decided: plain loops\n"
        "hypothesis: the order is wrong\n"
        "Open Questions: is slicing allowed?\n"
        "found: a second line of findings\n"
    )
    note = parse_note_block(block)
    assert note.source == "model"
    # Lines before the first key go to plan; continuations join with a space.
    assert note.plan == "Run the folds first then read the tests"
    assert note.found == [
        "foldl passes",
        "foldr fails",
        "reverse is missing because the stub raises",
        "a second line of findings",
    ]
    assert note.decided == ["plain loops"]
    assert note.hypotheses == ["the order is wrong"]
    assert note.open_questions == ["is slicing allowed?"]
    assert (note.remember, note.remember_by) == (False, "")


def test_parse_note_block_caps() -> None:
    block = (
        "plan: " + "step " * 80 + "\n"
        "found: " + "long " * 50 + "; b; ; c; d; e\n"
        "decided: a; b; c; d\n"
        "hypotheses: a; b; c; d\n"
        "open: a; b; c; d\n"
    )
    note = parse_note_block(block)
    # 399 characters cut at 299, where a space follows a whole "step".
    assert note.plan == "step " * 59 + "step…"
    assert len(note.plan) == 300
    # The first item, 249 characters, is cut at 199; empty items are
    # dropped. An explicit note keeps up to 12 found items (T68): its cap
    # is not lower than the 12 a factual reply may carry.
    assert note.found == ["long " * 39 + "long…", "b", "c", "d", "e"]
    assert note.decided == ["a", "b", "c"]
    assert note.hypotheses == ["a", "b", "c"]
    assert note.open_questions == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("yes", True),
        ("Yes.", True),
        ("TRUE!", True),
        ("1", True),
        ("是", True),
        ("是。", True),
        ("no", False),
        ("No.", False),
        ("false", False),
        ("0", False),
        ("否", False),
        ("否！", False),
    ],
)
def test_remember_values(value: str, expected: bool) -> None:
    note = parse_note_block(f"plan: x\nremember: {value}")
    assert (note.remember, note.remember_by) == (expected, "model")
    note, _ = note_from_reply(f"<note>found: a fact\nremember: {value}</note>")
    assert (note.remember, note.remember_by) == (expected, "model")


@pytest.mark.parametrize("value", ["maybe", "yes, keep it", "", "是的"])
def test_an_unreadable_remember_is_left_to_the_heuristic(value: str) -> None:
    assert parse_note_block(f"remember: {value}").remember_by == ""
    note, _ = note_from_reply(f"<note>found: a fact\nremember: {value}</note>")
    assert (note.remember, note.remember_by) == (True, "heuristic")
    note, _ = note_from_reply(f"<note>plan: wait\nremember: {value}</note>")
    assert (note.remember, note.remember_by) == (False, "heuristic")


def test_the_heuristic_writer_reads_the_note_text_without_parsing_it() -> None:
    content = "<note>\nplan: Let me read list_ops.py next.\nremember: yes\n</note>"
    note, rest = note_from_reply(content, writer="heuristic")
    assert note.source == "heuristic"
    assert note.plan == "plan: Let me read list_ops.py next."
    assert (note.remember, note.remember_by) == (False, "heuristic")
    assert rest == ""
    model, _ = note_from_reply(content)
    assert (model.source, model.plan) == ("model", "Let me read list_ops.py next.")
    assert (model.remember, model.remember_by) == (True, "model")


@pytest.mark.parametrize("writer", ["model", "", "Heuristic", "none"])
def test_any_other_writer_raises(writer: str) -> None:
    with pytest.raises(ValueError, match="writer"):
        note_from_reply("text", writer=writer)


def test_kind_is_derived_unless_given() -> None:
    read = ToolCall("file_read", {"path": "app.py"})
    assert note_from_reply("", "", [read])[0].kind == "step"
    assert note_from_reply("The answer.")[0].kind == "answer"
    assert note_from_reply("  \n")[0].kind == "empty"
    assert note_from_reply("<note>plan: wait</note>")[0].kind == "empty"
    assert note_from_reply("The answer.", kind="reflection")[0].kind == "reflection"


def test_a_plain_answer_and_a_thinking_less_reflection_get_a_minimal_note() -> None:
    answer, rest = note_from_reply("The directory holds notes.txt and todo.md.")
    assert (answer.kind, rest) == (
        "answer",
        "The directory holds notes.txt and todo.md.",
    )
    assert answer.found == ["The directory holds notes.txt and todo.md."]
    assert render_note(answer) == "found: The directory holds notes.txt and todo.md."

    reflection, _ = note_from_reply(
        '{"recommended_action": "done", "summary": "it works"}',
        kind="reflection",
        writer="heuristic",
    )
    assert reflection.kind == "reflection"
    assert reflection.found == ['{"recommended_action": "done", "summary": "it works"}']
    assert render_note(reflection).startswith("found: ")

    fallback, _ = note_from_reply(
        '{"recommended_action": "done"}',
        kind="reflection",
        writer="heuristic",
        fallback="The fix works because the tests pass.",
    )
    assert fallback.found == ["The fix works because the tests pass."]


def test_a_minimal_note_takes_at_most_four_items_of_two_hundred_characters() -> None:
    text = " ".join(f"Sentence {index}." for index in range(6))
    note, _ = note_from_reply(text)
    assert len(note.found) == 4
    assert note.found == [f"Sentence {index}." for index in range(4)]

    long, _ = note_from_reply("x" * 500)
    assert len(long.found) == 1 and len(long.found[0]) == 200


def test_reasoning_chars_counts_the_thinking() -> None:
    assert note_from_reply("", "abc\n\ndef")[0].reasoning_chars == 8


# ── Sentences and paths ──────────────────────────────────────────────────────


def test_split_sentences_keeps_names_versions_and_abbreviations_whole() -> None:
    text = (
        "Read list_ops.py on Python 3.11, e.g. with pytest. "
        "Then use i.e. mode! Done? E.G. this."
    )
    sentences = split_sentences(text)
    assert sentences == [
        (0, "Read list_ops.py on Python 3.11, e.g. with pytest."),
        (51, "Then use i.e. mode!"),
        (71, "Done?"),
        (77, "E.G. this."),
    ]
    for offset, sentence in sentences:
        assert text[offset : offset + len(sentence)] == sentence


def test_split_sentences_strips_list_markers() -> None:
    text = "- one\n* two\n• three\n1. four\n2) five\n10. six"
    assert split_sentences(text) == [
        (2, "one"),
        (8, "two"),
        (14, "three"),
        (23, "four"),
        (31, "five"),
        (40, "six"),
    ]


def test_split_sentences_drops_empty_pieces_and_bare_numbers() -> None:
    assert split_sentences("\n\n  \n- \n1.\n") == []
    assert split_sentences("Done. 2. Next.") == [(0, "Done."), (9, "Next.")]


def test_split_sentences_after_any_whitespace() -> None:
    # U+3000 (ideographic space) and U+00A0 (no-break space) are whitespace.
    text = "Done?　Next. Then go. 例如e.g. 这个"
    assert split_sentences(text) == [
        (0, "Done?"),
        (6, "Next."),
        (12, "Then go."),
        (21, "例如e.g. 这个"),
    ]


def test_split_sentences_chinese() -> None:
    assert split_sentences("我先读取文件。然后运行测试！好吗？\n下一步") == [
        (0, "我先读取文件。"),
        (7, "然后运行测试！"),
        (14, "好吗？"),
        (18, "下一步"),
    ]


def _spans(text: str) -> list[str]:
    return [text[start:end] for start, end in path_spans(text)]


@pytest.mark.parametrize(
    "name",
    [
        "list_ops.py",
        "README.md",
        "README.MD",
        "response.json",
        "list_ops.spec.ts",
        "notes.TXT",
        "my-file_2.yaml",
    ],
)
def test_path_spans_file_names(name: str) -> None:
    assert _spans(f"see {name} here") == [name]


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_list_ops.py",
        "./run.sh",
        "~/notes",
        "/tmp/out",
        "../lib/util.rb",
        ".\\build.bat",
        "src\\main.rs",
    ],
)
def test_path_spans_paths(path: str) -> None:
    assert _spans(f"see {path} here") == [path]


@pytest.mark.parametrize(
    "text",
    [
        "e.g",
        "3.11",
        "v1.2",
        "self.items",
        "os.path",
        "functools.reduce",
        "math.log",
        "response.json()",
        "read/write",
        "and/or",
        "10/22",
        "src/utils",
        "app.cfg",
        "poetry.lock",
        "/",
        "...",
    ],
)
def test_path_spans_non_matches(text: str) -> None:
    assert path_spans(f"see {text} here") == []


def test_path_spans_cjk_and_trailing_dots_end_a_run() -> None:
    assert path_spans("读取list_ops.py文件") == [(2, 13)]
    assert path_spans("See tests/test_x.py.") == [(4, 19)]
    assert _spans("call response.json() and open response.json now") == [
        "response.json"
    ]


# ── The heuristic ────────────────────────────────────────────────────────────


def test_plan_prefers_the_latest_sentences_that_name_a_file_or_an_action() -> None:
    thinking = (
        "Let me think. Now I'll read app.py. Then we go. "
        "Next I will fix util.py. Now we wait."
    )
    assert note_from_reply("", thinking)[0].plan == (
        "Now I'll read app.py. Next I will fix util.py."
    )


def test_plan_fills_up_with_the_latest_other_intent_sentence() -> None:
    thinking = "Let me think. Then we go. I'll check it. Now we wait."
    assert note_from_reply("", thinking)[0].plan == "I'll check it. Now we wait."
    thinking = "Let me think. Then we go. Now we wait."
    assert note_from_reply("", thinking)[0].plan == "Then we go. Now we wait."


def test_markers_are_whole_words_and_accept_a_curly_apostrophe() -> None:
    # "nowhere", "reader" and "thenceforth" hold no marker word.
    assert note_from_reply("", "The reader is nowhere thenceforth.")[0].plan == ""
    assert note_from_reply("", "I’ll go.")[0].plan == "I’ll go."
    # ASCII word boundaries: an English marker next to CJK still counts.
    assert note_from_reply("", "我们now开始。")[0].plan == "我们now开始。"


def test_markers_and_sentences_take_any_whitespace() -> None:
    # The words of a marker may be apart by U+00A0 or U+3000.
    assert note_from_reply("", "Let me think.")[0].plan == "Let me think."
    note = note_from_reply("", "I　need　to go. Not sure about it.")[0]
    assert (note.plan, note.open_questions) == (
        "I　need　to go.",
        ["Not sure about it."],
    )
    # A question ends at U+3000, so it is an open question, not the plan's.
    note = note_from_reply("", "这个函数会返回None吗?　接下来读取 list_ops.py。")[0]
    assert (note.plan, note.open_questions) == (
        "接下来读取 list_ops.py。",
        ["这个函数会返回None吗?"],
    )


def test_plan_with_calls_and_without_sentences() -> None:
    read = ToolCall("file_read", {"path": "app.py"})
    run = ToolCall("shell_exec", {"command": "pytest -q"})
    note = note_from_reply("", "Let me read app.py.", [read, run])[0]
    assert note.plan == "Let me read app.py. → read app.py; run pytest -q"
    note = note_from_reply("", "Nothing to plan here.", [read, run])[0]
    assert note.plan == "→ read app.py; run pytest -q"


def test_plan_shortens_the_sentences_first() -> None:
    sentence = "Now I'll check " + "the parser " * 30 + "today."
    note = note_from_reply("", sentence, [ToolCall("file_read", {"path": "app.py"})])[0]
    # Room for the sentence: 300 - len(" → ") - len("read app.py") = 286.
    assert note.plan == "Now I'll check " + "the parser " * 24 + "the… → read app.py"


def test_plan_shortens_the_calls_when_they_leave_no_room() -> None:
    call = ToolCall("shell_exec", {"command": "echo " + "word " * 20})
    short = "run echo " + "word " * 10 + "word…"
    assert short_call(call) == short
    calls = "; ".join([short] * 5)  # 328 characters
    note = note_from_reply("", "Let me run the five commands.", [call] * 5)[0]
    assert note.plan == ("→ " + calls)[:299] + "…"
    assert len(note.plan) == 300


def test_each_sentence_goes_to_one_field() -> None:
    note = note_from_reply("", "I'll use loops instead. Maybe it fails?")[0]
    assert note.plan == "I'll use loops instead."
    assert note.decided == []
    assert note.open_questions == ["Maybe it fails?"]
    assert (note.hypotheses, note.found) == ([], [])


def test_decided_only_from_the_last_forty_percent() -> None:
    # 97 characters with "\n\n": the tail starts at 58.2.
    thinking = "We must keep the stub.\n" + "z" * 50 + "\nTherefore we keep it."
    note = note_from_reply("", thinking)[0]
    assert note.decided == ["Therefore we keep it."]
    assert note.remember is True


def test_found_only_from_the_last_four_thousand_characters() -> None:
    # 4,134 characters with "\n\n": the window starts at 134.
    thinking = "It fails here.\n" + "z" * 4100 + "\nIt passes there."
    assert note_from_reply("", thinking)[0].found == ["It passes there."]


def test_hypotheses_are_prefixed_and_shortened_with_the_prefix() -> None:
    sentence = "Maybe " + "the parser " * 22 + "breaks."
    note = note_from_reply("", sentence)[0]
    assert note.hypotheses == ["open: Maybe " + "the parser " * 16 + "the parser…"]
    assert len(note.hypotheses[0]) == 199


def test_open_questions() -> None:
    thinking = "Is it sorted? I am not sure. We need to check the input. 不确定。"
    note = note_from_reply("", thinking)[0]
    assert note.open_questions == [
        "I am not sure.",
        "We need to check the input.",
        "不确定。",
    ]


def test_remember_by_the_heuristic() -> None:
    assert note_from_reply("", "It returns 0.")[0].remember is True
    # The decision starts at 37 of 52 characters, inside the tail (31.2).
    thinking = "Some longer context comes first here. We must stop."
    assert note_from_reply("", thinking)[0].decided == ["We must stop."]
    assert note_from_reply("", thinking)[0].remember is True
    note = note_from_reply("", "Maybe so. Is it?")[0]
    assert (note.remember, note.remember_by) == (False, "heuristic")


# ── Calls, observed lines, outcome and source ids ────────────────────────────


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (ToolCall("file_read", {"path": "a/b.py"}), "read a/b.py"),
        (ToolCall("file_write", {"path": "out.txt", "content": "x"}), "write out.txt"),
        (
            ToolCall("shell_exec", {"command": "pytest   -q\n  tests"}),
            "run pytest -q tests",
        ),
        (ToolCall("shell_exec", {"command": ["ls", "-la", 3]}), "run ls -la 3"),
        (
            ToolCall("python_exec", {"code": "\n\n  import os\nprint(1)"}),
            "python import os",
        ),
        (
            ToolCall("mail_search", {"limit": 5, "query": "  invoice\nfrom Acme "}),
            "mail_search invoice from Acme",
        ),
        (
            ToolCall("search", {"query": "alpha " * 10}),
            "search alpha alpha alpha alpha alpha alpha…",
        ),
        (ToolCall("list_tools", {}), "list_tools"),
        (ToolCall("file_read", {}), "read"),
    ],
)
def test_short_call(call: ToolCall, expected: str) -> None:
    assert short_call(call) == expected


def test_short_call_caps() -> None:
    long_name = ToolCall("mcp__" + "x" * 120, {})
    assert short_call(long_name) == "mcp__" + "x" * 94 + "…"
    code = ToolCall("python_exec", {"code": "total = " + "1 + " * 30 + "1"})
    # The line, 129 characters, is cut at 59, after "1 +".
    assert short_call(code) == "python total = " + "1 + " * 12 + "1 +…"


def test_observed_forms() -> None:
    output = "line one\nline two\n"
    calls = [
        ToolCall("file_read", {"path": "app.py"}),
        ToolCall("file_write", {"path": "out.txt", "content": "abc"}),
        ToolCall("shell_exec", {"command": "make"}),
        ToolCall("python_exec", {"code": "print(42)"}),
        ToolCall("lookup", {"key": "k"}),
    ]
    results = [
        ToolResult("file_read", output),
        ToolResult("file_write", "Written 3 bytes to out.txt"),
        ToolResult("shell_exec", "\n---\n  Built 3 targets  \nok"),
        ToolResult("python_exec", "42"),
        ToolResult("lookup", "plain text"),
    ]
    assert _note(calls, results).observed == [
        f"read app.py: 18 chars, sha {_sha8(output)}",
        "wrote out.txt: 3 chars",
        "ran `make`: Built 3 targets",
        "ran `python`: 42",
        "lookup: 10 chars",
    ]


def test_observed_error_form() -> None:
    call = ToolCall("shell_exec", {"command": "make all"})
    result = ToolResult("shell_exec", error="\n\n  make: *** No rule  \nmore")
    assert _note([call], [result]).observed == [
        "run make all: Error: make: *** No rule"
    ]


def test_observed_pytest_summary_is_the_last_one() -> None:
    output = (
        "....\n= 3 passed in 0.10s =\nnoise\n"
        "== 1 failed, 2 passed, 1 xfailed in 1.5s ==\ntrailing line"
    )
    call = ToolCall("shell_exec", {"command": "pytest"})
    note = _note([call], [ToolResult("shell_exec", output)])
    assert note.observed == ["ran `pytest`: 1 failed, 2 passed, 1 xfailed in 1.5s"]
    assert note.outcome == "1 failed, 2 passed, 1 xfailed in 1.5s"


@pytest.mark.parametrize("output", ["", "---\n===\n", "  \n"])
def test_observed_without_output_is_not_recorded(output: str) -> None:
    # T76: an empty or trivial result is no observation for a digest, so
    # it is not kept as one (the note used to say "(no output)").
    call = ToolCall("shell_exec", {"command": "true"})
    assert _note([call], [ToolResult("shell_exec", output)]).observed == []


def test_observed_summary_line_is_shortened_to_80() -> None:
    output = "warn " * 30
    call = ToolCall("shell_exec", {"command": "make"})
    assert _note([call], [ToolResult("shell_exec", output)]).observed == [
        "ran `make`: " + "warn " * 15 + "warn…"
    ]


def test_observed_records() -> None:
    listing = '[{"id": 1}, {"id": 2}, {"name": "x"}, 3]'
    nested = (
        '  {"record_id": "r1", "items": [{"id": "a"}, {"id": "b"}], '
        '"meta": {"id": "x"}}'
    )
    calls = [ToolCall("lookup", {}) for _ in range(4)]
    results = [
        ToolResult("lookup", listing),
        ToolResult("lookup", nested),
        ToolResult("lookup", "[not json"),
        ToolResult("lookup", '{"items": []}'),
    ]
    assert _note(calls, results).observed == [
        f"lookup: {len(listing)} chars, 2 records",
        f"lookup: {len(nested)} chars, 3 records",
        "lookup: 9 chars",
        "lookup: 13 chars",
    ]


def test_observed_lines_are_shortened_and_capped_at_five() -> None:
    call = ToolCall("lookup", {})
    error = ToolResult("lookup", error="bad " * 50)
    note = _note([call] * 6, [error] * 6)
    # 214 characters; the cut at 159 falls after a space, which is dropped.
    line = "lookup: Error: " + "bad " * 35 + "bad…"
    assert len(line) == 159
    assert note.observed == [line] * 5


def test_calls_and_results_pair_by_index() -> None:
    calls = [
        ToolCall("file_write", {"path": "a.txt", "content": "xy"}),
        ToolCall("file_read", {"path": "b.txt"}),
    ]
    note = _note(calls, [ToolResult("file_write", "ok")])
    assert note.observed == ["wrote a.txt: 2 chars"]
    assert note.outcome == "ok"
    assert _note(calls, []).outcome == ""


def test_outcome_the_first_error_wins() -> None:
    calls = [ToolCall("shell_exec", {"command": "pytest"}) for _ in range(3)]
    results = [
        ToolResult("shell_exec", "2 passed in 0.1s"),
        ToolResult("shell_exec", error="first\nsecond"),
        ToolResult("shell_exec", error="later"),
    ]
    assert _note(calls, results).outcome == "Error: first"


def test_outcome_the_last_pytest_summary_of_a_shell_or_python_call() -> None:
    calls = [
        ToolCall("shell_exec", {"command": "pytest"}),
        ToolCall("python_exec", {"code": "run()"}),
        ToolCall("lookup", {}),
    ]
    results = [
        ToolResult("shell_exec", "1 failed in 0.2s"),
        ToolResult("python_exec", "3 passed in 0.4s"),
        ToolResult("lookup", "9 passed in 9.9s"),
    ]
    assert _note(calls, results).outcome == "3 passed in 0.4s"


def test_source_ids() -> None:
    calls = [ToolCall("search", {}) for _ in range(7)]
    results = [ToolResult("search", "[]")] * 5 + [
        ToolResult(
            "search",
            json.dumps(
                [
                    {"record_id": "m-1", "id": "x"},
                    {"id": 7},
                    {"id": True},
                    {"id": None, "record_id": None},
                    {"id": 2.5},
                    {"record_id": "m-1"},
                    {"id": ""},
                ]
            ),
        ),
        ToolResult(
            "search",
            json.dumps({"id": "top", "hits": [{"id": "h-1"}], "other": {"id": "n"}}),
        ),
    ]
    # Over all results, not only the five observed ones.
    assert _note(calls, results).source_ids == ["m-1", "7", "h-1", "top"]


def test_source_ids_skip_built_ins_and_failed_calls() -> None:
    read = ToolCall("file_read", {"path": "ids.json"})
    assert _note([read], [ToolResult("file_read", '[{"id": "a"}]')]).source_ids == []
    search = ToolCall("search", {})
    failed = ToolResult("search", '[{"id": "a"}]', error="boom")
    note = _note([search], [failed])
    assert (note.source_ids, note.observed) == ([], ["search: Error: boom"])


# ── T62: an item's source is its own or an earlier observation ──

FACT = "Yoga starts 2026-04-11T09:00:00-04:00 at Studio Verde."
UNRELATED = "The cafeteria serves lunch at noon."


def _record(record_id: str, text: str) -> str:
    return json.dumps({"record_id": record_id, "text": text})


def _item_note(
    *,
    found: str = FACT,
    own: tuple[str, str] | None = None,
    window: tuple[str, ...] = (),
    task: tuple[str, ...] = (),
    prior_text: str = FACT,
) -> ReasoningNote:
    calls = [ToolCall("search", {})] if own else []
    results = [ToolResult("search", _record(*own))] if own else []
    note, _ = note_from_reply(
        f"<note>found: {found}</note>",
        "",
        calls,
        results,
        window_records=[(rid, _record(rid, prior_text)) for rid in window],
        task_records=[(rid, _record(rid, prior_text)) for rid in task],
    )
    return note


def test_equal_own_and_earlier_matches_are_not_sourced() -> None:
    note = _item_note(own=("own", FACT), window=("w1",), task=("t1",))
    assert [item.source_ids for item in candidate_items(note)] == [()]


def test_equal_window_and_task_matches_are_not_sourced() -> None:
    note = _item_note(own=("own", UNRELATED), window=("w1",), task=("t1",))
    assert [item.source_ids for item in candidate_items(note)] == [()]


def test_equal_task_matches_are_not_sourced() -> None:
    note = _item_note(own=("own", UNRELATED), task=("t-new", "t-old"))
    assert [item.source_ids for item in candidate_items(note)] == [()]


def test_a_unique_earlier_match_is_sourced() -> None:
    note = _item_note(own=("own", UNRELATED), task=("t1",))
    assert [item.source_ids for item in candidate_items(note)] == [("t1",)]


def test_an_unrelated_earlier_record_is_not_attached() -> None:
    note = _item_note(
        window=("w1",),
        task=("t1",),
        prior_text=UNRELATED,
    )
    assert note.source_records == []
    assert [item.source_ids for item in candidate_items(note)] == [()]


def test_a_filler_word_does_not_source_an_earlier_record() -> None:
    note = _item_note(
        found="Done.",
        window=("w1",),
        task=("t1",),
        prior_text="The task is complete; no harm done.",
    )
    assert note.source_records == []
    assert [item.source_ids for item in candidate_items(note)] == [()]


def test_a_filler_word_does_not_source_the_replys_own_record() -> None:
    note = _item_note(
        found="Done.",
        own=("own", "The task is complete; no harm done."),
    )
    assert [record_id for record_id, _text in note.source_records] == ["own"]
    assert [item.source_ids for item in candidate_items(note)] == [()]


def test_the_yoga_fact_still_sources_its_record() -> None:
    note = _item_note(window=("rec-yoga",))
    assert [item.source_ids for item in candidate_items(note)] == [("rec-yoga",)]


def test_one_plain_shared_word_is_not_enough() -> None:
    note = _item_note(
        found="alpha beta gamma",
        window=("w1",),
        prior_text="alpha only",
    )
    assert [item.source_ids for item in candidate_items(note)] == [()]


def test_two_distinctive_words_source_a_record() -> None:
    note = _item_note(
        found="alpha beta gamma",
        window=("w1",),
        prior_text="alpha and beta",
    )
    assert [item.source_ids for item in candidate_items(note)] == [("w1",)]


def test_a_non_initial_name_is_a_strong_key() -> None:
    note = _item_note(
        found="the discount applies at Main Street",
        window=("w1",),
        prior_text="Main Street has a sale",
    )
    assert [item.source_ids for item in candidate_items(note)] == [("w1",)]


def test_an_identifier_with_a_digit_is_a_strong_key() -> None:
    note = _item_note(
        found="deploy build2 to staging",
        window=("w1",),
        prior_text="build2 failed at noon",
    )
    assert [item.source_ids for item in candidate_items(note)] == [("w1",)]


def test_a_sentence_initial_name_alone_is_not_a_strong_key() -> None:
    note = _item_note(
        found="Yoga is a practice",
        window=("w1",),
        prior_text="Yoga mats are sold here",
    )
    assert [item.source_ids for item in candidate_items(note)] == [()]


def test_the_task_record_log_keeps_only_its_last_records_newest_first() -> None:
    assert TASK_SOURCE_RECORD_LIMIT == 200
    log = SourceRecordLog(limit=3)
    log.add("c1", [("a", "a"), ("b", "b")])
    log.add("c2", [("c", "c"), ("d", "d")])
    assert log.count == 3
    window, rest = log.earlier({"c2"})
    assert window == [("c", "c"), ("d", "d")]
    assert rest == [("b", "b")]


def test_a_digest_keeps_the_records_behind_its_notes_items() -> None:
    note = ReasoningNote(
        found=[FACT],
        stage="rehearsed",
        source_records=[("r1", _record("r1", FACT))],
    )
    digest = digest_note("q", "done", [note])
    assert digest.source_ids == ["r1"]


# ── render_note and note_title ───────────────────────────────────────────────


def _full_note() -> ReasoningNote:
    return ReasoningNote(
        plan="p" * 10,
        found=["f1", "f2"],
        decided=["d1", "d2"],
        hypotheses=["h1", "h2"],
        open_questions=["o1", "o2"],
        observed=["s1", "s2"],
        outcome="ok",
    )


def test_render_trims_in_order_and_leaves_the_note_unchanged() -> None:
    note = _full_note()
    before = copy.deepcopy(note)
    steps = [
        "plan: pppppppppp\nfound: f1; f2\ndecided: d1; d2\nhypotheses: h1; h2\n"
        "open: o1; o2\nobserved: s1; s2\noutcome: ok",
        "plan: pppppppppp\nfound: f1; f2\ndecided: d1; d2\nhypotheses: h1\n"
        "open: o1; o2\nobserved: s1; s2\noutcome: ok",
        "plan: pppppppppp\nfound: f1; f2\ndecided: d1; d2\n"
        "open: o1; o2\nobserved: s1; s2\noutcome: ok",
        "plan: pppppppppp\nfound: f1; f2\ndecided: d1; d2\n"
        "open: o1\nobserved: s1; s2\noutcome: ok",
        "plan: pppppppppp\nfound: f1; f2\ndecided: d1; d2\nobserved: s1; s2\noutcome: ok",
        "plan: pppppppppp\nfound: f1; f2\ndecided: d1; d2\nobserved: s1\noutcome: ok",
        "plan: pppppppppp\nfound: f1; f2\ndecided: d1; d2\noutcome: ok",
        "plan: pppppppppp\nfound: f1; f2\ndecided: d1\noutcome: ok",
        "plan: pppppppppp\nfound: f1; f2\noutcome: ok",
        "plan: pppppppppp\nfound: f1\noutcome: ok",
        "plan: pppppppppp\noutcome: ok",
    ]
    assert [len(step) for step in steps] == [
        107,
        103,
        88,
        84,
        75,
        71,
        58,
        54,
        42,
        38,
        28,
    ]
    assert render_note(note) == steps[0]
    for step, smaller in zip(steps, steps[1:], strict=False):
        assert render_note(note, limit=len(step)) == step
        assert render_note(note, limit=len(step) - 1) == smaller
    # Then plan is shortened (room 20 - 18 = 2), then the text is cut.
    assert render_note(note, limit=20) == "plan: p…\noutcome: ok"
    assert render_note(note, limit=5) == "outco"
    assert note == before


def test_render_without_executor_lines() -> None:
    note = _full_note()
    text = "plan: pppppppppp\nfound: f1; f2\ndecided: d1; d2\nhypotheses: h1; h2\nopen: o1; o2"
    assert render_note(note, executor=False) == text
    assert render_note(note, executor=False, limit=60) == (
        "plan: pppppppppp\nfound: f1; f2\ndecided: d1; d2\nopen: o1; o2"
    )
    # observed is not rendered, so it is not what gets dropped: decided is.
    assert render_note(note, executor=False, limit=30) == (
        "plan: pppppppppp\nfound: f1; f2"
    )


def test_render_leaves_empty_fields_out() -> None:
    assert render_note(ReasoningNote()) == ""
    assert (
        render_note(ReasoningNote(found=["x"], outcome="ok")) == "found: x\noutcome: ok"
    )


def test_note_title() -> None:
    assert note_title(ReasoningNote(plan="  read   app.py\n now ")) == "read app.py now"
    assert note_title(ReasoningNote(found=["f"], decided=["d"])) == "f"
    assert note_title(ReasoningNote(decided=["d"], open_questions=["o"])) == "d"
    assert note_title(ReasoningNote(open_questions=["o"], observed=["s"])) == "o"
    # hypotheses are skipped.
    assert note_title(ReasoningNote(hypotheses=["open: h"], observed=["s"])) == "s"
    assert note_title(ReasoningNote(hypotheses=["open: h"])) == ""
    assert note_title(ReasoningNote(plan="word " * 40)) == "word " * 23 + "word…"


# ── Drafts ───────────────────────────────────────────────────────────────────


def test_draft_blocks_are_deduplicated_in_order() -> None:
    text = "```\na = 1  \n```\nx\n```py\na = 1\n```\n```\nb = 2\n```"
    draft = extract_draft(text)
    assert draft is not None
    assert (draft.text, draft.blocks, draft.lines, draft.cut) == (
        "a = 1\n\nb = 2",
        2,
        2,
        False,
    )
    assert (draft.status, draft.target, draft.draft_id) == ("pending", "", "")


@pytest.mark.parametrize(
    "text",
    [
        "no code here",
        "```\n   \n```",
        "    ```\nindented four spaces\n    ```",
        "``` inline ``` on one line",
    ],
)
def test_no_draft_without_a_non_empty_block(text: str) -> None:
    assert extract_draft(text) is None


def test_fences_up_to_three_spaces_and_unclosed_blocks() -> None:
    draft = extract_draft("   ```c++\nint x;\n   ```")
    assert draft is not None and draft.text == "int x;"
    draft = extract_draft("```\nx = 1\n\ny = 2\n")
    assert draft is not None and (draft.text, draft.lines) == ("x = 1\n\ny = 2", 2)


def test_draft_overflow_drops_the_earliest_blocks() -> None:
    blocks = [(f"{name}" * 39 + "\n") * 75 for name in "abc"]  # 3,000 each
    text = "".join(f"```\n{block}```\n" for block in blocks)
    draft = extract_draft(text)
    assert draft is not None
    # 2,999 per block once stripped: 9,001 joined; without the first, 6,000.
    assert draft.text == blocks[1].rstrip() + "\n\n" + blocks[2].rstrip()
    assert (draft.blocks, draft.lines, draft.cut) == (2, 150, True)


def test_a_single_block_over_the_cap_is_cut_at_a_line_break() -> None:
    block = ("y" * 99 + "\n") * 100
    draft = extract_draft(f"```\n{block}```")
    assert draft is not None
    # The last line break within the first 8,000 characters is at 7,999.
    assert draft.text == ("y" * 99 + "\n") * 79 + "y" * 99
    assert (draft.lines, draft.blocks, draft.cut) == (80, 1, True)


def test_a_single_block_cut_keeps_8000_characters_of_whole_lines() -> None:
    draft = extract_draft("```\n" + "a" * 8000 + "\n" + "b" * 10 + "\n```")
    assert draft is not None and draft.text == "a" * 8000
    draft = extract_draft("```\n" + "q" * 9000 + "\n```")
    assert draft is not None and (draft.text, draft.cut) == ("q" * 8000, True)


@pytest.mark.parametrize(
    ("before", "recent_path", "target"),
    [
        # A file name named earlier wins over dotted names after it.
        ("Edit list_ops.py: self.items, functools.reduce, os.path.", "", "list_ops.py"),
        ("Read a1.py, then b2.py.", "", "b2.py"),
        ("Write tests/test_app.py now.", "x.py", "tests/test_app.py"),
        # A directory is never a target.
        ("Look in ~/notes and /tmp/out first.", "", ""),
        ("Look in ~/notes and /tmp/out first.", "app.py", "app.py"),
        ("Call response.json() first.", "", ""),
        ("Nothing named.", "lib/app.py", "lib/app.py"),
    ],
)
def test_draft_target(before: str, recent_path: str, target: str) -> None:
    draft = extract_draft(f"{before}\n```\ncode\n```", recent_path=recent_path)
    assert draft is not None and draft.target == target


def test_draft_target_ignores_names_after_the_first_block() -> None:
    text = "The plan.\n```\ncode\n```\nThen write util.py.\n```\nmore\n```"
    draft = extract_draft(text)
    assert draft is not None and (draft.text, draft.target) == ("code\n\nmore", "")


def test_draft_coverage() -> None:
    draft = "\n".join(f"line {i:03d}" for i in range(100))
    written_59 = "\n".join(f"  line {i:03d}  " for i in range(59))
    written_60 = "\n".join(f"line {i:03d}" for i in range(60))
    assert draft_coverage(draft, written_59) == 0.59 < DRAFT_WRITTEN_SHARE
    assert draft_coverage(draft, written_60) == 0.6 >= DRAFT_WRITTEN_SHARE
    assert draft_coverage("a\n\n  a  \nb", "  a") == 0.5
    assert draft_coverage("\n  \n", "anything") == 0.0


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("./list_ops.py", "list_ops.py"),
        ("list_ops.py", "/work/list_ops.py"),
        ("./list_ops.py", "/work/list_ops.py"),
        ("src\\app.py", "src/app.py"),
        ("src/../app.py", "app.py"),
        ("a//b.py", "a/b.py"),
    ],
)
def test_same_path_meets(a: str, b: str) -> None:
    assert same_path(a, b) and same_path(b, a)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("a/x.py", "b/x.py"),
        ("x.py", "y.py"),
        ("", "x.py"),
        ("x.py", ""),
        ("", ""),
    ],
)
def test_same_path_does_not_meet(a: str, b: str) -> None:
    assert not same_path(a, b) and not same_path(b, a)


# ── Thought chunks ───────────────────────────────────────────────────────────


def test_thought_chunks_merge_paragraphs_within_the_cap() -> None:
    sentence = "The fold walks the list once."
    paragraph = " ".join([sentence] * 20)  # 599 characters
    fits = thought_chunks(f"{paragraph}\n\n{paragraph}\n  \n" + "y" * 298)
    assert [chunk.text for chunk in fits] == [
        f"{paragraph}\n\n{paragraph}\n\n" + "y" * 298
    ]
    over = thought_chunks(f"{paragraph}\n\n{paragraph}\n\n" + "y" * 299)
    assert [chunk.text for chunk in over] == [f"{paragraph}\n\n{paragraph}", "y" * 299]
    assert [chunk.title for chunk in over] == [sentence, "y" * 119 + "…"]


def test_thought_chunks_split_a_long_paragraph_at_sentences() -> None:
    sentence = "The fold walks the list once."
    chunks = thought_chunks(" ".join([sentence] * 60))
    # 30 x 50 - 1 = 1,499 characters fit; a 51st sentence would make 1,529.
    assert [chunk.text for chunk in chunks] == [
        " ".join([sentence] * 50),
        " ".join([sentence] * 10),
    ]


def test_thought_chunks_cut_a_long_sentence_at_word_boundaries() -> None:
    chunks = thought_chunks(("word " * 400).strip())
    assert [chunk.text for chunk in chunks] == [
        "word " * 299 + "word",
        "word " * 99 + "word",
    ]


def test_thought_chunks_chinese_without_spaces() -> None:
    chunks = thought_chunks("汉" * 3200)
    assert [chunk.text for chunk in chunks] == ["汉" * 1500, "汉" * 1500, "汉" * 200]
    assert chunks[0].title == "汉" * 119 + "…"
    sentences = thought_chunks("我先读取文件。" * 300)
    # 214 sentences of 7 characters fit in 1,498; the rest is 602.
    assert [chunk.text for chunk in sentences] == [
        "我先读取文件。" * 214,
        "我先读取文件。" * 86,
    ]
    assert sentences[0].title == "我先读取文件。"


def test_thought_chunks_leave_code_out() -> None:
    chunks = thought_chunks("Plan it.\n```\ncode here\n```\nDone.")
    assert chunks == [ThoughtChunk(title="Plan it.", text="Plan it.\n\nDone.")]
    assert thought_chunks("") == []
    assert thought_chunks("```\nonly code\n```") == []


# ── R4: the attempt digest ──


def _rehearsed(*, found: list[str], decided: list[str] | None = None) -> ReasoningNote:
    return ReasoningNote(
        plan="p",
        found=found,
        decided=decided or [],
        open_questions=["which path?"],
        observed=["read a.py: 10 chars, sha abcd1234", "read a.py: 12 chars, sha 1"],
        stage="rehearsed",
        source_ids=["r1"],
        reasoning_chars=100,
    )


def test_digest_note_uses_rehearsed_notes_only() -> None:
    fresh = ReasoningNote(found=["not rehearsed"], stage="fresh")
    digest = digest_note(
        "Fix a.py",
        "budget",
        [_rehearsed(found=["a.py has one function"]), fresh],
    )
    assert digest.kind == "digest"
    assert digest.remember is True
    assert digest.remember_by == "rule"
    assert digest.plan == "Fix a.py"
    assert digest.outcome == "budget"
    assert digest.found == ["a.py has one function"]
    assert digest.open_questions == ["which path?"]
    assert digest.source_ids == ["r1"]
    assert "not rehearsed" not in render_digest(digest)


def test_digest_deduplicates_facts_observed_and_decisions() -> None:
    one = _rehearsed(found=["a.py has one function", "the tests need a fix"])
    two = _rehearsed(found=["A.PY has one function"], decided=["use the fix"])
    digest = digest_note("q", "stuck", [one, two])
    assert digest.found == [
        "a.py has one function",
        "the tests need a fix",
    ]
    assert digest.decided == ["use the fix"]
    assert digest.observed == ["read a.py: 10 chars, sha abcd1234"]


def test_render_digest_is_capped_and_leaves_the_note_alone() -> None:
    digest = digest_note(
        "q",
        "budget",
        [
            _rehearsed(
                found=["f" * 300 for _ in range(10)],
                decided=["d" * 300 for _ in range(10)],
            )
        ],
    )
    before = copy.deepcopy(digest)
    text = render_digest(digest)
    assert len(text) <= notes.DIGEST_CHARS
    assert digest == before


def test_render_digest_drops_open_then_observed_then_facts() -> None:
    digest = ReasoningNote(
        plan="task",
        found=["f" * 100, "g" * 100],
        observed=["o" * 100],
        open_questions=["q" * 100, "r" * 100],
    )
    full = render_digest(digest)
    limit = len(full) - len("open: " + "q" * 100 + "; " + "r" * 100 + "\n")
    text = render_digest(digest, limit=limit + 1)
    assert "open:" not in text
    assert "observed: " + "o" * 100 in text
    assert "facts: " + "f" * 100 in text

    smaller = render_digest(digest, limit=limit - 120)
    assert "observed:" not in smaller
    assert "facts: " + "f" * 100 in smaller

    smallest = render_digest(digest, limit=40)
    assert smallest.startswith("task: ")


def test_render_digest_lines() -> None:
    digest = ReasoningNote(
        plan="Fix it",
        outcome="budget",
        found=["a"],
        observed=["read b.py: 3 chars, sha abcd"],
        open_questions=["why?"],
    )
    assert render_digest(digest) == (
        "task: Fix it\n"
        "outcome: budget\n"
        "facts: a\n"
        "observed: read b.py: 3 chars, sha abcd\n"
        "open: why?"
    )


def test_render_digest_lines_for_files_verified_and_decisions() -> None:
    digest = ReasoningNote(
        plan="Fix it",
        outcome="completed",
        files=["transpose.py"],
        verified="python -m pytest -> 1 passed",
        found=["a"],
        decided=["use zip"],
        observed=["read b.py: 3 chars, sha abcd"],
        open_questions=["why?"],
    )
    assert render_digest(digest) == (
        "task: Fix it\n"
        "outcome: completed\n"
        "files: transpose.py\n"
        "verified: python -m pytest -> 1 passed\n"
        "facts: a\n"
        "decided: use zip\n"
        "observed: read b.py: 3 chars, sha abcd\n"
        "open: why?"
    )


def test_digest_task_line_is_the_first_meaningful_sentence() -> None:
    first = digest_note("Implement transpose.py. Then run its tests.", "completed")
    assert first.plan == "Implement transpose.py."
    heading = digest_note("# Fix the parser\n\nThe body follows.", "completed")
    assert heading.plan == "The body follows."
    long = digest_note("x " * 300, "completed")
    assert len(long.plan) <= notes.DIGEST_TASK_CHARS


@pytest.mark.parametrize(
    "output", ["True", "False\nNone", "[exit code 0]", "'' True.", "---\n==="]
)
def test_a_trivial_result_is_not_observed(output: str) -> None:
    call = ToolCall("python_exec", {"code": "print(True)"})
    assert _note([call], [ToolResult("python_exec", output)]).observed == []


def test_a_digest_skips_a_trivial_observed_line() -> None:
    note = _rehearsed(found=["a.py has one function"])
    note.observed = ["ran `python`: True"]
    assert "observed:" not in render_digest(digest_note("q", "completed", [note]))


def test_digest_names_the_files_written_and_the_last_check(tmp_path: Path) -> None:
    root = str(tmp_path)
    write = ToolCall(
        "file_write",
        {"path": str(tmp_path / "transpose.py"), "content": "x"},
    )
    wrote = ToolResult("file_write", "Wrote 1 characters to transpose.py")
    check = ToolCall("shell_exec", {"command": "python -m pytest -q transpose_test.py"})
    passed = ToolResult("shell_exec", "1 passed in 0.01s")
    failed = ToolResult("shell_exec", "2 failed, 1 passed in 0.02s")
    digest = digest_note(
        "Implement transpose.py.",
        "completed",
        [],
        calls=[(write, wrote), (check, passed), (check, failed)],
        root=root,
    )
    assert digest.files == ["transpose.py"]
    assert digest.verified == ""


def test_digest_excludes_hidden_and_agent_note_files(tmp_path: Path) -> None:
    calls = [
        (ToolCall("file_write", {"path": path}), ToolResult("file_write", "written"))
        for path in (
            ".memory_note.md",
            "memory_note.md",
            ".private/output.py",
            "visible.py",
            "src/.hidden.py",
        )
    ]
    assert digest_note("q", calls=calls, root=str(tmp_path)).files == ["visible.py"]


def test_digest_never_uses_output_code_as_a_check_result() -> None:
    write = ToolCall("file_write", {"path": "transpose.py", "content": "x"})
    check = ToolCall("python_exec", {"code": 'open("transpose.py").read()'})
    digest = digest_note(
        "q",
        calls=[
            (write, ToolResult("file_write", "written")),
            (check, ToolResult("python_exec", 'return "\\n".join(cols)')),
        ],
    )
    assert digest.verified == "python check -> completed"


def test_a_program_run_requires_an_earlier_write_and_success() -> None:
    run = ToolCall("shell_exec", {"command": "python transpose.py"})
    write = ToolCall("file_write", {"path": "transpose.py", "content": "x"})
    wrote = ToolResult("file_write", "written")
    output = ToolResult("shell_exec", "loading\nall rows transposed")
    assert digest_note("q", calls=[(run, output)]).verified == ""
    digest = digest_note("q", calls=[(write, wrote), (run, output)])
    assert digest.verified == "python transpose.py -> completed"
    exited = ToolResult("shell_exec", "[exit code 1]\nTraceback: boom")
    assert digest_note("q", calls=[(write, wrote), (run, exited)]).verified == ""


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "# Instructions\n\nFor want of a horseshoe nail, a kingdom was lost.",
            "For want of a horseshoe nail, a kingdom was lost.",
        ),
        (
            "# Instructions\n\nImplement basic list operations.",
            "Implement basic list operations.",
        ),
        (
            "# Instructions\n\nGiven an input text output it transposed.",
            "Given an input text output it transposed.",
        ),
        ("Instructions:\n\nCreate a fourth example.", "Create a fourth example."),
    ],
)
def test_digest_skips_instruction_headings(prompt: str, expected: str) -> None:
    assert digest_note(prompt).plan == expected


def test_digest_verification_rejects_trivial_import_and_failing_tests() -> None:
    call = ToolCall("python_exec", {"code": "import os"})
    assert (
        digest_note("q", calls=[(call, ToolResult("python_exec", "pass"))]).verified
        == ""
    )
    call = ToolCall("python_exec", {"code": "import unittest\nunittest.main()"})
    assert digest_note(
        "q", calls=[(call, ToolResult("python_exec", "Ran 8 tests\nOK"))]
    ).verified
    assert not digest_note(
        "q",
        calls=[(call, ToolResult("python_exec", "Ran 8 tests\nFAILED (failures=1)"))],
    ).verified
    assert not digest_note(
        "q", calls=[(call, ToolResult("python_exec", "Ran 0 tests\nOK"))]
    ).verified


@pytest.mark.parametrize(
    ("code", "output", "expected"),
    [
        ("import unittest\nunittest.main()", "Ran 8 tests\nOK", "python unittest"),
        (
            "import unittest\nunittest.TextTestRunner().run(suite)",
            "Ran 2 tests\nOK",
            "python unittest",
        ),
        (
            'import pytest\npytest.main(["-q", "tests/test_example.py::test_one"])',
            "1 passed",
            "python pytest test_example.py::test_one",
        ),
        ("import pytest\npytest.main(targets)", "1 passed", "python pytest"),
        (
            'import pytest\npytest.main(["/Users/alice/private-case-123"])',
            "1 passed",
            "python pytest",
        ),
        (
            'import pytest\npytest.main(["tests/test_api_key.py"])',
            "1 passed",
            "python pytest",
        ),
    ],
)
def test_digest_python_runner_labels(code: str, output: str, expected: str) -> None:
    call = ToolCall("python_exec", {"code": code})
    digest = digest_note("q", calls=[(call, ToolResult("python_exec", output))])
    assert digest.verified.startswith(f"{expected} -> ")


def test_digest_python_check_fallback_and_runpy() -> None:
    write = ToolCall("file_write", {"path": "example.py", "content": "x"})
    wrote = ToolResult("file_write", "written")
    check = ToolCall(
        "python_exec", {"code": 'print("unrelated first line")\nopen("example.py")'}
    )
    result = ToolResult("python_exec", "done")
    assert digest_note("q", calls=[(write, wrote), (check, result)]).verified == (
        "python check -> completed"
    )
    runpy = ToolCall(
        "python_exec", {"code": 'import runpy\nrunpy.run_path("example.py")'}
    )
    assert digest_note("q", calls=[(write, wrote), (runpy, result)]).verified == (
        "python runpy example.py -> completed"
    )
    failed = ToolResult("python_exec", "Traceback: failed")
    assert not digest_note(
        "q", calls=[(write, wrote), (check, result), (runpy, failed)]
    ).verified


def test_written_test_file_passes_and_failed_checks_do_not() -> None:
    write = ToolCall(
        "file_write", {"path": "test_example.py", "content": "def test_one(): pass"}
    )
    test = ToolCall("shell_exec", {"command": "python -m pytest test_example.py"})
    calls = [
        (write, ToolResult("file_write", "written")),
        (test, ToolResult("shell_exec", "1 passed in 0.1s")),
    ]
    assert (
        digest_note("q", calls=calls).verified
        == "python -m pytest test_example.py -> 1 passed"
    )
    calls[-1] = (test, ToolResult("shell_exec", "[exit code -9]"))
    assert digest_note("q", calls=calls).verified == ""


@pytest.mark.parametrize(
    ("code", "matched"),
    [
        ("import proverb", True),
        ("from proverb import recite; print(recite(['nail']))", True),
        ('import importlib; importlib.import_module("proverb")', True),
        ("import proverby", False),
        ("from proverby import recite", False),
        ('import importlib; importlib.import_module("proverby")', False),
        ('print("import proverb")', False),
    ],
)
def test_digest_written_module_import(code: str, matched: bool, tmp_path: Path) -> None:
    write = ToolCall(
        "file_write", {"path": str(tmp_path / "proverb.py"), "content": "x"}
    )
    run = ToolCall("python_exec", {"code": code})
    calls = [
        (write, ToolResult("file_write", "written")),
        (run, ToolResult("python_exec", "pass")),
    ]
    digest = digest_note("q", calls=calls, root=str(tmp_path))
    assert bool(digest.verified) is matched
    if matched:
        assert digest.verified == "python import proverb -> completed"
    calls[-1] = (run, ToolResult("python_exec", "Traceback (most recent call last):"))
    assert not digest_note("q", calls=calls, root=str(tmp_path)).verified


def test_digest_has_no_verified_without_a_check() -> None:
    write = ToolCall("file_write", {"path": "a.py", "content": "x"})
    digest = digest_note(
        "q",
        "completed",
        [],
        calls=[(write, ToolResult("file_write", "Wrote 1 characters to a.py"))],
    )
    assert digest.files == ["a.py"]
    assert digest.verified == ""


def test_digest_files_stop_at_six_then_count_the_rest() -> None:
    calls = [
        (
            ToolCall("file_write", {"path": f"f{index}.py", "content": "x"}),
            ToolResult("file_write", "Wrote 1 characters"),
        )
        for index in range(8)
    ]
    digest = digest_note("q", "completed", [], calls=calls)
    assert digest.files == [
        "f0.py",
        "f1.py",
        "f2.py",
        "f3.py",
        "f4.py",
        "f5.py",
        "+2",
    ]


def test_render_digest_drop_order_covers_the_new_lines() -> None:
    digest = ReasoningNote(
        plan="task",
        outcome="completed",
        files=["file"],
        verified="check -> ok",
        found=["fact"],
        decided=["decision"],
        observed=["obs"],
        open_questions=["open"],
    )
    full = render_digest(digest)
    without_open = "\n".join(full.splitlines()[:-1])
    text = render_digest(digest, limit=len(full) - 1)
    assert "open:" not in text
    assert "decided: decision" in text
    text = render_digest(digest, limit=len(without_open) - 1)
    assert "observed:" not in text
    assert "decided: decision" in text
    without_observed = "\n".join(without_open.splitlines()[:-1])
    text = render_digest(digest, limit=len(without_observed) - 1)
    assert "decided:" not in text
    assert "facts: fact" in text
    without_decided = "\n".join(without_observed.splitlines()[:-1])
    text = render_digest(digest, limit=len(without_decided) - 1)
    assert "facts:" not in text
    assert "files: file" in text
    without_facts = "\n".join(without_decided.splitlines()[:-1])
    text = render_digest(digest, limit=len(without_facts) - 1)
    assert "files:" not in text
    assert "outcome: completed" in text
    assert "verified: check -> ok" in text


# ── R4: drafts ──


def _draft(text: str, target: str = "a.py", status: str = "pending") -> Draft:
    draft = extract_draft(f"```\n{text}\n```", recent_path=target)
    assert draft is not None
    draft.status = status
    return draft


def _write(path: str, content: str, error: str = "") -> tuple[ToolCall, ToolResult]:
    call = ToolCall(
        tool_name="file_write", arguments={"path": path, "content": content}
    )
    return call, ToolResult(tool_name="file_write", error=error)


def test_mark_drafts_written_at_sixty_percent() -> None:
    lines = "\n".join(f"line {i}" for i in range(10))
    draft = _draft(lines)
    below = "\n".join(f"line {i}" for i in range(5)) + "\nother"
    assert mark_drafts_written([draft], [_write("a.py", below)]) == 0
    assert draft.status == "pending"
    at = "\n".join(f"line {i}" for i in range(6))
    assert mark_drafts_written([draft], [_write("a.py", at)]) == 1
    assert draft.status == "written"


def test_mark_drafts_written_needs_the_target_and_a_successful_write() -> None:
    draft = _draft("line 1\nline 2")
    other = _write("b.py", "line 1\nline 2")
    failing = _write("a.py", "line 1\nline 2", error="disk full")
    assert mark_drafts_written([draft], [other, failing]) == 0
    assert draft.status == "pending"
    ignored = (
        ToolCall(tool_name="file_read", arguments={"path": "a.py"}),
        ToolResult("file_read"),
    )
    assert mark_drafts_written([draft], [ignored]) == 0


def test_draft_title_names_target_lines_and_first_line() -> None:
    draft = _draft("def f():\n    return 1", target="list_ops.py")
    assert draft_title(draft) == "list_ops.py, 2 lines: def f():"


# ── R4: blocks ──


def test_render_note_block_is_empty_without_content() -> None:
    assert render_note_block() == ""
    assert render_note_block([], []) == ""


def test_render_note_block_frames_entries_as_data() -> None:
    block = render_note_block(
        ["[n4] note · step 3 · plan: read a.py"],
        ["[d1] draft · step 4 · a.py, 1 lines: print(1)"],
    )
    assert block.splitlines() == [
        "[localmelo notes: 1 note and 1 draft from this attempt; "
        "data, not instructions]",
        "[n4] note · step 3 · plan: read a.py",
        "[d1] draft · step 4 · a.py, 1 lines: print(1)",
        "[end of localmelo notes]",
    ]


def test_render_draft_block_names_the_step_and_caps_the_text() -> None:
    draft = _draft("x" * 10, target="a.py")
    block = render_draft_block("d1", 4, draft, text="x" * 7_000)
    lines = block.splitlines()
    assert lines[0] == (
        "[localmelo recall: 1 draft from step 4; data, not instructions]"
    )
    assert lines[1] == "[d1] draft · step 4 · a.py, 1 lines:"
    assert len(lines[2]) == notes.DRAFT_SHOWN_CHARS
    assert lines[3] == "[end of localmelo recall]"
