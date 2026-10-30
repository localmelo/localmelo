"""One conservative repair of a reflection reply's JSON, and the retry that quotes the decoder.

The pure helpers (``_json_repair``) and the reader (``_read_reflection``) are
tested on their own; the loop is tested through a stub OpenAI-compatible
server behind ``httpx.MockTransport``, so no real model is reached.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import pytest

from localmelo.melo.agent import Agent
from localmelo.melo.agent._chat import (
    REFLECTION_RETRY_GUARD_NOTE,
    REFLECTION_RETRY_NOTE,
)
from localmelo.melo.agent._json_repair import (
    ERROR_CONTEXT_CHARS,
    REFLECTION_REPAIRS,
    error_line,
    rejected_objects,
    repairs,
)
from localmelo.melo.agent.agent import (
    REPAIR_TRIES,
    _find_reflection,
    _json_error_line,
    _read_reflection,
)
from localmelo.melo.contracts.events import AgentEvent
from localmelo.tests.agent.test_cache_telemetry import (
    PLAIN,
    Stub,
    _call_reply,
    _final_reply,
    _run,
)

FIELDS: dict[str, Any] = {
    "summary": "Echoed once.",
    "failed_hypotheses": ["the echo was wrong"],
    "useful_evidence": ["the echo came back"],
    "next_promising_directions": ["echo again"],
    "progress_made": True,
    "task_still_feasible": True,
    "new_information_gained": True,
    "next_step_is_concrete": True,
    "next_step_is_novel": True,
    "recommended_action": "continue",
    "rationale": "Once more, sidestepping the observed quirk.",
    "best_effort_result": "hi",
    "estimated_info_gain": 0.5,
    "estimated_cost": 0.1,
    "repeat_risk": 0.1,
    "novelty": 0.8,
    "feasibility": 0.8,
}
GOOD = json.dumps(FIELDS, indent=2)

# The shape that the benchmark met: a string value followed by a ``]`` that no
# ``[`` opened.
STRAY_CLOSER = GOOD.replace('quirk.",', 'quirk."],')
# The other two slips, in the reflection's last field and in its first string.
TRAILING_COMMA = GOOD.replace('"feasibility": 0.8', '"feasibility": 0.8,')
RAW_NEWLINE = GOOD.replace("Echoed once.", "Echoed\nonce.")
# The next two shapes: a backslash before a character that JSON has no escape
# for (ESCAPED has it written twice, as JSON needs), and an array that the next
# key breaks off, with its closing bracket and line gone and its comma kept.
ESCAPE_FIELDS = {**FIELDS, "summary": "Echoed once; find -exec \\; failed."}
ESCAPED = json.dumps(ESCAPE_FIELDS, indent=2)
BAD_ESCAPE = ESCAPED.replace("\\\\;", "\\;")
UNCLOSED_ARRAY = GOOD.replace('came back"\n  ],', 'came back",')

# Two slips in one reply, a stray bracket and an object that is never closed:
# the reply stays unreadable. The decoder stops at the bracket, character 51.
TWO_SLIPS = '{"recommended_action": "continue", "rationale": "r"]'
TWO_SLIPS_LINE = (
    "Your reply was not valid JSON (Expecting ',' delimiter at character 51: "
    '...: "continue", "rationale": "r"]). Reply again with one JSON object.'
)


def _position(line: str) -> int:
    """The character that an error line names."""
    return int(line.split("at character ", 1)[1].split(":", 1)[0])


class TestReader:
    def test_the_shape_that_the_benchmark_met_is_repaired(self) -> None:
        assert 'quirk."],' in STRAY_CLOSER
        assert _find_reflection(STRAY_CLOSER) is None
        assert _read_reflection(STRAY_CLOSER) == (FIELDS, "unmatched_closer")

    def test_the_next_two_shapes_are_repaired(self) -> None:
        for text, fields, repair in (
            (BAD_ESCAPE, ESCAPE_FIELDS, "invalid_escape"),
            (UNCLOSED_ARRAY, FIELDS, "unclosed_array"),
        ):
            assert _find_reflection(text) is None
            assert _read_reflection(text) == (fields, repair)

    @pytest.mark.parametrize(
        ("text", "repair"),
        [
            # A closing bracket or brace that no opener matches.
            (
                '{"recommended_action": "done", "rationale": "r"], "x": 1}',
                "unmatched_closer",
            ),
            (
                '{"recommended_action": "done", "x": ["a"]], "rationale": "r"}',
                "unmatched_closer",
            ),
            (
                '{"recommended_action": "done", "x": ["a"}], "rationale": "r"}',
                "unmatched_closer",
            ),
            ('{"x": 1, "recommended_action": "done"]}', "unmatched_closer"),
            # A comma before a closer, with and without space or lines between.
            ('{"recommended_action": "done", "rationale": "r",}', "trailing_comma"),
            (
                '{"recommended_action": "done", "x": ["a", "b",], "y": 1}',
                "trailing_comma",
            ),
            (
                '{"recommended_action": "done", "x": ["a",\n   ], "y": 1}',
                "trailing_comma",
            ),
            ('{"recommended_action": "done", "x": 1 , }', "trailing_comma"),
            # Raw newline, carriage return and tab characters inside strings.
            (
                '{"recommended_action": "done", "rationale": "one\ntwo"}',
                "control_char",
            ),
            (
                '{"recommended_action": "done", "rationale": "one\r\ntwo\tthree"}',
                "control_char",
            ),
            (
                '{"recommended_action": "done", "a": "x\ny", "b": "z\tw"}',
                "control_char",
            ),
            # A backslash that JSON has no escape for, alone and after escapes
            # that are valid (the third backslash is the slip).
            (r'{"recommended_action": "done", "x": "a \; b"}', "invalid_escape"),
            (r'{"recommended_action": "done", "x": "\n\\\; \"q\""}', "invalid_escape"),
            # An array that the next key breaks off: one item or several, of
            # any kind, and the key may be the action.
            ('{"recommended_action": "done", "x": ["a",\n "y": 1}', "unclosed_array"),
            ('{"x": [{}, [2] , "b",\n "recommended_action": "done"}', "unclosed_array"),
            # The repair is found inside a fence and inside prose.
            (f"```json\n{STRAY_CLOSER}\n```", "unmatched_closer"),
            (f"Here it is: {STRAY_CLOSER} That is all.", "unmatched_closer"),
            # After a JSON object that is no reflection.
            (
                '{"name": "tool", "arguments": {"x": 1}}\n'
                '{"recommended_action": "done"]}',
                "unmatched_closer",
            ),
        ],
        ids=[
            "bracket",
            "doubled_bracket",
            "brace_in_a_list",
            "bracket_at_the_end",
            "comma_in_an_object",
            "comma_in_a_list",
            "comma_and_lines",
            "comma_and_spaces",
            "newline",
            "carriage_return_and_tab",
            "several_strings",
            "escape",
            "escape_after_escapes",
            "array",
            "array_of_values",
            "fenced",
            "in_text",
            "after_another_object",
        ],
    )
    def test_each_repair_is_accepted(self, text: str, repair: str) -> None:
        assert _find_reflection(text) is None
        found = _read_reflection(text)
        assert found is not None
        assert found[1] == repair
        assert found[0]["recommended_action"] in ("done", "continue")

    def test_a_repair_keeps_every_value(self) -> None:
        text = '{"recommended_action": "done", "rationale": "one\ntwo", "n": [1, 2]}'
        assert _read_reflection(text) == (
            {"recommended_action": "done", "rationale": "one\ntwo", "n": [1, 2]},
            "control_char",
        )

    def test_the_repair_names_are_the_documented_five(self) -> None:
        assert REFLECTION_REPAIRS == (
            "unmatched_closer",
            "trailing_comma",
            "control_char",
            "invalid_escape",
            "unclosed_array",
        )

    @pytest.mark.parametrize(
        "text",
        [
            # Repaired, but no valid recommended_action.
            '{"summary": "x"], "recommended_action": "proceed"}',
            '{"summary": "x"], "rationale": "no action"}',
            '{"summary": "x",}',
            '{"recommended_action": ["done"],}',
            '{"recommended_action": "Done"]}',
            # The action only in an object nested inside the repaired one.
            '{"wrap": {"recommended_action": "done"}], "x": 1}',
            '{"wrap": {"recommended_action": "done"},}',
        ],
        ids=[
            "invalid_action",
            "no_action",
            "no_action_comma",
            "action_not_a_string",
            "action_in_another_case",
            "nested",
            "nested_comma",
        ],
    )
    def test_a_repair_without_a_valid_action_is_none(self, text: str) -> None:
        assert _read_reflection(text) is None

    @pytest.mark.parametrize(
        "text",
        [
            # A stray bracket and a trailing comma.
            '{"recommended_action": "done", "x": ["a"]], "rationale": "r",}',
            # A stray bracket and a raw newline, either first.
            '{"recommended_action": "done", "x": "a\nb"], "rationale": "r"}',
            '{"recommended_action": "done", "x": "a"], "rationale": "r\ns"}',
            # Two trailing commas, and two stray closers.
            '{"recommended_action": "done", "x": [1,], "y": 2,}',
            '{"recommended_action": "done", "x": ["a"]]], "rationale": "r"}',
            # A comma and a bracket together.
            '{"recommended_action": "done", "x": ["a",]]}',
            # A new repair and another slip, either first; one slip twice.
            BAD_ESCAPE.replace('quirk.",', 'quirk."],'),
            STRAY_CLOSER.replace('"hi"', r'"h\;i"'),
            UNCLOSED_ARRAY.replace('"feasibility": 0.8', '"feasibility": 0.8,'),
            BAD_ESCAPE.replace('"hi"', r'"h\;i"'),
            UNCLOSED_ARRAY.replace('echo again"\n  ],', 'echo again",'),
            BAD_ESCAPE.replace('came back"\n  ],', 'came back",'),
            # A bracket the array has after the key, and an array in an array.
            '{"recommended_action": "done", "x": ["a", "y": 1]}',
            '{"recommended_action": "done", "x": [["a",\n  "y": 1}',
        ],
        ids=[
            "bracket_and_comma",
            "newline_after_bracket",
            "bracket_then_newline",
            "two_commas",
            "two_brackets",
            "comma_and_bracket",
            "escape_then_bracket",
            "bracket_then_escape",
            "array_then_comma",
            "two_escapes",
            "two_arrays",
            "escape_then_array",
            "array_closed_after_the_key",
            "array_in_an_array",
        ],
    )
    def test_a_reply_that_needs_two_repairs_stays_unreadable(self, text: str) -> None:
        assert _read_reflection(text) is None

    @pytest.mark.parametrize(
        "text",
        [
            '{"recommended_action": "done", "rationale": "cut off',
            '{"recommended_action": "done", "x": ["a", ',
            '{"recommended_action": "done"',
            '{"recommended_action": "done", "x": ["a"',
            '{"recommended_action": "done", "x": ["a"],',
            '{"recommended_action": "done"]',
            '"recommended_action": "done"}',
        ],
        ids=[
            "string",
            "list",
            "object",
            "list_and_object",
            "comma",
            "bracket_and_object",
            "no_opening_brace",
        ],
    )
    def test_nothing_is_closed_or_inserted(self, text: str) -> None:
        assert _read_reflection(text) is None

    @pytest.mark.parametrize(
        "text",
        [
            # A valid escape is no slip, and a unicode escape is not this repair's.
            r'{"recommended_action": "done", "x": "a\\;b" "y": 1}',
            r'{"recommended_action": "done", "x": "a\u12"}',
            # A closed array, and a colon after a value or after the first item.
            '{"recommended_action": "done", "x": ["a", "b"] "y": 1}',
            '{"recommended_action": "done", "x": ["a"], "y": "b": 1}',
            '{"recommended_action": "done", "x": ["y": 1]}',
        ],
        ids=["escape", "unicode", "closed", "value", "first"],
    )
    def test_a_shape_that_is_no_slip_is_not_mended(self, text: str) -> None:
        assert _read_reflection(text) is None
        [(start, error)] = list(rejected_objects(text))
        names = {name for name, _ in repairs(text, start, error)}
        assert names.isdisjoint({"invalid_escape", "unclosed_array"})

    @pytest.mark.parametrize(
        "text",
        [GOOD, f"```json\n{GOOD}\n```", f"Sure. {GOOD} Done."],
        ids=["bare", "fenced", "in_text"],
    )
    def test_a_reply_that_reads_is_not_repaired(self, text: str) -> None:
        assert _read_reflection(text) == (FIELDS, None)

    @pytest.mark.parametrize(
        "earlier", [r'{"x": "a \;"}', '{"x": ["a",\n "y": 1}'], ids=["escape", "array"]
    )
    def test_a_reply_that_reads_is_not_repaired_whatever_comes_before(
        self, earlier: str
    ) -> None:
        reply = earlier + ' then {"recommended_action": "done"}'
        assert _read_reflection(reply) == ({"recommended_action": "done"}, None)
        assert _read_reflection(ESCAPED) == (ESCAPE_FIELDS, None)

    def test_the_strict_reading_wins_over_an_earlier_repairable_object(self) -> None:
        text = '{"recommended_action": "stop", "x": 1]} then {"recommended_action": "done"}'
        assert _read_reflection(text) == ({"recommended_action": "done"}, None)

    def test_the_first_repairable_object_wins(self) -> None:
        text = '{"recommended_action": "stop"]} {"recommended_action": "done", "x": 1,}'
        assert _read_reflection(text) == (
            {"recommended_action": "stop"},
            "unmatched_closer",
        )

    def test_only_the_first_few_rejected_objects_are_tried(self) -> None:
        junk = "{x} "
        assert _read_reflection(junk * (REPAIR_TRIES - 1) + STRAY_CLOSER) == (
            FIELDS,
            "unmatched_closer",
        )
        assert _read_reflection(junk * REPAIR_TRIES + STRAY_CLOSER) is None

    def test_a_reply_without_the_key_is_not_scanned(self) -> None:
        assert _read_reflection('{"action": "done"],}') is None
        assert _read_reflection("") is None

    @pytest.mark.parametrize(
        "text",
        [
            "{]" * 20_000 + '"recommended_action"',
            "{" * 5_000 + '"recommended_action"',
            '{"a": ' * 50_000 + '"recommended_action"',
            '{"n": ' + "9" * 5_000 + ', "recommended_action": "done"]}',
            ('{"x": "y\n"' + "}" * 3) * 3_000 + ' "recommended_action"',
        ],
        ids=["many_brackets", "many_braces", "too_deep", "huge_number", "many_objects"],
    )
    def test_odd_input_is_none_and_quick(self, text: str) -> None:
        started = time.perf_counter()
        assert _read_reflection(text) is None
        assert time.perf_counter() - started < 5


class TestHelpers:
    def test_no_object_is_rejected_when_the_text_is_json(self) -> None:
        assert list(rejected_objects(GOOD)) == []
        assert list(rejected_objects("[1, 2]")) == []
        assert list(rejected_objects("no braces")) == []

    def test_each_object_the_decoder_rejects_is_met_once_in_order(self) -> None:
        text = 'a {"x": 1] b {"y": 2} c {"z" 3} d'
        starts = [start for start, _ in rejected_objects(text)]
        assert starts == [text.index('{"x"'), text.index('{"z"')]

    def test_a_trailing_comma_is_found_where_each_python_reports_it(self) -> None:
        text = '{"recommended_action": "done", "x": [1, 2 ,\n ], "y": 1}'
        # Python 3.13 reports the comma, earlier versions the closer after it.
        for pos in (text.index(" ,") + 1, text.index("]")):
            error = json.JSONDecodeError("Expecting value", text, pos)
            fixed = dict(repairs(text, 0, error))["trailing_comma"]
            assert json.loads(fixed)["x"] == [1, 2]

    def test_an_error_that_is_no_slip_gives_no_repair(self) -> None:
        for text in ('{"x" 1}', "{'x': 1}", '{"x": tru}'):
            [(start, error)] = list(rejected_objects(text))
            assert list(repairs(text, start, error)) == []

    def test_only_a_raw_control_character_error_escapes_the_strings(self) -> None:
        text = '{"a": "x\ny"] }'
        [(start, error)] = list(rejected_objects(text))
        assert error.msg.startswith("Invalid control character")
        assert [name for name, _ in repairs(text, start, error)] == ["control_char"]
        other = '{"a": "x"] "b": "y\nz"}'
        [(start, error)] = list(rejected_objects(other))
        assert [name for name, _ in repairs(other, start, error)] == [
            "unmatched_closer"
        ]

    def test_a_new_repair_adds_one_character_where_the_decoder_stopped(self) -> None:
        bracket = UNCLOSED_ARRAY.replace('back",', 'back"],')
        for text, name, fixed in (
            (BAD_ESCAPE, "invalid_escape", ESCAPED),
            (UNCLOSED_ARRAY, "unclosed_array", bracket),
        ):
            [(start, error)] = list(rejected_objects(text))
            assert list(repairs(text, start, error)) == [(name, fixed)]
            assert len(fixed) == len(text) + 1

    def test_a_key_of_an_object_is_no_unclosed_array(self) -> None:
        text = '{"recommended_action": "done", "x": 1, "y": 2}'
        error = json.JSONDecodeError("Expecting ',' delimiter", text, text.index(": 2"))
        assert list(repairs(text, 0, error)) == []

    def test_the_error_line_quotes_the_error_and_the_text_around_it(self) -> None:
        line = _json_error_line(STRAY_CLOSER)
        assert _position(line) == STRAY_CLOSER.index('quirk."]') + len('quirk."')
        assert line.startswith(
            "Your reply was not valid JSON (Expecting ',' delimiter at character "
        )
        assert line.endswith("...). Reply again with one JSON object.")
        quote = line.split(": ...", 1)[1].rsplit("...)", 1)[0]
        assert 'quirk."], "best_effort_result"' in quote
        assert len(quote) <= 2 * ERROR_CONTEXT_CHARS
        assert "\n" not in line

    def test_the_error_line_counts_the_fence_of_the_reply(self) -> None:
        plain = _json_error_line(STRAY_CLOSER)
        fenced = _json_error_line(f"```json\n{STRAY_CLOSER}\n```")
        assert _position(fenced) == _position(plain) + len("```json\n")

    def test_a_short_reply_is_quoted_whole(self) -> None:
        assert _json_error_line('{"recommended_action": "done"]') == (
            "Your reply was not valid JSON (Expecting ',' delimiter at character 29: "
            '{"recommended_action": "done"]). Reply again with one JSON object.'
        )

    def test_an_error_message_that_ends_in_at_reads_once(self) -> None:
        cut = '{"recommended_action": "done", "rationale": "cut off'
        start = cut.index('"cut')
        assert f"Unterminated string starting at character {start}:" in (
            _json_error_line(cut)
        )
        raw = '{"recommended_action": "a\nb"}'
        newline = raw.index("\n")
        assert f"Invalid control character at character {newline}:" in (
            _json_error_line(raw)
        )

    def test_no_line_without_a_rejected_object(self) -> None:
        for text in (
            "I would rather use words.",
            "",
            GOOD,
            '{"summary": "x"}',
            "[1, 2",
        ):
            assert _json_error_line(text) == ""

    def test_the_line_comes_from_the_first_rejected_object(self) -> None:
        assert "at character 7:" in _json_error_line('{"a": 1] {"b": 2,}')

    def test_error_line_quotes_around_the_position_only(self) -> None:
        body = "x" * 200 + "]" + "y" * 200
        error = json.JSONDecodeError("Expecting ',' delimiter", body, 200)
        assert error_line(body, error) == (
            "Your reply was not valid JSON (Expecting ',' delimiter at character 200: "
            f"...{'x' * 30}]{'y' * 29}...). Reply again with one JSON object."
        )


# ── The loop ────────────────────────────────────────────────────────────────


def _text(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}], "usage": PLAIN}


def _stub(*reflection: str | dict[str, Any]) -> Stub:
    """Two steps, the given reflection replies, then one more attempt."""
    replies = [_text(r) if isinstance(r, str) else r for r in reflection]
    return Stub(
        [
            _call_reply(PLAIN),
            _call_reply(PLAIN),
            *replies,
            _call_reply(PLAIN),
            _final_reply(PLAIN),
        ]
    )


def _reflection_event(events: list[AgentEvent]) -> Mapping[str, Any]:
    [event] = [e for e in events if e.type == "reflection"]
    return event.data


def _asked(body: dict[str, Any]) -> str:
    return str(body["messages"][-1]["content"])


class TestInTheLoop:
    async def test_a_repairable_reply_is_one_request_and_is_reported(self) -> None:
        stub = _stub(STRAY_CLOSER)
        events = await _run(stub, steps_per_attempt=2)
        data = _reflection_event(events)
        assert (data["requests"], data["readable"]) == (1, True)
        assert (data["unreadable_reason"], data["repaired"]) == (
            None,
            "unmatched_closer",
        )
        assert data["action"] == "continue" and data["summary"] == "Echoed once."
        assert len(stub.bodies) == 5

    async def test_a_reply_that_reads_reports_no_repair(self) -> None:
        events = await _run(_stub(GOOD), steps_per_attempt=2)
        assert _reflection_event(events)["repaired"] is None

    @pytest.mark.parametrize(
        ("reply", "repair"),
        [
            (TRAILING_COMMA, "trailing_comma"),
            (RAW_NEWLINE, "control_char"),
            (BAD_ESCAPE, "invalid_escape"),
            (UNCLOSED_ARRAY, "unclosed_array"),
        ],
        ids=["comma", "newline", "escape", "array"],
    )
    async def test_each_repair_is_reported_by_name(
        self, reply: str, repair: str
    ) -> None:
        events = await _run(_stub(reply), steps_per_attempt=2)
        data = _reflection_event(events)
        assert (data["requests"], data["repaired"]) == (1, repair)

    async def test_the_repair_of_the_retry_is_reported_with_the_first_reason(
        self,
    ) -> None:
        stub = _stub("I would rather say it in words.", STRAY_CLOSER)
        events = await _run(stub, steps_per_attempt=2)
        data = _reflection_event(events)
        assert (data["requests"], data["readable"]) == (2, True)
        assert (data["unreadable_reason"], data["repaired"]) == (
            "no_json",
            "unmatched_closer",
        )

    async def test_an_unreadable_reflection_reports_no_repair(self) -> None:
        events = await _run(_stub(TWO_SLIPS, TWO_SLIPS), steps_per_attempt=2)
        data = _reflection_event(events)
        assert (data["requests"], data["readable"], data["repaired"]) == (
            2,
            False,
            None,
        )

    async def test_utility_reads_and_retries_as_it_did(self) -> None:
        stub = _stub(STRAY_CLOSER, GOOD)
        events = await _run(stub, steps_per_attempt=2, reflection_gate="utility")
        data = _reflection_event(events)
        # No repair under utility: the reply holds no JSON object, so it is
        # asked again, with the note alone.
        assert (data["requests"], data["repaired"]) == (2, None)
        assert _asked(stub.bodies[3]).endswith(f"\n\n{REFLECTION_RETRY_NOTE}")

    async def test_the_retry_quotes_the_decoder_after_the_note(self) -> None:
        stub = _stub(TWO_SLIPS, GOOD)
        events = await _run(stub, steps_per_attempt=2)
        data = _reflection_event(events)
        assert (data["unreadable_reason"], data["readable"], data["requests"]) == (
            "no_json",
            True,
            2,
        )
        retry = _asked(stub.bodies[3])
        assert retry.endswith(f"\n\n{REFLECTION_RETRY_NOTE}\n{TWO_SLIPS_LINE}")
        assert retry.count(REFLECTION_RETRY_NOTE) == 1

    async def test_the_retry_quotes_the_decoder_for_an_invalid_reply_too(self) -> None:
        reply = f'{{"summary": "no action"}} {TWO_SLIPS}'
        stub = _stub(reply, GOOD)
        events = await _run(stub, steps_per_attempt=2)
        assert _reflection_event(events)["unreadable_reason"] == "invalid"
        assert _asked(stub.bodies[3]).endswith(
            f"{REFLECTION_RETRY_NOTE}\nYour reply was not valid JSON (Expecting ',' "
            f"delimiter at character {reply.index(']')}: "
            '...: "continue", "rationale": "r"]). Reply again with one JSON object.'
        )

    @pytest.mark.parametrize(
        "first",
        ["I would rather say it in words.", '{"summary": "x"}', '{"a": {"b": 1}}'],
        ids=["prose", "no_action", "nested"],
    )
    async def test_a_reply_with_nothing_to_quote_is_asked_as_before(
        self, first: str
    ) -> None:
        stub = _stub(first, GOOD)
        await _run(stub, steps_per_attempt=2)
        assert _asked(stub.bodies[3]).endswith(f"\n\n{REFLECTION_RETRY_NOTE}")
        assert "not valid JSON" not in _asked(stub.bodies[3])

    async def test_a_tool_call_is_asked_as_before_whatever_its_text_holds(self) -> None:
        call = _call_reply(PLAIN)
        call["choices"][0]["message"]["content"] = TWO_SLIPS
        stub = _stub(call, GOOD)
        events = await _run(stub, steps_per_attempt=2)
        assert _reflection_event(events)["unreadable_reason"] == "tool_call"
        assert _asked(stub.bodies[3]).endswith(f"\n\n{REFLECTION_RETRY_NOTE}")
        assert "not valid JSON" not in _asked(stub.bodies[3])

    async def test_the_retry_on_the_same_prefix_quotes_it_after_its_note(self) -> None:
        options: dict[str, Any] = {
            "steps_per_attempt": 2,
            "fixed_prefix": True,
            "reflect_tool_guard": True,
            "cache_layout": "v3",
        }
        stub = _stub(TWO_SLIPS, GOOD)
        await _run(stub, **options)
        reflect, retry = stub.bodies[2], stub.bodies[3]
        unread, note = retry["messages"][len(reflect["messages"]) :]
        assert unread == {"role": "assistant", "content": TWO_SLIPS}
        assert note["content"] == f"{REFLECTION_RETRY_GUARD_NOTE}\n{TWO_SLIPS_LINE}"
        # Without a line to quote the note is the one it always was.
        prose = _stub("I would rather say it in words.", GOOD)
        await _run(prose, **options)
        assert _asked(prose.bodies[3]) == REFLECTION_RETRY_GUARD_NOTE

    async def test_a_reflection_that_reads_asks_nothing_new(self) -> None:
        stub = _stub(GOOD)
        await _run(stub, steps_per_attempt=2)
        assert len(stub.bodies) == 5
        assert all("not valid JSON" not in json.dumps(body) for body in stub.bodies)


class TestWhereTheReaderIsUsed:
    def test_the_summary_of_a_repaired_reply_is_its_own(self) -> None:
        assert Agent._reflection_summary(STRAY_CLOSER) == "Echoed once."
        assert Agent._reflection_summary("[1,") == ""

    def test_the_notes_that_a_repaired_reply_cites_are_read(self) -> None:
        text = '{"recommended_action": "done", "used_notes": ["n1", 2, ""],}'
        assert Agent._used_notes(text, True) == ["n1"]
        # Under utility the reply is read as before: no JSON object, no notes.
        assert Agent._used_notes(text, False) == []
