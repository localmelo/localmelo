"""The importer turns proxy records into traces and counts everything it drops.

Requirements 53-55: files are found recursively and read in path order;
calls sort by start; endpoints, roles and the ``X-Localmelo-Call-Kind``
headers side file map to call kinds; status, clocks, tokens, service
sources, retries and overlap follow the rules; bad lines are dropped and
counted so the report's identities hold; a file of another schema
version is rejected whole; ``ttfb_s`` and ``ttft_s`` are never read;
traces round-trip through schema 1 and become templates.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import pytest

from localmelo.melo.contracts.providers import LLM_REQUEST_KINDS
from localmelo.support.scheduler.costs import (
    RATES_OMLX_OQ5,
    RATES_REC_FLINT,
    alone_service_s,
)
from localmelo.support.scheduler.model import CallKind
from localmelo.support.scheduler.sim import (
    ImportReport,
    TaskTrace,
    TraceCall,
    import_calls,
    read_trace,
    write_trace,
)
from localmelo.support.scheduler.sim.importer import (
    REQUEST_KIND_TO_CALL_KIND,
    CallsFormatError,
)
from localmelo.support.scheduler.sim.trace import (
    TraceFormatError,
    templates_from_traces,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
WALL = 1_793_000_000.0


def rec(index: int, start: float, end: float, run: str = "r1", **kw: Any) -> Any:
    """A localmelo chat record of schema 2, *start* and *end* in task seconds."""
    r = {
        "schema_version": 2,
        "harness": "localmelo",
        "run_id": run,
        "call_index": index,
        "endpoint": "chat_completions",
        "role": "main",
        "method": "POST",
        "path": "/v1/chat/completions",
        "mono_start": 500.0 + start,
        "mono_end": 500.0 + end,
        "ts_start": WALL + start,
        "ts_end": WALL + end,
        "duration_s": end - start,
        "status": 200,
        "error": None,
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
    }
    r.update(kw)
    return r


def write(root: Path, records: Sequence[Any], rel: str = "calls.jsonl") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r if isinstance(r, bytes) else json.dumps(r).encode() for r in records]
    path.write_bytes(b"\n".join(lines) + b"\n")
    return root


def check_identities(
    traces: Sequence[TaskTrace], report: ImportReport, kept_tasks: int | None = None
) -> None:
    """Requirement 55's identities; *kept_tasks* is the distinct tasks kept, when known."""
    assert report.lines == report.records_kept + sum(n for _, n in report.dropped)
    assert report.tasks == len(traces)
    if kept_tasks is not None:
        assert kept_tasks == (
            report.tasks + report.tasks_without_chat + report.tasks_overlapping
        )


def calls_of(traces: Sequence[TaskTrace], run: str = "r1") -> list[TraceCall]:
    return list(next(t for t in traces if t.task_key.endswith("#" + run)).calls)


def test_files_are_found_recursively_and_sorted_by_relative_path(
    tmp_path: Path,
) -> None:
    for rel in ("b/calls.jsonl", "a/x/y/calls.jsonl", "calls.jsonl"):
        write(tmp_path, [rec(0, 0.0, 1.0)], rel)
    write(tmp_path, [rec(0, 0.0, 1.0)], "a/other.jsonl")
    (tmp_path / "c" / "calls.jsonl").mkdir(parents=True)  # a directory, not a file
    traces, report = import_calls(tmp_path)
    assert report.files == 3
    assert [t.source for t in traces] == [
        "a/x/y/calls.jsonl",
        "b/calls.jsonl",
        "calls.jsonl",
    ]
    assert [t.task_key for t in traces][0] == "a/x/y/calls.jsonl#r1"
    nested, _ = import_calls(FIXTURES / "calls_nested")
    assert [t.task_key for t in nested] == ["a/b/calls.jsonl#n1"]


def test_calls_are_ordered_by_start_then_call_index(tmp_path: Path) -> None:
    traces, _ = import_calls(FIXTURES / "calls_nested")
    assert [c.call_index for c in traces[0].calls] == [0, 1]  # written in end order
    write(tmp_path, [rec(3, 5.0, 6.0), rec(1, 0.0, 2.0), rec(2, 5.0, 5.0)])
    traces, _ = import_calls(tmp_path, include_overlapping=True)
    calls = traces[0].calls
    assert [c.call_index for c in calls] == [1, 2, 3]
    assert [c.start_s for c in calls] == [0.0, 5.0, 5.0]
    assert [c.gap_before_s for c in calls] == [0.0, 3.0, 0.0]
    write(tmp_path, [rec(1, 5.0, 6.0), rec(2, 0.0, 1.0)])  # a lower index, later
    traces, _ = import_calls(tmp_path)
    assert [c.call_index for c in traces[0].calls] == [2, 1]


def test_endpoints_map_to_call_kinds_and_other_records_are_dropped_and_counted() -> (
    None
):
    traces, report = import_calls(FIXTURES / "calls_basic")
    kinds = {c.call_index: c.kind for c in calls_of(traces)}
    assert kinds == {
        0: "plan",
        1: "reflect",
        2: "plan",
        3: "plan",
        4: "plan",
        5: "plan",
        6: "embed",
    }
    # count_tokens, hello, GET /v1/models, another `other`, null and unknown.
    assert dict(report.dropped) == {"not_a_model_call": 6, "harness": 1}
    assert (report.lines, report.records_kept, report.tasks) == (14, 7, 1)
    check_identities(traces, report, kept_tasks=1)
    claude, _ = import_calls(FIXTURES / "calls_basic", harnesses=("claude-code",))
    assert [t.harness for t in claude] == ["claude-code"]


def test_status_is_ok_refused_or_failed_and_a_client_disconnect_alone_is_ok() -> None:
    traces, _ = import_calls(FIXTURES / "calls_basic")
    status = {c.call_index: c.status for c in calls_of(traces)}
    # 2: HTTP 500; 3: handled locally; 4: a stream that did not complete;
    # 5: a complete stream whose client disconnected after [DONE].
    assert status == {
        0: "ok",
        1: "ok",
        2: "failed",
        3: "refused",
        4: "failed",
        5: "ok",
        6: "ok",
    }


def test_nan_infinity_negative_and_reversed_times_are_rejected_and_counted(
    tmp_path: Path,
) -> None:
    good = json.dumps(rec(0, 0.0, 1.0))
    records: list[Any] = [
        good.replace('"duration_s": 1.0', '"duration_s": NaN').encode(),
        good.replace('"mono_end": 501.0', '"mono_end": Infinity').encode(),
        good.replace('"mono_start": 500.0', '"mono_start": -Infinity').encode(),
        rec(1, 2.0, 3.0, duration_s=-0.5),
        rec(2, 4.0, 3.5),
        rec(3, 5.0, 6.0, duration_s="1"),
        rec(4, 7.0, 8.0),
    ]
    traces, report = import_calls(write(tmp_path, records))
    assert dict(report.dropped) == {
        "bad_duration": 2,
        "end_before_start": 1,
        "nan_or_infinity": 3,
    }
    assert [c.call_index for c in traces[0].calls] == [4]
    check_identities(traces, report, kept_tasks=1)


def test_a_file_with_another_schema_version_is_rejected_whole(tmp_path: Path) -> None:
    traces, report = import_calls(FIXTURES / "calls_bad_version")
    assert (traces, report.files, report.files_rejected) == ((), 1, 1)
    assert report.lines == report.records_kept == 0
    assert report.errors == ("calls.jsonl: schema_version 3, expected 2",)
    with pytest.raises(CallsFormatError, match="schema_version 3"):
        import_calls(FIXTURES / "calls_bad_version", strict=True)
    # A record without schema_version rejects its file too; others still import.
    missing = rec(1, 2.0, 3.0)
    del missing["schema_version"]
    write(tmp_path, [rec(0, 0.0, 1.0), missing], "x/calls.jsonl")
    write(tmp_path, [rec(0, 0.0, 1.0, run="ok")], "y/calls.jsonl")
    traces, report = import_calls(tmp_path)
    assert [t.task_key for t in traces] == ["y/calls.jsonl#ok"]
    assert report.errors == ("x/calls.jsonl: schema_version None, expected 2",)
    assert (report.files, report.files_rejected, report.lines) == (2, 1, 1)


def test_null_cached_tokens_stay_unknown(tmp_path: Path) -> None:
    usage = {"prompt_tokens": 1000, "completion_tokens": 100}
    records = [
        rec(0, 0.0, 1.0, usage={**usage, "cached_tokens": None}),
        rec(1, 1.0, 2.0, usage={**usage, "cached_tokens": 0}),
        rec(2, 2.0, 3.0, usage_raw={"prompt_tokens_details": {"cached_tokens": 900}}),
        rec(3, 3.0, 4.0, usage_raw={"prompt_tokens_details": None}),
    ]
    traces, _ = import_calls(write(tmp_path, records))
    assert [c.cached_tokens for c in traces[0].calls] == [None, 0, 900, None]


def test_null_usage_bad_token_values_unknown_endpoints_and_bad_utf8_are_handled(
    tmp_path: Path,
) -> None:
    records: list[Any] = [
        rec(0, 0.0, 1.0, usage=None, usage_raw="text"),
        rec(1, 1.0, 2.0, usage={"prompt_tokens": -1, "completion_tokens": 2.5}),
        rec(2, 2.0, 3.0, usage={"prompt_tokens": True, "completion_tokens": "7"}),
        rec(3, 3.0, 4.0, endpoint=None),
        rec(4, 4.0, 5.0, endpoint="batch"),
        b'{"schema_version": 2, "run_id": "r1", "note": "\xff\xfe"}',
        b"[1, 2]",
        b"{not json",
        rec(5, 5.0, 6.0, run=""),
        rec(6, 6.0, 7.0, call_index=-1),
        rec(7, 7.0, 8.0, call_index=True),
        rec(8, 8.0, 9.0),
        rec(8, 9.0, 9.5),  # a repeated index: a trace keys calls by it
        rec(
            9,
            10.0,
            11.0,
            usage={"prompt_tokens": 10**400, "completion_tokens": 3},
            usage_raw={"prompt_eval_duration": 1e308, "generation_duration": 1e308},
        ),
    ]
    traces, report = import_calls(write(tmp_path, records))
    calls = traces[0].calls
    assert [c.call_index for c in calls] == [0, 1, 2, 8, 9]
    assert [(c.prompt_tokens, c.completion_tokens) for c in calls[:3]] == [
        (None, None)
    ] * 3
    assert (calls[4].prompt_tokens, calls[4].server_prompt_s) == (None, None)
    assert (calls[4].service_s, calls[4].service_source) == (1.0, "wall")
    assert [c.tokens_unknown for c in calls] == [True, True, True, False, True]
    write_trace(tmp_path / "trace.jsonl", traces)  # every value is finite
    assert read_trace(tmp_path / "trace.jsonl") == traces
    assert dict(report.dropped) == {
        "bad_call_index": 3,
        "no_run_id": 1,
        "not_a_model_call": 2,
        "not_json": 2,
        "not_object": 1,
    }
    check_identities(traces, report, kept_tasks=1)


def test_a_task_uses_mono_times_only_when_every_record_has_them(
    tmp_path: Path,
) -> None:
    no_mono: dict[str, Any] = {"mono_start": None, "mono_end": None}
    no_ts: dict[str, Any] = {"ts_start": None, "ts_end": None}
    write(tmp_path, [rec(0, 0.0, 1.0), rec(1, 2.0, 3.0)], "mono/calls.jsonl")
    write(tmp_path, [rec(0, 0.0, 1.0), rec(1, 2.0, 3.0, **no_mono)], "wall/calls.jsonl")
    both_missing = [
        rec(0, 0.0, 1.0, **no_ts),
        rec(1, 2.0, 3.0, **no_mono),
        rec(2, 4.0, 5.0),
    ]
    write(tmp_path, both_missing, "mixed/calls.jsonl")
    # A record that is no model call, or of another harness, sets no clock
    # (the lead's decision 6): this task stays on mono times.
    write(
        tmp_path,
        [
            rec(0, 0.0, 1.0),
            rec(
                1,
                1.0,
                1.1,
                endpoint="other",
                method="GET",
                path="/v1/models",
                **no_mono,
            ),
            rec(2, 1.0, 1.1, harness="hermes", **no_mono),
        ],
        "order/calls.jsonl",
    )
    huge = [rec(0, 0.0, 1.0, mono_start=-1.7e308), rec(1, 2.0, 3.0, mono_end=1.7e308)]
    write(tmp_path, huge, "zhuge/calls.jsonl")  # beyond ±2**53: not a clock value
    traces, report = import_calls(tmp_path)
    by_source = {t.source.split("/")[0]: t.calls for t in traces}
    assert [c.start_s for c in by_source["zhuge"]] == [0.0, 2.0]
    assert [c.wall_start_s for c in by_source["wall"]] == [WALL, WALL + 2.0]
    assert [c.start_s for c in by_source["wall"]] == [0.0, 2.0]
    assert [c.call_index for c in by_source["mixed"]] == [0, 2]
    assert [c.start_s for c in by_source["mixed"]] == [0.0, 4.0]
    assert [c.call_index for c in by_source["order"]] == [0]
    assert report.wall_time_tasks == 2
    assert dict(report.dropped) == {
        "bad_times": 1,
        "harness": 1,
        "not_a_model_call": 1,
    }
    check_identities(traces, report, kept_tasks=5)


def test_service_prefers_server_timings_then_tokens_then_wall(tmp_path: Path) -> None:
    raw = {"prompt_eval_duration": 0.25, "generation_duration": 1.5}
    records = [
        rec(0, 0.0, 2.0, usage_raw={**raw, "time_to_first_token": 0.3}),
        rec(1, 2.0, 3.0, usage_raw={**raw, "generation_duration": None}),
        rec(2, 3.0, 4.5, usage={"prompt_tokens": 1000}),
    ]
    traces, report = import_calls(write(tmp_path, records))
    a, b, c = traces[0].calls
    assert (a.service_s, a.service_source, a.server_ttft_s) == (1.75, "server", 0.3)
    assert a.residual_s == pytest.approx(0.25)
    assert (b.service_source, b.residual_s) == ("tokens", None)
    assert b.service_s == alone_service_s(1000, None, 100, RATES_OMLX_OQ5)
    assert (c.service_s, c.service_source) == (1.5, "wall")
    assert report.service_sources == (("server", 1), ("tokens", 1), ("wall", 1))
    rec_traces, _ = import_calls(tmp_path, rates=RATES_REC_FLINT)
    assert rec_traces[0].calls[1].service_s == alone_service_s(
        1000, None, 100, RATES_REC_FLINT
    )


def test_overlapping_runs_are_marked_and_left_out_by_default(tmp_path: Path) -> None:
    write(tmp_path, [rec(0, 0.0, 3.0), rec(1, 2.0, 4.0)], "over/calls.jsonl")
    write(tmp_path, [rec(0, 0.0, 1.0), rec(1, 1.0, 2.0)], "seq/calls.jsonl")
    embed: dict[str, Any] = {"endpoint": "other", "path": "/v1/embeddings"}
    write(
        tmp_path,
        [rec(0, 0.0, 3.0, **embed), rec(1, 1.0, 2.0, **embed)],
        "e/calls.jsonl",
    )
    traces, report = import_calls(tmp_path)
    assert [t.source for t in traces] == ["seq/calls.jsonl"]
    assert (report.tasks_overlapping, report.tasks_without_chat) == (1, 1)
    check_identities(traces, report, kept_tasks=3)
    traces, report = import_calls(tmp_path, include_overlapping=True)
    assert [(t.source, t.overlapping) for t in traces] == [
        ("over/calls.jsonl", True),
        ("seq/calls.jsonl", False),
    ]
    assert traces[0].calls[1].gap_before_s == -1.0
    assert report.tasks == 2 and report.tasks_overlapping == 1


def test_ttfb_and_ttft_are_never_read(tmp_path: Path) -> None:
    write(tmp_path, [rec(0, 0.0, 1.0), rec(1, 1.5, 2.0)], "a/calls.jsonl")
    # Plausible values that a read would accept, as times or as a fallback.
    seen: dict[str, Any] = {"ttfb_s": 0.4, "ttft_s": 0.7, "ttft_valid": True}
    write(
        tmp_path,
        [rec(0, 0.0, 1.0, **seen), rec(1, 1.5, 2.0, **seen)],
        "b/calls.jsonl",
    )
    traces, report = import_calls(tmp_path)
    a, b = traces
    assert [replace(c, task_key="") for c in a.calls] == [
        replace(c, task_key="") for c in b.calls
    ]
    assert report.dropped == ()


def test_a_retry_within_5_s_of_a_failure_is_the_next_attempt(tmp_path: Path) -> None:
    failed: dict[str, Any] = {"status": 500, "error": "boom"}
    records = [
        rec(0, 0.0, 1.0, **failed),
        rec(1, 6.0, 7.0, **failed),  # exactly 5.0 s later: attempt 2
        rec(2, 12.0, 13.0),  # 5.0 s after a failure: attempt 3
        rec(3, 13.5, 14.0),  # after an ok call: attempt 1
        rec(4, 14.5, 15.0, **failed),
        rec(5, 20.25, 21.0),  # 5.25 s later: a new call
    ]
    traces, _ = import_calls(write(tmp_path, records))
    assert [c.attempt for c in traces[0].calls] == [1, 2, 3, 1, 1, 1]


def test_a_trace_round_trips_and_an_s2_style_line_reads_as_a_trace_call(
    tmp_path: Path,
) -> None:
    traces, _ = import_calls(FIXTURES / "calls_basic")
    nested, _ = import_calls(FIXTURES / "calls_nested")
    both = tuple(sorted(traces + nested, key=lambda t: t.task_key))
    path = tmp_path / "trace.jsonl"
    write_trace(path, both)
    assert read_trace(path) == both
    first = path.read_bytes()
    write_trace(path, both)
    assert path.read_bytes() == first
    (s2,) = read_trace(FIXTURES / "trace_s2_style.jsonl")
    (c,) = s2.calls
    assert (s2.task_key, s2.source, s2.overlapping) == (
        "chat-7f3a",
        "gate/calls-2026-10-27.jsonl",
        False,
    )
    assert (c.session_kind, c.level, c.submit_s, c.admit_s) == ("chat", 2, 0.0, 1.5)
    assert c.inflight_at_submit == 3 and isinstance(c.start_s, float)
    lines = first.decode().splitlines()
    other = lines[0].replace('"call_index":0', '"call_index":99')
    for bad, message in (
        (lines + [lines[0]], "repeated call"),
        (lines + [other.replace('"harness":"localmelo"', '"harness":"x"')], "differ"),
        ([lines[0].replace('"version":1', '"version":2')], "version 1"),
        ([lines[0].replace('"attempt":1', '"attempt":1.5')], "attempt"),
        ([lines[0].replace('"version":1', '"version":true')], "version 1"),
        ([lines[0].replace('"version":1', '"version":1.0')], "version 1"),
        ([lines[0].replace('"start_s":0.0', '"start_s":' + "9" * 400)], "start_s"),
        (["[1]"], "not a JSON object"),
    ):
        path.write_text("\n".join(bad) + "\n")
        with pytest.raises(TraceFormatError, match=message):
            read_trace(path)
    path.write_bytes(first + b"\xff\n")
    with pytest.raises(TraceFormatError, match=f":{len(lines) + 1}: not JSON"):
        read_trace(path)


def test_templates_drop_refused_calls_and_fill_failed_tokens() -> None:
    traces, _ = import_calls(FIXTURES / "calls_basic")
    (template,) = templates_from_traces(traces)
    assert [c.kind for c in template.calls] == [
        "plan",
        "reflect",
        "plan",
        "plan",
        "plan",
        "embed",
    ]
    tokens = [(c.prompt_tokens, c.completion_tokens) for c in template.calls]
    # Call 2 failed with no usage: the previous chat prompt and 0; call 4
    # failed with a prompt only; the embedding has no completion count.
    assert tokens == [
        (1704, 120),
        (1900, 80),
        (1900, 0),
        (2100, 0),
        (2104, 150),
        (12, 0),
    ]
    assert [c.status for c in template.calls] == [
        "ok",
        "ok",
        "failed",
        "failed",
        "ok",
        "ok",
    ]
    # The refused call 3 (8.0-8.01) folds into call 4's gap: 1.0 + 0.01 + 0.99.
    assert template.calls[3].gap_before_s == pytest.approx(2.0)
    assert [c.warm_cached_tokens for c in template.calls[:2]] == [0, 1697]
    calls = calls_of(traces)
    unknown = replace(traces[0], calls=(replace(calls[0], prompt_tokens=None),))
    overlapping = replace(traces[0], overlapping=True)
    assert templates_from_traces([unknown, overlapping]) == ()
    first = replace(calls[2], prompt_tokens=None)
    (alone,) = templates_from_traces([replace(traces[0], calls=(first,))])
    assert alone.calls[0].prompt_tokens == 3235


# ── Real call kinds ──


def headers_file(
    root: Path,
    index: int,
    value: str | None,
    *,
    name: str = "X-Localmelo-Call-Kind",
) -> dict[str, Any]:
    """A record's ``bodies`` dict with a headers side file, as the proxy writes it.

    *value* ``None`` writes a headers file without the kind header.
    """
    rel = f"bodies/{index:06d}-headers.json"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    request = [] if value is None else [[name, value]]
    path.write_text(json.dumps({"request": request, "response": []}))
    return {"bodies": {"headers": rel}}


@pytest.mark.parametrize("value", list(REQUEST_KIND_TO_CALL_KIND))
def test_the_call_kind_header_names_the_kind_of_a_chat_call(
    tmp_path: Path, value: str
) -> None:
    # A plan call keeps the task one with a chat call; retrieve and
    # estimate are not chat kinds on their own. role aux would guess
    # reflect, so the second call asserts the header won.
    records = [
        rec(0, 0.0, 1.0),
        rec(1, 1.0, 2.0, role="aux", **headers_file(tmp_path, 1, value)),
    ]
    traces, report = import_calls(write(tmp_path, records))
    assert [c.kind for c in traces[0].calls] == ["plan", value]
    assert report.kind_sources == (("header", 1), ("history", 0), ("guess", 1))
    assert report.kind_unknown == 0


def test_the_call_kind_header_name_is_case_insensitive(tmp_path: Path) -> None:
    records = [
        rec(0, 0.0, 1.0),
        rec(
            1,
            1.0,
            2.0,
            **headers_file(tmp_path, 1, "estimate", name="x-LOCALMELO-call-KIND"),
        ),
    ]
    traces, _ = import_calls(write(tmp_path, records))
    assert [c.kind for c in traces[0].calls] == ["plan", "estimate"]


def test_an_embeddings_record_keeps_embed_whatever_a_headers_file_says(
    tmp_path: Path,
) -> None:
    # localmelo sends no kind header on embeddings, so a stray one is not
    # evidence about the call.
    records = [
        rec(0, 0.0, 1.0),
        rec(
            1,
            1.0,
            2.0,
            endpoint="other",
            path="/v1/embeddings",
            **headers_file(tmp_path, 1, "plan"),
        ),
    ]
    traces, report = import_calls(write(tmp_path, records))
    assert [c.kind for c in traces[0].calls] == ["plan", "embed"]
    assert report.kind_sources == (("header", 0), ("history", 0), ("guess", 2))


@pytest.mark.parametrize(("role", "guess"), [("aux", "reflect"), ("main", "plan")])
def test_an_unknown_call_kind_keeps_the_guess_and_is_counted(
    tmp_path: Path, role: str, guess: str
) -> None:
    record = rec(0, 0.0, 1.0, role=role, **headers_file(tmp_path, 0, "summarize"))
    traces, report = import_calls(write(tmp_path, [record]))
    assert traces[0].calls[0].kind == guess
    assert report.kind_sources == (("header", 0), ("history", 0), ("guess", 1))
    assert report.kind_unknown == 1


@pytest.mark.parametrize(
    "bodies",
    [
        None,
        {},
        {"headers": None},
        {"headers": 7},
        {"headers": "bodies/no-such-headers.json"},
        {"headers": "bodies/000000-headers.json"},  # the file below is malformed
    ],
)
def test_a_record_without_a_readable_headers_file_keeps_the_guess(
    tmp_path: Path, bodies: dict[str, Any] | None
) -> None:
    (tmp_path / "bodies").mkdir()
    (tmp_path / "bodies/000000-headers.json").write_text("{not json")
    kw: dict[str, Any] = {} if bodies is None else {"bodies": bodies}
    record = rec(0, 0.0, 1.0, role="aux", **kw)
    write(tmp_path, [record])
    traces, report = import_calls(tmp_path)
    assert traces[0].calls[0].kind == "reflect"
    assert report.kind_sources == (("header", 0), ("history", 0), ("guess", 1))
    assert report.kind_unknown == 0


def test_a_headers_file_without_the_kind_header_keeps_the_guess(
    tmp_path: Path,
) -> None:
    record = rec(0, 0.0, 1.0, role="aux", **headers_file(tmp_path, 0, None))
    traces, report = import_calls(write(tmp_path, [record]))
    assert traces[0].calls[0].kind == "reflect"
    assert dict(report.kind_sources)["header"] == 0
    assert report.kind_unknown == 0


@pytest.mark.parametrize(
    "relative",
    ["../outside-headers.json", "/etc/passwd", "bodies/../../outside-headers.json"],
)
def test_a_headers_path_outside_the_calls_directory_is_not_read(
    tmp_path: Path, relative: str
) -> None:
    outside = tmp_path.parent / "outside-headers.json"
    outside.write_text(json.dumps({"request": [["X-Localmelo-Call-Kind", "retrieve"]]}))
    record = rec(0, 0.0, 1.0, role="aux", bodies={"headers": relative})
    (tmp_path / "bodies").mkdir()
    write(tmp_path, [record])
    traces, report = import_calls(tmp_path)
    assert traces[0].calls[0].kind == "reflect"
    assert report.kind_sources == (("header", 0), ("history", 0), ("guess", 1))
    assert report.kind_unknown == 0


def test_kind_sources_count_every_kept_record(tmp_path: Path) -> None:
    records = [
        rec(0, 0.0, 1.0, role="aux", **headers_file(tmp_path, 0, "retrieve")),
        rec(1, 1.0, 2.0, role="aux", **headers_file(tmp_path, 1, "plan")),
        rec(2, 2.0, 3.0),  # no headers file: guessed plan
        rec(3, 3.0, 4.0, role="aux", **headers_file(tmp_path, 2, "made-up")),
    ]
    traces, report = import_calls(write(tmp_path, records))
    assert [c.kind for c in traces[0].calls] == [
        "retrieve",
        "plan",
        "plan",
        "reflect",
    ]
    assert report.kind_sources == (("header", 2), ("history", 0), ("guess", 2))
    assert report.kind_unknown == 1
    # Every kept record names one source, as service_sources does for service.
    assert sum(n for _, n in report.kind_sources) == report.records_kept == 4
    check_identities(traces, report, kept_tasks=1)


def test_pre_r1_records_import_exactly_as_before(tmp_path: Path) -> None:
    # Before R1 there is no kind header: chat with tools is plan, chat
    # without tools reflect, and the embeddings endpoint embed.
    records = [
        rec(0, 0.0, 1.0),
        rec(1, 1.0, 2.0, role="aux"),
        rec(2, 2.0, 3.0, endpoint="other", path="/v1/embeddings"),
    ]
    traces, report = import_calls(write(tmp_path, records))
    assert [c.kind for c in traces[0].calls] == ["plan", "reflect", "embed"]
    assert report.kind_sources == (("header", 0), ("history", 0), ("guess", 3))
    assert report.kind_unknown == 0
    # The pre-R1 fixture is the same: its records have no bodies.
    old, old_report = import_calls(FIXTURES / "calls_basic")
    assert [c.kind for c in calls_of(old)] == [
        "plan",
        "reflect",
        "plan",
        "plan",
        "plan",
        "plan",
        "embed",
    ]
    assert dict(old_report.kind_sources)["header"] == 0


def test_history_rows_cannot_name_a_proxy_record_so_kinds_stay_guessed(
    tmp_path: Path,
) -> None:
    # No sound join exists: llm_calls has task_id (a fresh uuid), seq,
    # timestamps and durations, but no run id or request id, and durations
    # and arrival times differ from the proxy's by a few ms. The importer
    # therefore reads only the calls.jsonl header side file, and this pins
    # that a history.db beside the calls changes no kind.
    db = tmp_path / "mem" / "history.db"
    db.parent.mkdir()
    connection = sqlite3.connect(db)
    try:
        connection.execute(
            "CREATE TABLE llm_calls (task_id TEXT, seq INTEGER, kind TEXT)"
        )
        connection.execute("INSERT INTO llm_calls VALUES ('t', 0, 'retrieve')")
        connection.commit()
    finally:
        connection.close()
    traces, report = import_calls(write(tmp_path, [rec(0, 0.0, 1.0, role="aux")]))
    assert traces[0].calls[0].kind == "reflect"
    assert report.kind_sources == (("header", 0), ("history", 0), ("guess", 1))


def test_the_importer_knows_every_request_kind_the_agent_sends() -> None:
    # support/scheduler may not import melo/, so the vocabulary is repeated
    # there; this catches the two drifting apart.
    assert tuple(REQUEST_KIND_TO_CALL_KIND) == LLM_REQUEST_KINDS
    assert set(REQUEST_KIND_TO_CALL_KIND.values()) <= set(get_args(CallKind))


def test_real_bench_traces_import_when_the_path_is_set() -> None:
    root = os.environ.get("LOCALMELO_SCHED_TRACE_ROOT", "")
    if not root or not Path(root).is_dir():
        pytest.skip("LOCALMELO_SCHED_TRACE_ROOT does not name a directory")
    traces, report = import_calls(Path(root))
    assert report.records_kept > 0
    check_identities(traces, report)
    with_overlap, report2 = import_calls(Path(root), include_overlapping=True)
    assert report2.tasks == report.tasks + report.tasks_overlapping
    assert report2.tasks_without_chat == report.tasks_without_chat
    check_identities(with_overlap, report2)
