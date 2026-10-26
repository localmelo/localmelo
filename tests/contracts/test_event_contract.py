"""The event contract: AgentEvent, EVENT_FIELDS and the payload rules."""

from __future__ import annotations

import enum
import json
import math
import sys
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

import localmelo
from localmelo.melo import contracts
from localmelo.melo.contracts import events
from localmelo.melo.contracts.events import (
    EVENT_FIELDS,
    EVENT_SCHEMA_VERSION,
    EVENT_TEXT_LIMIT,
    EVENT_TYPES,
    AgentEvent,
    json_safe_payload,
)
from localmelo.melo.schema import ImagePart

PNG = ImagePart.from_bytes(b"\x89PNG\r\n\x1a\n pixels", "image/png")


def _event(**data: Any) -> AgentEvent:
    return AgentEvent(
        type="step.started", run_id="r1", seq=3, ts=12.5, data=data, images=(PNG,)
    )


def _nested(levels: int, inner: Any = None, *, key: str | None = None) -> Any:
    """*inner* inside *levels* lists, or dicts under *key*, one in another.

    Built without recursion, so *levels* may pass the recursion limit.
    """
    for _ in range(levels):
        inner = [inner] if key is None else {key: inner}
    return inner


class TestCatalogue:
    def test_the_types_are_the_catalogued_ones(self) -> None:
        assert frozenset(EVENT_FIELDS) == EVENT_TYPES
        assert {
            "run.started",
            "step.started",
            "assistant.message",
            "tool.started",
            "tool.finished",
            "notice",
            "reflection.started",
            "reflection",
            "run.finished",
        } == EVENT_TYPES
        assert (EVENT_SCHEMA_VERSION, EVENT_TEXT_LIMIT) == (1, 100_000)

    def test_the_catalogue_cannot_be_changed(self) -> None:
        with pytest.raises(TypeError):
            EVENT_FIELDS["custom"] = frozenset()  # type: ignore[index]

    def test_the_names_are_exported(self) -> None:
        for name in (
            "EVENT_FIELDS",
            "EVENT_SCHEMA_VERSION",
            "EVENT_TEXT_LIMIT",
            "EVENT_TYPES",
            "AgentEvent",
            "EventSink",
            "json_safe_payload",
        ):
            assert name in contracts.__all__
            assert hasattr(contracts, name)
        assert localmelo.AgentEvent is AgentEvent
        assert "AgentEvent" in localmelo.__all__


class TestAgentEvent:
    def test_to_dict_is_the_envelope_without_images(self) -> None:
        event = _event(step=1, attempt=1)
        assert event.to_dict() == {
            "v": 1,
            "type": "step.started",
            "run_id": "r1",
            "seq": 3,
            "ts": 12.5,
            "data": {"step": 1, "attempt": 1},
        }
        assert json.loads(json.dumps(event.to_dict(), allow_nan=False))

    def test_to_dict_copies_data(self) -> None:
        event = _event(step=1)
        first = event.to_dict()
        first["data"]["step"] = 2
        assert event.data == {"step": 1}
        assert event.to_dict()["data"] is not event.to_dict()["data"]

    def test_images_stay_out_of_the_repr_and_of_comparisons(self) -> None:
        event = _event(step=1)
        assert event.images == (PNG,)
        assert PNG.data not in repr(event)
        assert event == AgentEvent(
            type="step.started", run_id="r1", seq=3, ts=12.5, data={"step": 1}
        )

    def test_it_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            _event().seq = 4  # type: ignore[misc]


class _Color(enum.StrEnum):
    RED = "red"


class _Size(enum.IntEnum):
    BIG = 3


class _Unprintable:
    def __str__(self) -> str:
        raise RuntimeError("no text")


class TestJsonSafePayload:
    def test_json_values_are_kept(self) -> None:
        data = {
            "s": "text",
            "i": 3,
            "f": 0.5,
            "b": True,
            "n": None,
            "l": [1, "two", [3.0]],
            "d": {"k": {"deep": False}},
        }
        assert json_safe_payload(data) == data

    def test_non_finite_floats_become_text(self) -> None:
        payload = json_safe_payload({"values": [math.nan, math.inf, -math.inf, -0.0]})
        assert payload == {"values": ["nan", "inf", "-inf", -0.0]}

    def test_other_values_become_their_text(self) -> None:
        thing = object()
        payload = json_safe_payload(
            {
                "object": thing,
                "bytes": b"ab",
                "set": {1},
                "nested": {"in": [thing]},
                "unprintable": _Unprintable(),
            }
        )
        assert payload["object"] == str(thing)
        assert payload["bytes"] == "b'ab'"
        assert payload["set"] == "{1}"
        assert payload["nested"] == {"in": [str(thing)]}
        assert payload["unprintable"].startswith("<")
        assert "_Unprintable object at" in payload["unprintable"]

    def test_tuples_become_lists_and_keys_become_text(self) -> None:
        payload = json_safe_payload({"t": (1, (2,)), "m": {1: "a", None: "b"}})
        assert payload == {"t": [1, [2]], "m": {"1": "a", "None": "b"}}

    def test_subclasses_become_plain_values(self) -> None:
        payload = json_safe_payload(
            {"color": _Color.RED, "size": _Size.BIG, _Color.RED: 1}
        )
        assert payload == {"color": "red", "size": 3, "red": 1}
        assert type(payload["color"]) is str
        assert type(payload["size"]) is int
        assert all(type(key) is str for key in payload)

    def test_the_result_is_json_without_nan(self) -> None:
        payload = json_safe_payload(
            {"a": [math.nan, object(), {"b": (math.inf,)}], "m": {2: _Size.BIG}}
        )
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload

    def test_long_strings_are_clipped_at_any_depth(self) -> None:
        exact = "x" * EVENT_TEXT_LIMIT
        long = "y" * (EVENT_TEXT_LIMIT + 50_000)
        payload = json_safe_payload(
            {"exact": exact, "long": long, "deep": [{"in": long}], long: "key"}
        )
        clipped = "y" * EVENT_TEXT_LIMIT + "… [+50000 chars]"
        assert payload["exact"] == exact
        assert payload["long"] == clipped
        assert payload["deep"] == [{"in": clipped}]
        assert payload[clipped] == "key"

    def test_long_text_of_other_values_is_clipped(self) -> None:
        class Wordy:
            def __str__(self) -> str:
                return "w" * (EVENT_TEXT_LIMIT + 1)

        assert json_safe_payload({"w": Wordy()})["w"] == (
            "w" * EVENT_TEXT_LIMIT + "… [+1 chars]"
        )

    def test_the_copy_shares_nothing_and_leaves_the_input_alone(self) -> None:
        inner = {"list": [1, 2]}
        data: dict[str, Any] = {"inner": inner, "n": math.nan}
        payload = json_safe_payload(data)
        payload["inner"]["list"].append(3)
        assert inner == {"list": [1, 2]}
        assert math.isnan(data["n"])

    def test_lists_and_dicts_nest_at_most_100_levels(self) -> None:
        assert events._DEPTH_LIMIT == 100
        # The values of data are on level 1. A list, dict or tuple on level
        # 100 is kept; one on level 101 becomes its text.
        kept = {
            "list": _nested(100, 1),
            "dict": _nested(100, 1, key="k"),
            "tuple": _nested(99, (1,)),
        }
        assert json_safe_payload(kept) == {
            "list": _nested(100, 1),
            "dict": _nested(100, 1, key="k"),
            "tuple": _nested(100, 1),
        }
        past = {
            "list": _nested(101, 1),
            "dict": _nested(101, 1, key="k"),
            "tuple": _nested(100, (1,)),
            "empty": _nested(100, {}),
        }
        assert json_safe_payload(past) == {
            "list": _nested(100, "[1]"),
            "dict": _nested(100, "{'k': 1}", key="k"),
            "tuple": _nested(100, "(1,)"),
            "empty": _nested(100, "{}"),
        }

    def test_data_nested_past_the_recursion_limit_gives_a_payload(self) -> None:
        loop: list[Any] = []
        loop.append(loop)
        payload = json_safe_payload(
            {"deep": _nested(2 * sys.getrecursionlimit()), "loop": loop}
        )
        json.dumps(payload, allow_nan=False)
        leaves = []
        for value in (payload["deep"], payload["loop"]):
            for _ in range(100):
                assert isinstance(value, list) and len(value) == 1
                value = value[0]
            leaves.append(value)
        # How much of the deep list's text str() gives depends on the
        # Python; a list that holds itself shows the reference as [...].
        assert isinstance(leaves[0], str)
        assert leaves[1] == "[[...]]"
