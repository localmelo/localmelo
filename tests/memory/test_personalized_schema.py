"""Guard tests freezing the PersonalizedSample v1 schema.

These tests must fail when a v1 field is renamed, removed, reordered, or its
default/default_factory changes. They also pin the read-side contract that
SleepPreprocessor relies on. If you intentionally evolve the schema, bump
PERSONALIZED_SAMPLE_SCHEMA_VERSION in the same commit and update these tests.
"""

from __future__ import annotations

import time
from dataclasses import MISSING, fields

from localmelo.melo.memory.personalized import (
    PERSONALIZED_SAMPLE_SCHEMA_VERSION,
    PersonalizedMemory,
    PersonalizedSample,
)
from localmelo.melo.sleep.preprocess import SleepPreprocessor

V1_FIELD_NAMES = (
    "input_text",
    "target_text",
    "signal",
    "metadata",
    "timestamp",
)
V1_FIELD_TYPES = {
    "input_text": "str",
    "target_text": "str",
    "signal": "str",
    "metadata": "dict[str, Any]",
    "timestamp": "float",
}


def test_schema_version_is_v1() -> None:
    assert PERSONALIZED_SAMPLE_SCHEMA_VERSION == "v1"


def test_field_names_and_order_frozen() -> None:
    assert tuple(f.name for f in fields(PersonalizedSample)) == V1_FIELD_NAMES


def test_field_type_annotations_frozen() -> None:
    actual = {f.name: f.type for f in fields(PersonalizedSample)}
    assert actual == V1_FIELD_TYPES


def test_field_defaults_frozen() -> None:
    by_name = {f.name: f for f in fields(PersonalizedSample)}

    # input_text is required: no default, no default_factory
    assert by_name["input_text"].default is MISSING
    assert by_name["input_text"].default_factory is MISSING  # type: ignore[misc]

    # target_text default is the empty string
    assert by_name["target_text"].default == ""
    assert by_name["target_text"].default_factory is MISSING  # type: ignore[misc]

    # signal default is the empty string
    assert by_name["signal"].default == ""
    assert by_name["signal"].default_factory is MISSING  # type: ignore[misc]

    # metadata uses a default_factory producing a fresh dict (no shared mutable)
    metadata_factory = by_name["metadata"].default_factory  # type: ignore[misc]
    assert metadata_factory is not MISSING
    first = metadata_factory()
    second = metadata_factory()
    assert first == {} and second == {}
    assert first is not second  # fresh dict each call

    # timestamp factory is exactly time.time (identity, not just "callable")
    timestamp_factory = by_name["timestamp"].default_factory  # type: ignore[misc]
    assert timestamp_factory is time.time


def test_constructed_sample_default_values() -> None:
    sample = PersonalizedSample(input_text="hi")
    assert sample.input_text == "hi"
    assert sample.target_text == ""
    assert sample.signal == ""
    assert sample.metadata == {}
    assert isinstance(sample.timestamp, float)


def test_metadata_default_is_not_shared_between_instances() -> None:
    a = PersonalizedSample(input_text="a")
    b = PersonalizedSample(input_text="b")
    a.metadata["k"] = 1
    assert b.metadata == {}


def test_personalized_memory_is_append_only_and_list_all_returns_copy() -> None:
    mem = PersonalizedMemory()
    assert mem.list_all() == []

    s1 = PersonalizedSample(input_text="one")
    s2 = PersonalizedSample(input_text="two")
    s3 = PersonalizedSample(input_text="three")

    mem.add(s1)
    assert mem.list_all() == [s1]

    mem.extend([s2, s3])
    assert mem.list_all() == [s1, s2, s3]

    # list_all returns a defensive copy: mutating the result must not leak in.
    snapshot = mem.list_all()
    snapshot.clear()
    snapshot.append(PersonalizedSample(input_text="evil"))
    assert mem.list_all() == [s1, s2, s3]

    mem.clear()
    assert mem.list_all() == []


def test_personalized_memory_public_surface_is_frozen() -> None:
    # v1 PersonalizedMemory exposes exactly these mutators/readers. Adding a
    # destructive op (pop/remove/delete/update) should fail this test until
    # the schema version is intentionally bumped.
    public = {
        name
        for name in dir(PersonalizedMemory)
        if not name.startswith("_") and callable(getattr(PersonalizedMemory, name))
    }
    assert public == {"add", "extend", "list_all", "clear"}


def test_sleep_preprocessor_consumes_v1_shape() -> None:
    sample = PersonalizedSample(
        input_text="prompt-text",
        target_text="completion-text",
        signal="success",
        metadata={"task_id": "t-1", "score": 0.9},
        timestamp=1234567890.0,
    )

    artifacts = SleepPreprocessor().build([sample])

    assert len(artifacts.training_samples) == 1
    row = artifacts.training_samples[0]
    assert set(row.keys()) == {"input_text", "target_text", "signal", "metadata"}
    assert row["input_text"] == "prompt-text"
    assert row["target_text"] == "completion-text"
    assert row["signal"] == "success"
    assert row["metadata"] == {"task_id": "t-1", "score": 0.9}
    # Preprocessor must hand back an isolated metadata dict, not the original.
    assert row["metadata"] is not sample.metadata
    assert artifacts.metadata == {"source_sample_count": 1}
