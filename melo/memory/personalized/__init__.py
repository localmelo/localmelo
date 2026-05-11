from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

PERSONALIZED_SAMPLE_SCHEMA_VERSION = "v1"
# v1 frozen field tuple: any rename/reorder/add/remove is a breaking change.
# Bump the version constant in the same commit so downstream (Track 4 sleep
# pipeline) can branch on schema version. Guarded by
# tests/memory/test_personalized_schema.py.


@dataclass
class PersonalizedSample:
    """Training-only personalization sample (schema v1, frozen).

    v1 field contract: input_text, target_text, signal, metadata, timestamp.
    These samples are not queried during normal online inference. They are
    collected during runtime and later consumed by the sleep pipeline.
    """

    input_text: str
    target_text: str = ""
    signal: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class PersonalizedMemory:
    """Append-only store for sleep-time personalization samples."""

    def __init__(self) -> None:
        self._samples: list[PersonalizedSample] = []

    def add(self, sample: PersonalizedSample) -> None:
        self._samples.append(sample)

    def extend(self, samples: list[PersonalizedSample]) -> None:
        self._samples.extend(samples)

    def list_all(self) -> list[PersonalizedSample]:
        return list(self._samples)

    def clear(self) -> None:
        self._samples.clear()


__all__ = [
    "PERSONALIZED_SAMPLE_SCHEMA_VERSION",
    "PersonalizedMemory",
    "PersonalizedSample",
]
