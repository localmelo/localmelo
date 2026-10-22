from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class SleepStage(StrEnum):
    IDLE = "idle"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SleepState:
    """Current sleep-mode execution state for one personalization job."""

    stage: SleepStage = SleepStage.IDLE
    dataset_version: str = ""
    adapter_version: str = ""
    started_at: float = 0.0
    updated_at: float = field(default_factory=time.time)
    last_error: str = ""


class SleepStateStore:
    """In-memory state holder for sleep-mode runs."""

    def __init__(self) -> None:
        self._state = SleepState()

    def get(self) -> SleepState:
        return self._state

    def set(self, state: SleepState) -> None:
        state.updated_at = time.time()
        self._state = state

    def update_stage(self, stage: SleepStage) -> SleepState:
        self._state.stage = stage
        self._state.updated_at = time.time()
        return self._state


__all__ = [
    "SleepStage",
    "SleepState",
    "SleepStateStore",
]
