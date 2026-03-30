"""Declarative manifest describing a desired local serving deployment.

This module defines the placeholder data structures that future runtime
management code will consume.  Today these are inert configuration objects
with no side-effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ServingRuntime = Literal["vllm", "sglang", "mlc", "ollama"]


@dataclass(frozen=True)
class ServingBackend:
    """Describes a single model endpoint that a local runtime should expose."""

    runtime: ServingRuntime
    model_id: str
    port: int = 0  # 0 = let the runtime pick a free port
    extra_args: dict[str, str] = field(default_factory=dict)


@dataclass
class ServingManifest:
    """Collection of backends that should be deployed together.

    A manifest is purely declarative — it captures *what* should run, not
    *how* to start it.  Future runtime managers will consume this to
    orchestrate local processes.
    """

    backends: list[ServingBackend] = field(default_factory=list)
    label: str = ""
