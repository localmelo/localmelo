"""Optional scaffold for future local-backend deployment and runtime management.

This package is **not** part of localmelo core behaviour today.  It exists as a
stable placeholder so that future work on local model serving (e.g. lifecycle
management for vLLM / SGLang / MLC processes) has a clear integration point
without needing to touch the core agent, memory, or executor subsystems.

No runtime logic, subprocess management, or network calls live here.
"""

from localmelo.support.melo_serving.manifest import (
    ServingBackend,
    ServingManifest,
)

__all__ = [
    "ServingBackend",
    "ServingManifest",
]
