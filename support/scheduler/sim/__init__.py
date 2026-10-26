"""The trace-driven simulator and the importer of real traces (design 8).

Traces of real runs (the bench proxy's ``calls.jsonl``, localmelo's
``history.db``) are imported into one trace format that S2's call log
also writes. Nothing here is wired into the gateway or the CLI; only
tests import it.
"""

from __future__ import annotations

from localmelo.support.scheduler.sim.importer import (
    ImportReport,
    import_calls,
    import_history,
)
from localmelo.support.scheduler.sim.trace import (
    TaskTemplate,
    TaskTrace,
    TraceCall,
    read_trace,
    write_trace,
)

__all__ = [
    "import_calls",
    "import_history",
    "ImportReport",
    "TraceCall",
    "TaskTrace",
    "TaskTemplate",
    "read_trace",
    "write_trace",
]
