"""The trace-driven simulator, its workloads and metrics, and the trace importer.

The simulator replays workloads through the policy core over a server
model (design 8): synthetic workloads or traces of real runs (the bench
proxy's ``calls.jsonl``, localmelo's ``history.db``), imported into the
trace format that S2's call log also writes. It is synchronous and
deterministic, and every result says ``validated: false`` until a live
window passes the 15% check (design 8.1). Nothing here is wired into the
gateway or the CLI; only tests import it.
"""

from __future__ import annotations

from localmelo.support.scheduler.sim.compare import compare, run_sweep
from localmelo.support.scheduler.sim.engine import SimResult, run
from localmelo.support.scheduler.sim.importer import (
    ImportReport,
    import_calls,
    import_history,
)
from localmelo.support.scheduler.sim.metrics import compute_metrics
from localmelo.support.scheduler.sim.server import ServerParams
from localmelo.support.scheduler.sim.trace import (
    TaskTemplate,
    TaskTrace,
    TraceCall,
    read_trace,
    write_trace,
)
from localmelo.support.scheduler.sim.workloads import SEEDS, Workload, make_workload

__all__ = [
    "run",
    "SimResult",
    "Workload",
    "make_workload",
    "SEEDS",
    "ServerParams",
    "import_calls",
    "import_history",
    "ImportReport",
    "TraceCall",
    "TaskTrace",
    "TaskTemplate",
    "read_trace",
    "write_trace",
    "compute_metrics",
    "run_sweep",
    "compare",
]
