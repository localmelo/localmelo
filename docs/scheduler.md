# Scheduler: policy core and simulator

localmelo will share one local model server between your conversation with
the front (the lane), worker tasks and chats. The scheduler decides which
model calls go to the server, in what order, and why a task waits. This page
is for maintainers and for the bench: what the code under
`support/scheduler/` is, how to run a simulation, and how to import bench
traces.

> [!NOTE]
> The web workspace's chats are gated: the gateway wraps each chat's chat and
> embedding providers with the gate described below. `[scheduler] policy`
> defaults to `off`, which only records; `fcfs` and `class` are the two
> enforcing arms. Direct runs and the terminal CLI are not gated.

## What is here

- **The policy core** (`localmelo.support.scheduler`). `decide()` is a pure
  function of an in-memory `State`: machine gates, levels with aging and
  inheritance, the lane's reserve, the background cap, credit floors for P2
  and P3, park and yield marks, and one reason per waiting task.
  `dispatch()` admits the lane first and falls back to first-come order if
  `decide()` raises. `advance()` and `apply_dispatch()` hold every change
  over time, so nothing else keeps state.
- **The simulator** (`localmelo.support.scheduler.sim`). `run()` replays a
  workload through the same functions over a simulated server: slots, its
  own first-come queue, throughput shared as Θ(n) = n^0.4, and a prefix
  cache. `make_workload()` builds the synthetic workloads W1 (a day), W2 (a
  burst), W2c (chats), W3 (a deadline crunch) and W5 (a long tail);
  `compute_metrics()` scores a run; `run_sweep()` and `compare()` apply the
  decision rule.
- **The importer**. `import_calls()` reads the bench proxy's `calls.jsonl`
  files and `import_history()` a run's `history.db`, for templates and tool
  times. `write_trace()` and `read_trace()` keep traces in one JSONL format,
  which the gate's call log will write too.

## Arms and mode off

Each arm is a `Params` preset (`params_for_arm()`), and each adds one family
of mechanisms to the one before it, so comparing neighbours isolates it.

| Mode | Order of task calls | Floors | Yield marks and resident-slot claims |
|---|---|---|---|
| `off` | No gate: every call goes straight to the server's queue | — | — |
| `fifo` | One queue by submission time; classes ignored | Off | Off |
| `class` | Strict classes P0 to P4; first come within each | Off | On |
| `las2` (default) | Strict classes; P1 by deadline, P2 and P3 by least attained service, P4 oldest first | On | On |

Every arm keeps the lane's fast path, the reserve, the background cap, the
machine gates and the park marks. Mode `off` still books charges and levels,
so its metrics compare with the gated modes; with no gate, nothing answers
the lane by code.

## The gate on chats (S2)

Every model call a web workspace chat makes passes through one in-process
gate (`support/gateway/chat_gate.py`). The gate submits the call to the
policy and either sends it at once or holds it until the policy admits it.
The page polls its chats every 2.5 seconds, so a held chat shows
`Waiting: <reason>` in the status label and `● Waiting` in the sidebar; the
reason is one of the policy's codes (`Resources` for the first blocked call,
`Priority` for later ones). A hold is a wait: it never counts against the
600-second bound on a turn, and a Stop during a hold ends the turn as
cancelled. A held call also wakes at least every 30 seconds (the focus
window) to re-run the policy, so a hold that only waited for the typing
window to expire ends on its own, with no new request and no timer in the
gate.

The `[scheduler]` table of `config.toml` sets the policy:

| Key | Default | Meaning |
|---|---|---|
| `policy` | `"off"` | `off` records only; `fcfs` is first come, first served; `class` admits P1 before P2, P2 before P3, P3 before P4, first come within a class |
| `n_srv` | `8` | the model server's slots |
| `r_f` | `1` | slots kept for callers the gate cannot see (the CLI and direct runs); it is forced to 0 when `n_srv <= 2` |
| `n_fg` | `1` | background P2-P4 calls allowed in flight while a chat is focused (P1 is exempt) |
| `call_log` | `true` | write one JSONL call log per gateway run |
| `call_log_dir` | `""` | the log's parent directory; `~/.cache/localmelo/scheduler` when empty |

The lane is the chat being typed in: `POST /v1/playground/presence` sets it
while you type, and every run sets it too. Its next call is admitted before
the background calls and bypasses their limit, but never the server's
`n_srv`. The non-lane limit is `max(1, n_srv - 1 - r_f)`; embedding,
`retrieve` and `estimate` calls are never held and never orderable, but they
count in flight: a foreground embedding can fill the last `n_srv` slot, and
the lane then waits for it.

A level is P1-P4 per chat, P2 by default; the chip in the chat header sets
it, and `fcfs` records it and ignores it. A level lives in the gateway's
memory: it is not written to `sessions.db`.

The gate writes one file per gateway run,
`<call_log_dir>/runs/<UTC yyyymmddThhmmss>/calls.jsonl` (mode 0600, run
directories older than the thinking retention, 30 days by default, are
removed at startup). Each line is S1's schema-1 trace record for one call,
with the submit, admit and end times, the hold (`admit_s - submit_s`), the
tokens and the charge. Read a file back with `read_trace()`:

```bash
uv run python -c '
import sys
from pathlib import Path
from localmelo.support.scheduler.sim.trace import read_trace
for task in read_trace(Path(sys.argv[1])):
    print(task.task_key, len(task.calls))
' "$(ls -t ~/.cache/localmelo/scheduler/runs/*/calls.jsonl | head -1)"
```

```text
84dd6b2355c5 1
```

Not gated, exactly as before: direct mode (`melo --base-url … -- <task>` and
`melo` with a config), the interactive CLI, and the legacy `/v1/agent/run`
route. They build their providers in their own process or inside `melo/`, so
the gate never sees their calls, and `r_f` keeps a slot for them.

## Run one simulation

A 30-minute W2 under each mode takes about a second:

```bash
uv run python -c '
from dataclasses import replace
from localmelo.support.scheduler.sim import compute_metrics, make_workload, run

wl = replace(make_workload("W2", 1), horizon_us=30 * 60 * 10**6)
for mode in ("off", "fifo", "class", "las2"):
    result = run(wl, mode)
    m = compute_metrics(result)
    print(mode, "reply p95", m.reply_whole_s[1], "s; finished",
          sum(m.finished_by_class), "of", len(result.tasks))
'
```

```text
off reply p95 9.211797 s; finished 6 of 6
fifo reply p95 9.494473 s; finished 6 of 6
class reply p95 10.787873 s; finished 6 of 6
las2 reply p95 7.690171 s; finished 6 of 6
```

`run()` takes `Params` and `ServerParams` (slots, Θ's exponent, the cache
model, foreign calls the gate cannot see), and `keep_log=True` keeps every
decision as a JSON line. A full sweep (five workloads, four modes, 20 seeds)
is CPU-heavy: run it outside a bench window.

## Import bench traces

Point `import_calls()` at a directory: it finds every `calls.jsonl` below it.
Only localmelo's runs are kept by default (`harnesses=("localmelo",)`),
runs with overlapping calls are left out, and every line it drops is counted
by reason.

A chat call's kind is the `X-Localmelo-Call-Kind` header localmelo sends.
The proxy keeps request headers only with `--log-bodies`, in the side file
that the record's `bodies.headers` names (relative to the record's run
directory); the importer reads only that file for the kind, and a record
without it keeps the endpoint/role guess. `history.db` is not read for
kinds: no `llm_calls` row names the proxy record it answered, and a
retrieval request bypasses the agent's call log and makes no row at all,
so no sound join exists. `report.kind_sources` counts kept records for
each source (`header`, `history`, `guess`; `history` is 0 today), and
`report.kind_unknown` counts the header values the importer does not know,
each of which keeps its guess.

```bash
uv run python -c '
import sys
from pathlib import Path
from localmelo.support.scheduler.sim import import_calls, make_workload
from localmelo.support.scheduler.sim.trace import templates_from_traces

traces, report = import_calls(Path(sys.argv[1]))
print(report.files, "files,", report.records_kept, "records kept,", report.tasks, "tasks")
print("dropped:", dict(report.dropped), "errors:", report.errors)
print("kinds:", dict(report.kind_sources), "unknown:", report.kind_unknown)
templates = templates_from_traces(traces)
print(len(make_workload("W2", 1, templates=templates).tasks), "W2 tasks on",
      len(templates), "imported templates")
' <runs-directory>
```

A run's tool times come from its `history.db` (bench runs keep it at
`<run>/mem/history.db`), opened read-only, also while a run still writes it.
Only tool durations are read: no query, thought or tool output. The pool
feeds `synthetic_templates(seed, tool_pool=...)` in `sim/workloads.py`.

```bash
uv run python -c '
import sys
from pathlib import Path
from localmelo.support.scheduler.sim import import_history
from localmelo.support.scheduler.sim.importer import tool_time_pool

tasks, report = import_history(Path(sys.argv[1]), label="run-1")
print(report.tasks, "tasks; tool times", tool_time_pool(tasks))
' <run>/mem/history.db
```

## Results are not validated

Every `SimResult` and every `compare()` report says `validated: false`. The
server model is not validated yet: no default changes because of a
simulation until a live window of the gate passes the 15% check against the
simulator. `compare()` gives each candidate arm a verdict
(`fails_guardrail`, `below_margin`, `insufficient_data` or `meets_margin`),
never "adopt".

## Determinism

A workload draws every random number before `run()`, from `random.Random`
streams seeded through sha256, and the scheduler reads no clock, so equal
inputs give the same decision log (`log_sha256`) in any process and with
any hash seed. Inputs are sorted before use and JSON is written with sorted
keys, so a written trace or report is the same file every time.

## Known gap: prefill is not shared evenly

Probe P-1 (oQ5 with MTP on oMLX 0.7.1rc1, 2026-10-26) measured Θ(n)/Θ(1) of
1.5, 1.6, 2.0, 1.8, 2.0 and 2.2 for n = 2, 3, 4, 5, 7 and 8, so the server
model uses Θ(n) = n^0.4 (a least-squares fit; 0.5 and 1.0 remain arms).
Decode is shared evenly and a call can join a running batch.

Prefill is not shared evenly. A front call's time to first token grows by
about 4.6 s for each 3k-token background prefill ahead of it (5.7, 10.5,
18.4 and 32.5 s with n = 1, 2, 4 and 7 background calls), and one 16k-token
background prefill holds it 5.3 s (n = 1) or 10.5 s (n = 2). The simulator
still shares prefill and decode at one rate, so it understates how long a
lane call waits behind long background prefills. Modelling prefill as a
queue in arrival order belongs to the gate's phase, where holding new long
prefills while a front call waits is the protection these numbers call for.
