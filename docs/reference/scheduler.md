# Scheduler: policy core and simulator

localmelo will share one local model server between your conversation with
the front (the lane), worker tasks and chats. The scheduler decides which
model calls go to the server, in what order, and why a task waits. This page
is for maintainers and for the bench: what the code under
`support/scheduler/` is, how to run a simulation, and how to import bench
traces.

> [!NOTE]
> The web workspace's chats are gated: the gateway wraps each chat's chat and
> embedding providers with the [scheduler gate](../guides/scheduler.md#the-gate-on-chats-s2). `[scheduler] policy`
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

## Tasks, workers and the scheduler store (S3a)

The gateway's `sessions.db` has a `scheduler` component beside the session
tables. `support/gateway/task_store.py` creates version 3 with `tasks`,
`task_deps`, `priority_events`, `questions`, `sched_rules`, `sched_log`,
`idle_windows`, `park_priority` and recovery storage, with indexes on `tasks(state, level_eff)`,
`priority_events(task_id, ts)` and `sched_log(ts)`, in one `BEGIN IMMEDIATE`
transaction. `TaskStore.open()` also migrates versions 1 and 2 atomically and
records the component version in `schema_components`. `PRAGMA user_version`
stays the session schema's, and a terminal `melo` never creates the component
or reads its tables; a component newer than 3 is refused. The console calls
`TaskStore.open()` after opening the store. A
gateway takes the session file's owner
lock: a second gateway process is refused rather than concurrently migrating
or writing the same file.

Version 3 adds dormant recovery provenance, durable operation receipts and
process identity records. Recovery prepare and commit both advance the shared
park generation; interrupted recovery preserves a pending pause until explicit
resume or a terminal outcome. Missing provenance and missing process records
are unknown, not proof that a task never ran or that its processes are gone.
The internal legacy-adoption seam records explicit orphan-risk confirmation but
does not itself queue or launch a task. Process capture has no completeness
proof. Startup classification uses the recovery journal; process verification,
controlled restart scheduling and user-facing confirmation remain later work.

A task row's id is its `kind="task"` session's id. `create_task()` writes the
session, the task, its first `priority_events` row and a `sched_log` row in
the same transaction, so a crash leaves none of them. Pass the already
validated real `workspace` to `create_task()`; it is saved on the session,
while the original request stays in `tasks.spec`. The current priority is in
`tasks.level`, not the create-time value in `tasks.spec`. `UNIQUE(origin,
request_key)` makes a repeated create idempotent: the same payload returns
the stored row with `repeated=True`, a different payload raises
`TaskConflictError`, which the task route maps to 409. `set_level()` records actual
level changes in `priority_events` and writes `sched_log`; unchanged levels
do not add priority labels. The user's P0 is single, so a
second user P0 returns the prior one to P1 with no `user_rank` in the same
transaction, regardless of task origin. Finished tasks cannot be reprioritized.
`set_priority()` also takes `auto` (P2, computed) and `first`
(keeps the level, sets a rank that sorts earlier first choices first).
`apply_retention()` clears `sched_log` and
`idle_windows` older than the session store's thinking retention — 30 days
by default. The task store registers it as a hook of the session store's
sweep, so the gateway's daily `apply_retention()` prunes both, and
`TaskStore.open()` prunes them too after downtime. Tasks,
`priority_events`, `questions` and rules are not pruned.

`support/gateway/task_policy.py` is the code-side boundary, never a model's
JSON:

- `TaskSpec.from_payload()` normalizes an untrusted request (accepting `prompt`
  as an alias for `objective`) at computed P2, ignoring an untrusted `level`
  and rejecting a payload-supplied `level_source`. Deferred optional fields,
  including `inputs`, `deadline`, and `budget_s`, are rejected rather than
  silently discarded; the task route also rejects `deadline`. After authorizing the actor,
  `TaskSpec.from_user_payload()` pins an explicit user-selected P0–P5; without
  an explicit level it uses computed P2. The adapter does not itself
  authenticate or authorize a caller.
- `validate_task_workspace()` resolves the real folder. The configured root
  must be absolute or `~/` and cannot resolve to `/` or the home folder. A
  user's task may use any existing folder strictly inside `~` with no dot
  component and nothing under `~/Library`. For other origins an explicit
  folder must be strictly inside the configured root; when omitted, pass a
  trusted `task_id` to create a non-symlink child of that root. The root
  defaults to `~/localmelo-workspace` when not configured.
- `worker_tool_allowlist()` returns the gateway built-ins
  (`file_read`, `file_write`, `python_exec`; never `shell_exec`) and, for
  `user` and `user_rule` only, remembered approvals, unmarked MCP tools and
  session grants. `front`, `harness`, `agent`, `channel` and `sleep` get
  none of those, so every marked call needs a fresh approval.

`ChatGate.register_task()` gives a worker its task lifecycle. Its calls are
submitted with `task_id` and `session_kind="task"`, never take the lane, and
a call ending does not finish the worker. The runner calls `finish_task()`
with `done`, `failed`, `cancelled` or `expired`.
It cancels parked calls but retains admitted calls until their providers end;
new calls for a terminal task are refused and logged. Ending a live task must
go through `finish_task()`; a repeat does not change its terminal phase.
`update_task()` carries caller-owned fields such as nonterminal phase and wait
reason, but not level or source. Re-registering an existing task preserves its
current priority, and forgetting it alone preserves its identity;
`set_task_level()` handles the chip's `P0`-`P5`, `auto` and `first`, and
`held(task_id)` reports the gate's reason as it does for a chat. Task routes
write priority changes through the store.

S3a.2 adds [task routes](api-v2.md#scheduler-tasks-s3a2) and in-process workers.
New user tasks are queued by effective priority, rank and creation time, with
at most four resident workers; a successfully parked worker releases its slot.
Each uses `mode="gateway"`, `memory=False` and the console's shared gate for
LLM and embedding calls. Stop cancels the runner and finishes the gate task.
A retention sweep discards terminal gate identities whose task rows have
passed the retention cutoff, even though the rows remain available for
inspection; forgetting alone does not discard an identity.

Rows survive gateway restart. Startup preserves terminal tasks and proved
never-started queued or parked tasks, and interrupts uncertain tasks without
replaying them. An already-started parked task retains its recorded card; an
uncertain task without usable evidence stays without a card. An execution-linked
committed run outcome can be reconciled without rerunning it. Pending pause intent
and its priority snapshot survive interruption. Recovered tasks with uncertain
history are fenced from resume and priority changes; Stop remains available.
Proved-never-started runnable queued tasks can start after a durable launch fence
and workspace validation; ordinary creation does not yet enable this proof.
Orphan verification and informed Resume are later
work; a missing process record is not proof that a process has stopped.
A repeat `request_key` only returns the existing task. Workspace
validity is checked at creation and again at run or resume; a folder that
becomes invalid fails the task with `invalid_workspace`. Deadlines are rejected
and service, wait and tool accounting is deferred. Pause requests P5 and
waits for a safe point while running providers and tools finish. A held plan,
reflection or undelivered approval can park and release its resident worker;
undelivered approvals are withdrawn and re-asked on resume. Resume uses a card
built from persisted evidence. A task paused before it starts does not
start a worker until resumed. Resume restores the prior priority unless a new
level is supplied; without a saved pause snapshot (such as a task created or
set at P5), resume falls back to computed P2. Resuming an unpaused task returns
409 `not_paused`. The task PATCH route rejects `first` with 400; it is deferred
to S6b.

## S3b.1a foundations

Pause saves the prior level, source, reason and rank in `park_priority` in the
same transaction as the P5 transition. This snapshot survives `sched_log`
retention and is consumed on resume. Migration reconstructs snapshots for v1
paused P5 rows when the retained audit trail supports it; without a usable
pause record there is no snapshot and resume falls back to P2. Migration checks
for later `task.level`, `task.first` and `task.resume` audit entries; a later
runnable priority may instead leave the task no longer paused. Explicit
priority changes can update or invalidate the snapshot. The row also carries a
monotonic `generation` and an `operation_id` for guarded transitions; it remains
across resume and is removed with the task. `pause_pending` records a pause
requested while a worker is still running. Before the first identity row exists,
the worker coordinator fences stale safe-point callbacks.

`ChatGate.try_park()` is an internal, synchronous primitive for a single
pending, unadmitted task plan or reflection call bound to a turn. On the same
event loop it fences the call from admission and asks the runner to cancel that
exact turn with cause `park`. A stale or already-stopping turn cannot win. If
cancellation is refused or raises, the fence is rolled back unless the call has
ended. The primitive does not itself integrate parking with a worker.

`SessionStore.component_event_transaction()` and `append_component_event()`
allow gateway component rows and reserved out-of-turn session events to commit
atomically. Task-store pause emits `sched.pause_requested`; resume and terminal
transitions also emit committed events with their identity. `ParkStore` uses the
same transaction seam for internal hold, resume and park transitions,
recording hold coordinates supplied by the caller and emitting `sched.held`,
`sched.resumed`, `sched.parked`, `sched.park_retrying` or `sched.interrupted`;
its returned record includes the current
`pause_pending` marker. It does not change priority. Its card builder
uses the task spec and persisted events only, bounds and redacts text, omits
tool outputs and arguments other than file paths, and treats completion as
unverified. Worker integration holds the provider or approval delivery fence
until the matching durable transition is committed, and resumes a parked worker
through its card. An unsaved park is retried without opening the fence; after
three failures an interrupted card is saved. If persistence remains unavailable,
the worker stays held and resume/stop return 503 until finalization succeeds.
Cleanup failures quarantine the worker, retain its slot, and report interruption
if retries fail; shutdown gives quarantined cleanup a bounded final attempt.

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
