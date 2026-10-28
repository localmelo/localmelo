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

## Task command

`melo-task` is a separate gateway client; it never starts the gateway or a
model server. It connects to the configured gateway or an explicit loopback
`--gateway` URL, uses the private gateway bearer file and refuses redirects
and environment proxies. Start the gateway first. A missing or unsafe token,
unreachable gateway or rejected request is an error, not a reason to fall
back to unauthenticated requests. The token is not accepted as an argument.

- `new TEXT`, `new --brief FILE` or `new --stdin` submits one objective;
  optional `--title`, `--workspace`, `--level P0` through `P5` and
  `--request-key` set fields explicitly. A file or stdin is task content,
  not a trusted urgency signal. Input is nonempty UTF-8, at most 1 MiB and
  100,000 characters. A generated request key allows safe retry; on an
  ambiguous network error it is printed to stderr. Keep the same text,
  options and key when retrying. `new` reports admission, not completion.
- `list [--state STATE] [--level Pn]` lists matching tasks, not their admission
  order. `show ID` inspects one. `pause ID`, `resume ID [--level Pn]`,
  `stop ID` and `level ID Pn|auto` control tasks. `first` and deadlines are
  not supported. Pause does not interrupt a running tool.
- `follow ID [--after EVENT_ID] [--wait]` streams stored session events and
  rechecks task state. It reconnects using durable event IDs, not run-local
  sequence numbers. A terminal state is a snapshot; retention and finalization
  races mean event completeness is not guaranteed. Repeated disconnects or
  invalid events are errors; Ctrl-C does not stop the task. Without `--wait`, a
  terminal observation exits 0; with `--wait`, done exits 0, failed 1, and
  cancelled or expired 2. Other successful operations exit 0; usage errors
  exit 2 and network, authentication or output failures exit 1.
- `--json` emits one versioned JSON object per non-stream command, or NDJSON
  for follow. Argument parser errors print usage to stderr instead of JSON.
  Check the terminal record and process exit; EOF alone is not success. Error
  details and terminal output may contain sensitive task data;
  review output before sharing. This is not the direct worker result schema.

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
tables. `support/gateway/task_store.py` creates version 2 with `tasks`,
`task_deps`, `priority_events`, `questions`, `sched_rules`, `sched_log`,
`idle_windows` and `park_priority`, with indexes on `tasks(state, level_eff)`,
`priority_events(task_id, ts)` and `sched_log(ts)`, in one `BEGIN IMMEDIATE`
transaction. `TaskStore.open()` also migrates version 1 to 2 atomically and
records the component version in `schema_components`. `PRAGMA user_version`
stays the session schema's, and a terminal `melo` never creates the component
or reads its tables; a component newer than 2 is refused. The console calls
`TaskStore.open()` after opening the store. A
gateway takes the session file's owner
lock: a second gateway process is refused rather than concurrently migrating
or writing the same file.

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
at most four resident workers; a parked running worker still occupies its slot.
Each uses `mode="gateway"`, `memory=False` and the console's shared gate for
LLM and embedding calls. Stop cancels the runner and finishes the gate task.
A retention sweep discards terminal gate identities whose task rows have
passed the retention cutoff, even though the rows remain available for
inspection; forgetting alone does not discard an identity.

Rows survive gateway restart, but startup marks surviving nonterminal tasks
failed with `gateway restarted; recovery arrives in S3b`. Full recovery is
deferred. A repeat `request_key` only returns the existing task. Workspace
validity is checked at creation and again at run or resume; a folder that
becomes invalid fails the task with `invalid_workspace`. Deadlines are rejected
and service, wait and tool accounting is deferred. Pause parks a task at P5
and takes effect at the next model call: running tools finish and an answered
pending approval waits until resume. A task paused before it starts does not
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
across resume and is removed with the task. `pause_pending` is reserved for
S3b.1b and is not set by worker code in S3b.1a. Before a task's first pause or
park creates its identity row, preventing stale worker safe-point callbacks relies
on the S3b.1b worker coordinator; these foundations do not provide that fence.

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
`sched.resumed` or `sched.parked`; its returned record includes the current
`pause_pending` marker. It does not change priority. Its card builder
uses the task spec and persisted events only, bounds and redacts text, omits
tool outputs and arguments other than file paths, and treats completion as
unverified. These primitives do not implement worker hold/park integration or
ship a new API. Worker API pause behavior remains as in #119; S3b.1b follows.

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
