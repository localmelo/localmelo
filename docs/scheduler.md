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
gate (`support/gateway/chat_gate.py`). A web console session's calls pass
through the same gate, which the gateway builds once from that table: both
surfaces share one `n_srv` budget and write one `calls.jsonl`. The gate submits the call to the
policy and either sends it at once or holds it until the policy admits it.
The page polls its chats every 2.5 seconds, so a held chat shows
`Waiting: <reason>` in the header's status pill and the status label, and
`● Waiting` in the sidebar; the reason is one of the policy's codes
(`Resources` for the first blocked call, `Priority` for later ones). A hold
is a wait: it never counts against the 600-second bound on a turn, and a
Stop during a hold ends the turn as cancelled. A held call also wakes at
least every 30 seconds (the focus window) to re-run the policy, so a hold
that only waited for the typing window to expire ends on its own, with no
new request and no timer in the gate. The gateway has one monitor task for
the GPU lock and one for the battery
([below](#the-gpu-lock-and-the-battery-s2b)); the gate itself still has no
timer.

The `[scheduler]` table of `config.toml` sets the policy:

| Key | Default | Meaning |
|---|---|---|
| `policy` | `"off"` | `off` records only; `fcfs` is first come, first served; `class` admits P1 before P2, P2 before P3, P3 before P4, first come within a class |
| `n_srv` | `8` | the model server's slots |
| `r_f` | `1` | slots kept for callers the gate cannot see (the CLI and direct runs); it is forced to 0 when `n_srv <= 2` |
| `n_fg` | `1` | background P2-P4 calls allowed in flight while a chat is focused (P1 is exempt) |
| `call_log` | `true` | write one JSONL call log per gateway run |
| `call_log_dir` | `""` | the log's parent directory; `~/.cache/localmelo/scheduler` when empty |
| `gpu_lock_file` | `""` | the shared GPU lock to read; empty means no lock reader |
| `lock_self` | `""` | the exact `holder:` (or `holder:run`) that is this bench's own lock |
| `lock_gate` | `"record"` | `record` only shows the lock; `wait` holds every workspace call while a foreign lock is present |

The lane is the chat being typed in: `POST /api/presence` sets it while you
type (the legacy `POST /v1/playground/presence` still does), and every run
sets it too — a turn through the turn routes, a retry, and a create with a
first turn. Its next call is admitted before
the background calls and bypasses their limit, but never the server's
`n_srv`. The non-lane limit is `max(1, n_srv - 1 - r_f)`; embedding,
`retrieve` and `estimate` calls are never held by the policy and never
orderable, but they count in flight (and a foreign lock under `lock_gate =
"wait"` holds them before the policy sees them, below): a foreground
embedding can fill the last `n_srv` slot, and the lane then waits for it.

A level is P1-P4 per chat, P2 by default; the chip in the chat header sets
it, and `fcfs` records it and ignores it. A level lives in the gateway's
memory: it is not written to `sessions.db`.

The gate writes one file per gateway run,
`<call_log_dir>/runs/<UTC yyyymmddThhmmss>/calls.jsonl` (mode 0600; a `-N`
suffix when two runs start in the same second; run directories older than the
thinking retention, 30 days by default, are removed at startup). Each line is
S1's schema-1 trace record for one call,
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

## The GPU lock and the battery (S2b)

The gateway reads two machine sources and records them; it never writes to
either. One task watches the shared GPU lock, one polls the battery, and both
are read-only. The lock is the four-line file the benchmark writers create
(`~/Developer/mlsys/experiments/.gpu-lock` on this Mac); the battery comes
from `pmset -g batt`. Direct runs and the interactive CLI do not read
either. The gateway app imports the monitor and starts it with the gateway
(`/v1/agent/run`'s requests and answers are unchanged), and `melo --serve`
logs one INFO line at startup and per change.

```toml
[scheduler]
gpu_lock_file = "~/Developer/mlsys/experiments/.gpu-lock"
lock_self = "local_ai-benchmark session"   # exact holder, or holder:run
lock_gate = "record"                       # "wait" to hold chats
```

The path is expanded with `Path.expanduser`; a missing file is no lock, not
an error. `open()` failures, invalid UTF-8, an empty file, a file above
64 KiB and a file without `holder:` are all **unreadable**, and an unreadable
lock is treated as foreign (fail closed). A stale lock (its ETA passed by
600 s) is still foreign; it is only displayed with `· ETA passed`, because
another session may be overrunning and localmelo must never delete its lock.
The two writers this Mac uses print:

```text
holder: local_ai-benchmark session
purpose: Night chain: Core AI llm-server gate, M8 ...
start: 2026-10-20T23:30:09-0400
eta: 2026-10-21T12:00:09-0400 (750 min)
```

```text
holder: agent-benchmark session (personal-agent-bench)
purpose: real-web UI eval: 27B oQ5 then Flash-Next oQ4e (untimed, port 8510)
start: 2026-10-27 03:06:08 EDT
eta: 330 min
```

Keys are `key: value`, one per line, first occurrence wins: `holder` is
required; `purpose`, `start`, `eta`, `run`, `mem_gib` are optional, and any
other key is ignored. `lock_self` matching is exact and case-sensitive: either
the whole `holder:` value, or `holder:run` when the file carries a `run:`
line. There is no prefix or substring matching, so
`lock_self = "local_ai-benchmark session"` does not match
`local_ai-benchmark session (copy)`; a typo makes every lock foreign and the
page names the holder. `mem_gib:` is parsed into the snapshot only (not
into `machine.jsonl`); nothing is reserved for it yet.

`lock_gate = "record"` (the default) is observation only: the page and the
machine log see the lock, holder and battery band, but the policy keeps its
default `Machine()` and no call waits, under `off`, `fcfs` and `class` alike.
`lock_gate = "wait"` holds every call the workspace gate sees while the lock
is foreign — planning, reflection **and embedding**, a deliberate exception
to S2's never-held rule, because an embedding is still a model-server call
that uses the same GPU. A call already pending in the policy sees the lock
through one `MachineChanged` and waits under `reservation_policy="respect"`;
a call already in flight is not stopped and finishes. When the lock clears,
the parked calls are submitted in `(submit time, call id)` order and the
policy decides them; the 600-second bound counts active time, so a lock hold
pauses it. The battery is record-only: `battery_gate_enforce` stays `False`,
and the table below is what S7 would enforce, not what S2b blocks.

| Battery | Band | What the class rule would block |
|---|---|---|
| 60% or more, or AC/unknown | 4 | nothing |
| 50-60% | 2 | P3 and P4 |
| 20-50% | 1 | P2, P3 and P4 |
| below 20% | 0 | every chat level |

The band is S1's `battery_band()` with its hysteresis: it falls to 2 at 50%
and 1 at 20%, and releases to 4 at 80% (or on AC). A `pmset` failure keeps the
last good reading and logs `battery unknown: <reason>` once; before the first good
reading the state is AC with no percentage, so a battery read never blocks a
chat. Chats have `Task.size = ""`, so the size arms of the battery rule cannot
fire for an S2 chat.

The lock watcher uses `kqueue` on macOS (a directory watch plus a file watch)
and a 2-second safety poll; where `kqueue` is unavailable it polls every
second. After a wake it waits 200 ms, re-reads and re-arms, so an O_EXCL
create and an atomic `mv -f` replacement are one state each, not an
absent-then-present pair. A re-read with no changed field emits nothing
(the ETA's `stale` flip is a change). The
battery poll is every 30 seconds in a thread with a 2-second `pmset` timeout.

Every changed snapshot is one line in `machine.jsonl`, beside `calls.jsonl` in
the same run directory (`0600`, pruned with the run directory, disabled with
`call_log = false`):

```json
{"at_s":0.0,"battery_pct":100.0,"battery_state":"charged","band":4,"eta":"","event":"start","harness":"localmelo","holder":"","lock":"none","on_ac":true,"purpose":"","readable":true,"run":"","schema":"localmelo.sched.machine","stale":false,"version":1,"wall_s":1793000000.0}
```

`event` is `start`, `lock`, `battery` or `stop`; the schema is its own, never
the call trace's. The page's machine line comes from the existing 2.5-second
poll: a foreign lock shows `GPU lock: <holder> · <purpose> (recorded)` or
`(chats wait)`, a self lock `GPU lock: yours`, an unreadable lock
`GPU lock unreadable`, and battery power `Battery 55% · discharging`; AC with
no lock shows nothing. A seeding create under a foreign lock in `wait` fails
fast with HTTP 409 naming the holder, because the seed embeddings would
otherwise send a model call before the chat exists.

## Tasks, workers and the scheduler store (S3a)

The gateway's `sessions.db` gains a `scheduler` component beside the session
tables. `support/gateway/task_store.py` creates version 1 of it — `tasks`,
`task_deps`, `priority_events`, `questions`, `sched_rules`, `sched_log` and
`idle_windows`, with indexes on `tasks(state, level_eff)`,
`priority_events(task_id, ts)` and `sched_log(ts)` — in one
`BEGIN IMMEDIATE` transaction, then records `('scheduler', 1)` in
`schema_components`. `PRAGMA user_version` stays the session schema's, and a
terminal `melo` never creates the component or reads its tables; a component
newer than 1 is refused like a newer `user_version`. `TaskStore.open()` is called by the console after it opens the store. A
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
current priority, and forgetting it preserves its identity while its row exists;
`set_task_level()` handles the chip's `P0`-`P5`, `auto` and `first`, and
`held(task_id)` reports the gate's reason as it does for a chat. Task routes
write priority changes through the store.

S3a.2 adds [task routes](api.md#scheduler-tasks-s3a2) and in-process workers.
New user tasks are queued by effective priority, rank and creation time, with
at most four resident workers; a parked running worker still occupies its slot.
Each uses `mode="gateway"`, `memory=False` and the console's shared gate for
LLM and embedding calls. Stop cancels the runner and finishes the gate task.
The gate registry discards a task after retention removes its row, not merely
when it is forgotten.

Rows survive gateway restart, but startup marks surviving nonterminal tasks
failed with `gateway restarted; recovery arrives in S3b`. Full recovery is
deferred. A repeat `request_key` only returns the existing task. Workspace
validity is checked at creation and again at run or resume; a folder that
becomes invalid fails the task with `invalid_workspace`. Deadlines are rejected
and service, wait and tool accounting is deferred. Pause parks a task at P5
and takes effect at the next model call: running tools finish and an answered
pending approval waits until resume. Resume restores the prior priority unless
a new level is supplied; if retention pruned the pause log it falls back to
computed P2. Resuming an unpaused task returns 409 `not_paused`. The task
PATCH route rejects `first` with 400; it is deferred to S6b.

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
