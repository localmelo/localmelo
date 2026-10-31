# Guide: Scheduler

This guide explains how the scheduler decides when chat requests may reach a shared model server, and how to run a simulation. Read it before you change scheduling; the policy, the store and the simulator are described in the [scheduler reference](../reference/scheduler.md).

## Tasks from the main chat

![The main chat starts tasks at a level from your words](../assets/img/main-chat-tasks.webp)

- **Who starts a task.** The main chat, when you ask it for work, owns the task (`front_id`); `/task …` or **Run as a task** starts one without a model call. The [web console guide](web-console.md#hand-work-to-the-main-chat) shows how it looks.
- **Its level.** Code reads only your own message: urgency words or a deadline today give Today (P1), "no rush" gives Whenever (P3), anything else Normal (P2). The reason is kept with the task (`you said "urgent"`). A page, a file or a task's report never sets a level.
- **Your level stays.** A level you set (the line's **Change**, the panel, `melo task level`) is yours: the scheduler leaves it alone.
- **Changes need your words.** The main chat changes a task (stop, pause, resume, a level) only when your latest message asks for it and names the task; otherwise it shows a suggestion you apply with one tap.
- **Words in default mode.** Levels read Now, Today, Normal, Whenever, If idle and Later everywhere a level shows; developer mode adds the P-codes. `melo task` takes a word wherever it takes a P-code (`melo task level ID today`).

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
type (the legacy `POST /v1/playground/presence` does so only when the
playground is enabled, under the in-progress security-1b changes), and every run
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

Not gated, exactly as before: direct mode (`melo --url … -- <task>` and
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
(`/v1/agent/run`'s requests and answers are unchanged), and `melo serve`
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
