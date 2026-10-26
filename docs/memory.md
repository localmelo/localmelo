# Memory

This page explains what localmelo remembers, how to keep memory between runs,
and how to inspect, reset and protect it.

The maintainer's memory design, parts of which are not built yet, is on the
website's [Memory design](https://localmelo.github.io/localmelo/en/memory-design.html)
page.

## Quick reference

| To | Do this |
|---|---|
| Keep memory between runs | `export LOCALMELO_PERSIST_MEMORY=1` |
| Choose the directory | `export LOCALMELO_MEMORY_DIR=…` (default `~/.cache/localmelo/memory`) |
| Keep the memory of one run in a directory | `melo --memory-dir ~/melo-memory "<task>"` |
| Turn long-term memory off for one run | `melo --memory off "<task>"` |
| See past runs | `sqlite3` queries; see [Inspect memory](#inspect-memory) |
| Delete memory | Stop localmelo, then remove the files; see [Reset memory](#reset-memory) |

## What localmelo remembers

| Memory | What it holds | Where |
|---|---|---|
| Working memory | The conversation of a run, or of a gateway session: the last 20 messages. | RAM only. |
| History | Every task with its status and result, and every step with its tool call and result. | RAM, or `history.db`. |
| Long-term memory | Short summaries of steps, and the reflections of past runs. On each step, the 5 entries closest to the task are recalled. Needs an embedding backend. | RAM, or `long_term.db`. |

By default everything stays in RAM and is lost when the process ends. In RAM,
long-term memory lasts as long as its agent: one direct run, or one gateway
session.

## Keep memory between runs

```bash
export LOCALMELO_PERSIST_MEMORY=1
melo "What is 6*7?"
```

- `LOCALMELO_PERSIST_MEMORY` turns persistence on with any non-empty value,
  even `0` or `false`. To turn it off, run `unset LOCALMELO_PERSIST_MEMORY`.
- The files go to `~/.cache/localmelo/memory/`. `LOCALMELO_MEMORY_DIR`
  chooses another directory. Its value is used as given: a `~` that the
  shell does not expand, such as a quoted one, is not expanded either.
- `--memory-dir DIR` turns persistence on for one direct run, in `DIR`,
  whatever the variables say. `~` is expanded:

  ```bash
  melo --memory-dir ~/melo-memory "What is 6*7?"
  ```

- A gateway uses the variables it was started with. All its sessions save to
  the same files.
- Recalling long-term memory in a later run needs both persistence and an
  embedding backend.

## Turn long-term memory off

`--memory off`, or `AgentOptions(memory=False)` in Python, turns long-term
memory off for a run: nothing is recalled from it or written to it, and
reflections are not promoted. Working memory and history work as usual.
`embedding_backend = "none"` in the config file has the same effect for every
run.

## Files on disk

```text
~/.cache/localmelo/memory/
  history.db      tasks and steps
  long_term.db    long-term memory, with an embedding backend and memory on
```

- SQLite may also create `history.db-wal` and `history.db-shm` (and the same
  for `long_term.db`). They belong to the database; leave them in place.
- The directory is created when an agent starts: at the start of a direct
  run, or when a gateway session is created.
- The gateway's web UI keeps its own files in
  `~/.cache/localmelo/playground/`.

## Inspect memory

The files are SQLite databases. Read them with the `sqlite3` command-line
tool, and use only `SELECT` queries:

```bash
cd ~/.cache/localmelo/memory

# The tables
sqlite3 history.db ".tables"

# The 10 most recently saved tasks
sqlite3 history.db "SELECT task_id, status, substr(query, 1, 60) FROM tasks ORDER BY rowid DESC LIMIT 10"

# How runs ended
sqlite3 history.db "SELECT status, COUNT(*) FROM tasks GROUP BY status"

# The steps of one task
sqlite3 history.db "SELECT seq, tool_call_json, substr(tool_result_json, 1, 200) FROM steps WHERE task_id = '<task_id>' ORDER BY seq"

# Long-term memory: the number of entries, and a preview
sqlite3 long_term.db "SELECT COUNT(*) FROM long_term"
sqlite3 long_term.db "SELECT id, substr(text, 1, 80), metadata FROM long_term LIMIT 5"
```

Replace `<task_id>` with an id from the task list, or with the `task_id` of
the JSON line.

## Task records

Every run saves one row in the `tasks` table of history:

| Column | Value |
|---|---|
| `task_id` | The run's `task_id`: 32 hex characters. |
| `query` | The task as given. |
| `status` | `running` during the run, then how it ended; see [Run results](run-results.md#statuses). `cancelled` appears only for interactive runs in the Python API. |
| `result` | The run's result, or its error when the result is empty. |

- The row is saved again when the run ends, whatever ended it. A direct run
  ended by SIGTERM or SIGHUP is saved as `crashed`, with the result
  `Cancelled by SIGTERM` or `Cancelled by SIGHUP`.
- An interactive run, one started in the Python API with `on_event` or
  `approval`, that is cancelled is saved as `cancelled`, with the reason as
  its result. Its interrupted step gets one row per tool call in `steps`
  (see [Stop a run](python-api.md#stop-a-run)). Other cancelled runs are
  saved as `crashed`.
- A row that still reads `running` belongs to a run in progress, or to a
  process that was killed (for example with SIGKILL) before it could save
  the row.
- Rows from earlier versions use only `running`, `completed` and `failed`.
  There, `failed` also covers runs that now end as `gate_stopped`,
  `budget_exhausted` or `stuck`, and a run that crashed kept `running`.

The `steps` table holds one row per tool call: `step_id`, `task_id`,
`thought` (the model's text), `tool_call_json`, `tool_result_json`,
`timestamp` and `seq` (the order within the task).

## Sessions

The session store of the Python API ([Sessions](sessions.md)) keeps
conversations in its own files, apart from `history.db`. No command uses it
yet; the files appear once a program opens a store.

```text
~/.cache/localmelo/gateway/sessions.db   the gateway's sessions
~/.cache/localmelo/cli/sessions.db       the sessions of every terminal melo
```

- **One writer each.** A file records whether the gateway or a terminal
  wrote it, and a store of the other kind refuses to open it, as it refuses
  any other SQLite file, without writing to it. The gateway's file has one
  owner, which holds a lock on `sessions.db.owner` while it runs; the
  terminals share theirs, with a lease on each running turn. Next to
  `sessions.db`, SQLite may also create `sessions.db-wal` and
  `sessions.db-shm`.
- **Private.** The directories are created with mode `0700` and the files
  with `0600`; the writer's own directory (`gateway/` or `cli/`) is set to
  `0700` when it is looser.
- **Durability.** The gateway's file uses `synchronous=FULL`: what it has
  committed survives an operating-system crash. The CLI file uses `NORMAL`,
  as `history.db` does: a crash of the process loses nothing committed, an
  operating-system crash may lose the last changes.
- **What they hold.** Each turn's text, its events (the model's text and
  thinking, the tool calls and their output), the images the tools returned,
  the final answer, and a window checkpoint per session: the working window
  at the end of the latest turn, with the tool calls and their results but
  not the tools' output, which the events keep.
- **Redaction.** The final answer is stored three times: in the turn's
  `run.finished` event, and as the turn's answer and result. Each copy, and
  the assistant text and thinking of the checkpoint, passes the minimal
  redaction of [events](events.md#assistantmessage) before it is stored.
- **Retention.** When a store opens, and every 24 hours in a gateway that
  keeps it open, what is older than 30 days is cleared: the thinking of
  `assistant.message` events, and a `<think>` block at the start of their
  text; the `<think>` block of the final answer's copies; window
  checkpoints, and in a newer checkpoint the thinking of each message older
  than 30 days; and withdrawn messages. The turns, their answers and their
  events stay until you delete the session.
- **A killed run.** A run killed with its process keeps the row `running` in
  `history.db`'s `tasks` table (see [Task records](#task-records)); its
  turn's status in `sessions.db` is the one to trust. The turn's `run_id` is
  the run's `task_id`, or `""` when the run never started.

## Reset memory

localmelo has no reset command. To delete memory:

1. Stop every localmelo process: direct runs, `melo --serve`, and the
   background service. Stop the service with `melo --daemon uninstall`; a
   plain `kill` is undone by launchd.
2. Remove the files:

   ```bash
   # Everything
   rm -rf ~/.cache/localmelo/memory

   # Only long-term memory
   rm ~/.cache/localmelo/memory/long_term.db*

   # Only history
   rm ~/.cache/localmelo/memory/history.db*

   # The web UI's memory
   rm -rf ~/.cache/localmelo/playground
   ```

3. Start localmelo again. It creates new files when it needs them.

If you set `LOCALMELO_MEMORY_DIR` or used `--memory-dir`, remove the files in
that directory instead.

> [!WARNING]
> Removing the files while localmelo runs can corrupt them. Stop localmelo
> first.

## Privacy

- The files are not encrypted. `history.db` holds your tasks, the model's
  text, every tool call with its arguments, and every tool result.
  `long_term.db` holds step summaries and reflections, with their embedding
  vectors.
- Images that tools return are stored only as references: the MIME type, the
  SHA-256 digest and the size, never the image itself.
- The files are as private as your home directory. To remove them, see
  [Reset memory](#reset-memory).
- The gateway's web UI always saves its sessions, in
  `~/.cache/localmelo/playground/`.

## How memory works

**Working memory** holds the conversation:

- It keeps the last 20 messages and drops the oldest first. A tool call is
  never separated from its results.
- A planning prompt may hold up to 100,000 characters. When it would hold
  more, the oldest turns are dropped first. The task always stays: when its
  own message has been dropped, localmelo puts it back in front. A task too
  long to fit ends the run as `failed`, with the error
  `Plan check failed: Prompt too large: <n> chars`.
- One turn of tool calls and results may use up to 50,000 characters. A
  longer result is cut and ends with `... [truncated: <n> more characters]`;
  a result left less than 200 characters is replaced by a short notice.
  History keeps the result as the tool returned it, up to 50,000 characters.

**Long-term memory** needs an embedding backend:

- After each tool call, a summary of the step is written to it: the model's
  text, the tool's name and the start of the tool's output.
- When a run ends, its latest reflection is written to it, with the task's
  id. This may take at most 10 seconds.
- On each step, the 5 entries closest to the task are recalled into the
  prompt.

**History** is a record. localmelo writes every task and step to it, but does
not read it back during a run.

**Reflections** are kept in working memory, one at a time: each new
reflection replaces the last one. See
[How localmelo works](how-it-works.md#reflection).
