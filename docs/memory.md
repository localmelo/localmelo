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
| Keep the model's reasoning N days | `export LOCALMELO_THINKING_RETENTION_DAYS=N` (default 30) |
| See past runs | `sqlite3` queries; see [Inspect memory](#inspect-memory) |
| Delete memory | In Python, `await hippo.forget_all()` (one task: `forget(task_id)`); no command yet. See [Reset memory](#reset-memory) |

## What localmelo remembers

| Memory | What it holds | Where |
|---|---|---|
| Working memory | The conversation of a run, of the `melo ""` loop, of a gateway session or of a web workspace chat: the last 20 messages. | RAM only. |
| History | Every task with its status and result, and every step with its tool call and result. In `history.db`, also every reply of the model with its reasoning; the history in RAM keeps no model calls. | RAM, or `history.db`. |
| Long-term memory | Short summaries of steps, and the reflections of past runs. On each step, the 5 entries closest to the task are recalled. Needs an embedding backend. | RAM, or `long_term.db`. |
| Procedural memory | What the runs of each recurring task did: the classes and short labels of their tool calls, their step counts and times. Written at the end of every run; never put in a prompt. See [Procedural memory](procedures.md). | RAM, or `procedures.db`. |

By default everything stays in RAM and is lost when the process ends. In RAM,
long-term memory lasts as long as its agent: one direct run, one `melo ""`
loop, or one gateway session.

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
- History keeps the model's reasoning for 30 days; see
  [Model calls](#model-calls). `LOCALMELO_THINKING_RETENTION_DAYS` sets
  another number of days, such as `7` or `0.5`, for `history.db` and the
  session store's `sessions.db` alike (see [Sessions](#sessions)). A value
  that is not a number above 0 stops `melo` before it runs anything; see
  [Usage errors](cli.md#usage-errors).
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
  history.db      tasks, steps and model calls
  long_term.db    long-term memory, with an embedding backend and memory on
  procedures.db   procedural memory, written at the end of every run
```

- SQLite may also create `history.db-wal` and `history.db-shm` (and the same
  for `long_term.db` and `procedures.db`). They belong to the database;
  leave them in place.
- The directory is created when an agent starts: at the start of a direct
  run or of the `melo ""` loop, or when a gateway session is created.
- A new directory gets mode 0700, so only you can open the files in it, and
  a new `history.db`, `long_term.db` or `procedures.db`, with its `-wal` and
  `-shm`, gets mode 0600.
- The default directory, `~/.cache/localmelo/memory`, is also set to 0700
  when it exists, you own it, and it is open to group or others, as earlier
  versions left it. One INFO line in the log says so.
- A directory you chose (`LOCALMELO_MEMORY_DIR`, `--memory-dir`), and every
  existing file, keep their mode. To close them to other users:

  ```bash
  chmod 700 ~/melo-memory
  chmod 600 ~/melo-memory/history.db*
  chmod 600 ~/melo-memory/long_term.db*
  ```

- `procedures.db` holds [procedural memory](procedures.md). With
  persistence on, the first run that ends creates it, and every run adds
  its record at its end, `--memory off` included. With persistence off,
  procedural memory stays in RAM, and nothing is written.
- The gateway's web workspace keeps its own files in
  `~/.cache/localmelo/playground/`. Its `history.db` keeps the model's
  reasoning for the same number of days, and clears older reasoning when a
  new chat first uses it: each chat opens its own connection to the shared
  file. Its chats share a `procedures.db` there too.

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

# The model calls of one task, with the start of their reasoning
sqlite3 history.db "SELECT seq, kind, outcome, step, substr(thinking, 1, 80) FROM llm_calls WHERE task_id = '<task_id>' ORDER BY seq"

# Each step of one task, with the reasoning of the call that made it
sqlite3 history.db "SELECT s.seq, c.seq, substr(c.thinking, 1, 80), s.tool_call_json FROM llm_calls c, json_each(c.step_ids_json) j JOIN steps s ON s.step_id = j.value WHERE c.task_id = '<task_id>' ORDER BY s.seq"

# Long-term memory: the number of entries, and a preview
sqlite3 long_term.db "SELECT COUNT(*) FROM long_term"
sqlite3 long_term.db "SELECT id, substr(text, 1, 80), metadata FROM long_term LIMIT 5"

# Procedural memory: the routines, and how their latest runs went
sqlite3 procedures.db "SELECT routine_id, substr(query_text, 1, 60), place FROM routines"
sqlite3 procedures.db "SELECT routine_id, status, steps, level, model FROM episodes ORDER BY created_at DESC LIMIT 10"
```

`melo-procedures export` prints all of procedural memory as JSON lines; see
[melo-procedures](procedures.md#export-and-forget).

Replace `<task_id>` with an id from the task list, or with the `task_id` of
the JSON line.

## Task records

Every run saves one row in the `tasks` table of history:

| Column | Value |
|---|---|
| `task_id` | The run's `task_id`: 32 hex characters. |
| `query` | The task as given. |
| `status` | `running` during the run, then how it ended; see [Run results](run-results.md#statuses). `cancelled` appears only for interactive runs: in the Python API, and in the web workspace's history. |
| `result` | The run's result, or its error when the result is empty. |

- The row is saved again when the run ends, whatever ended it. A direct run
  ended by SIGTERM or SIGHUP is saved as `crashed`, with the result
  `Cancelled by SIGTERM` or `Cancelled by SIGHUP`.
- An interactive run that is cancelled is saved as `cancelled`, with the
  reason as its result: a run started in the Python API with `on_event` or
  `approval`, or a web workspace chat stopped with Stop or by its 600-second
  limit (its history is in `~/.cache/localmelo/playground/`). Its
  interrupted step gets one row per tool call in `steps` (see
  [Stop a run](python-api.md#stop-a-run)). Other cancelled runs are saved as
  `crashed`.
- A row that still reads `running` belongs to a run in progress, or to a
  process that was killed (for example with SIGKILL) before it could save
  the row.
- Rows from earlier versions use only `running`, `completed` and `failed`.
  There, `failed` also covers runs that now end as `gate_stopped`,
  `budget_exhausted` or `stuck`, and a run that crashed kept `running`.

The `steps` table holds one row per tool call: `step_id`, `task_id`,
`thought` (the model's text), `tool_call_json`, `tool_result_json`,
`timestamp` and `seq` (the order within the task).

## Model calls

With persistence on, history also keeps every reply of the model, with its
reasoning, in the `llm_calls` table of `history.db`: one row per reply, for
planning and reflection requests alike. The history in RAM keeps none.

| Column | Value |
|---|---|
| `task_id` | The run's `task_id`. |
| `seq` | The reply's number in the run, from 0, in the order the replies arrived. |
| `kind` | `plan` for a planning request, `reflect` for a reflection request. |
| `step`, `attempt` | The run's step and attempt when the reply arrived. A reflection has the last step of the attempt it reflects on. |
| `outcome` | For `plan`: `tools` when the reply has a tool call, else `answer` when it has text, else `empty`. `aux` for every other kind. It describes the reply, not what the run did with it. |
| `content`, `thinking` | The reply's text and its reasoning, redacted (below). |
| `reasoning_key` | Where the reasoning came from: `reasoning_content` or `reasoning` (an OpenAI-compatible server), `thinking` (Ollama), or `""` when the reply has none or its provider does not say. |
| `step_ids_json` | For a reply with tool calls, the `step_id`s of its `steps` rows as a JSON list, in call order; `NULL` for a reply that produced none. |
| `redactions` | The number of spans the redaction replaced. |
| `tool_calls_json` | The reply's tool calls as a JSON list of `{"tool_name", "arguments", "id"}`, or `NULL`. |
| `usage_json` | The token counts the server reported, as JSON, `cached_tokens` and `reasoning_tokens` included when an OpenAI-compatible server sends them; or `NULL`. |
| `duration_s` | How long the request took, without earlier failed attempts and the waits before retries. |
| `timestamp` | When the reply arrived, in seconds since the epoch. |

- The working-memory window keeps each reply's reasoning, including an
  empty reply's and a turn whose text a size cut cleared; `llm_calls`
  records it as received (R1's `thinking` and `reasoning_key`). In
  `legacy` the provider decides whether it is sent again: OpenAI-compatible
  no, Ollama native yes. `drop` strips it at send. `replay` sends it on the
  assistant turns in scope, under one request key (`reasoning_content`,
  `reasoning`, or Ollama's `thinking`), at most 40,000 characters per turn.
  Both explicit modes skip `[RECALL]` when they build the prompt, so an
  embedding provider's long-term memory is written but not read back;
  `legacy` reads it as before.
- A request that was retried gives one row, for the reply that came back.
  So the rows of a run number its `n_llm`, unless a row could not be
  written: a row that waits more than 5 seconds for another process's write
  lock on `history.db`, or fails otherwise, is left out with one warning on
  stderr (later failures of the run are logged at debug level), and the run
  goes on as if nothing happened. A missing row leaves a gap in `seq`.
- `step_ids_json` stays `NULL` for answers, empty replies, reflections, a
  reply that the plan check refused, and a planning reply whose run was cut
  during its tools (by `--timeout`, or a cancellation): such a reply gets no
  `steps` rows. A run cut while it wrote a step's rows can leave those rows
  unlinked; each belongs to the task's latest `plan` row before its
  `timestamp`. A cancelled interactive run in the Python API links the rows
  that its clean-up writes.
- **Redaction.** `content` and `thinking` pass a minimal redaction of
  secrets, the same as the Python API's events: well-known key shapes
  (`sk-` keys, GitHub tokens, AWS access key ids), the token after `Bearer`,
  and a code of 4 to 8 digits within 20 characters of `verification code`,
  `passcode`, `OTP` or `验证码`, on the same line, each become `[redacted]`.
  It is minimal: tool-call arguments are not redacted, in
  `tool_calls_json` or in `steps`, and a key glued to the text before it
  stays: an `sk-` key right after a letter, a digit, `_` or `-`, a GitHub
  token right after a letter, a digit or `_`, and an AWS access key id
  right after a letter or a digit. `tasks` and `steps` keep every text as
  it was.
- **Retention.** Reasoning older than 30 days, or than
  `LOCALMELO_THINKING_RETENTION_DAYS`, is cleared: `thinking` becomes `""`,
  and the reasoning that `content` holds goes too, by the row's
  `reasoning_key`. For `thinking` (Ollama) that is the `<think>` block at
  the start of `content`, which Ollama's provider adds. For `""` (a server
  without a reasoning parser) it is a `<think>` block at the start, an
  unclosed `<think>` at the start (the output was cut off while the model
  thought), or the text up to a `</think>` with no `<think>` before it, as
  chat templates that open the block in the prompt leave it. For
  `reasoning_content` and `reasoning`, `content` holds no reasoning and
  stays. The row and its other columns stay. Old reasoning is cleared when
  an agent first uses `history.db`: at the start of a direct run, or at a
  gateway session's first run.
- **Joining events.** In the Python API, a `plan` row and the
  [`assistant.message`](events.md#assistantmessage) event of its step have
  the same `task_id` (the event's `run_id`) and `step`. The `reflect` rows
  of an attempt, one or two, belong to its
  [`reflection`](events.md#reflection) event; the last by `seq` is the reply
  it read. The event's `seq` and the row's `seq` are different counters.
- **Export.** `melo-retrieve --export-run` turns a run's model calls into
  documents for retrieval experiments; see
  [Export a history database](retrieval.md#export-a-history-database).
- **Schema version.** `history.db` has `PRAGMA user_version` 1. A file from
  an earlier version (version 0) gets the empty table and version 1 when this
  version first opens it, with one INFO line in the log; its tasks and steps
  stay as they are. Earlier versions still read and write the tasks and steps of
  a version-1 file, and ignore `llm_calls`. A file with a newer version ends
  every run of this version as `crashed`, with the error
  `HistorySchemaError: <path> has schema version <n>; this localmelo reads version 1`.

## Sessions

The session store ([Sessions](sessions.md)) keeps conversations in its own
files, apart from `history.db`. Bare `melo` opens
`~/.cache/localmelo/cli/sessions.db` at its first message; the gateway does
not use a store yet.

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
- **Retention.** When a store opens, and whenever its program calls
  `apply_retention()` (a gateway that keeps the store open is to call it
  every 24 hours), what is older than 30 days, or than
  `LOCALMELO_THINKING_RETENTION_DAYS` as for `history.db`, is cleared: the
  thinking of `assistant.message` events, and the `<think>` block in their
  text; the `<think>` block of the final answer's copies; window
  checkpoints, and in a newer checkpoint the thinking and `<think>` block
  of each message older than that; and withdrawn messages. The turns, their
  answers and their events stay until you delete the session.
- **A killed run.** A run killed with its process keeps the row `running` in
  `history.db`'s `tasks` table (see [Task records](#task-records)); its
  turn's status in `sessions.db` is the one to trust. The turn's `run_id` is
  the run's `task_id`, or `""` when the run never started.

## Reset memory

The Python API can forget memory. There is no command yet; a
`melo --forget task|all` command comes with the UI work.

```python
import asyncio
from pathlib import Path

from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.long.sqlite import SqliteLongTerm


async def main() -> None:
    memory = Path.home() / ".cache" / "localmelo" / "memory"
    hippo = Hippo(
        history=SqliteHistory(memory / "history.db"),
        long=SqliteLongTerm(memory / "long_term.db"),
    )
    try:
        await hippo.forget_all()      # one task: await hippo.forget(task_id)
    finally:
        await hippo.aclose()


asyncio.run(main())
```

`agent.hippo` is the same facade over the stores an agent has opened, so a
running script can call `forget` there too; the script above opens both
files even when the agent has no embedding provider. `SqliteHistory` and
`SqliteLongTerm` have `forget(task_id)` and `forget_all()` themselves, as
do the in-memory `History` and `LongTerm`.

- **What `forget(task_id)` removes.** The task's rows from history's
  `tasks`, `steps` and `llm_calls` tables. In long-term memory, the entries
  whose metadata names the task: a reflection's metadata has `"task_id"`,
  a step summary's has only `"step_id"`, so a per-task forget removes the
  task's reflections from long-term memory and leaves its step summaries.
  `forget_all()` removes every long-term entry and every row of the three
  history tables.
- **What it does not touch.** Working memory (the current conversation),
  procedural memory (`melo-procedures forget` clears that), and the session
  store (`sessions.db`; the web workspace's chats are in
  `~/.cache/localmelo/playground/`).
- **How.** One `BEGIN IMMEDIATE` transaction under the store's write lock.
  The deletes overwrite what they free (`PRAGMA secure_delete=ON`), and a
  checkpoint then truncates the WAL, so the forgotten text is not left in
  the database file or its `-wal` file. If another connection keeps the WAL
  from being truncated, the checkpoint is skipped with a WARNING in the log
  and a later checkpoint clears it. A forget is safe while localmelo runs;
  removing the files is not.

To delete everything, including what the API does not touch, stop every
localmelo process and remove the files:

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

   # Only procedural memory
   rm ~/.cache/localmelo/memory/procedures.db*

   # The web workspace's memory
   rm -rf ~/.cache/localmelo/playground
   ```

3. Start localmelo again. It creates new files when it needs them.

To clear the model's stored reasoning, with localmelo stopped:

```bash
# The records of the calls, with all their reasoning
sqlite3 ~/.cache/localmelo/memory/history.db "DELETE FROM llm_calls"

# Only the thinking column, keeping the records of the calls
sqlite3 ~/.cache/localmelo/memory/history.db "UPDATE llm_calls SET thinking = ''"
```

The second leaves the reasoning that `content` holds: the `<think>` copy at
the start of an Ollama reply, which retention then no longer removes, and
the inline block of a reply from a server without a reasoning parser. To
remove all the stored reasoning, use the first.

If you set `LOCALMELO_MEMORY_DIR` or used `--memory-dir`, remove the files in
that directory instead.

> [!WARNING]
> Removing the files while localmelo runs can corrupt them. Stop localmelo
> first.

## Privacy

- The files are not encrypted. `history.db` holds your tasks, the model's
  text, every tool call with its arguments, and every tool result. It also
  holds the model's reasoning for every call, redacted and kept 30 days by
  default; see [Model calls](#model-calls).
  `long_term.db` holds step summaries and reflections, with their embedding
  vectors. `procedures.db` holds each task's query, normalized, the folder
  it ran in, and its tool calls' labels: file names and program names,
  never their content or output. The query and the labels pass a minimal
  redaction: the key shapes, bearer tokens and one-time codes of
  [Model calls](#model-calls), then a URL's password
  (`postgres://app:<redacted>@host`, the user name kept) and the value of
  `NAME=value` or `NAME: value` when NAME contains `key`, `token`,
  `secret`, `password`, `passwd` or `pwd` (`API_KEY=<redacted>`).
  Anything else stays as written, such as host names, paths and other
  values.
- Images that tools return are stored only as references: the MIME type, the
  SHA-256 digest and the size, never the image itself.
- The interactive UI keeps every turn, with its tool calls and output, in
  `~/.cache/localmelo/cli/sessions.db`, and your messages in
  `~/.cache/localmelo/cli_history` (both `0600`, plain text, not
  encrypted); `~/.cache/localmelo/logs/cli.log` records request URLs and
  errors.
- The files are as private as your home directory. To remove them, see
  [Reset memory](#reset-memory). The Python API's `forget` overwrites the
  cells it deletes and truncates the WAL, so the text does not stay in the
  database file's free pages or in `history.db-wal`/`long_term.db-wal`.
- The gateway's web workspace always saves the history and procedural
  memory of its chats, and their long-term memory when an embedding model
  is selected, in `~/.cache/localmelo/playground/`.

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

**History** is a record. localmelo writes every task and step to it, and,
with persistence on, every model call, but never reads it back.

**Procedural memory** learns the steps of recurring tasks:

- When a run ends, after its last event and before `melo` prints the
  result, the run's record is added: its task's routine, its step counts and
  the labels of its tool calls. On a file of 1,000 runs this takes a few
  milliseconds, under 50 ms at the 95th percentile, and never more than 2
  seconds;
  a write that fails, such as on a locked file, is skipped with an INFO
  line in the log. It is not part of the run's `duration_s`.
- A run after SIGTERM or SIGHUP, and a cancelled Python run, add nothing.
- Nothing reads it during a run, unless progress is asked for: by a Python
  caller ([`run.progress`](events.md#runprogress)), or by `melo` drawing
  its [progress line](cli.md#progress-line) on a terminal.
- To keep it off the disk, leave persistence off; to delete it, see
  [Reset memory](#reset-memory) or `melo-procedures forget`.

**Reflections** are kept in working memory, one at a time: each new
reflection replaces the last one. See
[How localmelo works](how-it-works.md#reflection).

**Memory notes and retrievers** are not used during a run yet.
`melo-retrieve` compares the `bm25`, `llm` and `embedding` retrievers
offline, over documents built from a run's model calls: notes on the
model's reasoning, code drafts, reflections and chunks of its thinking. See
[Offline retrieval](retrieval.md).
