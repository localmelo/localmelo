# Guide: Memory

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
  [Model calls](../reference/memory.md#model-calls). `LOCALMELO_THINKING_RETENTION_DAYS` sets
  another number of days, such as `7` or `0.5`, for `history.db` and the
  session store's `sessions.db` alike (see [Sessions](../reference/memory.md#sessions)). A value
  that is not a number above 0 stops `melo` before it runs anything; see
  [Usage errors](../reference/cli.md#usage-errors).
- Recalling long-term memory in a later process needs persistence, memory on
  and an earlier stored entry. In `legacy`, the persistent long-term store
  also needs an embedding backend. Explicit reasoning modes can persist
  text-only entries without one and find them with BM25. In explicit modes,
  recall must be enabled and include the `long` kind; `drop` and `replay`
  leave `auto` recall off by default.

## Turn long-term memory off

`--memory off`, or `AgentOptions(memory=False)` in Python, turns long-term
memory off for a run: nothing is recalled from it or written to it, and
reflections are not promoted. Working memory and history work as usual.
`embedding_backend = "none"` disables the embedding backend, not memory
itself; explicit reasoning modes can still store and recall text-only entries.
To set an embedding model, see [Set up an embedding model](embedding.md).

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
[melo-procedures](../reference/procedures.md#export-and-forget).

Replace `<task_id>` with an id from the task list, or with the `task_id` of
the JSON line.

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
  stores (the console's chats are in
  `~/.cache/localmelo/gateway/sessions.db`), or the legacy playground's
  memory in `~/.cache/localmelo/playground/` when enabled.
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

   # The console's session store (including chats)
   rm -f ~/.cache/localmelo/gateway/sessions.db*

   # The legacy playground's memory, if it was enabled
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
  text, every tool call with its arguments, and every tool result, each
  after the redaction of [Model calls](../reference/memory.md#model-calls). It also
  holds the model's reasoning for every call, redacted and kept 30 days by
  default; see [Model calls](../reference/memory.md#model-calls).
  `long_term.db` holds step summaries and reflections, redacted, with their
  embedding vectors. `procedures.db` holds each task's query, normalized, the folder
  it ran in, and its tool calls' labels: file names and program names,
  never their content or output. The query and the labels pass a minimal
  redaction: the key shapes, bearer tokens and one-time codes of
  [Model calls](../reference/memory.md#model-calls), then a URL's password
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
- The console stores its session rows and turns in
  `~/.cache/localmelo/gateway/sessions.db`. The legacy playground is disabled
  by default; when enabled with `melo --serve --enable-playground`, it saves
  its chats' history and procedural memory, and their long-term memory when
  an embedding model is selected, in `~/.cache/localmelo/playground/`.
