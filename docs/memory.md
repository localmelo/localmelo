# Memory persistence

LocalMelo keeps four conceptual memory layers behind the `Hippo` coordinator:
working memory (short-lived LRU), long-term memory (retrieved at planning
time), history (raw record/debug layer), and `PersonalizedMemory` (selected
sleep-input samples). This page documents the persistence options users can
configure and inspect on disk. See [architecture.html](architecture.html) for
the conceptual diagram.

Sections:

- [Quick reference](#quick-reference)
- [Environment variables](#environment-variables)
- [File layout](#file-layout)
- [Retention semantics](#retention-semantics)
- [No-embedding mode](#no-embedding-mode)
- [Reset workflow](#reset-workflow)
- [Inspection workflow](#inspection-workflow)
- [Real-backend smoke commands](#real-backend-smoke-commands)
- [PersonalizedSample v1 schema](#personalizedsample-v1-schema)

## Quick reference

| Concern              | Knob / location                                                          |
|----------------------|--------------------------------------------------------------------------|
| Enable persistence   | `LOCALMELO_PERSIST_MEMORY=1`                                             |
| Storage directory    | `LOCALMELO_MEMORY_DIR` (default `~/.cache/localmelo/memory`)             |
| History DB           | `<dir>/history.db` (always when persistence is on)                       |
| Long-term DB         | `<dir>/long_term.db` (only if an embedding provider is configured)       |
| Reset                | stop the localmelo process → `rm` the chosen file(s) → restart           |
| Schema version guard | `PERSONALIZED_SAMPLE_SCHEMA_VERSION = "v1"` (frozen)                     |

## Environment variables

### `LOCALMELO_PERSIST_MEMORY`

Set to a non-empty value to switch memory from in-memory backends to
SQLite-backed persistence. Unset (or empty) means memory lives only in
process RAM and is lost on exit.

The flag is read in `localmelo/melo/agent/agent.py` with
`os.environ.get("LOCALMELO_PERSIST_MEMORY")` and tested for truthiness — so
**any non-empty string enables persistence**, including `0` and `false`.
Use `1` for clarity, and `unset LOCALMELO_PERSIST_MEMORY` (or
`export LOCALMELO_PERSIST_MEMORY=""`) to disable.

```bash
export LOCALMELO_PERSIST_MEMORY=1
melo "what is 2+2"
```

### `LOCALMELO_MEMORY_DIR`

Override the directory that holds the SQLite files. Default:
`~/.cache/localmelo/memory`. The directory is created with
`os.makedirs(..., exist_ok=True)` on first write, so any path you can write
to works.

```bash
export LOCALMELO_PERSIST_MEMORY=1
export LOCALMELO_MEMORY_DIR=/var/lib/localmelo/memory
melo --serve
```

## File layout

When persistence is enabled with an embedding provider configured:

```text
$LOCALMELO_MEMORY_DIR/
  history.db          # raw task/step record layer
  history.db-wal      # SQLite write-ahead log (auto, managed by SQLite)
  history.db-shm      # SQLite shared-memory (auto, managed by SQLite)
  long_term.db        # retrieved long-term memory (only if embedding present)
  long_term.db-wal
  long_term.db-shm
```

`-wal` and `-shm` are SQLite sidecars from `PRAGMA journal_mode=WAL` (set in
`melo/memory/_sqlite.py`). They are owned by SQLite — do not remove them
while a localmelo process is running.

In no-embedding mode the `long_term.db*` files are not created; only
`history.db*` exist.

## Retention semantics

LocalMelo deliberately separates "what happened" from "what to recall":

- **History (`history.db`)** — append-only record of every task and step.
  This is the raw debug / replay / audit layer. The agent and memory
  subsystems do **not** read history during normal memory recall or write
  placement.
- **Long-term (`long_term.db`)** — content extracted from agent runs and
  indexed for retrieval. This is the layer `Hippo.retrieve_context` reads
  at planning time, gated by the embedding provider.
- **PersonalizedMemory** — filtered sleep-input samples for the Track 4
  sleep pipeline. Not part of normal online recall. See
  [PersonalizedSample v1 schema](#personalizedsample-v1-schema).
- **Working memory** — short-lived per-session LRU cache. Lives in RAM,
  not persisted, lost on process exit.

In short: history is the raw event log, long-term is the queryable memory,
and `PersonalizedMemory` is selected sleep input — not a dump of history.

## No-embedding mode

When `embedding_backend = "none"` (or no embedding provider is wired up):

- History persistence still works — `history.db` is written normally.
- Long-term recall is silently a no-op: `retrieve_context` returns `[]`.
- Long-term store is silently a no-op: `memorize` writes only to working
  memory, not to `long_term.db`.
- `long_term.db` is not created on disk.

The agent loop still runs end-to-end; only retrieval-augmented recall is
disabled. See [quickstart.md → No-embedding mode](quickstart.md#no-embedding-mode)
for the matching config and smoke command.

## Reset workflow

LocalMelo does not ship a built-in "reset" command. To wipe persistent
memory, stop the process first and remove the files yourself:

1. Stop any running localmelo process (CLI, `melo --serve`, or the
   `--daemon` launchd service via `melo --daemon uninstall` / `kill`).
2. Remove the file(s) you want to clear. Wildcards cover the SQLite
   sidecars:

   ```bash
   # Default location — wipe everything
   rm -rf ~/.cache/localmelo/memory

   # Or just the long-term recall, keeping history for replay
   rm ~/.cache/localmelo/memory/long_term.db*

   # Or just history, keeping long-term recall
   rm ~/.cache/localmelo/memory/history.db*
   ```

3. Restart localmelo. The directory and SQLite files are recreated on the
   first write.

> Removing the `.db-wal` / `.db-shm` sidecars while the process is still
> running is unsafe and can corrupt the database. Always stop first.

## Inspection workflow

The SQLite files are plain database files. Open them read-only with
the `sqlite3` CLI to inspect state:

```bash
# List tables in history.db
sqlite3 ~/.cache/localmelo/memory/history.db '.tables'

# Recent tasks (rowid is monotonically increasing in insertion order)
sqlite3 ~/.cache/localmelo/memory/history.db \
  'SELECT task_id, status, substr(query, 1, 60) AS query
     FROM tasks ORDER BY rowid DESC LIMIT 10'

# Recent steps across all tasks
sqlite3 ~/.cache/localmelo/memory/history.db \
  'SELECT task_id, step_id, timestamp
     FROM steps ORDER BY timestamp DESC LIMIT 10'

# Count long-term entries
sqlite3 ~/.cache/localmelo/memory/long_term.db \
  'SELECT COUNT(*) FROM long_term'

# Preview long-term entries (vector column is a float32 BLOB — skip it)
sqlite3 ~/.cache/localmelo/memory/long_term.db \
  'SELECT id, substr(text, 1, 80), metadata FROM long_term LIMIT 5'
```

Treat these as read-only. Running `UPDATE` / `DELETE` while localmelo is
attached can leave the database in an inconsistent state (writes may be
buffered in WAL).

## Real-backend smoke commands

LocalMelo ships two real-backend smoke paths. The canonical runbook lives
in [`tests/smoke/README.md`](../tests/smoke/README.md); the summary below
points at the right command per question.

### Chat round-trip (pytest, marker-gated)

Marker-gated pytest cases exercise a single chat round-trip against a
local backend with `embedding_backend = "none"`. Skipped automatically
when the backend is not reachable, so collection is safe on any laptop.

```bash
# Both local backends — whichever is reachable runs, the other is skipped
python -m pytest -m smoke_backend tests/smoke/test_backend_smoke.py -q

# Ollama only — needs an Ollama server at http://127.0.0.1:11434
python -m pytest -m smoke_backend \
  tests/smoke/test_backend_smoke.py::test_smoke_ollama_chat_round_trip -q

# MLC-LLM only — needs an MLC serve at http://127.0.0.1:8400
python -m pytest -m smoke_backend \
  tests/smoke/test_backend_smoke.py::test_smoke_mlc_chat_round_trip -q

# Globally skip the backend smoke (useful in CI on a machine without backends)
python -m pytest -m 'not smoke_backend' -q
```

Backend URLs and model ids come from `tests/smoke/data/backends.json`.

### Long-term retrieval (CLI script)

`tests/smoke/core_loop_test.py` is the real-backend long-term-retrieval
evidence path. It seeds memories, runs scenario queries against a real
chat + embedding backend, persists into SQLite, and emits per-backend
evidence JSON plus a cross-backend markdown report. The filename does
**not** start with `test_`, so pytest never collects it — invoke it
explicitly:

```bash
# From localmelo/
python tests/smoke/core_loop_test.py --backends ollama
python tests/smoke/core_loop_test.py --backends mlc
SMOKE_ONLINE_API_KEY=sk-... python tests/smoke/core_loop_test.py --backends online
python tests/smoke/core_loop_test.py --backends all
```

The `online` backend is skipped (before any network probe) unless
`SMOKE_ONLINE_API_KEY` is exported, so `--backends all` stays
offline-safe by default. Per-backend evidence lands in
`tests/smoke/output/{ollama,mlc,online}_test.json`; cross-backend
comparison in `tests/smoke/output/compare_test.md`. See
[`tests/smoke/README.md`](../tests/smoke/README.md) for scenario
selection, env-var overrides, and online safety notes.

## PersonalizedSample v1 schema

`PersonalizedMemory` is a staging area for the Track 4 sleep pipeline. It
stores **selected** runtime samples (e.g. user corrections, task-success
signals), not raw history. The selection policy is still being designed,
but the on-disk record shape is frozen as v1 so downstream Track 4 code
can rely on it.

Source: `localmelo/melo/memory/personalized/__init__.py`.

| Field         | Type             | Default            | Purpose                                                                  |
|---------------|------------------|--------------------|--------------------------------------------------------------------------|
| `input_text`  | `str`            | — (required)       | The runtime input the sample was built from                              |
| `target_text` | `str`            | `""`               | Desired completion or corrected response                                 |
| `signal`      | `str`            | `""`               | Why the sample was selected (e.g. `"user_correction"`, `"task_success"`) |
| `metadata`    | `dict[str, Any]` | `{}`               | Provenance: `task_id`, attempt, tool, source                             |
| `timestamp`   | `float`          | `time.time()`      | Unix epoch at sample creation                                            |

Schema version constant: `PERSONALIZED_SAMPLE_SCHEMA_VERSION = "v1"`.

**Compatibility rule.** Any rename, reorder, type change, or field
add/remove is a **breaking change**. Bump the version constant in the same
commit so the Track 4 sleep pipeline can branch on schema version. The
v1 contract is guarded by `tests/memory/test_personalized_schema.py` —
that test will fail on any field-shape change, which is intentional.

The public import path is:

```python
from localmelo.melo.memory.personalized import (
    PERSONALIZED_SAMPLE_SCHEMA_VERSION,
    PersonalizedMemory,
    PersonalizedSample,
)
```
