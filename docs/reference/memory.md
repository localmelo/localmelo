# Memory

This page explains what localmelo remembers, how to keep memory between runs,
and how to inspect, reset and protect it.

The maintainer's memory design, parts of which are not built yet, is on the
website's [Memory design](https://localmelo.github.io/localmelo/en/memory-design.html)
page.

## Quick reference

| To | Do this |
|---|---|
| Keep each reply as a short note with default recall | `melo --reasoning compress "<task>"` |
| Use short notes and BM25 recall throughout an interactive chat | `melo --reasoning compress --retriever bm25 --note-writer heuristic` |
| Keep memory between runs | `export LOCALMELO_PERSIST_MEMORY=1` |
| Choose the directory | `export LOCALMELO_MEMORY_DIR=…` (default `~/.cache/localmelo/memory`) |
| Keep the memory of one run in a directory | `melo --memory-dir ~/melo-memory "<task>"` |
| Turn long-term memory off for one run | `melo --memory off "<task>"` |
| Keep the model's reasoning N days | `export LOCALMELO_THINKING_RETENTION_DAYS=N` (default 30) |
| See past runs | `sqlite3` queries; see [Inspect memory](../guides/memory.md#inspect-memory) |
| Delete memory | In Python, `await hippo.forget_all()` (one task: `forget(task_id)`); no command yet. See [Reset memory](../guides/memory.md#reset-memory) |

## Long-term memory

With `--reasoning compress` (or `compress+replay`) and memory on,
`long_term.db` is a **typed** store: every entry has a kind, a source, a
status, the signal that admitted it and its provenance, beside the text and
the vector the earlier store already had. With persistence on, the file
exists in those modes even without an embedding provider; a text-only entry
stores an empty vector and is found by the in-process BM25 index. In
`legacy` the file still exists only with an embedding provider.

| Kind | What it holds | Source |
|---|---|---|
| `semantic` | A fact about a named third party (person, place, organization, event), usually an attribute such as an address or a date. | `user`, `reasoning_note` |
| `episodic` | What happened in this task: a note's ordinary item, every digest. | `reasoning_note`, `digest` |
| `profile` | A fact the user states about themselves (first person: I/my/we/our; 我/我的/我们; `call me`/`叫我`). Only a user message gives it. | `user` |
| `reflection` | A reflection: the attempt summary, what failed, the evidence, what to avoid, the next directions and the open questions. | `reflection` |

`source` is one of `reasoning_note`, `digest`, `reflection`, `user`, `sleep`
or `step_summary` (a legacy row). `status` is `active`, `superseded` or
`conflicted`; v1 writes `active` for every new entry and marks the replaced
entry `superseded`; `conflicted` is reserved for the sleep pipeline.
`admitted_by` is the signal that wrote the entry, or empty for a legacy row.
`provenance` carries the task and session, the step and attempt, the run's
text, the paths and commands the turn touched, the record ids the item came
from (`source_ids`), and the candidate id and item index. Every one of those
texts is redacted and bounded before the write. The session id is `""` in direct mode, which does
not thread one; the gateway does not thread its own id into the run yet. A
search returns the legacy metadata merged with these columns, so an older
reader still sees the same text and metadata.

The source of a fact is the observation that established it, which may be
earlier in the task than the note that states it. The candidate records are
this reply's tool results, the results still in the working-memory window,
and the rest of the same task's results. Only the task's last 200 records are
kept, and another task's records are never used. A record matches when it
shares a strong key or at least two distinctive non-stopwords of four or more
characters. The uniquely best score wins; equal top scores cite nothing.
Each matching typed key adds to the score, and conflicting dates, years when
both are present, currencies or phone numbers reject a record. Amounts retain
their unit or currency. A strong key is a date, a time with a colon or am/pm,
an amount with a currency or unit (or an amount field), a phone number, an
identifier of three or more characters containing a digit, a number of three
or more digits, or a capitalised name that does not start its sentence.
Dates include ISO, `April 9`, `9 April` and bare slash dates such as `4/9`.
Ambiguous slash dates use US month-first order; unambiguous `13/4` is
day-first. Thus `offsite 4/9` matches `2026-04-09`. A slash pair is not a
date when followed by a quantity or count word, or preceded by a number and
space (a mixed number such as `2 1/2`); `24/7`, invalid dates and lowercase
`may`/`march` are not dates either. These are heuristic rules, so other
fraction formats can still be ambiguous. Bare one- or two-digit numbers and
ordinals are not keys. Fillers such
as `done`, `ok`, `good`, `thanks`, `好的`, `完成` and `可以` are stopwords,
so an item that says only "Done." takes no source. When the note has no record
text at all, the item keeps the note's own `source_ids`; when it has records
and none matches, the item gets none, including after admission.
A digest's ids are the union of its notes' ids and the records behind their
items. A completion or self-work claim with no fact beyond its frame is
`IGNORE`d with the reason `meta claim about the agent's own work`, whether it
comes from a note or from a digest. A `task:` line, a file, path or function
name makes the text content instead. A final digest admitted by `outcome`
for a completed run is exempt from this filter entirely.

Nothing is written without a signal:

| Signal | When | What it admits |
|---|---|---|
| `rehearsal` | A restatement, a successful call that acted on the note's path or command, or a reflection's `used_notes`. | That candidate, at once. A second admission of the unchanged candidate writes nothing. |
| `user_request` | The user's message asks to remember (`remember that …`, `记住…`). | The fact the request carries after its marker, this run's `remember` notes and its final digest at the run's end; never the final reflection. The fact is classified by `classify_kind`, so it is `semantic` unless it is first-person (`profile`), and the meta-noun and question rules below apply to it. |
| `user_statement` | A declarative sentence of the user's message states a stable fact without a remember request, or corrects one and carries the corrected value. A question (`?`/`？`) never counts, and `my … is`/`我的…是` refuses a meta noun (task, question, request, goal, plan, problem, issue, job, point; 任务, 问题, 请求, 目标, 计划). | That sentence, at once. |
| `outcome` | The end of the run, only with `--admit-on-outcome on`. | The final attempt digest, with the terminal status in its provenance; never the final reflection. For a completed run the digest is exempt from the meta-claim filter. |
| `sleep` | Reserved for the sleep pipeline's probe gate. | Nothing in v1. |

Only the user's own text gives a user signal: a tool result, a note, a recall
block or an assistant message never does, and a note's first-person item is
never a profile entry. The candidate boundary runs again with target `long`
before the write: a refused item writes no entry and no decision. Every text
written, including the provenance's task text, paths and commands, is redacted
by `redact_secrets` and bounded first.

The policy compares the item with the active entries of its kind and picks
one action:

| Action | Rule | Write |
|---|---|---|
| `IGNORE` | A short fact (under 20 normalised characters), a fact that names only this task's files, a completion or self-work claim with no fact beyond its frame, or a duplicate (normalised equality, text overlap >= 0.9, or cosine >= 0.95). | No entry; a duplicate target's `uses` grows. |

An episodic item's subject is its candidate plus its item index, so two
different items of one note are two entries; a restatement re-proposes the
same candidate and item order, so it stays an `UPDATE`.
| `UPDATE` | Same subject and a compatible value, or a change marker (`moved`, `now`, `搬到` and so on). | A new entry that supersedes the old one; the old one becomes `superseded`. |
| `CONFLICT` | Same subject, values that contradict, and no change marker. | A new active entry linked to the old one; both stay recallable. |
| `RELATED` | A different subject with text overlap >= 0.35 or cosine >= 0.75. | A new active entry linked to the old one. |
| `NEW` | Anything else. | A new active entry. |

**The decision log.** `long_term.db` also has a `decisions` table: one row per
action, with the candidate id, the action, the kind, the signal, the entry and
target ids, a one-line reason, the item's `source_ids`, the SHA-256 of the
checked text and the task id. The text itself is stored only with
`--keep-decision-text` (an evaluation switch, refused in the interactive UI);
the hash lets a later join without keeping the text. `forget(task_id)` removes
a task's entries and decisions; `forget_all()` clears both tables. Reset the
file by deleting `long_term.db` with localmelo stopped. There is no
age-based retention in this version; the sleep pipeline's policy comes later.

**The migration.** A `long_term.db` from the earlier store is migrated in place
when it is opened: `user_version` becomes 1, the typed columns and the
`decisions` table are added with `ALTER TABLE` (no table rebuild), a legacy
reflection becomes `kind='reflection'`/`source='reflection'`, every other row
`episodic`/`step_summary`, and its text, vector and metadata bytes stay as they
were. An older localmelo can still open and search the file. A file from a
newer localmelo raises one WARNING and the run continues without long-term
memory. The read-only exports never migrate a file; see
[Offline retrieval](retrieval.md#the-typed-long-term-store).

At `start`, the long leg reads the active entries of other tasks of every kind:
`--recall-min-score` applies to a cosine score, while the BM25 first-stage
pool has no absolute floor. A doc is `long:<entry id>`, not a hash of its text.
`memory=False` removes the store, the writes and the leg.

## Files on disk

```text
~/.cache/localmelo/memory/
  history.db      tasks, steps and model calls
  long_term.db    long-term memory, when memory is on and either an embedding
                  backend is configured or the reasoning mode is not legacy
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

- `procedures.db` holds [procedural memory](../concepts/procedures.md). With
  persistence on, the first run that ends creates it, and every run adds
  its record at its end, `--memory off` included. With persistence off,
  procedural memory stays in RAM, and nothing is written.
- The gateway's web workspace keeps its own files in
  `~/.cache/localmelo/playground/`. Its `history.db` keeps the model's
  reasoning for the same number of days, and clears older reasoning when a
  new chat first uses it: each chat opens its own connection to the shared
  file. Its chats share a `procedures.db` there too.

## Task records

Every run saves one row in the `tasks` table of history:

| Column | Value |
|---|---|
| `task_id` | The run's `task_id`: 32 hex characters. |
| `query` | The task as given, with the secrets of [Redaction](#model-calls) replaced. |
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
  [Stop a run](../guides/python.md#stop-a-run)). Other cancelled runs are saved as
  `crashed`.
- A row that still reads `running` belongs to a run in progress, or to a
  process that was killed (for example with SIGKILL) before it could save
  the row.
- Rows from earlier versions use only `running`, `completed` and `failed`.
  There, `failed` also covers runs that now end as `gate_stopped`,
  `budget_exhausted` or `stuck`, and a run that crashed kept `running`.

The `steps` table holds one row per tool call: `step_id`, `task_id`,
`thought` (the model's text), `tool_call_json`, `tool_result_json`,
`timestamp` and `seq` (the order within the task). The `thought`, every
string in the call's arguments and the result's output and error are stored
with the secrets of [Redaction](#model-calls) replaced; the run itself and
the model keep the text.

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
  assistant turns in scope (by default the current task's; a provider that
  requires replay gets every turn in the window), under one request key
  (`reasoning_content`, `reasoning`, or Ollama's `thinking`), at most
  40,000 characters per turn (no cap on a provider that requires replay).
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
- **Redaction.** Every text that memory stores passes a minimal redaction
  of secrets, the same as the Python API's events: well-known key shapes
  (`sk-` keys, GitHub tokens, AWS access key ids), the token after `Bearer`,
  and a code of 4 to 8 digits within 20 characters of `verification code`,
  `passcode`, `OTP` or `验证码`, on the same line, each become `[redacted]`.
  It covers `tasks.query`, `tasks.result`, `steps.thought`, every string
  in `steps.tool_call_json`'s arguments, `steps.tool_result_json`'s output
  and error, `llm_calls.content` and `llm_calls.thinking`, and the text
  long-term memory keeps (each step's summary and the promoted
  reflections). What the model reads is not redacted: working memory, the
  requests a run sends and a run's `result` keep the original text, so
  only the stored copies change. The run's events redact their own fields:
  `assistant.message`'s `content` and `thinking`, and `reflection`'s
  `rationale`, `summary` and `best_effort_result`; see
  [Events](events.md#assistantmessage).
  It is minimal: other secret shapes stay, and so does a key glued to the
  text before it: an `sk-` key right after a letter, a digit, `_` or `-`, a
  GitHub token right after a letter, a digit or `_`, and an AWS access key
  id right after a letter or a digit.
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
  [Export a history database](../guides/offline-retrieval.md#export-a-history-database).
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
  not the tools' output, which the events keep. Each turn's `note` column
  holds the final attempt digest of a compress-mode turn (empty in the
  other modes).
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
