# Run results

Every run ends with a status and a record of what it did. This page lists
the statuses, the fields of the record, and how failed model requests are
retried.

## Statuses

| Status | Exit code | Meaning |
|---|---|---|
| `completed` | 0 | The model gave a final answer, or a reflection decided that the task was done. |
| `failed` | 2 | A check refused the run, or the model gave two empty replies in a row. |
| `gate_stopped` | 2 | After an attempt, the reflection decided not to continue. |
| `budget_exhausted` | 2 | The run used all its steps or attempts without a final answer. |
| `context_budget_exceeded` | 2 | The pinned request cannot fit the configured input budget, or a smaller retry still overflowed. |
| `stuck` | 2 | The same tool calls, or the same errors, repeated three steps in a row. |
| `provider_error` | 2 | A request to the model server failed. |
| `setup_error` | 2 | The tool providers, such as MCP servers, could not be set up, or a worker inbox could not be read safely. |
| `timeout` | 2 | The run was still going when its time limit (`--timeout`) expired. |
| `crashed` | 1 | An unexpected error, Ctrl+C, SIGTERM or SIGHUP ended the run. |
| `cancelled` | none | An interactive run was stopped: a run in the Python API with `on_event` or `approval`, or a web workspace chat stopped with Stop or by its 600-second limit. Direct mode, the `melo ""` loop and `POST /v1/agent/run` never report it. |

Details:

- **`completed`.** When a reflection decides the task is done, the answer is
  its best effort or, if that is blank, the latest text the model wrote in
  the run.
- **`failed`.** The error says which check refused the run, for example
  `Plan check failed: Prompt too large: <n> chars` or
  `Response check failed: …`. A reply with neither a tool call nor text is
  an empty reply: localmelo asks the model to continue and plans again, and
  the step counts. Two empty replies in a row end the run with the error
  `empty replies`.
- **`gate_stopped`.** The reflection chose `stop` or `decompose`, or its
  decision to continue did not pass the continuation gate (see
  [Reflection](../concepts/how-it-works.md#reflection)). `result` holds the reflection's
  best effort, if any, and `error` its reason, or
  `Stopped after attempt <n> (stop reason: <code>)` when it gave none. The
  `reflection` event always carries the stop reason and the decision's flags
  (see [Events](events.md#reflection)). With `--reflection-gate utility`, an unreadable
  reflection stops the run too, and so does one over the size limit unless
  it chose `done`.
- **Unreadable reflections.** A reflection reply is readable when it holds a
  JSON object whose `recommended_action` is `done`, `continue`, `stop` or
  `decompose`; other JSON, such as a tool call written out as text, does not
  count. An unreadable reply is asked for once more, without tools: the
  attempt's turns are written out as text, and a closing line asks for the
  JSON object only. Both requests count in `n_aux`. When the second reply is
  unreadable too, nothing of it is kept and the next attempt starts. If none
  can start, the run ends as `budget_exhausted`, with the latest text the
  model wrote in the run as `result`. With `--reflection-gate utility`, any
  JSON object counts as readable, and a second reply without one ends the
  run as `gate_stopped`: the error is `Could not parse reflection response`
  when the reply could not be read, or
  `Reflection response was not a JSON object` when it was valid JSON that
  is not an object. With no attempt left the run ends as
  `budget_exhausted` instead.
- **`budget_exhausted`.** The error is `Max steps reached` or
  `Max attempts reached`. With `--attempt-policy reflect`, a last reflection
  can still choose `done`; otherwise its best effort becomes `result`.
- **`stuck`.** Only with `--attempt-policy none`; with `reflect`, a
  reflection decides what happens next. The error is
  `Stuck: the same tool calls or errors repeated`. Repeated calls that return
  new results, such as polling a job, are not stuck.
- **`provider_error`.** See [Provider errors and retries](#provider-errors-and-retries).
- **`setup_error`.** The error names the problem, for example
  `MCP server 'notes' exited (code 1)`. An agent does not retry an MCP or tool
  provider setup failure: its later runs end as `setup_error` too. A worker
  inbox that is inside the workspace, not a regular file, oversized, replaced,
  truncated or invalid also ends `setup_error` with the plain message; that
  failure is about the file, so a later run with a readable inbox proceeds.
- **`timeout`.** The error is `Timed out after <n> s`. A model request in
  flight is abandoned, and a `shell_exec` command or `python_exec` child is
  killed with every process it started.
- **`crashed`.** The error is `<ExceptionType>: <message>`, and the
  traceback is logged. Ctrl+C ends a run as `crashed` with the error
  `CancelledError`, and SIGTERM or SIGHUP with `Cancelled by SIGTERM` or
  `Cancelled by SIGHUP`. See [Signals](cli.md#signals).
- **`cancelled`.** Only for an interactive run whose task was cancelled: one
  started with `on_event` or `approval` in the Python API (see
  [Stop a run](../guides/python.md#stop-a-run)), or a web workspace chat stopped
  with Stop or by its 600-second limit. The gateway runs each web chat with
  an event sink (see [Web workspace](../guides/web-console.md#web-workspace)). `result` is
  empty, and `error` is the reason: the reason given to `Agent.begin_exit`
  when the process is exiting, else the message of `task.cancel(msg)`, such
  as `Timed out after 600 s` for a web chat, else `Cancelled by the caller`.
  A cancelled run without `on_event` or `approval` ends as `crashed`, so
  direct mode, the `melo ""` loop and `POST /v1/agent/run` never report
  `cancelled`.

[How localmelo works](../concepts/how-it-works.md) explains attempts, reflection and the
continuation gate.

### A session turn's cause

A turn of a [session](sessions.md) ends with one of these statuses too, and
with a `cause` that says what ended it:

| `cause` | What ended the turn | Status |
|---|---|---|
| `run` | The run itself, whatever its status. | any |
| `user_stop` | A person stopped it. | `cancelled` |
| `closed` | The session's runner was closed. | `cancelled` |
| `shutdown` | The process was asked to stop. | `cancelled` |
| `crash` | The process died; recovery recorded the end. | `crashed`, or `cancelled` when a stored `run.finished` said so |
| `takeover` | Another terminal took over a turn whose lease, held on another host, had expired. | `cancelled` |
| `park` | Reserved for the scheduler. | `cancelled` |

"Interrupted" is a label that a surface shows for the causes `shutdown` and
`crash`, never a status: a turn's status is always one of the statuses
above. `crashed` with the cause `run` is a crash inside the run, a bug;
`crashed` with `crash` means that the process stopped.

## Fields

`--output-format json` prints these fields in this order, followed by
`localmelo_version`. The Python API returns them as a `RunResult`.

| Field | Meaning |
|---|---|
| `task_id` | The id of the run's task record in history: 32 hex characters. |
| `status` | How the run ended; see [Statuses](#statuses). |
| `result` | The final answer or a best effort. It may be empty. |
| `error` | Why the run did not complete. Empty when it completed. |
| `steps` | Planning steps made, at most `max_steps`. |
| `attempts` | Attempts started: 1 when the first attempt answered. |
| `n_llm` | Model requests that returned a response. |
| `n_main` | Of those, requests that offered tools, other than reflections. |
| `n_aux` | Of those, the rest: reflections (also under `--fixed-prefix`, where they offer the plan's tools to share its prefix), recall's retrieve requests, and planning requests in a step that offered no tool. |
| `n_tool` | Tool calls handed to the executor, including refused calls. |
| `llm_errors` | Model requests that failed, counting each failed attempt of a retried request. `n_llm + llm_errors` is the number of requests sent. |
| `usage` | Token counts of all responses, summed by name, such as `prompt_tokens`, `completion_tokens` and `total_tokens`. Outside `legacy`, it also has `reasoning_chars` (the reasoning received, after an inline `<think>` block is split out) and `cached_tokens` or `reasoning_tokens` when the server reported them. |
| `duration_s` | Wall-clock time of the run, in seconds. |
| `config` | The options the run used (below). |
| `prefix_cache` | With fixed prefix enabled: version, observed planning schema changes, and optional server-reported cached tokens summed across planning and reflection responses; schema changes are expected invalidations, not measured cache misses. |
| `context_budget` | When configured, a list of per-attempt planning/reflection accounting records; absent when disabled. See [context accounting](configuration.md#context-accounting). |
| `memory_stats` | Outside `legacy`, the replay counters and the note counters; absent from a `legacy` line. |

`config` holds:

| Key | Meaning |
|---|---|
| `max_steps`, `attempt_policy`, `steps_per_attempt`, `max_attempts`, `reflection_gate`, `tool_exposure`, `memory`, `timeout_s` | The run options. `timeout_s` is `null` when no time limit is set. |
| `builtins` | The registered built-in tools, sorted by name. |
| `memory_dir` | The directory of the SQLite files, or `null` when memory stays in RAM. |
| `tools` | Every registered tool, built-in and MCP, sorted by name. |
| `window_eviction` | `chunk` or `boundary` when the run used one; absent for the default `slide`. |
| `reflect_tool_guard` | `true` when the run used `--reflect-tool-guard on`; absent otherwise. |
| `reasoning`, `replay_turns`, `note_writer` | Outside `legacy`, the effective reasoning mode, its turn cap (`null` for every turn in scope) and who writes a reply's note in the compress modes. They are absent from a `legacy` line. |
| `retriever`, `retriever_effective`, `recall_at` | Outside `legacy`, the requested retriever (`auto` when the default), the enabled-trigger map (`{"start": "llm", "attempt": "llm", "step": "bm25"}`, sorted) and the sorted effective trigger list. |
| `recall_k`, `recall_kinds`, `recall_placement`, `recall_min_score` | Outside `legacy`, the task-local hit cap, the sorted enabled kinds, the step placement and the embedding/long score floor. |
| `cross_task_ratio`, `cross_task_cap`, `cross_task_cold` | Outside `legacy`, the digest-recall gate: the same-backend multiple a cross-task digest must reach, the per-recall cap (0–2) and the cold-start allowance (0 or 1). |
| `admit_on_outcome`, `keep_decision_text` | Outside `legacy`, the two R6-lite switches: admit only the final digest with its terminal status at the run's end, and keep a decision's text beside its hash. Both are `false` by default and absent from a `legacy` line. |

`memory_stats` holds, in order:

| Key | Meaning |
|---|---|
| `replayed_chars` | Characters of reasoning the planning and reflection requests carried. |
| `replay_skipped` | Assistant turns whose reasoning was over the 40,000-character per-turn cap and was left out, summed over the requests sent. |
| `replay_stripped` | Assistant turns whose reasoning the fit-to-cap eviction cleared, summed over the requests sent. |
| `notes`, `notes_model`, `notes_heuristic` | Notes stored in the compress modes, and how many the model and the heuristic wrote. `notes` counts reflections too. |
| `note_chars` | Characters of the checked note texts this run stored (a restatement that grows an earlier note is not counted again). |
| `drafts`, `drafts_written` | Drafts stored, and how many a write covered. |
| `candidates` | Texts offered to the candidate boundary: each document (a note, a draft, a digest, a reflection, a user text, a recall block) and each candidate item that passes it again. |
| `remember_model`, `remember_heuristic` | Notes whose `remember` flag the model or the heuristic set. |
| `candidates_refused` | Texts the boundary or the note store refused, counted like `candidates`. |
| `redactions` | Secret spans the boundary replaced. |
| `rehearsed` | `{"restated": n, "acted": n, "cited": n}`: how often each rehearsal signal fired. |
| `digests` | Attempt digests stored. |
| `recalls` | Recall blocks first sent in this run. |
| `recalls_by_trigger` | The same count by `start`, `attempt`, `step` and `tool`. |
| `recall_hits` | Hits those blocks carried. |
| `recall_same_task`, `recall_cross_task` | Delivered long-store digests from this task and from other tasks; unscoped long entries count in neither. |
| `recall_task_notes` | Delivered task-local note-store hits (notes, drafts, thoughts and in-task digests). |
| `recall_chars` | Characters of the checked blocks. |
| `recall_ms` | Wall time of the recalls, summed. |
| `recall_fallbacks` | Recalls a retriever fallback answered, and `retrieve_errors` the retrieval failures behind them. |
| `recall_skipped` | Recalls that found nothing that fit a budget. |
| `recall_refused` | Blocks the candidate boundary refused. |
| `recall_empty` | Triggered recalls that had no candidate or no hit, so no block was built. |
| `aux_retrieve_calls` | Context-free retrieval requests sent (the `llm` retriever), already counted in `n_aux`; `retrieve_prompt_tokens` and `retrieve_completion_tokens` are their tokens. |
| `thought_chunks` | Thought chunks stored (at most 6 per task). |
| `recall_log` | One compact record per block first sent, without item text: trigger, placement, retriever, query, k, candidates, hits (ids, aliases, kinds, scores, origin task ids, chars, omitted), same/cross/task-note counts, chars, ms, fallback. |
| `admitted` | `{"rehearsal": n, "user_request": n, "user_statement": n, "outcome": n, "sleep": n}`: decisions written per signal. |
| `written` | `{"semantic": n, "episodic": n, "profile": n, "reflection": n}`: entries written per kind (`IGNORE` writes no entry). |
| `ignored` | Decisions that wrote no entry because the policy chose `IGNORE`. A boundary refusal counts in `candidates_refused`, not here. |

The replay counters are `0` in `drop`; the note, recall and admission
counters are `0` in every mode that does not use them, so the shape is the
same across the explicit arms. A `legacy` line has no `memory_stats`, no `reasoning_chars`
and no `cached_tokens`/`reasoning_tokens` in `usage`, and no
`reasoning`/`replay_turns`/`note_writer` or `retriever`/`recall_*` in
`config`, so it is byte-identical to the line before these fields.

The chat request timeout is a setting of the model connection, not of the
run, so it is not part of `config`.

## Provider errors and retries

When a request to the model server fails, localmelo retries it only if a
second try may succeed:

| Failure | Retried |
|---|---|
| A timeout, a refused connection or another network error | Yes |
| HTTP 429, 502, 503 or 504 | Yes |
| Any other HTTP error status, such as 400, 404 or 500 | No |
| A response that is not JSON or lacks the expected fields | No |

- A chat request is retried at most twice: after 1 second, then after 4 more
  seconds. If the last attempt fails too, the run ends as `provider_error`.
- Embedding requests are not retried.
- The error names the request and, when the server answered, gives the HTTP
  status and the first 500 characters of the response body, for example
  `POST /chat/completions failed with HTTP 500: <html> …`.
- Each attempt waits the full [chat request timeout](configuration.md#chat-request-timeout).
  A server that sends nothing therefore ends the run after about
  3 × the timeout + 5 seconds: about 365 seconds with the default of
  120 seconds.
- Every chat request carries its attempt number (1, 2 or 3) in the
  `X-Localmelo-Attempt` HTTP header, so a proxy's request log can be matched
  with `n_llm` and `llm_errors`. `X-Localmelo-Call-Kind` says `plan` for a
  planning request and `reflect` for a reflection request.

## Where results appear

- **Command line:** the result and the status line, or the JSON line. See
  [Output](cli.md#output).
- **Gateway:** `POST /v1/agent/run` answers HTTP 200 with `result`: the
  answer or best effort, or `[<status>] <error>` when that is empty. The
  status is not reported otherwise. See [HTTP API](http-v1.md#http-api).
- **Python:** `Agent.run_task()` returns a `RunResult`, and the
  `run.finished` event carries it too. See
  [Python API](python-api.md#results) and [Events](events.md#runfinished).
- **History:** each run's task record is saved with its status. See
  [Task records](memory.md#task-records).
