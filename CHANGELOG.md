# Changelog

All notable changes to localmelo are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

localmelo has no release yet. The v0.1.0 milestone is due 2026-12-20.

## [Unreleased]

### Security

- Gateway authentication is in progress: loopback-only binding, a private bearer
  token for API clients, and a one-use terminal sign-in link for browser sessions.
  Existing unauthenticated API clients must migrate; see the gateway guide.
  Same-user processes and localhost services on other ports remain outside the
  isolation guarantee. ([#136])

- Gateway security-1b is in progress: disable the legacy playground by default;
  `melo --serve --enable-playground` opts in behind authentication and uses
  saved local backend/model settings. Authenticated settings and model probes
  accept typed HTTP/HTTPS endpoints without following redirects. Explicit
  close/delete refuses running or queued starts with 409 rather than cancelling
  them ([#141]).

### Fixed

- Correct the API v2 response formats and tools example, and update the bilingual
  quickstart to reflect the shipped console settings and approvals ([#122], [#124]).
  Clarify task validation and read-only worker details ([#119], [#123], [#130], [#133]).
- Clarify that explicit reasoning modes can use text-only long-term memory
  without an embedding backend, in the embedding guide and Chinese architecture
  boundary ([#132]).

- Attempt digests skip task headings and report verification only for passing
  checks, rather than treating trivial Python imports as tests ([#128]).

### Added

- Scheduler component v4 adds dormant process capture state, identity checks
  and a gated subprocess primitive; worker behavior and Resume are unchanged. [#NN]

- Startup recovery classifies task rows without replay, retains pending pause
  and reconciles execution-linked terminal results. Recovered tasks with
  uncertain history remain fenced until controlled recovery is wired; Stop is
  available. Ordinary task creation does not yet establish first-start proof. [#NN]

- Scheduler component v3 adds dormant recovery provenance, guarded durable
  intents and receipts, and process identity records. Interrupted recovery
  retains pending pause intent; explicit legacy adoption records orphan-risk
  acceptance. Process capture completeness and controlled Resume are deferred. [#147]
- Experimental opt-in provider streaming and OpenAI-compatible SSE assembly;
  display filtering buffers complete replies, and no CLI or agent uses it yet ([#145]).
- Direct workers can read a brief file and emit a versioned JSON result or an
  NDJSON event stream; inbox steering remains future work. ([#143])

- Opt-in fixed system prefix with a local-date snapshot and schema-change telemetry;
  dynamic recall and reflection remain labelled data ([#142]).
- Opt-in bounded, pattern-redacted captured tool-result history, separate from the
  model preview, with capture and acquisition omission metadata ([#144]).
- Heuristic notes retain a bounded set of explicit English negative and Chinese
  residence/cancellation assertions, and item attribution rejects conflicting
  event times, rejects opposing detected polarity and requires the full negative
  assertion ([#137]). Similar opposing assertions are retained as conflicts;
  structured polarity-aware supersession is not yet implemented.

- Worker park integration fences held provider and approval delivery, persists
  pause intent and evidence-based cards, releases cleanly parked worker slots,
  and re-asks withdrawn approvals on resume. Failed park writes retry and can
  interrupt with a card; unavailable storage keeps the fence closed, and failed
  cleanup quarantines the slot until restart. Restart replay and accounting are
  deferred. [#135]
- Opt-in context fitting shortens request copies and retries one classified overflow without rerunning tools; impossible requests end with a specific status ([#131]).
- Read-only worker history in the web workspace, with server-side filters,
  paging, recorded transcripts and explicit refresh for unfinished runs. ([#130])

- Opt-in context-window request accounting reports estimated and measured planning/reflection
  tokens in events and run results; it does not prune or enforce an output cap ([#126]).

- S3b.1a foundations: scheduler component v2 adds a durable pause priority
  snapshot and atomic v1 migration; durable generation and operation IDs guard
  transitions and identify committed receipts. The internal synchronous gate
  `try_park` fences an eligible pending call and cancels only its bound turn.
  An atomic component event transaction supports `ParkStore` hold/resume/park
  records, including the current pending-pause flag. Guarded terminal retries
  return their matching committed record. An evidence-only card builder derives
  bounded, redacted cards from the task spec and persisted events. Worker API
  pause behavior remains as in #119; there is no worker hold/park integration or
  new API in this slice.
  S3b.1b follows ([#125]).

- Workspace settings gain connection testing and an immediately applied theme;
  a session details drawer shows tools, grant revocation, memory and runs.
  Store, model and stream banners no longer replace other notices. ([#124])

- Optional direct worker-session recording and dsh JSONL import, with a
  read-only, filterable and paginated gateway worker-history API. [#123]

- Workspace approval cards with a 500 ms guard, explicit allow/deny choices,
  and a jump-to-approval control; failure and stopped states with retry,
  continuation when eligible, copy-error and details; older-turn paging;
  revision-checked connection settings; and per-tab outbox recovery. Session
  update events also carry a display-safe workspace label. ([#122])
- A documentation inventory, bilingual style guide and experimental offline
  preview builder lay the groundwork for a later site migration; published
  pages and existing URLs are unchanged ([#120]).

- S3a.2 adds `/api/tasks` creation, inspection and control and in-process
  workers sharing the console's scheduler gate. Creation is idempotent with
  `request_key`; workers disable memory, omit `shell_exec`, and have at most
  four resident at once. On startup, surviving nonterminal rows fail with
  `gateway restarted; recovery arrives in S3b`; full recovery and accounting
  are deferred, deadlines are rejected, and an invalid workspace fails at run
  or resume. Pause takes effect at the next model call; running tools finish
  and answered pending approvals wait until resume; a task paused before
  start starts no worker until resumed. Resume restores the pre-pause priority,
  or computed P2 if the pause log is missing (including a task created or set
  at P5) or pruned by retention; an unpaused task returns 409 `not_paused`.
  Retention discards old terminal gate identities while keeping their rows.
  PATCH rejects `first` until S6b
  ([#119]).

- Reasoning and recall options now work in the interactive UI, including
  `--reasoning-field` and `--retrieval-extra-body`; explicit flags override
  `[agent]`/`[retrieval]` defaults. Console sessions honor the configured
  reasoning mode (default `legacy`) without changing legacy `/v1`. Session
  replay uses the retained window; pairs resume without old thinking,
  while checkpoint resume can replay retained, redacted thinking using the
  current process's mode. Scheduler wrappers preserve replay capabilities;
  rejected mode switches keep the old Agent and MCP state and close new
  providers; successful switches retain the RAM-only notes/drafts.
  Explicit-mode requests also remove legacy inline thinking
  copies after resume without changing the stored window ([#115]).
- `[agent] tool_env` names extra environment variables inherited by
  shell/Python tools in config-backed direct runs, the UI and console.
  Names are validated and missing variables skipped; only names are saved.
  SSH-agent and proxy access are explicit capability grants; defaults and
  `--base-url` runs/sessions expose nothing new ([#115]).

- The gateway's `sessions.db` can hold the scheduler's `scheduler` component
  version 1: `tasks`, `task_deps`, `priority_events`, `questions`,
  `sched_rules`, `sched_log` and `idle_windows` are created in one
  transaction in that file only, `PRAGMA user_version` stays the session
  schema's, and a newer component is refused. A task and its `kind="task"`
  session are created together behind `UNIQUE(origin, request_key)`, so a
  repeated request returns its task and a reused key with a different
  payload is a conflict; actual level changes write `priority_events` and
  30-day retention clears the log and idle windows.
  The task boundary normalizes the `TaskSpec`, pins a user-selected level
  only through a trusted adapter, and resolves a worker's folder under the
  user's home or `[gateway] workspace`; a non-user task with no folder gets
  its own child workspace. Only `user` and
  `user_rule` origins get remembered approvals, unmarked MCP tools and session
  grants. `ChatGate` can register a worker task, so its calls are logged as
  `session_kind="task"`, its P0 is the user's single task P0, and only the
  runner ends it. S3a.1 supplied the store and task policy boundary ([#113]).

- The workspace page on API v2. `http://127.0.0.1:8401/` now talks only to
  `/api`: the session list with its machine snapshot, one chat's turns, its
  event stream (SSE, reopened from the last applied event id), keyed sends
  that the store is asked about before `Retry` is offered, stop, close, the
  level chip and the typing heartbeat. A refused or unknown send goes back
  to the chat's draft, so a reload keeps it, and `#/s/<id>` opens its chat
  even when the list's first page does not hold it. A turn reads as the
  terminal's: the
  user's message, a work group with one card per tool call (`Running`,
  `Waiting: <held>`, `Needs approval`, `[ok]` with a non-zero `exit`,
  `blocked` with the approval tag, `not run`, `Interrupted: may have partly
  run`), reasoning and reflections folded away, notices, the live row
  (`step 4 · attempt 2 · Verify (cycle 1) · tests 3/8`, and on a first turn
  the routine's `last time …` facts) and the answer with its duration,
  steps, tokens and `Copy response`. That needs the S2 facts on `/api`:
  `level` and `held` on a session, `machine` on the list,
  `POST /api/sessions/{id}/level` and `POST /api/presence` on the one gate
  both surfaces share, and a console turn sets the gate's turn facts as a
  chat's message does. The page is plain ES modules with no build step
  (`workspace.js`, the new `turns.mjs` holding the event reducer and the
  terminal's wording, `markdown.mjs`). The follow-up adds approval cards,
  the failure block's actions and the settings dialog.
  `/` and every `/v1/*` route keep their behaviour ([#111]).

- API v2 settings, model listing, guard body limits and unconfigured gateway
  startup: `GET /api/settings`, `PUT /api/settings`,
  `POST /api/settings/test`, `GET /api/settings/key-status` and
  `GET /api/models`; `[gateway] workspace` in `config.toml` (validated by
  `Config.validate()`, read by console sessions, and written only when
  non-empty) and atomic `config.save()` replacement via `os.replace`
  (preserving file permissions and following a symlinked `config.toml`);
  `onboard.backend_defaults()`; a 1 MiB request body cap (`413 too_large`),
  `Content-Type: application/json` enforcement on non-empty `/api` bodies
  (`415 unsupported_media_type`), UTF-8/lone-surrogate validation, the
  console's Content-Security-Policy on `/api` and baseline security headers
  on every response; and `melo --serve --allow-unconfigured` so the gateway
  can boot before a backend is configured ([#106]).

- Terminal session commands: `/title [text]` shows or sets a title, including
  before the first message; `/undo` forgets the latest finished turn and
  keeps its typed history without reversing file changes (a committed undo
  reports success even after Esc, and `/retry` re-runs that undone turn
  without removing the previous live one); `/usage` shows
  recorded work and totals including undone turns, with missing counters
  left unknown and a narrow-terminal layout; `/memory` inspects the copied working window and explains
  its actual restore mode. `/copy [turn]` copies stored, redacted Markdown
  without thinking, preserving indentation and newlines, via bounded turn
  reads and macOS `pbcopy` or an OSC 52 request when
  remote, unsupported locally or on other platforms. Clipboard requests
  only happen on an explicit command; direct runs are unchanged ([#101]).

- `--workspace DIR` in direct mode: a task run keeps its file tools inside
  the folder and starts `shell_exec` and `python_exec` there. Without the
  flag a run keeps the current folder, exactly as before; `~` is expanded
  and the path is resolved like the interactive UI's. `melo ""` and
  `--serve` refuse it, and a path that is not a folder exits 2 with one
  line ([#98]).
- A remember request (`Remember that …`, `记住…`) now also stores the fact
  it carries after the marker, admitted by `user_request` beside the run's
  episodic digest, instead of only the digest. The fact is classified as
  `semantic` unless it is first-person (`profile`), and the redaction,
  meta-noun and question rules apply to it; a to-do such as "remember to
  buy milk" still stores no fact ([#100]).
- R6-lite's typed long-term store: `long_term.db` migrates in place to
  `user_version` 1 with `kind`, `source`, `status`, `admitted_by`,
  provenance, task/session ids, links and uses beside the text and vector,
  plus a `decisions` table. A candidate is admitted only by a rehearsal
  (restatement, a successful call it acted on, a cited note), a user
  request or statement, or `--admit-on-outcome on` for the final digest;
  the policy picks `NEW`, `UPDATE`, `RELATED`, `CONFLICT` or `IGNORE`, and
  every stored text, provenance included, passes the candidate boundary and
  `redact_secrets` (with its task text, paths and commands bounded). A long
  entry's llm catalog line carries a bounded text snippet, a BM25 row is
  exempt from the cosine floor, two items of one note stay two entries, a
  restatement merges the new note's records and supersedes its candidate, a
  newer file keeps a legacy run alive with one WARNING, a user signal needs
  a declarative, non-meta sentence, `call me` is a profile, and the three
  store modes accept and ignore model flags.
  `--keep-decision-text` keeps a decision's text beside its hash, and the
  `[memory]` table saves the outcome default. The `start` recall reads
  active entries of other tasks of every kind, by cosine or BM25, with
  `long:<entry id>` doc ids; `memory_stats` gains `admitted`, `written` and
  `ignored`, and a `memory` event reports each decision. `melo-retrieve`
  gains `--export-store`, `--export-decisions` and `--route` for the
  bench. Legacy requests, JSON line, window, `tasks`/`steps` and search
  order are unchanged; an existing `long_term.db` migrates with its rows
  intact. ([#97])
- Resume saved terminal chats with `melo -c`, `melo -r [ID|TITLE]`,
  `/sessions [filter]` or `/resume [ID|TITLE]`. The inline picker filters
  eligible chats by folder or across folders before the startup banner,
  which names the chosen chat; the recap explains what
  was restored with `[sessions] resume_mode = "pairs"` (default) or
  `"window"`. Resuming keeps the current model/workspace and clears
  approval grants, while retaining `/verbose` and `/thinking` preferences.
  Changes in another terminal are reloaded before the
  next turn; a busy or deleted session keeps the draft. `/retry` preserves
  a resumed turn's typed text, `/expand` reads stored tool output across
  event pages, and quitting prints a saved session's resume command.
  Direct runs keep their legacy parsing and output ([#95]).
- Recall in the loop: `--retriever auto|off|bm25|llm|embedding`,
  `--recall-at`, `--recall-k`, `--recall-kinds`, `--recall-placement`,
  `--recall-min-score` and `--retrieval-extra-body`, plus the `[retrieval]`
  config table and the `recall` tool. The compress modes recall the task's
  notes, drafts and thought chunks, and earlier tasks' long-term entries,
  at `start`, `attempt` and `step`: a frozen boundary block on the task
  query, one sticky block after a turn's tool results (or in its own user
  message), and the model-requested tool. `auto` uses the embedding
  retriever when configured and otherwise `llm` at `start`/`attempt` and
  `bm25` at `step`/`tool`; every failure falls back to `bm25`. Blocks are
  delimited, framed as data, checked by the candidate boundary and bounded
  (9,500 per turn, 12,000 live, 4,000 boundary); the system prompt gains
  one static data note. Each block is recorded once in
  `llm_calls.recall_json` with its hits' `origin_task_ids`, and in the
  `recall` event and `memory_stats.recall_log`. A step's query carries the
  latest note's plan, what it found and its open questions and outcome, and
  its candidates include a superseded note whose turn the window no longer
  shows, so a later attempt's earlier thinking stays reachable; such a line
  names its attempt and is marked `(superseded)`, and `recall_json` records
  each hit's `stage`.
  `memory_stats.recall_empty` counts a triggered recall that found nothing.
  `history.db` is schema
  version 2: a v1 file gains the `recall_json` column in the same
  transaction, and the column is `NULL` in `legacy`. `legacy` keeps its
  requests, JSON line, window and `tasks`/`steps` rows byte-identical; the
  new flags, the block and the data note are off there (and `drop`/`replay`
  recall only when a retriever is named). The JSON line gains
  `config.retriever`/`retriever_effective`/`recall_*` and the recall
  counters ([#94]).
- Console session details over API v2, and the scheduler gate in front of
  a console turn: `GET /api/sessions/{id}/tools` lists each tool's source,
  whether it asks first and whether the session allows it, `/memory` the
  conversation the gateway holds (thinking stripped, every text redacted,
  text cut at 2,000 characters) and `/runs` each finished turn's `RunResult`
  without its answer, while `DELETE /api/sessions/{id}/grants/{tool}` and `/grants`
  take a session grant back. Every model call of a console turn now waits
  for its slot under `[scheduler]`, like a workspace chat's: the console
  shares the gateway's one `ChatGate` and wraps the Agent's providers
  with `GatedLLM`/`GatedEmbedding` through `build_session_agent`'s hooks,
  so the call log records the session's id and `lock_gate = "wait"` holds
  a turn while a foreign GPU lock is present. A create now makes its
  session row before it builds the Agent, because the gate needs the id;
  its key still lands in the create scope, so a client without the id finds
  it, and a create whose Agent or first turn never came up, or was
  cancelled, deletes the row it made. Console runners ask each turn for
  `run.progress`. `/`, the workspace page and every `/v1/*` route are
  unchanged ([#92]).
- Console turns over API v2: `POST /api/sessions` (with an optional first
  turn), `POST /api/sessions/{id}/turns`, `/retry` and `/turns/{id}/cancel`,
  the two cancel-by-key routes, `POST /api/approvals/{request_id}`, and
  `PATCH`/`DELETE` of a session. A turn starts in the background and its
  events arrive on the session's stream: `approval.requested` waits for an
  answer given through the API, and the turn finishes `turn.finished` with
  the result read from its row. A `client_request_id` makes a create or a
  turn safe to repeat, one turn runs per session (409 `session_busy`), and
  a stop during an approval wait withdraws the card. The console keeps the
  `ApprovalBroker` and its `RememberingPolicy` through `runner.observe`, so
  the approvals a console turn asks show on the stream. A console turn
  does not pass the scheduler gate yet; that arrives with the tools and
  grants change. `docs/api.md` gains the routes and a tested Python script
  that creates a session, follows the stream, answers an approval and reads
  the answer; every request string must be valid UTF-8 (a lone surrogate
  is 400 `invalid_request`), a session's workspace is stored as its real
  path, and `/`, the workspace page and every `/v1/*` route are unchanged
  ([#91]).
- Reasoning notes for the compress modes: `--reasoning compress` and
  `compress+replay`, `--note-writer auto|heuristic`, and the `[agent]`
  `note_writer` key. `compress` sends no reasoning; instead each reply is
  compressed into a short note (what it planned, found, decided and left
  open, plus what its tools returned), with its code drafts, in a RAM-only
  note store beside the working window. A pending targeted draft is pinned
  once after its turn's tool results; a successful later write of 60% of
  its lines marks it written. Notes are rehearsed when a later note
  restates them, a successful call acts on a path or command they name, or
  a reflection cites them, and each attempt gets a digest from its rehearsed
  notes. `compress+replay` also sends thinking as `replay` does; a backend
  that requires replay runs `compress` as `compress+replay` and the JSON
  line reports that. Every document passes the new
  `check_memory_candidate` boundary (redact, sanitize, cap) before it
  reaches the window or the store. Nothing is recalled from the notes yet
  (retrieval is a later release), no notes table or schema change reaches
  `history.db`, the explicit modes stop writing raw step summaries and
  promoting reflections to long-term memory, and a session turn's `note`
  column gets the final digest. Legacy is byte-identical, goldens included
  ([#85]).
- The gateway opens its console session store and serves API v2 reads under
  `/api`: health with the versions, limits, store state and a cached
  model-server check; the sessions list and one session, a pending
  `file_write` approval's `target` and each session grant's `granted_at`;
  a session's turns with their events; a Server-Sent Events stream with
  `Last-Event-ID` and one-shot catch-up; image blobs whose CSP merges into
  the gateway's framing rule; and submission lookups by `client_request_id`.
  The console holds each session's Agent, keeps its conversation when an
  idle MCP session or the open cap closes it, and resumes it on reopen.
  When the store cannot open, health still serves and the reads answer 503
  `store_unavailable` while the gateway retries the open every 5 s. `/api`
  is loopback-only, carries the one error envelope and `Cache-Control:
  no-store`, and is additive-only and documented in
  [docs/api.md](docs/api.md); `/` and every `/v1/*` route are unchanged, and
  no turn starts yet ([#84]).
- A reasoning mode for direct runs: `--reasoning legacy|drop|replay`
  (default `legacy`), `--reasoning-field auto|reasoning_content|reasoning`
  and `--replay-turns N`, or the same `[agent]` keys in `config.toml`.
  `legacy` is byte-identical to before everywhere, including a provider
  built from the config by the gateway, a session or the Python API; `drop`
  never sends reasoning, and `replay` sends it on the assistant turns in
  scope (by default only the current task's) under one key, with a
  40,000-character per-turn cap and strip-first eviction. A backend whose
  `requires_reasoning_replay` is true gets every assistant turn in the
  window and keeps the final answer's thinking with `record_answers`. The
  JSON line reports the mode, `usage.reasoning_chars`, and the
  `memory_stats` replay counters outside `legacy`. The interactive UI
  refuses `--reasoning-field` until R2b wires it ([#78]).
- Diffs in the approval panel: a `file_write` over an existing file shows
  the change (`+a -b lines`, removed lines in red and added in green),
  a new file its line count, and a file too large or not text says why.
  `/approvals` shows the mode and this session's grants and switches
  between `ask`, `auto` and `deny` (or forgets one tool's grant). The
  terminal UI accepts `--mcp-config`, and the config file's top-level
  `mcp_config` key names an MCP servers file the UI loads (direct mode
  never reads it); `/mcp` lists the servers and their tools and says why
  MCP is off after a failure, `/status` shows the model and server, the
  session, the tools, memory, limits and files, and `/retry` runs the
  newest turn again. The hints name them where they apply ([#75]).
- Setup can add the embedding model that long-term memory needs. The first
  run offers an embedding model the chat server already lists, the wizard
  appends such models as options `6+`, the banner names the configured
  backend and model (or the flag that adds one), and a stored vector from
  another model is skipped instead of failing recall. A new
  [guide](docs/embedding.md) covers Ollama, oMLX, vLLM, SGLang and MLC
  ([#74]).
- The terminal and the web console share one set of words for a session.
  `support/sessions/display.py` re-exports `split_thinking` and adds the
  status table `STATUS_TEXT`, keyed by status and by `(status, cause)` and
  read through `status_text` (each with its word, severity and title), and
  `approval_tag`, which gives the short tag and long text of how a call's
  approval ended. `build_session_agent` takes optional `wrap_llm` and
  `wrap_embedding` callables, called with the providers it builds before
  the Agent's `Chat` is made, so a surface can put its scheduler gate in
  front of every model call; with neither, the builder is unchanged. The
  terminal reads its status severities from `STATUS_TEXT`, and its rendered
  lines and goldens are unchanged ([#72]).
- The web workspace's chats go through the scheduler: each chat call is
  admitted at once or held according to `[scheduler] policy` (`off`
  by default, which only records), each embedding call is recorded and counts
  in flight but is never held, a chat's level chip (P1-P4), the focused
  chat's lane, the server's `n_srv` slots and the `r_f`/`n_fg` reserves. A
  held chat shows `Waiting: <reason>` in the page, and every call leaves one
  schema-1 line in `~/.cache/localmelo/scheduler/runs/<UTC>/calls.jsonl`.
  Direct runs, the CLI and `/v1/agent/run` are unchanged and gated nowhere
  ([#69]).
- The gateway reads the shared GPU lock and `pmset -g batt`, records both in
  `machine.jsonl` beside the call log, and shows the holder and the battery
  in the workspace header. `[scheduler] lock_gate` defaults to `record`,
  which changes no request; `wait` holds every workspace call, embeddings
  included, while a foreign lock is present, and shows
  `Waiting: Reservation`. `lock_self` is an exact holder (or `holder:run`);
  unreadable and stale locks are foreign, and localmelo never writes,
  renames or removes a lock. Direct runs, the CLI and `/v1/agent/run` are
  unchanged and read neither source ([#76]).
- Tools have a permission tier, and an approval policy asks by tier rather
  than by a fixed list of names. The built-ins keep their tiers (`file_read`
  read, `file_write` write, `shell_exec` and `python_exec` exec), and a
  provided tool's raw MCP `annotations` are kept on `ToolDef.annotations`
  and read cautiously: `openWorldHint` gives network, `readOnlyHint` gives
  read unless `destructiveHint` contradicts it, and anything else gives
  write. So an unannotated MCP tool now asks in a run with a policy, while
  `require_approval` only adds caution. The terminal approval panel names
  the tier in its header, `ToolInfo.tier` and `ApprovalRequest.tier` report
  it, and `--yolo` still means no policy, not read-only ([#66]).
- Memory can be forgotten from the Python API. `await hippo.forget(task_id)`
  deletes that task's `tasks`, `steps` and `llm_calls` rows and the
  long-term entries whose metadata names it (its reflections), and
  `await hippo.forget_all()` empties history and long-term memory, step
  summaries included. `Hippo` has the facade; `SqliteHistory`,
  `SqliteLongTerm`, `History` and `LongTerm` have the two methods. A forget
  runs in one `BEGIN IMMEDIATE` transaction under the write lock, overwrites
  what it deletes and truncates the WAL, and leaves working memory,
  procedural memory and sessions alone. There is no command or UI yet
  ([#64]).
- A session runner can ask its turns for progress events, and the progress
  line's past facts are available alone. `SessionRunner(..., progress=True)`
  asks each turn's run for `run.progress` and stores it with the turn; the
  flag stays on the runner, `replace_agent` included. `run_turn` and
  `start_turn` take `routine="NAME"` and pass it to the run only when
  given. `localmelo.support.progress_line.format_past(payload, width)`
  gives exactly the direct-mode line's past-facts-and-items segment, with
  the same wording and width rules. A session turn emits `run.progress`
  only when its runner was built with `progress=True`; the interactive UI
  does not ask yet (its render commit does) ([#60]).
- An interactive UI in the terminal: `melo` without a task opens a chat
  that shows its work as it happens. A banner names the model, the
  workspace and the tools; each tool call prints as a block with the first
  lines of its output, a spinner shows the step and the time while the
  model works, and the answer prints as Markdown. `/` completes the
  commands with their descriptions: `/help`, `/new`, `/model` (one of
  the models the server lists), `/backend` and `/exit`; `/model` and
  `/backend` save the choice to the config file, except in a `--base-url`
  session. `shell_exec`,
  `python_exec` and `file_write` ask first (yes, allow for the session, or
  no with a reason); `--yolo` runs them without asking, and
  `--workspace DIR` sets the folder the file tools stay in. The input box
  edits Chinese and other wide characters whole, recalls earlier messages
  (↑↓, Ctrl+R) and makes a new line with Alt+Enter or Ctrl+J; Esc stops a
  turn. Sessions are kept in `~/.cache/localmelo/cli/sessions.db`;
  resuming them comes later ([#53]).
- A turn's work in the interactive UI, as it happens: the model's
  thinking (`✻ Thought for 1.2s` and 3 rows; 40 with `/verbose`, none with
  `/thinking off`), intermediate text and one held message printed only
  when it differs from the answer, `↻` retries, `·` empty replies, `⚠`
  stuck and budget notices, and reflections (including the unreadable
  ones). Tool blocks take the call's arguments and 20 output lines under
  `/verbose` and print images as `▪ image/png · 312 KB · not shown`. The
  live area now draws the run's `run.progress` facts, the running tool's
  clock and limit, and on a session's first turn a dim row of the past
  runs (the direct-mode line's wording). `/expand [turn.call]` prints a
  call's whole output, `/progress` the latest facts with the memory they
  rest on ([#68]).
- A first run in a terminal: `melo` with no config file looks for a model
  server on this computer, asks for its URL and for a model from the
  server's list, saves them and opens the chat. It looks with
  `GET /v1/models` on ports 8000, 8400, 11434 and 30000 of `127.0.0.1`
  (`LOCALMELO_DETECT_URLS` replaces them), and sends the chosen server one
  `GET /api/version` when saving, to tell Ollama apart; a server that cannot be used is
  named with the reason, and you can retry, start without saving or quit
  ([#52]).
- A progress line in the terminal. While `melo "<task>"` runs with stderr
  on a terminal and text output, one dimmed line on stderr shows the step,
  the phase, the running tool and the time; once the task has run before
  with memory kept, it adds what the last run took and its steps (or a
  similar task's, marked as such). The line is erased before the answer,
  draws nothing in the background, never wraps, and never holds up the
  run. `--progress off` turns it off, and `--routine NAME` groups
  differently worded runs of one task. Pipes, files, `--output-format json`
  and the benchmark harness see no change ([#51]).
- Runs learn the steps of recurring tasks. Each run that ends normally
  records an episode in procedural memory, and a completed run appends a
  new version of its routine's step template: in `procedures.db` in the
  memory directory with persistence on, in RAM otherwise. In the Python
  API, `run_task(..., progress=True)` with a sink adds `run.progress`
  events (levels 0 and 1; the percent bar is off until replayed runs
  support it), `observe=` watches a run without making it interactive,
  `routine=` names a routine, and `BaseLLMProvider.model` names a chat
  provider's model. The query text and labels procedural memory keeps
  also mask URL passwords and the values of names such as `API_KEY` or
  `password`, and `melo-procedures forget` leaves nothing of a forgotten
  routine in the file. Request bodies, the
  output, the JSON line, exit codes and history rows are unchanged
  ([#46]).
- Procedural memory (experimental), `localmelo.melo.memory.procedures`, and
  the offline `melo-procedures` command (`episodes`, `replay`, `export`,
  `forget`). From the runs in a `history.db` it learns the steps of a task
  you run again and again: the classes of its tool calls, short labels, step
  counts and times, never the model's text. Templates are versioned and kept
  in `procedures.db`. `replay` measures the rules of a planned progress
  display on recorded runs. Runs, the gateway and the Python API do not use
  it yet ([#45]).
- With persistence on, `history.db` keeps every reply of the model with its
  reasoning, in a new `llm_calls` table: one row per planning or reflection
  reply, with its kind, outcome, step, tool calls, token usage and duration,
  and the tool steps it made. Text and reasoning pass a minimal redaction of
  secrets. Reasoning is kept 30 days, or the days in
  `LOCALMELO_THINKING_RETENTION_DAYS`, in `history.db` and `sessions.db`
  alike. `melo-retrieve --export-run` exports these calls from a run's
  directory ([#42]).
- Every chat request carries an `X-Localmelo-Call-Kind` header (`plan` or
  `reflect`; `retrieve` for `melo-retrieve`), so a proxy can tell the kinds
  of calls apart. Request bodies are unchanged ([#42]).
- Sessions in the Python API (experimental), `localmelo.support.sessions`.
  A session keeps a conversation in SQLite: each turn begins with a stored
  row, outlives the request that started it, and ends exactly once with a
  status and a cause, also when the process that ran it dies; a fresh agent
  can resume the conversation. The package has the store, a runner for an
  agent's turns, approval helpers for surfaces that ask a person, an event
  stream and a read-only change feed. Neither `melo` nor the gateway uses
  sessions yet ([#41]).
- `docs/for-agents.md`, one compact page that tells AI coding agents and LLM
  tools what localmelo is and how to use it, and `docs/llms.txt`, an index of
  the docs in the llms.txt format, served at
  <https://localmelo.github.io/localmelo/llms.txt> ([#40]).
- The scheduler's policy core and simulator, `localmelo.support.scheduler`
  (experimental). It decides which model calls reach a shared local model
  server, in what order and why a task waits, and its simulator compares the
  `fifo`, `class` and `las2` arms on synthetic workloads or imported bench
  traces. Nothing uses it yet: runs and the gateway are unchanged
  ([#39]).
- `melo-retrieve`, installed next to `melo`, compares the `bm25`, `llm` and
  `embedding` memory retrievers offline on labelled queries: recall@k, mean
  reciprocal rank, latency and token use. `--export-run` turns a run's model
  calls or a history database into a corpus of reasoning notes, code drafts,
  thought chunks and reflections ([#38]).
- The web workspace replaces the gateway's playground page, at
  `http://127.0.0.1:8401/` by default. It discovers your server's models,
  keeps a draft for each chat, searches chats, renders Markdown with code,
  tables and links, can stop a run, and fits narrow screens. Chats last while
  the gateway runs, and a reload restores them ([#35]).
- `melo ""` opens a conversation in the terminal, with the configured backend
  or the server given by `--base-url`. `/help` lists its commands, and
  `/exit` or Ctrl-D leaves ([#35]).
- Run events for Python code: `Agent.run_task(..., on_event=sink)` passes
  each run, step, model reply, tool call, notice and reflection to the sink
  as an `AgentEvent`, with secrets redacted ([#37]).
- Cancelling a run that has an event sink or an approval policy ends it as
  `cancelled`, with the interrupted step closed in history and in the
  conversation. `Agent.begin_exit()` bounds the clean-up before the process
  exits ([#37]).
- Approval policies: `run_task(..., approval=policy)` is asked before each
  call of `shell_exec`, `python_exec`, `file_write` and other tools marked
  as requiring approval. An MCP server can mark its tools with
  `require_approval` ([#37]).
- `AgentOptions(workspace_root=...)` keeps `file_read` and `file_write`
  inside a directory, where `shell_exec` and `python_exec` also start
  ([#37]).
- `AgentOptions(record_answers=True)`, `Agent.conversation()` and
  `Agent.load_conversation()`, to carry a conversation across runs and
  agents; `Agent.start()`, `Agent.list_tools()` and `Agent.run_tool()`
  ([#37]).
- Direct-mode run options `--max-steps`, `--attempt-policy`,
  `--tool-exposure`, `--builtins`, `--memory`, `--memory-dir` and
  `--timeout`, and `AgentOptions` in the Python API ([#30]).
- `--embedding-base-url` and `--embedding-model`, for long-term memory in a
  `--base-url` run ([#30]).
- Nine run statuses and the `RunResult` record. `--output-format json` prints
  the record as one JSON line, with `localmelo_version` ([#30]).
- Exit codes: 0 for `completed`, 1 for `crashed`, 2 for any other status or a
  usage error ([#30]).
- A reflection can end a run as `completed` with the `done` action ([#30]).
- Tools from stdio MCP servers: `McpToolProvider` ([#27]), and `--mcp-config`
  and `--mcp-only` in direct mode ([#30]).
- Tools can return images. History and working memory keep only references
  to them: MIME type, SHA-256 digest and size ([#27]).
- The chat request timeout: `--request-timeout` and the `request_timeout_s`
  config key ([#30]).
- SIGTERM and SIGHUP end a direct run within 8 seconds, with its task record
  saved ([#30]).
- `extra_body` in each backend's config table, for extra chat request fields
  ([#26]).
- `--reflection-gate flags|utility` and `AgentOptions.reflection_gate`. The
  JSON line's `config` gains `reflection_gate` ([#34]).
- The MIT license ([#28]).
- A uv development environment with a committed `uv.lock` ([#25]).

### Changed

- Organize Markdown documentation into start, guides, reference, concepts and
  project sections. Old paths retain readable links for each historical
  fragment; the agent entry points and printed CLI paths remain stable ([#129]).

- An attempt's `compress` digest now says what the attempt did: the paths its
  successful `file_write` calls wrote (at most six, workspace-relative, then
  `+N`), the last `shell_exec` or `python_exec` check and its result (test
  counts, exit code or last output line), up to three decisions, and the
  task's first sentence or heading instead of a fixed 300-character cut.
  Empty or trivial observations (a bare boolean, `None`, an exit code or
  punctuation) no longer enter a digest or make one a rehearsal candidate.
  Over the limit, open questions drop first, then observed lines, decisions,
  facts and files, then the task shortens; the outcome and the verification
  stay. `legacy` and direct mode without memory are unchanged ([#116]).
- Direct mode and the Python API's default `python` mode no longer offer
  `shell_exec`: their default built-ins are `file_read`, `file_write` and
  `python_exec`, and an `exec` call that was registered anyway is refused
  with a reason naming the mode and the way to enable it. `--builtins all`
  (or naming `shell_exec`) restores the old set, and a named shell then
  runs. A call to a built-in the run does not offer returns
  `shell_exec is not available in this run. Available tools: …`, naming the
  run's tools; an unknown name keeps the registry message. `--builtins` is
  still a usage error with `--serve` and `--daemon`. The JSON line's
  `config` gains `mode` (`"direct"` in a direct run), and `config.builtins`
  and `config.tools` lose `shell_exec` there. The six-pattern command
  blocklist, `validate_executor_request` and `ExecutorRequest` are gone: the
  tool's tier and the mode decide, not the command's spelling. The `melo ""`
  loop says once at start that `shell_exec` is off and that
  `melo "" --builtins all` turns it on ([#110]).
- `shell_exec` and `python_exec` start their child with a minimal
  environment instead of inheriting melo's: `PATH`, `HOME`, `USER`,
  `LOGNAME`, `SHELL`, `TERM`, `LANG`, every `LC_*` variable and `TMPDIR`,
  then the new `AgentOptions(tool_env={...})`, and nothing else. An API key
  in melo's environment reaches no tool; `VIRTUAL_ENV`, `PYTHONPATH`,
  `SSH_AUTH_SOCK` and the proxy variables pass only through `tool_env`. MCP
  servers keep their minimal environment, which now takes every `LC_*`
  variable too, from the same list ([#81]).
- Tool hints come from the model's tool calls only. A tool named in the
  task, in a tool result, in recalled memory or in a reflection no longer
  offers that tool, so text in a run cannot put a tool the model never
  called back in front of it; the built-ins, the BM25 matches and the
  re-offer after a refused call are unchanged ([#80]).
- `python_exec` runs its code in a child Python in every mode, not only with
  a workspace root. Code that loops, sleeps or calls `os._exit()` no longer
  holds or ends `melo`: the tool's time limit (60 seconds by default, 10
  minutes in the interactive UI) and a run's `timeout_s` kill it with every
  process it started. The code reaches the child on its standard input, a
  pipe, never in its `-c` argument, so `ps` does not show it and a call has
  no `ARG_MAX` limit; once the fixed wrapper has read the code it points
  fd 0 at `/dev/null`, so `input()` gets EOF. The child gets a POSIX
  CPU-time limit of one hour and an open-file limit of 1,024. Success
  output is unchanged. A failure now reads
  `Python exited with code <n>:` and the end of the child's output instead
  of the exception's text, and `sys.exit(0)` is a success. The code runs as
  `__main__` with `sys.argv == ["-c"]`, in a fresh module namespace: module
  state no longer carries from one call to the next, and writes to file
  descriptor 1 or 2, from a subprocess too, are the call's output instead of
  melo's. The `tool.finished` event's `exit_code` now carries the child's
  status without a root too ([#79]).
- The README is short, and the website carries the documentation. Its header
  is two badge rows: one that goes to the website's pages (Website,
  Quickstart, Docs, Developers, 简体中文) and a quieter row of facts
  (pre-alpha, Python 3.11+, MIT, llms.txt for AI agents). Below it are what
  localmelo is, install, a few commands, a "Learn more" list of the
  website's pages and the security warning; everything else that was there
  is on the website or in `docs/`. `CONTRIBUTING.md` is short too, and links
  to a new developer guide, `docs/en/developers.html` and
  `docs/zh-CN/developers.html`, linked from every page's navigation: the
  repository layout and the `melo/` ↔ `support/` rule, the package layout,
  setup, tests, ruff/black/mypy, adding a backend, a built-in tool or a
  check, commits and pull requests, and the docs and website layout
  ([#67]).
- The `dev` extra installs `pytest-xdist` and `pytest-testmon`, so
  `uv run pytest -n auto -q` runs the offline suite in parallel before a
  pull request and `uv run pytest --testmon -q` runs only the tests a
  change affects while working. CONTRIBUTING and the agent docs name both,
  and `.testmondata` is ignored ([#58]).
- A plain `pytest` no longer opens sockets to local model servers while
  collecting `tests/smoke/test_backend_smoke.py`; its two real-backend tests
  are skipped unless `LOCALMELO_SMOKE=1` is set. With the switch on they probe
  and send requests as before, and
  `--ignore=tests/smoke/test_backend_smoke.py` is no longer needed for an
  offline run ([#57]).
- `melo` without a task, in a terminal, opens the interactive UI on the
  saved backend; it used to start the gateway, which `melo --serve` still
  does. `melo --base-url URL` and `melo --chat-model NAME` without a task
  open it too, where they used to be ignored; `--base-url` without
  `--chat-model` uses the first chat model the server lists.
  `melo --reconfigure` opens it after the wizard ([#52], [#53]).
- Without a terminal (stdin and stdout must be TTYs and `TERM` not
  `dumb`), bare `melo` says what to run instead and exits with code 2;
  `melo --reconfigure` runs the wizard and exits. The UI takes the run
  options, such as `--max-steps` and `--memory-dir`, for every turn;
  without a task, `--tool-exposure`, `--timeout`, `--output-format`,
  `--progress`, `--routine`, `--mcp-config` and `--mcp-only` are usage
  errors, and with a task, `--workspace` and `--yolo` are. `melo ""`
  keeps the plain conversation loop ([#53]).
- The setup wizard offers the config file's values as its defaults, and
  `None` for embeddings when none is set. A failed chat check can be
  retried with the values you typed; a failed embedding check saves with
  embedding off instead of cancelling the setup ([#52]).
- A memory directory that localmelo creates gets mode 0700, and a new
  `history.db` mode 0600. The default directory is tightened to 0700 when it
  is open to other users ([#42]).
- `--chat-model` without `--base-url` replaces the configured chat model for
  that run; it used to be ignored ([#35]).
- A follow-up question in a gateway session, a web chat or the `melo ""`
  loop sees the earlier answers, without their `<think>` blocks and cut to
  8,000 characters each ([#35]).
- A step runs only the tools it offered. A call to any other tool is refused
  and does not run ([#30]).
- A call to a built-in tool whose arguments have the wrong JSON type is
  refused before it runs ([#30]).
- A failed model request is retried when a retry may succeed. A request that
  still fails ends the run as `provider_error` ([#30]).
- In `retrieve` tool exposure, every step offers all registered built-in
  tools, then the extra tools found by hints and by Okapi BM25 with IDF, both
  sorted by name ([#32]).
- After a reflection, the next attempt starts when the reflection chooses
  `continue` and its three checks hold; its utility score no longer has to
  reach 0.1. A reply that still holds no reflection after the retry no longer
  ends a run that has attempts and steps left, and a reflection over 2,000
  characters is trimmed instead of ending the run. A reply whose only JSON is
  not a reflection, such as a tool call written out as text, is asked for
  again too. `--reflection-gate utility` keeps the previous behavior
  ([#34]).

### Fixed

- Correct documentation of retrieval option validation, persistent memory,
  interface event streams and current architecture status ([#121]).

- The heuristic note writer reads factual replies without a `<note>` into up
  to 12 `found` items, and an explicit note can also hold 12. Questions,
  options, plans and negated or cancelled statements are not promoted as
  positive facts. Self-work claims are filtered unless a keyed content
  clause follows the frame; `task:`/code markers and completed outcome
  digests remain exempt. Source matching scores typed keys independently,
  retains date years, currencies and whole phone numbers, rejects conflicts,
  and leaves ties and unmatched items unsourced even after admission. Bare
  slash dates use US order when ambiguous; quantity/count contexts, mixed
  numbers and `24/7` are excluded. Digit runs over 20 characters are ignored
  as keys.
  Factual extraction remains heuristic and does not extract CJK-only keys.
  ([#114])
- A note's items now take their `source_ids` from the observation that
  established them, even when it came in an earlier reply of the same task:
  the reply's own tool results first, then the results still in the window,
  then the rest of the task's (newest first, last 200 records). A record
  matches only on a strong key (a number, a date or time, an identifier
  containing a digit, or a capitalised name that is not sentence-initial)
  or two distinctive non-filler words, and a note with records but no
  match gets no item source, so "Done." no longer cites an unrelated
  "no harm done" record. This fills `provenance.source_ids` for facts
  copied from an earlier tool result and for the digest built from them;
  another task's records are never used ([#103]).
- The model-facing recall header no longer carries the retrieval latency:
  it is `(retriever bm25)`, not `(retriever bm25, 12 ms)`. Identical runs
  now produce identical prompt bytes, so a block kept in the window does
  not break the prefix cache. `ms` remains in `llm_calls.recall_json`, the
  `recall` event and `memory_stats` ([#103]).
- A thought chunk of an attempt is now superseded at the attempt boundary
  together with that attempt's notes, so a chunk recalled later renders
  with the `(superseded)` marker and names its attempt instead of passing
  for the current attempt's thinking ([#100]).
- `melo-retrieve --export-store DIR` and `--export-decisions DIR` print
  zero entries and exit 0 when the directory has no `long_term.db` yet,
  instead of failing with "No such file"; a path that is not a directory
  keeps the missing-file usage error ([#100]).
- A thought chunk recalled with `--recall-kinds thought` no longer replays
  the reply that produced it: its alias stays with that step's turn, so
  step recall and the `recall` tool return it only after the turn left the
  window, as they already did for notes and drafts ([#96]).
- The test suite no longer writes into the user's own `~/.cache/localmelo`:
  every test runs with `HOME` in a temporary directory, the defaults of the
  gateway's call log, the web workspace's memory and `config.toml` resolve
  when they are used instead of at import, and a session guard fails a test
  that resolves the real home or writes the real cache
  (`@pytest.mark.allow_real_cache` opts a test out of the home rule)
  ([#90]).
- `POST /v1/agent/run` answers a non-string `session_id` (a number, a list)
  with the route's own 400 `Invalid session_id format` instead of FastAPI's
  plain-text 500, and its 100,000-character limit counts the query after the
  surrounding whitespace is removed, which is the text the agent runs: a
  whitespace-padded query whose trimmed text fits is accepted ([#89]).
- A held chat's header still said `● Working` while the sidebar, status
  label and message area said it was waiting, and a chat that had just sent
  kept `Working on your request…` in the message area for the whole hold
  (the poll skipped the detail refresh while the send was pending). The
  header's status pill now says `Waiting: <reason>`, and while the send is
  pending the message area follows the polled hold as it changes; both
  return to Working when the hold ends ([#71]).
- The scheduler simulator's importer reads a chat call's kind from the
  `X-Localmelo-Call-Kind` header that localmelo sends, in the headers side
  file a proxy record names with `bodies.headers`. Retrieval and
  step-estimate calls, which have no tools, no longer replay as `reflect`;
  pre-R1 records and records without `--log-bodies` keep the old guess, an
  unknown header value keeps the guess and is counted, and the import
  report counts the records of each kind source ([#61]).
- `DELETE /v1/sessions/{id}` no longer closes a session whose task is still
  running: it answers 409 with the gateway's error body
  (`{"error", "session_id"}`) and leaves the task and the session alone;
  idle sessions close as before. The `/v1/agent` API still has no cancel
  route, so a caller waits for the task. Stopping the gateway still closes
  every session, a busy one included, and a task still running then ends as
  `crashed` ([#63]).
- A `shell_exec` command no longer inherits melo's standard input: it reads
  `/dev/null` in every mode. A shell whose stdin was a pty that was not
  melo's controlling terminal took that pty as its controlling terminal at
  start-up and hung it up when it exited, so `melo` ended a completed run
  with exit code 120 instead of 0 ([#59]).
- A server picked at the first run, at
  `Save URL · MODEL as the default backend?` or with `/backend` was always
  saved as the `vllm` backend. An Ollama server, saved that way, ran with
  vLLM's 120 s chat timeout instead of Ollama's 300 s, and the banner said
  vLLM. A server that answers `GET /api/version` like Ollama is saved as the
  `ollama` backend now; any other OpenAI-compatible server is still saved as
  `vllm` ([#55]).
- A similar task's past steps on the progress line said `like a similar
  task:`; they name the task they are from, `like <routine title>:` (the
  routine's name, else the start of its query) ([#56]).
- A `shell_exec` call whose `command` arrived as an argv array (a list of
  strings, which the tool refuses) was labelled `{command:` on the
  progress line, and procedural memory stored that label; the call is now
  classified and labelled by the command's words, joined with
  `shlex.join`, so a read command such as `["ls", "-la"]` is R rather
  than X ([#56]).
- `melo --serve` with no config file and no terminal, as under launchd,
  ended in an `EOFError` traceback, again every few seconds; it prints one
  line and exits with code 1. Ctrl-D or Ctrl+C in the setup wizard prints
  `Setup cancelled.` instead of a traceback ([#52]).
- The config file escapes quotation marks and backslashes in URLs, model
  names and other values; such a value made the file unreadable, so every
  later `melo`, `melo --reconfigure` included, failed ([#52]).
- An embedding URL that is not valid, such as a mistyped port, ended the
  setup wizard with a traceback after the chat check passed; the config is
  now saved with embedding off ([#52]).
- `--base-url` and `--chat-model` were silently ignored with `--serve` and
  `--daemon`; they are usage errors now ([#52]).
- An HTTP 401 or 403 from a local server during setup says that localmelo
  sends no API key to local servers (oMLX: "Allow inference without API
  key") ([#52]).
- In the `melo ""` loop, Backspace deletes whole characters: it no longer
  leaves part of a Chinese or other wide character on the screen, or in the
  line, which made the loop stop with a `UnicodeDecodeError`. The prompt is
  edited with readline on a terminal, so the Up arrow recalls earlier lines
  ([#50]).
- A request that disconnects during a run no longer leaves its gateway
  session busy for good ([#35]).
- Stopping the gateway closes the web workspace's model connections
  ([#35]).
- `--host '[::1]'` starts the gateway, and the startup lines show an IPv6
  address in brackets, as in `http://[::1]:8401/` ([#36]).
- Tool calls are sent to OpenAI-compatible models as real tool-call turns,
  and every call of a step runs ([#26]).
- Empty replies, oversized prompts and long tool results no longer end a run
  early or silently ([#30]).
- A reflection reply without a JSON object is asked for once more, without
  tools ([#31]).

### Security

- Gateway and console sessions no longer offer `shell_exec`: both run in the
  new `gateway` tool mode, which registers `file_read`, `file_write` and
  `python_exec` only: legacy `/v1` has no approval policy; the console asks for
  `file_write` and `python_exec` using `RememberingPolicy`, and
  the console's approvals arrive over the same unauthenticated loopback API,
  so any local process could approve its own request. `AgentOptions` gains
  `mode` (`direct`, `interactive`, `gateway` or `python`; `python` is the
  API default) and the CLI sets its own surface's mode; the interactive UI
  keeps `shell_exec` behind its existing approval panel. Direct mode and the
  library's `python` default follow in the entry above. An explicit
  `AgentOptions(builtins=…)` or `--builtins` still names `shell_exec` in
  `direct` and `python` mode, and the call then runs; `gateway` refuses it
  even when named ([#104]).
- Secrets no longer stay in stored memory. `tasks.query` and `tasks.result`,
  `steps.thought`, a tool call's arguments and a tool result's output and
  error, and the text long-term memory keeps (each step's summary and the
  promoted reflections) are written with the same redaction `llm_calls` and
  events already use: `sk-` keys, GitHub tokens, AWS access key ids, the
  token after `Bearer` and a one-time code next to its keyword. The run and
  the model keep the text: working memory, the requests a run sends, its
  events and its `result` are unchanged, and the stored images stay
  references ([#83]).
- On `localhost` or a loopback IP address, the gateway refuses requests that
  are not addressed to it, before any route runs. A `Host` other than
  `127.0.0.1`, `localhost` or `[::1]` with the gateway's port gets HTTP 421
  `misdirected_request`. A `POST`, `PUT`, `PATCH`, `DELETE` or `OPTIONS`
  request from a web page at another origin, and a `/v1/` request that the
  browser marks as cross-site or same-site, get HTTP 403 `foreign_origin`.
  Web pages could otherwise reach the gateway through DNS rebinding or
  cross-site requests and run tasks with its tools. The standalone
  playground server applies the same checks ([#36]).
- No web page can show the gateway's pages in a frame: every response
  carries `X-Frame-Options: DENY` and
  `Content-Security-Policy: frame-ancestors 'none'` ([#36]).
- On any other host, such as `0.0.0.0`, the gateway makes none of these
  checks and logs a warning when it starts ([#36]).
- A new `long_term.db` is created with mode 0600, like `history.db` and
  `procedures.db`, with its `-wal` and `-shm`; it got the default mode
  before, usually 0644, so other users of the computer could read the step
  summaries and reflections it holds. An existing file keeps its mode
  ([#62]).

Changes before these are not listed here. See the
[updates page](https://localmelo.github.io/localmelo/en/updates.html) for
earlier milestones.

[Unreleased]: https://github.com/localmelo/localmelo/commits/main
[#25]: https://github.com/localmelo/localmelo/pull/25
[#26]: https://github.com/localmelo/localmelo/pull/26
[#27]: https://github.com/localmelo/localmelo/pull/27
[#28]: https://github.com/localmelo/localmelo/pull/28
[#30]: https://github.com/localmelo/localmelo/pull/30
[#31]: https://github.com/localmelo/localmelo/pull/31
[#32]: https://github.com/localmelo/localmelo/pull/32
[#34]: https://github.com/localmelo/localmelo/pull/34
[#35]: https://github.com/localmelo/localmelo/pull/35
[#36]: https://github.com/localmelo/localmelo/pull/36
[#37]: https://github.com/localmelo/localmelo/pull/37
[#38]: https://github.com/localmelo/localmelo/pull/38
[#39]: https://github.com/localmelo/localmelo/pull/39
[#40]: https://github.com/localmelo/localmelo/pull/40
[#41]: https://github.com/localmelo/localmelo/pull/41
[#42]: https://github.com/localmelo/localmelo/pull/42
[#45]: https://github.com/localmelo/localmelo/pull/45
[#46]: https://github.com/localmelo/localmelo/pull/46
[#50]: https://github.com/localmelo/localmelo/pull/50
[#51]: https://github.com/localmelo/localmelo/pull/51
[#52]: https://github.com/localmelo/localmelo/pull/52
[#53]: https://github.com/localmelo/localmelo/pull/53
[#55]: https://github.com/localmelo/localmelo/pull/55
[#56]: https://github.com/localmelo/localmelo/pull/56
[#57]: https://github.com/localmelo/localmelo/pull/57
[#58]: https://github.com/localmelo/localmelo/pull/58
[#59]: https://github.com/localmelo/localmelo/pull/59
[#60]: https://github.com/localmelo/localmelo/pull/60
[#61]: https://github.com/localmelo/localmelo/pull/61
[#62]: https://github.com/localmelo/localmelo/pull/62
[#63]: https://github.com/localmelo/localmelo/pull/63
[#64]: https://github.com/localmelo/localmelo/pull/64
[#66]: https://github.com/localmelo/localmelo/pull/66
[#67]: https://github.com/localmelo/localmelo/pull/67
[#68]: https://github.com/localmelo/localmelo/pull/68
[#69]: https://github.com/localmelo/localmelo/pull/69
[#71]: https://github.com/localmelo/localmelo/pull/71
[#72]: https://github.com/localmelo/localmelo/pull/72
[#74]: https://github.com/localmelo/localmelo/pull/74
[#75]: https://github.com/localmelo/localmelo/pull/75
[#76]: https://github.com/localmelo/localmelo/pull/76
[#78]: https://github.com/localmelo/localmelo/pull/78
[#79]: https://github.com/localmelo/localmelo/pull/79
[#80]: https://github.com/localmelo/localmelo/pull/80
[#81]: https://github.com/localmelo/localmelo/pull/81
[#83]: https://github.com/localmelo/localmelo/pull/83
[#84]: https://github.com/localmelo/localmelo/pull/84
[#85]: https://github.com/localmelo/localmelo/pull/85
[#91]: https://github.com/localmelo/localmelo/pull/91
[#92]: https://github.com/localmelo/localmelo/pull/92
[#89]: https://github.com/localmelo/localmelo/pull/89
[#94]: https://github.com/localmelo/localmelo/pull/94
[#95]: https://github.com/localmelo/localmelo/pull/95
[#96]: https://github.com/localmelo/localmelo/pull/96
[#90]: https://github.com/localmelo/localmelo/pull/90
[#97]: https://github.com/localmelo/localmelo/pull/97
[#100]: https://github.com/localmelo/localmelo/pull/100
[#101]: https://github.com/localmelo/localmelo/pull/101
[#103]: https://github.com/localmelo/localmelo/pull/103
[#104]: https://github.com/localmelo/localmelo/pull/104
[#106]: https://github.com/localmelo/localmelo/pull/106
[#110]: https://github.com/localmelo/localmelo/pull/110
[#111]: https://github.com/localmelo/localmelo/pull/111
[#113]: https://github.com/localmelo/localmelo/pull/113
[#114]: https://github.com/localmelo/localmelo/pull/114
[#115]: https://github.com/localmelo/localmelo/pull/115
[#116]: https://github.com/localmelo/localmelo/pull/116
[#119]: https://github.com/localmelo/localmelo/pull/119
[#120]: https://github.com/localmelo/localmelo/pull/120
[#121]: https://github.com/localmelo/localmelo/pull/121
[#122]: https://github.com/localmelo/localmelo/pull/122
[#123]: https://github.com/localmelo/localmelo/pull/123
[#124]: https://github.com/localmelo/localmelo/pull/124
[#125]: https://github.com/localmelo/localmelo/pull/125
[#126]: https://github.com/localmelo/localmelo/pull/126
[#128]: https://github.com/localmelo/localmelo/pull/128
[#129]: https://github.com/localmelo/localmelo/pull/129
[#130]: https://github.com/localmelo/localmelo/pull/130
[#131]: https://github.com/localmelo/localmelo/pull/131
[#132]: https://github.com/localmelo/localmelo/pull/132
[#133]: https://github.com/localmelo/localmelo/pull/133
[#135]: https://github.com/localmelo/localmelo/pull/135
[#136]: https://github.com/localmelo/localmelo/pull/136
[#137]: https://github.com/localmelo/localmelo/pull/137
[#141]: https://github.com/localmelo/localmelo/pull/141
[#142]: https://github.com/localmelo/localmelo/pull/142
[#143]: https://github.com/localmelo/localmelo/pull/143
[#144]: https://github.com/localmelo/localmelo/pull/144
[#145]: https://github.com/localmelo/localmelo/pull/145
[#147]: https://github.com/localmelo/localmelo/pull/147
