# Changelog

All notable changes to localmelo are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

localmelo has no release yet. The v0.1.0 milestone is due 2026-12-20.

## [Unreleased]

### Added

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

- The scheduler simulator's importer reads a chat call's kind from the
  `X-Localmelo-Call-Kind` header that localmelo sends, in the headers side
  file a proxy record names with `bodies.headers`. Retrieval and
  step-estimate calls, which have no tools, no longer replay as `reflect`;
  pre-R1 records and records without `--log-bodies` keep the old guess, an
  unknown header value keeps the guess and is counted, and the import
  report counts the records of each kind source ([#61]).
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
  ([#NN]).

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
