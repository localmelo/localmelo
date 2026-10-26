# Changelog

All notable changes to localmelo are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

localmelo has no release yet. The v0.1.0 milestone is due 2026-12-20.

## [Unreleased]

### Added

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
