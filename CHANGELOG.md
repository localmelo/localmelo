# Changelog

All notable changes to localmelo are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

localmelo has no release yet. The v0.1.0 milestone is due 2026-12-20.

## [Unreleased]

### Added

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
- The MIT license ([#28]).
- A uv development environment with a committed `uv.lock` ([#25]).

### Changed

- A step runs only the tools it offered. A call to any other tool is refused
  and does not run ([#30]).
- A call to a built-in tool whose arguments have the wrong JSON type is
  refused before it runs ([#30]).
- A failed model request is retried when a retry may succeed. A request that
  still fails ends the run as `provider_error` ([#30]).
- In `retrieve` tool exposure, every step offers all registered built-in
  tools, then the extra tools found by hints and by Okapi BM25 with IDF, both
  sorted by name ([#32]).

### Fixed

- Tool calls are sent to OpenAI-compatible models as real tool-call turns,
  and every call of a step runs ([#26]).
- Empty replies, oversized prompts and long tool results no longer end a run
  early or silently ([#30]).
- A reflection reply without a JSON object is asked for once more, without
  tools ([#31]).

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
