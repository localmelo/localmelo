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
- `--reflection-gate flags|utility` and `AgentOptions.reflection_gate`. The
  JSON line's `config` gains `reflection_gate` ([#34]).
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
- After a reflection, the next attempt starts when the reflection chooses
  `continue` and its three checks hold; its utility score no longer has to
  reach 0.1. A reply that still holds no reflection after the retry no longer
  ends a run that has attempts and steps left, and a reflection over 2,000
  characters is trimmed instead of ending the run. A reply whose only JSON is
  not a reflection, such as a tool call written out as text, is asked for
  again too. `--reflection-gate utility` keeps the previous behavior
  ([#34]).

### Fixed

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
[#36]: https://github.com/localmelo/localmelo/pull/36
