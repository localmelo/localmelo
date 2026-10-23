# Quickstart

This guide covers the minimum needed to run localmelo, verify the
installation, and run the test suite.

Sections:

- [Installation](#installation)
- [Usage modes](#usage-modes)
- [Run results](#run-results) — statuses and the `RunResult` record of a run
- [Direct CLI options](#direct-cli-options) — flags, output, exit codes and
  signals of a direct query
- [Minimum smoke](#minimum-smoke) — verify each mode end-to-end after install
- [Backend options](#backend-options) — backends, `extra_body` and the
  [chat request timeout](#chat-request-timeout)
- [No-embedding mode](#no-embedding-mode)
- [Memory persistence](#memory-persistence)
- [Running tests](#running-tests)

## Installation

```bash
# Core + dev dependencies (tests, linting)
pip install -e ".[dev]"

# Core + dev + gateway dependencies (FastAPI, uvicorn)
pip install -e ".[dev,gateway]"
```

## Usage modes

### Direct CLI mode

Run a one-shot query without starting a server. With `--base-url` (below)
no configuration file is needed; otherwise the query uses the backends of
the configuration file, so run `melo --reconfigure` first. Without either,
melo prints `No backend configured` and exits with code 1.

```bash
melo "what is 2+2"
```

Override the backend URL and model on the fly:

```bash
melo --base-url http://localhost:11434/v1 --chat-model qwen3:8b "what is 2+2"
```

Further flags set the run's limits, its tools and memory, and how it
reports its outcome; see [Direct CLI options](#direct-cli-options).

### Gateway mode

Start the HTTP gateway, which serves an agent-as-a-service API. Requires
a configuration file (run `melo --reconfigure` to create one).

```bash
melo --serve
```

Override host and port:

```bash
melo --serve --host 0.0.0.0 --port 5000
```

### Gateway API

Once the gateway is running, interact via HTTP:

```bash
curl -X POST http://localhost:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "hello"}'
```

Sessions are created implicitly by `/v1/agent/run`. Pass an optional
`session_id` to reuse an existing session or create a named one:

```bash
# Run a query (creates a new session automatically)
curl -X POST http://localhost:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "hello"}'

# Reuse a named session across requests
curl -X POST http://localhost:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "follow up", "session_id": "my-session"}'

# List active sessions
curl http://localhost:8401/v1/sessions

# Delete a session
curl -X DELETE http://localhost:8401/v1/sessions/my-session
```

### Python API

`Agent`, `AgentOptions` and `RunResult` are importable from the package
root:

```python
import asyncio

from localmelo import Agent, AgentOptions
from localmelo.support.config import load


async def main() -> None:
    agent = Agent(config=load(), options=AgentOptions(max_steps=10))
    try:
        result = await agent.run_task("What is 6*7?")
        print(result.status, result.result or result.error)
    finally:
        await agent.close()


asyncio.run(main())
```

`AgentOptions` sets the step budget (`max_steps`), the attempt policy
(`attempt_policy`, `steps_per_attempt`, `max_attempts`), tool exposure
(`tool_exposure`), the built-in tools (`builtins`), memory (`memory`,
`memory_dir`) and a wall-clock bound for each run in seconds (`timeout_s`,
off by default); its docstring describes each option and its default.
The HTTP timeout of each chat request is not one of them: it belongs to
the chat provider. `Config.request_timeout_s` sets it for
`Agent(config=...)` (see [Chat request timeout](#chat-request-timeout)),
and the `timeout` argument of `OpenAICompatLLM` or `OllamaNativeChat` for
a provider passed as `llm`.
`Agent.run_task(query)` returns a [`RunResult`](#run-results).
`Agent.run(query)` returns the run's outcome as text, which is what the
gateway and its web pages show: the result when it is non-empty, otherwise
`[<status>] <error>`, for example `[budget_exhausted] Max steps reached`.
One agent runs one task at a time.

`Agent(..., tool_providers=[...])` adds tools from outside the core, such
as the tools of MCP servers (`McpToolProvider` from `localmelo.support.mcp`,
named `mcp__<server>__<tool>`). The first run starts every provider and
registers its tools next to the built-ins; a name that is already taken
ends the run as `setup_error`. Provided tools pass the same executor checks
as the built-ins, each call bounded by the timeout its provider gives, with
one exception: localmelo checks the JSON types of the built-ins' arguments
only, since a provider's server checks its own.
`Agent.close()` closes the providers.

## Run results

Every run ends with one of these statuses:

| Status | Meaning |
|---|---|
| `completed` | The model gave a final answer, or a reflection decided that the task was already done (`done`). The answer of a `done` is the reflection's `best_effort_result` or, when that is blank, the latest non-blank text the model wrote in this run. |
| `failed` | A checker refused the plan or the run (the prompt outgrew the pre-plan cap even with only the task and the newest working-memory turn left, tool resolution failed, or the model's response failed its check), or the model gave two empty replies in a row: `error` is then `empty replies`. An empty reply, with neither a tool call nor text, is no answer; the agent records it in the conversation as the assistant message `[empty reply]`, adds the user message "Your last reply was empty. Continue the task: call a tool or give your final answer." after it and plans again, and the step counts. Neither message goes to history. |
| `gate_stopped` | With `attempt_policy="reflect"`, the reflection after an attempt decided not to continue while attempts and steps were left: it chose `stop` or `decompose`, its `continue` did not pass the continuation gate, or its reply could not be parsed. `result` holds the reflection's best effort, if any; `error` is its rationale (`Could not parse reflection response` for an unreadable reply), or `Stopped after attempt <n>` when the rationale is blank. |
| `budget_exhausted` | `max_steps` or `max_attempts` ran out before a final answer. With `attempt_policy="reflect"`, the reflection that follows is told that no further attempt can start; unless it chooses `done`, the run ends here, and `result` holds its best effort, if any. |
| `stuck` | Stuck detection fired (the same tool calls with the same results, or the same errors, three steps in a row) with `attempt_policy="none"`. With `"reflect"`, a reflection decides what happens next instead. Repeated calls that return new results, such as polling a job, are not stuck. |
| `provider_error` | A request to the model server failed: an HTTP error status, a timeout, a connection failure or a malformed response body (see [Provider errors](#provider-errors)). `error` describes the failed request; no traceback is logged. |
| `setup_error` | The agent's tool providers (see [Python API](#python-api)) could not be set up at the start of its first run: one failed to start, for example an MCP server that could not be launched, or a provided tool has the name of a tool already registered. `error` says which. No provider is left running, and setup is not retried: every later run on the same agent ends as `setup_error` too, with `error` naming the first failure. |
| `timeout` | The run was still going after `AgentOptions.timeout_s` seconds and was cancelled where it stood: an LLM request in flight is abandoned and counts in `llm_errors`, and a `shell_exec` command is killed together with every process it started. `python_exec` is the exception: it runs its code inside the agent itself, which can do nothing else until the code returns, so neither `timeout_s` nor the executor's timeout can stop it; a run that times out during that code ends only once the code returns, and code that never returns keeps the run from ever ending. Tool providers still being set up are closed, so later runs end as `setup_error`. Closing them is never cut short, also when the deadline expires while a failed setup is closing them (the run then ends as `timeout` rather than `setup_error`): the run returns once every provider is closed, which can be after `timeout_s`. `error` is `Timed out after <timeout_s> s`. |
| `crashed` | An unexpected exception ended the run. `error` is `<ExceptionType>: <message>` and the traceback is logged. A run cancelled from outside is recorded as `crashed` too, and the cancellation propagates. A direct-mode run ended by SIGTERM or SIGHUP is recorded as `crashed` with the error `Cancelled by SIGTERM` or `Cancelled by SIGHUP` (see [SIGTERM and SIGHUP](#sigterm-and-sighup)). Code that `python_exec` runs cannot end the run this way: `sys.exit()` and `exit()` fail that tool call, with an error the model sees, and the run goes on. |

`Agent.run_task()` returns the run as a `RunResult`; `RunResult.to_dict()`
gives the same fields as a plain dict, in this order:

| Field | Meaning |
|---|---|
| `task_id` | id of the run's task record in history: 32 hex characters (a UUID4) |
| `status` | one of the statuses above |
| `result` | the final answer or a best effort; may be empty |
| `error` | why the run did not complete; empty when it completed |
| `steps` | planning calls made, at most `max_steps` |
| `attempts` | attempts started (1 when the first attempt answered) |
| `n_llm` | LLM requests that returned a response |
| `n_main` | of those, requests that offered tools |
| `n_aux` | of those, requests without tools: reflections, and planning calls in a step that offered no tool |
| `n_tool` | tool calls handed to the executor, including calls it refused |
| `llm_errors` | LLM requests that raised instead of responding, each failed attempt of a retried request and a request cut short by `timeout_s` included, so `n_llm + llm_errors` is the number of requests sent |
| `usage` | integer usage fields of all responses, summed by name (`prompt_tokens`, `completion_tokens`, `total_tokens`) |
| `duration_s` | wall-clock time of the run, in seconds |
| `config` | the effective run configuration (below) |

`config` holds `max_steps`, `attempt_policy`, `steps_per_attempt`,
`max_attempts`, `tool_exposure`, `builtins` (sorted names of the registered
built-in tools), `memory`, `memory_dir` (the resolved directory, or `null`),
`timeout_s` (or `null`) and `tools` (sorted names of every registered tool,
built-in and provided), so every record says which mode it ran in. The
[chat request timeout](#chat-request-timeout) (`request_timeout_s`,
`--request-timeout`) is a setting of the chat provider, not of the run, so
it is not part of `config`.

Whatever the status, the run's task record is saved with it; see
[memory.md → Task records](memory.md#task-records).

### Provider errors

The chat and embedding providers report a failed request as
`ProviderError` (from `localmelo.melo.contracts.providers`), which says
whether sending the same request again may succeed:

| Failure | Retried |
|---|---|
| timeout, connection failure or another transport error | yes |
| HTTP 429, 502, 503 or 504 | yes |
| any other HTTP error status, for example a proxy's 400 or a 500 | no |
| a response body that is not JSON or lacks the expected fields | no |

A chat request that may succeed is retried at most twice, after 1 s and
then after 4 s more; if the last attempt fails too, the run ends as
`provider_error`. Embedding requests are not retried. The error names the
request and, when the server answered, gives its HTTP status and the first
500 characters of its response body, for example
`POST /chat/completions failed with HTTP 400: {"error": ...}`.

A chat request that times out is retried the same way, and each attempt
waits the full [chat request timeout](#chat-request-timeout) again. So a
model server that sends nothing ends the run as `provider_error` after
about 3 × the timeout + 5 s: 365 s with the default of 120 s, and 905 s
with the 300 s of the `ollama` backend. A server that sends keep-alive
bytes while it generates may never time out (see
[Chat request timeout](#chat-request-timeout)).

Every chat request carries its attempt number (1, then 2 and 3 for the
retries) in the `X-Localmelo-Attempt` HTTP header, so a proxy in front of
the model server can match its request log against a run's `n_llm` and
`llm_errors`. A custom `BaseLLMProvider` gets the same retries by raising
`ProviderError(message, status_code=..., retryable=True)` for a failure
worth retrying.

## Direct CLI options

A direct query (`melo "<query>"` without `--serve`) takes the flags below,
each of which sets the [`AgentOptions`](#python-api) field it names, adds
tools or an embedding endpoint, or sets the chat request timeout. Given
with `--serve`, or without a query, they are a usage error.

| Flag | Default | Effect |
|---|---|---|
| `--max-steps N` | `30` | Cap on planning calls for the whole run (`max_steps`), at least 1. |
| `--attempt-policy reflect\|none` | `reflect` | `reflect`: attempts with a reflection between them. `none`: one attempt of up to `--max-steps` steps and no reflection calls. |
| `--tool-exposure retrieve\|all` | `retrieve` | `retrieve`: each step offers the tools that hint extraction and BM25 search resolve for it. `all`: every registered tool, sorted by name, on every step. |
| `--builtins all\|none\|NAME,...` | `all` | The built-in tools to register: all, none, or a comma-separated list out of `file_read`, `file_write`, `python_exec` and `shell_exec`. |
| `--mcp-config PATH` | none | Add the tools of the MCP servers in this file (see below). |
| `--mcp-only` | off | Only the MCP tools: the same as `--builtins none --tool-exposure all`. Requires `--mcp-config`, and cannot be combined with `--builtins` or `--tool-exposure`. |
| `--memory on\|off` | `on` | `off`: no long-term recall, no long-term writes and no reflection promotion. |
| `--memory-dir DIR` | none | Keep history (and long-term memory, when there is an embedding and memory is on) in SQLite files in `DIR`; see [Memory persistence](#memory-persistence). |
| `--timeout SECONDS` | none | Wall-clock limit of the run (`timeout_s`); a run still going then ends as `timeout`. |
| `--request-timeout SECONDS` | `request_timeout_s` of the config file, else `120` (`300` for the `ollama` backend) | HTTP timeout of each chat request, a finite number of seconds > 0. It overrides `request_timeout_s` for this run, so a configured value that is not valid is not reported, and does not change the config file; with `--base-url`, which does not read the config file, it goes to the OpenAI-compatible provider. A request that times out is retried twice, each time with the full timeout. It is not an `AgentOptions` field and not part of the run's `config`. See [Chat request timeout](#chat-request-timeout). |
| `--output-format text\|json` | `text` | How the outcome is reported; see [Output](#output). |
| `--embedding-base-url URL` | none | An OpenAI-compatible embedding endpoint, for long-term memory. Requires `--base-url`: a query with `--base-url` has no embedding without it, and one without `--base-url` uses the backends of the config file, its embedding backend included. |
| `--embedding-model NAME` | `default` | The embedding model to request. Requires `--embedding-base-url`. |

`--mcp-config` reads the `mcpServers` JSON format that other MCP clients
use. Only stdio servers are supported:

```json
{
  "mcpServers": {
    "notes": {
      "command": "python",
      "args": ["-m", "notes_mcp"],
      "env": {"NOTES_DIR": "/srv/notes"}
    }
  }
}
```

Each server may also set `cwd`, `timeout` (seconds per tool call, default
120), `startup_timeout` (seconds to start and list tools, default 30),
`include_tools` (the names of the tools to register; the others are left
out) and `disabled` (`true` skips the server). A server gets only a
minimal environment (`PATH`, `HOME`, the locale and a few more) plus its
`env`. The servers start when the run starts, and their tools are
registered as `mcp__<server>__<tool>`; a server that cannot be started
ends the run as `setup_error`.

A flag value that is not valid (for example `--max-steps 0`, an unknown
built-in name, or a `--memory-dir` that cannot be created, such as the
path of a file), a flag combination that is not allowed (such as
`--mcp-only` without `--mcp-config`), and an MCP config file that cannot
be read or is malformed (text that is not UTF-8 included) end melo before
the run, with exit code 2 and a `melo: error: ...` message on stderr.

### Output

With `--output-format text`, the default, the run's result, if it is not
empty, goes to stdout. A run that did not complete adds one line on
stderr, for example:

```text
melo: run ended with status budget_exhausted: Max steps reached
```

The error in that line has its whitespace and line breaks collapsed to
single spaces and is cut to 500 characters (ending in `...`), so an HTML
error page that a provider error quotes stays on one line; a blank error
reads `no error message`.

With `--output-format json`, melo prints exactly one line to stdout: the
`RunResult.to_dict()` object, with the fields of [Run results](#run-results)
in that order, and a last key `localmelo_version`. That is the output of
`git describe --tags --always --dirty` when localmelo runs from a git
checkout of its own, and otherwise the installed package version (or
`unknown`). Nothing else goes to stdout, and there is no status line:

```json
{"task_id": "3f0c1d2e4b5a46978899aabbccddeeff", "status": "completed", "result": "42", "error": "", "steps": 1, "attempts": 1, "n_llm": 1, "n_main": 1, "n_aux": 0, "n_tool": 0, "llm_errors": 0, "usage": {"prompt_tokens": 812, "completion_tokens": 9, "total_tokens": 821}, "duration_s": 1.84, "config": {"max_steps": 30, "attempt_policy": "reflect", "steps_per_attempt": 10, "max_attempts": 5, "tool_exposure": "retrieve", "builtins": ["file_read", "file_write", "python_exec", "shell_exec"], "memory": true, "memory_dir": null, "timeout_s": null, "tools": ["file_read", "file_write", "python_exec", "shell_exec"]}, "localmelo_version": "v0.1.0-42-g1a2b3c4"}
```

In both formats, stdout carries only the report. What a tool writes to
stdout by itself during the run, such as a command that `python_exec`
code starts without capturing its output, goes to stderr instead.
Warnings go to stderr one line each, for example a tool provider that
could not be closed or a task record that could not be saved. A traceback
appears only for a run that ended as `crashed`, or, before any run, for an
error that melo does not report in one line, such as a configuration file
that cannot be parsed (exit code 1).

### Exit codes

| Code | Meaning |
|---|---|
| `0` | The run ended as `completed`. |
| `2` | The run ended with any other status except `crashed` (`failed`, `gate_stopped`, `budget_exhausted`, `stuck`, `provider_error`, `setup_error`, `timeout`), or a usage error stopped melo before the run. |
| `1` | The run ended as `crashed`; its traceback is on stderr. A query without `--base-url` also exits with 1 before the run when no backend is configured or the config file is not valid, apart from a `request_timeout_s` that `--request-timeout` replaces; a config file that is not valid TOML prints a traceback. |

### SIGTERM and SIGHUP

SIGTERM and SIGHUP cancel a direct run, and melo's clean-up ends within
8 seconds, so a supervisor that sends SIGKILL 10 seconds after its SIGTERM
finds nothing left to kill. Shell commands and MCP servers run in sessions
of their own, so a signal sent to melo's process group does not reach
them: melo stops them itself, in this order:

1. A running `shell_exec` command is killed with its process group as the
   cancellation lands.
2. The MCP servers are closed, those still starting included. A server
   that has not stopped within 6 seconds of the signal is killed with its
   process group (a server started through a wrapper such as `sh -c`
   too, after the wrapper has exited), so a server that will not stop
   cannot keep the task record from being saved.
3. The task record is saved as `crashed`, with the error `Cancelled by
   SIGTERM` or `Cancelled by SIGHUP`; a run that had already ended, and
   was only saving its record, keeps its own status. The reflections are
   not promoted to long-term memory: that calls the embedding server and
   could take longer than the time left.
4. melo exits by the signal itself (a shell reports exit status 143 for
   SIGTERM and 129 for SIGHUP), and prints no result, no status line and
   no JSON line.

If the clean-up is still running after 8 seconds, melo kills the MCP
servers and exits by the signal at once. Further SIGTERM or SIGHUP signals
during the clean-up change nothing, and a signal that melo was started
with ignored stays ignored (`nohup` ignores SIGHUP). `python_exec` is the
exception here, as it is for `--timeout`: its code runs inside melo, which
handles the signal only once that code returns.

## Minimum smoke

After installing and configuring localmelo, run the commands below to
verify each path works end-to-end. These are the canonical smoke commands
referenced from issue [#3](https://github.com/localmelo/localmelo/issues/3)
and [#13](https://github.com/localmelo/localmelo/issues/13).

The smoke is split into four parts:

1. [Direct CLI smoke](#1-direct-cli-smoke)
2. [Gateway smoke](#2-gateway-smoke)
3. [Session reuse smoke](#3-session-reuse-smoke)
4. [Per-backend minimum config + chat verify](#4-per-backend-minimum-config--chat-verify)

A passing run for parts 1-3 plus at least one backend in part 4 is the
minimum localmelo "works on this machine" signal.

### 1. Direct CLI smoke

```bash
melo "What is 6*7?"
```

Pass criteria:

- the process exits cleanly with status `0`
- the final answer mentions `42`
- no traceback is printed

### 2. Gateway smoke

```bash
# Start the gateway in a separate shell (or background it)
melo --serve

# Health check — confirms the configured backend is reachable
curl http://127.0.0.1:8401/v1/health

# Run a query through the gateway
curl -X POST http://127.0.0.1:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query":"Say hello briefly"}'
```

Pass criteria:

- `/v1/health` returns JSON with `"status": "ok"` and shows the active
  backend / model
- `/v1/agent/run` returns JSON with both `result` and `session_id`
- the gateway log line on startup names the backend, model, embedding
  status, and bound URL

### 3. Session reuse smoke

```bash
# First call — creates a named session
curl -X POST http://127.0.0.1:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query":"remember this is session test","session_id":"demo123"}'

# Second call — reuses the same session
curl -X POST http://127.0.0.1:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query":"continue","session_id":"demo123"}'

# Confirm the session is listed, then delete it
curl http://127.0.0.1:8401/v1/sessions
curl -X DELETE http://127.0.0.1:8401/v1/sessions/demo123
```

Pass criteria:

- both `POST` calls return the same `session_id` (`demo123`)
- `GET /v1/sessions` lists the session before delete and not after
- no session-transition errors appear in the gateway log

### 4. Per-backend minimum config + chat verify

For each backend you want to support, run `melo --reconfigure`, supply
the minimum config below, then re-run the [Direct CLI smoke](#1-direct-cli-smoke)
to confirm chat works.

#### `mlc` (local MLC-LLM server)

Minimum config:

| Field | Value |
|---|---|
| `chat_backend` | `mlc` |
| `chat_url` | `http://127.0.0.1:8000/v1` |
| `chat_model` | a served model id, e.g. `Qwen3-1.7B` or `Qwen3-4B` |
| `embedding_backend` | `mlc` (or `none` to skip long-term memory) |
| `embedding_model` | e.g. `Qwen3-Embedding-0.6B` (only if `embedding_backend != "none"`) |

Verify:

```bash
melo "What is 6*7?"
```

#### `ollama` (local Ollama server)

Minimum config:

| Field | Value |
|---|---|
| `chat_backend` | `ollama` |
| `chat_url` | `http://127.0.0.1:11434/v1` |
| `chat_model` | e.g. `qwen3:8b` |
| `embedding_backend` | `ollama` (or `none`) |
| `embedding_model` | the embedding model name your Ollama instance has pulled (only if not `none`) |

Verify:

```bash
melo "What is 6*7?"
```

#### `online` (cloud chat — `openai`, `anthropic`, `gemini`, `nvidia`)

Minimum config:

| Field | Value |
|---|---|
| `chat_backend` | one of `openai`, `anthropic`, `gemini`, `nvidia` |
| `chat_model` | a vendor model id, e.g. `gpt-4o-mini`, `claude-sonnet-4-20250514`, `gemini-2.0-flash` |
| API key env | the env var named in `api_key_env` for the chosen backend, e.g. `OPENAI_API_KEY` |
| `embedding_backend` | a local backend (e.g. `ollama`) for long-term memory, or `none` |

Verify:

```bash
export OPENAI_API_KEY=sk-...   # or the key env var for your chosen vendor
melo "What is 6*7?"
```

### No-embedding smoke

When you do not have a local embedding server, set
`embedding_backend = "none"`. The agent loop still runs end-to-end —
only long-term memory is disabled. Verify with the same direct-CLI
command:

```bash
melo "What is 6*7?"
```

Pass criteria:

- the answer is returned without errors
- the gateway `/v1/health` (if running) reports embedding as disabled, not
  failed

See [No-embedding mode](#no-embedding-mode) below for the full behavior
contract.

## Backend options

localmelo uses a **split backend model**: `chat_backend` and
`embedding_backend` are configured independently via `melo --reconfigure`.

### Supported backends

| Backend | Type | Description |
|---------|------|-------------|
| `ollama` | local | Ollama-compatible server |
| `mlc` | local | MLC-LLM server |
| `vllm` | local | vLLM server |
| `sglang` | local | SGLang server |
| `openai` | cloud | OpenAI API |
| `gemini` | cloud | Google Gemini API |
| `anthropic` | cloud | Anthropic API |
| `nvidia` | cloud | NVIDIA API |
| `none` | — | No backend (embedding only: disables long-term memory) |

**Local backends** run as external processes managed by the user.
localmelo connects to them via a configured URL — it does not
host, compile, or serve models itself.

**Cloud backends** connect to vendor APIs directly. Set the API key
via the environment variable named in `api_key_env`
(e.g. `OPENAI_API_KEY`).

### Extra request fields (`extra_body`)

Every chat backend section in `~/.cache/localmelo/config.toml` accepts an
optional `extra_body` table. Its keys are merged into the top level of
every chat request, which is how server-specific options such as
`chat_template_kwargs` reach an OpenAI-compatible server:

```toml
[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B"

[vllm.extra_body.chat_template_kwargs]
enable_thinking = false
```

- Inline form: `extra_body = { chat_template_kwargs = { reasoning_effort = "low" } }`.
- A flat `chat_template_kwargs = { ... }` key in the section is accepted as
  shorthand and is rewritten as `extra_body` the next time the config is saved.
- `model`, `messages`, `tools`, `tool_choice` and `stream` are managed by
  localmelo and cannot be set through `extra_body`.
- An invalid `extra_body`, such as one that sets one of those fields, is
  reported as a configuration error when `melo --serve` or a direct query
  without `--base-url` starts, and as a failed probe in `melo --reconfigure`.
- For the `ollama` backend the table is merged into the native `/api/chat`
  body, e.g. `extra_body = { think = false }`.
- Direct `--base-url` mode does not read the config file, so `extra_body`
  does not apply there.

### Chat request timeout

The HTTP timeout of each chat request is 120 seconds with the
OpenAI-compatible chat provider, which every backend but `ollama` uses,
and 300 seconds with the `ollama` backend, which talks to Ollama's native
`/api/chat`. It limits inactivity, not the whole request: it bounds
connecting, sending the request and each wait for the next data from the
server, and starts again whenever data arrives. localmelo does not stream
replies, so with a server that sends nothing until its reply is complete,
the last wait lasts until the model has written its whole reply, which
can take a local thinking model longer than the timeout. A server that
sends keep-alive bytes while it generates restarts the timeout with each
of them. oMLX, for example, sends a space every 10 seconds while it writes
a reply that is not streamed, so against it a timeout longer than 10
seconds never fires, however long the reply takes; only `--timeout`
(`AgentOptions.timeout_s`), when set, limits the run. Set a longer
timeout with a top-level `request_timeout_s` in
`~/.cache/localmelo/config.toml`, in the block before the first table: a
key written below a `[table]` header belongs to that table, and
localmelo does not read it there.

```toml
chat_backend = "vllm"
embedding_backend = "none"
request_timeout_s = 600

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B"
```

- The value is a finite number of seconds greater than 0, such as `600`
  or `90.5`. Anything else (a string such as `"600"`, a boolean, 0, a
  negative number, `inf` or `nan`) is reported as an error naming
  `request_timeout_s`: as a configuration error when `melo --serve` or a
  direct query without `--base-url` starts, and as a failed probe, before
  any probe request is sent, in `melo --reconfigure`. Such a direct query
  exits with code 1, unless `--request-timeout` replaces the value for
  that run: the configured value is then not checked, and the query runs.
- Without the key, every provider keeps its default. `melo --reconfigure`
  keeps the key, and the config file has the line only when it is set.
- It applies to every chat request of the configured backend, for all
  eight backends: planning and reflection requests of direct queries
  without `--base-url` and of gateway sessions, and the connectivity probe
  of `melo --reconfigure`.
- Embedding requests keep their own timeout of 120 seconds. The setting is
  for chat requests only, on purpose: long thinking is what needs a longer
  timeout, and long-term recall and reflection promotion keep their
  current bounds.
- `--request-timeout SECONDS` sets it for one direct query: it overrides
  `request_timeout_s` for that run and does not change the config file.
  With `--base-url` the config file is not read, so the timeout is the
  flag's value or 120 seconds, also for an Ollama server's OpenAI-compatible
  `/v1` URL.
- A chat request that times out is retried twice, after 1 s and then 4 s,
  each attempt with the full timeout (see [Provider errors](#provider-errors)).
  Against a server that sends nothing, one chat request can therefore take
  about 3 × the timeout + 5 s before the run ends as `provider_error`:
  about 30 minutes with `request_timeout_s = 600`. Against one that sends
  keep-alive bytes, it can take as long as the reply. `--timeout`
  (`AgentOptions.timeout_s`) bounds the whole run either way.
- It is a setting of the chat provider, not a run option: it does not
  appear in `RunResult.config` or in the JSON line of `--output-format json`.

### Deployment matrix

Any local or cloud backend can be used for chat. Embedding can use
any local backend or `none`. Example combinations:

| Chat         | Embedding   | Notes                              |
|--------------|-------------|------------------------------------|
| `ollama`     | `ollama`    | Fully Ollama-based stack           |
| `openai`     | `ollama`    | Cloud LLM + local Ollama embedding |
| `mlc`        | `mlc`       | Fully local MLC-LLM stack          |
| `vllm`       | `vllm`      | Fully local vLLM stack             |
| `anthropic`  | `ollama`    | Cloud LLM + local embedding        |
| `gemini`     | `none`      | Cloud LLM, no long-term memory     |

> **Future:** a separate repository for local backend deployment and
> runtime management is being considered, but it is not part of
> localmelo core yet.

## Backend architecture

Backend-specific logic lives under `support/backends/`. Each backend
implements the `BaseBackend` contract:

```text
support/backends/
  base.py            # BaseBackend ABC
  registry.py        # register(), get_backend(), list_backends()
  openai_compat.py   # Shared OpenAI-compatible provider helpers
  local/             # Local backends (user-managed external processes)
    mlc.py           # MLC-LLM
    ollama.py        # Ollama
    vllm.py          # vLLM
    sglang.py        # SGLang
  cloud/             # Cloud vendor backends
    openai_api.py    # OpenAI
    gemini_api.py    # Google Gemini
    anthropic_api.py # Anthropic
    nvidia_api.py    # NVIDIA
```

The app layer (`agent.py`, `gateway/__init__.py`, `onboard.py`) uses the
registry to dispatch to the correct backend -- no hardcoded if/elif chains.

### Adding a new backend

1. Create `support/backends/local/mybackend.py` or `support/backends/cloud/mybackend.py` implementing `BaseBackend`
2. Register it in `support/backends/registry.py`'s `ensure_defaults_registered()`
3. Add the backend key to `VALID_CHAT_BACKENDS` and/or `VALID_EMBEDDING_BACKENDS` in `support/config.py`
4. Add a config dataclass in `support/config.py` if needed
5. Add tests in `tests/backends/test_mybackend.py`

## No-embedding mode

When `embedding_backend` is set to `"none"` (or `embedding=None` in
code), localmelo:

- Skips all long-term memory operations (store and retrieve)
- Uses only short-term (sliding window) context for conversations
- Still supports tool use, history recording, and the full agent loop
- The `Config.has_embedding` property reflects this at config time

This is useful when you want to use a cloud LLM without running a local
embedding server.

## Memory persistence

By default, working, history, and long-term memory live only in RAM and are
lost on process exit. To persist memory across runs, set
`LOCALMELO_PERSIST_MEMORY=1`. SQLite files are written to
`~/.cache/localmelo/memory` by default (`history.db` always,
`long_term.db` only when an embedding provider is configured), and
`LOCALMELO_MEMORY_DIR` overrides the directory.

```bash
export LOCALMELO_PERSIST_MEMORY=1
# optional: export LOCALMELO_MEMORY_DIR=/var/lib/localmelo/memory
melo --serve
```

A direct query can name the directory itself: `--memory-dir DIR` turns
persistence on for that run, in `DIR` (`~` is expanded), whatever the
environment variables say (`AgentOptions.memory_dir` in the Python API).

```bash
melo --memory-dir ~/melo-memory "what is 2+2"
```

See [memory.md](memory.md) for the full operational guide: env vars, file
layout, retention semantics, reset and inspection workflows, real-backend
smoke commands, and the frozen `PersonalizedSample` v1 schema.

## Running tests

### Dev-safe tests (no gateway dependencies)

These tests require only `.[dev]` and cover the core agent loop,
executor, checker, CLI, config, memory, and integration:

```bash
python -m pytest tests/agent tests/checker tests/executor \
                 tests/cli tests/memory tests/integration -q
```

### Gateway tests (require FastAPI + uvicorn)

These tests require `.[dev,gateway]`:

```bash
python -m pytest tests/gateway -q
```

Note: `tests/gateway/test_api.py` uses `pytest.importorskip` for both
`fastapi` and `uvicorn`, so it is safe to collect even when either is
missing -- it will simply skip.

### Full suite

```bash
python -m pytest tests/ -q
```

### Test matrix summary

| Directory              | Install target    | Notes                                |
|------------------------|-------------------|--------------------------------------|
| `tests/agent/`         | `.[dev]`          | Agent loop with fake providers       |
| `tests/backends/`      | `.[dev]`          | Backend adapter contract + implementations |
| `tests/executor/`      | `.[dev]`          | Executor + builtins                  |
| `tests/checker/`       | `.[dev]`          | Checker boundary validators          |
| `tests/cli/`           | `.[dev]`          | CLI entrypoint, config, direct + gateway mode |
| `tests/memory/`        | `.[dev]`          | Memory subsystems + SQLite persistence |
| `tests/integration/`   | `.[dev]`          | Cross-module integration tests       |
| `tests/gateway/`       | `.[dev,gateway]`  | Session management + HTTP API tests  |

## Linting

```bash
python -m ruff check .
```
