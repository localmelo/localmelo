# Quickstart

This guide covers the minimum needed to run localmelo, verify the
installation, and run the test suite.

Sections:

- [Installation](#installation)
- [Usage modes](#usage-modes)
- [Run results](#run-results) — statuses and the `RunResult` record of a run
- [Minimum smoke](#minimum-smoke) — verify each mode end-to-end after install
- [Backend options](#backend-options)
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

Run a one-shot query without starting a server. No configuration file is
needed -- localmelo defaults to a local Ollama backend at
`http://127.0.0.1:11434/v1`.

```bash
melo "what is 2+2"
```

Override the backend URL and model on the fly:

```bash
melo --base-url http://localhost:11434/v1 --chat-model qwen3:8b "what is 2+2"
```

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
`Agent.run_task(query)` returns a [`RunResult`](#run-results).
`Agent.run(query)` returns the run's outcome as text, which is what the
gateway and its web pages show: the result when it is non-empty, otherwise
`[<status>] <error>`, for example `[budget_exhausted] Max steps reached`.
One agent runs one task at a time.

## Run results

Every run ends with one of these statuses:

| Status | Meaning |
|---|---|
| `completed` | The model gave a final answer. |
| `failed` | A checker refused the plan or the run (the prompt outgrew the pre-plan cap, tool resolution failed, or the model's response failed its check), or the reflection gate decided not to continue; `result` then holds the reflection's best effort, if any. |
| `budget_exhausted` | `max_steps` or `max_attempts` ran out before a final answer. |
| `stuck` | Stuck detection fired (the same tool calls, or the same errors, three steps in a row) with `attempt_policy="none"`. With `"reflect"`, a reflection decides what happens next instead. |
| `provider_error` | A request to the model server failed: an HTTP error status, a timeout, a connection failure or a malformed response body (see [Provider errors](#provider-errors)). `error` describes the failed request; no traceback is logged. |
| `timeout` | The run was still going after `AgentOptions.timeout_s` seconds and was cancelled where it stood: an LLM request in flight is abandoned and counts in `llm_errors`, and a `shell_exec` command is killed together with every process it started. `error` is `Timed out after <timeout_s> s`. |
| `crashed` | An unexpected exception ended the run. `error` is `<ExceptionType>: <message>` and the traceback is logged. A run cancelled from outside is recorded as `crashed` too, and the cancellation propagates. |

`Agent.run_task()` returns the run as a `RunResult`; `RunResult.to_dict()`
gives the same fields as a plain dict, in this order:

| Field | Meaning |
|---|---|
| `task_id` | id of the run's task record in history |
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
`timeout_s` (or `null`) and `tools` (sorted names of every registered tool),
so every record says which mode it ran in.

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

Every chat request carries its attempt number (1, then 2 and 3 for the
retries) in the `X-Localmelo-Attempt` HTTP header, so a proxy in front of
the model server can match its request log against a run's `n_llm` and
`llm_errors`. A custom `BaseLLMProvider` gets the same retries by raising
`ProviderError(message, status_code=..., retryable=True)` for a failure
worth retrying.

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
