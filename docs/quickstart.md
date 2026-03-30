# Quickstart

This guide covers the minimum needed to run localmelo, verify the
installation, and run the test suite.

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
