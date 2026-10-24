# Configuration

This page explains how localmelo connects to your model servers: the setup
wizard, the config file, the supported backends, and the settings you can
change.

## Set up with the wizard

```bash
melo --serve --reconfigure
```

The wizard asks for a chat backend, its URL and model, and an embedding
backend; [Save a configuration](quickstart.md#save-a-configuration) lists the
prompts. It sends a short test request to each server, saves the config
file, and then starts the gateway. Press Ctrl+C to stop the gateway.

- `melo --reconfigure` saves the configuration without starting the gateway.
- `melo --serve` runs the wizard by itself when there is no config file.
- Run the wizard again to change backends. It keeps the other settings it
  knows, such as `request_timeout_s` and `extra_body`, and rewrites the file
  in the current format; comments and unknown keys are dropped.

## The config file

The configuration lives in `~/.cache/localmelo/config.toml`. A minimal file
for Ollama without embeddings:

```toml
chat_backend = "ollama"
embedding_backend = "none"

[ollama]
chat_url = "http://127.0.0.1:11434"
chat_model = "qwen3:8b"
```

> [!WARNING]
> Tools that clean caches may delete `~/.cache`, which also holds your saved
> memory. Keep a copy of a config file you edited by hand.

The wizard writes a table for every backend, but localmelo reads only the
tables of the chosen backends.

| Key | Table | Default | Description |
|---|---|---|---|
| `chat_backend` | top level | not set | The chat backend; see [Backends](#backends). Required. |
| `embedding_backend` | top level | not set | `mlc`, `ollama`, `vllm`, `sglang` or `none`. Required. |
| `request_timeout_s` | top level | not set | Chat request timeout in seconds; see [Chat request timeout](#chat-request-timeout). |
| `chat_url` | local backend | not set | The model server's URL. |
| `chat_model` | local or cloud backend | not set | The model to use. |
| `embedding_url` | local backend | not set | The embedding server's URL. |
| `embedding_model` | local backend | not set | The embedding model to use. |
| `extra_body` | local or cloud backend | not set | Extra fields for every chat request; see [Extra request fields](#extra-request-fields-extra_body). |
| `api_key_env` | cloud backend | not set | The name of the environment variable that holds the API key. |
| `base_url` | cloud backend | the vendor's URL | The API's URL. |
| `host` | `[gateway]` | `"127.0.0.1"` | The address the gateway listens on. |
| `port` | `[gateway]` | `8401` | The port the gateway listens on. |

Top-level keys must come before the first `[table]` line. How URLs are used:

- `mlc`, `vllm` and `sglang`: `/v1` is added to `chat_url` and
  `embedding_url` when it is missing.
- `ollama`: chat uses Ollama's own `/api/chat`, so `chat_url` is used without
  `/v1` (a trailing `/v1` is removed). `embedding_url` gets `/v1` added.
- Cloud backends: `base_url` is used as given.

## Backends

| Key | Type | Server or API | Default URL | Embeddings |
|---|---|---|---|---|
| `ollama` | local | [Ollama](https://ollama.com) | `http://localhost:11434` (wizard) | yes |
| `mlc` | local | [MLC LLM](https://llm.mlc.ai) | `http://localhost:8400/v1` (wizard) | yes |
| `vllm` | local | [vLLM](https://docs.vllm.ai) | `http://localhost:8000/v1` (wizard) | yes |
| `sglang` | local | [SGLang](https://docs.sglang.ai) | `http://localhost:30000/v1` (wizard) | yes |
| `openai` | cloud | OpenAI API | `https://api.openai.com/v1` | no |
| `anthropic` | cloud | Anthropic API | `https://api.anthropic.com/v1` | no |
| `gemini` | cloud | Google Gemini API | `https://generativelanguage.googleapis.com/v1beta/openai` | no |
| `nvidia` | cloud | NVIDIA API | `https://integrate.api.nvidia.com/v1` | no |

The wizard suggests the local URLs; use your server's address and port.
localmelo only connects to a local server: you start and stop it yourself.
`mlc`, `vllm` and `sglang` use the plain OpenAI-compatible API, so another
OpenAI-compatible server works with any of them.

### Ollama with embeddings

```toml
chat_backend = "ollama"
embedding_backend = "ollama"

[ollama]
chat_url = "http://127.0.0.1:11434"
chat_model = "qwen3:8b"
embedding_url = "http://127.0.0.1:11434"
embedding_model = "nomic-embed-text"
```

Pull both models first: `ollama pull qwen3:8b` and
`ollama pull nomic-embed-text`.

### MLC LLM, vLLM or SGLang

```toml
chat_backend = "vllm"
embedding_backend = "none"

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen/Qwen3-8B"
```

- For MLC LLM or SGLang, use the table `[mlc]` or `[sglang]` and the matching
  `chat_backend`.
- Set `chat_model` to a model that your server serves. The server's
  `GET /v1/models` lists them.
- To get embeddings from the same kind of server, set
  `embedding_backend = "vllm"` (or `mlc`, `sglang`) and add `embedding_url`
  and `embedding_model` to the table.

### A cloud API

```toml
chat_backend = "openai"
embedding_backend = "none"

[openai]
api_key_env = "OPENAI_API_KEY"
chat_model = "gpt-4o"
```

Export the key in the shell that runs `melo`:

```bash
export OPENAI_API_KEY="<your key>"
melo "What is 6*7?"
```

For another vendor, use the table `[anthropic]`, `[gemini]` or `[nvidia]`.
The wizard suggests the variables `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` and
`NVIDIA_API_KEY`.

### Mix backends

Chat and embedding backends are chosen separately. Any chat backend works
with an embedding backend of `mlc`, `ollama`, `vllm`, `sglang` or `none`.
For example, a cloud model with local Ollama embeddings:

```toml
chat_backend = "openai"
embedding_backend = "ollama"

[openai]
api_key_env = "OPENAI_API_KEY"
chat_model = "gpt-4o"

[ollama]
embedding_url = "http://127.0.0.1:11434"
embedding_model = "nomic-embed-text"
```

## Embedding and long-term memory

Long-term memory needs an embedding backend. With
`embedding_backend = "none"`:

- nothing is recalled from long-term memory, and nothing is written to it;
- reflections are not promoted to long-term memory;
- the run loop, tools, working memory and history work as usual.

`GET /v1/health` on the gateway then shows `"embedding_backend": "none"`.
See [Memory](memory.md).

## Cloud API keys

- The config file stores the *name* of the environment variable
  (`api_key_env`), never the key.
- Export the variable in the shell that runs `melo`.
- If it is not set, `melo` stops before the run with exit code 1:

  ```text
    Configuration errors:
    - [openai] environment variable $OPENAI_API_KEY is not set. Export it before starting the gateway.
  ```

- `--base-url` sends no API key, so use a config file for cloud APIs.
- The macOS background service does not get your shell's variables; see
  [Run the gateway at login](gateway.md#run-the-gateway-at-login-macos).

## Extra request fields (extra_body)

Each backend table accepts an `extra_body` table. Its fields are added to
every chat request. Use it for server options such as `chat_template_kwargs`:

```toml
[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen/Qwen3-8B"

[vllm.extra_body.chat_template_kwargs]
enable_thinking = false
```

- The inline form works too:
  `extra_body = { chat_template_kwargs = { enable_thinking = false } }`.
- A flat `chat_template_kwargs = { … }` key in the table is read as part of
  `extra_body`; the wizard writes it back inside `extra_body`.
- localmelo sets `model`, `messages`, `tools`, `tool_choice` and `stream`
  itself. Setting one of them in `extra_body` is a configuration error:

  ```text
    - [vllm] extra_body must not set 'model'; localmelo manages that field
  ```

- For the `ollama` backend, the fields go into the body of Ollama's
  `/api/chat`. localmelo asks Ollama for the model's thinking by default;
  `extra_body = { think = false }` turns that off.
- `--base-url` does not read the config file, so `extra_body` does not apply
  there.

## Chat request timeout

A chat request waits up to 120 seconds for the server (300 seconds with the
`ollama` backend). To change this for every run, set `request_timeout_s` at
the top of the config file, before the first table:

```toml
chat_backend = "vllm"
embedding_backend = "none"
request_timeout_s = 600

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen/Qwen3-8B"
```

- Below a `[table]` line, the key belongs to that table, and localmelo does
  not read it there.
- The value is a number of seconds greater than 0, such as `600` or `90.5`.
  Anything else stops `melo` with exit code 1, for example
  `request_timeout_s must be a finite number of seconds > 0, got '600'` for
  a quoted value.
- `--request-timeout SECONDS` overrides it for one direct run and does not
  change the file.
- It applies to chat requests only. Embedding requests always wait up to
  120 seconds.
- The timer restarts whenever the server sends data, so a server that sends
  keep-alive bytes while it generates never times out.
- A request that times out is retried twice, each time with the full
  timeout; see [Provider errors and retries](run-results.md#provider-errors-and-retries).
- To limit a whole run, use `--timeout`. See
  [Timeouts](how-it-works.md#timeouts).

## Environment variables

| Variable | Effect |
|---|---|
| `LOCALMELO_PERSIST_MEMORY` | Any non-empty value, even `0`, saves history and long-term memory in SQLite files. |
| `LOCALMELO_MEMORY_DIR` | The directory for those files (default `~/.cache/localmelo/memory`). It is used exactly as given: `~` is not expanded. |
| The variable that `api_key_env` names | The API key of a cloud backend, such as `OPENAI_API_KEY`. |

See [Keep memory between runs](memory.md#keep-memory-between-runs).

## Files and directories

| Path | What it holds | Created |
|---|---|---|
| `~/.cache/localmelo/config.toml` | The configuration. | By the setup wizard. |
| `~/.cache/localmelo/memory/` | `history.db`, and `long_term.db` with an embedding backend. | When an agent starts with `LOCALMELO_PERSIST_MEMORY` set. |
| The `--memory-dir` directory | The same files, for that run. | When the run starts. |
| `~/.cache/localmelo/playground/` | The web UI's memory. | When the gateway starts. |
| `~/.localmelo/logs/` | The background service's logs. | By `melo --daemon install`. |
| `~/Library/LaunchAgents/ai.localmelo.gateway.plist` | The background service's definition. | By `melo --daemon install`. |

## Update localmelo

In your clone, pull the latest code and update the environment:

```bash
git pull
uv sync --extra gateway
```

If you installed with pip, repeat the last command of
[Install with pip](quickstart.md#install-with-pip) instead of `uv sync`.

## Upgrading from an older config

Older config files still load. localmelo reads:

- a single `backend` key, the oldest format;
- the backend name `mlc-llm`, now `mlc`;
- `[cloud_api]` and `[online]` tables, now one table per vendor;
- `chat_port` in `[mlc]`, now `chat_url`;
- a flat `chat_template_kwargs`, now inside `extra_body`.

Running the wizard again writes the file in the current format.

## Uninstall localmelo

1. If you installed the background service, run `melo --daemon uninstall`.
2. Delete the clone.
3. Delete `~/.cache/localmelo` (your configuration and saved memory),
   `~/.localmelo` (the background service's logs), and any directory you
   used with `LOCALMELO_MEMORY_DIR` or `--memory-dir`.
