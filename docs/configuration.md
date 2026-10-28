# Configuration

This page explains how localmelo connects to your model servers: the first
run, the setup wizard, the config file, the supported backends, and the
settings you can change.

## First run

In a terminal with no config file, `melo` asks two questions, saves the
answers and opens the interactive UI:

```text
$ melo
No model server answered on this machine (ports 8000, 8400, 11434, 30000).
Model server URL [http://127.0.0.1:8000/v1]: http://127.0.0.1:8080/v1
Models on http://127.0.0.1:8080/v1:
  [1] nomic-embed-text
  [2] Qwen3-8B-4bit
  [3] Qwen3-32B-4bit
Model [2]:
✓ http://127.0.0.1:8080/v1 lists 3 models; using Qwen3-8B-4bit.
Saved to ~/.cache/localmelo/config.toml (melo --reconfigure changes it).
```

- A server on this computer that lists a model on port 8000, 8400, 11434 or
  30000 fills in the URL; one with a chat model comes first. Listing a
  server's models sends `GET /v1/models` only, and saving the choice asks
  `GET /api/version` once, to tell an Ollama server apart: no chat request.
- It saves an Ollama server (one that answers `GET /api/version`) as the
  `ollama` backend, and any other OpenAI-compatible server as `vllm`, with
  `embedding_backend = "none"`, and keeps the file's other settings:

  ```toml
  # an Ollama server
  chat_backend = "ollama"
  embedding_backend = "none"

  [ollama]
  chat_url = "http://127.0.0.1:11434/v1"
  chat_model = "qwen3:8b"
  ```

  ```toml
  # any other OpenAI-compatible server
  chat_backend = "vllm"
  embedding_backend = "none"

  [vllm]
  chat_url = "http://127.0.0.1:8080/v1"
  chat_model = "Qwen3-8B-4bit"
  ```

  The `ollama` backend chats on Ollama's native API with a 300 s request
  timeout; `vllm` uses 120 s. A `chat_url` with or without `/v1` works for
  Ollama: its version endpoint is asked at the root, and chat requests drop
  a trailing `/v1`.

- A server that does not answer, refuses the request without an API key,
  or lists no models is named with the reason, and you can retry, start
  without saving or quit. localmelo never asks for an API key for a local
  server.
- The [command-line reference](cli.md#first-run) has every prompt and
  reason.

For embeddings, a cloud API or another backend, use the wizard.

## Set up with the wizard

```bash
melo --reconfigure          # the wizard, then the interactive UI
melo --serve --reconfigure  # the wizard, then the gateway
```

The wizard asks for a chat backend, its URL and model, and an embedding
backend; [Save a configuration](quickstart.md#save-a-configuration) lists the
prompts. It sends a short test request to each server, saves the config
file, and then opens the interactive UI or, with `--serve`, starts the
gateway. Press Ctrl+C to stop the gateway.

- Each question offers the config file's current value in brackets: press
  Enter to keep it. The embedding question offers `None` when no embedding
  backend is set.
- When the chat server's check fails, the wizard prints `Probe failed: …`
  and asks `[r]etry, [s]ave anyway, [N]o [N]:`. `r` asks for the URL and
  model again, with what you typed as the defaults; `s` saves anyway;
  Enter or `N` saves nothing.
- An HTTP 401 or 403 from a local server adds a note: localmelo sends no
  API key to local servers. In oMLX, turn on
  **Allow inference without API key**.
- When the embedding server's check fails, the wizard saves with embedding
  off (`embedding_backend = "none"`, no long-term memory) and says so.
- `melo --serve` runs the wizard by itself when there is no config file.
- Without a terminal the wizard reads its answers from stdin. With no
  config file and no terminal, `melo --serve` exits with code 1 and one
  line on stderr instead of waiting for answers that cannot come, as under
  launchd.
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
| `reasoning_field` | local or cloud backend | not set | The request key reasoning is sent back under: `auto`, `reasoning_content` or `reasoning`; `[ollama]` also takes `thinking`. See [Reasoning](#reasoning). |
| `requires_reasoning_replay` | local or cloud backend | `false` | `true` when the server rejects a request that omits a reply's reasoning. See [Reasoning](#reasoning). |
| `reasoning` | `[agent]` | `"legacy"` | `legacy`, `drop` or `replay`; the default is direct mode's `--reasoning`. See [Reasoning](#reasoning). |
| `replay_turns` | `[agent]` | not set | With `replay`, the newest N assistant turns whose reasoning is sent; omit for every turn in scope. |
| `api_key_env` | cloud backend | not set | The name of the environment variable that holds the API key. |
| `base_url` | cloud backend | the vendor's URL | The API's URL. |
| `host` | `[gateway]` | `"127.0.0.1"` | The address the gateway listens on: keep `127.0.0.1`, or use `localhost` or `::1`. On a host that is not `localhost` or a loopback IP address, such as `0.0.0.0`, the gateway does not check the `Host` and `Origin` of requests; see [Keep the gateway private](gateway.md#keep-the-gateway-private). |
| `port` | `[gateway]` | `8401` | The port the gateway listens on. |

Top-level keys must come before the first `[table]` line. How URLs are used:

- `mlc`, `vllm` and `sglang`: `/v1` is added to `chat_url` and
  `embedding_url` when it is missing.
- `ollama`: chat uses Ollama's own `/api/chat`, so `chat_url` is used without
  `/v1` (a trailing `/v1` is removed). `embedding_url` gets `/v1` added.
- Cloud backends: `base_url` is used as given.

### Reasoning

`legacy` is the default: an OpenAI-compatible backend sends no reasoning,
and Ollama native sends its own `thinking` field and inline copy as before.
`drop` never sends reasoning; `replay` sends it on the assistant turns in
scope, under one key, at most 40,000 characters per turn (a longer turn's
reasoning is skipped and counted). Both explicit modes also stop reading
`[RECALL]` and remove an inline `<think>` block from the answer.

```toml
[agent]
reasoning = "replay"      # legacy, drop, replay
replay_turns = 3          # optional; omit for every turn
```

A direct-mode flag wins over `[agent]`, and a `--base-url` run uses only the
flags (it does not read this file). The interactive UI and the gateway do
not read `[agent]` in this release.

Each backend section can name the request key and say that the server
requires replay:

```toml
[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "MODEL"
reasoning_field = "auto"              # or reasoning_content, reasoning
requires_reasoning_replay = false     # true for a server that rejects a
                                      # request without the reasoning
```

`auto` follows the key the last reply's reasoning came from, so a backend
that emits either `reasoning_content` or `reasoning` works as it is; it
sends no key before the first reply. The `[ollama]` backend fixes its field
to `thinking` (it may still write `"thinking"` or `"auto"`). With
`requires_reasoning_replay = true`, `--reasoning drop` is refused and the
agent evicts whole working-memory units instead of cutting reasoning.

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

### oMLX

[oMLX](https://github.com/jundot/omlx) has an OpenAI-compatible API, so it
uses the `vllm` backend. [Serve a model with oMLX](quickstart.md#serve-a-model-with-omlx)
shows how to start it.

```toml
chat_backend = "vllm"
embedding_backend = "none"
request_timeout_s = 600

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B-4bit"

[vllm.extra_body.chat_template_kwargs]
enable_thinking = false
```

- `chat_model` is the model's id in oMLX's `GET /v1/models`: its folder
  name, or the alias set for it in oMLX.
- [First run](#first-run) writes the `chat_backend`, `embedding_backend`,
  `chat_url` and `chat_model` lines: run `melo` in a terminal with no
  config file, while oMLX runs. Add the other lines by hand.
- `request_timeout_s = 600` gives oMLX time to load the model when a
  request finds it unloaded; see [Chat request timeout](#chat-request-timeout).
- The `extra_body` table turns off a Qwen3 model's thinking, which makes
  each step faster. Leave it out to keep thinking on. oMLX applies it unless
  the model's settings in oMLX force `enable_thinking`.
- For long-term memory, oMLX can also serve an embedding model from the
  same directory: set `embedding_backend = "vllm"` and add
  `embedding_url = "http://127.0.0.1:8000/v1"` and
  `embedding_model = "<its folder name>"` to `[vllm]`. Keep `embedding` in
  the folder name of a Qwen3-Embedding model, as in its repository name:
  oMLX tells it from a chat model by that name.

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
| `LOCALMELO_DETECT_URLS` | The URLs that the [first run](#first-run) asks for a model list, comma-separated, in place of ports 8000, 8400, 11434 and 30000 of `127.0.0.1`. An empty value turns the search off. |
| `LOCALMELO_THINKING_RETENTION_DAYS` | The days that `history.db` and the session store's `sessions.db` keep the model's reasoning (default 30): any number above 0, such as `7` or `0.5`. Any other value stops `melo` with exit code 2. See [Model calls](memory.md#model-calls). |
| The variable that `api_key_env` names | The API key of a cloud backend, such as `OPENAI_API_KEY`. |

See [Keep memory between runs](memory.md#keep-memory-between-runs).

## Files and directories

| Path | What it holds | Created |
|---|---|---|
| `~/.cache/localmelo/config.toml` | The configuration. | By the setup wizard. |
| `~/.cache/localmelo/memory/` | `history.db`, `procedures.db` (procedural memory, written at the end of every run), and `long_term.db` with an embedding backend. | When an agent starts with `LOCALMELO_PERSIST_MEMORY` set, with mode 0700. An existing one that is open to group or others is set to 0700. |
| The `--memory-dir` directory | The same files, for that run. | When the run starts, with mode 0700 when it is new. |
| `~/.cache/localmelo/playground/` | The web workspace's memory. | When the gateway starts. |
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
