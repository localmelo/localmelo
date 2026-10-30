# Guide: Configure

This guide covers the first run, the setup wizard, and how to update or uninstall localmelo. Read it when you set localmelo up on a machine or change its configuration; the settings are in the [configuration reference](../reference/configuration.md).

## First run

In a terminal with no config file, `melo` asks for a model server's URL and
a chat model, saves the answers and opens the interactive UI. When the
server also lists an embedding model, it asks once whether to use it. Here
oMLX runs on port 8000 with three models, and the first chat model is loaded:

```text
$ melo
melo uses a model server on this computer (oMLX, Ollama, LM Studio…).
Found a model server at http://127.0.0.1:8000/v1 (3 models).
Server URL [http://127.0.0.1:8000/v1]:
 Choose a model · http://127.0.0.1:8000/v1
 Filter ›
 ❯ Qwen3-8B-4bit  (loaded)
   Qwen3-32B-4bit
 2 models · enter chooses · esc cancels
✓ http://127.0.0.1:8000/v1 lists 3 models; using Qwen3-8B-4bit.
Use Qwen3-Embedding-0.6B for long-term memory? [Y/n]:
Saved to ~/.cache/localmelo/config.toml (melo setup changes it).
```

- A server on this computer that lists a model on port 8000, 8400, 11434 or
  30000 fills in the URL; one with a chat model comes first. Listing a
  server's models sends `GET /v1/models`. On a terminal the models are a list
  to choose from (type to filter, ↑ ↓ to move, Enter to choose, Esc to
  cancel; the list is erased once you choose) with the chat models only, as
  above. Without a terminal, when answers are piped in, they are numbered
  (`Model [1]:`), chat models first; an embedding or reranking model is
  tagged `(embedding)` or `(reranking)` and is never the default. With
  several chat models, setup
  also asks which are in memory, with `GET /v1/models/status` (oMLX) and
  `GET /api/ps` (Ollama), one second each and without a key: it marks the
  loaded ones `(loaded)` and makes the first the default (highlighted), and
  with none loaded there is no default: nothing is highlighted, so move to a
  model with ↑ ↓ (or type a number, piped). Saving the choice asks
  `GET /api/version` once, to tell an Ollama server apart, and a `y` sends
  one `POST /v1/embeddings` probe of the offered model.
- The embedding question is asked only when the list holds an embedding
  model other than the chat model. `y` saves three embedding values: the
  chat backend's key, its URL and the offered model. A failed probe, `n`, or
  no candidate saves `embedding_backend = "none"` as before.
- It saves an Ollama server (one that answers `GET /api/version`) as the
  `ollama` backend, an oMLX server (its model listing names oMLX as the owner
  of every model) as `omlx`, and any other OpenAI-compatible server as
  `vllm`, and keeps the file's other settings. `omlx` is another name for the
  `vllm` backend, so the settings stay in the `[vllm]` table. With no
  embedding it writes:

  ```toml
  # an Ollama server
  chat_backend = "ollama"
  embedding_backend = "none"

  [ollama]
  chat_url = "http://127.0.0.1:11434/v1"
  chat_model = "qwen3:8b"
  ```

  ```toml
  # an oMLX server (another OpenAI-compatible server: "vllm")
  chat_backend = "omlx"
  embedding_backend = "none"

  [vllm]
  chat_url = "http://127.0.0.1:8000/v1"
  chat_model = "Qwen3-8B-4bit"
  ```

  A `y` writes the chat backend's embedding fields instead of `"none"`:

  ```toml
  chat_backend = "omlx"
  embedding_backend = "omlx"

  [vllm]
  chat_url = "http://127.0.0.1:8000/v1"
  chat_model = "Qwen3-8B-4bit"
  embedding_url = "http://127.0.0.1:8000/v1"
  embedding_model = "Qwen3-Embedding-0.6B"
  ```

  The `ollama` backend chats on Ollama's native API with a 300 s request
  timeout; `vllm` and `omlx` use 120 s. A `chat_url` with or without `/v1` works for
  Ollama: its version endpoint is asked at the root, and chat requests drop
  a trailing `/v1`.

- With no server running, it prints the commands that start one (oMLX on a Mac,
  Ollama elsewhere) and looks again each time you press Enter; the web console's
  setup card shows the same commands and looks again every three seconds.
- A server that does not answer, refuses the request without an API key,
  or lists no models is named with the reason, and you can retry, start
  without saving or quit. localmelo never asks for an API key for a local
  server.
- The [command-line reference](terminal.md#first-run) has every prompt and
  reason.

For an embedding model, see [Set up an embedding model](embedding.md). For a
cloud API or another backend, use the wizard.

## Set up with the wizard

```bash
melo setup                  # the wizard, then the interactive UI
melo --serve --reconfigure  # the wizard, then the gateway
```

The wizard asks for a chat backend, its URL and model, and an embedding
backend; [Save a configuration](../start/quickstart.md#save-a-configuration) lists the
prompts. It sends a short test request to each server, saves the config
file, and then opens the interactive UI or, with `--serve --reconfigure` (the
one setup that has no command-word form), starts the gateway. Press Ctrl+C to
stop the gateway.

In default mode, `melo setup` starts as the first
run does: it prints `Found a model server at …` and asks `Use it? [Y/n]:`. The
full menu follows only after `n`, with `--developer`, or when no server is found
while a backend is saved. With nothing saved and no server running, it prints the
commands that start one instead. `melo serve` before any setup does the same.

- Each question offers the config file's current value in brackets: press
  Enter to keep it. The embedding question offers the saved backend; with
  none saved, the first candidate the chat server lists, else `None`.
- When no embedding is saved, the wizard lists the chat server's models
  once (only for a local chat backend) and appends each embedding model it
  finds as options `6`, `7`, … after the fixed five. The first candidate is
  the default; choosing one saves the chat backend's URL and model and asks
  no endpoint question. The fixed `1`–`4` options still ask for another
  server's URL and model, and `5` is `None`. A server whose list cannot be
  read falls back to those five silently; the end-of-wizard check reports a
  dead server.
- When the chat server's check fails, the wizard prints `Probe failed: …`
  and asks `[r]etry, [s]ave anyway, [N]o [N]:`. `r` asks for the URL and
  model again, with what you typed as the defaults; `s` saves anyway;
  Enter or `N` saves nothing.
- An HTTP 401 or 403 from a local server adds a note: localmelo sends no
  API key to local servers. In oMLX, turn on
  **Allow inference without API key**.
- When the embedding server's check fails, the wizard saves with embedding
  off (`embedding_backend = "none"`, no long-term memory) and says so. A
  detected model that fails its probe is not left in the config's table.
- `melo serve` runs the wizard by itself when there is no config file.
- Without a terminal the wizard reads its answers from stdin. With no
  config file and no terminal, `melo serve` exits with code 1 and one
  line on stderr instead of waiting for answers that cannot come, as under
  launchd. Pass `--force` with `melo serve` to start the
  gateway anyway and configure it over `/api/settings`.
- Run the wizard again to change backends. It keeps the other settings it
  knows, such as `request_timeout_s` and `extra_body`, and rewrites the file
  atomically in the current format (following a symlink when `config.toml`
  is one, keeping an existing file's permissions, and creating a new file
  with mode `0600`); comments and unknown keys are dropped.

## Update localmelo

In your clone, pull the latest code and update the environment:

```bash
git pull
uv sync --extra gateway
```

If you installed with pip, repeat the last command of
[Install with pip](../start/quickstart.md#install-with-pip) instead of `uv sync`.

## Uninstall localmelo

1. If you installed the background service, run `melo serve uninstall`.
2. Delete the clone.
3. Delete `~/.cache/localmelo` (your configuration and saved memory),
   `~/.localmelo` (the background service's logs), and any directory you
   used with `LOCALMELO_MEMORY_DIR` or `--memory-dir`.
