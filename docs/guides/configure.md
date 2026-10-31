# Guide: Configure

Set localmelo up on a machine, change its configuration, update it or remove it. The settings are in the [configuration reference](../reference/configuration.md).

## First run

In a terminal with no config file, `melo` offers the model server it finds, asks for a chat model, saves the answers and opens the chat. Here oMLX runs on port 8000 with three models, and the first chat model is loaded:

```text
$ melo
melo uses a model server on this computer (oMLX, Ollama, LM Studio…).
Found a model server at http://127.0.0.1:8000/v1 (3 models).
 Choose a model · http://127.0.0.1:8000/v1
 Filter ›
 ❯ Qwen3-8B-4bit  (loaded)
   Qwen3-32B-4bit
 2 models · enter chooses · esc cancels
✓ http://127.0.0.1:8000/v1 lists 3 models; using Qwen3-8B-4bit.
Saved to ~/.cache/localmelo/config.toml (melo setup changes it).
```

- melo looks on this computer at ports 8000, 8400, 11434 and 30000 (`GET /v1/models`) and offers the first server that lists a model, a chat model first.
- Every question with a fixed set of answers is a list: ↑ ↓ or a number move, Enter confirms, Esc takes the safe answer (see [Questions are lists](terminal.md#questions-are-lists)).
- The model list shows chat models only, and marks the ones the server says are in memory `(loaded)`: setup asks `GET /v1/models/status` (oMLX) or `GET /api/ps` (Ollama), one second each and without a key.
- The first loaded model is the default; with none loaded there is no default, so move to a model with ↑ ↓ (or type a number, piped).
- Piped answers are numbered instead (`Model [1]:`); an embedding or reranking model is tagged and is never the default.
- No chat request is sent, so the server does not load a model for the check.

If the server also lists an embedding model, one more question asks `Remember across chats with <model>?`:

- `Yes, with it` sends one `POST /v1/embeddings` probe and saves the chat backend's key, its URL and that model as the embedding backend.
- `No, by words only` (or Esc), a failed probe or no candidate saves `embedding_backend = "none"`.

What it saves, with no embedding model:

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

`Yes, with it` writes the chat backend's embedding fields instead of `"none"`:

```toml
chat_backend = "omlx"
embedding_backend = "omlx"

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B-4bit"
embedding_url = "http://127.0.0.1:8000/v1"
embedding_model = "Qwen3-Embedding-0.6B"
```

- A server that answers `GET /api/version` is saved as `ollama`, one whose model list names oMLX as `omlx`, any other as `vllm`; `omlx` is another name for `vllm`, so its settings stay in `[vllm]`.
- `ollama` chats on Ollama's native API with a 300 s request timeout, `vllm` and `omlx` with 120 s; Ollama's `chat_url` works with or without `/v1`.
- The file's other settings are kept.

When no server is ready:

- With none running, supported Apple-silicon Macs show the terminal's [read-only setup preview](model-servers.md#setup-preview): the recommendation, pinned download sizes, cache/disk readings and lifetime. `Set up and start` currently shows an unavailable notice and manual steps, never an install. The list also lets you check again, enter an address or quit; pipes retain their existing typed answers. Other platforms and the web/app setup card keep manual-server guidance.
- A server that does not answer, needs an API key or lists no models is named with the reason, and you can try again, start without saving or quit.
- localmelo never asks for an API key for a local server.
- [The terminal guide](terminal.md#first-run) lists every prompt and reason.

For an embedding model, see [Set up an embedding model](embedding.md). For a cloud API or another backend, use the wizard.

## Set up with the wizard

```bash
melo setup                  # the wizard, then the interactive UI
melo --serve --reconfigure  # the wizard, then the gateway
```

- The wizard asks for a chat backend, its URL and model, and an embedding backend; [Save a configuration](../start/quickstart.md#save-a-configuration) lists the prompts.
- It sends a short test request to each server, saves the config file, then opens the chat or, with `--serve --reconfigure`, starts the gateway (Ctrl+C stops it).
- In default mode `melo setup` starts as the first run does (`Found a model server at …`, `Use this model server?`); the full menu follows `Use another server` or developer mode. With none found, the read-only setup preview leaves any saved connection unchanged.
- In developer mode (`[ui] mode = "developer"`; `melo config set ui.mode developer` saves it) the full menu comes first.
- Each question offers the file's current value in brackets: press Enter to keep it.

When something fails:

- A failed check of the chat server prints `Probe failed: …` and asks `The check failed. What now?`: `Try again`, `Save anyway` or `Cancel setup` (selected first; Esc also saves nothing). Piped, the question is `[r]etry, [s]ave anyway, [N]o [N]:`.
- An HTTP 401 or 403 from a local server adds a note that localmelo sends no API key to local servers; in oMLX, turn on **Allow inference without API key**.
- A failed check of the embedding server saves `embedding_backend = "none"` (no long-term memory) and says so.

More about the wizard:

- `melo serve` runs the wizard by itself when there is no config file; with no terminal either it exits with code 1 and one line on stderr, and `--force` starts the gateway anyway, to be configured over `/api/settings`.
- Run the wizard again to change backends: it keeps settings it knows, such as `request_timeout_s` and `extra_body`, and rewrites the file atomically with mode `0600` for a new file; comments and unknown keys are dropped.
- When more embedding models are found, the wizard appends them as options `6`, `7`, … after the fixed five; the first is the default and asks no endpoint question.

## Update localmelo

In your clone, pull the latest code and update the environment:

```bash
git pull
uv sync --extra gateway
```

If you installed with pip, repeat the last command of [Install with pip](../start/quickstart.md#install-with-pip) instead of `uv sync`.

## Uninstall localmelo

1. If you installed the background service, run `melo serve uninstall`.
2. Delete the clone.
3. Delete `~/.cache/localmelo` (your configuration and saved memory), `~/.localmelo` (the background service's logs), and any directory you used with `LOCALMELO_MEMORY_DIR` or `--memory-dir`.
