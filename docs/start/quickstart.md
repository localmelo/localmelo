# Quickstart

This tutorial installs localmelo, runs a first task against a local model
server, saves a configuration and starts the gateway.

## Before you begin

You need:

- macOS (developed and tested) or Linux (should work, untested). Windows is
  not supported.
- Python 3.11 or later, Git, and [uv](https://docs.astral.sh/uv/) (or pip,
  see [Install with pip](#install-with-pip)).
- A model server. This tutorial uses [Ollama](https://ollama.com): install
  it, start it, and pull a model with `ollama pull qwen3:8b`. Any server
  with an OpenAI-compatible API works too: use its `/v1` URL and its model
  name in the commands below. On a Mac with Apple silicon, you can
  [serve a model with oMLX](../guides/model-servers.md#serve-a-model-with-omlx) instead.

## Install

```bash
git clone https://github.com/localmelo/localmelo.git
cd localmelo
uv sync --extra gateway
source .venv/bin/activate
```

uv creates the virtual environment `.venv/` (installing Python 3.13 if you
do not have it) and installs localmelo with the gateway's dependencies.

Keep the directory name `localmelo`: the package is built from it. A renamed
folder or a ZIP download (`localmelo-main`) fails with
`ModuleNotFoundError: No module named 'localmelo'`.

The rest of this guide calls `melo` directly. Activation lasts for this
terminal only: in a new terminal, run `source <clone>/.venv/bin/activate`
again, where `<clone>` is your `localmelo` directory. With uv you can also
run `uv run melo …` inside the clone, or `uv run --project <clone> melo …`
from anywhere else.

### Install with pip

```bash
git clone https://github.com/localmelo/localmelo.git
cd localmelo
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[gateway]"
```

`python3` must be Python 3.11 or later. The `-e` flag keeps the installation
linked to this checkout, so source changes apply without reinstalling.

## Run your first task

> [!WARNING]
> The model can run shell commands and Python code, and read or write any
> file you can: there is no sandbox. To run a task without tools, add
> `--builtins none`. See the
> [security policy](https://github.com/localmelo/localmelo/blob/main/SECURITY.md)
> and [What the checks do](../guides/tools-mcp.md#what-the-checks-do).

```bash
melo --base-url http://127.0.0.1:11434/v1 --chat-model qwen3:8b "What is 6*7?"
```

The answer is printed on stdout, and the command exits with code 0
(`echo $?` prints `0`). `--base-url` points localmelo at an OpenAI-compatible
server for this one run, without a configuration file. It sends no API key,
so use it for local servers; for a cloud API, save a configuration (below).

For every option, see the [command-line reference](../reference/cli.md).

## Get the result as JSON

```bash
melo --base-url http://127.0.0.1:11434/v1 --chat-model qwen3:8b --output-format json "What is 6*7?"
```

Example output (values vary):

```json
{"task_id": "f52e785a84a144ff8c3b8e70e0451eb1", "status": "completed", "result": "42", "error": "", "steps": 1, "attempts": 1, "n_llm": 1, "n_main": 1, "n_aux": 0, "n_tool": 0, "llm_errors": 0, "usage": {"prompt_tokens": 120, "completion_tokens": 12, "total_tokens": 132}, "duration_s": 1.9, "config": {"max_steps": 30, "attempt_policy": "reflect", "steps_per_attempt": 10, "max_attempts": 5, "reflection_gate": "flags", "tool_exposure": "retrieve", "mode": "direct", "builtins": ["file_read", "file_write", "python_exec"], "memory": true, "memory_dir": null, "timeout_s": null, "tools": ["file_read", "file_write", "python_exec"]}, "localmelo_version": "603f40d"}
```

stdout holds exactly this one line. The exit code is 0 only when `status`
is `completed`.

### Run results

`status` says how the run ended, for example `completed`, `budget_exhausted`
or `provider_error`. The counters say how many steps, model requests and tool
calls the run took, and `config` records the options it ran with. See
[Run results](../reference/run-results.md) for every status and field.

## Save a configuration

A configuration file saves the backends, so that `melo`, `melo "<task>"`
and the gateway work without `--base-url`.

For a local server, run `melo` in a terminal. With no configuration yet, it
looks for a model server on this computer, asks for its URL (press Enter to
accept the one it found) and for a model from the server's list, saves
`~/.cache/localmelo/config.toml` and opens the interactive UI:

```text
$ melo
Found a model server at http://127.0.0.1:11434/v1 (1 model).
Model server URL [http://127.0.0.1:11434/v1]:
✓ http://127.0.0.1:11434/v1 lists 1 model; using qwen3:8b.
No embedding model on this server. ollama pull qwen3-embedding:0.6b, then melo --reconfigure.
Saved to ~/.cache/localmelo/config.toml (melo --reconfigure changes it).
╭─ melo 0.1.0 ─────────────────────────────────────────────────────────────╮
│ model      qwen3:8b · Ollama · http://127.0.0.1:11434/v1                 │
│            saved as the default                                          │
│ workspace  ~/demo-project                                                │
│ tools      4 built-in (3 ask first)                                      │
│ memory     no long-term memory · melo --reconfigure adds one             │
│ session    new                                                           │
╰──────────────────────────────────────────────────────────────────────────╯
  / commands · esc stop · ctrl+d quit
```

If the server also lists an embedding model, the first run adds one question,
`Use <model> for long-term memory? [Y/n]:`, and `y` saves it after one
embeddings probe. Type a message and press Enter, or `/exit` to leave. From
then on, `melo` goes straight in and `melo "<task>"` uses the saved server.
The server here is Ollama, so it is saved as the `ollama` backend; any other
OpenAI-compatible server is saved as `vllm`, the key for those. See
[First run](../guides/configure.md#first-run) and
[Set up an embedding model](../guides/embedding.md).

### With the setup wizard

The setup wizard also sets up an embedding server or a cloud API, and can
start the gateway when it is done. For a cloud API, first run
`export OPENAI_API_KEY="<your key>"` (or your vendor's variable) in this
terminal: the wizard's test request uses that key.

```bash
melo --serve --reconfigure
```

The wizard asks:

| Prompt | What to enter |
|---|---|
| `Chat backend` | The number of your server type, such as `2` for Ollama. The default is the saved backend, else `[1]`, MLC. For another OpenAI-compatible server, choose vLLM (`3`). |
| `Base URL` | Your server's URL. Press Enter to accept the default shown in brackets: the saved URL, else the backend's usual one. |
| `Chat model name` | The model, such as `qwen3:8b`. |
| `Start?` | Press Enter (yes) to continue. |
| `Embedding backend` | `5` for `None (no long-term memory)`. Without a saved embedding, the chat server's own embedding models are appended as `6`, `7`, … with the first as the default; choosing one is enough. |
| `Embedding endpoint URL`, `Embedding model name` | Only asked when you chose one of the fixed embedding backends `1`–`4`, not a detected model. |

For a cloud API, choose the vendor. The wizard then asks for
`API key env var`, the name of the variable you exported, and `Chat model`
instead of `Base URL` and `Chat model name`.

For Ollama, the default URL `http://localhost:11434` has no `/v1`. That is
correct: the `ollama` backend uses Ollama's own API, not the
OpenAI-compatible one.

Next, the wizard sends a short test request to each server
(`Probing backends ...`). If the chat server does not answer, it prints
`Probe failed: …` and asks `[r]etry, [s]ave anyway, [N]o [N]:`; `r` asks for
the URL and model again, and pressing Enter saves nothing. If only the
embedding server fails, the wizard saves with embedding off and says so. On
success it prints `Config saved to …`, the path of
`~/.cache/localmelo/config.toml`, and then **starts the gateway** on
`http://127.0.0.1:8401`. The gateway keeps running in this terminal.
Without `--serve`, `melo --reconfigure` opens the interactive UI instead.

## Start the gateway

If you saved the configuration with `melo`, start the gateway with
`melo --serve`; the wizard above starts it by itself. From a second
terminal, check the gateway and run a task through it:

```bash
curl http://127.0.0.1:8401/v1/health
curl -X POST http://127.0.0.1:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 6*7?"}'
```

Example output (values vary):

```json
{"status":"ok","uptime":12.3,"sessions":{"active":0,"busy":0,"max":50},"chat_backend":"ollama","embedding_backend":"none"}
{"result":"42","session_id":"a7e8f646ca2e"}
```

The health check does not contact the model server; a task does. Open
`http://127.0.0.1:8401/` in a browser for the web workspace (see
[Interactive conversations](#interactive-conversations)).

Press Ctrl+C in the first terminal to stop the gateway. Tasks use the saved
configuration:

```bash
melo "What is 6*7?"
```

To start the gateway again without the wizard, run `melo --serve`.

## Interactive conversations

Run `melo` without a task to chat in the terminal. Each tool call shows as
it finishes, a spinner shows the step and the time while the model works,
and the answer prints as Markdown:

```bash
melo
# Or talk to an OpenAI-compatible server that is not saved, with its first
# model or the one you name:
melo --base-url http://127.0.0.1:8000/v1
melo --base-url http://127.0.0.1:8000/v1 --chat-model YOUR_MODEL_ID
```

- Type `/` for the commands: `/model` lists the server's models and
  switches to one, `/backend` switches to another server (both saved as the
  default, except in a `melo --base-url` session), `/new` starts over,
  `/exit` or Ctrl+D leaves.
- melo asks before it runs a shell command, Python code or a file write:
  press `1` to allow it, `2` to allow that tool for the session, `3` to
  refuse and tell melo why. Esc stops a turn.
- melo works in the current folder (`--workspace DIR` picks another): file
  tools stay in it; commands and Python can reach other files, so melo asks
  first.
- A chat can run in a reasoning and recall mode: `--reasoning`,
  `--reasoning-field`, `--replay-turns`, `--note-writer`, `--retriever`,
  `--recall-*` and `--retrieval-extra-body`, or the saved `[agent]` and
  `[retrieval]` tables. Explicit flags win, and retrieval flags stay inert
  in `legacy`.

[The interactive UI](../guides/terminal.md#the-interactive-ui) has the keys and the rest.
`melo ""` opens the plain conversation loop of earlier releases. Without a
terminal, `melo` alone exits with code 2; start the gateway with
`melo --serve`.

The web workspace at `http://127.0.0.1:8401/` is the console's live page and
talks only to `/api`: it uses the gateway's configured model, so
`melo --reconfigure` changes it. It lists your chats newest first, with
search and a draft for each, and draws a turn as the terminal does: your
message, a work card for each tool call, folded reasoning and reflections,
notices, the live progress row and the answer with its facts and
`Copy response`. Stop cancels the turn, a waiting approval shows as
`Needs approval`, and the chats are saved in the gateway's `sessions.db`, so
a restart keeps them. Other browser tabs on the same gateway see the same
chats. Approval cards, connection settings and the session details drawer are
available in the workspace; see [Web workspace](../guides/web-console.md#web-workspace).

## Next steps

- [Configuration](../reference/configuration.md): backends, cloud API keys, timeouts.
- [Tools and MCP](../reference/tools.md): choose built-in tools and add MCP servers.
- [Gateway and HTTP API](../guides/gateway.md): sessions, the web workspace, running at login.
- [Python API](../reference/python-api.md): run tasks from Python, follow their events, approve tool calls.
- [Memory](../reference/memory.md): keep memory between runs, inspect and reset it.
- [Command-line reference](../reference/cli.md): every option, exit codes, scripting.
- [Troubleshooting](../reference/troubleshooting.md): error messages and fixes.
