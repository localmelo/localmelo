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
  name in the commands below.

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
> and [What the checks do](tools.md#what-the-checks-do).

```bash
melo --base-url http://127.0.0.1:11434/v1 --chat-model qwen3:8b "What is 6*7?"
```

The answer is printed on stdout, and the command exits with code 0
(`echo $?` prints `0`). `--base-url` points localmelo at an OpenAI-compatible
server for this one run, without a configuration file. It sends no API key,
so use it for local servers; for a cloud API, save a configuration (below).

For every option, see the [command-line reference](cli.md).

## Get the result as JSON

```bash
melo --base-url http://127.0.0.1:11434/v1 --chat-model qwen3:8b --output-format json "What is 6*7?"
```

Example output (values vary):

```json
{"task_id": "f52e785a84a144ff8c3b8e70e0451eb1", "status": "completed", "result": "42", "error": "", "steps": 1, "attempts": 1, "n_llm": 1, "n_main": 1, "n_aux": 0, "n_tool": 0, "llm_errors": 0, "usage": {"prompt_tokens": 120, "completion_tokens": 12, "total_tokens": 132}, "duration_s": 1.9, "config": {"max_steps": 30, "attempt_policy": "reflect", "steps_per_attempt": 10, "max_attempts": 5, "reflection_gate": "flags", "tool_exposure": "retrieve", "builtins": ["file_read", "file_write", "python_exec", "shell_exec"], "memory": true, "memory_dir": null, "timeout_s": null, "tools": ["file_read", "file_write", "python_exec", "shell_exec"]}, "localmelo_version": "603f40d"}
```

stdout holds exactly this one line. The exit code is 0 only when `status`
is `completed`.

### Run results

`status` says how the run ended, for example `completed`, `budget_exhausted`
or `provider_error`. The counters say how many steps, model requests and tool
calls the run took, and `config` records the options it ran with. See
[Run results](run-results.md) for every status and field.

## Save a configuration

A configuration file saves the backends, so that `melo "<task>"` and the
gateway work without `--base-url`. The setup wizard writes it. For a cloud
API, first run `export OPENAI_API_KEY="<your key>"` (or your vendor's
variable) in this terminal: the wizard's test request uses that key.

```bash
melo --serve --reconfigure
```

The wizard asks:

| Prompt | What to enter |
|---|---|
| `Chat backend` | The number of your server type, such as `2` for Ollama. The default, `[1]`, is MLC. For another OpenAI-compatible server, choose vLLM (`3`). |
| `Base URL` | Your server's URL. Press Enter to accept the default shown in brackets. |
| `Chat model name` | The model, such as `qwen3:8b`. |
| `Start?` | Press Enter (yes) to continue. |
| `Embedding backend` | `5` for `None (no long-term memory)` if you have no embedding server. |
| `Embedding endpoint URL`, `Embedding model name` | Only asked when you chose an embedding backend. |

For a cloud API, choose the vendor. The wizard then asks for
`API key env var`, the name of the variable you exported, and `Chat model`
instead of `Base URL` and `Chat model name`.

For Ollama, the default URL `http://localhost:11434` has no `/v1`. That is
correct: the `ollama` backend uses Ollama's own API, not the
OpenAI-compatible one.

Next, the wizard sends a short test request to each server
(`Probing backends ...`). If a server does not answer, it prints
`Probe failed: …` and asks `Save config anyway? [N]`; pressing Enter saves
nothing. On success it prints `Config saved to …`, the path of
`~/.cache/localmelo/config.toml`, and then **starts the gateway** on
`http://127.0.0.1:8401`. The gateway keeps running in this terminal.

## Start the gateway

From a second terminal, check the gateway and run a task through it:

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
`http://127.0.0.1:8401/` in a browser for the web UI (playground).

Press Ctrl+C in the first terminal to stop the gateway. Tasks now use the
saved configuration:

```bash
melo "What is 6*7?"
```

To start the gateway again without the wizard, run `melo --serve`.

## Interactive conversations

Give an empty task to enter a terminal conversation using the same agent across
turns:

```bash
melo ""
# Or use the exact model ID from an OpenAI-compatible server:
melo --base-url http://127.0.0.1:8500/v1 --chat-model YOUR_MODEL_ID ""
```

Use `/help` for commands and `/exit` or Ctrl-D to leave. Without the `""`,
`melo` starts the gateway, as `melo --serve` does.

The web workspace at `http://127.0.0.1:8401/` supports model discovery, separate
chat drafts, code formatting, cancellation, and responsive navigation. Connected
browser clients synchronize conversation state, and reloads recover live chats.
Sessions do not survive gateway restarts, and direct CLI chats are independent.

## Next steps

- [Configuration](configuration.md): backends, cloud API keys, timeouts.
- [Tools and MCP](tools.md): choose built-in tools and add MCP servers.
- [Gateway and HTTP API](gateway.md): sessions, the web UI, running at login.
- [Memory](memory.md): keep memory between runs, inspect and reset it.
- [Command-line reference](cli.md): every option, exit codes, scripting.
- [Troubleshooting](troubleshooting.md): error messages and fixes.
