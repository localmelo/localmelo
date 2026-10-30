# Quickstart

This tutorial installs localmelo, runs a first task against a local model
server, saves a configuration and starts the gateway.

## Before you begin

You need:

- macOS (developed and tested) or Linux (should work, untested). Windows is
  not supported.
- Python 3.11 or later, Git, and [uv](https://docs.astral.sh/uv/) (or pip,
  see [Install with pip](#install-with-pip)).
- A model server. The commands below use [Ollama](https://ollama.com):
  install it, start it, and pull a model with `ollama pull qwen3:8b`. On a
  Mac with Apple silicon, [oMLX](../guides/model-servers.md#serve-a-model-with-omlx)
  is the recommended server: use `http://127.0.0.1:8000/v1` and your model's
  folder name in the commands instead. Any server with an OpenAI-compatible
  API works too: use its `/v1` URL and its model name.

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
> file you can: there is no sandbox. To run a task without tools, see
> [Choose the built-in tools](../guides/tools-mcp.md#choose-the-built-in-tools).
> See also the [security policy](https://github.com/localmelo/localmelo/blob/main/SECURITY.md)
> and [What the checks do](../guides/tools-mcp.md#what-the-checks-do).

```bash
melo --url http://127.0.0.1:11434/v1 --model qwen3:8b "What is 6*7?"
```

The answer is printed on stdout, and the command exits with code 0
(`echo $?` prints `0`). `--url` points localmelo at an OpenAI-compatible
server for this one run, without a configuration file. It sends no API key,
so use it for local servers; for a cloud API, save a configuration (below).
`-u` and `-m` are the letters of `--url` and `--model`; the old spellings
`--base-url` and `--chat-model` still work ([Old spelling to new](../reference/cli.md#old-spelling-to-new)).

Without `--model`, melo first asks the server which models it has and
uses its only chat model, or the one the server says is in memory. With
several chat models and none known to be in memory it stops with exit code 2
and lists them, because asking for a model that is not loaded can make the
server load a large one: pass `--model <id>`.

With [oMLX](../guides/model-servers.md#serve-a-model-with-omlx), use its URL
and the name of a model folder:

```bash
melo --url http://127.0.0.1:8000/v1 --model Qwen3-8B-4bit "What is 6*7?"
```

Direct tasks write files and run Python without asking, and have no
`shell_exec` tool unless you pass `--builtins all`. That is not a sandbox:
`python_exec` can still start programs (see the
[warning above](#run-your-first-task)). A brief's check (see
[Run a worker task](#run-a-worker-task)) runs anyway.

For every option, see the [command-line reference](../reference/cli.md).

## Get the result as JSON

```bash
melo --url http://127.0.0.1:11434/v1 --model qwen3:8b --output json "What is 6*7?"
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

## Run a worker task

A worker task is a brief in a file. A line that starts with `Check:` (no
leading space) holds one shell command that tells whether the task is done:

```bash
cat > brief.md <<'EOF'
Write hello.py so that it prints the word hello.
Check: python3 hello.py | grep -x hello
EOF
melo worker brief.md --url http://127.0.0.1:11434/v1 --model qwen3:8b > result.json
echo "exit code: $?"
python3 -c 'import json; r = json.load(open("result.json")); print(r["status"], r["tests"]["state"])'
```

When the model has finished, `melo` runs the check in the workspace (the
current folder, or `--workspace`), whatever tools the run offers the model.
If it fails, the model is sent back to fix it, twice at most. Example output
of the last two commands (the model's work varies):

```text
exit code: 0
completed passed
```

`tests.state` is `passed` when every check of the last round exited with 0,
and `failed` when one did not, or could not run. A run that ends `completed`
with a failed check exits with code 4, so a script can stop on it:

```bash
melo worker brief.md --url http://127.0.0.1:11434/v1 --model qwen3:8b > result.json && echo "check passed"
```

The experimental setting `agent.brief_items` (`-X agent.brief_items=true`) also names the brief's numbered items that
no edit has touched yet in the model's reflection requests and before its last
steps (see the [reference](../reference/cli.md#brief-items)).

On macOS, the worker's shell commands and Python code can write only in the
workspace: a write elsewhere, even after `cd ..`, fails with `Operation not
permitted` and a line telling the model to write in the workspace, and scratch
files go to `$TMPDIR`, which is `.melo-tmp` in the workspace. The caches of uv,
black and npm stay writable, and in a git worktree `git status`, `git diff` and
`git add` work. Reads and the network are not limited. To run a worker without
this, add `-X agent.confine_writes=false`; see
[Writes stay in the workspace](../reference/cli.md#confined-writes).

The result is one JSON line with `status`, `exit_code`, `tests` and the run
record. For every field and exit code, see the
[command-line reference](../reference/cli.md#output).

## Save a configuration

A configuration file saves the backends, so that `melo`, `melo "<task>"`
and the gateway work without `--url`.

For a local server, run `melo` in a terminal. With no configuration yet, it
first says what it needs, a model server on this computer. It then looks for
one (oMLX on port 8000, Ollama on 11434), asks whether to use the one it found
(`Use this model server?`, a list: Enter takes the selected `Use it`) and for a
model from the server's list, saves
`~/.cache/localmelo/config.toml` and opens the interactive UI. The models are
a list to choose from with the arrow keys (type to filter, Enter to choose,
Esc to cancel), chat models only; with the answers piped in they are numbered
instead, chat models first, and an embedding or reranking model is tagged
`(embedding)` or `(reranking)` and is never the default. When the list holds
several chat models and the server says which are in memory (oMLX, Ollama),
they are marked `(loaded)` and the first of them is highlighted, so Enter does
not load a large model that is not in memory. When it does not say (a server
with an API key answers 401, others have no such route) or none is loaded,
nothing is highlighted and Enter does nothing until you move to a model,
because a model that is not loaded yet can take minutes and a lot of memory
to load (piped, there is no default either: Enter alone asks again, and an
empty answer is an error). With one chat model there is nothing to choose:

```text
$ melo
melo uses a model server on this computer (oMLX, Ollama, LM Studio…).
Found a model server at http://127.0.0.1:11434/v1 (1 model).
✓ http://127.0.0.1:11434/v1 lists 1 model; using qwen3:8b.
Long-term memory is off; melo setup turns it on.
Saved to ~/.cache/localmelo/config.toml (melo setup changes it).
melo 0.1.0 · qwen3:8b (/model to change)
~/demo-project
Try: What is this project? · / commands · ctrl+d quit
```

The first row shows the package version, and on a git checkout the short commit, as in `melo 0.1.0 (baab21e)`; `melo --version` prints the same line and exits 0. `/status` checks that the model answers and shows the folder, session, approvals and context use; `/status all` adds the tools, memory, limits and files.

With no model server running, `melo` prints this computer's commands for
starting one (oMLX on a Mac, Ollama elsewhere, the ones in the
[model servers guide](../guides/model-servers.md)) and asks
`No model server found. What now?` with `Check again`, `Enter an address` and
`Quit` (piped: `Press Enter to check again, type an address, or q to quit:`).
Start the server in another terminal, then choose `Check again`; melo installs
and starts nothing itself.

If the server also lists an embedding model, the first run adds one question,
`Use <model> for long-term memory?`, and `Use it` (selected first) saves it
after one embeddings probe.

Every question with a fixed set of answers is a list on a terminal: ↑ ↓ or a
number move, Enter confirms, Esc takes the safe answer, and the list is erased
once you answer (see [Questions are lists](../guides/terminal.md#questions-are-lists)).
Piped answers and scripts keep the typed questions (`Use it? [Y/n]:`).

Type a message and press Enter, or `/exit` to leave. From then on, `melo` goes straight in and `melo "<task>"` uses the saved server.
The server here is Ollama, so it is saved as the `ollama` backend; an oMLX
server is saved as `omlx` and any other OpenAI-compatible server as `vllm`,
the key for those. See
[First run](../guides/configure.md#first-run) and
[Set up an embedding model](../guides/embedding.md).

### With the setup wizard

The setup wizard also sets up an embedding server or a cloud API. For a
cloud API, first run `export OPENAI_API_KEY="<your key>"` (or your vendor's
variable) in this terminal: the wizard's test request uses that key. Run the
wizard, then start the gateway:

```bash
melo setup
melo serve
```

In default mode `melo setup` first offers the server it finds (`Use this model
server?`); choose `Use another server` for the full menu, which developer mode
(`melo config set ui.mode developer`) shows at once. The wizard asks:

| Prompt | What to enter |
|---|---|
| `Chat backend` | Your server type, such as `2. Ollama (local endpoint)`: move with ↑ ↓ and press Enter, or press the number (piped: type it). The saved backend is selected first, else MLC (`1`). For oMLX or another OpenAI-compatible server, choose oMLX or vLLM (`3`). |
| `Base URL` | Your server's URL. Press Enter to accept the default shown in brackets: the saved URL, else the backend's usual one. |
| `Chat model name` | The model, such as `qwen3:8b`. |
| `Save these settings?` | Press Enter on `Save and continue`, or choose `Cancel setup` (piped: `Save and continue? [Y]:`). |
| `Embedding backend` | `5. None (no long-term memory)`. Without a saved embedding, the chat server's own embedding models are appended as `6`, `7`, … with the first selected; choosing one is enough. |
| `Embedding endpoint URL`, `Embedding model name` | Only asked when you chose one of the fixed embedding backends `1`–`4`, not a detected model. |

For a cloud API, choose the vendor. The wizard then asks for
`API key env var`, the name of the variable you exported, and `Chat model`
instead of `Base URL` and `Chat model name`.

For Ollama, the default URL `http://localhost:11434` has no `/v1`. That is
correct: the `ollama` backend uses Ollama's own API, not the
OpenAI-compatible one.

Next, the wizard sends a short test request to each server
(`Probing backends ...`). If the chat server does not answer, it prints
`Probe failed: …` and asks `The check failed. What now?` with `Try again`,
`Save anyway` and `Cancel setup` (piped: `[r]etry, [s]ave anyway, [N]o [N]:`);
`Try again` asks for the URL and model again, and Enter, which takes the
selected `Cancel setup`, saves nothing. If only the
embedding server fails, the wizard saves with embedding off and says so. On
success it prints `Config saved to …`, the path of
`~/.cache/localmelo/config.toml`, and opens the interactive UI; leave it with
Ctrl+D. Then `melo serve` starts the gateway on `http://127.0.0.1:8401`, and
it keeps running in this terminal. In default mode `melo setup` starts with the
first run's question: it prints `Found a model server at …` and asks
`Use this model server?`. The full menu above follows only after
`Use another server`, in developer mode, or when no server is found while a
backend is saved; with nothing saved, it prints the commands that start a
server instead.

## Start the gateway

Once the configuration is saved, by `melo` or by the wizard above, start the
gateway with `melo serve`. From a second terminal, check the gateway and run a
task through it:

The examples pipe the gateway bearer token from its private file to curl, keeping
it out of command-line arguments and shell history. Do not enable verbose curl
output or log the header; same-user processes can still read the token file.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/v1/health
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 6*7?"}'
```

Example output (values vary):

```json
{"status":"ok","uptime":12.3,"sessions":{"active":0,"busy":0,"max":50},"chat_backend":"ollama","embedding_backend":"none"}
{"result":"42","session_id":"a7e8f646ca2e"}
```

The health check does not contact the model server; a task does. The web
workspace opens in your browser when `melo serve` starts at a terminal on this
computer; `melo console` opens it again. Over SSH, or when no browser starts,
`melo console` prints a one-use link to open within 60 seconds. Do not share the
link. See [Interactive conversations](#interactive-conversations).

Press Ctrl+C in the first terminal to stop the gateway. Tasks use the saved
configuration:

```bash
melo "What is 6*7?"
```

To start the gateway again, run `melo serve`. With
`melo serve --port 0` the system picks a free port: the startup lines show
the real address, and `melo console --port PORT` and
`melo-task --gateway URL` need it.

## Run a task through the gateway

With the gateway running, `melo-task` sends a brief to it, as in
[Run a worker task](#run-a-worker-task), and lists the tasks:

```bash
melo-task new --brief brief.md
melo-task list
```

Example output (the id and the model's work vary; the task works in the
default workspace, `~/localmelo-workspace`):

```text
c5b2d38ceecc  queued
ID            LEVEL  STATE  TITLE
c5b2d38ceecc  P2     held   Write hello.py so that it prints the word hello.
  Waiting for your approval: file_write hello.py (answer it in the web console, Tasks)
```

With the default settings a file write waits for your approval, so the task
is `held` until you answer it. The task waits once for each tool the model
asks to use: a model that runs its own check asks for `python_exec` too, and
the task is `held` again until you answer that. `melo-task show` names the
tool it waits for. Run `melo console` in a terminal to open the console,
and answer the request in the Tasks view. Then show the
task:

```bash
melo-task show c5b2d38ceecc
```

```text
c5b2d38ceecc  done
  Title: Write hello.py so that it prints the word hello.
  Level: P2
```

`show` adds `Reason:` and `Waiting:` lines only when the task has a reason or
waits for something. `melo-task --help` lists the other commands.

## Interactive conversations

Run `melo` without a task to chat in the terminal. Each tool call shows as
it finishes, a spinner shows what melo is doing and the time while the model
works, and the answer prints as Markdown, with the files a turn changed named
under its footer:

```bash
melo
# Or talk to an OpenAI-compatible server that is not saved. Without
# --model it uses the server's only chat model or its one loaded model;
# with several chat models and not exactly one loaded, it exits 2 and lists them:
melo --url http://127.0.0.1:8000/v1
melo --url http://127.0.0.1:8000/v1 --model YOUR_MODEL_ID
```

- Type `/` for the commands (`/help` lists the everyday ones, `/help all` every
  one): `/model` opens a list of the server's models to
  switch to one, `/backend` switches to another server (both saved as the
  default, except in a `melo --url` session), `/new` starts over,
  `/exit` or Ctrl+D leaves.
- melo asks before it runs a shell command, Python code or a file write:
  press `1` to allow it, `2` to allow that tool for the session, `3` to
  refuse and tell melo why. Esc stops a turn.
- melo works in the current folder (`--workspace DIR` picks another): file
  tools stay in it; commands and Python can reach other files, so melo asks
  first.

[The interactive UI](../guides/terminal.md#the-interactive-ui) has the keys and the rest.
`melo ""` opens the plain conversation loop of earlier releases. Without a
terminal, `melo` alone exits with code 2; start the gateway with
`melo serve`.

The web workspace at `http://127.0.0.1:8401/` is the console's live page and
talks only to `/api`: it uses the gateway's configured model, so
`melo setup` changes it. On a first run, with no model set up, its welcome
shows a card that finds the model server already running on this Mac and connects
it with **Use this model**, without opening settings. It lists your chats newest first, with
search and a draft for each, and draws a turn as the terminal does: your
message, a work card for each tool call, folded reasoning and reflections,
notices, the live progress row and the answer with its facts and
`Copy response`. Stop cancels the turn, a waiting approval shows as
`Needs approval`, and the chats are saved in the gateway's `sessions.db`, so
a restart keeps them. Other browser tabs on the same gateway see the same
chats. Approval cards, settings and the session details drawer are
available in the workspace; see [Web workspace](../guides/web-console.md#web-workspace).

## Next steps

- [Configuration](../reference/configuration.md): backends, cloud API keys, timeouts.
- [Tools and MCP](../reference/tools.md): choose built-in tools and add MCP servers.
- [Gateway and HTTP API](../guides/gateway.md): sessions, the web workspace, running at login.
- [Python API](../reference/python-api.md): run tasks from Python, follow their events, approve tool calls.
- [Memory](../reference/memory.md): keep memory between runs, inspect and reset it.
- [Command-line reference](../reference/cli.md): every option, exit codes, scripting.
- [Troubleshooting](../reference/troubleshooting.md): error messages and fixes.
