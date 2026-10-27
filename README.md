<p align="center">
  <img src="docs/assets/melo.png" alt="Melo, the localmelo mascot" width="200">
</p>

# localmelo

[English](README.md) | [简体中文](README.zh-CN.md)

A local personal agent that runs tasks with tools and memory on your own model server or a cloud API.
Use it to run agent tasks from the shell, from scripts and benchmarks (each
run gives one JSON line and a fixed exit code), or over HTTP.

[Website](https://localmelo.github.io/localmelo/) ·
[Quickstart](docs/quickstart.md) ·
[Documentation](docs/README.md) ·
[Roadmap](https://localmelo.github.io/localmelo/en/updates.html#roadmap)

**Status:** pre-alpha. Running tasks from the command line (direct mode) and
the HTTP gateway work today. Commands, the HTTP API and the Python API may
change before v0.1.0 (milestone due 2026-12-20).

## What it does

- **Runs one task per command.** `melo "<task>"` works within a step budget,
  reflects between attempts, and can have a time limit.
- **Uses tools.** Four built-in tools (`shell_exec`, `python_exec`,
  `file_read`, `file_write`) and the tools of MCP servers (stdio servers, in
  direct mode and the Python API).
- **Reports every run** as text or as one JSON line, with fixed exit codes,
  so scripts and benchmarks can rely on it.
- **Connects to your models.** A model server you run (Ollama, MLC LLM, vLLM,
  SGLang) or a cloud API (OpenAI, Anthropic, Google Gemini, NVIDIA).
- **Remembers.** Working memory during a run, a history of tasks and steps,
  and long-term memory when an embedding server is configured. History and
  long-term memory are saved to SQLite when you turn persistence on.
- **Serves the same agent over HTTP** with sessions (the gateway), plus a web
  workspace for running the agent in a browser, on a model server you choose.

localmelo does not download, start or serve models: you run the model server
yourself.

## Requirements

- macOS (developed and tested) or Linux (should work, untested). Windows is
  not supported.
- Python 3.11 or later, and Git.
- [uv](https://docs.astral.sh/uv/) (recommended) or pip.
- A model server that you run, such as [Ollama](https://ollama.com), or an
  API key for a cloud API.

## Install

```bash
git clone https://github.com/localmelo/localmelo.git
cd localmelo
uv sync --extra gateway
source .venv/bin/activate
```

Keep the directory name `localmelo`: the package is built from it. A renamed
folder or a ZIP download (`localmelo-main`) fails with
`ModuleNotFoundError: No module named 'localmelo'`.

Without uv, see [Install with pip](docs/quickstart.md#install-with-pip).

The examples call `melo` directly. Activation lasts for this terminal only:
in a new terminal, run `source <clone>/.venv/bin/activate` again, where
`<clone>` is your `localmelo` directory, or use `uv run melo …` inside the
clone. To update or uninstall localmelo, see
[Configuration](docs/configuration.md#update-localmelo).

## Quick start

1. Start a model server. This example uses [Ollama](https://ollama.com):
   install it, start it, and pull a model with `ollama pull qwen3:8b`. Any
   OpenAI-compatible server works with its `/v1` URL.

2. Run a task. The model can use tools with your account's permissions; see
   [Security](#security).

   ```bash
   melo --base-url http://127.0.0.1:11434/v1 --chat-model qwen3:8b "What is 6*7?"
   ```

   The answer is printed on stdout, and the exit code is 0 when the run
   completed. Add `--output-format json` to get one JSON line with the run's
   record instead. `--base-url` sends no API key; for a cloud API, skip to
   step 3.

3. Save a configuration and start the gateway. For a cloud API, first run
   `export OPENAI_API_KEY="<your key>"` (or your vendor's variable) in the
   same terminal: the wizard's test request uses that key.

   ```bash
   melo --serve --reconfigure
   ```

   The setup wizard asks for the chat backend, its URL and model, and an
   embedding backend (`None` if you have no embedding server). It checks
   that the servers answer, saves `~/.cache/localmelo/config.toml`, and then
   serves the HTTP API and the web workspace on `http://127.0.0.1:8401`.
   Press Ctrl+C to stop it. From then on, `melo "<task>"` uses the saved
   configuration.

The [Quickstart](docs/quickstart.md) walks through each step in more detail,
with the wizard's choices and a first request to the HTTP API.

## Three ways to run it

| Command | What it does |
|---|---|
| `melo "<task>"` | Runs one task, prints the result and exits. |
| `melo ""` (an empty task) | Opens a conversation loop in the terminal. `/help` lists its commands; `/exit` leaves. |
| `melo --serve`, or `melo` alone | Starts the gateway: the HTTP API and the web workspace at `http://127.0.0.1:8401/`. |

To hold a conversation with oMLX or another OpenAI-compatible server without
a saved configuration:

```bash
melo --base-url http://127.0.0.1:8000/v1 --chat-model YOUR_MODEL_ID ""
```

In the web workspace, enter your server's URL, pick one of its models and
create a chat. You can keep several chats, search them, read answers rendered
as Markdown and stop a run. Chats last only as long as the gateway process,
and the workspace is for your own computer only; see
[Web workspace](docs/gateway.md#web-workspace). The terminal loop's
conversation is separate from the web chats.

## Documentation

| Page | What it covers |
|---|---|
| [Quickstart](docs/quickstart.md) | Install, run a first task, save a configuration, start the gateway |
| [Command-line reference](docs/cli.md) | `melo` modes, options, output, exit codes and signals |
| [Configuration](docs/configuration.md) | Setup wizard, config file, backends, timeouts, environment variables |
| [Tools and MCP](docs/tools.md) | Built-in tools, MCP servers, which tools a step offers, the checks |
| [Gateway and HTTP API](docs/gateway.md) | HTTP API, sessions, web workspace, running at login on macOS |
| [Memory](docs/memory.md) | What localmelo remembers, keeping, inspecting and resetting it |
| [Python API](docs/python-api.md) | Run tasks from Python code, with events and approvals (experimental) |
| [Troubleshooting](docs/troubleshooting.md) | Error messages and how to fix them |

[All documentation](docs/README.md) lists every page, including the reference
for events, sessions and offline retrieval. [Evaluations](docs/evaluations.md)
reports the comparisons behind the defaults, such as the reflection gate.

AI agents and LLM tools: read [localmelo for agents](docs/for-agents.md), one
compact page, or the [llms.txt](https://localmelo.github.io/localmelo/llms.txt)
index.

## Roadmap

The roadmap is tracked in [#2](https://github.com/localmelo/localmelo/issues/2),
and the v0.1.0 milestone is due 2026-12-20. The
[website](https://localmelo.github.io/localmelo/en/updates.html#roadmap) has
its phases and the dated updates.

Planned:

- an interactive CLI that opens when you run `melo` without a task and shows
  the agent's tool calls as they happen;
- conversations that survive a gateway restart, with the agent's work shown
  live in the web workspace;
- keeping the model's reasoning as memory notes, with switchable retrieval
  (`bm25`, `llm`, `embedding`);
- running several tasks at once on one local model server, with your
  conversation answered first;
- a progress display for tasks you run again and again, with their steps
  learned from earlier runs (a first run shows only facts, such as its step
  count).

## Security

> [!WARNING]
> Tools run with your account's permissions, and there is no sandbox: the
> model can run shell commands and Python code, and read or write any file
> you can. The gateway has no authentication, so keep it on `127.0.0.1` (the
> default): there it refuses requests to its API that web pages at other
> sites send through your browser. See [SECURITY.md](SECURITY.md) and
> [What the checks do](docs/tools.md#what-the-checks-do).

The experimental [Python API](docs/python-api.md) can ask for approval before
a tool runs and keep the file tools inside one directory; direct mode and the
gateway do neither. Memory and session files are not encrypted; see
[Privacy](docs/memory.md#privacy).

## Getting help

Search the [issues](https://github.com/localmelo/localmelo/issues) or open a
new one. Include the JSON line from `--output-format json` (it names the
localmelo version), the exit code, the backend and model, your operating
system and Python version, and the steps to reproduce. Remove API keys and
personal paths first. Report security problems
as described in [SECURITY.md](SECURITY.md), not in a public issue.

## Contributing

Bug reports, documentation fixes and small, focused pull requests are
welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and
the checks to run.

## License

MIT © The localmelo Authors. See [LICENSE](LICENSE).
