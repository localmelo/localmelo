<p align="center">
  <img src="docs/assets/melo.png" alt="Melo, the localmelo mascot" width="200">
</p>

# localmelo

[English](README.md) | [简体中文](README.zh-CN.md)

A local personal agent that runs tasks with tools and memory on your own model server or a cloud API.

[Website](https://localmelo.github.io/localmelo/) ·
[Quickstart](docs/quickstart.md) ·
[Documentation](docs/README.md) ·
[Roadmap](https://localmelo.github.io/localmelo/en/updates.html#roadmap)

**Status:** pre-alpha. Direct mode and the gateway work today. Commands, the
HTTP API and the Python API may change before v0.1.0 (milestone due
2026-12-20).

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
- **Serves the same agent over HTTP** with sessions, plus a small web page
  for trying it out (the gateway).

localmelo does not download, start or serve models: you run the model server
yourself.

## Requirements

- macOS. localmelo is developed and tested on macOS. It should also work on
  Linux. Windows is not supported.
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

Without uv, see [Install with pip](docs/quickstart.md#install-with-pip). The
examples call `melo` directly; if you did not activate the environment, put
`uv run` in front of them.

## Quick start

1. Start a model server. This example uses [Ollama](https://ollama.com):
   install it, start it, and pull a model with `ollama pull qwen3:8b`. Any
   OpenAI-compatible server works with its `/v1` URL.

2. Run a task:

   ```bash
   melo --base-url http://127.0.0.1:11434/v1 --chat-model qwen3:8b "What is 6*7?"
   ```

   The answer is printed on stdout, and the exit code is 0 when the run
   completed. Add `--output-format json` to get one JSON line with the run's
   record instead.

3. Save a configuration and start the gateway:

   ```bash
   melo --serve --reconfigure
   ```

   The setup wizard asks for the chat backend, its URL and model, and an
   embedding backend; choose `None (no long-term memory)` if you have no
   embedding server. It checks that the servers answer, saves
   `~/.cache/localmelo/config.toml`, and then serves on
   `http://127.0.0.1:8401`. In another terminal:

   ```bash
   curl http://127.0.0.1:8401/v1/health
   curl -X POST http://127.0.0.1:8401/v1/agent/run \
     -H "Content-Type: application/json" \
     -d '{"query": "What is 6*7?"}'
   ```

   Press Ctrl+C to stop the gateway. From then on, `melo "<task>"` uses the
   saved configuration.

Next, the [Quickstart](docs/quickstart.md) walks through each step in more
detail.

## Documentation

| Page | What it covers |
|---|---|
| [Quickstart](docs/quickstart.md) | Install, run a first task, save a configuration, start the gateway |
| [Command-line reference](docs/cli.md) | `melo` modes, options, output, exit codes and signals |
| [Run results](docs/run-results.md) | How a run can end, the run record, retries |
| [Configuration](docs/configuration.md) | Setup wizard, config file, backends, timeouts, environment variables |
| [Tools and MCP](docs/tools.md) | Built-in tools, MCP servers, which tools a step offers, the checks |
| [Gateway and HTTP API](docs/gateway.md) | HTTP API, sessions, web UI, running at login on macOS |
| [Memory](docs/memory.md) | What localmelo remembers, keeping, inspecting and resetting it |
| [Python API](docs/python-api.md) | `Agent`, `AgentOptions` and `RunResult` (experimental) |
| [How localmelo works](docs/how-it-works.md) | The run loop, reflection, memory and timeouts |
| [Troubleshooting](docs/troubleshooting.md) | Error messages and how to fix them |
| [Architecture](https://localmelo.github.io/localmelo/en/architecture.html) | How the parts fit together (website) |
| [Roadmap](https://localmelo.github.io/localmelo/en/updates.html#roadmap) | Plans and dated updates (website) |

## Roadmap

The roadmap is tracked in [#2](https://github.com/localmelo/localmelo/issues/2).
The v0.1.0 milestone is due 2026-12-20.

| Phase | Dates | Scope |
|---|---|---|
| Phase 1 | 2026-10-19 to 2026-11-01 | Core loop v2 and the command line ([#3](https://github.com/localmelo/localmelo/issues/3)) |
| Phase 2 | 2026-11-02 to 2026-11-22 | Memory: notes, retrieval and long-term promotion |
| Phase 3 | 2026-11-23 to 2026-12-06 | Safety and privacy ([#29](https://github.com/localmelo/localmelo/issues/29)) |

Planned:

- an interactive CLI (running `melo` without a task will open it; today that
  starts the gateway);
- a web console;
- keeping the model's reasoning as memory notes, with switchable retrieval
  (`bm25`, `llm`, `embedding`).

## Security

> [!WARNING]
> Tools run with your account's permissions, and there is no sandbox: the
> model can run shell commands and Python code, and read or write any file
> you can. The gateway has no authentication, so keep it on `127.0.0.1` (the
> default). See [SECURITY.md](SECURITY.md) and
> [What the checks do](docs/tools.md#what-the-checks-do).

## Getting help

Search the [issues](https://github.com/localmelo/localmelo/issues) or open a
new one. Include the JSON line from `--output-format json` (it names the
localmelo version), the exit code, the backend and model, and the steps to
reproduce. Remove API keys and personal paths first. Report security problems
as described in [SECURITY.md](SECURITY.md), not in a public issue.

## Contributing

Bug reports, documentation fixes and small, focused pull requests are
welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and
the checks to run.

## License

MIT © The localmelo Authors. See [LICENSE](LICENSE).
