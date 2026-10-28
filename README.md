<p align="center">
  <img src="docs/assets/melo.png" alt="Melo, the localmelo mascot" width="200">
</p>

# localmelo

<p align="center">
  <a href="https://localmelo.github.io/localmelo/"><img alt="Website" src="https://img.shields.io/badge/Website-255EDB?style=for-the-badge"></a>
  <a href="https://localmelo.github.io/localmelo/en/quickstart.html"><img alt="Quickstart" src="https://img.shields.io/badge/Quickstart-255EDB?style=for-the-badge"></a>
  <a href="https://localmelo.github.io/localmelo/en/index.html#features"><img alt="Documentation: guides and reference" src="https://img.shields.io/badge/Docs-255EDB?style=for-the-badge"></a>
  <a href="https://localmelo.github.io/localmelo/en/developers.html"><img alt="Developer guide" src="https://img.shields.io/badge/Developers-255EDB?style=for-the-badge"></a>
  <a href="README.zh-CN.md"><img alt="简体中文" src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-255EDB?style=for-the-badge"></a>
  <br>
  <a href="https://localmelo.github.io/localmelo/en/updates.html#roadmap"><img alt="Status: pre-alpha, v0.1.0 due 2026-12-20" src="https://img.shields.io/badge/status-pre--alpha_%C2%B7_v0.1.0_due_2026--12--20-9A5A00?style=flat-square&labelColor=F0F3F7"></a>
  <a href="pyproject.toml"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-5C6573?style=flat-square&labelColor=F0F3F7&logo=python&logoColor=3776AB"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0F766E?style=flat-square&labelColor=F0F3F7"></a>
  <a href="docs/llms.txt"><img alt="For AI agents: llms.txt" src="https://img.shields.io/badge/for_agents-llms.txt-5C6573?style=flat-square&labelColor=F0F3F7"></a>
</p>

A local personal agent that runs tasks with tools and memory on your own model
server or a cloud API: from the shell, from scripts and benchmarks (one JSON
line and a fixed exit code per run), or over HTTP.

**Status:** pre-alpha. Direct mode, the interactive CLI and the gateway work
today; commands and APIs may change before v0.1.0 (due 2026-12-20).

## Install

```bash
git clone https://github.com/localmelo/localmelo.git
cd localmelo
uv sync --extra gateway
source .venv/bin/activate
```

Keep the directory name `localmelo`: the package is built from it. Without
uv, see [Install with pip](https://localmelo.github.io/localmelo/en/quickstart.html#install-with-pip).

## Quick start

Start a model server (here [Ollama](https://ollama.com), with
`ollama pull qwen3:8b`), then run a task:

```bash
melo --base-url http://127.0.0.1:11434/v1 --chat-model qwen3:8b "What is 6*7?"
```

Run `melo` in a terminal to chat and save a configuration on the first run;
`melo --serve` starts the gateway; run `melo --gateway-login` in a terminal
and open the one-use link within 60 seconds. The HTTP API and web workspace are at
`http://127.0.0.1:8401/`.
The interactive UI accepts the same [reasoning and recall options](docs/reference/cli.md#interactive-ui)
as a task, with saved `[agent]` and `[retrieval]` defaults; console sessions
also honor the saved reasoning mode (default `legacy`).

## Learn more

- [Website](https://localmelo.github.io/localmelo/): what localmelo is and
  what you can do with it.
- [Quickstart](https://localmelo.github.io/localmelo/en/quickstart.html):
  install, a first task, a saved configuration and the gateway, step by step.
- [Architecture](https://localmelo.github.io/localmelo/en/architecture.html):
  how the agent, its checks, memory and tools fit together.
- [Roadmap and updates](https://localmelo.github.io/localmelo/en/updates.html):
  the v0.1.0 plan and dated changes.
- [Memory design](https://localmelo.github.io/localmelo/en/memory-design.html):
  the memory design, parts of which are not built yet.
- [Developer guide](https://localmelo.github.io/localmelo/en/developers.html):
  the development setup, tests, checks and how to send a change.
- [Reference documentation](docs/README.md): every guide and reference page,
  from the command line to memory and the gateway.

## Security

> [!WARNING]
> Tools run with your account's permissions, and there is no sandbox: the
> model can run shell commands and Python code, and read or write any file you
> can. The gateway requires authentication and refuses non-loopback binds, but
> its token does not protect against malicious processes under your own OS user. See [SECURITY.md](SECURITY.md) and
> [what the checks do](docs/guides/tools-mcp.md#what-the-checks-do).

## Contributing

Bug reports, documentation fixes and small pull requests are welcome. The
[developer guide](https://localmelo.github.io/localmelo/en/developers.html)
has the setup, the tests and the checks to run.

## License

MIT © The localmelo Authors. See [LICENSE](LICENSE).
