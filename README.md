<p align="center">
  <img src="docs/assets/melo.png" alt="Melo" width="240">
</p>

# localmelo

[English](./README.md) | [简体中文](./README.zh-CN.md)

`localmelo` is a local-first agent runtime for explicit memory, checked tool use,
and user-managed model backends. It is designed to keep the core agent loop
separate from infrastructure so local and cloud backends can be configured
without changing runtime behavior.

**Status:** pre-alpha. The online core loop is usable for development and smoke
testing, but public APIs, memory policy, and personalization workflows are still
evolving.

**[Docs](https://localmelo.github.io/localmelo/en/index.html)** |
**[Quickstart](https://localmelo.github.io/localmelo/en/quickstart.html)** |
**[Architecture](https://localmelo.github.io/localmelo/en/architecture.html)** |
**[Updates](https://localmelo.github.io/localmelo/en/updates.html)**

## What It Does

localmelo provides:

- an online agent loop with planning, tool execution, reflection, and memory writeback
- explicit memory layers for working context, history, long-term retrieval, and tool lookup
- checker boundaries for ingress, planning, execution, output size, and memory writes
- backend connectors for local and cloud model providers
- gateway mode for HTTP sessions around the same runtime
- scaffolding for a later sleep-time personalization pipeline

It does not manage local model runtimes. Start Ollama, MLC, vLLM, SGLang, or other
local services yourself, then point localmelo at the configured endpoint.

## Current Maturity

Ready to verify:

- direct CLI mode
- gateway mode
- backend registry and provider construction
- built-in tool execution with workspace policy
- async SQLite history and long-term memory persistence
- focused tests and smoke checks

Still experimental:

- long-memory retrieval and promotion policy
- product-level deployment shell
- sleep-time training, evaluation, and personalization
- stable public API guarantees

## Quickstart

Requirements:

- Python 3.11+
- a configured chat backend such as Ollama, MLC, vLLM, SGLang, or a supported cloud provider

Install from the repository root:

```bash
pip install -e ".[dev,gateway]"
```

Configure providers:

```bash
melo --reconfigure
```

Run one direct query:

```bash
melo "What is 6*7?"
```

Run the gateway:

```bash
melo --serve
```

Call the gateway:

```bash
curl http://127.0.0.1:8401/v1/health

curl -X POST http://127.0.0.1:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query":"Say hello briefly"}'
```

See the [Quickstart](https://localmelo.github.io/localmelo/en/quickstart.html) for
gateway smoke checks, session reuse, no-embedding mode, and backend-specific
verification.

## Architecture

The main design rule is:

> `melo/` owns runtime behavior and contracts. `support/` owns infrastructure.
> Core runtime code should not import infrastructure implementations directly.

```text
localmelo/
  melo/       # agent, memory, checker, executor, sleep scaffolding
  support/    # backends, providers, gateway, config, onboarding
  tests/      # regression, integration, and smoke coverage
  docs/       # static documentation site
```

Core modules:

| Area | Responsibility |
|---|---|
| Agent | Retrieval, planning, tool calls, reflection, final answers |
| Memory | Working memory, history, long-term storage, tool registry |
| Checker | Structured validation across runtime boundaries |
| Executor | Built-in tool execution with timeout and workspace policy |
| Support | Backend registry, provider implementations, config, gateway |

For the full runtime map, see the
[Architecture docs](https://localmelo.github.io/localmelo/en/architecture.html).

## Safety Boundaries

localmelo treats checks as runtime control flow. Requests, plans, tool calls,
executor results, and memory writes are validated before they cross major
boundaries.

Current safeguards include:

- workspace policy for file operations
- blocked shell command patterns
- output size limits
- prompt and memory write size limits
- provider probing during onboarding

This is still pre-alpha software. Review configuration and tool permissions
before giving it access to important files, accounts, or long-running processes.

## Backends

Supported backend families:

| Backend | Type | Notes |
|---|---|---|
| `ollama` | local | User-managed Ollama server |
| `mlc` | local | User-managed MLC endpoint |
| `vllm` | local | User-managed vLLM endpoint |
| `sglang` | local | User-managed SGLang endpoint |
| `openai` | cloud | OpenAI API |
| `anthropic` | cloud | Anthropic API |
| `gemini` | cloud | Google Gemini API |
| `nvidia` | cloud | NVIDIA API |

Chat and embedding backends are configured separately. Set the embedding backend
to `none` when you do not want to run embedding retrieval.

## Development

Install development dependencies:

```bash
pip install -e ".[dev,gateway]"
```

Run focused checks:

```bash
python -m pytest tests/agent tests/checker tests/executor \
                 tests/cli tests/memory tests/integration -q
python -m pytest tests/gateway -q
python -m ruff check .
```

Run the full suite before merging:

```bash
python -m pytest tests/ -q
```

Development priorities are tracked in
[Updates](https://localmelo.github.io/localmelo/en/updates.html) and GitHub issues.

## Contributing

Issues and pull requests are welcome. Because the project is still pre-alpha,
small focused changes are easier to review than broad feature drops.

Please keep changes aligned with the runtime/infrastructure boundary:

- put core agent behavior in `melo/`
- put backend, gateway, config, and onboarding implementation in `support/`
- preserve checker-based control flow for safety boundaries
- add tests for behavior that changes public commands, storage, tool execution, or provider contracts
