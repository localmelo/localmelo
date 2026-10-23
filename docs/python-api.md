# Python API

Use localmelo from your own Python code: build an `Agent`, run tasks, and
read their results.

> [!NOTE]
> The Python API is experimental and may change before v0.1.0.

## Run a task

```python
import asyncio

from localmelo import Agent, AgentOptions
from localmelo.support.config import load


async def main() -> None:
    agent = Agent(config=load(), options=AgentOptions(max_steps=10))
    try:
        result = await agent.run_task("What is 6*7?")
        print(result.status, result.result or result.error)
    finally:
        await agent.close()


asyncio.run(main())
```

`load()` reads `~/.cache/localmelo/config.toml`, so save a configuration
first ([Configuration](configuration.md)). The script prints the status and
the answer, such as `completed 42`.

## Agent

There are two ways to build an agent:

- `Agent(config=load())` uses the backends of a configuration, as `melo` and
  the gateway do. `Config`, `load` and `save` come from
  `localmelo.support.config`.
- `Agent(llm=..., embedding=...)` uses model providers that you build. The
  embedding provider is optional; without it there is no long-term memory.

```python
from localmelo import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
agent = Agent(llm=llm, options=AgentOptions(builtins=["file_read"], timeout_s=60))
```

- Both forms take `options=AgentOptions(...)` and `tool_providers=[...]`.
- An agent runs one task at a time. Tasks on the same agent share its working
  memory.
- `await agent.close()` stops the tool providers and closes the model
  connections and memory files. Call it when you are done.

## AgentOptions

| Field | Default | Description |
|---|---|---|
| `max_steps` | `30` | The most planning steps per run. |
| `attempt_policy` | `"reflect"` | `"reflect"`: attempts with a reflection after each. `"none"`: one attempt. |
| `steps_per_attempt` | `10` | The most steps per attempt, with `"reflect"`. |
| `max_attempts` | `5` | The most attempts per run, with `"reflect"`. |
| `reflection_gate` | `"flags"` | `"flags"` or `"utility"`, with `"reflect"`; see [Reflection](how-it-works.md#reflection). |
| `tool_exposure` | `"retrieve"` | `"retrieve"` or `"all"`; see [How tools are offered](tools.md#how-tools-are-offered). |
| `builtins` | `None` | The built-in tools to register: `None` for all, an empty collection for none, or a collection of names. |
| `memory` | `True` | `False` turns long-term memory off. |
| `memory_dir` | `None` | A directory for SQLite memory, with `~` expanded. `None` follows the environment variables; see [Memory](memory.md#keep-memory-between-runs). |
| `timeout_s` | `None` | A time limit for each run, in seconds. `None` means no limit. |

`AgentOptions` cannot be changed after it is created. A value that is not
valid raises `ValueError` and names the field:

```text
ValueError: max_steps must be an int >= 1, got 0
```

The [command-line options](cli.md#direct-mode) set the same fields, except
`steps_per_attempt` and `max_attempts`.

## Results

- `await agent.run_task(query)` returns a `RunResult`.
  [Run results](run-results.md) describes its statuses and fields, and
  `result.to_dict()` returns them as a dict, in the same order.
- `await agent.run(query)` returns text: the result, or `[<status>] <error>`
  when the result is empty. The gateway returns this text.

`Agent`, `AgentOptions`, `RunResult`, `Message`, `ToolCall`, `ToolDef` and
`ToolResult` can be imported from `localmelo`.

## Tool providers and MCP

Tool providers add tools to an agent. `McpToolProvider` adds the tools of MCP
servers:

```python
from localmelo import Agent
from localmelo.support.config import load
from localmelo.support.mcp import McpToolProvider, load_mcp_config

servers = load_mcp_config("mcp.json")
agent = Agent(config=load(), tool_providers=[McpToolProvider(servers)])
```

- The file uses the format in [MCP config file](tools.md#mcp-config-file).
  `load_mcp_config` raises `McpConfigError` for a file that cannot be read or
  is not valid.
- The first run starts the providers and registers their tools next to the
  built-ins.
- If a provider cannot start, or one of its tools has the name of a tool that
  is already registered, the run ends as `setup_error`. Later runs on that
  agent end the same way.
- `agent.close()` stops the MCP servers.

## Custom model providers

- The chat request timeout belongs to the provider: pass `timeout=` (in
  seconds) to `OpenAICompatLLM` or to `OllamaNativeChat`
  (`localmelo.support.providers.llm.ollama_chat`). With
  `Agent(config=...)`, `request_timeout_s` sets it.
- To connect another kind of server, subclass `BaseLLMProvider` (or
  `BaseEmbeddingProvider`) from `localmelo.melo.contracts.providers`, and
  implement `chat()` and `close()` (or `embed()` and `close()`).
- Raise `ProviderError(message, status_code=..., retryable=True)` from
  `chat()` for a failure worth retrying. The agent then retries it like the
  built-in providers do; see
  [Provider errors and retries](run-results.md#provider-errors-and-retries).
