# Python API

Use localmelo from your own Python code: build an `Agent`, run tasks, read
their results, follow them as they run, stop them, and decide which tool
calls may run.

> [!NOTE]
> The Python API is experimental and may change before v0.1.0.

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

[Agent methods](#agent-methods) lists the other methods.

## AgentOptions

| Field | Default | Description |
|---|---|---|
| `max_steps` | `30` | The most planning steps per run. |
| `attempt_policy` | `"reflect"` | `"reflect"`: attempts with a reflection after each. `"none"`: one attempt. |
| `steps_per_attempt` | `10` | The most steps per attempt, with `"reflect"`. |
| `max_attempts` | `5` | The most attempts per run, with `"reflect"`. |
| `reflection_gate` | `"patient"` | `"patient"`, `"flags"` or `"utility"`, with `"reflect"`; see [Reflection](../concepts/how-it-works.md#reflection). |
| `answer_on_stop` | `True` | `True`: a run that ends `gate_stopped` or `budget_exhausted` with no answer makes one more request, without tools, for the answer from the evidence gathered, and an answer that lacks the closing line the task asks for gets it from one more; `False` sends neither. See [Run results](run-results.md#statuses). |
| `max_calls` | `None` | An int >= 2: the most model requests a run sends; before the one that would pass it the run ends `budget_exhausted` (with `answer_on_stop`, the last call is kept for the answer). `None`: no limit. See [Run results](run-results.md#statuses). |
| `tool_exposure` | `"retrieve"` | `"retrieve"` or `"all"`; see [How tools are offered](tools.md#how-tools-are-offered). |
| `builtins` | `None` | The built-in tools to register: `None` for the mode's default set, an empty collection for none, or a collection of names. |
| `memory` | `True` | `False` turns long-term memory off. |
| `memory_dir` | `None` | A directory for SQLite memory, with `~` expanded. `None` follows the environment variables; see [Memory](../guides/memory.md#keep-memory-between-runs). |
| `timeout_s` | `None` | A time limit for each run, in seconds. `None` means no limit. It also interrupts a `python_exec` call, whose child process it kills. |
| `mode` | `"python"` | The surface the run belongs to: `"direct"`, `"interactive"`, `"gateway"` or `"python"`. It decides what `builtins=None` registers: only `"interactive"` offers `shell_exec`; `"interactive"` and `"gateway"` also offer `list_files`; `"direct"` and `"python"` register `file_read`, `file_write` and `python_exec`. Naming `shell_exec` in `builtins` is a `direct` or `python` caller's opt-in and lets the run use it; the checker refuses an unnamed one there, and refuses it in `gateway` mode even when named; the refusal names the mode and, in direct or python mode, the way to enable it. An explicit `builtins` wins over the default registration. |
| `workspace_root` | `None` | The directory the built-in tools work in, and the only one the file tools reach; see [Work in a directory](../guides/python.md#work-in-a-directory). |
| `tool_timeout_s` | `None` | The time limit of `shell_exec` and `python_exec`, in seconds. Needs a `workspace_root`. `None` keeps 30 seconds for a shell command and 60 seconds for Python. |
| `tool_env` | `None` | Extra variables for the `shell_exec` and `python_exec` children, on top of the minimal environment; see [Work in a directory](../guides/python.md#work-in-a-directory). |
| `record_answers` | `False` | `True` keeps each run's answer in working memory for the next run; see [Keep answers between runs](../guides/python.md#keep-answers-between-runs). |
| `reasoning` | `"legacy"` | How a reply's reasoning is sent back: `"legacy"`, `"drop"`, `"replay"`, `"compress"` or `"compress+replay"`; see [Reasoning modes](cli.md#direct-mode). |
| `replay_turns` | `None` | With `"replay"` or `"compress+replay"`, the newest N assistant turns whose reasoning is replayed; `None` keeps every turn in scope. |
| `replay_scope` | `"task"` | `"task"` replays only the current task's turns; `"window"` replays every turn in the window, earlier tasks included. API-only. |
| `note_writer` | `"auto"` | In the compress modes, `"auto"` reads the model's `<note>` block, else the heuristic; `"heuristic"` always writes the note from the reply's text and thinking. |

`AgentOptions` cannot be changed after it is created. A value that is not
valid raises `ValueError` and names the field:

```text
ValueError: max_steps must be an int >= 1, got 0
```

The [command-line options](cli.md#direct-mode) set the same fields, except
`steps_per_attempt`, `max_attempts`, `replay_scope` and `mode` (the CLI sets
the surface's mode). Direct mode never sets `workspace_root`,
`tool_timeout_s`, `tool_env` or `record_answers`.

## Results

- `await agent.run_task(query)` returns a `RunResult`.
  [Run results](run-results.md) describes its statuses and fields, and
  `result.to_dict()` returns them as a dict, in the same order.
- `await agent.run(query)` returns text: the result, or `[<status>] <error>`
  when the result is empty. The gateway returns this text.

`Agent`, `AgentOptions`, `AgentEvent`, `RunResult`, `Message`, `ToolCall`,
`ToolDef` and `ToolResult` can be imported from `localmelo`. The event and
approval types, such as `ApprovalPolicy`, `ApprovalRequest`,
`ApprovalDecision`, `AllowAllPolicy` and `DenyAllPolicy`, come from
`localmelo.melo.contracts`.

## Agent methods

| Method | What it does |
|---|---|
| `await agent.run_task(query, *, on_event=None, approval=None, observe=None, progress=False, routine=None)` | Runs one task and returns its `RunResult`; see [Results](#results) and [Follow a run with events](../guides/python.md#follow-a-run-with-events). |
| `await agent.run(query)` | Runs one task and returns its text. |
| `await agent.start()` | Starts the tool providers and registers their tools now, instead of at the first run. A second call does nothing. It raises the setup's error. |
| `agent.list_tools()` | The registered tools, sorted by name, as `ToolInfo`: `name`, `description`, `parameters`, `source` (`"builtin"`, `"provided"` or `"custom"`), `requires_approval`, and `tier` (`"read"`, `"write"`, `"exec"` or `"network"`; see [Permission tiers](../guides/tools-mcp.md#permission-tiers)). |
| `await agent.run_tool(name, arguments, *, approval=None)` | Runs one tool call outside a run and returns its `ToolResult`. |
| `agent.conversation()` | A copy of the working-memory window, as a list of `Message`. |
| `agent.load_conversation(messages)` | Replaces the working-memory window and returns the number of messages kept. |
| `agent.workspace_root` | The real path of the workspace root, or `None`. |
| `agent.begin_exit(reason, *, within_s)` | Prepares for the process to exit; see [Stop a run](../guides/python.md#stop-a-run). |
| `await agent.close()` | Stops the tool providers and closes the model connections and memory files. |

- **Start the agent first** when you may cancel its first run: a
  cancellation while the tool providers start fails their setup for the
  agent's life, and later runs end as `setup_error`.
- **`run_tool`** passes the call through the same checks as a model's call,
  except that any registered tool may run: the arguments, the mode's
  `exec` refusal, the workspace root, the approval policy, the
  tool's time limit and the output limit. A refused or failed call returns
  its reason as `error`. The
  call writes nothing to working memory or history and emits no event. For
  the policy, `run_id` is `""`.
- **`load_conversation`** clears working memory and its reflection, then
  adds the messages. A tool call goes in only together with results that
  answer each of its calls once; an incomplete call, or a tool result
  without its call, is dropped with a WARNING. `[]` clears the
  conversation. Use it with `conversation()` to move a conversation to a
  new agent.
- `run_tool` and `load_conversation` raise `RuntimeError` while a run is
  under way, including the clean-up of a cancelled interactive run.

```python
import asyncio

from localmelo import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


async def main() -> None:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm, options=AgentOptions(workspace_root="."))
    try:
        await agent.start()
        for tool in agent.list_tools():
            print(tool.name, tool.source, tool.requires_approval)
    finally:
        await agent.close()


asyncio.run(main())
```

It prints:

```text
file_read builtin False
file_write builtin True
python_exec builtin True
```

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

- The file uses the format in [MCP config file](../guides/tools-mcp.md#mcp-config-file).
  `load_mcp_config` raises `McpConfigError` for a file that cannot be read or
  is not valid.
- The first run starts the providers and registers their tools next to the
  built-ins.
- If a provider cannot start, or one of its tools has the name of a tool that
  is already registered, the run ends as `setup_error`. Later runs on that
  agent end the same way.
- `agent.close()` stops the MCP servers.
- Each MCP tool's raw `annotations` object is kept on its
  `ToolDef.annotations`, and the checker derives the tool's tier from it;
  see [Permission tiers](../guides/tools-mcp.md#permission-tiers). A `ToolDef` you build
  yourself starts with `{}`, which is `write`.
- A provided tool with `requires_approval=True` asks the run's approval
  policy before each call; see [Approve tool calls](../guides/python.md#approve-tool-calls).
  For an MCP server, set `require_approval` in its entry.

## Custom model providers

- The chat request timeout belongs to the provider: pass `timeout=` (in
  seconds) to `OpenAICompatLLM` or to `OllamaNativeChat`
  (`localmelo.support.providers.llm.ollama_chat`). With
  `Agent(config=...)`, `request_timeout_s` sets it.
- To connect another kind of server, subclass `BaseLLMProvider` (or
  `BaseEmbeddingProvider`) from `localmelo.melo.contracts.providers`, and
  implement `chat()` and `close()` (or `embed()` and `close()`).
- An optional `stream_chat(messages, tools=None)` returns an async generator of
  `StreamFragment` values: provisional `content` and `reasoning` text, `tool`
  indices (never partial arguments), `usage`, then one `reply` containing a
  completed `Message` and `finish_reason`. Providers without streaming call
  `chat()` and yield one `reply` with an empty `finish_reason`. `OpenAICompatLLM`
  sends `stream=true` only for this method and validates complete tool calls
  before returning them. A server that does not stream still answers: a
  response that is one JSON body is read as the completion `chat()` returns, and
  a stream that ends before it delivered anything is asked for again, once,
  without `stream`. The `reply` then has `whole=True`, and the provider's
  `sends_whole_replies` is true, so it does not ask that server to stream again;
  a stream that breaks after a fragment was yielded keeps its error. Usage is
  returned only if the server sends it; where
  supported, request it with `extra_body={"stream_options": {"include_usage": True}}`.
  Failures raise `ProviderError`; after any fragment is yielded, transport or
  premature-end errors are not retryable. Cancellation propagates. Close the generator
  with `aclose()` on early exit. Deltas and the final message can contain
  sensitive text: do not persist raw deltas, and do not display them directly.
  `localmelo.melo.checker.stream_filter.StreamFilter` is the separate display
  policy. Without a declared framing hint it buffers the whole response; with
  `reasoning_tags="explicit"`, `"closing_only"` or `"none"`, it can release
  redacted complete lines. An unreviewed redaction rule change forces whole
  buffering. These hints are trusted guarantees for a model and template, not
  guesses: `none` treats tags literally, `closing_only` treats everything before
  the first closing tag as thinking, and `explicit` recognizes a leading opening
  tag. `split_thinking(..., reasoning_tags=...)` uses the same rule for the
  completed reply; omitting it keeps the existing whole-reply behavior. A trusted
  separate reasoning field can instead use `StreamFilter(separate_reasoning=True)`
  and `feed_reasoning()`; it never enters content. Each `feed()` and
  `feed_reasoning()` returns `(thinking_delta, content_delta)`; `finish()` returns
  the remaining tails. For each completed channel, concatenate its deltas in
  order, including the tail. Thinking is held until finish.
  Redaction rules are snapshotted at stream start; changes apply to the next
  stream. At commit, apply `redact_for_storage()` separately to each concatenated
  display channel: it applies the snapshot and then the current rules, so newer
  rules can add masking. Redaction is pattern-based and does not guarantee secret
  removal. The interactive terminal UI is the only consumer (`[cli] stream =
  true`): it streams planning requests only, draws released text in its live
  area and stores the completed reply, never the deltas; existing `chat()`
  behavior is unchanged.
  Callers must not silently retry a stream after showing provisional output.
  Call `finish()` only after a completed reply; call `abort()` on error or
  cancellation to discard buffered text (already released text cannot be recalled).
- Set `model` on a chat provider to the id of the model it uses, as the
  built-in providers do. Procedural memory keeps what runs did per model;
  with the default `""`, or `"default"`, past runs' facts are shown but a
  bar never would be. Keep `model` a plain attribute, or a property with a
  setter, as a wrapper that forwards its inner provider's id may: mypy
  refuses a read-only property in place of `BaseLLMProvider.model`.
- Raise `ProviderError(message, status_code=..., retryable=True)` from
  `chat()` for a failure worth retrying. The agent then retries it like the
  built-in providers do; see
  [Provider errors and retries](run-results.md#provider-errors-and-retries).
