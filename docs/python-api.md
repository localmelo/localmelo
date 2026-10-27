# Python API

Use localmelo from your own Python code: build an `Agent`, run tasks, read
their results, follow them as they run, stop them, and decide which tool
calls may run.

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

[Agent methods](#agent-methods) lists the other methods.

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
| `timeout_s` | `None` | A time limit for each run, in seconds. `None` means no limit. Without a `workspace_root`, `python_exec` code cannot be interrupted, so a run ends only when that code returns. |
| `workspace_root` | `None` | The directory the built-in tools work in, and the only one the file tools reach; see [Work in a directory](#work-in-a-directory). |
| `tool_timeout_s` | `None` | The time limit of `shell_exec` and `python_exec`, in seconds. Needs a `workspace_root`. `None` keeps 30 seconds for a shell command and 60 seconds for Python. |
| `record_answers` | `False` | `True` keeps each run's answer in working memory for the next run; see [Keep answers between runs](#keep-answers-between-runs). |

`AgentOptions` cannot be changed after it is created. A value that is not
valid raises `ValueError` and names the field:

```text
ValueError: max_steps must be an int >= 1, got 0
```

The [command-line options](cli.md#direct-mode) set the same fields, except
`steps_per_attempt` and `max_attempts`. Direct mode never sets
`workspace_root`, `tool_timeout_s` or `record_answers`.

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

## Follow a run with events

`await agent.run_task(query, on_event=sink)` calls `sink` once for each event
of the run, synchronously and in order: the start of each step, each reply of
the model, each tool call and its result, each reflection, and
`run.finished`, which carries the `RunResult`. The sink must not block.
[Events](events.md) lists the events and the rules for the sink.

- `observe=sink` instead of `on_event=sink` gives the sink the same events
  without making the run interactive, so a cancelled run still ends
  `crashed` (see [Stop a run](#stop-a-run)). Use it for a display that
  only watches.
- `progress=True`, with either, adds
  [`run.progress`](events.md#runprogress) events: the step, attempt, phase
  and test counts, and, for a task run before on this model, what past runs
  did. Without a sink it does nothing.
- `routine="name"` makes runs one routine in
  [procedural memory](procedures.md) whatever their wording and folder;
  without it, a routine is the query, dates and times masked, in the folder
  the run's tools work in. A blank name raises `ValueError`, before the run.

Before `run_task` returns, after `run.finished`, the run is recorded in
procedural memory: at most 2 seconds, usually a few milliseconds (see
[Memory](memory.md#how-memory-works)). A run that raises, such as a
cancelled one, is not recorded, and neither is one that ends after
`agent.begin_exit()`.

## Stop a run

To stop a run, cancel the task that awaits `run_task`, or let a scope cancel
it, such as `asyncio.timeout`, a `TaskGroup` or `asyncio.run` on Ctrl+C.
`run_task` then raises `CancelledError`; it never swallows a cancellation.

How the run ends depends on whether it is interactive, that is, whether it
has `on_event` or `approval` (`observe` does not count):

- **An interactive run** ends as `cancelled`. Its `RunResult` comes on the
  `run.finished` event, with an empty `result` and the reason as `error`:
  the message of `task.cancel(msg)` when it is a non-empty string, else
  `Cancelled by the caller`.
- **Any other run** ends as `crashed`, as in direct mode, and there is no
  `RunResult`.

```python
import asyncio

from localmelo import Agent, AgentEvent, RunResult
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


async def main() -> None:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm)
    finished: list[RunResult] = []

    def on_event(event: AgentEvent) -> None:
        if event.type == "run.finished":
            finished.append(RunResult(**event.data["result"]))

    try:
        run = asyncio.create_task(
            agent.run_task("Write a long story.", on_event=on_event)
        )
        await asyncio.sleep(2)
        run.cancel("Stopped by the user")
        try:
            await run
        except asyncio.CancelledError:
            pass
        print(finished[0].status, finished[0].error)
    finally:
        await agent.close()


asyncio.run(main())
```

If the model is still answering after 2 seconds, the script prints
`cancelled Stopped by the user`.

After an interactive cancellation:

- The task record is saved as `cancelled`, and history gets a row for each
  call of the step that was under way, so each call appears once.
- Working memory keeps the task and that step. A call that had not returned
  reads
  `The run was stopped before this call finished; it may have partly run.`,
  and a call that had not started, or that an approval policy was still
  deciding about, reads
  `The run was stopped before this call started; it did not run.` The next
  run on the agent sees this.
- Reflections are cleared without being promoted to long-term memory, so no
  network request follows the cancellation.
- A `shell_exec` command, or a `python_exec` child process with a workspace
  root, is killed with every process it started.
- This clean-up runs in a task of its own, which a second cancellation does
  not stop. Writing history and the task record may take up to 10 seconds.
  `run.finished` still comes, and `agent.close()` waits for the clean-up.

`timeout_s` still ends a run as `timeout`, interactive or not.

If your process must exit, as on SIGTERM, call
`agent.begin_exit(reason, within_s=...)` first, then cancel the run and
await `agent.close()`. The run ends with `reason` as its error, `cancelled`
when it is interactive and `crashed` otherwise, and its clean-up keeps to
the deadline: tool providers that do not close in time are killed, and
reflections are not promoted. Direct mode does this on SIGTERM and SIGHUP;
see [Signals](cli.md#signals).

## Approve tool calls

An approval policy decides whether a tool call may run. Pass one as
`approval` to `run_task` or `run_tool`. These tools require approval:

- the built-ins `shell_exec`, `python_exec` and `file_write`;
- a provided tool whose `ProvidedTool.requires_approval` is `True`;
- the tools of an MCP server whose entry sets `require_approval`; see
  [MCP config file](tools.md#mcp-config-file).

`file_read` does not: a [workspace root](#work-in-a-directory) limits it
instead. Without a policy, a tool that requires approval runs like any
other, as in direct mode, which never sets a policy.

A policy is any object with an `async def decide(request)` method that
returns an `ApprovalDecision`. This one asks in the terminal:

```python
import asyncio

from localmelo import Agent
from localmelo.melo.contracts import ApprovalDecision, ApprovalRequest
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


class AskInTerminal:
    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        question = f"Run {request.tool_name}: {request.preview}? [y/N] "
        answer = await asyncio.to_thread(input, question)
        if answer.strip().lower() == "y":
            return ApprovalDecision(True)
        return ApprovalDecision(False, "the user said no")


async def main() -> None:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm)
    try:
        result = await agent.run_task("Which files are here?", approval=AskInTerminal())
        print(result.status, result.result or result.error)
    finally:
        await agent.close()


asyncio.run(main())
```

When the run is cancelled while this policy waits, the `CancelledError` ends
`decide` at its `await` and propagates, as the rules below require, but the
question stays in the terminal until Enter is pressed. A policy with a prompt
it can take down, such as a dialog, must take it down.

`run_task` raises `TypeError` before the run starts when `approval` has no
`async def decide`. `AllowAllPolicy()` allows every call, and
`DenyAllPolicy(reason)` denies every call with that reason.

When the policy is asked:

- After every check that refuses a call by itself: the step must have
  offered the tool, the arguments must be valid, a shell command must pass
  the blocklist, and a file path must be inside the workspace root. So the
  policy is asked only about a call that would otherwise run.
- Only when the arguments match the tool's parameters. A missing argument,
  or one the tool does not take, is refused with
  `Arguments for <tool> do not match its parameters: <message>`, and the
  policy is not asked.
- Just before the tool runs. Its time to decide never counts against the
  tool's time limit, but it counts toward the run's `timeout_s`.
- Once per call. A denial refuses only that call; the step's other calls
  are asked about on their own.

The request (`ApprovalRequest`):

| Field | Meaning |
|---|---|
| `request_id` | A new random id for each request: 32 hex characters. |
| `run_id` | The run's `task_id`, or `""` for `run_tool`. |
| `call_id` | The tool call's id. |
| `tool_name` | The tool. |
| `arguments` | A copy of exactly what will run. A file tool's `path` is the resolved path inside the workspace root. |
| `preview` | A one-line summary of the call; see [tool.started](events.md#toolstarted). |
| `workspace_root` | The agent's workspace root, or `None`. |

The decision (`ApprovalDecision`):

| Field | Default | Meaning |
|---|---|---|
| `allow` | required | Whether the call may run. |
| `reason` | `""` | Why it may not, for the model. |
| `remember` | `False` | Allow this tool for the rest of the session. The run reports it on `tool.finished`, but does not act on it: a policy that grants a tool keeps the grant itself, and answers later requests with `origin="session"`. |
| `origin` | `"user"` | Who decided: `"user"`, `"session"` (a grant), `"policy"` or `"timeout"`. |

A field of the wrong type raises `TypeError`. `remember=True` or
`origin="session"` on a denial, and `origin="timeout"` on an allow, raise
`ValueError`.

A denied call does not run. The model reads one of these texts as the call's
error, and the run goes on:

| Origin | Without a reason | With a reason |
|---|---|---|
| `user` | `The user denied this call. <ending>` | `The user denied this call: <reason>. <ending>` |
| `policy` | `The approval policy denied this call. <ending>` | `The approval policy denied this call: <reason>. <ending>` |
| `timeout` | `The call was not approved in time, so it did not run. Silence is not consent: do not retry it without asking the user.` | The same. |

`<ending>` is
`Do not run it again or reach the same result another way unless the user asks.`
The reason loses surrounding whitespace and a final `.`, and keeps at most
500 characters. `tool.finished` reports the call as `blocked`, with the
error category `approval_denied`. The same denied call three steps in a row
ends the attempt as stuck.

The policy fails closed:

- A policy that raises an exception, returns something other than an
  `ApprovalDecision`, or raises `CancelledError` while the run is not being
  cancelled denies the call with
  `Approval failed, so the call did not run: <ExceptionType>: <message>`.
  The run goes on.
- When the run is cancelled while the policy decides, `decide` receives the
  `CancelledError`. It must withdraw its prompt and re-raise. The call does
  not run, and `tool.finished` reports the approval's origin as
  `withdrawn`.
- With a workspace root, when the policy allows a file tool's call, its
  path is resolved again. If the result has changed, for example because a
  symlink took the file's place while the policy decided, the call is
  refused with
  `Path '<path>' changed while waiting for approval; the call did not run.`

## Work in a directory

`AgentOptions(workspace_root=DIR)` gives the built-in tools a directory:

```python
import asyncio
import os

from localmelo import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


async def main() -> None:
    os.makedirs(os.path.expanduser("~/melo-work"), exist_ok=True)
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    options = AgentOptions(workspace_root="~/melo-work", tool_timeout_s=600)
    agent = Agent(llm=llm, options=options)
    try:
        print(agent.workspace_root)
        result = await agent.run_tool("shell_exec", {"command": "pwd"})
        print(result.output)
    finally:
        await agent.close()


asyncio.run(main())
```

Both lines print the real path of `~/melo-work`, such as
`/Users/you/melo-work`.

- The agent resolves the root to its real path, with `~` expanded. A root
  that is not a directory raises
  `ValueError: workspace_root is not a directory: '<root>'` when the agent is
  built.
- **`file_read` and `file_write`** resolve a relative path against the root
  and reach only files inside it. `..` and symlinks are resolved first: a
  symlink inside the root that points out is refused, and a symlink outside
  that points in is allowed. A path outside is refused with
  `Path '<path>' is outside allowed workspace root`, and an empty path with
  `Argument 'path' for <tool> must be a non-empty string without NUL characters.`
  They open the resolved path without following symlinks, so a symlink that
  takes the file's place after the check is refused, and they refuse
  anything but a regular file, such as a FIFO, with
  `Not a regular file: <path>`. `file_read` reads at most 200,000 bytes, as
  UTF-8. A file over 200,000 bytes, or whose text is longer than 50,000
  characters, is cut to 50,000 characters, which end with
  `... [truncated: the file has <size> bytes]`. `file_write` reports the
  resolved path.
- **`shell_exec` and `python_exec`** start in the root, each in a child
  process with a process group of its own, which a timeout or a
  cancellation kills as a whole. A shell command's standard input is
  `/dev/null`. Python code runs in a child Python, so a time limit can stop
  it, and `sys.exit()` or `os.chdir()` stays in the child.
- **They can still reach any path.** Only the file tools are limited to the
  root. Use an [approval policy](#approve-tool-calls) to control
  `shell_exec` and `python_exec`.
- **Exit status.** When a shell command exits with a non-zero status, its
  text starts with the line `[exit code <n>]`, and the call still succeeds:
  the model reads the status even when the output is cut. Python code that
  exits with a non-zero status fails with
  `Python exited with code <n>`, followed by the last 4,000 characters of
  its output.
- **Time limits.** `tool_timeout_s` sets the time limit of `shell_exec` and
  `python_exec`, for builds and test suites that take longer than the
  defaults of 30 seconds for a shell command and 60 seconds for Python. A
  shell command that runs too long fails with
  `Command timed out after <n> s`. `tool_timeout_s` without a
  `workspace_root` raises `ValueError`.
- **The model is told.** Each planning request says:
  `Working directory: <root>. Relative paths and shell and Python commands run there; file_read and file_write refuse paths outside it.`
- `RunResult.config` does not include the root: `agent.workspace_root` and
  the `run.started` event give it.

Without a root, the built-in tools behave as in direct mode, as described in
[Tools and MCP](tools.md#built-in-tools).

## Keep answers between runs

By default, working memory keeps the tasks and tool calls of earlier runs on
the agent, but not their answers. With `AgentOptions(record_answers=True)`,
each run ends by appending its answer to working memory, so the next run sees
the earlier questions and answers as a conversation:

```python
import asyncio

from localmelo import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


async def main() -> None:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm, options=AgentOptions(record_answers=True))
    try:
        await agent.run_task("What is the capital of France?")
        for message in agent.conversation():
            print(message.role, message.content)
    finally:
        await agent.close()


asyncio.run(main())
```

It prints, for example:

```text
user What is the capital of France?
assistant Paris is the capital of France.
```

- The answer is the run's `result` without its reasoning: the first
  `<think>…</think>` block, in any letter case, is left out, and so is the
  rest of the text after a `<think>` at the start that is never closed, or
  the text before a `</think>` that has no `<think>` before it (templates
  that open the block in the prompt, such as DeepSeek-R1's and Qwen3's
  thinking models, leave only the closing tag). Ollama, and
  OpenAI-compatible servers without a reasoning parser, put the reasoning
  there. The `RunResult` keeps the whole text.
- An answer longer than 8,000 characters is cut and ends with
  `... [<n> more characters not shown]`.
- A run without an answer leaves a label instead:
  `[interrupt: no assistant output for this turn]` for a `cancelled` run,
  and `[no answer: the run ended with status <status>]` for any other.

To keep a conversation in a file, across processes and restarts, see
[Sessions](sessions.md).

## Agent methods

| Method | What it does |
|---|---|
| `await agent.run_task(query, *, on_event=None, approval=None, observe=None, progress=False, routine=None)` | Runs one task and returns its `RunResult`; see [Results](#results) and [Follow a run with events](#follow-a-run-with-events). |
| `await agent.run(query)` | Runs one task and returns its text. |
| `await agent.start()` | Starts the tool providers and registers their tools now, instead of at the first run. A second call does nothing. It raises the setup's error. |
| `agent.list_tools()` | The registered tools, sorted by name, as `ToolInfo`: `name`, `description`, `parameters`, `source` (`"builtin"`, `"provided"` or `"custom"`) and `requires_approval`. |
| `await agent.run_tool(name, arguments, *, approval=None)` | Runs one tool call outside a run and returns its `ToolResult`. |
| `agent.conversation()` | A copy of the working-memory window, as a list of `Message`. |
| `agent.load_conversation(messages)` | Replaces the working-memory window and returns the number of messages kept. |
| `agent.workspace_root` | The real path of the workspace root, or `None`. |
| `agent.begin_exit(reason, *, within_s)` | Prepares for the process to exit; see [Stop a run](#stop-a-run). |
| `await agent.close()` | Stops the tool providers and closes the model connections and memory files. |

- **Start the agent first** when you may cancel its first run: a
  cancellation while the tool providers start fails their setup for the
  agent's life, and later runs end as `setup_error`.
- **`run_tool`** passes the call through the same checks as a model's call,
  except that any registered tool may run: the arguments, the blocklist,
  the workspace root, the approval policy, the tool's time limit and the
  output limit. A refused or failed call returns its reason as `error`. The
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
shell_exec builtin True
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

- The file uses the format in [MCP config file](tools.md#mcp-config-file).
  `load_mcp_config` raises `McpConfigError` for a file that cannot be read or
  is not valid.
- The first run starts the providers and registers their tools next to the
  built-ins.
- If a provider cannot start, or one of its tools has the name of a tool that
  is already registered, the run ends as `setup_error`. Later runs on that
  agent end the same way.
- `agent.close()` stops the MCP servers.
- A provided tool with `requires_approval=True` asks the run's approval
  policy before each call; see [Approve tool calls](#approve-tool-calls).
  For an MCP server, set `require_approval` in its entry.

## Custom model providers

- The chat request timeout belongs to the provider: pass `timeout=` (in
  seconds) to `OpenAICompatLLM` or to `OllamaNativeChat`
  (`localmelo.support.providers.llm.ollama_chat`). With
  `Agent(config=...)`, `request_timeout_s` sets it.
- To connect another kind of server, subclass `BaseLLMProvider` (or
  `BaseEmbeddingProvider`) from `localmelo.melo.contracts.providers`, and
  implement `chat()` and `close()` (or `embed()` and `close()`).
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
