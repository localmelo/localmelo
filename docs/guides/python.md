# Guide: Python

This guide shows how to use localmelo from your own Python code: run a task, follow its events, stop a run, approve tool calls, work in a directory and keep answers between runs. Read it when you write a Python script; the classes and their fields are in the [Python API reference](../reference/python-api.md).

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

- `load()` reads `~/.cache/localmelo/config.toml`, so save a configuration first ([Configuration](../reference/configuration.md)).
- The script prints the status and the answer, such as `completed 42`.

## Follow a run with events

`await agent.run_task(query, on_event=sink)` calls `sink` once for each event of the run, synchronously and in order. The sink must not block; [Events](../reference/events.md) lists the events and the rules for the sink.

- The events are the start of each step, each reply of the model, each tool call and its result, each reflection, and `run.finished`, which carries the `RunResult`.

| Argument | What it does |
|---|---|
| `observe=sink` | Instead of `on_event`: the same events without making the run interactive, so a cancelled run still ends `crashed` (see [Stop a run](#stop-a-run)). For a display that only watches. |
| `progress=True` | With either: adds [`run.progress`](../reference/events.md#runprogress) events (step, attempt, phase and test counts, and, for a task run before on this model, what past runs did). Without a sink it does nothing. |
| `routine="name"` | Makes runs one routine in [procedural memory](../concepts/procedures.md) whatever their wording and folder. Without it, a routine is the query, dates and times masked, in the folder the run's tools work in. A blank name raises `ValueError`, before the run. |

- Before `run_task` returns, after `run.finished`, the run is recorded in procedural memory: at most 2 seconds, usually a few milliseconds (see [Memory](../concepts/memory.md#how-memory-works)).
- A run that raises, such as a cancelled one, is not recorded, and neither is one that ends after `agent.begin_exit()`.

## Stop a run

Cancel the task that awaits `run_task`, or let a scope cancel it: `asyncio.timeout`, a `TaskGroup` or `asyncio.run` on Ctrl+C. `run_task` then raises `CancelledError`; it never swallows a cancellation.

How the run ends depends on whether it is interactive, that is, whether it has `on_event` or `approval` (`observe` does not count):

| Run | Ends as | `RunResult` |
|---|---|---|
| Interactive | `cancelled` | On the `run.finished` event, with an empty `result` and the reason as `error`: the message of `task.cancel(msg)` when it is a non-empty string, else `Cancelled by the caller`. |
| Any other | `crashed`, as in direct mode | None. |

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

If the model is still answering after 2 seconds, the script prints `cancelled Stopped by the user`.

After an interactive cancellation:

| What | State |
|---|---|
| The task record | Saved as `cancelled`; history gets a row for each call of the step that was under way, so each call appears once. |
| Working memory | Keeps the task and that step. A call that had not returned reads `The run was stopped before this call finished; it may have partly run.` A call that had not started, or that an approval policy was still deciding about, reads `The run was stopped before this call started; it did not run.` The next run on the agent sees this. |
| Reflections | Cleared without being promoted to long-term memory, so no network request follows. |
| Tool processes | A `shell_exec` command, or a `python_exec` child process, is killed with every process it started. |
| The clean-up | Runs in a task of its own, which a second cancellation does not stop. Writing history and the task record may take up to 10 seconds. `run.finished` still comes, and `agent.close()` waits for it. |

- `timeout_s` still ends a run as `timeout`, interactive or not.
- If your process must exit, as on SIGTERM, call `agent.begin_exit(reason, within_s=...)` first, then cancel the run and await `agent.close()`.
- The run then ends with `reason` as its error, `cancelled` when it is interactive and `crashed` otherwise; tool providers that do not close in time are killed, and reflections are not promoted.
- Direct mode does this on SIGTERM and SIGHUP; see [Signals](../reference/cli.md#signals).

## Approve tool calls

An approval policy decides whether a tool call may run. Pass one as `approval` to `run_task` or `run_tool`. A built-in or provided tool is asked about when its permission tier is not `read` (see [Permission tiers](tools-mcp.md#permission-tiers)):

| Tool | Asked? |
|---|---|
| The built-ins the run offers | `python_exec` (`exec`) and `file_write` (`write`) in the default `python` mode; `shell_exec` (`exec`) too when the caller names it in `builtins`. No mode offers `shell_exec` by default except `interactive`. |
| A provided tool | With no annotations, or annotations that make it `write` or `network`; or with `ProvidedTool.requires_approval` set to `True`, which also makes a `read` tool ask. |
| An MCP server's tools | When its entry sets `require_approval`; see [MCP config file](tools-mcp.md#mcp-config-file). |
| `file_read`, `list_files` | No: they are `read`, and a [workspace root](#work-in-a-directory) limits them instead. |
| A read-only MCP tool | No (`readOnlyHint: true`), unless `require_approval` names it. |
| `agent.executor.register` | Keeps its explicit `requires_approval`; its tier is a label only. |

- The interactive UI always has a policy (unless `--yolo`); direct mode, the gateway and a Python run without `approval=` have none, which is why those modes do not offer `shell_exec` by default.
- Without a policy, a tool that would ask runs like any other, as in direct mode.
- A `direct` or `python` caller who names `shell_exec` in `builtins` asks for it explicitly, and without a policy it runs unapproved; `gateway` still refuses it.

A policy is any object with an `async def decide(request)` method that returns an `ApprovalDecision`. This one asks in the terminal:

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

- When the run is cancelled while this policy waits, the `CancelledError` ends `decide` at its `await` and propagates, but the question stays in the terminal until Enter is pressed.
- A policy with a prompt it can take down, such as a dialog, must take it down.
- `run_task` raises `TypeError` before the run starts when `approval` has no `async def decide`.
- `AllowAllPolicy()` allows every call, and `DenyAllPolicy(reason)` denies every call with that reason.

When the policy is asked:

- After every check that refuses a call by itself: the step must have offered the tool, the arguments must be valid, an `exec` tool the mode does not offer must have been named in `builtins`, and a file path must be inside the workspace root. So it is asked only about a call that would otherwise run.
- Only when the arguments match the tool's parameters. A missing argument, or one the tool does not take, is refused with `Arguments for <tool> do not match its parameters: <message>`, and the policy is not asked.
- Just before the tool runs. Its time to decide never counts against the tool's time limit, but it counts toward the run's `timeout_s`.
- Once per call. A denial refuses only that call; the step's other calls are asked about on their own.

The request (`ApprovalRequest`):

| Field | Meaning |
|---|---|
| `request_id` | A new random id for each request: 32 hex characters. |
| `run_id` | The run's `task_id`, or `""` for `run_tool`. |
| `call_id` | The tool call's id. |
| `tool_name` | The tool. |
| `arguments` | A copy of exactly what will run. A file tool's `path` is the resolved path inside the workspace root. |
| `preview` | A one-line summary of the call; see [tool.started](../reference/events.md#toolstarted). |
| `workspace_root` | The agent's workspace root, or `None`. |
| `tier` | The tool's permission tier (`read`, `write`, `exec` or `network`), or `None` when the request was built by hand; an agent always fills it. |

The decision (`ApprovalDecision`):

| Field | Default | Meaning |
|---|---|---|
| `allow` | required | Whether the call may run. |
| `reason` | `""` | Why it may not, for the model. |
| `remember` | `False` | Allow this tool for the rest of the session. The run reports it on `tool.finished`, but does not act on it: a policy that grants a tool keeps the grant itself, and answers later requests with `origin="session"`. |
| `origin` | `"user"` | Who decided: `"user"`, `"session"` (a grant), `"policy"` or `"timeout"`. |

- A field of the wrong type raises `TypeError`.
- `remember=True` or `origin="session"` on a denial, and `origin="timeout"` on an allow, raise `ValueError`.

A denied call does not run. The model reads one of these texts as the call's error, and the run goes on:

| Origin | Without a reason | With a reason |
|---|---|---|
| `user` | `The user denied this call. <ending>` | `The user denied this call: <reason>. <ending>` |
| `policy` | `The approval policy denied this call. <ending>` | `The approval policy denied this call: <reason>. <ending>` |
| `timeout` | `The call was not approved in time, so it did not run. Silence is not consent: do not retry it without asking the user.` | The same. |

- `<ending>` is `Do not run it again or reach the same result another way unless the user asks.`
- The reason loses surrounding whitespace and a final `.`, and keeps at most 500 characters.
- `tool.finished` reports the call as `blocked`, with the error category `approval_denied`.
- The same denied call three steps in a row ends the attempt as stuck.

The policy fails closed:

| Case | What happens |
|---|---|
| The policy raises an exception, returns something other than an `ApprovalDecision`, or raises `CancelledError` while the run is not being cancelled | The call is denied with `Approval failed, so the call did not run: <ExceptionType>: <message>`. The run goes on. |
| The run is cancelled while the policy decides | `decide` receives the `CancelledError`; it must withdraw its prompt and re-raise. The call does not run, and `tool.finished` reports the approval's origin as `withdrawn`. |
| A workspace root, and the policy allows a file tool's call | Its path is resolved again. If the result has changed, for example because a symlink took the file's place while the policy decided, the call is refused with `Path '<path>' changed while waiting for approval; the call did not run.` |

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
    options = AgentOptions(
        workspace_root="~/melo-work",
        tool_timeout_s=600,
        # A Python run's default set leaves shell_exec out.
        builtins=["file_read", "file_write", "python_exec", "shell_exec"],
    )
    agent = Agent(llm=llm, options=options)
    try:
        print(agent.workspace_root)
        result = await agent.run_tool("shell_exec", {"command": "pwd"})
        print(result.output)
    finally:
        await agent.close()


asyncio.run(main())
```

- Both lines print the real path of `~/melo-work`, such as `/Users/you/melo-work`.
- The agent resolves the root to its real path, with `~` expanded. A root that is not a directory raises `ValueError: workspace_root is not a directory: '<root>'` when the agent is built.
- Without a root, the built-in tools behave as in direct mode, as described in [Tools and MCP](../reference/tools.md#built-in-tools).

What the root does to each tool:

| Tool | What happens |
|---|---|
| `file_read`, `file_write` | The fence, the resolved path, regular files only and the size cut are the ones in [With a workspace root](tools-mcp.md#with-a-workspace-root). A symlink inside the root that points out is refused; one outside that points in is allowed. `file_read` reads UTF-8, and `file_write` reports the resolved path. |
| `list_files` | Resolves its path the same way (no path lists the root) and refuses one outside with the same message. It never enters a symlink and never reads a file. |
| `shell_exec` | Runs in a child process with a process group of its own, which a timeout or a cancellation kills as a whole; the root is the directory it starts in. Its standard input is `/dev/null`. |
| `python_exec` | Runs in a child Python, so a time limit can stop it, and `sys.exit()` or `os.chdir()` stays in the child. It receives its code on a pipe (never in argv) and then points fd 0 at `/dev/null`, so `input()` gets EOF. |

Both exec tools:

| Topic | What happens |
|---|---|
| Limits | The child gets a CPU-time limit of one hour and an open-file limit of 1,024 (`RLIMIT_CPU` and `RLIMIT_NOFILE` on POSIX). They only stop code that outlives the call; no address-space limit is set, because macOS refuses one. |
| Environment | Not melo's: `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `LANG`, every `LC_*` variable and `TMPDIR`, plus `tool_env`. An API key in melo's environment reaches no tool; `VIRTUAL_ENV`, `PYTHONPATH`, `SSH_AUTH_SOCK` and the proxy variables pass only through `tool_env`. |
| Reach | They can still reach any path. Only the file tools are limited to the root; use an [approval policy](#approve-tool-calls) to control them. |
| Exit status | A shell command that exits non-zero starts its text with `[exit code <n>]`, and the call still succeeds: the model reads the status even when the output is cut. Python that exits non-zero fails with `Python exited with code <n>`, followed by the last 4,000 characters of its output. |
| Time limits | `tool_timeout_s` sets the limit, for builds and test suites that take longer than the defaults of 30 seconds (shell) and 60 seconds (Python). A shell command fails with `Command timed out after <n> s`; a Python call ends as a timeout with `Execution timed out after <n>s`. `tool_timeout_s` without a `workspace_root` raises `ValueError`. |

- Each planning request tells the model: `Working directory: <root>. Relative paths and shell and Python commands run there; file_read and file_write refuse paths outside it.`
- `RunResult.config` does not include the root: `agent.workspace_root` and the `run.started` event give it.

## Keep answers between runs

By default, working memory keeps the tasks and tool calls of earlier runs on the agent, but not their answers. With `AgentOptions(record_answers=True)`, each run ends by appending its answer to working memory, so the next run sees the earlier questions and answers as a conversation:

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

- The answer is the run's `result` without its reasoning. The `RunResult` keeps the whole text.
- The first `<think>…</think>` block, in any letter case, is left out. So is the text after a `<think>` at the start that is never closed, or the text before a `</think>` with no `<think>` before it.
- That last case is for templates that open the block in the prompt, such as Qwen3's thinking models, which leave only the closing tag. Ollama, and OpenAI-compatible servers without a reasoning parser, put the reasoning there.
- An answer longer than 8,000 characters is cut and ends with `... [<n> more characters not shown]`.
- A run without an answer leaves a label instead: `[interrupt: no assistant output for this turn]` for a `cancelled` run, and `[no answer: the run ended with status <status>]` for any other.
- The label stays in `agent.conversation()` and in a saved turn. The model is shown a user message, `[The previous request ended without an answer.]`, in its place, joined to the user messages beside it, because a model copies an assistant message that says it gave no answer.

To keep a conversation in a file, across processes and restarts, see [Sessions](../reference/sessions.md).
