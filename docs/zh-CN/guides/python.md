<a id="guide-python"></a>

# 指南：Python

<a id="run-a-task"></a>

## 运行任务

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

`load()` 读取 `~/.cache/localmelo/config.toml`，因此请先保存配置（[配置（英文）](../../reference/configuration.md)）。脚本输出状态和答案，例如 `completed 42`。

<a id="follow-a-run-with-events"></a>

## 通过事件跟踪运行

`await agent.run_task(query, on_event=sink)` 对运行中的每个事件按顺序同步调用一次 `sink`：每一步的开始、模型的每次回复、每次工具调用及其结果、每次反思，以及携带 `RunResult` 的 `run.finished`。接收函数不能阻塞。[事件（英文）](../../reference/events.md)列出了事件和接收函数的规则。

- 用 `observe=sink` 代替 `on_event=sink` 会向接收函数提供相同事件，但不会让运行变成交互式运行，因此被取消的运行仍以 `crashed` 结束（见[停止运行](#stop-a-run)）。适用于仅观察的展示界面。
- 两种方式都可搭配 `progress=True`，以增加 [`run.progress`（英文）](../../reference/events.md#runprogress)事件：步数、尝试、阶段和测试计数，以及此前在该模型上运行过这个任务时，过去的运行都做了什么。没有接收函数时，它不起作用。
- 无论任务的措辞和目录如何，`routine="name"` 都会将这些运行归为[程序记忆](../concepts/procedures.md)中的同一个例程；不指定时，例程由屏蔽日期和时间后的查询以及运行工具所使用的目录决定。空白名称会在运行前引发 `ValueError`。

在 `run.finished` 之后、`run_task` 返回之前，运行会记录到程序记忆中：最多耗时 2 秒，通常只有几毫秒（见[记忆](../concepts/memory.md#how-memory-works)）。引发异常的运行（如被取消的运行）不会被记录，在 `agent.begin_exit()` 之后结束的运行也不会。

<a id="stop-a-run"></a>

## 停止运行

要停止运行，可以取消等待 `run_task` 的任务，或让作用域取消它，例如 `asyncio.timeout`、`TaskGroup`，或在 `asyncio.run` 中按 Ctrl+C。此时 `run_task` 会引发 `CancelledError`，绝不会吞掉取消异常。

运行如何结束取决于它是否是交互式运行，即是否设置了 `on_event` 或 `approval`（`observe` 不算）：

- **交互式运行**以 `cancelled` 结束。它的 `RunResult` 随 `run.finished` 事件返回，`result` 为空，`error` 为原因：若 `task.cancel(msg)` 的消息是非空字符串，则使用该消息，否则使用 `Cancelled by the caller`。
- **其他运行**与直接模式一样以 `crashed` 结束，且没有 `RunResult`。

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

如果模型在 2 秒后仍在回答，脚本会输出 `cancelled Stopped by the user`。

交互式运行被取消后：

- 任务记录以 `cancelled` 保存；历史记录为正在执行的那一步中的每次调用各写入一行，因此每次调用只出现一次。
- 工作记忆保留任务和那一步。尚未返回的调用显示 `The run was stopped before this call finished; it may have partly run.`；尚未开始的调用或审批策略仍在决定的调用显示 `The run was stopped before this call started; it did not run.`。该智能体的下一次运行会看到这些信息。
- 反思被清除，不会提升为长期记忆，因此取消后不会发起网络请求。
- `shell_exec` 命令或 `python_exec` 子进程会连同其启动的所有进程一起被终止。
- 清理在独立任务中运行，第二次取消不会停止它。写入历史记录和任务记录最多可能需要 10 秒。`run.finished` 仍会发出，`agent.close()` 会等待清理完成。

无论是否交互式运行，`timeout_s` 仍会让运行以 `timeout` 结束。

如果进程必须退出，例如收到 SIGTERM 时，请先调用 `agent.begin_exit(reason, within_s=...)`，再取消运行并等待 `agent.close()`。运行以 `reason` 作为错误结束：交互式运行状态为 `cancelled`，否则为 `crashed`。清理遵守截止时间：未及时关闭的工具提供方会被终止，反思不会被提升。直接模式在收到 SIGTERM 和 SIGHUP 时也这样做；见[信号（英文）](../../reference/cli.md#signals)。

<a id="approve-tool-calls"></a>

## 审批工具调用

审批策略决定工具调用是否可以运行。将策略作为 `approval` 传给 `run_task` 或 `run_tool`。内置工具或提供的工具在权限级别不是 `read` 时会请求审批（见[权限级别](tools-mcp.md#permission-tiers)）：

- 本次运行提供的内置工具：默认 `python` 模式下的 `python_exec`（`exec`）和 `file_write`（`write`）；调用方在 `builtins` 中指定 `shell_exec`（`exec`）时，也包括它。除 `interactive` 外，没有模式默认提供 `shell_exec`；
- 没有注解，或注解使其归为 `write` 或 `network` 的提供的工具；
- `ProvidedTool.requires_approval` 为 `True` 的提供的工具，这也会让 `read` 工具请求审批；
- 配置项设置了 `require_approval` 的 MCP 服务器的工具；见 [MCP 配置文件](tools-mcp.md#mcp-config-file)。

交互式界面始终有审批策略（除非使用 `--yolo`）；直接模式、网关和没有 `approval=` 的 Python 运行则没有，这也是这些模式默认不提供 `shell_exec` 的原因。`file_read` 属于 `read`，不请求审批：它由[工作区根目录](#work-in-a-directory)限制。只读 MCP 工具（`readOnlyHint: true`）也不请求审批，除非 `require_approval` 指定了它。通过 `agent.executor.register` 直接注册的工具保留其明确的 `requires_approval`；它的级别只是标签。没有策略时，本会请求审批的工具也像其他工具一样运行，正如从不设置策略的直接模式。`direct` 或 `python` 调用方在 `builtins` 中指定 `shell_exec` 是明确启用它，没有策略时它无需审批即可运行；`gateway` 仍会拒绝它。

策略可以是任何带有 `async def decide(request)` 方法并返回 `ApprovalDecision` 的对象。下面的策略在终端中询问：

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

如果运行在此策略等待时被取消，`CancelledError` 会在 `await` 处结束 `decide` 并向上传播，符合下述规则；但终端中的问题会一直保留到按下 Enter。能够撤回提示的策略（例如对话框）必须撤回提示。

当 `approval` 没有 `async def decide` 时，`run_task` 在运行开始前引发 `TypeError`。`AllowAllPolicy()` 允许所有调用，`DenyAllPolicy(reason)` 则以指定原因拒绝所有调用。

请求策略的时机：

- 在所有能够直接拒绝调用的检查之后：本步必须提供了该工具，参数必须有效，模式未提供的 `exec` 工具必须在 `builtins` 中指定，文件路径必须在工作区根目录内。因此，只有其他条件都允许执行的调用才会请求策略决策。
- 仅在参数符合工具参数定义时请求。缺少参数或包含工具不接受的参数时，会以 `Arguments for <tool> do not match its parameters: <message>` 拒绝，不请求策略决策。
- 就在工具运行之前。策略的决策时间不计入工具时限，但计入运行的 `timeout_s`。
- 每次调用请求一次。一次拒绝只阻止该调用；该步的其他调用各自请求决策。

请求（`ApprovalRequest`）：

| 字段 | 含义 |
|---|---|
| `request_id` | 每次请求新生成的随机 ID：32 个十六进制字符。 |
| `run_id` | 运行的 `task_id`；对于 `run_tool` 则为 `""`。 |
| `call_id` | 工具调用的 ID。 |
| `tool_name` | 工具名称。 |
| `arguments` | 将要执行的确切参数的副本。文件工具的 `path` 是工作区根目录内解析后的路径。 |
| `preview` | 调用的单行摘要；见 [tool.started（英文）](../../reference/events.md#toolstarted)。 |
| `workspace_root` | 智能体的工作区根目录，或 `None`。 |
| `tier` | 工具的权限级别（`read`、`write`、`exec` 或 `network`）；手动构造请求时可为 `None`，智能体总会填写它。 |

决策（`ApprovalDecision`）：

| 字段 | 默认值 | 含义 |
|---|---|---|
| `allow` | 必填 | 是否允许运行调用。 |
| `reason` | `""` | 不允许的原因，提供给模型。 |
| `remember` | `False` | 在本会话剩余时间内允许此工具。运行在 `tool.finished` 中报告该值，但不会据此操作：授予工具权限的策略自行保存授权，并以 `origin="session"` 回答后续请求。 |
| `origin` | `"user"` | 决策者：`"user"`、`"session"`（授权）、`"policy"` 或 `"timeout"`。 |

字段类型错误时引发 `TypeError`。拒绝时设置 `remember=True` 或 `origin="session"`，以及允许时设置 `origin="timeout"`，都会引发 `ValueError`。

被拒绝的调用不会运行。模型将以下文本之一作为调用错误读取，运行继续：

| 来源 | 无原因 | 有原因 |
|---|---|---|
| `user` | `The user denied this call. <ending>` | `The user denied this call: <reason>. <ending>` |
| `policy` | `The approval policy denied this call. <ending>` | `The approval policy denied this call: <reason>. <ending>` |
| `timeout` | `The call was not approved in time, so it did not run. Silence is not consent: do not retry it without asking the user.` | 相同。 |

`<ending>` 是 `Do not run it again or reach the same result another way unless the user asks.`。原因会去掉首尾空白和末尾的 `.`，最多保留 500 个字符。`tool.finished` 将该调用报告为 `blocked`，错误类别为 `approval_denied`。连续三步出现同一个被拒绝的调用，会使本次尝试以卡住结束。

策略出现故障时默认拒绝：

- 策略引发异常、返回的不是 `ApprovalDecision`，或在运行未被取消时引发 `CancelledError`，都会以 `Approval failed, so the call did not run: <ExceptionType>: <message>` 拒绝调用。运行继续。
- 策略决策期间运行被取消时，`decide` 会收到 `CancelledError`，必须撤回提示并重新引发异常。调用不会运行，`tool.finished` 将审批来源报告为 `withdrawn`。
- 设置了工作区根目录时，策略允许文件工具调用后，路径会再次解析。如果结果发生变化，例如策略决策期间符号链接取代了文件，调用会以 `Path '<path>' changed while waiting for approval; the call did not run.` 拒绝。

<a id="work-in-a-directory"></a>

## 在目录中工作

`AgentOptions(workspace_root=DIR)` 为内置工具指定一个目录：

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
        # P13b: a Python run's default set leaves shell_exec out.
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

两行都输出 `~/melo-work` 的真实路径，例如 `/Users/you/melo-work`。

- 智能体展开 `~` 并将根目录解析为真实路径。如果根目录不是目录，创建智能体时会引发 `ValueError: workspace_root is not a directory: '<root>'`。
- **`file_read` 和 `file_write`** 相对于根目录解析相对路径，只能访问其中的文件。`..` 和符号链接会先解析：根目录内指向外部的符号链接会被拒绝，而外部指向内部的符号链接会被允许。外部路径会以 `Path '<path>' is outside allowed workspace root` 拒绝，空路径会以 `Argument 'path' for <tool> must be a non-empty string without NUL characters.` 拒绝。它们打开解析后的路径时不跟随符号链接，因此检查后取代文件的符号链接会被拒绝；非普通文件，例如 FIFO，也会以 `Not a regular file: <path>` 拒绝。`file_read` 最多按 UTF-8 读取 200,000 字节。超过 200,000 字节或文本超过 50,000 字符的文件会截断至 50,000 字符，末尾为 `... [truncated: the file has <size> bytes]`。`file_write` 报告解析后的路径。
- **`shell_exec` 和 `python_exec`** 始终在拥有独立进程组的子进程中运行，超时或取消会终止整个进程组；根目录是它们的启动目录。Shell 命令的标准输入是 `/dev/null`；`python_exec` 通过管道接收代码（绝不通过 argv），随后将 fd 0 指向 `/dev/null`，因此 `input()` 得到 EOF，两者都不读取终端。Python 代码在子 Python 进程中运行，因此时限可以停止它，`sys.exit()` 或 `os.chdir()` 的作用也留在子进程中。子进程还有一小时 CPU 时间限制和 1,024 个打开文件的限制（POSIX 上为 `RLIMIT_CPU` 和 `RLIMIT_NOFILE`），这些限制只会停止超过调用时长继续运行的代码；由于 macOS 不接受地址空间限制，因此没有设置。子进程的环境不是 melo 的环境：它获得 `PATH`、`HOME`、`USER`、`LOGNAME`、`SHELL`、`TERM`、`LANG`、所有 `LC_*` 变量和 `TMPDIR`，以及 `tool_env`，因此 melo 环境中的 API 密钥不会传给任何工具；`VIRTUAL_ENV`、`PYTHONPATH`、`SSH_AUTH_SOCK` 和代理变量只能通过 `tool_env` 传入。
- **它们仍可访问任何路径。** 只有文件工具受根目录限制。使用[审批策略](#approve-tool-calls)控制 `shell_exec` 和 `python_exec`。
- **退出状态。** Shell 命令以非零状态退出时，其文本以 `[exit code <n>]` 行开头，调用仍然成功：即使输出被截断，模型也能读取状态。Python 代码以非零状态退出时，调用以 `Python exited with code <n>` 失败，后面附上输出的最后 4,000 个字符。
- **时限。** `tool_timeout_s` 设置 `shell_exec` 和 `python_exec` 的时限，适用于超过默认时限的构建和测试套件：Shell 命令默认为 30 秒，Python 默认为 60 秒。运行过久的 Shell 命令以 `Command timed out after <n> s` 失败；运行过久的 Python 调用以超时结束，提示 `Execution timed out after <n>s`。设置 `tool_timeout_s` 而不设置 `workspace_root` 会引发 `ValueError`。
- **模型会得到提示。** 每次规划请求都包含：`Working directory: <root>. Relative paths and shell and Python commands run there; file_read and file_write refuse paths outside it.`
- `RunResult.config` 不包含根目录：可从 `agent.workspace_root` 和 `run.started` 事件获取。

未设置根目录时，内置工具的行为与直接模式相同，见[工具和 MCP](../reference/tools.md#built-in-tools)。

<a id="keep-answers-between-runs"></a>

## 在运行之间保留答案

默认情况下，工作记忆会保留该智能体先前运行的任务和工具调用，但不保留答案。设置 `AgentOptions(record_answers=True)` 后，每次运行结束时会将答案追加到工作记忆，让下一次运行以对话形式看到先前的问题和答案：

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

例如输出：

```text
user What is the capital of France?
assistant Paris is the capital of France.
```

- 答案是运行的 `result` 去除推理后的内容：第一个 `<think>…</think>` 块无论字母大小写都会被省略；开头的 `<think>` 若一直未闭合，其后的文本也会被省略；没有前置 `<think>` 的 `</think>` 之前的文本同样会被省略（在提示中打开该块的模板，例如 DeepSeek-R1 和 Qwen3 的思考模型，只留下闭合标签）。Ollama 和没有推理解析器的 OpenAI 兼容服务器会将推理放在那里。`RunResult` 保留完整文本。
- 超过 8,000 字符的答案会被截断，末尾为 `... [<n> more characters not shown]`。
- 没有答案的运行会留下标签：`cancelled` 运行使用 `[interrupt: no assistant output for this turn]`，其他运行使用 `[no answer: the run ended with status <status>]`。

要将对话保存在文件中，供跨进程和重启使用，见[会话（英文）](../../reference/sessions.md)。
