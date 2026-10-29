<a id="python-api"></a>

# 参考：Python API

在你自己的 Python 代码中使用 localmelo：构建一个 `Agent`、运行任务、读取它们的结果、在运行时跟踪它们、停止它们，并决定哪些工具调用可以运行。

> [!NOTE]
> Python API 是实验性的，在 v0.1.0 之前可能改变。

<a id="agent"></a>

## 智能体（Agent）

构建智能体有两种方式：

- `Agent(config=load())` 使用一份配置中的后端，与 `melo` 和网关的做法相同。`Config`、`load` 和 `save` 来自 `localmelo.support.config`。
- `Agent(llm=..., embedding=...)` 使用你自己构建的模型提供方。嵌入提供方是可选的；没有它就没有长期记忆。

```python
from localmelo import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
agent = Agent(llm=llm, options=AgentOptions(builtins=["file_read"], timeout_s=60))
```

- 两种形式都接受 `options=AgentOptions(...)` 和 `tool_providers=[...]`。
- 一个智能体一次运行一个任务。同一智能体上的任务共享它的工作记忆。
- `await agent.close()` 停止工具提供方，并关闭模型连接和记忆文件。用完后请调用它。

[智能体方法](#agent-methods)列出了其他方法。

<a id="agentoptions"></a>

## 选项（AgentOptions）

| 字段 | 默认值 | 说明 |
|---|---|---|
| `max_steps` | `30` | 每次运行最多的规划步骤数。 |
| `attempt_policy` | `"reflect"` | `"reflect"`：分多次尝试，每次之后有一次反思。`"none"`：只有一次尝试。 |
| `steps_per_attempt` | `10` | 使用 `"reflect"` 时每次尝试最多的步骤数。 |
| `max_attempts` | `5` | 使用 `"reflect"` 时每次运行最多的尝试次数。 |
| `reflection_gate` | `"flags"` | 使用 `"reflect"` 时为 `"flags"` 或 `"utility"`；见[反思](../concepts/how-it-works.md#reflection)。 |
| `tool_exposure` | `"retrieve"` | `"retrieve"` 或 `"all"`；见[如何提供工具](tools.md#how-tools-are-offered)。 |
| `builtins` | `None` | 要注册的内置工具：`None` 表示该模式的默认集合，空集合表示没有，或者给出一个名称集合。 |
| `memory` | `True` | `False` 关闭长期记忆。 |
| `memory_dir` | `None` | 存放 SQLite 记忆的目录，会展开 `~`。`None` 表示遵循环境变量；见[记忆](../guides/memory.md#keep-memory-between-runs)。 |
| `timeout_s` | `None` | 每次运行的时间限制，以秒计。`None` 表示没有限制。它也会中断 `python_exec` 调用，并终止其子进程。 |
| `mode` | `"python"` | 运行所属的界面：`"direct"`、`"interactive"`、`"gateway"` 或 `"python"`。它决定 `builtins=None` 注册什么：只有 `"interactive"` 提供 `shell_exec`；其他三个注册 `file_read`、`file_write` 和 `python_exec`。在 `builtins` 中点名 `shell_exec` 是 `direct` 或 `python` 调用方的显式启用，让运行可以使用它；未点名时检查器会拒绝它，在 `gateway` 模式下即使点名也拒绝；拒绝信息会说明模式，以及在 direct 或 python 模式下启用它的方法。显式的 `builtins` 优先于默认注册。 |
| `workspace_root` | `None` | 内置工具工作的目录，也是文件工具能到达的唯一目录；见[在目录中工作](../guides/python.md#work-in-a-directory)。 |
| `tool_timeout_s` | `None` | `shell_exec` 和 `python_exec` 的时间限制，以秒计。需要 `workspace_root`。`None` 保持 shell 命令 30 秒、Python 60 秒。 |
| `tool_env` | `None` | 在最小环境之上，为 `shell_exec` 和 `python_exec` 子进程提供的额外变量；见[在目录中工作](../guides/python.md#work-in-a-directory)。 |
| `record_answers` | `False` | `True` 把每次运行的答案保留在工作记忆中供下一次运行使用；见[在运行之间保留答案](../guides/python.md#keep-answers-between-runs)。 |
| `reasoning` | `"legacy"` | 回复中的推理如何回传：`"legacy"`、`"drop"`、`"replay"`、`"compress"` 或 `"compress+replay"`；见[推理模式](../../reference/cli.md#direct-mode)（英文）。 |
| `replay_turns` | `None` | 使用 `"replay"` 或 `"compress+replay"` 时，回放推理的最近 N 个助手轮次；`None` 保留范围内所有轮次。 |
| `replay_scope` | `"task"` | `"task"` 只回放当前任务的轮次；`"window"` 回放窗口中的每个轮次，包括更早的任务。仅限 API。 |
| `note_writer` | `"auto"` | 在压缩模式下，`"auto"` 先读模型的 `<note>` 块，否则用启发式；`"heuristic"` 总是从回复的文本和思考写笔记。 |

`AgentOptions` 创建后不能更改。无效的值会抛出 `ValueError` 并指出字段：

```text
ValueError: max_steps must be an int >= 1, got 0
```

[命令行选项](../../reference/cli.md#direct-mode)（英文）设置同样的字段，但 `steps_per_attempt`、`max_attempts`、`replay_scope` 和 `mode` 除外（命令行设置界面的模式）。直接模式从不设置 `workspace_root`、`tool_timeout_s`、`tool_env` 或 `record_answers`。

<a id="results"></a>

## 结果

- `await agent.run_task(query)` 返回一个 `RunResult`。[运行结果](run-results.md)介绍它的状态和字段，`result.to_dict()` 按同样的顺序把它们作为字典返回。
- `await agent.run(query)` 返回文本：结果，或者结果为空时的 `[<status>] <error>`。网关返回的就是这段文本。

`Agent`、`AgentOptions`、`AgentEvent`、`RunResult`、`Message`、`ToolCall`、`ToolDef` 和 `ToolResult` 都可以从 `localmelo` 导入。事件和审批类型，例如 `ApprovalPolicy`、`ApprovalRequest`、`ApprovalDecision`、`AllowAllPolicy` 和 `DenyAllPolicy`，来自 `localmelo.melo.contracts`。

<a id="agent-methods"></a>

## 智能体方法

| 方法 | 作用 |
|---|---|
| `await agent.run_task(query, *, on_event=None, approval=None, observe=None, progress=False, routine=None)` | 运行一个任务并返回它的 `RunResult`；见[结果](#results)和[通过事件跟踪运行](../guides/python.md#follow-a-run-with-events)。 |
| `await agent.run(query)` | 运行一个任务并返回它的文本。 |
| `await agent.start()` | 现在就启动工具提供方并注册它们的工具，而不是等到第一次运行。第二次调用不做任何事。它抛出设置阶段的错误。 |
| `agent.list_tools()` | 已注册的工具，按名称排序，返回 `ToolInfo`：`name`、`description`、`parameters`、`source`（`"builtin"`、`"provided"` 或 `"custom"`）、`requires_approval`，以及 `tier`（`"read"`、`"write"`、`"exec"` 或 `"network"`；见[权限级别](../guides/tools-mcp.md#permission-tiers)）。 |
| `await agent.run_tool(name, arguments, *, approval=None)` | 在一次运行之外运行一个工具调用，并返回它的 `ToolResult`。 |
| `agent.conversation()` | 工作记忆窗口的副本，是一个 `Message` 列表。 |
| `agent.load_conversation(messages)` | 替换工作记忆窗口，并返回保留的消息数。 |
| `agent.workspace_root` | 工作区根目录的真实路径，或 `None`。 |
| `agent.begin_exit(reason, *, within_s)` | 为进程退出做准备；见[停止运行](../guides/python.md#stop-a-run)。 |
| `await agent.close()` | 停止工具提供方，并关闭模型连接和记忆文件。 |

- **先启动智能体**，如果你可能要取消它的第一次运行：在工具提供方启动期间发生的取消会让它们的设置对智能体的整个生命周期都失败，之后的运行以 `setup_error` 结束。
- **`run_tool`** 让调用经过与模型调用相同的检查，区别在于任何已注册的工具都可以运行：参数、模式的 `exec` 拒绝、工作区根目录、审批策略、工具的时间限制和输出限制。被拒绝或失败的调用以 `error` 返回原因。该调用不向工作记忆或历史记录写入任何内容，也不发出任何事件。对策略而言，`run_id` 是 `""`。
- **`load_conversation`** 清空工作记忆及其反思，然后添加消息。工具调用只有连同逐条回答其每次调用的结果时才会被放入；不完整的调用，或者没有对应调用的工具结果，会被丢弃并给出一条 WARNING。`[]` 清空对话。把它和 `conversation()` 一起使用，可以把一段对话移到新的智能体。
- `run_tool` 和 `load_conversation` 在运行进行期间抛出 `RuntimeError`，包括在被取消的交互式运行的清理过程中。

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

它打印：

```text
file_read builtin False
file_write builtin True
python_exec builtin True
```

<a id="tool-providers-and-mcp"></a>

## 工具提供方与 MCP

工具提供方向智能体添加工具。`McpToolProvider` 添加 MCP 服务器的工具：

```python
from localmelo import Agent
from localmelo.support.config import load
from localmelo.support.mcp import McpToolProvider, load_mcp_config

servers = load_mcp_config("mcp.json")
agent = Agent(config=load(), tool_providers=[McpToolProvider(servers)])
```

- 文件使用 [MCP 配置文件](../guides/tools-mcp.md#mcp-config-file)中的格式。对于无法读取或无效的文件，`load_mcp_config` 抛出 `McpConfigError`。
- 第一次运行会启动这些提供方，并把它们的工具注册在内置工具旁边。
- 如果某个提供方无法启动，或者它的某个工具与已注册的工具同名，运行以 `setup_error` 结束。该智能体之后的运行也会以同样方式结束。
- `agent.close()` 停止 MCP 服务器。
- 每个 MCP 工具原始的 `annotations` 对象保存在它的 `ToolDef.annotations` 上，检查器据此推导工具的权限级别；见[权限级别](../guides/tools-mcp.md#permission-tiers)。你自己构建的 `ToolDef` 以 `{}` 开头，即 `write`。
- `requires_approval=True` 的提供工具会在每次调用前询问本次运行的审批策略；见[审批工具调用](../guides/python.md#approve-tool-calls)。对 MCP 服务器，在其条目中设置 `require_approval`。

<a id="custom-model-providers"></a>

## 自定义模型提供方

- 对话请求超时属于提供方：向 `OpenAICompatLLM` 或向 `OllamaNativeChat`（`localmelo.support.providers.llm.ollama_chat`）传入 `timeout=`（以秒计）。使用 `Agent(config=...)` 时，由 `request_timeout_s` 设置。
- 要连接另一种服务器，请继承 `localmelo.melo.contracts.providers` 中的 `BaseLLMProvider`（或 `BaseEmbeddingProvider`），并实现 `chat()` 和 `close()`（或 `embed()` 和 `close()`）。
- 可选的 `stream_chat(messages, tools=None)` 返回一个 `StreamFragment` 值的异步生成器：临时的 `content` 和 `reasoning` 文本、`tool` 索引（绝不是不完整的参数）、`usage`，然后是一个 `reply`，包含完整的 `Message` 和 `finish_reason`。没有流式能力的提供方调用 `chat()`，并产生一个 `reply`，其 `finish_reason` 为空。`OpenAICompatLLM` 只为这个方法发送 `stream=true`，并在返回之前验证完整的工具调用。只有服务器发送用量时才会返回它；在支持的地方，用 `extra_body={"stream_options": {"include_usage": True}}` 请求它。失败会抛出 `ProviderError`；一旦产生了任何片段，传输或提前结束错误就不可重试。取消会向外传播。提前退出时用 `aclose()` 关闭生成器。增量和最终消息可能包含敏感文本：不要持久化原始增量，也不要直接显示它们。`localmelo.melo.checker.stream_filter.StreamFilter` 是独立的显示策略。没有声明的分帧提示时，它缓冲整个回复；在 `reasoning_tags="explicit"`、`"closing_only"` 或 `"none"` 时，它可以放出已脱敏的完整行。未审阅的脱敏规则变更会强制整体缓冲。这些提示是针对某个模型和模板的可信保证，而不是猜测：`none` 按字面处理标签，`closing_only` 把第一个闭合标签之前的一切当作思考，`explicit` 识别开头的打开标签。`split_thinking(..., reasoning_tags=...)` 对完整回复使用同一规则；省略它会保留现有的整条回复行为。可信的独立推理字段可以改用 `StreamFilter(separate_reasoning=True)` 和 `feed_reasoning()`；它绝不进入内容。每次 `feed()` 和 `feed_reasoning()` 返回 `(thinking_delta, content_delta)`；`finish()` 返回剩余的尾部。对每个已完成的通道，按顺序拼接它的增量，包括尾部。思考会保留到 finish。脱敏规则在流开始时快照；更改从下一个流生效。提交时，对每个拼接后的显示通道分别应用 `redact_for_storage()`：它先应用快照，再应用当前规则，因此更新的规则可以增加掩码。脱敏基于模式，不保证移除秘密。唯一的消费者是交互式终端界面（`[cli] stream = true`）：它只对规划请求流式，在实时区域绘制已放出的文本，并且只存储完成的回复，从不存储增量；现有的 `chat()` 行为不变。调用方不得在展示临时输出之后悄悄重试流。只在回复完成之后调用 `finish()`；在出错或取消时调用 `abort()` 以丢弃缓冲的文本（已经放出的文本无法收回）。
- 在对话提供方上把 `model` 设置为它所用模型的 id，就像内置提供方那样。程序记忆按模型保存运行做过的事；使用默认的 `""` 或 `"default"` 时，过去运行的事实仍会显示，但绝不会显示进度条。让 `model` 保持为普通属性，或者像转发内部提供方 id 的包装器那样，保持为带 setter 的属性：mypy 不接受用只读属性代替 `BaseLLMProvider.model`。
- 对值得重试的失败，在 `chat()` 中抛出 `ProviderError(message, status_code=..., retryable=True)`。智能体随后会像对待内置提供方那样重试它；见[提供方错误与重试](run-results.md#provider-errors-and-retries)。
