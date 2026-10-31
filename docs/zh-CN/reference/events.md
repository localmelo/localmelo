<a id="events"></a>

# 事件

运行可以在执行过程中报告它做了什么：每个步骤、模型的每次回复、每次工具调用及其结果、每次反思，以及运行如何结束。本页列出各个事件、它们出现的顺序，以及接收事件的函数所遵循的规则。

> [!NOTE]
> 事件属于实验性的 Python API，可能会在 v0.1.0 之前变化。直接模式的任务在其进度行启用时使用 `observe` 接收函数；`melo ""` 循环不使用接收函数运行。交互式界面和网关的 `/api` 控制台以事件接收函数运行各轮次，并把事件保存在会话中；控制台通过 SSE 公开这些事件，并在网页工作区中显示。旧版 `/v1/playground` 接收函数只保留 Stop 所需的 `run.finished` 结果，不显示实时事件。

<a id="follow-a-run"></a>

## 跟踪一次运行

把一个函数作为 `on_event` 传给 `Agent.run_task`。每个事件都会调用它一次：

```python
import asyncio

from localmelo import Agent, AgentEvent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


def show(event: AgentEvent) -> None:
    print(event.seq, event.type)


async def main() -> None:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm, options=AgentOptions(builtins=["shell_exec"]))
    try:
        result = await agent.run_task("Which files are here?", on_event=show)
        print(result.status)
    finally:
        await agent.close()


asyncio.run(main())
```

模型调用一次 `shell_exec` 然后作答的运行会打印：

```text
1 run.started
2 step.started
3 assistant.message
4 tool.started
5 tool.finished
6 step.started
7 assistant.message
8 run.finished
completed
```

带有 `on_event` 或审批策略的运行是*交互式*运行。这只在运行被取消时才有区别：它会以 `cancelled` 而不是 `crashed` 结束（见[停止运行](../guides/python.md#stop-a-run)）。

要在不让运行成为交互式的情况下观察它，请把函数作为 `observe` 传入。它接收的事件相同，但被取消的运行会以 `crashed` 结束，与没有接收函数的运行一样。

`on_event` 和 `observe` 只能给出其中一个，不能两个都给。

要同时了解运行所处的位置，请与其中任一项一起传入 `progress=True`：运行随后还会报告 [`run.progress`](#runprogress) 事件。

<a id="the-event"></a>

## 事件的结构

每个事件都是一个 `AgentEvent`，`localmelo` 导出了它：

| 字段 | 含义 |
|---|---|
| `type` | 事件的种类；见[事件目录](#catalogue)。 |
| `run_id` | 运行的 `task_id`：与 `RunResult.task_id` 以及历史记录中的任务记录 id 相同。 |
| `seq` | 事件在运行中的编号：1、2、3……中间没有空缺。 |
| `ts` | 事件发出的时间，即 `time.time()` 的值。 |
| `data` | 事件的字段：恰好是目录为该类型列出的键。 |
| `images` | 仅 `tool.finished`：工具返回的完整图片，供同一进程中的代码使用。 |

`event.to_dict()` 把事件返回为一个 JSON 对象，不含 `images`：

```json
{"v": 1, "type": "step.started", "run_id": "<task_id>", "seq": 2, "ts": 1761465600.0, "data": {"step": 1, "attempt": 1}}
```

`v` 是架构版本，即 1。`json.dumps(event.to_dict(), allow_nan=False)` 总是会成功。

<a id="rules-for-the-sink"></a>

## 接收函数的规则

传给 `on_event` 或 `observe` 的函数就是接收函数（sink）。

- **同步，按顺序。** 接收函数每个事件调用一次，按 `seq` 顺序，且同一时刻只由一个任务调用：运行 `run_task` 的任务，或者在取消之后负责收尾的任务。它绝不会被两个任务同时调用，也不会从其他线程调用。
- **必须是普通函数。** 当 `on_event` 或 `observe` 不可调用、是 `async def` 函数，或两者都给出时，`run_task` 会在运行开始前抛出 `TypeError`。此时不发出任何事件，也不写入任何任务记录。
- **不要阻塞。** 没有缓冲区，也没有超时：慢的接收函数会拖慢运行。把耗时的工作放进队列，并在另一个任务中处理（见下面的示例）。
- **返回的可等待对象会被忽略。** 它不会被等待，协程会被关闭。在一次运行中第一次出现时，localmelo 会记录警告 `on_event returned an awaitable; async sinks are not supported and the event was dropped`。
- **接收函数抛出异常会丢失该事件。** 当接收函数抛出 `Exception` 或 `CancelledError` 时，运行继续，下一个事件仍然使用下一个 `seq`。一次运行中第一次失败会以 WARNING 级别记录，带有 traceback，记录器为 `localmelo.melo.agent._events`；之后的失败记录在 DEBUG 级别。其他异常（例如 `KeyboardInterrupt`）会继续向外传播。
- **没有接收函数时不构建任何内容。** 没有 `on_event` 或 `observe` 的运行不会构建事件，也不构建载荷。

要对事件做耗时的处理，例如写入文件或发送给客户端，请把队列的 `put_nowait` 传给运行，并在你自己的任务中读取该队列：

```python
import asyncio
import json

from localmelo import Agent, AgentEvent
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


async def main() -> None:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm)
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

    async def write_events() -> None:
        with open("events.jsonl", "a") as log:
            while True:
                event = await queue.get()
                log.write(json.dumps(event.to_dict()) + "\n")
                if event.type == "run.finished":
                    return

    writer = asyncio.create_task(write_events())
    try:
        await agent.run_task("What is 6*7?", on_event=queue.put_nowait)
        await writer
    finally:
        await agent.close()


asyncio.run(main())
```

<a id="catalogue"></a>

## 事件目录

| 类型 | 何时 | `data` 键 |
|---|---|---|
| `run.started` | 首先发出，在检查 `run_task` 的参数之后。 | `query`, `max_steps`, `workspace_root`, `approval` |
| `inbox.accepted` | 请求拟合之后、发送规划请求之前。 | `offset`, `text`, `step`, `attempt` |
| `step.started` | 每个规划步骤开始时。 | `step`, `attempt` |
| `assistant.message` | 步骤的规划回复到达时。 | `step`, `content`, `thinking`, `tool_calls`, `usage`, `prefix`, `duration_s` |
| `note` | 在压缩模式中，当笔记、反思的笔记或尝试摘要进入仅驻留 RAM 的笔记存储时。 | `step`, `attempt`, `note_id`, `kind`, `source`, `text`, `chars`, `drafts`, `remember`, `stage` |
| `tool.started` | 每次工具调用交给执行器之前，也包括之后被检查拒绝的调用。 | `step`, `call_id`, `name`, `arguments`, `preview` |
| `tool.finished` | 该调用完成、被拒绝、被取消或异常终止时。 | `step`, `call_id`, `name`, `status`, `error_category`, `output`, `output_chars`, `error`, `duration_s`, `images`, `approval`, `exit_code` |
| `notice` | 一次尝试结束时、模型给出空回复或只含未完成思考的回复时，或模型请求被重试时。 | `kind`, `message`, `step`, `attempt`, `detail` |
| `recall` | 召回块进入请求时：在 `start`、`attempt`、`step`，或由 `recall` 工具触发。 | `step`, `attempt`, `trigger`, `retriever`, `query`, `hits`, `chars`, `ms`, `fallback`, `placement` |
| `reflection.started` | 在一次尝试之后的反思开始之前。 | `attempt`, `failure_type` |
| `reflection` | 反思的效果确定之后。 | `attempt`, `action`, `continue_run`, `flags`, `gate`, `utility`, `stop_reason`, `rationale`, `summary`, `best_effort_result`, `requests`, `readable`, `unreadable_reason`, `repaired`, `usage`, `duration_s` |
| `run.progress` | 仅当设置 `progress=True` 时：紧接在 `run.started` 之后；每次 `step.started`、`tool.finished`、`reflection.started` 和 `stuck` 通知之后；以及紧接在 `run.finished` 之前。 | `level`, `state`, `step`, `attempt`, `limits`, `phase`, `cycle`, `tests`, `items`, `fraction`, `expected_steps`, `past`, `off_reason`, `basis` |
| `context_budget` | 启用上下文记账时，每次规划或反思尝试一次，包括失败的尝试。 | `kind`, `model`, `attempt`, `proxy_version`, `source`, `estimate`, `measured_prompt_tokens`, `input_limit`, `over_limit`, `image_allowance`, `pruned` |
| `run.finished` | 最后一个。 | `result` |

<a id="runstarted"></a>

### run.started：运行开始

- `query`：任务。
- `max_steps`：运行的步骤预算。
- `workspace_root`：智能体的工作区根目录，或 `null`；见[在目录中工作](../guides/python.md#work-in-a-directory)。
- `approval`：运行带有审批策略时为 `true`。

<a id="inboxaccepted"></a>

### inbox.accepted：收件箱消息已接受

- `offset`：在本次运行的收件箱中，被接受的换行符之后紧接的字节偏移。
- `text`：经过基于模式的敏感信息脱敏之后的指令。Worker 的 `stream-json` 省略此字段；保存的 worker 事件则包含它。
- `step`、`attempt`：规划步骤及其尝试。接受发生在 `step.started` 之后、请求拟合之后，但它并不证明模型已经收到或执行了该指令。

<a id="stepstarted"></a>

### step.started：步骤开始

- `step`：步骤在运行中的编号，从 1 开始。`step.started` 事件的数量就是 `RunResult.steps`。
- `attempt`：尝试的编号，从 1 开始。

<a id="assistantmessage"></a>

### assistant.message：助手回复

- `step`：回复所属的步骤。
- `content`：回复的文本。
- `thinking`：服务器在正文之外发送的推理，否则为 `""`。兼容 OpenAI 的服务器以 `reasoning_content` 或 `reasoning` 发送，Ollama 以 `thinking` 发送。Ollama 的提供方还会把它放在 `content` 的开头，作为一个 `<think>` 块；没有推理解析器的服务器只把它留在那里。只含未完成思考的回复（见 `unfinished_answer` 通知）的全部正文都在这里，`content` 为 `""`。
- `tool_calls`：回复中的调用，每个都是 `{id, name, arguments, preview}`。`preview` 与 `tool.started` 上的相同。
- `usage`：回复的整数 token 计数，例如 `prompt_tokens`；服务器没有发送时为 `{}`。兼容 OpenAI 的服务器发送它们时，其中还包括 `cached_tokens`（来自 `prompt_tokens_details`）和 `reasoning_tokens`（来自 `completion_tokens_details`），在请求要求服务器返回用量（`stream_options.include_usage`）时，流式回复也包含它们。服务器未发送的计数不在 `usage` 中，因此缺少 `cached_tokens` 表示“未报告”，绝不是 0。`reflection` 事件的 `usage` 同样包含它们（对其各个请求求和）；运行的 `RunResult.usage` 保留三个标准计数。
- `prefix`：本次请求中有多少内容与该运行较早的某个请求（规划或反思）已经发送过的内容相同，以便把缓存未命中追溯到发生了变化的部分；运行的第一个请求为 `null`。智能体根据它发送的消息和工具计算该值，而请求体无论有没有它都相同。它是一个对象：
  - `messages`：有多少条开头的消息与最佳的更早请求相同。最佳请求是：开头相同消息最多的；若相同，则字符数最多的；若仍相同，则最近的。服务器的前缀缓存保留的不止最后一个请求，因此这就是它可以复用的部分。
  - `chars`：这些消息中的内容字符数，外加第一条不同的消息与其对应消息共享的开头字符数。
  - `against`：该请求，`{"kind": "plan" | "reflect", "request": n}`；`n` 从 1 开始计算运行中的规划请求和反思请求（一次被请求两次的反思占两个编号）。
  - `previous_messages`、`total_messages`：该请求和本次请求的消息数。
  - `first_changed`：第一条不同消息的角色（`system`、`user`、`assistant` 或 `tool`）；如果该请求正好是本次请求的前缀（运行中常见的增长），则为 `null`。
  - `tools_changed`：提供的工具定义不同时为 `true`。
  - `fresh_chars`：本次请求的消息中，该运行较早的任何请求都没有包含过的字符数，按每条序列化消息（角色、内容、thinking、调用、调用 id、图片）的哈希判断。该运行较早的请求都没有发送过它们：它们是新的工具结果、回复和反思块。它们是序列化后的字符数，而不是 `chars` 统计的内容字符数。`request_chars`：该请求的序列化消息和工具 schema 的全部字符数。对询问了两次的 `reflection`，两者都对其各个请求求和。早于它们的日志没有这两项。

  服务器按 token 前缀进行缓存，因此 `messages` 很高而 `cached_tokens` 很低，说明问题出在服务器；`first_changed: "system"` 则指向提示词。`scripts/cache_report.py LOG` 会根据 `stream-json` 日志逐个请求打印这两项，然后给出可复用份额：`cached / (prompt - fresh)`，其中 `fresh` 是把 `fresh_chars` 换算成的 token，按该请求自己的每字符提示 token 数估算（`fresh_chars / request_chars * prompt_tokens`；这是估算值，因此份额可能超过 100%）。它统计每个运行第一个请求之后、报告了 `cached_tokens` 的请求，分别按运行和合并给出，与所有提示 token 的份额并列。
- `duration_s`：规划请求的耗时，包括其重试及重试前的等待。启用持久化时，它还包括把回复记录到历史中的时间；在另一个进程写入 `history.db` 时，这一步最多可能等待 5 秒。

在构建事件之前，`thinking` 和 `content` 会经过最小化的敏感信息脱敏，把每一处替换为 `[redacted]`：

- PEM 私钥块的正文，从 `-----BEGIN … PRIVATE KEY-----` 开始（RSA、EC、DSA、OPENSSH、ENCRYPTED 或普通格式；也包括 PGP 的 `PRIVATE KEY BLOCK`），到 END 行为止；如果缺少 END 行，则到文本末尾为止：两行标记保留，正文变成一行 `[redacted]`。公钥和证书保持不变；
- 已知的密钥格式：`sk-` 密钥、GitHub 令牌（`ghp_…`、`github_pat_…` 等），以及 AWS 访问密钥 id（`AKIA…`、`ASIA…`）；
- `Bearer` 之后的令牌；
- 与 `verification code`、`passcode`、`OTP` 或 `验证码` 位于同一行且相距 20 个字符以内的 4 到 8 位数字。

脱敏作用于整段文本，发生在截断之前，并且只针对带有接收函数的运行。工作记忆、运行发送的请求、`RunResult`、`run.finished` 保留模型写出的原文；历史记录则以同样的脱敏方式存储其任务和步骤，模型调用也一样（见[模型调用](memory.md#model-calls)）。

<a id="toolstarted"></a>

### tool.started：工具调用开始

- `step`、`call_id` 和 `name`：步骤、工具调用的 id 和工具名。
- `arguments`：模型发送的参数，原样保留。
- `preview`：调用的一行摘要，最多 200 个字符；更长的保留前 197 个字符，并以 `...` 结尾。

| 工具 | 预览 |
|---|---|
| `shell_exec` | 命令，连续的空白会被压缩。 |
| `python_exec` | 代码所运行的命令：代码以字面命令调用 `subprocess.run`、`check_output`、`check_call`、`call` 或 `Popen`（`grep -rn DEFAULT_WORKSPACE .`），或以字面参数调用它自己定义的、会这样做的函数（`run("git", "status")` 写作 `git status`）；否则是第一个不是空行、注释或 import 的行（都没有时取第一行）。若后面还有更多行，则附加 ` (+<n> more lines)`。 |
| `file_read` | 路径。 |
| `list_files` | 路径（不带路径时为 `.`），深度大于 1 时后接 ` (<n> levels)`。 |
| `file_write` | `<path> (<n> chars)`。 |
| 其他任何工具，或上述缺失或不是字符串的参数 | 参数，以紧凑 JSON 形式输出，键已排序。 |

`tool.started` 事件的数量就是 `RunResult.n_tool`。

<a id="toolfinished"></a>

### tool.finished：工具调用结束

- `step`、`call_id` 和 `name`：与该调用的 `tool.started` 相同。
- `status`：调用如何结束：
  - `success`、`error`、`timeout` 或 `blocked`，按执行器报告的结果；
  - `cancelled`：运行被取消，或其 `timeout_s` 在调用期间到期；
  - `crashed`：意外错误使调用离开了执行器或结果检查，运行随之以 `crashed` 结束。
- `error_category`：调用失败的原因，例如 `tool_not_offered`、`invalid_arguments`、`blocked_by_checker`、`path_policy_violation`、`approval_denied`、`timeout` 或 `runtime_error`；成功时为 `none`；状态为 `cancelled` 或 `crashed` 时则为相应的值。
- `output` 和 `error`：经过结果检查之后的结果；结果检查会截断超过 50,000 个字符的输出。`error` 不带 `Error: ` 前缀（模型看到的文本前面会有这个前缀）。被取消的调用，其错误为 `The run was stopped before this call finished; it may have partly run.`；若审批策略仍在对它做决定，则为 `The run was stopped before this call started; it did not run.`。崩溃的调用，其错误为 `<ExceptionType>: <message>`。
- `output_chars`：`output` 的长度，即使事件中的副本被截断也如此（见[载荷规则](#payload-rules)）。
- `duration_s`：工具运行的时间：未运行时为 `0`；对于被取消或崩溃的调用，则为自 `tool.started` 起算的时间。
- `images`：工具返回的图片，每张都是 `{sha256, mime_type, size_bytes}`。`AgentEvent.images` 保存图片本身。
- `approval`：未询问审批策略时为 `null`：对于不需要审批的工具、没有策略的运行，以及在策略会被询问之前就被拒绝的调用。否则为 `{request_id, allowed, reason, remember, origin}`，其中 `origin` 是决定的来源（`user`、`session`、`policy` 或 `timeout`）；策略失败时为 `error`；运行在策略做决定期间被取消时为 `withdrawn`。见[审批工具调用](../guides/python.md#approve-tool-calls)。
- `exit_code`：`shell_exec` 或 `python_exec` 所运行的子进程的退出状态；负值 `-N` 表示被信号 `N` 结束。对于其他工具、子进程从未启动的调用，以及工具因超时或取消而杀死子进程的情况，它为 `null`。非零退出不会改变 `status`：`shell_exec` 仍然成功，其文本以 `[exit code <n>]` 开头；`python_exec` 则以 `Python exited with code <n>` 失败。

<a id="note"></a>

### note：笔记

- `step`、`attempt`：笔记所属的规划步骤和尝试。
- `note_id`：笔记的记录 id，事件、会话和之后的长期存储都使用它。提示词改用短别名（`n4`、`d1`、`g1`）。
- `kind`：`step`（带工具调用的回复）、`answer`、`empty`、`reflection` 或 `digest`。
- `source`：`model` 或 `heuristic`，即谁写下了笔记。
- `text`：经过检查的笔记文本，已脱敏并净化；`chars` 是它的长度。
- `drafts`：回复中草稿的别名，为字符串。
- `remember`、`stage`：之后的记忆是否可以保留它，以及它所处的阶段（`fresh`、`rehearsed` 或 `superseded`）。

CLI 在其计划行出现之前会忽略此事件；会话运行器会保存它，而轮次的最后一条 `digest` 笔记会写入该轮次的 `note` 列。

<a id="memory"></a>

### memory：记忆

每次准入决定都会发出一个事件，在其存储事务提交之后。

- `candidate_id`：笔记 id，或用户事实的新 id。
- `action`：`NEW`、`UPDATE`、`RELATED`、`CONFLICT` 或 `IGNORE`。
- `kind`：条目的类型（`semantic`、`episodic`、`profile`、`reflection`）。
- `admitted_by`：`rehearsal`、`user_request`、`user_statement`、`outcome` 或 `sleep`。
- `entry_id`：新条目的 id，对于 `IGNORE` 为 `null`。

它不携带文本，也不携带条目 id 之外的 id；决策日志才是审计的依据。边界拒绝不发出任何事件，而是计入 `candidates_refused`。

在 `legacy` 中不发出 `memory` 事件。CLI 会忽略它。

<a id="notice"></a>

### notice：通知

| `kind` | 何时 | `message` | `detail` |
|---|---|---|---|
| `stuck` | 因相同的工具调用或错误重复出现，一次尝试结束。 | `The same tool calls or errors repeated; the attempt ends.` | `{}` |
| `budget` | 因一次尝试用完了它的步数上限，该尝试结束。 | `The attempt used its step budget.` | `{}` |
| `empty_reply` | 模型给出了空回复，localmelo 要求它继续。连续第二次空回复会让运行以 `failed` 结束，且不发出通知。 | `The model returned an empty reply; asking it to continue.` | `{"count": <empty replies in a row>}` |
| `unfinished_answer` | 回复只含模型未完成的思考（见[工作原理](../concepts/how-it-works.md)）：它的 `assistant.message` 把正文放在 `thinking` 中，`content` 为空，localmelo 再问一次。它按空回复计数；如果它会是连续第二次，运行改以 `failed` 结束，错误为 `melo did not finish an answer`，且不发出通知。 | `The model's reply ended inside its thinking; asking it for its answer.` | `{}` |
| `no_change` | 仅在 `--brief` 运行中：在一次尝试边界，下一次尝试开始时发现工作区与运行开始时相比没有变化（`git status`、`git diff HEAD` 和未跟踪文件的哈希相同；在 git 树之外从不发生）。下一次尝试会加入一行用户消息 "Nothing has changed in the workspace yet. Start this attempt by writing the part of the task you can already write, then continue."，若反思有 next_promising_directions，则以 "Start with: <first next_promising_directions of the reflection>" 结尾。当一次尝试中连续 6 个步骤里哈希自上一次变化（或开始）以来都没有变化时，同样的 kind 带上 `scope: "step"` 出现；此时加入的一行是 "You have read for 6 steps without changing anything. Write the part of the task you can already write now (the smallest item first), then continue."。 | `Nothing has changed in the workspace yet.` | `{}`（边界）或 `{"scope": "step"}` |
| `brief_check` | 仅在 `--brief` 运行中（`AgentOptions.brief_check`）：在运行会以 `completed` 结束之前，关卡运行了简报中的一条 `Check:` 命令。 | `The task's check passed.`、`The task's check failed.` 或 `The task's check was refused and did not run.` | `{"round", "command", "exit_code", "duration_s", "refused", "output_tail"}`：`round` 从 1 开始计数关卡的运行次数（在最后一次尝试结束前两步运行的检查也计入）；`command` 已脱敏；`output_tail` 是检查输出的最后 20 行，已脱敏，最多 4,000 个字符；对于被拒绝的检查或没有退出状态的检查（超时），`exit_code` 为 `null`；`refused` 是被拒绝的检查未能运行的原因，否则为 `""`。 |
| `brief_check_failed` | 关卡放弃：检查失败且已没有重试机会（每次运行允许两次）、没有可供重试的步骤或尝试，或检查被拒绝。运行以 `completed` 结束。 | `The task's check still fails; the run ends completed with failing tests.` | `{"retries": <retries used>}` |
| `retry` | 模型请求失败，并在延迟后重新发送；见[提供方错误与重试](run-results.md#provider-errors-and-retries)。 | `Model request failed; retrying in <delay> s: <the first 200 characters of the error>` | `{"request_attempt", "delay_s", "status_code"}` |

`step` 和 `attempt` 是最近一个 `step.started` 的值。

<a id="recall"></a>

### recall：召回

- `step`、`attempt`：召回块所在的规划步骤和尝试。
- `trigger`：`start`、`attempt`、`step` 或 `tool`。
- `retriever`：回答的检索器；发生回退时为 `bm25`。
- `query`：检索器排序所用的状态查询，已脱敏。
- `hits`：每个命中一个对象，`{doc_id, alias, kind, score, rank, origin_task_ids, chars, omitted}`，不含条目文本。`kind` 为 `note`、`draft`、`thought`、`digest`、`reflection` 或 `long`；`origin_task_ids` 指出产生它的任务；草稿变成指针行时，`omitted` 为 true。
- `chars`：块的字符数；`ms`：它的挂钟时间；`fallback`：`""`，或 `timeout`、`parse` 或 `provider_error`；`placement`：`query`、`tool`、`message` 或 `tool-result`。

该事件是附加的，CLI 目前不为它绘制任何内容。同一条记录也在 `memory_stats.recall_log` 中；带有条目文本的完整记录则在 `llm_calls.recall_json` 中。

<a id="reflectionstarted"></a>

### reflection.started：反思开始

- `attempt`：刚刚结束的尝试，从 1 开始。
- `failure_type`：结束它的原因：`stuck` 或 `budget`。

<a id="reflection"></a>

### reflection：反思

- `attempt`：与 `reflection.started` 中相同。
- `action`：反思的选择：`continue`、`stop`、`decompose` 或 `done`。使用 `reflection_gate="utility"` 时，对于超过大小限制且未选择 `done` 的反思，它为 `decompose`；默认的 `"flags"` 下，这样的反思会被裁剪，并保留其选择。
- `continue_run`：仅当另一次尝试开始时为 `true`。在选择 `done` 之后、最后一次尝试之后，或 `max_steps` 用完之后，以及继续判定停止运行时，它为 `false`。
- `flags`：决定中的 `task_still_feasible`、`next_step_is_concrete` 和 `next_step_is_novel`，均为布尔值；回复无法读取时为 `null`。
- `gate`：生效的 `reflection_gate`，即 `flags`、`utility` 或 `patient`。
- `utility`：使用 `"utility"` 时为决定的效用值（见该选项）；否则为 `null`。
- `stop_reason`：当且仅当 `continue_run` 为 `true` 时为 `null`；否则为下列中第一个适用的原因，顺序如下：`unreadable_reply`（没有决定，且无法开始任何尝试）、`action_not_continue`（动作为 `done`）、`no_attempt_left`（已用完 `max_attempts`；运行以 `Max attempts reached` 结束）、`step_budget_used`（已用完 `max_steps`；`Max steps reached`），然后是继续判定的各项检查：`action_not_continue`（动作为 `stop` 或 `decompose`）、`not_feasible`、`not_concrete`、`not_novel`，以及仅在 `"utility"` 下才有的 `utility_below_threshold`。它由决定它的代码推导得出，因此绝不会与 `continue_run` 相矛盾。`gate_stopped` 运行中若理由为空，则在 `error` 中报告：`Stopped after attempt <n> (stop reason: <code>)`；有理由时，错误就是该理由本身。在 `--brief` 运行中，只要还有步骤和尝试，`continue` 就会继续，因此 `flags` 中可以有 `false`，而 `stop_reason` 仍为 `null`；上面的其他检查不变。
- `rationale`、`summary` 和 `best_effort_result`：反思的理由、它对该次尝试的总结，以及它迄今为止最佳的答案。它们与 `assistant.message` 的 `thinking` 和 `content` 一样经过敏感信息脱敏；`RunResult` 保留模型写出的最佳答案原文。
- `requests`：反思请求的数量，为 1 或 2。若回复中不含反思，则不带工具再请求一次；第二次请求没有自己的事件。默认的 `reflection_gate="flags"` 下，若第一次回复含有解码器拒绝的 JSON 且不是工具调用，重试说明之后会多一行：解码器的错误及其前后最多 60 个字符，例如 `Your reply was not valid JSON (Expecting ',' delimiter at character 51: ...: "continue", "rationale": "r"]). Reply again with one JSON object.` 在压缩模式中，提示拟合可能在请求发出之前就拒绝它，这会被视为无法读取的回复：此时 `requests` 为 0；若只有重试被拒绝，则为 1。
- `readable`：在默认的 `reflection_gate="flags"` 下，第二次回复仍不含反思时为 `false`。此时 `requests` 为 2；`action`、`rationale`、`summary` 和 `best_effort_result` 为 `null`；`continue_run` 为 `true`，除非无法开始任何尝试，那时运行以 `budget_exhausted` 结束。使用 `"utility"` 时，它始终为 `true`：这样的回复计为 `stop`，理由为 `Could not parse reflection response`；若它是有效的 JSON 但不是对象，则理由为 `Reflection response was not a JSON object`。
- `unreadable_reason`：第一次回复为何不含反思：`tool_call`（它带有工具调用，其文本中没有反思；窗口以工具轮次结尾时的常见情况）、`invalid`（其文本含有 `"flags"` 不接受的 JSON 对象）或 `no_json`；可读时为 `null`。无论哪种，之后都会重试，因此 `readable` 可以为 `true` 同时带有原因。
- `repaired`：为使给出该反思的回复可读而修补的那一处 JSON 小错误：`unmatched_closer`（去掉一个没有开括号与之匹配的 `]` 或 `}`）、`trailing_comma`（去掉 `]` 或 `}` 之前的逗号）、`control_char`（转义字符串内原始的换行符、回车符和制表符）、`invalid_escape`（JSON 中不存在的转义，例如 `\;`，其反斜杠被加倍，因此读作一个反斜杠加其后的字符）或 `unclosed_array`（闭合被下一个键打断的数组：在该键之前的逗号前插入一个 `]`）；回复本来就可读、没有读到反思，或使用 `"utility"` 时为 `null`。只有回复本来不含反思时才会尝试修补，且只针对解码器拒绝的 `{...}`（每条回复的前八个）；只有结果是带有效 `recommended_action` 的对象时，修补才算数。需要两处修补的回复（包括有第二个无效转义的回复）仍然读不懂，除这一个反斜杠或 `]` 之外也不会补全或插入任何内容，因此中途截断的文本不会被修补。第一次回复经修补后无需重试：`requests` 为 1，`unreadable_reason` 为 `null`。
- `usage`、`prefix` 和 `duration_s`：两次请求的 token 计数和耗时，包括它们的重试。若服务器发送了 `cached_tokens` 和 `reasoning_tokens`，`usage` 中也会包含它们，并对各次请求求和。`prefix` 是 `assistant.message` 下所描述的对象，针对反思的第一次请求（当它是运行的第一个请求，或未发送任何请求时为 `null`）。启用持久化时，`duration_s` 还包括把回复记录到历史中的时间；在另一个进程写入 `history.db` 时，这一步最多可能等待 5 秒。

另见[反思](../concepts/how-it-works.md#reflection)和[运行结果](run-results.md#statuses)。

<a id="runprogress"></a>

### run.progress：运行进度

用于显示运行所处的位置。只有当 `run_task` 同时收到 `progress=True` 和接收函数时才会出现。

会话轮次的接收函数属于其运行器，因此只有当其 `SessionRunner` 以 `progress=True` 构建时，该轮次才会存储并转发 `run.progress` 事件（见[会话](../guides/sessions.md#run-a-conversation)）。每个键始终存在，未知时为 `null`。

它显示的内容依据运行的例程，即[程序记忆](../concepts/procedures.md)在该模型和这些选项下记住的内容：

| 键 | 含义 |
|---|---|
| `level` | 可以显示的内容：0，仅显示事实（首次运行、另一个模型或其他选项）；1，还显示过去运行的事实以及命名的步骤。第 2 级（进度条）在本版本中不显示。 |
| `state` | `running`，然后在 `run.finished` 之前的事件中为 `ended`。 |
| `step`、`attempt` | 最近一个 `step.started` 的值；第一步之前为 0。 |
| `limits` | `{max_steps, steps_per_attempt, max_attempts, timeout_s}`：运行的预算；运行没有某项预算时为 `null`（当 `attempt_policy="none"` 时为尝试次数限制）。`max_steps` 是预算，而不是预期的长度。 |
| `phase`、`cycle` | `Explore`、`Write`、`Verify` 或 `Finishing`，由运行中工具调用的类别得出（见[类别与阶段](../concepts/procedures.md#classes-and-phases)），以及一次写入之后的测试数量；第一次调用之前为 `null` 和 0。 |
| `tests` | `{passed, failed, total}`，来自测试调用输出中最近一次 pytest 或 unittest 的汇总；否则为 `null`。 |
| `items` | 第 1 级：例程中命名的步骤，每个为 `{label, state, repeat}`，其中 `state` 为 `done`、`current`、`pending` 或 `skipped`；它们只会向前推进。第 0 级时为 `null`。 |
| `fraction`、`expected_steps` | 在本版本中始终为 `null`。 |
| `past` | 第 1 级：`{runs, steps_lo, steps_median, steps_hi, wall_lo_s, wall_hi_s, last_steps, last_wall_s}`，即过去已完成运行的事实。 |
| `off_reason` | 在本版本中始终为 `null`。 |
| `basis` | 第 1 级：`{routine_id, template_id, version, match, model, title}`，即该显示所依据的记忆；`match` 为 `exact`、`routine`（命名的例程）或 `similar`（相似任务的步骤，仅作提示）；`title` 指出 `similar` 匹配借用了哪个例程，否则为 `null`。 |

首次运行没有可供依据的过去：它的事件只有事实。下面的接收函数会打印每个事件的级别、状态、步骤和阶段：

```python
import asyncio

from localmelo import Agent, AgentEvent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


def show(event: AgentEvent) -> None:
    if event.type == "run.progress":
        data = event.data
        print(data["level"], data["state"], data["step"], data["phase"])


async def main() -> None:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm, options=AgentOptions(builtins=["shell_exec"]))
    try:
        await agent.run_task("Which files are here?", observe=show, progress=True)
    finally:
        await agent.close()


asyncio.run(main())
```

当模型运行 `ls` 然后作答时，它会打印：

```text
0 running 0 None
0 running 1 None
0 running 1 Explore
0 running 2 Explore
0 ended 2 Explore
```

在第 1 级，应把各项显示为最近几次运行做了什么，而不是待办清单：这次运行可能会做其他工作。运行开始时，会在至多 1 秒内查找一次该例程；这段时间计入运行的 `timeout_s`。

若查找失败或耗时更长，则为第 0 级。进度跟踪器内部的失败只会结束运行的 `run.progress` 事件，而不会结束运行本身。

<a id="runfinished"></a>

### run.finished：运行结束

- `result`：运行的 `RunResult`，即 `RunResult.to_dict()`。其中没有任何内容被截断，因此 `RunResult(**event.data["result"])` 可以重建它。

当 `run_task` 因运行被取消而抛出异常时，也会发出 `run.finished`；此时它保存 `run_task` 本会返回的 `RunResult`。

<a id="guarantees"></a>

## 保证

- **一次开始，一次结束。** 每个通过参数检查的运行都会先发出 `run.started`（`seq` 为 1），最后发出 `run.finished`。无论状态如何，二者各只发出一次，即使第二次取消打断了收尾也是如此。
- **工具调用成对出现。** 每个 `tool.started` 之后，都恰好跟着一个具有相同 `call_id` 的 `tool.finished`，且它出现在下一个 `tool.started`、`step.started` 或 `run.finished` 之前，无论状态如何。回复中因运行在其之前停止而从未开始的调用，不会有工具事件：它只出现在 `assistant.message` 的 `tool_calls` 中。
- **步骤内的顺序。** 先是 `step.started`；然后是规划请求的 `retry` 通知；接着是 `assistant.message`（除非规划失败）；随后是工具事件对，按调用顺序排列；最后是 `stuck`、`budget`、`empty_reply` 或 `unfinished_answer` 通知。在默认的 `attempt_policy="reflect"` 下，结束一次尝试的通知之后紧跟 `reflection.started`、反思请求的 `retry` 通知，以及 `reflection`，除非其中某个请求失败或运行被取消。若为 `"none"`，则之后紧跟 `run.finished`。`no_change` 通知在该 `reflection` 之后出现，即下一次尝试开始时。直接回答的 `brief_check` 通知紧跟其 `assistant.message`；`done` 反思的通知则出现在其 `reflection` 事件之前，当失败的检查推翻了 `done` 时，该事件会显示 `continue_run: true` 和 `stop_reason: null`。
- **进度与其报告的内容一致。** 设置 `progress=True` 时，`run.progress` 紧接在 `run.started` 之后出现，紧接在每个 `step.started`、`tool.finished`（针对已返回的调用）、`reflection.started` 和 `stuck` 通知之后出现，并紧接在 `state` 为 `ended` 的 `run.finished` 之前出现，在所有路径上都是如此。若不设置 `progress=True`，运行的事件与 `run.progress` 出现之前完全相同。
- **计数与运行相符。** `step.started` 事件的数量就是 `RunResult.steps`，`tool.started` 事件的数量就是 `RunResult.n_tool`。`run.finished` 的 `result` 等于 `run_task` 返回的（或本会返回的）`RunResult` 的 `to_dict()`。运行中的 `assistant.message` 和 `reflection` 事件的 `usage` 加起来，等于 `RunResult.usage` 的 token 计数（`prompt_tokens`、`completion_tokens`、`total_tokens`）。显式推理模式还会在 `RunResult.usage` 中放入 `reasoning_chars`；服务器报告时还会有 `cached_tokens` 或 `reasoning_tokens`，这些都不由任何事件携带。在下列情况下，事件的数量会少于运行的计数，因为回复被计入却没有任何事件携带它：
  - 回复到达之后，结果检查抛出异常，这使运行以 `crashed` 结束；
  - 反思的第二次请求在第一次返回之后失败；
  - 启用持久化时，取消中断了回复写入历史记录的过程。
- **一个 id。** 每个事件的 `run_id` 都是运行的 `task_id`，与 `RunResult` 和任务记录中相同。

<a id="payload-rules"></a>

## 载荷规则

`data` 遵循以下规则，由 `localmelo.melo.contracts` 中的 `json_safe_payload` 应用：

- **JSON 安全。** `data` 只包含字符串、整数、有限浮点数、布尔值和 `null`，且位于以字符串为键的列表和对象中。工具调用参数中的其他值会变成其 `str()` 文本。非有限的浮点数会变成 `"nan"`、`"inf"` 或 `"-inf"`。
- **截断。** 超过 100,000 个字符的字符串保留前 100,000 个字符，并以 `… [+N chars]` 结尾，其中 `N` 为被省略的字符数。这适用于任务、回复的文本和思考、参数、输出、错误、反思的各字段以及其他所有字符串，但 `run.finished` 除外，其中的 `RunResult` 是完整的。脱敏先于截断进行。
- **嵌套。** 列表和对象的嵌套最多 100 层，`data` 的值算作第 1 层。比这更深的列表或对象会变成其 `str()` 文本，并像其他字符串一样被截断。模型的工具调用参数可能嵌套得比 Python 能复制的还要深，而事件仍然会被构建。
- **图片以哈希标识。** `data` 只以图片的 SHA-256 哈希、MIME 类型和大小来标识图片。
- **只读。** `to_dict()` 返回一个新字典，其中 `data` 为浅拷贝。请把载荷当作只读数据。

<a id="session-events"></a>

## 会话事件

[会话](sessions.md)会存储每一轮的事件：即运行的事件（如上所述），位于会话自身添加的两个事件之间，并且会话还会增加少量事件。每个存储的事件都有全局的 `id`、`session_id` 和 `turn`，其 `event` 是 `to_dict()` 的外层结构；会话事件的 `"seq": null` 且 `"run_id": ""`。

| 类型 | 何时 | `data` 键 |
|---|---|---|
| `turn.started` | 一轮开始，先于其运行的事件。 | `turn`, `turn_id`, `user_text`, `client_request_id` |
| `turn.finished` | 该轮结束：它的最后一个事件。 | `turn`, `turn_id`, `status`, `error`, `cause` |
| `turn.undone` | 某轮被撤销；该事件不属于任何轮次。 | `turn`, `turn_id` |
| `turn.pictures` | 控制台的一轮产生了图片：在其运行的最后一个事件之后、`turn.finished` 之前。没有图片时不存在，终端的文件里也没有。 | `turn`, `turn_id`, `pictures` |
| `turn.memory` | [跨对话记忆](../concepts/memory.md#memory-across-chats)为控制台的一轮做了什么，在其运行的第一个事件之前；开启记忆的每一轮都有；对话的模型每用 `remember` 记一条笔记或试图保存密钥时，会以同样的形状完整地再发送一次（以最新的一条为准）：`used`（`[{kind: record\|chat, id, text, source, session_id, turn, at}]`，召回到消息中的内容）、`remembered`（`[{id, text, topic}]`，这一轮记下的笔记：先是输入的文字请求记住的，然后是模型记的）、`refused`（某些内容没有被保存的原因，或 `""`）和 `recall`（搜索的过程：`{retriever: bm25\|embedding\|llm, fallback: none\|timeout\|error\|unreadable\|no-embedding, embed_ms, search_ms, select_ms, select_tokens: {prompt, completion}, candidates, used, block_chars, context: off\|used\|none}`（`block_chars`：记忆加到消息里的块的长度，没有时为 0；`context`：`[memory] context_query` 有没有把上一条消息加进搜索，没有上一条时为 `none`），即比较 `[memory] retriever` 所用的数字；该轮已保存的 `result` 里以 `memory_recall` 重复一份）。记忆关闭时不存在，终端的文件里也没有。 | `turn`, `turn_id`, `used`, `remembered`, `refused`, `recall` |
| `session.title` | 第一轮设置会话标题，或标题被重命名。 | `title` |
| `session.updated` | 会话的模型、工作区或 `meta` 发生变化。 | `model`, `workspace`, `workspace_label`, `meta` |
| `approval.requested` | 向用户询问某个工具调用（`SessionRunner.observe`）。 | `request_id`, `call_id`, `tool_name`, `arguments`, `preview`, `workspace_root` |
| `approval.resolved` | 该请求得到答复或被撤回。 | `request_id`, `call_id`, `tool_name`, `outcome`, `origin`, `reason`, `remember` |

- `workspace_label` 是把主目录缩写为 `~` 后的工作区，未设置工作区时为空字符串。较早存储的 `session.updated` 事件可能没有此键；客户端必须容忍它的缺失。
- `client_request_id` 是该轮[提交](sessions.md#message-ids-and-submissions)的消息 id，或为 `null`。
- `cause` 说明是什么结束了该轮：`run`、`user_stop`、`closed`、`shutdown`、`crash`、`takeover` 或 `park`；见[轮次、状态与原因](sessions.md#turns-statuses-and-causes)。
- `pictures` 最多列出 24 项：该轮在其文件夹里创建或修改的图片文件（`.png`、`.jpg`、`.jpeg`、`.gif`、`.webp`、`.svg`，通过比较运行前后的文件夹得知），以及回答中提到且位于该文件夹内的图片文件。每项为 `{"path", "size", "mtime", "state"}`：`path` 是相对于该文件夹、以 `/` 分隔的路径，`size` 的单位是字节，`mtime` 的单位是秒，`state` 为 `new`、`modified` 或 `mentioned`。这些文件由 [`GET /api/sessions/{id}/pictures`](api-v2.md#get-apisessionsidpictures) 提供。
- `outcome` 为 `allowed`、`denied` 或 `withdrawn`。`origin` 是决定的来源，与 `tool.finished` 的 `approval` 中相同：`user`、`session`、`policy` 或 `timeout`；询问失败时为 `error`；运行在决定之前停止，以及存储为一个从未存储答复的请求写入事件时，为 `withdrawn`（见下面的保证）。
- `turn` 对于不属于任何轮次的事件为 `null`：`turn.undone`、重命名时的 `session.title`、`session.updated`，以及在轮次之外存储的调度器事件。
- **调度器的类型。** 以 `task.`、`sched.` 或 `attention.` 开头的类型保留给调度器。只有网关的存储会保存它们，作为不属于任何轮次的事件（`SessionStore.append_session_event`），或属于某一轮（`SessionRunner.publish`）。它们的键尚未固定；其 `data` 遵循[载荷规则](#payload-rules)。存储会拒绝其他所有类型。
- **`attention.card`** 是其中唯一键已固定的类型：主对话的卡片（`source_id`、`source_kind`、`title`、`state`、`line`、`at`，等待或已答复的卡片还有 `request_id`），在旁支对话的运行或任务有结果时存入主对话，不属于任何轮次；见[主对话中的卡片](api-v2.md#cards-in-the-main-chat)。
- **`attention.echo`** 是主对话关于它开始、改变或交给消息的任务的一行回执（`echo`、`line`，以及按种类需要的 `task_id`、`title`、`level`、`action`、`undo`、`of`、`typed`、`text`、`turn_id`、`how`），不属于任何轮次；见[主对话中的回执行](api-v2.md#echoes-in-the-main-chat)。
- **存储的 `run.finished` 保存脱敏后的答案。** 在存储之前，它的 `RunResult` 中的 `result` 会经过与 `assistant.message` 文本相同的脱敏，该轮存储的答案和结果也一样。运行返回的 `RunResult` 保留原文。

存储提供以下保证：

- **一次开始，一次结束。** 每个 `turn.started` 之后，在该轮不再运行时，都会恰好跟着该轮的一个 `turn.finished`，即使在接管、关闭或崩溃之后也是如此；该轮不会再有任何事件出现在它之后。
- **`turn.finished` 关闭该轮。** 应把它理解为关闭该轮中所有仍然打开的卡片：只有 `tool.started` 而没有 `tool.finished`，读作“已中断：可能已部分运行”；没有结束的步骤则已经结束。崩溃之后，运行自身的事件可能在任何位置停止，存储不会为该运行补写事件。
- **每个请求都会得到答复。** 每个 `approval.requested` 之后，都恰好跟着一个 `approval.resolved`。崩溃之后，恢复过程会写入它，结果为 `withdrawn`，原因为 `melo stopped before this approval was answered.`；当某一轮结束时，若其中某个请求的答复尚未存储，则由该轮的结束过程写入它，结果为 `withdrawn`，原因为 `The turn ended before this approval's answer was stored.`
- **被撤回的消息不在流中留下痕迹：** 没有轮次编号，也没有事件。
- **id 在重启后仍然有效。** 事件 id 只会增长，跨会话、进程和重启都是如此，因此以其最后一个 id 重连（`Last-Event-ID`）的客户端，会收到它错过的事件，包括恢复过程写入的事件。

<a id="stability"></a>

## 稳定性

在架构版本 1（`v` 为 1）内，类型和键只会增加，绝不会删除或重命名。忽略你不认识的类型和键。

在 `prefix` 出现之前存储的 `assistant.message` 或 `reflection` 不包含该键（`localmelo.melo.contracts.events` 中的 `OPTIONAL_EVENT_FIELDS` 列出了这类键，会话存储也接受缺少它们的事件）。`localmelo.melo.contracts` 中的 `EVENT_FIELDS` 把每个类型映射到其键，`localmelo.support.sessions.store` 中的 `SESSION_EVENT_FIELDS` 则映射每个会话类型。
