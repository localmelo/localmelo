<a id="how-localmelo-works"></a>

# localmelo 如何工作

一次运行中发生的事：步骤循环、尝试之间的反思、每一步提供的工具、记忆如何参与，以及时间限制如何工作。各参考页面列出选项；本页解释它们如何配合。

<a id="the-run-loop"></a>

## 运行循环

```text
task
 └─ step: build a request ─► model reply ─┬─ text, no tool call ─► the answer (completed)
        ▲                                 └─ tool calls ─► checks ─► run ─► results
        └───────────────── next step ◄─────────────────────────────────────┘
an attempt that runs out of steps, or repeats itself ─► reflection ─► done, continue or stop
```

一次运行分步处理一项任务。每一步：

1. 根据任务、最近的对话（工作记忆）、相关长期记忆和最新反思（如果有）构造规划请求；
2. 向模型提供工具列表（见[每一步的工具](#tools-in-each-step)）；
3. 发送请求并读取回复。

含有工具调用的回复会依次经过检查并执行调用，结果加入对话。只有文本、没有工具调用的回复是最终答案：运行以 `completed` 结束。

运行有预算：

- 总共最多 30 步（`--steps`）。
- 默认 `reflect` 策略下，步骤分为多次尝试，每次最多 10 步，一次运行最多 5 次尝试。
- 使用 `--attempt-policy none` 时，一次运行只有一次尝试，最多 `--steps` 步；相同的工具调用或相同的错误连续三步重复，运行以 `stuck` 结束。

有三种回复不算回答：

- 既没有文本也没有工具调用的回复是空回复：localmelo 会要求模型继续，这一步仍计入预算，连续两个空回复会使运行以 `failed` 结束。
- 只含模型未完成思考的回复绝不会被当作回答。localmelo 只凭事实识别它，从不看措辞：一个始终没有关闭的 `<think>`、去掉思考后没有回答、流式回复重复了服务器发送的推理，或者对于用标记（`</think>`）结束思考的模型，回复既没有思考也没有标记。模型的格式取 `[cli.reasoning_tags]` 的声明（在终端中），否则取它先前回复所表现的格式。
- 这样的回复会成为它的思考，并按空回复处理：localmelo 再问一次，第二个回复仍没有回答时，运行以 `failed` 结束，错误为 `melo did not finish an answer`。
- 被告知不要思考（请求中为 `enable_thinking = false`）的模型没有思考可发送。对它来说，普通回复就是答案；回复没有其他文字时，服务器放在推理字段里的文字也是答案。只有一个始终没有关闭的 `<think>`，或服务器计入的推理 token，才仍会让回复被视为未完成。
- 把工具调用写成文本的回复同样绝不会被当作回答。模型的服务器可能吞掉一个格式错误的调用，只在回复里留下 `text`（第一个参数的名字）或 `<function_calls>` 这样的标记，却没有真正的工具调用。localmelo 按形状识别它：代码之外出现的标记，或回复开头是本步提供的某个工具的参数名，后面紧跟空、`=`、`<`、不带空格的冒号或一个符号。它按空回复处理，并带有告诉模型哪里出错的专用消息；连续第二次会使运行以 `failed` 结束，错误为 `tool call written as text`。在代码块中讨论这类标记的回复是正常回答。

使用 `--reasoning compress` 时，每个回复会在下一步之前被压缩成一条简短笔记：

- 笔记记录回复的计划、发现、决定和未解决的问题，以及工具返回的内容；模型也可以在 `<note>` 块中自行编写。
- 笔记、回复中的代码草稿和每次尝试的一份摘要，保存在对话窗口旁边仅驻留 RAM 的笔记存储中。
- 循环随后进行召回：每次尝试开始时，一块冻结的先前思考附在任务查询上；每个工具轮次附上窗口不再显示的笔记和草稿；`recall` 工具让模型请求更多内容。
- 每个块都有边界，被标记为数据并经过候选项边界检查；首次携带它的请求将其记录在 `history.db` 的 `llm_calls.recall_json` 中。
- `legacy` 保持原有路径不变。见[循环中的召回](memory.md#recall-in-the-loop)。

<a id="reflection"></a>

## 反思

默认 `reflect` 策略下，一次尝试用完步骤，或连续三步重复自身后，会进行反思：这是一次不带工具的请求，要求模型回顾本次尝试并回复一个 JSON 对象（摘要、失败之处、有帮助的内容和一个行动）。

| 行动 | 结果 |
|---|---|
| `done` | 任务完成。运行以 `completed` 结束，并以反思中的答案作为结果。 |
| `continue` | 如果继续判定同意，就开始新尝试，并将反思放入其提示。 |
| `stop` | 运行以 `gate_stopped` 结束。 |
| `decompose` | 任务应当拆分。localmelo 不拆分任务，因此运行以 `gate_stopped` 结束。 |

- 只有当反思认为任务仍可行，并提出具体的下一步时，继续判定才允许运行继续；否则运行以 `gate_stopped` 结束。
- 默认的判定很有耐心：不新的下一步仍会继续，每次运行一次；第二次则结束运行。
- 使用 `--reflection-gate flags` 时，不新的下一步总会结束运行。
- 使用 `--reflection-gate utility`（该选项出现前的行为）时，反思还必须估计再次尝试的收益超过成本和重复风险：效用分数至少为 0.1（[评测](../project/evaluations.md#reflection-gate)比较了这些判定）。
- 反思的 `useful_evidence`、`failed_hypotheses` 和 `recommended_avoids` 各最多保留 8 项，每项最多 300 个字符（使用 `--reflection-gate flags -X agent.answer_on_stop=off` 时没有这个限制）。

关于反思的更多说明：

- 只带着最新的反思继续：每次反思替换上一条。默认情况下，它只作为一条普通消息加入对话一次；使用 `--cache-layout v2` 时，它会被重建为每个请求的最后一条消息。
- 超过约 2,000 个字符的反思会被截短以适应预算，决定不变（先删去列表末尾的条目，再截短摘要末尾）；使用 `--reflection-gate utility` 时，超长反思会直接结束运行，除非它选择了 `done`。
- 如果无法开始下一次尝试，反思会收到这一信息；它仍可选择 `done`，否则运行以 `budget_exhausted` 结束，并以反思给出的最佳答案作为结果。
- 如果回复没有反思（没有包含四种行动之一的 JSON 对象），localmelo 会再请求一次；如果第二次回复仍没有，则不带新反思开始下一次尝试，或以 `budget_exhausted` 结束（见[运行结果](../reference/run-results.md#statuses)）。
- 如果回复的 JSON 只有一处常见小错误，例如字符串值后多出一个 `]`、JSON 没有对应转义的字符前多出的反斜杠，或被下一个键打断而没有闭合的数组，localmelo 会先修补再读取，无需第二次请求。

<a id="ending-with-an-answer"></a>

### 以答案收尾

- 以 `gate_stopped` 或 `budget_exhausted` 结束且没有答案的运行，会再发出一次不带工具的请求，根据已收集的内容索要最终答案；允许回答 `unknown`。
- 任务要求有收尾行（例如 `FINAL ANSWER: ...`）而答案里没有时，会再发出一次请求索要这一行，并把它加在答案之后。
- 没有指明收尾行的任务保持原样；请求失败时，答案保持原样。
- `-X agent.answer_on_stop=off` 会关闭这两种请求。

<a id="tools-in-each-step"></a>

## 每一步的工具

- 每一步都向模型提供一份工具列表，只有列表中的工具能在该步运行。
- 默认列表包含所有内置工具，以及与任务匹配或模型已经在本次对话中调用过的少数其他工具。
- 见[如何提供工具](../reference/tools.md#how-tools-are-offered)。

<a id="memory-in-the-loop"></a>

## 循环中的记忆

- 每一步的规划请求都包含工作记忆中的最近对话，每个工具调用都会记录在历史记录中。
- 在 `legacy` 下，启用记忆并配置嵌入后端时，请求还会包含与任务最接近的长期记忆；每个工具调用会被总结并写入长期记忆，运行结束时最新反思也会写入其中。
- 运行结束时，运行所做的事（步骤数量以及工具调用的类别和标签）也会记入[程序记忆](procedures.md)，它不会被放入任何提示。
- 在显式模式下，回复的笔记、尝试摘要和反思是长期记忆的**候选项**：只有信号准入时才会写入（排练、用户陈述、涵盖本次运行的用户请求，或对最终摘要使用 `--admit-on-outcome`）。
- 经过检查的写入会对事实分类，与同类活动条目比较，并在存储的决策日志中记录一个动作（`NEW`、`UPDATE`、`RELATED`、`CONFLICT` 或 `IGNORE`）；所有存储文本都会先脱敏。

限制和确切行为见[记忆如何工作](memory.md#how-memory-works)和[长期记忆](../../reference/memory.md#long-term-memory)（英文）。

<a id="timeouts"></a>

## 超时

三个限制分别适用于不同层级：

| 限制 | 取值 |
|---|---|
| 对话请求超时 | 120 秒（`ollama` 后端为 300 秒）：localmelo 等待模型服务器数据的时间，每次收到数据时计时器都会重置；超时的请求会重试两次。见[对话请求超时](../reference/configuration.md#chat-request-timeout)。 |
| 工具时限 | `shell_exec` 为 30 秒，`python_exec` 为 60 秒，`file_read`、`file_write` 和 `list_files` 为 60 秒，MCP 工具使用其服务器的 `timeout`；交互式界面给前两者 10 分钟。`tool_timeout_s` 见[在目录中工作](../guides/python.md#work-in-a-directory)。 |
| 运行时限（`--timeout`） | 对整次运行唯一的时限；到期时，运行以 `timeout` 结束。 |

- `shell_exec` 和 `python_exec` 在带进程组的子进程中运行；超时或取消会终止进程组。
- 直接运行、worker 和网关不流式接收回复，所以直到完整回复生成后才发送数据的服务器必须在超时内完成；思考较慢的模型可能需要更多时间。

<a id="the-main-chat-and-tasks"></a>

## 主对话与任务

![主对话开始任务并显示它们的级别](../../assets/img/main-chat-tasks.webp)

- 主对话是一次简短的运行：最多 4 步，不做反思，关闭思考，带有开始、列出和改变任务、读取文件和搜索记忆的工具。
- 代码先读你的消息。`/task` 和控制短语（"pause the slides"、"暂停 Fix CI"）不需要调用模型；问候或简短的问题会立刻得到回答，不会被作为任务开始。
- 较大的工作成为任务：一个 worker 在对话旁边运行它，级别由你的话决定，结果以卡片回来。
- 4 步内没有办完的运行会以 "This needs more than a quick answer, so I started it as a task" 结束；如果你提的是问题，会先问你。
- 旁支对话、直接运行和 worker 照旧运行上面完整的循环。

<a id="code-layout"></a>

## 代码布局

- `melo/` 包含智能体本身：运行循环、检查、工具执行器、记忆和模型提供方实现的接口。
- `support/` 负责连接外部：后端和模型提供方、网关及其网页工作区、MCP 服务器、配置和配置向导。
- 它还包含离线比较记忆检索器的 `melo-retrieve` 命令，以及调度器（网页工作区的对话使用其门控；直接运行不使用）。
- 见[架构页面](https://localmelo.github.io/localmelo/en/architecture.html)（英文）和[项目布局](https://github.com/localmelo/localmelo/blob/main/CONTRIBUTING.md#project-layout)（英文）。
