<a id="run-results"></a>

# 运行结果

每次运行都以一个状态和一份它做过什么的记录结束。本页列出状态、记录中的字段，以及失败的模型请求如何重试。

<a id="statuses"></a>

## 状态

| 状态 | 退出码 | 含义 |
|---|---|---|
| `completed` | 0 | 模型给出了最终答案，或者一次反思判定任务已完成。 |
| `failed` | 2 | 某项检查拒绝了运行，或者模型连续两次给出空回复。 |
| `gate_stopped` | 2 | 在一次尝试之后，反思决定不再继续。 |
| `budget_exhausted` | 2 | 运行用完了所有步骤或尝试次数，却没有最终答案。 |
| `context_budget_exceeded` | 2 | 固定的请求放不进配置的输入预算，或者缩小后的重试仍然溢出。 |
| `stuck` | 2 | 相同的工具调用或相同的错误连续三步重复出现。 |
| `provider_error` | 2 | 对模型服务器的一次请求失败。 |
| `setup_error` | 2 | 无法设置工具提供方（例如 MCP 服务器），或者无法安全读取 worker 收件箱。 |
| `timeout` | 2 | 时间限制（`--timeout`）到期时，运行仍在进行。 |
| `crashed` | 1 | 意外错误、Ctrl+C、SIGTERM 或 SIGHUP 结束了运行。 |
| `cancelled` | 无 | 交互式运行被停止：Python API 中带 `on_event` 或 `approval` 的运行，或者用 Stop 或其 600 秒上限停止的网页工作区聊天。直接模式、`melo ""` 循环和 `POST /v1/agent/run` 从不报告它。 |

细节：

- **`completed`。** 当反思判定任务已完成时，答案是它的尽力结果；如果那是空的，就是该次运行中模型写下的最新文本。
- **`failed`。** 错误说明是哪项检查拒绝了运行，例如 `Plan check failed: Prompt too large: <n> chars` 或 `Response check failed: …`。既没有工具调用也没有文本的回复是空回复：localmelo 要求模型继续并重新规划，这一步照常计数。连续两次空回复会以错误 `empty replies` 结束运行。
- **`gate_stopped`。** 反思选择了 `stop` 或 `decompose`，或者它继续的决定没有通过继续判定（见[反思](../concepts/how-it-works.md#reflection)）。`result` 保存反思的尽力结果（如果有），`error` 保存它的理由；如果它没有给出理由，则是 `Stopped after attempt <n>`。使用 `--reflection-gate utility` 时，读不懂的反思也会结束运行；超过大小限制的反思同样如此，除非它选择了 `done`。
- **读不懂的反思。** 当反思回复包含一个 `recommended_action` 为 `done`、`continue`、`stop` 或 `decompose` 的 JSON 对象时，它才是可读的；其他 JSON（例如写成文本的工具调用）不算。读不懂的回复会被再请求一次，且不带工具：该次尝试的轮次以文本写出，结尾一行只要求那个 JSON 对象。两次请求都计入 `n_aux`。如果第二次回复也读不懂，其中任何内容都不保留，下一次尝试随即开始。如果无法开始任何尝试，运行以 `budget_exhausted` 结束，并把该次运行中模型写下的最新文本作为 `result`。使用 `--reflection-gate utility` 时，任何 JSON 对象都算可读；第二次回复若没有 JSON 对象，则以 `gate_stopped` 结束运行：回复无法读取时错误是 `Could not parse reflection response`，是有效 JSON 但不是对象时是 `Reflection response was not a JSON object`。如果没有剩余尝试，运行反而以 `budget_exhausted` 结束。
- **`budget_exhausted`。** 错误是 `Max steps reached` 或 `Max attempts reached`。使用 `--attempt-policy reflect` 时，最后一次反思仍可选择 `done`；否则它的尽力结果成为 `result`。
- **`stuck`。** 只出现在 `--attempt-policy none` 下；使用 `reflect` 时，由反思决定接下来做什么。错误是 `Stuck: the same tool calls or errors repeated`。返回新结果的重复调用（例如轮询一个任务）不算卡住。
- **`provider_error`。** 见[提供方错误与重试](#provider-errors-and-retries)。
- **`setup_error`。** 错误会指出问题，例如 `MCP server 'notes' exited (code 1)`。智能体不会重试 MCP 或工具提供方的设置失败：它之后的运行也会以 `setup_error` 结束。位于工作区内、不是常规文件、过大、被替换、被截断或无效的 worker 收件箱，也会以 `setup_error` 结束并给出普通消息；该失败只关乎这个文件，因此之后有一次可读收件箱的运行会正常进行。
- **`timeout`。** 错误是 `Timed out after <n> s`。在途的模型请求被放弃，`shell_exec` 命令或 `python_exec` 子进程连同它启动的所有进程一起被终止。
- **`crashed`。** 错误是 `<ExceptionType>: <message>`，并且会记录 traceback。Ctrl+C 以 `crashed` 和错误 `CancelledError` 结束运行，SIGTERM 或 SIGHUP 则以 `Cancelled by SIGTERM` 或 `Cancelled by SIGHUP` 结束。见[信号](../../reference/cli.md#signals)（英文）。
- **`cancelled`。** 只用于任务被取消的交互式运行：在 Python API 中用 `on_event` 或 `approval` 启动的运行（见[停止运行](../guides/python.md#stop-a-run)），或者用 Stop 或其 600 秒上限停止的网页工作区聊天。网关为每个网页聊天用一个事件接收器运行（见[网页工作区](../guides/web-console.md#web-workspace)）。`result` 为空，`error` 是原因：进程正在退出时是传给 `Agent.begin_exit` 的原因；否则是 `task.cancel(msg)` 的消息，例如网页聊天的 `Timed out after 600 s`；再否则是 `Cancelled by the caller`。没有 `on_event` 或 `approval` 的取消运行以 `crashed` 结束，因此直接模式、`melo ""` 循环和 `POST /v1/agent/run` 从不报告 `cancelled`。

[localmelo 如何工作](../concepts/how-it-works.md)解释了尝试、反思和继续判定。

<a id="a-session-turns-cause"></a>

### 会话一轮的原因

一次[会话](../../reference/sessions.md)（英文）的一轮也以这些状态之一结束，并带有一个说明是什么结束了它的 `cause`：

| `cause` | 结束该轮的原因 | 状态 |
|---|---|---|
| `run` | 运行本身，无论它的状态是什么。 | 任意 |
| `user_stop` | 有人停止了它。 | `cancelled` |
| `closed` | 会话的运行器被关闭了。 | `cancelled` |
| `shutdown` | 进程被要求停止。 | `cancelled` |
| `crash` | 进程死掉；恢复过程记录了结束。 | `crashed`，或者存储的 `run.finished` 如此说明时为 `cancelled` |
| `takeover` | 另一个终端接管了一轮，其租约在另一台主机上已过期。 | `cancelled` |
| `park` | 预留给调度器。 | `cancelled` |

“Interrupted”是界面为 `shutdown` 和 `crash` 两个原因显示的标签，绝不是状态：一轮的状态始终是上面的状态之一。原因为 `run` 的 `crashed` 是运行内部的崩溃，属于缺陷；原因为 `crash` 的 `crashed` 表示进程停止了。

<a id="fields"></a>

## 字段

`--output-format json` 按此顺序打印这些字段，最后是 `localmelo_version`。Python API 把它们作为 `RunResult` 返回。

| 字段 | 含义 |
|---|---|
| `task_id` | 本次运行在历史记录中的任务记录 id：32 个十六进制字符。 |
| `status` | 运行如何结束；见[状态](#statuses)。 |
| `result` | 最终答案或尽力结果。它可能为空。 |
| `error` | 运行为何没有完成。完成时为空。 |
| `steps` | 做出的规划步骤数，最多 `max_steps`。 |
| `attempts` | 开始的尝试次数：第一次尝试就给出答案时为 1。 |
| `n_llm` | 返回了回复的模型请求数。 |
| `n_main` | 其中提供了工具的请求数。 |
| `n_aux` | 其中不带工具的请求数：反思、召回发出的检索请求，以及步骤中没有提供任何工具时的规划请求。 |
| `n_tool` | 交给执行器的工具调用数，包括被拒绝的调用。 |
| `llm_errors` | 失败的模型请求数，重试请求的每次失败尝试都计一次。`n_llm + llm_errors` 是发出的请求总数。 |
| `usage` | 所有回复的 token 计数，按名称求和，例如 `prompt_tokens`、`completion_tokens` 和 `total_tokens`。在 `legacy` 之外，它还有 `reasoning_chars`（收到的推理，已拆出内联的 `<think>` 块）以及服务器报告时的 `cached_tokens` 或 `reasoning_tokens`。 |
| `duration_s` | 运行的挂钟时间，以秒计。 |
| `config` | 本次运行使用的选项（见下文）。 |
| `prefix_cache` | 启用固定前缀时：版本、观察到的规划 schema 变化，以及可选的、服务器报告的、在规划和反思回复上求和的缓存 token 数；schema 变化是预期内的失效，不是实测到的缓存未命中。 |
| `context_budget` | 已配置时，是每次尝试的规划/反思记账记录列表；关闭时不存在。见[上下文记账](../../reference/configuration.md#context-accounting)（英文）。 |
| `memory_stats` | 在 `legacy` 之外，是回放计数和笔记计数；`legacy` 行中没有此字段。 |

`config` 包含：

| 键 | 含义 |
|---|---|
| `max_steps`, `attempt_policy`, `steps_per_attempt`, `max_attempts`, `reflection_gate`, `tool_exposure`, `memory`, `timeout_s` | 运行选项。没有设置时间限制时，`timeout_s` 为 `null`。 |
| `builtins` | 已注册的内置工具，按名称排序。 |
| `memory_dir` | SQLite 文件所在的目录；记忆留在 RAM 中时为 `null`。 |
| `tools` | 所有已注册的工具，内置的和 MCP 的，按名称排序。 |
| `reasoning`, `replay_turns`, `note_writer` | 在 `legacy` 之外，是生效的推理模式、它的轮次上限（`null` 表示范围内的所有轮次）以及压缩模式下由谁写回复的笔记。它们不出现在 `legacy` 行中。 |
| `retriever`, `retriever_effective`, `recall_at` | 在 `legacy` 之外，是请求的检索器（取默认值时为 `auto`）、已启用触发点的映射（`{"start": "llm", "attempt": "llm", "step": "bm25"}`，已排序）以及排序后的生效触发点列表。 |
| `recall_k`, `recall_kinds`, `recall_placement`, `recall_min_score` | 在 `legacy` 之外，是任务内的命中上限、排序后的已启用类型、步骤中的放置位置，以及嵌入/长期条目的分数下限。 |
| `cross_task_ratio`, `cross_task_cap`, `cross_task_cold` | 在 `legacy` 之外，是摘要召回的门：跨任务摘要必须达到的同后端倍数、每次召回的条数上限（0–2）以及冷启动配额（0 或 1）。 |
| `admit_on_outcome`, `keep_decision_text` | 在 `legacy` 之外，是两个 R6-lite 开关：在运行结束时只准入带最终状态的最终摘要，以及在决定的哈希旁保留其文本。两者默认都是 `false`，且不出现在 `legacy` 行中。 |

`memory_stats` 按顺序包含：

| 键 | 含义 |
|---|---|
| `replayed_chars` | 规划和反思请求携带的推理字符数。 |
| `replay_skipped` | 推理超过每轮 40,000 字符上限而被略去的助手轮次，在发出的请求上求和。 |
| `replay_stripped` | 推理被适配上限的淘汰清空的助手轮次，在发出的请求上求和。 |
| `notes`, `notes_model`, `notes_heuristic` | 压缩模式下存储的笔记数，以及其中由模型和启发式分别写下的数量。`notes` 也计入反思。 |
| `note_chars` | 本次运行存储的、通过检查的笔记文本的字符数（重述并扩充早先笔记的内容不再计入）。 |
| `drafts`, `drafts_written` | 存储的草稿数，以及其中被一次写入覆盖的数量。 |
| `candidates` | 提交给候选项边界的文本：每个文档（一条笔记、一份草稿、一份摘要、一次反思、一段用户文本、一个召回块）以及每个再次通过该边界的候选项条目。 |
| `remember_model`, `remember_heuristic` | 由模型或启发式设置了 `remember` 标志的笔记数。 |
| `candidates_refused` | 边界或笔记存储拒绝的文本，计数方式与 `candidates` 相同。 |
| `redactions` | 边界替换的秘密片段数。 |
| `rehearsed` | `{"restated": n, "acted": n, "cited": n}`：每个排练信号触发的次数。 |
| `digests` | 存储的尝试摘要数。 |
| `recalls` | 本次运行中首次发送的召回块数。 |
| `recalls_by_trigger` | 按 `start`、`attempt`、`step` 和 `tool` 划分的同一计数。 |
| `recall_hits` | 这些块携带的命中数。 |
| `recall_same_task`, `recall_cross_task` | 交付的本任务和其他任务的长期存储摘要；没有作用域的长期条目两者都不计入。 |
| `recall_task_notes` | 交付的任务内笔记存储命中数（笔记、草稿、思考和任务内摘要）。 |
| `recall_chars` | 通过检查的块的字符数。 |
| `recall_ms` | 各次召回的挂钟时间之和。 |
| `recall_fallbacks` | 由检索器回退回答的召回数，`retrieve_errors` 是其背后的检索失败数。 |
| `recall_skipped` | 没有找到符合预算内容的召回数。 |
| `recall_refused` | 候选项边界拒绝的块数。 |
| `recall_empty` | 触发了却没有候选项或没有命中、因而没有构造块的召回数。 |
| `aux_retrieve_calls` | 发出的无上下文检索请求数（`llm` 检索器），已经计入 `n_aux`；`retrieve_prompt_tokens` 和 `retrieve_completion_tokens` 是它们的 token 数。 |
| `thought_chunks` | 存储的思考片段数（每个任务最多 6 个）。 |
| `recall_log` | 每个首次发送的块一条紧凑记录，不含条目文本：触发点、放置位置、检索器、查询、k、候选项数、命中数（id、别名、类型、分数、来源任务 id、字符数、已略去数）、同任务/跨任务/任务笔记计数、字符数、毫秒数、回退。 |
| `admitted` | `{"rehearsal": n, "user_request": n, "user_statement": n, "outcome": n, "sleep": n}`：每个信号写入的决定数。 |
| `written` | `{"semantic": n, "episodic": n, "profile": n, "reflection": n}`：按类型写入的条目数（`IGNORE` 不写条目）。 |
| `ignored` | 因策略选择 `IGNORE` 而没有写入条目的决定数。边界拒绝计入 `candidates_refused`，不计入这里。 |

在 `drop` 模式下，回放计数为 `0`；在不使用它们的每种模式中，笔记、召回和准入计数为 `0`，因此各显式模式下的结构相同。`legacy` 行没有 `memory_stats`，`usage` 中没有 `reasoning_chars`，也没有 `cached_tokens`/`reasoning_tokens`，`config` 中没有 `reasoning`/`replay_turns`/`note_writer` 或 `retriever`/`recall_*`，因此与加入这些字段之前的行逐字节相同。

对话请求超时是模型连接的设置，而不是运行的设置，因此不属于 `config`。

<a id="provider-errors-and-retries"></a>

## 提供方错误与重试

当对模型服务器的请求失败时，localmelo 只在第二次尝试可能成功时才重试：

| 失败 | 是否重试 |
|---|---|
| 超时、连接被拒绝或其他网络错误 | 是 |
| HTTP 429、502、503 或 504 | 是 |
| 其他任何 HTTP 错误状态，例如 400、404 或 500 | 否 |
| 回复不是 JSON，或缺少预期的字段 | 否 |

- 对话请求最多重试两次：1 秒后一次，再过 4 秒又一次。如果最后一次尝试也失败，运行以 `provider_error` 结束。
- 嵌入请求不重试。
- 错误会指出请求，并在服务器有回答时给出 HTTP 状态和回复体的前 500 个字符，例如 `POST /chat/completions failed with HTTP 500: <html> …`。
- 每次尝试都等待完整的[对话请求超时](../../reference/configuration.md#chat-request-timeout)（英文）。因此，一个什么都不发送的服务器会在约 3 × 超时 + 5 秒后结束运行：使用默认的 120 秒时约为 365 秒。
- 每个对话请求都在 `X-Localmelo-Attempt` HTTP 头中携带它的尝试序号（1、2 或 3），因此可以把代理的请求日志与 `n_llm` 和 `llm_errors` 对应起来。`X-Localmelo-Call-Kind` 对规划请求是 `plan`，对反思请求是 `reflect`。

<a id="where-results-appear"></a>

## 结果出现在哪里

- **命令行：** 结果和状态行，或者 JSON 行。见[输出](../../reference/cli.md#output)（英文）。
- **网关：** `POST /v1/agent/run` 以 HTTP 200 和 `result` 作答：答案或尽力结果，在它为空时是 `[<status>] <error>`。其他情况不报告状态。见 [HTTP API](http-v1.md#http-api)。
- **Python：** `Agent.run_task()` 返回一个 `RunResult`，`run.finished` 事件也携带它。见 [Python API](python-api.md#results)和[事件](../../reference/events.md#runfinished)（英文）。
- **历史记录：** 每次运行的任务记录都连同其状态一起保存。见[任务记录](../../reference/memory.md#task-records)（英文）。
