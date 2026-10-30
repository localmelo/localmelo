<a id="configuration"></a>

# 配置

本页说明 localmelo 如何连接到你的模型服务器：首次运行、设置向导、配置文件、受支持的后端，以及你可以更改的设置。

<a id="the-config-file"></a>

## 配置文件

配置位于 `~/.cache/localmelo/config.toml`。一个不带嵌入、供 Ollama 使用的最小文件：

```toml
chat_backend = "ollama"
embedding_backend = "none"

[ollama]
chat_url = "http://127.0.0.1:11434"
chat_model = "qwen3:8b"
```

> [!WARNING]
> 清理缓存的工具可能删除 `~/.cache`，那里也保存着你的记忆。请为你手工编辑过的配置文件保留一份副本。

向导会为每个后端写一张表，但 localmelo 只读取所选后端的表。

| 键 | 表 | 默认值 | 说明 |
|---|---|---|---|
| `chat_backend` | 顶层 | 未设置 | 对话后端；见[后端](../guides/model-servers.md#backends)。对于 oMLX，使用 `omlx`（服务器自称 oMLX 时，首次运行的设置会写入它）或 `vllm`：两者都按 `vllm` 加载，并按原样保存。必需。 |
| `embedding_backend` | 顶层 | 未设置 | `mlc`、`ollama`、`vllm`、`sglang` 或 `none`；`omlx` 会被当作 `vllm`。必需。 |
| `request_timeout_s` | 顶层 | 未设置 | 对话请求超时，以秒计；见[对话请求超时](#chat-request-timeout)。 |
| `chat_url` | 本地后端 | 未设置 | 模型服务器的 URL。 |
| `chat_model` | 本地或云端后端 | 未设置 | 要使用的模型。 |
| `embedding_url` | 本地后端 | 未设置 | 嵌入服务器的 URL。 |
| `embedding_model` | 本地后端 | 未设置 | 要使用的嵌入模型。 |
| `extra_body` | 本地或云端后端 | 未设置 | 每次对话请求的额外字段；见[额外请求字段](#extra-request-fields-extra_body)。 |
| `reasoning_field` | 本地或云端后端 | 未设置 | 推理随之发回的请求键：`auto`、`reasoning_content` 或 `reasoning`；`[ollama]` 改用 `auto` 或 `thinking`。见[推理](#reasoning)。 |
| `requires_reasoning_replay` | 本地或云端后端 | `false` | 当服务器拒绝省略了回复推理的请求时为 `true`。见[推理](#reasoning)。 |
| `context_window`、`output_reserve`、`safety_margin` | `[agent]` | 未设置 | 可选的上下文请求记账；三个正整数必须都能确定，且输入上限为正。见[上下文记账](#context-accounting)。 |
| `reasoning` | `[agent]` | `"legacy"` | `legacy`、`drop`、`replay`、`compress` 或 `compress+replay`；直接运行、交互式界面和 API v2 控制台会话保存的默认值。见[推理](#reasoning)。 |
| `replay_turns` | `[agent]` | 未设置 | 使用 `replay` 或 `compress+replay` 时，发送多少条最近助手轮次的推理；省略则发送范围内的每一轮。 |
| `note_writer` | `[agent]` | `"auto"` | 压缩模式下，`auto`（模型的 `<note>`，否则用启发式）或 `heuristic`。 |
| `tool_env` | `[agent]` | `[]` | 配置支持的直接运行和会话中，shell/Python 工具额外继承的环境变量名称。值从不保存；见[工具环境](#tool-environment)。 |
| `fixed_prefix` | `[agent]` | `true` | `false` 会发送稳定前缀成为默认值之前的请求。见[上下文统计](#context-accounting)。 |
| `window_eviction` | `[agent]` | `"chunk"` | `chunk`、`boundary` 或 `slide`：工作记忆何时丢弃旧轮次。见 [Memory](../../reference/memory.md#working-memory-window)。 |
| `brief_items` | `[agent]` | `false` | 实验性。`true` 让 `--brief` 运行指出简报中尚未被任何编辑触及的编号条目；见 [`--brief-items`](cli.md#brief-items)。 |
| `reflect_tool_guard` | `[agent]` | `true` | `false` 发送不带工具和结尾句子的反思请求。见[上下文统计](#context-accounting)。 |
| `confine_writes` | `[agent]` | `true` | 在 macOS 上，`--brief` worker 运行和保存的任务会在沙箱中启动 `shell_exec` 和 `python_exec`，拒绝工作区之外的文件写入；`false` 关闭它（`-X agent.confine_writes=false` 为一次运行关闭）。其他运行从不受此限制。见[写入留在工作区内](cli.md#confined-writes)。 |
| `confine_direct_runs` | `[agent]` | `false` | 开发者设置。`true` 让普通直接运行也受限制，就像 `confine_writes` 限制 `--brief` 运行那样：在 macOS 上，命令行给出的任务和 `melo ""` 循环会在沙箱中启动 `shell_exec` 和 `python_exec`，拒绝工作区（`--workspace`，否则为当前文件夹）之外的文件写入，结果中带有 `confinement`。没有对应的标志，`-X agent.confine_direct_runs=true` 为一次运行设置它。`--brief` 运行遵循 `confine_writes`；交互式界面、控制台、应用和网关会忽略它。见[写入留在工作区内](cli.md#confined-writes)。 |
| `lean_output` | `[agent]` | `false` | 实验性。`true` 让命令行给出的任务少输出 token；没有对应的标志，`-X agent.lean_output=true` 为一次运行设置它。交互式界面和网关会忽略它。见[上下文统计](#context-accounting)。 |
| `cache_layout` | `[agent]` | `"v3"` | `"v2"` 会发送此前的请求，在每个规划请求中把反思重建为最后一条消息；`"v3"` 与 `fixed_prefix` 一起使用时，只追加一次反思。见[上下文统计](#context-accounting)。 |
| `window_messages` | `[agent]` | 本地服务器或 URL 上为 `64`，云厂商上为 `20` | 工作窗口的消息数上限，8 到 256；`20` 会发送此前的上限。见[上下文统计](#context-accounting)。 |
| `retriever` | `[retrieval]` | `"auto"` | `auto`、`off`、`bm25`、`llm` 或 `embedding`：召回如何为文档排序。见[循环中的召回](../concepts/memory.md#recall-in-the-loop)。 |
| `recall_at` | `[retrieval]` | 未设置 | 一个由 `start`、`attempt`、`step` 和 `tool` 组成的列表。省略则为 `start`、`attempt` 和 `step`；除非点名，`tool` 从不开启。 |
| `k` | `[retrieval]` | `3` | 每次召回的任务内命中数，从 1 到 10。 |
| `kinds` | `[retrieval]` | 未设置 | 一个由 `note`、`draft`、`thought`、`digest`、`reflection` 和 `long` 组成的列表。省略则为除 `thought` 之外的所有类型。 |
| `placement` | `[retrieval]` | `"tool"` | `tool`（附加到最后一个工具结果）或 `message`（单独的用户消息）。`replay` 和 `compress+replay` 中拒绝 `message`。 |
| `min_score` | `[retrieval]` | `0.35` | 嵌入和长期余弦命中的余弦下限，从 0 到 1。对任务内的 `bm25` 和 `llm` 排序，以及只有 BM25 打分的长期条目，均不起作用。 |
| `cross_task_ratio` | `[retrieval]` | `1.1` | 跨任务情节摘要需要达到同一长期存储后端中最佳同任务摘要分数的这个倍数；1–10。 |
| `cross_task_cap` | `[retrieval]` | `1` | 每次召回的跨任务摘要上限，0–2；既有的长期命中总数上限仍为两条。 |
| `cross_task_cold` | `[retrieval]` | `0` | 没有匹配的同任务摘要时，允许 0 或 1 条跨任务摘要；仍受上限和后端下限约束。一般记忆绕过这道门。 |
| `admit_on_outcome` | `[memory]` | `false` | R6-lite：只在运行结束时准入最后一次尝试的摘要，并在其来源中带上最终状态。`--admit-on-outcome` 旗标对一次运行优先；直接模式会读这张表，交互式界面和网关不读，且在 `legacy` 中被接受但不生效。 |
| `api_key_env` | 云端后端 | 未设置 | 保存 API 密钥的环境变量名称。 |
| `base_url` | 云端后端 | 厂商的 URL | API 的 URL。 |
| `host` | `[gateway]` | `"127.0.0.1"` | 回环监听地址；拒绝非回环绑定。 |
| `port` | `[gateway]` | `8401` | 网关监听的端口。 |
| `workspace` | `[gateway]` | `""` | 网页控制台会话的默认工作区文件夹（为空时为 `~/localmelo-workspace`）；仅在设置时写入。 |
| `stream` | `[cli]` | `true` | 仅在交互式界面使用的实时流式，除非关闭否则默认开启（`/stream off`、`-X cli.stream=false`、`melo config set cli.stream false`）；没有该界面的运行从不流式。`/new` 会把 `/stream` 恢复为此值。无效值在界面中警告并回退；直接运行和网关不会因此失败。见[流式回复](../guides/terminal.md#streamed-replies)。 |
| `status_line` | `[cli]` | `"auto"` | `"auto"`：交互式状态栏的资源区根据已收到的事件显示 `cache`、`ctx` 和 `tasks` 数字，没有数据的隐藏。`"off"`：状态栏保留各区但不显示这些数字（模型、工作区、运行区：标题或运行中轮次的步骤，以及上一次请求的 `ctx`）。高级选项：没有命令行参数。其他值在界面中警告并回退；直接运行和网关不会因此失败。见[状态栏](../guides/terminal.md#the-status-bar)。 |
| `colors` | `[cli]` | `"auto"` | 交互式界面的颜色数量：`"auto"`（根据 `COLORTERM` 和 `TERM`）、`"16"`、`"256"` 或 `"truecolor"`。`NO_COLOR` 仍会关闭颜色。开发者设置：`-X cli.colors=256` 只用于本次运行。其他值会警告并回退。见[终端颜色](../guides/terminal.md#terminal-colours)。 |
| `background` | `[cli]` | `"auto"` | 交互式界面所用的终端背景：`"auto"`（先询问终端，再看 `COLORFGBG`），什么都不回答的终端（tmux、screen、ssh）可设为 `"dark"` 或 `"light"`。开发者设置：`-X cli.background=light`。见[终端颜色](../guides/terminal.md#terminal-colours)。 |
| `reasoning_tags` | `[cli]` | 未设置 | 一张 `MODEL = "explicit" \| "closing_only" \| "none"` 表：某模型回复的可信分帧声明；`stream = false` 时也会用它切分完整回复。没有条目的模型使用保守规则（内容一直保留到回复结束）。目前需要手工设置，引导流程还不会询问它。 |
| `mode` | `[ui]` | `"default"` | `"default"`：`melo --help` 只显示用户选项。`"developer"`：显示全部选项，按用户、高级、实验分组；`melo --developer --help` 对单次运行有同样效果。模式只改变显示的内容（帮助文字，以及 Web 控制台的开发者视图，其设置中的开关会保存到这里），不改变任何选项的行为。其他取值会在 stderr 警告并显示默认帮助；运行和网关不会因此失败。见[默认模式与开发者模式](../../reference/cli.md#default-and-developer-mode)（英文）。 |
| `resume_mode` | `[sessions]` | `"pairs"` | 终端对话恢复：`pairs` 加载最近的用户/答案对；`window` 加载保存的工作窗口加上之后的问答对，没有可用的检查点时回退。无效值只在界面中警告或回退；直接运行不读取此设置。 |
| `record_direct` | `[sessions]` | `true` | 把每次直接运行（`melo "<task>"`、`melo --brief FILE`）记录为 `~/.cache/localmelo/cli/sessions.db` 中的 worker 会话，网页控制台的 Workers 视图和 macOS 应用会读取它。只有 `false` 会关闭；`--no-save-session` 对单次运行起同样作用，`--save-session` 或 `--parent-run` 仍会记录。只在为 `false` 时写入。见 [Worker 历史记录](cli.md#worker-历史记录)。 |
| `policy` | `[scheduler]` | `"off"` | 网页工作区的调用如何到达模型服务器：`off` 只记录，`fcfs` 先到先得，`class` 让 P1 先于 P2 得到准入（依此类推）；见[调度器](../guides/scheduler.md#the-gate-on-chats-s2)。 |
| `driver` | `[browser]` | `"off"` | melo 是否可以在浏览器中读取网页：`off` 不提供浏览工具；`safari-automation` 为终端对话和网页控制台提供八个只读工具，在 Safari 的自动化窗口中打开网页（每个网站每次对话询问一次；见[用 melo 浏览网页](../guides/browsing.md)）。终端中用 `/browser` 切换；`-X browser.driver=safari-automation` 仅用于一次运行。直接运行、worker 运行和已保存的任务从不获得这些工具。其他任何值都是配置错误。 |
| `n_srv` | `[scheduler]` | `8` | 模型服务器的槽位数。 |
| `r_f` | `[scheduler]` | `1` | 为命令行和直接运行保留的槽位数；`n_srv <= 2` 时强制为 0。 |
| `n_fg` | `[scheduler]` | `1` | 有聊天处于焦点时允许在途的后台调用数。 |
| `call_log` | `[scheduler]` | `true` | 每次网关运行写一份 JSONL 调用日志。 |
| `call_log_dir` | `[scheduler]` | `""` | 日志的父目录；默认为 `~/.cache/localmelo/scheduler`。 |
| `gpu_lock_file` | `[scheduler]` | `""` | 要读取的共享 GPU 锁；为空表示不读取锁。网关中的电池读取器始终开启。 |
| `lock_self` | `[scheduler]` | `""` | 这个 bench 自己的持有者：确切的 `holder:` 值，或锁带有 `run:` 行时的 `holder:run`。匹配是精确且区分大小写的。 |
| `lock_gate` | `[scheduler]` | `"record"` | `record` 从不改变请求；`wait` 在存在外部锁时扣住每一次工作区调用，包括嵌入。见[调度器](../guides/scheduler.md#the-gpu-lock-and-the-battery-s2b)。 |

全是默认值的 `[memory]` 表不会被写入。它唯一的键 `admit_on_outcome` 必须是布尔值；其他任何值都是配置错误（退出码 1）。它在 `legacy` 中被接受但不生效。`--keep-decision-text` 开关只能作为旗标使用，因为它会把可能涉及个人内容的文本写入存储，而保存为默认值会带来隐私隐患。

全是默认值的 `[scheduler]` 表不会被写入，未知的调度器键会被丢弃，因此没有这张表的配置文件保持其确切字节；向导会保留非默认的策略。

顶层键必须位于第一个 `[table]` 行之前。URL 的用法如下：

- `mlc`、`vllm`（oMLX 也用它）和 `sglang`：`chat_url` 和 `embedding_url` 缺少 `/v1` 时会补上。
- `ollama`：对话使用 Ollama 自己的 `/api/chat`，因此 `chat_url` 不带 `/v1` 使用（结尾的 `/v1` 会被去掉）。`embedding_url` 会补上 `/v1`。
- 云端后端：`base_url` 按原样使用。

<a id="reasoning"></a>

### 推理

`legacy` 是默认值：OpenAI 兼容后端不发送推理，Ollama 原生后端像以前一样发送它自己的 `thinking` 字段和内联副本。`drop` 从不发送推理；`replay` 在范围内的助手轮次上发送推理，使用一个键，每轮最多 40,000 个字符（更长的轮次推理会被跳过并计数）。`compress` 不发送推理，并把每条回复的笔记和草稿保留在工作记忆中；`compress+replay` 两者都做。`requires_reasoning_replay = true` 的后端把 `compress` 当作 `compress+replay` 运行。每种显式模式还会停止读取 `[RECALL]`、从答案中移除内联的 `<think>` 块，并停止把原始步骤摘要写入长期记忆。压缩模式默认召回笔记、草稿和早先的思考；见[循环中的召回](../concepts/memory.md#recall-in-the-loop)。

```toml
[agent]
reasoning = "replay"      # legacy, drop, replay, compress, compress+replay
replay_turns = 3          # optional; omit for every turn
note_writer = "auto"      # optional: auto or heuristic
```

`[retrieval]` 为直接运行、交互式界面和控制台会话设置召回的默认值；显式的命令行旗标优先于对应的键。这张表在 `legacy` 中被接受但不生效。超出范围的值（未知的检索器或名称、`k` 超出 1-10、`min_score` 超出 0-1）是配置错误，以一行输出和退出码 1 结束。

```toml
[retrieval]
retriever = "auto"        # auto, off, bm25, llm or embedding
recall_at = ["start", "attempt", "step"]
k = 3                     # 1-10
kinds = ["note", "draft"] # omit for all but thought
placement = "tool"        # tool or message
min_score = 0.35          # 0-1
cross_task_ratio = 1.1    # 1-10
cross_task_cap = 1        # 0-2
cross_task_cold = 0       # 0 or 1
```

`save` 只在某个值与其默认值不同时才写入这张表，因此现有的配置文件保持不变。

在直接模式和交互式界面中，显式旗标优先于 `[agent]`。`--base-url` 会话或运行使用旗标，而不使用这些保存的默认值。API v2 控制台会话使用保存的模式，未设置时默认为 `legacy`；它们保留网关的工具策略。旧版 `/v1` API 不变，也不继承推理/检索两张表。`--retrieval-extra-body` 只能作为旗标使用，从不是保存的键，且只影响 `retrieve` 请求。

每个后端小节可以命名请求键，并说明服务器要求回放：

```toml
[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "MODEL"
reasoning_field = "auto"              # or reasoning_content, reasoning
requires_reasoning_replay = false     # true for a server that rejects a
                                      # request without the reasoning
```

`auto` 跟随上一条回复的推理来自哪个键，因此发出 `reasoning_content` 或 `reasoning` 的后端可以原样工作；在第一条回复之前它不发送任何键。对 OpenAI 兼容后端，该字段只用于 `replay` 和 `compress+replay`：`legacy`、`drop` 和 `compress` 会忽略它，通过调度器包装器时也一样。Ollama 原生保留其既有的 legacy 线上行为。`replay` 在直接运行和 `melo ""` 循环中使用当前任务的助手轮次；会话构建器改为选择保留的对话窗口。会话恢复使用当前进程的模式：问答对不含旧思考，而窗口检查点可以回放它保留的、已脱敏的思考。`[ollama]` 后端把它的字段固定为 `thinking`（它仍然可以写 `"thinking"` 或 `"auto"`）。在 `requires_reasoning_replay = true` 时，`--reasoning drop` 会被拒绝，窗口中的每个助手轮次都会回放（任务范围和 `replay_turns` 不适用），并且智能体会整块淘汰工作记忆单元，而不是截断推理。

<a id="tool-environment"></a>

### 工具环境

Shell 和 Python 工具通常只继承 `HOME`、`LOGNAME`、`PATH`、`SHELL`、`TERM`、`USER`、`LANG`、`TMPDIR` 以及所有 `LC_*` 变量。要从 melo 的进程中显式暴露额外变量：

```toml
[agent]
tool_env = ["SSH_AUTH_SOCK", "HTTPS_PROXY"]
```

这是一个**名称**列表，而不是 `NAME=value` 赋值。名称必须匹配 `[A-Za-z_][A-Za-z0-9_]*`；缺失的变量会被跳过。基于配置的直接运行、交互式界面和控制台会话在启动时解析存在的值。只有名称会写入 TOML；选项值不会出现在运行的配置记录中。`--base-url` 运行或会话不读取此设置。Python 调用方可以改为传入显式的 `AgentOptions(tool_env={...})` 映射。这不会改变 MCP 各自独立的按服务器 `env` 配置。

请把每个名称视为一项能力授权。`SSH_AUTH_SOCK` 让工具子进程使用你的 SSH 代理进行认证或签名，代理变量则可能暴露它的地址和凭据。工具以你的权限运行，可以打印、复制或传输它们收到的变量；过滤环境不是沙箱。除非某个工具需要这项能力，否则请让列表保持为空，并且绝不要把密钥值放进 TOML。

<a id="context-accounting"></a>

## 上下文记账

把 `[agent] context_window`、`output_reserve` 和 `safety_margin` 设为正整数；窗口必须大于它们之和。直接运行和交互式界面的旗标会覆盖单个值；API v2 控制台会话使用配置。旧版 `/v1` 不变。未设置时不记录任何记账。

对每次规划或反思尝试，`utf8-bytes-v2` 以 UTF-8 字节数除以三向上取整来估算组装后的消息和工具 schema 表示。尺寸已知时，每张图片增加 `ceil(width / 28) * ceil(height / 28)` 个 token，上限为 16,384；尺寸未知时使用该上限。同一模型和请求类型的最近一次正数的提示 token 计数会向上校准这一估算；缺失或为零的用量会重置该校准。估算是估计值，不是界限：模板和视觉模型各不相同。

当估算超过输入上限时，智能体会适配一份请求副本：它把可裁剪的旧工具结果变成预览，移除可选的旧思考，然后淘汰旧的完整单元，同时保留任务、工具 schema、匹配的调用/结果对、图片和失败证据。它不改历史记录和保留的对话。如果固定的上下文无法装下，运行以 `context_budget_exceeded` 和一条修复提示结束。被归类为上下文溢出的回复最多获得一次更小的请求重试，且不重新运行工具；其他提供方错误保持其既有的重试策略。事件和 JSON 结果报告估算值、可用时的实际提示计数，以及裁剪计数，但不保存提示文本。输出预留不是提供方的生成长度上限；独立的字符限制仍然适用。

`[agent] fixed_prefix`（默认 `true`）让直接运行、交互式界面和 API v2 控制台运行采用 `fixed-prefix-v2`；`--fixed-prefix on|off` 为一次直接运行或交互式运行覆盖它，`fixed_prefix = false`（或 `--fixed-prefix off`）会发送它成为默认值之前的请求。它在每次运行开始时（包括跨越午夜时）快照本地日期和 UTC 偏移，并把会变化的召回和反思放进带标签的用户侧数据：召回在任务之前；使用 `cache_layout = "v2"` 时，最新的反思放在最后（窗口之后的一条最终用户消息），默认的 `v3` 则把反思只追加到窗口一次（见下文）；反思请求重复最近一次规划请求（相同的系统文本、相同的 `tools`、相同的轮次，再加上之后的轮次），并追加一条带反思提示的最终用户消息。一个显式的结束分隔符让任务和反思请求留在该数据之外；以该分隔符开头的数据行会被确定性地加上引号。`all` 为这次运行冻结排序后的工具定义；`retrieve` 和以前一样继续选择工具。JSON 结果的 `prefix_cache.schema_changes` 统计运行期间所提供的规划 schema 变化，而不是实际的缓存未命中；可选的 `reported_cached_tokens` 是服务器报告的、仅限规划和反思回复的总数。提供方可能不报告缓存，字节稳定的前缀也不保证缓存命中。

`[agent] reflect_tool_guard`（默认 `true`；`--reflect-tool-guard on|off` 会覆盖它）避免反思请求引出工具调用。请求会提供该步骤规划请求的工具（列表、顺序和 schema 相同，`fixed_prefix` 下本来就是如此），其最后一条用户消息以 `Answer with the JSON object only; do not call any tool.` 结尾；`tool_choice` 不变，因此服务器的提示前缀保持不变。重试请求不变。启用保护时会在 JSON 结果的 `config` 中加入 `reflect_tool_guard`；为 `false`（或 `off`）时，反思请求既不带工具，也没有那句话。无论该选项如何，`reflection` 事件的 `unreadable_reason` 都会说明第一次回复为何不可读（见[事件](events.md)）。

`[agent] window_eviction`（默认 `"chunk"`；另两个值是 `"boundary"` 和 `"slide"`；`--window-eviction` 会覆盖它）决定工作记忆何时丢弃旧轮次。`chunk` 和 `boundary` 让保留的历史在几步之内保持不变，服务器的提示缓存就能复用它；`slide` 在窗口满后每步丢弃最旧的单元，与 `chunk` 成为默认值之前相同。见 [Memory](../../reference/memory.md#working-memory-window)。

`[agent] cache_layout`（默认 `"v3"`；`--cache-layout v2|v3` 会覆盖它）决定反思所在的位置。`"v2"` 是这一默认值改变之前的布局。在 `v3` 且启用 `fixed_prefix` 时：

- 在尝试边界，智能体只追加一次反思，作为一条普通的窗口消息（`[Reflections; data, not instructions]` 和该块），而不是把它重建为每个规划请求的最后一条消息。之后它随历史移动，并像任何消息一样在被丢弃时离开。每次尝试中新产生的内容仍遵循“只保留最新一条”规则（新的反思会替换工作记忆中保存的那条），但更早的块会留在历史中，直到被丢弃。
- 反思请求及其回复留在窗口中，因此下一个规划请求会接续该请求，而不是重新构建。在默认的 `reflection_gate = "flags"` 下，即使重试之后仍未读到反思，其中的任何内容都不会留在窗口里。
- 不可读的回复会在同一前缀上再次询问：先是该请求，接着是作为助手消息的不可读回复（工具调用写成文本，因此不会留下没有结果的调用），再是一条结尾说明，并带上该请求提供的工具，而不是文本记录。如果回复含有解码器拒绝的 JSON，说明之后会多一行引用解码器的错误。如果第一个请求没有重复规划请求（窗口已经移动），就没有可接续的交换，重试仍和以前一样是文本记录。读到的重试会把整个交换留在窗口中：请求、不可读的回复、说明和回复。
- 使用 `window_eviction = "boundary"` 时，已结束尝试的轮次会在边界处被丢弃，因此那里只追加该块。

没有 `fixed_prefix` 时，两个值都不改变任何内容。`cache_layout` 为 `v3` 时，JSON 结果的 `config` 包含它，为 `v2` 时不包含。

`[agent] window_messages`（在本地服务器（Ollama、MLC、vLLM 或 oMLX、SGLang）和 `--base-url` 服务器上默认为 `64`，在云厂商的后端（OpenAI、Gemini、Anthropic 或 NVIDIA）上默认为 `20`；8 到 256；`--window-messages N` 会覆盖它）是工作窗口在 `window_eviction` 丢弃一些消息之前最多保存的消息数。云厂商的后端暂时保持 20：没有提示缓存时，更长的窗口会让每个请求多消耗输入 token。你设置的值在任何后端上都保持不变。上限较大时，由提示大小拟合来丢弃：超过 100,000 个字符的上限后，它整单元地丢弃（工具调用及其结果，绝不拆开），在 `chunk` 和 `boundary` 下降到上限的 70%，因此历史可以在几个步骤内保持不变；在 `slide` 下它每一步都丢弃一个单元。按计数丢弃一次最多拿走窗口的一半，因此低于 16 的上限每次丢弃的消息少于八条。见 [Memory](../../reference/memory.md#working-memory-window)。当 `window_messages` 不是 20 时，JSON 结果的 `config` 包含它。

`cache_layout = "v3"` 和 64 条消息的窗口，是在两轮用 Qwen 模型在 oMLX 上对编码任务做的基准测试之后成为默认值的：48 次运行中通过 46 次，此前的默认值为 44 次；86.8% 的提示 token 由缓存提供，此前为 78.9%。`--cache-layout v2 --window-messages 20` 会发送此前的请求。`melo --developer --help` 把这两个标志列在“高级”下。

`fixed_prefix`、`window_eviction` 和 `reflect_tool_guard` 是在同一版本上，用本地模型对基准测试的编码任务做 A/B 比较后成为默认值的。你在 `[agent]` 中设置的键保持原样，这三个和上面两个都一样；没有该键的文件使用默认值，保存时只有与默认值不同的键才会写入（`window_messages` 只要设置了就会写入）。`melo --developer --help` 把这三个标志列在“实验性”下，`--fixed-prefix off --window-eviction slide --reflect-tool-guard off` 会发送此前默认值的请求。API v2 控制台会话、控制台工作进程和旧版 `/v1` 会话会从配置读取这五个键；旧版 `/v1` API 不读取其他 `[agent]` 键或 `[retrieval]`。

用 `[agent] captured_history = true` 或 `--captured-history on` 选择把捕获的工具历史单独保存；`off` 覆盖该设置。经过检查的预览使用 50,000 个字符的上限，包括截断和图片提示。历史记录为每个结果保存一份独立的、按模式脱敏的副本，最多 200,000 个字符（错误优先），并给出相对于脱敏后捕获的精确略去计数。计数保存在 `capture_metadata` 中，而不是已存文本里的带内标记。`acquisition_omitted_chars` 为 null，因为工具可能已经限制了自己的输出。捕获不是完整的文件或无限制的标准输出；脱敏基于模式，可能漏掉密钥。模型看到的经过检查的预览仍可能包含原始文本；`Message.tool_result`、`tool.finished` 事件和 trace 使用脱敏后的副本。既有的历史记录保留和权限仍然适用。`Agent.run_tool()` 不写入捕获的历史，并保留其现有的结果检查。该选项默认关闭。

用 `[agent] lean_output = true` 或 `-X agent.lean_output=true` 选择让直接运行少输出 token（实验性；没有对应的标志）。命令行给出的任务（`melo "<task>"` 或 `melo worker BRIEF`，输出为 `json` 或 `stream-json` 时也一样）随后发送的系统提示，会用下面几条取代 `Think step by step. Be concise.`：工具调用之间不写文字，最终答案至多三行，使用相对路径而不设置 `sys.path`，并在列目录之前先读任务点名的文件。在默认的 `reflection_gate = "flags"` 下，反思提示还会去掉五个估计值、`progress_made` 和 `new_information_gained`（只有 `utility` 会读估计值，另外两项没有任何代码读取），并且只在 `done` 或 `stop` 时要求给出 `best_effort_result`；继续判定读取的动作和三个标志仍然会被要求给出，旧的长格式回复也照常读取。`--reflection-gate utility` 保留完整的反思提示，压缩模式保留系统提示的最后一行，因为它们的笔记就是工具调用之间的文字。交互式界面、`melo ""` 循环和网关会忽略该键。关闭（默认）时，每个请求都与该选项出现之前发送的请求相同；开启时，一次运行中各轮请求的开头保持不变。

<a id="embedding-and-long-term-memory"></a>

## 嵌入与长期记忆

长期记忆需要嵌入后端；见[设置嵌入模型](../guides/embedding.md)。使用 `embedding_backend = "none"` 时：

- 不会从长期记忆召回任何内容，也不会向它写入任何内容；
- 反思不会提升到长期记忆；
- 运行循环、工具、工作记忆和历史记录照常工作。

此时网关上的 `GET /v1/health` 显示 `"embedding_backend": "none"`。见[记忆](../../reference/memory.md)（英文）。

<a id="cloud-api-keys"></a>

## 云端 API 密钥

- 配置文件只保存环境变量的名称（`api_key_env`），从不保存密钥本身。
- 在运行 `melo` 的 shell 中导出该变量。
- 如果它没有设置，`melo` 会在运行之前停止，退出码为 1：

  ```text
    Configuration errors:
    - [openai] environment variable $OPENAI_API_KEY is not set. Export it before starting the gateway.
  ```

- `--base-url` 不发送 API 密钥，因此云端 API 请使用配置文件。
- macOS 后台服务拿不到你 shell 的变量；见[登录时运行网关](../guides/gateway.md#run-the-gateway-at-login-macos)。

<a id="extra-request-fields-extra_body"></a>

## 额外请求字段（extra_body）

每个后端表都接受一个 `extra_body` 表。它的字段会加入每次对话请求。用它传递服务器选项，例如 `chat_template_kwargs`：

```toml
[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen/Qwen3-8B"

[vllm.extra_body.chat_template_kwargs]
enable_thinking = false
```

- 内联写法也可以：`extra_body = { chat_template_kwargs = { enable_thinking = false } }`。
- 表中的扁平 `chat_template_kwargs = { … }` 键会被读作 `extra_body` 的一部分；向导会把它写回 `extra_body` 之内。
- localmelo 自己设置 `model`、`messages`、`tools`、`tool_choice` 和 `stream`。在 `extra_body` 中设置其中之一是配置错误：

  ```text
    - [vllm] extra_body must not set 'model'; localmelo manages that field
  ```

- 对 `ollama` 后端，这些字段进入 Ollama `/api/chat` 的请求体。localmelo 默认向 Ollama 请求模型的思考；`extra_body = { think = false }` 会关闭它。
- `--base-url` 不读取配置文件，因此 `extra_body` 在那里不适用。

驱动为 `off` 的 `[browser]` 表不会写入。

<a id="mcp-servers"></a>

## MCP 服务器

`config.toml` 顶部的 `mcp_config = "~/mcp.json"` 指定交互式界面加载的 MCP 服务器 JSON 文件（网关将读取同一个键）；`--mcp-config FILE` 为一次运行覆盖它。直接模式从不读取它。

<a id="chat-request-timeout"></a>

## 对话请求超时

对话请求最多等待服务器 120 秒（使用 `ollama` 后端时为 300 秒）。要为每次运行更改这一点，请把 `request_timeout_s` 设在配置文件顶部、第一个表之前：

```toml
chat_backend = "vllm"
embedding_backend = "none"
request_timeout_s = 600

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen/Qwen3-8B"
```

- 在 `[table]` 行之下，该键属于那张表，localmelo 不会在那里读取它。
- 该值是一个大于 0 的秒数，例如 `600` 或 `90.5`。其他任何值都会让 `melo` 以退出码 1 停止，例如带引号的值会得到 `request_timeout_s must be a finite number of seconds > 0, got '600'`。
- `--request-timeout SECONDS` 为一次直接运行覆盖它，且不更改文件。
- 在第一个请求到达时才加载模型的服务器（如 oMLX），处理这个请求可能需要超过 120 秒；上面的 `600` 为它留出了余地。
- 它只适用于对话请求。嵌入请求始终最多等待 120 秒。
- 每当服务器发送数据，计时器都会重置，因此生成过程中发送保活字节的服务器永远不会超时。
- 超时的请求会重试两次，每次都使用完整的超时时间；见[提供方错误与重试](run-results.md#provider-errors-and-retries)。
- 要限制整个运行，请使用 `--timeout`。见[超时](../concepts/how-it-works.md#timeouts)。

<a id="environment-variables"></a>

## 环境变量

| 变量 | 作用 |
|---|---|
| `LOCALMELO_PERSIST_MEMORY` | 任何非空值，甚至 `0`，都会把历史记录和长期记忆保存到 SQLite 文件中。 |
| `LOCALMELO_MEMORY_DIR` | 这些文件的目录（默认 `~/.cache/localmelo/memory`）。它按原样使用：不会展开 `~`。 |
| `LOCALMELO_DETECT_URLS` | [首次运行](../guides/configure.md#first-run)请求模型列表的 URL，逗号分隔，用来代替 `127.0.0.1` 的 8000、8400、11434 和 30000 端口。空值会关闭搜索。 |
| `LOCALMELO_THINKING_RETENTION_DAYS` | `history.db` 和会话存储的 `sessions.db` 保留模型推理的天数（默认 30）：任何大于 0 的数，例如 `7` 或 `0.5`。其他任何值都会让 `melo` 以退出码 2 停止。见[模型调用](../../reference/memory.md#model-calls)（英文）。 |
| `api_key_env` 指定的变量 | 云端后端的 API 密钥，例如 `OPENAI_API_KEY`。 |

见[在运行之间保留记忆](../guides/memory.md#keep-memory-between-runs)。

<a id="files-and-directories"></a>

## 文件与目录

| 路径 | 内容 | 创建时机 |
|---|---|---|
| `~/.cache/localmelo/config.toml` | 配置。 | 由设置向导创建。 |
| `~/.cache/localmelo/memory/` | `history.db`、`procedures.db`（程序性记忆，每次运行结束时写入），以及使用嵌入后端时的 `long_term.db`。 | 智能体在设置了 `LOCALMELO_PERSIST_MEMORY` 时启动，权限为 0700。已存在且允许组或其他用户读写的目录会被设为 0700。 |
| `--memory-dir` 指定的目录 | 同上的文件，供那次运行使用。 | 运行开始时创建；新目录的权限为 0700。 |
| `~/.cache/localmelo/playground/` | 旧版 playground 的记忆。 | 默认关闭的 playground 启动时（security-1b 进行中）。 |
| `~/.localmelo/logs/` | 后台服务的日志。 | 由 `melo --daemon install` 创建。 |
| `~/Library/LaunchAgents/ai.localmelo.gateway.plist` | 后台服务的定义。 | 由 `melo --daemon install` 创建。 |

<a id="upgrading-from-an-older-config"></a>

## 从旧版配置升级

较旧的配置文件仍然能加载。localmelo 会读取：

- 单个 `backend` 键，最旧的格式；
- 后端名称 `mlc-llm`，现在是 `mlc`；
- `[cloud_api]` 和 `[online]` 表，现在每个厂商一张表；
- `[mlc]` 中的 `chat_port`，现在是 `chat_url`；
- 扁平的 `chat_template_kwargs`，现在位于 `extra_body` 之内。

再次运行向导会以当前格式写入该文件。
