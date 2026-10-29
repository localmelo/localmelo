<a id="command-line-reference"></a>

# 命令行参考

本页说明 `melo` 命令：它的模式、选项、输出、退出码和信号，以及如何在脚本中使用它。`melo-retrieve` 是 localmelo 安装的另一个命令，用于离线比较记忆检索器；见[离线检索](../../reference/retrieval.md)。

<a id="synopsis"></a>

## 概要

```text
melo [options] -- "<task>"                   # run one task (direct mode)
melo [options]                               # the interactive UI (in a terminal)
melo --base-url URL [--chat-model NAME]      # the interactive UI on this server
melo [--base-url URL] [--chat-model NAME] "" # the plain conversation loop
melo --reconfigure                           # run setup, then the interactive UI
melo --serve [--host HOST] [--port PORT]     # start the gateway (--port 0: a free port)
melo --serve --enable-playground             # opt in to legacy playground (in progress)
melo --serve --reconfigure                   # run setup, then start the gateway
melo --import-session dsh-jsonl FILE [--title TITLE] # import worker history
melo --daemon install|uninstall|status       # macOS background service
melo --developer --help                      # every option, grouped by tier
```

用引号括起任务，并把它放在 `--` 之后，这样形似选项的文字就不会产生歧义。不加引号的多个单词会以空格连接。既有的任务命令行、缩写以及 `melo ""` 循环，仍保持原有的解析方式。

<a id="modes"></a>

## 模式

| 命令 | 作用 |
|---|---|
| `melo -- "<task>"` | 直接模式：运行一个任务，打印结果后退出。 |
| `melo`（无任务） | 在终端中打开[交互式界面](../guides/terminal.md#the-interactive-ui)，使用已保存的后端。若没有已保存的后端，会先询问并保存一个模型服务器和一个模型；见[首次运行](../guides/terminal.md#first-run)。没有终端时以退出码 2 退出，并说明应改为运行什么。 |
| `melo --base-url URL`（无任务） | 在该服务器上打开交互式界面，模型为 `--chat-model`，或服务器列出的第一个对话模型。 |
| `melo --reconfigure` | 运行设置向导，然后打开交互式界面；默认模式下会先提供它找到的服务器（`Use it? [Y/n]:`），只有回答 `n`、没有找到服务器或使用 `--developer` 时才显示完整的后端菜单。没有终端时，设置向导从标准输入读取答案，完成后退出。 |
| `melo --serve` | 启动网关。若尚无配置，则先运行设置向导（需要终端；没有终端时以退出码 1 退出）。在仍在进行的 security-1b 改动中，`--enable-playground` 会启用经过认证的旧版 playground 路由，且只在 `--serve` 下有效。 |
| `melo ""`（空任务） | 打开纯对话循环：你输入的每一行都是同一个智能体的一项任务，只打印回答。`/help` 列出它的命令；`/exit`、`/quit`、`exit`、`quit` 或 Ctrl-D 退出。 |
| `melo --daemon …` | 管理在 macOS 上运行网关的后台服务；见[登录时运行网关](../guides/gateway.md#run-the-gateway-at-login-macos)。 |

在 `melo ""` 循环中，各轮共享同一个智能体及其工作记忆，因此每个回答都会保留，后面的轮次能看到之前的内容。未完成的轮次会在 stderr 上打印状态行，然后循环等待下一个任务。`/exit`、`/quit`、`exit` 或 `quit`（不分大小写），或 Ctrl-D，都会以退出码 0 结束循环；Ctrl+C 立即结束它（shell 显示 130）。循环启动时会打印一行，说明直接模式下 `shell_exec` 处于关闭状态，并说明 `melo "" --builtins all` 可以重新打开启用了它的循环（若该工具已启用，则不打印此行）。循环接受 `--base-url`、`--chat-model` 和 `--builtins`；与它一起使用时，其他直接模式选项都是用法错误。它的对话与网页工作区的对话彼此独立。在终端中，提示符由 readline 编辑：退格键删除一个完整字符（包括中文等宽字符），上箭头可以调出本循环中更早的输入行。不是有效 UTF-8 的输入行会打印 `melo: that line was not valid UTF-8; type it again.`，然后循环再次询问。

<a id="worker-histories"></a>

## Worker 历史记录

每次直接运行（`melo "<task>"` 和 `melo --brief FILE`）默认都会记录为 CLI 会话文件 `~/.cache/localmelo/cli/sessions.db` 中的一个 `worker` 会话，因此网页控制台的 Workers 视图和 macOS 应用能看到它，运行中的也一样。记录不改变运行发送或打印的任何内容：模型请求、stdout、JSON 结果和退出码都相同，首个请求之前大约多花 20 毫秒。把任务放在 `--` 之后；第一行任务作为标题，`--save-session TITLE` 可另设标题。`--parent-run ID` 把任务关联到启动器，不需要其他旗标。单次运行可用 `--no-save-session` 退出，整个主目录可在配置的 `[sessions]` 下设置 `record_direct = false`（见[配置参考](configuration.md)）；明确给出 `--save-session` 或 `--parent-run` 时，无论配置如何都会记录。存储无法打开或写入时，运行不会失败：stderr 会输出一行 `melo: could not record this run (<reason>); the run is unchanged`，运行继续，只是不记录。`-c`、`-r` 和对话选择器都不提供 worker 会话。提供了 `--workspace` 时，工作区和 git 元数据使用它；否则使用当前工作目录。

`--import-session dsh-jsonl FILE [--title TITLE]` 不需要任务，即可导入 dsh 的 JSONL 日志。重复导入内容相同的文件不产生任何变化；无法识别或无效的行会被计数。网关可以通过只读的 [worker API](api-v2.md#get-apisessions) 列出并读取 worker 历史，但不能继续或编辑它。

worker 记录保存在本地、未加密，文件权限为 0600。保存的文本会经过现有的密钥脱敏器处理；可识别的敏感文件读取会被替换为带路径标签的占位符。脱敏基于模式匹配：命令打印的、模型复述的，或通过无法识别的工具加载的任意密钥，仍可能被保留。分享前请先检查这些数据。推理内容受会话保留策略约束。

<a id="default-and-developer-mode"></a>

## 默认模式与开发者模式

`melo` 有两种模式。它们只改变 `melo --help` 的显示内容，从不改变任何选项的作用。

| 模式 | 如何选择 | `--help` 显示的内容 |
|---|---|---|
| 默认 | 不做任何设置，或在[配置文件](configuration.md#the-config-file)中写 `[ui] mode = "default"` | 用户选项，然后是这一行：`More options: melo --developer --help (or set [ui] mode = "developer").` |
| 开发者 | 在配置中写 `[ui] mode = "developer"`，或单次运行时加 `--developer` | 所有选项，分为 Options、Advanced 和 Experimental 三组；每个实验性选项的那一行都会注明决定它去留的 issue。 |

每个选项在两种模式下都能解析和生效，无论来自命令行还是配置文件，因此脚本和基准测试不受影响：模式只是把选项从帮助中隐藏，而不是删除它。`--developer` 无论放在命令行的哪个位置都对帮助生效（`melo --help --developer` 也可以），且必须写全名。既不是 `"default"` 也不是 `"developer"` 的 `[ui] mode` 会在 stderr 上打印一行并显示默认帮助；它从不会使运行或网关停止。

<a id="option-tiers"></a>

### 选项层级

| 层级 | 含义 |
|---|---|
| user | 新用户所需的选项；除这些之外不需要任何设置。默认模式的帮助只显示这些选项。 |
| advanced | 供脚本、网关、worker 和调优使用的稳定选项。 |
| experimental | 在基准测试中比较不同设计，可能改变或被移除。每个选项都注明决定其去留的 issue；获胜的方案会成为默认值，而该旗标会在一个版本之后移除。 |

下表是完整的分类，顺序与开发者帮助相同。它与 `support/option_tiers.py` 中的表格一致，并由一项测试保证两者相等。

| 选项 | 层级 | 决定于 |
|---|---|---|
| `-h`, `--help` | user |  |
| `--version` | user |  |
| `--base-url` | user |  |
| `--chat-model` | user |  |
| `--workspace` | user |  |
| `--max-steps` | user |  |
| `--timeout` | user |  |
| `--output-format` | user |  |
| `--memory` | user |  |
| `--brief` | user |  |
| `-c`, `--continue` | user |  |
| `-r`, `--resume` | user |  |
| `--yolo` | user |  |
| `--serve` | user |  |
| `--gateway-login` | user |  |
| `--reconfigure` | user |  |
| `--developer` | advanced |  |
| `--gateway-rotate-token` | advanced |  |
| `--enable-playground` | advanced |  |
| `--host` | advanced |  |
| `--port` | advanced |  |
| `--allow-unconfigured` | advanced |  |
| `--daemon` | advanced |  |
| `--import-session` | advanced |  |
| `--title` | advanced |  |
| `--save-session` | advanced |  |
| `--no-save-session` | advanced |  |
| `--parent-run` | advanced |  |
| `--inbox` | advanced |  |
| `--inbox-fd` | advanced |  |
| `--context-window` | advanced |  |
| `--output-reserve` | advanced |  |
| `--safety-margin` | advanced |  |
| `--builtins` | advanced |  |
| `--mcp-config` | advanced |  |
| `--mcp-only` | advanced |  |
| `--memory-dir` | advanced |  |
| `--request-timeout` | advanced |  |
| `--embedding-base-url` | advanced |  |
| `--embedding-model` | advanced |  |
| `--progress` | advanced |  |
| `--routine` | advanced |  |
| `--attempt-policy` | experimental | #185 |
| `--reflection-gate` | experimental | #185 |
| `--fixed-prefix` | experimental | #183 |
| `--window-eviction` | experimental | #183 |
| `--reflect-tool-guard` | experimental | #183 |
| `--cache-layout` | experimental | #183 |
| `--window-messages` | experimental | #183 |
| `--reasoning` | experimental | #183 |
| `--reasoning-field` | experimental | #183 |
| `--replay-turns` | experimental | #183 |
| `--note-writer` | experimental | #183 |
| `--tool-exposure` | experimental | #3 |
| `--captured-history` | experimental | #4 |
| `--retriever` | experimental | #4 |
| `--recall-at` | experimental | #4 |
| `--recall-k` | experimental | #4 |
| `--recall-kinds` | experimental | #4 |
| `--recall-placement` | experimental | #4 |
| `--recall-min-score` | experimental | #4 |
| `--admit-on-outcome` | experimental | #4 |
| `--keep-decision-text` | experimental | #4 |
| `--retrieval-extra-body` | experimental | #4 |

比较设计的新选项，在添加它的同一个 pull request 中以实验性选项的身份出现，并附上将移除它的 issue。新增的用户选项是有意的改动：测试会固定默认模式帮助所显示的选项数量。

<a id="options"></a>

## 选项

<a id="general"></a>

### 通用

| 选项 | 默认值 | 说明 |
|---|---|---|
| `query`（位置参数） | 未设置 | 要运行的任务。给出它即选择直接模式；给出空字符串（`""`）则打开纯对话循环；都不给则打开交互式界面。 |
| `-h`, `--help` | | 显示帮助并退出：默认模式下显示用户选项，[开发者模式](#default-and-developer-mode)下显示全部选项。 |
| `--version` | | 打印 `melo <version>`（包版本号；在 git 检出中其后附上短提交号，例如 `melo 0.1.0 (baab21e)`）并以 0 退出。 |
| `--developer` | 关闭 | 本次运行时，在 `--help` 中按层级分组显示全部选项；与 `[ui] mode = "developer"` 效果相同。它只改变帮助。 |
| `--base-url URL` | 未设置 | 把对话请求发送到这个兼容 OpenAI 的服务器，而不是已配置的后端。 |
| `--chat-model NAME` | 与 `--base-url` 同用时，任务默认为 `default`，无任务时为服务器列出的第一个模型；否则为已配置的模型 | 向 `--base-url` 服务器请求的模型，或代替已配置的对话模型。 |

- `--base-url` 按原样使用，因此要包含 `/v1`，例如 `http://127.0.0.1:11434/v1`。
- `--base-url` 不发送 API 密钥，也不读取配置文件，因此 `extra_body` 和 `request_timeout_s` 不适用。
- 不带 `--base-url` 时，`--chat-model` 只替换本次运行的已配置对话模型，配置文件不会改变。
- 没有任务时，两个选项都会打开交互式界面；见[使用哪个模型服务器](../guides/terminal.md#which-model-server)。与 `--serve` 或 `--daemon` 一起使用时，两者都是用法错误：网关使用配置文件。

<a id="direct-mode"></a>

### 直接模式

这些选项适用于不带 `--serve` 的任务运行；交互式界面也接受[交互式界面](#interactive-ui)一节中列出的选项。

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--brief FILE` | 未设置 | 从一个非空的 UTF-8 文件读取直接任务，文件不超过 1 MiB，路径相对于调用目录。不需要位置参数形式的任务，默认输出 worker 的 JSON，并拒绝文本输出。 |
| `--inbox FILE` | 未设置 | 与 `--brief` 一起使用时，在规划边界接受完整的、追加写入的指令行。路径必须显式给出，相对于调用目录，并且必须解析到工作区之外。 |
| `--inbox-fd N` | 未设置 | 与 `--brief` 一起使用时，从继承的、可读的管道文件描述符 `N` 接受同样的指令行。父进程必须保留写端；读端在工具启动前设置为 close-on-exec。不能与 `--inbox` 同时使用。 |
| `--workspace DIR` | 当前文件夹 | 运行所在的文件夹：文件工具只能在其中操作，`shell_exec` 和 `python_exec` 从这里启动；见[带工作区根目录](../guides/tools-mcp.md#with-a-workspace-root)。 |
| `--fixed-prefix on\|off` | `on` | 使用带版本号的固定系统前缀，本地日期在运行开始时冻结；`off` 会发送此前的请求。它会覆盖 `[agent] fixed_prefix`。 |
| `--window-eviction slide\|chunk\|boundary` | `chunk` | 工作记忆如何丢弃旧轮次：`chunk` 每次丢弃八条消息；`slide`（此前的默认值）在窗口满后每步丢弃一个单元；`boundary` 还会在下一次尝试时丢弃已结束尝试的轮次。它会覆盖 `[agent] window_eviction`；见[记忆](../../reference/memory.md#working-memory-window)。 |
| `--reflect-tool-guard on\|off` | `on` | 为 `on` 时，每个反思请求都提供该步骤规划请求的工具（列表、顺序和 schema 相同，因此服务器的提示前缀保持不变），且其最后一条用户消息以 `Answer with the JSON object only; do not call any tool.` 结尾；`tool_choice` 不变。重试请求不变。为 `off` 时，请求既不带工具，也没有那句话。它会覆盖 `[agent] reflect_tool_guard`；见[配置](configuration.md#context-accounting)。 |
| `--cache-layout v2\|v3` | `v2` | 启用固定前缀时，`v3` 在尝试边界把反思作为一条窗口消息只追加一次，让反思请求及其回复留在窗口中，并在同一前缀上再次询问不可读的反思；`v2` 是今天的请求。没有固定前缀时它不改变任何内容。它会覆盖 `[agent] cache_layout`；见[配置](configuration.md#context-accounting)。 |
| `--window-messages N` | `20` | 工作窗口的消息数上限，8 到 256。它会覆盖 `[agent] window_messages`；见[记忆](../../reference/memory.md#working-memory-window)。 |
| `--captured-history on\|off` | `off` | 把有界的、经过模式脱敏的捕获结果与模型预览分开存储；见[配置](configuration.md#context-accounting)。 |
| `--context-window N`、`--output-reserve N`、`--safety-margin N` | 未设置 | 选择启用逐次请求的上下文记账；三者都必须解析为正整数，且窗口必须大于另外两者之和。 |
| `--max-steps N` | `30` | 整个运行最多可以使用的规划步数。 |
| `--attempt-policy reflect\|none` | `reflect` | `reflect`：每次尝试之后都进行一次反思。`none`：只做一次尝试，不进行反思。 |
| `--reflection-gate flags\|utility` | `flags` | 反思之后，当它选择 `continue` 且三项检查都成立时（`flags`），开始下一次尝试；或者还要求其效用分数达到 0.1（`utility`，即本选项出现之前的行为）。 |
| `--tool-exposure retrieve\|all` | `retrieve` | 每个步骤向模型提供哪些工具。 |
| `--builtins all\|none\|NAME,...` | 所在模式的默认集合 | 要注册的内置工具。不带该旗标时，直接运行获得 `file_read`、`file_write` 和 `python_exec`；`all` 会加上 `shell_exec`（交互式界面默认保留全部四个）。 |
| `--mcp-config PATH` | 未设置 | 添加此 JSON 文件中列出的 MCP 服务器提供的工具。 |
| `--mcp-only` | 关闭 | 只使用 MCP 工具：等同于 `--builtins none --tool-exposure all`。 |
| `--memory on\|off` | `on` | `off` 会在本次运行中关闭长期记忆。 |
| `--memory-dir DIR` | 未设置 | 把历史记录、长期记忆和程序性记忆保存为 `DIR` 中的 SQLite 文件：`history.db`、`long_term.db` 和 `procedures.db`。 |
| `--timeout SECONDS` | 未设置 | 经过这么多秒后以状态 `timeout` 结束运行。 |
| `--request-timeout SECONDS` | `120` | 每次对话请求最多等待服务器的时间。 |
| `--output-format text\|json\|stream-json` | `text`（与 `--brief` 同用时为 `json`） | 文本、一个 JSON 结果，或以换行分隔的事件与结果（`stream-json` 需要 `--brief`）。 |
| `--embedding-base-url URL` | 未设置 | 供长期记忆使用的、兼容 OpenAI 的嵌入服务器；见[设置嵌入模型](../guides/embedding.md)。 |
| `--embedding-model NAME` | `default` | 向 `--embedding-base-url` 服务器请求的模型。 |
| `--progress auto\|off` | `auto` | 任务运行时在 stderr 上绘制[进度行](#progress-line)：当 stderr 是终端且输出为文本时使用 `auto`，`off` 则从不绘制。 |
| `--routine NAME` | 未设置 | 把此任务的步骤作为例程 `NAME` 学习和召回，无论在哪个文件夹运行，也无论任务的措辞如何。 |
| `--reasoning legacy\|drop\|replay\|compress\|compress+replay` | `legacy` | 回复的推理如何发回。`legacy` 保留提供方自身的行为（兼容 OpenAI 的服务器不发送推理，Ollama 原生接口使用它自己的字段）；`drop` 从不发送推理；`replay` 在范围内的助手轮次上以一个键发送推理；`compress` 不发送推理，并把每条回复的笔记和草稿保留在工作记忆中；`compress+replay` 两者都做。 |
| `--reasoning-field auto\|reasoning_content\|reasoning` | 未设置 | `--reasoning replay` 使用的请求键。`auto` 跟随上一条回复的推理来自哪个键；其他取值则固定为对应的键。与 `--base-url` 一起使用时需要它，或者在对话后端的配置小节中设置 `reasoning_field`。 |
| `--replay-turns N` | 范围内的每一轮 | 与 `replay` 或 `compress+replay` 同用时，保留最新 `N` 个助手轮次的推理；更早的轮次则不带推理发送。在 `legacy`、`drop` 和 `compress` 中它不起作用。 |
| `--note-writer auto\|heuristic` | `auto` | 在压缩模式中，由谁写每条回复的笔记：`auto` 读取模型的 `<note>` 块，没有时回退到启发式方法；`heuristic` 总是根据回复的文本和推理来写。在其他模式中接受但不起作用。 |
| `--retriever auto\|off\|bm25\|llm\|embedding` | `auto` | 召回如何为文档排序。配置了嵌入提供方时，`auto` 使用 `embedding`；否则在 `start`/`attempt` 使用 `llm`，在 `step`/`tool` 使用 `bm25`。`off` 关闭召回。见[循环中的召回](../concepts/memory.md#recall-in-the-loop)。 |
| `--recall-at LIST` | `start,attempt,step` | 召回的触发点：以逗号分隔的 `start`、`attempt`、`step` 和 `tool` 列表。`tool` 会注册 `recall` 工具，除非明确指定，否则从不开启。 |
| `--recall-k N` | `3` | 每次召回的任务内命中数，范围 1 到 10（另外最多 2 条长期记忆命中）。 |
| `--recall-kinds LIST` | 除 `thought` 外的全部 | 以逗号分隔的 `note`、`draft`、`thought`、`digest`、`reflection` 和 `long` 列表。 |
| `--recall-placement tool\|message` | `tool` | 步骤块放在哪里：附加到最后一个工具结果之后，或放在它自己的用户消息中。在 `replay` 和 `compress+replay` 中会拒绝 `message`。 |
| `--recall-min-score F` | `0.35` | 嵌入命中和长期记忆余弦命中的余弦下限，范围 0 到 1。对任务内的 `bm25` 和 `llm` 排序，以及只由 BM25 打分的长期记忆行，它不起作用。 |
| `--retrieval-extra-body JSON` | 未设置 | 仅用于 `retrieve` 请求（即 `llm` 检索器）的额外顶层字段，以 JSON 对象形式给出。在 `legacy` 之外，任务或交互式界面在 `llm`/`auto` 下可用；它从不保存到配置文件中。 |
| `--admit-on-outcome on\|off` | `off` | R6-lite：运行结束时，只准入最后一次尝试的摘要，并在其来源中记录终态；off 时在没有信号准入任何内容的情况下不写入任何内容。在 `legacy` 中接受但不起作用，在交互式界面中可以接受。 |
| `--keep-decision-text` | 关闭 | R6-lite：同时保存每个准入决定所检查的文本；SHA-256 始终保留。这是评估开关，在交互式界面中会被拒绝。 |

- `--workspace`：不指定时，任务在当前文件夹中运行，与以前完全一致。指定后，`~` 会被展开，路径的解析方式与交互式界面相同，运行在该文件夹中：文件工具只能在其中操作，`shell_exec` 和 `python_exec` 从这里启动。它不增加任何审批（见[带工作区根目录](../guides/tools-mcp.md#with-a-workspace-root)）。不是文件夹的路径是用法错误（退出码 2，只输出一行）；见[用法错误](#usage-errors)。`melo ""` 和 `--serve` 拒绝该旗标。
- 三个上下文旗标分别覆盖对应的 `[agent]` 键，在直接模式和交互式界面中均如此。超出预算的请求会在副本上做适配，否则运行以 `context_budget_exceeded` 结束；输出预留不是生成长度上限。见[上下文记账](configuration.md#context-accounting)。
- `--max-steps`：至少为 1。使用 `reflect` 时，一次运行的每次尝试最多 10 个步骤，最多 5 次尝试，总步骤数最多为 `--max-steps`。见[localmelo 如何工作](../concepts/how-it-works.md#the-run-loop)。
- `--reflection-gate`：见[反思](../concepts/how-it-works.md#reflection)。与 `--attempt-policy none` 一起使用时无效。
- `--tool-exposure`：见[工具的提供方式](tools.md#how-tools-are-offered)。
- `--builtins`：`all`、`none`，或以逗号分隔的 `file_read`、`file_write`、`python_exec` 和 `shell_exec` 列表。直接运行的默认集合不包含 `shell_exec`：`--builtins all`（或点名 `shell_exec`）会重新开启它，而且由于直接模式没有策略，它会在无需审批的情况下运行。`recall` 只有与 `--recall-at tool` 同用时才被接受，后者会注册它；没有该触发点时，它是用法错误。
- `--mcp-only` 需要 `--mcp-config`，且不能与 `--builtins` 或 `--tool-exposure` 同用。见[工具与 MCP](../guides/tools-mcp.md#add-tools-from-mcp-servers)。
- `--memory-dir` 会为本次运行开启持久化，不论环境变量如何设置。`~` 会被展开。见[记忆](../guides/memory.md#keep-memory-between-runs)。
- `--request-timeout`：默认值来自配置文件中的 `request_timeout_s`，没有则为 120 秒（`ollama` 后端为 300 秒）。与 `--base-url` 一起使用时为 120 秒，对 Ollama 的 `/v1` 地址也是如此。它不会更改配置文件。见[对话请求超时](configuration.md#chat-request-timeout)。
- `--embedding-base-url` 同样按原样使用：要包含 `/v1`，例如 `http://127.0.0.1:11434/v1`。
- `--embedding-base-url` 需要 `--base-url`，`--embedding-model` 需要 `--embedding-base-url`。不带 `--base-url` 时，使用已配置的嵌入后端。
- `--progress`：见[进度行](#progress-line)。`off` 使运行的启动方式与没有终端时的任务运行完全相同。
- `--routine`：不带它时，任务由其措辞（日期和时间已屏蔽）及其运行的文件夹来识别。给定相同 `NAME` 的运行属于同一个例程，因此 `--routine daily-mail` 会把在任何文件夹中以不同措辞提出的请求归入同一例程。空白名称是用法错误。见[程序性记忆](../concepts/procedures.md#how-it-works)。
- `--reasoning`：`legacy` 为默认值，线上的请求保持不变。`drop`、`replay`、`compress` 和 `compress+replay` 不读取 `[RECALL]`，并从显式模式回复的文本中移除内联的 `<think>` 块，因此得到的是干净的答案。`replay` 在范围内的助手轮次上发送推理，每轮最多 `40,000` 个字符（超过的轮次推理会被跳过并计数）；`requires_reasoning_replay` 为 true 的后端，其推理从不被截断，并且在那里 `drop` 会被拒绝。即使在这样的后端上，`legacy` 也从不被拒绝，但它仍可能返回 HTTP 400。`compress` 不发送推理，并在每次工具调用之前请求模型写一份简短的笔记；笔记和草稿保留在工作记忆中（见[推理笔记](../concepts/memory.md#reasoning-notes)）。`compress+replay` 两者都做。`requires_reasoning_replay` 为 true 的后端会把 `compress` 当作 `compress+replay` 运行，JSON 行会报告实际生效的模式。`--replay-turns` 计数带有推理的助手轮次；工具结果不占用轮次。默认的任务范围只包括当前任务的轮次：`melo ""` 循环或会话中更早的轮次不会被回放。在 `requires_reasoning_replay` 为 true 的后端上，窗口中的每个助手轮次都会被回放，任务范围和 `--replay-turns` 都不适用。`--base-url` 运行使用这些旗标，且不读取配置文件。
- `--retriever`、`--recall-*` 和 `--retrieval-extra-body`：见[循环中的召回](../concepts/memory.md#recall-in-the-loop)。`--retriever`、`--recall-*` 和 `--retrieval-extra-body` 在交互式界面中也被接受，但在 `legacy` 中没有作用：在那里，它们中的每一个都是用法错误（退出码 2，只输出一行）；`--retriever embedding` 没有嵌入提供方或与 `--memory off` 同用时，同样是用法错误；`--recall-placement message` 与 `replay`/`compress+replay` 同用、`--recall-at tool` 与 `--mcp-only` 同用或 `--builtins` 列表中没有 `recall`、`--builtins recall` 没有 `--recall-at tool`，以及解析出的检索器为 `off`、`bm25` 或 `embedding`（包括已保存的默认值）时使用 `--retrieval-extra-body`，也都是用法错误。在 `auto` 下，只有召回真的发出了 `llm` 检索请求时才会使用额外请求体。已注册的工具名为 `recall` 时，若又提供了一个同名工具，则是设置错误。这些模式之间的区别仅在于这些旗标：

  ```bash
  melo --base-url http://127.0.0.1:11434/v1 --chat-model MODEL \
    --reasoning compress --retriever bm25 -- "List the files"
  melo --base-url http://127.0.0.1:11434/v1 --chat-model MODEL \
    --reasoning compress --retriever off -- "List the files"
  ```
- `--admit-on-outcome` 和 `--keep-decision-text`：见[长期记忆](../../reference/memory.md#long-term-memory)。它们在 `legacy` 中被接受，但在那里没有任何作用。`--keep-decision-text` 会把可能涉及个人内容的文本写入存储，因此交互式界面会拒绝它（退出码 2，只输出一行）。在直接模式中，`[memory] admit_on_outcome` 是第一个开关的已保存默认值（交互式界面和网关不读取此表）；第二个开关只能作为旗标使用。
- `--note-writer`：`auto`（默认）会解析模型写出的 `<note>plan: … found: … open: … remember: …</note>` 块，没有时回退到启发式方法；`heuristic` 则忽略该块。笔记最多约 1,200 个字符，`remember: yes` 会把它标记为日后记忆的候选。这个旗标在每种模式下都被接受，但只有压缩模式会使用它。在 `config.toml` 中，它对应 `[agent]` 的 `note_writer` 键，而 `[agent] reasoning` 接受同样的五个值。在 `compress` 和 `compress+replay` 中，默认的 `--retriever auto` 也会在 `start`、`attempt` 和 `step` 召回笔记和草稿；见[循环中的召回](../concepts/memory.md#recall-in-the-loop)。
- `--reasoning-field`：`auto` 会解析为上一条回复的推理所来自的键，在第一条这样的回复之前不发送任何键。只有 `replay` 和 `compress+replay` 使用此旗标以及后端的 `reasoning_field`；在 `legacy`、`drop` 和 `compress` 中，线上行为与未设置任何字段相同。`ollama` 后端把该字段固定为 `thinking`，因此在那里使用此旗标是用法错误。交互式界面遵循同样的规则，包括切换后端之后。

<a id="interactive-ui"></a>

### 交互式界面

这些选项适用于不带任务的 `melo`。

| 选项 | 默认值 | 说明 |
|---|---|---|
| `-c`, `--continue` | 关闭 | 恢复此工作区中最新的一个符合条件的对话；没有时新建一个。 |
| `-r`, `--resume [ID\|TITLE]` | 关闭 | 恢复指定的对话；不带值则打开会话选择器。 |
| `--yolo` | 关闭 | 运行每个工具时都不再询问审批。 |

- 交互式界面同样接受 `--context-window`、`--output-reserve`、`--safety-margin`、`--workspace`、`--max-steps`、`--attempt-policy`、`--reflection-gate`、`--builtins`、`--memory`、`--memory-dir`、`--request-timeout`、`--reasoning`、`--reasoning-field`、`--replay-turns`、`--note-writer`、`--retriever`、`--recall-*`、`--retrieval-extra-body` 以及 `--embedding-*` 选项，对会话的每一轮都生效。显式旗标优先于已保存的 `[agent]` 和 `[retrieval]` 默认值；检索设置在 `legacy` 中保持不起作用，而在那里显式的检索旗标是错误。`--base-url` 会话从旗标获取这些值，而不是已保存的默认值。`--tool-exposure`（界面提供全部工具）、`--timeout`、`--output-format`、`--progress`、`--routine` 和 `--keep-decision-text` 是用法错误；`--mcp-config` 和 `--mcp-only` 被接受。
- 例如，`melo --reasoning compress --retriever bm25 --note-writer heuristic` 会启动一个带短笔记和召回的会话。没有 `/reasoning` 命令：请在启动进程时选择模式。`/model` 和 `/backend` 会保留模式、完整对话以及仅存于内存中的笔记和草稿。要求回放的提供方会拒绝 `drop` 会话，且不会改变其模型、对话或 MCP 状态；在这类提供方上，`compress` 会作为 `compress+replay` 运行。
- 会话回放覆盖保留的对话窗口，而不只是当前任务。恢复时，`pairs` 恢复回答但不含思考；`window` 可以回放检查点中保留的、已脱敏的思考。当前进程的模式优先于旧轮次上记录的模式。两种路径都不会重建仅存于内存中的笔记存储。见[恢复](../guides/sessions.md#resume)。
- 实时回复流式输出需要主动开启：设置 `[cli] stream = true`，并为每个模型设置 `[cli.reasoning_tags]` 提示；`/stream on|off` 会为之后的轮次切换它。界面只在实时区域中绘制经过脱敏并已放行的文本，为未成为最终答案的临时文本加上标签，并且只打印一次最终答案。不支持流式输出的提供方会静默回退为完整回复。直接模式和 stdin 已关闭的运行从不流式输出。见[流式回复](../guides/terminal.md#streamed-replies)。
- 交互式界面中的文本附件：`/attach PATH` 暂存一个文件（路径含空格时请加引号），`/attachments` 列出已暂存的文件，`/detach N` 或 `/detach all` 移除它们。从 `@` 文件补全中显式选择会暂存一个文件；只在正文中输入的 `@file` 不会。仅接受工作区内的普通 UTF-8 文本，不接受符号链接、路径穿越、URL、已识别的密钥文件、二进制文件或图片；清理后的文件名最多 200 个字符。硬链接可以指向工作区之外的文件；这不是沙箱。最多 8 个文件，每个 256 KiB，原始字节总计 512 KiB，同时还受组合后的提示词上限约束。脱敏是尽力而为的；如果清理或脱敏改变了快照或路径，请检查并确认。快照会以未加密的形式随会话保存，`/retry` 使用已保存的快照，而不是当前的文件。配对恢复有 32,000 字符的用户/回答预算，可能会省略某一轮；窗口恢复也可能截断或省略它。见[附加文本文件](../guides/terminal.md#attach-text-files)。
- 交互式界面中的 shell 命令：在空闲提示符下，`! COMMAND` 会以当前用户身份，在工作区中、于 `$SHELL`（否则为 `/bin/sh`）里运行 COMMAND，并把终端交给它（Ctrl+C 和 Esc 发给该命令）。它不是沙箱：不应用审批、检查器或策略。输出只留在终端中，从不进入模型上下文、历史记录或已存事件；`/attach !` 会把上一次输出的脱敏快照附加到下一条消息。单独的 `!` 会显示帮助，`\! text` 会发送一个字面上的前导 `! `；在一轮进行中、审批、切换或另一个命令期间，它会被拒绝，草稿保持不变。不会保存到输入历史中。见[运行 shell 命令](../guides/terminal.md#run-a-shell-command)。
- `--workspace` 在这里同样会被展开和解析；不是文件夹的路径与任务模式给出相同的一行用法错误，见[用法错误](#usage-errors)。

<a id="gateway-and-setup"></a>

### 网关与设置

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--serve` | 关闭 | 启动网关：HTTP API 和网页工作区。 |
| `--host HOST` | `127.0.0.1` | 网关监听的地址。 |
| `--port PORT` | `8401` | 网关监听的端口。与 `--serve` 同用时，`0` 表示让系统选一个空闲端口：启动信息会打印真实地址，登录提示行写明供 `melo --gateway-login --port PORT` 使用的端口，`melo-task` 则需要 `--gateway URL`。与 `--daemon`、网关认证命令同用，以及写在 `[gateway] port` 中，`0` 都会被拒绝。 |
| `--allow-unconfigured` | 关闭 | 与 `--serve` 一起使用时，即使没有配置任何后端或配置文件有校验错误，也启动网关，以便通过 `/api/settings` 进行配置。 |
| `--reconfigure` | 关闭 | 运行设置向导（默认模式下不带 `--serve` 时，先显示找到的服务器和 `Use it? [Y/n]:`；回答 `n`、没有找到服务器或使用 `--developer` 时显示完整菜单），然后启动网关（与 `--serve` 一起）或打开交互式界面（不带它时）。没有终端时，它从标准输入读取答案；不带 `--serve` 时，之后会退出。 |
| `--daemon install\|uninstall\|status` | 未设置 | 管理 macOS 后台服务；`install` 也接受 `--port`。 |

- `--host` 和 `--port` 的默认值来自配置文件中的 `[gateway]` 表。这些选项只为本次运行改变它们。
- 给出任务时，`--host`、`--port` 和 `--reconfigure` 会被忽略。
- 没有任务时，`--host` 和 `--port` 需要 `--serve`、`--daemon`、`--gateway-login` 或 `--gateway-rotate-token` 之一。
- 没有终端且没有配置文件时，`melo --serve` 会在 stderr 上打印一行并以退出码 1 退出，与 launchd 下的行为一致：`melo: no backend is configured; run melo --reconfigure in a terminal`。没有终端时，`--reconfigure` 仍会运行设置向导，并从标准输入读取答案，因此脚本可以通过管道输入。`--allow-unconfigured` 同时跳过设置向导和启动时校验，因此 `melo --serve --allow-unconfigured` 可以在没有配置文件的情况下启动；它不能与 `--reconfigure` 同用，也不能不带 `--serve` 使用。
- 在设置向导的问题处按 Ctrl-D 会打印 `Setup cancelled.` 并以退出码 0 退出；按 Ctrl+C 会打印相同内容，并以 130 退出。
- `melo --gateway-login [--port PORT]` 会在终端中打印一次性登录链接；`melo --gateway-rotate-token` 轮换共享的本地网关令牌。网关拒绝非回环地址的绑定，并要求认证；见[保持网关私有](../guides/gateway.md#keep-the-gateway-private)。

<a id="usage-errors"></a>

## 用法错误

`melo` 在运行任何内容之前会检查命令行。出现问题时，它会在 stderr 上打印 `melo: error: …`（通常在用法摘要之后），并以退出码 2 退出：

```text
melo: error: argument --max-steps: must be at least 1, got 0
```

以下情况属于用法错误：

- 取值无效，例如 `--max-steps 0`、`--timeout 0`、未知的内置工具名称、空白的 `--routine`，或者 `--memory-dir` 指向一个文件；
- `--workspace` 不是现有文件夹，在两种模式下都会以一行输出报告，不带用法摘要：`melo: --workspace: not a directory: DIR`；
- 无法组合的选项：没有 `--mcp-config` 的 `--mcp-only`，与 `--builtins` 或 `--tool-exposure` 同用的 `--mcp-only`，没有 `--embedding-base-url` 的 `--embedding-model`，没有 `--base-url` 的 `--embedding-base-url`；
- 直接模式选项（包括 `--workspace`）与 `--serve`、`--daemon` 或 `melo ""` 同用（但 `--builtins` 除外，循环会把它用于各轮），报告为 `only for direct mode (a query without --serve): --max-steps`；没有任务时，只报告交互式界面无法使用的那些选项（`--tool-exposure`、`--timeout`、`--output-format`、`--progress`、`--routine`）；
- `-c` 与 `-r` 同用，报告为 `-c and -r cannot be combined`；
- `--yolo`、`-c` 或 `-r` 与任务、`--serve` 或 `--daemon` 同用，报告为 `only for the interactive UI (melo without a query, --serve or --daemon): --yolo`；
- `--base-url` 或 `--chat-model` 与 `--serve` 或 `--daemon` 同用，报告为 `--base-url and --chat-model are for a task or the interactive UI; --serve uses the config file (melo --reconfigure)`；
- 没有任务时使用 `--host` 或 `--port`，并且没有 `--serve`、`--daemon` 或网关认证命令，报告为 `--host and --port are for --serve and --daemon; bare melo opens the interactive UI`；
- `--port 0` 与 `--daemon` 或网关认证命令同用，报告为 `--port 0 picks a free port for --serve only; …`；
- 没有任务的 `--reconfigure` 与 `--base-url` 同用，报告为 `--reconfigure sets up the config file, which --base-url does not use; give one of them`；
- `legacy` 中的召回旗标，报告为 `argument --retriever/--recall-*/--retrieval-*: retrieval is not available in legacy; use --reasoning compress, compress+replay, drop or replay`；
- 没有嵌入提供方或使用 `--memory off` 时的 `--retriever embedding`；
- 与 `--reasoning replay` 或 `compress+replay` 同用的 `--recall-placement message`；
- 与 `--mcp-only` 同用、`--builtins` 中没有 `recall`，或与 `--retriever off` 同用的 `--recall-at tool`；
- 与 `--retriever off`、`bm25` 或 `embedding` 同用的 `--retrieval-extra-body`；
- 与 `--base-url` 同用且没有 `--reasoning-field` 的 `--reasoning replay` 或 `--reasoning compress+replay`，报告为 `argument --reasoning: replay and compress+replay need a reasoning field: give --reasoning-field (auto, reasoning_content or reasoning) with --base-url, or set reasoning_field in the backend's config`；
- 在 `reasoning_field` 未设置的已配置后端上使用的 `--reasoning replay` 或 `--reasoning compress+replay`，报告为 `argument --reasoning: replay and compress+replay need a reasoning field: set reasoning_field in the chat backend's config section`；
- 在 `requires_reasoning_replay = true` 的后端上使用的 `--reasoning drop`，报告为 `argument --reasoning: this backend requires reasoning replay, so drop is not possible: use replay or compress+replay`；
- 与 `ollama` 后端同用的 `--reasoning-field`，报告为 `argument --reasoning-field: the ollama backend always reads and sends the thinking field`；
- 无法读取或无效的 MCP 配置文件；
- 无效的 `LOCALMELO_THINKING_RETENTION_DAYS`，在 `--serve` 或 `melo ""` 中同样如此；它只打印一行 `melo: error:`，不带用法摘要：`melo: error: LOCALMELO_THINKING_RETENTION_DAYS must be a number of days greater than 0, not 'abc'`。

<a id="output"></a>

## 输出

**文本**（默认）：

- 结果输出到 stdout。
- 状态不是 `completed` 时，stderr 上会输出一行：

  ```text
  melo: run ended with status budget_exhausted: Max steps reached
  ```

  错误信息会被合并为一行，并截断到 500 个字符。空的错误信息显示为 `no error message`。

**不带 `--brief` 的 JSON**（`--output-format json`）：

- stdout 恰好输出一行：运行记录，其中的[运行结果字段](run-results.md#fields)按顺序排列，末尾加上 `localmelo_version`。没有状态行。
- 当 localmelo 从 git 检出中运行时，`localmelo_version` 是 `git describe --tags --always --dirty` 的输出（例如 `603f40d`）；否则为已安装的包版本（例如 `0.1.0`），或 `unknown`。
- 在 `--reasoning legacy` 之外，记录还包含 `config.reasoning`、`config.replay_turns`、`config.note_writer`、`usage.reasoning_chars`（服务器报告时还有 `usage.reasoning_tokens` 或 `usage.cached_tokens`）以及 `memory_stats`；在不使用笔记计数器的模式中，这些笔记计数器为零。见[运行结果](run-results.md#fields)。`legacy` 的记录没有上述任何字段。

**带 `--brief`**：JSON 是一个带版本的 worker 结果：`v`、`status`、`run_id`、`exit_code`、`head_sha`、`git_dirty`、`tests`、`run` 和 `error`。HEAD 和脏状态取自工作区，不可用时为 null。`run` 是去掉 `config` 的运行记录；启动或传输错误时为 null。如果简报中没有 `Check:` 行，`tests` 只有在最后一次运行中最后一个通过的、可识别的测试检查时才会被记录为 `recorded`，否则为 `not_recorded`。已记录的项目包含原始命令或 Python 代码，经过脱敏并限制在 2,000 个字符之内（超出时带 `command_truncated`），其可为 null 的退出码，以及 `tool_event` 来源。它只是工具输出的证据，并不保证任务正确。若有 `Check:` 行，则 `tests` 改为由工具自身的结果给出（见下文）：`passed` 或 `failed`，每个命令一项，取自门禁最后一次运行，每项包含 `command`、`command_truncated`、`exit_code`、`duration_s` 以及来源 `brief_check`；`output_tail`（检查输出的最后 20 行，经过脱敏，最多 4,000 个字符；旧版本的记录中没有该字段）；若策略不允许某个命令运行，则还包含 `refused`（原因）。

带 `--brief --output-format stream-json` 时，stdout 为 NDJSON：每个智能体事件都包装为 `{ "v": 1, "type": "event", "event": ... }`；运行正常结束时，随后输出 `{ "v": 1, "type": "result", "result": ... }` 的 worker 结果。输出管道损坏或未被读完是失败；信号、进程终止或管道损坏都可能导致没有结果记录。应读取完整的 stdout，并同时检查最后的记录和退出码。在有效简报之后发生的智能体构造失败（包括智能体会拒绝的配置文件值，以及无效的 TOML 文件）会产生 `run: null` 的 `startup_error`，并以退出码 1 退出。对于事件输出失败，worker 会尝试输出 `run: null` 的 `transport_error` 结果，并以退出码 1 退出；输出失败或被阻塞时，该记录可能无法送达。写入启动记录时若管道损坏，仍以退出码 1 退出。用法错误不产生结果记录。Worker 输出使用基于模式的脱敏，并省略部分可识别的敏感文件读取，包括简单的 shell 和 Python 读取。启动错误的文本同样会脱敏，用户的主目录显示为 `~`。任意命令、代码和模型输出仍可能暴露密钥：该流不是密钥边界。分享前请先检查。直接运行（包括这一次）默认都会被记录（见上文）。

带 `--brief` 的运行与普通任务在四个方面有所不同。查询内容是简报，后面附加一段固定的段落，每次请求都相同：一旦知道改动该放在哪里，就做出满足任务的最小改动；每次改动之后运行任务的检查命令；修复你破坏的内容（包括断言旧行为的测试）；先读取即将编辑的行，因为较早的工具输出大约在十个步骤之后会离开上下文；并在最终回答中说明运行了哪些检查以及结果。若反思的动作是 `continue`，那么只要还有步骤和尝试，就会开始下一次尝试，不论其三个标志如何（`reflection` 事件仍会记录它们）。当某次尝试边界发现工作区自运行开始以来没有变化时，下一次尝试的第一个请求会带上这样一行：“Nothing has changed in the workspace yet. Start this attempt by writing the part of the task you can already write, then continue.”，并发出一条 `no_change` 通知；不是 git 工作树的工作区两者都不会得到。不带 `--brief` 的运行不变。

另外三项辅助机制会让 brief 运行持续写作。自上一次看到的变化（或开始时）以来，连续 6 个步骤中工作区摘要没有变化后，会加入一条用户行：“You have read for 6 steps without changing anything. Write the part of the task you can already write now (the smallest item first), then continue.”，并发出 detail 为 `{"scope": "step"}` 的 `no_change` 通知；计数在每次尝试开始时以及每次探测之后重置，因此这一行至多每 6 个步骤出现一次（边界行依然保留）。当窗口或适配丢弃了工具输出时，运行会为每个被丢弃的结果记住一行（工具、其关键参数缩短到 80 个字符、输出长度，以及最近的 8 条）；每个规划请求的末尾会附上“Earlier tool output no longer in your context: file_read support/sessions/runner.py (49,999 chars); shell_exec `grep -n ...` (2,310 chars). Read again what you need before you edit.”（若该步骤带有 inbox 行，则不附加）。反思请求的末尾附有一句话，要求在 `useful_evidence` 中给出带行号的确切文件路径，并要求把第一条要运行的命令或编辑作为 `next_promising_directions` 的第一项；当添加了上面的边界行时，它以“Start with: <that direction>”结尾（经过脱敏，最多 300 个字符）。

第四个差异是运行会自我检查。简报通过以 `Check:` 开头的行（行首没有空格）来指定其检查项；该行其余部分是一条 shell 命令，例如 `Check: uv run pytest tests/cli -q`。在运行可以结束为 `completed` 之前，工具会按顺序运行这些命令，每条都作为运行执行器的一次 `shell_exec` 调用，无论运行是否向模型提供 `shell_exec`（直接模式下不提供，除非指定 `--builtins all` 或 `--builtins shell_exec`；`--builtins none` 也不会阻止检查：要跳过检查，请删除该行）。只有工具自身的这次调用能越过这一过滤：模型的工具列表和它自己的 `shell_exec` 调用保持不变。检查器的 `pre_execute`、工作区根目录（`--workspace`）、shell 超时和脱敏与模型自己的调用一样适用。退出状态为 0 即通过。若某项失败，且该次尝试还有剩余步骤（或一个 `done` 反思可以开始另一次尝试），运行不会结束：下一个规划请求会带上一条用户消息，内容包括命令、其退出码以及输出的最后 60 行（经过脱敏），然后是“The task's check failed. Fix what it shows, run it again, then finish.”。每个失败的检查消耗一次重试，一次运行共有两次。一轮中的第一个失败会终止该轮。被拒绝的检查不会重试。若检查在最后仍然失败，运行仍以 `completed` 结束，但 `tests` 为 `failed`，会发出一条 `brief_check_failed` 通知，退出码为 4（见[退出码](#exit-codes)；stderr 会输出一行 `melo: the brief's check failed: <command> (exit N)`，被拒绝的检查显示 `not run`，超时显示 `no exit status`）；若最后一轮中所有检查都通过，则 `tests` 为 `passed`。每次运行检查都会发出一条 `brief_check` 通知，包含其命令、退出码和耗时。当运行处于最后一次尝试、总步数预算恰好还剩两步、工作区自检查上次运行（或运行开始）以来有变化，且门禁在该次尝试中尚未运行时，工具会在下一个规划请求之前运行一次检查，使模型仍有机会修复检查所示的问题。通过则不改变任何行为；失败则发送与门禁相同的消息，并消耗两次重试中的一次。`max_steps` 仍是硬性上限。没有 `Check:` 行的简报与以前的行为一致，模型对其自身检查的说明仍然只是说明。

使用显式的 `--inbox FILE` 时，worker 会在每个规划步骤开始时读取完整的、以 LF 结尾的非空 UTF-8 行。未完成的最后一行会等待其换行符。该文件可以最初不存在，但除此之外必须是只追加的普通文件（不能是符号链接），总共不超过 1 MiB，每行不超过 16 KiB。写入者需要串行化，每次以一次写入追加完整的一行，并且每次运行使用新的收件箱。没有默认的收件箱：只使用 `--brief` 时不读取任何文件，收件箱路径也从不进入模型的上下文。

使用 `--inbox-fd N` 时，同样的行、总量和每行限制以及确认机制，改由一个继承的、可读的管道文件描述符提供。父进程打开管道，传入读端（例如在 `subprocess.Popen` 中使用 `pass_fds=(read_fd,)`），并保持写端打开；它每次追加一条完整的行。worker 在第一次模型调用之前检查 `N` 是否为管道，并在那里把读端设置为 close-on-exec，因此 `python_exec`、`shell_exec` 和 MCP 服务器子进程都拿不到副本，也无法写入该通道。对于不可信的任务，推荐使用这种收件箱。已关闭的 fd 或不是管道的 fd，会在同一时间点以干净的 `setup_error`（退出码 2）被拒绝，且不进行模型调用。fd 没有路径，因此下面的工作区检查只适用于文件收件箱。`--inbox` 和 `--inbox-fd` 不能同时使用。

对于文件收件箱，路径在解析符号链接之后必须位于运行的工作区之外。位于工作区之内的路径会在任何模型调用之前被拒绝，因为默认的 `file_write` 和 `python_exec` 工具可能会改写它；符号链接、FIFO、目录、超过大小限制的文件和不可读的路径也一样，并在同一时间点检查。空的、非 UTF-8 的或超大的消息，或者在一行被接受之后文件被替换或截断，会在读取它的边界处结束运行。这些都是退出码为 2 的 `setup_error` 运行，只给出普通消息，没有堆栈跟踪，而不是 `crashed`。读取器要求已读取的字节仍是文件的前缀：保留这些字节的同一 inode 重写会被当作追加的内容接受，而更短的文件或更早的字节被改动则会失败。它无法检测保留了所有先前已读字节的截断并重写。

文件收件箱是引导通道，而不是信任边界：同一用户的进程，以及以用户权限运行的 `python_exec`/`shell_exec` 代码，仍然可以写入它或任何其他文件。把它放在工作区之外，只是阻止了受限的文件工具改写它。fd 通道关闭了工具写入的这条路径，但它不是身份认证：已经持有写端的进程可以引导运行，因此父进程不得把写端传给 worker 的工具或其他不可信的进程。

在 `stream-json` 中，`inbox.accepted` 会报告每条插入行的结束字节 `offset`、步骤和尝试，但不包含其文本。确认只表示指令进入了运行中的上下文，并不表示模型收到或遵循了它。确认从不会被丢弃：读取慢或停滞的读取者会让 worker 等待写入者，而不是跳过记录。最后一个规划边界之后追加的消息可能永远不会被接受。偏移量按每次实时运行提交一次，不跨崩溃或重启；应当核对确认记录，并为新的运行使用新的文件或管道。已保存的 worker 会把脱敏后的文本记录在其事件历史中。脱敏可能会改变该文本，而且会话恢复不会把收件箱事件重建为对话消息；不要把已保存的历史当作精确的回放。

**两种格式都适用：**

- stdout 只承载报告。工具自行写入 stdout 的任何内容（例如 `python_exec` 代码启动的、未捕获输出的程序），都会改为发送到 stderr。
- 警告输出到 stderr，每条一行。
- 只有当运行以 `crashed` 结束、Ctrl+C 之后，或者配置文件不是有效 TOML 且没有给出 `--brief` 时，才会出现堆栈跟踪。

<a id="progress-line"></a>

## 进度行

在终端中以文本形式输出时，`melo -- "<task>"` 会在 stderr 上用一行显示任务所处的位置。每次步骤和工具调用时，以及每秒，该行都会重绘；在输出结果和状态行之前，它会被擦除，因此不会留在滚动历史中：

```text
step 4 · Verify (cycle 1) · pytest 0:41 · tests 3/8 · 1m03s
```

- **何时显示。** 使用默认的 `--progress auto` 时，只有当 stderr 是终端、`TERM` 不是 `dumb` 且输出为文本时才显示。使用 `--output-format json`（无论是否在终端上）时不绘制任何内容；stderr 为管道或文件时（例如在脚本和基准测试中）、使用 `--progress off` 时，或在 `melo ""` 循环中，也都不绘制，运行的启动方式与没有该选项时完全相同。stdout 在两种情况下都一样。当 `melo` 在后台运行时（`melo "<task>" &`，或 Ctrl+Z 然后 `bg`），该行不会绘制，因此不会覆盖 shell 的提示符；使用 `fg` 切回前台后它会再次出现。
- **首次运行：事实。** `step 4` 统计规划步骤；`attempt 2` 从第二次尝试开始出现。阶段根据目前的工具调用读取：编辑之前为 `Explore`，编辑中为 `Write`，运行测试中为 `Verify`，测试通过之后为 `Finishing`；从第一次编辑之后的第一次测试运行开始，带有 `(cycle n)`，n 统计这样的编辑后再测试的轮次。之后是正在进行的工具调用及其耗时（`pytest 0:41`），最近一次测试摘要中的计数（`tests 3/8`：通过数 / 总数），以及运行的耗时（`1m03s`）。
- **重复的任务：也显示过去的运行。** 一旦某个任务的[例程](../concepts/procedures.md#how-it-works)在同一模型上以相同选项完成过（或者作为提示，同一文件夹中相似的任务完成过），该行就会添加那些运行做了什么，从不显示百分比：

  ```text
  step 4 · Write (cycle 1) · 1m03s · last time 6 steps, 2m10s: Read test_parser.py ✓ · Edit parser.py + pytest (×2) ◐ · git commit
  step 4 · Write (cycle 1) · 1m03s · past 5-8 steps, 2-3 min · last time: Read test_parser.py ✓ · Edit parser.py ◐ · git commit
  ```

  这些步骤来自过去的运行：`✓` 表示已完成，`◐` 表示当前步骤，`–` 表示跳过，尚未到达的步骤没有标记（本次运行不必全部完成）；`(×2)` 标记重复出现的模式。本版本不显示百分比进度条。步骤数从不与 `--max-steps` 对照显示，因为后者是限制，而不是计划。

  相似任务的运行只是提示，因此该行从不对它们显示 `last time`；它们会跟在 `like <routine title>:` 之后，并注明来源任务（例程的名称，否则为查询的开头部分），其步骤则用 `its steps:` 标示：

  ```text
  step 4 · Write (cycle 1) · 1m03s · like Fix the parser tests: 6 steps, 2m10s: Read test_parser.py ✓ · Edit parser.py ◐ · git commit
  ```
- **过去的运行需要持久化。** 每次 `melo` 都是一个新进程，除非在运行之间保留记忆，否则程序性记忆只存在于内存中。使用 `LOCALMELO_PERSIST_MEMORY=1` 或 `--memory-dir DIR` 运行（见[记忆](../guides/memory.md#keep-memory-between-runs)）；否则每次运行只显示事实。
- **窄终端。** 当该行比终端宽时，它会按以下顺序省略：过去运行的耗时、测试计数、过去运行中的具名步骤、尝试次数，然后用 `…` 截断其余部分。在中文、日文或韩文的区域设置中（`LC_ALL`、`LC_CTYPE` 和 `LANG` 中第一个已设置的值以 `zh`、`ja` 或 `ko` 开头），`·` 和 `…` 等标记按两个单元格计算，与把模糊宽度字符设为宽字符的终端绘制方式一致，因此该行在那里从不会换行。
- **颜色。** 该行为暗色；设置非空的 `NO_COLOR` 可使其以普通颜色绘制。
- **警告** 会在其上方单独一行输出。工具直接写到终端的输出（例如 `python_exec` 代码启动的、未捕获输出的程序）可能会覆盖该行，直到下一次重绘（一秒之内）。
- **信号。** SIGTERM、SIGHUP 和 Ctrl+C 会在 `melo` 停止之前擦除该行。
- **暂停的输出。** 当终端的输出被暂停（Ctrl+S）时，任务会在没有该行的情况下继续进行，信号仍会使 `melo` 停止；此时最后绘制的那一行会留在屏幕上。

<a id="exit-codes"></a>

## 退出码

| 代码 | 含义 |
|---|---|
| 0 | 运行以状态 `completed` 结束。 |
| 1 | 运行以 `crashed` 结束；或者因为没有配置后端或配置文件无效，`melo` 在运行之前就停止了；或者设置向导需要终端但没有终端；或者交互式界面无法打开其会话文件或启动其智能体。 |
| 2 | 运行以任何其他状态结束；或者用法错误；或者交互式界面没有终端；或者无论哪种模式下 `--workspace` 都不是文件夹。 |
| 3 | 网关无法启动，例如其端口已被占用。 |
| 4 | 一次 `--brief` 运行以 `completed` 结束，但其简报的检查最终失败：`tests.state` 为 `failed`，包括被拒绝的检查。`status` 仍为 `completed`，记录的 `exit_code` 为 4。没有 `Check:` 行的简报永远不会以 4 退出。 |
| 129 | SIGHUP 结束了运行。 |
| 130 | Ctrl+C（SIGINT）结束了运行。 |
| 143 | SIGTERM 结束了运行。 |

129、130 和 143 是 shell 为被该信号结束的进程报告的值。取消设置向导，或者在首次运行的问题处输入 `q`，会以 0 退出。在交互式界面打开之前，在设置问题处按 Ctrl-D 会以 0 退出，按 Ctrl+C 则以 130 退出，两者都会在输出 `Setup cancelled.` 之后发生。使用 `/exit`、Ctrl+D 或连按两次 Ctrl+C 离开界面，无论其轮次如何结束，都以 0 退出。

<a id="signals"></a>

## 信号

SIGTERM 和 SIGHUP 会在 8 秒内干净地结束直接运行：

1. 正在运行的 `shell_exec` 命令或 `python_exec` 子进程会与其进程组一起被终止。
2. MCP 服务器被关闭。信号发出 6 秒后仍在运行的服务器会被终止。
3. 任务记录以状态 `crashed` 保存，错误信息为 `Cancelled by SIGTERM`（或 `Cancelled by SIGHUP`）。反思不会被提升到长期记忆。
4. `melo` 因该信号退出（shell 显示 143 或 129），并且不在 stdout 上输出任何内容：没有结果、状态行或 JSON 行。

有一个例外：

- 若 `melo` 启动时某个信号已被忽略，它将保持被忽略。例如，`nohup` 会忽略 SIGHUP。

Ctrl+C（SIGINT）会立即停止运行。`melo` 会打印堆栈跟踪，把任务记录保存为 `crashed`，错误信息为 `CancelledError`，shell 显示退出码 130。

在交互式界面中，SIGTERM 和 SIGHUP（以及关闭终端窗口）会停止正在运行的轮次，该轮次会被记录为 `cancelled`，错误信息为 `Cancelled by SIGTERM`，并在同样的 8 秒内关闭工具，随后由该信号结束 `melo`。那里的 Ctrl+C 是一个按键（见[按键](../guides/terminal.md#keys)）。

<a id="use-melo-in-scripts"></a>

## 在脚本中使用 `melo`

读取 JSON 行以获取详情，读取退出码以获取结果：

```bash
line=$(melo --output-format json --timeout 600 -- "List the files in this directory")
code=$?
if [ -n "$line" ]; then
  status=$(printf '%s\n' "$line" | python3 -c 'import json, sys; print(json.load(sys.stdin)["status"])')
  echo "status: $status, exit code: $code"
else
  echo "no run record, exit code: $code" >&2
fi
```

- 退出码 2 涵盖多种状态。请读取 `status` 来区分它们；见[运行结果](run-results.md#statuses)。
- 使用 `--brief` 时，退出码 4 表示运行已完成，但其 `Check:` 命令失败：`melo --brief brief.md && next-step` 不会运行 `next-step`。详情请读取记录中的 `tests.state`。
- 应设置 `--timeout`。不设置时，没有任何东西会限制整个运行：一个在生成过程中持续发送保活字节的服务器永远不会超时。
- 若要从监督进程停止 `melo`，请发送 SIGTERM，并等待约 10 秒后再发送 SIGKILL；`melo` 的清理最多需要 8 秒。
