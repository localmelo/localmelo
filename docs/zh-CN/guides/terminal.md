<a id="guide-terminal"></a>

# 指南：终端

在终端里与 melo 对话：它的按键、命令、审批和工作区。不带任务运行 `melo` 时请阅读本指南；命令行选项见[命令行参考](../reference/cli.md)。

<a id="the-interactive-ui"></a>

## 交互式界面

![终端里的对话：思考、一个工具调用和答案。](../../assets/img/interactive-ui.webp)

- 在终端中不带任务运行 `melo`，会打开一个聊天界面，它安静地记录工作过程，并让回答成为屏幕上最醒目的内容。
- 每次思考一行，每次工具调用一行并附上其输出的第一行；回答以 Markdown 格式输出。
- 输出内容保留在终端的滚动缓冲区中；输入框和状态栏留在底部。
- 开发者模式保留完整的记录（见[一轮的输出](#a-turns-output)）。

```text
melo 0.1.0 · Qwen3-8B (/model to change)
~/demo-project
Try: What is this project? · / commands · ctrl+d quit

› What is in this folder?

✓ shell_exec  ls -1 · 0.1s · allowed
  └ app.py · +3 lines (/expand 1.1)
● The folder holds app.py, notes.txt, tests and todo.md.
✻ Worked for 6.2s · 1 tool call · ↓ 412 tokens

────────────────────────────────────────────────────────────────────────────
› Ask melo · / for commands
────────────────────────────────────────────────────────────────────────────
  Qwen3-8B · ~/demo-project · context 2.1k tokens · What is in this folder?
```

- 欢迎信息共三行：版本和模型，melo 工作的文件夹，以及一件可以试试的事，后面是命令和退出的按键。
- 警告紧随其后；`/status` 会检查模型是否有回应，并显示文件夹、会话、审批和上下文用量。
- 开发者模式（`--developer`，或 `[ui] mode = "developer"`）下，欢迎信息是列出所有行的方框。
- 它需要终端：标准输入和标准输出都必须是终端，且 `TERM` 不能是 `dumb`。否则 `melo` 会向标准错误输出以下内容，并以退出码 2 退出：

```text
melo: the interactive UI needs a terminal (stdin and stdout must be TTYs).
  Run one task:      melo "your task"
  Start the gateway: melo serve
```

<a id="the-main-chat"></a>

### 主对话

- `melo` 打开主对话：与网页控制台和 App 相同的对话、卡片和任务。
- melo 通过 `~/.cache/localmelo/gateway/endpoint.json` 找到正在运行的网关；无论由谁启动，每个网关启动时都会写入这个文件。
- 没有网关在运行时，melo 在后台用配置的端口启动一个，并用一行说明。它在 melo 退出后继续运行，任务因此不会中断，直到 `melo serve stop`；日志在 `~/.cache/localmelo/logs/gateway.log`。

```text
Started the gateway in the background on port 8401; it keeps running until melo serve stop.
```

- melo 从不启动第二个网关：App 的网关在自己的端口上持有主对话时，melo 就与它对话。App 遇到 melo 启动的网关时也一样。
- 连不上网关时，一行说明原因，终端继续使用自己的对话：

```text
⚠ No main chat (port 8401 is used by another program); this is the terminal's own chat.
```

- `/chats` 列出主对话和它的侧边对话（方向键和 Enter），`/side [title]` 在当前文件夹开始一个侧边对话，`/main` 回到主对话。状态栏显示当前打开的对话。
- 轮次请求的审批仍是熟悉的选项列表，答复经网关送达；在控制台答复的审批，这里的列表会关闭。
- 卡片是一行彩色文字：一个侧边对话或任务已完成、未完成或在等你，以及如何打开它。`/task …` 会在当前对话里开始一个任务，与控制台相同；关于别的对话的任务的回执，行尾会写上 `from` 和那个对话的名字。
- 任务向对话求助时，会有一行灰色文字 `Task “Fix tests” asks: <problem>`（中文目录里是「任务「Fix tests」在问：…」），随后是 melo 的回答，上面没有你的消息。接着是 `→ melo answered “Fix tests” · <answer>`，或者 `! “Fix tests” needs your answer: <question>`：只有这一个任务在等时，在这里发一条消息回答（`→ Sent to “Fix tests” as your answer`；10 秒内可在 `melo console` 里撤销）；有几个在等时，这一行会写 `several wait: /answer Fix tests …`。
- `/model`、`/backend`、`/undo`、`/copy`、`/sessions`、`/resume`、`/title`、`/usage`、`/attach`、`/approvals`、`/browser` 和 `/mcp` 只在终端自己的对话里可用。`melo -X cli.chat=local`（或 `[cli] chat = "local"`）打开它，`-u URL`、`-c`、`-r` 和 `--yolo` 也是如此。

<a id="which-model-server"></a>

### 选择模型服务器

![输入框下方的模型选择器：三个模型、勾选的当前模型，以及一个筛选框。](../../assets/img/model-picker.webp)

按以下顺序使用第一项适用的配置：

| | 来源 | 做法 |
|---|---|---|
| 1 | `melo setup` | 运行配置向导，然后使用它保存的后端。默认模式下，它先提供在本机找到的服务器（`Found a model server at …`、`Use this model server?`）；只有选择 `Use another server`、处于开发者模式，或已保存后端但没有找到服务器时，才出现完整菜单。不带任务时不能与 `--url` 同用。 |
| 2 | `--url URL` | 使用该服务器，不使用配置文件。模型使用 `--model` 指定的模型，否则使用 `GET URL/models` 列出的唯一对话模型（ID 包含 `embed` 或 `rerank` 的不是对话模型），再否则使用服务器说明已加载的那一个；有多个且不知道哪个已加载时，退出码为 2，列出这些模型并要求使用 `--model <id>`。 |
| 3 | 已保存的后端 | 若指定 `--model`，则在本会话中用它替代保存的模型。界面打开前不做检查：之后 melo 会列出一次服务器模型，如果服务器没有响应或没有列出该模型，就会警告。 |
| 4 | 没有已保存的后端 | 先回答[首次运行](#first-run)的问题，然后打开界面。 |

- 任务、`melo worker` 和 `melo ""` 在第一个请求之前以同样的方式选择模型（见[命令行参考](../reference/cli.md#general)）。
- 没有 `/v1` 的 URL 会通过 `URL/v1/models` 列出模型，该地址有响应时，界面也使用 `URL/v1`。
- 没有配置文件时，`--url` 会以[列表](#questions-are-lists)询问一次 `Save URL · MODEL as the default backend?`；默认选中 `Save it`，选择 `Not now` 或按 Esc 则什么都不保存。
- 服务器没有响应模型列表，或没有列出任何模型时，模型为 `default`，横幅下方的警告会说明原因。
- 配置文件写 `omlx`，或服务器的模型列表显示它是 oMLX 时，`/status` 把服务器称为 `oMLX`，否则称为 `OpenAI-compatible server`（Ollama、MLC LLM 和 SGLang 保留各自的名称）。

当模型不是保存的默认模型时，文件夹之后的一行会给出服务器的 URL 及来源（`from --url (not saved)`、`not saved; melo asks again next time`）。第一行显示软件包版本，在 git 检出中还会显示短提交号；`melo --version` 输出同一行。

<a id="what-melo-says-about-itself"></a>

### melo 如何介绍自己

- 每个对话都会先告诉模型它是谁，这样问“你是谁？”时，回答的是 melo：localmelo 的 melo，一个开源智能体，在你的文件夹里用工具工作，改动任何东西之前先询问你。
- 被问到时，它说出自己运行在哪个模型上；不把自己说成该模型原本的助手；并用你书写的语言和文字回答。
- 它像横幅那样写出服务器的名称（`oMLX`、`Ollama`、`OpenAI-compatible server`，或云服务商的名称），并说明服务器是否在本机；`/model` 和 `/backend` 会让下一轮用新的选择开始。
- 它还会写明对话记得什么：默认只记得本次对话最近的消息，不记得其他对话；如果记忆会在多次运行之间保留（见[记忆](memory.md#keep-memory-between-runs)），还会有 localmelo 在本机保存、并与你的问题相符的简短笔记。你文件夹里的文件不是记忆。
- 网页控制台和 macOS 应用也这样做；直接运行（`melo "task"`）、worker 和基准测试不发送这段说明。

<a id="keys"></a>

### 按键

| 按键 | 作用 |
|---|---|
| Enter | 发送消息。轮次运行期间将消息加入队列，在该轮完成时发送。 |
| Alt+Enter、Ctrl+J 或 `\` 后按 Enter | 在消息中换行。 |
| Esc | 关闭补全菜单；否则停止当前轮次。 |
| Ctrl+C | 依次尝试关闭菜单、清除草稿、清除排队消息、停止当前轮次；2 秒内再次按 Ctrl+C 则退出。 |
| Ctrl+D | 没有运行中的轮次且草稿为空时退出；有文本时删除光标下的字符。 |
| ↑ ↓ | 在各行之间移动，然后浏览之前的消息。轮次运行期间草稿为空时，↑ 可取回排队的消息。 |
| Ctrl+R | 搜索之前的消息。 |
| Ctrl+O | 列出本轮的步骤（轮次运行期间；轮次结束后列出上一轮）；Esc 关闭列表。 |
| Ctrl+T | 列出聊天之外的任务（网关的任务）；在某个任务上按 Enter 列出它的操作。 |
| Tab | 补全命令或列出的模型；↑ ↓ 在菜单中选择，Enter 确认所选项。 |
| Ctrl+L | 清屏。 |
| Ctrl+Z | 无操作（此版本的 melo 不会挂起）。 |

- **队列。** 一次只能有一条消息等待，显示为 `↳ Queued for the next turn: …`；如果当前轮次以 `completed` 以外的状态结束，消息会返回输入框，不会发送。
- **停止。** Esc 立即停止当前轮次（`■ Stopped`）；回答输出后，melo 最多花 2 秒记录运行步骤（参见[程序性记忆](../concepts/procedures.md)），下一条消息会等待记录完成。
- **历史记录。** 消息和命令保存在 `~/.cache/localmelo/cli_history` 中（权限为 0600；超过 1 MB 时保留最新的 1,000 条），文件内容绝不会保存到其中；无法打开该文件时，会话在内存中保留历史记录并提示你。

<a id="commands"></a>

### 命令

![终端里的 /status：模型及其能否回答、文件夹、会话、审批和上下文用量。](../../assets/img/status.webp)

- 输入 `/`，命令会显示在输入框下方并附上各自的作用；多输入几个字母可以缩小范围，Tab 或 Enter 选用一个，Esc 关闭列表直到你的下一次编辑。
- `@` 之后的文件名和命令的参数（`/model NAME`）也是如此。
- 常用命令排在最前：`/model`、`/new`、`/status`、`/help`、`/exit`、`/undo`、`/copy` 和 `/retry`。
- 以 `/` 和小写名称开头的消息是命令；未知名称会输出 `Unknown command /NAME. /help lists the commands.` 并保留草稿，而 `/etc/hosts is broken` 是一条消息。
- 要发送以命令样式开头的消息，请在开头加一个空格。
- 命令的输出会挂在该命令之下，前面是一个淡色的 `└`，这样你就能分清哪些行属于哪个命令；后面的行与第一行对齐，警告和错误保留自己的颜色。
- 纯文本终端（`NO_COLOR`、`TERM=dumb`、ASCII 符号）得到同样的版式，只是连接符为 `|`；`/model` 这样的列表仍然是列表，它关闭时输出的内容同样挂在命令之下。

```text
› /status
  └ model      Qwen3-8B · oMLX · http://127.0.0.1:8000/v1 · ✓ answers
    folder     ~/demo-project
    chat       new
    approvals  asks before running commands or changing files
    context    not measured yet
```

| 命令 | 作用 |
|---|---|
| `/help [command]` | 三组常用命令；`/help all`（开发者模式下直接 `/help`）列出所有命令和按键；`/help NAME` 给出某个命令的详情。 |
| `/model [name\|number]` | 单独使用：列出服务器的对话模型，当前模型标 `✓`，已加载的标 `(loaded)`；Enter 切换并把该模型保存为默认值，Esc 保留当前模型。带列出的 ID 或编号：从下一条消息起切换。服务器没有列出的名称会被拒绝。 |
| `/backend [url]` | 切换到另一个模型服务器：询问它的 URL 和列表中的一个模型，用 `GET /models` 检查，切换并把两者保存为默认值。 |
| `/new` | 开始新对话：模型忘记之前的对话，本对话的授权结束；一条 `── New chat ──` 分隔线标出开始位置。 |
| `/clear` | 在空屏上执行 `/new`：屏幕及其滚动缓冲区被清空，欢迎信息重新绘制。 |
| `/sessions [filter]` | 选择本文件夹中保存的对话；输入可筛选，Tab 切换全部文件夹，Enter 恢复，Esc 关闭。 |
| `/resume [id\|title]` | 按 ID、ID 前缀（至少四个字符）或完整标题恢复；不带参数则打开选择器。 |
| `/title [text]` | 显示或设置 1–200 个字符的标题。 |
| `/undo` | 忘掉最近一个已完成的轮次，并把它输入的消息放进 ↑ 历史。工具改过的文件不会还原。 |
| `/retry` | 用保存的原始输入文本重新运行最近一个已完成的轮次。 |
| `/copy [turn]` | 复制已保存回答的 Markdown；不带编号时复制最近一个已完成、未被撤销的轮次。 |
| `/usage` | 每个轮次记录的步骤、工具调用、词元和时间（包括被撤销的工作），并给出合计；`?` 表示未记录。 |
| `/status [all]` | 五行说明 melo 是否就绪：模型及其服务器（`✓ answers`，或原因）、文件夹、会话、审批的作用，以及上下文有多满。`/status all`（开发者模式下直接 `/status`）输出所有行。 |
| `/memory` | melo 记下的关于你的笔记，以列表显示：在某条笔记上按 Enter 会提供 `Back`、`Edit it` 和 `Delete it`；`Forget everything…` 会先询问。`/memory window` 不调用模型，检查模型的工作窗口。 |
| `/attach PATH` | 暂存一个工作区文本文件；`/attachments` 列出它们，`/detach N\|all` 移除它们。 |
| `/verbose [on\|off]` | 详细显示：40 行思考、工具调用的参数、20 行输出、反思的完整理由。 |
| `/thinking [on\|off]` | 打开思考摘录，或关闭（默认，开发者模式除外）。 |
| `/stream [on\|off]` | 对之后的轮次打开或关闭实时流式回复；见[流式回复](#streamed-replies)。 |
| `/expand [turn.call]` | 工具调用的完整输出或错误：最近一次调用、最近一轮的第 N 次调用，或第 T 轮的第 C 次调用。 |
| `/progress` | 本轮最新的进度事实：级别、步骤、尝试、阶段、测试和过去的运行。 |
| `/approvals [ask\|auto\|deny\|forget [tool]]` | 审批模式和本会话的授权；切换它，或忘掉某个工具的授权（用 `forget` 忘掉全部）。 |
| `/browser [on\|off]` | melo 是否可以在 Safari 中读取网页（见[用 melo 浏览网页](browsing.md)）。 |
| `/mcp` | MCP 服务器、它们的工具和来源文件，以及失败后 MCP 为何关闭。 |
| `/chats` | 在主对话中：以列表列出主对话和它的侧边对话；按 Enter 打开，`New side chat` 新开一个。 |
| `/side [title]` | 在主对话中：在当前文件夹开始一个侧边对话，由第一条消息创建。在那里 `/new` 和 `/clear` 也是如此。 |
| `/main` | 从侧边对话回到主对话。 |
| `/task TEXT` | 在当前对话（主对话或旁支对话）里开始一个任务，与控制台里的 `/task` 相同；任务属于那个对话。旁支对话要先有第一条消息，`/task` 才能用。 |
| `/answer TASK TEXT` | 回答当前对话里在等你的任务，用它的标题指明（`/answer Fix tests: B`）；只有一个在等时，`/answer B` 就够了。 |
| `/exit`（也可用 `/quit`） | 退出；正在运行的轮次会先被停止。 |

- `/new`、`/clear`、`/sessions`、`/resume`、`/undo`、`/retry`、`/model` 和 `/backend` 在轮次运行期间会被拒绝：草稿保留，提示为 `/model is not available while a turn runs (esc stops it).`
- `/model` 保留已配置的后端并更换它的模型；`/backend` 在服务器响应 `GET /api/version` 时把它保存为 `ollama`，模型列表标出 oMLX 时保存为 `omlx`，否则保存为 `vllm`，并保留配置的其余部分。
- 无法启动的模型会让当前模型保持不变（`✗ Could not start NAME: …`）；无响应的服务器会让当前服务器保持不变（`✗ Could not reach URL: connection refused. Still using URL.`）。
- 在 `melo --url URL` 会话中，这些选择只在本会话中有效，除非你对 `Save URL · MODEL as the default backend?` 回答了是。

<a id="attach-text-files"></a>

### 附加文本文件

在空闲的对话中，发送消息之前先暂存文件：

```text
/attach "notes/my file.txt"
/attachments
/detach 1
/detach all
```

- 输入 `@` 和工作区相对路径前缀，再从菜单中选一个文件，同样可以附加；仅在正文里写 `@file` 不会附加。
- 发送一条非空消息才会带上暂存的文件；附件命令不能在轮次运行期间使用。
- 只接受工作区内的普通 UTF-8 文本文件：不接受 URL、`..`、符号链接、已识别的密钥文件、二进制数据或图片。
- 限制：8 个文件、每个 256 KiB、总共 512 KiB；组合后的消息还必须符合智能体的提示限制。
- 模型收到的是只读取一次的快照，放在标明为不可信数据的块中；密钥脱敏只是尽力而为，所以附加之前请检查文件，脱敏或清理改变了内容时，melo 会先询问再暂存。

快照随会话保存在 `~/.cache/localmelo/cli/sessions.db`（明文，未加密），`/retry` 会使用它们。`pairs` 恢复对用户文本和回答有 32,000 个字符的预算，可能省略一个附件轮次；模型需要时，请重新附加并再次发送。纯 `melo ""` 循环和直接模式不支持附件。

<a id="run-a-shell-command"></a>

### 运行 shell 命令

在空闲的提示符处，以 `! ` 开头的消息会把其余部分作为命令，在你自己的 shell 中、以你的身份、在工作区文件夹里运行：

```text
! echo from-the-child; exit 3
```

```text
! user shell · ~/demo-project
$ echo from-the-child; exit 3
from-the-child
  └ exit 3 · 0.0s
  └ /attach ! shows this output (redacted, 15 bytes) and asks before attaching it
```

- **不是沙箱。** 该命令不是模型的工具：不适用任何审批、检查器或工作区策略，它拥有你账户的全部权限。
- **运行期间终端归该命令所有：** 读取输入或绘制屏幕的程序都能正常工作，Ctrl+C 和 Esc 交给命令；这个块以退出码或信号以及耗时结束。
- **输出留在终端。** 它绝不会到达模型，命令也不会保存到输入历史；`/attach !` 会在一个以 `Don't attach it` 开头的[列表](#questions-are-lists)之后，附上上一次输出的脱敏快照。
- **忙碌时会被拒绝**（`! runs only when melo is idle`）；单独输入 `!` 会输出这份说明，`\! text` 则发送以 `! ` 开头的消息。
- melo 退出时，命令的进程组会收到 SIGTERM（终端关闭时为 SIGHUP），2 秒后收到 SIGKILL。直接模式和纯循环中不可用。

<a id="inspect-undo-or-copy-a-conversation"></a>

### 查看、撤销或复制对话

- `/title` 可以在发送任何消息之前为对话命名；标题会在第一轮创建会话时生效。
- `/undo` 把最近一个已完成的轮次从模型恢复的工作窗口中移除，并把输入的消息保留在 ↑ 历史中；它不会撤销 shell 命令、Python 代码或文件编辑。
- `/retry` 重新运行最近一个已完成的轮次；如果 `/undo` 已经移除了它，就不会动之前仍然有效的轮次。
- `/usage` 包含被撤销的行，因为那些工作确实做过；运行中的轮次或缺失的计数器显示为 `?`，而不是零。
- `/copy` 读取已保存的、脱敏后的回答（不含思考），所以 Markdown 和代码不会从折行后的显示里复制；在 SSH 之外的 macOS 上使用 `/usr/bin/pbcopy`，其他情况发送需要终端允许的 OSC 52 请求，没有明确的命令就不会复制任何内容。

<a id="a-turns-output"></a>

### 一轮的输出

![终端里的对话：每个工具调用安静地占一行，最后是彩色的答案。](../../assets/img/terminal-chat.webp)

| 部分 | 你看到的 |
|---|---|
| 实时区域 | 第一行说明 melo 在做什么（`Starting`、`Thinking`、`Replying`、`Running`、`Waiting for your approval`、`Retrying in 4s`、`Reflecting`、`Wrapping up`、`Stopping`），然后是时间、输出词元数和 `esc to stop`。第二行是正在运行的工具及其自己的计时，对 `shell_exec` 和 `python_exec` 还有它们的时限。melo 以前运行过的任务，第三行会显示过去的运行及其步骤。开发者模式还会加上步骤、尝试、阶段和测试。 |
| 思考 | `✻ Thought for 1.2s`，一行；`/thinking on` 和开发者模式会加上前 3 行，`/verbose` 加上 40 行。中间消息在 `●` 之后以 Markdown 输出。 |
| 工具调用 | 完成时输出一个块：`✓` 已运行、`✗` 失败、`⊘` 被阻止或拒绝、`◌` 已停止或未运行；类型（`→` 读取、`±` 写入、`$` 运行、`@` 网络）和名称；耗时；命令以非 0 退出码结束时的 `exit N`；审批的结果；输出的第一行和 `+N lines (/expand T.C)`。失败的调用最多显示 3 行；输出中的转义序列会被移除。 |
| 写文件 | 对已有文件的 `file_write` 会在审批中显示差异（删除的行为红色，新增的行为绿色，并附计数）。 |
| 通知 | `↻` 重试，`·` 空回复或结束在思考中的回复，`⚠` 卡住或预算用尽的通知。反思显示为 `✻ Reflected after attempt 1: continue · Read the test before editing. (2.3s)`。 |
| 回答 | `●` 之后的 Markdown。未完成的运行会先输出它的最佳结果，再用平实的话说明发生了什么以及该怎么做（`✗ Could not reach the model server at URL.`、`Start it, then /retry.`）；页脚显示 `✻ Stopped after 14s`。直接模式仍在标准错误上输出 `melo: run ended with status …` 一行。 |
| 改动的文件 | 用 `file_write` 写过文件的轮次，会在页脚下面多一行安静的说明（见下）。 |

```text
✓ ± file_write  hello.py (15 chars) · 0.0s · allowed
  └ new file, 1 line
● I saved hello.py. Run it with python hello.py.
✻ Worked for 4.7s · 1 tool call · ↓ 50 tokens
  Changed + hello.py (+1)
```

- 多个文件显示为 `Changed + hello.py (+1), ~ README.md (+3 −1)`：`+` 表示新建的文件，`~` 表示修改过的文件，并附上本轮增删的行数；最多显示三个名称，其余写 `and 2 more`。
- 这一行来自本轮自己的 `file_write` 调用：被拒绝或被驳回的写入不会列出，由 shell 命令或脚本改动的文件也不会列出。
- 本轮没有改动任何文件或你停止了它时，不输出任何内容；开发者模式显示同样的一行。

<a id="the-status-bar"></a>

### 状态栏

输入框下方的栏回答“用了多少”和“旁边在运行什么”，分成几个区，之间有竖线：用量、聊天之外的工作，以及运行（没有轮次运行时是对话的标题）。在本地服务器上：

```text
  ↑ 3.0k ↓ 15 · ctx 2.6k/32k ▮▯▯▯▯ · cache 61% (avg 72%) · idle 2m │ Fix the parser
```

| 部分 | 含义 |
|---|---|
| `↑ 3.0k ↓ 15` | 本会话发送和接收的词元数。 |
| `ctx 2.6k/32k ▮▯▯▯▯` | 最近一次测得的提示词占输入上限的比例，用五格表示：从 70% 起为琥珀色，从 90% 起为红色。没有上限时只显示词元数。 |
| `cache 61% (avg 72%)` | 服务器报告的、最近一次提示词命中缓存的比例，随后是本会话的平均值。从 50% 起为青绿色，第一次请求之后低于 20% 为琥珀色；不报告缓存的服务器不显示这个数字，绝不显示 0%。 |
| `idle 2m` | 距最近一次请求的时间，一分钟起显示。 |
| `$0.13 · ctx 100k · limit 42% ▮▮▯▯▯ ↻3m` | 在云端 API 上：按你配置的价格（见 [`[pricing]`](../reference/configuration.md#pricing)）算出的本会话费用，以及最近一次响应报告的最紧的速率限制和重置前的时间。melo 不内置价格。 |
| `2/~4 · ctrl+o steps` | 轮次运行期间：第 2 步，共约 4 步（该任务之前用了 4 步；`~` 表示估计值），以及列出这些步骤的按键。 |
| `2 running · 1 queued · P1 · ctrl+t tasks` | 网关有未完成的任务时：按状态统计的数量、最高的优先级（P0 最先运行，P1 是今天，P2 正常，直到 P5 暂停），以及列出它们的按键。 |

- Ctrl+T 列出任务；在某个任务上按 Enter 会提供 Pause 或 Resume、Raise to、Lower to 和 Stop（会先询问），效果与 `melo task pause|resume|level|stop` 完全相同。
- melo 每 5 秒向网关询问一次任务，并且只在存在网关令牌（`~/.cache/localmelo/gateway/auth`）时才询问。
- 终端较窄时，先去掉细节，再去掉数字；这一栏从不折行。只有模式不是 `ask` 时，它才显示在右边缘。
- 这些数字来自 melo 已经收到的事件；不会向模型服务器查询。`[cli] status_line = "off"` 只保留运行区、标题和最近一次请求的上下文。
- 设置 `NO_COLOR` 或 `LOCALMELO_ASCII=1` 时，各区仍然分开（`│` 或 `|`），ASCII 写作 `in 3.0k out 15`。

<a id="streamed-replies"></a>

### 流式回复

- 交互式界面默认在模型写回复的同时流式显示，只绘制发布过滤器已经脱敏的文本。
- `/stream off` 对之后的轮次关闭它，`melo -X cli.stream=false` 只对一次运行关闭，`melo config set cli.stream false` 则永久关闭。
- 不支持流式的服务器无需任何设置：melo 在回复完整后显示它，并只说明一次（`This server sends replies whole; melo shows them when they are complete.`）。
- 回复到达一部分后流中断，仍然是错误。
- 直接模式、worker 和网关从不流式输出。

回复能否边写边显示，取决于它的模型如何标记推理，这由 `config.toml` 中针对每个模型的提示来声明：

```toml
[cli.reasoning_tags]
"Qwen3-8B" = "closing_only"
```

- 提示是 `explicit`（开头的 `<think>` 开始推理）、`closing_only`（`</think>` 之前的一切都是推理）或 `none`（标签是字面文本）。
- 没有提示的模型沿用保守规则：内容会等到回复结束才显示，实时区域显示 `Thinking…`；自己用单独字段发送推理的服务器无需提示就能流式输出。
- 使用 `closing_only` 时，没有 `</think>` 的回复是未完成的思考：melo 会再问一次，第二次仍如此则以 `melo did not finish an answer.` 结束该轮。
- 流式文本是临时的：回复变成工具调用、重试、失败或停止时，melo 会把它输出一次，开发者模式会标上 `↳ partial reply (tool call)`。

<a id="approvals"></a>

### 审批

![终端里的审批面板：写文件将做出的改动，以及一组可选的回答。](../../assets/img/terminal-approval.webp)

`shell_exec`、`python_exec` 和 `file_write` 要等你回答后才运行，标记为需要审批的 MCP 工具也是如此；`file_read` 和 `list_files` 只读，所以从不询问。面板取代输入框的位置，你的草稿保持不变：

```text
╭─ Run a shell command? ───────────────────────────────────────────────────╮
│ shell_exec · in ~/demo-project                                           │
│   ls -1                                                                  │
╰──────────────────────────────────────────────────────────────────────────╯
❯ 1. Yes
  2. Yes, and allow every shell_exec call in this session
  3. No, and tell melo why
  "Allow every" lets routine shell commands run without asking for the rest of this session.
  Installs, downloads and changes outside ~/demo-project still ask.
  ↑↓ select · enter confirm · 1-3 · v view all · esc stop the turn
```

安装软件、下载文件、修改文件夹之外的文件或做其他[高影响](tools-mcp.md#high-impact-calls)操作的调用，会得到一个面板：问题点明了它是什么，理由写在代码上方，没有会话选项，且选中“否”，所以只按 Enter 绝不会允许它：

```text
╭─ Install software? ────────────────────────────────────────────────────────╮
│ shell_exec · in ~/demo-project                                             │
│ ⚠ melo asks because this call installs playwright with pip (downloads      │
│   about 42 MB).                                                            │
│   pip install playwright                                                   │
╰────────────────────────────────────────────────────────────────────────────╯
  1. Yes
❯ 2. No, and tell melo why
  Session grants never cover installs, downloads or changes outside the
  folder.
  ↑↓ select · enter confirm · 1-2 · v view all · esc stop the turn
```

- `1` 允许这次调用，`2` 在本会话余下时间内允许该工具的常规调用，`3` 打开一个填写理由的输入框（最多 500 个字符），模型会读到它；↑ ↓ 和 Enter 同样可以选择。
- 面板打开后的 0.4 秒内只有 Esc 和 Ctrl+C 有效，所以你正在输入的消息不会误回答它；Esc 或 Ctrl+C 会停止当前轮次。
- 命令、代码或文件内容按它将要运行的样子完整显示，折行但绝不截断；控制字符和会改变文字顺序的字符会转义显示并附一行警告。
- 面板最多显示正文的 8 行，之后是 `… +N lines (v views all)`。
- `--yolo` 不经询问运行所有工具，状态栏显示 `⚠ YOLO: no approvals`；开发者模式会加上工具的权限层级（见[权限层级](tools-mcp.md#permission-tiers)）。

<a id="auto-mode"></a>

### 自动模式

在自动模式下什么都不会询问：常规步骤直接运行，有风险的步骤或你没要求的步骤会被拦下并说明原因。

- **切换：** `/approvals auto` 为这个对话打开它，`/approvals ask` 关闭它。`[approvals] mode`（控制台的 **新对话开始时**，或单次运行用 `-X approvals.mode=auto`）让终端自己的对话以它开始。
- **会运行什么：** 在文件夹内、不安装、不下载、不发送、不删除任何东西的步骤；melo 无法判断的步骤先交给你的模型。它的 `shell_exec` 和 `python_exec` 调用在写入沙箱中运行，所以即使没有规则发现，写入文件夹之外也会失败，而且它们既读不到 melo 的登录令牌，也连不上 melo 自己的 API。自动模式是审查，不是围墙：把工作交给另一个应用或程序的步骤由审查来判断（拦下，或交给你的模型检查），而不是由沙箱隔离。
- **会拦下什么：** 安装、下载、更改文件夹外的文件、删除、密码和密钥、系统设置以及来自网络的代码，除非你自己的话要求这样做；更改 localmelo 本身的步骤永远不会运行。那一行会写明原因（`⊘ … · blocked: installs software`），melo 会在回答里说明它想做什么，以及还有哪些别的办法。
- **允许它：** 说“go ahead”（继续），melo 会再试一次；你的模型会核对你的话确实要求了它。只有点按钮才能允许的步骤（密码或密钥、系统设置、以后自动运行的东西、来自网络的代码或 melo 无法检查的代码）需要用控制台的 **Allow once**。

<a id="the-workspace"></a>

### 工作区

- melo 在当前文件夹中工作，或在 `--workspace DIR` 指定的文件夹中工作。
- 文件工具（`file_read`、`file_write`、`list_files`）只能在这个文件夹内；命令和 Python 可以访问其他文件，所以 melo 会先询问。
- 工作区是你的主目录或 `/` 时，横幅会发出警告。
- `shell_exec` 和 `python_exec` 在界面中 10 分钟后停止（任务中分别为 30 秒和 60 秒）。

<a id="chinese-and-other-wide-characters"></a>

### 中文及其他宽字符

输入框按字符编辑、按终端单元格排布文本：中文和其他宽字符能整字输入、删除和折行。已知的限制：

- 由多个码点组成的字符（用组合记号输入的重音、👩‍💻 这样的表情序列），每个码点需要按一次退格键。
- Ctrl+W 删除到上一个空格，在没有空格的中文文本里就是整段。
- 输入法（例如 macOS 拼音）仍在组字的文本，由终端绘制，而不是 melo。

<a id="terminal-colours"></a>

### 终端颜色

颜色把一轮的输出分门别类：回答、失败和改动过的文件一眼可见，读取则淡出。每种颜色都有一个符号或固定位置表达同样的意思，所以 `NO_COLOR` 和 `LOCALMELO_ASCII=1` 不会丢失任何信息。

| 内容 | 符号（ASCII） | 颜色 |
|---|---|---|
| 你的消息 | `›`（`>`） | 苔绿色标记，整行衬在浮起的色带上 |
| 回答 | `●`（`*`） | Melo 的橙色，唯一的橙色 |
| 不是回答的消息；思考 | `●`（`*`）；`✻`（`*`） | 暗淡色 |
| 读取的工具 | `→`（`>`） | 暗淡色，所以读取会淡出 |
| 写入的工具 | `±`（`~`） | 紫色 |
| 运行命令或代码的工具 | `$` | 蓝色 |
| 访问网络的工具 | `@` | 靛蓝色 |
| 已运行；失败或以非 0 退出码结束；被拒绝；未运行 | `✓`（`+`）；`✗`（`x`）；`⊘`（`!`）；`◌`（`-`） | 青绿色；红色，附 `exit N`；琥珀色；整行暗淡 |
| 改动的文件：新建；修改 | `+`；`~` | 青绿色；琥珀色 |

- 面板、列表和各种栏用同样的颜色：选中的行衬在苔绿色色带上并只有一个 `❯`，审批面板有琥珀色边框，差异中新增的行为青绿色、删除的行为红色。
- 在 256 色或更多、Unicode、80 列或更宽的终端中，欢迎信息旁边会有 Melo 的头像；否则欢迎信息只有那三行。
- **颜色数量。** `NO_COLOR`（任何非空值）或 `TERM=dumb`：不用颜色，只用粗体、暗淡和下划线。否则依次看 `[cli] colors` 设置、`COLORTERM=truecolor`（或 `24bit`）、以 `256color` 结尾的 `TERM`，再否则使用终端主题的 16 种命名颜色。
- **背景。** 先看 `[cli] background` 设置，再看终端自己的回答（melo 只问一次，最多等待 100 毫秒），最后看 `COLORFGBG`；知道确切颜色时，melo 只在你的文字保持 4.5:1 对比度的地方绘制色带和小块。
- **颜色看起来不对时。** 有些终端，以及 tmux、screen 或 ssh，什么都不回答：用 `-X cli.background=light`（或 `dark`）告诉 melo 你的背景，只用于本次运行，`melo config set cli.background light` 则保存下来；`-X cli.colors=16`、`256` 或 `truecolor` 设置颜色数量。

<a id="terminal-settings"></a>

### 终端设置

- 颜色：见[终端颜色](#terminal-colours)；你必须阅读或据此行动的句子绝不会被调暗。
- 终端编码不是 UTF-8，或设置了 `LOCALMELO_ASCII=1` 时，melo 用 ASCII 替代符号（`›` 用 `>`，`…` 用 `...`）。
- 审批面板打开时，以及 30 秒或更长的轮次结束时，melo 会让终端响铃；`LOCALMELO_BELL=0` 可关闭。

<a id="sessions-and-files"></a>

### 会话和文件

- 单独运行 `melo` 会开始新对话；只有第一条消息才会保存它。`melo -c`（`--continue`）恢复当前工作区中最近一个有已完成轮次的对话，`melo -r`（`--resume`）则让你选择一个。
- `melo -r REF` 按 ID、至少四个字符的前缀或不区分大小写的完整标题解析；多个匹配会打开选择器，未知引用以退出码 2 退出。`-c` 和 `-r` 不能同时使用，也不能与任务或 `melo serve` 一起使用。
- 选择器最多列出 50 个有已完成、未被撤销轮次的对话，最新的在前；Tab 切换本文件夹和全部文件夹，Enter 恢复，Esc 关闭。`/sessions FILTER` 和 `/resume REF` 使用同一个选择器。
- 恢复会沿用当前的模型和工作区并清除审批授权；回顾部分以纯文本显示最近三轮。`[sessions] resume_mode = "pairs"`（默认）恢复最近的用户消息和回答配对，`"window"` 恢复保存的工作记忆检查点加之后的配对。
- 退出时，有轮次的对话会说明如何回到它：`◆ Saved "What is this project?". Continue it with: melo -c`，或 `Resume it with: melo -r ID`。

文件放在哪里：

- 会话保存在 `~/.cache/localmelo/cli/sessions.db`：每个轮次，包括你的消息、回答和每次工具调用及其输出；`~/.cache/localmelo/cli_history` 保存你的消息。两者都是明文（权限 0600），未加密。
- 网页控制台的 Workers 视图在 **Terminal chats** 下以只读方式显示你的对话，读取时会脱敏；它不能修改这些对话，也不能向它们发送内容。
- 界面把日志写入 `~/.cache/localmelo/logs/cli.log`，从不打印回溯；不是有效 TOML 的配置文件会让 `melo` 以一行 `melo: …` 和退出码 1 停止。
- 程序性记忆遵循[记忆持久化](memory.md#keep-memory-between-runs)，与任务相同。

<a id="first-run"></a>

### 首次运行

在没有配置文件的终端中，`melo` 会提供它找到的模型服务器，询问模型，保存它们，然后打开界面。之后的运行直接进入。

```text
$ melo
melo uses a model server on this computer (oMLX, Ollama, LM Studio…).
Found a model server at http://127.0.0.1:8000/v1 (1 model).
✓ http://127.0.0.1:8000/v1 lists 1 model; using Qwen3-8B-4bit.
melo remembers across chats by words only; melo setup adds search by meaning.
Saved to ~/.cache/localmelo/config.toml (melo setup changes it).
```

- melo 通过 `GET /v1/models` 在 `127.0.0.1` 的端口 8000（vLLM、oMLX）、8400（MLC）、11434（Ollama）和 30000（SGLang）上查找本机服务器，并提供第一个列出模型的服务器（有对话模型的排在最前）；不会发送任何对话请求，所以服务器不会加载任何东西。
- 在第二行和第三行之间，melo 以列表询问 `Use this model server?`；有多个模型时会显示选择器，有嵌入模型时还会有一个列表询问 `Remember across chats with nomic-embed-text?`（`Yes, with it`、`No, by words only`）。
- 没有服务器在运行时，它会输出启动服务器的命令（Mac 上是 oMLX，其他系统是 Ollama；见[模型服务器指南](model-servers.md)），并询问 `No model server found. What now?`（`Check again`、`Enter an address`、`Quit`）。
- 无法使用的服务器会连同原因一起指出（`connection refused`、`no answer in 5 s`、`HTTP 401` 或 `HTTP 403` 并附 `the server requires an API key; localmelo sends none to local servers (oMLX: "Allow inference without API key")`、`it lists no models`），你可以重试、不保存直接启动或退出。
- Ollama 服务器保存为 `chat_backend = "ollama"`，oMLX 服务器保存为 `chat_backend = "omlx"`，其他保存为 `"vllm"`；嵌入模型的选择见[配置嵌入模型](embedding.md)。
- `LOCALMELO_DETECT_URLS` 用逗号分隔的列表替换这四个 URL；空值会关闭查找。

没有找到服务器时，melo 会输出：

```text
No model server is running yet. To start one with oMLX:

  brew tap jundot/omlx https://github.com/jundot/omlx
  brew install jundot/omlx/omlx
  omlx serve --model-dir ~/models --port 8000

Put a model in its own folder of ~/models first. The guide shows how:
https://github.com/localmelo/localmelo/blob/main/docs/guides/model-servers.md
```

<a id="questions-are-lists"></a>

#### 问题都是列表

在终端里，每个答案固定的问题都是列表，工具审批和 `/model` 也是如此：

```text
 Use this model server?
 ❯ 1. Use it
   2. Use another server
 ↑↓ select · enter confirm · esc cancel
```

- ↑ ↓ 移动，Enter 确认选中的行，数字键（1 到 9）直接选择对应的行。
- 安全或推荐的答案最先被选中，Esc 给出按键说明行末尾所写的答案（`esc cancel` 会结束设置；`esc not now`、`esc no, by words only` 和 `esc don't attach it` 表示拒绝）。
- 没有筛选框：列表很短，在其中输入字母不起作用。
- Ctrl+C 输出 `Setup cancelled.` 并以 130 退出，Ctrl+D 以 0 退出。
- 没有终端时（答案通过管道输入、脚本、测试），不绘制列表，每个问题仍读取它一贯读取的键入答案（`Use it? [Y/n]:`、后端的编号、默认值的 Enter）；直接运行（`melo "task"`）从不询问任何问题。
