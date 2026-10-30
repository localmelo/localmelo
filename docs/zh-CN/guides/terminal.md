<a id="guide-terminal"></a>

# 指南：终端

本指南说明交互式终端：如何对话、使用它的按键和命令、回应审批，以及在工作区中工作。不带任务运行 `melo` 时请阅读本指南；命令行选项见[命令行参考](../reference/cli.md)。

<a id="the-interactive-ui"></a>

## 交互式界面

在终端中不带任务运行 `melo`，会打开一个聊天界面，它安静地记录工作过程，并让回答成为屏幕上最醒目的内容：每次思考一行，每次工具调用一行并附上其输出的第一行，还有通知和反思，以及一个说明 melo 正在做什么、已用多久、哪个工具在运行的实时区域。回答以 Markdown 格式输出。输出内容保留在终端的滚动缓冲区中；输入框和状态栏留在底部。开发者模式保留完整的记录（见[一轮的输出](#a-turns-output)）。

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

欢迎信息共三行：版本和模型（并说明如何更换模型）；melo 工作的文件夹；以及一件可以试试的事，后面是命令和退出的按键。只有当模型不是保存的默认模型时，才会另起一行显示模型的服务器。警告紧随其后；`/status` 会检查模型是否有回应，并显示文件夹、会话、审批和上下文用量。开发者模式（`--developer`，或 `[ui] mode = "developer"`）下，欢迎信息是列出所有行的方框。

它需要终端：标准输入和标准输出都必须是终端，且 `TERM` 不能是 `dumb`。否则 `melo` 会向标准错误输出以下内容，并以退出码 2 退出：

```text
melo: the interactive UI needs a terminal (stdin and stdout must be TTYs).
  Run one task:      melo "your task"
  Start the gateway: melo serve
```

<a id="which-model-server"></a>

### 选择模型服务器

按以下顺序使用第一项适用的配置：

1. `melo setup`：运行配置向导，然后使用它保存的后端。默认模式下，它先用首次运行的问题提供在本机找到的模型服务器（`Found a model server at …`、`Use this model server?`）；只有选择 `Use another server`、处于开发者模式，或已保存后端但没有找到服务器时，才出现完整的后端菜单（什么都没保存时，会输出启动服务器的命令）。不带任务时，不能与 `--url` 同用。
2. `--url URL`：使用该服务器，不使用配置文件。模型使用 `--model` 指定的模型，否则使用 `GET URL/models` 列出的唯一对话模型（ID 包含 `embed` 或 `rerank` 的不是对话模型），再否则使用服务器说明已加载的那一个对话模型；有多个且不知道哪个已加载时，退出码为 2，列出这些模型并要求使用 `--model <id>`。任务、`melo worker` 和 `melo ""` 在第一个请求之前以同样的方式选择模型（见[命令行参考](../reference/cli.md#general)）。没有 `/v1` 的 URL 会通过 `URL/v1/models` 获取列表，如果该地址有响应，界面也会使用 `URL/v1`。服务器没有响应模型列表请求或没有列出模型时，模型为 `default`，横幅下方的警告会说明原因。没有配置文件时会以[列表](#questions-are-lists)询问一次 `Save URL · MODEL as the default backend?`：默认选中 `Save it`，按 Enter 保存，与首次运行相同；选择 `Not now` 或按 Esc 则不保存。
3. 已保存的后端；若指定 `--model`，则在本会话中用它替代保存的模型。界面打开前不做检查：显示横幅后，melo 会列出一次服务器模型（`GET /models`），如果服务器没有响应或没有列出该模型，就会警告：`⚠ Cannot reach the model server at URL (connection refused). Start it, then send your message.`
4. 没有已保存的后端：先回答[首次运行](#first-run)的问题，然后打开界面。

欢迎信息的第一行显示软件包版本；在 git 检出中还会显示短提交号（`melo 0.1.0 (baab21e)`），`melo --version` 输出同一行。当模型不是保存的默认模型时，文件夹之后的一行会给出服务器的 URL 及来源（`from --url (not saved)`、`not saved; melo asks again next time`）；如果 melo 从多个模型中选了服务器报告已加载的那个，还会显示 `the loaded one of N; /model lists them`。服务器没有响应时，欢迎信息下方会显示警告。配置文件写 `omlx`，或服务器的模型列表显示它是 oMLX 时，`/status` 把服务器称为 `oMLX`，否则称为 `OpenAI-compatible server`（Ollama、MLC LLM 和 SGLang 保持各自的名称）。开发者模式的方框有一行模型信息，写的就是这个名称；已保存的服务器在方框显示之后才列出模型，所以方框显示的是配置文件所写的名称；本次运行已经列出过该服务器时（首次运行就是如此）除外。

<a id="what-melo-says-about-itself"></a>

### melo 如何介绍自己

每个对话都会先告诉模型它是谁，这样问“你是谁？”时，回答的是 melo，而不是模型原本被训练成的那个助手。这段说明告诉模型：它是 localmelo 的 melo，一个开源智能体，在你的文件夹里用工具工作，改动任何东西之前先询问你；被问到时说出它运行在哪个模型上；不要把自己说成该模型原本的助手，也不要说自己是模型开发方做的；用你的语言回答。这段说明还会像横幅那样写出服务器的名称（`oMLX`、`Ollama`、`OpenAI-compatible server`，或云服务商的名称），并说明它是否在本机（`on this computer`；在另一台机器上的服务器则是 `at 192.168.1.20`）。`/model` 和 `/backend` 会让下一轮改用新的选择，那一轮会说出新的模型。这段说明还会要求模型对问候，或可以直接回答的问题，不使用工具直接回答。Web 控制台和 macOS 应用的做法相同；直接运行（`melo "task"`）、worker 和基准测试不发送这段说明，所以它们的请求和以前一样。

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

- **队列。**一次只能有一条消息等待，显示为 `↳ Queued for the next turn: …`。如果当前轮次以 `completed` 以外的状态结束（停止、失败），消息会返回输入框，不会发送。完全无法运行的轮次所对应的消息（`✗ The turn could not run: …`）也会返回，并排在它前面。
- **停止。**Esc 立即停止当前轮次；该轮以 `cancelled` 结束并输出 `■ Stopped`。回答输出后，melo 最多花 2 秒记录运行步骤（参见[程序性记忆](../concepts/procedures.md)）；此时 Esc 不起作用，下一条消息会等待记录完成。
- **历史记录。**消息和命令保存在 `~/.cache/localmelo/cli_history` 中（权限为 0600；超过 1 MB 时保留最新的 1,000 条），文件内容绝不会保存到其中。无法打开该文件时，会话在内存中保留历史记录并提示你。

<a id="commands"></a>

### 命令

输入 `/` 后，命令及其作用会显示在输入框下方；继续输入字母可缩小范围，删除字母则重新放宽范围（删掉 `/` 后列表关闭），按 Tab 或 Enter 选择。Esc 会关闭列表，直到你下一次编辑；仅移动光标不会打开它。`@` 后的文件名同理。这个列表没有背景色块：名称用你的文字颜色，说明为暗淡色，选中的行和其他列表一样，带 `❯` 并衬在苔绿色色带上，名称与你输入的文字对齐。最多显示八行，更多时右侧出现一条细细的滚动条。命令的参数（`/model NAME`）用同一个列表。常用命令排在最前，也就是菜单的第一屏：`/model`、`/new`、`/status`、`/help`、`/exit`、`/undo`、`/copy` 和 `/retry`。每一行写明命令的作用，需要参数的命令还写明参数（`/attach PATH`）。

| 命令 | 作用 |
|---|---|
| `/help [command]` | 分三组（Conversation、Model、Help）显示常用命令、`!` 一行和查看更多的办法。`/help all`，以及开发者模式下的 `/help`，列出所有命令和按键；`/help NAME` 给出某个命令的详情。比终端宽的一行会在其说明下面续行。 |
| `/attach PATH` | 暂存一个工作区文本文件；路径含空格时请加引号。 |
| `/attachments` | 列出已暂存的文件及其编号。 |
| `/detach N\|all` | 移除指定编号的已暂存文件，或移除全部已暂存文件。 |
| `/new`（也可用 `/clear`） | 开始新聊天：模型忘记当前对话，本聊天的授权结束。开头会画一条 `── New chat ──` 分隔线。 |
| `/sessions [filter]` | 在当前文件夹中选择已保存的聊天；输入文字筛选，Tab 切换到所有文件夹，↑ ↓ 选择，Enter 恢复，Esc 关闭。 |
| `/resume [id\|title]` | 按 ID、ID 前缀（至少四个字符）或完整标题恢复；不带参数时打开选择器。 |
| `/title [text]` | 显示或设置 1–200 个字符的标题。首条消息之前设置的标题会保留到聊天被保存时。 |
| `/undo` | 忘记最近一个已完成轮次，并将其输入的消息放入 ↑ 历史记录。工具更改的文件不会回滚。 |
| `/usage` | 显示每轮记录的步骤、工具调用、token 和耗时，以及总计；包含已撤销的工作。`?` 表示未记录。 |
| `/memory` | 查看模型工作窗口的副本及其计数、长期记忆后端和恢复限制；不调用模型，也不写入记忆。 |
| `/copy [turn]` | 复制已存储回答的 Markdown，而不是终端中换行后的显示内容。不带编号时使用最近已完成且未撤销的轮次。 |
| `/verbose [on\|off]` | 详细显示：40 行思考、工具调用参数、20 行输出、反思摘要及完整理由；不带参数时切换开关。 |
| `/thinking [on\|off]` | 开启或关闭思考摘录（默认关闭，开发者模式除外）；关闭时只保留 `Thought for 1.2s`；不带参数时切换开关。 |
| `/stream [on\|off]` | 开启或关闭后续轮次的实时流式回复；不带参数时切换开关。见[流式回复](#streamed-replies)。 |
| `/expand [turn.call]` | 显示工具调用的完整输出或错误，包括恢复会话后存储的调用：最近一次调用、最近一轮的第 N 次调用，或第 T 轮的第 C 次调用（最多 200 行，之后显示剩余行数）。 |
| `/progress` | 显示当前轮次最新的进度信息：级别、步骤、尝试、阶段和周期、测试、过去的运行、每个具名步骤及其依据的记忆。 |
| `/approvals [ask\|auto\|deny\|forget [tool]]` | 显示审批模式和本会话的授权；切换模式，或忘记某个工具的授权（仅用 `forget` 则忘记全部）。使用 `--yolo` 时说明如何开启询问。 |
| `/status [all]` | 用五行说明 melo 是否就绪：模型及其服务器，后面是 `✓ answers`（或 `✗ does not answer (connection refused). Start it, then /status checks again.`），文件夹，会话，用文字写出的审批行为，以及上下文用了多少（`8% used (2.6k of 32k tokens)`，不知道上限时为 `2.6k tokens used`）。`/status all`，以及开发者模式下的 `/status`，改为打印每一行：模型和服务器（`reachable` 或不可达的原因）、工作区及其边界、带 ID 的会话、审批模式和授权、工具、记忆（含工作窗口及其大小的原因，如 `64 (local server)`）、限制、用量块（cache、context、steps、tasks：一行有数据后才出现；开发者模式显示所有行）、文件和版本。比终端宽的一行会在其文字下面续行。其他任何参数都会显示 `✗ Use /status or /status all.`。 |
| `/mcp` | 显示 MCP 服务器、它们的工具和命令行，以及来源文件（`--mcp` 或 `mcp_config`）；失败后 MCP 关闭的原因；没有文件时如何指定文件。 |
| `/retry` | 重新运行最近一个已完成轮次，即使恢复会话后也保留存储的原始输入文本。只有尚未撤销该轮时才先撤销；之前的轮次和文件仍保持原有更改。 |
| `/model [name\|number]` | 单独使用时打开服务器的对话模型列表（嵌入和重排模型不在其中），当前模型标 `✓`，服务器说明已在内存中的模型标 `(loaded)`。输入即可筛选，↑ ↓ 移动，Enter 切换并将该模型保存为默认值，Esc 保留当前模型。指定服务器列表中的 ID 或编号时（尚未列出则 melo 先列出），从下一条消息起切换到该模型，并将其保存为默认值。没有列表可选时（云端 API），`/model` 会说明这一点，`/model NAME` 仍可切换。 |
| `/backend [url]` | 切换到另一台模型服务器：询问 URL（Enter 保留当前值）和列表中的模型，用 `GET /models` 检查服务器，切换并将两者保存为默认值（服务器以 Ollama 的方式响应时保存为 `ollama`，模型列表把 oMLX 标为所有者时保存为 `omlx`，否则保存为 `vllm`）。 |
| `/exit`（也可用 `/quit`） | 退出；先停止正在运行的轮次。 |

- 执行的命令会像消息一样回显在滚动缓冲区中（`› /model 2`），后面跟着输出。被拒绝的命令不会回显，`/exit` 不输出任何内容。
- 轮次运行期间，`/new`、`/sessions`、`/resume`、`/undo`、`/retry`、`/model` 和 `/backend` 会被拒绝：草稿保留，并显示提示 `/model is not available while a turn runs (esc stops it).`；等该轮结束后再次按 Enter。
- `/model` 只能在服务器列出的模型之间切换。未列出的名称会被拒绝，不会切换或保存任何内容：`✗ URL lists no model named 'NAME'. /model lists them.`；超出列表范围的编号也会被拒绝：`✗ Type a number from 1 to N, or a model id.` 只有没有列表时（服务器没有响应，或使用云端 API），`/model NAME` 才会仍然切换到指定模型；服务器没有响应时会先显示 `⚠ Could not list the models (REASON); switching anyway.`。
- 保存行为：`/model` 保留已配置的后端并更改其模型。`/backend` 在服务器以 Ollama 的方式响应（`GET /api/version`）时将其保存为 `ollama` 后端，模型列表把 oMLX 标为所有者时保存为 `omlx`，否则保存为 `vllm`；它写入 `chat_backend` 和对应表中的 `chat_url`、`chat_model`，并保留其余配置：嵌入、`request_timeout_s` 和 `extra_body`；会话按保存后的配置运行。在 `melo --url URL` 会话中，它们只为当前会话切换，除非 melo 询问了 `Save URL · MODEL as the default backend?`（仅当配置文件尚不存在时询问）且你回答了是。首次运行的问题中选择 `s`（仍然启动）后，`/model` 和 `/backend` 会像首次运行一样保存服务器。
- 模型启动失败时保留当前模型（`✗ Could not start NAME: …`）。新服务器没有响应或没有列出模型时，`/backend` 保留当前服务器：`✗ Could not reach URL: connection refused. Still using URL.`
- 以 `/` 和小写名称开头，后接空格或消息结束的文本是命令；未知名称会显示 `Unknown command /NAME. /help lists the commands.` 并保留草稿。`/etc/hosts is broken` 是一条消息。要发送看起来以命令开头的消息，请在开头加一个空格。

<a id="attach-text-files"></a>

### 附加文本文件

在空闲的交互式对话中，先暂存文件，再发送消息：

```text
/attach "notes/my file.txt"
/attachments
/detach 1
/detach all
```

也可以在消息中输入 `@` 和工作区内的路径前缀，然后从补全菜单中明确选择一个文件。仅仅在正文里输入 `@file` 不会附加文件。已暂存文件行会标明这些文件；发送一条非空消息即可把它们包含进去。轮次运行期间不能执行附加命令，切换会话前必须先移除已暂存的文件。

只接受当前工作区内的普通 UTF-8 文本文件；工作区内的绝对路径也可以。URL、`..` 穿越、符号链接（包括指向工作区内部的链接）、已识别的密钥文件和二进制数据都会被拒绝。清理后的名称不得超过 200 个字符。硬链接可能指向工作区之外的 inode；这不是沙箱。不支持图片附件。上限是 8 个文件、每个 256 KiB、合计 512 KiB，按原始字节计算。显示出的字节数是清理和脱敏后快照的 UTF-8 大小。组合后的消息仍受智能体的提示词上限约束，因此即使文件在这些限制之内，也可能被拒绝。

每个文件只读取一次，存入不可变的快照；之后修改或删除文件不会更新该快照。模型会在一个带分隔、标记为不可信数据的块中收到该快照，而不是作为指令。密钥脱敏是尽力而为，不是保证：附加前请先检查文件。如果脱敏或清理改变了内容或路径，melo 会显示该快照，并在暂存前请求确认。

已提交的快照随会话保存在 `~/.cache/localmelo/cli/sessions.db`（明文，未加密）；`/retry` 使用保存的快照，而不是重新打开文件。持久化并不意味着完整重放：`pairs` 恢复为用户文本和回答提供 32,000 个字符的预算，可能整轮省略附件；`window` 恢复在保留的检查点或尾部不完整包含它时，可能裁剪或省略它。如果模型需要的内容没有被恢复，请重新附加并再次发送。简单的 `melo ""` 循环和直接模式不支持附件。

<a id="run-a-shell-command"></a>

### 运行 shell 命令

在空闲的提示符下，以 `! `（感叹号加空格）开头的消息会把其余部分交给你自己的 shell（`$SHELL`，否则 `/bin/sh`）运行：以你的身份，工作目录为工作区文件夹：

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

- **不是沙箱。** 该命令不是模型工具：不经过审批、检查器或工作区策略，拥有你账户的全部权限，也可以离开工作区。melo 只把终端交给它。
- **命令运行期间终端归它所有：**读取输入、绘制屏幕或响应窗口大小变化的程序都能正常工作。Ctrl+C 和 Esc 交给命令，绝不会取消代理；命令结束后，Ctrl+C、草稿和队列的规则不变。块的末尾显示退出码（`exit 3`）或信号（`killed by SIGTERM`）和耗时。无论命令如何结束，光标、颜色、终端模式和提示符都只恢复一次。
- **输出只留在终端。**输出不会进入模型、对话或事件，命令也不会存入输入历史。`/attach !` 会把最近一次输出的脱敏快照（最后 256 KiB，不含终端从未显示的文本：隐藏文本和清屏之前的内容；截断后的开头不完整行会被丢弃）附加到你的下一条消息。它总会先打印将发送的原文，并以[列表](#questions-are-lists)询问 `Attach this output?`（有内容被脱敏时是 `Attach this redacted snapshot?`），列表默认停在 `Don't attach it`，即使没有脱敏；你不选 `Attach it`，就不会附加任何内容（Esc 也不附加）。附加后，快照会像 [`/attach PATH`](#attach-text-files) 的文件一样未加密地随会话保存。
- **忙时拒绝。**在轮次、审批、会话切换或另一条命令进行期间，`! ...` 会被拒绝（`! runs only when melo is idle`），草稿保留。
- 单独输入 `!` 会显示上述帮助。要发送以 `! ` 开头的消息，写成 `\! text`：反斜杠会被去掉。
- melo 退出时命令的进程组收到 SIGTERM，终端关闭时收到 SIGHUP，2 秒后收到 SIGKILL。只会向该进程组发信号；自行脱离该进程组的程序（`setsid`、`nohup`）会继续运行。
- 直接模式（`melo "task"`）和纯文本循环不支持此功能。

<a id="inspect-undo-or-copy-a-conversation"></a>

### 查看、撤销或复制对话

`/title` 可以在发送任何消息前为聊天命名；它本身不会创建已保存的会话。第一轮创建会话时才会应用所选标题。`/new` 或成功恢复会话会清除未保存的标题。轮次运行期间可以更改标题，但切换会话期间不能更改。

`/undo` 从模型恢复后的工作窗口中移除最近一个已完成轮次，并将原始输入的消息保留在输入历史记录（↑）中。它不会撤销 shell 命令、Python 代码或文件编辑。在另一个窗口中，运行中的轮次会阻止撤销；命令保留在草稿中。撤销提交后，Esc 无法回滚：操作会完成并输出 `Undid …` 行，忙碌提示会要求等待，而不会声称 Esc 能取消操作。预览只使用输入的第一行，按终端单元格宽度截断，不会再添加一个句点。

`/retry` 重新运行最近一个已完成轮次。如果 `/undo` 已经移除该轮，重试会保留前一个仍有效的轮次；否则先撤销目标轮次。它发送存储的完整消息，同时保留原始输入文本。

`/usage` 包含已撤销的记录，因为这些工作确实执行过。运行中的轮次或缺失的计数显示为 `?`，而非零；不完整的总计以 `+?` 结尾。终端宽度不足 72 列时，用带标签的换行记录显示用量，而不是截断表格中的数字单元格。`/memory` 读取窗口副本，因此它和 `/usage` 都可在轮次运行期间使用。记忆记录显示用户、助手和工具消息，不显示隐藏的思考；页脚说明恢复时使用的是配对，还是检查点加后续配对。它不会读取所有持久化记忆，也不会让模型对其进行总结。

`/copy` 读取移除思考后的**已存储脱敏回答**，因此 Markdown 和代码不是从终端换行后的显示内容中复制的；代码行首缩进和末尾换行都会保留。显式指定轮次编号时仅读取该轮；默认选择则按有限大小的分页向前查找。显式编号可以选中已撤销的回答。空回答、运行中的轮次和仅含状态的标签不能复制。在非 SSH 的 macOS 环境中，它使用 `/usr/bin/pbcopy`，超时为两秒；通过 SSH、在其他平台或该工具失败时，它发送 base64 编码的 OSC 52 请求。后者会显示 `Sent the answer …`，而不会声称复制成功：终端必须支持并允许剪贴板请求。只有显式使用 `/copy` 命令才会执行剪贴板操作。

<a id="a-turns-output"></a>

### 一轮的输出

- **审批显示更改内容。**对已有文件的 `file_write` 会显示差异（删除的行标红、添加的行标绿，并显示行数）；写入内容过大或不是文本时会说明原因。`/approvals` 可以在 `ask`、`auto` 和 `deny` 之间切换，并管理本会话的授权。
- **实时区域。**第一行显示 melo 正在做什么（`Starting`、`Thinking`、`Replying`、`Running`、`Waiting for your approval`、`Retrying in 4s`、`Reflecting`、`Wrapping up`、`Stopping`），接着显示时间；后面是输出 token 数和 `esc to stop`（`Thinking… · 14s · ↓ 180 tokens · esc to stop`）。开发者模式会在时间之前加上运行信息：步骤（`step 4`，不与 `--steps` 对比）、从第二次开始显示的尝试次数、阶段及其周期和已见到的测试。行宽不足时先省略提示，再省略 token 数，开发者模式下的信息用 `…` 截断。模型发来但要等完整后才显示的文本不会多出一行：第一行已经写着 `Thinking`。第二行显示正在运行的工具、其自身计时，以及 `shell_exec` 和 `python_exec` 的时限：`└ shell_exec  pytest -q tests · 0:25 of 10:00`。对于 melo 以前运行过的任务，第三行显示进度行会显示的过去运行及其步骤：`last time 6 steps, 2m10s: Run ls ✓ · Read app.py ✓`。它在会话的第一轮显示，开发者模式下每一轮都显示。
- **思考。**显示 `✻ Thought for 1.2s`，只有一行。以没有任何工具调用的回复结束的一轮显示 `✻ Thought`，因为回答下方的页脚已经给出了那段时间。`/thinking on` 和开发者模式会再显示思考的前 3 行（`/verbose` 为 40 行），按宽度换行。中间消息在 `●` 后以 Markdown 输出；最后一条会先暂存，到下一步才输出；如果直到结束时才输出，则只在它不同于最终回答时显示。
- **工具调用**在完成时各输出一个区块：`✓` 表示已运行，`✗` 表示失败，`⊘` 表示被拦截或拒绝，`◌` 表示已停止或未运行；接着是工具的类别（`→` 读取，`±` 写入，`$` 运行，`@` 网络）和名称；还显示耗时；命令运行后以非 0 退出码结束时显示 `exit N`，并以红色 `✗` 标记，尽管调用确实运行了（`grep` 以 1 退出并不代表失败）；审批结果；以及输出的第一行，同一行末尾显示 `+N lines (/expand T.C)`（若只剩一行则直接显示该行，不显示计数）。标记带结果的颜色，类别符号和名称带类别的颜色（读取是暗淡色，所以读取会淡出），耗时和标签是暗淡色；见[终端颜色](#terminal-colours)。失败的调用，或运行后以非 0 退出码结束的命令，最多显示 3 行，第一行为红色；开发者模式对每次调用显示 3 行，`/verbose` 显示 20 行并附上调用参数。图片则显示为 `▪ image/png · 312 KB · not shown`。输出中的转义序列会被移除，因此工具无法移动光标、设置标题或写入剪贴板。
- **通知和反思。**`↻` 标记重试：每轮一行，服务器无法连接时这一行会保留它的 URL（`↻ Could not reach URL; trying again in 1 s`），延迟在实时区域倒计时；`·` 表示空回复，或只含未完成思考的回复（`· The model's reply ended inside its thinking; asking it for its answer.`），这种回复显示为思考（`✻ Thought`），绝不作为回答；`⚠` 表示卡住或预算用尽通知。反思显示为 `✻ Reflected after attempt 1: continue · Read the test before editing. (2.3s)`；`/verbose` 还会显示其摘要和完整理由，以及在询问了两次时显示 `asked twice`；无法读取的反思会说明运行是否继续。
- **回答**在 `●` 后以 Markdown 输出。未完成的运行会先输出尽力生成的结果，再用平实的话说明发生了什么、该怎么做：`✗ Could not reach the model server at URL.`，其下是 `Start it, then /retry.`。第一行是共享表中该状态的标题（`Ran out of steps (30).`、`Stopped: the same calls kept repeating.`），并附上有帮助的原因（URL、模型、失败检查的错误）；第二行是操作建议，使用默认颜色。页脚此时显示 `✻ Stopped after 14s` 而不是 `Worked for`，且不显示为零的工具调用次数。直接模式仍在标准错误上输出 `melo: run ended with status …` 这一行。
- **更改的文件。**用 `file_write` 写过文件的一轮，会在页脚下方多出一行安静的提示，让你不必逐个查看工具块，就知道它动过哪些文件：

  ```text
  ✓ ± file_write  hello.py (15 chars) · 0.0s · allowed
    └ new file, 1 line
  ● I saved hello.py. Run it with python hello.py.
  ✻ Worked for 4.7s · 1 tool call · ↓ 50 tokens
    Changed + hello.py (+1)
  ```

  多个文件时显示为 `Changed + hello.py (+1), ~ README.md (+3 −1)`：`+` 表示新文件，`~` 表示修改过的文件；路径相对于 melo 工作的文件夹，每个文件后面是这一轮在其中增加和删除的行数。最多显示三个名字，其后是 `and 2 more`；终端较窄时显示得更少，并且不会换行。同一个文件写了两次只算一次，比较的是这一轮开始时的文件和结束时的文件；写回原样的文件不会列出。没有可统计或可比较的内容时，名字不带行数：新建的空文件、melo 无法比较的文件（太大或不是文本），以及 `--yolo` 轮次中的所有文件（没有人事先读取过旧文件）；后两种也没有 `+` 或 `~`，因为 melo 不知道文件是否是新建的。这一行来自这一轮自己的 `file_write` 调用：被拒绝的写入（审批被拒绝，或路径在文件夹之外）和失败的写入不会列出，shell 命令或脚本更改的文件也不会列出，因为 melo 从不扫描文件夹。这一轮没有更改任何文件，或者你停止了它（被停止的轮次没有页脚）时，不会输出这一行。开发者模式显示同样的这一行。

<a id="the-status-bar"></a>

### 状态栏

输入框下方的状态栏回答两个问题："这次用了多少"和"旁边在跑什么"。模型和文件夹在欢迎信息和 `/status` 里，不在这里。各区位置固定，区与区之间有一条竖线：用量、聊天之外的工作、运行（没有轮次在运行时是聊天的标题）。使用本地服务器时：

```text
  ↑ 3.0k ↓ 15 · ctx 2.6k/32k ▮▯▯▯▯ · cache 61% (avg 72%) · idle 2m │ Fix the parser
```

- `↑ 3.0k ↓ 15`：本次会话发送和接收的词元数。
- `ctx 2.6k/32k ▮▯▯▯▯`：最近一次测得的提示词词元数与输入上限，后面是按五分之一刻度的小条。达到 70% 变为琥珀色，达到 90% 变为红色。不知道上限时只显示词元数（`ctx 2.6k`），没有小条。
- `cache 61% (avg 72%)`：服务器报告的、上一次请求提示词中已缓存的比例；有两次请求报告过之后，后面加上本次会话的平均值。达到 50% 为青绿色；第一次请求之后低于 20% 为琥珀色（第一次请求总是冷的）。服务器没有报告已缓存词元时不显示缓存数字，绝不显示 0%。
- `idle 2m`：距上一次请求的时间，满一分钟后显示。本地服务器按内存而不是按时间淘汰前缀缓存，所以没有倒计时。

使用云端 API 时，用量区改为显示本次会话按你配置的价格算出的费用（见 [`[pricing]`](../reference/configuration.md#pricing)），以及上一次响应报告的最紧的速率限制：

```text
  $0.13 · ctx 100k · limit 42% ▮▮▯▯▯ ↻3m │ Fix the parser
```

`limit 42%` 是该时间窗口已用的比例，来自响应的 `x-ratelimit-*` 或 `anthropic-ratelimit-*` 头（不额外发请求），`↻3m` 是距重置的时间。没有配置价格的模型显示词元数（`↑ 100k ↓ 5`），不发送限制头的服务器不显示限制。melo 不内置任何价格。

- **运行区。**轮次运行期间，运行区简要给出步骤和列出步骤的按键：

  ```text
    ↑ 1.0k ↓ 5 · ctx 1.0k · cache 83% │ 2/~4 · ctrl+o steps
  ```

  `2/~4` 表示大约 4 步中的第 2 步：这个任务以前做过，当时用了 4 步，所以总数是估计值，用 `~` 标出。melo 没做过的任务显示 `step 2`，运行超过估计值时也是如此。（不带 `~` 的 `2/8` 留给自带计划的轮次；目前还没有。）Ctrl+O 把步骤列成列表：每次工具调用的工具、目标、状态和用时，然后是该任务上一次运行在此之后还有的步骤（`◌ Run the tests · to come`）；Esc 关闭。轮次结束后，同一个键列出上一轮的步骤。
- **聊天之外的工作。**网关（`melo serve`）有未完成的任务时，会出现一个区，按状态计数，并显示未暂停任务中的最高优先级和列出它们的按键：

  ```text
    … │ 2 running · 1 queued · P1 · ctrl+t tasks │ Fix the parser
  ```

  `running` 为苔绿色，`held`（等你处理的任务，例如等待审批）为琥珀色；`queued` 和 `paused` 为暗淡色。Ctrl+T 列出这些任务（`P2  running  Bench the cache`）；在某个任务上按 Enter 会列出它当前状态允许的操作：Pause（已暂停的任务为 Resume）、Raise to 和 Lower to 相邻的优先级（P0 最高；降低到 P4 为止，因为 P5 就是暂停），以及 Stop，它会先问 `Stop it` 还是 `Keep it`。每个操作都与 `melo task pause|resume|level|stop` 完全一样地发给网关，并用一条提示说明结果（`Paused: Bench the cache`）或网关拒绝的原因。终端打开期间，melo 每 5 秒向网关询问一次任务，且只在存在网关令牌（`~/.cache/localmelo/gateway/auth`）时询问；没有网关时不显示这一区，Ctrl+T 会说明如何添加任务。

终端较窄时，先去掉标题，然后是工作区的细节（先优先级，再按键），然后是用量区的细节（从右到左：空闲时间、平均值、小条），然后是运行中轮次的按键；之后数字从右到左去掉，运行中轮次的数字保留。状态栏不会换行。审批模式只在不是 `ask` 时显示在右端（`auto`、`deny`，或 `⚠ YOLO: no approvals` 警告）；开发者模式始终显示。补全菜单或列表打开时，状态栏上方会空出一行，与列表分开。

这些数字来自 melo 已收到的事件，不会询问模型服务器。状态栏在事件到达或按键时重绘，显示时钟（空闲时间、限制的重置时间）时每秒重绘一次；空闲时没有其他重绘。设置 `NO_COLOR` 或 `LOCALMELO_ASCII=1` 时各区仍然分开（`│` 或 `|`），ASCII 下写作 `in 3.0k out 15`、`##---` 和 `reset 3m`。`/status` 会完整打印这些数字。`[cli] status_line = "off"` 只保留运行区、标题和上一次请求的上下文（`ctx 1.0k`）。没有命令行参数。

<a id="streamed-replies"></a>

### 流式回复

交互式界面会在模型写出回复的同时把它流式显示出来，而且这是默认行为：它向提供方请求流式规划回复，并只在实时区域中绘制发布过滤器已经脱敏的文本，按应用的刷新频率重绘；它从不把原始增量直接写到终端。`/stream off` 为之后的轮次关闭流式，`melo -X cli.stream=false` 只关闭这一次运行，`melo config set cli.stream false` 则永久关闭。

不支持流式输出的模型服务器无需任何设置。当它用一个普通的 JSON 响应体回答流式请求，或在发出任何内容之前就关闭了流时，melo 会使用这个回复（或不带流式、把同一个问题再发送一次），并且在 melo 退出之前不再请求该服务器流式输出。此时回复在完成后才会显示，与 `/stream off` 时相同，melo 会只说明一次：`This server sends replies whole; melo shows them when they are complete.` 如果流在已经收到部分回复之后中断，仍然是错误：`Could not get an answer from the model server: Streaming response ended before completion`。

回复能否边写边显示，取决于模型如何标记推理，这由 `config.toml` 中按模型设置的提示声明：

```toml
[cli.reasoning_tags]
"Qwen3-8B" = "closing_only"
```

提示是 `explicit`（开头的 `<think>` 开始推理）、`closing_only`（`</think>` 之前都是推理）或 `none`（标签是普通文本）。没有提示的模型使用保守规则：内容一直保留到回复结束，因此实时区域只显示那一行转圈的 `Thinking…`，而不显示文本。在开发者模式下，它的第一条完整回复之后，melo 会说明一次：`This model's replies appear when complete until its reasoning format is set (see Streamed replies).` 新用户看不到这条说明。把推理放在单独字段里发送的服务器无需提示就能流式显示，该字段绝不会作为正文绘制；`/thinking on` 会显示它的摘录。同一个提示也会切分完整回复，因此流式显示和存储的回答不会不一致。这个提示还说明回复是否停在了思考中：声明为 `closing_only` 时，没有 `</think>` 的回复就是未完成的思考。这种回复被保留的文本绝不会释放，它的临时文本会被丢弃，它显示为思考；melo 再问模型一次，第二个回复仍没有回答时，这一轮以 `melo did not finish an answer.` 和 `/retry, or try another model.` 结束。

流式文本是临时的。如果回复变成工具调用、重试、失败的运行或停止，melo 会关闭实时文本并只打印一次，开发者模式下标为 `↳ partial reply (tool call)`、`(retrying)`、`(not the final answer)` 或 `(stopped)`；完成的最终回答会在 `●` 后只打印一次。不支持流式的提供方会发送完整回复，照常打印。`/stream on|off` 只切换后续轮次，不写入配置，`/new` 会把它恢复为 `[cli] stream`。没有交互式界面的运行（直接模式、worker、网关）从不流式，因此输出和请求体保持不变。无效的 `[cli]` 值不会让运行失败：交互式界面会发出警告并使用默认值，直接模式和网关根本不读取它。

<a id="approvals"></a>

### 审批

`shell_exec`、`python_exec` 和 `file_write` 只有在你回答后才会运行，标记为需要审批的 MCP 工具也是如此。`file_read` 和 `list_files` 只读取，因此从不询问：melo 可以查看你的文件夹来回答 “What is this project?”，不会出现审批面板。（交互式界面加载 `--mcp` 指定的 MCP 服务器；未指定时加载顶层 `mcp_config` 键中的服务器。服务器启动失败会在本会话中关闭 MCP，`/mcp` 会说明原因。）审批面板取代输入框，草稿会保留：

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

安装软件、下载文件、更改文件夹以外的文件或做其他[高影响](tools-mcp.md#high-impact-calls)事情的调用会得到另一种面板：标题用一个问题说出调用要做什么，代码上方列出原因，没有会话授权的选项，并默认选中 No，因此只按 Enter 不会允许它：

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

- `1` 允许本次调用，`2` 在本会话剩余时间内允许该工具的常规调用（安装、下载、更改文件夹以外的文件等[高影响调用](tools-mcp.md#high-impact-calls)仍会询问），`3` 打开理由输入框（最多 500 个字符），模型会读取该理由；Enter 发送。高影响面板没有会话授权的选项，所以那里的 `2` 是 No。也可以用 ↑ ↓ 和 Enter 选择。字母不能作为回答；面板打开后的 0.4 秒内只有 Esc 和 Ctrl+C 有效，避免你正在输入的消息意外回答审批。
- 标题显示工具名和文件夹。开发者模式会在工具名后加上其权限级别，对于 `file_write` 则加在路径后（`shell_exec · exec · in ~/demo-project`；MCP 工具可为 `write`、`exec` 或 `network`，前提是界面能加载该工具）。只读 MCP 工具仅在服务器的 `require_approval` 强制询问时显示 `read`；参见[权限级别](tools-mcp.md#permission-tiers)。
- 在选项处按 Esc 或 Ctrl+C 会停止当前轮次，2 秒内再次按 Ctrl+C 则退出；在理由输入框中按 Esc 返回选项。
- 命令、代码或文件的新内容会按实际运行时的原样显示，长行换行，绝不截断：控制字符和会重新排列或隐藏文本的字符以转义形式显示（`\r`、`\x1b`、`\u{202e}`），并附有警告行，防止内容被隐藏。
- 面板最多显示正文的 8 行，在较短的终端上更少，之后显示 `… +N lines (v views all)`；按 `v` 可在面板上方输出完整正文。
- `--yolo` 不经询问就运行所有工具（意味着没有策略限制，而不是只读），状态栏会显示 `⚠ YOLO: no approvals`。

<a id="the-workspace"></a>

### 工作区

melo 在当前文件夹或 `--workspace DIR` 指定的文件夹中工作。文件工具（`file_read`、`file_write`、`list_files`）被限制在这个文件夹内。命令和 Python 可以访问其他文件，因此 melo 会先询问。工作区为你的主目录或 `/` 时，横幅会发出警告。界面中 `shell_exec` 和 `python_exec` 在 10 分钟后停止（单次任务分别为 30 秒和 60 秒）。

<a id="chinese-and-other-wide-characters"></a>

### 中文及其他宽字符

输入框按字符编辑，按终端单元格放置文本：中文和其他宽字符的输入、删除和换行都保持字符完整，删除换行后的消息也不会留下空行。已知限制：

- 由多个代码点组成的字符（用组合标记输入的重音、👩‍💻 之类的 emoji 序列）需要每个代码点按一次 Backspace。
- Ctrl+W 向左删除到上一个空格，因此对于没有空格的中文文本，会删除整段连续文本。
- 输入法（如 macOS 拼音）仍在组合的文本由终端绘制，而不是由 melo 绘制。

<a id="terminal-colours"></a>

### 终端颜色

颜色把一轮的输出分门别类：回答、失败和改动过的文件一眼可见，读取则淡出。每种颜色都有一个符号或固定位置表达同样的意思，所以 `NO_COLOR` 和 `LOCALMELO_ASCII=1` 不会丢失任何信息。

| 内容 | 符号（ASCII） | 颜色 |
|---|---|---|
| 你的消息 | `›`（`>`） | 苔绿色标记，整行衬在浮起的色带上 |
| 回答及其标题和列表符号 | `●`（`*`） | Melo 的橙色（浅色终端上标题为藏青色），唯一的橙色 |
| 不是回答的消息；思考 | `●`（`*`）；`✻`（`*`） | 暗淡色 |
| 读取的工具 | `→`（`>`） | 暗淡色，所以读取会淡出 |
| 写入的工具 | `±`（`~`） | 紫色 |
| 运行命令或代码的工具 | `$` | 蓝色 |
| 访问网络的工具 | `@` | 靛蓝色 |
| 已运行；失败或以非 0 退出码结束；被拒绝；未运行 | `✓`（`+`）；`✗`（`x`）；`⊘`（`!`）；`◌`（`-`） | 青绿色；红色，附 `exit N`；琥珀色；整行暗淡 |
| 改动的文件：新建；修改 | `+`；`~` | 青绿色；琥珀色 |
| 耗时、计数、提示 | `·`（`-`） | 暗淡色 |

行内代码衬在浮起的小块上，代码块衬在浮起的面板上，关键字为苔绿色、字符串为金色、名称为藏青色。

面板、列表和各种栏用同样的颜色：

| 内容 | 样子 |
|---|---|
| 审批面板 | 琥珀色边框，标题为粗体；选中的答案 `❯ 1. Yes` 衬在苔绿色色带上；差异中新增的行衬在青绿色色带上，删除的行衬在红色色带上 |
| 所有列表（`/model`、`/sessions`、设置问题、附加问题、Ctrl+O 的步骤） | 标题为粗体，选中的行衬在苔绿色色带上并只有一个 `❯`，当前模型的 `✓` 为青绿色 |
| 输入框下方的列表（`/`、`@`、参数） | 没有背景；名称用你的文字颜色，说明为暗淡色，选中的行衬在苔绿色色带上并带一个 `❯`；只有行数多于显示的八行时，才在右侧出现一条细细的、规则线颜色的滚动条 |
| 按键说明行（`↑↓ select · enter confirm · esc stop the turn`） | 按键用你的文字颜色，说明文字为暗淡色（16 色时为普通文字，绝不使用 dim） |
| 实时行 | 转圈符号为苔绿色，melo 正在做的事用你的文字颜色，时间和词元数为暗淡色；等待审批时为琥珀色 `⚠` 和粗体文字 |
| 状态栏 | 暗淡色；费用为你的文字颜色，良好的缓存为青绿色，渐满的上下文或限制先琥珀色后红色，运行中的工作为苔绿色，等待中的工作为琥珀色，各区之间有竖线 |
| 建议中的命令（`Start it, then /retry.`） | 苔绿色粗体 |

不知道确切背景时色带去掉，选中的行为苔绿色粗体；设置 `NO_COLOR` 时为反色。

在 256 色或更多、Unicode、80 列或更宽的终端中，欢迎信息的几行旁边会有 Melo 的头像：五行半格方块，画出两只竖起的耳朵、眼睛、鼻子和奶油色的口鼻部，`melo` 为橙色（README 的截图里能看到）。头像由宠物的脸部图片（`macos/art/akita/face/idle@2x.png`）计算得出，不是手绘的；图片改变时，`scripts/make_terminal_mark.py` 会重新生成它。不足 80 列、16 色、设置 `NO_COLOR` 或 `LOCALMELO_ASCII=1` 时，欢迎信息只有那三行：行数更少的半格方块画不出耳朵。

melo 如何选择：

- **颜色数量。**`NO_COLOR`（任何非空值）或 `TERM=dumb`：不用颜色，只用粗体、暗淡和下划线，所有符号保留。否则依次看 `[cli] colors` 设置、`COLORTERM=truecolor`（或 `24bit`）表示 24 位色、以 `256color` 结尾的 `TERM` 表示 256 色，其余情况使用终端主题的 16 种命名颜色。
- **背景。**先看 `[cli] background` 设置，再看终端自己的回答：交互式界面启动时会向终端询问背景色和文字颜色（OSC 11 和 OSC 10），最多等待 100 毫秒（直接模式、`--json` 或没有终端时从不询问），最后看 `COLORFGBG`。知道确切颜色时，melo 绘制色带、小块和面板，且只在你的文字在其上保持 4.5:1 对比度时绘制，并调整画在其上的颜色直到清晰可读。只知道深色或浅色时，保留颜色、去掉衬底：行内代码为粗体，代码块左侧有 `│` 竖线。什么都不知道时，melo 使用 16 种命名颜色。
- **颜色看起来不对时。**有些终端，以及 tmux、screen 或 ssh，什么都不回答。告诉 melo 你的背景：`-X cli.background=light`（或 `dark`）只用于本次运行，`melo config set cli.background light` 则保存下来。`-X cli.colors=16`、`256` 或 `truecolor` 设置颜色数量。

<a id="terminal-settings"></a>

### 终端设置

- 颜色：见[终端颜色](#terminal-colours)；将 `NO_COLOR` 设为非空值可关闭颜色。你必须阅读或据以操作的句子（警告、审批的警告和按键、输入框下方的提示）从不使用暗淡色；暗淡色只用于耗时、计数和 `(/expand T.C)`。
- 在编码不是 UTF-8 的终端中，或设置 `LOCALMELO_ASCII=1` 时，melo 使用 ASCII 字符代替符号（用 `>` 代替 `›`，用 `...` 代替 `…`）。
- 打开审批面板以及持续 30 秒或更长的轮次结束时，melo 会鸣响终端提示音；`LOCALMELO_BELL=0` 可关闭提示音。

<a id="sessions-and-files"></a>

### 会话和文件

直接运行 `melo` 会开始新聊天；只有第一条消息才会保存它。状态栏在聊天有标题之前显示 `New chat`。使用 `melo -c`（`--continue`）恢复当前工作区中最近一个有已完成轮次的聊天，或使用 `melo -r`（`--resume`）选择聊天。没有可选聊天时，两者都会提示并开始新聊天；在启动选择器中按 Esc 也会开始新聊天，不会创建空记录。`melo -r REF` 根据 ID、至少四个字符的前缀或不区分大小写的完整标题查找；有多个匹配项，或回退到标题／消息搜索时，会打开选择器。未知引用以退出码 2 退出。启动选择（包括 `-r` 选择器）发生在横幅之前：回顾会点出选中的聊天（`◆ Resumed "TITLE"`，开发者模式的方框把它标为 `"TITLE" (resumed)`）；按 Esc 或列表为空时显示新聊天横幅。`-c` 和 `-r` 不能同时使用，也不能与任务或 `melo serve` 同用。

选择器最多列出 50 个具有已完成且未撤销轮次的聊天，最新的在前。它的标题行说明各个按键（`Resume a chat · ~/demo-project · tab: this folder or all · esc close`）；文件夹路径太长时从中间截断，按键提示始终保留。筛选在 150 毫秒后更新；Tab 在当前文件夹和所有文件夹之间切换。Enter 恢复，Esc 或 Ctrl+D 关闭选择器，普通草稿保持不变。`/sessions FILTER` 和 `/resume REF` 使用同一选择器和查找方式。网页及调度器会话不会出现在选项中。网页控制台的 Workers 视图会在 **Terminal chats** 下以只读方式显示你的这些聊天，读取时会对其脱敏（保存的副本不会被脱敏）；控制台不能修改它们，也不能向它们发送消息。恢复会话会保留当前模型和工作区，将其记录到会话中，并清除审批授权；`/verbose`、`/thinking` 和 `/stream` 偏好保持不变。`/new` 会重置这些显示偏好（`/stream` 恢复为 `[cli] stream`）。如果聊天之前在另一个文件夹中工作过，会显示提示。

横幅之后的回顾以纯文本显示存储的脱敏副本中最近三个符合条件的轮次：你的消息在 `›` 之后，每个回答只在开头有一个 `●`，去掉 Markdown 标记和空行，最多四行，标记下方的行缩进对齐。标题下方的一行说明有几个轮次回到了模型的记忆中（`1 turn is back in the model's memory.`）；开发者模式还会说明它们的工具调用怎么样了。`[sessions] resume_mode = "pairs"`（默认）恢复最近的用户消息／回答配对，不包含工具步骤；`"window"` 恢复已保存的工作记忆检查点及之后的配对，没有可用检查点时退回配对模式。在 `config.toml` 中设置此项；无效值会在界面中产生警告并使用配对模式，但不会在直接模式输出中增加警告。回顾中的计数反映淘汰后的恢复窗口，而不是存储的轮次总数。

两个终端可以轮流使用同一个 CLI 会话。发送前，melo 会重新加载另一个窗口的更改；会话忙碌或已删除时会保留草稿。退出时，有轮次的聊天会说明如何回到它：当它是这个文件夹中最新的聊天，且 melo 就在该文件夹中运行时，显示 `◆ Saved "What is this project?". Continue it with: melo -c`；否则显示 `Resume it with: melo -r ID`。

会话保存在 `~/.cache/localmelo/cli/sessions.db` 中：每一轮都包含你的消息、回答以及每次工具调用和输出。`~/.cache/localmelo/cli_history` 保存你的消息。两者都是明文（权限为 0600），未加密。界面将日志写入 `~/.cache/localmelo/logs/cli.log`（文件夹权限 0700、文件权限 0600、1 MB、两个备份），绝不会在屏幕上输出 traceback；出现问题时显示一条暗色提示，指出日志位置。配置文件不是有效的 TOML，或无法创建日志文件夹时，`melo` 会输出一行 `melo: …` 并以退出码 1 停止。程序性记忆与单次任务一样，遵循[记忆持久化](memory.md#keep-memory-between-runs)。

<a id="first-run"></a>

### 首次运行

在没有配置文件的终端中，`melo` 会提供找到的模型服务器，询问模型，保存它们，然后打开界面。服务器列出了嵌入模型时，还会询问一次是否使用它。之后运行会直接进入界面。

```text
$ melo
melo uses a model server on this computer (oMLX, Ollama, LM Studio…).
Found a model server at http://127.0.0.1:8000/v1 (1 model).
✓ http://127.0.0.1:8000/v1 lists 1 model; using Qwen3-8B-4bit.
Long-term memory is off; melo setup turns it on.
Saved to ~/.cache/localmelo/config.toml (melo setup changes it).
```

在第二行和第三行之间，melo 会以列表询问 `Use this model server?`（见[问题都是列表](#questions-are-lists)）；回答后列表会被擦除。如果服务器也列出了嵌入模型，还会多一个列表询问 `Use nomic-embed-text for long-term memory?`（`Use it`、`Leave it off`）。

<a id="questions-are-lists"></a>

#### 问题都是列表

在终端里（标准输入和输出都是终端，且 `TERM` 不是 `dumb`），每个答案固定的问题都是一个列表，与工具审批和 `/model` 一样：

```text
 Use this model server?
 ❯ 1. Use it
   2. Use another server
 ↑↓ select · enter confirm · esc cancel
```

↑ ↓ 移动，Enter 确认选中的行，数字键（1 到 9）直接选择对应的行。安全或推荐的答案默认选中，Esc 给出按键行末尾写明的答案（`esc cancel` 结束设置，与 Ctrl+C 相同；`esc not now`、`esc leave it off` 和 `esc don't attach it` 表示拒绝）。列表没有筛选框：列表很短，在其中输入字母不起作用。Ctrl+C 输出 `Setup cancelled.` 并以 130 退出，Ctrl+D 以 0 退出，与文字问题相同。这些问题是：`Use this model server?`；`Use <model> for long-term memory?`；URL 无法使用后的 `What now?`（`Try again`、`Start without saving`、`Quit`；Esc 退出）；`No model server found. What now?`（`Check again`、`Enter an address`，之后询问 `Server URL […]:`，以及 `Quit`；Esc 退出）；`Save URL · MODEL as the default backend?`；向导的后端列表、`Save these settings?`（`Save and continue`、`Cancel setup`）和 `The check failed. What now?`（`Try again`、`Save anyway`、`Cancel setup`，默认选中最后一项，与以前的 `[N]` 一样）；以及界面中的 `Attach this output?`。文字问题（`Server URL`、`Model name`、`Base URL`）仍是文字。没有终端时（答案通过管道输入、脚本、测试）不画列表，每个问题照旧读取键入的答案（`Use it? [Y/n]:`、`Enter tries again · s starts without saving · q quits:`、`[r]etry, [s]ave anyway, [N]o [N]:`、后端用编号、默认值按 Enter）。直接运行（`melo "task"`、`--output-format json`）从不提问。

- melo 在本机通过 `GET /v1/models` 检查 `127.0.0.1` 的 8000（vLLM、oMLX）、8400（MLC）、11434（Ollama）和 30000（SGLang）端口，查找服务器，并提供第一个列出模型的服务器：对 `Use this model server?` 按 Enter 即选中 `Use it`；选择 `Use another server` 会询问 `Server URL […]:`（预填该 URL），可改填另一个。探测时请求模型列表；有多个对话模型时，还会请求 `GET /v1/models/status`（oMLX）和 `GET /api/ps`（Ollama），了解哪些模型已加载；保存选择时会请求一次 `GET /api/version`，以识别 Ollama 服务器。不发送对话请求，因此服务器不会为此加载模型。列出对话模型的服务器优先；只列出嵌入或重排模型的服务器其次。对于响应但没有返回列表的服务器，melo 也会指出该服务器并提供选择，例如 `A model server at URL answered, but HTTP 401: …`。都没有响应时，会输出启动服务器的命令（Mac 上是 oMLX，其他系统是 Ollama，与[模型服务器指南](model-servers.md)中的一致）并等待：

  ```text
  No model server is running yet. To start one with oMLX:

    brew tap jundot/omlx https://github.com/jundot/omlx
    brew install jundot/omlx/omlx
    omlx serve --model-dir ~/models --port 8000

  Put a model in its own folder of ~/models first. The guide shows how:
  https://github.com/localmelo/localmelo/blob/main/docs/guides/model-servers.md
  ```

  随后出现列表 `No model server found. What now?`。`Check again`（默认选中）会再次检查同样的端口，有服务器响应时按上面的方式继续（仍然没有时输出 `✗ No model server found yet (looked at ports …).`）；`Enter an address` 询问 `Server URL [http://127.0.0.1:8000/v1]:`，并像 URL 一样检查你输入的内容（见下）；`Quit`（或 Esc）以退出码 0 退出。答案通过管道输入时，这一行是 `Press Enter to check again, type an address, or q to quit:`。`LOCALMELO_DETECT_URLS` 用逗号分隔的列表替换这四个 URL，按 Enter 时检查的就是这些 URL；空值会关闭搜索，此时 melo 询问 `Server URL [http://127.0.0.1:8000/v1]:`。
- 列出多个模型的服务器会以列表供你选择。在终端里（标准输入和输出都是终端，且 `TERM` 不是 `dumb`），它和 `/model` 是同一个列表：输入即可筛选，↑ ↓ 移动，Enter 选择，Esc 或 Ctrl+C 取消设置（`Setup cancelled.`）。列表只含对话模型（嵌入和重排模型不在其中，除非服务器只列出这些）。有多个对话模型时，已加载的会标上 `(loaded)` 并高亮；服务器没有说明哪些已加载时，不高亮任何一项，在按键移到某一行之前 Enter 不起作用，并且会先用一行说明：尚未加载的模型可能要花几分钟和大量内存来加载。没有终端时（答案通过管道输入），模型以编号列表显示，对话模型在前，并询问 `Model [N]:`；输入编号或 ID。嵌入或重排模型会标上 `(embedding)` 或 `(reranking)`，且绝不作为默认值。有多个对话模型时，已加载的会标上 `(loaded)` 并作为默认值；服务器没有说明哪些已加载时，提示符是没有默认值的 `Model:`，只按 Enter 会再次询问（`Type the number of the model to use. …`）；没有终端时，空答案会以错误退出，退出码为 2。`/backend` 显示同样的列表（`Choose a model`；按 Esc 不做任何更改）。只列出嵌入或重排模型的服务器也会以相同方式显示，并提示对话需要对话模型。提示文字会按终端宽度在词边界处换行。
- 对 `GET /api/version` 返回 `{"version": "…"}` 的 Ollama 服务器会保存为 `chat_backend = "ollama"`，并在其 `[ollama]` 表中保存 `chat_url` 和 `chat_model`；横幅随后显示 Ollama，对话请求使用 Ollama 原生 API 及其 300 秒超时。oMLX 服务器（它的模型列表把 oMLX 标为每个模型的所有者）保存为 `chat_backend = "omlx"`，其他任何兼容 OpenAI 的服务器保存为 `chat_backend = "vllm"`（适用于所有兼容 OpenAI 服务器的键）；两者都在 `[vllm]` 表中保存 `chat_url` 和 `chat_model`。文件中的其他设置保持不变。
- 服务器列出所选对话模型之外的嵌入模型时，melo 会询问 `Use <model> for long-term memory?`（管道输入时是 `[Y/n]:`），优先推荐 Qwen3-Embedding。`Use it` 发送一次 `POST /v1/embeddings` 探测，并保存 `embedding_backend = "<chat key>"` 及对应表中的 `embedding_url`（对话 URL）和 `embedding_model`。选择 `Leave it off`（或按 Esc）、探测失败或没有候选模型时，会保存 `embedding_backend = "none"` 并说明记忆已关闭（`Long-term memory is off; melo setup turns it on.`）；`melo setup` 还会说明如何添加模型：Ollama 服务器是 `ollama pull qwen3-embedding:0.6b`，否则是 `Serve Qwen3-Embedding-0.6B on it`。参见[设置嵌入模型](embedding.md)。
- 服务器无法使用时，melo 会说明原因，例如 `✗ URL: connection refused.`，然后询问 `What now?`（管道输入时是 `Enter tries again · s starts without saving · q quits:`）。`Try again`（默认选中）再次询问 URL，默认值为你输入的地址；`Start without saving` 询问 `Model name [default]:` 并打开界面但不保存（`Not saved: melo asks again next time.`）；`Quit`（或 Esc）以退出码 0 退出。原因包括 `connection refused`、`no answer in 5 s`、附有 `the server requires an API key; localmelo sends none to local servers (oMLX: "Allow inference without API key")` 的 `HTTP 401` 或 `HTTP 403`、其他 `HTTP` 状态（重定向会显示目标，melo 不会跟随）、`the reply is not a model list`、`the reply could not be read`、`not a URL` 和 `it lists no models`。
- 管道输入时，选项按首字母选择，因此 `s`、`start` 和 `start without saving` 是同一回答（`r` 或 `retry` 等同于 Enter）；其他回答会重新询问。
- 在问题处按 Ctrl-D 会输出 `Setup cancelled.` 并以退出码 0 退出；Ctrl+C 输出相同内容并以 130 退出。
