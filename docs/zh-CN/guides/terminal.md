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
  Qwen3-8B · ~/demo-project · ctx 2.1k · What is in this folder?
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

1. `melo setup`：运行配置向导，然后使用它保存的后端。默认模式下，它先用首次运行的问题提供在本机找到的模型服务器（`Found a model server at …`、`Use it? [Y/n]:`）；只有回答 `n`、使用 `--developer`，或已保存后端但没有找到服务器时，才出现完整的后端菜单（什么都没保存时，会输出启动服务器的命令）。不带任务时，不能与 `--url` 同用。
2. `--url URL`：使用该服务器，不使用配置文件。模型使用 `--model` 指定的模型，否则使用 `GET URL/models` 列出的唯一对话模型（ID 包含 `embed` 或 `rerank` 的不是对话模型），再否则使用服务器说明已加载的那一个对话模型；有多个且不知道哪个已加载时，退出码为 2，列出这些模型并要求使用 `--model <id>`。任务、`melo worker` 和 `melo ""` 在第一个请求之前以同样的方式选择模型（见[命令行参考](../reference/cli.md#general)）。没有 `/v1` 的 URL 会通过 `URL/v1/models` 获取列表，如果该地址有响应，界面也会使用 `URL/v1`。服务器没有响应模型列表请求或没有列出模型时，模型为 `default`，横幅下方的警告会说明原因。没有配置文件时会询问一次 `Save URL · MODEL as the default backend? [Y/n]`；按 Enter 保存，与首次运行相同。
3. 已保存的后端；若指定 `--model`，则在本会话中用它替代保存的模型。界面打开前不做检查：显示横幅后，melo 会列出一次服务器模型（`GET /models`），如果服务器没有响应或没有列出该模型，就会警告：`⚠ Cannot reach the model server at URL (connection refused). Start it, then send your message.`
4. 没有已保存的后端：先回答[首次运行](#first-run)的问题，然后打开界面。

欢迎信息的第一行显示软件包版本；在 git 检出中还会显示短提交号（`melo 0.1.0 (baab21e)`），`melo --version` 输出同一行。当模型不是保存的默认模型时，文件夹之后的一行会给出服务器的 URL 及来源（`from --url (not saved)`、`not saved; melo asks again next time`）；如果 melo 从多个模型中选了服务器报告已加载的那个，还会显示 `the loaded one of N; /model lists them`。服务器没有响应时，欢迎信息下方会显示警告。配置文件写 `omlx`，或服务器的模型列表显示它是 oMLX 时，`/status` 把服务器称为 `oMLX`，否则称为 `OpenAI-compatible server`（Ollama、MLC LLM 和 SGLang 保持各自的名称）。开发者模式的方框有一行模型信息，写的就是这个名称；已保存的服务器在方框显示之后才列出模型，所以方框显示的是配置文件所写的名称；本次运行已经列出过该服务器时（首次运行就是如此）除外。

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
| Tab | 补全命令或列出的模型；↑ ↓ 在菜单中选择，Enter 确认所选项。 |
| Ctrl+L | 清屏。 |
| Ctrl+Z | 无操作（此版本的 melo 不会挂起）。 |

- **队列。**一次只能有一条消息等待，显示为 `↳ Queued for the next turn: …`。如果当前轮次以 `completed` 以外的状态结束（停止、失败），消息会返回输入框，不会发送。完全无法运行的轮次所对应的消息（`✗ The turn could not run: …`）也会返回，并排在它前面。
- **停止。**Esc 立即停止当前轮次；该轮以 `cancelled` 结束并输出 `■ Stopped`。回答输出后，melo 最多花 2 秒记录运行步骤（参见[程序性记忆](../concepts/procedures.md)）；此时 Esc 不起作用，下一条消息会等待记录完成。
- **历史记录。**消息和命令保存在 `~/.cache/localmelo/cli_history` 中（权限为 0600；超过 1 MB 时保留最新的 1,000 条），文件内容绝不会保存到其中。无法打开该文件时，会话在内存中保留历史记录并提示你。

<a id="commands"></a>

### 命令

输入 `/` 后，命令及其作用会显示在输入框下方；继续输入字母可缩小范围，按 Tab 或 Enter 选择。常用命令排在最前，也就是菜单的第一屏：`/model`、`/new`、`/status`、`/help`、`/exit`、`/undo`、`/copy` 和 `/retry`。每一行写明命令的作用，需要参数的命令还写明参数（`/attach PATH`）。

| 命令 | 作用 |
|---|---|
| `/help [command]` | 分三组（Conversation、Model、Help）显示常用命令、`!` 一行和查看更多的办法。`/help all`，以及开发者模式下的 `/help`，列出所有命令和按键；`/help NAME` 给出某个命令的详情。比终端宽的一行会在其说明下面续行。 |
| `/attach PATH` | 暂存一个工作区文本文件；路径含空格时请加引号。 |
| `/attachments` | 列出已暂存的文件及其编号。 |
| `/detach N\|all` | 移除指定编号的已暂存文件，或移除全部已暂存文件。 |
| `/new`（也可用 `/clear`） | 开始新会话：模型忘记当前对话，会话授权结束。 |
| `/sessions [filter]` | 在当前文件夹中选择已保存的聊天；输入文字筛选，Tab 切换到所有文件夹，↑ ↓ 选择，Enter 恢复，Esc 关闭。 |
| `/resume [id\|title]` | 按 ID、ID 前缀（至少四个字符）或完整标题恢复；不带参数时打开选择器。 |
| `/title [text]` | 显示或设置 1–200 个字符的标题。首条消息之前设置的标题会保留到会话创建时。 |
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
- **输出只留在终端。**输出不会进入模型、对话或事件，命令也不会存入输入历史。`/attach !` 会把最近一次输出的脱敏快照（最后 256 KiB，不含终端从未显示的文本：隐藏文本和清屏之前的内容；截断后的开头不完整行会被丢弃）附加到你的下一条消息。它总会先打印将发送的原文并询问 `[y/N]`（默认 No），即使没有脱敏；你不回答 yes，就不会附加任何内容。附加后，快照会像 [`/attach PATH`](#attach-text-files) 的文件一样未加密地随会话保存。
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
- **实时区域。**第一行显示 melo 正在做什么（`Starting`、`Thinking`、`Replying`、`Running`、`Waiting for your approval`、`Retrying in 4s`、`Reflecting`、`Wrapping up`、`Stopping`），接着显示时间；后面是输出 token 数和 `esc to stop`（`Thinking… · 14s · ↓ 180 tokens · esc to stop`）。开发者模式会在时间之前加上运行信息：步骤（`step 4`，不与 `--steps` 对比）、从第二次开始显示的尝试次数、阶段及其周期和已见到的测试。行宽不足时先省略提示，再省略 token 数，开发者模式下的信息用 `…` 截断。第二行显示正在运行的工具、其自身计时，以及 `shell_exec` 和 `python_exec` 的时限：`└ shell_exec  pytest -q tests · 0:25 of 10:00`。会话的第一轮中，第三行显示进度行会显示的过去运行：`last time 6 steps, 2m10s: Run ls ✓ · Read app.py ✓`。
- **思考。**显示 `✻ Thought for 1.2s`，只有一行。`/thinking on` 和开发者模式会再显示思考的前 3 行（`/verbose` 为 40 行），按宽度换行。中间消息在 `●` 后以 Markdown 输出；最后一条会先暂存，到下一步才输出；如果直到结束时才输出，则只在它不同于最终回答时显示。
- **工具调用**在完成时各输出一个区块：`✓` 表示已运行，`✗` 表示失败，`⊘` 表示被拦截或拒绝，`◌` 表示已停止或未运行；还显示耗时；命令运行后以非 0 退出码结束时显示 `exit N`（`grep` 以 1 退出并不代表失败）；审批结果；以及输出的第一行，同一行末尾显示 `+N lines (/expand T.C)`（若只剩一行则直接显示该行，不显示计数）。只有标记带颜色（`✓` 绿色，`✗` 红色，`⊘` 在深色终端为黄色、其他情况为粗体）；这一行的其余部分是默认颜色，耗时和标签是暗色。失败的调用，或运行后以非 0 退出码结束的命令，最多显示 3 行，失败时第一行为红色；开发者模式对每次调用显示 3 行，`/verbose` 显示 20 行并附上调用参数。图片则显示为 `▪ image/png · 312 KB · not shown`。输出中的转义序列会被移除，因此工具无法移动光标、设置标题或写入剪贴板。
- **通知和反思。**`↻` 标记重试：每轮一行，服务器无法连接时这一行会保留它的 URL（`↻ Could not reach URL; trying again in 1 s`），延迟在实时区域倒计时；`·` 表示空回复；`⚠` 表示卡住或预算用尽通知。反思显示为 `✻ Reflected after attempt 1: continue · Read the test before editing. (2.3s)`；`/verbose` 还会显示其摘要和完整理由，以及在询问了两次时显示 `asked twice`；无法读取的反思会说明运行是否继续。
- **回答**在 `●` 后以 Markdown 输出。未完成的运行会先输出尽力生成的结果，再用平实的话说明发生了什么、该怎么做：`✗ Could not reach the model server at URL.`，其下是 `Start it, then /retry.`。第一行是共享表中该状态的标题（`Ran out of steps (30).`、`Stopped: the same calls kept repeating.`），并附上有帮助的原因（URL、模型、失败检查的错误）；第二行是操作建议，使用默认颜色。页脚此时显示 `✻ Stopped after 14s` 而不是 `Worked for`，且不显示为零的工具调用次数。直接模式仍在标准错误上输出 `melo: run ended with status …` 这一行。

<a id="the-status-bar"></a>

### 状态栏

输入框下方的状态栏显示本会话的用量，数据来自 melo 已收到的事件：不发请求、不轮询、没有定时器。
在模型和工作区之后从左到右，每一段只在有数据时出现：

```text
  stub-model · ~/demo-project · cache 83% · ctx 1.4k · List the files.
```

- `cache 83%`：服务器报告的、上一次请求提示词中已缓存的比例（服务器没有报告时显示 `cache --`，绝不显示 0%）。
  本地服务器在距上次请求超过一分钟后还会显示 `idle 3m`；本地服务器按内存而不是按时间淘汰前缀缓存，所以没有倒计时。
- `ctx 12k/32k`：最近一次测得的提示词词元数与输入上限；不知道上限时只显示 `ctx 12k`。
- `step 4 attempt 2 phase fix`：运行进行中的进度事实，从不显示百分比或 `n/max`。开发者模式显示它们，默认模式不显示。
- `tasks 1 running 2 queued`：仅在能访问任务存储时出现；有任务在等待（例如等待审批）时追加 `1 held`。

状态栏在事件到达或按键时重绘，所以 `idle` 是上一次重绘时的值。审批模式只在不是 `ask` 时显示在右端（`auto`、`deny`，或 `⚠ YOLO: no approvals` 警告）；开发者模式始终显示。终端较窄时先去掉标题，然后从右到左去掉各段，再截短工作区，且不会换行。
`/status all` 会完整打印同样的数字。`[cli] status_line = "off"` 让状态栏保持这些段出现之前的样子：模型、工作区、上一次请求的 `ctx` 和标题。没有命令行参数。

<a id="streamed-replies"></a>

### 流式回复

交互式界面会在模型写出回复的同时把它流式显示出来，而且这是默认行为：它向提供方请求流式规划回复，并只在实时区域中绘制发布过滤器已经脱敏的文本，按应用的刷新频率重绘；它从不把原始增量直接写到终端。`/stream off` 为之后的轮次关闭流式，`melo -X cli.stream=false` 只关闭这一次运行，`melo config set cli.stream false` 则永久关闭。

不支持流式输出的模型服务器无需任何设置。当它用一个普通的 JSON 响应体回答流式请求，或在发出任何内容之前就关闭了流时，melo 会使用这个回复（或不带流式、把同一个问题再发送一次），并且在 melo 退出之前不再请求该服务器流式输出。此时回复在完成后才会显示，与 `/stream off` 时相同，melo 会只说明一次：`This server sends replies whole; melo shows them when they are complete.` 如果流在已经收到部分回复之后中断，仍然是错误：`Could not get an answer from the model server: Streaming response ended before completion`。

回复能否边写边显示，取决于模型如何标记推理，这由 `config.toml` 中按模型设置的提示声明：

```toml
[cli.reasoning_tags]
"Qwen3-8B" = "closing_only"
```

提示是 `explicit`（开头的 `<think>` 开始推理）、`closing_only`（`</think>` 之前都是推理）或 `none`（标签是普通文本）。没有提示的模型使用保守规则：内容一直保留到回复结束，因此实时区域显示 `✻ thinking…` 而不显示文本。在开发者模式下，它的第一条完整回复之后，melo 会说明一次：`This model's replies appear when complete until its reasoning format is set (see Streamed replies).` 新用户看不到这条说明。把推理放在单独字段里发送的服务器无需提示就能流式显示，该字段绝不会作为正文绘制；`/thinking on` 会显示它的摘录。同一个提示也会切分完整回复，因此流式显示和存储的回答不会不一致。

流式文本是临时的。如果回复变成工具调用、重试、失败的运行或停止，melo 会关闭实时文本并只打印一次，开发者模式下标为 `↳ partial reply (tool call)`、`(retrying)`、`(not the final answer)` 或 `(stopped)`；完成的最终回答会在 `●` 后只打印一次。不支持流式的提供方会发送完整回复，照常打印。`/stream on|off` 只切换后续轮次，不写入配置，`/new` 会把它恢复为 `[cli] stream`。没有交互式界面的运行（直接模式、worker、网关）从不流式，因此输出和请求体保持不变。无效的 `[cli]` 值不会让运行失败：交互式界面会发出警告并使用默认值，直接模式和网关根本不读取它。

<a id="approvals"></a>

### 审批

`shell_exec`、`python_exec` 和 `file_write` 只有在你回答后才会运行，标记为需要审批的 MCP 工具也是如此。（交互式界面加载 `--mcp` 指定的 MCP 服务器；未指定时加载顶层 `mcp_config` 键中的服务器。服务器启动失败会在本会话中关闭 MCP，`/mcp` 会说明原因。）审批面板取代输入框，草稿会保留：

```text
╭─ Run a shell command? ───────────────────────────────────────────────────╮
│ shell_exec · in ~/demo-project                                           │
│   ls -1                                                                  │
╰──────────────────────────────────────────────────────────────────────────╯
❯ 1. Yes
  2. Yes, and allow every shell_exec call in this session
  3. No, and tell melo why
  "Allow every" lets any shell command run without asking for the rest of this session.
  Commands can reach files outside ~/demo-project.
  ↑↓ select · enter confirm · 1-3 · v view all · esc stop the turn
```

- `1` 允许本次调用，`2` 在本会话剩余时间内允许该工具，`3` 打开理由输入框（最多 500 个字符），模型会读取该理由；Enter 发送。也可以用 ↑ ↓ 和 Enter 选择。字母不能作为回答；面板打开后的 0.4 秒内只有 Esc 和 Ctrl+C 有效，避免你正在输入的消息意外回答审批。
- 标题显示工具名和文件夹。开发者模式会在工具名后加上其权限级别，对于 `file_write` 则加在路径后（`shell_exec · exec · in ~/demo-project`；MCP 工具可为 `write`、`exec` 或 `network`，前提是界面能加载该工具）。只读 MCP 工具仅在服务器的 `require_approval` 强制询问时显示 `read`；参见[权限级别](tools-mcp.md#permission-tiers)。
- 在选项处按 Esc 或 Ctrl+C 会停止当前轮次，2 秒内再次按 Ctrl+C 则退出；在理由输入框中按 Esc 返回选项。
- 命令、代码或文件的新内容会按实际运行时的原样显示，长行换行，绝不截断：控制字符和会重新排列或隐藏文本的字符以转义形式显示（`\r`、`\x1b`、`\u{202e}`），并附有警告行，防止内容被隐藏。
- 面板最多显示正文的 8 行，在较短的终端上更少，之后显示 `… +N lines (v views all)`；按 `v` 可在面板上方输出完整正文。
- `--yolo` 不经询问就运行所有工具（意味着没有策略限制，而不是只读），状态栏会显示 `⚠ YOLO: no approvals`。

<a id="the-workspace"></a>

### 工作区

melo 在当前文件夹或 `--workspace DIR` 指定的文件夹中工作。文件工具（`file_read`、`file_write`）被限制在这个文件夹内。命令和 Python 可以访问其他文件，因此 melo 会先询问。工作区为你的主目录或 `/` 时，横幅会发出警告。界面中 `shell_exec` 和 `python_exec` 在 10 分钟后停止（单次任务分别为 30 秒和 60 秒）。

<a id="chinese-and-other-wide-characters"></a>

### 中文及其他宽字符

输入框按字符编辑，按终端单元格放置文本：中文和其他宽字符的输入、删除和换行都保持字符完整，删除换行后的消息也不会留下空行。已知限制：

- 由多个代码点组成的字符（用组合标记输入的重音、👩‍💻 之类的 emoji 序列）需要每个代码点按一次 Backspace。
- Ctrl+W 向左删除到上一个空格，因此对于没有空格的中文文本，会删除整段连续文本。
- 输入法（如 macOS 拼音）仍在组合的文本由终端绘制，而不是由 melo 绘制。

<a id="terminal-settings"></a>

### 终端设置

- 界面使用 16 色 ANSI 配色，因此浅色和深色主题都可使用。将 `NO_COLOR` 设为非空值可关闭颜色。
- 浅色终端。melo 通过 `COLORFGBG`（以 ANSI 颜色编号表示的 `FG;BG`，有些终端会设置它，例如 iTerm2 和 rxvt）了解背景色，从不向终端查询。只有在背景报告为深色时，警告才用黄色；背景为浅色或未知时，警告是默认颜色的粗体，回答中的代码在浅色背景下使用浅色语法主题。不设置该变量的终端按未知处理：用 `COLORFGBG='0;15' melo` 启动表示浅色，用 `COLORFGBG='15;0' melo` 表示深色。你必须阅读或据以操作的句子（警告、审批的警告和按键、输入框下方的提示）从不使用暗色；暗色只用于耗时、计数和 `(/expand T.C)`。
- 在编码不是 UTF-8 的终端中，或设置 `LOCALMELO_ASCII=1` 时，melo 使用 ASCII 字符代替符号（用 `>` 代替 `›`，用 `...` 代替 `…`）。
- 打开审批面板以及持续 30 秒或更长的轮次结束时，melo 会鸣响终端提示音；`LOCALMELO_BELL=0` 可关闭提示音。

<a id="sessions-and-files"></a>

### 会话和文件

直接运行 `melo` 会开始新会话；只有第一条消息才会创建会话记录。使用 `melo -c`（`--continue`）恢复当前工作区中最近一个有已完成轮次的聊天，或使用 `melo -r`（`--resume`）选择聊天。没有可选聊天时，两者都会提示并开始新会话；在启动选择器中按 Esc 也会开始新会话，不会创建空记录。`melo -r REF` 根据 ID、至少四个字符的前缀或不区分大小写的完整标题查找；有多个匹配项，或回退到标题／消息搜索时，会打开选择器。未知引用以退出码 2 退出。启动选择（包括 `-r` 选择器）发生在横幅之前：回顾会点出选中的聊天（`◆ Resumed "TITLE"`，开发者模式的方框把它标为 `"TITLE" (resumed)`）；按 Esc 或列表为空时显示新会话横幅。`-c` 和 `-r` 不能同时使用，也不能与任务或 `melo serve` 同用。

选择器最多列出 50 个具有已完成且未撤销轮次的聊天，最新的在前。筛选在 150 毫秒后更新；Tab 在当前文件夹和所有文件夹之间切换。Enter 恢复，Esc 或 Ctrl+D 关闭选择器，普通草稿保持不变。`/sessions FILTER` 和 `/resume REF` 使用同一选择器和查找方式。网页及调度器会话不会出现在选项中。网页控制台的 Workers 视图会在 **Terminal chats** 下以只读方式显示你的这些聊天，读取时会对其脱敏（保存的副本不会被脱敏）；控制台不能修改它们，也不能向它们发送消息。恢复会话会保留当前模型和工作区，将其记录到会话中，并清除审批授权；`/verbose`、`/thinking` 和 `/stream` 偏好保持不变。`/new` 会重置这些显示偏好（`/stream` 恢复为 `[cli] stream`）。如果会话之前在另一个文件夹中工作过，会显示提示。

横幅之后的回顾显示存储的脱敏副本中最近三个符合条件的轮次。标题下方的一行说明进入模型记忆的内容：`[sessions] resume_mode = "pairs"`（默认）恢复最近的用户消息／回答配对，不包含工具步骤；`"window"` 恢复已保存的工作记忆检查点及之后的配对，没有可用检查点时退回配对模式。在 `config.toml` 中设置此项；无效值会在界面中产生警告并使用配对模式，但不会在直接模式输出中增加警告。回顾中的计数反映淘汰后的恢复窗口，而不是存储的轮次总数。

两个终端可以轮流使用同一个 CLI 会话。发送前，melo 会重新加载另一个窗口的更改；会话忙碌或已删除时会保留草稿。退出时，有轮次的会话会输出其 `melo -r ID` 恢复命令。

会话保存在 `~/.cache/localmelo/cli/sessions.db` 中：每一轮都包含你的消息、回答以及每次工具调用和输出。`~/.cache/localmelo/cli_history` 保存你的消息。两者都是明文（权限为 0600），未加密。界面将日志写入 `~/.cache/localmelo/logs/cli.log`（文件夹权限 0700、文件权限 0600、1 MB、两个备份），绝不会在屏幕上输出 traceback；出现问题时显示一条暗色提示，指出日志位置。配置文件不是有效的 TOML，或无法创建日志文件夹时，`melo` 会输出一行 `melo: …` 并以退出码 1 停止。程序性记忆与单次任务一样，遵循[记忆持久化](memory.md#keep-memory-between-runs)。

<a id="first-run"></a>

### 首次运行

在没有配置文件的终端中，`melo` 会提供找到的模型服务器，询问模型，保存它们，然后打开界面。服务器列出了嵌入模型时，还会询问一次是否使用它。之后运行会直接进入界面。

```text
$ melo
melo uses a model server on this computer (oMLX, Ollama, LM Studio…).
Found a model server at http://127.0.0.1:8000/v1 (1 model).
Use it? [Y/n]:
✓ http://127.0.0.1:8000/v1 lists 1 model; using Qwen3-8B-4bit.
Long-term memory is off; melo setup turns it on.
Saved to ~/.cache/localmelo/config.toml (melo setup changes it).
```

如果服务器也列出了嵌入模型：

```text
Use nomic-embed-text for long-term memory? [Y/n]:
```

- melo 在本机通过 `GET /v1/models` 检查 `127.0.0.1` 的 8000（vLLM、oMLX）、8400（MLC）、11434（Ollama）和 30000（SGLang）端口，查找服务器，并提供第一个列出模型的服务器：对 `Use it? [Y/n]:` 按 Enter 即同意，回答 `n` 会询问 `Server URL […]:`（预填该 URL），可改填另一个。探测时请求模型列表；有多个对话模型时，还会请求 `GET /v1/models/status`（oMLX）和 `GET /api/ps`（Ollama），了解哪些模型已加载；保存选择时会请求一次 `GET /api/version`，以识别 Ollama 服务器。不发送对话请求，因此服务器不会为此加载模型。列出对话模型的服务器优先；只列出嵌入或重排模型的服务器其次。对于响应但没有返回列表的服务器，melo 也会指出该服务器并提供选择，例如 `A model server at URL answered, but HTTP 401: …`。都没有响应时，会输出启动服务器的命令（Mac 上是 oMLX，其他系统是 Ollama，与[模型服务器指南](model-servers.md)中的一致）并等待：

  ```text
  No model server is running yet. To start one with oMLX:

    brew tap jundot/omlx https://github.com/jundot/omlx
    brew install jundot/omlx/omlx
    omlx serve --model-dir ~/models --port 8000

  Put a model in its own folder of ~/models first. The guide shows how:
  https://github.com/localmelo/localmelo/blob/main/docs/guides/model-servers.md
  Press Enter to check again, type an address, or q to quit:
  ```

  按 Enter 会再次检查同样的端口，有服务器响应时按上面的方式继续（仍然没有时输出 `✗ No model server found yet (looked at ports …).`）。改为输入的地址会像 URL 一样检查（见下）；`q` 以退出码 0 退出。`LOCALMELO_DETECT_URLS` 用逗号分隔的列表替换这四个 URL，按 Enter 时检查的就是这些 URL；空值会关闭搜索，此时 melo 询问 `Server URL [http://127.0.0.1:8000/v1]:`。
- 列出多个模型的服务器会以列表供你选择。在终端里（标准输入和输出都是终端，且 `TERM` 不是 `dumb`），它和 `/model` 是同一个列表：输入即可筛选，↑ ↓ 移动，Enter 选择，Esc 或 Ctrl+C 取消设置（`Setup cancelled.`）。列表只含对话模型（嵌入和重排模型不在其中，除非服务器只列出这些）。有多个对话模型时，已加载的会标上 `(loaded)` 并高亮；服务器没有说明哪些已加载时，不高亮任何一项，在按键移到某一行之前 Enter 不起作用，并且会先用一行说明：尚未加载的模型可能要花几分钟和大量内存来加载。没有终端时（答案通过管道输入），模型以编号列表显示，对话模型在前，并询问 `Model [N]:`；输入编号或 ID。嵌入或重排模型会标上 `(embedding)` 或 `(reranking)`，且绝不作为默认值。有多个对话模型时，已加载的会标上 `(loaded)` 并作为默认值；服务器没有说明哪些已加载时，提示符是没有默认值的 `Model:`，只按 Enter 会再次询问（`Type the number of the model to use. …`）；没有终端时，空答案会以错误退出，退出码为 2。`/backend` 用编号询问。只列出嵌入或重排模型的服务器也会以相同方式显示，并提示对话需要对话模型。提示文字会按终端宽度在词边界处换行。
- 对 `GET /api/version` 返回 `{"version": "…"}` 的 Ollama 服务器会保存为 `chat_backend = "ollama"`，并在其 `[ollama]` 表中保存 `chat_url` 和 `chat_model`；横幅随后显示 Ollama，对话请求使用 Ollama 原生 API 及其 300 秒超时。oMLX 服务器（它的模型列表把 oMLX 标为每个模型的所有者）保存为 `chat_backend = "omlx"`，其他任何兼容 OpenAI 的服务器保存为 `chat_backend = "vllm"`（适用于所有兼容 OpenAI 服务器的键）；两者都在 `[vllm]` 表中保存 `chat_url` 和 `chat_model`。文件中的其他设置保持不变。
- 服务器列出所选对话模型之外的嵌入模型时，melo 会询问 `Use <model> for long-term memory? [Y/n]:`，优先推荐 Qwen3-Embedding。`y` 发送一次 `POST /v1/embeddings` 探测，并保存 `embedding_backend = "<chat key>"` 及对应表中的 `embedding_url`（对话 URL）和 `embedding_model`。选择 `n`、探测失败或没有候选模型时，会保存 `embedding_backend = "none"` 并说明记忆已关闭（`Long-term memory is off; melo setup turns it on.`）；`melo setup` 还会说明如何添加模型：Ollama 服务器是 `ollama pull qwen3-embedding:0.6b`，否则是 `Serve Qwen3-Embedding-0.6B on it`。参见[设置嵌入模型](embedding.md)。
- 服务器无法使用时，melo 会说明原因，例如 `✗ URL: connection refused.`，然后询问 `Enter tries again · s starts without saving · q quits:`。Enter 再次询问 URL，默认值为你输入的地址；`s` 询问 `Model name [default]:` 并打开界面但不保存（`Not saved: melo asks again next time.`）；`q` 以退出码 0 退出。原因包括 `connection refused`、`no answer in 5 s`、附有 `the server requires an API key; localmelo sends none to local servers (oMLX: "Allow inference without API key")` 的 `HTTP 401` 或 `HTTP 403`、其他 `HTTP` 状态（重定向会显示目标，melo 不会跟随）、`the reply is not a model list`、`the reply could not be read`、`not a URL` 和 `it lists no models`。
- 选项按首字母选择，因此 `s`、`start` 和 `start without saving` 是同一回答（`r` 或 `retry` 等同于 Enter）；其他回答会重新询问。
- 在问题处按 Ctrl-D 会输出 `Setup cancelled.` 并以退出码 0 退出；Ctrl+C 输出相同内容并以 130 退出。
