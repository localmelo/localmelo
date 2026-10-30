<a id="guide-web-console"></a>

# 指南：网页控制台

使用网页控制台可以配置 localmelo 网关、进行对话以及查看已保存的 worker 运行记录。本指南面向在浏览器中使用控制台的用户。

开始前，请安装网关额外依赖并准备好模型服务器；参见[启动网关](gateway.md#start-the-gateway)。网关需要身份验证，且只监听回环地址。工具以你的权限运行，且没有沙箱；批准工具调用前请先审查任务。

<a id="web-workspace"></a>

## 网页工作区

在浏览器中打开 `http://127.0.0.1:8401/` 即可进入网页工作区。这是控制台的页面，只与 `/api`（[API v2](../../reference/api-v2.md)（英文））通信，因此网关必须运行才能提供该页面：先运行 `melo serve`，它会在浏览器中打开已登录的页面。没有链接就打开页面时，会显示 “Sign in to localmelo”：在终端运行 `melo console`，页面会以已登录的状态打开（非默认端口需加上 `--port PORT`）。通过 SSH 或没有浏览器能启动时，它会改为输出一次性链接，请在 60 秒内打开。不要分享该链接；见[网关安全](gateway.md#keep-the-gateway-private)。

首次运行时，如果还没有设置模型，欢迎页会用一张卡片代替起始提示。它像终端的首次运行一样，在这台 Mac 上寻找模型服务器，并显示找到的内容：**Found oMLX on this Mac**（或 Ollama，或某个模型服务器）及其模型、**A model server needs an API key**，或 **No model server is running on this Mac**。**Use this model** 会保存该服务器、测试它，并让欢迎页无需打开设置即可使用。如果有多个对话模型而且不知道哪个已加载，卡片会列出它们、标出已加载的那个，并等你选择，绝不会替你选。没有服务器在运行时，卡片会显示启动服务器的命令（Mac 上是 oMLX，其他系统是 Ollama，与[模型服务器指南](model-servers.md)中的一致），放在一个点按一次即可全选的方框里，并提供键盘可用的 **Copy commands** 按钮；卡片在屏幕上且标签页可见时，每三秒再次寻找，服务器一响应，模型就会出现。**Check again** 立即寻找；**Enter a server address…** 会打开连接设置；在设置好模型之前，输入框显示 `Connect a model to start`。已配置的服务器不再响应时也会显示这张卡片，而且只在它显示在屏幕上时才会查找服务器（[`GET /api/setup/detect`](../reference/api-v2.md#get-apisetupdetect)）。

页面以默认模式打开，不显示只有开发者才需要的内容：对话的级别标签、**Details**、GPU 锁状态行、**LOCAL** 徽标、状态行的缓存、步骤、阶段和队列段，以及任务里的调度优先级。它的措辞也更平实：正在运行的对话在 **Stop** 旁写 `Working…`，设置里把服务器叫 **Model server**、把它的地址叫 **Server address**。页眉显示运行中标签和一个模型标签（模型名称、`Model offline` 或 `No model`），点按它会打开设置。当有运行、终端对话或任务可看时，**Workers** 和 **Tasks** 才会出现在侧边栏。[开发者模式](#developer-mode)显示全部内容。这些元素都保留在页面中，只是隐藏。

新对话在你发送任何内容之前，就会显示 melo 将在哪里工作。输入框里的标签显示 `In localmelo-workspace`，点按它会打开一个面板：默认文件夹（`[gateway] workspace`，没有设置时是 `~/localmelo-workspace`）、最近对话的文件夹（从新到旧，最多四个）、**Choose a folder…**，以及在 [macOS 应用](macos-app.md)中的 **Show in Finder**。在浏览器里，**Choose a folder…** 会打开一个路径输入框（完整路径，或以 `~/` 开头的路径；在其中按回车绝不会发送你的消息），在应用里则打开系统的文件夹面板。文件工具只在该文件夹内活动；shell 命令和 Python 从那里启动。你的选择会随第一条消息作为 `POST /api/sessions` 中的 `workspace` 发出（使用默认文件夹的对话不发送它），并保留给下一个新对话，之后显示在对话的页眉里（`In ~/notes`；开发者模式会把它加在模型之后）。输入框下方有三个单行起始提示（**Summarize the files in this folder**、**Plan a small project with me**、**Explain a piece of code**），会替你填好输入框，可以直接发送或修改，输入内容后它们就让开。

1. 使用 **Settings**（开发者模式下叫 **Connection settings**）选择模型服务器、它的地址、模型和工作区（开发者模式下叫 **Chat backend** 和 **Chat API URL**）。更改会在你的下一条消息生效。尚未保存任何设置时，对话框停在空白的 **Choose a server…** 上，在你选择之前，其中的字段、**Discover models**、**Test connection** 和 **Save** 都不可用。本地模型发现功能会列出模型，但不会启动模型。云端后端会指定 API 密钥的环境变量，界面只显示该变量是否已设置，绝不显示密钥。保存时使用加载的设置修订版本；如果文件已更改，请重新加载对话框后重试。**Test connection** 会探测尚未保存的模型设置；**Appearance**（开发者模式下叫 **Theme**）会立即应用 System、Light 或 Dark 主题，并在当前浏览器中记住选择。默认模式把 Model（含它的测试）、Appearance 和 **Developer mode** 开关叠在同一个对话框里；开发者模式保留 Model、Test、Theme 和 Developer 标签页，并显示设置文件及其修订版本。你也可以使用 `melo setup`。
2. 输入并发送消息。第一条消息会创建对话：页面以该对话的第一轮回答你的草稿，绝不会自行创建对话。回答以 Markdown 渲染，包含代码块。
3. 一轮的呈现方式与终端类似：你的消息、包含已运行工具卡片的工作组、推理、通知和反思、运行时的实时行，以及带有 `Copy response` 的最终回答。工具输出中的转义序列会被清除，其图片（PNG、JPEG、GIF、WebP）会在页面内显示，这一轮产生的图片会显示在回答下面（见[回答里的图片](#pictures-in-the-answer)）。没有工具调用的一轮，其时间只在回答下方的一行里显示一次；它的推理（如果有）在 `Thought` 一行下展开。对话会告诉模型它是 localmelo 的 melo，以及它运行在哪个模型和服务器上，所以问“你是谁？”时会说出它们（见[melo 如何介绍自己](terminal.md#what-melo-says-about-itself)）。
4. 输入框上方有一行状态行：与终端开发者模式状态栏相同的段（终端的默认模式用文字表述这些数字），`cache 67% idle 3m`、`ctx 1.0k/8.0k`、`step 3`、`tasks 1 running 2 queued`，读取自 `GET /api/sessions/{id}/status`，因此两边一致。默认模式只显示上下文段。模型服务器没有报告缓存 token 时，缓存占比显示 `cache --`，绝不会显示 0%；本地服务器显示空闲时间，而不是倒计时。有事件到达时最多每秒读取一次，页面的轮询时也会读取；屏幕较窄时从右侧开始省略段。打开的 worker 运行显示同样的一行。
5. 页眉中的 **N running** 标签统计当前正在运行的内容：本控制台中有轮次在进行的对话、任务、直接运行和终端对话（运行或终端对话只在其进程持有有效租约时才计入）。数量为 0 时隐藏，屏幕较窄时仍然保留。点按它会显示一个简短的列表，每一行写明类别和标题，并打开该对话、该任务的卡片或 Workers 详情；按 Esc 或点按别处即可关闭。数字来自 `GET /api/health` 的 `status`，随页面每 4 秒的轮询读取；列表最多显示 20 行，并说明还有多少在运行。

侧边栏按从新到旧列出对话，支持搜索、切换和关闭对话。每个标题下写出对话的文件夹（开发者模式把模型放在前面），标题前的强调色圆点表示这个对话在等你批准。草稿在当前浏览器标签页中按对话保留。打开的对话通过事件流（SSE）接收轮次；页面重新加载后会恢复对话，读取最近十轮，并从那里重放已保存的事件，因此不会丢失内容或重复显示。**Show earlier turns** 加载更早的轮次，不会移动当前视图。浏览器放弃连接的事件流会在 1、2、4、8、16 秒后以及之后每 30 秒，从页面自己的游标处重新打开，但只会在重新读取对话之后进行：如果对话已不存在，页面会进入新对话，而不是无限重试。标签页可见时，对话列表每 4 秒重新读取一次，以获取其他浏览器的更改。**Reconnect now** 会先检查对话，再重新打开事件流。独立的横幅分别显示存储、模型和连接问题，不会替换其他通知。

<a id="pictures-in-the-answer"></a>
## 回答里的图片

一轮产生了图片，就会显示在回答下面，浏览器和 [macOS 应用](macos-app.md)里都一样，所以你要的图表不用打开文件夹就能看到。

- **哪些图片。** 这一轮在对话文件夹里创建或修改的 PNG、JPEG、GIF、WebP 和 SVG 文件，以及回答里用文件夹内路径提到的图片文件。网关在这一轮开始和结束时各看一次文件夹，所以 `python_exec` 写出的文件虽然没有任何工具列出，也会算进来。查看时会跳过隐藏文件夹、`node_modules`、链接和三层以下的文件夹，看过 4,000 个条目就停止，一轮最多保留 24 张图片。
- **这一行。** 每个预览显示图片、文件名和大小（这一轮改动了已有文件时写 `updated`）。显示四张；`+N more` 展开其余，**Show fewer** 收起。对预览点按、按回车或空格会以完整大小打开；按 Esc、点按 **Close** 或点按图片外面即可关闭。预览在两种主题下和手机宽度下都能用。
- **重新打开的对话。** 这个列表随这一轮一起保存（[`turn.pictures` 事件](../../reference/events.md#session-events)），所以重新打开的对话只要文件还在文件夹里，就仍会显示图片。文件已经不在的预览写 `No longer in the folder`；不再是图片的写 `Not a picture file`。
- **网关提供什么。** `GET /api/sessions/{id}/pictures?path=` 只在以下条件都满足时返回文件：它的真实路径（跟随链接后）位于该对话的文件夹内，名称和内容都是 PNG、JPEG、GIF、WebP 或 SVG，并且不超过 20 MB。带 `..` 的路径、文件夹外的绝对路径、指向外部的链接和另一个对话的文件夹都会被拒绝（403）；名为 `chart.png` 的文本文件会被拒绝（415）；不会写入或列出任何内容。这个路由和所有 `/api` 路由一样需要控制台登录。SVG 带有禁止脚本、不让它作为页面打开的策略（[API v2](../../reference/api-v2.md#get-apisessionsidpictures)（英文））。

<a id="melo-in-the-page"></a>
## 页面里的 Melo

页面在原来就有 Melo 图片的位置显示 Melo，这只秋田犬幼崽，在应用和浏览器里一样。所有图片都是静止的：页面不让任何东西动，所以不需要为“减弱动态效果”单独处理。

- **空对话。** Melo 坐在 *What would you like to work on?* 上方，神情平静。
- **回答。** 每条回答的 `localmelo` 旁是 Melo 的脸。最新一条回答上，它显示该轮的状态：运行时在工作（吐着舌头），等待批准时睁大眼睛，完成时开心，以带警告或错误结束时耷拉着一只耳朵。你自己停止的一轮不算失败，保持平静的脸；更早的回答也都是平静的脸。
- **批准栏。** 写着 “melo is waiting for your answer” 的栏显示睁大眼睛的脸。
- **品牌和标签页。** 侧边栏和登录页显示平静的脸，浏览器标签页以 16、32 和 64 px 使用它。

在文字已经说明状态的地方（实时行、工作组、失败卡片、批准栏），这些脸只是装饰，替代文字为空。上方没有工作组的已完成回答，没有别的东西说明它完成了，所以它的脸对屏幕阅读器读作 `Done`。画像保留替代文字 `Melo, the localmelo Akita mascot`。

图片是带一圈很细奶油色外沿的透明 PNG。在浅色画布上，深蓝轮廓让 Melo 显现出来（对控制台的浅色表面至少 12:1）；在深色画布上靠的是外沿（对深色表面至少 5.6:1，图片的要求是 3:1）。它们由 `/assets/akita/` 提供（`sit-idle.png`、`face-idle.png`、`face-working.png`、`face-waiting.png`、`face-done.png`、`face-error.png`，以及 `favicon-16.png`、`-32`、`-64`），共 150 KB，是应用自己在 `macos/art/akita` 的图片的副本，由同一个脚本生成（见 [macOS 应用指南](macos-app.md)）；因为 `melo.png` 和 `icon_small.png`（80 KB）已从页面资源中移除，净增约 70 KB。

<a id="inspect-workers"></a>

在手机上，或在宽度不超过 700 px 的窗口里，页面会去掉小触摸屏用不上的内容。页眉在模型标签只写出模型名时把它去掉（输入框里已经显示模型），而在它发出警告时（`Model offline`、`No model`）保留。键盘提示（`Enter to send · Shift + Enter for a new line` 和 `⌘ K`）在触摸屏和窄窗口中都不显示。失败轮次的下一步操作是至少 44 px 高的按钮。

## 查看 worker

**Workers**（默认模式下，有运行或终端对话后才出现在侧边栏）打开来自命令行会话文件的只读 worker 历史记录。你可以按状态、来源或精确的父运行筛选，使用服务器的游标加载更多记录，并打开一次运行以查看其记录的任务简报、元数据、工作过程和回答。每次直接运行的 `melo` 默认都会记录在这里（`--no-save-session` 或 `[sessions] record_direct = false` 可退出，见 [Worker 历史记录](../reference/cli.md#worker-历史记录)）。列表会显示仍在运行的任务，并每三秒重新读取一次。打开一个未完成的运行时，它先读取一次，然后跟随它的事件流：新的步骤、工具调用和文本约在一秒内出现，不会再次读取之前的工作，运行结束后事件流停止。隐藏的标签页会停止跟随，重新显示时从最后一个事件继续。**Refresh** 会立即重新读取该运行。包含笔记、已接受的消息、记忆决定或召回的运行，会把每一项显示为一行淡色文字。该视图不能发送消息、批准调用或更改授权；打开 Workers 时，关闭对话和级别控制均不可用。连接设置仍可单独使用。**Chats** 返回打开的对话。记录按模式进行脱敏，不能保证不含秘密信息；分享前请先检查。

**Worker runs** 旁边的 **Terminal chats** 标签会列出同一文件中交互式 `melo` 会话的对话，按最近活动排序；状态筛选保留，来源和父运行筛选不再显示，因为对话没有这两项。对话以只读方式打开，每一轮一个小节；当它的终端正在运行一个轮次时，页面会在其事件流上跟随，该轮结束后页面继续检查这个对话，并跟随下一轮。这些对话不会出现在侧边栏的对话列表里，控制台也不能向它们发送消息。终端以未脱敏的方式保存对话，所以控制台读取时会对每段文本脱敏：密钥、Bearer 令牌和私钥块显示为 `[redacted]`，读取过 `.env` 之类密钥文件的输出会被一条说明替换。图片不会被脱敏。

<a id="tasks"></a>

## 任务

**Tasks**（`#/tasks`；默认模式下，有任务后才出现在侧边栏）列出调度器的任务，按最近更新排在前面，显示状态、优先级（仅开发者模式）和标题。打开一个任务（`#/tasks/<id>`）可以看到它的状态、保存的恢复卡片（以纯文本显示）、证据状态，以及它当前状态允许的操作，还有与对话相同的状态行（任务运行过之前没有这一行）。和 Workers 一样，打开这个视图时它会取代对话。

- 已停放或中断的任务会显示 **Resume**。页面先询问服务器，只有服务器要求时才请你确认：对话框会写 "A process from before the restart may still be running. Resuming may repeat effects. Confirmation does not prove the process stopped or effects happened only once."，并提供 **Resume anyway** 和 **Cancel**。按 Escape 即取消。在你选择 **Resume anyway** 之前不会记录任何东西；确认在五分钟内有效，网关重启后失效。
- 请求运行工具的任务会显示状态 "Waiting for your approval"，以及和对话相同的批准卡片，包含 **Allow once**、**Allow for this task** 和 **Deny**（可附说明）。通过 `melo-task` 提交的任务也在这里等待批准，别处不会显示。回答后卡片消失，任务继续（允许）或收到拒绝（拒绝）；暂停或停止任务会撤回卡片。视图打开时，每 3 秒重新读取任务（或列表），只重绘有变化的部分，因此之后到达的请求会自动出现。
- 已开始的任务会在批准卡片下方显示 **Live** 区：它的步骤、工具调用和模型文本，画法与 Workers 视图显示一次运行相同。卡片先读取一次任务的会话（`GET /api/sessions/<task id>/turns?events=true`），任务运行或等待批准期间再跟随 `GET /api/sessions/<task id>/events?after=<last id>`。新事件合并后最多每 150 毫秒绘制一次，不会重新读取之前的工作，也只重绘 Live 区，因此在批准卡片里输入拒绝原因时焦点不会丢失。标签页隐藏时会关闭事件流，重新显示时从最后一个事件之后继续。任务完成、失败、取消、过期、中断或停放时，事件流在最后一次读取后关闭；任务恢复后会再次被跟随。从未开始的任务没有 Live 区，已结束的任务只显示一次它的工作过程。控制台显示任务事件的方式与显示对话事件相同：同一个需要身份验证的回环地址，不做额外脱敏。
- **Stop** 使用任务自己的停止操作，而不是对话的停止。在任务结束前都可用，并且优先于正在进行的 Resume。
- 已经开始但没有保存卡片的任务会显示 "Resume unavailable: recovery information is missing." 和 **Start again as a new task**，它会根据原始目标创建一个单独的新任务。之前那个任务的效果是未知的，也不会被撤销。

请求进行中时对应按钮不可用。被拒绝时（确认已过期、任务已结束、Stop 抢先）会显示服务器给出的文字。

<a id="inspect-a-chat"></a>

## 查看对话详情

在开发者模式下，**Details** 打开会话抽屉，显示工具和会话授权、工作记忆及长期记忆状态，以及已完成的运行；当页面提供轮次游标时，还可分页查看运行记录。你可以在那里撤销已授权的工具；工具视图可能会打开一个暂停的会话。在窄屏上，抽屉会排列在消息输入框下方。

<a id="developer-mode"></a>

## 开发者模式

**Connection settings → Developer**（默认模式下是 **Settings**，开关在最后）里有一个开关，“Developer mode: show advanced settings and debug views”。它保存为 `config.toml` 中的 `[ui] mode`，也就是 `melo --help` 读取的设置，并与其他设置一样做修订版本检查；手工写进文件的模式会在页面下次读取时生效。它只改变显示的内容，不改变任何运行行为。开启后：

- Developer 标签页以只读方式列出高级和实验性的 `melo` 选项，标明各自的层级；实验性选项还标明决定它的 issue（用命令行参数或在 `config.toml` 中设置）；
- **Details** 增加 **Events** 标签页，显示最近一轮保存的原始事件；**Memory** 标签页增加智能体状态和长期记忆后端；
- **Runs** 标签页增加每次运行记录的计数：请求数、工具调用数、token 用量和缓存数字；
- 页眉显示对话的级别标签、**Details**、GPU 锁状态行，以及显示页面状态（`Ready`、`Working`、`Approval needed`）的标签，侧边栏始终列出 **Workers** 和 **Tasks**；
- 状态行增加缓存、步骤、阶段和队列段，Tasks 显示每个任务的优先级；
- 侧边栏显示 **LOCAL** 徽标，正在运行的对话写 `Running on your backend`，设置保留它们的技术用语（**Chat backend**、**Chat API URL**、`Settings apply to subsequent turns…`）。

关闭时这些内容只是隐藏，不会被删除：再次开启即可看到。对话在等待其他持有者的 GPU 锁时，默认模式会在 GPU 状态行的位置显示一条横幅：“Waiting for this Mac's GPU, your chat starts when it is free.”对话、批准、Workers 和 Tasks 视图，以及设置所保存的内容，在两种模式下的行为相同。

<a id="recover-a-send"></a>

## 恢复发送

每次发送由一个 `client_request_id` 标识。如果回答丢失，页面会向存储查询消息是否已被接受，并仅在未被接受时提供 `Retry`；重试会复用同一个键，因此一条消息只产生一个轮次。未被接受的消息会返回该对话的草稿。发件箱及其键也保存在标签页的会话存储中；重新加载时仍在发送的消息，会先对照存储进行检查，再决定是否重试。页面会显示 `Not sent`（如果无法查询存储，则显示 `Outcome unknown`），以及 `Retry` 和 `Cancel and keep draft`。

<a id="scheduling-and-stopping"></a>

## 调度与停止

在开发者模式下，对话标题栏有一个级别标签（P1 至 P4，默认 P2），用于设置调度器如何相对于其他对话排列该对话的调用；级别只保存在内存中，不会持久化。标题和级别标签旁边的机器状态行会显示其他持有者的 GPU 锁（`GPU lock: <holder> · <purpose>`，附带 `(recorded)` 或 `(chats wait)`）、`GPU lock: yours`、`GPU lock unreadable`、`· ETA passed`，以及电池状态（`Battery 55% · discharging`）。使用交流电且没有锁时不显示任何内容。级别标签是菜单按钮：按 Enter 或向下箭头会打开菜单，焦点落在对话当前的级别上，方向键、Home 和 End 在各级别之间移动，Escape 关闭菜单并回到按钮。调用等待时（等待可用槽位、后台上限或其他持有者的锁），标题栏的状态标签和状态文字会显示 `Waiting: <reason>`，侧边栏显示 `● Waiting`，实时行也显示相同的等待状态；原因使用调度器自身的代码（`Resources`、`Priority` 等），只有等待其他持有者的 GPU 锁（`Reservation`）时显示 "Waiting for this Mac's GPU"，与横幅的说法一致。这些与终端显示的是同一组信息，通过网关共享的同一个门控机制读写：会话上的 `level` 和 `held`、列表上的 `machine`，以及用于设置它们的 `POST /api/sessions/{id}/level` 和 `POST /api/presence`。

停止操作会取消运行；光标在消息输入框内时按 Escape 也会停止，在其他位置按 Escape 只会关闭已打开的内容，在 Tasks 和 Workers 里与在对话中一样。正在进行的模型请求会被放弃，但模型服务器可能仍会完成计算；正在运行的 `python_exec` 子进程会连同其进程组被终止。`python_exec` 代码在子进程中运行，因此网关在其运行期间仍能响应其他请求。工具的限制参见[工具与 MCP](../reference/tools.md)。该轮次以 `cancelled` 状态保存，保存完成后对话即可接收下一条消息。每条消息都能看到该对话先前的回答，与[会话](gateway.md#sessions)相同。

<a id="approvals-and-failures"></a>

## 审批与失败

需要审批的工具调用会在其 `Needs approval` 工具下方显示卡片。控制台会话使用 `RememberingPolicy`，在执行 `file_write` 和 `python_exec` 前请求审批；旧版 `/v1` 没有审批策略。审批不是针对持有 bearer 令牌的人或同一操作系统账户下恶意进程的安全边界，他们可以响应审批请求。

卡片用一个简单的问题发问（"Run this command in demo-project?"、"Replace notes.txt?"、"Let melo use <tool>?"），问题下方是工具名和将要运行的确切内容。对已有文本文件的写入会显示它替换的内容：问题下面一行统计行数（"5 lines removed, 1 added"），框里列出有变化的行，每行带行号、− 或 + 符号，以及每侧两行上下文。删除的行标红，添加的行标绿，符号在没有颜色时表达同样的意思。较长的改动先显示前 20 行，**Show all N lines** 显示其余部分。行中的密钥读作 `[redacted]`。melo 无法比较时（文件不是文本、无法读取、超过 200 KB 或含有私钥，或者包括上下文在内的改动超过 1,000 行），卡片用一行文字说明，问题仍带着文件大小（"Replace logo.bin (480 bytes)?"），并只显示新文本。这个比较只供你阅读：下面的回答与其他文件写入的回答规则相同。实心按钮 **Allow once** 只运行这一次调用。**Allow for this chat**（在 Tasks 中为 **Allow for this task**）让该工具在这个对话的其余时间不再询问；按钮下方的一行会说明这一点，对命令或 Python 代码还会警告它可以访问工作区之外的文件。**Deny** 会打开一个可选的输入框 "Tell melo what to do instead"，并连同你写的内容一起发送拒绝；**Back** 或 Escape 回到三个回答。控件需等待 500 毫秒才能使用；请求到达不会移动焦点，也没有单键快捷方式可响应请求。卡片不在视图中时，消息输入框上方的提示条会写 "melo is waiting for your answer"，点按 **Review** 会打开其所在分组并移至 **Allow once**。如果无法加载审批详情，或参数被截断或转换，两个允许按钮将不可用；你可以重新加载详情、拒绝或停止运行。

失败的轮次会显示一张卡片：符号（错误为 ✗，警告为 ⚠）、简明的标题、错误的第一行和一条提示。最近的轮次用实心按钮给出一个下一步，其余操作以链接显示：模型服务器给出了错误回答（密钥被拒绝、模型不存在）时为 **Open model settings**，步数用尽或超时后为 **Continue**（没有运行过工具时为 **Try again**），卡住、失败或反思后停止时为 **Try again**，工具无法启动时为 **Show tools**，localmelo 内部崩溃后为 **Copy details**。**Try again** 和 **Continue** 只在最近完成的轮次上提供，Continue 还要求该轮次已运行工具。**Copy error** 复制一行文字和运行 ID，便于提交错误报告；技术字段仍放在 **Details** 中。已停止或中断的轮次使用中性提示行（■）。页面连不上网关时，提示条带有 **Try again**。

模型服务器完全连不上时，卡片写 `Could not reach the model server`，并问 `Is your model server still running at 127.0.0.1:8000? Start it, then try again.`，其中的地址就是失败的那个地址。卡片不重复错误本身：提供方的原始文字留在 **Details** 中。实心按钮是 **Try again**（该轮次已运行过工具时是 **Continue**，接着做而不是从头开始），**Change the server address** 是打开设置的链接。使用云端模型时，提示是 `Check your network connection, then try again.`，链接仍是 **Open model settings**，因为没有地址可改。以这种方式结束的运行，其工作框也不再列出重试行（卡片已经说过了）；运行仍在重试时它们照常显示。

只含模型未完成思考的回复绝不会显示为回答：它是工作框中折叠的思考项，附有一行 `The model's reply ended inside its thinking; asking it for its answer.`，模型会被再问一次。第二个回复仍没有回答时，卡片标题是 `melo did not finish an answer`，提示是 `The model's reply held only its thinking. Try again, or rephrase the request.`，实心按钮是 **Try again**。

<a id="persistence-and-limits"></a>

## 持久化与限制

页面中的对话是保存在网关自身会话文件 `~/.cache/localmelo/gateway/sessions.db` 中的会话；终端则使用自己的 `~/.cache/localmelo/cli/sessions.db`。对话会一直保留，直到你关闭：重启网关不会删除它们，重新打开对话时会从文件读取对话记录，但尚未构建智能体（`not_loaded`）；下一轮或重试开始时，才会根据文件中保存的对话构建智能体。

对话以网关模式运行：其智能体使用网关默认运行选项，包括四个内置工具 `file_read`、`file_write`、`list_files` 和 `python_exec`，不使用 MCP 工具，但会保留回答。它可以列出文件夹、读取和写入文件、运行 Python 代码；由于不提供 `shell_exec`，因此没有直接的 shell 命令（Python 代码仍可启动进程）。其模型使用网关从配置文件中读取的模型，页面也会读取该文件的 `[scheduler]` 表，包括 `policy`、各通道大小、调用日志以及锁和电池相关键。

在 `lock_gate = "record"`（默认值）模式下，其他持有者的锁会被显示和记录，但不会导致等待。在 `wait` 模式下，只要存在其他持有者的锁，对话发出的每个调用都会等待，包括嵌入调用。对话请求最多等待 300 秒；一个轮次在活跃时间达到 600 秒后停止：调度器暂扣调用的时间，包括因其他持有者的锁而暂扣的时间，不计入该限制。

页面不使用默认禁用的[旧版 playground 路由](../reference/http-v1.md#legacy-playground)。

工作区有以下限制：

- 仅供回环地址使用。按上文所述登录，然后在本机打开 `http://127.0.0.1:8401/` 或 `http://localhost:8401/`。不支持从其他设备或通过代理远程访问；参见[确保网关私密](gateway.md#keep-the-gateway-private)。
