# API v2

API v2 是网页控制台的 HTTP API。它在网关上以 `/api/` 为前缀提供，且仅当网关监听回环地址（`127.0.0.1`、`localhost` 或 `::1`）时才可用。

受保护的路由需要 bearer 令牌或已登录的浏览器 cookie；不要暴露网关。见[网关与 HTTP API](../guides/gateway.md#keep-the-gateway-private)。

除认证要求外，API 字段保持增量兼容：会新增端点、可选的请求字段和响应键。客户端可以忽略它不认识的键。

本页记录每一个 API v2 路由：健康检查、设置、密钥状态、连接测试、模型列表、会话、轮次、SSE 事件、blob、图片、提交记录、审批、工具、授权、记忆、运行，以及调度器任务。

`POST /api/auth/login` 用一个有效的一次性代码换取浏览器 cookie；`POST /api/auth/logout` 撤销该 cookie。`POST /api/auth/rotate` 需要 bearer 令牌，并会轮换它。

登录换取要求匹配的 Origin。当 macOS 应用启动了网关时，它还要求应用的 `X-LocalMelo-Instance` 请求头，并且代码只对签发它的那个实例有效；缺失、不同或绑定不一致的值会在代码被消费之前被拒绝。

见[网关安全](../guides/gateway.md#keep-the-gateway-private)。

控制台的一轮对话与工作区对话经过同样的调度器闸门：每次模型调用都要在 `[scheduler]` 下等待自己的槽位，调用日志会以会话的 id 记录它。见[调度器](../guides/scheduler.md#the-gate-on-chats-s2)。

JSON 请求体和 JSON 响应使用 UTF-8。事件流使用 `text/event-stream`，图片 blob 和图片文件使用其图片媒体类型，成功的 204 响应没有响应体。

非空请求体需要 `Content-Type: application/json`，且请求体最多 1 MiB（1,048,576 字节）。请求中的每个字符串都必须是有效的 UTF-8，且不含孤立代理项。

响应带有 `Cache-Control: no-store`、`Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; font-src 'self'; manifest-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`（blob 除外，其策略为 `default-src 'none'; sandbox; frame-ancestors 'none'`；图片见[它的路由](#get-apisessionsidpictures)）、`X-Content-Type-Options: nosniff`、`Referrer-Policy: no-referrer`、`Cross-Origin-Opener-Policy: same-origin`、`Cross-Origin-Resource-Policy: same-origin`、`X-Frame-Options: DENY` 和 `Permissions-Policy: camera=(), microphone=(), geolocation=()`。被拒绝的请求只有一种错误结构：

```json
{"error": {"code": "not_found", "message": "Session not found.", "details": {}}}
```

除非错误码另有说明，否则 `details` 为 `{}`。以下错误码可在下面的每个路由上返回：

| 状态 | `code` | 何时 |
|---|---|---|
| 400 | `invalid_request` | JSON 格式错误、类型错误、未知字段、控制字符或孤立代理项、设置中相对路径的 `workspace`、错误的查询参数，或格式错误的 `client_request_id` |
| 400 | `invalid_settings` | 候选设置未通过 `Config.validate()`；`details.errors` 列出各条消息 |
| 403 | `foreign_origin` | 来自其他站点的 `/api/` 请求（`Sec-Fetch-Site` 既不是 `same-origin` 也不是 `none`），或来自其他来源的、会改变状态的请求 |
| 403 | `outside_workspace` | 图片请求的真实路径不在会话的文件夹内 |
| 404 | `not_found` | 未知的会话、blob、图片或路由 |
| 405 | `method_not_allowed` | 已知路径，但不支持该方法；`Allow` 请求头列出它支持的方法 |
| 409 | `settings_changed` | 自 `GET /api/settings` 读取其 `revision` 以来，`config.toml` 在磁盘上已被改动 |
| 413 | `too_large` | 请求体超过 1 MiB，或所请求的图片超过 20 MB |
| 415 | `unsupported_media_type` | 非空请求体没有 `Content-Type: application/json` |
| 415 | `not_a_picture` | 图片请求的文件，其名称或内容不是 PNG、JPEG、GIF、WebP 或 SVG |
| 421 | `misdirected_request` | `Host` 请求头不是 `127.0.0.1:<port>`、`localhost:<port>` 或 `[::1]:<port>`，或由应用启动的网关的 `/api/auth/login` 没有匹配的 `X-LocalMelo-Instance` |
| 500 | `internal_error` | 意外失败；消息为 `Internal error; see the gateway log.`，从不包含堆栈跟踪或请求中的值 |
| 502 | `backend_unreachable` | 列出模型失败或没有返回任何模型 |
| 503 | `store_unavailable` | 网关的会话存储无法打开，或停止记录变更；第二种情况下请重启网关 |

网关会记录 500 错误的原因，其响应只带固定的消息。503 会重复存储的原因，并把主目录写作 `~`；它从不包含堆栈跟踪、租约持有者或密钥。

旧版 `/v1/*` 路由保留它们自己的响应体，包括 FastAPI 的 `422` 和 `{"detail": ...}`。

## GET /api/health

就绪状态、API 和事件结构的版本、设置中的模型、缓存的模型服务器检查、各项限制、存储的状态，以及当前正在运行的内容。存储不可用时仍提供健康检查。

示例把网关的 bearer 令牌从其私有文件通过管道传给 curl，使它不出现在命令行参数和 shell 历史中。不要开启 curl 的详细输出，也不要记录该请求头；同一用户的进程仍然可以读取令牌文件。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/health
```

```json
{
  "ok": true,
  "api": 2,
  "events_schema": 1,
  "version": "0.1.0",
  "configured": true,
  "config_errors": [],
  "model": {"backend": "ollama", "backend_label": "Ollama", "name": "qwen3:8b", "embedding": "none"},
  "model_server": {"url_label": "http://127.0.0.1:11434/v1", "reachable": true, "error": "", "checked_s_ago": 4},
  "limits": {"text_chars": 100000, "attachment_budget_chars": 8000, "body_bytes": 1048576},
  "sessions": {"open": 0, "running": 0, "pending_approvals": 0, "store": "ok", "store_reason": null, "store_error": "", "resume_mode": "pairs"},
  "status": {
    "running": {"chats": 0, "tasks": 1, "runs": 1, "terminal": 1},
    "tasks": {"running": 1, "queued": 1, "held": 0, "top_level": 2},
    "items": [
      {"kind": "task", "id": "a1b2c3d4e5f6", "title": "Index the repo", "activity": "Reading notes.md", "icon": "read"},
      {"kind": "run", "id": "c44733ec77e5", "title": "Summarize the changelog", "activity": "Thinking", "icon": "think"},
      {"kind": "terminal", "id": "d27a70f1bbd8", "title": "Debug the flaky test", "activity": "Running the tests", "icon": "run"}
    ],
    "waiting": [],
    "recent": {"kind": "chat", "id": "e5f60718293a", "title": "Plan the trip", "outcome": "done", "ended_s_ago": 3.2},
    "focus": {"kind": "terminal", "id": "d27a70f1bbd8", "title": "Debug the flaky test", "activity": "Running the tests", "icon": "run"}
  },
  "uptime_s": 132.4
}
```

- `version` 是已安装包的版本；在检出的代码中则为 `git describe` 的结果。
- `configured` 表示配置文件是否指定了对话后端；`config_errors` 是对所用设置执行 `Config.validate()` 的结果。
- `model.backend_label` 是后端注册表中的标签；对于 `vllm` 后端，配置文件写 `omlx` 或服务器的模型列表显示它是 oMLX 时为 `oMLX`，否则为 `OpenAI-compatible server`。未配置嵌入后端时，`model.embedding` 为 `none`。
- 对于云端后端，`model_server` 为 `null`。对于本地后端，它是 `GET <url>/models` 的缓存结果：在第一次检查结束之前，`reachable` 为 `null`，`checked_s_ago` 为该结果的时长。请求从不等待检查；过期的结果会在后台启动一次检查（30 秒之后，或服务器无响应时的 5 秒之后）。当 `chat_url` 不是绝对的 `http` 或 `https` URL 时，`url_label` 为空，`reachable` 为 `false`；这样的值从不被探测，其文本（包括用户信息）也不会被回显。
- `limits` 是入口文本上限、会话文本预算，以及以字节计的 API 请求体上限。
- `sessions.store` 取值为 `ok`、`unavailable`（存储无法打开，网关每 5 秒重试一次）或 `failed`（写入失败；请重启网关）。`store_error` 说明原因，并把主目录写作 `~`。`store_reason` 在健康时为 `null`，另一个网关拥有该存储时为 `owned`，其他任何打开失败为 `open-error`，写入失败之后为 `failed`。客户端应使用这个稳定的原因，而不要解析 `store_error`；存储成功打开后它会被清除。
- `sessions.resume_mode` 是当前使用的恢复方式；目前为 `pairs`。
- `status.running` 按类别统计当前正在运行的内容：`chats`（控制台中有轮次在进行的对话，与 `sessions.running` 相同）、`tasks`（状态为 `running` 的调度任务）、`runs`（直接运行和 `--brief` worker）和 `terminal`（交互式 `melo` 对话）。后两项来自命令行文件，只在租约有效时才计入一个轮次，因此进程已终止的运行不会被计入。`status.tasks` 是状态行所显示的调度队列：`running`、`queued`、`held` 和 `top_level`（其中最高的级别，没有则为 `null`）。`status.items` 列出正在运行的内容，先是对话，然后是任务、运行和终端对话，最多 20 项，每项为 `{"kind", "id", "title", "activity", "icon"}`（`kind` 为 `chat`、`task`、`run` 或 `terminal`；终端对话的标题已脱敏）。计数没有上限。无法读取的存储按没有运行任何内容计；一次调用通过索引读取任务状态，并通过索引读取命令行文件中正在运行的轮次。
- `activity` 用平实的话说明该项正在做什么，`icon` 是它的类别：`Waiting for your approval`（`wait`）、`Waiting for the model`（`held`，调度器暂缓了它的模型调用）、`Reading notes.md` 和 `Looking through
  ~/talk`（`read`）、`Writing slides.md`（`write`）、`Running the tests`（运行测试套件的命令）、`Running a command` 和 `Running Python`（`run`）、`Reading example.com`、`Reading the page` 和 `Following a link`（`web`）、`Remembering`（`memory`）、`Checking its work` 和 `Thinking`（`think`），其他任何工具为 `Using a tool`（`tool`）。它们来自该项最新的 40 个事件；只显示文件名、文件夹和网页的主机名，绝不显示命令行、代码或文本。无法读取的存储显示 `Thinking`。
- `status.focus` 是一眼看过去时显示的那一项（形式与 `status.items` 中的相同）：先取等待审批的一项，否则取最新事件最晚的一项；没有任何内容运行时为 `null`。
- `status.waiting` 列出有工具审批请求在等待回答的对话和任务，先是对话，最多 20 项，每项为 `{"kind", "id"}`（`kind` 为 `chat` 或 `task`）。控制台用 `#/s/<id>` 和 `#/tasks/<id>` 打开它们，macOS 应用的菜单栏在每次轮询时读取这个列表。`sessions.pending_approvals` 只统计已打开对话的请求。
- `status.recent` 说明最近结束的对话轮次、任务、运行或终端对话轮次是怎么结束的：`{"kind", "id", "title", "outcome", "ended_s_ago"}`；还没有结束过任何一个（或读不到任何存储）时为 `null`。`kind` 与 `status.items` 中的相同；`title` 是它的名称（最多 200 个字符，读不到时为空）。`outcome` 为 `done`（运行已完成，或任务已完成）、`stopped`（用户停止了它，或任务被取消或已过期）或 `failed`（运行的其他任何结局，例如 `provider_error`、`budget_exhausted` 或 `stuck`，以及失败的任务）；用词与控制台一致，控制台显示为警告或错误的都是 `failed`。`ended_s_ago` 是它结束至今的秒数，精确到十分之一秒。因关机、崩溃恢复、接管、关闭或挂起而结束的轮次不算结果，已撤销的轮次也不计。它供展示上一次运行结果的客户端使用：macOS 应用中的桌面 Melo 和活动胶囊会读取它。一次调用通过索引读取每个文件中最近更新的几个会话里最新结束的轮次。

## GET /api/activity

今天运行过的内容，供对话旁的面板使用：控制台的对话、调度器的任务，以及命令行文件中的 worker 运行和终端对话（只读，终端对话已脱敏），每个会话一行。带 `session_id` 时跟随一个对话，每一行还列出它的会话请求过的审批。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- "http://127.0.0.1:8401/api/activity?since=$(date -v0H -v0M -v0S +%s)&session_id=742f96513ef3"
```

```json
{
  "since": 1793332800.0,
  "scope": "chat",
  "rows": [
    {"kind": "chat", "id": "742f96513ef3", "title": "add the meeting date to the notes", "state": "waiting",
     "activity": "Waiting for your approval", "icon": "wait", "result": "", "at": 1793399003.68,
     "approvals": [{"id": "8c64f46b27384316a9e407bb52b4e6bc", "what": "Write notes.txt", "answer": "waiting",
                    "at": 1793399003.69}],
     "turns": 1}
  ],
  "more": false,
  "unavailable": []
}
```

不带 `session_id`（或带主对话的）时，同一个调用列出一整天，行例如：

```json
[
  {"kind": "task", "id": "a1b2c3d4e5f6", "title": "Slides for the 3 pm meeting", "state": "running",
   "activity": "Writing slides.md", "icon": "write", "result": "", "at": 1793376000.2, "approvals": [], "level": 2},
  {"kind": "run", "id": "c44733ec77e5", "title": "Check the build", "state": "done",
   "activity": "", "icon": "run", "result": "All 412 tests passed.", "at": 1793370000.5, "approvals": [], "turns": 1}
]
```

- `since`（自纪元起的秒数）是这一天的开始；默认是网关所在电脑的本地午夜，不能是将来的时间（`400 invalid_request`），最多往回一周。
- `session_id`（最多 128 个字符）是面板当前打开的对话。对于侧边对话，`scope` 为 `chat`，行是该对话自己的一行和它启动的任务（任务的 `front_id`）；未知的 id 视为没有任何内容的侧边对话。对于主对话（类型为 `front` 的会话，或 `meta` 里有 `main: true` 的会话），以及不带 `session_id` 时，`scope` 为 `all`，行是一整天的内容，包括 worker 运行和终端对话。
- 一行是会话在 `since` 或之后仍在运行或已结束的最新轮次（仍在运行的轮次一定在内，无论它多早开始），或在那之后有变化或尚未结束的任务。运行中 `state` 为 `running`、`waiting`（有审批在等待）、`held`（调度器暂缓了它的模型调用）、`queued` 或 `paused`，否则为 `done`、`failed` 或 `stopped`。运行中的行说明它在做什么（`activity` 和 `icon`，与 [`status.items`](#get-apihealth) 中的相同）；已结束的行在 `result` 中说明结果，一行，最多 140 个字符（该轮的摘要，否则是回答的第一句，失败时是原因），其 `icon` 是它最后一次工具调用的类别（没有工具调用时为 `done`，或为 `failed`、`stopped`）。`at` 是它结束的时间（运行中的行：开始的时间；任务：最后一次变化的时间），`turns` 是对话当天开始的轮次数，`level` 是任务的优先级（0 到 5）。结果始终是平实的话：运行自己的标记写作 `Stopped before answering` 或 `Ended without an answer`，路径写作 `Made a picture` 或文件名，代码让位给它周围的第一句话（没有时写 `Wrote some code`）。
- `approvals` 按从旧到新列出该行的会话今天请求做的事，最多 20 条：`{id, what, answer, at}`。`what` 是用平实的话描述的调用（`Write slides.md`、`Run the tests`、`Run a command`，从不显示命令行、代码或文本），`at` 是请求的时间，`answer` 为 `waiting`、`allowed once`、`allowed for this chat`、`allowed automatically`、`denied`、`no answer`（超时）或 `withdrawn`（轮次先结束了）。对话或任务中 `waiting` 的请求用 `POST /api/approvals/{id}` 回答，与对话的卡片相同。运行和终端对话的行与其余内容一样经过脱敏。
- 运行中的行在前，其余从新到旧，最多 100 行；有行被省略时 `more` 为 `true`。无法读取的来源不添加任何行，并在 `unavailable` 中写出（`chats`、`tasks`、`runs`、`approvals`）。

## GET /api/settings

来自 `~/.cache/localmelo/config.toml` 的已保存配置（不含 API 密钥值），以及设置向导按各节给出的默认值、后端选项和文件的 revision。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/settings
```

```json
{
  "chat_backend": "mlc",
  "embedding_backend": "none",
  "workspace": "~/localmelo-workspace",
  "sections": {
    "mlc": {"chat_url": "http://127.0.0.1:8400/v1", "chat_model": "qwen3-1.7b", "embedding_url": "", "embedding_model": ""},
    "openai": {"api_key_env": "OPENAI_API_KEY", "api_key_set": false, "chat_model": "gpt-4o", "base_url": ""}
  },
  "defaults": {
    "mlc": {"chat_url": "http://localhost:8400/v1", "chat_model": "qwen3-1.7b", "embedding_url": "http://localhost:8400/v1", "embedding_model": "nomic-embed"},
    "openai": {"api_key_env": "OPENAI_API_KEY", "chat_model": "gpt-4o"}
  },
  "errors": [],
  "choices": {
    "chat_backends": [{"key": "mlc", "label": "MLC (local endpoint)", "kind": "local"}, {"key": "openai", "label": "OpenAI", "kind": "cloud"}],
    "embedding_backends": [{"key": "mlc", "label": "MLC (local endpoint)"}, {"key": "none", "label": "None (no long-term memory)"}]
  },
  "mode": "default",
  "advanced_options": [
    {"option": "--host", "tier": "advanced", "issue": ""},
    {"option": "--retriever", "tier": "experimental", "issue": "#4"}
  ],
  "file_label": "~/.cache/localmelo/config.toml",
  "revision": "5d41402abc4b2a76"
}
```

- `sections` 包含全部四个本地节（`mlc`、`ollama`、`vllm`、`sglang`）和全部四个云端节（`openai`、`gemini`、`anthropic`、`nvidia`）。云端节报告 `api_key_env`（变量名）和 `api_key_set`（该变量在网关的环境中是否非空），从不包含密钥或其遮罩。
- `workspace` 是 `[gateway] workspace`；未设置时为 `~/localmelo-workspace`。
- `revision` 是文件 SHA-256 的前 16 个十六进制字符；文件尚不存在时为 `""`。
- `across_chats` 是 `[memory] across_chats`，即[跨对话记忆](#memory-across-chats)的开关。
- `mode` 是 `[ui] mode`（`default` 或 `developer`；保存的其他值都按 `default` 读取）。`advanced_options` 列出层级表中 `melo` 的高级和实验性选项（实验性选项的 `issue` 是决定它的 issue，其他为 `""`）；控制台在开发者模式下以只读方式显示它们。

## PUT /api/settings

以原子方式把设置保存到 `config.toml`，并应用到新的控制台会话。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X PUT http://127.0.0.1:8401/api/settings \
  -H "Content-Type: application/json" \
  -d '{"chat_backend":"ollama","embedding_backend":"none","workspace":"~/localmelo-workspace","sections":{"ollama":{"chat_url":"http://127.0.0.1:11434","chat_model":"qwen3:8b"}},"revision":""}'
```

- 每个字符串都必须是不含控制字符的单行，`workspace` 为必填项，必须是绝对路径或以 `~/` 开头，并且不能解析为 `/` 或主目录（包括经由符号链接）；否则返回 400 `invalid_request`。`sections` 中不接受 `api_key_set`。
- `mode` 可选：`default` 或 `developer` 会设置 `[ui] mode`（其他值返回 400 `invalid_request`）；省略时保留已保存的模式。模式只改变控制台和 `melo --help` 显示的内容。
- `across_chats` 可选：布尔值，设置 `[memory] across_chats`；省略时保留已保存的值。
- 与磁盘上的文件不匹配的 `revision` 会返回 409 `settings_changed`。两个并发的 `PUT` 请求若读取了同一个 `revision`，两者都可能通过检查；最后写入者胜出，`os.replace` 保证文件不会被撕裂。
- 服务器会重新加载 `config.toml`，因此命令行的 `--host` 和 `--port` 覆盖值从不写入磁盘；它会保留其他节、`extra_body`、`mcp_config`、`request_timeout_s`、`[agent]` 和 `[scheduler]`，校验候选设置（失败时返回 `400 invalid_settings` 并附 `details.errors`），以原子方式写入 `config.toml`，关闭空闲的会话智能体，使下一轮以新设置打开，并返回更新后的 `GET /api/settings` 响应体。

## POST /api/settings/test

用一条消息的对话 ping 检查候选设置；若选择了嵌入后端，还会做一个单词的嵌入 ping，时限为 30 秒。它接受 `PUT /api/settings` 的请求体，但不含 `revision`。

在仍在进行的 security-1b 改动中，经过认证的探测接受明确输入的绝对 HTTP 或 HTTPS URL，且不跟随重定向；重定向即视为探测失败。这些探测可能会连接所提供的主机。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/settings/test \
  -H "Content-Type: application/json" \
  -d '{"chat_backend":"ollama","embedding_backend":"none","workspace":"~/localmelo-workspace","sections":{"ollama":{"chat_url":"http://127.0.0.1:11434","chat_model":"qwen3:8b"}}}'
```

```json
{"ok": true}
```

- 字段或校验错误会先返回 400。探测失败或超时会返回 HTTP 200，并附 `{"ok": false, "error": "..."}`。

## GET /api/settings/key-status

某个环境变量在网关的环境中是否非空。从不返回该变量的值。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/settings/key-status?name=OPENAI_API_KEY'
```

```json
{"set": false}
```

- `name` 必须匹配 `^[A-Za-z_][A-Za-z0-9_]{0,127}$`；其他任何值都是 400 `invalid_request`。

## GET /api/models

列出本地后端的模型（`GET <url>/models`，超时为 5 秒）。在仍在进行的 security-1b 改动中，经过认证的请求可以提供绝对 HTTP 或 HTTPS URL；不跟随重定向。

所提供的主机可能会被连接。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/models?backend=ollama&url=http%3A%2F%2F127.0.0.1%3A11434%2Fv1'
```

```json
{"models": ["qwen3:8b"], "embedding_models": [], "suggested_embedding": null}
```

- `embedding_models` 是名称表明为嵌入模型的列出项，`suggested_embedding` 是优先提供的那个（Qwen3-Embedding 优先），否则为 `null`；设置对话框据此询问 `Remember across chats with <model>?`。

- 不带 `backend` 或 `url` 时，使用已保存的对话后端及其 `chat_url`。
- 云端后端返回 400 `invalid_request`：`Listing models works for local backends only; type the model name.`
- 无法连接的服务器或空的模型列表会返回 502 `backend_unreachable`。

## GET /api/setup/detect

像终端的首次运行一样，在本机寻找模型服务器。它向 `LOCALMELO_DETECT_URLS` 中的每个 URL 发送 `GET <url>/models`（超时 1 秒；默认是 `127.0.0.1` 的 8000、8400、11434 和 30000 端口；值为空则关闭检测），然后向找到的服务器发送 `GET <root>/api/version`，在多个对话模型之间还会读取它的状态路由。

尚未配置时也能使用，不带参数，也不写入任何内容。网页控制台的设置卡片会调用它，并且只在卡片显示时才调用。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/setup/detect
```

```json
{"state": "found", "name": "oMLX", "backend": "vllm", "url": "http://127.0.0.1:8000/v1", "models": ["Qwen3-32B-4bit", "qwen3"], "loaded": ["qwen3"], "suggested": "qwen3", "problem": "", "start": {"name": "oMLX", "commands": ["brew tap jundot/omlx https://github.com/jundot/omlx", "brew install jundot/omlx/omlx", "omlx serve --model-dir ~/models --port 8000"], "note": "Put a model in its own folder of ~/models first.", "guide": "https://github.com/localmelo/localmelo/blob/main/docs/guides/model-servers.md"}}
```

- `state` 是 `found`（有可提供的对话模型）、`needs_key`（某个服务器返回 401 或 403；localmelo 不会向本地服务器发送密钥）或 `none`。
- `name` 是 `Ollama`、`oMLX`（其模型列表把 oMLX 标为所有者）或 `""`。`backend` 是要保存的 `chat_backend`：`ollama`，或 `vllm`；当列表指明是 oMLX 时，`PUT /api/settings` 会把它写成 `omlx`。`state` 不是 `found` 时为 `""`。
- `models` 是按列表顺序排列的对话模型（不含嵌入和重排序模型）。`loaded` 是服务器说已在内存中的模型；只有在多个对话模型之间才会询问，`[]` 也可能表示它没有回答。
- `suggested` 是要提供的模型：唯一的对话模型，或一个已加载的模型。当提供了多个模型而且不知道哪个已加载时为 `null`，客户端必须让用户自己选择。
- 没有任何服务器回应时 `url` 为空。`problem` 说明已回应的服务器为什么不能使用。URL 中的用户名或密码绝不会返回。
- 每个回答都带有 `start`：如何在这台电脑上启动模型服务器，来源与终端的首次运行相同。`name` 在 Mac 上是 `oMLX`，其他系统是 `Ollama`；`commands` 是要运行的命令，`note` 是运行命令之外要做的一件事，`guide` 是模型服务器指南。`state` 为 `none` 时，卡片会显示它们。

## Scheduler tasks (S3a.2)

这些路由在网关进程中创建和控制用户发起的任务。它们与对话轮次分开，但使用同一个调度器闸门。

每个任务都有一个 `kind="task"` 的会话，使用相同的 id；可以通过现有的会话路由检查它的会话事件和轮次。worker 使用内置工具 `file_read`、`file_write` 和 `python_exec`（不含 `shell_exec` 和 `list_files`），并关闭记忆。

工具以用户的权限运行，并非在沙箱中运行。

- `POST /api/tasks` 接受 `objective`（或其别名 `prompt`）、可选的 `title`、`workspace`、`level` 和 `request_key`。objective 必须非空，且不超过 100,000 个字符；title 不超过 200 个字符，默认为从 objective 派生的标题。level 为 0–5 的整数或 `P0`–`P5`；省略时为计算出的 P2。提供的 `deadline`（即使为 null）都会被拒绝；截止时间尚不支持。其他字段也会被拒绝。objective 缺失、为空白或过长、title 过长或 level 无效，都返回 400 `invalid_task`；模式或类型错误返回 400 `invalid_request`。创建新任务时返回 202 `{"task": ...}`。显式为空的 `request_key` 返回 400 `invalid_request`。非空的 `request_key` 使同一规范化请求的重复提交返回 200 `{"task": ..., "repeated": true}`；将其用于不同的请求会返回 409 `submission_conflict`。重复提交不会启动新的 worker。
- `front_id`（可选）是发起该任务的对话：某个旁支对话或主对话的会话 id。任务属于它（即任务的 `front_id`，该对话的面板 Workers 标签页会列出它）；缺失时属于主对话。不指向任何控制台对话的 id（未知 id，或任务的会话）返回 400 `invalid_request`。
- `GET /api/tasks` 返回 `{"tasks": [...]}`；可选的 `state` 和整数 `level` 查询参数用于筛选行。`GET /api/tasks/{task_id}` 返回 `{"task": ...}`，或 404 `not_found`。
- `PATCH /api/tasks/{task_id}` 接受 `level`（0–5、`P0`–`P5` 或 `auto`）和/或 `state`（`pause`、`resume` 或 `stop`）。它返回 `{"task": ...}`。`first` 会被 400 `invalid_request` 拒绝；目前不支持。暂停会把任务停放在 P5；恢复会在暂停前恢复优先级，除非提供了新的可运行 level。保存的优先级会在审计日志保留期后仍然保留。如果没有保存的快照（例如，任务创建时或被设为 P5，或者其日志已丢失的 v1 暂停），恢复会回退为计算出的 P2。`auto` 选择计算出的 P2。暂停不能与 level 同用，恢复也不能与 P5 同用。既没有 level 也没有 state 的更新，或任一无效组合，都返回 400 `invalid_request`。恢复一个未被暂停的任务返回 409 `not_paused`。停止会取消正在运行的 worker，或取消排队中的任务。已结束的任务不能再修改（409 `invalid_transition`）。

暂停会关闭 worker 的围栏，并在一个已在运行的提供方或工具结束时记录待处理的暂停。在持有的计划、反思或未送达的审批处，worker 会停放，关闭其资源，并根据已持久化的证据保存一张卡片。

未送达的审批会被撤回；恢复会从卡片开始，并再次询问。尚未启动的已暂停任务在恢复之前不会启动 worker。

如果无法保存停放状态，worker 会在关闭围栏的情况下重试；三次失败之后，它会保存一个中断的任务和卡片。如果存储无法持久化该结果，恢复和停止会返回 503 `store_unavailable`，直到最终化成功。

资源关闭失败会隔离该 worker，并保持其槽位被计数；重试之后，中断的任务会报告 `cleanup failed; slot held until restart`。

任务对象包含任务 id、规范化的 `spec`、`state`、优先级和来源、`wait_reason`、`pending_approvals`（任务正在等待的批准，形状与会话的相同；没有时为空；与对话一样，用 `POST
/api/approvals/{request_id}` 回答其中一个）、时间戳、工作区和 `priority_events`，以及保留的计量字段。服务、等待和工具计量尚未实现。

明确指定的用户工作区必须是主目录之内的现有目录（严格位于主目录内部），且位于隐藏文件夹和 `~/Library` 之外；默认工作区的选择方式见[调度器](scheduler.md#tasks-workers-and-the-scheduler-store-s3a)。无效的工作区请求返回 400 `invalid_workspace`；在运行或恢复之前变为无效的工作区会使任务以 `wait_reason="invalid_workspace"` 失败，而不是运行它。

同一时间最多常驻四个 worker，包括被隔离的 worker。任务行在网关重启之后仍然保留。

启动时会中断状态不确定的任务，保留有效的已停放卡片，并保留待处理的暂停，但不进行重放。每次创建都会在任何启动之前、在创建事务之内记录一个持久的“从未启动”证明；带有该证明的可运行排队任务，在启动围栏之后可以运行一次。

历史状态不确定的已恢复任务会拒绝 `PATCH` 的恢复和优先级变更，直到使用受控恢复；停止仍然可用。无法持久化启动恢复会使存储保持不可用。

用同一个 `request_key` 重试会返回已有任务，并且不会重新启动它。

### Controlled Resume

`POST /api/tasks/{task_id}/resume` 接受 `operation_id`（必填，非空，最多 200 个字符）和可选的 `confirmation_token`。它是唯一能恢复启动时已恢复任务的路由；`PATCH` 对这类工作保持围栏。

当转换被提交时，它返回 202 `{"task": ..., "operation_id": ...}`。

该路由依据的是证据，而从来不是证据的缺失：

- 存储证明从未启动过的、已停放或已中断的任务，会通过持久的启动围栏排队等待首次启动，不需要卡片，也不需要确认。已排队且从未启动的创建由启动恢复准入，而不是由该路由准入，因此为它调用该路由会返回 `invalid_transition`。
- 已启动的任务需要卡片和有效的工作区。只有当其可信且完整的捕获覆盖了整个执行过程，并且每个已记录的进程组都能被验证为已不存在时，它才会在无需确认的情况下恢复。经过验证的启动变更（两个 id 都已读取、格式正确且不同）只有在记录该进程组的主机上才算已不存在；在另一台主机上读取的变更、没有主机的记录，以及无法读取的主机，都视为未知。
- 其他所有情况，包括缺失、不可信或不完整的捕获，或无法检查存活性的进程组，都返回 409 `confirmation_required`，并附带 `details.confirmation_token`、`details.expires_in_s`（300）、`details.warning` 和 `details.reason`。提交该令牌即接受已记录的风险并恢复。这一被接受的转换会写入一条 `sched.resume_confirmed` 事件；仅预览则不写入任何内容。该令牌只能使用一次，仅保存在内存中，并绑定到任务、源执行、证据标识和 CAS 代次。重启会丢弃它，而停止会使其失效。
- 已结束的任务、无效的工作区、已启动任务缺少卡片，或过时的执行标识，都会返回 409 `invalid_transition` 并附 `details.reason`；缺少卡片时会提示将其作为新任务重新开始。之前未解决的恢复操作，或尚未结算的 worker 清理，会返回 409 `recovery_unknown`。

操作 ID 标识一次操作。使用同一个 `operation_id` 重试，无论携带什么载荷，都会返回 202 和当前任务，并且从不会开始第二个轮次或重新授权一次转换；令牌最多只能被消费一次。

重启会丢弃令牌。在重启之前已提交的恢复，若其轮次从未开始，则会在启动时重新准入；一旦该轮次已经开始，对该操作的重试会返回 409 `invalid_transition`，而不是假装排队中的工作仍在等待。

停止始终优先：它会使待处理的令牌失效，并在 worker 启动提供方或工具之前取消一个已排队的恢复轮次。无论是免确认的恢复还是确认，都不能证明较早的进程已经停止，或其效果只发生了一次。

客户端若按版本号寻址 API v2，也可以通过 `/api/v2/tasks/{task_id}/resume` 访问该路由；其他任务路由使用 `/api` 前缀。

## GET /api/sessions

最新的会话排在最前面。默认只列出 `chat` 会话：即主对话的旁支对话，列表在 `main` 中给出主对话本身。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/sessions?query=notes&limit=50'
```

```json
{"sessions": [{"id": "8983e7f06ecf", "kind": "chat", "main": false, "title": "Plan the Tokyo trip", "created_at": 1793399544.338924, "updated_at": 1793399544.339885, "model": "stub-model", "workspace": "/home/me/demo-project", "workspace_label": "~/demo-project", "busy": false, "running_here": false, "pending_approvals": 0, "level": 2, "held": ""}], "machine": {"lock": "none", "holder": "", "purpose": "", "run": "", "eta": "", "stale": false, "readable": true, "on_ac": true, "battery_pct": null, "battery_state": "", "band": 4, "gate": "record"}, "has_more": false, "main": {"id": "313ad3fcb8fb", "kind": "front", "main": true, "title": "Main chat", "created_at": 1793399544.315079, "updated_at": 1793399544.315079, "model": "", "workspace": null, "workspace_label": "", "busy": false, "running_here": false, "pending_approvals": 0, "level": 2, "held": "", "last_card": {"id": 4, "ts": 1793399544.342355, "source_id": "8983e7f06ecf", "source_kind": "chat", "title": "Plan the Tokyo trip", "state": "done", "line": "Three days in Tokyo, then Kyoto.", "at": 1793399544.339885}}}
```

- **主对话。** 网关的会话文件中恰好有一个主对话：一个类型为 `front`、标题为 `Main chat` 的会话，在控制台第一次打开该文件时创建，早于它的第一条消息。它永远不会被删除。每个类型为 `chat` 的控制台对话都是它的旁支对话；主对话出现之前创建的对话也原样成为它的旁支对话。每个会话对象用 `main` 说明这一点：主对话为 `true`，其他会话为 `false`。
- `main`（仅在默认的 `kind=chat` 下出现）是主对话的会话对象，另加 `last_card`：它最新的一张[卡片](#cards-in-the-main-chat)，没有时为 `null`。只要 `last_card.id` 比客户端上次显示的卡片更新，客户端就把主对话标为未读。只有在存储无法读取时，`main` 才为 `null`。
- `limit` 为 1 到 200（默认 50）。`query` 最多 200 个字符，与会话的标题或任一轮次的用户文本匹配。`has_more` 表示该页之外还有更早的会话。
- 会话在拥有一个轮次之后即被列出，正在运行的轮次也算在内：创建操作一旦保存了第一个轮次的行，会话就会出现。创建操作留下的、从未开始其消息的行（例如中途关机）没有轮次，因此不会被列出；`GET /api/sessions/{id}` 仍会显示它。
- `kind` 为 `chat`（默认）、`front`、`front_bg`、`task`、`worker` 或 `terminal`；其他任何值都是 400 `invalid_request`。`front` 列出主对话，在它的第一条消息之前也会列出。`worker` 和 `terminal` 以 SQLite 只读模式读取 CLI 会话文件：`worker` 列出直接运行，`terminal` 列出交互式 `melo` 会话的对话，`kind=chat` 从不包含它们。CLI 文件不存在时返回空列表，且网关不会创建它。SQLite 可能会在读取期间创建 `-wal` 和 `-shm` 记账附属文件；读取器不以写入方式打开主数据库，也不使用 immutable 模式。
- worker 和 terminal 列表接受精确匹配的 `status`、`source` 和 `parent_run` 筛选条件，以及来自上一个响应 `next_cursor` 的不透明 `cursor`。结果按该行所用轮次的开始时间降序排列，再按会话 id 降序排列。到达末尾时 `next_cursor` 为 `null`；翻到后续页时，应保持相同的筛选条件和查询。无效的游标，或在没有 `kind=worker` 或 `kind=terminal` 的情况下使用这些参数，都返回 400。分页结果不是快照：并发的变更可能影响后续页面。
- worker 的列表和详情会话对象包含可为空的 `source`、`parent_run`、`git_branch`、`git_head`、`exit_code`、`status`、`steps`、`tokens`、`started_at` 和 `duration_s`。开始时间是第一轮次的时间戳；存在时，持续时间为其记录的结果持续时间。`tokens` 是整数的总 token 数，`meta.usage` 保存完整的用量映射。状态回退为第一轮次的状态；步骤数和 token 数回退为其保存的结果。缺失的值为 `null`，而不是零。详情响应顶层的 `tokens` 单独统计 completion token；对于 worker，若没有任何轮次带有整数的 `completion_tokens`，则为 `null`。
- 终端对话和 worker 运行一样以只读方式提供，列表和详情对象中的 `kind` 为 `terminal`。该行所用的轮次是对话的最新轮次，因此 `status`、`started_at`、`steps` 和 `duration_s` 描述的是那一轮，正在进行的对话排在最前。详情、轮次（包括 `events=true`）、`/events` 和 `/status` 与 worker 运行一样可以按 id 获取；当拥有该对话的终端持有有效租约并正在运行一个轮次时，`busy` 为 true。终端以未脱敏的方式保存对话，所以对话的每一段文本在读取时都会脱敏：标题、`meta`、轮次文本、回答、结果和事件载荷（包括工具输出和 shell 输出）中的密钥都变为 `[redacted]`，读取过 `.env` 之类密钥文件的调用，其输出被替换为一条说明，运行的 `config` 被省略。worker 运行是脱敏后保存的，按保存的内容读取。脱敏基于模式，不保证不含密钥；图片不会被脱敏。对于终端对话的 id，每个写入路由都返回 404，与 worker 的 id 相同。
- worker 会话是只读的。它们的会话详情和轮次（包括 `events=true`）、已保存的图片 blob、`/events` 和 `/status` 都可以通过 id 获取；写入路由返回 404（`POST /api/turns/{turn_id}/cancel` 返回 409 `turn_not_running`：worker 运行的轮次不在网关中运行）。对于只在 CLI 文件中找到的 worker id，`/tools`、`/memory`、`/runs` 以及限定会话的提交查询也返回 404 `not_found`。如果一个 id 同时存在于两个文件中，则网关会话优先。worker 会话对象会增加 `meta`，其中包含已保存的来源信息，它们的 `running_here` 始终为 false。`busy` 反映一个未过期的 worker 租约；网关不会恢复陈旧的轮次，也不会对 CLI 文件应用保留策略。
- `workspace_label` 是工作区，并把主目录写作 `~`。
- `busy` 表示有轮次正在运行。对于网关会话，`running_here` 等于 `busy`：只有本网关运行其自身会话文件中的轮次。
- `pending_approvals` 统计该会话中仍在等待答复的审批卡片数。
- `level` 和 `held` 是调度器对该对话的视图：它的 P1 到 P4（在芯片设置之前为 P2），以及它的模型调用正在等待的原因（`Resources`、`Priority`、`Reservation`、…），没有等待时为 `""`。它们与网页工作区标题栏显示的是同一批事实；见[调度器](../guides/scheduler.md#the-gate-on-chats-s2)。
- `machine` 是唯一闸门对机器的快照：GPU 锁的 `lock`（`none`、`self`、`foreign`）、`holder`、`purpose`、`stale`、`readable`、`on_ac`、`battery_pct`、`battery_state`、`band`，以及 `gate`（`record` 或 `wait`）——即页面机器行所读取的对象；见[GPU 锁与电池](../guides/scheduler.md#the-gpu-lock-and-the-battery-s2b)。它从不为 `null`：控制台总是有一个闸门。

## GET /api/sessions/{id}

一个会话，它的待处理审批、会话授权，以及输出 token 总数。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e
```

```json
{
  "session": {"id": "3f9c2a7b1d4e", "kind": "chat", "main": false, "title": "Release notes", "created_at": 1761141600.0, "updated_at": 1761141612.4, "model": "stub-model", "workspace": "/home/me/demo-project", "workspace_label": "~/demo-project", "busy": false, "running_here": false, "pending_approvals": 0, "level": 2, "held": ""},
  "pending_approvals": [{"request_id": "0d6c2a9e8f7b41c3a5e9d1b3f5a7c9e1", "call_id": "call_1", "tool_name": "file_write", "arguments": {"path": "/home/me/demo-project/todo.md"}, "arguments_complete": true, "preview": "~/demo-project/todo.md", "workspace_root": "/home/me/demo-project", "workspace_label": "~/demo-project", "target": {"path_label": "todo.md", "exists": false, "size_bytes": 0}}],
  "grants": ["file_write"],
  "granted_at": {"file_write": 1761141604.9},
  "mcp_error": "",
  "agent": "open",
  "tokens": 18512
}
```

- `grants` 是本会话剩余部分允许使用的工具；`granted_at` 把每一项映射到授权开始的时间。
- `mcp_error` 说明本会话为何在没有 MCP 工具的情况下运行，否则为 `""`。目前它始终为 `""`：控制台尚未实现 MCP。
- `agent` 为 `open`、`paused`（已关闭，但保留其对话）或 `not_loaded`（本网关尚未打开的会话，例如重启之后）。创建操作即使没有轮次也会打开会话；读取路由从不打开会话，而轮次或重试会打开。因此，`not_loaded` 会持续到网关启动后的第一次打开，而已关闭的智能体读作 `paused`。
- `tokens` 汇总该会话已结束轮次的 `completion_tokens`，与回答所报告的字段相同。
- `pending_approvals` 条目的 `arguments_complete` 仅当 JSON 安全的 `arguments` 保留了原始值、未经截断或转换时才为 true。当它为 false 或缺失时，客户端不应允许该请求；拒绝仍然可用。仅审批事件中的参数可能被截断，因此在允许之前请获取会话详情。
- `pending_approvals` 条目的 `review` 是 melo 的规则在 `shell_exec`、`python_exec` 或 `file_write` 调用中发现的内容，其他工具为 `null`：`verdict` 为 `routine`、`high` 或 `undecided`；`title` 是 `high` 调用的卡片问题（“Install software?”），其他为 `""`；`reasons` 是最多三句说明 melo 为何询问的话（“Installs playwright with pip into .signin_venv (downloads about 42 MB).”），其中的密钥已被遮盖。`high` 调用每次都会询问，卡片不提供 `session` 回答。
- `pending_approvals` 条目的 `target` 仅为 `file_write` 设置：路径 `path_label`（相对于工作区；否则以 `~` 代表主目录）、文件是否 `exists`，以及其 `size_bytes`（在生成响应时读取）。其他工具的 `target` 为 `null`。
- 对已有文件写入文本的 `file_write`，其 `target` 还带有 `diff`，即这次写入会替换的内容；它在生成响应时读取，显示方式与终端的审批面板相同。新文件或没有文本的调用，`target` 没有 `diff`。

  ```json
  {"path_label": "todo.md", "exists": true, "size_bytes": 86, "diff": {"added": 1, "removed": 1, "rows": [{"kind": "context", "line": 2, "text": ""}, {"kind": "context", "line": 3, "text": "- [x] Read data.csv"}, {"kind": "removed", "line": 4, "text": "- [ ] Add a --json flag to app.py"}, {"kind": "added", "line": 4, "text": "- [x] Add a --json flag to app.py"}, {"kind": "context", "line": 5, "text": "- [ ] Test empty files"}], "note": ""}}
  ```

  - `added` 和 `removed` 统计行数。`rows` 是有变化的行，每侧各带两行上下文，相距较远的变化之间有一个 `gap` 行。`kind` 为 `removed`、`added`、`context` 或 `gap`；`line` 对 `removed` 行是旧行号，对其他行是新行号，`gap` 行为 `null`；`text` 是不含行尾的该行文本，从不截断。
  - 每一行中的密钥都会脱敏，与已保存的事件相同：密钥读作 `[redacted]`。各行取自文本本身，所以只有密钥不同的一行，仍是一个 `removed` 行和一个 `added` 行。
  - 差异要么完整，要么没有。melo 无法比较时，`rows` 为空，`note` 用文字说明原因：文件不是文本、无法读取、不是常规文件或超过 200 KB（200,000 字节）；新文本超过 200,000 个字符；会有超过 1,000 行变化；或者文件或新文本中含有私钥。其他情况下 `note` 为 `""`，`rows` 为空表示文件已经是这段新文本。
  - `diff` 仅用于显示。审批的其他字段、回答及其规则与没有它时相同。

## GET /api/sessions/{id}/turns

会话的轮次，在页内按从旧到新排列。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/turns?limit=10&events=true'
```

```json
{"turns": [{"id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "turn": 3, "run_id": "4b1e7c9a2f3d4e5b8a6c0d1e2f3a4b5c", "user_text": "Summarize these notes", "status": "completed", "cause": "run", "answer": "The notes say ...", "result": {"task_id": "4b1e7c9a2f3d4e5b8a6c0d1e2f3a4b5c", "status": "completed", "result": "...", "error": "", "steps": 3, "attempts": 1, "n_llm": 3, "n_main": 3, "n_aux": 0, "n_tool": 4, "llm_errors": 0, "usage": {"prompt_tokens": 1180, "completion_tokens": 24, "total_tokens": 1204}, "duration_s": 12.2, "config": {}}, "undone": false, "started_at": 1761141600.2, "finished_at": 1761141612.4, "client_request_id": null, "running_here": false, "events": [{"id": 790, "session_id": "3f9c2a7b1d4e", "turn": 3, "v": 1, "type": "turn.started", "run_id": "", "seq": null, "ts": 1761141600.2, "data": {"turn": 3, "turn_id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "user_text": "Summarize these notes", "client_request_id": null}}]}], "has_earlier": false, "replay_from": 789}
```

- `limit` 为 1 到 50（默认 10）。不带 `before` 时，该页是最新的轮次，`replay_from` 是最旧返回轮次的 `turn.started` 之前的那个事件 id；会话没有轮次时为 0。对于网关会话，把它作为事件流的 `after` 传入，即可回放该页的事件。对于 worker 会话，`events=true` 是一次读取；`/events` 则会跟随它。
- 使用 `before=N`（1 到 SQLite 的最大整数）时，该页是编号低于 `N` 的最新轮次，并且没有 `replay_from`。
- `has_earlier` 表示该页之外还有更早的轮次。
- `events=true` 会把每个轮次已保存的事件作为 `events` 添加进来，按 id 排序，与事件流的 `data` 帧完全一致。
- 对于被 `Try again` 撤销的轮次，`undone` 为 `true`；此类轮次仍会被列出。
- `answer` 和 `result.result` 是保存的副本：密钥已脱敏，并且在保留期之后，内联的 `<think>` 块已被删除。
- 轮次运行期间，`cause` 为 `""`。
- 主对话的每一页另带 `cards`，即这一页轮次旁边的卡片，按从旧到新排列：从该页最旧的轮次（最旧的一页则从头开始）到下一页的第一个轮次（最新的一页则到现在为止），最多取最新的 50 张。没有轮次的主对话给出最新的 50 张卡片，`replay_from` 位于其中最旧一张之前，因此事件流不会回放更早的内容。

<a id="cards-in-the-main-chat"></a>

### 主对话中的卡片

旁支对话的一次运行结束，或任务完成、失败或等待审批时，主对话会得到一张卡片：一个不属于任何轮次的 `attention.card` 会话事件。卡片从来不是轮次：没有模型会读取它，主对话的对话内容也从不包含它。

它的 `data`：

```json
{"source_id": "8983e7f06ecf", "source_kind": "chat", "title": "Plan the Tokyo trip", "state": "done", "line": "Three days in Tokyo, then Kyoto.", "at": 1793399544.339885}
```

- `state` 为 `done`、`failed` 或 `waiting`（有审批在等待答复）。`source_kind` 为 `chat` 或 `task`，`source_id` 是要打开的旁支对话或任务。`line` 是一行平实的话：回答的第一句、运行结束的方式，或 `Waiting for your approval`。`at` 是事情发生的时间。
- 旁支对话的轮次由其运行结束时（`cause` 为 `run`）才算；本人停止、关机或崩溃都不产生卡片。任务以 `done`、`failed` 或 `expired` 结束时才算；本人停止不产生卡片。
- 网关在读取主对话或列表时写入卡片，另外每分钟一次；审批在此之前已答复的等待卡片永远不会写入。主对话的事件流以 `attention.card` 帧送来新卡片。
- 通过 `POST /api/tasks` 创建的任务属于它的 `front_id` 所指的对话，没有时属于主对话；它的卡片仍然来到主对话。

<a id="echoes-in-the-main-chat"></a>

### 主对话中的回执行

主对话对任务做了什么，是主对话里一个不属于任何轮次的 `attention.echo` 会话事件。由代码处理的消息（`/task TEXT`、带 `"as_task": true` 的轮次请求体，或点名一个未结束任务的简短控制短语）不开始轮次：`POST /api/sessions/{main}/turns` 以 200 返回该回执。

```json
{"echo": {"id": 41, "ts": 1793405483.9, "echo": "started", "task_id": "3f9c2a71b0de", "title": "summarize the readme", "level": 2, "reason": "you asked for it", "line": "Started: summarize the readme · Normal", "typed": "/task summarize the readme"}}
```

- `echo` 为 `started`、`control`（带 `action` 和 `undo`）、`level`（带 `undo`）、`suggest`（本人的话没有要求的改变，带 `action`，可能带 `level`）、`ask`（`Start this as a task?`，带完整消息 `text`、键入的文字 `words` 和 `turn_id`）、`undone` 或 `failed`。`line` 是平实的话；`typed` 是它回应的消息；`of` 是它回应的回执。
- 在主对话以外的会话上使用 `as_task` 返回 400 `invalid_request`。委派时重复的 `client_request_id` 返回同一条回执。
- 主对话的每一页另带 `echoes`，分页方式与 `cards` 相同。
- `POST /api/front/echoes/{id}` 带 `{"do": "undo" | "apply" | "start" | "level", "level"?: "Today" | "P1" | 1}`，返回 200 `{"echo"}`：撤销 `control` 或 `level` 回执（停止的撤销会以新任务重新开始它）、应用 `suggest`、开始 `ask`，或把 `started` 或 `level` 回执的任务设为本人的级别。撤销、应用和开始只答复一次（409 `already_answered`）；未知的回执或任务为 404。

## GET /api/sessions/{id}/events

会话事件的 Server-Sent Events 流：先是起点之后已保存的事件，然后是实时事件。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -N 'http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/events?after=789'
# catch up once and stop:
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/events?after=789&follow=false'
```

```text
retry: 2000

id: 790
data: {"id":790,"session_id":"3f9c2a7b1d4e","turn":3,"v":1,"type":"turn.started","run_id":"","seq":null,"ts":1761141600.2,"data":{"turn":3,"turn_id":"9c1f0e2d4b6a48c1a3e5f7091b2d4c6e","user_text":"Summarize these notes","client_request_id":null}}

: keepalive
```

- 每个帧有一行 `id:` 和一行 `data:`；没有 `event:` 字段。JSON 是事件信封，外加其持久的 `id`、`session_id` 和 `turn`。
- `retry: 2000` 只发送一次。每当静默 15 秒，就发送一次 `: keepalive`。
- `after` 默认为 0。当两者同时存在时，`Last-Event-ID` 请求头优先；这正是浏览器的 `EventSource` 在重连时发送的内容；此时无效的 `after` 不会被读取。否则它必须是有符号 64 位整数范围内的十进制数字：负数、`+1`、`1_0`、非数字以及过大的值，都是 400 `invalid_request`。
- `follow=false` 发送起点之后已保存的事件，直到该请求读取到的最新事件 id，然后结束响应；它是一次性的追赶，而在它分页期间开始的轮次不会增加任何帧。
- 当客户端断开连接、会话不再存在（此时重连会得到 404），以及网关关闭时，流即结束。浏览器使用 `Last-Event-ID` 重连，并且恰好一次地收到它错过的每个事件。
- worker 运行（CLI 文件中的会话）从 CLI 文件提供，帧、`after`、`follow` 和 `Last-Event-ID` 都相同。该文件由另一个进程写入，因此跟随的流约每秒读取一次，发送带 id 的新事件，并在运行不再持有有效租约且最后一个事件已发送后结束（会话对象中 `busy` 为 false）。没有有效租约的运行（例如崩溃后留下的）会被回放并立即结束。保活与对话相同；每个打开的流是该文件的一个读取者，客户端离开时关闭。
- 终端对话以同样方式流式提供，并经过脱敏。当它的某个轮次在终端的租约下运行时它处于活动状态，因此该轮次结束时流也结束；请再次读取会话对象，查看是否有另一个轮次在运行。

## GET /api/sessions/{id}/status

终端的 `/status` 和状态栏为一个对话、任务或 worker 会话显示的数字，由服务器用终端自己的代码对该会话已保存的事件折叠得出，因此两边一致。这些段是终端开发者模式的紧凑写法；终端的默认模式用文字表述同样的数字。

worker 运行或终端对话从 CLI 的文件读取。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/status
```

```json
{
  "session_id": "3f9c2a7b1d4e",
  "local": true,
  "segments": ["cache 67% idle 3m", "ctx 1.0k/8.0k", "step 3", "tasks 1 running 2 queued"],
  "cache": {"requests": 2, "last_percent": 67, "session_percent": 56, "idle_s": 200.0},
  "context": {"measured": 1000, "limit": 8000},
  "steps": {"active": true, "step": 3, "attempt": 1, "phase": ""},
  "tasks": {"running": 1, "queued": 2, "held": 0, "top_level": 1}
}
```

- `segments` 是终端状态栏在开发者模式下从左到右绘制的内容：缓存、上下文、步骤、任务。没有数据的段会被省略。
- 服务器没有报告缓存 token 时，`cache.last_percent` 和 `cache.session_percent` 为 `null`，该段显示 `cache --`：绝不会显示 0%。`idle_s` 只在本地后端设置（本地服务器的缓存因内存而被淘汰，而不是因时间，所以没有倒计时）。
- 在首次测量或一次运行的首个进度事实之前，`context` 和 `steps` 为 `null`。运行期间 `steps.active` 为 true。`tasks` 是任务存储中 running、queued 和 held 的数量，以及其中最高的优先级（`top_level`，没有任务时为 `null`）。
- 折叠结果按会话缓存在其最后一个事件 id 上：没有新事件的读取不会读取任何事件。

## GET /api/sessions/{id}/blobs/{sha256}

轮次中某个工具返回的图片，通过其 SHA-256 摘要获取。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -o shot.png http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/blobs/823ceb99fcef5252333ede1b2202341c3b287b6d47571963e6b0ddf393a24f82
```

- `sha256` 必须恰好是 64 个小写十六进制字符；其他任何值、未知的摘要和未知的会话都是 404 `not_found`。
- `image/png`、`image/jpeg`、`image/gif` 和 `image/webp` 以其类型内联提供。其他任何已保存的类型（包括 SVG）都以 `application/octet-stream` 提供，并附带 `Content-Disposition: attachment; filename="<sha256>.bin"`。
- 每个响应都带有 `X-Content-Type-Options: nosniff`、一个 `Content-Security-Policy` 响应头（其值为 `default-src 'none'; sandbox; frame-ancestors 'none'`，网关的 framing 规则会合并进 blob 自身的策略）以及 `Cache-Control: no-store`。

## GET /api/sessions/{id}/pictures

会话所在文件夹里的一个图片文件，按路径获取。控制台用它显示一轮产生的图片（[`turn.pictures` 事件](events.md#session-events)列出它们）。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -G --data-urlencode 'path=charts/fcf.png' -o fcf.png http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/pictures
```

- `path` 必填，最多 4,096 个字符：相对于会话文件夹的路径，或位于其中的绝对路径。路径中任何位置的 `..` 都会被拒绝。未知的会话、还没有文件夹的会话和不存在的文件都是 404 `not_found`。
- 文件的真实路径（跟随所有链接后）必须位于会话的真实文件夹内，并会在已打开的文件上再检查一次；否则返回 403 `outside_workspace`（带 `..` 的路径、其他位置的绝对路径、指向外部的链接、另一个会话的文件夹）。只提供普通文件。
- 名称必须以 `.png`、`.jpg`、`.jpeg`、`.gif`、`.webp` 或 `.svg` 结尾，内容也必须是这类图片（根据开头几个字节判断；`Content-Type` 取自内容而不是名称）；否则返回 415 `not_a_picture`（名为 `chart.png` 的文本文件，或实为 HTML 页面的 SVG）。超过 20 MB 的文件返回 413 `too_large`。
- 每个响应都带有 `Content-Disposition: inline`、`X-Content-Type-Options: nosniff` 和 `Cache-Control: private`。PNG、JPEG、GIF 或 WebP 带有 `Content-Security-Policy: default-src 'none'; sandbox; frame-ancestors 'none'`；SVG 带有 `Content-Security-Policy: sandbox; default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'`，因此它可以绘制，但绝不会运行脚本，也不会作为页面打开。该路由只读取一个文件，不写入也不列出任何内容。

## GET /api/sessions/{id}/submissions/{client_request_id}

客户端用 `client_request_id` 命名的消息最终如何。不带会话的同一查询为 `GET /api/submissions/{client_request_id}`，供从未得知新会话 id 的客户端使用。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/submissions/8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07
```

```json
{"submission": {"client_request_id": "8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07", "state": "accepted", "session_id": "3f9c2a7b1d4e", "turn_id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "status": "completed", "cause": "run", "turn": {"id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "turn": 3, "...": "..."}, "error": "", "created_at": 1761141600.2}}
```

- `client_request_id` 为 1 到 128 个 `[A-Za-z0-9._:-]` 字符，以字母或数字开头。格式错误的会返回 400 `invalid_request`。
- `state` 为 `accepted`、`withdrawn` 或 `unknown`。未知的键以 200 返回，其中 `state: "unknown"`、`error: ""`，其他所有键为 `null`：用它重试是安全的。这里的 404 表示会话未知或已删除，绝不表示该键未知。
- 对于已接受的提交，`turn_id`、`status`、`cause` 和 `turn` 属于其轮次。对于已撤回的提交，它们为 `null`，而 `error` 是存储的撤回原因。
- 会话路由同样能找到创建该会话第一轮次的那个键，因此丢失了创建请求响应的客户端，可以向任一路由查询。

## Writes

下列路由会改变某些状态。多数接受 JSON 请求体；`POST /api/turns/{turn_id}/cancel` 和 `DELETE /api/sessions/{id}` 不接受请求体，若提供则忽略。

从浏览器发起时，写操作必须是同源的：来自其他来源的、会改变状态的请求会返回 403 `foreign_origin`。请求体是严格的：未知字段会返回 400 `invalid_request`，并指出该字段（`body.model: Extra inputs are not permitted`）。

除上面的通用错误码外，以下错误码会出现在写入路由上：

| 状态 | `code` | 何时 |
|---|---|---|
| 400 | `invalid_workspace` | 文件夹不是绝对路径或 `~/` 开头，已不再是目录，或者无法创建默认工作区 |
| 400 | `text_rejected` | 检查器拒绝了消息文本，与 `/v1/agent/run` 请求的处理方式相同 |
| 409 | `session_busy` | 该会话中已有轮次正在运行 |
| 409 | `not_latest_turn` | 重试所指的轮次不是最新已结束的那一个 |
| 409 | `submission_conflict` | 该 `client_request_id` 已用于另一条消息 |
| 409 | `submission_withdrawn` | 该 `client_request_id` 在开始之前已被取消 |
| 409 | `turn_not_running` | 取消所指的轮次并未在运行 |
| 409 | `already_answered` | 该审批已经答复 |
| 502 | `agent_start_failed` | 会话的智能体无法启动；消息为构建器的错误 |

<a id="what-a-turn-answers-with"></a>

### 轮次的回答内容

带第一轮次的创建，以及两个轮次路由，都会返回该轮次的行和三个恢复键：

```json
{"turn_id": "9a840f8444e1452a9a25e5b6b804b20b", "turn": {"id": "9a840f8444e1452a9a25e5b6b804b20b", "turn": 1, "run_id": "", "user_text": "list the folder", "status": "running", "cause": "", "answer": "", "result": null, "undone": false, "started_at": 1793163770.17, "finished_at": null, "client_request_id": "curl-turn-1", "running_here": true}, "resumed_turns": null, "resume_mode": null, "resumed_checkpoint_turns": null}
```

- 轮次立即开始，并在后台运行：只要其行存在，回答即为 202；当创建操作启动了它时为 201。请通过事件流跟踪它——`turn.started`、运行的事件、其 `run.progress` 事件、带有 `status` 和 `cause` 的 `turn.finished`——并从轮次行中读取回答。
- `resumed_turns` 是在本请求打开会话时，智能体从会话文件中加载的更早轮次数；`resume_mode` 是加载它们所用的方式（`pairs` 或 `window`）；`resumed_checkpoint_turns` 是其中来自窗口检查点的数量。当会话已经打开、会话是从本网关保存在内存中的对话重新打开（此时不从文件读取任何内容），以及会话没有更早的轮次时，这三项都为 `null`。

## POST /api/sessions

创建一个会话，通常同时带上它的第一个轮次。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"workspace": "~/demo-project", "turn": {"text": "list the folder", "client_request_id": "8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07"}}'
```

```json
{"session": {"id": "3f9c2a7b1d4e", "kind": "chat", "main": false, "title": "list the folder", "created_at": 1793163786.21, "updated_at": 1793163786.21, "model": "qwen3:8b", "workspace": "/home/me/demo-project", "workspace_label": "~/demo-project", "busy": true, "running_here": true, "pending_approvals": 0, "level": 2, "held": ""}, "turn_id": "9a840f8444e1452a9a25e5b6b804b20b", "turn": {"id": "9a840f8444e1452a9a25e5b6b804b20b", "...": "..."}, "resumed_turns": null, "resume_mode": null, "resumed_checkpoint_turns": null}
```

- 创建得到的是[主对话](#get-apisessions)的一个旁支对话：`kind` 为 `chat`，`main` 为 false。主对话已经存在；向它发消息请用 `POST /api/sessions/{id}/turns`。
- `workspace` 是可选的。缺失或为 `null` 时为 `~/localmelo-workspace`，不存在时以权限 `0700` 创建。其他值必须是绝对路径或以 `~/` 开头，并且是现有目录；其他任何情况都是 400 `invalid_workspace`，且不会创建智能体。
- `turn` 是可选的。不带它时，回答是仅含 `session` 的 201，会话等待 `POST /api/sessions/{id}/turns`。
- `session.model` 是本请求中设置的对话模型。第一条消息的文本会成为标题：空白被折叠，最多 60 个字符，超出时以 `…` 结尾。
- `client_request_id` 使创建可以安全地重复。相同的键加相同的文本，返回 200，附带它所创建的会话和轮次以及 `"repeated": true`，且不再构建任何东西；文本不同则为 409 `submission_conflict`；在开始之前已撤回的键则为 409 `submission_withdrawn`。两个针对同一个键的竞争创建只会生成一个会话。
- `attachments` 目前必须为空：任何条目都是 400 `invalid_request`，即 `attachments: files are not supported yet; send the text alone`。文本由网关入口检查器检查；被拒绝则为 400 `text_rejected`。

## POST /api/sessions/{id}/turns

在已有会话中开始一个轮次。请求体与创建时的 `turn` 相同；此处没有 `workspace`——智能体使用的是会话自身的文件夹。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/turns \
  -H 'Content-Type: application/json' \
  -d '{"text": "list the folder", "client_request_id": "curl-turn-1"}'
```

- 返回 202，附带轮次和恢复键。当本次调用必须从会话文件重建智能体时，恢复键会被填充：即本网关的任何路由都未打开过的会话上的第一个轮次，例如重启之后的第一个轮次。本网关关闭并保存在内存中的会话，会从该对话重新打开，并报告 `null`。
- 任务会话返回 409 `task_session_read_only`；请使用任务路由，而不是开始一个对话轮次。
- 轮次运行期间返回 409 `session_busy`。每个会话同时只运行一个轮次；第二个请求会在任何东西进入队列之前被拒绝。
- 重复的 `client_request_id` 会以 200 返回该键已保存的轮次和 `"repeated": true`，无论该轮次是否仍在运行。
- 会话的文件夹必须仍然存在：文件夹被删除则返回 400 `invalid_workspace`，且轮次不会开始。

## POST /api/sessions/{id}/retry

再次运行最新已结束轮次的文本。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/retry \
  -H 'Content-Type: application/json' -d '{"turn_id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e"}'
```

- 该轮次被标记为已撤销——并发出一个 `turn.undone` 事件——它的文本会开始一个新的轮次，且没有 `client_request_id`。回答为 202，附带新的轮次，以及重放撤销所留下内容的那次打开的恢复键。
- 任务会话返回 409 `task_session_read_only`；它不能作为对话重试。
- 除非该 id 是最新的、已结束且未撤销的轮次，否则返回 409 `not_latest_turn`：丢失了某个轮次结束信息的客户端，应重试它所持有的那个 id。轮次运行期间返回 409 `session_busy`。

## POST /api/turns/{turn_id}/cancel

根据事件给出的 id 停止一个正在运行的轮次。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/turns/9c1f0e2d4b6a48c1a3e5f7091b2d4c6e/cancel
```

```json
{"turn_id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "cancelling": true}
```

- 本网关运行该轮次时返回 202；对于已结束的轮次，或本网关不运行的轮次，返回 409 `turn_not_running`。
- 轮次在其下一个边界处停止，并以 `cancelled` 结束，原因为 `user_stop`；`turn.finished` 会说明这一点。若它没有说任何话，其保存的 `answer` 为 `[interrupt: no assistant output for this turn]`。
- 在审批等待期间取消，会撤回该卡片：请求会被答复为 `withdrawn`，并离开 `pending_approvals`。

## POST /api/sessions/{id}/submissions/{client_request_id}/cancel

通过发送消息时使用的键取消该消息。`POST /api/submissions/{client_request_id}/cancel` 与此相同，但不带会话，供回答已丢失的创建操作使用。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/submissions/8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07/cancel \
  -H 'Content-Type: application/json' -d '{"text": "list the folder"}'
```

```json
{"submission": {"client_request_id": "8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07", "state": "withdrawn", "session_id": "3f9c2a7b1d4e", "turn_id": null, "status": null, "cause": null, "turn": null, "error": "Request cancelled before acceptance.", "created_at": 1793163793.49}, "cancelling": false}
```

- 请求体的 `text` 是消息的文本。对已保存的键使用另一段文本会返回 409 `submission_conflict`。
- 不带会话保存的撤回——即不带 id 的路由——其 `"session_id": null`。
- 当该键的轮次已被接受且仍在运行时，返回 202 并附带 `"cancelling": true`：该轮次会按上述方式停止。`submission` 是本次调用读取到的键，因此其中的轮次仍为 `running`：它会在回答之后结束。请跟踪其 `turn.finished`（原因为 `user_stop`），或再次查询以获知结束。
- 其他情况返回 200。已被接受且已经结束的轮次按原样返回。没有提交的键会被记录为已撤回，因此之后用它的创建或轮次会返回 409 `submission_withdrawn`；重复调用会返回相同的撤回结果。
- 键按其作用域判断：会话读取它自己的键，以及创建它的那个创建作用域键。不带会话保存的撤回会阻止使用该键的创建，而不会阻止某个会话中的轮次。

## POST /api/approvals/{request_id}

答复一张审批卡片。该 id 来自事件流中的 `approval.requested`，或来自 `pending_approvals`。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/approvals/386183e1811b4ba68c3768bfb96ac8e6 \
  -H 'Content-Type: application/json' -d '{"choice": "session"}'
```

```json
{"request_id": "386183e1811b4ba68c3768bfb96ac8e6", "outcome": "allowed", "remember": true}
```

- `choice` 为 `once`（仅本次调用）、`session`（本次调用及本会话中该工具之后的每一次常规调用）或 `deny`。对 `high` 调用（见 `GET /api/sessions/{id}` 的 `review`），`session` 只允许这一次调用：回复为 `"remember": false`，不会产生授权。
- `reason` 只属于 `deny`，最多 500 个字符；与其他 choice 一起使用的 reason 会返回 400 `invalid_request`。
- 未知或已过期的请求返回 404 `not_found`；已被答复的请求返回 409 `already_answered`，因此两个客户端竞争同一张卡片时，只会得到一个答复。
- 审批保存在网关的内存中。网关停止时，等待某个审批的轮次会以 `cancelled` 结束，原因为 `shutdown`，且其卡片被保存为已撤回。

## PATCH /api/sessions/{id}

重命名会话、把它移动到另一个文件夹，或两者同时进行。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X PATCH http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e \
  -H 'Content-Type: application/json' -d '{"title": "Release notes"}'
```

```json
{"session": {"id": "3f9c2a7b1d4e", "kind": "chat", "title": "Release notes", "...": "..."}}
```

- 任务会话对任一变更都返回 409 `task_session_read_only`。
- `title` 会被去除首尾空白，且必须为 1 到 200 个字符；其他长度返回 400 `invalid_request`。重命名的标题永远不会被自动标题替换。
- `workspace` 会移动会话：在新文件夹中，用会话的对话构建一个新的智能体，并且会话的工具授权会被遗忘。规则与创建时相同：绝对路径或 `~/`，且为现有目录，否则返回 400 `invalid_workspace`。轮次运行期间返回 409 `session_busy`；被拒绝的移动会使该行保持原样。
- 两项变更之后，回答为 200，附带会话。

## DELETE /api/sessions/{id}

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X DELETE http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e
```

- 任务会话返回 409 `task_session_read_only`；它不能作为对话被删除。
- 主对话返回 409 `main_chat`（`The main chat is always there: it cannot be deleted.`），并保持原样。
- 204 且无响应体：该会话的行、轮次、事件和 blob 都已删除，它在记忆对话索引中的条目也已删除。melo 在这个对话里产生的笔记会保留（笔记比对话活得久，之后显示为 `from a deleted chat`）；`?notes=forget` 会一并忘记。`notes` 的其他取值返回 400。
- 在仍在进行的 security-1b 改动中，轮次运行或启动排队期间，409 `session_busy` 会拒绝删除；它不会取消该轮次，也不会删除其行。请取消或等待它结束，然后重试。持有会话锁的维护操作也可能短暂地返回该错误，即使没有轮次在运行；请等待后重试。
- 未知会话返回 404 `not_found`。

## POST /api/sessions/{id}/level

设置对话的调度器 level，P1 到 P4。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/level \
  -H 'Content-Type: application/json' -d '{"level": 1}'
```

```json
{"session_id": "3f9c2a7b1d4e", "level": 1}
```

- `level` 是 1 到 4 的 JSON 整数。字符串或布尔值是 400 `invalid_request`（`body.level: Input should be a valid integer`）；`0` 和 `5` 是 400，并分别附带 `body.level: Input should be greater than or equal to 1` 和 `body.level: Input should be less than or equal to 4`。P1 是紧急通道，P4 是后台通道；见[调度器](../guides/scheduler.md#the-gate-on-chats-s2)。
- 该路由从不打开会话：level 是调度器的事实，而不是智能体的事实。未知会话返回 404 `not_found`，与下面每个 `/api/sessions/{id}` 之下的读取路由相同。
- level 保存在内存中，仅限本网关的运行期间，不写入会话文件；下一次 `GET /api/sessions` 会把它报告出来。

## POST /api/presence

页面的输入心跳：此对话是通道中被聚焦的对话。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/presence \
  -H 'Content-Type: application/json' -d '{"session_id": "3f9c2a7b1d4e"}'
```

```json
{"ok": true}
```

- 未知会话返回 404 `not_found`。该调用是针对被聚焦对话的一个调度器事件（通道会保留它的输入窗口）；它不会打开会话，也不改变其他任何东西。
- 通过轮次路由开始的轮次，以及带第一个轮次的创建，同样会重置该对话的保持时钟并使其获得焦点，与对话自身的消息一样。

## GET /api/sessions/{id}/tools

会话的智能体提供的每个工具，以及本会话是否允许它。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/tools
```

```json
{"tools": [{"name": "python_exec", "description": "Run Python code ...", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}, "source": "builtin", "requires_approval": true, "granted": false}], "mcp_config_label": "", "mcp_error": "MCP is not built in the console yet"}
```

- `source` 为 `builtin`、`provided`（MCP 服务器的工具）或 `custom`。`requires_approval` 表示调用它之前会先询问；`granted` 表示本会话在网关本次运行的剩余时间内允许它。
- 控制台会话处于网关模式：它列出 `file_read`、`file_write`、`list_files` 和 `python_exec`，从不列出 `shell_exec`。
- 任务会话返回 409 `task_session_read_only`；此路由不能把它作为对话打开。
- 此路由会打开会话，因为提供方的工具只有在其智能体启动之后才会出现。它可能需要数秒，而启动失败则返回 502 `agent_start_failed`。
- 控制台尚未实现 MCP，因此不加载任何 MCP 工具：`mcp_config_label` 始终为 `""`，`mcp_error` 始终为 `MCP is not built in the console yet`。

## DELETE /api/sessions/{id}/grants/{tool}

收回一项会话授权，使该工具下次再次询问。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X DELETE http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/grants/python_exec
```

- 204 且无响应体；`GET /api/sessions/{id}` 的 `grants` 和 `granted_at` 都会去掉它。
- 会话未打开或该工具没有授权时返回 404 `not_found`。`DELETE /api/sessions/{id}/grants` 不带工具名时，会忘记每一项授权，并返回 204，无论会话是否打开。

## GET /api/sessions/{id}/memory

会话的智能体在其窗口中记住的内容。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/memory
```

```json
{"agent": "open", "working": [{"role": "user", "text": "What is in this project?", "tool_calls": [], "tool_call_id": "", "images": 0}], "working_max": 20, "long_term": {"enabled": false, "backend": "none"}}
```

- 此路由从不打开会话。`working` 是已打开智能体的对话；在 `paused` 时为保留的副本；在 `not_loaded` 时为 `[]`。
- 助手行的 `text` 会移除其 `<think>` 块；每段文本超过 2,000 个字符时会在 `… (+N chars)` 处截断；页面会自行展开该消息。
- 每段文本都像已保存的事件一样脱敏：工具输出中携带的密钥会以 `[redacted]` 返回。工具行属于窗口的一部分。
- `tool_calls` 列出一条消息调用的名称，`images` 统计其图片引用数。
- `working_max` 是窗口的消息上限。`long_term.enabled` 表示设置中是否指定了嵌入后端，`backend` 是其键（未指定时为 `none`）。

## Memory across chats

melo 对用户保存的笔记，以及打开或关闭跨对话记忆的开关（见[跨对话记忆](../concepts/memory.md)）。面板的 Memory 标签页使用这些路由。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/memory/records
```

```json
{"on": true, "records": [{"id": 3, "text": "You prefer short answers.", "topic": "preferences", "quote": "I prefer short answers", "source": "you said it · chat “Planning”, Oct 30", "from": "from chat “Planning”, Oct 30", "chat_gone": false, "edited": false, "session_id": "3f9c2a7b1d4e", "created_at": 1793395000.0}], "documents": 12, "embedding_model": "", "retriever": "auto", "last_error": ""}
```

- `GET /api/memory/records` 按时间从新到旧列出笔记；`?session_id=ID` 只保留某个对话的。`topic` 为 `about`、`preferences`、`people`、`projects` 或 `decisions`。`from` 写明对话和日期；对话被删除后是 `from a deleted chat, Oct 30`（`chat_gone` 为 true）；用户改写过笔记后 `edited` 为 true。`documents` 是对话索引中的轮次数，`embedding_model` 是按含义搜索的模型（`""` 表示只按词，`retriever` 为 `bm25` 或 `llm` 时总是如此），`last_error` 是它最近一次没被使用的原因。
- `PATCH /api/memory/records/{id}`，请求体 `{"text": "..."}`，原地改写一条笔记，并返回上述形式的笔记。旧文字被覆盖（不保留），新文字重新嵌入，笔记的决策行中的文字被清空。空文字、超过 300 个字符或看起来像密码或密钥的文字返回 400 `invalid_request` 并给出原因；未知笔记返回 404 `not_found`。
- `DELETE /api/memory/records/{id}` 删除一条笔记、它的向量和它取代过的旧文字，并清空其决策行中的文字：204，或 404 `not_found`。带 `?undo=true`（消息下方的 “Undo”）时，被它替换的笔记恢复为当前。
- `DELETE /api/memory/records?all=true` 忘记所有笔记并清空对话索引，返回 `{"notes": 12, "turns": 48}`。没有 `all=true` 时返回 400。对话本身不受影响。
- `PUT /api/memory`，请求体 `{"on": false}`，切换并保存 `[memory] across_chats`；返回 `{"on": false}`。此后开始的对话会使用它。
- `POST /api/memory/reindex` 从对话重建对话索引（上次“忘记全部”之前结束的轮次不会进入），并给每条笔记补上当前嵌入模型的向量：`{"notes": 0, "turns": 48}`。

## GET /api/sessions/{id}/runs

每个已结束轮次的运行记录，与轮次列表一样，最新的页面排在最前。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/runs?limit=20'
```

```json
{"runs": [{"turn": 3, "turn_id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "status": "completed", "started_at": 1761141600.2, "finished_at": 1761141612.4, "result": {"task_id": "4b1e7c9a2f3d4e5b8a6c0d1e2f3a4b5c", "status": "completed", "error": "", "steps": 3, "...": "..."}}], "has_earlier": false}
```

- `limit` 为 1 到 50（默认 20）；`before=N` 分页到编号低于 `N` 的轮次，与 `GET /api/sessions/{id}/turns` 相同。
- `result` 是该轮次的 `RunResult`，不含其 `result` 文本，因为记录中已经显示了该文本。仍在运行的轮次没有运行记录，因而被略去；`has_earlier` 表示该页之外还有更早的轮次。

## Not in this version

`POST /api/sessions` 和 `POST /api/sessions/{id}/turns` 中的文件附件，以及控制台中的 MCP 工具，目前尚未实现；上面的结构是之后的改动将要扩展的形式。

FastAPI 的 `/docs`、`/redoc` 和 `/openapi.json` 已关闭：本页即为参考。非回环地址的绑定会在启动时被拒绝；见[网关与 HTTP API](../guides/gateway.md#keep-the-gateway-private)。
