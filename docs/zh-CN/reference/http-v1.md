<a id="reference-http-v1"></a>

# 参考：HTTP v1

本页列出网关的 HTTP v1 端点，并说明与之并列的 API v2 和旧版 playground 路由。你自己的程序通过 HTTP 调用网关时请使用本页；启动和保护网关的方法见[网关指南](../guides/gateway.md)。

<a id="http-api"></a>

## HTTP 接口

请求和回复都是 JSON。示例使用默认地址。

任何路由都要求 Bearer 认证，也可能回答 401、421 或 403；见[保持网关私有](../guides/gateway.md#keep-the-gateway-private)。

<a id="api-v2-web-console"></a>

### API v2（网页控制台）

网页控制台自己的 API 位于 `/api/` 下。它只在网关监听回环地址时提供，并单独记录在 [API v2](../../reference/api-v2.md)（英文）中。

认证对现有客户端而言是一项兼容性变更；除此之外，API 字段只增不改。

本版本提供读取：`GET /api/health`、`GET /api/settings`、`GET /api/settings/key-status`、`GET /api/models`、会话列表和单个会话、会话的轮次、它的 Server-Sent Events 流、它的 blob、它的工具、记忆和运行记录，以及提交查询；还提供配置网关和运行会话的写入：`PUT /api/settings`、`POST /api/settings/test`、创建会话（带或不带第一轮）、开始、重试和停止其轮次、按 key 取消消息、回答审批、忘记会话授权、重命名或移动会话，以及删除会话。调度器任务还有 `POST /api/tasks`、`GET /api/tasks`，以及 `GET` 和 `PATCH /api/tasks/{task_id}`；它们的字段和控制见[调度器任务](../../reference/api-v2.md#scheduler-tasks-s3a2)（英文）。

`/api` 路由给出与 `/v1/` 相同的 421、403 和 413（非 JSON 请求体另有 415），并为每个回复加上 `Cache-Control: no-store` 和统一错误结构 `{"error": {"code", "message", "details"}}`。

控制台会话在网关进程中运行它的轮次，一次一个：一轮运行时再来第二轮会得到 409 `session_busy`。它的智能体在会话第一次使用时构建并保留；当第九个智能体将要打开时，最久未使用的空闲智能体会被关闭，并保留其对话。

审批在网关内存中等待，没有超时，直到客户端回答或轮次停止；停止网关会让正在等待的轮次以 `cancelled` 结束，原因为 `shutdown`。控制台轮次像工作区聊天一样经过调度器的门：网关根据 `[scheduler]` 表构建一个门，两个界面共用它，因此它们的调用共享一份 `n_srv` 预算和一个调用日志，工作区页面运行的监视器为两者提供输入（外部 GPU 锁在 `lock_gate = "wait"` 下扣住控制台轮次；电池只为页面的状态胶囊读取，从不扣住调用）。

网关在关闭时最后关闭这个门，在关闭两个界面的会话之后，因此等待槽位的轮次以 `cancelled` 结束，原因为 `shutdown`。

任务 worker 对模型和嵌入调用使用同一个门，记忆关闭，工具是网关的工具（没有 `shell_exec`）。最多可以有四个 worker 驻留；已暂停的运行中 worker 仍可能占用一个槽位。

任务行在重启后保留；启动过程会中断不确定的工作，并保留已证明从未开始的排队任务，在普通创建尚未得到证明的期间可以使用 Stop。完整恢复、截止时间和 worker 记账被推迟。

工作区校验也在 worker 开始或恢复之前运行；不再有效的文件夹会让任务失败，而不是运行它。暂停要等到下一次模型调用：正在运行的工具会完成，已回答的待处理审批会等到恢复。

恢复时还原暂停前的优先级；如果保留机制清除了暂停日志，则使用计算出的 P2；未暂停的任务返回 409 `not_paused`。任务 PATCH 拒绝 `first`（400）；它被推迟到 S6b。

<a id="legacy-playground"></a>

### 旧版 playground

内部的 `/v1/playground/` 路由默认关闭（404），它们的管理器也不会启动。用 `melo --serve --enable-playground` 显式启用；认证和 Host/Origin 检查仍然适用。

发现和会话使用已保存的本地对话后端和模型，而不是请求中的覆盖设置。这些路由包括：

- `GET /v1/playground/sessions`，用于摘要；
- `GET /v1/playground/session/{id}`，用于对话记录；
- `POST /v1/playground/session/{id}/cancel`，用于取消一次聊天；
- `POST /v1/playground/session/{id}/level`，请求体为 `{"level": 1..4}`，用于设置级别；
- `POST /v1/playground/presence`，请求体为 `{"session_id": ...}`，用于输入心跳。

单独的 `/v1/agent/` API 也要求认证，但它的调用不经过 playground 和控制台共享的调度器门。

开发时，下面的命令用已保存的配置和配置好的回环地址，启动启用 playground 的已认证网关：

```bash
uv run python -m localmelo.support.gateway._test_webapp
```

<a id="get-v1health"></a>

### 健康检查：GET /v1/health

示例把网关的 Bearer 令牌从它的私有文件通过管道传给 curl，使它不出现在命令行参数和 shell 历史中。不要启用 curl 的详细输出，也不要记录该请求头；同一用户的其他进程仍然能读取令牌文件。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/v1/health
```

```json
{"status":"ok","uptime":12.3,"sessions":{"active":0,"busy":0,"max":50},"chat_backend":"ollama","embedding_backend":"none"}
```

| 字段 | 含义 |
|---|---|
| `status` | 网关能回答时为 `ok`。 |
| `uptime` | 网关启动以来的秒数。 |
| `sessions` | `active` 会话的数量、其中有多少 `busy` 运行任务，以及 `max`。 |
| `chat_backend`, `embedding_backend` | 配置的后端。 |

健康检查不联系模型服务器。

<a id="post-v1agentrun"></a>

### 运行任务：POST /v1/agent/run

在一个会话中运行一个任务，并返回它的结果。

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 6*7?", "session_id": "demo123"}'
```

```json
{"result":"42","session_id":"demo123"}
```

| 请求字段 | 必须 | 说明 |
|---|---|---|
| `query` | 是 | 任务，去掉首尾空白后最多 100,000 个字符。 |
| `session_id` | 否 | 1 到 64 个字母、数字、`_` 或 `-`。使用该会话，并在需要时创建它。不提供时，创建一个新会话。 |

| HTTP 状态 | 请求体 | 何时 |
|---|---|---|
| 200 | `{"result": "…", "session_id": "…"}` | 运行结束，无论其状态如何。 |
| 400 | `{"error": "…"}` | 查询为空或过长，或者 `session_id` 无效。 |
| 409 | `{"error": "…", "session_id": "…"}` | 少见：会话在这个请求能够开始之前被关闭了。 |
| 422 | `{"detail": […]}` | 请求体不是 JSON 对象。 |
| 500 | `{"error": "…", "session_id": "…"}` | 网关中的意外错误。 |

400 错误读作 `Query must be a non-empty string`、`Query too large: <n> chars (max 100000)` 或 `Invalid session_id format: '<id>'`；不是字符串的 id 也会得到最后一条。

`result` 是答案或本次运行的尽力结果。当它为空时，是 `[<status>] <error>`，例如 `[provider_error] POST /chat/completions failed with HTTP 500: …`。

没有完成的运行仍然返回 HTTP 200，回复中没有状态字段；见[运行结果](run-results.md#where-results-appear)。

<a id="get-v1sessions"></a>

### 会话列表：GET /v1/sessions

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/v1/sessions
```

```json
{"sessions":[{"session_id":"demo123","status":"idle","idle_seconds":4.2,"created_at":1792732915.88,"is_busy":false}]}
```

| 字段 | 含义 |
|---|---|
| `session_id` | 会话的 id。 |
| `status` | `idle` 或 `running`。 |
| `idle_seconds` | 会话上次使用以来的秒数。 |
| `created_at` | 会话创建的时间，以 1970 年以来的秒数（Unix 时间）计。 |
| `is_busy` | 会话中有任务运行时为 `true`。 |

<a id="delete-v1sessionssession_id"></a>

### 删除会话：DELETE /v1/sessions/{session_id}

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X DELETE http://127.0.0.1:8401/v1/sessions/demo123
```

回答是 HTTP 200 和 `{"closed": "demo123"}`，或者 HTTP 404 和 `{"error": "session not found"}`。在进行中的 security-1b 变更中，有运行中任务或排队开始的会话不会被处理：回答是 HTTP 409，`{"error": "Session demo123 is running a task; wait for it to finish before closing it.", "session_id": "demo123", "code": "session_busy"}`，任务继续进行。

[GET /v1/sessions](#get-v1sessions)中的 `is_busy` 显示哪些会话正忙。

`/v1/agent` API 没有停止运行的路由，所以请等它结束之后再关闭会话。启用后，`POST /v1/playground/session/{id}/cancel` 停止的是旧版 playground 聊天，不是 `/v1/agent` 运行。

当前的[网页工作区](../guides/web-console.md#web-workspace)则用 [`POST /api/turns/{turn_id}/cancel`](../../reference/api-v2.md#post-apiturnsturn_idcancel)（英文）停止它的轮次。停止网关仍会关闭每个会话，包括正忙的会话；见[停止网关](../guides/gateway.md#stop-the-gateway)。
