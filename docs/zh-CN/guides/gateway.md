<a id="gateway-and-http-api"></a>

# 网关与 HTTP API

网关通过 HTTP 提供智能体服务：既为你自己的程序提供 API，也提供可在浏览器中运行智能体的网页工作区。本页介绍如何启动网关、确保其私密、调用 API、使用网页工作区，以及在 macOS 登录时运行网关。

<a id="start-the-gateway"></a>

## 启动网关

```bash
melo serve
```

旧写法（`melo --serve`、`--daemon`、`--gateway-login`、`--gateway-rotate-token`、`--enable-playground`、`--allow-unconfigured`）仍然可用，macOS 应用和启动脚本就在用；完整对照见[旧写法对应的新写法](../reference/cli.md#old-spelling-to-new)。

- 如果还没有配置文件，`melo serve` 会先运行配置向导；参见[使用向导配置](configure.md#set-up-with-the-wizard)。如果没有终端，网关无法提问，因此会输出 `melo: no backend is configured; run melo setup in a terminal`，并以退出码 1 退出。
- 即使没有配置后端或配置文件存在验证错误，`melo serve --force` 也会跳过向导并启动网关，让客户端或网页控制台通过 `GET /api/health` 检查状态，并通过 `PUT /api/settings` 保存配置。
- `--url` 和 `--model` 与 `melo serve` 同时使用属于用法错误：网关使用配置文件。
- 网关监听 `127.0.0.1:8401`，即配置文件中 `[gateway]` 的默认值。`--host` 和 `--port` 可以更改单次运行的地址和端口，例如 `melo serve --port 9401`。
- 启动时会输出后端和地址：

  ```text
    Starting localmelo gateway ...
    Chat: ollama
    Embedding: none
    Gateway: http://127.0.0.1:8401
    Web console: opened in your browser (melo console opens it again).
  ```

  最后一行在网关开始监听后才出现。在本机终端里运行 `melo serve` 时，它会用默认浏览器打开已登录的网页控制台（`BROWSER` 环境变量可指定其他浏览器）。通过 SSH、管道、作为登录服务运行，或在 macOS 应用的网关里，它不会打开浏览器，而是改为输出 `Web UI:  http://127.0.0.1:8401/` 和 `Sign in: run melo console -p 8401 in another terminal`。如果没有浏览器能启动，它会像 `melo console` 一样输出一次性链接。

- 启动时不会连接模型服务器。运行一个任务来检查连接。
- 按 Ctrl+C 停止网关。

<a id="keep-the-gateway-private"></a>

## 确保网关私密

> [!WARNING]
> 网关只允许绑定回环地址，访问受保护的 API 需要 bearer 令牌或浏览器会话 cookie。持有令牌者可以使用你的权限执行操作。私有令牌文件无法防范以同一操作系统用户身份运行的恶意进程、浏览器扩展或被攻陷的同源代码。浏览器 cookie 不按端口隔离：针对特定端口的名称不能构成抵御同一主机上其他服务的安全边界。不要通过代理暴露网关。

启动网关：在本机终端里，`melo serve` 会在浏览器中打开已登录的网页控制台。运行 `melo console` 可再次打开；通过 SSH 或没有浏览器能启动时，它会输出一次性链接，请在 60 秒内打开，该端口上没有网关在监听时则以退出码 1 结束。如果使用非默认端口，请加上 `--port PORT`。不要分享该链接。页面会用链接的片段交换一个有效期最长为 30 天、具有 HttpOnly 和 SameSite=Strict 属性的 cookie，然后移除该片段。运行 `melo console --reset` 可替换私有令牌文件。所有使用该文件的网关都会拒绝旧凭证；对已打开的事件流撤销凭证约在 250 毫秒内生效。一个正在发送的事件仍可能发送完成。

bearer 令牌在首次使用时生成，并存储于 `~/.cache/localmelo/gateway/auth/token`（目录权限为 0700，文件权限为 0600）。非浏览器客户端的每个 `/api/` 和 `/v1/` 请求都应发送 `Authorization: Bearer <token>`，包括健康检查、流和 blob 请求。凭证缺失或无效时返回 401。页面和静态资源仅作为公开的引导资源。不要将令牌放入 URL、日志或 shell 历史记录；在命令行中提供请求头可能使其暴露于本地进程检查。此令牌与模型服务器凭证相互独立。

使用默认主机地址时，只有你电脑上的程序能访问网关。网页仍可通过你的浏览器向网关发送请求，因此网关会拒绝并非发往自身的请求：

| HTTP 状态码 | `code` | 情况 |
|---|---|---|
| 421 | `misdirected_request` | `Host` 请求头不是 `127.0.0.1:<port>`、`localhost:<port>` 或 `[::1]:<port>`。这可以阻止其他网站上的页面使用自己的域名访问网关（DNS 重绑定）。由 macOS 应用启动的网关还会拒绝不带应用匹配 `X-LocalMelo-Instance` 的 `/api/auth/login`，登录码也与该实例绑定。 |
| 403 | `foreign_origin` | 提供的 `Origin` 与请求的 Host 不完全匹配，或者 `/v1/` 或 `/api/` 下的请求带有不允许的 `Sec-Fetch-Site`。登录交换要求提供 Origin。 |
| 413 | `too_large` | 请求体超过 1 MiB（1,048,576 字节）。 |
| 415 | `unsupported_media_type` | `/api/` 下的请求带有非空请求体，但没有 `Content-Type: application/json`。 |

响应体是带有 `code` 的常规错误：

```json
{"error":"Host header must be 127.0.0.1:8401, localhost:8401 or [::1]:8401","code":"misdirected_request"}
```

被拒绝的 `/api/` 请求则使用 [API v2](../../reference/api-v2.md)（英文）的响应封装：`{"error": {"code": "misdirected_request", "message": "...", "details": {}}}`。

- 使用 `http://127.0.0.1:8401/` 或 `http://localhost:8401/` 打开网页工作区，并替换为你的端口；若设置了 `host = "::1"`，则使用 `http://localhost:8401/` 或 `http://[::1]:8401/`。其他能访问网关的地址，例如 `http://0.0.0.0:8401/`，会收到 421。
- curl 和脚本通常不发送 `Origin` 请求头；Host 和身份验证检查仍然适用。
- 每个响应都会通过 `X-Frame-Options: DENY` 和以 `frame-ancestors 'none'` 结尾的 `Content-Security-Policy` 告诉浏览器不要在框架中显示该响应，以防其他网站的页面嵌入网页工作区并诱骗你在其中点击。带有自身策略的响应，例如使用 `default-src 'none'; sandbox` 的 blob，会保留一个包含两者的响应头。每个响应还带有 `X-Content-Type-Options: nosniff`、`Referrer-Policy: no-referrer`、`Cross-Origin-Opener-Policy: same-origin`、`Cross-Origin-Resource-Policy: same-origin` 和 `Permissions-Policy: camera=(), microphone=(), geolocation=()`。
- 当主机为 `localhost` 或回环 IP 地址时，这些检查会启用：包括 `127.0.0.1`、`::1` 或 `127.0.0.0/8` 中的其他地址。请求仍须以 `127.0.0.1`、`localhost` 或 `[::1]` 指定网关，因此请使用这三个主机地址之一：例如使用 `127.0.0.2` 时，浏览器也会收到 421。
- 非回环地址绑定（包括 `0.0.0.0`）会在启动时被拒绝。
- 不支持通过反向代理或 Tailscale Serve 远程访问。

<a id="sessions"></a>

## 会话

一个会话对应一个有独立工作记忆的智能体，因此任务能看到同一会话中先前的任务及其回答。每个回答都会去掉 `<think>` 块并截断到 8,000 个字符，但较长的会话仍会产生更大的请求：对于无关的任务，请省略 `session_id`。

- 不指定 `session_id` 时，每个请求都会获得一个新会话，其生成的 ID 为 12 个十六进制字符。
- 一个会话同一时间只运行一个任务。对忙碌会话发出的第二个请求会等待第一个任务完成。
- 网关最多保留 50 个会话。再创建会话时，会关闭最近最少使用的空闲会话；如果没有空闲会话，仍会创建新会话。
- 创建会话时，还会关闭空闲超过 1 小时的会话。
- 每个会话的智能体使用配置文件中的后端和默认运行选项（三个内置工具 `file_read`、`file_write` 和 `python_exec`，没有 MCP 工具），但会保留回答（`AgentOptions(record_answers=True, mode="gateway")`）。会话不提供 `shell_exec`：旧版 `/v1` 没有审批策略，因此无法向任何人请求审批。
- 为网关设置 `LOCALMELO_PERSIST_MEMORY` 时，所有会话都会保存到同一组 SQLite 文件；参见[跨运行保留记忆](memory.md#keep-memory-between-runs)。

<a id="run-the-gateway-at-login-macos"></a>

## 登录时运行网关（macOS）

```bash
melo serve install
melo serve status
melo serve uninstall
```

- `install` 写入 `~/Library/LaunchAgents/ai.localmelo.gateway.plist` 并加载它。随后 launchd 使用你的环境中的 Python 运行 `<python> -m localmelo --serve --port 8401`，在登录时启动，并在其退出时重启（最多每 3 秒一次）。
- `install --port N` 使用其他端口。主机地址取自配置文件中的 `[gateway] host`；`--host` 会被忽略。
- 日志写入 `~/.localmelo/logs/gateway.stdout.log` 和 `~/.localmelo/logs/gateway.stderr.log`。
- `status` 输出服务是否为 `installed` 和 `running`，以及它的 `pid`。
- 在执行 `install` 前，使用 `melo` 或在终端中使用配置向导保存配置。服务无法回答问题：没有配置文件时，launchd 每次启动服务都会将 `melo: no backend is configured; run melo setup in a terminal` 写入 `gateway.stderr.log`。
- launchd 启动服务时不会带上你的 shell 环境变量，因此云端 API 密钥和 `LOCALMELO_PERSIST_MEMORY` 均不可用。请为该服务使用本地后端。
- 该服务需要 macOS；在其他系统上执行 `install` 会失败。

<a id="stop-the-gateway"></a>

## 停止网关

- 在网关所在的终端中按 Ctrl+C。
- 对于后台服务，运行 `melo serve uninstall`。单纯使用 `kill` 无济于事：launchd 会再次启动网关。

停止时会关闭所有会话，包括正在运行任务的会话：如果会话仍在忙碌，其智能体会被关闭，任务以 `crashed` 结束。收到 SIGTERM 后，网关会先等待运行中的任务最多 3 秒，然后取消剩余任务：客户端请求会失败，运行以 `crashed` 结束。`DELETE` 请求不再关闭忙碌的会话：它会返回 409，直到任务结束。
