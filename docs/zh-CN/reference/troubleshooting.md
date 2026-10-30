<a id="troubleshooting"></a>

# 故障排除

找到你看到的消息，然后阅读它的原因和修复方法。未完成的运行会在 `melo: run ended with status <status>: <error>` 这一行中，或在 JSON 行的 `status` 字段中，给出它的状态。

<a id="install"></a>

## 安装

### command not found: melo

```text
zsh: command not found: melo
```

（bash 显示 `bash: melo: command not found`。）或者在使用 `uv run` 时：

```text
error: Failed to spawn: `melo`
  cause: No such file or directory (os error 2)
```

- **原因：** 当前终端中没有激活虚拟环境（激活只在一个终端中有效），或者 `uv run` 在 localmelo 克隆目录之外运行，uv 找不到 localmelo。
- **修复：** 再次运行 `source <clone>/.venv/bin/activate`，其中 `<clone>` 是你的 `localmelo` 目录。使用 uv 时，请在克隆目录内运行 `uv run melo …`，或在其他位置运行 `uv run --project <clone> melo …`。

### No module named 'localmelo'

```text
ModuleNotFoundError: No module named 'localmelo'
```

- **原因：** 克隆目录的名称不是 `localmelo`，例如目录被重命名，或者 ZIP 包被解压为 `localmelo-main`。
- **修复：** 把目录重命名为 `localmelo`，然后重新安装，步骤见[安装](../start/quickstart.md#install)。

### Gateway requires fastapi and uvicorn

```text
ImportError: Gateway requires fastapi and uvicorn: pip install fastapi uvicorn
```

- **原因：** localmelo 安装时没有包含 `gateway` 扩展。
- **修复：** 运行 `uv sync --extra gateway`，或 `pip install -e ".[gateway]"`。

### pip refuses to install

```text
error: externally-managed-environment
```

- **原因：** pip 不会安装到系统或 Homebrew 的 Python 中。
- **修复：** 先按[使用 pip 安装](../start/quickstart.md#install-with-pip)中的说明创建虚拟环境，或使用 uv。

### FileNotFoundError when the gateway starts

```text
FileNotFoundError: [Errno 2] No such file or directory: '…/localmelo/tests/smoke/data/scenarios.json'
```

- **原因：** 旧版本的构建没有把这个必需的数据文件打包进 wheel。当前版本会把它与网页工作区的资源一起打包。
- **修复：** 按[更新 localmelo](../guides/configure.md#update-localmelo)中的说明更新并重新安装 localmelo。使用 `pip install -e ".[gateway]"` 进行的可编辑源码安装也包含该文件。

<a id="setup-and-configuration"></a>

## 设置与配置

### No backend configured

```text
No backend configured. Run melo in a terminal to set up,
or use --base-url to connect to an OpenAI-compatible endpoint.
```

- **原因：** 没有配置文件，且命令没有 `--base-url`。`melo` 以退出码 1 退出。
- **修复：** 在终端中运行 `melo`，它会询问模型服务器并保存（见[首次运行](../guides/terminal.md#first-run)）；或者用 `melo --serve --reconfigure` 保存配置；或者添加 `--base-url` 和 `--chat-model`。

### No backend is configured; run melo --reconfigure in a terminal

```text
melo: no backend is configured; run melo --reconfigure in a terminal
```

- **原因：** `melo --serve` 没有找到配置文件，而设置向导在没有终端时无法提问。在 launchd 下，这一行会进入网关的错误日志。`melo` 以退出码 1 退出。
- **修复：** 在终端中运行一次 `melo` 或 `melo --reconfigure`，然后再次启动网关。

### The interactive UI needs a terminal

```text
melo: the interactive UI needs a terminal (stdin and stdout must be TTYs).
  Run one task:      melo "your task"
  Start the gateway: melo --serve
```

- **原因：** 不带任务的 `melo` 会打开交互式界面，它需要 stdin 和 stdout 都是终端，并且 `TERM` 不是 `dumb`：这里其中之一是管道或文件（脚本、`melo | tee`），或者 `TERM=dumb`。`melo` 以退出码 2 退出。
- **修复：** 运行 `melo "<task>"` 执行一个任务，或运行 `melo --serve` 启动网关，或者在真正的终端中运行 `melo`。

### A model server cannot be used at the first run

```text
✗ http://127.0.0.1:8000/v1: connection refused.
[r]etry, [s]tart without saving, [q]uit [r]:
```

- **原因：** 对该 URL 的 `GET /v1/models` 失败。`connection refused`：那里没有进程在监听；`HTTP 401` 或 `HTTP 403`：服务器需要 API 密钥，而 localmelo 不会向本地服务器发送密钥；`it lists no models`：服务器尚未提供任何模型。
- **修复：** 启动服务器，检查其端口和 URL 末尾的 `/v1`，然后按 Enter 重试。在 oMLX 中，打开 **Allow inference without API key**。按 `q` 退出且不保存。

### Configuration errors

```text
  Configuration errors:
  - [openai] environment variable $OPENAI_API_KEY is not set. Export it before starting the gateway.
```

`melo` 会在运行之前停止，退出码为 1。每一行都指出一个问题：

| 消息包含 | 修复 |
|---|---|
| `environment variable $… is not set` | 在当前 shell 中导出 API 密钥。 |
| `embedding_url is empty`, `embedding_model is empty` | 在后端的表中添加这两项，或设置 `embedding_backend = "none"`。 |
| `request_timeout_s must be a finite number of seconds > 0` | 写成不带引号的数字，例如 `request_timeout_s = 600`。 |
| `extra_body must not set 'model'` | 从 `extra_body` 中删除该键。 |

### LOCALMELO_THINKING_RETENTION_DAYS must be a number of days

```text
melo: error: LOCALMELO_THINKING_RETENTION_DAYS must be a number of days greater than 0, not 'abc'
```

- **原因：** 该变量的值不是大于 0 的数字，例如 `abc`、`0` 或 `inf`。`melo` 在运行任何内容之前以退出码 2 退出，适用于直接模式、`melo ""` 和 `--serve`。
- **修复：** 设置一个天数，例如 `export LOCALMELO_THINKING_RETENTION_DAYS=7`；或运行 `unset LOCALMELO_THINKING_RETENTION_DAYS`，恢复默认的 30 天。见[模型调用](../../reference/memory.md#model-calls)。

### TOMLDecodeError

```text
tomllib.TOMLDecodeError: Illegal character '\n' (at line 1, column 21)
```

- **原因：** 配置文件不是有效的 TOML。消息给出了行和列，但没有给出文件名：文件是 `~/.cache/localmelo/config.toml`。
- **修复：** 更正该行，或删除该文件并重新运行设置向导。

### An embedding model cannot be used

```text
  Probe failed: embedding probe failed: …
  Saving with embedding off: no long-term memory (melo --reconfigure sets it again).
```

- **原因：** `POST /v1/embeddings` 探测失败。**HTTP 404** 表示服务器不提供该模型 id；**HTTP 400** 表示该 id 是对话模型，而不是嵌入模型；**connection refused** 表示该 URL 上没有进程在监听。
- **修复：** 用 `curl -sS <url>/v1/models` 列出服务器的模型，使用它列出的 id，并确认用该 id 发起的 `POST /v1/embeddings` 会返回非空的 `data[0].embedding`。然后再次运行 `melo --reconfigure`。见[设置嵌入模型](../guides/embedding.md#check-the-server)。
- **更换模型后召回为空：** 旧的向量维度不同，会被跳过。请恢复旧模型，或停止 localmelo，删除 `~/.cache/localmelo/memory/long_term.db*`，让存储重新填充。见[变更内容](../guides/embedding.md#what-changes)。

### Probe failed

```text
  Probe failed: chat probe failed: …
  [r]etry, [s]ave anyway, [N]o [N]:
```

- **原因：** 设置向导的测试请求失败。服务器没有运行，URL 或模型有误，或者缺少 API 密钥。对于本地服务器，HTTP 401 或 403 会附带一条说明：localmelo 不会向本地服务器发送 API 密钥。
- **修复：** 纠正问题，然后输入 `r`：向导会再次询问 URL 和模型，并以你之前输入的内容作为默认值。按 Enter 则不保存任何内容（`Setup cancelled — config not saved.`）。嵌入服务器检查失败不会中止设置：向导会在关闭嵌入的情况下保存。

网关的 HTTP 401 表示其 bearer 令牌或浏览器会话缺失或无效。请在终端中运行 `melo --gateway-login`，获取新的一次性链接；需要时加上 `--port PORT`。对于 API 客户端，请使用私有的网关令牌；轮换令牌会使现有的浏览器会话失效。见[网关安全](../guides/gateway.md#keep-the-gateway-private)。

### `melo --serve --reconfigure` does not return

- **原因：** 保存之后，设置向导会启动网关，而网关会持续运行。
- **修复：** 这并没有错。按 Ctrl+C 停止网关；之后 `melo "<task>"` 会使用已保存的配置。不带 `--serve` 时，`melo --reconfigure` 会打开交互式界面，输入 `/exit` 即可退出。

### request_timeout_s has no effect

- **原因：** 该键位于某个 `[table]` 行之下，而 localmelo 不会在那里读取它。
- **修复：** 把它移到文件顶部，放在第一个表之前。见[对话请求超时](configuration.md#chat-request-timeout)。

<a id="runs-that-do-not-complete"></a>

## 未完成的运行

### ConnectError

```text
melo: run ended with status provider_error: cannot reach the model server at http://127.0.0.1:11434/v1. Start it, or run melo to set one up. (ConnectError: All connection attempts failed)
```

- **原因：** 该 URL 上没有进程在监听：服务器没有运行，或主机或端口有误。localmelo 尝试了三次，历时约 5 秒。
- **修复：** 启动服务器并检查其 URL，例如用 `curl http://127.0.0.1:11434/v1/models` 检查。若服务器不在本机或局域网内，消息会改为 `Check the URL and the network.`。

### HTTP 404 or another error status

```text
melo: run ended with status provider_error: POST /chat/completions failed with HTTP 404: …
```

- **原因：** 服务器返回了错误状态码，其回答的开头紧跟在后面。404 表示 URL 错误（与 `--base-url` 一起使用时，要包含 `/v1` 部分），或者服务器没有该模型；回答通常会说明原因。500 表示服务器或其前面的代理出错。
- **修复：** 检查 URL 和模型名称；`GET /v1/models` 会列出模型。

### ReadTimeout

```text
melo: run ended with status provider_error: POST /chat/completions failed: ReadTimeout
```

- **原因：** 服务器在对话请求超时时间内没有发送任何内容，连续三次。这大约需要 3 × 超时时间 + 5 秒。在第一个请求到达时才加载模型的服务器（如 oMLX）可能还在加载。
- **修复：** 给慢速模型更多时间：使用 `--request-timeout 600`，或在配置文件顶部设置 `request_timeout_s = 600`。

### A run never ends

- **原因：** 有些服务器在生成时会持续发送保活字节，每个字节都会重新开始请求超时计时。
- **修复：** 用 `--timeout SECONDS` 限制整个运行。

### failed: empty replies

```text
melo: run ended with status failed: empty replies
```

- **原因：** 模型连续两次回复，既没有文本也没有工具调用。
- **修复：** 换一个模型，或检查模型服务器的日志。

### budget_exhausted: Max steps reached

```text
melo: run ended with status budget_exhausted: Max steps reached
```

- **原因：** 运行用完了所有步骤，却没有最终答案。
- **修复：** 提高 `--max-steps`，或让任务更小。要查看各步骤做了什么，请用 `--memory-dir` 保留运行历史，并[检查它](../guides/memory.md#inspect-memory)。

### stuck

```text
melo: run ended with status stuck: Stuck: the same tool calls or errors repeated
```

- **原因：** 使用 `--attempt-policy none` 时，相同的工具调用或相同的错误连续重复了三个步骤。
- **修复：** 使用默认策略 `reflect`，它会进行反思并再次尝试；或者在该运行的历史中查看是哪次调用一直失败。

### gate_stopped

```text
melo: run ended with status gate_stopped: Stopped after attempt 1 (stop reason: not_novel)
```

- **原因：** 一次尝试之后，反思决定停止，或者它继续的决定没有通过继续判定（见[反思](../concepts/how-it-works.md#reflection)）。错误信息是模型给出的理由，若它没有给出理由，则为 `Stopped after attempt <n> (stop reason: <code>)`；在所有情况下，`reflection` 事件的 `stop_reason` 都会指明是哪项检查。使用 `--reflection-gate utility` 时，它还可能显示为 `Could not parse reflection response`：模型两次回复时都没有反思所需的 JSON 对象。
- **修复：** 阅读其理由。如果模型无法写出反思，请换一个模型，或使用 `--attempt-policy none`。

### Tool not offered in this step

- **原因：** `Tool '<name>' was not offered in this step.` 是工具的错误，出现在历史记录中，而不是运行状态。模型调用了本步骤没有提供的工具，因此该调用没有运行。
- **修复：** 通常无需处理：之后的步骤可以提供该工具。若要在每个步骤都提供全部工具，请使用 `--tool-exposure all`。

<a id="gateway"></a>

## 网关

### Address already in use

```text
ERROR:    [Errno 48] error while attempting to bind on address ('127.0.0.1', 8401): [errno 48] address already in use
```

- **原因：** 另一个程序占用了该端口，通常是另一个网关或后台服务。`melo` 以退出码 3 退出。（Linux 显示的错误号不同。）
- **修复：** 停止另一个程序（`melo --daemon status` 会显示该服务），或用 `--port` 选择另一个端口。

### HTTP 400

```json
{"error":"Query must be a non-empty string"}
```

- **原因：** 请求体无效：`query` 必须是非空字符串，`session_id` 为 1 到 64 个字母、数字、`_` 或 `-`。
- **修复：** 更正请求体。见 [HTTP API](http-v1.md#http-api)。

### HTTP 421 misdirected_request

```json
{"error":"Host header must be 127.0.0.1:8401, localhost:8401 or [::1]:8401","code":"misdirected_request"}
```

- **原因：** 请求没有把网关写成 `127.0.0.1`、`localhost` 或 `[::1]` 加上端口。例如，网页工作区是以 `http://0.0.0.0:8401/` 打开的，或者通过一个会发送自己主机名的代理访问，或者因为网关监听另一个回环地址而以 `http://127.0.0.2:8401/` 打开。网关会拒绝这些请求，以防另一个站点上的网页通过 DNS 重绑定访问它。由 macOS 应用启动的网关，若 `/api/auth/login` 请求缺少应用匹配的 `X-LocalMelo-Instance`，也会返回这个错误；其登录代码与该次启动绑定。
- **修复：** 打开 `http://127.0.0.1:8401/` 或 `http://localhost:8401/`，并使用网关的端口。如果网关监听的是另一个回环地址，请让它改为监听 `127.0.0.1`：在 `[gateway]` 中设置 `host = "127.0.0.1"`，或运行 `melo --serve --host 127.0.0.1`。不支持代理和远程访问；见[保持网关私有](../guides/gateway.md#keep-the-gateway-private)。在 macOS 应用中，请使用 **Sign In Again**，而不要使用在其他地方签发的代码。

### HTTP 403 foreign_origin

```json
{"error":"Cross-origin request refused: only pages at http://127.0.0.1:8401, http://localhost:8401 or http://[::1]:8401 may send it","code":"foreign_origin"}
```

- **原因：** 一个并非由网关提供的网页通过浏览器发送了请求：可能是另一个站点或你电脑上另一个端口的页面，从文件打开的页面（`Origin: null`），或通过代理打开的网页工作区。从浏览器发出的请求，网关只接受来自它自己的网页工作区的 API 请求。
- **修复：** 使用 `http://127.0.0.1:8401/` 的网页工作区，或从 curl 或脚本等程序调用 API（它们不发送 `Origin` 请求头）。不支持从其他网页调用 API。

### A status in the result

```json
{"result":"[provider_error] POST /chat/completions failed with HTTP 500: …","session_id":"53a72edadef4"}
```

- **原因：** 运行没有完成，也没有回答，因此网关返回 `[<status>] <error>`，仍然使用 HTTP 200。
- **修复：** 见[未完成的运行](#runs-that-do-not-complete)中该状态对应的条目。

### The background service keeps restarting

- **原因：** 网关立即退出，launchd 每隔几秒就重新启动它。常见原因是服务缺少云端 API 密钥（它拿不到你 shell 中的变量），或者没有配置文件。
- **修复：** 阅读 `~/.localmelo/logs/gateway.stderr.log`。让服务使用本地后端，或运行 `melo --daemon uninstall`，然后在终端中运行 `melo --serve` 以查看错误。

<a id="mcp-servers"></a>

## MCP 服务器

### MCP server exited

```text
melo: run ended with status setup_error: MCP server 'notes' exited (code 1); stderr: …
```

- **原因：** 服务器在启动过程中停止。其错误输出的末尾紧跟在 `stderr:` 之后。
- **修复：** 在终端中运行该服务器的 `command` 和 `args`，以查看错误。请记住它只会得到一个最小化的环境：把它需要的变量添加到 `env` 中。

### Cannot start MCP server

```text
melo: run ended with status setup_error: Cannot start MCP server 'notes' (my-notes-server): [Errno 2] No such file or directory: 'my-notes-server'
```

- **原因：** 找不到 `command`。
- **修复：** 在配置文件中写出该命令的完整路径。

### HTTP/SSE servers are not supported

```text
melo: error: mcp.json: server 'docs': only stdio servers are supported (got type 'sse')
```

或者，对于没有 `type` 的条目：

```text
melo: error: mcp.json: server 'docs': 'command' must be a non-empty string (HTTP/SSE servers are not supported)
```

- **原因：** 该条目描述的是 HTTP 或 SSE 服务器：它的 `type`（或 `transport`）不是 `stdio`，或者它有 `url` 而没有 `command`。localmelo 只支持 stdio 服务器。
- **修复：** 如果服务器提供 stdio 模式，请使用它。

<a id="getting-help"></a>

## 获取帮助

如果以上内容都没有帮助，请搜索 [issues](https://github.com/localmelo/localmelo/issues)，或按照 [报告 bug](https://github.com/localmelo/localmelo/blob/main/CONTRIBUTING.md#report-a-bug) 中列出的细节开一个新的 issue。先删除 API 密钥和个人路径。
