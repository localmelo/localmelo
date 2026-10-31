<a id="guide-configure"></a>

# 配置指南

在一台机器上设置 localmelo、修改配置、更新或移除它。各项设置见[配置参考](../reference/configuration.md)。

<a id="first-run"></a>

## 首次运行

在没有配置文件的终端中运行 `melo`，它会提供找到的模型服务器，询问对话模型，保存答案，然后打开对话。下面的例子中，oMLX 在 8000 端口运行，提供三个模型，第一个对话模型已加载：

```text
$ melo
melo uses a model server on this computer (oMLX, Ollama, LM Studio…).
Found a model server at http://127.0.0.1:8000/v1 (3 models).
 Choose a model · http://127.0.0.1:8000/v1
 Filter ›
 ❯ Qwen3-8B-4bit  (loaded)
   Qwen3-32B-4bit
 2 models · enter chooses · esc cancels
✓ http://127.0.0.1:8000/v1 lists 3 models; using Qwen3-8B-4bit.
Saved to ~/.cache/localmelo/config.toml (melo setup changes it).
```

- melo 在本机的 8000、8400、11434 和 30000 端口查找（`GET /v1/models`），提供第一个列出模型的服务器，有对话模型的排在最前。
- 每个答案固定的问题都是列表：↑ ↓ 或数字键移动，Enter 确认，Esc 取安全的答案（见[问题都是列表](terminal.md#questions-are-lists)）。
- 模型列表只显示对话模型，并把服务器说已在内存中的标上 `(loaded)`：设置会询问 `GET /v1/models/status`（oMLX）或 `GET /api/ps`（Ollama），各最多等一秒，且不带密钥。
- 第一个已加载的模型是默认值；没有已加载的模型时没有默认值，请用 ↑ ↓ 移到某个模型（通过管道输入时输入编号）。
- 通过管道输入时，改为按编号列出（`Model [1]:`）；嵌入或重排模型会加标记，且绝不作为默认值。
- 不会发送对话请求，所以服务器不会为这次检查加载模型。

如果服务器还列出了嵌入模型，会再问一个问题 `Remember across chats with <model>?`：

- `Yes, with it` 会发送一次 `POST /v1/embeddings` 探测，并把对话后端的键、其 URL 和该模型保存为嵌入后端。
- `No, by words only`（或按 Esc）、探测失败或没有候选模型时，保存 `embedding_backend = "none"`。

没有嵌入模型时，保存的内容：

```toml
# an Ollama server
chat_backend = "ollama"
embedding_backend = "none"

[ollama]
chat_url = "http://127.0.0.1:11434/v1"
chat_model = "qwen3:8b"
```

```toml
# an oMLX server (another OpenAI-compatible server: "vllm")
chat_backend = "omlx"
embedding_backend = "none"

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B-4bit"
```

选择 `Yes, with it` 时，会写入对话后端的嵌入字段，而不是 `"none"`：

```toml
chat_backend = "omlx"
embedding_backend = "omlx"

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B-4bit"
embedding_url = "http://127.0.0.1:8000/v1"
embedding_model = "Qwen3-Embedding-0.6B"
```

- 响应 `GET /api/version` 的服务器保存为 `ollama`，模型列表把 oMLX 标为所有者的保存为 `omlx`，其他保存为 `vllm`；`omlx` 是 `vllm` 的另一个名字，所以它的设置仍在 `[vllm]` 中。
- `ollama` 通过 Ollama 原生 API 对话，请求超时 300 秒，`vllm` 和 `omlx` 为 120 秒；Ollama 的 `chat_url` 带不带 `/v1` 都可以。
- 文件中的其他设置保持不变。

没有可用的服务器时：

- 没有服务器在运行时，受支持的 Apple silicon Mac 会显示终端的[只读设置预览](model-servers.md#setup-preview)：推荐项、下载量上限、已缓存量和可用空间。摘要只显示一次，`Show how to set one up` 会列出最多六步的编号步骤（开发者模式保留带 `Details` 的检查菜单，可查看版本、路径和校验和）。自动设置即将推出，目前绝不会执行安装。列表也可以重新检查、输入地址或退出；管道输入保留原有的键入答案。其他平台以及网页/应用设置卡片保留手动启动说明。
- 服务器无响应、需要 API 密钥或没有列出模型时，会指出是哪个服务器并给出原因；你可以重试、不保存直接启动或退出。
- localmelo 从不询问本地服务器的 API 密钥。
- [终端指南](terminal.md#first-run)列出了所有提示和原因。

嵌入模型请参阅[配置嵌入模型](embedding.md)。如需云端 API 或其他后端，请使用向导。

<a id="set-up-with-the-wizard"></a>

## 使用向导配置

```bash
melo setup                  # the wizard, then the interactive UI
melo --serve --reconfigure  # the wizard, then the gateway
```

- 向导会询问对话后端、其 URL 和模型，以及嵌入后端；[保存配置](../start/quickstart.md#save-a-configuration)列出了这些提示。
- 它会向每个服务器发送简短的测试请求，保存配置文件，然后打开对话，或者在使用 `--serve --reconfigure` 时启动网关（Ctrl+C 停止）。
- 在默认模式下，`melo setup` 会先提供找到的服务器（`Found a model server at …`、`Use this model server?`）。选择 `Use another server`、处于开发者模式，或已经保存后端但没有找到服务器时，会显示完整后端菜单。只有尚未配置后端的首次运行才会在没有找到服务器时显示只读设置预览。
- 在开发者模式（`[ui] mode = "developer"`；`melo config set ui.mode developer` 可保存它）下，完整菜单最先出现。
- 每个问题都会在方括号中提供文件中的当前值：按 Enter 保留。

出问题时：

- 对话服务器检查失败时，会显示 `Probe failed: …` 并询问 `The check failed. What now?`：`Try again`、`Save anyway` 或 `Cancel setup`（默认选中；Esc 同样不保存任何内容）。通过管道输入时，问题是 `[r]etry, [s]ave anyway, [N]o [N]:`。
- 本地服务器返回 HTTP 401 或 403 时，会提示 localmelo 不向本地服务器发送 API 密钥；在 oMLX 中请启用 **Allow inference without API key**。
- 嵌入服务器检查失败时，会保存 `embedding_backend = "none"`（没有长期记忆），并告知你。

向导的其他说明：

- 没有配置文件时，`melo serve` 会自行运行向导；也没有终端时，它会以退出码 1 退出并在 stderr 输出一行，传入 `--force` 则照常启动网关，再通过 `/api/settings` 配置。
- 重新运行向导可以更换后端：它保留认识的设置，例如 `request_timeout_s` 和 `extra_body`，并以原子方式重写文件，新建文件使用 `0600` 权限；注释和未知键会被丢弃。
- 找到更多嵌入模型时，向导会把它们作为选项 `6`、`7` 等追加在固定的五个选项之后；第一个是默认值，不再询问端点。

<a id="update-localmelo"></a>

## 更新 localmelo

在克隆目录中拉取最新代码并更新环境：

```bash
git pull
uv sync --extra gateway
```

如果你使用 pip 安装，请改为重复[使用 pip 安装](../start/quickstart.md#install-with-pip)中的最后一条命令，而不是运行 `uv sync`。

<a id="uninstall-localmelo"></a>

## 卸载 localmelo

1. 如果安装了后台服务，运行 `melo serve uninstall`。
2. 删除克隆目录。
3. 删除 `~/.cache/localmelo`（配置和已保存的记忆）、`~/.localmelo`（后台服务的日志），以及通过 `LOCALMELO_MEMORY_DIR` 或 `--memory-dir` 使用的所有目录。
