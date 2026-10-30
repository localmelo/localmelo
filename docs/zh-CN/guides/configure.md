<a id="guide-configure"></a>

# 配置指南

本指南介绍首次运行、配置向导，以及如何更新或卸载 localmelo。在一台机器上设置 localmelo 或修改其配置时请阅读本指南；各项设置见[配置参考](../reference/configuration.md)。

<a id="first-run"></a>

## 首次运行

在没有配置文件的终端中运行 `melo`，它会询问模型服务器的 URL 和对话模型，保存答案，然后打开交互式界面。如果服务器还列出了嵌入模型，它会询问一次是否使用。下面的例子中，oMLX 在 8000 端口运行，提供三个模型，第二个已加载：

```text
$ melo
Found a model server at http://127.0.0.1:8000/v1 (3 models).
Model server URL [http://127.0.0.1:8000/v1]:
Models on http://127.0.0.1:8000/v1:
  [1] Qwen3-Embedding-0.6B
  [2] Qwen3-8B-4bit  (loaded)
  [3] Qwen3-32B-4bit
Model [2]:
✓ http://127.0.0.1:8000/v1 lists 3 models; using Qwen3-8B-4bit.
Use Qwen3-Embedding-0.6B for long-term memory? [Y/n]:
Saved to ~/.cache/localmelo/config.toml (melo setup changes it).
```

- 如果本机服务器在端口 8000、8400、11434 或 30000 上列出模型，提示中会填入它的 URL；有对话模型的服务器排在最前。列出服务器模型时发送 `GET /v1/models`。有多个对话模型时，设置还会询问哪些模型已在内存中：用 `GET /v1/models/status`（oMLX）和 `GET /api/ps`（Ollama），各等一秒，不带密钥；已加载的模型会标上 `(loaded)`，第一个作为默认值，没有已加载的模型时没有默认值，请输入编号。保存选择时会发送一次 `GET /api/version`，以区分 Ollama 服务器；回答 `y` 则会发送一次 `POST /v1/embeddings`，探测提供的模型。
- 只有列表中包含对话模型以外的嵌入模型时才会询问嵌入问题。回答 `y` 会保存三个嵌入值：对话后端的键、其 URL 和提供的模型。探测失败、回答 `n` 或没有候选模型时，会像以前一样保存 `embedding_backend = "none"`。
- Ollama 服务器（响应 `GET /api/version` 的服务器）保存为 `ollama` 后端，oMLX 服务器（它的模型列表把 oMLX 标为每个模型的所有者）保存为 `omlx`，其他 OpenAI 兼容服务器保存为 `vllm`，文件中的其他设置保持不变。`omlx` 是 `vllm` 后端的另一个名字，所以设置仍在 `[vllm]` 表中。没有嵌入时，写入：

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

  回答 `y` 时，会写入对话后端的嵌入字段，而不是 `"none"`：

  ```toml
  chat_backend = "omlx"
  embedding_backend = "omlx"

  [vllm]
  chat_url = "http://127.0.0.1:8000/v1"
  chat_model = "Qwen3-8B-4bit"
  embedding_url = "http://127.0.0.1:8000/v1"
  embedding_model = "Qwen3-Embedding-0.6B"
  ```

  `ollama` 后端通过 Ollama 原生 API 对话，请求超时为 300 秒；`vllm` 和 `omlx` 使用 120 秒。Ollama 的 `chat_url` 带不带 `/v1` 都可以：版本端点在根路径查询，对话请求会去掉末尾的 `/v1`。

- 服务器无响应、因没有 API 密钥而拒绝请求，或没有列出模型时，会指出是哪个服务器，并给出原因；你可以重试、不保存直接启动或退出。localmelo 从不询问本地服务器的 API 密钥。
- [命令行参考](terminal.md#first-run)列出了所有提示和原因。

嵌入模型请参阅[配置嵌入模型](embedding.md)。如需云端 API 或其他后端，请使用向导。

<a id="set-up-with-the-wizard"></a>

## 使用向导配置

```bash
melo setup                  # the wizard, then the interactive UI
melo --serve --reconfigure  # the wizard, then the gateway
```

向导会询问对话后端、其 URL 和模型，以及嵌入后端；[保存配置](../start/quickstart.md#save-a-configuration)列出了这些提示。它会向每个服务器发送简短的测试请求，保存配置文件，然后打开交互式界面，或者在使用 `--serve --reconfigure`（唯一没有命令词写法的设置方式）时启动网关。按 Ctrl+C 停止网关。

在默认模式下，`melo setup` 像首次运行一样开始：显示 `Found a model server at …` 并询问 `Use it? [Y/n]:`。只有回答 `n`、没有找到服务器或使用 `--developer` 时，才出现完整菜单。

- 每个问题都会在方括号中提供配置文件中的当前值：按 Enter 保留。嵌入问题提供已保存的后端；如果尚未保存，则提供对话服务器列出的第一个候选模型，没有候选时为 `None`。
- 没有保存嵌入配置时，向导会列出一次对话服务器的模型（仅限本地对话后端），并将找到的每个嵌入模型作为选项 `6`、`7` 等追加在固定的五个选项之后。第一个候选模型是默认值；选择它会保存对话后端的 URL 和模型，不再询问端点。固定的 `1`–`4` 选项仍会询问另一个服务器的 URL 和模型，`5` 是 `None`。无法读取模型列表时，向导会静默回退到这五个选项；向导结束时的检查会报告无法连接的服务器。
- 对话服务器检查失败时，向导显示 `Probe failed: …` 并询问 `[r]etry, [s]ave anyway, [N]o [N]:`。`r` 重新询问 URL 和模型，以你输入的值为默认值；`s` 仍然保存；Enter 或 `N` 不保存任何内容。
- 本地服务器返回 HTTP 401 或 403 时，向导会提示 localmelo 不向本地服务器发送 API 密钥。在 oMLX 中请启用 **Allow inference without API key**。
- 嵌入服务器检查失败时，向导会关闭嵌入并保存（`embedding_backend = "none"`，没有长期记忆），同时告知你。检测到但探测失败的模型不会留在配置表中。
- 没有配置文件时，`melo serve` 会自行运行向导。
- 没有终端时，向导从 stdin 读取答案。既没有配置文件也没有终端时，`melo serve` 会以退出码 1 退出，并在 stderr 输出一行，而不是等待不可能到来的答案（例如在 launchd 下）。给 `melo serve` 传入 `--force` 可照常启动网关，再通过 `/api/settings` 配置。
- 重新运行向导可以更换后端。它会保留认识的其他设置，例如 `request_timeout_s` 和 `extra_body`，并以当前格式原子重写文件（如果 `config.toml` 是符号链接则跟随链接，保留现有文件的权限，新建文件使用 `0600` 权限）；注释和未知键会被丢弃。

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
