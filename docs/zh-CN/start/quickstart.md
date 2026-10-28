<a id="quickstart"></a>

# 快速开始

本教程将安装 localmelo，针对本地模型服务器运行第一个任务，保存配置并启动网关。

<a id="before-you-begin"></a>

## 开始之前

你需要：

- macOS（已开发和测试）或 Linux（应该可以运行，但未经测试）。不支持 Windows。
- Python 3.11 或更新版本、Git 和 [uv](https://docs.astral.sh/uv/)（或 pip，见[使用 pip 安装](#install-with-pip)）。
- 模型服务器。本教程使用 [Ollama](https://ollama.com)：安装并启动它，然后使用 `ollama pull qwen3:8b` 拉取模型。任何提供 OpenAI 兼容 API 的服务器也可以使用：在下面的命令中填写它的 `/v1` URL 和模型名称。在搭载 Apple silicon 的 Mac 上，也可以[使用 oMLX 提供模型服务](../guides/model-servers.md#serve-a-model-with-omlx)。

<a id="install"></a>

## 安装

```bash
git clone https://github.com/localmelo/localmelo.git
cd localmelo
uv sync --extra gateway
source .venv/bin/activate
```

uv 会创建虚拟环境 `.venv/`（如果没有 Python 3.13，还会安装它），并安装 localmelo 及网关依赖。

请保留目录名 `localmelo`：软件包从该目录构建。重命名文件夹或下载 ZIP（`localmelo-main`）会导致 `ModuleNotFoundError: No module named 'localmelo'`。

本指南余下部分直接调用 `melo`。激活只在当前终端有效：在新终端中需要再次运行 `source <clone>/.venv/bin/activate`，其中 `<clone>` 是你的 `localmelo` 目录。使用 uv 时，也可以在克隆目录中运行 `uv run melo …`，或在其他任何位置运行 `uv run --project <clone> melo …`。

<a id="install-with-pip"></a>

### 使用 pip 安装

```bash
git clone https://github.com/localmelo/localmelo.git
cd localmelo
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[gateway]"
```

`python3` 必须是 Python 3.11 或更新版本。`-e` 选项使安装保持与此检出目录的链接，因此源代码更改无需重新安装即可生效。

<a id="run-your-first-task"></a>

## 运行第一个任务

> [!WARNING]
> 模型可以运行 shell 命令和 Python 代码，并读取或写入你有权访问的任何文件：没有沙箱。如需运行不使用工具的任务，添加 `--builtins none`。请参阅[安全策略](https://github.com/localmelo/localmelo/blob/main/SECURITY.md)（英文）和[检查的作用](../guides/tools-mcp.md#what-the-checks-do)。

```bash
melo --base-url http://127.0.0.1:11434/v1 --chat-model qwen3:8b "What is 6*7?"
```

答案会输出到 stdout，命令以退出码 0 结束（`echo $?` 输出 `0`）。`--base-url` 将 localmelo 指向一个 OpenAI 兼容服务器，仅用于本次运行，无需配置文件。它不发送 API 密钥，因此适用于本地服务器；要使用云端 API，请保存配置（见下文）。

所有选项见[命令行参考](../../reference/cli.md)（英文）。

<a id="get-the-result-as-json"></a>

## 以 JSON 获取结果

```bash
melo --base-url http://127.0.0.1:11434/v1 --chat-model qwen3:8b --output-format json "What is 6*7?"
```

输出示例（数值会有所不同）：

```json
{"task_id": "f52e785a84a144ff8c3b8e70e0451eb1", "status": "completed", "result": "42", "error": "", "steps": 1, "attempts": 1, "n_llm": 1, "n_main": 1, "n_aux": 0, "n_tool": 0, "llm_errors": 0, "usage": {"prompt_tokens": 120, "completion_tokens": 12, "total_tokens": 132}, "duration_s": 1.9, "config": {"max_steps": 30, "attempt_policy": "reflect", "steps_per_attempt": 10, "max_attempts": 5, "reflection_gate": "flags", "tool_exposure": "retrieve", "mode": "direct", "builtins": ["file_read", "file_write", "python_exec"], "memory": true, "memory_dir": null, "timeout_s": null, "tools": ["file_read", "file_write", "python_exec"]}, "localmelo_version": "603f40d"}
```

stdout 中恰好只有这一行。只有 `status` 为 `completed` 时退出码才是 0。

<a id="run-results"></a>

### 运行结果

`status` 表示运行如何结束，例如 `completed`、`budget_exhausted` 或 `provider_error`。计数器表示运行消耗的步数、模型请求次数和工具调用次数，`config` 记录运行时使用的选项。所有状态和字段见[运行结果](../../reference/run-results.md)（英文）。

<a id="save-a-configuration"></a>

## 保存配置

配置文件保存后端设置，使 `melo`、`melo "<task>"` 和网关无需 `--base-url` 即可工作。

对于本地服务器，在终端运行 `melo`。尚无配置时，它会在本机查找模型服务器，询问其 URL（按 Enter 接受找到的地址）和服务器列表中的模型，保存 `~/.cache/localmelo/config.toml` 并打开交互式界面：

```text
$ melo
Found a model server at http://127.0.0.1:11434/v1 (1 model).
Model server URL [http://127.0.0.1:11434/v1]:
✓ http://127.0.0.1:11434/v1 lists 1 model; using qwen3:8b.
No embedding model on this server. ollama pull qwen3-embedding:0.6b, then melo --reconfigure.
Saved to ~/.cache/localmelo/config.toml (melo --reconfigure changes it).
╭─ melo 0.1.0 ─────────────────────────────────────────────────────────────╮
│ model      qwen3:8b · Ollama · http://127.0.0.1:11434/v1                 │
│            saved as the default                                          │
│ workspace  ~/demo-project                                                │
│ tools      4 built-in (3 ask first)                                      │
│ memory     no long-term memory · melo --reconfigure adds one             │
│ session    new                                                           │
╰──────────────────────────────────────────────────────────────────────────╯
  / commands · esc stop · ctrl+d quit
```

如果服务器还列出嵌入模型，首次运行会多问一个问题 `Use <model> for long-term memory? [Y/n]:`；回答 `y` 后，会在一次嵌入探测通过后保存。输入消息并按 Enter，或输入 `/exit` 退出。此后，`melo` 会直接进入界面，`melo "<task>"` 会使用已保存的服务器。这里的服务器是 Ollama，因此保存为 `ollama` 后端；其他 OpenAI 兼容服务器保存为 `vllm`，即它们的后端键。见[首次运行](../guides/configure.md#first-run)和[配置嵌入模型](../guides/embedding.md)。

<a id="with-the-setup-wizard"></a>

### 使用配置向导

配置向导还可以配置嵌入服务器或云端 API，并在完成后启动网关。对于云端 API，先在当前终端运行 `export OPENAI_API_KEY="<your key>"`（或提供方对应的变量）：向导的测试请求会使用该密钥。

```bash
melo --serve --reconfigure
```

向导会询问：

| 提示 | 输入内容 |
|---|---|
| `Chat backend` | 服务器类型的编号，例如 Ollama 为 `2`。默认值为已保存的后端，否则为 `[1]`，即 MLC。对于其他 OpenAI 兼容服务器，选择 vLLM（`3`）。 |
| `Base URL` | 服务器的 URL。按 Enter 接受方括号中显示的默认值：已保存的 URL，否则为该后端的常用地址。 |
| `Chat model name` | 模型，例如 `qwen3:8b`。 |
| `Start?` | 按 Enter（是）继续。 |
| `Embedding backend` | 输入 `5` 选择 `None (no long-term memory)`。尚未保存嵌入配置时，对话服务器自身的嵌入模型会作为 `6`、`7` 等追加，第一个为默认值；选择一个即可。 |
| `Embedding endpoint URL`, `Embedding model name` | 仅当选择固定嵌入后端 `1`–`4` 之一，而不是检测到的模型时询问。 |

对于云端 API，选择提供方。向导会改为询问 `API key env var`（你导出的变量名称）和 `Chat model`，而不是 `Base URL` 和 `Chat model name`。

Ollama 的默认 URL `http://localhost:11434` 不带 `/v1`。这是正确的：`ollama` 后端使用 Ollama 自己的 API，而不是 OpenAI 兼容 API。

接下来，向导向每个服务器发送简短的测试请求（`Probing backends ...`）。如果对话服务器无响应，会显示 `Probe failed: …` 并询问 `[r]etry, [s]ave anyway, [N]o [N]:`；`r` 会重新询问 URL 和模型，按 Enter 则不保存任何内容。如果只有嵌入服务器失败，向导会关闭嵌入并保存，同时告知你。成功时会显示 `Config saved to …` 和 `~/.cache/localmelo/config.toml` 的路径，然后在 `http://127.0.0.1:8401` **启动网关**。网关会在此终端持续运行。不带 `--serve` 时，`melo --reconfigure` 会改为打开交互式界面。

<a id="start-the-gateway"></a>

## 启动网关

如果使用 `melo` 保存了配置，使用 `melo --serve` 启动网关；上面的向导会自行启动它。在第二个终端中，检查网关并通过它运行任务：

```bash
curl http://127.0.0.1:8401/v1/health
curl -X POST http://127.0.0.1:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 6*7?"}'
```

输出示例（数值会有所不同）：

```json
{"status":"ok","uptime":12.3,"sessions":{"active":0,"busy":0,"max":50},"chat_backend":"ollama","embedding_backend":"none"}
{"result":"42","session_id":"a7e8f646ca2e"}
```

健康检查不会联系模型服务器；任务会。在浏览器中打开 `http://127.0.0.1:8401/` 可进入网页工作区（见[交互式对话](#interactive-conversations)）。

在第一个终端中按 Ctrl+C 停止网关。任务使用已保存的配置：

```bash
melo "What is 6*7?"
```

要不运行向导而再次启动网关，运行 `melo --serve`。

<a id="interactive-conversations"></a>

## 交互式对话

不带任务运行 `melo`，即可在终端中对话。每次工具调用完成时都会显示，模型工作时会以旋转指示器显示步骤和用时，答案以 Markdown 输出：

```bash
melo
# Or talk to an OpenAI-compatible server that is not saved, with its first
# model or the one you name:
melo --base-url http://127.0.0.1:8000/v1
melo --base-url http://127.0.0.1:8000/v1 --chat-model YOUR_MODEL_ID
```

- 输入 `/` 查看命令：`/model` 列出服务器的模型并切换模型，`/backend` 切换到另一个服务器（两者都会保存为默认值，`melo --base-url` 会话除外），`/new` 重新开始，`/exit` 或 Ctrl+D 退出。
- melo 运行 shell 命令、Python 代码或写入文件前会询问：按 `1` 允许，按 `2` 在本会话中允许该工具，按 `3` 拒绝并告诉 melo 原因。Esc 停止当前轮次。
- melo 在当前文件夹中工作（`--workspace DIR` 可选择其他文件夹）：文件工具限制在该文件夹内；命令和 Python 可以访问其他文件，因此 melo 会先询问。
- 对话可以使用推理和召回模式：`--reasoning`、`--reasoning-field`、`--replay-turns`、`--note-writer`、`--retriever`、`--recall-*` 和 `--retrieval-extra-body`，或已保存的 `[agent]` 和 `[retrieval]` 表。显式选项优先，检索选项在 `legacy` 中不起作用。

[交互式界面](../../guides/terminal.md#the-interactive-ui)（英文）介绍了按键和其他内容。`melo ""` 打开早期版本的普通对话循环。没有终端时，单独运行 `melo` 会以退出码 2 退出；请使用 `melo --serve` 启动网关。

`http://127.0.0.1:8401/` 上的网页工作区是控制台的实时页面，只与 `/api` 通信：它使用网关配置的模型，因此 `melo --reconfigure` 会改变该模型。它按最新优先列出对话，支持搜索和每个对话各自的草稿，并像终端一样展示每一轮：你的消息、每次工具调用的工作卡片、折叠的推理和反思、提示、实时进度行，以及带事实和 `Copy response` 的答案。停止操作会取消当前轮次，等待中的审批显示为 `Needs approval`，对话保存在网关的 `sessions.db` 中，因此重启后仍然存在。同一网关上的其他浏览器标签页可看到相同的对话。工作区提供审批卡片、连接设置和会话详情抽屉；见[网页工作区](../../guides/web-console.md#web-workspace)（英文）。

<a id="next-steps"></a>

## 下一步

- [配置](../../reference/configuration.md)（英文）：后端、云端 API 密钥、超时。
- [工具和 MCP](../../reference/tools.md)（英文）：选择内置工具并添加 MCP 服务器。
- [网关和 HTTP API](../../guides/gateway.md)（英文）：会话、网页工作区、登录时运行。
- [Python API](../../reference/python-api.md)（英文）：从 Python 运行任务、跟踪事件、审批工具调用。
- [记忆](../../reference/memory.md)（英文）：跨运行保留记忆、检查和重置记忆。
- [命令行参考](../../reference/cli.md)（英文）：所有选项、退出码和脚本编写。
- [故障排查](../../reference/troubleshooting.md)（英文）：错误消息和解决办法。
