<a id="quickstart"></a>

# 快速开始

本教程将安装 localmelo，针对本地模型服务器运行第一个任务，保存配置并启动网关。

<a id="before-you-begin"></a>

## 开始之前

你需要：

- macOS（已开发和测试）或 Linux（应该可以运行，但未经测试）。不支持 Windows。
- Python 3.11 或更新版本、Git 和 [uv](https://docs.astral.sh/uv/)（或 pip，见[使用 pip 安装](#install-with-pip)）。
- 模型服务器。下面的命令使用 [Ollama](https://ollama.com)：安装并启动它，然后使用 `ollama pull qwen3:8b` 拉取模型。在搭载 Apple silicon 的 Mac 上，推荐使用 [oMLX](../guides/model-servers.md#serve-a-model-with-omlx)：在命令中改用 `http://127.0.0.1:8000/v1` 和你的模型文件夹名。任何提供 OpenAI 兼容 API 的服务器也可以使用：填写它的 `/v1` URL 和模型名称。

<a id="install"></a>

## 安装

```shell
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

```shell
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
> 模型可以运行 shell 命令和 Python 代码，并读取或写入你有权访问的任何文件：没有沙箱。如需运行不使用工具的任务，见[选择内置工具](../guides/tools-mcp.md#choose-the-built-in-tools)。另见[安全策略](https://github.com/localmelo/localmelo/blob/main/SECURITY.md)（英文）和[检查的作用](../guides/tools-mcp.md#what-the-checks-do)。

```shell
melo --url http://127.0.0.1:11434/v1 --model qwen3:8b "What is 6*7?"
```

答案会输出到 stdout，命令以退出码 0 结束（`echo $?` 输出 `0`）。`--url` 将 localmelo 指向一个 OpenAI 兼容服务器，仅用于本次运行，无需配置文件。它不发送 API 密钥，因此适用于本地服务器；要使用云端 API，请保存配置（见下文）。`-u` 和 `-m` 是 `--url` 和 `--model` 的字母写法；旧写法 `--base-url` 和 `--chat-model` 仍然可用（[旧写法对应的新写法](../reference/cli.md#old-spelling-to-new)）。

不带 `--model` 时，melo 会先询问服务器有哪些模型，使用其唯一的对话模型，或服务器说明已在内存中的那一个。有多个对话模型且不知道哪个在内存中时，它会以退出码 2 停止并列出这些模型，因为请求一个尚未加载的模型可能让服务器加载一个很大的模型：请传入 `--model <id>`。

使用 [oMLX](../guides/model-servers.md#serve-a-model-with-omlx) 时，填写它的 URL 和模型文件夹的名称：

```shell
melo --url http://127.0.0.1:8000/v1 --model Qwen3-8B-4bit "What is 6*7?"
```

直接任务会不经询问就写入文件、运行 Python，并且没有 `shell_exec` 工具，除非传入 `--builtins all`。这不是沙箱：`python_exec` 仍可启动程序（见[上面的警告](#run-your-first-task)）。任务简报中的检查（见[运行 worker 任务](#run-a-worker-task)）仍会运行。

所有选项见[命令行参考](../../reference/cli.md)（英文）。

<a id="get-the-result-as-json"></a>

## 以 JSON 获取结果

```shell
melo --url http://127.0.0.1:11434/v1 --model qwen3:8b --output json "What is 6*7?"
```

输出示例（数值会有所不同）：

```json
{"task_id": "f52e785a84a144ff8c3b8e70e0451eb1", "status": "completed", "result": "42", "error": "", "steps": 1, "attempts": 1, "n_llm": 1, "n_main": 1, "n_aux": 0, "n_tool": 0, "llm_errors": 0, "usage": {"prompt_tokens": 120, "completion_tokens": 12, "total_tokens": 132}, "duration_s": 1.9, "config": {"max_steps": 30, "attempt_policy": "reflect", "steps_per_attempt": 10, "max_attempts": 5, "reflection_gate": "flags", "tool_exposure": "retrieve", "mode": "direct", "builtins": ["file_read", "file_write", "python_exec"], "memory": true, "memory_dir": null, "timeout_s": null, "tools": ["file_read", "file_write", "python_exec"]}, "localmelo_version": "603f40d"}
```

stdout 中恰好只有这一行。只有 `status` 为 `completed` 时退出码才是 0。

<a id="run-results"></a>

### 运行结果

`status` 表示运行如何结束，例如 `completed`、`budget_exhausted` 或 `provider_error`。计数器表示运行消耗的步数、模型请求次数和工具调用次数，`config` 记录运行时使用的选项。所有状态和字段见[运行结果](../reference/run-results.md)。

<a id="run-a-worker-task"></a>

## 运行 worker 任务

worker 任务是写在文件里的任务简报（brief）。以 `Check:` 开头（前面没有空格）的一行写一条 shell 命令，用来判断任务是否完成：

```bash
cat > brief.md <<'EOF'
Write hello.py so that it prints the word hello.
Check: python3 hello.py | grep -x hello
EOF
melo worker brief.md --url http://127.0.0.1:11434/v1 --model qwen3:8b > result.json
echo "exit code: $?"
python3 -c 'import json; r = json.load(open("result.json")); print(r["status"], r["tests"]["state"])'
```

模型完成后，`melo` 在工作区（当前文件夹，或 `--workspace`）中运行该检查，无论这次运行向模型提供了哪些工具。检查失败时，模型会被送回去修复，最多两次。最后两条命令的输出示例（模型的工作结果会有所不同）：

```text
exit code: 0
completed passed
```

最后一轮的所有检查都以 0 退出时，`tests.state` 为 `passed`；有检查失败或无法运行时为 `failed`。运行以 `completed` 结束但检查失败时，退出码为 4，脚本可以据此停下：

```shell
melo worker brief.md --url http://127.0.0.1:11434/v1 --model qwen3:8b > result.json && echo "check passed"
```

实验性设置 `agent.brief_items`（`-X agent.brief_items=true`）还会在模型的反思请求和最后几步之前，指出简报中尚未被任何编辑触及的编号条目（见[参考](../../reference/cli.md#brief-items)）。

在 macOS 上，worker 的 shell 命令和 Python 代码只能写入工作区：写到其他地方（即使先 `cd ..`）会以 `Operation not permitted` 失败，并多出一行提示模型在工作区内写入；临时文件写到 `$TMPDIR`，即工作区里的 `.melo-tmp`。uv、black 和 npm 的缓存以及你自己的临时文件夹（`mktemp` 写入的位置）仍可写入，在 git worktree 中 `git status`、`git diff` 和 `git add` 照常可用。读取和网络不受限制。要不加这项限制地运行 worker，请加上 `-X agent.confine_writes=false`；见[写入留在工作区内（英文）](../../reference/cli.md#confined-writes)。

结果是一行 JSON，含 `status`、`exit_code`、`tests` 和运行记录。所有字段和退出码见[命令行参考](../../reference/cli.md#output)（英文）。

<a id="save-a-configuration"></a>

## 保存配置

配置文件保存后端设置，使 `melo`、`melo "<task>"` 和网关无需 `--url` 即可工作。

对于本地服务器，在终端运行 `melo`。尚无配置时，它先说明自己需要什么：本机上的一个模型服务器。随后在本机查找（oMLX 在 8000 端口，Ollama 在 11434 端口），询问是否使用找到的那个（`Use this model server?`，一个列表：按 Enter 即选中 `Use it`）和服务器列表中的模型，保存 `~/.cache/localmelo/config.toml` 并打开交互式界面。模型是一个用方向键选择的列表（输入即可筛选，Enter 选择，Esc 取消），其中只有对话模型；答案从管道输入时则改为编号列表，对话模型在前，嵌入或重排模型会标上 `(embedding)` 或 `(reranking)`，且绝不作为默认值。列表里有多个对话模型、且服务器会说明哪些已在内存中（oMLX、Ollama）时，这些模型会标上 `(loaded)`，其中第一个会被高亮，因此 Enter 不会去加载不在内存中的大模型。服务器没有说明（带 API 密钥的服务器返回 401，其他服务器没有这个路由）或没有任何模型已加载时，不会高亮任何一项，在你移到某个模型之前 Enter 不起作用，因为尚未加载的模型可能需要数分钟和大量内存才能加载（管道输入时同样没有默认值：只按 Enter 会再次询问，空答案是错误）。只有一个对话模型时无需选择：

```text
$ melo
melo uses a model server on this computer (oMLX, Ollama, LM Studio…).
Found a model server at http://127.0.0.1:11434/v1 (1 model).
✓ http://127.0.0.1:11434/v1 lists 1 model; using qwen3:8b.
melo remembers across chats by words only; melo setup adds search by meaning.
Saved to ~/.cache/localmelo/config.toml (melo setup changes it).
melo 0.1.0 · qwen3:8b (/model to change)
~/demo-project
Try: What is this project? · / commands · ctrl+d quit
```

第一行显示软件包版本；在 git 检出中还会显示短提交号，如 `melo 0.1.0 (baab21e)`。`melo --version` 输出同一行并以退出码 0 退出。`/status` 检查模型是否有回应，并显示文件夹、会话、审批和上下文用量；`/status all` 还会显示工具、记忆、限制和文件。

没有模型服务器在运行时，首次运行的 `melo` 和 `melo setup` 会在受支持的 Apple silicon Mac 上显示只读设置预览：oMLX 0.7.3、Qwen3 4B Instruct（4-bit，约 2.28 GB）、Hugging Face 缓存中仍缺少的模型文件、运行时下载量和可用磁盘空间。“随会话退出而停止”的生命周期也只是预览，修改后不会保存。选择 `Review setup plan` 可以预览生命周期或查看手动步骤，其中的 `Set up and start` 只说明自动设置尚未开放并显示手动步骤，不会安装、下载或启动任何东西。用方向键列表重新检查、输入地址或退出；管道输入保留 `Press Enter to check again, type an address, or q to quit:`，绝不代表同意设置。其他系统保留对应平台的手动启动说明。自行启动服务器后输入该地址：预览建议的高位端口不在默认发现范围内，`Check again` 只重新检查常用端口。保留缓存的手动路径见[模型服务器指南](../guides/model-servers.md#setup-preview)。已有连接和直接任务不受影响。

如果服务器还列出嵌入模型，首次运行会多问一个问题 `Remember across chats with <model>?`；选择默认选中的 `Yes, with it` 后，会在一次嵌入探测通过后保存。

在终端里，每个答案固定的问题都是列表：↑ ↓ 或数字键移动，Enter 确认，Esc 取安全答案，回答后列表会被擦除（见[问题都是列表](../guides/terminal.md#questions-are-lists)）。管道输入的答案和脚本仍使用键入的问题（`Use it? [Y/n]:`）。

输入消息并按 Enter，或输入 `/exit` 退出。此后，`melo` 会直接进入界面，`melo "<task>"` 会使用已保存的服务器。这里的服务器是 Ollama，因此保存为 `ollama` 后端；oMLX 服务器保存为 `omlx`，其他 OpenAI 兼容服务器保存为 `vllm`，即它们的后端键。见[首次运行](../guides/configure.md#first-run)和[配置嵌入模型](../guides/embedding.md)。

<a id="with-the-setup-wizard"></a>

### 使用配置向导

配置向导还可以配置嵌入服务器或云端 API。对于云端 API，先在当前终端运行 `export OPENAI_API_KEY="<your key>"`（或提供方对应的变量）：向导的测试请求会使用该密钥。先运行向导，再启动网关：

```shell
melo setup
melo serve
```

在默认模式下，`melo setup` 先提供它找到的服务器（`Use this model server?`）；选择 `Use another server` 才会出现完整菜单，开发者模式（`melo config set ui.mode developer`）则直接显示它。向导会询问：

向导会询问：

| 提示 | 输入内容 |
|---|---|
| `Chat backend` | 你的服务器类型，例如 `2. Ollama (local endpoint)`：用 ↑ ↓ 移动并按 Enter，或直接按数字键（管道输入时键入编号）。默认选中已保存的后端，否则为 MLC（`1`）。对于 oMLX 或其他 OpenAI 兼容服务器，选择 oMLX 或 vLLM（`3`）。 |
| `Base URL` | 服务器的 URL。按 Enter 接受方括号中显示的默认值：已保存的 URL，否则为该后端的常用地址。 |
| `Chat model name` | 模型，例如 `qwen3:8b`。 |
| `Save these settings?` | 在 `Save and continue` 上按 Enter，或选择 `Cancel setup`（管道输入时为 `Save and continue? [Y]:`）。 |
| `Embedding backend` | `5. None (no long-term memory)`。尚未保存嵌入配置时，对话服务器自身的嵌入模型会作为 `6`、`7` 等追加，默认选中第一个；选择一个即可。 |
| `Embedding endpoint URL`, `Embedding model name` | 仅当选择固定嵌入后端 `1`–`4` 之一，而不是检测到的模型时询问。 |

对于云端 API，选择提供方。向导会改为询问 `API key env var`（你导出的变量名称）和 `Chat model`，而不是 `Base URL` 和 `Chat model name`。

Ollama 的默认 URL `http://localhost:11434` 不带 `/v1`。这是正确的：`ollama` 后端使用 Ollama 自己的 API，而不是 OpenAI 兼容 API。

接下来，向导向每个服务器发送简短的测试请求（`Probing backends ...`）。如果对话服务器无响应，会显示 `Probe failed: …` 并询问 `The check failed. What now?`，选项有 `Try again`、`Save anyway` 和 `Cancel setup`（管道输入时为 `[r]etry, [s]ave anyway, [N]o [N]:`）；`Try again` 会重新询问 URL 和模型，按 Enter 会选中默认的 `Cancel setup`，不保存任何内容。如果只有嵌入服务器失败，向导会关闭嵌入并保存，同时告知你。成功时会显示 `Config saved to …` 和 `~/.cache/localmelo/config.toml` 的路径，然后打开交互式界面；用 Ctrl+D 离开它。接着运行 `melo serve`，它会在 `http://127.0.0.1:8401` 启动网关，并在此终端持续运行。在默认模式下，`melo setup` 先问首次运行的那个问题：显示 `Found a model server at …` 并询问 `Use this model server?`。只有选择 `Use another server` 或处于开发者模式时，才会出现上面的完整菜单；没有找到服务器时，会显示只读设置预览，并保留已有的连接设置。

<a id="start-the-gateway"></a>

## 启动网关

配置保存之后（无论是用 `melo` 还是用上面的向导保存的），用 `melo serve` 启动网关。在第二个终端中，检查网关并通过它运行任务：

示例把网关的 bearer token 从其私有文件通过管道传给 curl，使它不出现在命令行参数和 shell 历史中。不要开启 curl 的详细输出，也不要记录这个请求头；同一用户的其他进程仍能读取 token 文件。

```shell
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/v1/health
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 6*7?"}'
```

输出示例（数值会有所不同）：

```json
{"status":"ok","uptime":12.3,"sessions":{"active":0,"busy":0,"max":50},"chat_backend":"ollama","embedding_backend":"none"}
{"result":"42","session_id":"a7e8f646ca2e"}
```

健康检查不会联系模型服务器；任务会。在本机终端里运行 `melo serve` 时，网页工作区会在浏览器中自动打开；`melo console` 可再次打开它。通过 SSH 或没有浏览器能启动时，`melo console` 会输出一次性链接，请在 60 秒内打开。不要分享这个链接。见[交互式对话](#interactive-conversations)。

在第一个终端中按 Ctrl+C 停止网关。任务使用已保存的配置：

```shell
melo "What is 6*7?"
```

要跳过向导再次启动网关，运行 `melo serve`。用 `melo serve --port 0` 时由系统选一个空闲端口：启动信息会显示真实地址，`melo console --port PORT` 和 `melo-task --gateway URL` 需要用它。

<a id="run-a-task-through-the-gateway"></a>

## 通过网关运行任务

网关运行时，`melo-task` 把简报发给它（简报写法见[运行 worker 任务](#run-a-worker-task)），并列出任务：

```shell
melo-task new --brief brief.md
melo-task list
```

示例输出（任务 ID 和模型的工作每次不同；任务在默认工作区 `~/localmelo-workspace` 中进行）：

```text
c5b2d38ceecc  queued
ID            LEVEL  STATE  TITLE
c5b2d38ceecc  P2     held   Write hello.py so that it prints the word hello.
  Waiting for your approval: file_write hello.py (answer it in the web console, Tasks)
```

默认设置下，写文件要等你批准，所以任务处于 `held` 状态，直到你作出回答。模型每要用一个工具，任务就等待一次：自己运行检查的模型也会请求 `python_exec`，回答前任务会再次处于 `held` 状态。`melo-task show` 会指出它在等哪个工具。在终端中运行 `melo console` 打开控制台，然后在控制台的 Tasks 视图中回答该请求。之后查看任务：

```shell
melo-task show c5b2d38ceecc
```

```text
c5b2d38ceecc  done
  Title: Write hello.py so that it prints the word hello.
  Level: P2
```

只有任务有原因或在等待某事时，`show` 才会增加 `Reason:` 和 `Waiting:` 行。`melo-task --help` 列出其他命令。

<a id="interactive-conversations"></a>

## 交互式对话

不带任务运行 `melo`，即可在终端中对话。每次工具调用完成时都会显示，模型工作时会以旋转指示器显示 melo 在做什么和用时，答案以 Markdown 输出，该轮改动的文件会在页脚下点出：

```shell
melo
# Or talk to an OpenAI-compatible server that is not saved. Without
# --model it uses the server's only chat model or its one loaded model;
# with several chat models and not exactly one loaded, it exits 2 and lists them:
melo --url http://127.0.0.1:8000/v1
melo --url http://127.0.0.1:8000/v1 --model YOUR_MODEL_ID
```

- 输入 `/` 查看命令（`/help` 列出常用命令，`/help all` 列出全部）：`/model` 打开服务器的模型列表并切换模型，`/backend` 切换到另一个服务器（两者都会保存为默认值，`melo --url` 会话除外），`/new` 重新开始，`/exit` 或 Ctrl+D 退出。
- melo 运行 shell 命令、Python 代码或写入文件前会询问：按 `1` 允许，按 `2` 在本会话中允许该工具，按 `3` 拒绝并告诉 melo 原因。Esc 停止当前轮次。
- melo 在当前文件夹中工作（`--workspace DIR` 可选择其他文件夹）：文件工具限制在该文件夹内；命令和 Python 可以访问其他文件，因此 melo 会先询问。

[交互式界面](../guides/terminal.md#the-interactive-ui)介绍了按键和其他内容。`melo ""` 打开早期版本的普通对话循环。没有终端时，单独运行 `melo` 会以退出码 2 退出；请使用 `melo serve` 启动网关。

`http://127.0.0.1:8401/` 上的网页工作区是控制台的实时页面，只与 `/api` 通信：它使用网关配置的模型，因此 `melo setup` 会改变该模型。首次运行、还没有设置模型时，欢迎页会显示一张卡片，找到这台 Mac 上已在运行的模型服务器，点按 **Use this model** 即可连接，无需打开设置。它按最新优先列出对话，支持搜索和每个对话各自的草稿，并像终端一样展示每一轮：你的消息、每次工具调用的工作卡片、折叠的推理和反思、通知、实时进度行，以及带事实和 `Copy response` 的答案。停止操作会取消当前轮次，等待中的审批显示为 `Needs approval`，对话保存在网关的 `sessions.db` 中，因此重启后仍然存在。同一网关上的其他浏览器标签页可看到相同的对话。工作区提供审批卡片、连接设置和会话详情抽屉；见[网页工作区](../guides/web-console.md#web-workspace)。

<a id="next-steps"></a>

## 下一步

- [配置](../reference/configuration.md)：后端、云端 API 密钥、超时。
- [工具和 MCP](../reference/tools.md)：选择内置工具并添加 MCP 服务器。
- [网关和 HTTP API](../guides/gateway.md)：会话、网页工作区、登录时运行。
- [Python API](../reference/python-api.md)：从 Python 运行任务、跟踪事件、审批工具调用。
- [记忆](../../reference/memory.md)（英文）：跨运行保留记忆、检查和重置记忆。
- [命令行参考](../../reference/cli.md)（英文）：所有选项、退出码和脚本编写。
- [故障排查](../../reference/troubleshooting.md)（英文）：错误消息和解决办法。
