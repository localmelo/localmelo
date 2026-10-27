<p align="center">
  <img src="docs/assets/melo.png" alt="Melo，localmelo 的吉祥物" width="200">
</p>

# localmelo

[English](README.md) | [简体中文](README.zh-CN.md)

一个本地个人智能体：在你自己的模型服务器或云端 API 上，借助工具和记忆完成任务。你可以在命令行、脚本和基准测试中用它运行任务（每次运行输出一行 JSON 和固定的退出码），也可以通过 HTTP 调用它。

[网站](https://localmelo.github.io/localmelo/zh-CN/index.html) · [快速开始](https://localmelo.github.io/localmelo/zh-CN/quickstart.html) · [文档（英文）](docs/README.md) · [路线图](https://localmelo.github.io/localmelo/zh-CN/updates.html#roadmap)

**状态**：早期预览版（pre-alpha）。在命令行中运行任务（直接模式）和 HTTP 网关现在已经可以使用。在 v0.1.0（里程碑计划于 2026-12-20 完成）之前，命令、HTTP API 和 Python API 都可能变化。

## 功能

- **一条命令完成一个任务。** `melo "<task>"` 在步数上限内运行，两次尝试之间会进行反思，还可以设置运行时限。在终端里，一行[进度行（英文）](docs/cli.md#progress-line)显示任务进行到哪里；保留记忆时，任务再次运行会显示之前运行的步骤。
- **使用工具。** 四个内置工具（`shell_exec`、`python_exec`、`file_read`、`file_write`），以及 MCP 服务器提供的工具（stdio 服务器，可在直接模式和 Python API 中使用）。
- **报告每一次运行**：输出文本或一行 JSON，并使用固定的退出码，脚本和基准测试可以放心依赖。
- **连接你的模型。** 你自己运行的模型服务器（Ollama、oMLX、MLC LLM、vLLM、SGLang），或云端 API（OpenAI、Anthropic、Google Gemini、NVIDIA）。
- **有记忆。** 运行期间的工作记忆、任务和步骤的历史记录、配置了嵌入服务器时的长期记忆，以及程序记忆：每个反复运行的任务的各次运行做了什么，在每次运行结束时记录。开启持久化后，历史记录、长期记忆和程序记忆会保存到 SQLite。
- **通过 HTTP 提供同一个智能体**（网关），支持会话，并附带一个在浏览器中使用智能体的网页工作区（模型服务器由你选择）。

localmelo 不负责下载、启动或部署模型：模型服务器需要你自己运行。

## 环境要求

- macOS（在 macOS 上开发和测试）或 Linux（应该也能运行，但尚未测试）。不支持 Windows。
- Python 3.11 或更高版本，以及 Git。
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip。
- 一个你自己运行的模型服务器，例如 [Ollama](https://ollama.com)；或者一个云端 API 的密钥。

## 安装

```bash
git clone https://github.com/localmelo/localmelo.git
cd localmelo
uv sync --extra gateway
source .venv/bin/activate
```

请保留目录名 `localmelo`：软件包就是从这个目录构建的。重命名后的目录或 ZIP 下载（`localmelo-main`）会报错 `ModuleNotFoundError: No module named 'localmelo'`。

不使用 uv 时，请看快速开始中的[用 pip 安装](https://localmelo.github.io/localmelo/zh-CN/quickstart.html#install-with-pip)一节。

下面的示例直接调用 `melo`。激活虚拟环境只对当前终端有效：打开新终端后，需要再次运行 `source <clone>/.venv/bin/activate`（`<clone>` 是你的 `localmelo` 目录），或者在克隆目录中运行 `uv run melo …`。更新或卸载 localmelo 的方法见[配置（英文）](docs/configuration.md#update-localmelo)。

## 快速开始

1. 启动模型服务器。下面以 [Ollama](https://ollama.com) 为例：安装并启动它，然后用 `ollama pull qwen3:8b` 下载一个模型。任何兼容 OpenAI 接口的服务器都可以，使用它的 `/v1` 地址即可。

2. 运行一个任务。模型可以用你的账户权限使用工具，详见[安全](#安全)。

   ```bash
   melo --base-url http://127.0.0.1:11434/v1 --chat-model qwen3:8b "What is 6*7?"
   ```

   答案会输出到 stdout；运行完成时退出码为 0。加上 `--output-format json`，输出就会变成一行 JSON，记录这次运行的结果。`--base-url` 不会发送 API 密钥；使用云端 API 时，请直接跳到第 3 步。

3. 保存配置并启动网关。使用云端 API 时，请先在同一个终端中运行 `export OPENAI_API_KEY="<your key>"`（或你所用服务商对应的变量）：向导的测试请求会用到这个密钥。

   ```bash
   melo --serve --reconfigure
   ```

   配置向导会询问对话后端、它的地址和模型，以及嵌入后端（没有嵌入服务器时选择 `None`）。向导会检查服务器能否连通，保存 `~/.cache/localmelo/config.toml`，然后在 `http://127.0.0.1:8401` 上提供 HTTP API 和网页工作区。按 Ctrl+C 停止网关。之后运行 `melo "<task>"` 时会使用保存的配置。

[快速开始](https://localmelo.github.io/localmelo/zh-CN/quickstart.html)会更详细地介绍每一步，包括向导的选项和第一个 HTTP API 请求。

## 三种运行方式

| 命令 | 作用 |
|---|---|
| `melo "<task>"` | 运行一个任务，输出结果后退出。 |
| `melo ""`（任务为空字符串） | 在终端中打开对话循环。输入 `/help` 查看命令，输入 `/exit` 退出。 |
| `melo --serve`，或者只运行 `melo` | 启动网关：在 `http://127.0.0.1:8401/` 上提供 HTTP API 和网页工作区。 |

不保存配置，直接和 oMLX 或其他兼容 OpenAI 接口的服务器对话：

```bash
melo --base-url http://127.0.0.1:8000/v1 --chat-model YOUR_MODEL_ID ""
```

在网页工作区中，输入服务器地址，选择它的一个模型，再创建对话。你可以保留多个对话、搜索对话、阅读按 Markdown 渲染的回答，也可以停止运行。对话只在网关进程运行期间保留，工作区也只供你自己的电脑使用，见[网页工作区（英文）](docs/gateway.md#web-workspace)。终端对话循环与网页对话相互独立。

## 文档

| 页面 | 内容 |
|---|---|
| [快速开始](https://localmelo.github.io/localmelo/zh-CN/quickstart.html) | 安装、运行第一个任务、保存配置、启动网关 |
| [命令行参考（英文）](docs/cli.md) | `melo` 的模式、选项、输出、退出码和信号 |
| [配置（英文）](docs/configuration.md) | 配置向导、配置文件、后端、超时、环境变量 |
| [工具和 MCP（英文）](docs/tools.md) | 内置工具、MCP 服务器、每一步提供哪些工具、检查 |
| [网关和 HTTP API（英文）](docs/gateway.md) | HTTP API、会话、网页工作区、在 macOS 登录时自动运行 |
| [记忆（英文）](docs/memory.md) | localmelo 记住什么，以及如何保留、查看和重置记忆 |
| [Python API（英文）](docs/python-api.md) | 在 Python 代码中运行任务，支持事件和审批（实验性） |
| [故障排查（英文）](docs/troubleshooting.md) | 常见错误信息及解决方法 |

[全部文档（英文）](docs/README.md)列出了所有页面，包括事件、会话和离线检索的参考文档。[评测（英文）](docs/evaluations.md)报告了各项默认设置背后的对比实验，例如反思的继续判定。

AI 智能体和 LLM 工具请读 [localmelo for agents（英文）](docs/for-agents.md)，它用一页写清使用 localmelo 所需的内容；文档索引见 [llms.txt](https://localmelo.github.io/localmelo/llms.txt)。

## 路线图

路线图记录在 [#2](https://github.com/localmelo/localmelo/issues/2) 中，v0.1.0 里程碑计划于 2026-12-20 完成。各阶段和按日期排列的更新见[网站](https://localmelo.github.io/localmelo/zh-CN/updates.html#roadmap)。

计划中：

- 交互式命令行：不带任务运行 `melo` 时打开，并实时显示智能体的工具调用；
- 网关重启后仍然保留的对话，并在网页工作区里实时显示智能体的工作；
- 把模型的推理过程保存为记忆笔记，并可切换检索方式（`bm25`、`llm`、`embedding`）；
- 在一个本地模型服务器上同时运行多个任务，并优先回答你的对话；
- 进度行上的百分比进度条，用于反复运行的任务；在重放证明它可靠之前保持关闭。

## 安全

> [!WARNING]
> 工具以你的账户权限运行，并且没有沙箱：模型可以运行 shell 命令和 Python 代码，也可以读写你能访问的任何文件。网关没有身份验证，所以请让它只监听 `127.0.0.1`（默认值）：在这个地址上，它会拒绝其他网站的网页通过浏览器向它的 API 发来的请求。详见 [SECURITY.md（英文）](SECURITY.md)和[检查能做什么（英文）](docs/tools.md#what-the-checks-do)。

实验性的 [Python API（英文）](docs/python-api.md)可以在工具运行前请求审批，并把文件工具限制在一个目录内；直接模式和网关都没有这两项。记忆文件和会话文件都没有加密，详见[隐私（英文）](docs/memory.md#privacy)。

## 获取帮助

先在 [GitHub issues](https://github.com/localmelo/localmelo/issues) 中搜索，找不到再新建一个。请附上 `--output-format json` 输出的那一行 JSON（其中包含 localmelo 的版本）、退出码、后端和模型、你的操作系统和 Python 版本，以及复现步骤。提交前请删除 API 密钥和个人路径。安全问题请按 [SECURITY.md（英文）](SECURITY.md)中的方式报告，不要公开提交 issue。

## 参与贡献

欢迎提交问题报告、文档修正，以及小而聚焦的 pull request。开发环境的搭建和需要运行的检查，请看 [CONTRIBUTING.md（英文）](CONTRIBUTING.md)。

## 许可证

MIT © The localmelo Authors。详见 [LICENSE](LICENSE)。
