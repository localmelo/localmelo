<p align="center">
  <img src="docs/assets/melo.png" alt="Melo，localmelo 的吉祥物" width="200">
</p>

# localmelo

[English](README.md) | [简体中文](README.zh-CN.md)

一个本地个人智能体：在你自己的模型服务器或云端 API 上，借助工具和记忆完成任务。你可以在命令行、脚本和基准测试中用它运行任务（每次运行输出一行 JSON 和固定的退出码），也可以通过 HTTP 调用它。

[网站](https://localmelo.github.io/localmelo/zh-CN/index.html) · [快速开始](https://localmelo.github.io/localmelo/zh-CN/quickstart.html) · [文档（英文）](docs/README.md) · [路线图](https://localmelo.github.io/localmelo/zh-CN/updates.html#roadmap)

**状态**：早期预览版（pre-alpha）。在命令行中运行任务（直接模式）和 HTTP 网关现在已经可以使用。在 v0.1.0（里程碑计划于 2026-12-20 完成）之前，命令、HTTP API 和 Python API 都可能变化。

## 功能

- **一条命令完成一个任务。** `melo "<task>"` 在步数上限内运行，两次尝试之间会进行反思，还可以设置运行时限。
- **使用工具。** 四个内置工具（`shell_exec`、`python_exec`、`file_read`、`file_write`），以及 MCP 服务器提供的工具（stdio 服务器，可在直接模式和 Python API 中使用）。
- **报告每一次运行**：输出文本或一行 JSON，并使用固定的退出码，脚本和基准测试可以放心依赖。
- **连接你的模型。** 你自己运行的模型服务器（Ollama、MLC LLM、vLLM、SGLang），或云端 API（OpenAI、Anthropic、Google Gemini、NVIDIA）。
- **有记忆。** 运行期间的工作记忆、任务和步骤的历史记录，以及配置了嵌入服务器时的长期记忆。开启持久化后，历史记录和长期记忆会保存到 SQLite。
- **通过 HTTP 提供同一个智能体**（网关），支持会话，并附带一个用来试用的简单网页。

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

下面的示例直接调用 `melo`。激活虚拟环境只对当前终端有效：打开新终端后，需要再次运行 `source <clone>/.venv/bin/activate`，其中 `<clone>` 是你的 `localmelo` 目录。使用 uv 时，也可以在克隆目录中运行 `uv run melo …`，或在其他任何位置运行 `uv run --project <clone> melo …`。

更新或卸载 localmelo 的方法见[配置（英文）](docs/configuration.md#update-localmelo)。

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

   配置向导会询问对话后端、它的地址和模型，以及嵌入后端：

   - 如果第 1 步启动的是 Ollama，请选择 `2`，然后按回车接受建议的地址和模型。
   - 如果是其他兼容 OpenAI 接口的服务器，请选择 vLLM（`3`），然后输入它的地址和模型名。
   - 使用云端 API 时，请选择对应的服务商。
   - 如果没有嵌入服务器，嵌入后端请选择 `5`，即 `None (no long-term memory)`。

   向导会检查服务器能否连通，保存 `~/.cache/localmelo/config.toml`，然后在 `http://127.0.0.1:8401` 上提供服务。在另一个终端中运行：

   ```bash
   curl http://127.0.0.1:8401/v1/health
   curl -X POST http://127.0.0.1:8401/v1/agent/run \
     -H "Content-Type: application/json" \
     -d '{"query": "What is 6*7?"}'
   ```

   按 Ctrl+C 停止网关。之后运行 `melo "<task>"` 时会使用保存的配置。

接下来，[快速开始](https://localmelo.github.io/localmelo/zh-CN/quickstart.html)会更详细地介绍每一步。

## 交互式工作区

不带任务运行 `melo`，即可使用已配置的后端开始终端对话。用 `/help` 查看命令，或用 `/exit` 退出。也可以直接连接 oMLX 或其他兼容 OpenAI 接口的服务器：

```bash
melo --base-url http://127.0.0.1:8500/v1 --chat-model YOUR_MODEL_ID
```

运行 `melo --serve`，然后打开 `http://127.0.0.1:8401/`，即可使用网页工作区。发现服务器上的模型，选择模型，再创建对话。嵌入和初始记忆都是可选项。对话支持搜索、独立草稿、代码格式化、取消运行和响应式导航，并使用 Melo 现有的图像。同一网关上的浏览器客户端会同步对话状态；只要网关进程仍在运行，刷新页面就能恢复对话。直接运行的命令行对话与网页对话相互独立。

如果想先试用网页工作区，而不配置主网关：

```bash
uv run python -m localmelo.support.gateway._test_webapp
```

拟议的 iPhone/Mac 架构，以及经过身份验证的远程访问还需要哪些工作，见 [Apple 客户端方向（英文）](docs/mobile-client-direction.md)。

## 文档

| 页面 | 内容 |
|---|---|
| [快速开始](https://localmelo.github.io/localmelo/zh-CN/quickstart.html) | 安装、运行第一个任务、保存配置、启动网关 |
| [命令行参考（英文）](docs/cli.md) | `melo` 的模式、选项、输出、退出码和信号 |
| [运行结果（英文）](docs/run-results.md) | 一次运行有哪些结束方式、运行记录、重试 |
| [配置（英文）](docs/configuration.md) | 配置向导、配置文件、后端、超时、环境变量 |
| [工具和 MCP（英文）](docs/tools.md) | 内置工具、MCP 服务器、每一步提供哪些工具、检查 |
| [网关和 HTTP API（英文）](docs/gateway.md) | HTTP API、会话、网页界面、在 macOS 登录时自动运行 |
| [Apple 客户端方向（英文）](docs/mobile-client-direction.md) | 拟议的 iPhone/Mac 客户端、共享状态、经过身份验证的远程访问 |
| [工作区评审（英文）](docs/ui-review-2026-10-24.md) | 界面与智能体运行框架的修改、验证结果、当前限制 |
| [记忆（英文）](docs/memory.md) | localmelo 记住什么，以及如何保留、查看和重置记忆 |
| [Python API（英文）](docs/python-api.md) | `Agent`、`AgentOptions` 和 `RunResult`（实验性） |
| [localmelo 的工作原理（英文）](docs/how-it-works.md) | 运行循环、反思、记忆和超时 |
| [故障排查（英文）](docs/troubleshooting.md) | 常见错误信息及解决方法 |
| [架构](https://localmelo.github.io/localmelo/zh-CN/architecture.html) | 各部分如何协同工作（网站） |
| [路线图](https://localmelo.github.io/localmelo/zh-CN/updates.html#roadmap) | 计划和更新记录（网站） |

## 路线图

路线图记录在 [#2](https://github.com/localmelo/localmelo/issues/2) 中。v0.1.0 里程碑计划于 2026-12-20 完成。

| 阶段 | 日期 | 范围 |
|---|---|---|
| 第 1 阶段 | 2026-10-19 至 2026-11-01 | 核心循环 v2 和命令行（[#3](https://github.com/localmelo/localmelo/issues/3)） |
| 第 2 阶段 | 2026-11-02 至 2026-11-22 | 记忆：笔记、检索，以及提升为长期记忆 |
| 第 3 阶段 | 2026-11-23 至 2026-12-06 | 安全和隐私（[#29](https://github.com/localmelo/localmelo/issues/29)） |

计划中：

- 由命令行、网页控制台和未来的移动客户端共享的持久化对话；
- 把模型的推理过程保存为记忆笔记，并可切换检索方式（`bm25`、`llm`、`embedding`）。

## 安全

> [!WARNING]
> 工具以你的账户权限运行，并且没有沙箱：模型可以运行 shell 命令和 Python 代码，也可以读写你能访问的任何文件。网关没有身份验证，所以请让它只监听 `127.0.0.1`（默认值）。详见 [SECURITY.md（英文）](SECURITY.md)和[检查能做什么（英文）](docs/tools.md#what-the-checks-do)。

## 获取帮助

先在 [GitHub issues](https://github.com/localmelo/localmelo/issues) 中搜索，找不到再新建一个。请附上 `--output-format json` 输出的那一行 JSON（其中包含 localmelo 的版本）、退出码、后端和模型、你的操作系统和 Python 版本，以及复现步骤。提交前请删除 API 密钥和个人路径。安全问题请按 [SECURITY.md（英文）](SECURITY.md)中的方式报告，不要公开提交 issue。

## 参与贡献

欢迎提交问题报告、文档修正，以及小而聚焦的 pull request。开发环境的搭建和需要运行的检查，请看 [CONTRIBUTING.md（英文）](CONTRIBUTING.md)。

## 许可证

MIT © The localmelo Authors。详见 [LICENSE](LICENSE)。
