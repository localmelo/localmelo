<p align="center">
  <img src="docs/assets/melo.png" alt="Melo" width="240">
</p>

# localmelo

[English](./README.md) | [简体中文](./README.zh-CN.md)

`localmelo` 是一个本地优先的 agent runtime，面向显式 memory、受检查的
tool use，以及由用户自行管理的模型 backend。它把核心 agent loop 和基础设施
分开，让本地和云端 backend 可以通过配置接入，而不是改变 runtime 本身。

**状态：** pre-alpha。当前 online core loop 已经可以用于开发和 smoke test，
但公共 API、memory policy 和 personalization workflow 仍在演进。

**[文档](https://localmelo.github.io/localmelo/zh-CN/index.html)** |
**[快速开始](https://localmelo.github.io/localmelo/zh-CN/quickstart.html)** |
**[架构](https://localmelo.github.io/localmelo/zh-CN/architecture.html)** |
**[更新](https://localmelo.github.io/localmelo/zh-CN/updates.html)**

## 项目能力

localmelo 提供：

- online agent loop：planning、tool execution、reflection、memory writeback
- 显式 memory 分层：working context、history、long-term retrieval、tool lookup
- checker 边界：ingress、planning、execution、output size、memory write
- 本地和云端模型 provider 的 backend connector
- 围绕同一个 runtime 的 HTTP gateway session 模式
- 后续 sleep-time personalization pipeline 的基础骨架

localmelo 不负责管理本地模型 runtime。Ollama、MLC、vLLM、SGLang 或其他本地
服务需要用户自行启动，然后把 localmelo 指向对应 endpoint。

## 当前成熟度

现在可以验证：

- direct CLI 模式
- gateway 模式
- backend registry 和 provider 构建
- 带 workspace policy 的内置 tool execution
- async SQLite history 和 long-term memory 持久化
- focused tests 和 smoke checks

仍处于实验阶段：

- long-memory retrieval 和 promotion policy
- 产品级 deployment shell
- sleep-time training、evaluation 和 personalization
- 稳定公共 API 保证

## 快速开始

环境要求：

- Python 3.11+
- 已配置的 chat backend，例如 Ollama、MLC、vLLM、SGLang，或支持的云端 provider

从仓库根目录安装：

```bash
pip install -e ".[dev,gateway]"
```

配置 provider：

```bash
melo --reconfigure
```

运行一次 direct query：

```bash
melo "What is 6*7?"
```

启动 gateway：

```bash
melo --serve
```

调用 gateway：

```bash
curl http://127.0.0.1:8401/v1/health

curl -X POST http://127.0.0.1:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query":"Say hello briefly"}'
```

Gateway smoke checks、session 复用、no-embedding mode 和 backend-specific 验证
见 [快速开始](https://localmelo.github.io/localmelo/zh-CN/quickstart.html)。

## 架构

核心设计规则：

> `melo/` 负责 runtime behavior 和 contracts。`support/` 负责基础设施。
> Core runtime code 不应直接 import infrastructure implementations。

```text
localmelo/
  melo/       # agent, memory, checker, executor, sleep scaffolding
  support/    # backends, providers, gateway, config, onboarding
  tests/      # regression, integration, smoke coverage
  docs/       # static documentation site
```

核心模块：

| 区域 | 职责 |
|---|---|
| Agent | Retrieval、planning、tool calls、reflection、final answers |
| Memory | Working memory、history、long-term storage、tool registry |
| Checker | Runtime boundary 的结构化校验 |
| Executor | 带 timeout 和 workspace policy 的内置工具执行 |
| Support | Backend registry、provider implementations、config、gateway |

完整 runtime map 见
[架构文档](https://localmelo.github.io/localmelo/zh-CN/architecture.html)。

## 安全边界

localmelo 把 checks 作为 runtime control flow。Request、plan、tool call、
executor result 和 memory write 在跨越主要边界前都会被校验。

当前 safeguards 包括：

- 文件操作的 workspace policy
- blocked shell command patterns
- `shell_exec` 命令在 timeout 或 cancellation 时会连同其 process group（它启动的所有 process）一起被 kill
- execution allowlist：tool call 只有在其所在 planning step 向模型提供了该工具时才会执行
- output size limits
- prompt 和 memory write size limits
- onboarding 阶段的 provider probing

在默认的 `retrieve` tool exposure 下，每个 step 只提供 hint extraction 和 BM25
search 为它解析出的工具。调用其他已注册的工具会被拒绝，返回
"Tool '<name>' was not offered in this step."，工具不会执行；之前的版本会执行模型
指名的任何已注册工具。这个检查只在单个 step 内有效：被拒绝的调用会留在对话中，
所以 hint extraction 可能在下一个 step 提供该工具。使用 `tool_exposure="all"` 时，
每个 step 都提供所有已注册的工具。

这仍然是 pre-alpha software。在让它访问重要文件、账号或长期运行的 process
之前，请先 review 配置和 tool permissions。

## Backends

支持的 backend families：

| Backend | 类型 | 说明 |
|---|---|---|
| `ollama` | 本地 | 用户自行管理的 Ollama server |
| `mlc` | 本地 | 用户自行管理的 MLC endpoint |
| `vllm` | 本地 | 用户自行管理的 vLLM endpoint |
| `sglang` | 本地 | 用户自行管理的 SGLang endpoint |
| `openai` | 云端 | OpenAI API |
| `anthropic` | 云端 | Anthropic API |
| `gemini` | 云端 | Google Gemini API |
| `nvidia` | 云端 | NVIDIA API |

Chat backend 和 embedding backend 独立配置。如果不想运行 embedding retrieval，
可以把 embedding backend 设为 `none`。

每个 chat request 默认 120 s 后 timeout（`ollama` backend 为 300 s）。模型较慢时
（例如本地 thinking model），可在 `~/.cache/localmelo/config.toml` 顶层设置
`request_timeout_s`，或给 direct query 加 `--request-timeout SECONDS`；详见
[docs/quickstart.md](docs/quickstart.md#chat-request-timeout)。

## 开发

用 [uv](https://docs.astral.sh/uv/) 安装开发依赖，会按仓库里的 `uv.lock`
创建 `.venv/`：

```bash
uv sync --extra dev --extra gateway
```

运行 focused checks：

```bash
uv run pytest tests/agent tests/checker tests/executor \
              tests/cli tests/memory tests/integration -q
uv run pytest tests/gateway -q
uv run ruff check .
```

合并前运行完整测试：

```bash
uv run pytest tests/ -q
```

开发进度见 [更新](https://localmelo.github.io/localmelo/zh-CN/updates.html)
和 GitHub issues。

## 贡献

欢迎提交 issue 和 pull request。由于项目仍处于 pre-alpha，小而聚焦的改动会比
大范围 feature drop 更容易 review。

请保持 runtime/infrastructure 边界清晰：

- core agent behavior 放在 `melo/`
- backend、gateway、config、onboarding implementation 放在 `support/`
- 安全边界保持 checker-based control flow
- 修改 public command、storage、tool execution 或 provider contracts 时补测试

## 许可证

localmelo 以 [MIT License](LICENSE) 发布。
