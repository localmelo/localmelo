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
- output size limits
- prompt 和 memory write size limits
- onboarding 阶段的 provider probing

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

## 开发

安装开发依赖：

```bash
pip install -e ".[dev,gateway]"
```

运行 focused checks：

```bash
python -m pytest tests/agent tests/checker tests/executor \
                 tests/cli tests/memory tests/integration -q
python -m pytest tests/gateway -q
python -m ruff check .
```

合并前运行完整测试：

```bash
python -m pytest tests/ -q
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
