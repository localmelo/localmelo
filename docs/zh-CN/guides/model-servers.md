<a id="guide-model-servers"></a>

# 模型服务器指南

本指南说明如何自行运行模型服务器并让 localmelo 连接到它，涵盖 oMLX、Ollama、MLC LLM、vLLM、SGLang 和云端 API。选择后端时请阅读本指南；设置名称和默认值见[配置参考](../reference/configuration.md)。

<a id="serve-a-model-with-omlx"></a>

## 使用 oMLX 提供模型服务

[oMLX](https://github.com/jundot/omlx) 在搭载 Apple silicon 且运行 macOS 15 或更新版本的 Mac 上，通过 OpenAI 兼容 API 提供 MLX 模型服务。要用它替代 Ollama：

1. 使用 Homebrew 安装 oMLX：

   ```bash
   brew tap jundot/omlx https://github.com/jundot/omlx
   brew install jundot/omlx/omlx
   ```

   oMLX macOS 应用也会安装该命令，路径为 `~/.omlx/bin/omlx`。其他安装方式见其 [README](https://github.com/jundot/omlx#install)（英文）。

2. 在一个模型目录中，为每个模型分别建立文件夹。oMLX 会提供每个包含模型的子文件夹（其中有 `config.json` 和权重），子文件夹名就是模型 ID。例如，使用 Hugging Face 命令行（`hf`，通过 `uv tool install huggingface_hub` 安装）：

   ```bash
   hf download mlx-community/Qwen3-8B-4bit --local-dir ~/models/Qwen3-8B-4bit
   ```

3. 启动服务器，让它在自己的终端中持续运行：

   ```bash
   omlx serve --model-dir ~/models --port 8000
   ```

   它监听 `http://127.0.0.1:8000`。oMLX 会将 `--model-dir` 和 `--port` 保存在 `settings.json` 中（默认为 `~/.omlx` 下），省略选项时会使用这些值。因此，先前保存的端口会覆盖默认的 8000；显式指定 `--port` 可以避免这个问题。保持默认主机地址，不要设置 `--api-key`：localmelo 不向本地服务器发送 API 密钥。如果请求返回 HTTP 401（`API key required`），说明之前在 oMLX 管理页面或通过 `OMLX_API_KEY` 设置过密钥：请在管理页面的设置中开启 **Allow inference without API key**，或取消设置 `OMLX_API_KEY`。

4. 检查服务器是否列出模型：

   ```bash
   curl -sS http://127.0.0.1:8000/v1/models
   ```

   响应的 `data` 列表中有一项包含 `"id": "Qwen3-8B-4bit"`。它也可能列出 Hugging Face 缓存中的 MLX 模型。`Failed to connect` 表示该端口没有服务监听：请检查 oMLX 启动日志中的端口。`API key required` 表示已设置密钥：见第 3 步。

5. 使用该 ID 运行任务：

   ```bash
   melo --base-url http://127.0.0.1:8000/v1 --chat-model Qwen3-8B-4bit "What is 6*7?"
   ```

   oMLX 在首次请求时加载模型，大模型可能需要一分钟。如果该请求超时，添加 `--request-timeout 600`。不带任务运行 `melo --base-url http://127.0.0.1:8000/v1` 会使用 oMLX 列出的第一个模型打开交互式界面。

要将 oMLX 设为默认值，在保存任何配置之前在终端运行 `melo`：它会在端口 8000 找到 oMLX，列出模型并保存你的选择。已有配置时，先运行第 5 步以加载模型，再运行向导（`melo --reconfigure`）：选择 vLLM（`3`）作为对话后端，输入 `http://127.0.0.1:8000/v1` 和模型 ID；除非 oMLX 也提供嵌入模型，否则选择 `5`（None）作为嵌入后端。[oMLX 配置](model-servers.md#omlx)展示了保存后的文件，以及关闭 Qwen3 模型思考时要添加的表。网页工作区随后会使用该模型；它会读取配置文件。

<a id="backends"></a>

## 后端

| 键 | 类型 | 服务器或 API | 默认 URL | 嵌入 |
|---|---|---|---|---|
| `ollama` | 本地 | [Ollama](https://ollama.com) | `http://localhost:11434`（向导） | 支持 |
| `mlc` | 本地 | [MLC LLM](https://llm.mlc.ai) | `http://localhost:8400/v1`（向导） | 支持 |
| `vllm` | 本地 | [vLLM](https://docs.vllm.ai) | `http://localhost:8000/v1`（向导） | 支持 |
| `sglang` | 本地 | [SGLang](https://docs.sglang.ai) | `http://localhost:30000/v1`（向导） | 支持 |
| `openai` | 云端 | OpenAI API | `https://api.openai.com/v1` | 不支持 |
| `anthropic` | 云端 | Anthropic API | `https://api.anthropic.com/v1` | 不支持 |
| `gemini` | 云端 | Google Gemini API | `https://generativelanguage.googleapis.com/v1beta/openai` | 不支持 |
| `nvidia` | 云端 | NVIDIA API | `https://integrate.api.nvidia.com/v1` | 不支持 |

向导会建议本地 URL；请使用你的服务器地址和端口。localmelo 只负责连接本地服务器：你需要自行启动和停止服务器。`mlc`、`vllm` 和 `sglang` 使用普通的 OpenAI 兼容 API，因此其他 OpenAI 兼容服务器可以搭配其中任何一个使用。

### oMLX

[oMLX](https://github.com/jundot/omlx) 提供 OpenAI 兼容 API，因此使用 `vllm` 后端。[使用 oMLX 提供模型服务](model-servers.md#serve-a-model-with-omlx)介绍了如何启动它。

```toml
chat_backend = "vllm"
embedding_backend = "none"
request_timeout_s = 600

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B-4bit"

[vllm.extra_body.chat_template_kwargs]
enable_thinking = false
```

- `chat_model` 是模型在 oMLX 的 `GET /v1/models` 中的 ID：文件夹名称，或在 oMLX 中为其设置的别名。
- [首次运行](configure.md#first-run)会写入 `chat_backend`、`chat_url` 和 `chat_model` 行：在没有配置文件的终端中运行 `melo`，同时让 oMLX 运行。当它提供嵌入模型时，还会写入 `embedding_backend = "vllm"`，以及该表的 `embedding_url` 和 `embedding_model`。其他行请手动添加。
- `request_timeout_s = 600` 为 oMLX 留出时间，以便在请求遇到未加载的模型时加载它；见[对话请求超时](../reference/configuration.md#chat-request-timeout)。
- `extra_body` 表会关闭 Qwen3 模型的思考，使每一步更快。省略该表可保持思考开启。除非模型在 oMLX 中的设置强制指定 `enable_thinking`，否则 oMLX 会应用上述表中的设置。
- 要使用长期记忆，oMLX 也可以从同一目录提供嵌入模型：设置 `embedding_backend = "vllm"`，并在 `[vllm]` 中添加 `embedding_url = "http://127.0.0.1:8000/v1"` 和 `embedding_model = "<its id>"`。oMLX 的 `GET /v1/models` 返回 ID 但不返回类型，因此无法区分嵌入模型和对话模型：需要使用 ID 发送一次 `POST /v1/embeddings` 进行检查。见[配置嵌入模型](embedding.md#omlx)。

<a id="ollama-with-embeddings"></a>

### Ollama 配嵌入

```toml
chat_backend = "ollama"
embedding_backend = "ollama"

[ollama]
chat_url = "http://127.0.0.1:11434"
chat_model = "qwen3:8b"
embedding_url = "http://127.0.0.1:11434"
embedding_model = "qwen3-embedding:0.6b"
```

先拉取两个模型：`ollama pull qwen3:8b` 和 `ollama pull qwen3-embedding:0.6b`。`nomic-embed-text` 是更小的替代模型，但仅支持英文。

<a id="mlc-llm-vllm-or-sglang"></a>

### MLC LLM、vLLM 或 SGLang

```toml
chat_backend = "vllm"
embedding_backend = "none"

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen/Qwen3-8B"
```

- 对于 MLC LLM 或 SGLang，使用 `[mlc]` 或 `[sglang]` 表及对应的 `chat_backend`。
- 将 `chat_model` 设置为服务器提供的模型。服务器的 `GET /v1/models` 会列出这些模型。
- 要从同类服务器获取嵌入，设置 `embedding_backend = "vllm"`（或 `mlc`、`sglang`），并在表中添加 `embedding_url` 和 `embedding_model`。

<a id="a-cloud-api"></a>

### 云端 API

```toml
chat_backend = "openai"
embedding_backend = "none"

[openai]
api_key_env = "OPENAI_API_KEY"
chat_model = "gpt-4o"
```

在运行 `melo` 的 shell 中导出密钥：

```bash
export OPENAI_API_KEY="<your key>"
melo "What is 6*7?"
```

对于其他提供方，使用 `[anthropic]`、`[gemini]` 或 `[nvidia]` 表。向导建议使用变量 `ANTHROPIC_API_KEY`、`GEMINI_API_KEY` 和 `NVIDIA_API_KEY`。

<a id="mix-backends"></a>

### 混用后端

对话后端和嵌入后端分别选择。任何对话后端都可以搭配 `mlc`、`ollama`、`vllm`、`sglang` 或 `none` 嵌入后端。例如，云端模型搭配本地 Ollama 嵌入：

```toml
chat_backend = "openai"
embedding_backend = "ollama"

[openai]
api_key_env = "OPENAI_API_KEY"
chat_model = "gpt-4o"

[ollama]
embedding_url = "http://127.0.0.1:11434"
embedding_model = "qwen3-embedding:0.6b"
```
