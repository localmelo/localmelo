<a id="guide-model-servers"></a>

# 模型服务器指南

自行运行模型服务器并让 localmelo 连接到它：oMLX、Ollama、MLC LLM、vLLM、SGLang 或云端 API。设置名称和默认值见[配置参考](../reference/configuration.md)。

- 在搭载 Apple silicon 的 Mac 上，推荐使用本地服务器 oMLX。
- 在其他系统上，可以用 Ollama 作为替代。
- 云端 API 只需要把密钥放进一个环境变量。

<a id="setup-preview"></a>

## 设置预览

- 未找到服务器时，终端首次运行和 `melo setup` 会在受支持的 Apple silicon Mac 上预览 oMLX 0.7.3 与 Qwen3 4B Instruct（4-bit）。固定版本的模型约 2.28 GB；官方运行时 DMG 在 macOS 15 上约 0.84 GB，在 macOS 26/27 上约 0.88 GB。已缓存的模型文件会减少仍需下载的字节数。磁盘负载只是下限，展开和暂存开销尚未测量；读数不保证空间足够，也不代表运行内存需求。
- 预览显示缓存位置、可用磁盘空间和拟采用的生命周期（默认随会话退出而停止）。它不写设置、不预留端口。`Review setup plan` 中的 `Set up and start` 目前只说明自动设置尚未开放并显示手动步骤，不会安装、下载、启动服务器，也不保存长期授权。
- 手动步骤由同一个固定版本配方生成：模型保留在标准 Hugging Face 缓存中，服务器使用独立的设置目录，并关闭 oMLX 的额外缓存扫描，让这个实例只提供所选快照。自行安装官方应用，准备好后再复制显示的命令。建议的高位端口只是示例，并未预留；启动前请选一个空闲端口，然后在 melo 中输入该地址。
- 直接指向快照时，服务器提供的模型 ID 可能是修订哈希，而不是仓库名称，请使用服务器实际列出的 ID。重新检查地址只列出模型，不会加载模型。网页/应用设置卡片以及已有配置或显式指定地址的运行保持原有行为。

<a id="serve-a-model-with-omlx"></a>

## 使用 oMLX 提供模型服务

- [oMLX](https://github.com/jundot/omlx) 在搭载 Apple silicon 且运行 macOS 15 或更新版本的 Mac 上，通过 OpenAI 兼容 API 提供 MLX 模型服务。
- 首次运行时的设置会在 8000 端口找到它，并询问哪些模型已加载。
- 在配置文件中它是 `vllm` 后端，也可写作 `omlx`（[原因](model-servers.md#omlx)）。

1. 使用 Homebrew 安装 oMLX：

   ```bash
   brew tap jundot/omlx https://github.com/jundot/omlx
   brew install jundot/omlx/omlx
   ```

   oMLX macOS 应用也会安装该命令，路径为 `~/.omlx/bin/omlx`，并内置了少数模型系列（例如 Qwen3.5）全速运行所需的可选原生内核；其他安装方式见其 [README](https://github.com/jundot/omlx#install)（英文）。

2. 在一个模型目录中，为每个模型分别建立文件夹：oMLX 会提供每个包含模型的子文件夹（其中有 `config.json` 和权重），子文件夹名就是模型 ID。使用 Hugging Face 命令行（`hf`，通过 `uv tool install huggingface_hub` 安装）：

   ```bash
   hf download mlx-community/Qwen3-8B-4bit --local-dir ~/models/Qwen3-8B-4bit
   ```

3. 启动服务器，让它在自己的终端中持续运行：

   ```bash
   omlx serve --model-dir ~/models --port 8000
   ```

   - 它监听 `http://127.0.0.1:8000`；oMLX 会把 `--model-dir` 和 `--port` 保存在 `settings.json` 中（在 `~/.omlx` 下），所以先前保存的端口会盖过默认的 8000：显式指定 `--port` 可以避免这个问题。
   - oMLX 应用和 `omlx start` 会改在后台运行服务器，使用 oMLX 中保存的文件夹和端口（Homebrew 服务使用 `~/.omlx/models` 和 8000 端口）：此时跳过这条命令。
   - 保持默认主机地址，不要设置 `--api-key`：localmelo 不向本地服务器发送 API 密钥。
   - HTTP 401（`API key required`）说明之前设置过密钥，在 oMLX 管理页面或通过 `OMLX_API_KEY`：请在管理页面的设置中开启 **Allow inference without API key**，或取消设置 `OMLX_API_KEY`。
   - 开启该设置后，列出模型和对话都可以使用，但 oMLX 仍要求用密钥访问报告哪些模型已加载的路由，因此设置无法标出已加载的模型。

4. 检查服务器是否列出模型：

   ```bash
   curl -sS http://127.0.0.1:8000/v1/models
   ```

   响应的 `data` 列表中有一项包含 `"id": "Qwen3-8B-4bit"`（它也可能列出 Hugging Face 缓存中的 MLX 模型）。`Failed to connect` 表示该端口没有服务监听：请检查 oMLX 启动日志中的端口。`API key required` 表示已设置密钥：见第 3 步。

5. 使用该 ID 运行任务：

   ```bash
   melo --url http://127.0.0.1:8000/v1 --model Qwen3-8B-4bit "What is 6*7?"
   ```

   - oMLX 在首次请求时加载模型，大模型可能需要一分钟；如果该请求超时，添加 `--request-timeout 600`。
   - 不指定 `--model` 时，melo 会询问 oMLX 有哪些模型，并使用唯一的对话模型或已加载的那一个，对任务和不带任务的 `melo --url http://127.0.0.1:8000/v1`（交互式界面）都是如此。
   - 有多个对话模型、且不是恰好一个已加载时，会以退出码 2 退出，列出这些模型，并要求使用 `--model <id>`。

要将 oMLX 设为默认值，在保存任何配置之前在终端运行 `melo`：它会在端口 8000 找到 oMLX，列出模型并保存你的选择（[首次运行](configure.md#first-run)展示了整个过程）。oMLX 在第一个请求到达时才加载模型，因此列出的模型不一定在内存中。列出多个对话模型时，设置会向 oMLX 询问哪些模型已加载（`GET /v1/models/status`，最多等一秒），并按下面的方式选择：

| 服务器列出 | 设置的做法 |
|---|---|
| 一个对话模型 | 它就是默认值。 |
| 有多个，且 oMLX 说明了哪些已加载 | 给已加载的模型标上 `(loaded)`；其中第一个是默认值。 |
| 有多个，但没有已加载的，或 oMLX 没有说明 | 没有默认值：不高亮任何一项，在你用 ↑ ↓ 移到某个模型之前 Enter 不起作用（答案从管道输入时：输入模型的编号），因为替你选择可能会开始漫长的加载。 |

- 设置了密钥时，即使开启了 **Allow inference without API key**，oMLX 也不会说明哪些已加载，此时设置会输出 `The server did not say which models are loaded (it needs a key for that).`
- 在 oMLX 中设置了别名的模型会以别名列出，状态回复会在文件夹名旁边给出该别名（`model_alias`），所以该模型会被标出，并且可以成为默认值；状态回复中没有别名的服务器无法标出这样的模型。

已有配置时，先运行第 5 步以加载模型，再运行 `melo setup`：

- 在默认模式下，它会用首次运行的问题提供它找到的服务器。
- 在完整菜单中（选择 `Use another server` 之后，或处于开发者模式），选择 oMLX 或 vLLM（`3`），输入 `http://127.0.0.1:8000/v1` 和模型 ID；除非 oMLX 也提供嵌入模型，否则选择 `5`（None）作为嵌入后端。
- [oMLX 配置](model-servers.md#omlx)展示了保存后的文件，以及关闭 Qwen3 模型思考时要添加的表；网页工作区随后会使用该模型。

<a id="serve-a-model-with-ollama"></a>

## 使用 Ollama 提供模型服务

[Ollama](https://ollama.com) 是其他系统上的替代方案。从其官网安装后，启动服务器并拉取一个模型：

```bash
ollama serve
ollama pull qwen3:8b
```

- `ollama serve` 会一直运行，所以请在另一个终端运行 `ollama pull`。
- 如果 Ollama 已经以应用或服务的形式运行，跳过第一条命令。
- 首次运行的设置会在 11434 端口找到 Ollama 并列出它的模型。

<a id="backends"></a>

## 后端

| 键 | 类型 | 服务器或 API | 默认 URL | 嵌入 |
|---|---|---|---|---|
| `ollama` | 本地 | [Ollama](https://ollama.com) | `http://localhost:11434`（向导） | 支持 |
| `mlc` | 本地 | [MLC LLM](https://llm.mlc.ai) | `http://localhost:8400/v1`（向导） | 支持 |
| `vllm` | 本地 | [oMLX](https://github.com/jundot/omlx)、[vLLM](https://docs.vllm.ai) 或其他 OpenAI 兼容服务器 | `http://localhost:8000/v1`（向导） | 支持 |
| `sglang` | 本地 | [SGLang](https://docs.sglang.ai) | `http://localhost:30000/v1`（向导） | 支持 |
| `openai` | 云端 | OpenAI API | `https://api.openai.com/v1` | 不支持 |
| `anthropic` | 云端 | Anthropic API | `https://api.anthropic.com/v1` | 不支持 |
| `gemini` | 云端 | Google Gemini API | `https://generativelanguage.googleapis.com/v1beta/openai` | 不支持 |
| `nvidia` | 云端 | NVIDIA API | `https://integrate.api.nvidia.com/v1` | 不支持 |

- 向导会建议本地 URL；请使用你的服务器地址和端口。
- localmelo 只负责连接本地服务器：你需要自行启动和停止服务器。
- `mlc`、`vllm` 和 `sglang` 使用普通的 OpenAI 兼容 API，因此其他 OpenAI 兼容服务器可以搭配其中任何一个使用。
- 在配置文件中，`omlx` 是 `vllm` 的另一种写法（[详情](model-servers.md#omlx)）。

### oMLX

[oMLX](https://github.com/jundot/omlx) 是受支持的本地服务器；[使用 oMLX 提供模型服务](model-servers.md#serve-a-model-with-omlx)介绍了如何启动它。

- 它的 API 与 OpenAI 兼容，而 `vllm` 键代表任何 OpenAI 兼容服务器，所以 oMLX 就是 `vllm` 后端，配置文件里写作 `omlx`。
- 服务器的模型列表把 oMLX 标为所列模型的所有者时，首次运行的设置会保存 `chat_backend = "omlx"`，使用该服务器的嵌入模型时还会保存 `embedding_backend = "omlx"`；其他服务器保存为 `vllm`。
- 两种写法按同一个后端加载，设置在 `[vllm]` 表中，并按原样保存回去。
- 配置文件写 `omlx`，或服务器的模型列表显示它是 oMLX 时，`/status`、开发者模式的横幅和网页控制台把这台服务器称为 oMLX，否则称为 OpenAI 兼容服务器。

```toml
chat_backend = "omlx"
embedding_backend = "none"
request_timeout_s = 600

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B-4bit"

[vllm.extra_body.chat_template_kwargs]
enable_thinking = false
```

- `chat_model` 是模型在 oMLX 的 `GET /v1/models` 中的 ID：文件夹名称，或在 oMLX 中为其设置的别名；`--model ID` 可在一次运行中替换它。
- [首次运行](configure.md#first-run)会写入 `chat_backend`、`chat_url` 和 `chat_model` 行（有嵌入模型时还会写入 `embedding_backend = "omlx"`，以及该表的 `embedding_url` 和 `embedding_model`）；其他行请手动添加。
- `request_timeout_s = 600` 为 oMLX 留出时间，加载尚未加载的模型；见[对话请求超时](../reference/configuration.md#chat-request-timeout)。
- `extra_body` 表会关闭 Qwen3 模型的思考，使每一步更快；省略该表可保持思考开启。除非模型在 oMLX 中的设置强制指定 `enable_thinking`，否则 oMLX 会应用它。
- 要使用长期记忆，oMLX 可以从同一目录提供嵌入模型：设置 `embedding_backend = "omlx"`，并在 `[vllm]` 中添加 `embedding_url = "http://127.0.0.1:8000/v1"` 和 `embedding_model = "<its id>"`；由于 `GET /v1/models` 不给出类型，需要使用 ID 发送一次 `POST /v1/embeddings` 进行检查。见[配置嵌入模型](embedding.md#omlx)。

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
