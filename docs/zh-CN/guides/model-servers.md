<a id="guide-model-servers"></a>

# 模型服务器指南

自行运行模型服务器并让 localmelo 连接到它：oMLX、Ollama、MLC LLM、vLLM、SGLang、melo_serve 或云端 API。设置名称和默认值见[配置参考](../reference/configuration.md)。

- 在搭载 Apple silicon 的 Mac 上，推荐使用本地服务器 oMLX。
- 在其他系统上，可以用 Ollama 作为替代。
- 云端服务只需要一个密钥：在设置里粘贴一次，它会保存在你 Mac 的钥匙串里。见[云端模型](hosted-models.md)。

<a id="first-run-setup"></a>

## 首次运行的设置

- 已经在运行的服务器总是优先提供。尚未保存后端且没有找到服务器时，搭载 Apple silicon 的 Mac（macOS 15 或更新版本）上的首次运行会提议帮你设置一个。这是可选的：`Use a server I already run` 和 `Use a hosted service` 在同一个菜单里，选择 `Set up <model> (<size>)` 之前不会安装任何东西。若已经保存后端，`melo setup` 会改为打开完整后端菜单。
- melo 会向 [melo_serve](https://github.com/localmelo/melo_serve) 询问这台 Mac 能运行什么（`melo-serve plan`，只读），然后打印一个列表：芯片、内存、macOS 和可用磁盘空间，推荐的模型，以及每一项的大小和去向。已经有的项显示 `✓ already here`，不计入下载量；下载了一半的模型会显示已有多少。若尚未安装 melo_serve，uv 会把固定版本的副本取到自己的缓存里，只为了这次询问。
- 各项是：melo_serve 本身（用 `uv tool install` 安装固定版本）、对话后端（oMLX：固定的下载文件，会核对大小和校验和，再复制到 `~/Applications/oMLX.app`；Splash 通过 Homebrew 安装）、对话模型和嵌入模型（Hugging Face 缓存，遵循 `HF_HOME`，绝不放进项目文件夹），以及 melo_serve 的登录项（macOS 会显示“登录项”通知）。缓存所在的卷上必须放得下下载量，另加 5 GiB。
- `Choose what to install` 会逐项询问尚未安装的项。melo_serve 本身不提供跳过：不想用它就选 `Use a server I already run`。没有对话模型或它的后端，melo_serve 暂时无法对话，之后用 `melo-serve get <pick>` 补齐。没有嵌入模型或它的后端，记忆只按字词查找。没有登录项，`melo serve` 会在需要时启动 melo_serve。`Choose another model` 会从同一目录规划另一个选择，并重新显示列表。
- 同意之后，每一步打印一行，下载带进度条。按 Ctrl-C 会停下并保留已下载的内容；再选 `Set up` 即可继续。某一步失败时会说明原因并回到列表。对话模型能回答后，melo 会把 melo_serve 保存为对话服务器并进入对话。
- 管道和 `TERM=dumb` 只会看到列表和 `Run melo in a terminal to set this up.`：同意必须来自真人，所以管道从不开始设置。其他系统、Intel Mac、内存太小的 Mac 以及检查失败时，会用一行说明原因，并给出手动方式：下面的命令，加上 `Check again`、`Enter an address`、`Use a hosted service` 和 `Quit`。
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
| `melo_serve` | 本地 | [melo_serve](https://github.com/localmelo/melo_serve) | `http://127.0.0.1:8321/v1` | 支持 |
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

### melo_serve

[melo_serve](https://github.com/localmelo/melo_serve) 是单独的守护进程，每个用户一个，用一个与 OpenAI 兼容的地址 `http://127.0.0.1:8321/v1` 提供本地模型（Splash 或 oMLX），并用一条命令切换。

1. 安装它，也可以让它成为登录项：

   ```bash
   uv tool install git+https://github.com/localmelo/melo_serve
   melo-serve install
   ```

2. 设置后端，不写模型：

   ```toml
   chat_backend = "melo_serve"
   embedding_backend = "none"

   [melo_serve]
   chat_url = "http://127.0.0.1:8321/v1"
   ```

3. 想换模型时随时切换：在任一聊天里用 `/model`、控制台的模型标签、应用的 **Model** 菜单，或 `melo-serve use <pick>`；见[切换模型](#switch-the-model)。

- `chat_model` 为空表示 melo_serve 当前提供的模型。localmelo 在创建提供方时从 melo_serve 的状态文件读取它，因此切换会被跟上；`embedding_model` 也一样。正在运行的聊天会在下一轮跟上：网关的聊天（`melo` 主聊天、侧聊天、网页控制台和应用），以及启动时没有带 `--model` 的本地终端聊天（`[cli] chat = "local"`、`-c`、`-r`）。终端聊天会打印 `The model changed to <model> (via melo_serve).`，网关运行该轮时也一样。在某个模型上构建的聊天，若现在提供的是另一个模型，会在下一轮重新构建；正在运行一轮的聊天先完成这一轮。只有控制套接字接受连接，并且套接字和状态文件都属于你自己、所在文件夹只有你能写入时，melo_serve 才算在运行，所以崩溃或停止后留下的状态什么也不提供。
- 没有提供任何模型时，`melo "task"`（以及 `POST /v1/agent/run`，返回 502 `agent_start_failed`）会以一行说明要运行什么命令结束：`melo-serve up`、安装命令，或 `melo-serve use <pick>`。直接运行的退出码为 1。
- 首次运行的设置优先选择正在运行的 melo_serve，并且不写入模型。设置、设置页和启动时的检查都不需要已有模型在提供：是否有模型，在需要模型时才检查。
- 配置使用 melo_serve 时，`melo serve` 会检查它；见[网关](gateway.md#start-the-gateway)。
- localmelo 运行第一个只有你能写入的 `melo-serve` 文件，依次查找 PATH（须为绝对路径）和 `uv tool install` 的安装文件夹（`$UV_TOOL_BIN_DIR`，其次 `$XDG_BIN_HOME`，再其次 `~/.local/bin`；登录项的 PATH 里没有它）。
- melo_serve 的 [README](https://github.com/localmelo/melo_serve#readme) 介绍可选项（pick）、登录项和其他命令。

<a id="switch-the-model"></a>

#### 切换模型

- **终端：** `/model` 列出 melo_serve 的对话模型，当前模型标 `✓`，每行写出后端、需要的内存，以及 `fits` 或 `tight fit`。Enter 切换，Esc 保留当前模型。`/model NAME`（短名、ID 或名称）不打开列表直接切换。主聊天里也可以用。
- **控制台：** 页头的模型标签和输入框下方的模型按钮打开同一个列表，**设置 → 模型服务器** 和 `/model` 也一样。
- **应用：** 菜单栏菜单的 **Model** 子菜单。
- 本机的每个聊天都会跟上切换。切换先等待正在运行一轮的聊天（`Waiting for 1 chat to finish`，最多 10 分钟）；这期间发出的一轮会等待（`Waiting for the model switch`），然后在新模型上运行。**Cancel**（终端里按 Esc）结束等待；已经开始的加载会在 melo_serve 中继续。
- 需要的内存超过空闲内存的模型显示为灰色，写出它需要多少和还有多少空闲，不能选择。任务不会被挂起：模型加载期间，它的请求会等待并再次询问，然后用新模型继续。被终端里的 `melo-serve use` 赶上的聊天也一样。加载慢不会让它失败，Stop 会立即结束等待。只有一直不回答的 melo_serve 才会到达 45 分钟的上限（高于它自身最坏的情况）：切换以 `failed` 结束，被挂起的一轮继续运行；正在等待的任务以 `failed` 结束，这一轮以 `provider_error` 结束，并带上 melo_serve 的回答。
- 网关在运行时，终端通过网关切换，使用主聊天已在用的令牌，因此网关的聊天也会等待；没有网关时，终端直接请求 melo_serve。开发者模式会加上每个模型在磁盘上的大小。

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

在设置里，从 **Hosted services** 下选择一个服务并粘贴一次密钥，见[云端模型](hosted-models.md)。如果想把密钥放在环境变量里：

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

对于其他提供方，使用 `[anthropic]`、`[gemini]` 或 `[nvidia]` 表。设置会建议使用变量 `ANTHROPIC_API_KEY`、`GEMINI_API_KEY` 和 `NVIDIA_API_KEY`。变量优先于保存在钥匙串里的密钥。

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
