<a id="set-up-an-embedding-model"></a>

# 配置嵌入模型

嵌入模型是可选的。localmelo 不会下载或启动模型：你需要运行模型服务器，localmelo 会调用该服务器已经提供的模型。

<a id="what-it-buys"></a>

## 有什么作用

没有嵌入后端时，工作记忆和历史记录仍然可用。记忆开启时，显式推理模式可以接纳纯文本长期记忆条目，并通过 BM25 找到它们；开启持久化后，这些条目可以跨进程保留。显式模式的召回要求触发点已启用，并且包含 `long` 类型。在 `legacy` 模式中，长期记忆的写入和召回需要嵌入后端。见[记忆](../concepts/memory.md#what-localmelo-remembers)。

嵌入后端为带有向量的条目增加向量相似度搜索；纯文本条目仍通过 BM25 查找。在 `compress` 和 `compress+replay` 模式下，`auto` 检索器还会使用嵌入提供方对任务内召回排序；没有嵌入提供方时，`auto` 在 start 和 attempt 触发点使用 LLM，在 step 和 tool 触发点使用 BM25。[召回参考](../concepts/memory.md#recall-in-the-loop)说明了触发点和其他模式。

一项离线比较在 97 个金标准查询上以封顶 grade-2 recall@3 衡量三种检索器：`bm25` 为 0.284、`llm` 为 0.576、`embedding` 为 0.636。`embedding` 与 `llm` 相差 +0.06，95% 置信区间为 [−0.055, +0.179]，因此在该样本规模下两者无法区分；只有相对于 `bm25` 的提升可以区分。所以优先选择嵌入模型是产品选择——点估计最佳、不需要额外的对话调用，而且模型可以小巧并在本地运行——而不是已证明的优势。

localmelo 本身不会下载、启动或提供模型。以下命令用于你自己的服务器。

<a id="check-the-server"></a>

## 检查服务器

先查询服务器提供哪些模型，再向其中一个模型请求嵌入：

```bash
curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3-Embedding-0.6B", "input": "ping"}'
```

第二个响应必须包含非空的 `data[0].embedding` 数组。模型列表只能提供候选项：包括 oMLX 和 Ollama 在内的 OpenAI 兼容服务器不会说明哪些 ID 是嵌入模型，因此需要用嵌入请求检查。localmelo 使用相同请求进行探测，绝不会保存对该探测请求没有响应的嵌入模型。

## Ollama

推荐模型为 `qwen3-embedding:0.6b`；`nomic-embed-text` 是更小的替代模型，但仅支持英文。

```bash
ollama pull qwen3-embedding:0.6b
curl -sS http://127.0.0.1:11434/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-embedding:0.6b", "input": "ping"}'
melo setup    # Embedding backend: 2 (Ollama); URL http://localhost:11434; model qwen3-embedding:0.6b
```

在向导中选择 `2`，即 `Ollama (local endpoint)`，接受 `http://localhost:11434`，并输入模型名 `qwen3-embedding:0.6b`。向导写入：

```toml
chat_backend = "ollama"
embedding_backend = "ollama"

[ollama]
chat_url = "http://127.0.0.1:11434"
chat_model = "qwen3:8b"
embedding_url = "http://127.0.0.1:11434"
embedding_model = "qwen3-embedding:0.6b"
```

Ollama 的 `embedding_url` 不带 `/v1`，与它的 `chat_url` 一样；后端会为嵌入请求添加 `/v1`（`POST /v1/embeddings`）。对话使用 Ollama 自己的 `/api/chat` API。如需较小的模型，拉取 `nomic-embed-text` 并改用该 ID。

## oMLX

oMLX 在同一个端点提供嵌入和对话服务。将模型放入其模型目录，保持对话服务器不变。请在 Qwen3-Embedding 模型的文件夹名中保留 `embed`，如 `Qwen3-Embedding-0.6B`：oMLX 通过名称识别这个模型。然后检查 oMLX 列出的 ID：

```bash
curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3-Embedding-0.6B", "input": "ping"}'
```

`GET /v1/models` 返回 ID 但不返回类型，因此无法告诉你哪些 ID 是嵌入模型：上面的嵌入请求才是检查。模型 ID 由 oMLX 决定，通常是文件夹名称。然后把嵌入字段添加到保存 oMLX 对话设置的 `[vllm]` 表中：

```toml
chat_backend = "omlx"
embedding_backend = "omlx"

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen3-8B-4bit"
embedding_url = "http://127.0.0.1:8000/v1"
embedding_model = "Qwen3-Embedding-0.6B"
```

其余配置见 [oMLX](model-servers.md#omlx)。

## vLLM

vLLM 将嵌入模型作为独立端点提供。这一步没有在这台 Mac 上运行（本机没有 vLLM）；请在 `vllm serve --help` 中检查你所用版本的选项。

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B --task embed --port 8001
curl -sS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-Embedding-0.6B", "input": "ping"}'
```

然后设置 `embedding_backend = "vllm"`，并填入该 URL 和服务提供的 ID：

```toml
embedding_backend = "vllm"

[vllm]
embedding_url = "http://127.0.0.1:8001/v1"
embedding_model = "Qwen/Qwen3-Embedding-0.6B"
```

<a id="sglang-and-mlc"></a>

## SGLang 和 MLC

同样的方式适用于 SGLang；此处未运行：

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-Embedding-0.6B --is-embedding --port 30001
curl -sS http://127.0.0.1:30001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-Embedding-0.6B", "input": "ping"}'
melo setup    # Embedding backend: 4 (SGLang); URL http://localhost:30001/v1
```

MLC 需要编译后的嵌入产物；对话产物不能提供嵌入。以下 `<artifact>` 表示编译后的 Qwen3-Embedding 目录：

```bash
mlc_llm serve <artifact> --port 8080
curl -sS http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "<the served id>", "input": "ping"}'
melo setup    # Embedding backend: 1 (MLC); URL http://localhost:8080/v1; model <the served id>
```

上面的 SGLang 和 MLC 命令都没有在这台 Mac 上运行。归档的后端冒烟测试在 MLC 上使用了编译后的 `Qwen3-Embedding-0.6B-q0f16-MLC` 产物，服务提供的 ID 为 `qwen3-embedding`。

<a id="a-second-server"></a>

## 使用第二个服务器

对话后端和嵌入后端相互独立。云端对话模型可以使用本地嵌入服务器，本地对话服务器也可以使用另一个本地服务器提供嵌入。在向导中，通用的 `1`–`4` 选项会询问嵌入端点的 URL 和模型，因此应选择嵌入服务器的类型，而不是对话服务器的类型：

```toml
chat_backend = "openai"
embedding_backend = "ollama"

[openai]
api_key_env = "OPENAI_API_KEY"
chat_model = "gpt-4o"

[ollama]
embedding_url = "http://127.0.0.1:11434"
embedding_model = "nomic-embed-text"
```

见[混用后端](model-servers.md#mix-backends)。

<a id="what-changes"></a>

## 会有哪些变化

- `/status` 和开发者模式横幅的记忆行会显示后端和模型，例如 `long-term memory · OpenAI-compatible server · Qwen3-Embedding-0.6B`（配置文件写 `omlx` 时为 `oMLX`）。没有嵌入后端时，`/status` 显示 `no long-term memory (no embedding backend)`，横幅显示 `no long-term memory · melo setup adds one`（在 `--url` 会话中则显示 `--embedding-base-url adds one`；首次运行刚说明过怎么加，所以它的横幅不再重复）。这一行描述的是嵌入配置，而非显式模式是否可以使用纯文本长期记忆。
- 新建的长期记忆存储初始为空。运行结束和步骤完成时，符合条件的条目可能被接纳；已有的纯文本条目不会重新生成嵌入。
- 更换模型会改变向量空间。已存向量的维度与新模型不同时，该向量会被跳过，因此旧条目不再参与排序，而不会破坏召回。要回收空间，请停止 localmelo 并删除 `~/.cache/localmelo/memory/long_term.db*`；见[重置记忆](memory.md#reset-memory)。
- 网页工作区也有连接设置。见[网页工作区](web-console.md#web-workspace)。

<a id="check-it-works"></a>

## 检查是否生效

- 在终端启动 `melo` 并输入 `/status`：其记忆行会写出嵌入模型。
- `/status` 可用后，其 `memory` 行来自相同的文本。
- 运行一个使用工具的任务并结束，然后运行一个相关任务。持久化可以让已接纳的条目在后续运行中可用；召回还取决于模式、启用的触发点和相关匹配。见[跨运行保留记忆](memory.md#keep-memory-between-runs)。
- 要在自己的语料上衡量模型，使用 `melo-retrieve --retriever embedding …`；见[离线检索](../../reference/retrieval.md)（英文）。

<a id="troubleshooting"></a>

## 故障排查

- 嵌入请求返回 **HTTP 404**：服务器不提供该模型 ID。列出模型并使用其中的 ID。
- 嵌入请求返回 **HTTP 400**：该 ID 是对话模型，而不是嵌入模型。请选择嵌入模型，而不是对话模型。
- **Connection refused**：服务器未在该 URL 上运行，或 URL 有误。
- **更换模型后召回为空**：旧向量的维度不同，因而被跳过。恢复旧模型，或删除 `long_term.db*` 并让它重新填充。

更多错误见[故障排查](../../reference/troubleshooting.md#an-embedding-model-cannot-be-used)（英文）。
