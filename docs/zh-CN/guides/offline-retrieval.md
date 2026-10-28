<a id="guide-offline-retrieval"></a>

# 指南：离线检索

<a id="run-a-first-evaluation"></a>

## 运行首次评测

`bm25` 不需要服务器。写入包含三篇文档的语料库和两个带标注的查询：

```bash
mkdir retrieval-demo
cd retrieval-demo
cat > corpus.jsonl <<'EOF'
{"doc_id": "demo:n1", "kind": "note", "text": "plan: Let me read list_ops.py to find the stubs.\nfound: append(list1, list2) must return a new list; foldr is still a stub", "meta": {"task_id": "demo", "step": 1}}
{"doc_id": "demo:d2", "kind": "draft", "text": "def append(list1, list2):\n    return list1 + list2", "meta": {"task_id": "demo", "step": 2}}
{"doc_id": "demo:t2.1", "kind": "thought", "text": "foldr folds from the right, so the accumulator starts with the last item.", "meta": {"task_id": "demo", "step": 2}}
EOF
cat > queries.jsonl <<'EOF'
{"qid": "q1", "query": "how append joins list1 and list2", "scope": {"task_id": "demo"}, "relevant": ["d2"]}
{"qid": "q2", "query": "foldr accumulator order", "relevant": ["demo:t2.1"]}
EOF
```

语料库包含一个任务 `demo` 的一条笔记、一份代码草稿和一个思维片段。每个查询在 `relevant` 中列出回答它的文档。由于 `q1` 的 `scope` 指定了任务，它可以用 `d2` 代表 `demo:d2`。

在这些数据上评测 `bm25`：

```bash
melo-retrieve --retriever bm25 --corpus corpus.jsonl --queries queries.jsonl
```

输出示例（数值会有所不同）：

```json
{"type": "query", "qid": "q1", "source": "", "candidates": 3, "relevant": ["demo:d2"], "hits": [{"doc_id": "demo:d2", "score": 2.1301, "rank": 1}, {"doc_id": "demo:n1", "score": 1.0976, "rank": 2}], "recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "rr": 1.0, "ms_cold": 0, "ms_warm": [0, 0, 0], "ms_interleaved": null, "fallbacks": {"timeout": 0, "parse": 0, "provider_error": 0}, "usage": {}}
{"type": "query", "qid": "q2", "source": "", "candidates": 3, "relevant": ["demo:t2.1"], "hits": [{"doc_id": "demo:t2.1", "score": 1.5471, "rank": 1}], "recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "rr": 1.0, "ms_cold": 0, "ms_warm": [0, 0, 0], "ms_interleaved": null, "fallbacks": {"timeout": 0, "parse": 0, "provider_error": 0}, "usage": {}}
{"type": "summary", "retriever": "bm25", "model": null, "dim": null, "n": 2, "calls": 8, "recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "mrr": 1.0, "ms_cold_p50": 0, "ms_warm_p50": 0, "ms_warm_p95": 0, "ms_interleaved_p50": null, "ms_interleaved_p95": null, "prompt_tokens_mean": null, "cached_tokens_mean": null, "completion_tokens_mean": null, "fallbacks": {"timeout": 0, "parse": 0, "provider_error": 0}}
```

- 标准输出（stdout）为每个查询输出一行，最后输出一行汇总。
- `q1` 在第 1 位找到了相关文档 `demo:d2`，因此其 `recall@1` 和 `rr`（倒数排名）均为 1.0。笔记 `demo:n1` 也匹配，在第 2 位。
- 每个查询运行了一次冷调用和三次热调用，总共 8 次 `calls`。耗时以毫秒为单位，记录在 `ms_` 开头的键中。
- 汇总中的 `recall@<k>`（召回率）和 `mrr`（平均倒数排名）是各查询的平均值。

[输出](../../reference/retrieval.md#output)（英文）介绍了每个键。

<a id="compare-retrievers-on-a-model-server"></a>

## 在模型服务器上比较检索器

`llm` 和 `embedding` 需要兼容 OpenAI 的服务器。对同一组文件运行它们，并将其汇总行与 `bm25` 的汇总行比较。这些示例使用 vLLM 进行对话，使用 Ollama 生成嵌入；请使用你自己的服务器地址和模型。

```bash
melo-retrieve --retriever llm --corpus corpus.jsonl --queries queries.jsonl \
  --base-url http://127.0.0.1:8000/v1 --chat-model Qwen/Qwen3-8B \
  --retrieval-extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
melo-retrieve --retriever embedding --corpus corpus.jsonl --queries queries.jsonl \
  --embedding-base-url http://127.0.0.1:11434/v1 --embedding-model nomic-embed-text
```

- `--retrieval-extra-body` 向每个检索请求添加字段。在接受 `chat_template_kwargs` 的服务器上，例如 vLLM，此设置会关闭 Qwen3 等模型的思考功能，让模型不先进行推理就直接回答。
- 使用 Ollama 时，先拉取嵌入模型：`ollama pull nomic-embed-text`。
- 从 `recall@<k>` 和 `mrr` 了解准确率，从 `ms_` 数值了解速度。`fallbacks` 统计改由 `bm25` 回答的调用次数；参见[警告](../../reference/retrieval.md#warnings)（英文）。

<a id="measure-latency-between-other-requests"></a>

### 测量穿插其他请求时的延迟

缓存提示词的服务器回答重复请求更快，因此连续调用看起来可能比穿插其他工作时更快。每个查询先运行一次冷调用，然后连续运行 `--repeat` 次热调用（默认：`3`）。使用 `--interleave` 时，每个查询随后还会运行 `--repeat` 次穿插调用，并在每次调用前将文件中的请求发送到 `--base-url`，模拟智能体在中间发送自己的请求：

```bash
cat > request.json <<'EOF'
{"messages": [{"role": "system", "content": "You are a coding agent. Use the tools to edit files."}, {"role": "user", "content": "Implement foldr in list_ops.py."}], "max_tokens": 1}
EOF
melo-retrieve --retriever llm --corpus corpus.jsonl --queries queries.jsonl \
  --base-url http://127.0.0.1:8000/v1 --chat-model Qwen/Qwen3-8B \
  --retrieval-extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --interleave request.json
```

- 穿插请求本身的耗时不计入结果。穿插调用的耗时记录在 `ms_interleaved`、`ms_interleaved_p50` 和 `ms_interleaved_p95` 中。
- 要得到接近智能体实际情况的数值，请让请求类似它的规划请求：使用相同的系统提示词和工具、相近的长度，并且不设置 `max_tokens`（参见[穿插请求](../../reference/retrieval.md#interleave-request)（英文））。
- 冷调用在新的检索器上运行，客户端没有任何缓存：一次冷 `embedding` 调用会对每个候选文档生成嵌入。服务器仍可能缓存之前查询发送的内容。

<a id="export-a-run"></a>

## 导出一次运行

`--export-run` 将一次运行的模型调用转为语料库行并输出到 stdout，可直接用于 `--corpus`。以下是一个简短任务的原始调用：

```bash
cat > calls.jsonl <<'EOF'
{"task_id": "demo", "seq": 0, "step": 1, "thinking": "Let me read list_ops.py to find the stubs.", "tool_calls": [{"tool_name": "file_read", "arguments": {"path": "list_ops.py"}, "id": "call_1"}], "tool_results": [{"output": "def append(list1, list2):\n    pass\n"}], "timestamp": 1761200000.0}
{"task_id": "demo", "seq": 1, "step": 2, "thinking": "append must return a new list, so concatenation is enough. Now I'll write it to list_ops.py:\n```python\ndef append(list1, list2):\n    return list1 + list2\n```", "tool_calls": [{"tool_name": "file_write", "arguments": {"path": "list_ops.py", "content": "def append(list1, list2):\n    return list1 + list2\n"}, "id": "call_2"}], "tool_results": [{"output": "Wrote 51 characters to list_ops.py"}], "timestamp": 1761200010.0}
{"task_id": "demo", "seq": 2, "kind": "reflect", "step": 2, "content": "{\"recommended_action\": \"done\", \"summary\": \"append returns a new list, so the task is done.\"}", "timestamp": 1761200020.0}
EOF
melo-retrieve --export-run calls.jsonl > run-corpus.jsonl
```

stderr 输出：

```text
melo-retrieve: exported 6 documents (2 notes, 1 reflections, 1 drafts, 2 thought chunks) from 1 tasks
```

`run-corpus.jsonl` 按顺序包含 `demo:n0`、`demo:t0.1`、`demo:n1`、`demo:d1`、`demo:t1.1` 和 `demo:r2`。第二次调用的笔记如下：

```json
{"doc_id": "demo:n1", "kind": "note", "title": "Now I'll write it to list_ops.py: → write list_ops.py", "text": "plan: Now I'll write it to list_ops.py: → write list_ops.py\nfound: append must return a new list, so concatenation is enough.\nobserved: wrote list_ops.py: 51 chars\noutcome: ok", "meta": {"task_id": "demo", "step": 2, "attempt": 1, "seq": 1, "created_at": 1761200010.0, "source": "heuristic", "note_kind": "step", "remember": true, "remember_by": "heuristic"}}
```

<a id="documents-of-a-run"></a>

### 一次运行的文档

任务按其首次调用的顺序排列，同一任务的调用按 `seq` 排列。每次调用产生以下文档：

| 调用 `kind` | 文档 |
|---|---|
| `plan` | 一条笔记 `<task_id>:n<seq>`；当推理或回复包含代码块时，一份草稿 `<task_id>:d<seq>`；思维片段 `<task_id>:t<seq>.<i>`，其中 i 从 1 开始。 |
| `reflect` | 当回复包含反思时，产生一条反思 `<task_id>:r<seq>`（即智能体读取的 `recommended_action` 为 `continue`、`stop`、`decompose` 或 `done` 的 JSON 对象），以及思维片段。不含反思的回复只产生思维片段。 |
| `retrieve`、`estimate` | 无。 |

- **笔记**在 `plan:`、`found:`、`decided:`、`hypotheses:` 和 `open:` 行中记录回复计划、发现、决定、怀疑和尚未解决的内容，在 `observed:` 和 `outcome:` 中记录其工具调用的返回结果。如果回复文本包含 `<note>` 块，笔记从该块中提取这些行（`meta.source` 为 `model`）。否则，根据中英文标记词从推理和回复中选取句子（`heuristic`）。
- **反思**根据智能体读取的反思自身字段构建，而不是根据回复文本：`plan:` 是建议的操作和摘要，`found:` 是有用证据，`decided:` 是接下来有希望的方向（`next: …`），`hypotheses:` 是已被否定的假设（`refuted: …`）和应避免的内容（`avoid: …`）。`meta.source` 为 `model`。
- **草稿**包含推理和回复中的代码块。`meta.target` 是第一个代码块之前最后提到的文件，否则是任务最近读取或写入的文件。当对该文件的一次写入包含草稿至少 60% 的行时，`meta.status` 为 `written`；后续草稿具有相同目标时为 `superseded`；其他情况下为 `pending`。
- **思维片段**包含去掉代码块后的推理，按段落和句子切分，每片最多 1,500 个字符。
- 每段文本和每个标题都会处理成可安全作为纯文本展示给模型的形式：`<|im_start|>` 等角色 token 和 `<tool_call>` 等标签会被转义，`<think>` 标签会被移除。处理后没有文本的文档会被跳过。
- `meta` 包含 `task_id`、`step`、`attempt`、`seq` 和 `created_at`（调用的 `timestamp`）。其他键取决于类型：
  - 笔记和反思：`source`（`model` 或 `heuristic`；反思始终为 `model`）、`note_kind`（根据调用结果为 `step`、`answer` 或 `empty`，或者为 `reflection`），以及 `remember` 和 `remember_by`。`remember` 表示笔记看起来是否值得保留：若块中有 `remember:` 行，则使用该行的值（`model`）；否则，当笔记有 `found:` 或 `decided:` 行时为 `true`（`heuristic`）。反思的这两个值为 `false` 和 `""`；
  - 草稿：`target`、`status`、`lines`（非空行数）、`blocks` 和 `cut`；为将草稿限制在 8,000 个字符以内而省略代码时，该值为 `true`；
  - 思维片段：`chunk`，即片段编号。

ID 遵循 `seq`，因此你可以在导出前根据自己保存的调用记录编写带标注的查询。

<a id="export-a-history-database"></a>

### 导出历史数据库

`--export-run` 也可读取一次运行的 `history.db`：提供包含该文件的记忆目录（运行时的 `--memory-dir` 或 `LOCALMELO_MEMORY_DIR`），或直接提供文件本身。[在多次运行之间保留记忆](memory.md#keep-memory-between-runs)时，`melo` 默认将其保存在 `~/.cache/localmelo/memory`。文件以 SQLite 文件头开头时按数据库读取，否则按原始调用读取。

导出会从数据库的 `llm_calls` 表读取模型调用；`melo` 在保留记忆时会写入该表（参见[模型调用](../../reference/memory.md#model-calls)（英文））：

```bash
LOCALMELO_PERSIST_MEMORY=1 melo "List the files"
melo-retrieve --export-run ~/.cache/localmelo/memory > run-corpus.jsonl
```

```text
melo-retrieve: exported 4 documents (2 notes, 0 reflections, 0 drafts, 2 thought chunks) from 1 tasks
```

数字取决于具体运行。此例中，模型先列出目录然后回答，两次回复都有推理，因此两次调用各产生一条笔记和一个思维片段。

- 从未被支持模型调用记录的版本打开过的 `history.db` 没有该表，导出会以退出码 2 停止：

  ```text
  melo-retrieve: error: <path> has no llm_calls table: it was written before localmelo recorded model calls. Export the calls as raw-call JSONL instead (see docs/retrieval.md).
  ```

  `<path>` 是数据库路径。要导出这样的运行，请根据 localmelo 之外的记录（例如代理记录的请求和回复）将其模型调用写为[原始调用](../../reference/retrieval.md#raw-calls)（英文）。
- 旧版本的 `history.db` 被当前版本打开后会获得一张空表：之前的任务不会导出任何内容，结束行只统计有模型调用的任务。
- 早于 `LOCALMELO_THINKING_RETENTION_DAYS` 的行（默认 30 天）没有 `thinking`，因此不会产生思维片段。
- `content` 和 `thinking` 在存储时经过脱敏，工具调用的参数则没有。如果草稿的代码中包含密钥，脱敏后可能不再匹配其 `file_write`；如果反思的 JSON 在代码关键词附近包含未加引号的 4 至 8 位数字，脱敏后可能不再能够解析。

包含该表的数据库按以下方式读取：

- 读取的列为 `task_id`、`seq`、`kind`、`step`、`attempt`、`outcome`、`content`、`thinking`、`tool_calls_json`、`step_ids_json` 和 `timestamp`，遵循原始调用的规则。`step` 或 `attempt` 为 `NULL` 时分别读作 0 或 1。缺少列会导致导出停止：`melo-retrieve: error: <path>: llm_calls has no column <names>`。
- 当 `thinking` 非空，且 `content` 以 `<think>`、换行、推理、换行、`</think>` 和空行开头时（即 Ollama 提供方添加的副本），读取内容时会去掉该副本，避免重复读取推理。
- 工具结果来自 `step_ids_json` 列出的 `steps` 行，按 `id` 与调用匹配。列出的行缺失或与任何调用都不匹配时，会向 stderr 输出一条警告：`melo-retrieve: warning: <n> step ids in llm_calls.step_ids_json had no matching steps row; their results are left out`。
- 任务按最早的 `timestamp` 排列。
- 导出绝不会写入数据库，也不会在数据库旁创建文件。如果存在 `history.db-wal` 文件（例如运行正在写入时），导出会通过它读取数据，SQLite 可能会创建 `history.db-shm`。请导出已经结束的运行。
