<a id="offline-retrieval"></a>

# 离线检索

`melo-retrieve` 离线比较记忆检索器：它对一组文档运行已标注的查询，报告每个检索器找到正确文档的频率及其速度。它也能把一次运行的模型调用转换为这样的语料库。

localmelo 有三个检索器。每个检索器都会针对一个查询为文档排序，这些文档包括关于模型推理的笔记、代码草稿以及思考片段：

| 检索器 | 排序方式 |
|---|---|
| `bm25` | Okapi BM25 文本排序，不使用模型。 |
| `llm` | 对话模型从文档目录中挑选文档。 |
| `embedding` | 嵌入向量的余弦相似度。 |

`melo-retrieve` 离线评估这些检索器；压缩模式则在运行循环中使用同样的 `bm25`、`llm` 和 `embedding` 检索器。`melo-retrieve` 随 localmelo 一起安装，与 `melo` 并列，运行方式相同：例如在克隆目录中运行 `uv run melo-retrieve --help`。

<a id="options"></a>

## 选项

```text
melo-retrieve --retriever bm25|llm|embedding --corpus FILE --queries FILE [options]
melo-retrieve --export-run DIR|FILE
```

| 选项 | 默认值 | 说明 |
|---|---|---|
| `-h`, `--help` | | 显示帮助并退出。 |
| `--retriever bm25\|llm\|embedding` | 未设置 | 要评估的检索器。 |
| `--corpus FILE` | 未设置 | 文档，每行一个；见[语料库](#corpus)。 |
| `--queries FILE` | 未设置 | 已标注的查询，每行一个；见[已标注的查询](#labelled-queries)。 |
| `--k K[,K...]` | `1,3,5` | `recall@k` 所用的 k 值。每次搜索都请求其中最大的那个。 |
| `--repeat N` | `3` | 每个查询的热调用次数，以及使用 `--interleave` 时穿插调用的次数。 |
| `--retrieval-timeout SECONDS` | `30` | 一次 `llm` 检索请求最多可用的时间；超时则改由 `bm25` 回答。 |
| `--interleave REQUEST.json` | 未设置 | 在每次穿插调用之前发送的对话请求；见[穿插请求](#interleave-request)。 |
| `--base-url URL` | 未设置 | 兼容 OpenAI 的对话服务器，用于 `llm` 和 `--interleave`。 |
| `--chat-model NAME` | `default` | 向 `--base-url` 请求的模型。 |
| `--retrieval-extra-body JSON` | 未设置 | 要添加到每个 `llm` 检索请求中的字段，为一个 JSON 对象。其中的 `temperature`（如有）会替换默认的 0。 |
| `--embedding-base-url URL` | 未设置 | 兼容 OpenAI 的嵌入服务器，用于 `embedding`。 |
| `--embedding-model NAME` | `default` | 向 `--embedding-base-url` 请求的模型。 |
| `--export-run DIR\|FILE` | 未设置 | 把一次运行的模型调用打印为语料库行；见[导出运行](../guides/offline-retrieval.md#export-a-run)。 |
| `--export-store DIR\|FILE` | 未设置 | 打印类型化 `long_term.db` 中的有效条目；见[类型化长期存储](#the-typed-long-term-store)。 |
| `--export-decisions DIR\|FILE` | 未设置 | 按 `seq` 顺序打印其决策日志。 |
| `--route CANDIDATES.jsonl` | 未设置 | 通过边界和策略为合成候选项打分；需要 `--memory-dir`。 |
| `--memory-dir DIR` | 未设置 | `--route` 写入的临时目录；绝不会是用户的记忆目录。 |

- 运行评估需要 `--retriever`、`--corpus` 和 `--queries`。`--export-run` 不接受其他选项。
- `--retriever llm` 需要 `--base-url`，`--retriever embedding` 需要 `--embedding-base-url`，而 `--interleave` 需要 `--base-url`。
- `--retrieval-timeout` 和 `--retrieval-extra-body` 只适用于 `--retriever llm`；`--embedding-base-url` 和 `--embedding-model` 只适用于 `--retriever embedding`；`--base-url` 和 `--chat-model` 只适用于 `--retriever llm` 或 `--interleave`。
- `--k`：用逗号分隔的正整数，例如 `1,3,10`。重复的值只计一次，`recall@<k>` 键按 k 递增的顺序排列。
- `--repeat 0` 只运行每个查询的冷调用。
- `--retrieval-timeout`：大于 0 的秒数，且只适用于 `--retriever llm`。检索请求在该时间内仍未返回，就由 `bm25` 回答。运行的 HTTP 超时为 120 秒或 `--retrieval-timeout`，取较长者，并适用于检索请求和穿插请求；嵌入请求最多等待 120 秒。
- `--retrieval-extra-body` 必须是 JSON 对象，且不能设置 `model`、`messages`、`tools`、`tool_choice` 或 `stream`。它不适用于穿插请求。
- URL 按原样使用，因此要包含 `/v1`，例如 `http://127.0.0.1:8000/v1`。
- `melo-retrieve` 不发送 API 密钥，也不读取配置文件。要针对云端 API 评估，请使用 [Python](#from-python)，并提供带有你自己密钥的提供方。

不合法的值、缺少的选项，或互不兼容的选项，都是用法错误：`melo-retrieve` 会在 stderr 上打印其用法摘要和一行错误信息，并以退出码 2 结束。

```text
melo-retrieve: error: --retriever llm needs --base-url
```

<a id="input-formats"></a>

## 输入格式

语料库、已标注的查询和原始调用都是 JSON Lines 文件：每行一个 JSON 对象，使用 UTF-8 编码。空行会被跳过，未知的键会被忽略。其他任何问题都会让运行以退出码 2 停止，并输出一行，指明文件和行号：

```text
melo-retrieve: error: queries.jsonl:1: relevant id 'demo:d9' is not in the corpus
```

<a id="corpus"></a>

### 语料库

每行一个文档，是 `--corpus` 的输入，也是 `--export-run` 的输出：

```json
{"doc_id": "demo:n1", "kind": "note", "title": "Find the stubs", "text": "plan: Let me read list_ops.py to find the stubs.", "meta": {"task_id": "demo", "step": 1}}
```

| 键 | 必需 | 值 |
|---|---|---|
| `doc_id` | 是 | 文档的 id：在文件中唯一，且不含空白字符。 |
| `kind` | 是 | `note`、`draft`、`thought`、`reflection`、`digest` 或 `long`。 |
| `text` | 是 | 要搜索的文本，不能为空。 |
| `title` | 否 | 一行标题。`llm` 的目录会显示它，否则显示 `text` 的第一行。 |
| `meta` | 否 | 一个对象。其中的 `task_id`（字符串）和 `step`（整数，0 或更大）决定哪些查询能看到该文档。其他键原样保留。 |

导出会写入笔记、草稿、思考片段和反思；见[导出运行](../guides/offline-retrieval.md#export-a-run)。`digest`（多条笔记的摘要）和 `long`（一条长期记忆条目）是为你自己构建的语料库预留的。

<a id="labelled-queries"></a>

### 已标注的查询

每行一个查询，是 `--queries` 的输入：

```json
{"qid": "q3", "query": "how append joins list1 and list2", "scope": {"task_id": "demo", "before_step": 3, "exclude_ids": ["n1"]}, "relevant": ["d2"], "grades": {"d2": 2}, "source": "gold"}
```

| 键 | 必需 | 值 |
|---|---|---|
| `qid` | 是 | 查询的 id，在文件中唯一。 |
| `query` | 是 | 要搜索的文本，不能为空白。 |
| `relevant` | 是 | 能回答该查询的文档的 id：一个非空列表。 |
| `scope` | 否 | 该查询搜索哪些文档；见下文。 |
| `grades` | 否 | 分级相关性：为 `relevant` 中 id 的整数，1 或更大。它会被检查，但不参与评分。 |
| `source` | 否 | 一个标签，例如 `gold`，该查询的输出行会重复它。 |

`scope` 有三个键：

- `task_id`：只搜索其 `meta.task_id` 为该值的文档。没有它时，查询会搜索每个文档。
- `before_step`：只搜索其 `meta.step`（不存在时为 0）更小的文档，即该步骤之前已存在的文档。它是一个 1 或更大的整数；它需要 `task_id`。
- `exclude_ids`：检索器不得返回的文档。它们不计为候选项，但 `llm` 会把它们保留在目录中。

在 `relevant`、`grades` 和 `exclude_ids` 中，没有 `:` 的 id 是 `<task_id>:<id>` 的简写，其中的 `task_id` 取自 scope；没有 scope 时则是错误。每个相关 id 都必须在语料库中，并且位于该查询的 scope 之内，且不能出现在 `exclude_ids` 中。

<a id="raw-calls"></a>

### 原始调用

运行中的每个模型调用占一行，是 `--export-run` 的输入：

```json
{"task_id": "demo", "seq": 0, "kind": "plan", "step": 1, "attempt": 1, "content": "", "thinking": "Let me read list_ops.py to find the stubs.", "tool_calls": [{"tool_name": "file_read", "arguments": {"path": "list_ops.py"}, "id": "call_1"}], "tool_results": [{"tool_name": "file_read", "output": "def append(list1, list2):\n    pass\n", "error": "", "duration_ms": 3.1}], "outcome": "tools", "timestamp": 1761200000.0}
```

| 键 | 默认值 | 值 |
|---|---|---|
| `task_id` | 必需 | 任务的 id：不能为空，且不含空白字符。 |
| `seq` | 必需 | 调用在其任务中的编号：0 或更大的整数，每个任务内唯一。 |
| `kind` | `plan` | `plan`（规划调用）、`reflect`（反思调用），或 `retrieve` 和 `estimate`（其他调用，不产生文档）。 |
| `step` | `0` | 运行的步骤。 |
| `attempt` | `1` | 运行的尝试。 |
| `content` | `""` | 回复的文本。 |
| `thinking` | `""` | 回复的推理。 |
| `tool_calls` | `[]` | 回复中的工具调用，每个都有 `tool_name`、`arguments` 和 `id`。 |
| `tool_results` | `[]` | 它们的结果，顺序相同，每个都有 `tool_name`、`output`、`error` 和 `duration_ms`。 |
| `outcome` | 推导得出 | 规划调用为 `tools`、`answer` 或 `empty`，其他类型为 `aux`。 |
| `timestamp` | `0.0` | 调用发生的时间，以自纪元以来的秒数计。 |

- `step` 和 `attempt` 可以为 `null`，视为默认值。
- `arguments` 是一个对象，或是保存对象的字符串，与 OpenAI 请求体中的形式相同。其他任何字符串都会保存为 `{"_raw_arguments": "<the string>"}`。
- 结果的 `tool_name` 默认与其调用相同。结果从不多于调用：没有结果的调用就是没有运行。
- 若没有 `outcome`，规划调用的结果为：有工具调用时为 `tools`；否则 `content` 不为空白时为 `answer`；否则为 `empty`。

<a id="interleave-request"></a>

### 穿插请求

`--interleave` 的文件保存一个 JSON 对象：一个对话补全请求的请求体，其中 `messages` 列表不能为空。

```json
{"messages": [{"role": "system", "content": "You are a coding agent. Use the tools to edit files."}, {"role": "user", "content": "Implement foldr in list_ops.py."}], "max_tokens": 1}
```

- `melo-retrieve` 把 `model` 设为 `--chat-model`，把 `stream` 设为 `false`，并把其余内容原样发送到 `<base-url>/chat/completions`。请求体可以包含 `tools` 以及你的服务器接受的任何其他字段。
- `--retrieval-extra-body` 不适用于它。
- `max_tokens` 让回复保持简短，但在有 `tools` 时不要使用：截断工具调用的上限可能会让服务器请求失败（oMLX 对不完整的工具调用返回 HTTP 500），于是运行就会停止。有 `tools` 时，请不要设置 `max_tokens`；请求的时间本来就不计入统计。
- 如果请求失败，运行会以退出码 1 停止：

  ```text
  melo-retrieve: error: interleave request failed: POST /chat/completions failed: ConnectError: All connection attempts failed
  ```

<a id="output"></a>

## 输出

stdout 为每个查询输出一行 JSON，在该查询完成时打印，之后再输出一行汇总。警告和错误输出到 stderr。

<a id="query-lines"></a>

### 查询行

| 键 | 值 |
|---|---|
| `type` | `"query"` |
| `qid` | 查询的 `qid`。 |
| `source` | 查询的 `source`，或 `""`。 |
| `candidates` | 该查询搜索了多少文档：即其 scope 内的文档，不含 `exclude_ids`。 |
| `relevant` | 完整的相关 id。 |
| `hits` | 冷调用的命中，最佳者在前，每个包含 `doc_id`、`score`（保留 4 位小数）和 `rank`（从 1 开始）。 |
| `recall@<k>` | 每个 `--k` 值一个键：相关 id 在前 k 个命中中所占的比例。 |
| `rr` | 倒数排名：1 / 第一个相关命中的排名；若没有则为 0。 |
| `ms_cold` | 冷调用的耗时，以毫秒计。 |
| `ms_warm` | 热调用的耗时，或在 `--repeat 0` 时为 `null`。 |
| `ms_interleaved` | 穿插调用的耗时，或在没有 `--interleave` 或使用 `--repeat 0` 时为 `null`。 |
| `fallbacks` | 该查询的调用中有多少次被 `bm25` 回答而非原检索器回答，按原因计：`timeout`、`parse` 和 `provider_error`。 |
| `usage` | 服务器为该查询的调用报告的 token 计数，按键求和；对于 `bm25` 和 `embedding` 为 `{}`。 |

- `recall@<k>` 和 `rr` 是该查询所有调用（冷调用、热调用和穿插调用）的均值。它们会平滑 `llm` 的回答，因为它在各次调用之间可能有所不同。`hits` 只显示冷调用，因此只根据 `hits` 算出的召回率可能与此不同。
- 对于 `bm25`，`score` 是 BM25 分数；对于 `llm`，是 1 / 排名；对于 `embedding`，是余弦相似度。发生回退后，它是 BM25 分数。

<a id="summary-line"></a>

### 汇总行

| 键 | 值 |
|---|---|
| `type` | `"summary"` |
| `retriever` | 被评估的检索器，即使其调用发生了回退也是如此。 |
| `model` | `llm` 为 `--chat-model`，`embedding` 为 `--embedding-model`，`bm25` 为 `null`。 |
| `dim` | `embedding` 的嵌入向量长度，否则为 `null`。 |
| `n` | 查询的数量。 |
| `calls` | 所有查询的搜索次数。 |
| `recall@<k>` | 各查询 `recall@<k>` 的均值。 |
| `mrr` | 各查询 `rr` 的均值：即平均倒数排名。 |
| `ms_cold_p50` | 冷调用耗时的中位数。 |
| `ms_warm_p50`、`ms_warm_p95` | 热调用耗时的第 50 和第 95 百分位数。 |
| `ms_interleaved_p50`、`ms_interleaved_p95` | 穿插调用的相同指标。 |
| `prompt_tokens_mean`、`cached_tokens_mean`、`completion_tokens_mean` | 每次调用的平均 token 计数，只计报告了该值的调用，保留 1 位小数。 |
| `fallbacks` | 所有调用的回退，按原因计。 |

- 每个查询的 `calls` 计入 1 次冷调用、`--repeat` 次热调用，以及使用 `--interleave` 时的 `--repeat` 次穿插调用。
- 每个查询在均值中的权重相同。`recall@<k>`、`rr` 和 `mrr` 保留 4 位小数。
- 耗时以整毫秒计，在每次搜索前后测量，包括回退。百分位数采用最近秩法：n 个耗时的第 p 百分位数是第 ⌈p × n / 100⌉ 小的那个。
- `null` 表示不适用的数值：没有 `--interleave` 时的穿插数值，`--repeat 0` 时的热调用和穿插数值，以及 `bm25` 和 `embedding` 的 token 均值。
- `cached_tokens_mean` 在 `llm` 中，只有服务器报告 `prompt_tokens_details.cached_tokens` 时才会填写，否则为 `null`。

<a id="warnings"></a>

### 警告

当某个 `llm` 或 `embedding` 运行的每个调用都回退到 `bm25` 时，汇总行之后会在 stderr 上输出一条警告，列出原因及其计数：

```text
melo-retrieve: warning: every llm call fell back to bm25 (provider_error 8); check --base-url
```

此时的数值就是 `bm25` 的数值，退出码仍为 0。`provider_error` 通常表示无法连接服务器，`timeout` 表示其耗时超过了 `--retrieval-timeout`，`parse` 表示模型没有以 ID 列表作答。

<a id="exit-codes"></a>

## 退出码

| 退出码 | 含义 |
|---|---|
| 0 | 评估或导出已完成，包括发生回退的情况。 |
| 1 | 穿插请求失败。 |
| 2 | 用法错误或输入错误：无法读取的文件、违反格式的行，或无法导出的数据库。 |
| 130 | Ctrl+C 停止了运行。 |

用法错误会打印用法摘要和一行错误信息；其他错误只打印一行。不会输出回溯（traceback）。已经打印到 stdout 的行保持完整，因此被停止的运行会留下完整的查询行。

<a id="the-typed-long-term-store"></a>

## 类型化长期存储

有三种模式会读取或写入 R6-lite 的类型化 `long_term.db`。每种模式都与除模型相关选项之外的评估标志互斥，并且与 `--export-run` 也互斥；且都不联系模型：`--base-url`、`--chat-model`、`--embedding-base-url`、`--embedding-model`、`--retrieval-timeout` 和 `--retrieval-extra-body` 会被接受并忽略，因此基准脚本可以在所有地方传入相同的标志。[循环中的召回](../concepts/memory.md#recall-in-the-loop)在 `start` 时读取的正是同一个存储，`--export-store` 则把它展示给基准测试。`DIR` 可以是记忆目录（文件为 `long_term.db`），也可以是文件本身。

```bash
melo-retrieve --export-store DIR|FILE
melo-retrieve --export-decisions DIR|FILE
melo-retrieve --route CANDIDATES.jsonl --memory-dir DIR
```

<a id="--export-store"></a>

### `--export-store`：导出存储

每个**有效**条目输出一行 JSON，按 `(created_at, id)` 顺序排列：

```json
{"id": 1, "text": "I live in Berlin near the river", "kind": "profile",
 "source": "user", "status": "active", "admitted_by": "user_statement",
 "provenance": {"task_id": "t1", "source_ids": ["r1"]},
 "created_at": 0.0, "last_used_at": 0.0, "uses": 0, "supersedes": null,
 "links": [], "embed_model": "", "embed_dim": 0}
```

基准测试的 T4 读取器读取 `text` 和 `provenance.source_ids`。计数输出到 stderr。文件以只读方式打开：v0 文件以其旧列和旧的推导方式读取（反思依据 `metadata.type` 判断，其他则为 episodic/step_summary），并且**不会**被迁移；比当前版本更新的文件，则在 stderr 输出一行并以退出码 2 结束。没有 `long_term.db` 的目录（即运行没有写入任何内容的基准目录）是一个空存储：stdout 没有输出，stderr 输出 `exported 0 active entries from DIR/long_term.db`，退出码为 0，因此基准脚本无需先检查文件是否存在。不是目录的路径仍保留缺少文件的错误：输出一行，退出码 2。

<a id="--export-decisions"></a>

### `--export-decisions`：导出决策

每个决策行输出一行 JSON，按 `seq` 顺序排列：

```json
{"seq": 1, "candidate_id": "n1", "action": "NEW", "kind": "profile",
 "admitted_by": "user_statement", "entry_id": 1, "target_id": null,
 "reason": "new", "source_ids": ["r1"], "text_sha256": "…", "text": null,
 "timestamp": 0.0, "task_id": "t1"}
```

`source_ids` 是解析后的数组；除非运行使用了 `--keep-decision-text`，否则 `text` 为 `null`。v0 文件没有决策表，因此不输出任何内容；没有 `long_term.db` 的目录不输出任何内容并以 0 退出，stderr 输出 `exported 0 decisions from DIR/long_term.db`。

<a id="--route"></a>

### `--route`：路由

每个输入行是一个候选项：

```json
{"candidate_id": "n1", "source": "digest", "text": "task: fix it\nfiles: a.py",
 "source_ids": ["r1"], "signal": "rehearsal", "task_id": "t1",
 "session_id": "", "task_text": "plan the move", "task_key": "…"}
```

`source` 是 `reasoning_note`、`digest`、`reflection`、`user` 或 `sleep` 之一；`signal` 是 `rehearsal`、`user_request`、`user_statement`、`outcome` 或 `sleep` 之一；未知的值会以退出码 2 结束。可选的 `task_key` 字符串保存在候选项的来源信息中，因此经路由的摘要可以携带运行的召回所查找的同一个键；没有该键的经路由摘要则跨任务。候选项经过与运行相同的边界和策略，写入 `--memory-dir` 下一个仅含文本的存储。每个输入行输出一行，顺序与输入相同：

```json
{"candidate_id": "n1", "action": "NEW", "kind": "profile",
 "admitted_by": "user_statement", "entry_id": 1, "target_id": null,
 "reason": "new", "source_ids": ["r1"]}
```

边界或准入拒绝时输出 `"action": "REFUSED"` 及原因；没有非空 `files:` 或 `verified:` 行的摘要会在准入时被拒绝。`--memory-dir` 是必需的，并且该命令拒绝默认的 `~/.cache/localmelo/memory` 以及 `LOCALMELO_MEMORY_DIR` 的值（以解析后的路径为准）；目录以 `0700` 权限创建。`--route` 会追加到已有的临时存储，而不是清空它，因此基准测试每次运行都使用一个新目录。对该目录执行 `--export-store`，即可看到路由写入的内容。

<a id="from-python"></a>

## 从 Python 使用

> [!NOTE]
> 检索器的 Python 接口是实验性的，可能会在 v0.1.0 之前变化。

`localmelo.melo.memory.retrieval` 包含检索器、评估以及文件格式。下面的脚本会搜索并评估[第一次评估](../guides/offline-retrieval.md#run-a-first-evaluation)中的文件：

```python
import asyncio

from localmelo.melo.memory.retrieval import (
    build_retriever,
    evaluate,
    load_corpus,
    load_queries,
)


async def main() -> None:
    with open("corpus.jsonl", encoding="utf-8") as file:
        corpus = load_corpus(file)
    with open("queries.jsonl", encoding="utf-8") as file:
        queries = load_queries(file, corpus)

    # One search.
    retriever = build_retriever("bm25")
    result = await retriever.search("foldr accumulator order", corpus, k=3)
    for hit in result.hits:
        print(hit.rank, hit.doc_id, round(hit.score, 4))

    # The evaluation that melo-retrieve runs, with a new retriever per query.
    report = await evaluate(
        lambda: build_retriever("bm25"),
        corpus=corpus,
        queries=queries,
        ks=(1, 3),
    )
    print(report.recall, report.mrr)


asyncio.run(main())
```

输出：

```text
1 demo:t2.1 1.5471
{1: 1.0, 3: 1.0} 1.0
```

- `build_retriever(name, *, send=None, embedding=None, **options)` 返回一个带有空缓存的新检索器。`llm` 需要 `send`，`embedding` 需要 `embedding`（一个嵌入提供方）。未知的名称或选项、缺少 `send` 或提供方，或取值超出范围，都会抛出 `ValueError`。
- `await retriever.search(query, docs, *, k, exclude=frozenset())` 返回一个 `RetrievalResult`：`hits`（每个都是一个 `Hit`，包含 `doc_id`、`score` 和 `rank`）、`retriever`（回答的检索器）、`ms`、`fallback` 和 `usage`。`docs` 是一个序列，其元素为 `Doc(doc_id, kind, text, title="", meta={})`。
- `await evaluate(retriever, *, corpus, queries, ks=(1, 3, 5), repeats=3, interleave=None, model=None, on_query=None)` 返回一个 `EvalReport`。传入如上所述的构建检索器的函数，可让每个查询从冷启动开始；传入一个检索器，则可在所有查询之间共享它。`interleave` 是在每次穿插调用之前要等待的异步函数，`model` 是汇总行报告的名称。`on_query` 会在每个查询完成后立即收到它的 `QueryResult`。
- `report.queries` 保存每个查询的 `QueryResult`。`report.summary_line()` 和 `QueryResult.to_line(ks)` 以字典形式返回[输出行](#output)。
- `load_corpus`、`load_queries`、`load_raw_calls`、`docs_from_calls` 和 `doc_to_dict` 读写上述[格式](#input-formats)。错误的行会抛出 `CorpusError`，其 `line` 为从 1 开始计的行号。

`build_retriever` 接受以下选项：

| 选项 | 检索器 | 默认值 | 说明 |
|---|---|---|---|
| `floor_terms` | 全部 | `2` | `bm25` 命中必须包含的不同查询单元数；`0` 关闭此下限。 |
| `relative_cut` | 全部 | `0.2` | 丢弃得分低于最佳命中此比例的 `bm25` 命中；取值从 0 起，不含 1。 |
| `timeout_s` | `llm` | `30.0` | 超过此秒数后由 `bm25` 回答。 |
| `min_score` | `embedding` | `0.0` | 保留的最低余弦相似度。 |
| `batch_size` | `embedding` | `32` | 每次嵌入请求的文本数。 |

对于 `llm` 和 `embedding`，`floor_terms` 和 `relative_cut` 适用于回退时回答的 `bm25`。

有模型服务器时，`send_from_provider` 可以把对话提供方转换为 `llm` 所需的 `send`：

```python
import asyncio

from localmelo.melo.memory.retrieval import (
    build_retriever,
    load_corpus,
    send_from_provider,
)
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


async def main() -> None:
    with open("corpus.jsonl", encoding="utf-8") as file:
        corpus = load_corpus(file)

    llm = OpenAICompatLLM(
        "http://127.0.0.1:8000/v1",
        "Qwen/Qwen3-8B",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    try:
        retriever = build_retriever("llm", send=send_from_provider(llm), timeout_s=30.0)
        result = await retriever.search("foldr accumulator order", corpus, k=3)
        hits = [hit.doc_id for hit in result.hits]
        print(result.retriever, result.fallback or "-", hits)
    finally:
        await llm.close()


asyncio.run(main())
```

- `send_from_provider(llm)` 把每个检索请求作为一次不带工具的 `chat()` 调用发送，且不重试。
- 对于云端 API，请通过 `api_key=` 把密钥传给 `OpenAICompatLLM`。
- 对于 `embedding`，请传入一个提供方，例如从 `localmelo.support.providers.embedding.openai_compat` 导入的 `OpenAICompatEmbedding("http://127.0.0.1:11434/v1", "nomic-embed-text")`。
- 使用完毕后，请对每个提供方调用 `await provider.close()` 将其关闭。

<a id="see-also"></a>

## 另见

- [记忆](memory.md)：localmelo 记住什么，以及保存在哪里。
- [额外请求字段](configuration.md#extra-request-fields-extra_body)：`chat_template_kwargs` 及其他服务器选项。
- [命令行参考](../../reference/cli.md)（英文）：`melo` 命令。
