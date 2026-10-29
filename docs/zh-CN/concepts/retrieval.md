<a id="concepts-retrieval"></a>

# 概念：检索

本页说明每个检索器如何为记忆候选排序，以及一次运行如何召回这些候选。需要弄清某条记忆为何被召回时请阅读本页；选项和离线比较见[检索参考](../reference/retrieval.md)。

<a id="how-each-retriever-ranks"></a>

## 各检索器如何排序

**`bm25`** 用 Okapi BM25 为查询的候选文档评分，因此词语的稀有程度只取决于这些文档。随后：

- **最低要求。** 命中项必须包含查询中至少两个不同的单元。单元包括两个或更多字符的词，以及相邻的中文、日文或韩文字符对；`list_ops` 这样的标识符算一个单元。文件名和路径不计入，常见虚词以及笔记行中的 `plan`、`found`、`read`、`run` 等词也不计入。查询不足两个单元时没有命中。
- **类型。** 每个任务步骤的思考片段最多保留得分最高的一个命中；其他所有类型都排在所有思考片段之前。
- **相对截断。** 分别在思考片段和其他类型中，去掉得分低于该组最高分 0.2 倍的命中。

**`llm`** 发送一次不带工具的对话请求：先是固定的系统提示，再是文档目录、不得返回的别名、查询以及要返回的 ID 数量。`melo-retrieve` 以 `temperature` 0 发送，因此相同的模型和文件会给出相同选择；`--retrieval-extra-body` 可以设置其他值。

- 每行目录格式为 `[<alias>] <label> · step <n> · <title>`，例如 `[n1] note · step 1 · Find the stubs`。别名由类型字母（`n` note、`d` draft、`t` thought、`r` reflection、`g` digest、`m` long）和文档在该类型中的位置组成。
- 被排除的文档仍保留在目录中的位置，因此只有文档变化时目录才变化，服务器可在多次搜索之间复用提示缓存。超过 12,000 个字符时，每次丢弃最前面的若干整行，合计约 3,000 个字符。
- 模型以 `{"ids": ["n2", "d1"]}` 回答，最佳项在前。JSON 外面有文本或代码围栏也可接受，单独的列表也可以。命中得分为 1 / 排名。
- 超时、服务器报错或回复中没有 ID 列表时，改由 `bm25` 回答，结果注明原因：`timeout`、`provider_error` 或 `parse`。不重试。

**`embedding`** 对每篇文档嵌入一次，每次请求 32 条文本，并保留向量；每次搜索时对查询进行嵌入。文档按其向量与查询向量的余弦相似度排序，负相似度被排除。服务器报错、回复中的向量数量错误、值不是有限数字或向量长度不一致时，改由 `bm25` 回答，原因标为 `provider_error`。

<a id="recall-in-the-loop"></a>

## 循环中的召回

同样的三个检索器对一次运行自身的文档排序。`--reasoning compress`（以及 `compress+replay`）默认在 `start`、`attempt` 和 `step` 召回；`--recall-at` 替换这些触发点，`tool` 则添加 `recall` 工具。`--retriever auto` 在配置嵌入时使用 `embedding`，否则在 `start`/`attempt` 使用 `llm`、在 `step`/`tool` 使用 `bm25`；`off` 关闭召回，指定检索器则在用户指定的每个触发点运行。检索器失败时回退到 `bm25`，记录会注明原因。

[离线命令](../../guides/offline-retrieval.md)（英文）保持不变：它在已记录调用组成的语料上衡量检索器。运行中的块记录在 `llm_calls.recall_json` 中（触发点、检索器、查询、命中、`ms`、回退），`memory_stats` 则分别记录循环中的 `recall_ms`、`aux_retrieve_calls` 和 `retrieve_*` token 数。见[循环中的召回](memory.md#recall-in-the-loop)。
