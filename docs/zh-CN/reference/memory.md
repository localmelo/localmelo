<a id="memory"></a>

# 记忆

本页说明 localmelo 记住什么、如何在运行之间保留记忆，以及如何检查、重置和保护记忆。

维护者的记忆设计（其中部分尚未实现）位于网站的[记忆设计](https://localmelo.github.io/localmelo/en/memory-design.html)页面（英文）。

<a id="quick-reference"></a>

## 快速参考

| 要做什么 | 怎么做 |
|---|---|
| 把每个回复保留为简短笔记，并使用默认召回 | `melo --reasoning compress "<task>"` |
| 在整个交互式对话中使用简短笔记和 BM25 召回 | `melo --reasoning compress --retriever bm25 --note-writer heuristic` |
| 在运行之间保留记忆 | `export LOCALMELO_PERSIST_MEMORY=1` |
| 选择目录 | `export LOCALMELO_MEMORY_DIR=…`（默认 `~/.cache/localmelo/memory`） |
| 把一次运行的记忆保存在某个目录中 | `melo --memory-dir ~/melo-memory "<task>"` |
| 对一次运行关闭长期记忆 | `melo --memory off "<task>"` |
| 把模型的推理保留 N 天 | `export LOCALMELO_THINKING_RETENTION_DAYS=N`（默认 30） |
| 查看过去的运行 | 使用 `sqlite3` 查询；见[检查记忆](../guides/memory.md#inspect-memory) |
| 删除记忆 | 在 Python 中调用 `await hippo.forget_all()`（单个任务：`forget(task_id)`）；目前还没有命令。见[重置记忆](../guides/memory.md#reset-memory) |

<a id="working-memory-window"></a>

## 工作记忆窗口

工作记忆窗口最多保存 20 条消息：先是任务，然后是每个回复的轮次（工具调用消息及其结果，绝不拆开）。默认的（`slide`）模式在窗口满了之后，每个步骤都会移除最旧的单元，因此从大约第十一步起，每个请求的历史开头都比前一个晚一个单元，服务器的提示缓存只能复用系统提示、工具和任务。窗口还会受到提示上限（100,000 个字符）的限制；使用 `--context-window` 时，还会受到请求拟合的限制，方式同样是每个步骤移除一个单元。

`--window-eviction chunk|boundary`（或 `[agent] window_eviction`）是实验性的，用于衡量提示缓存的复用情况；默认值保持不变。

| 模式 | 窗口移除的内容 |
|---|---|
| `slide`（默认） | 窗口满后，每个步骤移除最旧的单元。 |
| `chunk` | 超过 20 条消息时，整单元地移除，直到剩下 12 条消息（上限减 8），因此历史的开头在大约四个步骤内都从同一个单元开始。随后，提示上限的拟合会降到上限的 70%，而不仅仅是低于上限。 |
| `boundary` | 在一次尝试内与 `chunk` 相同；在反思之后下一次尝试开始时，移除已完成尝试的轮次（反思会携带它们）；任务和收件箱消息则保留。 |

在 JSON 结果的 `config` 中，非 `slide` 的模式会以 `window_eviction` 出现；`slide` 则不增加任何内容。请求拟合（`--context-window`）在 `chunk` 和 `boundary` 下同样降到输入上限的 70%，但它在每次请求时都作用于整个窗口的副本，因此它不像窗口和提示上限拟合那样在步骤之间保持切割点。如果上下文预算小到每个步骤都会移除，请求仍会滑动。

<a id="digest-recall"></a>

## 摘要召回

只有当摘要经过检查的文本含有非空的 `files:` 或 `verified:` 行时，它才有资格进入长期准入。被准入的摘要会在其来源信息中保存其运行的规范化、已脱敏的任务行的哈希，以及其工作区真实路径的哈希，因此经过清理或截断的渲染行不会改变键，而同一任务文本在另一个文件夹中则是另一个任务；哈希并不能防止有人猜测内容，因此它不是隐私保证。没有键的摘要（包括旧条目）是跨任务的。第一个有意义的句子是有界的，因此在同一工作区中，任务行相同的不同任务可能共享一个键。如果运行的最后一次尝试没有证据，而较早的尝试有证据，则准入的是最后一个有证据的尝试摘要，从而不丢失已检查的工作。

在长期候选池中，同一任务的情节性摘要排在最前。存储还会在该候选池之外搜索最匹配的同任务摘要。跨任务摘要的得分必须至少是同一长期存储后端中最佳同任务摘要得分的 `cross_task_ratio` 倍；BM25 和余弦的得分不互相比较。最多发送 `cross_task_cap` 条跨任务摘要。若没有同任务摘要匹配，则 `cross_task_cold` 设定冷启动配额（默认为 0；若选择加入且匹配超过现有后端下限，则为 1）。它们仍受现有的两条长期命中限制、检索器选择和块预算的约束。非摘要的记忆（包括用户事实和笔记）不经过此门。

配置键位于 `[retrieval]` 下：ratio 为 1–10（默认 1.1），cap 为 0–2（默认 1），cold 为 0–1（默认 0）。它们在 `legacy` 中不起作用。交付的召回记录会为长期存储的摘要包含 `same_task` 和 `cross_task` 计数，并为交付的任务内笔记存储命中（笔记、草稿、思考及任务内摘要）包含 `task_notes`；无作用域的长期命中不计入这三者中的任何一个。`memory_stats.recall_same_task`、`recall_cross_task` 和 `recall_task_notes` 汇总已交付的记录；未发送的块不计数。

<a id="long-term-memory"></a>

## 长期记忆

在 `--reasoning compress`（或 `compress+replay`）且启用记忆时，`long_term.db` 是一个**类型化**的存储：每个条目都有类型、来源、状态、促使其写入的信号以及来源信息，同时保留文本和向量（早先的存储已经有这些）。启用持久化时，即使没有嵌入提供方，这些模式下也会存在该文件；纯文本条目保存空向量，并由进程内的 BM25 索引找到。在 `legacy` 中，只有配置了嵌入提供方时才存在该文件。

| 类型 | 保存的内容 | 来源 |
|---|---|---|
| `semantic` | 关于某个具名第三方（人、地点、组织、事件）的事实，通常是地址或日期等属性。 | `user`、`reasoning_note` |
| `episodic` | 本任务中发生的事：笔记中的普通条目，以及每个摘要。 | `reasoning_note`、`digest` |
| `profile` | 用户就自己所陈述的事实（第一人称：I/my/we/our；我/我的/我们；`call me`/`叫我`）。只有用户消息能产生它。 | `user` |
| `reflection` | 反思：尝试摘要、失败之处、证据、应避免的事项、后续方向和未解决的问题。 | `reflection` |

`source` 是 `reasoning_note`、`digest`、`reflection`、`user`、`sleep` 或 `step_summary`（旧行）之一。`status` 为 `active`、`superseded` 或 `conflicted`；v1 为每个新条目写入 `active`，并把被替换的条目标记为 `superseded`；`conflicted` 保留给睡眠流水线使用。`admitted_by` 是写入该条目的信号，对于旧行则为空。`provenance` 保存任务和会话、步骤和尝试、运行的文本、该轮触及的路径和命令、该条目来自的记录 id（`source_ids`），以及候选项 id 和条目索引。这些文本在写入前都经过脱敏并有长度限制。直接模式下会话 id 为 `""`，因为它不会传递会话 id；网关尚未把自己的 id 传入运行。搜索返回的是旧的元数据与这些列合并后的结果，因此旧的读取者仍能看到相同的文本和元数据。

事实的来源是确立该事实的观察，它可能早于陈述它的那条笔记。候选记录包括本次回复的工具结果、仍在工作记忆窗口中的工具结果，以及同一任务的其余工具结果。只保留任务的最后 200 条记录，绝不使用其他任务的记录。当一条记录共享一个强键，或至少共享两个四个字符或更长的、有区分度的非停用词时，它即匹配。唯一的最佳得分获胜；最高分相同则不引用任何来源。每个匹配的类型化键都会增加得分；日期冲突、两者都存在时的年份冲突、货币、电话号码或事件时间冲突都会拒绝一条记录。对于 JSON 记录，时间比较使用正文、文本、笔记、开始和结束，而不使用消息时间戳。极性不匹配会在任一方向上拒绝一条记录；该检测是启发式的，也可能拒绝带有无关否定子句的记录。否定条目必须完整地出现在记录中（末尾标点除外），否则它保持无来源；转述可能会被漏掉。其他匹配同样是启发式的，不是已验证的证据。金额保留其单位或货币。强键包括：日期、带冒号或 am/pm 的时间、带货币或单位的金额（或金额字段）、电话号码、含数字且长度为三个或更多字符的标识符、三位或更多位的数字，以及不在句首的大写专有名词。日期包括 ISO 格式、`April 9`、`9 April`，以及像 `4/9` 这样的裸斜杠日期。有歧义的斜杠日期采用美式的月在前顺序；无歧义的 `13/4` 为日在前。因此 `offsite 4/9` 会匹配 `2026-04-09`。斜杠对后面跟着数量或计数词，或前面是数字加空格（如带分数的 `2 1/2`），则不算日期；`24/7`、无效日期以及小写的 `may`/`march` 也不是日期。这些都是启发式规则，因此其他分数格式仍可能有歧义。一位或两位的裸数字和序数词不是键。诸如 `done`、`ok`、`good`、`thanks`、`好的`、`完成` 和 `可以` 之类的填充词是停用词，因此仅说 "Done." 的条目不会获得来源。当笔记完全没有记录文本时，该条目保留笔记自身的 `source_ids`；当它有记录而没有任何一条匹配时，该条目则不获得来源，准入之后也是如此。摘要的 id 是其笔记的 id 与其条目背后记录的并集。仅有框架而没有超出框架事实的完成或自我工作声明，无论来自笔记还是摘要，都会以原因 `meta claim about the agent's own work` 被 `IGNORE`。`task:` 行、文件、路径或函数名会使文本成为内容。由 `outcome` 为已完成的运行所准入的最终摘要完全免于此过滤，但仍需要非空的文件或验证行才能准入。

<a id="experimental-typed-assertions"></a>

## 实验性类型化断言

Python 调用方可以通过 `AgentOptions(typed_assertions=True)` 选择启用；没有 CLI 或配置开关。默认保持关闭。在显式推理模式中，被识别的简短用户断言以及受支持的、有来源的笔记，可以在 `provenance` 中携带带版本号的断言元数据，并与其原始的已检查文本并存。识别范围有意保持狭窄，它不是通用的事实抽取器。有来源的笔记需要与一个精确匹配的文本字段和记录 ID；不受支持或未经验证的断言不会被自动取代。源记录和断言仍要经过基于模式的脱敏，而这种脱敏可能会漏掉敏感信息。

在同一任务内，一个明确标记的、对已验证类型键的更改，可以在其已验证的来源时间（`timestamp` 或 `date`）晚于某个较早的活动断言时，取代该断言；用户陈述则按其任务内的消息和句子位置排序。用户计数器与来源时间戳是不同的时钟，重新开始的任务不会链接到较早的用户陈述（它们仍然是冲突）。缺少来源时间或时间为无时区的（naive）时不排序；标记包括中文 `取消` 以及结尾的 `now`/`instead`。存在多个活动候选项、缺少排序或没有更改标记的不同值，都会保留为冲突。这不会跨任务或会话链接实体。未链接的断言会被保留，但不声称它指的是同一实体；相对的 `tonight` 只有从匹配记录的带时区时间戳中才会得到有效日期。旧行没有推断出的极性。

启用该选项后，BM25 长期搜索可以包含已被取代的条目；召回会为这些条目和未链接的断言加上标签，依据记录的来源日期而非机器时区为其标题标注日期，把所选的长期命中按来源观察或用户条目的挂钟时间，从新到旧排序，并且只忽略完全相同的规范化重复项；内容不同的近似重复项会被保留（`NEW`，未链接）。搜索仍然有相关性和条目数量的限制，不保证每个相关事实都会出现。长期存储上的 `history_entries(task_id)` 按从新到旧的顺序返回该任务的活动条目和已取代条目；forget 会同时删除二者。在非持久化的显式模式智能体中，该选项会启用一个 RAM 存储；持久化则需要 `memory_dir` 或持久化环境变量。

除非有信号，否则不会写入任何内容：

| 信号 | 何时 | 准入的内容 |
|---|---|---|
| `rehearsal` | 重述，成功的调用对笔记的路径或命令采取了行动，或反思的 `used_notes` 引用了它。 | 该候选项，立即准入。未改变的候选项再次准入则不写入任何内容。 |
| `user_request` | 用户消息要求记住（`remember that …`、`记住…`）。 | 请求标记之后携带的事实、本次运行的 `remember` 笔记，以及运行结束时的最终摘要；绝不包括最终反思。该事实由 `classify_kind` 分类，因此除非它是第一人称（`profile`），否则为 `semantic`；下文的元名词和问句规则同样适用于它。 |
| `user_statement` | 用户消息中的陈述句在没有记住请求的情况下陈述了一个稳定事实，或者纠正了某个事实并给出了更正后的值。问句（`?`/`？`）从不计入，`my … is`/`我的…是` 会拒绝元名词（task、question、request、goal、plan、problem、issue、job、point；任务、问题、请求、目标、计划）。 | 该句子，立即准入。 |
| `outcome` | 运行结束时，仅在 `--admit-on-outcome on` 时。 | 最后一次尝试的摘要，其终止状态记录在来源信息中；绝不包括最终反思。对于已完成的运行，该摘要免于元声明过滤。 |
| `sleep` | 为睡眠流水线的探测门保留。 | v1 中没有内容。 |

只有用户自己的文本才会提供用户信号：工具结果、笔记、召回块或助手消息都不会；笔记中的第一人称条目绝不会成为用户资料条目。候选项边界在写入前会以目标 `long` 再次运行：被拒绝的条目不写入条目，也不写入决定。写入的所有文本，包括来源信息中的任务文本、路径和命令，在写入之前都先经过 `redact_secrets` 脱敏并限制长度。

策略会将该条目与同类型的活动条目比较，并选择一个动作：

| 动作 | 规则 | 写入 |
|---|---|---|
| `IGNORE` | 简短的事实（规范化后少于 20 个字符）、只命名本任务文件的事实、仅有框架而没有超出框架事实的完成或自我工作声明，或重复项（规范化后相等、文本重叠 >= 0.9，或余弦 >= 0.95）。 | 不写入条目；重复的目标其 `uses` 增加。 |
| `UPDATE` | 主题相同且值兼容，或有更改标记（`moved`、`now`、`搬到` 等）。 | 写入一个新条目来取代旧条目；旧条目变为 `superseded`。 |
| `CONFLICT` | 主题相同，值相互矛盾，且没有更改标记。 | 写入一个新的活动条目并与旧条目链接；两者都仍可被召回。 |
| `RELATED` | 主题不同，但文本重叠 >= 0.35 或余弦 >= 0.75。 | 写入一个新的活动条目并与旧条目链接。 |
| `NEW` | 其他所有情况。 | 写入一个新的活动条目。 |

情节性条目的主题是其候选项加上其条目索引，因此同一笔记的两个不同条目是两个条目；重述会重新提出相同的候选项和条目顺序，因此它仍然是 `UPDATE`。

**决策日志。** `long_term.db` 还有一个 `decisions` 表：每个动作一行，包括候选项 id、动作、类型、信号、条目和目标 id、一行原因、该条目的 `source_ids`、已检查文本的 SHA-256，以及任务 id。文本本身仅在使用 `--keep-decision-text` 时才会保存（这是一个评估开关，在交互式界面中会被拒绝）；哈希使之后无需保留文本也能进行关联。`forget(task_id)` 会删除任务的条目和决策；`forget_all()` 会清空两张表。若要重置文件，请在停止 localmelo 后删除 `long_term.db`。本版本没有基于年龄的保留策略；睡眠流水线的策略将在之后提供。

**迁移。** 早期存储的 `long_term.db` 在打开时会被就地迁移：`user_version` 变为 1，类型化的列和 `decisions` 表通过 `ALTER TABLE` 添加（不重建表），旧的反思变为 `kind='reflection'`/`source='reflection'`，其他每一行变为 `episodic`/`step_summary`，其文本、向量和元数据字节保持不变。较早版本的 localmelo 仍然可以打开并搜索该文件。比当前版本更新的文件会输出一条 WARNING，运行继续进行但没有长期记忆。只读导出从不迁移文件；见[离线检索](retrieval.md#the-typed-long-term-store)。

在 `start` 时，长期部分读取其他任务中各类型的活动条目：`--recall-min-score` 适用于余弦得分，而 BM25 的第一阶段候选池没有绝对下限。文档的标识为 `long:<entry id>`，而不是其文本的哈希。`memory=False` 会移除存储、写入和这一部分。

<a id="files-on-disk"></a>

## 磁盘上的文件

```text
~/.cache/localmelo/memory/
  history.db      tasks, steps and model calls
  long_term.db    long-term memory, when memory is on and either an embedding
                  backend is configured or the reasoning mode is not legacy
  procedures.db   procedural memory, written at the end of every run
```

- SQLite 还可能创建 `history.db-wal` 和 `history.db-shm`（`long_term.db` 和 `procedures.db` 也是如此）。它们属于数据库，请让它们保留在原处。
- 目录在智能体启动时创建：直接运行或 `melo ""` 循环开始时，或网关会话创建时。
- 新目录的权限为 0700，因此只有你能打开其中的文件；新的 `history.db`、`long_term.db` 或 `procedures.db`，连同其 `-wal` 和 `-shm`，权限为 0600。
- 默认目录 `~/.cache/localmelo/memory` 存在、归你所有，且对组或其他用户开放（早期版本就是这样留下的）时，会被设为 0700。日志中会有一行 INFO 说明这一点。
- 你自己选择的目录（`LOCALMELO_MEMORY_DIR`、`--memory-dir`），以及每个已存在的文件，保持其权限不变。要对其他用户关闭它们，请运行：

  ```bash
  chmod 700 ~/melo-memory
  chmod 600 ~/melo-memory/history.db*
  chmod 600 ~/melo-memory/long_term.db*
  ```

- `procedures.db` 保存[程序记忆](../concepts/procedures.md)。启用持久化后，第一个结束的运行会创建它，每次运行都会在结束时添加其记录，包括 `--memory off` 的运行。关闭持久化时，程序记忆保留在 RAM 中，且不写入任何内容。
- 旧的 playground 默认禁用；`melo --serve --enable-playground` 会启用它，并创建 `~/.cache/localmelo/playground/`。其 `history.db` 会在相同的天数内保留模型的推理，并在某个聊天第一次使用它时清除更早的推理：每个聊天都会打开自己到共享文件的连接。其聊天也共享那里的 `procedures.db`。

```text
~/.cache/localmelo/gateway/sessions.db   the gateway's sessions
~/.cache/localmelo/cli/sessions.db       the sessions of every terminal melo
```

<a id="task-records"></a>

## 任务记录

每次运行都会在历史记录的 `tasks` 表中保存一行：

| 列 | 值 |
|---|---|
| `task_id` | 运行的 `task_id`：32 个十六进制字符。 |
| `query` | 任务原样保存，其中的秘密按[脱敏](#model-calls)规则替换。 |
| `status` | 运行期间为 `running`，之后为它结束的方式；见[运行结果](run-results.md#statuses)。`cancelled` 只出现在交互式运行中：Python API 中，以及网页工作区的历史记录中。 |
| `result` | 运行的结果；若结果为空，则为其错误。 |

- 无论什么原因结束，运行结束时都会再次保存该行。被 SIGTERM 或 SIGHUP 结束的直接运行会保存为 `crashed`，结果为 `Cancelled by SIGTERM` 或 `Cancelled by SIGHUP`。
- 被取消的交互式运行会保存为 `cancelled`，原因作为其结果：即在 Python API 中以 `on_event` 或 `approval` 启动的运行，或者由 Stop 或其 600 秒上限停止的旧版 playground 聊天（启用时，其历史记录位于 `~/.cache/localmelo/playground/` 中）。其被中断的步骤在 `steps` 中每个工具调用有一行（见[停止运行](../guides/python.md#stop-a-run)）。其他被取消的运行保存为 `crashed`。
- 仍显示 `running` 的行属于一个进行中的运行，或属于在能保存该行之前就被杀死的进程（例如被 SIGKILL 杀死）。
- 较早版本的行只使用 `running`、`completed` 和 `failed`。在那些版本中，`failed` 也涵盖了现在会以 `gate_stopped`、`budget_exhausted` 或 `stuck` 结束的运行，而异常终止的运行则保持 `running`。

`steps` 表每个工具调用一行：`step_id`、`task_id`、`thought`（模型的文本）、`tool_call_json`、`tool_result_json`、`timestamp` 以及 `seq`（任务内的顺序）。`thought`、调用参数中的每个字符串，以及结果的输出和错误，都以替换了[脱敏](#model-calls)内容的形式保存；运行本身和模型保留原文。

<a id="model-calls"></a>

## 模型调用

启用持久化时，历史记录还会在 `history.db` 的 `llm_calls` 表中保存模型的每次回复及其推理：规划请求和反思请求的回复各一行。内存中的历史记录则不保存任何模型调用。

| 列 | 值 |
|---|---|
| `task_id` | 运行的 `task_id`。 |
| `seq` | 该回复在运行中的编号，从 0 开始，按回复到达的顺序。 |
| `kind` | 规划请求为 `plan`，反思请求为 `reflect`。 |
| `step`、`attempt` | 回复到达时运行的步骤和尝试。反思的步骤是它所反思的那次尝试的最后一步。 |
| `outcome` | 对于 `plan`：回复有工具调用时为 `tools`，否则有文本时为 `answer`，再否则为 `empty`。其他所有类型为 `aux`。它描述的是回复本身，而不是运行对它的处理。 |
| `content`、`thinking` | 回复的文本和推理，已脱敏（见下文）。 |
| `reasoning_key` | 推理的来源：`reasoning_content` 或 `reasoning`（兼容 OpenAI 的服务器）、`thinking`（Ollama），或当回复没有推理或其提供方未说明时为 `""`。 |
| `step_ids_json` | 对于带工具调用的回复，为其 `steps` 行的 `step_id`，以 JSON 列表形式按调用顺序排列；对于没有产生任何行的回复，为 `NULL`。 |
| `redactions` | 脱敏替换的片段数。 |
| `tool_calls_json` | 回复的工具调用，以 JSON 列表形式表示，每项为 `{"tool_name", "arguments", "id"}`；否则为 `NULL`。 |
| `usage_json` | 服务器报告的 token 计数，以 JSON 表示；兼容 OpenAI 的服务器发送时包括 `cached_tokens` 和 `reasoning_tokens`；否则为 `NULL`。 |
| `duration_s` | 请求耗时，不含之前失败的尝试以及重试前的等待。 |
| `timestamp` | 回复到达的时间，以自纪元以来的秒数计。 |

- 工作记忆窗口保留每个回复的推理，包括空回复的推理，以及被大小截断清空了文本的轮次的推理；`llm_calls` 按接收到时的样子记录它（R1 的 `thinking` 和 `reasoning_key`）。在 `legacy` 中，由提供方决定它是否会再次发送：兼容 OpenAI 的不发送，Ollama 原生的发送。`drop` 在发送时将其去除。`replay` 会在范围内的助手轮次上发送它（默认为当前任务的轮次；要求重放的提供方会获得窗口中的每个轮次），使用一个请求键（`reasoning_content`、`reasoning` 或 Ollama 的 `thinking`），每个轮次最多 40,000 个字符（要求重放的提供方没有上限）。两种显式模式在构建提示时都跳过 `[RECALL]`，因此嵌入提供方的长期记忆会被写入，但不会被读回；`legacy` 则照旧读取它。
- 被重试的请求只产生一行，即返回的那次回复的行。因此一次运行的行数等于其 `n_llm`，除非某行无法写入：等待超过 5 秒才拿到另一个进程对 `history.db` 写锁的行，或因其他原因失败的行，会被遗漏，并在 stderr 上输出一条警告（该运行之后的失败以调试级别记录），然后运行照常进行，仿佛什么也没发生。缺失的行会在 `seq` 中留下空缺。
- `step_ids_json` 对于回答、空回复、反思、被计划检查拒绝的回复，以及在其工具运行期间被中断（由 `--timeout` 或取消中断）的规划回复，保持为 `NULL`：这类回复不产生 `steps` 行。在运行写入某个步骤的行时被中断，可能留下未关联的行；每一行都属于其 `timestamp` 之前该任务最近的 `plan` 行。在 Python API 中，被取消的交互式运行会关联其清理过程所写入的行。
- **脱敏。** 记忆存储的每段文本都会经过与 Python API 的事件相同的最小化秘密脱敏：已知的密钥格式（`sk-` 密钥、GitHub 令牌、AWS 访问密钥 id）、PEM 私钥块的正文（`-----BEGIN … PRIVATE KEY-----`，或 PGP 的 `PRIVATE KEY BLOCK`；保留 BEGIN 和 END 行，没有 END 行的块会一直脱敏到文本末尾）、`Bearer` 之后的令牌，以及与 `verification code`、`passcode`、`OTP` 或 `验证码` 位于同一行且相距 20 个字符以内的 4 到 8 位数字，都会变为 `[redacted]`。它涵盖 `tasks.query`、`tasks.result`、`steps.thought`、`steps.tool_call_json` 的参数中的每个字符串、`steps.tool_result_json` 的输出和错误、`llm_calls.content` 和 `llm_calls.thinking`，以及长期记忆保存的文本（每个步骤的摘要和被提升的反思）。模型读取的内容不会脱敏：工作记忆、运行发送的请求以及运行的 `result` 保留原始文本，因此只有存储的副本会改变。运行的事件会对自己的字段进行脱敏：`assistant.message` 的 `content` 和 `thinking`，以及 `reflection` 的 `rationale`、`summary` 和 `best_effort_result`；见[事件](events.md#assistantmessage)。它是最小化的：其他形式的秘密会保留，粘连在前一段文本上的密钥也会保留：紧跟字母、数字、`_` 或 `-` 之后的 `sk-` 密钥，紧跟字母、数字或 `_` 之后的 GitHub 令牌，以及紧跟字母或数字之后的 AWS 访问密钥 id。
- **保留期。** 早于 30 天（或 `LOCALMELO_THINKING_RETENTION_DAYS`）的推理会被清除：`thinking` 变为 `""`，并且 `content` 中的推理也会被清除，清除方式取决于该行的 `reasoning_key`。对于 `thinking`（Ollama），即 Ollama 提供方添加的、位于 `content` 开头的 `<think>` 块。对于 `""`（没有推理解析器的服务器），则是开头的 `<think>` 块、开头未闭合的 `<think>`（输出在模型思考时被截断），或没有前置 `<think>` 的 `</think>` 之前的文本（这是那些在提示中打开该块的聊天模板所留下的情况）。对于 `reasoning_content` 和 `reasoning`，`content` 不含推理，保持不变。该行及其其他列保持不变。旧推理会在智能体第一次使用 `history.db` 时清除：即直接运行开始时，或网关会话的第一次运行时。
- **关联事件。** 在 Python API 中，一个 `plan` 行与其步骤的 [`assistant.message`](events.md#assistantmessage) 事件具有相同的 `task_id`（即事件的 `run_id`）和 `step`。一次尝试的 `reflect` 行（一行或两行）属于其 [`reflection`](events.md#reflection) 事件；按 `seq` 排在最后的那行是它所读取的回复。事件的 `seq` 与行的 `seq` 是两个不同的计数器。
- **导出。** `melo-retrieve --export-run` 把运行的模型调用转换为用于检索实验的文档；见[导出历史数据库](../guides/offline-retrieval.md#export-a-history-database)。
- **架构版本。** `history.db` 的 `PRAGMA user_version` 为 1。较早版本（版本 0）的文件在本版本首次打开时会获得空表并变为版本 1，并在日志中输出一行 INFO；其任务和步骤保持不变。较早版本仍可读写版本 1 文件中的任务和步骤，并忽略 `llm_calls`。比本版本更新的文件会使本版本的每次运行都以 `crashed` 结束，错误为 `HistorySchemaError: <path> has schema version <n>; this localmelo reads version 1`。

<a id="sessions"></a>

## 会话

会话存储（[会话](sessions.md)）把对话保存在它自己的文件中，与 `history.db` 分开。直接运行的 `melo` 会在其第一条消息时打开 `~/.cache/localmelo/cli/sessions.db`；网关的控制台使用它自己的存储。

- **每种只有一个写入者。** 文件会记录其写入者是网关还是终端，而另一类存储拒绝打开它，如同它拒绝任何其他 SQLite 文件一样，并且不写入它。网关的文件只有一个所有者，它在运行期间持有 `sessions.db.owner` 上的锁；终端则共享各自的文件，并为每个正在进行的轮次持有租约。在 `sessions.db` 旁边，SQLite 还可能创建 `sessions.db-wal` 和 `sessions.db-shm`。
- **私密。** 目录以 `0700` 权限创建，文件以 `0600` 权限创建；写入者自己的目录（`gateway/` 或 `cli/`）如果权限比 0700 宽松，则会被设为 `0700`。
- **持久性。** 网关的文件使用 `synchronous=FULL`：已提交的内容在操作系统崩溃后仍然保留。CLI 的文件使用 `NORMAL`，与 `history.db` 相同：进程崩溃不会丢失任何已提交的内容，而操作系统崩溃可能会丢失最后的更改。
- **所保存的内容。** 每轮的文本、其事件（模型的文本和推理、工具调用及其输出）、工具返回的图片、最终答案，以及每个会话的一个窗口检查点：即最近一轮结束时的工作窗口，包含工具调用及其结果，但不包含工具的输出（输出由事件保存）。每轮的 `note` 列保存压缩模式下该轮最后一次尝试的摘要（在其他模式下为空）。
- **脱敏。** 最终答案会保存三份：在该轮的 `run.finished` 事件中，以及作为该轮的答案和结果。每一份副本，以及检查点中助手的文本和思考，在保存之前都会经过[事件](events.md#assistantmessage)的最小化脱敏。
- **保留期。** 存储打开时，以及其程序调用 `apply_retention()` 时（保持存储打开的网关应每 24 小时调用一次），超过 30 天（或与 `history.db` 相同的 `LOCALMELO_THINKING_RETENTION_DAYS`）的内容会被清除：`assistant.message` 事件的思考和其文本中的 `<think>` 块；最终答案副本中的 `<think>` 块；窗口检查点，以及较新的检查点中比该时间更早的每条消息的思考和 `<think>` 块；以及被撤回的消息。轮次、它们的答案和事件会保留，直到你删除该会话。
- **被终止的运行。** 随进程一起被杀死的运行，会在 `history.db` 的 `tasks` 表中保持 `running` 状态（见[任务记录](#task-records)）；其轮次在 `sessions.db` 中的状态才是可信的。轮次的 `run_id` 是运行的 `task_id`，若运行从未开始，则为 `""`。
