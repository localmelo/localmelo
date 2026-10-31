<a id="guide-memory"></a>

# 指南：记忆

在多次运行之间保留记忆、检查和重置记忆、关闭记忆，并了解记忆会保存哪些关于你的内容。为项目启用记忆之前请阅读本指南；文件和选项见[记忆参考](../reference/memory.md)。

<a id="keep-memory-between-runs"></a>

## 在多次运行之间保留记忆

```bash
export LOCALMELO_PERSIST_MEMORY=1
melo "What is 6*7?"
```

- `LOCALMELO_PERSIST_MEMORY` 的值只要非空就会启用持久化，即使是 `0` 或 `false` 也一样；`unset LOCALMELO_PERSIST_MEMORY` 会关闭它。
- 文件存放在 `~/.cache/localmelo/memory/`；使用 `LOCALMELO_MEMORY_DIR` 可以指定其他目录，其值会原样使用（shell 没有展开的 `~` 也不会被展开）。
- `--memory-dir DIR` 为一次直接运行启用持久化并存入 `DIR`，不受变量值的影响；此处会展开 `~`：

  ```bash
  melo --memory-dir ~/melo-memory "What is 6*7?"
  ```

- 网关使用启动时的变量，其所有会话都保存到同一组文件中。
- 历史记录将模型的推理保留 30 天（参见[模型调用](../../reference/memory.md#model-calls)（英文））；`LOCALMELO_THINKING_RETENTION_DAYS` 可为 `history.db` 和会话存储的 `sessions.db` 设置其他保留天数，例如 `7` 或 `0.5`（参见[会话](../../reference/memory.md#sessions)（英文））；如果值不是大于 0 的数字，`melo` 会在运行任何操作之前停止（参见[用法错误](../../reference/cli.md#usage-errors)（英文））。

后续运行会召回什么：

- 在后续进程中召回长期记忆需要启用持久化、开启记忆，并且已有存储的条目。
- 在 `legacy` 中，持久化长期存储还需要嵌入后端；显式推理模式无需嵌入后端即可持久化纯文本条目，并通过 BM25 找到它们。
- 在显式模式下，必须启用召回并包含 `long` 类型；`drop` 和 `replay` 默认关闭 `auto` 召回。
- 尝试摘要只有在包含已确认的文件写入或符合条件的通过检查时才会写入长期记忆；召回优先使用同一任务、同一工作区的摘要，新任务在同任务摘要匹配之前不会获得跨任务摘要（见[摘要召回](../../reference/memory.md#digest-recall)（英文））。

<a id="experimental-typed-facts"></a>

## 实验性的类型化事实

Python 调用方可以在显式推理模式中启用 `AgentOptions(typed_assertions=True)`。

- 它会保留受支持的简短事实；当一个经过验证、有先后顺序的变更取代旧断言时，旧断言可以作为带标签的历史保留下来。
- 有歧义或未关联的事实不会被悄悄关联；解析器和来源检查都是保守且不完整的，召回受相关性限制。
- 参见[类型化断言参考](../../reference/memory.md#experimental-typed-assertions)（英文）。

<a id="notes-melo-keeps-about-you"></a>

## melo 记下的关于你的笔记

![“记忆”标签页：melo 记下的关于你的笔记，每条都有来源对话以及编辑和删除按钮。](../../assets/img/memory-tab.webp)

在网页控制台和 App 中，melo 会跨对话保存关于你的简短笔记（见[跨对话记忆](../concepts/memory.md#memory-across-chats)）。它用两种方式记一条笔记，都不会请你确认：

- **请它记。** "Remember that Sam has the lab keys"、"keep in mind that my group meeting is on Thursdays"、"记住：我的车停在B2层"、"别忘了…" 或 "帮我记一下…" 会立刻被记下。
- **告诉它。** "I prefer short answers" 或 "call me Ray" 会在 melo 判断值得记时被记下，对话里一个已完成任务的结果也是如此；只有网页、文件、工具输出或任务向对话提的问题说过的内容它从不记，密码或密钥也从不记。
- melo 被告知：关于你的明确、长期的事实（"I live in Utrecht"、"我的导师是周教授"），要在你说出的那一轮就记下。
- **改掉它。** 在任意对话里说出新的值（"I live in Utrecht now"、"我的健身教练是小宇"）：它会替换关于同一件事的旧笔记，召回时排在最前，回答只给出新的值：只有你问起过去（"before"、"used to"、"以前"）时，你说旧值的那个对话才会重新出现，并被标为较旧。
- **长对话。** 对话只让最新的消息留在视野里。你在同一个对话里更早说过的话，匹配你的消息时会回来，并标为 `this chat`。
- **任务从它们开始。** 你从一个对话开始的任务（或你保存的定时事项）会带着与它相关的笔记开始：关于你和你偏好的笔记、它所在文件夹的笔记，以及与它的请求匹配的笔记。人物和决定不包括在内，因为任务要调用工具。任务页面把它们折叠在“melo 告诉任务的内容”下。关闭记忆，或没有匹配的笔记时，任务不带这些内容开始。继续或重启时使用同样的内容。
- 每条笔记会以 `Remembered: … · Undo` 显示在你的消息下方；对新值点 Undo 会恢复旧笔记。
- **其他助手。** 主对话第一次打开时，或在终端里第一次打开 `melo` 时，melo 会提议导入这台 Mac 上其他助手保存的关于你的笔记，以文件列表的形式（见[导入其他助手的记忆](../concepts/memory.md#bring-in-memory-from-other-assistants)）。在你选择之前不会导入任何内容，**Not now** 会结束这个提议。之后可以用 Memory 标签页的 **From other assistants**、`/memory import` 或 `melo memory import`。
- 之后要查看、改写或忘掉笔记，使用控制台的 **Memory** 标签页、终端里的 `/memory`，或 `melo memory list` 和 `melo memory forget`；要停止记笔记，运行 `melo config set memory.across_chats false`，然后重启 `melo serve` 或 App。

<a id="turn-long-term-memory-off"></a>

## 关闭长期记忆

- `--memory off` 或 Python 中的 `AgentOptions(memory=False)` 可在一次运行中关闭长期记忆：不从中召回内容，也不向其写入内容，反思也不会被提升为长期记忆。
- 工作记忆和历史记录照常工作。
- `embedding_backend = "none"` 禁用的是嵌入后端，而不是记忆本身；显式推理模式仍可存储和召回纯文本条目。
- 设置嵌入模型的方法见[设置嵌入模型](embedding.md)。

<a id="inspect-memory"></a>

## 查看记忆

这些文件是 SQLite 数据库：使用 `sqlite3` 命令行工具读取，并且只使用 `SELECT` 查询：

```bash
cd ~/.cache/localmelo/memory

# The tables
sqlite3 history.db ".tables"

# The 10 most recently saved tasks
sqlite3 history.db "SELECT task_id, status, substr(query, 1, 60) FROM tasks ORDER BY rowid DESC LIMIT 10"

# How runs ended
sqlite3 history.db "SELECT status, COUNT(*) FROM tasks GROUP BY status"

# The steps of one task
sqlite3 history.db "SELECT seq, tool_call_json, substr(tool_result_json, 1, 200) FROM steps WHERE task_id = '<task_id>' ORDER BY seq"

# The model calls of one task, with the start of their reasoning
sqlite3 history.db "SELECT seq, kind, outcome, step, substr(thinking, 1, 80) FROM llm_calls WHERE task_id = '<task_id>' ORDER BY seq"

# Each step of one task, with the reasoning of the call that made it
sqlite3 history.db "SELECT s.seq, c.seq, substr(c.thinking, 1, 80), s.tool_call_json FROM llm_calls c, json_each(c.step_ids_json) j JOIN steps s ON s.step_id = j.value WHERE c.task_id = '<task_id>' ORDER BY s.seq"

# Long-term memory: the number of entries, and a preview
sqlite3 long_term.db "SELECT COUNT(*) FROM long_term"
sqlite3 long_term.db "SELECT id, substr(text, 1, 80), metadata FROM long_term LIMIT 5"

# Procedural memory: the routines, and how their latest runs went
sqlite3 procedures.db "SELECT routine_id, substr(query_text, 1, 60), place FROM routines"
sqlite3 procedures.db "SELECT routine_id, status, steps, level, model FROM episodes ORDER BY created_at DESC LIMIT 10"
```

- `melo-procedures export` 将全部程序性记忆以 JSON 行形式输出（参见 [melo-procedures](../reference/procedures.md#export-and-forget)）。
- 将 `<task_id>` 替换为任务列表中的 ID，或 JSON 行中的 `task_id`。

<a id="reset-memory"></a>

## 重置记忆

Python API 可以遗忘记忆。目前还没有相应命令；`melo --forget task|all` 命令将随界面工作推出。

```python
import asyncio
from pathlib import Path

from localmelo.melo.memory.coordinator import Hippo
from localmelo.melo.memory.history.sqlite import SqliteHistory
from localmelo.melo.memory.long.sqlite import SqliteLongTerm


async def main() -> None:
    memory = Path.home() / ".cache" / "localmelo" / "memory"
    hippo = Hippo(
        history=SqliteHistory(memory / "history.db"),
        long=SqliteLongTerm(memory / "long_term.db"),
    )
    try:
        await hippo.forget_all()      # one task: await hippo.forget(task_id)
    finally:
        await hippo.aclose()


asyncio.run(main())
```

`agent.hippo` 是对智能体已打开存储的同一封装接口，因此运行中的脚本也可以通过它调用 `forget`。`SqliteHistory` 和 `SqliteLongTerm` 本身也有 `forget(task_id)` 和 `forget_all()`，内存中的 `History` 和 `LongTerm` 同样如此。

- **`forget(task_id)` 删除的内容。** 历史记录的 `tasks`、`steps` 和 `llm_calls` 表中属于该任务的行，以及长期记忆中元数据指明该任务的条目：它的反思（`"task_id"`），而不是各步摘要（只有 `"step_id"`）。
- **`forget_all()` 删除的内容。** 所有长期记忆条目以及历史记录中这三张表的所有行。
- **不涉及的内容。** 工作记忆（当前对话）、程序性记忆（由 `melo-procedures forget` 清除）、会话存储（控制台的聊天记录位于 `~/.cache/localmelo/gateway/sessions.db`），以及旧版 playground 保存在 `~/.cache/localmelo/playground/` 中的记忆。
- **实现方式。** 在存储的写锁下执行一次 `BEGIN IMMEDIATE` 事务；删除操作会覆盖释放的空间（`PRAGMA secure_delete=ON`），检查点会截断 WAL，因此文本不会留在数据库文件或其 `-wal` 文件中；如果另一个连接阻止 WAL 截断，检查点会跳过并在日志中记录 WARNING，之后的检查点会将其清除。
- localmelo 运行时可以安全地遗忘，但不能安全地删除文件。

要删除全部内容，包括 API 不涉及的内容，请停止所有 localmelo 进程并删除文件：

1. 停止所有 localmelo 进程：直接运行、`melo serve` 和后台服务（使用 `melo serve uninstall` 停止服务；单纯使用 `kill` 后 launchd 会重新启动它）。
2. 删除文件：

   ```bash
   # Everything
   rm -rf ~/.cache/localmelo/memory

   # Only long-term memory
   rm ~/.cache/localmelo/memory/long_term.db*

   # Only history
   rm ~/.cache/localmelo/memory/history.db*

   # Only procedural memory
   rm ~/.cache/localmelo/memory/procedures.db*

   # The console's session store (including chats)
   rm -f ~/.cache/localmelo/gateway/sessions.db*

   # The legacy playground's memory, if it was enabled
   rm -rf ~/.cache/localmelo/playground
   ```

3. 重新启动 localmelo。它会在需要时创建新文件。

要清除模型存储的推理，请先停止 localmelo：

```bash
# The records of the calls, with all their reasoning
sqlite3 ~/.cache/localmelo/memory/history.db "DELETE FROM llm_calls"

# Only the thinking column, keeping the records of the calls
sqlite3 ~/.cache/localmelo/memory/history.db "UPDATE llm_calls SET thinking = ''"
```

- 第二条命令会留下 `content` 中的推理：Ollama 回复开头的 `<think>` 副本（保留期机制之后不再删除该副本），以及没有推理解析器的服务器回复中的内嵌块；要删除所有存储的推理，请使用第一条命令。
- 如果设置了 `LOCALMELO_MEMORY_DIR` 或使用了 `--memory-dir`，请改为删除该目录中的文件。

> [!WARNING]
> localmelo 运行时删除文件可能导致文件损坏。请先停止 localmelo。

<a id="privacy"></a>

## 隐私

| 文件 | 保存的内容 |
|---|---|
| `~/.cache/localmelo/memory/history.db` | 你的任务、模型文本、每次工具调用及其参数以及每个工具结果，均经过[模型调用](../../reference/memory.md#model-calls)（英文）所述的脱敏处理；还有每次调用中经过脱敏的模型推理，默认保留 30 天。 |
| `~/.cache/localmelo/memory/long_term.db` | 经过脱敏的各步摘要和反思及其嵌入向量。 |
| `~/.cache/localmelo/memory/procedures.db` | 每个任务经规范化的查询、运行目录以及工具调用的标签（文件名和程序名，不保存其内容或输出）。 |
| `~/.cache/localmelo/cli/sessions.db` 和 `~/.cache/localmelo/cli_history` | 终端的每个轮次及其工具调用和输出；你的消息（两者均为 `0600`）。 |
| `~/.cache/localmelo/gateway/sessions.db` | 控制台的会话行和轮次。 |
| `~/.cache/localmelo/playground/` | 仅在使用 `melo serve --playground` 时（默认禁用）：聊天的历史记录和程序性记忆；选择了嵌入模型时还有长期记忆。 |

- 这些文件没有加密；它们的私密程度取决于你的主目录。
- `procedures.db` 中的查询和标签经过最基本的脱敏：先处理[模型调用](../../reference/memory.md#model-calls)（英文）所述的密钥格式、bearer token 和一次性验证码，再处理 URL 中的密码（`postgres://app:<redacted>@host`，保留用户名），以及 NAME 包含 `key`、`token`、`secret`、`password`、`passwd` 或 `pwd` 时 `NAME=value` 或 `NAME: value` 中的值（`API_KEY=<redacted>`）；其他内容均原样保留，例如主机名、路径和其他值。
- 工具返回的图像只存储引用：MIME 类型、SHA-256 摘要和大小，绝不存储图像本身。
- `~/.cache/localmelo/logs/cli.log` 记录请求 URL 和错误。
- 删除文件的方法见[重置记忆](#reset-memory)；Python API 的 `forget` 会覆盖所删除的单元并截断 WAL，因此文本不会留在 `history.db-wal` 或 `long_term.db-wal` 中。
