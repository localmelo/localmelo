<a id="guide-memory"></a>

# 指南：记忆

<a id="keep-memory-between-runs"></a>

## 在多次运行之间保留记忆

```bash
export LOCALMELO_PERSIST_MEMORY=1
melo "What is 6*7?"
```

- `LOCALMELO_PERSIST_MEMORY` 的值只要非空就会启用持久化，即使是 `0` 或 `false` 也一样。要关闭它，请运行 `unset LOCALMELO_PERSIST_MEMORY`。
- 文件存放在 `~/.cache/localmelo/memory/`。使用 `LOCALMELO_MEMORY_DIR` 可以指定其他目录。其值会原样使用：如果 shell 没有展开 `~`，例如将它放在引号中，程序也不会展开它。
- `--memory-dir DIR` 为一次直接运行启用持久化并存入 `DIR`，不受变量值的影响。此处会展开 `~`：

  ```bash
  melo --memory-dir ~/melo-memory "What is 6*7?"
  ```

- 网关使用启动时的变量。其所有会话都保存到同一组文件中。
- 历史记录将模型的推理保留 30 天；参见[模型调用](../../reference/memory.md#model-calls)（英文）。`LOCALMELO_THINKING_RETENTION_DAYS` 可为 `history.db` 和会话存储的 `sessions.db` 设置其他保留天数，例如 `7` 或 `0.5`（参见[会话](../../reference/memory.md#sessions)（英文））。如果值不是大于 0 的数字，`melo` 会在运行任何操作之前停止；参见[用法错误](../../reference/cli.md#usage-errors)（英文）。
- 在后续进程中召回长期记忆需要启用持久化、开启记忆，并且已有存储的条目。在 `legacy` 中，持久化长期存储还需要嵌入后端。显式推理模式无需嵌入后端即可持久化纯文本条目，并通过 BM25 找到它们。在显式模式下，必须启用召回并包含 `long` 类型；`drop` 和 `replay` 默认关闭 `auto` 召回。

<a id="turn-long-term-memory-off"></a>

## 关闭长期记忆

`--memory off` 或 Python 中的 `AgentOptions(memory=False)` 可在一次运行中关闭长期记忆：不从中召回内容，也不向其写入内容，反思也不会被提升为长期记忆。工作记忆和历史记录照常工作。`embedding_backend = "none"` 禁用的是嵌入后端，而不是记忆本身；显式推理模式仍可存储和召回纯文本条目。设置嵌入模型的方法见[设置嵌入模型](embedding.md)。

<a id="inspect-memory"></a>

## 查看记忆

这些文件是 SQLite 数据库。使用 `sqlite3` 命令行工具读取，并且只使用 `SELECT` 查询：

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

`melo-procedures export` 将全部程序性记忆以 JSON 行形式输出；参见 [melo-procedures](../../reference/procedures.md#export-and-forget)（英文）。

将 `<task_id>` 替换为任务列表中的 ID，或 JSON 行中的 `task_id`。

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

`agent.hippo` 是对智能体已打开存储的同一封装接口，因此运行中的脚本也可以通过它调用 `forget`；上面的脚本即使在智能体没有嵌入提供方时也会打开这两个文件。`SqliteHistory` 和 `SqliteLongTerm` 本身也有 `forget(task_id)` 和 `forget_all()`，内存中的 `History` 和 `LongTerm` 同样如此。

- **`forget(task_id)` 删除的内容。** 历史记录的 `tasks`、`steps` 和 `llm_calls` 表中属于该任务的行。长期记忆中则删除元数据指明该任务的条目：反思的元数据有 `"task_id"`，各步摘要只有 `"step_id"`，因此按任务遗忘会从长期记忆中删除该任务的反思，但保留其各步摘要。`forget_all()` 删除所有长期记忆条目以及历史记录中这三张表的所有行。
- **不涉及的内容。** 工作记忆（当前对话）、程序性记忆（由 `melo-procedures forget` 清除）以及会话存储（`sessions.db`；网页工作区的聊天记录位于 `~/.cache/localmelo/playground/`）。
- **实现方式。** 在存储的写锁下执行一次 `BEGIN IMMEDIATE` 事务。删除操作会覆盖释放的空间（`PRAGMA secure_delete=ON`），随后检查点会截断 WAL，因此遗忘的文本不会留在数据库文件或其 `-wal` 文件中。如果另一个连接阻止 WAL 截断，检查点会跳过，并在日志中记录 WARNING；之后的检查点会将其清除。localmelo 运行时可以安全地遗忘，但不能安全地删除文件。

要删除全部内容，包括 API 不涉及的内容，请停止所有 localmelo 进程并删除文件：

1. 停止所有 localmelo 进程：直接运行、`melo --serve` 和后台服务。使用 `melo --daemon uninstall` 停止服务；单纯使用 `kill` 后 launchd 会重新启动它。
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

   # The web workspace's memory
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

第二条命令会留下 `content` 中的推理：Ollama 回复开头的 `<think>` 副本（保留期机制之后不再删除该副本），以及没有推理解析器的服务器回复中的内嵌块。要删除所有存储的推理，请使用第一条命令。

如果设置了 `LOCALMELO_MEMORY_DIR` 或使用了 `--memory-dir`，请改为删除该目录中的文件。

> [!WARNING]
> localmelo 运行时删除文件可能导致文件损坏。请先停止 localmelo。

<a id="privacy"></a>

## 隐私

- 这些文件没有加密。`history.db` 保存你的任务、模型文本、每次工具调用及其参数以及每个工具结果，这些内容均经过[模型调用](../../reference/memory.md#model-calls)（英文）所述的脱敏处理。它还保存每次调用中经过脱敏的模型推理，默认保留 30 天；参见[模型调用](../../reference/memory.md#model-calls)（英文）。`long_term.db` 保存经过脱敏的各步摘要和反思及其嵌入向量。`procedures.db` 保存每个任务经规范化的查询、运行目录以及工具调用的标签：文件名和程序名，不保存其内容或输出。查询和标签经过最基本的脱敏：先处理[模型调用](../../reference/memory.md#model-calls)（英文）所述的密钥格式、bearer token 和一次性验证码，再处理 URL 中的密码（`postgres://app:<redacted>@host`，保留用户名），以及 NAME 包含 `key`、`token`、`secret`、`password`、`passwd` 或 `pwd` 时 `NAME=value` 或 `NAME: value` 中的值（`API_KEY=<redacted>`）。其他内容均原样保留，例如主机名、路径和其他值。
- 工具返回的图像只存储引用：MIME 类型、SHA-256 摘要和大小，绝不存储图像本身。
- 交互式界面将每个轮次及其工具调用和输出保存在 `~/.cache/localmelo/cli/sessions.db`，将你的消息保存在 `~/.cache/localmelo/cli_history`（两者均为 `0600`，明文，未加密）；`~/.cache/localmelo/logs/cli.log` 记录请求 URL 和错误。
- 这些文件的私密程度取决于你的主目录。删除方法见[重置记忆](#reset-memory)。Python API 的 `forget` 会覆盖所删除的单元并截断 WAL，因此文本不会留在数据库文件的空闲页或 `history.db-wal`/`long_term.db-wal` 中。
- 网关的网页工作区始终将聊天的历史记录和程序性记忆保存在 `~/.cache/localmelo/playground/`；选择了嵌入模型时，也会在那里保存长期记忆。
