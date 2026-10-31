<a id="sessions"></a>

# 会话

会话是保存在 SQLite 中的一段对话：每一轮都从一条已存储的记录开始，比启动它的请求活得更久，并且恰好以一种状态和一个原因结束一次——运行它的进程停止或死掉时也是如此。之后一个新的智能体可以恢复这段对话。

`localmelo.support.sessions` 提供存储器、智能体轮次的运行器、供人使用的审批策略、事件流和只读的变更订阅。

> [!NOTE]
> 会话属于实验性的 Python API，在 v0.1.0 之前可能变化。交互式界面（单独的
> `melo`）把它的对话保存在这里，即 `~/.cache/localmelo/cli/sessions.db`；直接模式、`melo ""`
> 和网关的 `/v1` API 目前还不使用会话。网页控制台在网关自己的文件
> `~/.cache/localmelo/gateway/sessions.db` 中运行并读取它的会话，通过 [API v2](../../reference/api-v2.md)（英文）。

<a id="the-agent-a-session-runs-on"></a>

## 会话运行的智能体

`build_session_agent(...)` 创建并启动某个界面运行其轮次所用的智能体，并在启动失败时关闭它。它把会话的规则设在选项上：工作区、`record_answers=True`、`tool_exposure="all"`、600 秒的工具时限和 `replay_scope="window"`。

调用方的工具 `mode`（`interactive` 或 `gateway`）保持不变。它接受带 `base_url` 的 URL（以及 `model`，为 None 时用 `default`；嵌入只来自 `embedding_base_url` 和 `embedding_model`），或使用 `cfg` 的后端——在一份副本上，其对话模型为 `model`、`request_timeout_s` 为给定的值；`cfg` 本身不变。

当 `options=None` 时，构建器解析 `[agent]` 和 `[retrieval]` 的默认值，包括点名的 `tool_env` 变量；显式的 `AgentOptions` 对象优先。界面让显式旗标优先于这些默认值，控制台使用这些默认值，同时保持 `mode="gateway"`；它的推理模式默认为 `legacy`。

`reasoning_field` 覆盖配置的回放键，但对 Ollama 原生固定的 `thinking` 字段会被拒绝。`retrieval_extra_body` 只适用于 `legacy` 之外使用 `llm`/`auto` 的 `retrieve` 请求，且不会被保存。

`wrap_llm` 和 `wrap_embedding` 在智能体构建之前，以构建器创建的提供方调用；智能体用它们的返回值创建，因此它的 `Chat` 持有的提供方就是被包装的那个，每次模型调用都会经过包装器。非回放的 OpenAI 兼容提供方在包装之前会清空它的推理键；Ollama 保留其原生字段。

不带钩子的 legacy 保持原有的 `Agent(config=cfg)` 路径。显式模式在构建之前就拥有自己的提供方，因此被拒绝的模式/提供方组合会关闭它们，而不禁用 MCP。

没有创建的嵌入不会被包装。

```python
import asyncio
import tempfile

from localmelo import AgentOptions
from localmelo.support.sessions.agents import build_session_agent


def show_wrapped(llm):
    print("wrapped", type(llm).__name__)
    return llm


async def main() -> None:
    agent = await build_session_agent(
        cfg=None,
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3:8b",
        workspace=tempfile.mkdtemp(),
        options=AgentOptions(memory=False),
        wrap_llm=show_wrapped,
    )
    try:
        print(agent.options.tool_timeout_s)
    finally:
        await agent.close()


asyncio.run(main())
```

它打印：

```text
wrapped OpenAICompatLLM
600.0
```

<a id="the-two-files"></a>

## 两个文件

一种存储器实现服务于两个文件。每个界面只打开自己的那个：

| | 网关的文件 | 命令行界面的文件 |
|---|---|---|
| 打开方式 | `SessionStore(writer="gateway")` | `SessionStore(writer="cli")` |
| 默认路径 | `~/.cache/localmelo/gateway/sessions.db` | `~/.cache/localmelo/cli/sessions.db` |
| 写入者 | 一个持有所有者锁的网关进程 | 用户的每个终端 `melo`；每一轮运行都有一份租约，防止其中两个同时运行同一个会话的轮次 |
| 打开时发现的运行中轮次 | 在 `open()` 返回之前全部由恢复结束 | 在其持有者的进程消失后结束；持有者在另一台主机上时，则在其租约过期后结束 |
| `PRAGMA synchronous` | `FULL`：操作系统崩溃不会丢失任何已提交的内容 | `NORMAL` |
| 事件流（`SessionHub`） | 有 | 无 |

worker 会话也使用命令行界面的文件：每个直接任务默认都会保存一个（`--no-save-session` 或 `[sessions] record_direct = false` 可退出），dsh 日志可以按内容哈希导入。它们只有一轮，并被排除在交互式聊天选择之外。

网关可以通过只读连接读取这个文件中的 worker 记录；它不会恢复、修改或跟随它们。读取期间 SQLite 可能创建 `-wal` 和 `-shm` 记账附属文件。

脱敏限制见 [worker 历史记录](../../reference/cli.md#worker-histories)（英文）。

- **写入者标记。** 新文件会记录它的写入者。打开另一个写入者的文件会抛出 `SessionStoreError`，例如 `<path> belongs to the gateway (writer 'gateway'); open it with writer='gateway' or use the default path`；标记缺失或未知的文件抛出 `<path> has no valid writer marker`；不是会话文件的文件（例如 `history.db`）抛出 `<path> is not a localmelo sessions file`。存储器在写入任何内容之前先在只读连接上读取这些信息，因此被拒绝的文件保持原样、逐字节不变；在 WAL 模式下，SQLite 仍可能在文件旁创建它的 `-wal` 和 `-shm` 文件。schema 更新的文件抛出 `<path> has schema version <n>; this localmelo reads version 3`。较旧的存储器会原地迁移：版本 1 添加可为空的 `turns.attachments` 元数据，版本 1 和 2 添加 `turns.origin`（本人开始的每一轮为 `''`；对话回答任务问题的内部轮次为 `ask_owner`，即 `TurnInfo.origin`，它不给会话起标题）。
- **所有者锁。** 网关的存储器在其整个生命周期内持有 `<path>.owner` 上的锁。同一文件上的第二个网关会得到 `StoreOwnedError`：`Another melo gateway (pid <n>) is using <path>. Stop it to use the console here.` 进程以任何方式结束时系统都会释放该锁，因此崩溃后下一个网关可以立即启动。文件必须位于本地磁盘上。
- **私有文件。** 缺失的目录以 `0700` 权限创建，文件以 `0600` 权限创建。写入者自己的目录（`gateway/` 或 `cli/`）在权限更宽松时会被设为 `0700`；你自己选择的目录保持它的权限，其他用户可读时会有一条 WARNING。
- **打开。** 存储器在第一次被等待的调用时打开，或者用 `await store.open()` 打开。每次打开失败都会抛出 `SessionStoreError`（或 `StoreOwnedError`），并且可以重试：文件在写入任何内容之前就被拒绝，打开的每一步写入各自是一个事务，失败（例如磁盘已满）会回滚该事务；它之前的步骤（例如已恢复的轮次）保持完成。在 `await store.aclose()` 之后，每次调用都会抛出 `SessionStoreError("The session store is closed")`，仍在进行中的打开也一样。
- **保留期。** 打开存储器还会清除超过 30 天的思考、`<think>` 块、窗口检查点和已撤回的消息；轮次和事件保留。`LOCALMELO_THINKING_RETENTION_DAYS` 为 `sessions.db` 和 `history.db` 设置另一个天数，在创建存储器时读取：一个值让模型推理在每个文件中保留同样长的时间。无效值会抛出 `ValueError`，并指出该变量名。用 `thinking_retention_days=` 创建的存储器改用那个数。见[记忆](../../reference/memory.md#sessions)（英文）。

<a id="turns-statuses-and-causes"></a>

## 轮次、状态与原因

一轮的 `status` 在结束之前是 `running`，之后是一个[运行状态](run-results.md#statuses)。它的 `cause` 说明是什么结束了这一轮，运行期间为 `""`：

| `cause` | 何时 | 状态 |
|---|---|---|
| `run` | 运行自行结束，包括 `timeout` 或运行内部的崩溃（一个缺陷），以及运行结束后才到来的 Stop。 | 任意 |
| `user_stop` | 有人停止了它：`runner.cancel()`。 | `cancelled` |
| `closed` | 会话的运行器被关闭了：`runner.close()`，例如删除会话时。 | `cancelled` |
| `shutdown` | 进程被要求停止：`agent.begin_exit(...)` 之后的 `runner.close(cause="shutdown")`，或来自运行器外部的取消，例如 Ctrl+C 时 `asyncio.run` 发出的取消。 | `cancelled` |
| `crash` | 进程死掉；恢复过程记录了结束。 | `crashed`，或者存储的 `run.finished` 如此说明时为 `cancelled` |
| `takeover` | 另一个终端接管了一轮，其租约在另一台主机上已过期（仅限命令行界面的文件）。 | `cancelled` |
| `park` | 预留给调度器。 | `cancelled` |

- **停止。** `runner.cancel(reason, cause="user_stop", turn_id=None)` 停止正在运行的轮次，并返回它是否停止了一轮；带 `turn_id` 时只停止该 id 的轮次，因此迟到的 Stop 绝不会停止下一轮。`runner.close(reason="Session closed", cause="closed")` 停止它，并最多等待 12 秒。两种方式下运行都以 `cancelled` 结束，其 `RunResult` 的 `error` 是那个原因（见[停止运行](../guides/python.md#stop-a-run)）。
- **崩溃之后。** 进程死掉的轮次在磁盘上保持 `running`，直到恢复过程结束它。网关的存储在 `open()` 返回之前结束每一个这样的轮次。命令行界面的存储在打开时、在会话的下一轮或撤销开始时，以及在 `recover_stale_turns()` 中结束这样的轮次：一旦其持有者在本次主机上的进程消失，或者，对于另一台主机上的持有者，一旦它的租约过期。恢复过程撤回没人回答的审批请求，并用存储的 `run.finished` 的结果结束该轮（如果有的话）；否则，对于另一台主机已过期的租约，以 `cancelled` 结束，原因为 `takeover`、错误为 `The session was taken over by another process`，旧终端的运行也以同样的方式结束；再否则，以 `crashed` 结束，错误为 `The process running this turn stopped before it finished.`。
- **“Interrupted”是一个标签，不是状态。** 界面把 `shutdown` 和 `crash` 两个原因显示为“Interrupted”；状态始终是一个运行状态。
- **写入失败。** 当网关的存储无法记录一个运行中轮次的事件或结束，它会以 `SessionStoreError("The session store could not record a change; restart the gateway")` 拒绝新工作，直到它重启，读取照常工作；重启会结束该轮。对于命令行界面的文件，运行器会在 0.5、1 和 2 秒后重试结束该轮（关闭期间只重试一次），并在会话的下一轮或撤销之前再试一次；在此之前会话保持忙碌。

会话和轮次的存储方法：`create_session`、`get_session`、`find_sessions`、`list_sessions`（最新的在前；除非传入 `kinds`，否则只列出聊天）、`rename_session`、`update_session`、`delete_session`（有轮次运行时被拒绝）、`list_turns(session_id, limit=, before=)`、`list_events(session_id, after_id=, limit=, turns=)`、`undo_last_turn` 和 `get_window`。`runner.undo()` 撤销最新的已完成轮次并恢复；期间开始的轮次会等待恢复完成。

<a id="message-ids-and-submissions"></a>

## 消息 id 与提交

通过网络发送消息的客户端，可以在知道消息是否到达之前用一个 `client_request_id` 命名它。id 由 1 到 128 个字母、数字和 `. _ : -` 组成，以字母或数字开头，例如 `crypto.randomUUID()` 给出的那种。

带 id 时，存储器在一个事务中记录消息、它的轮次和 `turn.started`，因此丢失了响应的客户端可以询问这条消息后来怎样了：

```python
import asyncio
import tempfile
from pathlib import Path

from localmelo import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.support.sessions.submissions import SubmissionRepeated

KEY = "8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07"


async def main() -> None:
    store = SessionStore(Path(tempfile.mkdtemp()) / "sessions.db", writer="gateway")
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm, options=AgentOptions(workspace_root=".", record_answers=True))
    await agent.start()
    runner = SessionRunner(agent, store, None, new_session=NewSession())
    try:
        turn = await runner.start_turn("Summarize my notes", client_request_id=KEY)
        await runner.wait()
        try:  # the same message again, as a client retries
            await runner.start_turn("Summarize my notes", client_request_id=KEY)
        except SubmissionRepeated as repeated:
            print("already accepted as turn", repeated.turn.turn)
        submission = await store.get_submission(turn.session_id, KEY)
        print(submission.state, submission.status, submission.cause)
    finally:
        await runner.close()
        await agent.close()
        await store.aclose()


asyncio.run(main())
```

它打印：

```text
already accepted as turn 1
accepted completed run
```

| 状态 | `get_submission` 返回 | 含义 |
|---|---|---|
| unknown | `None` | 消息从未到达，也没有运行任何东西：用同一个 id 重试是安全的。 |
| `accepted` | 带轮次的 `SubmissionInfo` | 消息启动了那一轮；`turn_id`、`status` 和 `cause` 都是该轮的。 |
| `withdrawn` | 不带轮次的 `SubmissionInfo` | `withdraw_submission` 在消息被接受之前取消了它。 |

- **重复的提交不运行任何东西。** 对已接受的 id 和相同文本调用 `start_turn`（以及 `store.begin_turn`）会抛出 `SubmissionRepeated`，它的 `turn` 是已存储的轮次，该轮正在运行时也一样；文本不同时抛出 `SubmissionConflict`；id 已被撤回时抛出 `SubmissionWithdrawn`。这些异常来自 `localmelo.support.sessions.submissions`，并不是 `SessionStoreError`。`check_submission(info, text)` 对你读取的提交应用同样的规则。
- **`store.withdraw_submission(session_id, client_request_id, *, user_text)`** 取消一条未被接受的消息：它存储一次撤回，这次撤回不占用轮次编号、不写事件，并在此后拒绝该 id，重启之后也一样。对于已被接受的消息，它按原样返回提交；请用 `runner.cancel(turn_id=submission.turn_id)` 停止它的轮次。
- **作用域。** id 在每个会话内唯一。新会话的第一轮使用创建作用域：`get_submission(None, key)`，用会话自己的 id 也能找到它。
- `run_turn` 不接受 id。`get_submission` 对未知会话抛出 `SessionStoreError`，因此被删除的聊天不会被误认为丢失的消息。

<a id="approvals-for-a-person"></a>

## 供人使用的审批

`localmelo.support.sessions.approvals` 为会询问人的界面提供两个辅助工具：

- **`RememberingPolicy(ask)`**，一个审批策略：它通过 `ask(request)` 询问，并把“本会话内允许”（带 `remember=True` 的回答）保留为该工具名称的一项授权。`allowed_tools` 列出这些授权，`forget(tool_name=None)` 收回一项或全部。授权从不保存。
- **`ApprovalBroker(timeout_s=None)`** 保存等待来自别处（例如网页）的回答的请求：`asker(session_id)` 是一个询问函数，`pending(session_id)` 列出等待中的请求，`answer(request_id, decision)` 返回 `resolved`、`already_resolved` 或 `unknown`，`withdraw_all(reason=...)` 拒绝等待中的请求。

用 `runner.observe(ask)` 包装询问函数，例如 `RememberingPolicy(runner.observe(broker.asker(session_id)))`，并把该策略作为 `approval` 传给 `run_turn` 或 `start_turn`：会话的事件随后会显示 `approval.requested` 和 `approval.resolved`。

<a id="stream-a-sessions-events"></a>

## 流式读取会话事件

`SessionHub(store)` 流式读取网关存储的事件，它只接受这一个：

```python
from localmelo.support.sessions.hub import SessionHub

hub = SessionHub(store)  # a SessionStore(writer="gateway")
async for event in hub.subscribe(session_id, after_id=last_event_id):
    print(event.id, event.event["type"])
```

它回放 `after_id` 之后已存储的事件，然后随着提交发送新事件，不遗漏也不重复，对慢速客户端也是如此。事件 id 只增不减，跨重启也是如此，因此用最后一个 id 重连的客户端会得到它错过的事件，包括恢复过程写入的那些。

会话事件列在[事件](../../reference/events.md#session-events)（英文）中。

<a id="export-finished-turns"></a>

## 导出已完成的轮次

`SessionReader(path)` 以只读方式读取任一文件，写入者打开它时也可以，且从不写入。`changes()` 为每个已完成或已撤销的轮次返回一条记录，顺序与存储器记录它们时相同；每条记录的事件 id 是一个由你保存的游标：

```python
import asyncio
import io
import tempfile
from pathlib import Path

from localmelo import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions.reader import SessionReader, export_turns
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore


async def main() -> None:
    path = Path(tempfile.mkdtemp()) / "sessions.db"
    store = SessionStore(path, writer="cli")
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm, options=AgentOptions(workspace_root=".", record_answers=True))
    await agent.start()
    runner = SessionRunner(agent, store, None, new_session=NewSession())
    await runner.run_turn("What is the capital of France?")
    await runner.close()
    await agent.close()
    await store.aclose()

    reader = SessionReader(path)
    for change in await reader.changes(after_event_id=0):
        print(change.event_id, change.change, change.status, change.answer)
    await reader.aclose()

    out = io.StringIO()
    cursors = await export_turns([path], out)
    print(len(out.getvalue().splitlines()), "line(s); cursor", cursors[str(path)])


asyncio.run(main())
```

例如，它打印：

```text
7 finished completed Paris is the capital of France.
1 line(s); cursor 7
```

- `changes(after_event_id=0, limit=500, kinds=None, with_events=False)` 最多返回 `limit` 条记录（1 到 5,000）。每条 `TurnChange` 保存该轮当前的字段、其会话的类型、标题、工作区和模型，以及该轮的 `run_id`，即它在 `history.db` 中运行的 `task_id`。对话回答任务问题的轮次 `origin` 为 `ask_owner`，其 `user_text` 是任务的话，不是用户的；其他轮次以及 schema 2 文件中的轮次为 `""`。
- 之后被撤销的轮次会再次以 `undone` 记录出现，因此按顺序应用这些记录总是正确的。运行中的轮次在结束后出现。
- `export_turns(paths, out, after=None)` 为每条记录写一行 JSON，`{"kind": "localmelo.session_turn", "v": 1, ...}`，并返回每个路径的新游标；下次把这些游标作为 `after` 传回。
- 读取器不加锁、不运行恢复，并拒绝更新的 schema。schema 版本为 0 的文件（例如空文件）没有记录。

<a id="limits"></a>

## 限制

| 项目 | 限制 |
|---|---|
| 一轮的文本 | 1 到 100,000 个字符 |
| `client_request_id` | 1 到 128 个字符 |
| 标题 | 手工设置时 1 到 200 个字符；从文本设置时为 60 |
| 撤回 | 每个会话 1,000 条，新会话的第一轮为 1,000 条；没有超过保留期（默认 30 天）的 |
| 窗口检查点 | 每个会话一个，最多 2,000,000 个字符 |
| 轮次的租约 | 30 秒，每 10 秒刷新一次 |
| `list_events` 每页 | 1 到 1,000 个事件 |
