<a id="guide-sessions"></a>

# 指南：会话

<a id="run-a-conversation"></a>

## 运行对话

```python
import asyncio
import tempfile
from pathlib import Path

from localmelo import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore


def new_agent() -> Agent:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    return Agent(llm=llm, options=AgentOptions(workspace_root=".", record_answers=True))


async def main() -> None:
    store = SessionStore(Path(tempfile.mkdtemp()) / "sessions.db", writer="cli")
    agent = new_agent()
    await agent.start()
    runner = SessionRunner(
        agent, store, None, new_session=NewSession(workspace=agent.workspace_root)
    )
    try:
        result = await runner.run_turn("What is the capital of France?")
        print(result.status, result.result)
        session_id = runner.session_id
        await runner.close()
        await agent.close()

        # Later, or in another process: a fresh agent resumes the session.
        agent = new_agent()
        await agent.start()
        runner = SessionRunner(agent, store, session_id)
        print("resumed", await runner.resume(), "turn(s)")
        await runner.run_turn("How many people live there?")
        for turn in await store.list_turns(session_id):
            print(turn.turn, turn.status, turn.cause, turn.answer)
    finally:
        await runner.close()
        await agent.close()
        await store.aclose()


asyncio.run(main())
```

输出例如：

```text
completed Paris is the capital of France.
resumed 1 turn(s)
1 completed run Paris is the capital of France.
2 completed run About 2.1 million people live in Paris itself.
```

- **智能体**需要设置 `AgentOptions(record_answers=True)`，使每个回答留在其工作记忆中，还需要设置 `workspace_root`。对于不满足条件的智能体，`SessionRunner` 会抛出 `ValueError`。将智能体传给运行器之前，先调用 `await agent.start()`。
- **每个会话、每个智能体各用一个运行器。**运行器一次只运行一轮：有一轮正在运行时，`run_turn` 和 `start_turn` 会抛出 `SessionBusyError`。如果在 `runner.undo()` 或 `runner.resume()` 加载对话时开始一轮，它会等待加载完成。
- **`run_turn`** 在你的任务中运行这一轮并返回其 `RunResult`；即使运行器停止了这一轮（通过 `cancel()`、`close()` 或接管），也会返回结果，此时状态为 `cancelled`，若在运行开始前停止则没有计数。运行器外部取消你的任务时，取消会在这一轮结束后传播；如果取消恰好在该轮结束时发生，则会立即传播，而该轮自行结束。**`start_turn`** 在存储该轮记录后返回其 `TurnInfo`，这一轮继续在运行器的任务中运行；`await runner.wait()` 返回其 `RunResult`，与 `run_turn` 返回的结果相同。
- **`title`** 是传给 `run_turn` 和 `start_turn` 的用户输入文本。会话的第一轮以它设置标题；若未提供，则使用该轮文本，因此附带文件的消息也能获得简短标题。
- **`routine`** 是传给 `run_turn` 和 `start_turn` 的运行过程例程名称（[`run_task` 的 `routine=`](python.md#follow-a-run-with-events)）；仅在提供时才会传给运行。
- **进度。**以 `progress=True` 创建的运行器会向每轮运行请求 [`run.progress`](../../reference/events.md#runprogress)（英文）事件：这些事件与该轮一起存储，并像其他事件一样传给 `on_event`。该标志保留在运行器上，包括调用 `replace_agent` 后，因此之后每轮都会请求。不启用时，会话轮次和以前一样不报告此类事件；交互式界面始终请求进度事件。
- **监听器。**`turn.started` 和 `turn.finished` 是存储器自身的事件，而非运行事件；当一轮结束时为尚未存储回答的请求写入 `approval.resolved`，该事件也是如此。`store.add_listener(fn)` 在事务提交后对每个已存储事件调用 `fn`，包括上述事件。
- 使用 `None` 和 `new_session=NewSession(...)` 创建的运行器会在第一轮所在的同一事务中创建会话，因此第一轮被拒绝时不会留下空会话。该轮开始后，`runner.session_id` 即被设置。

`SessionRunner` 参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `agent`, `store`, `session_id` | 必填 | 智能体、存储器和会话 ID；也可将 ID 设为 `None` 并提供 `new_session`。 |
| `new_session` | `None` | 第一轮创建的会话：`NewSession(model="", workspace=None, meta=None, kind="chat")`。 |
| `on_event` | `None` | 每个轮次的运行事件及其 `approval.*` 事件提交后都会调用此函数，传入包含 ID 的 `StoredEvent`；存储器无法记录的事件，其 ID 为 `None`。函数可返回可等待对象，等待时间最多为 5 秒。 |
| `holder` | 当前进程 | 租约持有者，格式为 `"<host>:<pid>:<suffix>"`。 |
| `resume_mode` | `"pairs"` | [`resume()`](#resume) 加载对话的方式：`"pairs"` 或 `"window"`。 |
| `before_turn` | `None` | 获取该轮租约后、模型运行前执行的异步钩子。它接收一个无参数异步重载函数；如果另一个终端更改了对话，就等待该函数完成，不要保留它。命令行界面利用这个边界避免轮次在检查后、获取租约前结束所产生的竞态。 |
| `progress` | `False` | 向每轮运行请求 [`run.progress`](../../reference/events.md#runprogress)（英文）事件；该设置保留在运行器上，包括调用 `replace_agent` 后。 |

<a id="resume"></a>

## 恢复会话

`await runner.resume(max_turns=10)` 将已存储的对话加载到运行器的智能体中，并返回加载的轮数。运行一轮期间调用会抛出 `SessionBusyError`；会话已删除时会抛出 `UnknownSessionError`（`SessionStoreError` 的子类）。要求会话存在的存储操作也会抛出同一种类型的错误；`list_turns` 对未知 ID 仍返回空列表。`TurnInfo.typed_text` 单独提供通过 `title=` 传入的原始输入文本，与组合后的 `user_text` 分开；`/retry` 在重启后仍保留两者。`resume_mode` 选择加载方式：

- **`pairs`**，默认值：加载最近已完成且未撤销的轮次，每轮包含用户文本和存储的回答，最多 10 轮、32,000 个字符。异常终止的轮次显示为 `[no answer: the run ended with status crashed]`。
- **`window`**：加载最近一个带检查点的轮次结束时智能体持有的工作窗口，包括工具调用和结果，再以配对方式加载之后的轮次。没有可用检查点时（不存在、版本未知、属于已撤销轮次或被保留策略删除）则加载配对。

`runner.last_resume` 说明加载了什么：`mode`、`fell_back`（`window` 恢复时退回加载配对）、`turns`、`checkpoint_turns`（`turns` 中来自检查点的轮数）、`messages` 和 `chars`。无论设置哪种模式，每个以 `run.finished` 结束的轮次都会保存检查点，因此可以随时切换恢复模式。这不会改变智能体的推理模式：推理模式来自当前进程的配置或标志，绝不来自旧轮次的 `RunResult.config`。

配对不包含旧的思考内容。窗口检查点包含经过脱敏的思考内容，直到保留策略清除它：重放模式会在各自的重放限制内发送保留部分，而 `drop` 和 `compress` 会从请求副本中移除思考字段以及旧版内联 `<think>` 副本。存储的对话保持不变。`turns.note` 仍是存储的摘要，不会恢复为仅驻留内存的笔记存储；窗口可以保留模型原始的 `<note>` 文本，但不会重建笔记条目。

在基准测试使用 `scripts/resume_ab.py` 决定两种模式的取舍之前，`pairs` 仍是默认值。只有在 90 对中断状态上，`window` 的下一步成功率通过双侧精确 McNemar 检验以 p < 0.05 优于 `pairs`，其对先前回答中事实的召回率至多低 5 个百分点，且提示词 token 数的中位数至多为 `pairs` 的 1.5 倍时，才会将其设为默认值。默认值改变时会更新本页。
