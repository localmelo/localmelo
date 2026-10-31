<a id="scheduler-policy-core-and-simulator"></a>

# 调度器：策略核心与模拟器

localmelo 将把你的前台对话（通道）、worker 任务和聊天共享在同一个本地模型服务器上。调度器决定哪些模型调用发往服务器、按什么顺序，以及一个任务为什么在等待。本页面向维护者和 bench：`support/scheduler/` 下的代码是什么、如何运行一次模拟，以及如何导入 bench 轨迹。

> [!NOTE]
> 网页工作区的聊天经过调度门：网关用[调度门](../guides/scheduler.md#the-gate-on-chats-s2)包装每个聊天的对话和嵌入提供方。`[scheduler] policy`
> 默认为 `off`，只做记录；`fcfs` 和 `class` 是两个会强制执行的模式。直接运行和终端命令行不经过门。

<a id="task-command"></a>

## 任务命令

`melo-task` 是一个独立的网关客户端；它从不启动网关或模型服务器。它连接到配置的网关或显式的回环 `--gateway` URL，使用私有的网关 bearer 文件，并拒绝重定向和环境代理。请先启动网关。缺失或不安全的令牌、无法到达的网关或被拒绝的请求都是错误，而不是回退到未认证请求的理由。令牌不接受作为参数传入。显式的 `--gateway` 会绕过网关配置加载。网关接受的环回名称和数字写法无需 DNS 查询即可使用。

- `new TEXT`、`new --brief FILE` 或 `new --stdin` 提交一个目标；可选的 `--title`、`--workspace`、`--level P0` 到 `P5` 和 `--request-key` 显式设置字段。文件和标准输入是任务内容，不是可信的紧急程度信号。输入是非空 UTF-8，最多 1 MiB 和 100,000 个字符。生成的请求键允许安全重试；在网络错误不确定时，它会被打印到标准错误。重试时请保持相同的文本、选项和键。`new` 报告的是准入，而不是完成。
- `list [--state STATE] [--level Pn]` 列出匹配的任务（而不是它们的准入顺序），以表头行标出 ID、级别、状态和标题列的表格呈现；除标题外每一列都补齐，使各行对齐。`show ID` 查看一个。等待工具批准的挂起任务会打印 `Waiting for your approval: <tool> <target> (answer it in the web console, Tasks)`（在 `list` 中位于该行下方，在 `show` 中代替 `Waiting:` 行），并显示 `Reason: approval`；没有用于回答它的 `melo-task` 子命令。`pause ID`、`resume ID [--level Pn]`、`stop ID` 和 `level ID Pn|auto` 控制任务。不支持 `first` 和截止时间。暂停不会中断正在运行的工具。`stop ID` 输出任务的状态；任务尚未变为 `cancelled` 时还会加上 `stop requested; the task stops when its worker ends`，因为网关会在 worker 结束之前就接受对仍有 worker 的任务的停止请求。
- `follow ID [--after EVENT_ID] [--wait]` 流式读取已存储的会话事件，并重新检查任务状态。默认每个事件打印一行可读文本：本地时间、事件类型和简短摘要，绝不是原始 JSON。它使用持久的事件 ID 重连，而不是运行内部的序号。最终状态是一份快照；只有在这次 follow 重连过，或 `--after` 跳过了更早的事件时，最终一行才附带事件完整性的提示。保留期和收尾阶段的竞态仍然意味着，安静的 follow 不是完整回放的证明。反复断开连接或无效事件是错误；Ctrl-C 以 130 退出，但不停止任务。不带 `--wait` 时，观察到最终状态以 0 退出；带 `--wait` 时，done 退出 0、failed 退出 1、cancelled 或 expired 退出 2。其他成功的操作退出 0；用法错误退出 2，网络、认证或输出失败退出 1。
- `--json` 为每个非流式命令输出一个带版本的 JSON 对象，follow 则输出 NDJSON；follow 的 NDJSON 记录不因人类可读格式而改变。follow 的诊断信息和参数解析错误进入标准错误，而不是 JSON 标准输出。检查最终记录和进程退出码；仅有 EOF 不是成功。错误详情和最终输出可能包含敏感的任务数据；分享之前请先审阅输出。这不是直接 worker 的结果 schema。

<a id="what-is-here"></a>

## 这里有什么

- **策略核心**（`localmelo.support.scheduler`）。`decide()` 是内存中 `State` 的纯函数：机器门、带老化（aging）和继承的级别、通道的预留、后台上限、P2 和 P3 的信用下限、停放和让出标记，以及每个等待任务的一条原因。`dispatch()` 先准入通道，如果 `decide()` 抛出异常则回退到先来先服务顺序。`advance()` 和 `apply_dispatch()` 保存所有随时间发生的变化，因此没有别处保存状态。
- **模拟器**（`localmelo.support.scheduler.sim`）。`run()` 在一个模拟服务器上通过同样的函数重放一份工作负载：槽位、它自己的先来先服务队列、按 Θ(n) = n^0.4 共享的吞吐量，以及一个前缀缓存。`make_workload()` 构造合成工作负载 W1（一天）、W2（一次突发）、W2c（聊天）、W3（截止时间紧迫）和 W5（长尾）；`compute_metrics()` 为一次运行打分；`run_sweep()` 和 `compare()` 应用决策规则。
- **导入器**。`import_calls()` 读取 bench 代理的 `calls.jsonl` 文件，`import_history()` 读取一次运行的 `history.db`，用于模板和工具时长。`write_trace()` 和 `read_trace()` 以同一种 JSONL 格式保存轨迹，门的调用日志之后也会写这种格式。

<a id="arms-and-mode-off"></a>

## 模式与 off 模式

每个模式是一组 `Params` 预设（`params_for_arm()`），每个都在前一个的基础上增加一类机制，因此比较相邻模式就能把它单独隔离出来。

| 模式 | 任务调用的顺序 | 下限 | 让出标记与常驻槽位占用 |
|---|---|---|---|
| `off` | 不设门：每次调用都直接进入服务器的队列 | — | — |
| `fifo` | 按提交时间排成一个队列；忽略类别 | 关 | 关 |
| `class` | 严格的 P0 到 P4 类别；每类内先来先服务 | 关 | 开 |
| `las2`（默认） | 严格类别；P1 按截止时间，P2 和 P3 按最小已获服务，P4 最旧的优先 | 开 | 开 |

每个模式都保留通道的快速路径、预留、后台上限、机器门和停放标记。`off` 模式仍然记录计费和级别，因此它的指标可以与带门的模式比较；没有门时，没有任何代码为通道作答。

<a id="tasks-workers-and-the-scheduler-store-s3a"></a>

## 任务、worker 与调度器存储（S3a）

网关的 `sessions.db` 在会话表旁边有一个 `scheduler` 组件。`support/gateway/task_store.py` 在一个 `BEGIN IMMEDIATE` 事务中创建版本 4，包含 `tasks`、`task_deps`、`priority_events`、`questions`、`sched_rules`、`sched_log`、`idle_windows`、`park_priority` 和恢复存储，并在 `tasks(state, level_eff)`、`priority_events(task_id, ts)` 和 `sched_log(ts)` 上建立索引。`TaskStore.open()` 还会原子地迁移版本 1、2 和 3，并把组件版本记录在 `schema_components` 中。`PRAGMA user_version` 仍属于会话 schema，终端 `melo` 从不创建该组件或读取它的表；比 4 更新的组件会被拒绝。控制台在打开存储器之后调用 `TaskStore.open()`。网关取得会话文件的所有者锁：第二个网关进程会被拒绝，而不是并发迁移或写入同一个文件。

版本 3 增加了休眠的恢复来源信息、持久的操作回执和进程身份记录。恢复的准备和提交都推进共享的停放代次；被中断的恢复会把待处理的暂停保留到显式恢复或出现最终结果为止。缺失的来源信息和缺失的进程记录都表示未知，而不是任务从未运行或其进程已经消失的证明。内部的旧版收养接缝会记录显式的孤儿风险确认，但它本身不排队也不启动任务。版本 4 增加了每次执行的捕获状态、进程身份检查和带门的子进程原语。worker 在释放子进程之前通过那道门记录一次启动的进程组，并在第一个工具提供方启动之前（MCP 提供方可以自己启动一个服务器）以及在每个任意代码工具的入口标记捕获未完成；存储中断时会重试该标记最多 10 秒，然后拒绝那一次调用。不受支持的平台，或身份/记录失败，只有在标记提交之后才释放子进程。目前还没有任何东西读取这些证据，因此 Resume 仍然被栅栏挡住；进程验证、受控的重启调度和面向用户的确认仍是后续工作。

任务记录的 id 就是它 `kind="task"` 会话的 id。`create_task()` 在同一个事务中写入会话、任务、它的第一条 `priority_events` 记录和一条 `sched_log` 记录，因此崩溃不会留下其中任何一项。请把已经验证过的真实 `workspace` 传给 `create_task()`；它保存在会话上，而原始请求留在 `tasks.spec` 中。当前优先级在 `tasks.level` 中，而不是 `tasks.spec` 中创建时的值。`UNIQUE(origin, request_key)` 让重复的创建具有幂等性：相同的负载返回已存储的记录，并带 `repeated=True`；不同的负载抛出 `TaskConflictError`，任务路由把它映射为 409。`set_level()` 把实际的级别变化记录在 `priority_events` 中，并写入 `sched_log`；未变化的级别不会增加优先级标签。用户的 P0 是唯一的，因此第二个用户 P0 会在同一事务中把先前那个退回 P1，且不带 `user_rank`，无论任务来源如何。已完成的任务不能再调整优先级。`set_priority()` 也接受 `auto`（计算出的 P2）和 `first`（保留级别，设置一个让较早的第一选择排在更前面的排名）。`apply_retention()` 清除早于会话存储思考保留期（默认 30 天）的 `sched_log` 和 `idle_windows`。任务存储把它注册为会话存储清扫的一个钩子，因此网关每天的 `apply_retention()` 会清理两者，`TaskStore.open()` 也会在停机之后清理它们。任务、`priority_events`、`questions` 和规则不会被清理。

`support/gateway/task_policy.py` 是代码侧的边界，绝不是模型的 JSON：

- `TaskSpec.from_payload()` 规范化一个不可信的请求（接受 `prompt` 作为 `objective` 的别名），按计算出的 P2 处理，忽略不可信的 `level`，并拒绝负载提供的 `level_source`。推迟的可选字段（包括 `inputs`、`deadline` 和 `budget_s`）会被拒绝，而不是被悄悄丢弃；任务路由也会拒绝 `deadline`。在授权操作者之后，`TaskSpec.from_user_payload()` 固定用户显式选择的 P0–P5；没有显式级别时它使用计算出的 P2。适配器本身不对调用方进行认证或授权。
- `validate_task_workspace()` 解析真实的文件夹。配置的根必须是绝对路径或 `~/`，且不能解析为 `/` 或主文件夹。用户的任务可以使用 `~` 内任何严格位于其下、不含点分量的现有文件夹，且不能在 `~/Library` 之下。对其他来源，显式指定的文件夹必须严格位于配置的根之内；省略时，传入可信的 `task_id` 以创建该根的一个非符号链接子目录。未配置时，根默认为 `~/localmelo-workspace`。
- `worker_tool_allowlist()` 返回网关的内置工具（`file_read`、`file_write`、`python_exec`；绝不返回 `shell_exec`），并且只对 `user` 和 `user_rule` 返回记住的审批、未标记的 MCP 工具和会话授权。`front`、`harness`、`agent`、`channel` 和 `sleep` 都得不到这些，因此每个标记过的调用都需要一次新的审批。

`ChatGate.register_task()` 给 worker 一个任务生命周期。它的调用以 `task_id` 和 `session_kind="task"` 提交，绝不占用通道，一次调用的结束也不会结束 worker。运行器用 `done`、`failed`、`cancelled` 或 `expired` 调用 `finish_task()`。它会取消停放的调用，但保留已准入的调用，直到它们的提供方结束；终止任务的新调用会被拒绝并记录。结束一个活动的任务必须走 `finish_task()`；重复调用不会改变它的最终阶段。`update_task()` 携带调用方拥有的字段，例如非最终阶段和等待原因，但不携带级别或来源。重新注册现有任务会保留它当前的优先级，只忘记它则会保留它的身份；`set_task_level()` 处理芯片上的 `P0`-`P5`、`auto` 和 `first`，`held(task_id)` 像对聊天那样报告门的原因。任务路由通过存储写入优先级更改。

网关有[任务路由](../../reference/api-v2.md#scheduler-tasks-s3a2)（英文）和进程内 worker。新的用户任务按有效优先级、排名和创建时间排队，最多四个常驻 worker；成功停放的 worker 会释放它的槽位。每个 worker 都使用 `mode="gateway"`、`memory=False` 和控制台共享的同一个门来处理 LLM 和嵌入调用。Stop 会取消运行器并结束门任务。保留期清扫会丢弃任务记录已过保留期截止时间的最终门身份，尽管这些记录仍可供查看；只忘记它并不会丢弃身份。

记录在网关重启后仍然存在。启动时保留已完成的任务和已证明从未启动过的排队或停放任务，并中断情况不确定的任务而不重放它们。已经启动过的停放任务保留其记录的工作卡片；没有可用证据的不确定任务保持没有卡片。与执行关联、已提交的运行结果可以不重新运行就得到调和。待处理的暂停意图及其优先级快照在中断后仍然保留。历史不确定的恢复任务被挡在恢复和优先级更改之外；Stop 仍然可用。已证明从未启动过、可运行的排队任务可以在持久的启动栅栏和工作区验证之后启动；每次创建都在自己的事务里、在任何启动之前写入这份证明，因此普通的任务也适用。已提交、但其回合从未开始的恢复任务在重启后会被重新准入，不需要再次审阅；无论网关退出时该任务仍在排队，还是已经进入运行状态。孤儿验证和知情恢复是后续工作；缺失的进程记录不是进程已经停止的证明。重复的 `request_key` 只返回现有的任务。工作区有效性在创建时检查，并在运行或恢复时再次检查；变得无效的文件夹会让任务以 `invalid_workspace` 失败；恢复会返回 409 `invalid_transition` 并让记录保持中断状态。截止时间会被拒绝，服务、等待和工具的记账被推迟。暂停会把任务请求为 P5，并等待运行中的提供方和工具完成的安全点。被扣住的规划、反思或未送达的审批可以停放并释放其常驻 worker；未送达的审批会被撤回，并在恢复时重新询问。恢复使用一张由持久证据构建的卡片。在开始之前被暂停的任务不会在恢复之前启动 worker。恢复会还原先前的优先级，除非提供了新的级别；没有保存的暂停快照时（例如以 P5 创建或设置的任务），恢复回退到计算出的 P2。恢复一个未暂停的任务返回 409 `not_paused`。任务 PATCH 路由以 400 拒绝 `first`；它被推迟到 S6b。

<a id="s3b1a-foundations"></a>

## S3b.1a 基础

暂停会在与 P5 转换相同的事务中，把先前的级别、来源、原因和排名保存在 `park_priority` 中。这份快照在 `sched_log` 的保留期之后仍然存在，并在恢复时被消耗。迁移在保留的审计轨迹支持时为 v1 的停放 P5 记录重建快照；没有可用的暂停记录时没有快照，恢复回退到 P2。迁移会检查后续的 `task.level`、`task.first` 和 `task.resume` 审计条目；后续可运行的优先级也可能让任务不再处于暂停状态。显式的优先级更改可以更新或使快照失效。该记录还带有一个单调的 `generation` 和一个用于受保护转换的 `operation_id`；它在恢复后仍然保留，并随任务一起删除。`pause_pending` 记录在 worker 仍在运行时请求的暂停。在第一条身份记录存在之前，worker 协调器会挡住过期的安全点回调。

`ChatGate.try_park()` 是一个内部的同步原语，用于绑定到一轮的单个待处理、未准入的任务规划或反思调用。在同一个事件循环上，它把该调用挡在准入之外，并请求运行器以原因 `park` 取消那一轮。过期或已经在停止的轮次无法抢先。如果取消被拒绝或抛出异常，除非该调用已经结束，否则这道栅栏会被回滚。该原语本身不把停放与 worker 集成起来。

`SessionStore.component_event_transaction()` 和 `append_component_event()` 让网关的组件记录和预留的轮外会话事件能够原子地提交。任务存储的暂停会发出 `sched.pause_requested`；恢复和最终转换也会带着自己的身份发出已提交的事件。`ParkStore` 用同一个事务接缝处理内部的扣住、恢复和停放转换，记录调用方提供的扣住坐标，并发出 `sched.held`、`sched.resumed`、`sched.parked`、`sched.park_retrying` 或 `sched.interrupted`；它返回的记录包含当前的 `pause_pending` 标记。它不改变优先级。它的卡片构建器只使用任务规格和持久事件，对文本设限并脱敏，省略工具输出和文件路径以外的参数，并把完成视为未经验证。worker 集成会一直持有提供方或审批投递的栅栏，直到对应的持久转换提交，并通过它的卡片恢复停放的 worker。未保存的停放会在不打开栅栏的情况下重试；三次失败后保存一张中断卡片。如果持久化仍然不可用，worker 保持被扣住，恢复/停止返回 503，直到收尾成功。清理失败会隔离 worker、保留它的槽位，并在重试失败时报告中断；关闭时会为被隔离的清理给出一次有界的最终尝试。

<a id="controlled-resume-s3b2-pr3b-1"></a>

## 受控恢复（S3b.2 PR3b-1）

`POST /api/tasks/{task_id}/resume`（也可用 `/api/v2/tasks/{task_id}/resume`）接收 `{"operation_id": str, "confirmation_token"?: str}`。它是恢复启动恢复过的任务的唯一路由；对这些工作，`PATCH` 路由保持栅栏。被接受的转换是 202 `{"task", "operation_id"}`。一个 `operation_id` 只命名一次操作：用它重试会返回当前记录，无论负载是什么，绝不启动第二个回合，也绝不重新授权一次转换。

每次创建都在自己的创建事务里、在任何启动之前写入从未启动的证明，因此在首次启动前失败的任务是被证明从未启动，而不是未知。重启之前提交的恢复会预订它将使用的回合编号；启动时该任务重新排队，在没有对应回合时被重新准入（启动恢复在它进入运行状态之后把它中断的情形同样适用），而在该回合已开始之后重试会返回 409 `invalid_transition`，而不是重放该操作。

在行动之前，路由读取持久化的任务、卡片、来源、活动捕获和已记录的进程组，并验证工作区。分类期间它绝不启动提供方、工具或 worker。无法读取或不支持的检查是未知，而不是进程已停止的证明。

| 证据 | 动作 |
|---|---|
| 已完成任务、无效工作区、已启动任务缺少卡片、过期执行身份、未解决的恢复操作或未结算的 worker 清理 | 阻止恢复（`409 invalid_transition` 或 `409 recovery_unknown`）；Stop 和新建任务仍然可用 |
| 已证明从未启动过的排队或中断任务 | 通过持久启动栅栏入队，首次启动，不需要卡片或确认。排队中的从未启动创建由启动流程准入，而不是这个路由 |
| 已启动任务，有有效卡片、覆盖整个执行的受信任完整捕获，且每个已记录进程组都经过验证已消失 | 无需确认即入队 |
| 其他一切情况，包括缺失、不受信任、部分或畸形捕获，仍然存活的进程组，或无法检查存活性的进程组 | 只有在明确接受风险确认之后才入队 |

经过验证的启动（boot）变化（两个启动标识都能读取、格式良好且不同）只在其记录该进程组的主机上视为已消失：没有进程能在该机器的重启后存活。捕获在启动标识旁边保存持久的主机标识，因此在另一台主机上读到的启动变化、在保存主机标识之前写入的记录，或无法读取的主机标识都是未知，该路径需要确认。在 macOS 上主机标识是硬件 UUID（`IOPlatformUUID`），只以 SHA-256 摘要保存；在 Linux 上是 `/etc/machine-id`。无法读取的启动标识、权限错误、不受支持的平台或畸形记录同样是未知。只有捕获受信任且完整时，空的进程组集合才算数；仅缺少记录证明不了任何东西。

未经确认的 `confirmation_required`（`409`，带 `details.reason` 和 `details.warning`）在 `details.confirmation_token` 中携带一个一次性令牌，`details.expires_in_s` 为 300。令牌只存在于内存中，并绑定到预览所读取的任务、来源执行、证据身份和 CAS 代次。出示它即接受风险；被接受的转换会还原保存的优先级（暂停的 P5 任务没有快照时用计算出的 P2），清除暂停快照，把任务入队，写入恢复收据和一个 `sched.resume_confirmed` 事件。预览不写入任何东西，令牌会被 Stop 或重启作废。变化的证据身份、已使用的令牌或过期令牌返回 409 `stale_confirmation`。该操作的未知收据返回 409 `recovery_unknown`，只阻止恢复。

Stop 总是获胜且始终可用。它与恢复在共享的停放身份上串行，作废待处理的令牌，排队中的恢复回合会在注册 worker 或启动提供方、工具之前被取消。无论是自由恢复还是确认，都不证明旧进程已停止、不清理它，也不让效果只发生一次。

<a id="console-task-card"></a>

### 控制台任务卡片

控制台的 **Tasks** 视图（`#/tasks`、`#/tasks/<id>`）是上述路由的客户端：`GET /api/tasks`、`GET /api/tasks/{id}`、`POST /api/tasks/{id}/resume`，以及带 `{"state": "stop"}` 的 `PATCH /api/tasks/{id}`。它先不带令牌发送 Resume，只有收到 `confirmation_required` 时才打开确认对话框，然后用同一个 `operation_id` 带上 `details.confirmation_token` 重试。收到其他带状态的回答后，操作 id 会被丢弃，所以过期或被拒绝的尝试绝不会复用授权；网络失败后则保留它，所以重试不会启动第二个回合。已经开始但没有卡片的任务，或者收到 `missing_card` 拒绝时，Resume 会换成 **Start again as a new task**（用目标、标题、工作区和一个跨重试保留的请求键调用 `POST /api/tasks`）。恢复卡片用 `textContent` 渲染，从不作为 HTML。Stop 走任务 API，从不调用 `POST /api/turns/{id}/cancel`。

<a id="import-bench-traces"></a>

## 导入 bench 轨迹

让 `import_calls()` 指向一个目录：它会找到其中每一个 `calls.jsonl`。默认只保留 localmelo 的运行（`harnesses=("localmelo",)`），调用互相重叠的运行会被排除，丢弃的每一行都按原因计数。

一次聊天调用的类型是 localmelo 发送的 `X-Localmelo-Call-Kind` 头。代理只在带 `--log-bodies` 时保存请求头，保存在记录名的 `bodies.headers` 所指的旁文件中（相对于该记录所在的运行目录）；导入器只从那个文件中读取类型，没有它的记录保留端点/猜测。`history.db` 不用于读取类型：没有 `llm_calls` 记录指明它回答的是哪条代理记录，而检索请求绕过智能体的调用日志、根本不产生记录，因此不存在可靠的连接。`report.kind_sources` 按来源（`header`、`history`、`guess`；今天 `history` 为 0）统计保留的记录数，`report.kind_unknown` 统计导入器不认识的头部值，其中每一个都保留它的猜测。

```bash
uv run python -c '
import sys
from pathlib import Path
from localmelo.support.scheduler.sim import import_calls, make_workload
from localmelo.support.scheduler.sim.trace import templates_from_traces

traces, report = import_calls(Path(sys.argv[1]))
print(report.files, "files,", report.records_kept, "records kept,", report.tasks, "tasks")
print("dropped:", dict(report.dropped), "errors:", report.errors)
print("kinds:", dict(report.kind_sources), "unknown:", report.kind_unknown)
templates = templates_from_traces(traces)
print(len(make_workload("W2", 1, templates=templates).tasks), "W2 tasks on",
      len(templates), "imported templates")
' <runs-directory>
```

一次运行的工具时长来自它的 `history.db`（bench 运行把它保存在 `<run>/mem/history.db`），以只读方式打开，运行仍在写入时也可以。只读取工具时长：没有查询、思考或工具输出。这个池供给 `sim/workloads.py` 中的 `synthetic_templates(seed, tool_pool=...)`。

```bash
uv run python -c '
import sys
from pathlib import Path
from localmelo.support.scheduler.sim import import_history
from localmelo.support.scheduler.sim.importer import tool_time_pool

tasks, report = import_history(Path(sys.argv[1]), label="run-1")
print(report.tasks, "tasks; tool times", tool_time_pool(tasks))
' <run>/mem/history.db
```

<a id="results-are-not-validated"></a>

## 结果尚未验证

每一份 `SimResult` 和每一份 `compare()` 报告都写着 `validated: false`。服务器模型尚未验证：在门的一次真实窗口通过相对模拟器的 15% 检查之前，不会因为一次模拟而更改任何默认值。`compare()` 为每个候选模式给出一个裁决（`fails_guardrail`、`below_margin`、`insufficient_data` 或 `meets_margin`），从不给出“采纳”。

<a id="determinism"></a>

## 确定性

工作负载在 `run()` 之前就抽完所有随机数，来自用 sha256 播种的 `random.Random` 流，而调度器不读取时钟，因此相等的输入在任何进程、任何哈希种子下都会给出相同的决定日志（`log_sha256`）。输入在使用前排序，JSON 以排序的键写出，因此写出的轨迹或报告每次都是同一个文件。

<a id="known-gap-prefill-is-not-shared-evenly"></a>

## 已知缺口：预填充没有平均共享

探针 P-1（2026-10-26，oMLX 0.7.1rc1 上启用 MTP 的 oQ5）测得 n = 2、3、4、5、7 和 8 时 Θ(n)/Θ(1) 分别为 1.5、1.6、2.0、1.8、2.0 和 2.2，因此服务器模型使用 Θ(n) = n^0.4（最小二乘拟合；0.5 和 1.0 仍属候选）。解码是平均共享的，调用可以加入正在运行的批次。

预填充没有平均共享。排在它前面的每 3k token 后台预填充会让前台调用的首个 token 时间增加约 4.6 秒（n = 1、2、4 和 7 个后台调用时分别为 5.7、10.5、18.4 和 32.5 秒），而一个 16k token 的后台预填充会扣住它 5.3 秒（n = 1）或 10.5 秒（n = 2）。模拟器仍以同一个速率共享预填充和解码，因此它低估了通道调用在长后台预填充之后要等多久。把预填充建模为按到达顺序的队列属于门的阶段；在那里，在前台调用等待时扣住新的长预填充，正是这些数字所要求的保护。
