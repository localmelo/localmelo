<a id="reference-procedures"></a>

# 参考：程序记忆

本页是 `melo-procedures` 的参考：回放已记录的运行、把运行转为片段，并导出或遗忘程序记忆。需要从命令行管理程序记忆时请阅读本页；概念见[程序记忆](../concepts/procedures.md)。

<a id="melo-procedures"></a>

## 命令 melo-procedures

`melo-procedures` 随 localmelo 一起安装，与 `melo` 相邻，例如在克隆中运行 `uv run melo-procedures --help`。它有四个命令。

<a id="replay-recorded-runs"></a>

### 回放已记录的运行

`replay` 读取导出格式的单次记录，并衡量这些规则。写同一个例程的五次运行：

```bash
mkdir procedures-demo
cd procedures-demo
python3 - <<'EOF'
import json
rows = [{"table": "routines", "routine_id": "r1", "key": "fix-tests", "name": None,
         "query_text": "run pytest and fix the failures", "place": "/home/me/proj",
         "similar_to": None, "created_at": 0}]
for i, (steps, status) in enumerate([(6, "completed"), (8, "completed"), (7, "completed"),
                                     (9, "completed"), (5, "budget_exhausted")], 1):
    segments = [
        {"cls": "R", "label": "Read test_parser.py", "target": "test_parser.py", "steps": [1, 1], "pass": False},
        {"cls": "W", "label": "Edit parser.py", "target": "parser.py", "steps": [2, steps - 2], "pass": False},
        {"cls": "T", "label": "pytest", "target": "pytest", "steps": [steps - 1, steps - 1], "pass": status == "completed"},
    ]
    rows.append({"table": "episodes", "task_id": f"t{i}", "routine_id": "r1",
                 "template_id": "r1:demo", "model": "qwen", "status": status, "steps": steps,
                 "first_pass_step": steps - 1 if status == "completed" else None,
                 "segments_json": json.dumps(segments), "created_at": i})
with open("episodes.jsonl", "w") as f:
    f.writelines(json.dumps(row) + "\n" for row in rows)
EOF
melo-procedures replay episodes.jsonl
```

输出（一行）：

```json
{"protocol": "online", "arm": "default", "params": {"window": 8, "stats": "completed", "drop": "conditional", "min_episodes": 2, "spread": 0.5, "cap": 0.95, "denominator": "dynamic", "demotion": 0.18, "demotion_window": 10, "demotion_min": 5}, "key_build": false, "runs": 5, "levels": {"0": 1, "1": 1, "2": 3}, "model_demoted": 0, "level2": {"runs": 3, "bar_error": 0.105, "cap_share": 0.0, "end_jump": 0.1181, "backward_moves": 0, "longer_rate": 0.3333, "longer_expected": 0.2611}, "held": {"runs": 0}, "separation": null, "level2_share": 0.75, "level2_by_prior_runs": {"2": {"runs": 1, "bar_error": 0.0179}, "3": {"runs": 1, "bar_error": 0.0972}, "4": {"runs": 1, "bar_error": 0.2}}, "phase_agreement": 1.0, "thresholds": {"bar_error": true, "separation": null, "backward_moves": true, "cap_share": true, "end_jump": true, "longer_rate": false, "phase_agreement": true}}
```

第一次运行还没有模板（级别 0），第二次有了一次过去的运行（级别 1），最后三次有进度条（级别 2），它们在各步骤上的平均误差是 0.105。

`replay FILE... [--protocol loo|first-k|online] [--arms] [--denominator D] [--spread S] [--drop D] [--stats S] [--key-build]`：

| 选项 | 含义 |
|---|---|
| `--protocol online` | 默认值。按时间顺序把每个单次记录送进存储和跟踪器，如同在线存储会看到它们一样。每个 arm 一行。 |
| `--protocol loo` | 对运行次数达到 3 次的模板，每次运行都用它的其他运行来预测，不带门。每个 denominator 一行。 |
| `--protocol first-k` | 模板的每次后续运行，只根据它的前 2 次运行来评判，并保持不变，带 spread 门。每个 denominator 一行。 |
| `--arms` | 另外运行每个只改一项默认值的 arm：denominator 的 `median`、`milestone` 和 `hybrid`，spread 的 0.4 和 0.6，drop 的 `off` 和 `unconditional`，以及统计的 `normal`。 |
| `--denominator` | 进度条的规则：[`dynamic`](../concepts/procedures.md#how-it-works)（默认值）、`median`（k ÷ m）、`milestone`（以第一次写入、测试和通过为锚点）或 `hybrid`（`dynamic`，在第一次通过后跳到 90%）。`loo` 和 `first-k` 不带它时报告全部四种。 |
| `--spread S` | 门的 spread（默认 0.5）。 |
| `--drop` | 探索性丢弃：`conditional`（默认）、`off` 或 `unconditional`。 |
| `--stats` | 统计使用的运行：`completed`（默认）或 `normal`，后者会加入以 `budget_exhausted`、`stuck` 或 `gate_stopped` 结束的运行。 |
| `--key-build` | 把每个单次记录的 localmelo 版本放进其模板的 key。 |

`--arms`、`--drop` 和 `--stats` 只适用于 `online`；与 `loo` 或 `first-k` 一起使用是用法错误。这两个协议从模板的每次运行取统计，无论其状态如何，也不做丢弃。

一行 `online` 输出包含：

| 字段 | 含义 |
|---|---|
| `levels` | 每个级别上的运行数，按查找时的结果计。 |
| `model_demoted` | 因模型被降级而停在级别 1 的运行数。 |
| `level2` | 在统计中至少有 2 次更早运行的运行里、处于级别 2 的那些：`bar_error`（对每次运行的各步骤，进度条与真实比例之间距离的均值）、`cap_share`（在这些运行上，处于 95% 的步骤所占份额的均值）、`end_jump`（在已完成的运行上为 1 − 最后一步的进度条；没有已完成运行时为 `null`）、`backward_moves`、`longer_rate`（超出自己窗口的运行比例）和 `longer_expected`（1 ÷ (window + 1) 的均值）。 |
| `held` | 对被门拦下的运行做同样的计算，并给出它们本会有的进度条。 |
| `separation` | `held.bar_error` − `level2.bar_error`；两者不齐时为 `null`。 |
| `level2_share` | 级别 2 的运行数 ÷ 级别 1 或 2 的运行数。用一个 key 回放一个族的全部任务时（`episodes --key <family>`），它是一个族会错误地显示进度条的运行比例；规格的第 10 节要求最多 10%。 |
| `level2_by_prior_runs` | 按模板统计中更早运行的数量划分的 `level2` 运行数和误差。 |
| `phase_agreement` | 阶段与其调用类别相符的步骤所占份额。 |
| `thresholds` | 每个阈值是否满足：进度条误差 ≤ 0.15、separation ≥ 0.05、没有后退、最多 5% 的步骤停在 95%、结束跳变 ≤ 0.30、超出比例 ≤ 期望 + 0.05、阶段一致 ≥ 0.9；无法衡量时为 `null`。 |

一行 `loo` 输出为所有运行给出 `level2` 字段；一行 `first-k` 输出包含 `shown`、`held` 和 `separation`。

<a id="turn-a-run-into-episodes"></a>

### 把一次运行变成单次记录

`episodes --history DB [--model M] [--key K] [--place P] [--result-json FILE]` 把一次运行的 `history.db` 以导出格式的行打印出来：一行 `routines`，以及每个任务一行 `episodes`。`DB` 是该文件，或者是存放它的目录。

- `--result-json` 是该次运行的 `--output-format json` 行（文件的最后一行）。它给出任务、它的状态、步骤、尝试次数、时长和版本，以及作为其模板 key 的选项。
- `--model` 是该次运行的模型 id，历史记录并不保存它（默认：无）。`--key` 设置例程的 key；默认值是带 `--place` 的标准化查询（默认：空）。
- 文件中有 `llm_calls` 表时，工具调用的步骤取自该表。没有它时，步骤只是近似：智能体只在一个步骤的第一条历史行上写下该步骤的文本，而一个步骤的各行几乎同时写入，因此有文本的行，或者在上一行之后超过 0.5 秒才写入的行（中间发生过一次规划调用），都开始一个新步骤。`--result-json` 随后给出真实的步骤总数。

把多次运行的行合并起来并回放它们：

```bash
for run in runs/*; do
  melo-procedures episodes --history "$run/memory" --model qwen \
    --result-json "$run/result.json"
done > episodes.jsonl
melo-procedures replay episodes.jsonl --arms
```

<a id="export-and-forget"></a>

### 导出并遗忘

```bash
melo-procedures export [--db FILE | --memory-dir DIR]
melo-procedures forget (--routine ID | --all) [--db FILE | --memory-dir DIR]
```

`export` 把 `procedures.db` 的每一行打印为一行 JSON：先是 `table`，然后按名称给出该行的各列，JSON 列作为文本。`forget` 删除一个例程及其单次记录和模板，或者删除每一行，打印 `{"deleted": <rows>}`，并重建文件（`VACUUM`），因此文件及其 `-wal` 都不会保留被删除内容的任何副本。

请在没有任何 localmelo 进程运行时执行它。两者都在 `--memory-dir` 中读取 `procedures.db`，否则在 `LOCALMELO_MEMORY_DIR` 中，否则在 `~/.cache/localmelo/memory` 中；两者都不会创建它。

<a id="exit-codes"></a>

### 退出码

| 代码 | 何时 |
|---|---|
| 0 | 完成。 |
| 2 | 用法错误，或者无法使用的输入：缺失或不可读的文件、格式错误的行（会给出文件和行号）、重复的任务。 |
| 130 | Ctrl+C。 |
