# Reference: Procedures

This page is the reference for `melo-procedures`, which replays recorded runs, turns runs into episodes, and exports or forgets procedural memory. Read it when you manage procedural memory from the command line; the concept is explained in [procedural memory](../concepts/procedures.md).

## melo-procedures

`melo-procedures` is installed with localmelo, next to `melo`, for example
as `uv run melo-procedures --help` inside the clone. It has four commands.

### Replay recorded runs

`replay` reads episodes in the export format and measures the rules. Write
five runs of one routine:

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

Output (one line):

```json
{"protocol": "online", "arm": "default", "params": {"window": 8, "stats": "completed", "drop": "conditional", "min_episodes": 2, "spread": 0.5, "cap": 0.95, "denominator": "dynamic", "demotion": 0.18, "demotion_window": 10, "demotion_min": 5}, "key_build": false, "runs": 5, "levels": {"0": 1, "1": 1, "2": 3}, "model_demoted": 0, "level2": {"runs": 3, "bar_error": 0.105, "cap_share": 0.0, "end_jump": 0.1181, "backward_moves": 0, "longer_rate": 0.3333, "longer_expected": 0.2611}, "held": {"runs": 0}, "separation": null, "level2_share": 0.75, "level2_by_prior_runs": {"2": {"runs": 1, "bar_error": 0.0179}, "3": {"runs": 1, "bar_error": 0.0972}, "4": {"runs": 1, "bar_error": 0.2}}, "phase_agreement": 1.0, "thresholds": {"bar_error": true, "separation": null, "backward_moves": true, "cap_share": true, "end_jump": true, "longer_rate": false, "phase_agreement": true}}
```

The first run had no template (level 0), the second one past run (level 1),
and the last three a bar (level 2), whose mean error over their steps was
0.105.

`replay FILE... [--protocol loo|first-k|online] [--arms] [--denominator D]
[--spread S] [--drop D] [--stats S] [--key-build]`:

| Option | Meaning |
|---|---|
| `--protocol online` | The default. Every episode in time order through the store and the tracker, as a live store would see them. One line per arm. |
| `--protocol loo` | Each run of a template with 3 or more runs, predicted from its other runs, with no gate. One line per denominator. |
| `--protocol first-k` | Every later run of a template, judged on its first 2 runs only, held fixed, with the spread gate. One line per denominator. |
| `--arms` | Also each arm that changes one default: the denominators `median`, `milestone` and `hybrid`, the spread 0.4 and 0.6, the drop `off` and `unconditional`, and the statistics `normal`. |
| `--denominator` | The bar's rule: [`dynamic`](../concepts/procedures.md#how-it-works) (the default), `median` (k ÷ m), `milestone` (anchored on the first write, test and pass) or `hybrid` (`dynamic`, jumping to 90% after the first pass). `loo` and `first-k` report all four without it. |
| `--spread S` | The gate's spread (default 0.5). |
| `--drop` | The exploration drop: `conditional` (default), `off` or `unconditional`. |
| `--stats` | The runs the statistics use: `completed` (default) or `normal`, which adds runs that ended `budget_exhausted`, `stuck` or `gate_stopped`. |
| `--key-build` | Put each episode's localmelo version in its template's key. |

`--arms`, `--drop` and `--stats` apply to `online` only; with `loo` or
`first-k` they are a usage error. Those two protocols take the statistics
from every run of a template, whatever its status, with no drop.

An `online` line holds:

| Field | Meaning |
|---|---|
| `levels` | Runs at each level, as their lookup found them. |
| `model_demoted` | Runs held at level 1 because their model was demoted. |
| `level2` | The runs at level 2 among those with at least 2 earlier runs in the statistics: `bar_error` (mean over each run's steps of the distance between the bar and the true fraction), `cap_share` (the mean over the runs of the share of their steps at 95%), `end_jump` (1 − the bar at the last step, over the completed runs; `null` without one), `backward_moves`, `longer_rate` (runs that outlasted their window) and `longer_expected` (the mean of 1 ÷ (window + 1)). |
| `held` | The same for the runs the gate held back, with the bar they would have had. |
| `separation` | `held.bar_error` − `level2.bar_error`; `null` without both. |
| `level2_share` | Level-2 runs ÷ runs at level 1 or 2. Replayed with one key for every task of a family (`episodes --key <family>`), it is the share of runs that a family would wrongly show a bar to; section 10 of the spec wants at most 10%. |
| `level2_by_prior_runs` | `level2` runs and error by the number of earlier runs in the template's statistics. |
| `phase_agreement` | The share of steps whose phase matches their calls' classes. |
| `thresholds` | Each threshold met or not: bar error ≤ 0.15, separation ≥ 0.05, no backward move, at most 5% of steps at 95%, end jump ≤ 0.30, longer rate ≤ expected + 0.05, phase agreement ≥ 0.9; `null` when not measurable. |

A `loo` line has the `level2` fields for all the runs; a `first-k` line has
`shown`, `held` and `separation`.

### Turn a run into episodes

`episodes --history DB [--model M] [--key K] [--place P] [--result-json
FILE]` prints a run's `history.db` as rows of the export format: a
`routines` row and an `episodes` row per task. `DB` is the file or the
directory that holds it.

- `--result-json` is the run's `--output-format json` line (the file's last
  line). It gives the task, its status, steps, attempts, duration and
  version, and the options that key its template.
- `--model` is the run's model id, which history does not hold (default:
  none). `--key` sets the routine's key; by default it is the normalized
  query with `--place` (default: empty).
- A tool call's step comes from the `llm_calls` table when the file has it.
  Without it, the step is an approximation: the agent writes a step's text
  on its first history row only and a step's rows within moments of each
  other, so a row with text, or one written more than 0.5 s after the row
  before it (a planning call came between), starts a new step.
  `--result-json` then gives the true step total.

Join the lines of several runs and replay them:

```bash
for run in runs/*; do
  melo-procedures episodes --history "$run/memory" --model qwen \
    --result-json "$run/result.json"
done > episodes.jsonl
melo-procedures replay episodes.jsonl --arms
```

### Export and forget

```bash
melo-procedures export [--db FILE | --memory-dir DIR]
melo-procedures forget (--routine ID | --all) [--db FILE | --memory-dir DIR]
```

`export` prints every row of a `procedures.db` as a JSON line: the
`table`, then the row's columns by name, JSON columns as text. `forget`
deletes one routine with its episodes and templates, or every row, prints
`{"deleted": <rows>}`, and rebuilds the file (`VACUUM`), so that neither it
nor its `-wal` keeps any copy of what was deleted.

Run it while no
localmelo process is running. Both read
`procedures.db` in `--memory-dir`, else in `LOCALMELO_MEMORY_DIR`, else in
`~/.cache/localmelo/memory`; neither creates it.

### Exit codes

| Code | When |
|---|---|
| 0 | Done. |
| 2 | A usage error, or an input it cannot use: a missing or unreadable file, a malformed line (named by file and line), a repeated task. |
| 130 | Ctrl+C. |
