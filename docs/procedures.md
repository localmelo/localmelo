# Procedural memory

Procedural memory learns the steps of tasks you run again and again, such as
a test suite you fix every day or a daily report, so that a progress display
can show how far a run is. It keeps what earlier runs did, never what the
model said: the classes of their tool calls, short labels, step counts and
times.

Every run writes it: when a run ends, through `melo`, the `melo ""` loop,
the gateway or the Python API, its record is added. With persistence on
(see [Memory](memory.md#keep-memory-between-runs)) that is `procedures.db`
in the memory directory; otherwise procedural memory stays in RAM for the
agent's life and nothing is written. To keep it off the disk, leave
persistence off. Nothing here calls a model, reaches a prompt or changes a
request.

What a run may show from it comes as [`run.progress`](events.md#runprogress)
events, to a Python caller that asks with `progress=True`; `melo` does not
show progress yet. Level 2, the bar, is off in this release: replayed on
recorded runs, the gate did not yet keep bad bars back by the margin it
must (0.05) on both test sets, so a run shows at most level 1 until a
later release turns level 2 on, once a set of repeated runs passes the
replay. `melo-procedures` replays
the rules on recorded runs, so that they are measured before a run shows
anything.

## What a run does

- **At its start**, only when a Python caller asked for progress, the run
  looks its routine up, once, for at most 1 second; without a template, or
  if the lookup fails, it shows facts only (level 0).
- **During the run**, it keeps a trace in RAM: each step, and the class and
  label of each tool call. It keeps no tool output; a test call's last 400
  characters are read for its summary and dropped.
- **At its end**, after its last event, it adds its routine (if new), its
  episode and, when it completed, its template's next version. This takes a
  few milliseconds and at most 2 seconds; a write that fails is skipped
  with one INFO line in the log. A run that is cancelled, or that ends
  after SIGTERM or SIGHUP, adds nothing.
- The episode records the level the gate gives, also when that is level 2
  and level 1 is shown, and, when the gate passed, the error the bar would
  have had: evidence for turning level 2 on. The gate is the one the
  run's start found when it asked for progress; otherwise, since such a
  run reads nothing at its start, the one its routine's template gives
  as the run ends.
- `run_task(..., routine="name")` in Python makes runs one routine, named,
  in any folder and whatever their wording.

## How it works

- **Routine.** A recurring task. A task given as a query is known by its
  normalized query and its *place*, the directory its tools work in: the
  workspace root, or the working directory without one. Normalizing applies
  NFKC, lowercases, collapses whitespace and masks dates and times as
  `<date>` and `<time>` (ISO dates, `10/26`, `10/26/2026`, `26.10.2026`,
  month names with a day or year, weekday names, `2026年`, `10月26日`,
  `星期一`, `周一`, and `HH:MM[:SS]`). Other numbers stay: "fix issue 123" and
  "fix issue 456" are two routines. A named routine is one routine in any
  folder.
- **Similar task.** A new routine in a folder that holds at least 5 others
  is compared, by BM25 over their normalized queries, with the folder's
  latest 1,000. A clear best match lends the run its named steps and past
  facts, at level 1: a hint, never a bar. The run still starts a routine
  of its own.
- **Episode.** One finished run of a routine: its status, steps, attempts,
  duration, and its *segments*, the runs of consecutive tool calls with the
  same class and target.
- **Template.** What runs of one routine on one model with one set of
  length-shaping options (`attempt_policy`, `reflection_gate`, `max_steps`,
  `steps_per_attempt`, `max_attempts`) have in common: step statistics and
  named steps. Every completed run appends a new version, computed from the
  last 8 completed runs.
- **Level.** What a display may show from a template:

  | Level | When | Shows |
  |---|---|---|
  | 0 | No template: a first run, another model or other options | Facts only: step, attempt, phase, tests |
  | 1 | One completed run; or the gate fails; or a similar task; or the model is demoted or has no id | Past runs' steps and times, and their named steps, without a percent |
  | 2 | The gate passes | A bar as well (off in this release: a run shows level 1) |

- **The gate.** Level 2 needs the routine's own key or name, a model id,
  at least 2 completed runs in the window, and a *spread* (longest − shortest
  steps ÷ median) of at most 0.5. A model whose bar was off by more than
  0.18 on average over its last 10 scored runs (at least 5) is demoted to
  level 1 until it does better. The first run of a template is exploration:
  once 2 later runs completed, it leaves the statistics if it completed and
  took more steps than all of them.
- **The bar.** After step k starts, with m the median of the window: k ÷ m
  while k < m; else k ÷ (k + the median remaining steps of the longer runs),
  or k ÷ (k + 1) when no run was longer. It never moves back, never passes
  95% while running, and is 100% only when a run completes. A run longer than
  every run in the window, a reflection that the window does not expect, or
  a stuck notice freezes it for the rest of the run. Only a bar can leave
  the template: at level 1 the run stays `running`.

### Classes and phases

Each tool call gets one class:

| Class | Calls |
|---|---|
| `R` | `file_read`; a `shell_exec` that starts with a reading program (`ls`, `cat`, `grep`, …); `python_exec` code that reads or prints |
| `W` | `file_write`; a `shell_exec` that writes (`>`, `mv`, `cp`, `mkdir`, `rm`, `sed -i`, `tee`, `touch`, …); `python_exec` code that writes a file |
| `T` | A call other than `file_read` and `file_write` whose command, code or arguments mention `pytest`, `unittest`, `_test.py`, `verify`, `check` or `assert `; this is checked before the `R` and `W` rules of `shell_exec` and `python_exec` |
| `X` | Everything else |

A `T` call *passes* when the last 400 characters of its output hold a pytest
or unittest summary with at least one pass and no failure or error. Its exit
status alone never makes a pass: agents pipe tests into `tail` or `grep`. A
call that failed, or whose output starts with `[exit code <n>]`, never
passes.

The phase is read from the classes: **Explore** before the first write,
**Write**, **Verify**, and **Finishing** after the first pass. A *cycle* is a
test after a write.

A call's label names it without its arguments' content: `Read parser.py`,
`Edit parser.py`, the program and its first subcommand (`pytest`,
`git status`, `npm test`), `Python`, or an MCP tool's name. Labels pass the
same secret redaction as model text and are cut to 40 characters.

## The store

`SqliteProcedures` keeps procedural memory in `procedures.db`, in the memory
directory, with mode 0600; `MemoryProcedures` keeps it in RAM. Both live in
`localmelo.melo.memory.procedures.store`. The file has three tables, in
schema version 1 (`PRAGMA user_version`):

| Table | One row per | Changes |
|---|---|---|
| `routines` | Routine: its key, name, normalized and redacted query (at most 2,000 characters), place, and the routine it was similar to | Never |
| `episodes` | Finished run: routine, template, model, build, match, level, status, steps, attempts, tool calls, duration, first pass, cycles, segments and shadow bar error | Never |
| `templates` | Version of a template: statistics, named steps, parameters, the runs it came from, its parent version, status, source | Only `status`: `active` → `superseded`, `candidate` → `active` or `rejected`; a rejected candidate's `eval_json` may record why |

Triggers in the file refuse any other change, and at most one version of a
template is `active`. A run's end writes in one `BEGIN IMMEDIATE`
transaction, so two processes that finish runs of one routine at once wait
for each other (up to 500 ms). Rows leave the file only through
`melo-procedures forget`.

A later sleep job can propose a version: it appends a `candidate` with its
replay on held-out runs (`heldout_task_ids`, `bar_error`,
`parent_bar_error`, `cap_share`, `longer_rate`, `reason`). It becomes active
only when its error is below its parent's, at most 5% of its steps sat at
95%, and its parent is still the active version; otherwise it is
`rejected`, with the reason `stale` when another version came first. Later
online versions keep a sleep job's named steps and refresh only the
statistics.

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
| `--denominator` | The bar's rule: `dynamic` (above, the default), `median` (k ÷ m), `milestone` (anchored on the first write, test and pass) or `hybrid` (`dynamic`, jumping to 90% after the first pass). `loo` and `first-k` report all four without it. |
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
nor its `-wal` keeps any copy of what was deleted. Run it while no
localmelo process is running. Both read
`procedures.db` in `--memory-dir`, else in `LOCALMELO_MEMORY_DIR`, else in
`~/.cache/localmelo/memory`; neither creates it.

### Exit codes

| Code | When |
|---|---|
| 0 | Done. |
| 2 | A usage error, or an input it cannot use: a missing or unreadable file, a malformed line (named by file and line), a repeated task. |
| 130 | Ctrl+C. |

## See also

- [Memory](memory.md): history, long-term memory and where the files live.
- [Run results](run-results.md): the statuses and the JSON line.
