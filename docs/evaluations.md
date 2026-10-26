# Evaluations

This page reports the controlled comparisons behind localmelo's defaults.
Each one is small: one model on one machine and a fixed set of tasks. Read
it as the evidence for one decision, not as a general benchmark. New
comparisons are added when they settle a default.

## Reflection gate

On 2026-10-23, on 12 coding exercises run 5 times in each setting, the
default `flags` gate passed **59/60 runs, against 50/60 for the earlier
`utility` gate: 15 percentage points more**. Runs without reflection passed
59/60 as well.

| Setting | Option | Passed |
|---|---|---:|
| `flags`, the default since [#34](https://github.com/localmelo/localmelo/pull/34) | `--reflection-gate flags` | 59/60 |
| `utility`, the earlier gate | `--reflection-gate utility` | 50/60 |
| No reflection | `--attempt-policy none` | 59/60 |

### What the two gates do

After a reflection, `flags` starts the next attempt when the reflection
chooses `continue` and its three checks hold. `utility`, the behavior before
#34, also requires a utility score computed from the reflection's own
estimates to reach 0.1. They also differ in two other cases:

- A reflection that still cannot be read after the retry: with `flags`, the
  next attempt starts if attempts and steps are left; `utility` ends the
  run.
- A reflection longer than about 2,000 characters: `flags` trims it and
  keeps its decision; `utility` ends the run, unless the reflection chose
  `done`.

See [Reflection](how-it-works.md#reflection).

The comparison measures these differences together. It does not say how much
each one contributes.

### Conditions

- **Code:** localmelo `c295ce2`, the merge commit of #34. As of #42
  (2026-10-26), the code that reads reflections and decides whether to
  continue is unchanged since that commit.
- **Tasks:** 12 Python exercises from the Aider polyglot benchmark
  ([`Aider-AI/polyglot-benchmark`](https://github.com/Aider-AI/polyglot-benchmark)
  at `7e0611e`): affine-cipher, grade-school, list-ops, phone-number,
  pig-latin, proverb, react, robot-name, scale-generator, transpose,
  variable-length-quantity and wordy. A run passes when the exercise's
  original unit tests pass on the solution it leaves.
- **Runs:** 5 passes. In each pass, each exercise ran once in each setting,
  in a seeded random order. That makes 180 scored runs, after one warm-up:
  repeated runs of 12 exercises, not 180 different tasks.
- **Options:** `--max-steps 30 --timeout 900 --request-timeout 1800`, with
  the default tool exposure.
- **Model:** `Qwen3.8-27B-oQ5-mtp`, served by oMLX 0.7.1rc1 with a local
  patch. Multi-token prediction on, thinking off, a 131,072-token context,
  temperature 1.0, top-p 0.95 and top-k 20.
- **System:** macOS 27.2 (26B5101f), Xcode 27.1 (27A9275).

### Caveats

- **The default ties running without reflection.** Paired by exercise and
  pass, `flags` and no reflection each passed one pair the other failed.
  This comparison shows no gain from reflection itself.
- The difference comes from 4 exercises. The other 8 passed all 5 runs in
  every setting:

  | Exercise | `flags` | `utility` | No reflection |
  |---|---:|---:|---:|
  | pig-latin | 5/5 | 4/5 | 5/5 |
  | scale-generator | 4/5 | 2/5 | 5/5 |
  | transpose | 5/5 | 2/5 | 4/5 |
  | wordy | 5/5 | 2/5 | 5/5 |
  | The other 8 | 40/40 | 40/40 | 40/40 |

- Paired by exercise and pass, `flags` alone passed 9 pairs and `utility`
  alone none (exact sign test, p = 0.004).
- "The earlier gate" is the `utility` setting run at the same commit, which
  keeps the behavior from before #34. It is not a run of the older code.
- All 10 runs that `utility` lost ended at a reflection with the task still
  unsolved: 3 because the reflection could not be read, and 7 on the
  reflection's own decision. Six of those 7 said the work was close or the
  next step was clear.
- `flags` stopped no run. Its one failure, on scale-generator, had a passing
  solution midway, broke it later and used all 30 steps: the model did not
  notice that its solution passed. The gate does not help the model notice
  that a task is done.
- The result holds for this model, server and task set. Other models or
  tasks may behave differently.
- This comparison makes no claim about speed.
- The raw records (run logs, requests and replies) are not published. They
  stay with the maintainer's benchmark, which is not public.
