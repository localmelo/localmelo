# How localmelo works

This page explains what happens during a run: the loop of steps, reflection
between attempts, the tools a step offers, how memory takes part, and how the
time limits work. The reference pages list the options; this page explains
how they fit together.

## The run loop

A run works on one task in steps. Each step:

1. builds a planning request from the task, the recent conversation (working
   memory), related long-term memories and the latest reflection, if any;
2. offers the model a list of tools (see [Tools in each step](#tools-in-each-step));
3. sends the request and reads the reply.

A reply with tool calls runs them, in order, through the checks. Their
results join the conversation, and the next step begins. A reply with text
and no tool call is the final answer: the run ends as `completed`.

The run has a budget:

- At most 30 steps in total (`--max-steps`).
- With the default `reflect` policy, steps come in attempts of up to 10
  steps, and a run has at most 5 attempts.
- With `--attempt-policy none`, the run is one attempt of up to
  `--max-steps` steps. If the same tool calls, or the same errors, repeat
  three steps in a row, it ends as `stuck`.

A reply with neither text nor a tool call is an empty reply. localmelo asks
the model to continue, and the step still counts. Two empty replies in a row
end the run as `failed`.

## Reflection

With the default `reflect` policy, an attempt that runs out of steps, or
repeats itself three steps in a row, is followed by a reflection. That is one
request without tools, which asks the model to review the attempt and reply
with a JSON object: a summary, what failed, what helped, and one action.

| Action | What happens |
|---|---|
| `done` | The task is complete. The run ends as `completed` with the reflection's answer. |
| `continue` | A new attempt starts, with the reflection in its prompt, if the continuation gate agrees. |
| `stop` | The run ends as `gate_stopped`. |
| `decompose` | The task should be split up. localmelo does not split tasks, so the run ends as `gate_stopped`. |

The continuation gate lets a run continue only when the reflection says the
task is still feasible and names a concrete and new next step. Otherwise the
run ends as `gate_stopped`. With `--reflection-gate utility`, the behavior
before that option, the reflection must also estimate that the gain of
another attempt outweighs its cost and the risk of repeating itself: a
utility score computed from its own estimates must reach 0.1.

More about reflections:

- Only the latest reflection is kept: each one replaces the one before, so
  the prompt stays small. A reflection longer than about 2,000 characters is
  trimmed to fit, and its decision stands: the last items of its lists go
  first, then the end of its summary. With `--reflection-gate utility`, such
  a reflection ends the run instead, unless it chose `done`.
- When no attempt can follow, because the steps or attempts are used up, the
  reflection is told so. It can still choose `done`; otherwise the run ends
  as `budget_exhausted` with the reflection's best answer as its result.
- A reply that holds no reflection, that is no JSON object with one of the
  four actions, is asked for once more. If the second reply holds none
  either, the next attempt starts without a new reflection, or the run ends
  as `budget_exhausted` when none can start; see
  [Run results](run-results.md#statuses).

## Tools in each step

Each step offers the model a list of tools, and only those tools can run in
that step. By default the list holds every built-in tool plus the few other
tools that match the task or the conversation. See
[How tools are offered](tools.md#how-tools-are-offered).

## Memory in the loop

On each step, the planning request includes the recent conversation from
working memory and, with an embedding backend, the long-term memories closest
to the task. After each tool call, the step is recorded in history and, with
an embedding backend, summarized into long-term memory, which also receives
the run's latest reflection when the run ends. See
[How memory works](memory.md#how-memory-works) for the limits and the exact
behavior.

## Timeouts

Three limits apply at different levels:

- **Chat request timeout** (120 seconds; 300 seconds with the `ollama`
  backend). It limits how long localmelo waits for data from the model
  server, and the timer restarts whenever data arrives. localmelo does not
  stream replies, so a server that sends nothing until the whole reply is
  written must finish within the timeout; a slow thinking model may need
  more. A server that sends keep-alive bytes while it generates never times
  out. A request that times out is retried twice. See
  [Chat request timeout](configuration.md#chat-request-timeout).
- **Tool time limits.** `shell_exec` 30 seconds, `file_read` and
  `file_write` 60 seconds, MCP tools their server's `timeout`. `python_exec`
  has no limit, because its code runs inside localmelo and cannot be
  interrupted.
- **Run time limit** (`--timeout`). It is the only limit on a whole run. When
  it expires, the run ends as `timeout`.

## Code layout

The code has two parts. `melo/` holds the agent itself: the run loop, the
checks, the tool executor, memory, and the interfaces that model providers
implement. `support/` connects it to the outside: backends and model
providers, the gateway, MCP servers, configuration and the setup wizard. See
the [architecture page](https://localmelo.github.io/localmelo/en/architecture.html)
and the
[project layout](https://github.com/localmelo/localmelo/blob/main/CONTRIBUTING.md#project-layout).
