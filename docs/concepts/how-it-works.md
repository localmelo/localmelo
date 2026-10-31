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

- At most 30 steps in total (`--steps`).
- With the default `reflect` policy, steps come in attempts of up to 10
  steps, and a run has at most 5 attempts.
- With `--attempt-policy none`, the run is one attempt of up to
  `--steps` steps. If the same tool calls, or the same errors, repeat
  three steps in a row, it ends as `stuck`.

A reply with neither text nor a tool call is an empty reply. localmelo asks
the model to continue, and the step still counts. Two empty replies in a row
end the run as `failed`.

A reply that is only the model's unfinished thinking is never the answer.
Thinking models end their thinking with a mark (`</think>`); when the model
stops before it, a server often returns the thinking as the reply's text.
localmelo tells such a reply by facts, never by its wording: a `<think>` that
is never closed; no answer once the thinking is set aside, with reasoning
tokens counted; a streamed reply whose text repeats the reasoning the server
sent; or, for a model whose thinking ends with a mark, no thinking and no
mark. The model's format is the one `[cli.reasoning_tags]` declares (in the
terminal), else the one its earlier replies showed: a server that sent its
reasoning in a field of its own, and counted no reasoning tokens in this
reply. The reply's text becomes its thinking (folded in the terminal
and the console), and it goes on as an empty reply: localmelo asks once
more, and a second reply without an answer ends the run as `failed` with
the error `melo did not finish an answer`.

With `--reasoning compress`, each reply is compressed into a short note
before the next step: the note asks what the reply planned, found, decided
and left open, plus what its tools returned, and the model may write it
itself in a `<note>` block. The notes, the reply's code drafts and one digest
per attempt stay in a RAM-only note store beside the window; a pending draft
is pinned once after its tool results. The loop then recalls: a frozen block
of earlier thinking rides on the task query at the start of an attempt, each
tool turn appends a block of the notes and drafts the window no longer
shows, and the `recall` tool lets the model ask for more. Every block is
delimited, framed as data and checked by the candidate boundary, and the
first request that carries one records it in `history.db`'s
`llm_calls.recall_json`. `legacy` keeps its own path unchanged. See
[Recall in the loop](memory.md#recall-in-the-loop).

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
[Evaluations](../project/evaluations.md#reflection-gate) compares the two gates and runs
without reflection on coding exercises.

A developer setting, `--reflection-gate patient`, is `flags` with one exception:
a `continue` that is feasible and concrete but not novel still starts the next
attempt, once per run. A second one stops the run as `flags` does, and the
reflection's `recommended_avoids` reach the next attempt as usual. With it,
and with `answer_on_stop`, the reflection's `useful_evidence`, `failed_hypotheses`
and `recommended_avoids` keep at most 8 items of at most 300 characters each.

More about reflections:

- Only the latest reflection is carried forward: each one replaces the one
  before. By default a reflection is added to the conversation once, as an
  ordinary message, so an earlier one stays until it leaves with the old turns;
  with `--cache-layout v2` it is rebuilt as the last message of every request
  and the prompt holds only the latest. A reflection longer than about 2,000
  characters is trimmed to fit, and its decision stands: the last items of its lists go
  first, then the end of its summary. With `--reflection-gate utility`, such
  a reflection ends the run instead, unless it chose `done`.
- When no attempt can follow, because the steps or attempts are used up, the
  reflection is told so. It can still choose `done`; otherwise the run ends
  as `budget_exhausted` with the reflection's best answer as its result.
- A reply that holds no reflection, that is no JSON object with one of the
  four actions, is asked for once more. If the second reply holds none
  either, the next attempt starts without a new reflection, or the run ends
  as `budget_exhausted` when none can start; see
  [Run results](../reference/run-results.md#statuses).
- A reply whose JSON has one common slip, such as a stray `]` after a string
  value, a backslash before a character that JSON has no escape for, or an
  array that the next key breaks off, is read after the slip is mended and
  needs no second request. If the decoder rejected the first reply's JSON, the
  second request quotes its error.

## Tools in each step

Each step offers the model a list of tools, and only those tools can run in
that step. By default the list holds every built-in tool plus the few other
tools that match the task or that the model has already called in this
conversation. See [How tools are offered](../reference/tools.md#how-tools-are-offered).

## Memory in the loop

On each step, the planning request includes the recent conversation from
working memory. Each tool call is recorded in history. In `legacy`, with an
embedding backend and memory on, the request also includes the long-term
memories closest to the task; each tool call is summarized into long-term
memory, which also receives the run's latest reflection when the run ends.
Also when the run ends, what it did (its step counts and the classes and
labels of its tool calls) is recorded in [procedural memory](procedures.md),
which no prompt includes.
See [How memory works](memory.md#how-memory-works) for the limits and the
exact behavior.

In the explicit modes, a reply's note, the attempt digest and the reflection
are also **candidates** for long-term memory. A candidate is written only
when a signal admits it: a rehearsal (a restatement, a successful call that
acted on it, a reflection's `used_notes`), a user statement, a user request
covering the run, or `--admit-on-outcome` for the final digest. The checked
write classifies the fact, compares it with the active entries of its kind
and records one action (`NEW`, `UPDATE`, `RELATED`, `CONFLICT` or `IGNORE`)
in the store's decision log. The text passes the candidate boundary with
target `long`, so every stored text is redacted first. `legacy` keeps its
old path: a step summary and the reflection promotion, unchanged. See
[Long-term memory](../reference/memory.md#long-term-memory).

## Timeouts

Three limits apply at different levels:

- **Chat request timeout** (120 seconds; 300 seconds with the `ollama`
  backend). It limits how long localmelo waits for data from the model
  server, and the timer restarts whenever data arrives. localmelo does not
  stream replies, so a server that sends nothing until the whole reply is
  written must finish within the timeout; a slow thinking model may need
  more. A server that sends keep-alive bytes while it generates never times
  out. A request that times out is retried twice. See
  [Chat request timeout](../reference/configuration.md#chat-request-timeout).
- **Tool time limits.** `shell_exec` 30 seconds, `python_exec` 60 seconds,
  `file_read`, `file_write` and `list_files` 60 seconds, MCP tools their server's
  `timeout`. `shell_exec` and `python_exec` run in a child process with a
  process group, which a timeout or a cancellation kills; the interactive
  UI gives them 10 minutes. See
  [Work in a directory](../guides/python.md#work-in-a-directory) for the Python
  API's `tool_timeout_s`.
- **Run time limit** (`--timeout`). It is the only limit on a whole run. When
  it expires, the run ends as `timeout`.

## The main chat and tasks

![The main chat starts tasks and shows their levels](../assets/img/main-chat-tasks.webp)

- The main chat is a short run: at most 4 steps, no reflection, thinking off, and tools to start, list and change tasks, read files and search memory.
- Code reads your message first. `/task` and control phrases ("pause the slides") need no model call; a greeting or a short question goes straight to the answer.
- Bigger work becomes a task: a worker runs it beside the conversation at a level from your words, and its result comes back as a card.
- A run that does not settle in 4 steps ends with "This needs more than a quick answer, so I started it as a task", or asks first when you asked a question.
- Side chats, direct runs and workers run the full loop above, as before.

## Code layout

The code has two parts. `melo/` holds the agent itself: the run loop, the
checks, the tool executor, memory, and the interfaces that model providers
implement. `support/` connects it to the outside: backends and model
providers, the gateway and its web workspace, MCP servers, configuration and
the setup wizard. It also holds the `melo-retrieve` command, which compares
memory retrievers offline, and the scheduler (the web workspace's chats
use its gate; direct runs do not). See
the [architecture page](https://localmelo.github.io/localmelo/en/architecture.html)
and the
[project layout](https://github.com/localmelo/localmelo/blob/main/CONTRIBUTING.md#project-layout).
