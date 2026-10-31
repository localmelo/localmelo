# How localmelo works

What happens during a run: the loop of steps, reflection between attempts, the tools a step offers, how memory takes part and how the time limits work. The reference pages list the options; this page explains how they fit together.

## The run loop

```text
task
 └─ step: build a request ─► model reply ─┬─ text, no tool call ─► the answer (completed)
        ▲                                 └─ tool calls ─► checks ─► run ─► results
        └───────────────── next step ◄─────────────────────────────────────┘
an attempt that runs out of steps, or repeats itself ─► reflection ─► done, continue or stop
```

A run works on one task in steps. Each step:

1. builds a planning request from the task, the recent conversation (working memory), related long-term memories and the latest reflection, if any;
2. offers the model a list of tools (see [Tools in each step](#tools-in-each-step));
3. sends the request and reads the reply.

A reply with tool calls runs them, in order, through the checks, and their results join the conversation. A reply with text and no tool call is the final answer: the run ends as `completed`.

The run has a budget:

- At most 30 steps in total (`--steps`).
- With the default `reflect` policy, steps come in attempts of up to 10 steps, and a run has at most 5 attempts.
- With `--attempt-policy none`, the run is one attempt of up to `--steps` steps; the same tool calls, or the same errors, three steps in a row end it as `stuck`.

Two kinds of reply are not answers:

- A reply with neither text nor a tool call is an empty reply, and so is one with only dots, dashes or emoji (no letter or digit): localmelo asks the model to continue, the step still counts, and two in a row end the run as `failed`.
- A reply that is only the model's unfinished thinking is never the answer. localmelo tells it by facts, never by wording: a `<think>` that is never closed, no answer once the thinking is set aside, a streamed reply that repeats the server's reasoning, or, for a model whose thinking ends with a mark (`</think>`), no thinking and no mark. The model's format is the one `[cli.reasoning_tags]` declares (in the terminal), else the one its earlier replies showed.
- Such a reply becomes its thinking and goes on as an empty reply: localmelo asks once more, and a second reply without an answer ends the run as `failed` with the error `melo did not finish an answer`.
- A model that was told not to think (`enable_thinking = false` in the request) has no thinking to send. For it, a plain reply is the answer, and so are the words a server puts in its reasoning field when the reply has no other text. Only a `<think>` that is never closed, or reasoning tokens the server counted, still make a reply unfinished.

With `--reasoning compress`, each reply is compressed into a short note before the next step:

- The note records what the reply planned, found, decided and left open, plus what its tools returned; the model may write it itself in a `<note>` block.
- The notes, the reply's code drafts and one digest per attempt stay in a RAM-only note store beside the window.
- The loop then recalls: a frozen block of earlier thinking rides on the task query at the start of an attempt, each tool turn appends the notes and drafts the window no longer shows, and the `recall` tool lets the model ask for more.
- Every block is delimited, framed as data and checked by the candidate boundary; the first request that carries one records it in `history.db`'s `llm_calls.recall_json`.
- `legacy` keeps its own path unchanged. See [Recall in the loop](memory.md#recall-in-the-loop).

## Reflection

With the default `reflect` policy, an attempt that runs out of steps, or repeats itself three steps in a row, is followed by a reflection: one request without tools that asks the model to review the attempt and reply with a JSON object (a summary, what failed, what helped, and one action).

| Action | What happens |
|---|---|
| `done` | The task is complete. The run ends as `completed` with the reflection's answer. |
| `continue` | A new attempt starts, with the reflection in its prompt, if the continuation gate agrees. |
| `stop` | The run ends as `gate_stopped`. |
| `decompose` | The task should be split up. localmelo does not split tasks, so the run ends as `gate_stopped`. |

- The continuation gate lets a run continue only when the reflection says the task is still feasible and names a concrete next step; otherwise the run ends as `gate_stopped`.
- By default the gate is patient: a next step that is not new still continues, once per run. A second one ends the run.
- With `--reflection-gate flags`, a next step that is not new always ends the run.
- With `--reflection-gate utility`, the behavior before that option, the reflection must also estimate that another attempt's gain outweighs its cost and the risk of repeating itself: a utility score of at least 0.1 ([Evaluations](../project/evaluations.md#reflection-gate) compares the gates).
- The reflection's `useful_evidence`, `failed_hypotheses` and `recommended_avoids` keep at most 8 items of at most 300 characters each (not with `--reflection-gate flags -X agent.answer_on_stop=off`).

More about reflections:

- Only the latest reflection is carried forward: each replaces the one before. By default it joins the conversation once as an ordinary message; with `--cache-layout v2` it is rebuilt as the last message of every request.
- A reflection longer than about 2,000 characters is trimmed to fit and its decision stands (the last items of its lists go first, then the end of its summary); with `--reflection-gate utility`, such a reflection ends the run instead, unless it chose `done`.
- When no attempt can follow, the reflection is told so; it can still choose `done`, otherwise the run ends as `budget_exhausted` with its best answer as the result.
- A reply with no reflection (no JSON object with one of the four actions) is asked for once more; if the second has none either, the next attempt starts without a new reflection, or the run ends as `budget_exhausted` (see [Run results](../reference/run-results.md#statuses)).
- A reply whose JSON has one common slip, such as a stray `]` after a string value, a backslash before a character that JSON has no escape for, or an array that the next key breaks off, is mended and read without a second request.

### Ending with an answer

- A run that stops (`gate_stopped` or `budget_exhausted`) with no answer is asked once more, without tools, for its final answer from what it gathered; `unknown` is allowed.
- When the task asks for a closing line, such as `FINAL ANSWER: ...`, and the answer lacks it, one more request asks for that line and adds it after the answer.
- A task that names no closing line is left as it is, and a request that fails leaves the answer as it was.
- `-X agent.answer_on_stop=off` turns both requests off.

## Tools in each step

- Each step offers the model a list of tools, and only those tools can run in that step.
- By default the list holds every built-in tool plus the few other tools that match the task or that the model has already called in this conversation.
- See [How tools are offered](../reference/tools.md#how-tools-are-offered).

## Memory in the loop

- On each step, the planning request includes the recent conversation from working memory, and each tool call is recorded in history.
- In `legacy`, with an embedding backend and memory on, the request also includes the long-term memories closest to the task; each tool call is summarized into long-term memory, which also receives the run's latest reflection when the run ends.
- When the run ends, what it did (its step counts and the classes and labels of its tool calls) is recorded in [procedural memory](procedures.md), which no prompt includes.
- In the explicit modes, a reply's note, the attempt digest and the reflection are **candidates** for long-term memory: one is written only when a signal admits it (a rehearsal, a user statement, a user request covering the run, or `--admit-on-outcome` for the final digest).
- The checked write classifies the fact, compares it with the active entries of its kind and records one action (`NEW`, `UPDATE`, `RELATED`, `CONFLICT` or `IGNORE`) in the store's decision log; every stored text is redacted first.

See [How memory works](memory.md#how-memory-works) and [Long-term memory](../reference/memory.md#long-term-memory) for the limits and the exact behavior.

## Timeouts

Three limits apply at different levels:

| Limit | Value |
|---|---|
| Chat request timeout | 120 seconds (300 with the `ollama` backend): how long localmelo waits for data from the model server, restarting whenever data arrives; a request that times out is retried twice. See [Chat request timeout](../reference/configuration.md#chat-request-timeout). |
| Tool time limits | `shell_exec` 30 seconds, `python_exec` 60 seconds, `file_read`, `file_write` and `list_files` 60 seconds, MCP tools their server's `timeout`; the interactive UI gives the first two 10 minutes. See [Work in a directory](../guides/python.md#work-in-a-directory) for `tool_timeout_s`. |
| Run time limit (`--timeout`) | The only limit on a whole run; when it expires, the run ends as `timeout`. |

- `shell_exec` and `python_exec` run in a child process with a process group, which a timeout or a cancellation kills.
- Direct runs, workers and the gateway do not stream replies, so a server that sends nothing until the whole reply is written must finish within the timeout; a slow thinking model may need more.

## The main chat and tasks

![The main chat starts tasks and shows their levels](../assets/img/main-chat-tasks.webp)

- The main chat is a short run: at most 4 steps, no reflection, thinking off, and tools to start, list and change tasks, read files and search memory.
- Code reads your message first. `/task` and control phrases ("pause the slides") need no model call; a greeting or a short question is answered at once, never started as a task.
- Bigger work becomes a task: a worker runs it beside the conversation at a level from your words, and its result comes back as a card.
- A run that does not settle in 4 steps ends with "This needs more than a quick answer, so I started it as a task", or asks first when you asked a question.
- Side chats, direct runs and workers run the full loop above, as before.

## Code layout

- `melo/` holds the agent itself: the run loop, the checks, the tool executor, memory, and the interfaces that model providers implement.
- `support/` connects it to the outside: backends and model providers, the gateway and its web workspace, MCP servers, configuration and the setup wizard.
- It also holds the `melo-retrieve` command, which compares memory retrievers offline, and the scheduler (the web workspace's chats use its gate; direct runs do not).
- See the [architecture page](https://localmelo.github.io/localmelo/en/architecture.html) and the [project layout](https://github.com/localmelo/localmelo/blob/main/CONTRIBUTING.md#project-layout).
