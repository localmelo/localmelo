# How localmelo works

What happens during a run: the loop of steps, reflection between attempts, the tools a step offers, how memory takes part and how the time limits work. The reference pages list the options; this page explains how they fit together.

## The run loop

```text
task
 └─ step: build a request ─► model reply ─┬─ text, no tool call ─► the answer (completed)
        ▲                                 └─ tool calls ─► checks ─► run ─► results
        └───────────────── next step ◄─────────────────────────────────────┘
an attempt that runs out of steps (50), or repeats itself ─► reflection ─► done, continue or stop
```

A run works on one task in steps. Each step:

1. builds a planning request from the task, the recent conversation (working memory), what recall finds (in the reasoning modes other than `legacy`) and the latest reflection, if any;
2. offers the model a list of tools (see [Tools in each step](#tools-in-each-step));
3. sends the request and reads the reply.

A reply with tool calls runs them, in order, through the checks, and their results join the conversation. A reply with text and no tool call is the final answer: the run ends as `completed`.

The run has a budget:

- At most 100 steps in total (`--steps`).
- With reflection on, the default, steps come in attempts of up to 50 steps (`-X agent.steps_per_attempt=N`), and a run has at most 5 attempts. A task that needs 20 or 40 steps is never interrupted: the first reflection comes only after 50 steps without an answer.
- With reflection off (`[agent] reflection = false`, `-X agent.reflection=off` or `--attempt-policy none`), the run is one attempt of up to `--steps` steps and never reflects; the same tool calls, or the same errors, three steps in a row end it as `stuck`.

Three kinds of reply are not answers:

- A reply with neither text nor a tool call is an empty reply, and so is one with only dots, dashes or emoji (no letter or digit): localmelo asks the model to continue, the step still counts, and two in a row end the run as `failed`.
- A reply that is only the model's unfinished thinking is never the answer. localmelo tells it by facts, never by wording: a `<think>` that is never closed, no answer once the thinking is set aside, a streamed reply that repeats the server's reasoning, or, for a model whose thinking ends with a mark (`</think>`), no thinking and no mark. The model's format is the one `[cli.reasoning_tags]` declares (in the terminal), else the one its earlier replies showed.
- Such a reply becomes its thinking and goes on as an empty reply: localmelo asks once more, and a second reply without an answer ends the run as `failed` with the error `melo did not finish an answer`.
- When the service reported the model's reasoning, the run does not fail: before the second reply would end it, localmelo asks one last time with thinking off, where the service has a switch for that. If the reply is still empty, the last lines of the thinking become the answer, ending with `(from melo's thinking)`, and the run completes. A thinking that was cut off inside the reply, or no thinking at all, still fails as above. The result and the events say when this happened (`empty_reply`, and the `thinking_off_retry` and `answer_from_thinking` notices), and `-X agent.empty_reply_fallback=off` turns it off, so such a run fails as above.
- A model that was told not to think (`enable_thinking = false` in the request) has no thinking to send. For it, a plain reply is the answer, and so are the words a server puts in its reasoning field when the reply has no other text. Only a `<think>` that is never closed, or reasoning tokens the server counted, still make a reply unfinished.
- A reply that writes a tool call out as text is never the answer either. A model's server can swallow a malformed call and leave `text` (the first argument's name), or markup such as `<function_calls>`, in the reply with no real tool call. localmelo tells it by shape, outside code blocks: call markup at the start of the reply or of a line (or followed by `{`, `<` or the end of a line, never in the middle of a sentence), or `to=functions.` and a name anywhere; or, against the tools offered on the step, a reply that is a JSON call object (a tool as its `name`, with `arguments` or `parameters`) or a list of them, or has a line that is wholly one, or that starts with a tool's name, `(` and an argument (`text=`, a quote, `{`, `[`) or `)`, or with a parameter name followed by nothing, `=`, `<`, a line break, a colon with no space, or an emoji. It goes on as an empty reply, with a message of its own that tells the model what went wrong; a second one in a row ends the run as `failed` with the error `tool call written as text`. A short reply that is a call's id (`text call_00_CfS77joytXzYfQlV455l3697`), or one that starts with a parameter name and holds such an id, is taken for a broken call too. A reply that discusses such markup in a code block or in a sentence is an answer, and so is JSON a person asked for (`{"text": "hello"}`) or a sentence about a tool (`file_read(path) reads a file`). A one-word reply that is just a parameter name (`path`) is taken for a broken call too, and the model is asked once more.

With `--reasoning compress`, each reply is compressed into a short note before the next step:

- The note records what the reply planned, found, decided and left open, plus what its tools returned; the model may write it itself in a `<note>` block.
- The notes, the reply's code drafts and one digest per attempt stay in a RAM-only note store beside the window.
- The loop then recalls: a frozen block of earlier thinking rides on the task query at the start of an attempt, each tool turn appends the notes and drafts the window no longer shows, and the `recall` tool lets the model ask for more.
- Every block is delimited, framed as data and checked by the candidate boundary; the first request that carries one records it in `history.db`'s `llm_calls.recall_json`.
- `legacy` has no recall in the loop. See [Recall in the loop](memory.md#recall-in-the-loop).

## Reflection

Reflection is an option, on by default (**Reflect when stuck after many steps** in Settings, `[agent] reflection`; see [Evaluations](../project/evaluations.md#long-attempts-and-reflection-as-an-option) for why). With it on, an attempt that runs out of its 50 steps, or repeats itself three steps in a row, is followed by a reflection: one request without tools that asks the model to review the attempt and reply with a JSON object (a summary, what failed, what helped, and one action). A run that is not done after 50 steps reflects once and goes on, up to 100 steps. A reflection costs one more model request and about 170 output tokens. With it off, nothing reflects and the run goes straight to its 100 steps.

| Action | What happens |
|---|---|
| `done` | The task is complete. The run ends as `completed` with the reflection's answer. |
| `continue` | A new attempt starts, with the reflection in its prompt, if the continuation gate agrees. |
| `stop` | The run ends as `gate_stopped`. |
| `decompose` | The task should be split up. localmelo does not split tasks, so the run ends as `gate_stopped`. |

- The continuation gate lets a run continue only when the reflection says the task is still feasible and names a concrete next step; otherwise the run ends as `gate_stopped`.
- The gate is patient: a next step that is not new still continues, once per run. A second one ends the run ([Evaluations](../project/evaluations.md#reflection-gate) compares it with the earlier gates).
- The reflection's `useful_evidence`, `failed_hypotheses` and `recommended_avoids` keep at most 8 items of at most 300 characters each.

More about reflections:

- Only the latest reflection is carried forward: each replaces the one before. It joins the conversation once, as an ordinary message.
- A reflection longer than about 2,000 characters is trimmed to fit and its decision stands (the last items of its lists go first, then the end of its summary).
- When no attempt can follow, the reflection is told so; it can still choose `done`, otherwise the run ends as `budget_exhausted` with its best answer as the result.
- A reply with no reflection (no JSON object with one of the four actions) is asked for once more; if the second has none either, the next attempt starts without a new reflection, or the run ends as `budget_exhausted` (see [Run results](../reference/run-results.md#statuses)).
- A reply whose JSON has one common slip, such as a stray `]` after a string value, a backslash before a character that JSON has no escape for, or an array that the next key breaks off, is mended and read without a second request.

### Ending with an answer

- A run that stops (`gate_stopped`, `budget_exhausted` or `stuck`) with no answer is asked once more, without tools, for its final answer from what it gathered; `unknown` is allowed (a `stuck` run is asked to answer what it can in plain words).
- When the task asks for a closing line, such as `FINAL ANSWER: ...`, and the answer lacks it, one more request asks for that line and adds it after the answer.
- A task that names no closing line is left as it is, and a request that fails leaves the answer as it was.
- `-X agent.answer_on_stop=off` turns both requests off.

## Tools in each step

- Each step offers the model a list of tools, and only those tools can run in that step.
- By default the list holds every built-in tool plus the few other tools that match the task or that the model has already called in this conversation.
- See [How tools are offered](../reference/tools.md#how-tools-are-offered).

## Memory in the loop

- On each step, the planning request includes the recent conversation from working memory, and each tool call is recorded in history.
- When the run ends, what it did (its step counts and the classes and labels of its tool calls) is recorded in [procedural memory](procedures.md), which no prompt includes.
- In the explicit modes, a reply's note, the attempt digest and the reflection are **candidates** for long-term memory: one is written only when a signal admits it (a rehearsal, a user statement or a user request covering the run).
- The checked write classifies the fact, compares it with the active entries of its kind and records one action (`NEW`, `UPDATE`, `RELATED`, `CONFLICT` or `IGNORE`) in the store's decision log; every stored text is redacted first.

See [How memory works](memory.md#how-memory-works) and [Long-term memory](../reference/memory.md#long-term-memory) for the limits and the exact behavior.

## Timeouts

Three limits apply at different levels:

| Limit | Value |
|---|---|
| Chat request timeout | 300 seconds: how long localmelo waits for data from the model server, restarting whenever data arrives; a request that times out is retried twice. See [Chat request timeout](../reference/configuration.md#chat-request-timeout). |
| Tool time limits | `shell_exec` 30 seconds, `python_exec` 60 seconds, `file_read`, `file_write` and `list_files` 60 seconds, MCP tools their server's `timeout`; the interactive UI gives the first two 10 minutes. See [Work in a directory](../guides/python.md#work-in-a-directory) for `tool_timeout_s`. |
| Run time limit (`--timeout`) | The only limit on a whole run; when it expires, the run ends as `timeout`. |

- `shell_exec` and `python_exec` run in a child process with a process group, which a timeout or a cancellation kills.
- Direct runs and workers do not stream replies, so a server that sends nothing until the whole reply is written must finish within the timeout; a slow thinking model may need more. A chat's turns in the gateway ask the server to stream.

## The main chat and tasks

![The main chat starts tasks and shows their levels](../assets/img/main-chat-tasks.webp)

- Every chat decides first who does the work, and does short work itself (long work goes to a task): no reflection, tools for files, Python, the web and the browser, tools to start, list and change tasks, and memory. Every step passes the same checks and approvals.
- Code reads your message first, for explicit commands only: `/task` and control phrases ("pause the slides") need no model call. Everything else is the model's to read.
- A task is work beside the chat: ask for one, or the chat starts one for long work. A worker runs it at a level from your words, with the browser and while it makes progress, to about 500 steps, and its result comes back as a card.
- A turn melo began itself (a task's question, a follow-up) reads and answers, and changes nothing.
- Direct runs and workers run the full loop above, as before.

## Code layout

- `melo/` holds the agent itself: the run loop, the checks, the tool executor, memory, and the interfaces that model providers implement.
- `support/` connects it to the outside: backends and model providers, the gateway and its web workspace, MCP servers, configuration and the setup wizard.
- It also holds the `melo-retrieve` command, which compares memory retrievers offline, and the scheduler (the web workspace's chats use its gate; direct runs do not).
- See the [architecture page](https://localmelo.github.io/localmelo/en/architecture.html) and the [project layout](https://github.com/localmelo/localmelo/blob/main/CONTRIBUTING.md#project-layout).
