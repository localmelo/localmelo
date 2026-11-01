# Events

A run can report what it does while it runs: each step, each reply of the
model, each tool call and its result, each reflection, and how the run ended.
This page lists the events, the order they come in, and the rules for the
function that receives them.

> [!NOTE]
> Events are part of the experimental Python API and may change before
> v0.1.0. A direct task uses an `observe` sink when its progress line is
> enabled; the `melo ""` loop runs without a sink. The interactive UI and
> the gateway's `/api` console run turns with event sinks and save events in
> sessions; the console exposes them over SSE and shows them in the web
> workspace. The legacy `/v1/playground` sink keeps only the `run.finished`
> outcome for Stop and does not show live events.

## Follow a run

Pass a function as `on_event` to `Agent.run_task`. It is called once for
each event:

```python
import asyncio

from localmelo import Agent, AgentEvent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


def show(event: AgentEvent) -> None:
    print(event.seq, event.type)


async def main() -> None:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm, options=AgentOptions(builtins=["shell_exec"]))
    try:
        result = await agent.run_task("Which files are here?", on_event=show)
        print(result.status)
    finally:
        await agent.close()


asyncio.run(main())
```

A run in which the model calls `shell_exec` once and then answers prints:

```text
1 run.started
2 step.started
3 assistant.message
4 tool.started
5 tool.finished
6 step.started
7 assistant.message
8 run.finished
completed
```

A run with `on_event` or an approval policy is an *interactive* run. That
matters only when the run is cancelled: it then ends as `cancelled` instead
of `crashed` (see [Stop a run](../guides/python.md#stop-a-run)).

To watch a run without making it interactive, pass the function as
`observe` instead. It receives the same events, but a cancelled run ends as
`crashed`, as a run without a sink does.

Give `on_event` or `observe`, not
both.

To also learn where the run stands, pass `progress=True` with either: the
run then reports [`run.progress`](#runprogress) events as well.

## The event

Each event is an `AgentEvent`, which `localmelo` exports:

| Field | Meaning |
|---|---|
| `type` | The kind of event; see [Catalogue](#catalogue). |
| `run_id` | The run's `task_id`: the same as `RunResult.task_id` and the task record's id in history. |
| `seq` | The event's number in the run: 1, 2, 3 and so on, with no gaps. |
| `ts` | When the event was emitted, as `time.time()`. |
| `data` | The event's fields: exactly the keys that the catalogue lists for its type. |
| `images` | `tool.finished` only: the full images that the tool returned, for code in the same process. |

`event.to_dict()` returns the event as a JSON object, without `images`:

```json
{"v": 1, "type": "step.started", "run_id": "<task_id>", "seq": 2, "ts": 1761465600.0, "data": {"step": 1, "attempt": 1}}
```

`v` is the schema version, 1. `json.dumps(event.to_dict(), allow_nan=False)`
always succeeds.

## Rules for the sink

The function you pass as `on_event` or `observe` is the sink.

- **Synchronous, in order.** The sink is called once per event, in `seq`
  order, from one task at a time: the task that runs `run_task`, or, after a
  cancellation, the task that finalizes the run. It is never called from two
  tasks at once or from another thread.
- **A plain function.** `run_task` raises `TypeError` before the run starts
  when `on_event` or `observe` is not callable, when it is an `async def`
  function, or when both are given. Then no event is emitted and no task
  record is written.
- **Do not block.** There is no buffer and no timeout: a slow sink holds up
  the run. Put slow work on a queue and do it in another task (see the
  example below).
- **A returned awaitable is ignored.** It is not awaited, and a coroutine is
  closed. The first time in a run, localmelo logs the warning
  `on_event returned an awaitable; async sinks are not supported and the event was dropped`.
- **A sink that raises loses that event.** When the sink raises an
  `Exception` or a `CancelledError`, the run goes on and the next event
  still gets the next `seq`. The first failure in a run is logged at
  WARNING, with a traceback, on the logger `localmelo.melo.agent._events`,
  and later ones at DEBUG. Any other exception, such as
  `KeyboardInterrupt`, propagates.
- **Without a sink, nothing is built.** A run without `on_event` or
  `observe` builds no event and no payload.

To do slow work with the events, such as writing them to a file or sending
them to a client, give the run a queue's `put_nowait` and read the queue in
a task of your own:

```python
import asyncio
import json

from localmelo import Agent, AgentEvent
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


async def main() -> None:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm)
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

    async def write_events() -> None:
        with open("events.jsonl", "a") as log:
            while True:
                event = await queue.get()
                log.write(json.dumps(event.to_dict()) + "\n")
                if event.type == "run.finished":
                    return

    writer = asyncio.create_task(write_events())
    try:
        await agent.run_task("What is 6*7?", on_event=queue.put_nowait)
        await writer
    finally:
        await agent.close()


asyncio.run(main())
```

## Catalogue

| Type | When | `data` keys |
|---|---|---|
| `run.started` | First, once the arguments of `run_task` are checked. | `query`, `max_steps`, `workspace_root`, `approval` |
| `inbox.accepted` | After request fitting and before sending the planning request. | `offset`, `text`, `step`, `attempt` |
| `step.started` | At the start of each planning step. | `step`, `attempt` |
| `assistant.message` | When the step's planning reply arrives. | `step`, `content`, `thinking`, `tool_calls`, `usage`, `prefix`, `duration_s` |
| `note` | In the compress modes, when a note, a reflection's note or an attempt digest enters the RAM-only note store. | `step`, `attempt`, `note_id`, `kind`, `source`, `text`, `chars`, `drafts`, `remember`, `stage` |
| `tool.started` | Before each tool call goes to the executor, also a call that the checks then refuse. | `step`, `call_id`, `name`, `arguments`, `preview` |
| `tool.finished` | When that call is done, refused, cancelled or crashed. | `step`, `call_id`, `name`, `status`, `error_category`, `output`, `output_chars`, `error`, `duration_s`, `images`, `approval`, `exit_code` |
| `notice` | When an attempt ends, the model gave an empty reply or one that is only unfinished thinking, or a model request is retried. | `kind`, `message`, `step`, `attempt`, `detail` |
| `recall` | When a recall block enters a request: at `start`, `attempt`, `step`, or from the `recall` tool. | `step`, `attempt`, `trigger`, `retriever`, `query`, `hits`, `chars`, `ms`, `fallback`, `placement` |
| `reflection.started` | Before the reflection that follows an attempt. | `attempt`, `failure_type` |
| `reflection` | Once the reflection's effect is known. | `attempt`, `action`, `continue_run`, `flags`, `gate`, `utility`, `stop_reason`, `rationale`, `summary`, `best_effort_result`, `requests`, `readable`, `unreadable_reason`, `repaired`, `usage`, `duration_s` |
| `run.progress` | Only with `progress=True`: right after `run.started`; after each `step.started`, `tool.finished`, `reflection.started` and `stuck` notice; and right before `run.finished`. | `level`, `state`, `step`, `attempt`, `limits`, `phase`, `cycle`, `tests`, `items`, `fraction`, `expected_steps`, `past`, `off_reason`, `basis` |
| `context_budget` | With context accounting enabled, once per planning or reflection attempt, including failures. | `kind`, `model`, `attempt`, `proxy_version`, `source`, `estimate`, `measured_prompt_tokens`, `input_limit`, `over_limit`, `image_allowance`, `pruned` |
| `run.finished` | Last. | `result`, `reviews` |

### run.started

- `query`: the task.
- `max_steps`: the step budget of the run.
- `workspace_root`: the agent's workspace root, or `null`; see
  [Work in a directory](../guides/python.md#work-in-a-directory).
- `approval`: `true` when the run has an approval policy.

### inbox.accepted

- `offset`: byte offset immediately after the accepted newline in this run's inbox.
- `text`: the instruction after pattern-based secret redaction. Worker
  `stream-json` omits this field; a saved worker event includes it.
- `step`, `attempt`: the planning step and its attempt. Acceptance
  follows `step.started` and request fitting, but does not prove model receipt
  or execution.

### step.started

- `step`: the step's number in the run, from 1. The number of
  `step.started` events is `RunResult.steps`.
- `attempt`: the attempt's number, from 1.

### assistant.message

- `step`: the step of the reply.
- `content`: the reply's text.
- `thinking`: the reasoning that the server sent apart from the text, else
  `""`. OpenAI-compatible servers send it as `reasoning_content` or
  `reasoning`, Ollama as `thinking`. Ollama's provider also puts it at the
  start of `content`, as a `<think>` block, and a server without a reasoning
  parser leaves it only there. A reply that is only unfinished thinking
  (see the `unfinished_answer` notice) has all its text here and `""` in
  `content`.
- `tool_calls`: the calls of the reply, each as
  `{id, name, arguments, preview}`. `preview` is as on `tool.started`.
- `usage`: the integer token counts of the reply, such as `prompt_tokens`,
  or `{}` when the server sent none. When an OpenAI-compatible server sends
  them, `cached_tokens` (from `prompt_tokens_details`) and
  `reasoning_tokens` (from `completion_tokens_details`) are in it too, on a
  streamed reply as well when the request asks the server for usage
  (`stream_options.include_usage`). A count the server did not send is not
  in `usage`, so a missing `cached_tokens` means "not reported", never 0.
  The `reflection` event's `usage` has them too (summed over its
  requests); the run's `RunResult.usage` keeps the three standard counts.
- `prefix`: how much of this request an earlier request of the run
  (planning or reflection) already sent, so a cache miss can be traced to
  what changed; `null` for the run's first request. The agent works it out from the
  messages and tools it sends, and the request body is the same with or
  without it. An object:
  - `messages`: how many leading messages are identical to those of the
    best earlier request: the one with the most identical leading messages,
    then the most characters, then the most recent. A server's prefix cache
    keeps more than the last request, so this is what it could reuse.
  - `chars`: the characters of content in those messages, plus the leading
    characters that the first differing message shares with its
    counterpart.
  - `against`: that request, `{"kind": "plan" | "reflect", "request": n}`;
    `n` counts the run's planning and reflection requests from 1 (a
    reflection asked twice takes two numbers).
  - `previous_messages`, `total_messages`: the message counts of that
    request and this one.
  - `first_changed`: the role of the first message that differs
    (`system`, `user`, `assistant` or `tool`), or `null` when that
    request is a plain prefix of this one (the usual growth of a run).
  - `tools_changed`: `true` when the tool definitions offered differ.
  - `fresh_chars`: the characters of this request's messages that no earlier
    request of the run held, found by the hash of each serialized message
    (role, content, thinking, calls, call id, images). No earlier request of
    this run sent them: they are new tool results, replies and reflection
    blocks. They are serialized characters, not the content characters that
    `chars` counts. `request_chars`: all
    the characters of the request's serialized messages and of the tools'
    schema. On a `reflection` that asked twice, both are summed over its
    requests. A log from before them lacks both.

  A server caches by token prefix, so a high `messages` with a low
  `cached_tokens` points at the server, and `first_changed: "system"` at
  the prompt. `scripts/cache_report.py LOG` prints both per request from a
  `stream-json` log, then the reusable share: `cached / (prompt - fresh)`,
  where `fresh` is `fresh_chars` as tokens, estimated with the request's own
  prompt tokens per character (`fresh_chars / request_chars * prompt_tokens`;
  an estimate, so the share can pass 100%). It counts the requests after a
  run's first that reported `cached_tokens`, per run and pooled, beside the
  share of all prompt tokens.
- `duration_s`: how long the planning request took, its retries and the
  waits before them included. With persistence on, it also includes
  recording the reply in history, which can wait up to 5 seconds while
  another process writes to `history.db`.

Before the event is built, `thinking` and `content` pass a minimal
redaction of secrets, which replaces each with `[redacted]`:

- the body of a PEM private key block, from `-----BEGIN … PRIVATE KEY-----`
  (RSA, EC, DSA, OPENSSH, ENCRYPTED or plain; also PGP's `PRIVATE KEY BLOCK`)
  to its END line, or to the end of
  the text when the END line is missing: the two marker lines stay and the
  body becomes one `[redacted]` line. Public keys and certificates stay;
- well-known key shapes: `sk-` keys, GitHub tokens (`ghp_…`,
  `github_pat_…` and the like) and AWS access key ids (`AKIA…`, `ASIA…`);
- the token after `Bearer`;
- a code of 4 to 8 digits within 20 characters of `verification code`,
  `passcode`, `OTP` or `验证码`, on the same line.

The redaction runs on the whole text, before it is clipped, and only for a
run with a sink. Working memory, the requests a run sends, the `RunResult`
and `run.finished` keep the text as the model wrote it, while history
stores its tasks and steps with the same redaction, and its model calls
too (see [Model calls](memory.md#model-calls)).

### tool.started

- `step`, `call_id` and `name`: the step, the tool call's id and the tool.
- `arguments`: the arguments as the model sent them.
- `preview`: a one-line summary of the call, at most 200 characters; a
  longer one keeps its first 197 characters and ends with `...`.

| Tool | Preview |
|---|---|
| `shell_exec` | The command, with runs of whitespace collapsed. |
| `python_exec` | The command the code runs, when it calls `subprocess.run`, `check_output`, `check_call`, `call` or `Popen` with a literal command (`grep -rn DEFAULT_WORKSPACE .`), or calls a function it defines that does, with literal arguments (`run("git", "status")` reads `git status`); else the first line that is not blank, a comment or an import (the first line when there is none). Followed by ` (+<n> more lines)` when more lines follow. |
| `file_read` | The path. |
| `list_files` | The path (`.` without one), followed by ` (<n> levels)` for a depth above 1. |
| `file_write` | `<path> (<n> chars)`. |
| Any other tool, or an argument above that is missing or not a string | The arguments as compact JSON, with sorted keys. |

The number of `tool.started` events is `RunResult.n_tool`.

### tool.finished

- `step`, `call_id` and `name`: as on the call's `tool.started`.
- `status`: how the call ended:
  - `success`, `error`, `timeout` or `blocked`, as the executor reports it;
  - `cancelled`: the run was cancelled, or its `timeout_s` expired, during
    the call;
  - `crashed`: an unexpected error left the executor or the result check,
    and the run ends as `crashed`.
- `error_category`: why the call failed, such as `tool_not_offered`,
  `invalid_arguments`, `blocked_by_checker`, `path_policy_violation`,
  `approval_denied`, `timeout` or `runtime_error`; `none` for a success;
  `cancelled` or `crashed` with those statuses.
- `output` and `error`: the result after the result check, which cuts output
  longer than 50,000 characters. `error` has no `Error: ` prefix, which the
  model sees in front of it. A cancelled call's error is
  `The run was stopped before this call finished; it may have partly run.`,
  or `The run was stopped before this call started; it did not run.` when an
  approval policy was still deciding about it. A crashed call's error is
  `<ExceptionType>: <message>`.
- `output_chars`: the length of `output`, also when the event's copy is
  clipped (see [Payload rules](#payload-rules)).
- `duration_s`: how long the tool ran: `0` when it did not run, and the time
  since `tool.started` for a cancelled or crashed call.
- `images`: the images that the tool returned, each as
  `{sha256, mime_type, size_bytes}`. `AgentEvent.images` holds the images
  themselves.
- `approval`: `null` when no approval policy was asked: for a tool that does
  not require approval, in a run without a policy, and for a call refused
  before the policy would have been asked. Otherwise it is
  `{request_id, allowed, reason, remember, origin}`, where `origin` is the
  decision's
  (`user`, `session`, `policy`, `timeout` or `auto`), `error` when the policy
  failed, or `withdrawn` when the run was cancelled while the policy decided.
  In auto mode, and for a call whose review is not plainly routine, it also has
  `mode` (`"auto"`), `review` (the review: `verdict`, `categories`, `reasons`,
  `paths`, `by` (`rules` or `model`), `rules_version` and, when the model was
  asked, `model`: `{verdict, category, fits_request, user_asked, reason, ms}` or
  `{error, ms}`) and `step_key` (32 hex characters: the same step in the same
  folder); a step allowed once has `allowance: "once"`. Origin `auto` with
  `allowed: false` is a step auto mode blocked; `reason` then names it in a few
  words (`installs software`); the console's
  [Allow once](api-v2.md#ask-and-auto) finds it by `request_id`. See
  [Approve tool calls](../guides/python.md#approve-tool-calls).
- `exit_code`: the exit status of the child process that `shell_exec` or
  `python_exec` ran; a negative value `-N` means that signal `N` ended it. It
  is `null` for every other tool, for a call whose child never started, and
  when the tool killed its child, after a timeout or a cancellation. A
  non-zero exit does not change `status`: `shell_exec` succeeds, and its text
  starts with `[exit code <n>]`; `python_exec` fails with
  `Python exited with code <n>`.

### note

- `step`, `attempt`: the planning step and attempt the note belongs to.
- `note_id`: the note's record id, which events, sessions and a later
  long-term store use. Prompts use the short alias instead (`n4`, `d1`,
  `g1`).
- `kind`: `step` (a reply with tool calls), `answer`, `empty`,
  `reflection` or `digest`.
- `source`: `model` or `heuristic`, who wrote the note.
- `text`: the checked note text, already redacted and sanitized; `chars` is
  its length.
- `drafts`: the aliases of the reply's drafts, as strings.
- `remember`, `stage`: whether a later memory may keep it, and its stage
  (`fresh`, `rehearsed` or `superseded`).

The CLI ignores the event until its plan line lands; the session runner
stores it, and a turn's last `digest` note goes to the turn's `note` column.

### memory

One event per admission decision, after its store transaction commits.

- `candidate_id`: the note id, or the fresh id of a user fact.
- `action`: `NEW`, `UPDATE`, `RELATED`, `CONFLICT` or `IGNORE`.
- `kind`: the entry's kind (`semantic`, `episodic`, `profile`, `reflection`).
- `admitted_by`: `rehearsal`, `user_request`, `user_statement`, `outcome` or
  `sleep`.
- `entry_id`: the new entry's id, or `null` for `IGNORE`.

It carries no text and no ids beyond the entry id; the decision log is the
audit surface. A boundary refusal emits nothing and counts
`candidates_refused`.

No `memory` event is emitted in `legacy`. The CLI
ignores it.

### notice

| `kind` | When | `message` | `detail` |
|---|---|---|---|
| `stuck` | An attempt ends because the same tool calls or errors repeated. | `The same tool calls or errors repeated; the attempt ends.` | `{}` |
| `budget` | An attempt ends because it used its steps. | `The attempt used its step budget.` | `{}` |
| `empty_reply` | The model gave an empty reply, and localmelo asks it to continue. A second empty reply in a row ends the run as `failed` instead, with no notice. | `The model returned an empty reply; asking it to continue.` | `{"count": <empty replies in a row>}` |
| `unfinished_answer` | The reply was only the model's unfinished thinking (see [How it works](../concepts/how-it-works.md)): its `assistant.message` has the text as `thinking` and no `content`, and localmelo asks once more. It counts as an empty reply; when it would be the second in a row, the run ends as `failed` with the error `melo did not finish an answer` instead, with no notice. | `The model's reply ended inside its thinking; asking it for its answer.` | `{}` |
| `tool_call_as_text` | The reply has no tool call but writes one out as text (a parameter name such as `text` and nothing else, or tool-call markup such as `<function_calls>`; see [How it works](../concepts/how-it-works.md)): its `assistant.message` has what the model sent, and localmelo asks once more. It counts as an empty reply. The notice also comes with the reply that ends the run, before `run.finished`, so a display can drop that reply's text (its message is then `The model wrote a tool call as text, so it did not run.`); when the reply would be the second in a row, the run ends as `failed` with the error `tool call written as text` (`melo did not finish an answer` when the row also held unfinished thinking). | `The model wrote a tool call as text, so it did not run; asking it again.` | `{"count": <empty replies in a row>}` |
| `no_change` | Only in a `--brief` run: an attempt boundary where another attempt starts finds the workspace unchanged since the run started (the digest of `git status`, `git diff HEAD` and untracked files is equal; never outside a git tree). A user line, "Nothing has changed in the workspace yet. Start this attempt by writing the part of the task you can already write, then continue.", is added for the next attempt, and ends with "Start with: <first next_promising_directions of the reflection>" when it has one. The same kind comes with `scope: "step"` after 6 steps of an attempt in a row in which the digest has not changed since the last change seen (or the start): the line added is "You have read for 6 steps without changing anything. Write the part of the task you can already write now (the smallest item first), then continue." | `Nothing has changed in the workspace yet.` | `{}` (boundary) or `{"scope": "step"}` |
| `brief_check` | Only in a `--brief` run (`AgentOptions.brief_check`): the gate ran one of the brief's `Check:` commands before the run would end `completed`. | `The task's check passed.`, `The task's check failed.` or `The task's check was refused and did not run.` | `{"round", "command", "exit_code", "duration_s", "refused", "output_tail"}`: `round` counts the gate's runs from 1 (a check run two steps before the end of the last attempt counts too); `command` is redacted; `output_tail` is the last 20 lines of the check's output, redacted, at most 4,000 characters; `exit_code` is `null` for a refused check or one with no status (a timeout); `refused` is the reason a refused check did not run, else `""`. |
| `brief_check_failed` | The gate gives up: a check failed with no retry left (two are allowed per run), with no step or attempt left to retry in, or was refused. The run ends `completed`. | `The task's check still fails; the run ends completed with failing tests.` | `{"retries": <retries used>}` |
| `retry` | A model request failed and is sent again after a delay; see [Provider errors and retries](run-results.md#provider-errors-and-retries). | `Model request failed; retrying in <delay> s: <the first 200 characters of the error>` | `{"request_attempt", "delay_s", "status_code"}` |

`step` and `attempt` are those of the latest `step.started`.

### recall

- `step`, `attempt`: the planning step and attempt the block was built in.
- `trigger`: `start`, `attempt`, `step` or `tool`.
- `retriever`: the retriever that answered, `bm25` after a fallback.
- `query`: the state query the retriever ranked, redacted.
- `hits`: one object per hit, `{doc_id, alias, kind, score, rank,
  origin_task_ids, chars, omitted}`, without item text. `kind` is `note`,
  `draft`, `thought`, `digest`, `reflection` or `long`; `origin_task_ids`
  names the task that produced it; `omitted` is true for a draft turned
  into a pointer line.
- `chars`: the block's characters; `ms`: its wall time; `fallback`: `""` or
  `timeout`, `parse` or `provider_error`; `placement`: `query`, `tool`,
  `message` or `tool-result`.

The event is additive and the CLI draws nothing for it yet. The same record
is in `memory_stats.recall_log` and, with its item text, in
`llm_calls.recall_json`.

### reflection.started

- `attempt`: the attempt that just ended, from 1.
- `failure_type`: what ended it: `stuck` or `budget`.

### reflection

- `attempt`: as on `reflection.started`.
- `action`: the reflection's choice: `continue`, `stop`, `decompose` or
  `done`. With `reflection_gate="utility"` it is `decompose` for a
  reflection over the size limit that did not choose `done`; with the
  default `"patient"` or `"flags"`, such a reflection is trimmed and keeps its choice.
- `continue_run`: `true` only when another attempt starts. It is `false`
  after `done`, after the last attempt or once `max_steps` is used up, and
  when the continuation gate stops the run.
- `flags`: the decision's `task_still_feasible`, `next_step_is_concrete` and
  `next_step_is_novel`, as booleans; `null` when the reply was not readable.
- `gate`: the `reflection_gate` in force, `patient` (the default), `flags` or `utility`.
- `utility`: with `"utility"`, the decision's utility (see the option); else
  `null`.
- `stop_reason`: `null` exactly when `continue_run` is `true`; otherwise the
  first of these that applies, in this order:
  `unreadable_reply` (no decision, and no attempt could start),
  `action_not_continue` (the action is `done`),
  `no_attempt_left` (`max_attempts` used; the run ends with `Max attempts
  reached`), `step_budget_used` (`max_steps` used; `Max steps reached`),
  then the gate's checks: `action_not_continue` (the action is `stop` or
  `decompose`), `not_feasible`, `not_concrete`, `not_novel`, and with
  `"utility"` only `utility_below_threshold`. It is derived by the
  code that decides, so it never disagrees with `continue_run`. A
  `gate_stopped` run whose rationale is blank reports it in `error`:
  `Stopped after attempt <n> (stop reason: <code>)`; with a rationale, the
  error is the rationale alone. In a `--brief` run a `continue` goes on while
  steps and attempts remain, so `flags` can hold a `false` while `stop_reason`
  is `null`; the other checks above are unchanged.
- `rationale`, `summary` and `best_effort_result`: the reflection's
  reasons, its summary of the attempt, and its best answer so far. They
  pass the same redaction of secrets as `assistant.message`'s `thinking`
  and `content`; the `RunResult` keeps the best answer as the model wrote
  it.
- `requests`: the number of reflection requests, 1 or 2. A reply that holds
  no reflection is asked for once more, without tools; the second request
  gets no event of its own. With the default `reflection_gate="patient"` or `"flags"`, a
  first reply that has JSON the decoder rejected, and is not a tool call, puts
  one more line after the retry note: the decoder's error and up to 60
  characters around it, for example `Your reply was not valid JSON (Expecting
  ',' delimiter at character 51: ...: "continue", "rationale": "r"]). Reply
  again with one JSON object.` In the compress modes the prompt fit can
  refuse a request before it is sent, which reads as an unreadable reply:
  `requests` is then 0, or 1 when only the retry was refused.
- `readable`: `false` when, with the default `reflection_gate="patient"` or `"flags"`, the
  second reply still held no reflection. `requests` is then 2; `action`,
  `rationale`, `summary` and `best_effort_result` are `null`; and
  `continue_run` is `true`, unless no attempt could start, in which case the
  run ends as `budget_exhausted`. With `"utility"` it is always `true`: such
  a reply counts as `stop`, with the rationale
  `Could not parse reflection response`, or
  `Reflection response was not a JSON object` when it was valid JSON that
  is not an object.
- `unreadable_reason`: why the first reply held no reflection: `tool_call` (it has
  tool calls and no reflection in its text; the usual case when the window
  ends in tool turns), `invalid` (its text has a JSON object that
  `"flags"` does not accept) or `no_json`; `null` when it read. The retry
  follows either way, so `readable` can be `true` with a reason.
- `repaired`: the one slip in the JSON of the reply that gave the reflection
  that was mended so that it reads: `unmatched_closer` (a `]` or `}` that no
  opener matches was dropped), `trailing_comma` (a comma before a `]` or `}`
  was dropped), `control_char` (raw newline, carriage return and tab
  characters inside strings were escaped), `invalid_escape` (the backslash of
  an escape that JSON does not have, such as `\;`, was doubled, so that it
  reads as a backslash and the character after it) or `unclosed_array` (an
  array that the next key broke off was closed: a `]` was inserted before the
  comma that precedes that key); `null` when the reply read as it was, when no
  reflection was read, and with `"utility"`. A repair is tried only when the
  reply holds no reflection as it stands, only on a `{...}` that the decoder
  rejects (the first eight of a reply), and counts only when the result is an
  object with a valid `recommended_action`. A reply that needs two repairs
  stays unreadable, a second invalid escape included, and nothing is closed or
  inserted beyond the one backslash or `]`, so text that stops short is not
  repaired. A repaired first reply needs no retry: `requests` is 1 and
  `unreadable_reason` is `null`.
- `usage`, `prefix` and `duration_s`: the token counts and the time of both
  requests, their retries included. `usage` also holds `cached_tokens` and
  `reasoning_tokens` when the server sent them, summed over the requests.
  `prefix` is the object described under `assistant.message`, for the
  first request of the reflection (`null` when it is the run's first request
  or no request was sent). With persistence on, `duration_s` also includes
  recording the replies in history, which can wait up to 5 seconds while
  another process writes to `history.db`.

See [Reflection](../concepts/how-it-works.md#reflection) and
[Run results](run-results.md#statuses).

### run.progress

Where the run stands, for a progress display. It comes only when
`run_task` gets `progress=True` and a sink.

A session turn's sink is its
runner's, so the turn stores and forwards `run.progress` events only when
its `SessionRunner` was built with `progress=True` (see
[Sessions](../guides/sessions.md#run-a-conversation)). Every key is always present,
`null` where it is unknown.

What it shows rests on the run's routine, as
[procedural memory](../concepts/procedures.md) remembers it on this model with these
options:

| Key | Meaning |
|---|---|
| `level` | What may be shown: 0, facts only (a first run, another model or other options); 1, past runs' facts and named steps as well. Level 2, a bar, is not shown in this release. |
| `state` | `running`, then `ended` on the event right before `run.finished`. |
| `step`, `attempt` | Those of the latest `step.started`; 0 before the first step. |
| `limits` | `{max_steps, steps_per_attempt, max_attempts, timeout_s}`: the run's budgets, `null` where the run has none (the attempt limits with `attempt_policy="none"`). `max_steps` is a budget, not the expected length. |
| `phase`, `cycle` | `Explore`, `Write`, `Verify` or `Finishing`, from the classes of the run's tool calls (see [Classes and phases](../concepts/procedures.md#classes-and-phases)), and the number of tests after a write; `null` and 0 before the first call. |
| `tests` | `{passed, failed, total}` from the latest pytest or unittest summary in a test call's output, or `null`. |
| `items` | Level 1: the routine's named steps, each `{label, state, repeat}`, `state` being `done`, `current`, `pending` or `skipped`; they tick forward only. `null` at level 0. |
| `fraction`, `expected_steps` | Always `null` in this release. |
| `past` | Level 1: `{runs, steps_lo, steps_median, steps_hi, wall_lo_s, wall_hi_s, last_steps, last_wall_s}`, facts about the past completed runs. |
| `off_reason` | Always `null` in this release. |
| `basis` | Level 1: `{routine_id, template_id, version, match, model, title}`, the memory the display rests on; `match` is `exact`, `routine` (a named routine) or `similar` (a similar task's steps, a hint); `title` names the routine a `similar` match borrowed from, `null` otherwise. |

A first run has no past to rest on: its events are facts only. This sink
prints the level, state, step and phase of each:

```python
import asyncio

from localmelo import Agent, AgentEvent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM


def show(event: AgentEvent) -> None:
    if event.type == "run.progress":
        data = event.data
        print(data["level"], data["state"], data["step"], data["phase"])


async def main() -> None:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm, options=AgentOptions(builtins=["shell_exec"]))
    try:
        await agent.run_task("Which files are here?", observe=show, progress=True)
    finally:
        await agent.close()


asyncio.run(main())
```

When the model runs `ls` and then answers, it prints:

```text
0 running 0 None
0 running 1 None
0 running 1 Explore
0 running 2 Explore
0 ended 2 Explore
```

At level 1, show the items as what the last runs did, not as a to-do list:
this run may do other work. The run's start looks the routine up once,
for at most 1 second, which counts toward the run's `timeout_s`; a lookup
that fails or takes longer gives level 0.

A failure inside the tracker
ends the run's `run.progress` events, never the run.

### run.finished

- `result`: the run's `RunResult`, as `RunResult.to_dict()`. Nothing in it
  is clipped, so `RunResult(**event.data["result"])` rebuilds it.
- `reviews`: only for a run with an approval policy: the calls the rules found
  `routine`, `high` and `undecided`, the `model` reviews the run sent, and
  `blocked`: the steps auto mode blocked, each `{call_id, title, step_key}`. A
  run without a policy, and an event stored before this key, has no `reviews`.

`run.finished` comes also when `run_task` raises because the run was
cancelled; it then holds the `RunResult` that `run_task` would have
returned.

## Guarantees

- **One start, one finish.** Every run that passes the argument checks emits
  `run.started` first, with `seq` 1, and `run.finished` last. Each comes
  exactly once, whatever the status, also when a second cancellation
  interrupts the clean-up.
- **Tool calls pair up.** Each `tool.started` is followed by exactly one
  `tool.finished` with the same `call_id`, before the next `tool.started`,
  `step.started` or `run.finished`, whatever the status. A call of the reply
  that never started, because the run stopped before it, gets no tool
  events: it appears only in `assistant.message`'s `tool_calls`.
- **Order within a step.** `step.started`; the `retry` notices of the
  planning request; `assistant.message`, unless planning failed; the tool
  pairs, in call order; then a `stuck`, `budget`, `empty_reply`,
  `unfinished_answer` or `tool_call_as_text` notice. A `tool_call_as_text`
  notice has no tool pairs before it: it comes right after `assistant.message`
  and the step's `note` event, and before the step's `recall` event, so a
  display that holds the reply's text can drop it first.
  With the default `attempt_policy="reflect"`, the notice that ends an
  attempt is followed by `reflection.started`, the `retry` notices of the
  reflection's requests, and `reflection`, unless one of its requests
  failed or the run was cancelled. With `"none"`, it is followed by
  `run.finished`. A `no_change` notice comes after that `reflection`, when the
  next attempt starts. The `brief_check` notices of a direct answer follow its
  `assistant.message`; those of a `done` reflection come before its
  `reflection` event, which then reads `continue_run: true` and
  `stop_reason: null` when a failing check overrules the `done`.
- **Progress follows what it reports.** With `progress=True`, a
  `run.progress` comes right after `run.started`, right after each
  `step.started`, `tool.finished` (of a call that returned),
  `reflection.started` and `stuck` notice, and right before `run.finished`
  with `state` `ended`, on every path. Without `progress=True`, a run's
  events are the same as before `run.progress` existed.
- **The counts match the run.** The number of `step.started` events is
  `RunResult.steps`, and the number of `tool.started` events is
  `RunResult.n_tool`. `run.finished`'s `result` equals the `to_dict()` of
  the `RunResult` that `run_task` returns, or would have returned. The
  `usage` of the run's `assistant.message` and `reflection` events adds up
  to the token counts of `RunResult.usage` (`prompt_tokens`,
  `completion_tokens`, `total_tokens`). The explicit reasoning modes also
  put `reasoning_chars` in `RunResult.usage`, and `cached_tokens` or
  `reasoning_tokens` when the server reports them, which no event carries.
  The events fall short of the run's counts in these cases too, where a
  reply is counted but no event carries it:
  - the response check raised an exception after the reply arrived, which
    ends the run as `crashed`;
  - the second request of a reflection failed after the first had returned;
  - with persistence on, a cancellation interrupted the recording of a
    reply in history.
- **One id.** Every event's `run_id` is the run's `task_id`, as in the
  `RunResult` and the task record.

## Payload rules

`data` follows these rules, which `json_safe_payload` in
`localmelo.melo.contracts` applies:

- **JSON-safe.** `data` holds only strings, integers, finite floats,
  booleans and `null`, in lists and objects with string keys. Other values
  in tool-call arguments become their `str()` text. A float that is not
  finite becomes `"nan"`, `"inf"` or `"-inf"`.
- **Clipping.** A string longer than 100,000 characters keeps its first
  100,000 and ends with `… [+N chars]`, where `N` counts the characters left
  out. This applies to the task, the reply's text and thinking, the
  arguments, the output, the error, the reflection's fields and every other
  string, except in `run.finished`, whose `RunResult` is whole. Redaction
  comes first.
- **Nesting.** Lists and objects nest at most 100 levels deep, counting the
  values of `data` as level 1. A list or object deeper than that becomes its
  `str()` text, clipped like any string. A model's tool-call arguments can
  nest deeper than Python could copy, and the event still gets built.
- **Images by digest.** `data` names images only by their SHA-256 digest,
  MIME type and size.
- **Read-only.** `to_dict()` returns a new dict with a shallow copy of
  `data`. Treat the payload as read-only.

## Session events

A [session](sessions.md) stores the events of each turn, the run's events
above, between two events of its own, and adds a few for the session. Each
stored event has a global `id`, the `session_id` and the `turn`, and its
`event` is the envelope of `to_dict()`; a session event has `"seq": null`
and `"run_id": ""`.

| Type | When | `data` keys |
|---|---|---|
| `turn.started` | A turn begins, before its run's events. | `turn`, `turn_id`, `user_text`, `client_request_id`; `origin` (`ask_owner`) only on a turn the person did not begin |
| `turn.finished` | The turn ends: its last event. | `turn`, `turn_id`, `status`, `error`, `cause` |
| `turn.undone` | A turn is undone; the event belongs to no turn. | `turn`, `turn_id` |
| `turn.pictures` | A console turn made pictures: after its run's last event, before `turn.finished`. Absent when it made none, and in the terminal's file. | `turn`, `turn_id`, `pictures` |
| `turn.memory` | What [memory across chats](../concepts/memory.md#memory-across-chats) did for a console turn, before its run's first event, on every turn with memory on, and again, whole and the same shape, each time the chat's model keeps a note with `remember`, tries to keep a secret, its `search_memory` finds something new, it opens a chat turn with `open_memory`, or a reply that claimed a save is kept or flagged (the newest one counts): `used` (`[{kind: record\|chat\|item, id, text, source, session_id, turn, at}]`, what was recalled into the message; an `item` entry, a memory item with developer setting `[memory] recall_source = "items"`, also has `item` (its id, the same as `id`) and `sources`, `[[chat id, turn], …]` for the turns it was written from), `searched` (`[{kind, id, text, source, session_id, turn, at}]`, what the model's own `search_memory` calls found in the turn, first found first, no repeats; absent until a search finds something), `opened` (`[{session_id, turn, id}]`, the chat turns the model opened in the turn with `open_memory`, first opened first, one entry for each turn, `id` the number of the list id (`t123`) it gave; developer setting `[memory] recall_form = "list"`; with `recall_source = "items"` an entry for a memory item is `{item, sources}`, `item` the number of the list id (`m12`) it gave, one entry for each item; absent until a turn or item is opened), `remembered` (`[{id, text, topic}]`, the notes the turn kept: those the typed text asked for, then the model's), `refused` (why something was not kept, or `""`), `unsaved_claim` (`true` when the reply said it noted something and no note was kept, even after melo tried to keep what it named, when what it named was not written to you, or when it named nothing ("Saved."); else `false`) and `recall` (how the search went: `{retriever: bm25\|embedding\|llm, fallback: none\|timeout\|error\|unreadable\|no-embedding, embed_ms, search_ms, select_ms, select_tokens: {prompt, completion}, candidates, used, block_chars}` (`block_chars`: the length of the block memory added to the message, 0 when none), the numbers `[memory] retriever` is compared by; the stored turn's `result` repeats it as `memory_recall`). `search_hint` (`true`, present only when the block carried the one line that tells the model to call the memory search tool because nothing was recalled: developer setting `[memory] search_when_empty = "hint"`). Absent with memory off, and in the terminal's file. | `turn`, `turn_id`, `used`, `searched`, `opened`, `remembered`, `refused`, `recall`, `unsaved_claim` |
| `session.title` | The first turn sets the session's title, or it is renamed. | `title` |
| `session.updated` | The session's model, workspace or `meta` changes (a console chat's mode is `meta.approvals`, `ask` or `auto`). | `model`, `workspace`, `workspace_label`, `meta` |
| `approval.requested` | A person is asked about a tool call (`SessionRunner.observe`). | `request_id`, `call_id`, `tool_name`, `arguments`, `preview`, `workspace_root` |
| `approval.resolved` | That request is answered or withdrawn. | `request_id`, `call_id`, `tool_name`, `outcome`, `origin`, `reason`, `remember` |

- `workspace_label` is the workspace with the home directory abbreviated to
  `~`, or an empty string when no workspace is set. Older stored
  `session.updated` events may not have this key; clients must tolerate its
  absence.
- `client_request_id` is the message id of the turn's
  [submission](sessions.md#message-ids-and-submissions), or `null`.
- `cause` says what ended the turn: `run`, `user_stop`, `closed`,
  `shutdown`, `crash`, `takeover` or `park`; see
  [Turns, statuses and causes](sessions.md#turns-statuses-and-causes).
- `pictures` lists, at most 24 of them, the picture files (`.png`, `.jpg`, `.jpeg`,
  `.gif`, `.webp`, `.svg`) the turn created or changed in its folder, seen by
  looking at the folder before and after the run, then the picture files its
  answer names that are inside the folder. Each is `{"path", "size", "mtime", "state"}`:
  `path` is relative to the folder with `/`, `size` is in bytes, `mtime` in
  seconds, and `state` is `new`, `modified` or `mentioned`. The files are served by
  [`GET /api/sessions/{id}/pictures`](api-v2.md#get-apisessionsidpictures).
- `outcome` is `allowed`, `denied` or `withdrawn`. `origin` is the
  decision's, as in `tool.finished`'s `approval`: `user`, `session`,
  `policy` or `timeout`; `error` when the ask failed; or `withdrawn` when
  the run stopped first, and when the store wrote the event for a request
  whose answer it never stored (see the guarantees below).
- `turn` is `null` for an event that belongs to no turn: `turn.undone`, a
  rename's `session.title`, `session.updated`, and the scheduler's events
  stored outside a turn.
- **The scheduler's types.** Types that start with `task.`, `sched.` or
  `attention.` are reserved for the scheduler. Only the gateway's store
  keeps them, as events that belong to no turn
  (`SessionStore.append_session_event`), or with a turn
  (`SessionRunner.publish`). Their keys are not pinned yet; their `data`
  follows the [payload rules](#payload-rules). The store refuses every other
  type.
- **`attention.card`** is the one such type with set keys: the main chat's
  card (`source_id`, `source_kind`, `title`, `state`, `line`, `at`, and `request_id` on a waiting or answered one), stored in
  the main chat with no turn when a side chat's run or a task comes back; see
  [Cards in the main chat](api-v2.md#cards-in-the-main-chat).
- **`attention.echo`** is the main chat's line about a task it started,
  changed or sent a message to (`echo`, `line`, and `task_id`, `title`,
  `level`, `action`, `undo`, `of`, `typed`, `text`, `turn_id`, `how` as the
  kind needs), with no turn; see
  [Echoes in the main chat](api-v2.md#echoes-in-the-main-chat).
- **A stored `run.finished` holds the answer redacted.** Before it is
  stored, the `result` of its `RunResult` passes the same redaction as
  `assistant.message`'s text, and so do the turn's stored answer and result.
  The `RunResult` that the run returns keeps the text.

The store guarantees:

- **One start, one end.** Every `turn.started` is followed by exactly one
  `turn.finished` for that turn once the turn is not running, also after a
  takeover, a shutdown or a crash, and no event of the turn comes after it.
- **`turn.finished` closes the turn.** Read it as closing every open card of
  its turn: a `tool.started` without a `tool.finished` reads "interrupted:
  may have partly run", and a step without its end is over. After a crash,
  the run's own events can stop anywhere: the store makes up no event of
  the run.
- **Every request is answered.** Every `approval.requested` is followed by
  exactly one `approval.resolved`. After a crash, recovery writes it, with
  the outcome `withdrawn` and the reason
  `melo stopped before this approval was answered.`; when a turn ends while
  the answer to one of its requests is not stored, the turn's end writes
  it, `withdrawn`, with the reason
  `The turn ended before this approval's answer was stored.`
- **A withdrawn message leaves no trace in the stream:** no turn number and
  no event.
- **Ids survive a restart.** Event ids only grow, across sessions,
  processes and restarts, so a client that reconnects with its last id
  (`Last-Event-ID`) receives what it missed, the events that recovery wrote
  included.

## Stability

Within schema version 1 (`v` 1), types and keys are only ever added, never
removed or renamed. Ignore types and keys you do not know.

A stored `assistant.message` or `reflection` from before `prefix` existed lacks that key
(`OPTIONAL_EVENT_FIELDS` in `localmelo.melo.contracts.events` lists such
keys, and a session store accepts events without them).
`EVENT_FIELDS` in `localmelo.melo.contracts` maps each type to its keys, and
`SESSION_EVENT_FIELDS` in `localmelo.support.sessions.store` each session
type; `OPTIONAL_SESSION_EVENT_FIELDS` there lists the session keys an event
may lack (`turn.started`'s `origin`).
