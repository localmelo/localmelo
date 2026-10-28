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
of `crashed` (see [Stop a run](python-api.md#stop-a-run)).

To watch a run without making it interactive, pass the function as
`observe` instead. It receives the same events, but a cancelled run ends as
`crashed`, as a run without a sink does. Give `on_event` or `observe`, not
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
| `step.started` | At the start of each planning step. | `step`, `attempt` |
| `assistant.message` | When the step's planning reply arrives. | `step`, `content`, `thinking`, `tool_calls`, `usage`, `duration_s` |
| `note` | In the compress modes, when a note, a reflection's note or an attempt digest enters the RAM-only note store. | `step`, `attempt`, `note_id`, `kind`, `source`, `text`, `chars`, `drafts`, `remember`, `stage` |
| `tool.started` | Before each tool call goes to the executor, also a call that the checks then refuse. | `step`, `call_id`, `name`, `arguments`, `preview` |
| `tool.finished` | When that call is done, refused, cancelled or crashed. | `step`, `call_id`, `name`, `status`, `error_category`, `output`, `output_chars`, `error`, `duration_s`, `images`, `approval`, `exit_code` |
| `notice` | When an attempt ends, the model gave an empty reply, or a model request is retried. | `kind`, `message`, `step`, `attempt`, `detail` |
| `recall` | When a recall block enters a request: at `start`, `attempt`, `step`, or from the `recall` tool. | `step`, `attempt`, `trigger`, `retriever`, `query`, `hits`, `chars`, `ms`, `fallback`, `placement` |
| `reflection.started` | Before the reflection that follows an attempt. | `attempt`, `failure_type` |
| `reflection` | Once the reflection's effect is known. | `attempt`, `action`, `continue_run`, `rationale`, `summary`, `best_effort_result`, `requests`, `readable`, `usage`, `duration_s` |
| `run.progress` | Only with `progress=True`: right after `run.started`; after each `step.started`, `tool.finished`, `reflection.started` and `stuck` notice; and right before `run.finished`. | `level`, `state`, `step`, `attempt`, `limits`, `phase`, `cycle`, `tests`, `items`, `fraction`, `expected_steps`, `past`, `off_reason`, `basis` |
| `context_budget` | With context accounting enabled, once per planning or reflection attempt, including failures. | `kind`, `model`, `attempt`, `proxy_version`, `source`, `estimate`, `measured_prompt_tokens`, `input_limit`, `over_limit`, `image_allowance` |
| `run.finished` | Last. | `result` |

### run.started

- `query`: the task.
- `max_steps`: the step budget of the run.
- `workspace_root`: the agent's workspace root, or `null`; see
  [Work in a directory](python-api.md#work-in-a-directory).
- `approval`: `true` when the run has an approval policy.

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
  parser leaves it only there.
- `tool_calls`: the calls of the reply, each as
  `{id, name, arguments, preview}`. `preview` is as on `tool.started`.
- `usage`: the integer token counts of the reply, such as `prompt_tokens`,
  or `{}` when the server sent none.
- `duration_s`: how long the planning request took, its retries and the
  waits before them included. With persistence on, it also includes
  recording the reply in history, which can wait up to 5 seconds while
  another process writes to `history.db`.

Before the event is built, `thinking` and `content` pass a minimal
redaction of secrets, which replaces each with `[redacted]`:

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
| `python_exec` | The first non-blank line of the code, followed by ` (+<n> more lines)` when more lines follow. |
| `file_read` | The path. |
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
  (`user`, `session`, `policy` or `timeout`), `error` when the policy failed,
  or `withdrawn` when the run was cancelled while the policy decided. See
  [Approve tool calls](python-api.md#approve-tool-calls).
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
`candidates_refused`. No `memory` event is emitted in `legacy`. The CLI
ignores it.

### notice

| `kind` | When | `message` | `detail` |
|---|---|---|---|
| `stuck` | An attempt ends because the same tool calls or errors repeated. | `The same tool calls or errors repeated; the attempt ends.` | `{}` |
| `budget` | An attempt ends because it used its steps. | `The attempt used its step budget.` | `{}` |
| `empty_reply` | The model gave an empty reply, and localmelo asks it to continue. A second empty reply in a row ends the run as `failed` instead, with no notice. | `The model returned an empty reply; asking it to continue.` | `{"count": <empty replies in a row>}` |
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
  default `"flags"`, such a reflection is trimmed and keeps its choice.
- `continue_run`: `true` only when another attempt starts. It is `false`
  after `done`, after the last attempt or once `max_steps` is used up, and
  when the continuation gate stops the run.
- `rationale`, `summary` and `best_effort_result`: the reflection's
  reasons, its summary of the attempt, and its best answer so far. They
  pass the same redaction of secrets as `assistant.message`'s `thinking`
  and `content`; the `RunResult` keeps the best answer as the model wrote
  it.
- `requests`: the number of reflection requests, 1 or 2. A reply that holds
  no reflection is asked for once more, without tools; the second request
  gets no event of its own. In the compress modes the prompt fit can
  refuse a request before it is sent, which reads as an unreadable reply:
  `requests` is then 0, or 1 when only the retry was refused.
- `readable`: `false` when, with the default `reflection_gate="flags"`, the
  second reply still held no reflection. `requests` is then 2; `action`,
  `rationale`, `summary` and `best_effort_result` are `null`; and
  `continue_run` is `true`, unless no attempt could start, in which case the
  run ends as `budget_exhausted`. With `"utility"` it is always `true`: such
  a reply counts as `stop`, with the rationale
  `Could not parse reflection response`, or
  `Reflection response was not a JSON object` when it was valid JSON that
  is not an object.
- `usage` and `duration_s`: the token counts and the time of both requests,
  their retries included. With persistence on, `duration_s` also includes
  recording the replies in history, which can wait up to 5 seconds while
  another process writes to `history.db`.

See [Reflection](how-it-works.md#reflection) and
[Run results](run-results.md#statuses).

### run.progress

Where the run stands, for a progress display. It comes only when
`run_task` gets `progress=True` and a sink. A session turn's sink is its
runner's, so the turn stores and forwards `run.progress` events only when
its `SessionRunner` was built with `progress=True` (see
[Sessions](sessions.md#run-a-conversation)). Every key is always present,
`null` where it is unknown. What it shows rests on the run's routine, as
[procedural memory](procedures.md) remembers it on this model with these
options:

| Key | Meaning |
|---|---|
| `level` | What may be shown: 0, facts only (a first run, another model or other options); 1, past runs' facts and named steps as well. Level 2, a bar, is not shown in this release. |
| `state` | `running`, then `ended` on the event right before `run.finished`. |
| `step`, `attempt` | Those of the latest `step.started`; 0 before the first step. |
| `limits` | `{max_steps, steps_per_attempt, max_attempts, timeout_s}`: the run's budgets, `null` where the run has none (the attempt limits with `attempt_policy="none"`). `max_steps` is a budget, not the expected length. |
| `phase`, `cycle` | `Explore`, `Write`, `Verify` or `Finishing`, from the classes of the run's tool calls (see [Classes and phases](procedures.md#classes-and-phases)), and the number of tests after a write; `null` and 0 before the first call. |
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
that fails or takes longer gives level 0. A failure inside the tracker
ends the run's `run.progress` events, never the run.

### run.finished

- `result`: the run's `RunResult`, as `RunResult.to_dict()`. Nothing in it
  is clipped, so `RunResult(**event.data["result"])` rebuilds it.

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
  pairs, in call order; then a `stuck`, `budget` or `empty_reply` notice.
  With the default `attempt_policy="reflect"`, the notice that ends an
  attempt is followed by `reflection.started`, the `retry` notices of the
  reflection's requests, and `reflection`, unless one of its requests
  failed or the run was cancelled. With `"none"`, it is followed by
  `run.finished`.
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
| `turn.started` | A turn begins, before its run's events. | `turn`, `turn_id`, `user_text`, `client_request_id` |
| `turn.finished` | The turn ends: its last event. | `turn`, `turn_id`, `status`, `error`, `cause` |
| `turn.undone` | A turn is undone; the event belongs to no turn. | `turn`, `turn_id` |
| `session.title` | The first turn sets the session's title, or it is renamed. | `title` |
| `session.updated` | The session's model, workspace or `meta` changes. | `model`, `workspace`, `workspace_label`, `meta` |
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
`EVENT_FIELDS` in `localmelo.melo.contracts` maps each type to its keys, and
`SESSION_EVENT_FIELDS` in `localmelo.support.sessions.store` each session
type.
