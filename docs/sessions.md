# Sessions

A session is a conversation that is kept in SQLite: each turn begins with a
stored row, outlives the request that started it, and ends exactly once with
a status and a cause, also when the process that ran it stops or dies. A
fresh agent can then resume the conversation. `localmelo.support.sessions`
provides the store, a runner for an agent's turns, approval policies for a
person, an event stream and a read-only change feed.

> [!NOTE]
> Sessions are part of the Python API, which is experimental and may change
> before v0.1.0. Neither `melo` nor the gateway uses them yet: direct mode,
> `melo ""` and the gateway's `/v1` API run as before. The planned
> interactive CLI and web console will keep their conversations here.

## Run a conversation

```python
import asyncio
import tempfile
from pathlib import Path

from localmelo import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore


def new_agent() -> Agent:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    return Agent(llm=llm, options=AgentOptions(workspace_root=".", record_answers=True))


async def main() -> None:
    store = SessionStore(Path(tempfile.mkdtemp()) / "sessions.db", writer="cli")
    agent = new_agent()
    await agent.start()
    runner = SessionRunner(
        agent, store, None, new_session=NewSession(workspace=agent.workspace_root)
    )
    try:
        result = await runner.run_turn("What is the capital of France?")
        print(result.status, result.result)
        session_id = runner.session_id
        await runner.close()
        await agent.close()

        # Later, or in another process: a fresh agent resumes the session.
        agent = new_agent()
        await agent.start()
        runner = SessionRunner(agent, store, session_id)
        print("resumed", await runner.resume(), "turn(s)")
        await runner.run_turn("How many people live there?")
        for turn in await store.list_turns(session_id):
            print(turn.turn, turn.status, turn.cause, turn.answer)
    finally:
        await runner.close()
        await agent.close()
        await store.aclose()


asyncio.run(main())
```

It prints, for example:

```text
completed Paris is the capital of France.
resumed 1 turn(s)
1 completed run Paris is the capital of France.
2 completed run About 2.1 million people live in Paris itself.
```

- **The agent** needs `AgentOptions(record_answers=True)`, so that each answer
  stays in its working memory, and a `workspace_root`. `SessionRunner`
  raises `ValueError` for any other agent. Call `await agent.start()` before
  you pass the agent to a runner.
- **One runner per session and per agent.** A runner runs one turn at a time:
  `run_turn` and `start_turn` raise `SessionBusyError` while a turn runs. A
  turn started while `runner.undo()` or `runner.resume()` loads the
  conversation waits until it is loaded.
- **`run_turn`** runs the turn in your task and returns its `RunResult`,
  also when the runner stopped the turn (`cancel()`, `close()`, a takeover):
  then the `cancelled` one, with no counters when the turn was stopped before
  its run began. A cancellation of your task from outside the runner
  propagates once the turn has ended, or at once when it comes while the
  turn ends, which then ends on its own. **`start_turn`** returns the turn's
  `TurnInfo` once its row is stored, and the turn runs on in a task of the
  runner's; `await runner.wait()` returns its `RunResult`, the same one
  `run_turn` would return.
- **`title`**, on `run_turn` and `start_turn`, is the text the person typed.
  A session's first turn sets its title from it, else from the turn's text,
  so a message with an attached file still gets a short title.
- **Listeners.** `turn.started` and `turn.finished` are the store's own
  events, not the run's, and so is an `approval.resolved` that a turn's end
  writes for a request whose answer was not stored: `store.add_listener(fn)`
  calls `fn` with every stored event, those included, once its transaction
  has committed.
- A runner with `None` and `new_session=NewSession(...)` creates the session
  in the same transaction as its first turn, so a refused first turn leaves
  no empty session. `runner.session_id` is set once that turn has begun.

`SessionRunner` arguments:

| Argument | Default | Meaning |
|---|---|---|
| `agent`, `store`, `session_id` | required | The agent, the store, and the session's id, or `None` with `new_session`. |
| `new_session` | `None` | The session that the first turn creates: `NewSession(model="", workspace=None, meta=None, kind="chat")`. |
| `on_event` | `None` | Called with each event of a turn's run, and with its `approval.*` events, once it is committed, as a `StoredEvent` with its id (`None` for an event the store could not record). It may return an awaitable, which is awaited for at most 5 seconds. |
| `holder` | this process | The lease holder, `"<host>:<pid>:<suffix>"`. |
| `resume_mode` | `"pairs"` | How [`resume()`](#resume) loads a conversation: `"pairs"` or `"window"`. |

## The two files

One store implementation serves two files. Each surface opens only its own:

| | The gateway's file | The CLI file |
|---|---|---|
| Opened with | `SessionStore(writer="gateway")` | `SessionStore(writer="cli")` |
| Default path | `~/.cache/localmelo/gateway/sessions.db` | `~/.cache/localmelo/cli/sessions.db` |
| Writers | One gateway process, which holds an owner lock | Every terminal `melo` of the user; a lease per running turn keeps two of them from running turns of one session at once |
| Turns found running when it opens | All ended by recovery before `open()` returns | Ended once their holder's process is gone, or, for a holder on another host, once its lease expired |
| `PRAGMA synchronous` | `FULL`: an OS crash loses nothing committed | `NORMAL` |
| Event stream (`SessionHub`) | Yes | No |

- **A writer marker.** A new file records its writer. Opening the other
  writer's file raises `SessionStoreError`, for example
  `<path> belongs to the gateway (writer 'gateway'); open it with writer='gateway' or use the default path`;
  a file whose marker is missing or unknown raises
  `<path> has no valid writer marker`; and a file that is not a sessions
  file, such as `history.db`, raises `<path> is not a localmelo sessions file`.
  The store reads these on a read-only connection before it writes anything,
  so a refused file is left byte for byte as it was; beside a file in WAL
  mode, SQLite may still create its `-wal` and `-shm` files. A file of a
  newer schema raises
  `<path> has schema version <n>; this localmelo reads version 1`.
- **The owner lock.** The gateway's store holds a lock on `<path>.owner` for
  its life. A second gateway on the file gets `StoreOwnedError`:
  `Another melo gateway (pid <n>) is using <path>. Stop it to use the console here.`
  The system releases the lock when the process ends in any way, so the next
  gateway can start at once after a crash. The file must be on a local disk.
- **Private files.** Missing directories are created with mode `0700`, and
  the file with mode `0600`. The writer's own directory (`gateway/` or `cli/`)
  is set to `0700` when it is looser; a directory you chose keeps its mode,
  with one WARNING when others can read it.
- **Opening.** The store opens on its first awaited call, or with
  `await store.open()`. Every open failure raises `SessionStoreError` (or
  `StoreOwnedError`) and can be retried: a file is refused before anything
  is written to it, and each write of the open is a transaction of its own,
  which a failure, such as a full disk, rolls back; the steps before it,
  such as turns already recovered, stay done. After
  `await store.aclose()`, every call raises
  `SessionStoreError("The session store is closed")`, also an open that was
  still under way.
- **Retention.** Opening the store also clears the thinking, the `<think>`
  blocks, the window checkpoints and the withdrawn messages that are older
  than 30 days (`thinking_retention_days`); turns and events stay. See
  [Memory](memory.md#sessions).

## Turns, statuses and causes

A turn's `status` is `running` until it ends, then a
[run status](run-results.md#statuses). Its `cause` says what ended the turn,
and is `""` while it runs:

| `cause` | When | Status |
|---|---|---|
| `run` | The run ended by itself, also with `timeout` or a crash inside the run (a bug), and when a Stop came after the run had ended. | any |
| `user_stop` | A person stopped it: `runner.cancel()`. | `cancelled` |
| `closed` | The session's runner was closed: `runner.close()`, as when a session is deleted. | `cancelled` |
| `shutdown` | The process was asked to stop: `runner.close(cause="shutdown")` after `agent.begin_exit(...)`, or a cancellation from outside the runner, such as `asyncio.run`'s on Ctrl+C. | `cancelled` |
| `crash` | The process died; recovery wrote the end. | `crashed`, or `cancelled` when the stored `run.finished` said so |
| `takeover` | Another terminal took over a turn whose lease, held on another host, had expired (the CLI file only). | `cancelled` |
| `park` | Reserved for the scheduler. | `cancelled` |

- **Stopping.** `runner.cancel(reason, cause="user_stop", turn_id=None)`
  stops the running turn and returns whether it stopped one; with `turn_id`,
  only a turn with that id, so a late Stop never stops the next turn.
  `runner.close(reason="Session closed", cause="closed")` stops it and
  waits for at most 12 seconds. Either way the run ends `cancelled`, with a
  `RunResult` whose `error` is the reason (see
  [Stop a run](python-api.md#stop-a-run)).
- **After a crash.** A turn whose process died stays `running` on disk until
  recovery ends it. The gateway's store ends every such turn before `open()`
  returns. The CLI store ends one when it opens, when the session's next
  turn or undo begins, and in `recover_stale_turns()`: once its holder's
  process on this host is gone, or, for a holder on another host, once its
  lease has expired. Recovery withdraws the approvals nobody answered, and
  ends the turn with the outcome of its stored `run.finished`, if any; else,
  for another host's expired lease, as `cancelled` with the cause `takeover`
  and the error `The session was taken over by another process`, as the old
  terminal's run ends too; else as `crashed` with the error
  `The process running this turn stopped before it finished.`
- **"Interrupted" is a label, not a status.** A surface shows the causes
  `shutdown` and `crash` as "Interrupted"; the status stays a run status.
- **Write failures.** When the gateway's store cannot record a running
  turn's events or end, it refuses new work with
  `SessionStoreError("The session store could not record a change; restart the gateway")`
  until it restarts, and reads keep working; the restart ends the turn. With
  the CLI file, a runner tries a turn's end again after 0.5, 1 and 2 seconds
  (once only while it closes), and once more before the session's next turn
  or undo; until then the session stays busy.

Store methods for sessions and turns: `create_session`, `get_session`,
`find_sessions`, `list_sessions` (newest first, chats unless you pass
`kinds`), `rename_session`, `update_session`, `delete_session` (refused
while a turn runs), `list_turns(session_id, limit=, before=)`,
`list_events(session_id, after_id=, limit=, turns=)`, `undo_last_turn` and
`get_window`. `runner.undo()` undoes the newest finished turn and resumes;
a turn started meanwhile waits for the resume.

## Message ids and submissions

A client that sends a message over a network can name it with a
`client_request_id` before it knows whether the message arrived. The id is
1 to 128 characters of letters, digits and `. _ : -`, starting with a letter
or digit, such as `crypto.randomUUID()` gives. With an id, the store records
the message, its turn and `turn.started` in one transaction, so a client
that lost the response can ask what became of the message:

```python
import asyncio
import tempfile
from pathlib import Path

from localmelo import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore
from localmelo.support.sessions.submissions import SubmissionRepeated

KEY = "8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07"


async def main() -> None:
    store = SessionStore(Path(tempfile.mkdtemp()) / "sessions.db", writer="gateway")
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm, options=AgentOptions(workspace_root=".", record_answers=True))
    await agent.start()
    runner = SessionRunner(agent, store, None, new_session=NewSession())
    try:
        turn = await runner.start_turn("Summarize my notes", client_request_id=KEY)
        await runner.wait()
        try:  # the same message again, as a client retries
            await runner.start_turn("Summarize my notes", client_request_id=KEY)
        except SubmissionRepeated as repeated:
            print("already accepted as turn", repeated.turn.turn)
        submission = await store.get_submission(turn.session_id, KEY)
        print(submission.state, submission.status, submission.cause)
    finally:
        await runner.close()
        await agent.close()
        await store.aclose()


asyncio.run(main())
```

It prints:

```text
already accepted as turn 1
accepted completed run
```

| State | `get_submission` returns | Meaning |
|---|---|---|
| unknown | `None` | The message never arrived, and nothing ran: a retry with the same id is safe. |
| `accepted` | A `SubmissionInfo` with the turn | The message started that turn; `turn_id`, `status` and `cause` are the turn's. |
| `withdrawn` | A `SubmissionInfo` without a turn | `withdraw_submission` cancelled the message before it was accepted. |

- **A repeat runs nothing.** `start_turn` (and `store.begin_turn`) with an
  accepted id and the same text raises `SubmissionRepeated`, whose `turn` is
  the stored turn, also while that turn runs; with another text, it raises
  `SubmissionConflict`; with a withdrawn id, `SubmissionWithdrawn`. These
  come from `localmelo.support.sessions.submissions`, and are not
  `SessionStoreError`s. `check_submission(info, text)` applies the same rules
  to a submission you read.
- **`store.withdraw_submission(session_id, client_request_id, *, user_text)`**
  cancels a message that was not accepted: it stores a withdrawal, which
  takes no turn number, writes no event and refuses the id later, also after
  a restart. For an accepted message it returns the submission as it is;
  stop its turn with `runner.cancel(turn_id=submission.turn_id)`.
- **Scope.** An id is unique per session. The first turn of a new session
  uses the create scope: `get_submission(None, key)`, and the session's own
  id finds it too.
- `run_turn` takes no id. `get_submission` raises `SessionStoreError` for an
  unknown session, so a deleted chat is not mistaken for a lost message.

## Resume

`await runner.resume(max_turns=10)` loads the stored conversation into the
runner's agent and returns the number of turns it loaded. It raises
`SessionBusyError` during a turn. `resume_mode` chooses how:

- **`pairs`**, the default: the newest finished turns that are not undone,
  each as the user's text and the stored answer, at most 10 turns and 32,000
  characters. A crashed turn reads
  `[no answer: the run ended with status crashed]`.
- **`window`**: the working window that the agent held at the end of the
  latest turn with a checkpoint, tool calls and results included, then the
  later turns as pairs. Without a usable checkpoint (none, one of an unknown
  version, an undone turn, or one that retention deleted) it loads pairs.

`runner.last_resume` says what was loaded: `mode`, `fell_back` (a `window`
resume that loaded pairs), `turns`, `checkpoint_turns` (how many of `turns`
came from the checkpoint), `messages` and `chars`. Every turn that ends with
a `run.finished` stores a checkpoint, whichever mode is set, so you can
switch modes at any time.

`pairs` stays the default until a benchmark decides between the two with
`scripts/resume_ab.py`. `window` becomes the default only if, over 90 paired
interrupted states, its next-step success beats `pairs` by an exact
two-sided McNemar test at p < 0.05, its recall of facts from earlier answers
is at most 5 points lower, and its median prompt tokens are at most 1.5
times those of `pairs`. A change of the default updates this page.

## Approvals for a person

`localmelo.support.sessions.approvals` has two helpers for surfaces that ask
a person:

- **`RememberingPolicy(ask)`**, an approval policy: it asks through
  `ask(request)` and keeps "allow for the session" (an answer with
  `remember=True`) as a grant of the tool's name. `allowed_tools` lists the
  grants, and `forget(tool_name=None)` takes one or all back. Grants are
  never stored.
- **`ApprovalBroker(timeout_s=None)`** keeps the requests that wait for an
  answer from elsewhere, such as a web page: `asker(session_id)` is an ask
  function, `pending(session_id)` lists the waiting requests,
  `answer(request_id, decision)` returns `resolved`, `already_resolved` or
  `unknown`, and `withdraw_all(reason=...)` denies the waiting ones.

Wrap the ask function with `runner.observe(ask)`, as in
`RememberingPolicy(runner.observe(broker.asker(session_id)))`, and pass the
policy as `approval` to `run_turn` or `start_turn`: the session's events
then show `approval.requested` and `approval.resolved`.

## Stream a session's events

`SessionHub(store)` streams the events of the gateway's store, which is the
only one it takes:

```python
from localmelo.support.sessions.hub import SessionHub

hub = SessionHub(store)  # a SessionStore(writer="gateway")
async for event in hub.subscribe(session_id, after_id=last_event_id):
    print(event.id, event.event["type"])
```

It replays the stored events after `after_id`, then the new ones as they are
committed, with no gap and no repeat, also for a slow client. Event ids only
grow, also across restarts, so a client that reconnects with its last id
gets the events it missed, those that recovery wrote included. The session
events are listed in [Events](events.md#session-events).

## Export finished turns

`SessionReader(path)` reads either file read-only, also while its writer has
it open, and never writes to it. `changes()` returns one record per finished
or undone turn, in the order the store recorded them; the event id of each
record is a cursor that you keep:

```python
import asyncio
import io
import tempfile
from pathlib import Path

from localmelo import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM
from localmelo.support.sessions.reader import SessionReader, export_turns
from localmelo.support.sessions.runner import SessionRunner
from localmelo.support.sessions.store import NewSession, SessionStore


async def main() -> None:
    path = Path(tempfile.mkdtemp()) / "sessions.db"
    store = SessionStore(path, writer="cli")
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b")
    agent = Agent(llm=llm, options=AgentOptions(workspace_root=".", record_answers=True))
    await agent.start()
    runner = SessionRunner(agent, store, None, new_session=NewSession())
    await runner.run_turn("What is the capital of France?")
    await runner.close()
    await agent.close()
    await store.aclose()

    reader = SessionReader(path)
    for change in await reader.changes(after_event_id=0):
        print(change.event_id, change.change, change.status, change.answer)
    await reader.aclose()

    out = io.StringIO()
    cursors = await export_turns([path], out)
    print(len(out.getvalue().splitlines()), "line(s); cursor", cursors[str(path)])


asyncio.run(main())
```

It prints, for example:

```text
7 finished completed Paris is the capital of France.
1 line(s); cursor 7
```

- `changes(after_event_id=0, limit=500, kinds=None, with_events=False)`
  returns at most `limit` records (1 to 5,000). Each `TurnChange` holds the
  turn's fields as they read now, its session's kind, title, workspace and
  model, and the turn's `run_id`, the `task_id` of its run in `history.db`.
- A turn undone later appears again as an `undone` record, so applying the
  records in order is always right. Running turns appear once they end.
- `export_turns(paths, out, after=None)` writes one JSON line per record,
  `{"kind": "localmelo.session_turn", "v": 1, ...}`, and returns each path's
  new cursor; pass the cursors back as `after` next time.
- The reader takes no lock, runs no recovery and refuses a newer schema. A
  file of schema version 0, such as an empty one, has no records.

## Limits

| What | Limit |
|---|---|
| A turn's text | 1 to 100,000 characters |
| `client_request_id` | 1 to 128 characters |
| A title | 1 to 200 characters by hand; 60 when set from the text |
| Withdrawals | 1,000 per session, and 1,000 for new sessions' first turns; none older than 30 days |
| Window checkpoints | One per session, at most 2,000,000 characters |
| A turn's lease | 30 seconds, refreshed every 10 |
| `list_events` page | 1 to 1,000 events |
