# Guide: Sessions

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
- **`routine`**, on `run_turn` and `start_turn`, names the run's procedural
  routine ([`run_task`'s `routine=`](python.md#follow-a-run-with-events));
  it is passed to the run only when given.
- **Progress.** A runner built with `progress=True` asks each turn's run
  for [`run.progress`](../reference/events.md#runprogress) events: they are stored with
  the turn and reach `on_event` like the rest. The flag stays on the
  runner, `replace_agent` included, so every later turn asks too. Without
  it a session turn reports none, as before; the interactive UI always
  requests progress events.
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
| `before_turn` | `None` | Async hook after the turn's lease is acquired, before the model runs. It receives a zero-argument async reload function; await that function if another terminal changed the conversation, and do not retain it. The CLI uses this boundary to avoid a finish-between-check-and-lease race. |
| `progress` | `False` | Ask each turn's run for [`run.progress`](../reference/events.md#runprogress) events; kept on the runner, `replace_agent` included. |

## Resume

`await runner.resume(max_turns=10)` loads the stored conversation into the
runner's agent and returns the number of turns it loaded. It raises
`SessionBusyError` during a turn, or `UnknownSessionError` (a
`SessionStoreError` subclass) if the session was deleted. Store operations
that require an existing session raise the same typed error; `list_turns`
keeps returning an empty list for an unknown ID. `TurnInfo.typed_text`
exposes the original typed text passed as `title=`, separately from the
composed `user_text`; `/retry` keeps both after a restart.
`resume_mode` chooses how:

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
switch restore modes at any time. This does not change the Agent's reasoning
mode: that comes from the current process's config/flags, never from an old
turn's `RunResult.config`.

Pairs contain no old thinking. A window checkpoint includes redacted
thinking until retention clears it: replay modes send the retained part,
subject to their replay limits, while `drop` and `compress` strip both the
thinking field and a legacy inline `<think>` copy from request copies.
The stored conversation stays intact. `turns.note` remains a stored digest,
not a restored RAM-only note store; a window may retain the original model
`<note>` text without reconstructing note entries.

`pairs` stays the default until a benchmark decides between the two with
`scripts/resume_ab.py`. `window` becomes the default only if, over 90 paired
interrupted states, its next-step success beats `pairs` by an exact
two-sided McNemar test at p < 0.05, its recall of facts from earlier answers
is at most 5 points lower, and its median prompt tokens are at most 1.5
times those of `pairs`. A change of the default updates this page.
