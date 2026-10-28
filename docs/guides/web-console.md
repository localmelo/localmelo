# Guide: Web console

## Web workspace

Open `http://127.0.0.1:8401/` in a browser for the web workspace. It is the
console's page and talks only to `/api` ([API v2](../reference/api-v2.md)), so the gateway
must be running to serve it: `melo --serve`. Run `melo --gateway-login` in a
terminal and open its single-use link within 60 seconds (add `--port PORT`
for a nondefault port). Keep the link private; see [gateway security](gateway.md#keep-the-gateway-private).

1. Use **Connection settings** to choose the chat backend, model and workspace;
   local model discovery lists models without starting one. A cloud backend
   names an API-key environment variable and shows only whether it is set,
   never the key. Saving uses the loaded settings revision; if the file changed,
   reload the dialog and retry. **Test connection** probes the unsaved model
   settings; **Theme** applies System, Light or Dark immediately and remembers
   it in this browser. You can also use `melo --reconfigure`.
2. Type a message and send. The first message creates the chat: the page
   answers your draft with the chat's first turn, and never creates a chat on
   its own. Answers are rendered as Markdown, with code blocks.
3. A turn reads like the terminal's: your message, a work group with the
   tool cards it ran, its reasoning, its notices and reflections, the live
   row while it runs, and the final answer with `Copy response`. Output that
   a tool printed is cleaned of escape sequences, and its images (PNG, JPEG,
   GIF, WebP) show inline.

The sidebar lists chats newest first, with search and switching, and closes a
chat. Drafts stay with each chat in the current browser tab. The open chat's
turns arrive on its event stream (SSE); after a page reload the page restores
the chat, reads its last ten turns, and replays the stored events from there,
so nothing is lost or drawn twice. **Show earlier turns** loads older turns
without moving the current view. A stream the browser gives up on reopens
after 1, 2, 4, 8, 16 and then 30 seconds from the page's own cursor, and only
after the chat is read again: one that is gone leaves the page on a new chat
rather than retrying for ever. The list is re-read every 4 seconds while the
tab is visible, for changes from another browser. **Reconnect now** checks the
chat before reopening its stream. Independent banners show store, model and
connection problems without replacing other notices.

**Workers** opens read-only worker history from the CLI sessions file. Filter
by status, source or exact parent run, load more with the server's cursor, and
open a run to inspect its recorded brief, metadata, work and answer. An
unfinished run is a snapshot: **Refresh** reads it again, without a live
stream. The view cannot send messages, approve calls or change grants; chat
close and level controls are disabled while Workers is open. Connection
settings remain available separately. **Chats** returns to the open chat.
Records are redacted by patterns, not guaranteed secret-free; review before sharing.

**Details** opens a session drawer with its tools and session grants, working
and long-term memory status, and finished runs with paging when a page supplies
a turn cursor. Granted tools can be
revoked there; the tools view may open a paused session. On narrow screens the
drawer stacks below the composer.

A send is named by a `client_request_id`. If its answer is lost the page asks
the store whether the message was accepted, and offers `Retry` only when it
was not; a retry repeats the same key, so one message makes one turn. A
message that was never accepted goes back to that chat's draft. The outbox is
also kept in the tab's session storage, including its key, and a send that
was in flight at reload is checked against the store before a retry. It says
`Not sent` (or `Outcome unknown` when the store could not be asked), with
`Retry` and `Cancel and keep draft`.

A chat's header has a level chip (P1 to P4, P2 by default) that sets how the
scheduler orders the chat's calls in front of other chats; it is kept in
memory and is not saved. Next to the title and the level chip, a machine line
shows a foreign GPU lock (`GPU lock: <holder> · <purpose>`, with
`(recorded)` or `(chats wait)`), `GPU lock: yours`, `GPU lock unreadable`,
`· ETA passed`, and the battery (`Battery 55% · discharging`). AC power with
no lock shows nothing. While a call waits (for a slot, the background cap or
a foreign lock), the header's status pill and the status label say
`Waiting: <reason>`, the sidebar says `● Waiting` and the live row says the
same; the reason is the scheduler's own code (`Resources`, `Priority`,
`Reservation`, …). These are the same facts the terminal shows, read from and
written to the one gate the gateway shares:
`level` and `held` on a session, `machine` on the list, and
`POST /api/sessions/{id}/level` and `POST /api/presence` to set them.

Stop cancels the run. A model request in flight is abandoned, although the
model server may finish computing it, and a running `python_exec` child is
killed with its process group. `python_exec` code runs
in a child process, so the gateway keeps answering other requests while it
runs. See [Tools and MCP](../reference/tools.md) for the tools' limits. The turn is saved
with the status `cancelled`, and the chat takes the next message once that
save is done. Each message sees the earlier answers of its chat, as in a
[session](gateway.md#sessions).

A tool call that needs approval shows a card below its `Needs approval` tool.
Console sessions use `RememberingPolicy` and ask before `file_write` and
`python_exec`; legacy `/v1` has no approval policy. Approvals are not a
security boundary against a bearer holder or a malicious process under the same
OS account, which can answer them. When a request is present, choose
`Allow once`, `Allow every …` for the session, or `Deny…` with an
optional reason. The controls wait 500 ms before they can be used; arrival
does not move focus and no single-key shortcut answers a request. When the
card is outside the view, **Go to approval** opens its group and moves to it.
If approval details cannot be loaded, or arguments were truncated or converted,
allow is disabled; you can reload the details, deny, or stop the run.
A failed turn shows its error and details; the newest finished turn can offer
**Try again**, and **Continue** when it has run tools and its status permits
continuation. Stopped or interrupted turns use a neutral line instead.

The page's chats are sessions in the gateway's own session file,
`~/.cache/localmelo/gateway/sessions.db`; the terminal keeps its own in
`~/.cache/localmelo/cli/sessions.db`. They last until you close them: a
gateway restart keeps them, and a chat reopens with its transcript read from
the file, no Agent built yet (`not_loaded`); its Agent is built on the
conversation the file holds when the next turn or retry starts.

A chat runs in gateway mode: its Agent gets the gateway's default run options,
the three built-in tools `file_read`, `file_write` and `python_exec`, and no
MCP tools, except that it keeps its answers. It can read and write files and
run Python code; `shell_exec` is not offered, so there is no direct shell
command (Python code can still start a process). Its model is the gateway's
own, from the config file, and the page reads that file's `[scheduler]` table
too — `policy`, the lane sizes, the call log and the lock and battery keys.

In `lock_gate = "record"` (the default) a foreign lock is shown and logged but
nothing waits. In `wait`, every call a chat makes waits while a foreign lock is
present, embeddings included. A chat request waits up to 300 seconds, and a
turn is stopped after 600 seconds of active time: a call the scheduler holds,
including a foreign-lock hold, does not count against that bound.

Gateway security-1b is in progress. The internal `/v1/playground/` routes are
disabled by default (404), and their manager is not started. Opt in with
`melo --serve --enable-playground`; authentication and Host/Origin checks
still apply. Playground discovery and sessions use the saved local chat
backend and model, not endpoint or model overrides in a request. The page
does not use these routes.
They include `GET /v1/playground/sessions` for summaries,
`GET /v1/playground/session/{id}` for a transcript,
`POST /v1/playground/session/{id}/cancel` for cancellation,
`POST /v1/playground/session/{id}/level` with `{"level": 1..4}` to set a
chat's level, and `POST /v1/playground/presence` with `{"session_id": …}` for
the page's typing heartbeat. The ordinary `/v1/agent/` API remains separate
and ungated.

For development, the following starts the authenticated gateway with the
playground enabled, using the saved configuration:

```bash
uv run python -m localmelo.support.gateway._test_webapp
```

It starts the normal gateway, including `/api`, on its configured loopback
address. It needs the gateway configuration and authentication.

The workspace has these limits:

- It is for loopback use only. Sign in as described above and open it on this
  computer as `http://127.0.0.1:8401/` or
  `http://localhost:8401/`. Remote access, from another device or through a
  proxy, is not supported; see
  [Keep the gateway private](gateway.md#keep-the-gateway-private).
