# Gateway and HTTP API

The gateway serves the agent over HTTP: an API for your own programs, and a
web workspace for running the agent in your browser. This page shows how to
start it, keep it private, call its API, use the web workspace, and run it at
login on macOS.

## Start the gateway

```bash
melo --serve
```

- When there is no config file yet, `melo --serve` runs the setup wizard
  first; see [Configuration](configuration.md#set-up-with-the-wizard).
  Without a terminal it cannot ask, so it prints
  `melo: no backend is configured; run melo --reconfigure in a terminal`
  and exits with code 1.
- `melo --serve --allow-unconfigured` starts the gateway without running
  the wizard even when no backend is configured or the config file has
  validation errors, so a client or the web console can inspect
  `GET /api/health` and save a configuration through `PUT /api/settings`.
  It cannot be combined with `--reconfigure`.
- `--base-url` and `--chat-model` are usage errors with `--serve`: the
  gateway uses the config file.
- It listens on `127.0.0.1:8401`, the `[gateway]` defaults of the config
  file. `--host` and `--port` change them for one run, as in
  `melo --serve --port 9401`.
- It prints the backends and its addresses:

  ```text
    Starting localmelo gateway ...
    Chat: ollama
    Embedding: none
    Gateway: http://127.0.0.1:8401
    Web UI:  http://127.0.0.1:8401/
  ```

- Starting does not contact the model server. Run a task to check it.
- Press Ctrl+C to stop the gateway.

## Keep the gateway private

> [!WARNING]
> The gateway has no authentication. Any program that can reach its port can
> run tasks with your tools and your account's permissions, and the web
> workspace makes the gateway fetch any URL it is given. Keep the default host
> `127.0.0.1`.

On the default host, only programs on your computer can reach the gateway.
Web pages can still send it requests through your browser, so the gateway
refuses requests that are not addressed to it:

| HTTP status | `code` | When |
|---|---|---|
| 421 | `misdirected_request` | The `Host` header is not `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>`. This stops a page on another site that reaches the gateway under its own domain name (DNS rebinding). |
| 403 | `foreign_origin` | A web page at another origin sent the request: a `POST`, `PUT`, `PATCH`, `DELETE` or `OPTIONS` request whose `Origin` is not `http://127.0.0.1:<port>`, `http://localhost:<port>` or `http://[::1]:<port>`, or a request under `/v1/` or `/api/` whose `Sec-Fetch-Site` is `cross-site` or `same-site`. |
| 413 | `too_large` | The request body exceeds 1 MiB (1,048,576 bytes). |
| 415 | `unsupported_media_type` | A request under `/api/` has a non-empty body without `Content-Type: application/json`. |

The body is the usual error with a `code`:

```json
{"error":"Host header must be 127.0.0.1:8401, localhost:8401 or [::1]:8401","code":"misdirected_request"}
```

A refused request under `/api/` uses [API v2](api.md)'s envelope instead:
`{"error": {"code": "misdirected_request", "message": "...", "details": {}}}`.

- Open the web workspace as `http://127.0.0.1:8401/` or
  `http://localhost:8401/`, with your port; with `host = "::1"`, use
  `http://localhost:8401/` or `http://[::1]:8401/`. Other addresses that
  reach the gateway, such as `http://0.0.0.0:8401/`, get 421.
- curl and scripts send no `Origin` header, so only the `Host` check applies
  to them.
- Every response tells the browser not to show it in a frame, with
  `X-Frame-Options: DENY` and a `Content-Security-Policy` that ends
  `frame-ancestors 'none'`, so a page on another site cannot embed the web
  workspace and trick you into clicking in it. A response with a policy of
  its own, such as a blob's `default-src 'none'; sandbox`, keeps one header
  with both. Every response also carries `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, `Cross-Origin-Opener-Policy: same-origin`,
  `Cross-Origin-Resource-Policy: same-origin` and
  `Permissions-Policy: camera=(), microphone=(), geolocation=()`.
- The checks are on when the host is `localhost` or a loopback IP address:
  `127.0.0.1`, `::1` or another address in `127.0.0.0/8`. Requests must
  still name the gateway as `127.0.0.1`, `localhost` or `[::1]`, so use one
  of these three as the host: on `127.0.0.2`, for example, your browser gets
  421 too.
- With `host = "0.0.0.0"`, or any other host, none of this applies, and the
  gateway logs a warning when it starts. With no checks and no
  authentication, anything that can reach the port, web pages included, can
  run tasks with your tools. This is not recommended.
- Remote access, through a reverse proxy or Tailscale Serve for example, is
  not supported until the gateway has authentication.

## HTTP API

Requests and responses are JSON. The examples use the default address. Any
route can also answer 421 or 403; see
[Keep the gateway private](#keep-the-gateway-private).

### API v2 (web console)

The web console's own API is under `/api/`. It is served only when the
gateway listens on a loopback address, and it is documented separately in
[API v2](api.md). API v2 only grows: endpoints, optional request fields and
response keys are added, never renamed or removed.

This version serves the reads — `GET /api/health`, `GET /api/settings`,
`GET /api/settings/key-status`, `GET /api/models`, the sessions list and
one session, a session's turns, its Server-Sent Events stream, its blobs,
its tools, memory and runs, and a submission lookup — and the writes that
configure the gateway and run a session: `PUT /api/settings`,
`POST /api/settings/test`, create a session with or without a first turn,
start, retry and stop its turns, cancel a message by its key, answer an
approval, forget a session grant, rename or move a session, and delete it.
Scheduler tasks also have `POST /api/tasks`, `GET /api/tasks`, and
`GET` and `PATCH /api/tasks/{task_id}`; see
[Scheduler tasks](api.md#scheduler-tasks-s3a2) for their fields and controls.
The `/api` routes answer the same 421, 403 and 413 as `/v1/` (plus 415 for
a non-JSON body), and add `Cache-Control: no-store` and the error shape
`{"error": {"code", "message", "details"}}` to every response.

A console session runs its turns in the gateway process, one at a time: a
second turn while one runs is 409 `session_busy`. Its Agent is built when
the session is first used and kept; when a ninth Agent would open, the
least recently used idle one is closed with its conversation kept.
Approvals wait in the gateway's memory with no timeout until a client
answers them or the turn stops, and stopping the gateway ends a waiting
turn `cancelled` with cause `shutdown`. A console turn passes the
scheduler gate like a workspace chat: the gateway builds one gate from the
`[scheduler]` table and both surfaces use it, so their calls share one
`n_srv` budget and one call log, and the monitor the workspace page runs
feeds both (a foreign GPU lock holds a console turn under
`lock_gate = "wait"`; the battery is read for the page's chip and never
holds a call). The gateway closes that gate last at shutdown, after both
surfaces' sessions, so a turn waiting for its slot ends `cancelled` with
cause `shutdown`.

S3a.2 task workers use that same gate for model and embedding calls, with
memory off and gateway tools (no `shell_exec`). Up to four workers can be
resident; a paused running worker can still occupy a slot. Task rows survive
restart, but startup marks surviving nonterminal tasks failed with the reason
`gateway restarted; recovery arrives in S3b`. Full recovery, deadlines and worker
accounting are deferred. Workspace validation also runs before a worker starts
or resumes; a no-longer-valid folder fails the task rather than running it.
Pause waits for the next model call: running tools finish, and an answered
pending approval waits until resume. Resume restores the priority before pause,
or computed P2 if retention pruned the pause log; an unpaused task returns
409 `not_paused`. Task PATCH rejects `first` (400); it is deferred to S6b.

### GET /v1/health

```bash
curl http://127.0.0.1:8401/v1/health
```

```json
{"status":"ok","uptime":12.3,"sessions":{"active":0,"busy":0,"max":50},"chat_backend":"ollama","embedding_backend":"none"}
```

| Field | Meaning |
|---|---|
| `status` | `ok` while the gateway answers. |
| `uptime` | Seconds since the gateway started. |
| `sessions` | The number of `active` sessions, how many are `busy` running a task, and the `max`. |
| `chat_backend`, `embedding_backend` | The configured backends. |

The health check does not contact the model server.

### POST /v1/agent/run

Runs one task in a session and returns its result.

```bash
curl -X POST http://127.0.0.1:8401/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 6*7?", "session_id": "demo123"}'
```

```json
{"result":"42","session_id":"demo123"}
```

| Request field | Required | Description |
|---|---|---|
| `query` | yes | The task, up to 100,000 characters after leading and trailing whitespace is removed. |
| `session_id` | no | 1 to 64 letters, digits, `_` or `-`. Uses that session, and creates it if needed. Without it, a new session is created. |

| HTTP status | Body | When |
|---|---|---|
| 200 | `{"result": "…", "session_id": "…"}` | The run ended, whatever its status. |
| 400 | `{"error": "…"}` | The query is empty or too long, or the `session_id` is not valid. |
| 409 | `{"error": "…", "session_id": "…"}` | Rare: the session was closed just before this request could start. |
| 422 | `{"detail": […]}` | The body is not a JSON object. |
| 500 | `{"error": "…", "session_id": "…"}` | An unexpected error in the gateway. |

The 400 errors read `Query must be a non-empty string`,
`Query too large: <n> chars (max 100000)` or
`Invalid session_id format: '<id>'`; an id that is not a string gets the
last one too.

`result` is the answer or the run's best effort. When that is empty, it is
`[<status>] <error>`, for example
`[provider_error] POST /chat/completions failed with HTTP 500: …`. A run
that did not complete still returns HTTP 200, and the response has no status
field; see [Run results](run-results.md#where-results-appear).

### GET /v1/sessions

```bash
curl http://127.0.0.1:8401/v1/sessions
```

```json
{"sessions":[{"session_id":"demo123","status":"idle","idle_seconds":4.2,"created_at":1792732915.88,"is_busy":false}]}
```

| Field | Meaning |
|---|---|
| `session_id` | The session's id. |
| `status` | `idle` or `running`. |
| `idle_seconds` | Seconds since the session was last used. |
| `created_at` | When the session was created, in seconds since 1970 (Unix time). |
| `is_busy` | `true` while a task runs in the session. |

### DELETE /v1/sessions/{session_id}

```bash
curl -X DELETE http://127.0.0.1:8401/v1/sessions/demo123
```

The answer is HTTP 200 with `{"closed": "demo123"}`, or HTTP 404 with
`{"error": "session not found"}`. A session whose task is still running is
left alone: the answer is HTTP 409 with the same error body as a refused run,
`{"error": "Session demo123 is running a task; wait for it to finish before closing it.", "session_id": "demo123"}`,
and the task goes on. `is_busy` in [GET /v1/sessions](#get-v1sessions)
shows which sessions are busy.

The `/v1/agent` API has no route that stops a run, so wait for it to finish
and close the session after. The web workspace's cancel route,
`POST /v1/playground/session/{id}/cancel`, stops a chat of the web workspace
only ([Web workspace](#web-workspace)); it does not reach an `/v1/agent` run.
Stopping the gateway still closes every session, a busy one included; see
[Stop the gateway](#stop-the-gateway).

## Sessions

A session is one agent with its own working memory, so a task sees the
earlier tasks of its session and their answers. Each answer is kept without
its `<think>` block and cut to 8,000 characters, but a long session still
makes larger requests: leave out `session_id` for an unrelated task.

- Without a `session_id`, each request gets a new session with a generated
  id of 12 hex characters.
- A session runs one task at a time. A second request for a busy session
  waits until the first one finishes.
- The gateway keeps up to 50 sessions. Creating one more closes the least
  recently used idle session; if no session is idle, the new one is created
  anyway.
- Creating a session also closes the sessions that have been idle for more
  than 1 hour.
- Each session's agent uses the config file's backends and the default run
  options (three built-in tools, `file_read`, `file_write` and
  `python_exec`, and no MCP tools), except that it keeps its answers
  (`AgentOptions(record_answers=True, mode="gateway")`). A session does not
  offer `shell_exec`: legacy `/v1` has no approval policy, so there is no one
  to ask.
- When `LOCALMELO_PERSIST_MEMORY` is set for the gateway, all sessions save
  to the same SQLite files; see [Memory](memory.md#keep-memory-between-runs).

## Web workspace

Open `http://127.0.0.1:8401/` in a browser for the web workspace. It is the
console's page and talks only to `/api` ([API v2](api.md)), so the gateway
must be running to serve it: `melo --serve`, then open the address it prints.

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
runs. See [Tools and MCP](tools.md) for the tools' limits. The turn is saved
with the status `cancelled`, and the chat takes the next message once that
save is done. Each message sees the earlier answers of its chat, as in a
[session](#sessions).

A tool call that needs approval shows a card below its `Needs approval` tool.
Console sessions use `RememberingPolicy` and ask before `file_write` and
`python_exec`; legacy `/v1` has no approval policy. Approvals are not a
security boundary: any local process that reaches the unauthenticated API
can answer them. When a request is present, choose
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

The web workspace's earlier routes under `/v1/playground/` are kept as they
were for compatibility and are still internal; the page no longer uses them.
They include `GET /v1/playground/sessions` for summaries,
`GET /v1/playground/session/{id}` for a transcript,
`POST /v1/playground/session/{id}/cancel` for cancellation,
`POST /v1/playground/session/{id}/level` with `{"level": 1..4}` to set a
chat's level, and `POST /v1/playground/presence` with `{"session_id": …}` for
the page's typing heartbeat. The ordinary `/v1/agent/` API remains separate
and ungated.

To serve the page's files and the legacy playground routes without the
configuration wizard:

```bash
uv run python -m localmelo.support.gateway._test_webapp
```

It always listens on `http://127.0.0.1:8401/`, so it cannot run while the
gateway uses that port. It mounts no `/api`, so the page itself needs the
gateway; this server only reaches the `/v1/playground/` routes.

The workspace has these limits:

- It is for loopback use only. Like the rest of the gateway, it has no
  authentication: open it on this computer as `http://127.0.0.1:8401/` or
  `http://localhost:8401/`. Remote access, from another device or through a
  proxy, is not supported; see
  [Keep the gateway private](#keep-the-gateway-private).

## Run the gateway at login (macOS)

```bash
melo --daemon install
melo --daemon status
melo --daemon uninstall
```

- `install` writes `~/Library/LaunchAgents/ai.localmelo.gateway.plist` and
  loads it. launchd then runs `<python> -m localmelo --serve --port 8401`
  with the Python of your environment, starts it at login, and restarts it
  whenever it exits (at most once every 3 seconds).
- `install --port N` uses another port. The address is `[gateway] host` from
  the config file; `--host` is ignored.
- Logs go to `~/.localmelo/logs/gateway.stdout.log` and
  `~/.localmelo/logs/gateway.stderr.log`.
- `status` prints whether the service is `installed` and `running`, and its
  `pid`.
- Save a configuration before `install`, with `melo` or the setup wizard in
  a terminal. The service cannot answer questions: without a config file it
  writes `melo: no backend is configured; run melo --reconfigure in a terminal`
  to `gateway.stderr.log` each time launchd starts it.
- launchd starts the service without your shell's environment variables, so
  cloud API keys and `LOCALMELO_PERSIST_MEMORY` are missing. Use a local
  backend with the service.
- The service needs macOS; on other systems `install` fails.

## Stop the gateway

- In its terminal, press Ctrl+C.
- For the background service, run `melo --daemon uninstall`. A plain `kill`
  does not help: launchd starts the gateway again.

Stopping closes every session, a session running a task included: if one is
still busy, its agent is closed and the task ends as `crashed`. SIGTERM
first waits up to 3 seconds for runs in flight, then cancels the ones left:
the client's request fails and the run ends `crashed`. A `DELETE` request no
longer closes a busy session: it returns 409 until the task ends.
