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
  options (all four built-in tools and no MCP tools), except that it keeps
  its answers (`AgentOptions(record_answers=True)`).
- When `LOCALMELO_PERSIST_MEMORY` is set for the gateway, all sessions save
  to the same SQLite files; see [Memory](memory.md#keep-memory-between-runs).

## Web workspace

Open `http://127.0.0.1:8401/` in a browser for the web workspace:

1. Enter the URL of a chat server, and optionally of an embedding server.
2. Discover models, pick a chat model, and optionally select an embedding model
   or seed memory set. Embeddings and seed memories are disabled by default.
3. Create a chat, then send a message. Creating a chat never sends your draft.
   Answers are rendered as Markdown, with code blocks.

The sidebar supports chat search and switching. Drafts stay with each chat in
the current browser tab. The workspace restores live chats after a page reload
and polls for changes from other browser clients every 2.5 seconds while visible.

A chat's header has a level chip (P1 to P4, P2 by default) that sets how the
scheduler orders the chat's calls in front of other chats; it is kept in
memory and is not saved. Next to the title and the level chip, a machine line
shows a foreign GPU lock (`GPU lock: <holder> · <purpose>`, with
`(recorded)` or `(chats wait)`), `GPU lock: yours`, `GPU lock unreadable`,
`· ETA passed`, and the battery (`Battery 55% · discharging`). AC power with
no lock shows nothing. While a call waits (for a slot, the background cap or
a foreign lock), the header's status pill and the status label say
`Waiting: <reason>`, the sidebar says `● Waiting` and the chat's bubble says
the same; the reason is the scheduler's own code (`Resources`, `Priority`,
`Reservation`, …). A lock hold uses `Reservation` and, in `wait` mode, does
not count against the 600-second active-time bound.

Stop cancels the run. A model request in flight is abandoned, although the
model server may finish computing it, and a running `shell_exec` command or
`python_exec` child is killed with its process group. `python_exec` code runs
in a child process, so the gateway keeps answering other requests while it
runs. See [Tools and MCP](tools.md) for the tools' limits. The task is saved
with the status `cancelled`, and the chat takes the next message once that
save is done. A Stop that comes after the answer, while the task is being
saved, leaves the answer in the chat. Each message sees the earlier answers
of its chat, as in a [session](#sessions).

The web workspace works apart from the HTTP API above:

- It does not use the config file's backend settings. It talks to the URL you
  enter, so `extra_body` and `request_timeout_s` do not apply. It does read
  the `[scheduler]` table: `policy` (default `off`), `n_srv`, `r_f`, `n_fg`,
  the call log, and the lock and battery keys (`gpu_lock_file`, `lock_self`,
  `lock_gate`); see [Scheduler](scheduler.md#the-gate-on-chats-s2) and
  [The GPU lock and the battery](scheduler.md#the-gpu-lock-and-the-battery-s2b).
  Chat requests wait up to 300 seconds, and a task is stopped after 600
  seconds of active time: a call the scheduler holds, including a foreign-lock
  hold, does not count against that bound.
- In `lock_gate = "record"` (the default) a foreign lock is shown and logged
  but nothing waits. In `wait`, every call a chat makes waits while a foreign
  lock is present, embeddings included, and a create that would seed memories
  answers HTTP 409 `{"error": "GPU lock held by <holder>; …"}` without sending
  an embedding or keeping a half-built chat. A create without seeding makes no
  model call and proceeds.
- Each chat's agent uses the default run options (all four built-in tools
  and no MCP tools), except that it keeps its answers. A chat can therefore
  run shell commands and Python code, and read and write files.
- It keeps up to 12 chats and closes those idle for 2 hours when creating a
  chat. At capacity, an idle chat is replaced only after preparation succeeds;
  if all chats are busy, creation fails with HTTP 409.
- Each chat has its own working memory. All chats save to one history and,
  when an embedding model is selected, one long-term memory, in
  `~/.cache/localmelo/playground/`; this is always on.
- Its routes under `/v1/playground/` are internal and may change.

The workspace routes include `GET /v1/playground/sessions` for summaries,
`GET /v1/playground/session/{id}` for a transcript,
`POST /v1/playground/session/{id}/cancel` for cancellation,
`POST /v1/playground/session/{id}/level` with `{"level": 1..4}` to set a
chat's level, and `POST /v1/playground/presence` with `{"session_id": …}` for
the page's typing heartbeat (the focused chat is the scheduler's lane). A
session summary and transcript also carry `level` and `held`, the scheduler's
reason while one of the chat's calls waits (`""` otherwise). The
`GET /v1/playground/sessions` response adds a top-level `machine` object
(`lock`, `holder`, `purpose`, `run`, `eta`, `stale`, `readable`, `on_ac`,
`battery_pct`, `battery_state`, `band`, `gate`), or `null` when the playground
has no monitor; the per-session keys do not change. Running or deleting a busy
chat returns HTTP 409. The ordinary `/v1/agent/` API remains separate and
ungated.

To start only the workspace without the configuration wizard:

```bash
uv run python -m localmelo.support.gateway._test_webapp
```

It always listens on `http://127.0.0.1:8401/`, so it cannot run while the
gateway uses that port.

The workspace has these limits:

- Chats last as long as the gateway process. They are kept in memory and are
  not restored after a gateway restart.
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
