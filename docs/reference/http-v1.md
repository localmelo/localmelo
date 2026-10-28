# Reference: Http v1

## HTTP API

Requests and responses are JSON. The examples use the default address. Any
route requires bearer authentication and can also answer 401, 421 or 403; see
[Keep the gateway private](../guides/gateway.md#keep-the-gateway-private).

### API v2 (web console)

The web console's own API is under `/api/`. It is served only when the
gateway listens on a loopback address, and it is documented separately in
[API v2](api-v2.md). Authentication is a compatibility change for existing
clients; API fields otherwise remain additive.

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
[Scheduler tasks](api-v2.md#scheduler-tasks-s3a2) for their fields and controls.
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

The examples read the gateway bearer token from its private file. A command-line
header can be visible to local processes; keep the token out of logs and history.

```bash
curl -H "Authorization: Bearer $(< ~/.cache/localmelo/gateway/auth/token)" http://127.0.0.1:8401/v1/health
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
curl -H "Authorization: Bearer $(< ~/.cache/localmelo/gateway/auth/token)" -X POST http://127.0.0.1:8401/v1/agent/run \
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
curl -H "Authorization: Bearer $(< ~/.cache/localmelo/gateway/auth/token)" http://127.0.0.1:8401/v1/sessions
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
curl -H "Authorization: Bearer $(< ~/.cache/localmelo/gateway/auth/token)" -X DELETE http://127.0.0.1:8401/v1/sessions/demo123
```

The answer is HTTP 200 with `{"closed": "demo123"}`, or HTTP 404 with
`{"error": "session not found"}`. In the in-progress security-1b changes, a session with a running task or queued start is
left alone: the answer is HTTP 409 with the same error body as a refused run,
`{"error": "Session demo123 is running a task; wait for it to finish before closing it.", "session_id": "demo123"}`,
and the task goes on. `is_busy` in [GET /v1/sessions](#get-v1sessions)
shows which sessions are busy.

The `/v1/agent` API has no route that stops a run, so wait for it to finish
and close the session after. The web workspace's cancel route,
`POST /api/turns/{turn_id}/cancel`, stops a console turn
only ([Web workspace](../guides/web-console.md#web-workspace)); it does not reach an `/v1/agent` run.
Stopping the gateway still closes every session, a busy one included; see
[Stop the gateway](../guides/gateway.md#stop-the-gateway).
