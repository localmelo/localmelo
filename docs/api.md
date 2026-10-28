# API v2

API v2 is the web console's HTTP API. It is served under `/api/` on the
gateway, only when the gateway listens on a loopback address
(`127.0.0.1`, `localhost` or `::1`). It has no login: do not expose the
gateway. See [Gateway and HTTP API](gateway.md#keep-the-gateway-private).

The API only grows: endpoints, optional request fields and response keys
are added, never renamed or removed. A client may ignore keys it does not
know. This page documents the reads that exist now; creating, running,
stopping and approving turns, settings and model listing come in later
changes.

Requests and responses are JSON in UTF-8. A successful read carries
`Cache-Control: no-store`. A refused request carries one error shape:

```json
{"error": {"code": "not_found", "message": "Session not found.", "details": {}}}
```

`details` is `{}` unless the code says otherwise. These codes can answer on
every route below:

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | a malformed query parameter, or a malformed `client_request_id` |
| 403 | `foreign_origin` | a request under `/api/` from another site (`Sec-Fetch-Site` is neither `same-origin` nor `none`), or a state-changing request from another origin |
| 404 | `not_found` | an unknown session, turn, blob or route |
| 405 | `method_not_allowed` | a known path with a method it does not serve |
| 421 | `misdirected_request` | the `Host` header is not `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>` |
| 500 | `internal_error` | an unexpected failure; the message is `Internal error; see the gateway log.` and never a traceback or a value from the request |
| 503 | `store_unavailable` | the gateway's session store could not open, or stopped recording changes; restart the gateway in the second case |

The gateway logs the cause of a 500, whose response carries a fixed
message. A 503 repeats the store's reason with the home folder written as
`~`; it never carries a traceback, a lease holder or a key. The legacy
`/v1/*` routes keep their own bodies, including FastAPI's `422` and
`{"detail": ...}`.

## GET /api/health

Readiness, the API and event schema versions, the settings' model, the
cached model-server check, the limits and the store's state. Health is
served while the store is unavailable.

```bash
curl http://127.0.0.1:8401/api/health
```

```json
{
  "ok": true,
  "api": 2,
  "events_schema": 1,
  "version": "0.1.0",
  "configured": true,
  "config_errors": [],
  "model": {"backend": "ollama", "backend_label": "Ollama", "name": "qwen3:8b", "embedding": "none"},
  "model_server": {"url_label": "http://127.0.0.1:11434/v1", "reachable": true, "error": "", "checked_s_ago": 4},
  "limits": {"text_chars": 100000, "attachment_budget_chars": 8000, "body_bytes": 1048576},
  "sessions": {"open": 0, "running": 0, "pending_approvals": 0, "store": "ok", "store_error": "", "resume_mode": "pairs"},
  "uptime_s": 132.4
}
```

- `version` is the installed package's version, or `git describe` in a
  checkout.
- `configured` is whether the config file names a chat backend;
  `config_errors` is `Config.validate()` on the settings in use.
- `model.backend_label` is the backend registry's label. `model.embedding`
  is `none` when no embedding backend is configured.
- `model_server` is `null` for a cloud backend. For a local one it is the
  cached result of `GET <url>/models`: `reachable` is `null` until the
  first check ends, and `checked_s_ago` is the age of the result. The
  request never waits for the check; a stale result starts one in the
  background (after 30 s, or 5 s while the server does not answer).
- `limits` are the ingress text limit, the session text budget and the API
  body limit in bytes.
- `sessions.store` is `ok`, `unavailable` (the store could not open and the
  gateway retries every 5 s) or `failed` (a write failed; restart the
  gateway). `store_error` says why, with the home folder written as `~`.
- `sessions.resume_mode` is the resume arm in use; it is `pairs` today.

## GET /api/sessions

The newest sessions first. Only `chat` sessions are listed by default.

```bash
curl 'http://127.0.0.1:8401/api/sessions?query=notes&limit=50'
```

```json
{"sessions": [{"id": "3f9c2a7b1d4e", "kind": "chat", "title": "Release notes", "created_at": 1761141600.0, "updated_at": 1761141612.4, "model": "stub-model", "workspace": "/home/me/demo-project", "workspace_label": "~/demo-project", "busy": false, "running_here": false, "pending_approvals": 0}], "has_more": false}
```

- `limit` is 1 to 200 (default 50). `query` is at most 200 characters and
  matches a session's title or any turn's user text. `has_more` says that
  older sessions exist beyond the page.
- `kind` is `chat` (default), `front`, `front_bg` or `task`; any other
  value is 400 `invalid_request`.
- `workspace_label` is the workspace with the home folder written as `~`.
- `busy` says a turn is running. `running_here` equals `busy`: only this
  gateway runs turns in its own sessions file.
- `pending_approvals` counts this session's approval cards that wait for an
  answer.

## GET /api/sessions/{id}

One session, its pending approvals, its session grants, and its output
token total.

```bash
curl http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e
```

```json
{
  "session": {"id": "3f9c2a7b1d4e", "kind": "chat", "title": "Release notes", "created_at": 1761141600.0, "updated_at": 1761141612.4, "model": "stub-model", "workspace": "/home/me/demo-project", "workspace_label": "~/demo-project", "busy": false, "running_here": false, "pending_approvals": 0},
  "pending_approvals": [{"request_id": "0d6c2a9e8f7b41c3a5e9d1b3f5a7c9e1", "call_id": "call_1", "tool_name": "file_write", "arguments": {"path": "/home/me/demo-project/todo.md"}, "preview": "~/demo-project/todo.md", "workspace_root": "/home/me/demo-project", "workspace_label": "~/demo-project", "target": null}],
  "grants": ["shell_exec"],
  "granted_at": {"shell_exec": 1761141604.9},
  "mcp_error": "",
  "agent": "open",
  "tokens": 18512
}
```

- `grants` are the tools allowed for the rest of the session; `granted_at`
  maps each one to when the grant began.
- `mcp_error` is why this session runs without MCP tools, or `""`. It is
  always `""` today: MCP in the console is not built yet.
- `agent` is `open`, `paused` (closed, its conversation kept) or
  `not_loaded`. This PR does not open a session by itself, so a console
  that has not run a turn reads `not_loaded`.
- `tokens` sums the `completion_tokens` of the session's finished turns,
  the same field the answers report.
- `pending_approvals` entries carry `target: null` until the shared
  approval-target helper lands; API v2 only adds keys, so a later object
  is compatible.

## GET /api/sessions/{id}/turns

The session's turns, oldest first within the page.

```bash
curl 'http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/turns?limit=10&events=true'
```

```json
{"turns": [{"id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "turn": 3, "run_id": "4b1e7c9a2f3d4e5b8a6c0d1e2f3a4b5c", "user_text": "Summarize these notes", "status": "completed", "cause": "run", "answer": "The notes say ...", "result": {"task_id": "4b1e7c9a2f3d4e5b8a6c0d1e2f3a4b5c", "status": "completed", "result": "...", "error": "", "steps": 3, "attempts": 1, "n_llm": 3, "n_main": 3, "n_aux": 0, "n_tool": 4, "llm_errors": 0, "usage": {"prompt_tokens": 1180, "completion_tokens": 24, "total_tokens": 1204}, "duration_s": 12.2, "config": {}}, "undone": false, "started_at": 1761141600.2, "finished_at": 1761141612.4, "client_request_id": null, "running_here": false}], "has_earlier": false, "replay_from": 789}
```

- `limit` is 1 to 50 (default 10). Without `before`, the page is the newest
  turns and `replay_from` is the event id just before the oldest returned
  turn's `turn.started`, or 0 when the session has no turn. Pass it as the
  event stream's `after` to replay the page's events.
- With `before=N`, the page is the newest turns numbered below `N`, and
  there is no `replay_from`.
- `has_earlier` says that older turns exist beyond the page.
- `events=true` adds each turn's stored events as `events`, in id order,
  exactly as the event stream's `data` frames them.
- `undone` is `true` for a turn that `Try again` undid; such a turn is
  still listed.
- `answer` and `result.result` are the stored copies: secrets are redacted
  and, after retention, an inline `<think>` block is gone.
- `cause` is `""` while a turn runs.

## GET /api/sessions/{id}/events

A Server-Sent Events stream of the session's events: the stored ones after
the start point, then live ones.

```bash
curl -N 'http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/events?after=789'
# catch up once and stop:
curl 'http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/events?after=789&follow=false'
```

```text
retry: 2000

id: 790
data: {"id":790,"session_id":"3f9c2a7b1d4e","turn":3,"v":1,"type":"turn.started","run_id":"","seq":null,"ts":1761141600.2,"data":{"turn":3,"turn_id":"9c1f0e2d4b6a48c1a3e5f7091b2d4c6e","user_text":"Summarize these notes","client_request_id":null}}

: keepalive
```

- Each frame has one `id:` line and one `data:` line; there is no `event:`
  field. The JSON is the event envelope plus its durable `id`, `session_id`
  and `turn`.
- `retry: 2000` is sent once. `: keepalive` follows every 15 seconds of
  silence.
- `after` defaults to 0. The `Last-Event-ID` header wins over it when both
  are present, which is what a browser's `EventSource` sends when it
  reconnects; an `after` that would be invalid is not read then. Otherwise
  it must be decimal digits: negative, `+1`, `1_0` and non-numeric values
  are 400 `invalid_request`.
- `follow=false` sends the stored events after the start point, up to the
  newest event id the request read, and then ends the response; it is
  one-shot catch-up, and a turn that starts while it pages adds no frames.
- The stream ends when the client disconnects, when the session no longer
  exists (a reconnect then gets 404), and when the gateway shuts down. A
  browser reconnects with `Last-Event-ID` and receives every event it
  missed, once.

## GET /api/sessions/{id}/blobs/{sha256}

An image a turn's tool returned, by its SHA-256 digest.

```bash
curl -o shot.png http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/blobs/823ceb99fcef5252333ede1b2202341c3b287b6d47571963e6b0ddf393a24f82
```

- `sha256` is exactly 64 lowercase hex characters; anything else, an
  unknown digest and an unknown session are 404 `not_found`.
- `image/png`, `image/jpeg`, `image/gif` and `image/webp` are served inline
  with their type. Any other stored type, SVG included, is
  `application/octet-stream` with
  `Content-Disposition: attachment; filename="<sha256>.bin"`.
- Every response has `X-Content-Type-Options: nosniff`,
  `Content-Security-Policy: default-src 'none'; sandbox` and
  `Cache-Control: no-store`.

## GET /api/sessions/{id}/submissions/{client_request_id}

What became of a message the client named with a `client_request_id`. The
same lookup without a session is `GET /api/submissions/{client_request_id}`,
for a client that never learned the new session's id.

```bash
curl http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/submissions/8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07
```

```json
{"submission": {"client_request_id": "8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07", "state": "accepted", "session_id": "3f9c2a7b1d4e", "turn_id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "status": "completed", "cause": "run", "turn": {"id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "turn": 3, "...": "..."}, "error": "", "created_at": 1761141600.2}}
```

- A `client_request_id` is 1 to 128 of `[A-Za-z0-9._:-]`, starting with a
  letter or digit. A malformed one is 400 `invalid_request`.
- `state` is `accepted`, `withdrawn` or `unknown`. An unknown key reads 200
  with `state: "unknown"`, `error: ""` and every other key `null`: a retry
  with it is safe. A 404 here means the session is unknown or deleted,
  never that the key is unknown.
- For an accepted submission, `turn_id`, `status`, `cause` and `turn` are
  its turn's. For a withdrawn one they are `null`, and `error` is the
  store's withdrawal reason.
- The session route also finds the key that created the session's first
  turn, so a client that lost the response to the create request can ask
  either route.

## Not in this version

Turns cannot be started, stopped or retried through API v2 yet, approvals
cannot be answered, and there are no settings, model-list or tools routes.
They arrive in later changes; the shapes above are the ones those changes
extend.

FastAPI's `/docs`, `/redoc` and `/openapi.json` are off: this page is the
reference. Run the API only on the loopback address. A gateway on `0.0.0.0`
mounts no console and no `/api`, and logs a warning at startup; see
[Gateway and HTTP API](gateway.md#keep-the-gateway-private).
