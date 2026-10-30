# API v2

API v2 is the web console's HTTP API. It is served under `/api/` on the
gateway, only when the gateway listens on a loopback address
(`127.0.0.1`, `localhost` or `::1`). Protected routes require a bearer
token or signed-in browser cookie; do not expose the gateway. See [Gateway and HTTP API](../guides/gateway.md#keep-the-gateway-private).

Apart from the authentication requirement, API fields remain additive:
endpoints, optional request fields and response keys are added. A client may ignore keys it does not
know. This page documents every API v2 route: health, settings, key status,
connection test, model listing, sessions, turns, SSE events, blobs,
submissions, approvals, tools, grants, memory, runs and scheduler tasks.

`POST /api/auth/login` exchanges a valid single-use code for a browser cookie;
`POST /api/auth/logout` revokes the cookie. `POST /api/auth/rotate` requires
the bearer and rotates it. The login exchange requires a matching Origin. When
the macOS app started the gateway, it also requires the app's
`X-LocalMelo-Instance` header, and a code is valid only for the instance it was
issued for; a missing, different or differently bound value is refused before
the code is consumed. See [gateway security](../guides/gateway.md#keep-the-gateway-private).

A console turn passes the same scheduler gate a workspace chat does: every
model call waits for its slot under `[scheduler]`, and the call log records
it under the session's id. See [Scheduler](../guides/scheduler.md#the-gate-on-chats-s2).

JSON request bodies and JSON responses use UTF-8. Event streams use
`text/event-stream`, image blobs use their image media type, and a successful
204 response has no body. A request with a non-empty body
needs `Content-Type: application/json`, and a body may have at most 1 MiB
(1,048,576 bytes). Every string in a request must be valid UTF-8 without
lone surrogates. Responses carry `Cache-Control: no-store`,
`Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; font-src 'self'; manifest-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`
(except blobs, whose policy is `default-src 'none'; sandbox; frame-ancestors 'none'`),
`X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`,
`Cross-Origin-Opener-Policy: same-origin`,
`Cross-Origin-Resource-Policy: same-origin`, `X-Frame-Options: DENY` and
`Permissions-Policy: camera=(), microphone=(), geolocation=()`. A refused
request carries one error shape:

```json
{"error": {"code": "not_found", "message": "Session not found.", "details": {}}}
```

`details` is `{}` unless the code says otherwise. These codes can answer on
every route below:

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | malformed JSON, a wrong type, an unknown field, a control character or lone surrogate, a relative `workspace` in settings, a bad query parameter, or a malformed `client_request_id` |
| 400 | `invalid_settings` | `Config.validate()` failed on the candidate settings; `details.errors` lists the messages |
| 403 | `foreign_origin` | a request under `/api/` from another site (`Sec-Fetch-Site` is neither `same-origin` nor `none`), or a state-changing request from another origin |
| 404 | `not_found` | an unknown session, blob or route |
| 405 | `method_not_allowed` | a known path with a method it does not serve; the `Allow` header names the methods it does |
| 409 | `settings_changed` | `config.toml` changed on disk since `GET /api/settings` read its `revision` |
| 413 | `too_large` | the request body is over 1 MiB |
| 415 | `unsupported_media_type` | a non-empty request body without `Content-Type: application/json` |
| 421 | `misdirected_request` | the `Host` header is not `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>`, or an app-started gateway's `/api/auth/login` has no matching `X-LocalMelo-Instance` |
| 500 | `internal_error` | an unexpected failure; the message is `Internal error; see the gateway log.` and never a traceback or a value from the request |
| 502 | `backend_unreachable` | listing models failed or returned no models |
| 503 | `store_unavailable` | the gateway's session store could not open, or stopped recording changes; restart the gateway in the second case |

The gateway logs the cause of a 500, whose response carries a fixed
message. A 503 repeats the store's reason with the home folder written as
`~`; it never carries a traceback, a lease holder or a key. The legacy
`/v1/*` routes keep their own bodies, including FastAPI's `422` and
`{"detail": ...}`.

## GET /api/health

Readiness, the API and event schema versions, the settings' model, the
cached model-server check, the limits, the store's state and what runs now.
Health is served while the store is unavailable.

The examples pipe the gateway bearer token from its private file to curl, keeping
it out of command-line arguments and shell history. Do not enable verbose curl
output or log the header; same-user processes can still read the token file.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/health
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
  "sessions": {"open": 0, "running": 0, "pending_approvals": 0, "store": "ok", "store_reason": null, "store_error": "", "resume_mode": "pairs"},
  "status": {
    "running": {"chats": 0, "tasks": 1, "runs": 1, "terminal": 1},
    "tasks": {"running": 1, "queued": 1, "held": 0, "top_level": 2},
    "items": [
      {"kind": "task", "id": "a1b2c3d4e5f6", "title": "Index the repo"},
      {"kind": "run", "id": "c44733ec77e5", "title": "Summarize the changelog"},
      {"kind": "terminal", "id": "d27a70f1bbd8", "title": "Debug the flaky test"}
    ]
  },
  "uptime_s": 132.4
}
```

- `version` is the installed package's version, or `git describe` in a
  checkout.
- `configured` is whether the config file names a chat backend;
  `config_errors` is `Config.validate()` on the settings in use.
- `model.backend_label` is the backend registry's label; for the `vllm` backend it
  is `oMLX` when the config file says `omlx` or a model listing of the server
  did, else `OpenAI-compatible server`. `model.embedding`
  is `none` when no embedding backend is configured.
- `model_server` is `null` for a cloud backend. For a local one it is the
  cached result of `GET <url>/models`: `reachable` is `null` until the
  first check ends, and `checked_s_ago` is the age of the result. The
  request never waits for the check; a stale result starts one in the
  background (after 30 s, or 5 s while the server does not answer).
  `url_label` is empty and `reachable` is `false` when `chat_url` is not an
  absolute `http` or `https` URL; such a value is never probed and its text,
  user info included, is not echoed.
- `limits` are the ingress text limit, the session text budget and the API
  body limit in bytes.
- `sessions.store` is `ok`, `unavailable` (the store could not open and the
  gateway retries every 5 s) or `failed` (a write failed; restart the
  gateway). `store_error` says why, with the home folder written as `~`.
  `store_reason` is `null` when healthy, `owned` when another gateway owns
  the store, `open-error` for any other open failure, or `failed` after a
  write failure. Clients should use this stable reason instead of parsing
  `store_error`; it clears when the store opens successfully.
- `sessions.resume_mode` is the resume arm in use; it is `pairs` today.
- `status.running` counts what runs now, by kind: `chats` (the console's chats
  with a turn in flight, as `sessions.running`), `tasks` (scheduler tasks in
  state `running`), `runs` (direct runs and `--brief` workers) and `terminal`
  (interactive `melo` chats). The last two come from the CLI file and count a
  turn only under a live lease, so a run whose process died is not counted.
  `status.tasks` is the scheduler's queue as the status row shows it:
  `running`, `queued`, `held` and `top_level` (the best level among them, or
  `null`). `status.items` lists what runs, chats first, then tasks, runs and
  terminal chats, at most 20, each `{"kind", "id", "title"}` (`kind` is `chat`,
  `task`, `run` or `terminal`; a terminal chat's title is redacted). The counts
  are not capped. A store that cannot be read counts as nothing running; one
  call reads the task states by their index and the running turns of the CLI
  file by theirs.
- `status.waiting` lists the chats and tasks with a tool-approval request that
  waits for an answer, chats first, at most 20, each `{"kind", "id"}` (`kind` is
  `chat` or `task`). The console opens them at `#/s/<id>` and `#/tasks/<id>`, and
  the macOS app's menu bar reads the list at each poll. `sessions.pending_approvals`
  counts the open chats' requests only.

## GET /api/settings

The saved configuration from `~/.cache/localmelo/config.toml`, without API
key values, plus the wizard's per-section defaults, backend choices and the
file's revision.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/settings
```

```json
{
  "chat_backend": "mlc",
  "embedding_backend": "none",
  "workspace": "~/localmelo-workspace",
  "sections": {
    "mlc": {"chat_url": "http://127.0.0.1:8400/v1", "chat_model": "qwen3-1.7b", "embedding_url": "", "embedding_model": ""},
    "openai": {"api_key_env": "OPENAI_API_KEY", "api_key_set": false, "chat_model": "gpt-4o", "base_url": ""}
  },
  "defaults": {
    "mlc": {"chat_url": "http://localhost:8400/v1", "chat_model": "qwen3-1.7b", "embedding_url": "http://localhost:8400/v1", "embedding_model": "nomic-embed"},
    "openai": {"api_key_env": "OPENAI_API_KEY", "chat_model": "gpt-4o"}
  },
  "errors": [],
  "choices": {
    "chat_backends": [{"key": "mlc", "label": "MLC (local endpoint)", "kind": "local"}, {"key": "openai", "label": "OpenAI", "kind": "cloud"}],
    "embedding_backends": [{"key": "mlc", "label": "MLC (local endpoint)"}, {"key": "none", "label": "None (no long-term memory)"}]
  },
  "mode": "default",
  "advanced_options": [
    {"option": "--host", "tier": "advanced", "issue": ""},
    {"option": "--retriever", "tier": "experimental", "issue": "#4"}
  ],
  "file_label": "~/.cache/localmelo/config.toml",
  "revision": "5d41402abc4b2a76"
}
```

- `sections` holds all four local sections (`mlc`, `ollama`, `vllm`,
  `sglang`) and all four cloud sections (`openai`, `gemini`, `anthropic`,
  `nvidia`). Cloud sections report `api_key_env` (the variable name) and
  `api_key_set` (whether that variable is non-empty in the gateway's
  environment), never a key or a mask of one.
- `workspace` is `[gateway] workspace`, or `~/localmelo-workspace` when it
  is unset.
- `revision` is the first 16 hex characters of the file's SHA-256, or `""`
  when the file does not exist yet.
- `mode` is `[ui] mode` (`default` or `developer`; any other saved value
  reads as `default`). `advanced_options` lists the advanced and experimental
  `melo` options from the tier table (`issue` is the deciding issue of an
  experimental one, `""` otherwise); the console shows them read-only in
  developer mode.

## PUT /api/settings

Save settings atomically to `config.toml` and apply them to new console
sessions.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X PUT http://127.0.0.1:8401/api/settings \
  -H "Content-Type: application/json" \
  -d '{"chat_backend":"ollama","embedding_backend":"none","workspace":"~/localmelo-workspace","sections":{"ollama":{"chat_url":"http://127.0.0.1:11434","chat_model":"qwen3:8b"}},"revision":""}'
```

- Every string must be a single line with no control characters, and
  `workspace` is required, must be an absolute path or start with `~/`, and
  must not resolve to `/` or the home folder (including through a symlink);
  otherwise 400 `invalid_request`. `api_key_set` is not accepted in
  `sections`.
- `mode` is optional: `default` or `developer` sets `[ui] mode` (other values
  are 400 `invalid_request`); omitted, the saved mode is kept. The mode changes
  only what the console and `melo --help` show.
- A `revision` that does not match the file on disk returns 409
  `settings_changed`. Two concurrent `PUT` requests that read the same
  `revision` can both pass the check; the last writer wins, and `os.replace`
  keeps the file from being torn.
- The server loads `config.toml` fresh so CLI `--host` and `--port`
  overrides are never written to disk, preserves other sections,
  `extra_body`, `mcp_config`, `request_timeout_s`, `[agent]` and
  `[scheduler]`, validates the candidate (`400 invalid_settings` with
  `details.errors` on failure), writes `config.toml` atomically, closes
  idle session Agents so the next turn opens with the new settings, and
  returns the updated `GET /api/settings` body.

## POST /api/settings/test

Check candidate settings with a one-message chat ping and, when an
embedding backend is selected, a one-word embedding ping, bounded to 30
seconds. Takes the `PUT /api/settings` body without `revision`. In the
in-progress security-1b changes, authenticated probes accept explicitly
typed absolute HTTP or HTTPS URLs and do not follow redirects; a redirect
is a failed probe. These probes can contact the supplied host.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/settings/test \
  -H "Content-Type: application/json" \
  -d '{"chat_backend":"ollama","embedding_backend":"none","workspace":"~/localmelo-workspace","sections":{"ollama":{"chat_url":"http://127.0.0.1:11434","chat_model":"qwen3:8b"}}}'
```

```json
{"ok": true}
```

- Field or validation errors return 400 first. A failed or timed-out probe
  returns HTTP 200 with `{"ok": false, "error": "..."}`.

## GET /api/settings/key-status

Whether an environment variable is non-empty in the gateway's environment.
Never returns the variable's value.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/settings/key-status?name=OPENAI_API_KEY'
```

```json
{"set": false}
```

- `name` must match `^[A-Za-z_][A-Za-z0-9_]{0,127}$`; any other value is 400
  `invalid_request`.

## GET /api/models

List the models of a local backend (`GET <url>/models` with a 5 s timeout).
In the in-progress security-1b changes, authenticated requests may supply an
absolute HTTP or HTTPS URL; redirects are not followed. The supplied host
can be contacted.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/models?backend=ollama&url=http%3A%2F%2F127.0.0.1%3A11434%2Fv1'
```

```json
{"models": ["qwen3:8b"]}
```

- Without `backend` or `url`, the saved chat backend and its `chat_url` are
  used.
- A cloud backend returns 400 `invalid_request`:
  `Listing models works for local backends only; type the model name.`
- An unreachable server or an empty model list returns 502
  `backend_unreachable`.

## GET /api/setup/detect

Look for a model server on this machine, as the terminal's first run does. It
asks `GET <url>/models` of each URL in `LOCALMELO_DETECT_URLS` with a 1 s
timeout (by default `127.0.0.1` ports 8000, 8400, 11434 and 30000; an empty
value turns detection off), then `GET <root>/api/version` of the server it
found, and, between several chat models, its status routes. It works before
anything is configured, takes no parameters and writes nothing. The web console's
setup card calls it, and only while the card is shown.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/setup/detect
```

```json
{"state": "found", "name": "oMLX", "backend": "vllm", "url": "http://127.0.0.1:8000/v1", "models": ["Qwen3-32B-4bit", "qwen3"], "loaded": ["qwen3"], "suggested": "qwen3", "problem": "", "start": {"name": "oMLX", "commands": ["brew tap jundot/omlx https://github.com/jundot/omlx", "brew install jundot/omlx/omlx", "omlx serve --model-dir ~/models --port 8000"], "note": "Put a model in its own folder of ~/models first.", "guide": "https://github.com/localmelo/localmelo/blob/main/docs/guides/model-servers.md"}}
```

- `state` is `found` (a chat model to offer), `needs_key` (a server answered 401
  or 403; localmelo sends no key to a local server) or `none`.
- `name` is `Ollama`, `oMLX` (its model listing names oMLX as the owner) or `""`.
  `backend` is the `chat_backend` to save: `ollama`, or `vllm`, which
  `PUT /api/settings` writes as `omlx` when a listing named oMLX; `""` unless
  `state` is `found`.
- `models` are the chat models in listing order (embedding and reranking models
  left out). `loaded` are those the server says are in memory; it is asked only
  between several chat models, and `[]` also means it did not say.
- `suggested` is the model to offer: the only chat model, or a loaded one. It is
  `null` when several are served and none is known to be loaded, and the client
  must let the user choose.
- `url` is empty when nothing answered. `problem` says why a server that answered
  cannot be used. A user name or password in a URL is never returned.
- `start` is in every answer: how to start a model server on this computer, from the
  same place as the terminal's first run. `name` is `oMLX` on a Mac and `Ollama`
  elsewhere, `commands` are the lines to run, `note` is the one thing to do besides
  running them, and `guide` is the model servers guide. The card shows them while
  `state` is `none`.

## Scheduler tasks (S3a.2)

These routes create and control user-origin tasks in the gateway process.
They are separate from chat turns but use the same scheduler gate. Each task
has a `kind="task"` session with the same id; its session events and turns
can be inspected through the existing session routes. A worker uses gateway
built-ins (`file_read`, `file_write`, `python_exec`, not `shell_exec`) with
memory off. Tools run with the user's permissions, not in a sandbox.

- `POST /api/tasks` accepts `objective` (or its alias `prompt`), optional
  `title`, `workspace`, `level` and `request_key`. The objective must be
  non-empty and at most 100,000 characters; title is at most 200 characters
  and defaults to a title derived from the objective. Level is an integer
  0–5 or `P0`–`P5`; omitted means computed P2. A supplied `deadline`, even
  null, is rejected; deadlines are not supported yet. Other fields are
  rejected. A missing, blank or overlong objective, overlong title or
  invalid level returns 400 `invalid_task`; schema/type errors return 400
  `invalid_request`. Returns 202 `{"task": ...}` for a new task. An explicitly
  empty `request_key` returns 400 `invalid_request`. A non-empty
  `request_key` makes a repeat of the same normalized request return 200
  `{"task": ..., "repeated": true}`; reusing it for a different request
  returns 409 `submission_conflict`. A repeat does not start a new worker.
- `GET /api/tasks` returns `{"tasks": [...]}`; optional `state` and integer
  `level` query parameters filter the rows. `GET /api/tasks/{task_id}`
  returns `{"task": ...}` or 404 `not_found`.
- `PATCH /api/tasks/{task_id}` accepts `level` (0–5, `P0`–`P5`, or
  `auto`) and/or `state` (`pause`, `resume`, or `stop`). It returns
  `{"task": ...}`. `first` is rejected with 400 `invalid_request`; it is
  deferred to S6b. Pause parks at P5; resume restores the priority before
  pause unless a new runnable level is supplied. The saved priority survives
  audit-log retention. If there is no saved snapshot (for example, a task
  created or set at P5, or a v1 pause whose log was already missing), resume
  falls back to computed P2. `auto` selects computed P2.
  Pause cannot be combined with a level, and resume cannot be combined with
  P5. An update with neither level nor state, or either invalid combination,
  returns 400 `invalid_request`. Resuming a task that is not paused
  returns 409 `not_paused`. Stop cancels a running worker or cancels a queued
  task. Finished tasks cannot be changed (409 `invalid_transition`).

Pause closes the worker fence and records pending pause while an already-running
provider or tool finishes. At a held plan, reflection or undelivered approval,
the worker parks, closes its resources and saves a card from persisted evidence.
An undelivered approval is withdrawn; resume starts from the card and asks again.
A task paused before it starts does not start a worker until resumed. If a park
cannot be saved, the worker retries with its fence closed; after three failed
attempts it saves an interrupted task and card. If the store cannot persist that
outcome, resume and stop return 503 `store_unavailable` until finalization succeeds.
A failed resource close quarantines the worker and keeps its slot counted; after
retries, the interrupted task reports `cleanup failed; slot held until restart`.

Task objects include the task id, normalized `spec`, `state`, priority and
source, `wait_reason`, `pending_approvals` (the approvals the task waits for, in
the shape of a session's; empty when none; answer one with `POST
/api/approvals/{request_id}`, as for a chat), timestamps, workspace and `priority_events`, along
with reserved accounting fields. Service, wait and tool accounting are not
implemented yet. Explicit user workspaces must be existing directories
strictly inside home, outside hidden folders and `~/Library`; default
workspace selection is described in [Scheduler](scheduler.md#tasks-workers-and-the-scheduler-store-s3a).
Invalid workspace requests return 400 `invalid_workspace`; a workspace that
becomes invalid before a run or resume fails the task with
`wait_reason="invalid_workspace"` instead of running it.

At most four workers are resident at once, including quarantined workers.
Task rows persist across gateway restarts. Startup interrupts uncertain tasks,
retains valid parked cards and preserves pending pause, without replay. Every
create records a durable never-started proof inside the create transaction,
before any launch; a runnable queued task with that proof may run once after a
start fence. Recovered tasks with
uncertain history reject `PATCH` resume and priority changes until controlled
Resume is used; Stop remains available. A failure to persist
startup recovery keeps the store unavailable. Retrying the same `request_key`
returns the existing task and does not restart it.

### Controlled Resume

`POST /api/tasks/{task_id}/resume` accepts `operation_id` (required, non-empty,
at most 200 characters) and an optional `confirmation_token`. It is the only
route that resumes a task recovered by startup; `PATCH` stays fenced for that
work. It returns 202 `{"task": ..., "operation_id": ...}` when the transition
is committed.

The route consumes evidence, never absence of evidence:

- A parked or interrupted task the store proves never started is queued for its
  first launch through the durable launch fence, with no card and no
  confirmation. A queued never-started create is admitted by startup recovery,
  not by this route, so calling the route for one returns `invalid_transition`.
- A started task needs a card and a valid workspace. It resumes without
  confirmation only when its trusted, complete capture covers the whole
  execution and every recorded process group is verifiably gone. A verified
  boot change (both ids read, well formed and different) counts as gone only on
  the host that recorded the group; a change read on another host, a record
  without a host, and an unreadable host are all unknown.
- Anything else, including missing, untrusted or incomplete capture or a group
  whose liveness cannot be checked, returns 409 `confirmation_required` with
  `details.confirmation_token`, `details.expires_in_s` (300), a
  `details.warning`, and `details.reason`. Present the token to accept the
  recorded risk and resume. That accepted transition writes one
  `sched.resume_confirmed` event; a preview writes
  nothing. The token is single use, held in memory only, and bound to the task,
  source execution, evidence identity and CAS generation. A restart drops it,
  and Stop invalidates it.
- A terminal task, invalid workspace, missing card for a started task, or stale
  execution identity returns 409 `invalid_transition` with a `details.reason`;
  a missing card says to start again as a new task. An unresolved earlier
  recovery operation or an unsettled worker cleanup returns 409
  `recovery_unknown`.

An operation ID names one operation. A retry with the same `operation_id`
returns 202 with the current task, whatever payload it carries, and never
starts a second turn or re-authorizes a transition; a token is consumed at most
once. A restart drops tokens. A Resume committed before a restart is re-admitted
on startup when its turn never started; once that turn has started, a retry of
the operation returns 409 `invalid_transition` instead of pretending the queued
work still waits. Stop always
wins: it invalidates pending tokens and cancels a queued resumed turn before a
worker starts a provider or tool. Neither a free Resume nor a confirmation
proves that an older process stopped or that its effects happened only once.

The route is also reachable at `/api/v2/tasks/{task_id}/resume` for clients
that address API v2 by its version; other task routes use the `/api` prefix.

## GET /api/sessions

The newest sessions first. Only `chat` sessions are listed by default.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/sessions?query=notes&limit=50'
```

```json
{"sessions": [{"id": "3f9c2a7b1d4e", "kind": "chat", "title": "Release notes", "created_at": 1761141600.0, "updated_at": 1761141612.4, "model": "stub-model", "workspace": "/home/me/demo-project", "workspace_label": "~/demo-project", "busy": false, "running_here": false, "pending_approvals": 0, "level": 2, "held": ""}], "machine": {"lock": "none", "holder": "", "purpose": "", "run": "", "eta": "", "stale": false, "readable": true, "on_ac": true, "battery_pct": 100, "battery_state": "charged", "band": 4, "gate": "record"}, "has_more": false}
```

- `limit` is 1 to 200 (default 50). `query` is at most 200 characters and
  matches a session's title or any turn's user text. `has_more` says that
  older sessions exist beyond the page.
- A session is listed once it has a turn, a running one included: a
  session appears as soon as its create has stored its first turn's row.
  A row a create made without ever starting its message (a shutdown in the
  middle, say) has no turn and is left out; `GET /api/sessions/{id}` still
  shows it.
- `kind` is `chat` (default), `front`, `front_bg`, `task`, `worker` or
  `terminal`; any other value is 400 `invalid_request`. `worker` and `terminal`
  read the CLI sessions file in SQLite read-only mode: `worker` lists direct
  runs and `terminal` lists the chats of interactive `melo` sessions, which
  `kind=chat` never includes. A
  missing CLI file gives an empty list and is not created by the gateway.
  SQLite may create `-wal` and `-shm` bookkeeping sidecars during a read;
  the reader does not open the main database for writing or use immutable mode.
- Worker and terminal lists accept exact-match `status`, `source`, and
  `parent_run` filters, and an opaque `cursor` from the previous response's
  `next_cursor`. Results are ordered by the start time of the row's turn
  descending, then session id descending. `next_cursor` is `null` at the end;
  keep the same filters and query on later pages. Invalid cursors or using
  these parameters without `kind=worker` or `kind=terminal` return 400. Pages are not a snapshot: concurrent changes may
  affect subsequent pages.
- Worker list and detail session objects include nullable `source`,
  `parent_run`, `git_branch`, `git_head`, `exit_code`, `status`, `steps`, and
  `tokens`, `started_at`, and `duration_s`. Start time is the first turn's
  timestamp; duration is its recorded result duration when present.
  `tokens` is the integer total token count, and `meta.usage` holds
  the full usage map. Status falls back to the first turn's status; steps and
  tokens fall back to its stored result. Missing values are `null`, not zero.
  The detail response's top-level `tokens` separately counts completion tokens;
  for a worker it is `null` if no turn has an integer `completion_tokens`.
- Terminal chats are served read-only, like worker runs, with `kind` set to
  `terminal` in list and detail objects. The row's turn is the chat's latest
  turn, so `status`, `started_at`, `steps` and `duration_s` describe that turn
  and an active chat comes first. Detail, turns (including `events=true`),
  `/events` and `/status` work by id as for a worker run, and `busy` is true
  while a turn runs under a live lease of the terminal that owns the chat.
  The terminal stores a chat unredacted, so every text of one is redacted as
  it is read: secrets in titles, `meta`, turn text, answers, results and event
  payloads (tool output and shell output included) become `[redacted]`, the
  output of a call that read a secret file such as `.env` is replaced by a
  note, and a run's `config` is left out. A worker run is stored redacted and
  is read as stored. Redaction is by patterns, not guaranteed secret-free;
  images are not redacted. Every write route returns 404 for a terminal chat
  id, as for a worker id.
- Worker sessions are read-only. Their session detail and turns (including
  `events=true`), stored image blobs, `/events` and `/status` are available by
  id; write routes return 404 (`POST /api/turns/{turn_id}/cancel` returns 409
  `turn_not_running`: no turn of a worker run runs in the gateway). For a
  worker id found only in the CLI file, `/tools`,
  `/memory`, `/runs` and the session-scoped submission lookup also return
  404 `not_found`. If an id exists in both files, the gateway session wins.
  Worker session objects add `meta` with the stored provenance, and their
  `running_here` is always false. `busy` reflects an unexpired worker lease;
  the gateway does not recover stale turns or apply retention to the CLI file.
- `workspace_label` is the workspace with the home folder written as `~`.
- `busy` says a turn is running. For gateway sessions, `running_here` equals
  `busy`: only this gateway runs turns in its own sessions file.
- `pending_approvals` counts this session's approval cards that wait for an
  answer.
- `level` and `held` are the scheduler's view of the chat: its P1 to P4
  (P2 until a chip sets it) and the reason its model call is waiting
  (`Resources`, `Priority`, `Reservation`, …), `""` when nothing waits. They
  are the same facts the web workspace's header shows; see
  [Scheduler](../guides/scheduler.md#the-gate-on-chats-s2).
- `machine` is the one gate's snapshot of the machine — the GPU lock's
  `lock` (`none`, `self`, `foreign`), `holder`, `purpose`, `stale`,
  `readable`, `on_ac`, `battery_pct`, `battery_state`, `band`, and `gate`
  (`record` or `wait`) — the object the page's machine line reads; see
  [The GPU lock and the battery](../guides/scheduler.md#the-gpu-lock-and-the-battery-s2b).
  It is not `null`: the console always has one gate.

## GET /api/sessions/{id}

One session, its pending approvals, its session grants, and its output
token total.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e
```

```json
{
  "session": {"id": "3f9c2a7b1d4e", "kind": "chat", "title": "Release notes", "created_at": 1761141600.0, "updated_at": 1761141612.4, "model": "stub-model", "workspace": "/home/me/demo-project", "workspace_label": "~/demo-project", "busy": false, "running_here": false, "pending_approvals": 0, "level": 2, "held": ""},
  "pending_approvals": [{"request_id": "0d6c2a9e8f7b41c3a5e9d1b3f5a7c9e1", "call_id": "call_1", "tool_name": "file_write", "arguments": {"path": "/home/me/demo-project/todo.md"}, "arguments_complete": true, "preview": "~/demo-project/todo.md", "workspace_root": "/home/me/demo-project", "workspace_label": "~/demo-project", "target": {"path_label": "todo.md", "exists": false, "size_bytes": 0}}],
  "grants": ["file_write"],
  "granted_at": {"file_write": 1761141604.9},
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
  `not_loaded` (a session this gateway has not opened, as after a restart).
  A create opens the session even without a turn; the read routes never
  open one, while a turn or a retry does. So `not_loaded` lasts until the
  first open since the gateway started, and a closed Agent reads `paused`.
- `tokens` sums the `completion_tokens` of the session's finished turns,
  the same field the answers report.
- A `pending_approvals` entry's `arguments_complete` is true only when the
  JSON-safe `arguments` preserve the original values without clipping or
  conversion. Clients should not allow a request when it is false or missing;
  denial remains available. Approval event arguments alone may be truncated,
  so fetch session details before allowing.
- A `pending_approvals` entry's `target` is set for a `file_write`: the
  path `path_label` (relative to the workspace, else with the home folder
  as `~`), whether the file `exists`, and its `size_bytes`, read when the
  response is made. Every other tool's `target` is `null`.

## GET /api/sessions/{id}/turns

The session's turns, oldest first within the page.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/turns?limit=10&events=true'
```

```json
{"turns": [{"id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "turn": 3, "run_id": "4b1e7c9a2f3d4e5b8a6c0d1e2f3a4b5c", "user_text": "Summarize these notes", "status": "completed", "cause": "run", "answer": "The notes say ...", "result": {"task_id": "4b1e7c9a2f3d4e5b8a6c0d1e2f3a4b5c", "status": "completed", "result": "...", "error": "", "steps": 3, "attempts": 1, "n_llm": 3, "n_main": 3, "n_aux": 0, "n_tool": 4, "llm_errors": 0, "usage": {"prompt_tokens": 1180, "completion_tokens": 24, "total_tokens": 1204}, "duration_s": 12.2, "config": {}}, "undone": false, "started_at": 1761141600.2, "finished_at": 1761141612.4, "client_request_id": null, "running_here": false, "events": [{"id": 790, "session_id": "3f9c2a7b1d4e", "turn": 3, "v": 1, "type": "turn.started", "run_id": "", "seq": null, "ts": 1761141600.2, "data": {"turn": 3, "turn_id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "user_text": "Summarize these notes", "client_request_id": null}}]}], "has_earlier": false, "replay_from": 789}
```

- `limit` is 1 to 50 (default 10). Without `before`, the page is the newest
  turns and `replay_from` is the event id just before the oldest returned
  turn's `turn.started`, or 0 when the session has no turn. For gateway
  sessions, pass it as the event stream's `after` to replay the page's events.
  For a worker session, `events=true` is one read; `/events` follows it.
- With `before=N` (1 to SQLite's largest integer), the page is the newest
  turns numbered below `N`, and there is no `replay_from`.
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
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -N 'http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/events?after=789'
# catch up once and stop:
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/events?after=789&follow=false'
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
  it must be decimal digits within a signed 64-bit integer: negative, `+1`,
  `1_0`, non-numeric and too-large values are 400 `invalid_request`.
- `follow=false` sends the stored events after the start point, up to the
  newest event id the request read, and then ends the response; it is
  one-shot catch-up, and a turn that starts while it pages adds no frames.
- The stream ends when the client disconnects, when the session no longer
  exists (a reconnect then gets 404), and when the gateway shuts down. A
  browser reconnects with `Last-Event-ID` and receives every event it
  missed, once.
- A worker run (a session of the CLI file) is served from the CLI file with
  the same frames, `after`, `follow` and `Last-Event-ID`. Another process
  writes that file, so a following stream reads it about once a second, sends
  the new events with their ids, and ends once the run holds no live lease
  and its last event is sent (`busy` is false in the session object). A run
  with no live lease, as one that crashed leaves, is replayed and ends at
  once. Keepalives are as for a chat; each open stream is one reader of the
  file, closed when the client goes away.
- A terminal chat is streamed the same way, redacted. It is live while one
  of its turns runs under the terminal's lease, so the stream ends when that
  turn ends; read the session object again to see whether another turn runs.

## GET /api/sessions/{id}/status

The numbers the terminal's `/status` and status bar show for one chat, task or
worker session, folded on the server from its stored events with the
terminal's own code, so both agree. The segments are the compact ones of the
terminal's developer mode; its default mode says the same figures in words. A worker run or terminal chat is read from
the CLI's file.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/status
```

```json
{
  "session_id": "3f9c2a7b1d4e",
  "local": true,
  "segments": ["cache 67% idle 3m", "ctx 1.0k/8.0k", "step 3", "tasks 1 running 2 queued"],
  "cache": {"requests": 2, "last_percent": 67, "session_percent": 56, "idle_s": 200.0},
  "context": {"measured": 1000, "limit": 8000},
  "steps": {"active": true, "step": 3, "attempt": 1, "phase": ""},
  "tasks": {"running": 1, "queued": 2, "held": 0, "top_level": 1}
}
```

- `segments` is what the terminal's status bar draws in developer mode, left to
  right: cache, context, step, tasks. A segment with no data is left out.
- `cache.last_percent` and `cache.session_percent` are `null`, and the
  segment reads `cache --`, when the server reported no cached tokens: never
  0%. `idle_s` is set for a local backend only (a local server's cache is
  evicted by memory, not by time, so there is no countdown).
- `context` and `steps` are `null` before a measurement or a run's first
  progress fact. `steps.active` is true while a run is going. `tasks` is the
  task store's running, queued and held counts and the highest priority
  among them (`top_level`, `null` with none).
- The fold is cached per session at its last event id: a read with no new
  event reads no events.

## GET /api/sessions/{id}/blobs/{sha256}

An image a turn's tool returned, by its SHA-256 digest.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -o shot.png http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/blobs/823ceb99fcef5252333ede1b2202341c3b287b6d47571963e6b0ddf393a24f82
```

- `sha256` is exactly 64 lowercase hex characters; anything else, an
  unknown digest and an unknown session are 404 `not_found`.
- `image/png`, `image/jpeg`, `image/gif` and `image/webp` are served inline
  with their type. Any other stored type, SVG included, is
  `application/octet-stream` with
  `Content-Disposition: attachment; filename="<sha256>.bin"`.
- Every response has `X-Content-Type-Options: nosniff`, one
  `Content-Security-Policy` header whose value is
  `default-src 'none'; sandbox; frame-ancestors 'none'` (the gateway's
  framing rule is merged into the blob's own policy) and
  `Cache-Control: no-store`.

## GET /api/sessions/{id}/submissions/{client_request_id}

What became of a message the client named with a `client_request_id`. The
same lookup without a session is `GET /api/submissions/{client_request_id}`,
for a client that never learned the new session's id.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/submissions/8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07
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

## Writes

The routes below change something. Most take a JSON body; `POST
/api/turns/{turn_id}/cancel` and `DELETE /api/sessions/{id}` take none and
ignore one. From a browser, a write must be same-origin: a state-changing
request from another origin is 403 `foreign_origin`. A body is strict — an
unknown field is 400 `invalid_request` with the field named
(`body.model: Extra inputs are not permitted`).

These codes answer on the write routes besides the shared ones above:

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_workspace` | the folder is not absolute or `~/`, is no longer a directory, or the default workspace cannot be created |
| 400 | `text_rejected` | the checker refused the message text, as it does for a `/v1/agent/run` request |
| 409 | `session_busy` | a turn is already running in the session |
| 409 | `not_latest_turn` | retry named a turn that is not the newest finished one |
| 409 | `submission_conflict` | the `client_request_id` was used for a different message |
| 409 | `submission_withdrawn` | the `client_request_id` was cancelled before it started |
| 409 | `turn_not_running` | cancel named a turn that is not running |
| 409 | `already_answered` | the approval was answered already |
| 502 | `agent_start_failed` | the session's Agent could not start; the message is the builder's error |

### What a turn answers with

A create with a first turn and both turn routes answer with the turn's row
and three resume keys:

```json
{"turn_id": "9a840f8444e1452a9a25e5b6b804b20b", "turn": {"id": "9a840f8444e1452a9a25e5b6b804b20b", "turn": 1, "run_id": "", "user_text": "list the folder", "status": "running", "cause": "", "answer": "", "result": null, "undone": false, "started_at": 1793163770.17, "finished_at": null, "client_request_id": "curl-turn-1", "running_here": true}, "resumed_turns": null, "resume_mode": null, "resumed_checkpoint_turns": null}
```

- The turn starts at once and runs in the background: the answer is 202 as
  soon as its row exists, 201 when a create started it. Follow it on the
  event stream — `turn.started`, the run's events, its `run.progress`
  events, `turn.finished` with `status` and `cause` — and read the answer
  from the turn row.
- `resumed_turns` is how many earlier turns the Agent loaded from the
  session file when this request opened the session; `resume_mode` is the
  arm that loaded them (`pairs` or `window`) and
  `resumed_checkpoint_turns` how many came from the window checkpoint. All
  three are `null` when the session was already open, when it was reopened
  from the conversation this gateway kept in memory (nothing is read from
  the file then), and when it had no earlier turn.

## POST /api/sessions

Create a session, usually with its first turn.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"workspace": "~/demo-project", "turn": {"text": "list the folder", "client_request_id": "8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07"}}'
```

```json
{"session": {"id": "3f9c2a7b1d4e", "kind": "chat", "title": "list the folder", "created_at": 1793163786.21, "updated_at": 1793163786.21, "model": "qwen3:8b", "workspace": "/home/me/demo-project", "workspace_label": "~/demo-project", "busy": true, "running_here": true, "pending_approvals": 0, "level": 2, "held": ""}, "turn_id": "9a840f8444e1452a9a25e5b6b804b20b", "turn": {"id": "9a840f8444e1452a9a25e5b6b804b20b", "...": "..."}, "resumed_turns": null, "resume_mode": null, "resumed_checkpoint_turns": null}
```

- `workspace` is optional. Absent or `null`, it is `~/localmelo-workspace`,
  created with mode `0700` when missing. Another value must be absolute or
  start with `~/` and be an existing directory; anything else is 400
  `invalid_workspace` and costs no Agent.
- `turn` is optional. Without it the answer is 201 with `session` alone,
  and the session waits for `POST /api/sessions/{id}/turns`.
- `session.model` is the settings' chat model of this request. The first
  message's text becomes the title: whitespace collapsed, at most 60
  characters with `…` when longer.
- `client_request_id` makes the create safe to repeat. The same key with
  the same text answers 200 with the session and turn it made and
  `"repeated": true`, building nothing; another text is 409
  `submission_conflict`; a key withdrawn before it started is 409
  `submission_withdrawn`. Two creates racing on one key make one session.
- `attachments` must be empty for now: an entry is 400 `invalid_request`,
  `attachments: files are not supported yet; send the text alone`. The
  text is checked by the gateway ingress checker; a refusal is 400
  `text_rejected`.

## POST /api/sessions/{id}/turns

Start a turn in an existing session. The body is a create's `turn`; there
is no `workspace` here — the session's own folder is the Agent's.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/turns \
  -H 'Content-Type: application/json' \
  -d '{"text": "list the folder", "client_request_id": "curl-turn-1"}'
```

- 202 with the turn and the resume keys. The resume keys are filled when
  this call had to rebuild the Agent from the session file: the first turn
  on a session no route of this gateway had opened, such as the first turn
  after a restart. A session this gateway closed and keeps in memory
  reopens from that conversation and reports `null`.
- A task session returns 409 `task_session_read_only`; use the task routes
  instead of starting a chat turn.
- 409 `session_busy` while a turn runs. One turn runs per session; a second
  is refused before anything is queued.
- A repeated `client_request_id` answers 200 with that key's stored turn
  and `"repeated": true`, whether the turn still runs or not.
- The session's folder must still exist: a deleted folder is 400
  `invalid_workspace`, and the turn does not start.

## POST /api/sessions/{id}/retry

Run the newest finished turn's text again.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/retry \
  -H 'Content-Type: application/json' -d '{"turn_id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e"}'
```

- The turn is marked undone — an event `turn.undone` — and its text starts
  a new turn with no `client_request_id`. The answer is 202 with the new
  turn and the resume keys of the open that replayed what the undo left.
- A task session returns 409 `task_session_read_only`; it cannot retry as a chat.
- 409 `not_latest_turn` unless the id is the newest finished, not undone
  turn: a client that lost a turn's end retries the id it holds. 409
  `session_busy` while a turn runs.

## POST /api/turns/{turn_id}/cancel

Stop a running turn, from the id the events gave.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/turns/9c1f0e2d4b6a48c1a3e5f7091b2d4c6e/cancel
```

```json
{"turn_id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "cancelling": true}
```

- 202 when this gateway runs the turn; 409 `turn_not_running` for a turn
  that finished, or one this gateway does not run.
- The turn stops at its next boundary and finishes `cancelled` with cause
  `user_stop`; `turn.finished` says so, and its stored `answer` is
  `[interrupt: no assistant output for this turn]` when it said nothing.
- Cancelling while an approval waits withdraws the card: the request is
  answered `withdrawn` and leaves `pending_approvals`.

## POST /api/sessions/{id}/submissions/{client_request_id}/cancel

Cancel a message by the key it was sent with.
`POST /api/submissions/{client_request_id}/cancel` is the same without a
session, for a create whose answer was lost.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/submissions/8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07/cancel \
  -H 'Content-Type: application/json' -d '{"text": "list the folder"}'
```

```json
{"submission": {"client_request_id": "8d2f6c1e-5b7a-4e0c-9a41-2f3b6d8e1c07", "state": "withdrawn", "session_id": "3f9c2a7b1d4e", "turn_id": null, "status": null, "cause": null, "turn": null, "error": "Request cancelled before acceptance.", "created_at": 1793163793.49}, "cancelling": false}
```

- The body's `text` is the message's text. Another text on a stored key is
  409 `submission_conflict`.
- A withdrawal stored without a session — the route without an id — has
  `"session_id": null`.
- 202 with `"cancelling": true` when the key's turn was accepted and still
  runs: the turn is stopped as above. `submission` is the key as this call
  read it, so the turn is still `running` there: it ends after the answer.
  Follow its `turn.finished` (cause `user_stop`), or ask again, for the end.
- 200 otherwise. An accepted turn that already finished comes back as it
  is. A key with no submission is recorded as withdrawn, so a later create
  or turn with it is 409 `submission_withdrawn`; repeating the call
  returns the same withdrawal.
- A key is judged in its scope: a session reads its own keys and the
  create-scope key that created it. A withdrawal stored without a session
  blocks a create with that key, not a turn in some session.

## POST /api/approvals/{request_id}

Answer an approval card. The id comes from `approval.requested` on the
event stream, or from `pending_approvals`.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/approvals/386183e1811b4ba68c3768bfb96ac8e6 \
  -H 'Content-Type: application/json' -d '{"choice": "session"}'
```

```json
{"request_id": "386183e1811b4ba68c3768bfb96ac8e6", "outcome": "allowed", "remember": true}
```

- `choice` is `once` (this call), `session` (this call and every later one
  of that tool in this session) or `deny`.
- `reason` belongs to a `deny` only, at most 500 characters; a reason with
  another choice is 400 `invalid_request`.
- 404 `not_found` for an unknown or expired request; 409 `already_answered`
  for one answered already, so two clients racing one card get one answer.
- Approvals live in the gateway's memory. A turn waiting on one when the
  gateway stops is finished `cancelled` with cause `shutdown`, and its card
  is stored as withdrawn.

## PATCH /api/sessions/{id}

Rename the session, move it to another folder, or both.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X PATCH http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e \
  -H 'Content-Type: application/json' -d '{"title": "Release notes"}'
```

```json
{"session": {"id": "3f9c2a7b1d4e", "kind": "chat", "title": "Release notes", "...": "..."}}
```

- A task session returns 409 `task_session_read_only` for either change.
- `title` is stripped and must be 1 to 200 characters; another length is
  400 `invalid_request`. A rename is never replaced by the automatic title.
- `workspace` moves the session: a new Agent is built in the new folder
  with the session's conversation, and the session's tool grants are
  forgotten. The rules are a create's: absolute or `~/`, an existing
  directory, else 400 `invalid_workspace`. 409 `session_busy` while a turn
  runs; a refused move leaves the row as it was.
- The answer is 200 with the session after both changes.

## DELETE /api/sessions/{id}

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X DELETE http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e
```

- A task session returns 409 `task_session_read_only`; it cannot be deleted as a chat.
- 204 with no body: the session's row, turns, events and blobs are gone.
- In the in-progress security-1b changes, 409 `session_busy` refuses a
  delete while a turn is running or a start is queued; it does not cancel
  the turn or delete its row. Cancel or wait for it to finish, then retry.
  A maintenance operation holding the session lock can also briefly return
  this error, even when no turn is running; wait and retry.
- 404 `not_found` for an unknown session.

## POST /api/sessions/{id}/level

Set a chat's scheduler level, P1 to P4.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/level \
  -H 'Content-Type: application/json' -d '{"level": 1}'
```

```json
{"session_id": "3f9c2a7b1d4e", "level": 1}
```

- `level` is a JSON integer from 1 to 4. A string or a boolean is 400
  `invalid_request` (`body.level: Input should be a valid integer`); `0` and
  `5` are 400 with `body.level: Input should be greater than or equal to 1`
  and `body.level: Input should be less than or equal to 4`. P1 is the urgent
  lane, P4 the background one; see
  [Scheduler](../guides/scheduler.md#the-gate-on-chats-s2).
- The route never opens the session: a level is a scheduler fact, not an
  Agent's. An unknown session is 404 `not_found`, as on every read below
  `/api/sessions/{id}`.
- The level is kept in memory for the gateway's run, not in the session
  file; the next `GET /api/sessions` reports it back.

## POST /api/presence

The page's typing heartbeat: this chat is the lane's focused chat.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/presence \
  -H 'Content-Type: application/json' -d '{"session_id": "3f9c2a7b1d4e"}'
```

```json
{"ok": true}
```

- 404 `not_found` for an unknown session. The call is a scheduler event for
  the focused chat (the lane keeps its typing window); it does not open the
  session and changes nothing else.
- A turn started through the turn routes, and a create with a first turn,
  also resets the chat's hold clock and focuses it, as a chat's own message
  does.

## GET /api/sessions/{id}/tools

Every tool the session's Agent offers, with whether this session allows it.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/tools
```

```json
{"tools": [{"name": "python_exec", "description": "Run Python code ...", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}, "source": "builtin", "requires_approval": true, "granted": false}], "mcp_config_label": "", "mcp_error": "MCP is not built in the console yet"}
```

- `source` is `builtin`, `provided` (an MCP server's tool) or `custom`.
  `requires_approval` says a call of it asks first; `granted` says this
  session allows it for the rest of the gateway run.
- A console session is gateway mode: it lists `file_read`, `file_write` and
  `python_exec`, never `shell_exec`.
- A task session returns 409 `task_session_read_only`; this route cannot open it as a chat.
- This route opens the session, because a provider's tools appear only once
  its Agent started. It can take seconds, and a start that fails is 502
  `agent_start_failed`.
- MCP in the console is not built yet, so no MCP tools load:
  `mcp_config_label` is always `""` and `mcp_error` always says
  `MCP is not built in the console yet`.

## DELETE /api/sessions/{id}/grants/{tool}

Take back one session grant, so the tool asks again next time.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X DELETE http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/grants/python_exec
```

- 204 with no body; both `grants` and `granted_at` of
  `GET /api/sessions/{id}` drop it.
- 404 `not_found` when the session is not open or the tool has no grant.
  `DELETE /api/sessions/{id}/grants` without a tool forgets every grant and
  answers 204, whether or not the session is open.

## GET /api/sessions/{id}/memory

What the session's Agent remembers in its window.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/memory
```

```json
{"agent": "open", "working": [{"role": "user", "text": "What is in this project?", "tool_calls": [], "tool_call_id": "", "images": 0}], "working_max": 20, "long_term": {"enabled": false, "backend": "none"}}
```

- This route never opens the session. `working` is the open Agent's
  conversation, the kept copy while `paused`, and `[]` when `not_loaded`.
- An assistant row's `text` has its `<think>` block removed, and every text
  stops at 2,000 characters with `… (+N chars)` when longer; the page
  expands the message itself.
- Every text is redacted like a stored event: a key a tool's output
  carried comes back as `[redacted]`. Tool rows are part of the window.
- `tool_calls` lists the names a message called, `images` counts its image
  references.
- `working_max` is the window's message bound. `long_term.enabled` is
  whether the settings name an embedding backend, and `backend` is its key
  (`none` when they do not).

## GET /api/sessions/{id}/runs

Each finished turn's run record, newest page first like the turns list.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/runs?limit=20'
```

```json
{"runs": [{"turn": 3, "turn_id": "9c1f0e2d4b6a48c1a3e5f7091b2d4c6e", "status": "completed", "started_at": 1761141600.2, "finished_at": 1761141612.4, "result": {"task_id": "4b1e7c9a2f3d4e5b8a6c0d1e2f3a4b5c", "status": "completed", "error": "", "steps": 3, "...": "..."}}], "has_earlier": false}
```

- `limit` is 1 to 50 (default 20); `before=N` pages to turns numbered below
  `N`, as on `GET /api/sessions/{id}/turns`.
- `result` is the turn's `RunResult` without its `result` text, which the
  transcript already shows. A turn that is still running has no run record
  and is left out; `has_earlier` says that older turns exist beyond the
  page.

## Not in this version

File attachments in `POST /api/sessions` and `POST /api/sessions/{id}/turns`
and MCP tools in the console are not built yet; the shapes above are the
ones later changes extend.

FastAPI's `/docs`, `/redoc` and `/openapi.json` are off: this page is the
reference. Non-loopback binds are refused at startup; see
[Gateway and HTTP API](../guides/gateway.md#keep-the-gateway-private).
