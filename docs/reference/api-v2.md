# API v2

API v2 is the web console's HTTP API. It is served under `/api/` on the
gateway, only when the gateway listens on a loopback address
(`127.0.0.1`, `localhost` or `::1`).

Protected routes require a bearer
token or signed-in browser cookie; do not expose the gateway. See [Gateway and HTTP API](../guides/gateway.md#keep-the-gateway-private).

Apart from the authentication requirement, API fields remain additive:
endpoints, optional request fields and response keys are added. A client may ignore keys it does not
know.

This page documents every API v2 route: health, settings, key status,
connection test, model listing, sessions, turns, SSE events, blobs, pictures,
submissions, approvals, tools, grants, memory, runs and scheduler tasks.

`POST /api/auth/login` exchanges a valid single-use code for a browser cookie;
`POST /api/auth/logout` revokes the cookie. `POST /api/auth/rotate` requires
the bearer and rotates it.

The login exchange requires a matching Origin. When
the macOS app started the gateway, it also requires the app's
`X-LocalMelo-Instance` header, and a code is valid only for the instance it was
issued for; a missing, different or differently bound value is refused before
the code is consumed.

See [gateway security](../guides/gateway.md#keep-the-gateway-private).

A console turn passes the same scheduler gate a workspace chat does: every
model call waits for its slot under `[scheduler]`, and the call log records
it under the session's id. See [Scheduler](../guides/scheduler.md#the-gate-on-chats-s2).

JSON request bodies and JSON responses use UTF-8. Event streams use
`text/event-stream`, image blobs and pictures use their image media type, and a successful
204 response has no body.

A request with a non-empty body
needs `Content-Type: application/json`, and a body may have at most 1 MiB
(1,048,576 bytes). Every string in a request must be valid UTF-8 without
lone surrogates.

Responses carry `Cache-Control: no-store`,
`Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; font-src 'self'; manifest-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`
(except blobs, whose policy is `default-src 'none'; sandbox; frame-ancestors 'none'`, and pictures, see
[their route](#get-apisessionsidpictures)),
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
| 403 | `outside_workspace` | a picture request whose real path is not inside the session's folder |
| 404 | `not_found` | an unknown session, blob, picture or route |
| 405 | `method_not_allowed` | a known path with a method it does not serve; the `Allow` header names the methods it does |
| 409 | `settings_changed` | `config.toml` changed on disk since `GET /api/settings` read its `revision` |
| 413 | `too_large` | the request body is over 1 MiB, or a requested picture is over 20 MB |
| 415 | `unsupported_media_type` | a non-empty request body without `Content-Type: application/json` |
| 415 | `not_a_picture` | a picture request for a file whose name or content is not a PNG, JPEG, GIF, WebP or SVG |
| 421 | `misdirected_request` | the `Host` header is not `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>`, or an app-started gateway's `/api/auth/login` has no matching `X-LocalMelo-Instance` |
| 500 | `internal_error` | an unexpected failure; the message is `Internal error; see the gateway log.` and never a traceback or a value from the request |
| 502 | `backend_unreachable` | listing models failed or returned no models |
| 503 | `store_unavailable` | the gateway's session store could not open, or stopped recording changes; restart the gateway in the second case |

The gateway logs the cause of a 500, whose response carries a fixed
message. A 503 repeats the store's reason with the home folder written as
`~`; it never carries a traceback, a lease holder or a key.

The legacy `/v1/*` routes keep their own bodies, including FastAPI's `422` and
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
      {"kind": "task", "id": "a1b2c3d4e5f6", "title": "Index the repo", "activity": "Reading notes.md", "icon": "read"},
      {"kind": "run", "id": "c44733ec77e5", "title": "Summarize the changelog", "activity": "Thinking", "icon": "think"},
      {"kind": "terminal", "id": "d27a70f1bbd8", "title": "Debug the flaky test", "activity": "Running the tests", "icon": "run"}
    ],
    "waiting": [],
    "recent": {"kind": "chat", "id": "e5f60718293a", "title": "Plan the trip", "outcome": "done", "ended_s_ago": 3.2},
    "focus": {"kind": "terminal", "id": "d27a70f1bbd8", "title": "Debug the flaky test", "activity": "Running the tests", "icon": "run"}
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
  terminal chats, at most 20, each `{"kind", "id", "title", "activity", "icon"}`
  (`kind` is `chat`, `task`, `run` or `terminal`; a terminal chat's title is
  redacted). The counts are not capped. A store that cannot be read counts as
  nothing running; one call reads the task states by their index and the
  running turns of the CLI file by theirs.
- `activity` says what the item is doing in plain words, and `icon` its kind:
  `Waiting for your approval` (`wait`), `Waiting for the model` (`held`, the
  scheduler holds its model call; when the reason is known it says why:
  `Waiting for a higher-priority task`, `Waiting for a free model slot`,
  `Waiting while another job uses the model`, `Waiting for free memory`,
  `Waiting for the battery`, `Waiting for the Mac to cool down` or `Waiting
  until the Mac is idle`), `Reading notes.md` and `Looking through
  ~/talk` (`read`), `Writing slides.md` (`write`), `Running the tests` (for a
  command that runs a test suite), `Running a command` and `Running Python`
  (`run`), `Reading example.com`, `Reading the page` and `Following a link`
  (`web`), `Remembering` (`memory`), `Checking its work` and `Thinking`
  (`think`), and `Using a tool` (`tool`) for any other tool. They come from the
  item's newest 40 events; only a file's name, a folder and a page's host are
  shown, never a command line, code or text. A store that cannot be read
  reads `Thinking`.
- `status.focus` is the one item a glance shows (the item as in
  `status.items`): one that waits for an approval first, else the one whose
  newest event is the latest; `null` when nothing runs.
- `status.waiting` lists the chats and tasks with a tool-approval request that
  waits for an answer, chats first, at most 20, each `{"kind", "id"}` (`kind` is
  `chat` or `task`). The console opens them at `#/s/<id>` and `#/tasks/<id>`, and
  the macOS app's menu bar reads the list at each poll. `sessions.pending_approvals`
  counts the open chats' requests only.
- `status.recent` says how the newest finished chat turn, task, run or terminal
  turn ended: `{"kind", "id", "title", "outcome", "ended_s_ago"}`, or `null` when none
  did (or when no store can be read). `kind` is as in `status.items`; `title` is
  what it was called (at most 200 characters, empty when it cannot be read). `outcome` is
  `done` (a run that completed, a task that is done), `stopped` (the person
  stopped it, or a task was cancelled or expired) or `failed` (any other end of
  a run, such as `provider_error`, `budget_exhausted` or `stuck`, and a failed
  task); the words are the console's, so what it shows as a warning or an error
  is `failed`. `ended_s_ago` is its age in seconds, to a tenth. A turn that a
  shutdown, a crash recovery, a takeover, a close or a park ended is no outcome,
  and an undone turn does not count. It is for a client that shows how the last
  run went: the macOS app's pet and activity pill read it. One call reads the newest finished turn of
  the most recently updated sessions of each file by their indexes.

## GET /api/activity

What ran today, for the panel beside a chat: the console's chats, the
scheduler's tasks, and the worker runs and terminal chats of the CLI file
(read-only, a terminal chat redacted), one row per session. With `session_id` it
follows one chat, and each row lists the approvals its session asked for.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- "http://127.0.0.1:8401/api/activity?since=$(date -v0H -v0M -v0S +%s)&session_id=742f96513ef3"
```

```json
{
  "since": 1793332800.0,
  "scope": "chat",
  "rows": [
    {"kind": "chat", "id": "742f96513ef3", "title": "add the meeting date to the notes", "state": "waiting",
     "activity": "Waiting for your approval", "icon": "wait", "result": "", "at": 1793399003.68,
     "approvals": [{"id": "8c64f46b27384316a9e407bb52b4e6bc", "what": "Write notes.txt", "answer": "waiting",
                    "at": 1793399003.69}],
     "turns": 1}
  ],
  "more": false,
  "unavailable": []
}
```

Without `session_id` (or with the main chat's) the same call lists the whole day, with rows such as:

```json
[
  {"kind": "task", "id": "a1b2c3d4e5f6", "title": "Slides for the 3 pm meeting", "state": "running",
   "activity": "Writing slides.md", "icon": "write", "result": "", "at": 1793376000.2, "approvals": [], "level": 2},
  {"kind": "run", "id": "c44733ec77e5", "title": "Check the build", "state": "done",
   "activity": "", "icon": "run", "result": "All 412 tests passed.", "at": 1793370000.5, "approvals": [], "turns": 1}
]
```

- `since` (seconds since the epoch) starts the day; it defaults to the
  gateway's local midnight, must not be in the future (`400 invalid_request`),
  and reaches back a week at most.
- `session_id` (at most 128 characters) is the chat the panel has open. For a
  side chat, `scope` is `chat` and the rows are that chat's own row and the
  tasks it started (a task's `front_id`); an unknown id is a side chat with
  nothing of its own. For the main chat (a session of kind `front`, or one with
  `main: true` in its `meta`), and with no `session_id`, `scope` is `all` and
  the rows are the whole day, worker runs and terminal chats included.
- A row is a session's newest turn that runs or ended at or after `since` (a
  turn that still runs is in, however early it began), or a task changed since
  then or not finished. `state` is `running`, `waiting` (an approval waits),
  `held` (the scheduler holds its model call), `queued` or `paused` while it is
  live, else `done`, `failed` or `stopped`. A live row says what it does
  (`activity` and `icon`, as in [`status.items`](#get-apihealth)); a finished one
  says how in `result`, one line of at most 140 characters (the turn's digest,
  else the answer's first sentence, else for a failure why), and its `icon` is
  the kind of its last tool call (`done` without one, `failed` or `stopped`).
  `at` is when it ended (a live row: when it started; a task: its last change),
  `turns` how many of a chat's turns started that day, `level` a task's
  priority (0 to 5). A result is always plain words: a run's own marker reads
  `Stopped before answering` or `Ended without an answer`, a path reads `Made a
  picture` or the file's name, and code gives way to the first sentence around
  it (`Wrote some code` when there is none).
- `approvals` lists what the row's session asked to do today, oldest first, at
  most 20: `{id, what, answer, at}`. `what` is the call in plain words (`Write
  slides.md`, `Run the tests`, `Run a command`; never a command line, code or
  text), `at` when it was asked, and `answer` is `waiting`, `allowed once`,
  `allowed for this chat`, `allowed automatically`, `denied`, `no answer` (it
  timed out) or `withdrawn` (the turn ended first). A `waiting` request of a
  chat or task is answered with `POST /api/approvals/{id}`, as the chat's card
  does. The rows of runs and terminal chats are redacted like the rest.
- Live rows come first, then the rest newest first, at most 100; `more` is
  `true` when some were left out. A source that cannot be read adds no rows and
  is named in `unavailable` (`chats`, `tasks`, `runs`, `approvals`).

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
  "across_chats": true,
  "memory_retriever": "auto",
  "language": "auto",
  "advanced_options": [
    {"option": "--host", "tier": "advanced", "issue": ""},
    {"option": "--retriever", "tier": "experimental", "issue": "#4"}
  ],
  "developer_settings": [
    {"section": "agent", "keys": [
      {"key": "agent.reasoning", "value": "legacy", "default": "legacy", "flag": "--reasoning", "issue": "#183", "file": true}
    ]}
  ],
  "run_options": [{"option": "--mcp PATH", "text": "MCP servers to use (mcpServers JSON file)"}],
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
- `across_chats` is `[memory] across_chats`, the switch of
  [memory across chats](#memory-across-chats).
- `mode` is `[ui] mode` (`default` or `developer`; any other saved value
  reads as `default`). `advanced_options` lists the advanced and experimental
  `melo` options from the tier table (`issue` is the deciding issue of an
  experimental one, `""` otherwise).
- `memory_retriever` is `[memory] retriever`, how a chat finds its notes:
  `auto`, `bm25`, `embedding` or `llm`.
- `language` is `[console] language`, the console's language: `auto`, `en`
  or `zh-Hans`.
- `approvals` is `[approvals] mode`, the mode a new chat starts in: `ask` or
  `auto` (another value in the file reads `ask`).
- `developer_settings` is every key `melo -h all` lists, grouped by section
  in its order (`""` is the top level): `value` is the one in use and
  `default` the default, both as the help prints them; `flag` is the old
  option that sets the key, `issue` the issue that decides a comparison, and
  a key with `file` false is for one run only (its value is its default).
  `run_options` are the help's option rows past the everyday ones. The
  console's Developer section shows both, read-only.

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
- `across_chats` is optional: a boolean that sets `[memory] across_chats`; omitted,
  the saved value is kept.
- `memory_retriever` is optional: `auto`, `bm25`, `embedding` or `llm` sets
  `[memory] retriever` (other values are 400 `invalid_request`; `embedding`
  without an embedding model is 400 `invalid_settings`); omitted, the saved
  value is kept.
- `language` is optional: `auto`, `en` or `zh-Hans` sets `[console] language`
  (other values are 400 `invalid_request`); omitted, the saved value is kept.
  The page loads again to show the new language.
- `approvals` is optional: `ask` or `auto` sets `[approvals] mode`, the mode
  new chats start in (other values are 400 `invalid_request`); omitted, the
  saved value is kept. Chats made before keep their own mode.
- The console saves one change at a time: the loaded `revision` with only the
  changed field, and its Undo is the same request with the earlier value.
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
seconds. Takes the `PUT /api/settings` body without `revision`.

In the in-progress security-1b changes, authenticated probes accept explicitly
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

- Field or validation errors return 400 first. A failed probe returns HTTP
  200 with `{"ok": false, "error": "...", "failed": "chat"}` (`failed` is
  `embedding` when the embedding probe failed); a timed-out one has no
  `failed`.

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
absolute HTTP or HTTPS URL; redirects are not followed.

The supplied host
can be contacted.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/models?backend=ollama&url=http%3A%2F%2F127.0.0.1%3A11434%2Fv1'
```

```json
{"models": ["qwen3:8b"], "embedding_models": [], "suggested_embedding": null}
```

- `embedding_models` are the listed ids that name embedding models and
  `suggested_embedding` the one to offer first (Qwen3-Embedding first), or
  `null`: the Settings dialog asks `Remember across chats with <model>?` from them.
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
found, and, between several chat models, its status routes.

It works before
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
They are separate from chat turns but use the same scheduler gate.

Each task has a `kind="task"` session with the same id; its session events and turns
can be inspected through the existing session routes. A worker uses the
built-ins `file_read`, `file_write` and `python_exec` (not `shell_exec` or
`list_files`) with memory off.

Tools run with the user's permissions, not in a sandbox.

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
- `front_id` (optional) is the chat that starts the task: a side chat's or the
  main chat's session id. The task belongs to it (the task's `front_id`, which
  the panel's Workers tab of that chat lists); absent, the main chat. An id
  that names no console chat (unknown, or a task's session) is 400
  `invalid_request`.
- `GET /api/tasks` returns `{"tasks": [...]}`; optional `state` and integer
  `level` query parameters filter the rows. `GET /api/tasks/{task_id}`
  returns `{"task": ...}` or 404 `not_found`.
- `PATCH /api/tasks/{task_id}` accepts `level` (0–5, `P0`–`P5`, or
  `auto`) and/or `state` (`pause`, `resume`, or `stop`). It returns
  `{"task": ...}`. `first` is rejected with 400 `invalid_request`; it is
  not supported yet. Pause parks at P5; resume restores the priority before
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
A task paused before it starts does not start a worker until resumed.

If a park cannot be saved, the worker retries with its fence closed; after three failed
attempts it saves an interrupted task and card. If the store cannot persist that
outcome, resume and stop return 503 `store_unavailable` until finalization succeeds.

A failed resource close quarantines the worker and keeps its slot counted; after
retries, the interrupted task reports `cleanup failed; slot held until restart`.

Task objects include the task id, normalized `spec`, `state`, priority and
source, `wait_reason`, `pending_approvals` (the approvals the task waits for, in
the shape of a session's; empty when none; answer one with `POST
/api/approvals/{request_id}`, as for a chat), timestamps, workspace and `priority_events`, along
with reserved accounting fields. Service, wait and tool accounting are not
implemented yet.

A task keeps the [mode](#ask-and-auto) its chat had when it was made
(`approvals`, `ask` or `auto`); a later switch of the chat does not change
it. A task in Auto asks nobody, so nothing holds it, and its calls run
under the write sandbox. Its object adds `blocked`, the steps Auto blocked
(oldest first: `request_id`, `call_id`, `name`, `step_key`, `reason`,
`review`, `preview`, `ts`, `turn`, `can_allow`, and `used_at`, when the
allowed step ran, else `null`), and `allowances` as a chat's. The task goes
on without a blocked step. `POST
/api/tasks/{task_id}/blocks/{request_id}/allow` allows it once (as for a
chat) and sends the go-ahead message to the task: 202 `{"task": ...,
"message": "queued"}` for a task that has not finished (it reads it at its
next step), `"continued"` for one that has (it runs again); 404 `not_found`
for no such block, 409 `not_allowable` for one that cannot be allowed, 409
`already_allowed` once the step ran on an allowance or while one waits. A
task's allowances go when it finishes.

Explicit user workspaces must be existing directories
strictly inside home, outside hidden folders and `~/Library`; default
workspace selection is described in [Scheduler](scheduler.md#tasks-workers-and-the-scheduler-store-s3a).
Invalid workspace requests return 400 `invalid_workspace`; a workspace that
becomes invalid before a run or resume fails the task with
`wait_reason="invalid_workspace"` instead of running it.

At most four workers are resident at once, including quarantined workers.
Task rows persist across gateway restarts.

Startup interrupts uncertain tasks,
retains valid parked cards and preserves pending pause, without replay. Every
create records a durable never-started proof inside the create transaction,
before any launch; a runnable queued task with that proof may run once after a
start fence.

Recovered tasks with
uncertain history reject `PATCH` resume and priority changes until controlled
Resume is used; Stop remains available. A failure to persist
startup recovery keeps the store unavailable.

Retrying the same `request_key`
returns the existing task and does not restart it.

### Controlled Resume

`POST /api/tasks/{task_id}/resume` accepts `operation_id` (required, non-empty,
at most 200 characters) and an optional `confirmation_token`. It is the only
route that resumes a task recovered by startup; `PATCH` stays fenced for that
work.

It returns 202 `{"task": ..., "operation_id": ...}` when the transition
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
once.

A restart drops tokens. A Resume committed before a restart is re-admitted
on startup when its turn never started; once that turn has started, a retry of
the operation returns 409 `invalid_transition` instead of pretending the queued
work still waits.

Stop always wins: it invalidates pending tokens and cancels a queued resumed turn before a
worker starts a provider or tool. Neither a free Resume nor a confirmation
proves that an older process stopped or that its effects happened only once.

The route is also reachable at `/api/v2/tasks/{task_id}/resume` for clients
that address API v2 by its version; other task routes use the `/api` prefix.

## Chores

Saved requests that run on a schedule (see [Chores that repeat](../guides/web-console.md#chores-that-repeat)); the panel's Scheduled tab uses these routes. A run is a task with origin `user_rule` in a folder of its own under the workspace, owned by the chat the chore belongs to (`front_id`: the main chat, or a side chat). Every chore carries a `revision` that each change bumps.

```json
{"chore": {"id": "2ab01e8df4f1", "title": "brief me on my calendar", "request": "brief me on my calendar", "schedule": {"kind": "weekdays", "at": "08:30"}, "when": "every weekday at 8:30", "level": 2, "level_word": "Normal", "enabled": true, "revision": 2, "front_id": "51742917baa0", "next_run": 1793262600.0, "last_run": 1793176260.0, "last": {"state": "done", "line": "Three meetings today.", "task_id": "edb413a9ca6d", "at": 1793176260.0}, "created_at": 1793170800.0, "late_min": null}}
```

- A chore's `level` (1 to 4) is its priority: the console shows it as `P` plus the number. `level_word` (`Today`, `Normal`, …) is kept for existing clients and is never what the console shows.
- `GET /api/chores` answers `{"chores": [...], "checkin": {"on": true, "every_min": 30, "next_at": 1793413200.0, "last_at": null}, "scope": "all"}`, oldest first. `?session_id=ID` of a side chat keeps that chat's own and gives `"scope": "chat"`; the main chat's id, an empty or no `session_id` lists all. An unknown or non-chat id is 404 `not_found`.
- `POST /api/chores` takes `request` (required, at most 100,000 characters), `schedule`, and optional `title` (at most 200; default derived from the request), `level` (`P1` to `P4`, 1 to 4, or the older words `Today`, `Normal`, `Whenever`, `If idle`; default from the request's words, as for a task) and `front_id` (a chat; default the main chat). `schedule` is `{"kind": "daily" | "weekdays", "at": "HH:MM"}` (local time), `{"kind": "hours", "every": 1..24}` or, for a reminder, `{"kind": "once", "at": "YYYY-MM-DDTHH:MM"}` (a local date and time to the minute; it must be ahead and at most 366 days away, else 400 `invalid_request`); other keys are refused. A bad body is 400 `invalid_request`; 50 chores are the most (409 `chore_limit`). Answers 201 `{"chore"}`; the first `next_run` is the schedule's next time plus a jitter of up to 120 s.
- `PATCH /api/chores/{id}` takes the `revision` read plus `title`, `request`, `schedule`, `level` and `enabled` (at least one). A stale revision is 409 `stale_revision`, an unknown chore 404 `not_found`. A new `schedule` or `enabled: true` plans `next_run` from now (a missed day is not run); `enabled: false` sets it to null. A new `once` schedule also switches a done reminder back on; `enabled: true` on one whose time has passed is 400 `invalid_request`.
- `DELETE /api/chores/{id}` answers `{"deleted": id}`; with `?revision=N` a stale revision is 409 `stale_revision`.
- `POST /api/chores/{id}/run` starts a task now and leaves the schedule: 202 `{"chore", "task"}`.
- `PUT /api/checkin` with `{"on": false}` switches the check-in (kept in the file) and answers `{"checkin": {...}}`; `next_at` is null while it is off.

A one-time chore (`"kind": "once"`) runs once. Its `when` reads `once, Fri Oct 30 at 9:00` (in the page's language for the console), its first `next_run` is its time plus the jitter, and after its run the gateway switches it off: `enabled` false, `next_run` null, `last_run` and `last` kept. One due while the computer slept or the gateway was down runs at the next tick, and `late_min` then says how many minutes after its time it started (null when it started less than five minutes after it). It never waits for an earlier run to end.

A one-time chore whose `request` starts with `Reminder: ` is a reminder. Its run's objective is that request plus a note that tells the model to say the reminder in one short sentence and not to do the task; the task's `reason` is `Reminder`, and its card in the main chat has `reminder: true` (the console words it `Reminder: <title>`, and for a failed run the card's line is the full words).

`last` is the chore's newest run as its task stands (`state` is the task's: `queued`, `running`, `held`, `waiting`, `parked`, `interrupted`, `done`, `failed`, `cancelled` or `expired`; `line` the result's first sentence, or why it did not finish). The gateway starts a due chore with a tick every 30 s that calls no model; a chore due while the computer slept runs once at the next tick, one whose last run has not ended skips that time, and a start repeated after a crash is the same task (the key is `chore:<id>:<planned time>`).

## GET /api/sessions

The newest sessions first. Only `chat` sessions are listed by default: the
side chats of the main chat, which the list names in `main`.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- 'http://127.0.0.1:8401/api/sessions?query=notes&limit=50'
```

```json
{"sessions": [{"id": "8983e7f06ecf", "kind": "chat", "main": false, "title": "Plan the Tokyo trip", "created_at": 1793399544.338924, "updated_at": 1793399544.339885, "model": "stub-model", "workspace": "/home/me/demo-project", "workspace_label": "~/demo-project", "busy": false, "running_here": false, "pending_approvals": 0, "level": 2, "held": ""}], "machine": {"lock": "none", "holder": "", "purpose": "", "run": "", "eta": "", "stale": false, "readable": true, "on_ac": true, "battery_pct": null, "battery_state": "", "band": 4, "gate": "record"}, "has_more": false, "main": {"id": "313ad3fcb8fb", "kind": "front", "main": true, "title": "Main chat", "created_at": 1793399544.315079, "updated_at": 1793399544.315079, "model": "", "workspace": null, "workspace_label": "", "busy": false, "running_here": false, "pending_approvals": 0, "level": 2, "held": "", "last_card": {"id": 4, "ts": 1793399544.342355, "source_id": "8983e7f06ecf", "source_kind": "chat", "title": "Plan the Tokyo trip", "state": "done", "line": "Three days in Tokyo, then Kyoto.", "at": 1793399544.339885}}}
```

- **The main chat.** The gateway's sessions file holds exactly one main chat:
  a session of kind `front`, titled `Main chat`, made when the console first
  opens the file, before its first message. It is never deleted. Every
  console chat of kind `chat` is one of its side chats; chats made before
  the main chat existed are its side chats too, unchanged. Each session
  object says so in `main`: `true` for the main chat, `false` for any other.
- `main` (with the default `kind=chat` only) is the main chat as a session
  object, plus `last_card`: its newest [card](#cards-in-the-main-chat), or
  `null`. A client marks the main chat unread while `last_card.id` is newer
  than the last card it showed. `main` is `null` only while the store cannot
  be read.
- `limit` is 1 to 200 (default 50). `query` is at most 200 characters and
  matches a session's title or any turn's user text. `has_more` says that
  older sessions exist beyond the page.
- A session is listed once it has a turn, a running one included: a
  session appears as soon as its create has stored its first turn's row.
  A row a create made without ever starting its message (a shutdown in the
  middle, say) has no turn and is left out; `GET /api/sessions/{id}` still
  shows it.
- `kind` is `chat` (default), `front`, `front_bg`, `task`, `worker` or
  `terminal`; any other value is 400 `invalid_request`. `front` lists the
  main chat, also before its first message. `worker` and `terminal`
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
  "session": {"id": "3f9c2a7b1d4e", "kind": "chat", "main": false, "title": "Release notes", "created_at": 1761141600.0, "updated_at": 1761141612.4, "model": "stub-model", "workspace": "/home/me/demo-project", "workspace_label": "~/demo-project", "busy": false, "running_here": false, "pending_approvals": 0, "level": 2, "held": ""},
  "pending_approvals": [{"request_id": "0d6c2a9e8f7b41c3a5e9d1b3f5a7c9e1", "call_id": "call_1", "tool_name": "file_write", "arguments": {"path": "/home/me/demo-project/todo.md"}, "arguments_complete": true, "preview": "~/demo-project/todo.md", "workspace_root": "/home/me/demo-project", "workspace_label": "~/demo-project", "target": {"path_label": "todo.md", "exists": false, "size_bytes": 0}}],
  "grants": ["file_write"],
  "granted_at": {"file_write": 1761141604.9},
  "mcp_error": "",
  "agent": "open",
  "tokens": 18512,
  "last_event_id": 412
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
- `last_event_id` is the id of the session's newest stored event (0 for
  none): a client that streams `GET /api/sessions/{id}/events?after=` this
  id hears only what comes next.
- A `pending_approvals` entry's `arguments_complete` is true only when the
  JSON-safe `arguments` preserve the original values without clipping or
  conversion. Clients should not allow a request when it is false or missing;
  denial remains available. Approval event arguments alone may be truncated,
  so fetch session details before allowing.
- A `pending_approvals` entry's `review` is what melo's rules found in a
  `shell_exec`, `python_exec` or `file_write` call, and `null` for any other
  tool: `verdict` is `routine`, `high` or `undecided`; `title` is the card's
  question for a `high` call ("Install software?") and `""` otherwise;
  `reasons` are at most three sentences that say why melo asks ("Installs
  playwright with pip into .signin_venv (downloads about 42 MB)."), secrets
  redacted. `categories` are the rules' categories, strongest first
  (`install`, `download`, `outside_write`, ...). A `high` call asks every time and the card offers no `session`
  answer.
- A `pending_approvals` entry's `target` is set for a `file_write`: the
  path `path_label` (relative to the workspace, else with the home folder
  as `~`), whether the file `exists`, and its `size_bytes`, read when the
  response is made. Every other tool's `target` is `null`.
- A `file_write` of text over an existing file also gives its `target` a `diff`, what the
  write replaces, read when the response is made, as the terminal's approval panel
  shows it. `target` has no `diff` for a new file or a call without text.

  ```json
  {"path_label": "todo.md", "exists": true, "size_bytes": 86, "diff": {"added": 1, "removed": 1, "rows": [{"kind": "context", "line": 2, "text": ""}, {"kind": "context", "line": 3, "text": "- [x] Read data.csv"}, {"kind": "removed", "line": 4, "text": "- [ ] Add a --json flag to app.py"}, {"kind": "added", "line": 4, "text": "- [x] Add a --json flag to app.py"}, {"kind": "context", "line": 5, "text": "- [ ] Test empty files"}], "note": ""}}
  ```

  - `added` and `removed` count the lines. `rows` are the changed lines with two lines of
    context each side, and a `gap` row between changes that are further apart. `kind` is
    `removed`, `added`, `context` or `gap`; `line` is the old line's number for a `removed`
    row and the new line's for any other, and `null` for a `gap`; `text` is the line without
    its ending, never cut.
  - Secrets are redacted in every row, as in a stored event: a key reads `[redacted]`. The
    rows come from the texts as they are, so a line that differs only in a secret is a
    `removed` row and an `added` row.
  - The diff is whole or absent. When melo cannot compare, `rows` is empty and `note` says why in
    words: the file is not text, cannot be read, is not a regular file or is over 200 KB
    (200,000 bytes); the new text is over 200,000 characters; more than 1,000 rows would
    change; or the file or the new text holds a private key. Otherwise `note` is `""`, and
    empty `rows` mean the file already holds the new text.
  - `diff` is for display. The approval's other fields, the answer and its rules are the
    same as without it.

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
- The main chat's pages add `cards`, its cards beside the page's turns, oldest
  first: from the page's oldest turn (from the start, on the oldest page) up to
  the next page's first turn (up to now, on the newest page), at most the newest
  50. A main chat without turns gives its newest 50 cards and a `replay_from`
  just before the oldest of them, so the stream replays nothing older.

### Cards in the main chat

When a side chat's run ends, or a task finishes, fails or waits for an
approval, the main chat gets a card: an `attention.card` session event that
belongs to no turn. A card is never a turn: no model reads it, and the main
chat's conversation never holds it.

Its `data`:

```json
{"source_id": "8983e7f06ecf", "source_kind": "chat", "title": "Plan the Tokyo trip", "state": "done", "line": "Three days in Tokyo.\nThen Kyoto.", "at": 1793399544.339885}
```

- `state` is `done`, `failed`, `waiting` (an approval waits for an answer) or
  `answered` (that approval was answered). `source_kind` is `chat` or `task`,
  and `source_id` is the side chat or the task to open. `title` is at most 48
  columns: a title the person or the main chat gave stays, a longer one is cut
  at a word. `at` is when it happened.
- `line` is plain words: the first two lines of the answer on a `done` card
  (an opening `Done.` is left out when more follows), why the run did not
  finish on a `failed` one (its status in words, then the first line of its
  error), what is asked on a `waiting` one (`Wants to run Python`, `Wants to
  write notes.md`), and the answer on an `answered` one (`Allowed once`,
  `Allowed for this chat`, `Allowed for this task`, `Allowed automatically`,
  `Denied`, `No answer in time`, `Not run: it stopped first`).
- A `waiting` card has a `request_id`. When that approval is answered, a card
  with state `answered`, the same `request_id` and the answer as its `line`
  follows it, and a client shows the one card with the answer in place of
  its Review. An answer is no news: it is never the list's `last_card`, and
  none is written when the waiting card never was.
- A side chat's run counts when its turn ends by its run (`cause` `run`); a
  stop by the person, a shutdown or a crash makes no card. A task counts when
  it ends `done`, `failed` or `expired`; a stop by the person makes none.
- The gateway writes the cards when the main chat or the list is read, and
  once a minute; a waiting card whose approval was answered first is never
  written, and an approval answered later is followed by its `answered` card. The main chat's stream carries new cards as `attention.card` frames.
- A task created through `POST /api/tasks` belongs to the chat its
  `front_id` names, else to the main chat; its cards still come to the main
  chat.

### Echoes in the main chat

What the main chat did with a task is an `attention.echo` session event in the
main chat, with no turn. A message that code handles (`/task TEXT`, a turn body
with `"as_task": true`, an imperative change to files it names, a short
control phrase that names one open task, or a short follow-up word with a
task's name) starts no turn: `POST /api/sessions/{main}/turns` answers 200
with the echo.

```json
{"echo": {"id": 41, "ts": 1793405483.9, "echo": "started", "task_id": "3f9c2a71b0de", "title": "summarize the readme", "level": 2, "reason": "you asked for it", "line": "Started: summarize the readme · P2", "typed": "/task summarize the readme"}}
```

- `echo` is `started`, `control` (with `action` and `undo`), `level` (with
  `undo`), `suggest` (a change the person's own words did not ask for, with
  `action` and maybe `level`; `action` `send` carries the message as `text`),
  `ask` (`Start this as a task?`, with `text`, the whole message, `words`, what
  was typed, and `turn_id`), `sent` (a message given to a task, `how`
  `queued` for its next step or `continued` as a new turn of a finished task;
  `line` `Sent to: <title>` or `Continued: <title>`), `chore` (a repeating
  phrase code read: `Run this every weekday at 8:30?`, with `title`, `text`
  (the work), `schedule`, `when` and `level`; a reminder cue with a time gives
  `Remind you on Fri Oct 30 at 9:00?` with `reminder: true`, `text`
  `Reminder: <words>`, `work` (the words), a `once` `schedule` and no
  `level`), `chore_saved` (with `chore_id`),
  `checkin` (the check-in's note, with `items`), `undone` or `failed`. A reminder
  cue with no usable time (none, one that has passed, one more than a year
  ahead) gives an `ask` echo with `ask: "when"` and `work` instead (`When should
  I remind you?`); it has no buttons, and the next message, if it is only a
  time (`Friday at 9`), within an hour, answers it with the `chore` echo
  (`of` is the ask). `line` is
  the plain words, and names a priority by its code (`Started: <title> · P2`,
  `<title> → P3`, `Undone: <title> is back to P2`); `typed` is the message it
  answers; `of` is the echo it answers.
- `as_task` on any session but the main chat is 400 `invalid_request`. A
  repeated `client_request_id` of a delegation answers the same echo.
- The main chat's turns pages add `echoes`, paged as `cards` are.
- `POST /api/front/echoes/{id}` with `{"do": "undo" | "apply" | "start" |
  "level" | "save" | "once" | "cancel", "level"?: "P1" | 1 | "Today"}` answers
  200 `{"echo"}`: Undo a `control` or `level` echo (a stop starts the task
  again as a new task) or a `chore_saved` one (the chore is deleted), apply a
  `suggest`, start an `ask`, or set a `started` or `level` echo's task to the
  person's level. On a `chore` echo, `save` keeps the chore (a `chore_saved`
  echo answers; a reminder is kept at P1, and one whose time passed meanwhile is 400),
  `once` starts its work as one task (not for a reminder: 400) and `cancel` keeps
  nothing (an `undone` echo answers). Each answers once (409
  `already_answered`); an unknown echo or task is 404, `save` with 50 chores
  already kept is 409 `invalid_transition`, and a `do` that does not fit the
  echo is 400 `invalid_request`.

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
terminal's developer mode; its default mode says the same figures in words.

A worker run or terminal chat is read from
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
  "tasks": {"running": 1, "queued": 2, "held": 0, "top_level": 1, "waiting": 0}
}
```

- `segments` is what the terminal's status bar draws in developer mode, left to
  right: cache, context, step, tasks, then a sentence such as `1 task waits for
  you` (`N tasks wait for you`) while a held task waits for your approval. A
  segment with no data is left out.
- `cache.last_percent` and `cache.session_percent` are `null`, and the
  segment reads `cache --`, when the server reported no cached tokens: never
  0%. `idle_s` is set for a local backend only (a local server's cache is
  evicted by memory, not by time, so there is no countdown).
- `context` and `steps` are `null` before a measurement or a run's first
  progress fact. `steps.active` is true while a run is going. `tasks` is the
  task store's running, queued and held counts, the highest priority
  among them (`top_level`, `null` with none) and `waiting`, how many held tasks
  wait for the person (an approval or an answer) rather than for the scheduler.
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

## GET /api/sessions/{id}/pictures

A picture file from the folder the session works in, by its path. The console
shows a turn's pictures with it (the [`turn.pictures` event](events.md#session-events)
lists them).

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -G --data-urlencode 'path=charts/fcf.png' -o fcf.png http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/pictures
```

- `path` is required, up to 4,096 characters: a path relative to the session's
  folder, or an absolute one inside it. `..` anywhere in it is refused. An
  unknown session, a session with no folder yet and a file that does not exist
  are 404 `not_found`.
- The file's real path, with every link followed, must be inside the session's
  real folder, and is checked again on the open file; otherwise the answer is 403
  `outside_workspace` (a `..` path, an absolute path elsewhere, a link that leads
  out, another session's folder). Only a regular file is served.
- The name must end in `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` or `.svg`, and
  the content must be that kind of picture, judged from its first bytes (the
  `Content-Type` is the content's, not the name's); otherwise 415 `not_a_picture`
  (a text file named `chart.png`, an SVG that is an HTML page). A file over
  20 MB is 413 `too_large`.
- Every answer has `Content-Disposition: inline`, `X-Content-Type-Options:
  nosniff` and `Cache-Control: private`. A PNG, JPEG, GIF or WebP has
  `Content-Security-Policy: default-src 'none'; sandbox; frame-ancestors 'none'`;
  an SVG has `Content-Security-Policy: sandbox; default-src 'none'; style-src
  'unsafe-inline'; frame-ancestors 'none'`, so it can draw but never run a
  script or open as a page. The route reads one file and writes and lists nothing.

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
ignore one.

From a browser, a write must be same-origin: a state-changing
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
{"session": {"id": "3f9c2a7b1d4e", "kind": "chat", "main": false, "title": "list the folder", "created_at": 1793163786.21, "updated_at": 1793163786.21, "model": "qwen3:8b", "workspace": "/home/me/demo-project", "workspace_label": "~/demo-project", "busy": true, "running_here": true, "pending_approvals": 0, "level": 2, "held": ""}, "turn_id": "9a840f8444e1452a9a25e5b6b804b20b", "turn": {"id": "9a840f8444e1452a9a25e5b6b804b20b", "...": "..."}, "resumed_turns": null, "resume_mode": null, "resumed_checkpoint_turns": null}
```

- A create makes a side chat of the [main chat](#get-apisessions): `kind`
  `chat`, `main` false. The main chat already exists; write to it with
  `POST /api/sessions/{id}/turns`.
- `approvals` is optional: `ask` or `auto`, the new chat's
  [mode](#ask-and-auto); absent, `[approvals] mode`.
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

- `choice` is `once` (this call), `session` (this call and every later
  routine one of that tool in this session) or `deny`. For a `high`
  call (see `review` under `GET /api/sessions/{id}`), `session` allows this
  call only: the reply says `"remember": false` and no grant is made.
- `reason` belongs to a `deny` only, at most 500 characters; a reason with
  another choice is 400 `invalid_request`.
- 404 `not_found` for an unknown or expired request; 409 `already_answered`
  for one answered already, so two clients racing one card get one answer.
- Approvals live in the gateway's memory. A turn waiting on one when the
  gateway stops is finished `cancelled` with cause `shutdown`, and its card
  is stored as withdrawn.

## PATCH /api/sessions/{id}

Rename the session, move it to another folder, switch it between Ask and
Auto, or several at once.

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
- `approvals` is `ask` or `auto`: the chat's [mode](#ask-and-auto) from its
  next step; a call already waiting for an answer keeps waiting. It is set
  in the session's `meta` in one transaction that keeps the other keys, and
  `session.updated` carries the new `meta`. A session that is not a console
  chat (`chat` or `front`) is 409 `not_a_chat`.
- The answer is 200 with the session after the changes.

## Ask and Auto

Each console chat has a mode (#339): `ask` (melo asks before commands and
file changes) or `auto` (nobody is asked; routine steps run, a risky or
off-request step is blocked and explained). Every session object carries:

- `approvals`: the chat's mode; a task's is the mode its chat had when the
  task was made. The main chat has `[approvals] mode` until it is set.
- `allowances`: the steps allowed once in this chat since the gateway
  started, `step_key` -> when the allowance runs out (Unix seconds); kept
  after it was used or ran out, so a page can say so.

A blocked step is the `tool.finished` record of its call with an
`approval` of origin `auto` and `allowed` false (see
[events](events.md)). In Auto the chat's calls run under the write sandbox
of `[agent] confine_writes`: a write outside the folder and the temporary
folders fails, and its output starts with a line that says so.

### POST /api/sessions/{id}/blocks/{request_id}/allow

Allow once the step blocked with `approval.request_id` *request_id*, and send
the person's message that asks melo to go ahead.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X POST http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e/blocks/5b1f0c2a9e7d4c3b8a6f1e2d3c4b5a69/allow
```

```json
{"step_key": "051aab7c5a9691e4d449b7a140164314", "until": 1793401344.2, "turn_id": "9a84…", "turn": {"user_text": "Go ahead with the step melo blocked: install software.", "...": "..."}, "resumed_turns": null, "resume_mode": null, "resumed_checkpoint_turns": null}
```

- The server finds the block in the chat's records; the client never sends
  a key. The allowance covers that exact step (`step_key`) with the kinds of
  step the block named, once, for 30 minutes, in memory: a restart forgets
  it. The retried call is recorded with origin `user` and
  `allowance: "once"`.
- The message (in the console's language) starts a turn as
  `POST /api/sessions/{id}/turns` does; 202 with that turn.
- 404 `not_found` for no such block in this chat; 409 `not_allowable` for a
  step that changes localmelo itself or a call a stopped turn did not run;
  409 `already_allowed` once the step ran on an allowance after the block,
  or while an allowance of it waits; 409 `session_busy` while a turn runs
  (nothing is allowed then); 409 `task_session_read_only` for a task (see
  below) and `not_a_chat` for another kind. An allowance the turn it
  started did not use goes when that turn ends.

## DELETE /api/sessions/{id}

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- -X DELETE http://127.0.0.1:8401/api/sessions/3f9c2a7b1d4e
```

- A task session returns 409 `task_session_read_only`; it cannot be deleted as a chat.
- The main chat returns 409 `main_chat` (`The main chat is always there: it cannot be deleted.`)
  and stays as it is.
- 204 with no body: the session's row, turns, events and blobs are gone, and so are
  its entries in memory's chat index. The notes melo made in the chat stay (a note
  outlives its chat, and then lists as `from a deleted chat`); `?notes=forget` forgets
  them too. Any other value of `notes` is 400.
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
  the focused chat (the lane keeps its typing window, `[scheduler] focus_s`,
  10 seconds by default); it does not open the session and changes nothing
  else. A chat that was only opened takes slots from tasks again once the
  window ends.
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
- A console session is gateway mode: it lists `file_read`, `file_write`,
  `list_files` and `python_exec`, never `shell_exec`.
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

## Memory across chats

The notes melo keeps about the user and the switch that turns memory across chats on or off
(see [Memory across chats](../concepts/memory.md#memory-across-chats)). The Memory tab of the
panel uses these routes.

```bash
{ printf 'Authorization: Bearer '; cat ~/.cache/localmelo/gateway/auth/token; printf '\n'; } | curl -H @- http://127.0.0.1:8401/api/memory/records
```

```json
{"on": true, "records": [{"id": 3, "text": "You prefer short answers.", "topic": "preferences", "quote": "I prefer short answers", "source": "you said it · chat “Planning”, Oct 30", "from": "from chat “Planning”, Oct 30", "chat_gone": false, "edited": false, "session_id": "3f9c2a7b1d4e", "created_at": 1793395000.0}], "documents": 12, "embedding_model": "", "retriever": "auto", "last_error": ""}
```

- `GET /api/memory/records` lists the notes newest first; `?session_id=ID` keeps one chat's.
  `topic` is `about`, `preferences`, `people`, `projects` or `decisions`. `from` names the
  chat and the day, or `from a deleted chat, Oct 30` (`chat_gone` true) once the chat is
  gone; `edited` is true once the user reworded the note. `documents` counts the chat turns in the
  index, `embedding_model` is the model that searches by meaning (`""`: words only, which
  `retriever` `bm25` or `llm` always is) and `last_error` the last reason it was not used.
- `PATCH /api/memory/records/{id}` with `{"text": "..."}` rewords a note in place and answers
  the note as above. The old wording is overwritten (not kept), the new one is embedded, and
  the text of the note's decision rows is blanked. A wording that is empty, over 300
  characters or looks like a password or key is 400 `invalid_request` with the reason; an
  unknown note is 404 `not_found`.
- `DELETE /api/memory/records/{id}` removes a note, its vector and the older wordings it
  replaced, and blanks the text of its decision rows: 204, or 404 `not_found`. With
  `?undo=true` (the "Undo" under a message) the note it replaced is current again.
- `DELETE /api/memory/records?all=true` forgets every note and empties the chat index, and
  answers `{"notes": 12, "turns": 48}`. Without `all=true` it is 400. Chats are not touched.
- `PUT /api/memory` with `{"on": false}` switches `[memory] across_chats` and saves it; the
  answer is `{"on": false}`. Chats started afterwards use it.
- `POST /api/memory/reindex` rebuilds the chat index from the chats (turns that ended
  before the last "forget everything" stay out) and gives every note a vector of the
  embedding model now set: `{"notes": 0, "turns": 48}`.

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
