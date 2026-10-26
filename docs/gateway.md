# Gateway and HTTP API

The gateway serves the agent over HTTP: an API for your own programs, and a
small web page for trying models by hand. This page shows how to start it,
keep it private, call its API, and run it at login on macOS.

## Start the gateway

```bash
melo --serve
```

- When there is no config file yet, `melo --serve` runs the setup wizard
  first; see [Configuration](configuration.md#set-up-with-the-wizard).
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
> run tasks with your tools and your account's permissions, and the web UI
> makes the gateway fetch any URL it is given. Keep the default host
> `127.0.0.1`.

On the default host, only programs on your computer can reach the gateway.
Web pages can still send it requests through your browser, so the gateway
refuses requests that are not addressed to it:

| HTTP status | `code` | When |
|---|---|---|
| 421 | `misdirected_request` | The `Host` header is not `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>`. This stops a page on another site that reaches the gateway under its own domain name (DNS rebinding). |
| 403 | `foreign_origin` | A web page at another origin sent the request: a `POST`, `PUT`, `PATCH`, `DELETE` or `OPTIONS` request whose `Origin` is not `http://127.0.0.1:<port>`, `http://localhost:<port>` or `http://[::1]:<port>`, or a request under `/v1/` whose `Sec-Fetch-Site` is `cross-site` or `same-site`. |

The body is the usual error with a `code`:

```json
{"error":"Host header must be 127.0.0.1:8401, localhost:8401 or [::1]:8401","code":"misdirected_request"}
```

- Open the web UI as `http://127.0.0.1:8401/` or `http://localhost:8401/`,
  with your port; with `host = "::1"`, use `http://localhost:8401/` or
  `http://[::1]:8401/`. Other addresses that reach the gateway, such as
  `http://0.0.0.0:8401/`, get 421.
- curl and scripts send no `Origin` header, so only the `Host` check applies
  to them.
- Every response tells the browser not to show it in a frame, with
  `X-Frame-Options: DENY` and
  `Content-Security-Policy: frame-ancestors 'none'`, so a page on another
  site cannot embed the web UI and trick you into clicking in it.
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
| `query` | yes | The task, up to 100,000 characters. Leading and trailing whitespace is removed. |
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
`Invalid session_id format: '<id>'`.

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
`{"error": "session not found"}`.

Closing a busy session closes its agent: the running task and any request
waiting for the session end as `crashed`, still with HTTP 200 and the result
`[crashed] RuntimeError: Cannot send a request, as the client has been closed.`
Close only idle sessions; `is_busy` in [GET /v1/sessions](#get-v1sessions)
shows which ones are busy.

## Sessions

A session is one agent with its own working memory, so a task sees the
earlier tasks of its session.

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
  options: all four built-in tools and no MCP tools.
- When `LOCALMELO_PERSIST_MEMORY` is set for the gateway, all sessions save
  to the same SQLite files; see [Memory](memory.md#keep-memory-between-runs).

## Web UI (playground)

Open `http://127.0.0.1:8401/` in a browser for the localmelo workspace:

1. Enter the URL of a chat server, and optionally of an embedding server.
2. Discover models, pick a chat model, and optionally select an embedding model
   or seed memory set. Embeddings and seed memories are disabled by default.
3. Create a chat, then send a message. Creating a chat never sends your draft.

The sidebar supports chat search and switching. Drafts stay with each chat in
the current browser tab. The workspace restores live chats after a page reload
and polls for changes from other browser clients every 2.5 seconds while visible.

Stop cancels the harness task and its current provider/tool request. Shell
process groups are killed; already dispatched model computation can still finish
inside a backend. In-process `python_exec` remains non-interruptible while Python
code blocks the event loop. See [Tools and MCP](tools.md) for execution limits.

The web UI works apart from the HTTP API above:

- It does not use the config file. It talks to the URL you enter, so
  `extra_body` and `request_timeout_s` do not apply. Chat requests wait up to
  300 seconds, and a task is stopped after 600 seconds.
- It keeps up to 12 sessions and closes those idle for 2 hours when creating a
  chat. At capacity, an idle chat is replaced only after preparation succeeds;
  if all chats are busy, creation fails with HTTP 409.
- All its sessions share one memory store in
  `~/.cache/localmelo/playground/`, which is always on.
- Its routes under `/v1/playground/` are internal and may change.

The workspace routes include `GET /v1/playground/sessions` for summaries,
`GET /v1/playground/session/{id}` for a transcript, and
`POST /v1/playground/session/{id}/cancel` for cancellation. Running or deleting
a busy chat returns HTTP 409. The ordinary `/v1/agent/` API remains separate.

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
- Run the setup wizard before `install`. The service cannot answer the
  wizard's questions.
- launchd starts the service without your shell's environment variables, so
  cloud API keys and `LOCALMELO_PERSIST_MEMORY` are missing. Use a local
  backend with the service.
- The service needs macOS; on other systems `install` fails.

## Stop the gateway

- In its terminal, press Ctrl+C.
- For the background service, run `melo --daemon uninstall`. A plain `kill`
  does not help: launchd starts the gateway again.
