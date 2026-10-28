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
  first; see [Configuration](configure.md#set-up-with-the-wizard).
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

A refused request under `/api/` uses [API v2](../reference/api-v2.md)'s envelope instead:
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
