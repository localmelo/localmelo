# Troubleshooting

Find the message you see, then read its cause and fix. A run that did not
complete names its status in the line
`melo: run ended with status <status>: <error>`, or in the `status` field of
the JSON line.

## Install

### command not found: melo

```text
zsh: command not found: melo
```

(bash prints `bash: melo: command not found`.) Or, with `uv run`:

```text
error: Failed to spawn: `melo`
  cause: No such file or directory (os error 2)
```

- **Cause:** the virtual environment is not active in this terminal
  (activation lasts for one terminal only), or `uv run` ran outside the
  clone, where uv cannot find localmelo.
- **Fix:** run `source <clone>/.venv/bin/activate` again, where `<clone>` is
  your `localmelo` directory. With uv, run `uv run melo …` inside the clone,
  or `uv run --project <clone> melo …` from anywhere else.

### No module named 'localmelo'

```text
ModuleNotFoundError: No module named 'localmelo'
```

- **Cause:** the clone's directory is not named `localmelo`, for example a
  renamed folder or a ZIP download unpacked as `localmelo-main`.
- **Fix:** rename the directory to `localmelo` and install again, as in
  [Install](quickstart.md#install).

### Gateway requires fastapi and uvicorn

```text
ImportError: Gateway requires fastapi and uvicorn: pip install fastapi uvicorn
```

- **Cause:** localmelo was installed without the `gateway` extra.
- **Fix:** run `uv sync --extra gateway`, or `pip install -e ".[gateway]"`.

### pip refuses to install

```text
error: externally-managed-environment
```

- **Cause:** pip will not install into the Python of your system or of
  Homebrew.
- **Fix:** create a virtual environment first, as in
  [Install with pip](quickstart.md#install-with-pip), or use uv.

### FileNotFoundError when the gateway starts

```text
FileNotFoundError: [Errno 2] No such file or directory: '…/localmelo/tests/smoke/data/scenarios.json'
```

- **Cause:** an older build omitted this required data file from its wheel.
  Current builds package it along with the web workspace's assets.
- **Fix:** update and reinstall localmelo as described in
  [Update localmelo](configuration.md#update-localmelo). An editable source
  install with `pip install -e ".[gateway]"` also includes the file.

## Setup and configuration

### No backend configured

```text
No backend configured. Run 'melo --serve --reconfigure' to set up,
or use --base-url to connect to an OpenAI-compatible endpoint.
```

- **Cause:** there is no config file, and the command has no `--base-url`.
  `melo` exits with code 1.
- **Fix:** save a configuration with `melo --serve --reconfigure`, or add
  `--base-url` and `--chat-model`.

### Configuration errors

```text
  Configuration errors:
  - [openai] environment variable $OPENAI_API_KEY is not set. Export it before starting the gateway.
```

`melo` stops before the run with exit code 1. Each line names one problem:

| Message contains | Fix |
|---|---|
| `environment variable $… is not set` | Export the API key in this shell. |
| `embedding_url is empty`, `embedding_model is empty` | Add both to the backend's table, or set `embedding_backend = "none"`. |
| `request_timeout_s must be a finite number of seconds > 0` | Write a number without quotes, such as `request_timeout_s = 600`. |
| `extra_body must not set 'model'` | Remove that key from `extra_body`. |

### TOMLDecodeError

```text
tomllib.TOMLDecodeError: Illegal character '\n' (at line 1, column 21)
```

- **Cause:** the config file is not valid TOML. The message gives the line
  and column, but not the file: it is `~/.cache/localmelo/config.toml`.
- **Fix:** correct that line, or delete the file and run the setup wizard
  again.

### Probe failed

```text
  Probe failed: chat probe failed: …
  Save config anyway? [N]:
```

- **Cause:** the wizard's test request failed. The server is not running,
  the URL or model is wrong, or the API key is missing.
- **Fix:** press Enter (nothing is saved:
  `Setup cancelled — config not saved.`), correct the problem, and run the
  wizard again.

### `melo --reconfigure` does not return

- **Cause:** after saving, the wizard starts the gateway, which keeps running.
- **Fix:** nothing is wrong. Press Ctrl+C to stop the gateway;
  `melo "<task>"` then uses the saved configuration.

### request_timeout_s has no effect

- **Cause:** the key is below a `[table]` line, where localmelo does not
  read it.
- **Fix:** move it to the top of the file, before the first table. See
  [Chat request timeout](configuration.md#chat-request-timeout).

## Runs that do not complete

### ConnectError

```text
melo: run ended with status provider_error: POST /chat/completions failed: ConnectError: All connection attempts failed
```

- **Cause:** nothing listens at the URL: the server is not running, or the
  host or port is wrong. localmelo tried three times, over about 5 seconds.
- **Fix:** start the server and check its URL, for example with
  `curl http://127.0.0.1:11434/v1/models`.

### HTTP 404 or another error status

```text
melo: run ended with status provider_error: POST /chat/completions failed with HTTP 404: …
```

- **Cause:** the server answered with an error status, and the start of its
  answer follows it. 404 means a wrong URL (with `--base-url`, include the
  `/v1` part) or a model that the server does not have; the answer usually
  says which. 500 is an error in the server or in a proxy in front of it.
- **Fix:** check the URL and the model name; `GET /v1/models` lists the
  models.

### ReadTimeout

```text
melo: run ended with status provider_error: POST /chat/completions failed: ReadTimeout
```

- **Cause:** the server sent nothing within the chat request timeout, three
  times in a row. This takes about 3 × the timeout + 5 seconds.
- **Fix:** give a slow model more time: `--request-timeout 600`, or
  `request_timeout_s = 600` at the top of the config file.

### A run never ends

- **Cause:** some servers send keep-alive bytes while they generate, and
  each one restarts the request timeout.
- **Fix:** limit the whole run with `--timeout SECONDS`.

### failed: empty replies

```text
melo: run ended with status failed: empty replies
```

- **Cause:** the model replied twice in a row with neither text nor a tool
  call.
- **Fix:** try another model, or check the model server's log.

### budget_exhausted: Max steps reached

```text
melo: run ended with status budget_exhausted: Max steps reached
```

- **Cause:** the run used all its steps without a final answer.
- **Fix:** raise `--max-steps`, or make the task smaller. To see what the
  steps did, keep the run's history with `--memory-dir` and
  [inspect it](memory.md#inspect-memory).

### stuck

```text
melo: run ended with status stuck: Stuck: the same tool calls or errors repeated
```

- **Cause:** with `--attempt-policy none`, the same tool calls, or the same
  errors, repeated three steps in a row.
- **Fix:** use the default policy, `reflect`, which reflects and tries
  again, or check which call kept failing in the run's history.

### gate_stopped

```text
melo: run ended with status gate_stopped: Stopped after attempt 1
```

- **Cause:** after an attempt, the reflection decided to stop, or its
  decision to continue did not pass the continuation gate. The error is the
  model's reason, or `Stopped after attempt <n>` when it gave none. With
  `--reflection-gate utility`, it can also be
  `Could not parse reflection response`: the model twice replied without the
  JSON object that a reflection asks for.
- **Fix:** read the reason. If the model cannot write reflections, try
  another model or `--attempt-policy none`.

### Tool not offered in this step

- **Cause:** `Tool '<name>' was not offered in this step.` is a tool's
  error, seen in history, not a run status. The model called a tool that the
  step did not offer, so the call did not run.
- **Fix:** usually none: a later step can offer that tool. To offer every
  tool on every step, use `--tool-exposure all`.

## Gateway

### Address already in use

```text
ERROR:    [Errno 48] error while attempting to bind on address ('127.0.0.1', 8401): [errno 48] address already in use
```

- **Cause:** another program uses the port, often another gateway or the
  background service. `melo` exits with code 3. (Linux shows another error
  number.)
- **Fix:** stop the other program (`melo --daemon status` shows the
  service), or choose another port with `--port`.

### HTTP 400

```json
{"error":"Query must be a non-empty string"}
```

- **Cause:** the request body is not valid: `query` must be a non-empty
  string, and `session_id` 1 to 64 letters, digits, `_` or `-`.
- **Fix:** correct the body. See [HTTP API](gateway.md#http-api).

### HTTP 421 misdirected_request

```json
{"error":"Host header must be 127.0.0.1:8401, localhost:8401 or [::1]:8401","code":"misdirected_request"}
```

- **Cause:** the request did not name the gateway as `127.0.0.1`,
  `localhost` or `[::1]` with its port. For example, the web workspace was
  opened as `http://0.0.0.0:8401/`, through a proxy that sends its own host
  name, or as `http://127.0.0.2:8401/` because the gateway listens on another
  loopback address. The gateway refuses these requests so that a web page on
  another site cannot reach it through DNS rebinding.
- **Fix:** open `http://127.0.0.1:8401/` or `http://localhost:8401/`, with the
  gateway's port. If the gateway listens on another loopback address, start
  it on `127.0.0.1`: set `host = "127.0.0.1"` in `[gateway]`, or run
  `melo --serve --host 127.0.0.1`. Proxies and remote access are not
  supported; see
  [Keep the gateway private](gateway.md#keep-the-gateway-private).

### HTTP 403 foreign_origin

```json
{"error":"Cross-origin request refused: only pages at http://127.0.0.1:8401, http://localhost:8401 or http://[::1]:8401 may send it","code":"foreign_origin"}
```

- **Cause:** a web page that the gateway did not serve sent the request
  through the browser: a page on another site or on another port of your
  computer, a page opened from a file (`Origin: null`), or the web workspace
  opened through a proxy. From a browser, the gateway accepts API requests
  only from its own web workspace.
- **Fix:** use the web workspace at `http://127.0.0.1:8401/`, or call the API
  from a program such as curl or a script, which sends no `Origin` header.
  Calling the API from another web page is not supported.

### A status in the result

```json
{"result":"[provider_error] POST /chat/completions failed with HTTP 500: …","session_id":"53a72edadef4"}
```

- **Cause:** the run did not complete and has no answer, so the gateway
  returns `[<status>] <error>`, still with HTTP 200.
- **Fix:** see the entry for that status in
  [Runs that do not complete](#runs-that-do-not-complete).

### The background service keeps restarting

- **Cause:** the gateway exits at once, and launchd starts it again every few
  seconds. Often the service lacks a cloud API key (it does not get your
  shell's variables), or there is no config file.
- **Fix:** read `~/.localmelo/logs/gateway.stderr.log`. Use a local backend
  with the service, or run `melo --daemon uninstall` and start
  `melo --serve` in a terminal to see the error.

## MCP servers

### MCP server exited

```text
melo: run ended with status setup_error: MCP server 'notes' exited (code 1); stderr: …
```

- **Cause:** the server stopped while it was starting. The end of its error
  output follows `stderr:`.
- **Fix:** run the server's `command` and `args` in a terminal to see the
  error. Remember that it gets only a minimal environment: add the variables
  it needs to `env`.

### Cannot start MCP server

```text
melo: run ended with status setup_error: Cannot start MCP server 'notes' (my-notes-server): [Errno 2] No such file or directory: 'my-notes-server'
```

- **Cause:** the `command` was not found.
- **Fix:** give the command's full path in the config file.

### HTTP/SSE servers are not supported

```text
melo: error: mcp.json: server 'docs': only stdio servers are supported (got type 'sse')
```

or, for an entry without `type`:

```text
melo: error: mcp.json: server 'docs': 'command' must be a non-empty string (HTTP/SSE servers are not supported)
```

- **Cause:** the entry describes an HTTP or SSE server: its `type` (or
  `transport`) is not `stdio`, or it has a `url` and no `command`.
  localmelo supports stdio servers only.
- **Fix:** use the server's stdio mode, if it has one.

## Getting help

If none of this helps, search the
[issues](https://github.com/localmelo/localmelo/issues) or open a new one
with the details that
[Report a bug](https://github.com/localmelo/localmelo/blob/main/CONTRIBUTING.md#report-a-bug)
lists. Remove API keys and personal paths first.
