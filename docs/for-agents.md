# localmelo for agents

The contract for AI agents and scripts: commands, exit codes, output, routes, files and rules. Details live in the reference pages this page links to. Humans: start at the [README](https://github.com/localmelo/localmelo#readme) or the [website](https://localmelo.github.io/localmelo/).

## What it is

- A local personal agent: it runs tasks with tools (shell, Python, files, a browser, MCP servers) and memory.
- It talks to a model server the user runs (oMLX, Ollama, any OpenAI-compatible server, melo_serve) or a hosted model; it serves no model itself.
- Surfaces: the `melo` command (one task per run, a fixed exit code, one JSON line), a gateway on 127.0.0.1 (web console, macOS app, HTTP API) and an experimental Python API.
- Pre-alpha: the command line, the HTTP API and the Python API may change before v0.1.0.
- macOS is tested; Linux is untested; Windows is not supported.

## Rules

- Tools run as the user, with no sandbox (a worker's and a saved task's writes are confined to their folder on macOS). Never run melo on untrusted input or near data the user cannot lose. Pass `--builtins none` when the task needs no tool.
- Never touch the user's `~/.cache/localmelo/` (config, memory, chats, keys). For tests and your own runs set a temporary `HOME`, with its own `config.toml` when you need one.
- Never run bare `melo` (no task) in a terminal under the user's `HOME`: it may ask setup questions and write the config. Without a terminal it exits 2 and asks nothing.
- Never expose the gateway: non-loopback binds are refused; no reverse proxy or tunnel. The default port is 8401; a test gateway takes another port or `-p 0`, never the user's.
- Never run `melo serve install` (the login service) in tests.
- No API keys in files: `api_key_env` names an environment variable, `api_key = "keychain:localmelo.<name>"` names a keychain item.
- Approvals are the person's: never answer an approval for them unless they told you to. `--yolo` runs tools without asking; use it only on a folder you may change.

## Install

```bash
git clone https://github.com/localmelo/localmelo.git
cd localmelo
uv sync --extra gateway        # Python 3.11+; the gateway extra is needed for melo serve
.venv/bin/melo --help
```

- `melo` lands in `.venv/bin/`, not on `PATH`. In scripts call `CLONE/.venv/bin/melo`: under `uv run`, a signal's `$!` is uv, and SIGKILL does not reach melo.
- Also installed: `melo-task` (gateway tasks), `melo-retrieve` and `melo-procedures` (offline tools, which runs do not use).

## Run one task

```bash
melo -u http://127.0.0.1:8000/v1 -m MODEL -o json -t 300 --builtins none -- "What is the capital of France?"
echo $?
```

- `-u`/`--url` is used as given (include `/v1`) and sends no API key. Without `-m`, melo asks `GET URL/models` and uses the only chat model, or the one the server says is loaded; with several and none known loaded it exits 2 and lists them.
- Without `-u`, the run uses the config file; with no config file it exits 1.
- Put the task after `--`, or quote it: a first argument that is a command word (`serve`, `setup`, …) is a command. A command word after options and no `--` exits 2 with a hint and no model call.
- Always set `-t`/`--timeout`: the chat request timeout counts per read, so a server that keeps sending bytes never trips it.
- Direct runs offer `file_read`, `file_write` and `python_exec` by default; `--builtins all` adds `shell_exec` and `list_files`, which then run unapproved.
- Every direct run is recorded as a worker session, which the console's Workers view lists; `-X sessions.record_direct=false` turns it off for one run.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | The run ended `completed`. |
| 1 | The run ended `crashed`, or melo stopped before the run: no backend, an invalid config file, a setup that needed a terminal. |
| 2 | Any other status, or a usage error (no JSON line), or no terminal for bare `melo`, or a bad value in the config's `[agent]` table. |
| 3 | The gateway could not start, for example because its port is in use. |
| 4 | A `--brief` run ended `completed` but its brief's check failed. |
| 129, 130, 143 | SIGHUP, Ctrl+C (SIGINT) or SIGTERM ended the run; nothing is printed on stdout. |

The full table, with every case, is in [Exit codes](reference/cli.md#exit-codes); signals are in [Signals](reference/cli.md#signals).

### Output

- Text (default): the result on stdout; a run that did not complete adds `melo: run ended with status STATUS: ERROR` on stderr.
- `-o json`: exactly one line on stdout, the run record, then `localmelo_version`. Exit 1 and 2 can come with no line (a usage or setup error): check that stdout is not empty before you parse it, then read `status`.

```json
{"task_id": "ee8b…", "status": "completed", "result": "Paris is the capital of France.", "error": "", "steps": 1, "attempts": 1, "n_llm": 1, "n_main": 0, "n_aux": 1, "n_tool": 0, "llm_errors": 0, "usage": {…}, "duration_s": 0.02, "config": {…}, "localmelo_version": "0.1.0"}
```

- Statuses: `completed`, `failed`, `gate_stopped`, `budget_exhausted`, `context_budget_exceeded`, `stuck`, `provider_error`, `setup_error`, `timeout`, `crashed` (`cancelled` only in the Python API and console chats). Meanings, every field and the retry rules: [Run results](reference/run-results.md).

### Worker runs

- `melo worker BRIEF [-w DIR] [-o json|stream-json]` (also `melo --brief FILE`) runs a brief: the task text, optionally with `Check:` lines that run when the model is done.
- `-o json` prints one versioned result: `v`, `status`, `run_id`, `exit_code`, `head_sha`, `git_dirty`, `tests`, `run`, `error`. `startup_error` and `transport_error` results have `run: null` and exit 1.
- `-o stream-json` prints NDJSON: `{"v": 1, "type": "event", "event": …}` per agent event, then `{"v": 1, "type": "result", "result": …}`. Drain stdout and check both the result and the exit code.
- `--inbox FILE` or `--inbox-fd N` adds instructions while it runs; `--parent ID` links the run to a launcher's run.
- Shapes: [Output](reference/cli.md#output); the events: [Events](reference/events.md).

## Commands, options and settings

This is `melo -h all`. `melo COMMAND -h` explains one command. `-X KEY=VALUE` (or `--set`) changes one setting for one run and never writes the file; `melo config set KEY VALUE` saves it. Old option spellings still parse: [Old spelling to new](reference/cli.md#old-spelling-to-new). Every option: [Command-line reference](reference/cli.md); every setting: [Configuration](reference/configuration.md).

<!-- agent-docs:help -->
```text
usage: melo [options] ["task"]
       melo COMMAND [options]

melo opens a chat here. melo "task" runs one task and prints the answer.

Options:
  -m, --model NAME     chat model for this run
  -u, --url URL        model server for this run (http://127.0.0.1:8000/v1)
  -w, --workspace DIR  folder to work in (default: this one)
  -c, --continue       continue the newest chat here
  -r, --resume [ID]    resume a chat, or pick one from a list
  -s, --steps N        most model calls per task (default 100)
  -t, --timeout SEC    stop the run after SEC seconds
  -o, --output FORMAT  text (default) or json
  -y, --yolo           run tools without asking
  -X, --set KEY=VALUE  change a setting for this run (list: melo config keys)
  -V, --version        show the version
  --memory on|off      long-term memory outside legacy reasoning (default on)
  --builtins SET       built-in tools: all, none or NAME,... (default: per mode)
  --mcp PATH           MCP servers to use (mcpServers JSON file)
  --mcp-only           only the MCP tools (--mcp PATH --builtins none)
  --routine NAME       learn and recall the task's steps as NAME
  --progress auto|off  progress line on stderr (default auto)
  --title T            title of this run's worker session
  --developer          with -h: every option and setting, as -h all does

Commands (melo COMMAND -h):
  setup               choose the model server again
  config              show or change the settings
  memory VERB         see and change the notes melo keeps (melo memory -h)
  serve [ACTION]      run the gateway; install | uninstall | status | stop
  console             open the web console in your browser
  task VERB           manage background tasks (melo task -h)
  worker BRIEF        run a worker brief and print its result
  import FORMAT FILE  import a worker history

Developer tools, own -h: melo-retrieve · melo-procedures · melo-task
Old spellings of the options: docs/reference/cli.md, "Old spelling to new".

Settings, by config.toml key (melo config keys shows the values in use):
-X KEY=VALUE changes one for this run and melo config set KEY VALUE saves
it; the option shown still works. A comparison names the issue that decides it.
  mcp_config                   ""
  request_timeout_s            unset       --request-timeout
  agent.reasoning              legacy      --reasoning (#183)
  agent.replay_turns           unset       --replay-turns (#183)
  agent.note_writer            auto        --note-writer (#183)
  agent.tool_env               []
  agent.window_eviction        chunk       --window-eviction (#183)
  agent.window_messages        unset       --window-messages
  agent.context_window         unset       --context-window
  agent.output_reserve         unset       --output-reserve
  agent.safety_margin          unset       --safety-margin
  agent.captured_history       false       --captured-history (#4)
  agent.confine_writes         true
  agent.confine_direct_runs    false
  agent.answer_on_stop         true         (#185)
  agent.max_calls              unset
  agent.tool_images            auto
  agent.tool_images_keep       3            (#280)
  agent.tool_images_retire     2            (#280)
  agent.empty_reply_fallback   on
  agent.reflection             true
  agent.steps_per_attempt      unset        (#185)
  agent.attempt_policy         reflect     --attempt-policy (#185)
  agent.tool_exposure          retrieve    --tool-exposure (#3)
  retrieval.retriever          auto        --retriever (#4)
  retrieval.recall_at          []          --recall-at (#4)
  retrieval.k                  3           --recall-k (#4)
  retrieval.kinds              []          --recall-kinds (#4)
  retrieval.placement          tool        --recall-placement (#4)
  retrieval.min_score          0.35        --recall-min-score (#4)
  retrieval.cross_task_ratio   1.1
  retrieval.cross_task_cap     1
  retrieval.cross_task_cold    0
  retrieval.extra_body         none        --retrieval-extra-body (#4)
  memory.across_chats          true
  memory.retriever             auto
  memory.sleep                 off          (#363)
  memory.sleep_lock            ""           (#363)
  memory.recall_min_cosine     0.55         (#349)
  memory.recall_min_coverage   0.6          (#349)
  memory.recall_excerpt        head         (#349)
  memory.recall_form           excerpt      (#428)
  memory.recall_list_max       20           (#428)
  memory.recall_open_chars     4000         (#428)
  memory.recall_source         turns        (#428)
  memory.compress_parallel     1
  memory.compress_cache        ""
  memory.project_instructions  on
  memory.keep_decision_text    false       --keep-decision-text (#4)
  memory.dir                   built in    --memory-dir
  endpoint.embedding_url       none        --embedding-base-url
  endpoint.embedding_model     none        --embedding-model
  endpoint.reasoning_field     auto        --reasoning-field (#183)
  sessions.resume_mode         pairs
  sessions.record_direct       true        --no-save-session
  cli.stream                   true
  cli.status_line              auto
  cli.colors                   auto
  cli.background               auto
  cli.chat                     main
  ui.mode                      default
  console.language             auto
  approvals.mode               ask
  approvals.auto_asks          true
  workspace.extra_folders      []
  scheduler.policy             off
  scheduler.n_srv              8
  scheduler.r_f                1
  scheduler.n_fg               1
  scheduler.focus_s            10.0
  scheduler.call_log           true
  scheduler.call_log_dir       ""
  scheduler.gpu_lock_file      ""
  scheduler.lock_self          ""
  scheduler.lock_gate          record
  browser.driver               off
  browser.keep_tabs            false
  browser.app_windows          true
  browser.live                 true
  browser.live_fps             15
  tools.web_fetch              true
  tools.web_fetch_allow_local  []
  web.search                   false
  web.search_order             []
  web.search_count             6
  tasks.follow_up              true
  chat.recent_digest_minutes   30
  gateway.port                 8401
  gateway.host                 127.0.0.1
  gateway.workspace            ""
```
<!-- /agent-docs -->

## The gateway

- `melo serve [-p PORT] [--host 127.0.0.1]` runs the gateway (web console, macOS app, HTTP API); `melo serve stop` ends it. It reads only `~/.cache/localmelo/config.toml`: for a test, give it a temporary `HOME` with its own file.
- Every route needs `Authorization: Bearer $(cat $HOME/.cache/localmelo/gateway/auth/token)` (or a browser session). The token does not isolate processes of the same user.
- A `Host` other than `127.0.0.1:PORT`, `localhost:PORT` or `[::1]:PORT` gets 421 `misdirected_request`; a foreign `Origin` gets 403 `foreign_origin`.
- `POST /v1/agent/run` `{"query", "session_id"?}` blocks until the run ends and returns `{"result", "session_id"}`; 200 means finished, not completed, and the gateway sets no time limit, so give your client one. For a status, use the console routes or `melo -o json`.
- The console's routes (`/api`) start chats and turns, stream a session's events, answer approvals and run tasks; a task's `ending` says how it really ended ([Task endings](reference/scheduler.md#task-endings)). Walkthrough: [API client guide](guides/api-client.md).
- `melo task list|show|new|pause|resume|stop|follow|level` (also `melo-task`) controls the gateway's tasks; `--gateway URL` picks another gateway. See [Task command](reference/scheduler.md#task-command).
- Stopping: send SIGTERM. The gateway waits up to 3 seconds for runs in flight, then cancels them (they end `crashed`) and exits 143.

Every route the gateway serves:

<!-- agent-docs:routes -->
| Route | Documented in |
|---|---|
| `GET /` | [api-v2](reference/api-v2.md) |
| `GET /api/activity` | [api-v2](reference/api-v2.md#get-apiactivity) |
| `GET /api/approvals/{request_id}` | [api-v2](reference/api-v2.md#get-apiapprovalsrequest_id) |
| `POST /api/approvals/{request_id}` | [api-v2](reference/api-v2.md#post-apiapprovalsrequest_id) |
| `POST /api/auth/login` | [api-v2](reference/api-v2.md) |
| `POST /api/auth/logout` | [api-v2](reference/api-v2.md) |
| `POST /api/auth/rotate` | [api-v2](reference/api-v2.md) |
| `PUT /api/checkin` | [api-v2](reference/api-v2.md) |
| `GET /api/chores` | [api-v2](reference/api-v2.md) |
| `POST /api/chores` | [api-v2](reference/api-v2.md) |
| `DELETE /api/chores/{chore_id}` | [api-v2](reference/api-v2.md) |
| `PATCH /api/chores/{chore_id}` | [api-v2](reference/api-v2.md) |
| `DELETE /api/chores/{chore_id}/keeps` | [api-v2](reference/api-v2.md) |
| `POST /api/chores/{chore_id}/run` | [api-v2](reference/api-v2.md) |
| `GET /api/commands` | [api-v2](reference/api-v2.md#get-apicommands) |
| `POST /api/front/echoes/{echo_id}` | [api-v2](reference/api-v2.md) |
| `GET /api/grants` | [api-v2](reference/api-v2.md#get-apigrants) |
| `DELETE /api/grants/{grant_id}` | [api-v2](reference/api-v2.md#delete-apigrantsid) |
| `GET /api/health` | [api-v2](reference/api-v2.md#get-apihealth) |
| `PUT /api/memory` | [api-v2](reference/api-v2.md) |
| `POST /api/memory/import` | [api-v2](reference/api-v2.md) |
| `DELETE /api/memory/import/offer` | [api-v2](reference/api-v2.md) |
| `GET /api/memory/import/offer` | [api-v2](reference/api-v2.md) |
| `GET /api/memory/import/scan` | [api-v2](reference/api-v2.md) |
| `DELETE /api/memory/import/{batch_id}` | [api-v2](reference/api-v2.md) |
| `GET /api/memory/items` | [api-v2](reference/api-v2.md) |
| `DELETE /api/memory/items/{item_id}` | [api-v2](reference/api-v2.md) |
| `POST /api/memory/items/{item_id}/restore` | [api-v2](reference/api-v2.md) |
| `DELETE /api/memory/records` | [api-v2](reference/api-v2.md) |
| `GET /api/memory/records` | [api-v2](reference/api-v2.md) |
| `DELETE /api/memory/records/{record_id}` | [api-v2](reference/api-v2.md) |
| `PATCH /api/memory/records/{record_id}` | [api-v2](reference/api-v2.md) |
| `POST /api/memory/reindex` | [api-v2](reference/api-v2.md) |
| `POST /api/memory/sleep` | [api-v2](reference/api-v2.md) |
| `DELETE /api/memory/sleep/{run_id}` | [api-v2](reference/api-v2.md) |
| `GET /api/models` | [api-v2](reference/api-v2.md#get-apimodels) |
| `POST /api/presence` | [api-v2](reference/api-v2.md#post-apipresence) |
| `GET /api/serve` | [api-v2](reference/api-v2.md#get-apiserve) |
| `POST /api/serve/switch` | [api-v2](reference/api-v2.md#post-apiserveswitch) |
| `POST /api/serve/switch/cancel` | [api-v2](reference/api-v2.md#post-apiserveswitchcancel) |
| `GET /api/sessions` | [api-v2](reference/api-v2.md#get-apisessions) |
| `POST /api/sessions` | [api-v2](reference/api-v2.md#post-apisessions) |
| `DELETE /api/sessions/{session_id}` | [api-v2](reference/api-v2.md#delete-apisessionsid) |
| `GET /api/sessions/{session_id}` | [api-v2](reference/api-v2.md#get-apisessionsid) |
| `PATCH /api/sessions/{session_id}` | [api-v2](reference/api-v2.md#patch-apisessionsid) |
| `GET /api/sessions/{session_id}/blobs/{sha256}` | [api-v2](reference/api-v2.md#get-apisessionsidblobssha256) |
| `POST /api/sessions/{session_id}/blocks/{request_id}/allow` | [api-v2](reference/api-v2.md#post-apisessionsidblocksrequest_idallow) |
| `GET /api/sessions/{session_id}/browser` | [api-v2](reference/api-v2.md#get-apisessionsidbrowser) |
| `POST /api/sessions/{session_id}/browser/done` | [api-v2](reference/api-v2.md#post-apisessionsidbrowserdone) |
| `WS /api/sessions/{session_id}/browser/live` | [api-v2](reference/api-v2.md#ws-apisessionsidbrowserlive) |
| `GET /api/sessions/{session_id}/browser/picture` | [api-v2](reference/api-v2.md#get-apisessionsidbrowserpicture) |
| `POST /api/sessions/{session_id}/browser/window` | [api-v2](reference/api-v2.md#post-apisessionsidbrowserwindow) |
| `GET /api/sessions/{session_id}/events` | [api-v2](reference/api-v2.md#get-apisessionsidevents) |
| `DELETE /api/sessions/{session_id}/folders` | [api-v2](reference/api-v2.md) |
| `DELETE /api/sessions/{session_id}/grants` | [api-v2](reference/api-v2.md) |
| `DELETE /api/sessions/{session_id}/grants/{tool:path}` | [api-v2](reference/api-v2.md#delete-apisessionsidgrantstool) |
| `POST /api/sessions/{session_id}/level` | [api-v2](reference/api-v2.md#post-apisessionsidlevel) |
| `GET /api/sessions/{session_id}/memory` | [api-v2](reference/api-v2.md#get-apisessionsidmemory) |
| `GET /api/sessions/{session_id}/pictures` | [api-v2](reference/api-v2.md#get-apisessionsidpictures) |
| `POST /api/sessions/{session_id}/retry` | [api-v2](reference/api-v2.md#post-apisessionsidretry) |
| `GET /api/sessions/{session_id}/runs` | [api-v2](reference/api-v2.md#get-apisessionsidruns) |
| `GET /api/sessions/{session_id}/status` | [api-v2](reference/api-v2.md#get-apisessionsidstatus) |
| `GET /api/sessions/{session_id}/submissions/{client_request_id}` | [api-v2](reference/api-v2.md#get-apisessionsidsubmissionsclient_request_id) |
| `POST /api/sessions/{session_id}/submissions/{client_request_id}/cancel` | [api-v2](reference/api-v2.md#post-apisessionsidsubmissionsclient_request_idcancel) |
| `GET /api/sessions/{session_id}/tools` | [api-v2](reference/api-v2.md#get-apisessionsidtools) |
| `GET /api/sessions/{session_id}/turns` | [api-v2](reference/api-v2.md#get-apisessionsidturns) |
| `POST /api/sessions/{session_id}/turns` | [api-v2](reference/api-v2.md#post-apisessionsidturns) |
| `POST /api/sessions/{session_id}/undo` | [api-v2](reference/api-v2.md#post-apisessionsidundo) |
| `GET /api/settings` | [api-v2](reference/api-v2.md#get-apisettings) |
| `PUT /api/settings` | [api-v2](reference/api-v2.md#put-apisettings) |
| `DELETE /api/settings/key` | [api-v2](reference/api-v2.md#delete-apisettingskey) |
| `PUT /api/settings/key` | [api-v2](reference/api-v2.md#put-apisettingskey) |
| `GET /api/settings/key-status` | [api-v2](reference/api-v2.md#get-apisettingskey-status) |
| `POST /api/settings/test` | [api-v2](reference/api-v2.md#post-apisettingstest) |
| `POST /api/setup/cancel` | [api-v2](reference/api-v2.md#post-apisetupcancel) |
| `GET /api/setup/detect` | [api-v2](reference/api-v2.md#get-apisetupdetect) |
| `GET /api/setup/plan` | [api-v2](reference/api-v2.md#get-apisetupplan) |
| `POST /api/setup/start` | [api-v2](reference/api-v2.md#post-apisetupstart) |
| `GET /api/submissions/{client_request_id}` | [api-v2](reference/api-v2.md) |
| `POST /api/submissions/{client_request_id}/cancel` | [api-v2](reference/api-v2.md) |
| `GET /api/tasks` | [api-v2](reference/api-v2.md) |
| `POST /api/tasks` | [api-v2](reference/api-v2.md) |
| `GET /api/tasks/{task_id}` | [api-v2](reference/api-v2.md) |
| `PATCH /api/tasks/{task_id}` | [api-v2](reference/api-v2.md) |
| `POST /api/tasks/{task_id}/blocks/{request_id}/allow` | [api-v2](reference/api-v2.md) |
| `POST /api/tasks/{task_id}/blocks/{request_id}/keep` | [api-v2](reference/api-v2.md) |
| `POST /api/tasks/{task_id}/continue` | [api-v2](reference/api-v2.md) |
| `POST /api/tasks/{task_id}/resume` | [api-v2](reference/api-v2.md) |
| `POST /api/turns/{turn_id}/cancel` | [api-v2](reference/api-v2.md#post-apiturnsturn_idcancel) |
| `POST /api/turns/{turn_id}/notes` | [api-v2](reference/api-v2.md#post-apiturnsturn_idnotes) |
| `POST /api/uploads` | [api-v2](reference/api-v2.md#post-apiuploads) |
| `DELETE /api/uploads/{upload_id}` | [api-v2](reference/api-v2.md) |
| `GET /api/uploads/{upload_id}` | [api-v2](reference/api-v2.md) |
| `POST /api/v2/tasks/{task_id}/resume` | [api-v2](reference/api-v2.md) |
| `POST /v1/agent/run` | [http-v1](reference/http-v1.md#post-v1agentrun) |
| `GET /v1/health` | [http-v1](reference/http-v1.md#get-v1health) |
| `GET /v1/sessions` | [http-v1](reference/http-v1.md#get-v1sessions) |
| `DELETE /v1/sessions/{session_id}` | [http-v1](reference/http-v1.md#delete-v1sessionssession_id) |
<!-- /agent-docs -->

## Python API (experimental)

```python
from localmelo import Agent, Message, ToolCall, ToolDef, ToolResult
from localmelo.melo.contracts.providers import BaseLLMProvider, BaseEmbeddingProvider
from localmelo.support.config import Config, load, save
```

- Only these imports are supported. Build providers with `providers_for(cfg)` (`localmelo.support.backends`) and pass them to `Agent(llm=, embedding=)`.
- One `Agent` per concurrent task: `run_task` has no guard against overlap.
- `run_task(..., on_event=, approval=)` follows a run and decides its tool calls; cancelling the awaiting task stops it. See [Python API](reference/python-api.md), [Events](reference/events.md) and [Sessions](reference/sessions.md).

## Files

| Path | What it holds |
|---|---|
| `~/.cache/localmelo/config.toml` | The configuration; a missing file means not configured. No flag or variable moves it: set `HOME`. |
| `~/.cache/localmelo/memory/` | `history.db`, `procedures.db`, and `long_term.db` outside `legacy`, with `LOCALMELO_PERSIST_MEMORY` set; `LOCALMELO_MEMORY_DIR` or `--memory-dir` moves it. |
| `~/.cache/localmelo/gateway/` | `sessions.db` (console chats, tasks, grants), `auth/token`, `endpoint.json` (the running gateway). |
| `~/.cache/localmelo/cli/sessions.db` | The terminal's own chats. |
| `~/.cache/localmelo/scheduler/runs/` | The gateway's model-call log. |
| `~/.cache/localmelo/logs/` | The terminal's and a background gateway's logs; the login service logs to `~/.localmelo/logs/`. |

Every file, its mode and who creates it: [Files and directories](reference/configuration.md#files-and-directories). Environment variables: [Environment variables](reference/configuration.md#environment-variables). The tables of the stores: [Memory](reference/memory.md#files-on-disk).

## Gotchas

- `melo serve` ignores a task; direct mode ignores `--host` and `--port`.
- A request timeout only counts silence: a silent server costs about 3 times the request timeout plus 5 s per request, because of two retries.
- The `ollama` backend's `result` can start with a `<think>…</think>` block.
- A `session_id` must match `^[a-zA-Z0-9_-]{1,64}$`; omit it for a new session. `DELETE /v1/sessions/{id}` on a running session is 409 `session_busy`, and the run goes on.
- A shell `&` job ignores SIGINT: stop a background `melo` with SIGTERM. A direct run exits 143 within 8 seconds.
- Every chat request carries the header `X-Localmelo-Call-Kind` (`plan`, `reflect`, `review` or `retrieve`).

## Working on the code

- The folder that holds `pyproject.toml` is the package `localmelo`: `melo/agent/agent.py` imports as `localmelo.melo.agent.agent`. Never create `localmelo/localmelo/`.
- `melo/` (the core) never imports `support/` (backends, gateway, config, CLI); `tests/test_layering.py` checks it.
- Setup `uv sync --extra dev --extra gateway`; while working `uv run pytest --testmon -q`; before a pull request `uv run pytest -n auto -q`, `uv run ruff check .`, `uv run black --check .`, `uv run mypy .`.
- Rules for changes and docs: [CONTRIBUTING.md](https://github.com/localmelo/localmelo/blob/main/CONTRIBUTING.md) and [Maintaining the docs](project/documentation.md). This page changes only when the contract changes; its two tables come from `uv run python scripts/agent_docs.py sync`.
- Every page: [llms.txt](llms.txt) and the [documentation index](README.md).
