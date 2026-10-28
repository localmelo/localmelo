# localmelo for agents

Humans: start at the [README](https://github.com/localmelo/localmelo#readme) or the [website](https://localmelo.github.io/localmelo/).

## What it is

- A local agent: it runs one task with tools (shell, Python, files, MCP) and memory, on a model server the user runs (any OpenAI-compatible server, Ollama) or a cloud API. It never downloads, starts or serves models.
- Three surfaces: the `melo` CLI (a fixed exit code per run, and one JSON line with `--output-format json`), the Python `Agent` (experimental), an HTTP gateway on 127.0.0.1.
- Pre-alpha: CLI, HTTP API and Python API may change before v0.1.0.

## Rules (read before running anything)

- Tools run as the user with no sandbox: never on untrusted input or near data the user cannot lose. Pass `--builtins none` (or `--mcp-only`) when no built-in tool is needed.
- Never expose the gateway: host `127.0.0.1`, `localhost` or `::1` only (check `--host` and the user's `[gateway] host`); no `0.0.0.0`, LAN IP, reverse proxy or Tailscale Serve. It has no authentication: whatever reaches the port runs `shell_exec`, `python_exec` and `file_write` as the user, unapproved, and makes the web workspace fetch any URL. Any other host only logs a warning and turns the Host/Origin guard off.
- Never edit the user's `~/.cache/localmelo/config.toml` or delete `~/.cache/localmelo/` (cleaning `~/.cache` deletes both). For tests or your own backend, run `melo` with a temporary `HOME`.
- No API keys in the config file; `api_key_env` holds the variable's name.
- Never run `--daemon install` in tests. Never run bare `melo` (no task) in a pty or terminal under the user's `HOME`: with no config it asks setup questions and writes `~/.cache/localmelo/config.toml`; it also writes `~/.cache/localmelo/cli/sessions.db`, `cli_history` and `logs/cli.log`. Without a TTY, bare `melo` exits 2 with three stderr lines, and `--serve` with no config exits 1 with one; neither asks anything. `--reconfigure` without a TTY reads the wizard's answers from stdin (EOF: `Setup cancelled.`, exit 0).

## Install

```bash
git clone https://github.com/localmelo/localmelo.git
cd localmelo                 # keep this name: the package is built from this directory
uv sync --extra gateway      # pip: python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[gateway]"
uv run melo --help
```

- `melo` lands in `.venv/bin/`, not on PATH. Below, `melo` means `uv run melo` in the clone, `uv run --project CLONE melo` elsewhere, or `CLONE/.venv/bin/melo` (for background jobs; see Gotchas). Scripts: `uv run [--project CLONE] python SCRIPT.py` or `CLONE/.venv/bin/python SCRIPT.py`. Under a temporary `HOME` prefer the `.venv/bin/` paths: `uv run` keeps its cache in `$HOME/.cache/uv` and may resolve and download again.
- Python >= 3.11; the `gateway` extra (fastapi, uvicorn) is needed for `--serve`. A renamed clone or ZIP folder gives `ModuleNotFoundError: No module named 'localmelo'`.
- Commands: `melo`, `melo-retrieve` and `melo-procedures` (offline only). macOS tested; Linux untested; no Windows.

## Use it

### CLI, one task (best for scripts)

```bash
melo --base-url http://127.0.0.1:8000/v1 --chat-model MODEL \
  --output-format json --timeout 300 --builtins none "What is the capital of France?"
echo $?   # 0 completed; 1 crashed (JSON line) or no/invalid config (no JSON line); 2 other status (JSON line) or usage error (no JSON line)
```

- `--base-url` is used verbatim (include `/v1`): no config file, no API key, a 120 s chat request timeout unless `--request-timeout`; with a task, `--chat-model` defaults to `default`. Cloud APIs need the config file.
- oMLX (Apple silicon, macOS 15+): `omlx serve --model-dir DIR --port 8000` (a port saved in oMLX's settings.json wins when `--port` is left out; each subfolder of DIR with a `config.json` is a model; its folder name, or an alias set in oMLX, is the model id), then `--base-url http://127.0.0.1:8000/v1 --chat-model ID`; in the config file it is the `vllm` backend. oMLX must accept requests without an API key (no `--api-key`, no key from its admin page or `OMLX_API_KEY`, or "Allow inference without API key" on): localmelo sends none to local servers, and HTTP 401 means a key is set.
- Without `--base-url`: the config file; `--chat-model` and `--request-timeout` override it for the run. No config: exit 1, `No backend configured`.
- Drop `--builtins none` to give the model the four built-in tools. Quote the task.
- Direct-mode flags (first value is the default):
  - `--max-steps N` (30 planning steps); `--attempt-policy reflect|none` (`reflect`: attempts of up to 10 steps, at most 5, a reflection between); `--reflection-gate flags|utility`; `--tool-exposure retrieve|all`.
  - `--builtins all|none|LIST` (comma list of `file_read,file_write,python_exec,shell_exec`); `--mcp-config PATH` (stdio MCP servers); `--mcp-only` = `--builtins none --tool-exposure all` (needs `--mcp-config`; do not also pass `--builtins` or `--tool-exposure`).
  - `--memory on|off` (long-term memory); `--memory-dir DIR` (save memory as SQLite); `--embedding-base-url URL`, `--embedding-model NAME` (embeddings for a `--base-url` run).
  - `--timeout S` (run wall clock; none); `--request-timeout S` (per chat request; `request_timeout_s`, else 120, 300 for `ollama`); `--output-format text|json`.
  - `--progress auto|off` (`auto`: a progress line on stderr, `\r\x1b[2K` redraws, erased before the result, only when stderr is a TTY, `TERM` is not `dumb` and the output is text; with JSON output or no TTY nothing is drawn and the run is unchanged; not drawn while melo is a background job, `tcgetpgrp(2) != getpgrp()`; cut to the width with East Asian Ambiguous characters counted 2 cells when the locale (`LC_ALL`, `LC_CTYPE`, `LANG`) starts with `zh`, `ja` or `ko`); `--routine NAME` (procedural routine; blank is a usage error). Past-run facts on the line need `LOCALMELO_PERSIST_MEMORY=1` or `--memory-dir`; a similar match's facts follow `like <routine title>:` (or `like a similar task:` when the event names no routine), never `last time`; a `shell_exec` command sent as an argv array (a list of strings, which the tool refuses) is labelled by its words (`shlex.join`), not by its JSON.

Other modes:

- `melo ""`: the plain conversation loop on stdin (`/help`, `/exit` or `/quit`, Ctrl-D). Takes only `--base-url` and `--chat-model`; exits 0 even if every turn failed. Do not parse its output.
- `melo` with no task, no `--serve`, no `--daemon`: the interactive UI (prompt_toolkit, inline). Needs stdin and stdout TTYs and `TERM` not `dumb`, else exit 2. Backend, first that applies: `--reconfigure` (wizard); `--base-url URL` (model: `--chat-model`, else the first id from `GET URL/models` without `embed`/`rerank`, else `default` with a warning under the banner; a URL without `/v1` is listed at `URL/v1/models` and, when that answers, chatted with at `URL/v1`; with no config file asks `Save URL · MODEL as the default backend? [Y/n]`); the config (no check before the UI; one `GET /models` after the banner, warning only); no config: `Model server URL [..]:` (prefilled from `GET /v1/models` on 127.0.0.1 ports 8000, 8400, 11434, 30000; `LOCALMELO_DETECT_URLS`, comma-separated, replaces them, empty turns it off) and `Model [N]:`, saves `chat_backend = "ollama"` with `[ollama] chat_url`/`chat_model` when the server answers `GET /api/version` like Ollama, else `chat_backend = "vllm"` with `[vllm] chat_url`/`chat_model`, and `embedding_backend = "none"`; failure: `[r]etry, [s]tart anyway, [q]uit [r]:`. Ctrl-D at a question: exit 0; Ctrl+C: 130. Takes `--workspace DIR`, `--yolo` (no approvals), `--max-steps`, `--attempt-policy`, `--reflection-gate`, `--builtins`, `--memory`, `--memory-dir`, `--request-timeout`, `--embedding-*`. Refuses (exit 2) `--tool-exposure`, `--timeout`, `--output-format`, `--progress`, `--routine`, `--mcp-config`, `--mcp-only`, `--host`, `--port`. Commands: `/help`, `/new` (`/clear`), `/model [NAME|N]` (lists; switches only to a listed id or number 1..N, else `✗ URL lists no model named 'x'. /model lists them.`, nothing saved; with no list available, switches anyway with a warning), `/backend [URL]` (switches server, saved as `ollama` when it answers `GET /api/version` like Ollama, else `vllm`; keeps the config's embedding, `request_timeout_s`, `extra_body`), `/exit` (`/quit`); both save the choice, except in a `--base-url` session that was not saved. A draft that starts with a space is a message. A turn that could not run puts its message back in the input box. Approval panel before `shell_exec`/`python_exec`/`file_write`: keys `1` once, `2` the tool for the session, `3` deny with a reason; no letters; keys ignored 0.4 s after it opens. Esc stops a turn (`cancelled`, cause `user_stop`). `shell_exec`/`python_exec` limit 600 s. Exit 0 on `/exit`, Ctrl+D, Ctrl+C twice; SIGTERM/SIGHUP: exit by the signal. Not for scripts: use direct mode.
- `melo --serve [--host H] [--port P]`, `melo --reconfigure` (the wizard, then the UI; with `--serve`, then the gateway; without a TTY it reads answers from stdin, then exits 0 unless `--serve`; with `--base-url` and no task: exit 2), `melo --daemon install|uninstall|status` (macOS launchd). Direct-mode flags with `--serve` or in `melo ""` are usage errors (exit 2), so are `--base-url`/`--chat-model` with `--serve`/`--daemon`, and `--workspace`/`--yolo` with a task, `--serve` or `--daemon`.
- Wizard: every prompt defaults to the config's value (embedding: `None` when unset); a failed chat check asks `[r]etry, [s]ave anyway, [N]o [N]:`; a failed embedding check saves with `embedding_backend = "none"`. Never asks for an API key for a local server.

### Python API (experimental)

```python
import asyncio
from localmelo import Agent, AgentOptions
from localmelo.support.providers.llm.openai_compat import OpenAICompatLLM

async def main() -> None:
    llm = OpenAICompatLLM(base_url="http://127.0.0.1:8000/v1", model="MODEL")  # include /v1
    agent = Agent(llm=llm, options=AgentOptions(builtins=[], timeout_s=300))  # builtins=[]: no tools
    try:
        r = await agent.run_task("What is the capital of France?")
        print(r.status, r.result or r.error)
    finally:
        await agent.close()  # there is no `async with` and no aclose()

asyncio.run(main())
```

- `localmelo` also exports `AgentEvent`, `Message`, `RunResult`, `ToolCall`, `ToolDef`, `ToolResult`; event and approval types are in `localmelo.melo.contracts`, `ProviderError` only in `localmelo.melo.contracts.providers`.
- `Agent(config=None, *, llm=None, embedding=None, options=None, tool_providers=())`; `llm=` wins; neither raises `TypeError`. From the config file: `from localmelo.support.config import load`; `cfg = load(); cfg.validate_or_raise(); Agent(config=cfg)`.
- `OpenAICompatLLM(base_url, model, api_key=None, timeout=120.0, *, extra_body=None)`: `api_key` goes out as a Bearer token; misspelled keywords are silently ignored; `timeout=None` means no timeout at all.
- `AgentOptions` (frozen; bad values raise `ValueError`): `max_steps=30`, `attempt_policy="reflect"`, `steps_per_attempt=10`, `max_attempts=5`, `reflection_gate="flags"`, `tool_exposure="retrieve"`, `builtins=None` (all four), `memory=True`, `memory_dir=None`, `timeout_s=None`, `workspace_root=None`, `record_answers=False`, `tool_timeout_s=None` (needs `workspace_root`).
- `await agent.run_task(query, *, on_event=None, approval=None, observe=None, progress=False, routine=None) -> RunResult`. `await agent.run(query) -> str`: the result, or `[STATUS] ERROR` when it is empty.
- `on_event`: a plain sync callable taking an `AgentEvent` (`type`, `run_id`, `seq`, `ts`, `data`); `async def` raises `TypeError`; never block in it. Types: `run.started`, `step.started`, `assistant.message`, `tool.started`, `tool.finished`, `notice`, `reflection.started`, `reflection`, `run.progress` (only with `progress=True`), `run.finished` (`RunResult(**event.data["result"])`).
- `observe`: a sink like `on_event` that does not make the run interactive (a cancelled run ends `crashed`); not both. `progress=True` with a sink adds `run.progress` (level 0 facts, or level 1 with past runs' facts and named steps; no percent in this release). `routine="NAME"`: one procedural routine in any folder, whatever the wording; blank raises `ValueError`. `localmelo.support.progress_line.format_past(payload, width)` gives that line's past-facts-and-items segment alone (same wording and widths; empty below level 1 or when there is nothing).
- `approval`: an object with `async def decide(request: ApprovalRequest) -> ApprovalDecision` (e.g. `ApprovalDecision(allow=False, reason="...")`; `AllowAllPolicy`, `DenyAllPolicy` exist), asked before `shell_exec`, `python_exec`, `file_write` and tools marked as needing approval; a raise denies. Without a policy every tool runs.
- `workspace_root=DIR`: `file_read` and `file_write` stay inside DIR; `shell_exec` and `python_exec` start in DIR as child processes but can reach any path.
- Stop a run: cancel the task awaiting `run_task` (it re-raises `CancelledError`). With `on_event` or `approval` the run ends `cancelled`, its result only in `run.finished`; with neither (`observe` does not count) it ends `crashed`, no result.
- MCP: `tool_providers=[McpToolProvider(load_mcp_config("mcp.json"))]` (`localmelo.support.mcp`).
- Sessions (experimental, `localmelo.support.sessions`): conversations in SQLite that survive a restart. The agent needs `record_answers=True` and a `workspace_root`, and `await agent.start()` first. `store = SessionStore(PATH, writer="cli")` (or `"gateway"`; a file belongs to one writer); `runner = SessionRunner(agent, store, None, new_session=NewSession(workspace=agent.workspace_root))`; `await runner.run_turn(text)` returns a `RunResult`; keep `runner.session_id`. Later, with a fresh agent: `SessionRunner(agent, store, session_id)`, `await runner.resume()`. Close the runner, then the agent, then `await store.aclose()`. `SessionRunner(..., progress=True)` asks each turn's run for `run.progress` and stores it (kept across `replace_agent`); without it a session turn reports none, and the interactive UI does not ask yet. `run_turn`/`start_turn` take `routine="NAME"` and pass it to `run_task` only when given. See [sessions.md](sessions.md).

### HTTP gateway

The gateway reads only `~/.cache/localmelo/config.toml`. For your own backend or a test, give it a temporary `HOME` with its own file (here for any OpenAI-compatible server):

```bash
T=$(mktemp -d); mkdir -p "$T/.cache/localmelo"
cat > "$T/.cache/localmelo/config.toml" <<'EOF'
chat_backend = "vllm"
embedding_backend = "none"

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "MODEL"
EOF
PORT=18401   # a free port that is not the user's gateway port (default 8401)
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null && { echo "port $PORT taken" >&2; exit 1; }   # run as a script
HOME=$T CLONE/.venv/bin/melo --serve --port $PORT > "$T/gateway.log" 2>&1 & GW=$!   # blocks until stopped
for i in $(seq 60); do curl -sf http://127.0.0.1:$PORT/v1/health >/dev/null && break; kill -0 $GW 2>/dev/null || break; sleep 0.5; done
curl -sf http://127.0.0.1:$PORT/v1/health >/dev/null || { echo "gateway did not start; see $T/gateway.log" >&2; kill $GW 2>/dev/null; exit 1; }
curl -s --max-time 600 -X POST http://127.0.0.1:$PORT/v1/agent/run -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
# {"result":"Paris is the capital of France.","session_id":"850f5874fd49"}
kill -TERM $GW; wait $GW   # exit 143 once runs in flight end; SIGKILL after your own deadline
```

- `POST /v1/agent/run` with `Content-Type: application/json` and `{"query": str, "session_id"?: str}`: 200 `{"result", "session_id"}`; 400 `{"error"}`; 422 `{"detail": [...]}` (non-object body or no JSON content type); 500 `{"error", "session_id"}`.
- `GET /v1/health`: `{"status": "ok", "uptime" (s), "sessions": {"active", "busy", "max": 50}, "chat_backend", "embedding_backend"}`.
- `GET /v1/sessions`: `{"sessions": [{"session_id", "status": "idle" or "running", "idle_seconds", "created_at" (Unix time, float), "is_busy"}]}`. `DELETE /v1/sessions/{id}`: 200 `{"closed": id}`; 409 `{"error", "session_id"}` while a task runs (the session and the task are left alone); 404 `{"error": "session not found"}`.
- `query`: non-empty, at most 100,000 characters. `session_id`: `^[a-zA-Z0-9_-]{1,64}$`; omit it for a new one (12 hex characters); an unknown id creates that session.
- The POST blocks until the run ends (no streaming) and the gateway sets no time limit: give the client a deadline (`curl --max-time`).
- 200 means the run finished, not that it completed; there is no status field (an empty answer is `"result": "[STATUS] ERROR"`). For the status, use the CLI with `--output-format json`.
- Sessions: one request at a time each (the next waits); RAM only; at most 50 (more only while all are busy); idle TTL 3600 s, checked when a session is created. Each has all four built-ins, no approval policy, workspace root, MCP or time limit, `max_steps` 30, `record_answers=True`.
- Loopback guard (loopback host only): a `Host` other than `127.0.0.1:PORT`, `localhost:PORT` or `[::1]:PORT` gets 421 `{"error", "code": "misdirected_request"}`; a foreign `Origin` on POST, PUT, PATCH, DELETE or OPTIONS, or a `Sec-Fetch-Site` other than `same-origin` or `none` under `/v1/`, gets 403 `{"error", "code": "foreign_origin"}`. curl passes.
- `/` is the web workspace (`/v1/playground/*` is internal). Starting the gateway creates `~/.cache/localmelo/playground/`, its history, always on disk, even if `/` is never opened. No `/v1/models`; health never contacts the model server.

## Output and results

The CLI example's JSON line (field order fixed; `RunResult.to_dict()` is the same without `localmelo_version`):

```json
{"task_id": "ee8befecca3b4fd4bae6499fa0c73704", "status": "completed", "result": "Paris is the capital of France.", "error": "", "steps": 1, "attempts": 1, "n_llm": 1, "n_main": 0, "n_aux": 1, "n_tool": 0, "llm_errors": 0, "usage": {"prompt_tokens": 120, "completion_tokens": 12, "total_tokens": 132}, "duration_s": 0.015203958842903376, "config": {"max_steps": 30, ..., "tools": []}, "localmelo_version": "<git describe or version>"}
```

- `n_llm` requests answered, `llm_errors` failed attempts (sum = requests sent), `n_main` with tools offered, `n_aux` without, `n_tool` tool calls (refused included). `config`: the `AgentOptions` fields except `workspace_root`, `record_answers` and `tool_timeout_s` (`memory_dir` as resolved, `builtins` as registered), and `tools`, every registered tool's name (not only those a step offered). `localmelo_version`: `git describe` of the checkout, else the installed version, else `unknown`.
- Text mode: `result` on stdout (tool writes to fd 1 go to stderr); a non-completed run adds `melo: run ended with status STATUS: ERROR` on stderr and may still print a best-effort result.
- Statuses, exit 2 unless noted:
  - `completed` (exit 0). `timeout`: `Timed out after N s`. `budget_exhausted`: `Max steps reached` or `Max attempts reached`.
  - `failed`: a check refused, or 2 empty replies in a row. `gate_stopped`: the reflection stopped the run with budget left. `stuck` (`--attempt-policy none` only): the same calls or errors repeated.
  - `provider_error`: a chat request still failed after its retries (network errors, timeouts and HTTP 429/502/503/504 are retried twice, after 1 s, then 4 s more), or an embedding request failed (never retried). `setup_error`: an MCP server or tool provider failed to start; permanent for that `Agent`.
  - `crashed` (exit 1, or the signal's code below): an unexpected exception or a signal (`Cancelled by SIGTERM` or `Cancelled by SIGHUP`; SIGINT records `CancelledError` and prints a traceback on stderr). `cancelled`: Python runs only.
- Other exit codes (no JSON line): 2 usage error, bad `--mcp-config`, `--memory-dir` or `LOCALMELO_THINKING_RETENTION_DAYS`; 1 no backend, invalid config or TOML, unset cloud key variable, the wizard needed without a TTY; 3 gateway port in use; 143, 129, 130 for SIGTERM, SIGHUP, SIGINT (the task record is saved as `crashed`); bare `melo`: 2 no TTY, `TERM=dumb` or `--workspace` not a directory, 1 its session file, agent start or log folder failed.

## Configuration

- File: `~/.cache/localmelo/config.toml`, fixed (no flag or env var; set `HOME` to isolate). Missing file = not configured.
- Top level, before any `[table]`: `chat_backend` (`mlc`, `ollama`, `vllm`, `sglang`, `openai`, `anthropic`, `gemini`, `nvidia`) and `embedding_backend` (`mlc`, `ollama`, `vllm`, `sglang`, `none`), both required; optional `request_timeout_s` (unquoted number > 0).
- Local table (`[mlc]`, `[ollama]`, `[vllm]`, `[sglang]`): `chat_url`, `chat_model`, `embedding_url`, `embedding_model`, `extra_body` (a table merged into every chat request). `mlc`, `vllm`, `sglang` add a missing `/v1`; `ollama` chat uses native `/api/chat`. `vllm` fits any OpenAI-compatible server.
- Cloud table (`[openai]`, `[anthropic]`, `[gemini]`, `[nvidia]`): `api_key_env` (the NAME of the env var holding the key; export it where `melo` runs), `chat_model`, `base_url` (used as given), `extra_body`. No cloud embeddings.
- `[gateway]`: `host` (`127.0.0.1`), `port` (`8401`); `--host` and `--port` override them for one run.
- Errors exit 1: `Configuration errors:` on stderr, or for invalid TOML a raw `TOMLDecodeError` traceback.
- `LOCALMELO_PERSIST_MEMORY` (any non-empty value, `0` included) saves memory as SQLite in `LOCALMELO_MEMORY_DIR` (default `~/.cache/localmelo/memory`; `~` not expanded).
- `LOCALMELO_THINKING_RETENTION_DAYS` (default 30; a number above 0, such as `7` or `0.5`): days of model reasoning kept in `history.db` and `sessions.db`. Any other value: exit 2 with one line before the run.
- Every chat request a run sends carries the header `X-Localmelo-Call-Kind: plan` or `reflect` (`melo-retrieve`'s: `retrieve`); embedding requests and the setup wizard's probe carry none.

## Tools and MCP

- `shell_exec(command)`: stdin `/dev/null`; process group killed after 30 s; refuses `rm -rf /...`, `mkfs`, `dd if=`, fork bombs, `shutdown`, `reboot`.
- `python_exec(code)`: `exec` in the agent's process, on its event loop, so it cannot be interrupted (a child process with a workspace root).
- `file_read(path)`: any file the user can read. `file_write(path, content)`: creates or replaces any file the user can write; in the interactive UI, only inside the workspace.
- A Python run with an approval policy, and the interactive UI (bare `melo`, unless `--yolo`), gate `shell_exec`, `python_exec` and `file_write`; direct mode, `melo ""` and the gateway have none. Output is cut at 50,000 characters.
- `retrieve` offers the built-ins every step, plus extra (MCP or provided) tools the conversation already named and the BM25 top 3 for the task; `all` offers every tool every step. Unoffered calls are refused (`Tool 'NAME' was not offered in this step.`).
- MCP file: `{"mcpServers": {"time": {"command": "uvx", "args": ["mcp-server-time"]}}}` (`servers` also accepted). Optional keys: `env`, `cwd`, `timeout` (120), `startup_timeout` (30), `include_tools`, `require_approval` (`true`, `false` or names), `disabled`.
- Only stdio servers. A server gets only `HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM`, `USER`, `LANG`, `LC_ALL`, `LC_CTYPE`, `TMPDIR` plus its `env` (no `${VAR}` expansion).
- Tool names: `mcp__SERVER__TOOL` (other characters become `_`; over 64 cut with a hash); `config.tools` lists them. Gateway sessions have no MCP tools.

## Memory

- Working memory: the last 20 messages, in RAM. History (task and step records) and long-term memory: RAM unless persisted.
- Persistence: `--memory-dir DIR` or `AgentOptions(memory_dir=DIR)` wins, else `LOCALMELO_PERSIST_MEMORY`. Files: `history.db` always (tasks, steps, and in table `llm_calls` every model reply with its redacted reasoning, kind `plan` or `reflect`, usage and the steps it made), `long_term.db` only with an embedding provider and memory on, `procedures.db` always (procedural memory: each run's step counts and tool-call labels, added after `run.finished`, a few ms, at most 2 s, skipped after SIGTERM; without persistence it stays in RAM). A directory localmelo creates is `0700`, a new `history.db`, `long_term.db` or `procedures.db` `0600`; an existing file keeps its mode. Export a run's calls: `melo-retrieve --export-run MEMORY_DIR`; procedural memory: `melo-procedures export --memory-dir DIR`.
- Long-term memory needs an embedding backend; without one it is inert although `config.memory` is `true`. A new `Agent` or `melo` process recalls earlier runs only through it; history is written but never read back into a run.
- Multi-turn chat: one `Agent` with `record_answers=True` (without it, later runs see earlier queries but not answers), or one gateway `session_id`.
- Sessions: used by the interactive UI (`~/.cache/localmelo/cli/sessions.db`, every turn with its tool calls and output; your messages also in `cli_history`); Python API experimental (`~/.cache/localmelo/gateway/` by default otherwise). Plain text, files `0600`.
- Delete memory with the Python API: `await agent.hippo.forget(task_id)` removes the task's `tasks`, `steps` and `llm_calls` rows and its long-term reflections (entries whose metadata `task_id` names it); `forget_all()` removes both stores, step summaries included. No command yet. Sessions, procedures and the web workspace's files (`~/.cache/localmelo/playground/`) go only by deleting them with localmelo stopped. Not encrypted. One gateway's sessions share one set of files.

## Gotchas

- Exit 1 and exit 2 can come with no JSON line (setup or usage error, message on stderr). Check that stdout is non-empty before parsing, then read `status`.
- Always set `--timeout` (or `timeout_s`): the request timeout counts per read, so a server that keeps sending bytes never trips it, and a silent server costs about 3 x the request timeout + 5 s per request (two retries). It cannot interrupt `python_exec` without a workspace root (direct mode has none).
- `--serve` silently ignores a positional task; direct mode silently ignores `--host`, `--port`, `--reconfigure`.
- A non-string `session_id` gets a plain-text HTTP 500; omit it or send `null` for a new session.
- One `Agent` per concurrent task: `run_task` has no overlap guard (runs share working memory and counters).
- `ollama` backend: `result` can start with a `<think>...</think>` block.
- Gateway: `python_exec` runs inside the gateway process and blocks every request until it returns. `DELETE` on a session running a task: 409 `{"error", "session_id"}`, and the run goes on; `/v1/agent` has no cancel route (the web workspace's cancel is for its own chats). At shutdown every session is closed, a busy one included, and a task still running then ends as `crashed`.
- Stopping: `&` jobs from a non-interactive shell ignore SIGINT; send SIGTERM. Direct mode exits 143 within 8 s; the gateway exits 143 only after its runs in flight finish (no time limit): SIGKILL it after your own deadline.
- Under `uv run`, `$!` is uv: SIGTERM reaches melo, SIGKILL does not (the gateway keeps listening). In scripts start `.venv/bin/melo`, or SIGKILL the child (`pkill -KILL -f 'melo --serve --port PORT'`).
- `--daemon install` takes `--port` but not `--host`, gets no shell env vars (no cloud keys), and its KeepAlive restarts a killed gateway: use `--daemon uninstall`.

## Status of features

- Available: direct mode, the gateway, the interactive UI (bare `melo` in a terminal, with first-run setup, live tool blocks, approvals, `/model`, `/backend`), the `melo ""` loop, MCP stdio servers (direct mode, Python API), the macOS background service.
- Experimental: the Python API. Offline only (runs do not use them): `melo-retrieve` and its `bm25`, `llm`, `embedding` retrievers; `melo-procedures` (`episodes`, `replay`, `export`, `forget`: replays the progress rules on recorded runs; every run itself writes procedural memory at its end); the scheduler core in `support/scheduler/`. Sessions (`support/sessions/`): used by the interactive UI (`cli/sessions.db`); Python API experimental; not used by direct mode or the gateway yet.
- Planned, not on main: the interactive UI's thinking excerpts, notices, diffs, `-c`/`-r` resume, attachments and `!` shell mode, MCP in it; gateway sessions that survive a restart; the model's reasoning kept as memory notes and retrieved during runs; several tasks on one model server, the user's conversation first; a percent bar, off until repeated runs pass the replay.

## Working on the code

- The directory that holds `pyproject.toml` IS the package `localmelo` (`package-dir`: `localmelo = "."`): `melo/agent/agent.py` imports as `localmelo.melo.agent.agent`. Keep the clone named `localmelo`; never create `localmelo/localmelo/`.
- `melo/` is the core (run loop, checks, executor, memory, contracts); `support/` holds backends, providers, gateway, MCP, config, wizard, `melo-retrieve`, `melo-procedures`, scheduler. `melo/` does not import `support/` (exceptions: `tests/test_layering.py`).
- Setup: `uv sync --extra dev --extra gateway`. Tests while working: `uv run pytest --testmon -q` (after its first run, only tests that cover the changed files run). Before a pull request: `uv run pytest -n auto -q` (the offline suite in parallel). A plain run opens no socket: `tests/smoke/test_backend_smoke.py` is skipped unless `LOCALMELO_SMOKE=1`, and `--ignore=tests/smoke/test_backend_smoke.py` still works; pre-commit hooks run it too.
- Lint: `uv run ruff check .`, `uv run black --check .`, `uv run mypy .`. Web workspace JS (Node from `.nvmrc`): `node --test tests/gateway/*.mjs`.
- Commits `[Area] Imperative summary`; PR titles add the type (`[Feature][Agent] ...`). Tests and docs (with `README.zh-CN.md`, `docs/zh-CN/`) in the same PR, in English ([CONTRIBUTING.md](https://github.com/localmelo/localmelo/blob/main/CONTRIBUTING.md)).

## Where to read more

- Use: [quickstart.md](quickstart.md), [cli.md](cli.md) ([exit codes](cli.md#exit-codes), [signals](cli.md#signals)), [run-results.md](run-results.md) ([retries](run-results.md#provider-errors-and-retries)), [python-api.md](python-api.md), [events.md](events.md), [sessions.md](sessions.md), [gateway.md](gateway.md) ([keep it private](gateway.md#keep-the-gateway-private)).
- Setup: [configuration.md](configuration.md), [tools.md](tools.md) ([MCP config file](tools.md#mcp-config-file), [what the checks do](tools.md#what-the-checks-do)), [memory.md](memory.md), [troubleshooting.md](troubleshooting.md).
- Background: [how-it-works.md](how-it-works.md), [evaluations.md](evaluations.md) (why `flags` is the default gate), [retrieval.md](retrieval.md), [procedures.md](procedures.md), [scheduler.md](scheduler.md), [SECURITY.md](https://github.com/localmelo/localmelo/blob/main/SECURITY.md), [CHANGELOG.md](https://github.com/localmelo/localmelo/blob/main/CHANGELOG.md).
