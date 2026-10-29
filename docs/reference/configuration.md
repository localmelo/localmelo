# Configuration

This page explains how localmelo connects to your model servers: the first
run, the setup wizard, the config file, the supported backends, and the
settings you can change.

## The config file

The configuration lives in `~/.cache/localmelo/config.toml`. A minimal file
for Ollama without embeddings:

```toml
chat_backend = "ollama"
embedding_backend = "none"

[ollama]
chat_url = "http://127.0.0.1:11434"
chat_model = "qwen3:8b"
```

> [!WARNING]
> Tools that clean caches may delete `~/.cache`, which also holds your saved
> memory. Keep a copy of a config file you edited by hand.

The wizard writes a table for every backend, but localmelo reads only the
tables of the chosen backends.

| Key | Table | Default | Description |
|---|---|---|---|
| `chat_backend` | top level | not set | The chat backend; see [Backends](../guides/model-servers.md#backends). Required. |
| `embedding_backend` | top level | not set | `mlc`, `ollama`, `vllm`, `sglang` or `none`. Required. |
| `request_timeout_s` | top level | not set | Chat request timeout in seconds; see [Chat request timeout](#chat-request-timeout). |
| `chat_url` | local backend | not set | The model server's URL. |
| `chat_model` | local or cloud backend | not set | The model to use. |
| `embedding_url` | local backend | not set | The embedding server's URL. |
| `embedding_model` | local backend | not set | The embedding model to use. |
| `extra_body` | local or cloud backend | not set | Extra fields for every chat request; see [Extra request fields](#extra-request-fields-extra_body). |
| `reasoning_field` | local or cloud backend | not set | The request key reasoning is sent back under: `auto`, `reasoning_content` or `reasoning`; `[ollama]` takes `auto` or `thinking` instead. See [Reasoning](#reasoning). |
| `requires_reasoning_replay` | local or cloud backend | `false` | `true` when the server rejects a request that omits a reply's reasoning. See [Reasoning](#reasoning). |
| `context_window`, `output_reserve`, `safety_margin` | `[agent]` | not set | Opt-in context request accounting; all three positive integers must resolve, with a positive input limit. See [Context accounting](#context-accounting). |
| `reasoning` | `[agent]` | `"legacy"` | `legacy`, `drop`, `replay`, `compress` or `compress+replay`; the saved default for direct runs, the interactive UI and API v2 console sessions. See [Reasoning](#reasoning). |
| `replay_turns` | `[agent]` | not set | With `replay` or `compress+replay`, the newest N assistant turns whose reasoning is sent; omit for every turn in scope. |
| `note_writer` | `[agent]` | `"auto"` | In the compress modes, `auto` (the model's `<note>`, else the heuristic) or `heuristic`. |
| `tool_env` | `[agent]` | `[]` | Extra environment variable names inherited by shell/Python tools in config-backed runs and sessions. Values are never saved; see [Tool environment](#tool-environment). |
| `retriever` | `[retrieval]` | `"auto"` | `auto`, `off`, `bm25`, `llm` or `embedding`: how recall ranks documents. See [Recall in the loop](../concepts/memory.md#recall-in-the-loop). |
| `recall_at` | `[retrieval]` | not set | A list of `start`, `attempt`, `step` and `tool`. Omit for `start`, `attempt` and `step`; `tool` is never on unless named. |
| `k` | `[retrieval]` | `3` | Task-local hits per recall, from 1 to 10. |
| `kinds` | `[retrieval]` | not set | A list of `note`, `draft`, `thought`, `digest`, `reflection` and `long`. Omit for all but `thought`. |
| `placement` | `[retrieval]` | `"tool"` | `tool` (append to the last tool result) or `message` (its own user message). `message` is refused in `replay` and `compress+replay`. |
| `min_score` | `[retrieval]` | `0.35` | The cosine floor for embedding and long-term cosine hits, from 0 to 1. Inert for the task-local `bm25` and `llm` ranking and for a long-term row that only BM25 scored. |
| `cross_task_ratio` | `[retrieval]` | `1.1` | Cross-task episodic digests need this multiple of the best same-task digest's score from the same long-store backend; 1–10. |
| `cross_task_cap` | `[retrieval]` | `1` | Maximum cross-task digests per recall, 0–2; the existing total long-hit limit remains two. |
| `cross_task_cold` | `[retrieval]` | `0` | Without a matching same-task digest, permit 0 or 1 cross-task digest; still subject to the cap and backend floor. General memories bypass this gate. |
| `admit_on_outcome` | `[memory]` | `false` | R6-lite: admit only the final attempt digest at the run's end, with the terminal status in its provenance. The `--admit-on-outcome` flag wins for one run; direct mode applies the table, the interactive UI and the gateway do not read it, and it is accepted and inert in `legacy`. |
| `api_key_env` | cloud backend | not set | The name of the environment variable that holds the API key. |
| `base_url` | cloud backend | the vendor's URL | The API's URL. |
| `host` | `[gateway]` | `"127.0.0.1"` | Loopback listen address; non-loopback binds are refused. |
| `port` | `[gateway]` | `8401` | The port the gateway listens on. |
| `workspace` | `[gateway]` | `""` | The default workspace folder for web console sessions (`~/localmelo-workspace` when empty); written only when set. |
| `stream` | `[cli]` | `false` | Opt-in live streaming in the interactive UI only; runs without that UI never stream. `/new` puts `/stream` back to this value. An invalid value warns and falls back in the UI; direct runs and the gateway never fail on it. See [Streamed replies](../guides/terminal.md#streamed-replies). |
| `reasoning_tags` | `[cli]` | not set | A `MODEL = "explicit" \| "closing_only" \| "none"` table: the trusted framing declaration for a model's replies, which also splits the completed reply when `stream = false`. A model with no entry keeps the conservative rule (content is held until the reply ends). Set by hand: no onboarding prompt asks for it yet. |
| `mode` | `[ui]` | `"default"` | `"default"`: `melo --help` shows the user options. `"developer"`: it shows every option, grouped as user, advanced and experimental; `melo --developer --help` does the same for one run. The mode changes only what is shown (the help text, and the web console's developer views, which its settings toggle saves here), never what an option does. Any other value warns on stderr and shows the default help; runs and the gateway never fail on it. See [Default and developer mode](cli.md#default-and-developer-mode). |
| `resume_mode` | `[sessions]` | `"pairs"` | Terminal conversation restoration: `pairs` loads recent user/answer pairs; `window` loads the saved working window plus later pairs, falling back when no checkpoint is usable. Invalid values warn/fall back only in the UI; direct runs do not consume this setting. |
| `policy` | `[scheduler]` | `"off"` | How the web workspace's calls reach the model server: `off` records only, `fcfs` admits first come and `class` admits P1 before P2 (and so on); see [Scheduler](../guides/scheduler.md#the-gate-on-chats-s2). |
| `n_srv` | `[scheduler]` | `8` | The model server's slots. |
| `r_f` | `[scheduler]` | `1` | Slots kept for the CLI and direct runs; forced to 0 when `n_srv <= 2`. |
| `n_fg` | `[scheduler]` | `1` | Background calls allowed in flight while a chat is focused. |
| `call_log` | `[scheduler]` | `true` | Write one JSONL call log per gateway run. |
| `call_log_dir` | `[scheduler]` | `""` | The log's parent directory; the default is `~/.cache/localmelo/scheduler`. |
| `gpu_lock_file` | `[scheduler]` | `""` | The shared GPU lock to read; empty means no lock reader. The battery reader is always on in the gateway. |
| `lock_self` | `[scheduler]` | `""` | This bench's own holder: the exact `holder:` value, or `holder:run` when the lock carries a `run:` line. Matching is exact and case-sensitive. |
| `lock_gate` | `[scheduler]` | `"record"` | `record` never changes a request; `wait` holds every workspace call, embeddings included, while a foreign lock is present. See [Scheduler](../guides/scheduler.md#the-gpu-lock-and-the-battery-s2b). |

A `[memory]` table that is all defaults is not written. Its one key,
`admit_on_outcome`, must be a bool; any other value is a configuration error
(exit 1). It is accepted and inert in `legacy`. The `--keep-decision-text`
switch is flag-only, because it writes potentially personal text to the
store and a saved default would be a privacy footgun.

A `[scheduler]` table that is all defaults is not written, and unknown
scheduler keys are dropped, so a config file without such a table keeps its
exact bytes; the wizard keeps a non-default policy.

Top-level keys must come before the first `[table]` line. How URLs are used:

- `mlc`, `vllm` and `sglang`: `/v1` is added to `chat_url` and
  `embedding_url` when it is missing.
- `ollama`: chat uses Ollama's own `/api/chat`, so `chat_url` is used without
  `/v1` (a trailing `/v1` is removed). `embedding_url` gets `/v1` added.
- Cloud backends: `base_url` is used as given.

### Reasoning

`legacy` is the default: an OpenAI-compatible backend sends no reasoning,
and Ollama native sends its own `thinking` field and inline copy as before.
`drop` never sends reasoning; `replay` sends it on the assistant turns in
scope, under one key, at most 40,000 characters per turn (a longer turn's
reasoning is skipped and counted). `compress` sends none and keeps each
reply's note and drafts in working memory; `compress+replay` does both. A
backend with `requires_reasoning_replay = true` runs `compress` as
`compress+replay`. Every explicit mode also stops reading `[RECALL]` and
removes an inline `<think>` block from the answer, and stops writing the
raw step summary to long-term memory. The compress modes recall notes,
drafts and earlier thinking by default; see
[Recall in the loop](../concepts/memory.md#recall-in-the-loop).

```toml
[agent]
reasoning = "replay"      # legacy, drop, replay, compress, compress+replay
replay_turns = 3          # optional; omit for every turn
note_writer = "auto"      # optional: auto or heuristic
```

`[retrieval]` sets recall's defaults for direct runs, the interactive UI and
console sessions; an explicit CLI flag wins over the matching key. The table
is accepted and inert in `legacy`. A value outside its range (an
unknown retriever or name, `k` outside 1-10, `min_score` outside 0-1) is a
configuration error, exit 1 with one line.

```toml
[retrieval]
retriever = "auto"        # auto, off, bm25, llm or embedding
recall_at = ["start", "attempt", "step"]
k = 3                     # 1-10
kinds = ["note", "draft"] # omit for all but thought
placement = "tool"        # tool or message
min_score = 0.35          # 0-1
cross_task_ratio = 1.1    # 1-10
cross_task_cap = 1        # 0-2
cross_task_cold = 0       # 0 or 1
```

`save` writes the table only when a value differs from its default, so an
existing config file is unchanged.

An explicit flag wins over `[agent]` in direct mode and the interactive UI.
A `--base-url` session or run uses flags, not these saved defaults. API v2
console sessions use the saved mode, defaulting to `legacy` when none is set;
they keep the gateway's tool policy. The legacy `/v1` API is unchanged and
does not inherit the reasoning/retrieval tables. `--retrieval-extra-body`
is flag-only, never a saved key, and affects `retrieve` requests only.

Each backend section can name the request key and say that the server
requires replay:

```toml
[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "MODEL"
reasoning_field = "auto"              # or reasoning_content, reasoning
requires_reasoning_replay = false     # true for a server that rejects a
                                      # request without the reasoning
```

`auto` follows the key the last reply's reasoning came from, so a backend
that emits either `reasoning_content` or `reasoning` works as it is; it
sends no key before the first reply. For OpenAI-compatible backends, the
field is used only by `replay` and `compress+replay`: `legacy`, `drop` and
`compress` ignore it, including through scheduler wrappers. Native Ollama
keeps its existing legacy wire behavior. `replay` uses the
current task's assistant turns in direct runs and the `melo ""` loop;
session builders select the retained conversation window instead. Session
resume uses the current process's mode: pairs contain no old thinking,
while a window checkpoint can replay its retained, redacted thinking.
The `[ollama]` backend fixes its field
to `thinking` (it may still write `"thinking"` or `"auto"`). With
`requires_reasoning_replay = true`, `--reasoning drop` is refused, every
assistant turn in the window is replayed (the task scope and
`replay_turns` do not apply), and the agent evicts whole working-memory
units instead of cutting reasoning.

### Tool environment

Shell and Python tools normally inherit only `HOME`, `LOGNAME`, `PATH`,
`SHELL`, `TERM`, `USER`, `LANG`, `TMPDIR` and every `LC_*` variable. To
explicitly expose additional variables from melo's process:

```toml
[agent]
tool_env = ["SSH_AUTH_SOCK", "HTTPS_PROXY"]
```

This is a list of **names**, not `NAME=value` assignments. Names must match
`[A-Za-z_][A-Za-z0-9_]*`; missing variables are skipped. Config-backed direct
runs, the interactive UI and console sessions resolve the present values at
startup. Only names are written to TOML; the option values are omitted from
run configuration records. A `--base-url` run or session does not read this
setting. Python callers can instead pass an explicit
`AgentOptions(tool_env={...})` mapping. This does not change MCP's separate
per-server `env` configuration.

Treat each name as a capability grant. `SSH_AUTH_SOCK` lets tool subprocesses
use your SSH agent to authenticate or sign, and a proxy variable can expose
its address and credentials. Tools run with your permissions and can print,
copy or transmit variables they receive; filtering the environment is not a
sandbox. Leave the list empty unless a tool needs the capability, and never
put secret values in TOML.

## Context accounting

Set `[agent] context_window`, `output_reserve` and `safety_margin` to positive integers; the window
must be greater than their sum. Direct and interactive flags override individual values; API v2
console sessions use the config. Legacy `/v1` is unchanged. When unset, no accounting is recorded.

For each planning or reflection attempt, `utf8-bytes-v2` estimates the assembled message and tool
schema representation at UTF-8 bytes divided by three, rounded up. Each image adds
`ceil(width / 28) * ceil(height / 28)` tokens when dimensions are known, capped at 16,384;
unknown dimensions use the cap. The last positive prompt-token count for the same model and request
kind calibrates the proxy upward; missing or zero usage resets that calibration. The proxy is an
estimate, not a bound: templates and vision models differ.

When the estimate exceeds the input limit, the agent fits a request copy: it previews eligible old
tool outputs, removes optional old thinking and then evicts old complete units, while keeping the
task, tool schemas, matching call/result pairs, images and failing evidence. It leaves history and
the retained conversation unchanged. If the pinned context cannot fit, the run ends with
`context_budget_exceeded` and a remediation message. A classified context-overflow response gets at
most one smaller request retry without re-running tools; other provider errors keep their existing
retry policy. Events and JSON results report the estimate, actual prompt count if available, and
pruning counts without storing prompt text. The output reserve is not a provider generation cap;
the independent character guard still applies.

`[agent] fixed_prefix = true` opts direct, interactive and API v2 console runs into
`fixed-prefix-v2`; `--fixed-prefix on|off` overrides it for a direct or interactive
run. It snapshots the local date and UTC offset at each run's start, including across
midnight, and places changing recall and reflections in labelled user-side data.
Recall goes before the task; the latest reflection goes last, in a final user message
after the window, so a new attempt changes only what follows the shared history.
A reflection request repeats the latest planning request (same system text, same
`tools`, same turns, then the turns since) and adds one final user message with the
reflection prompt; it offers the tools, and a reply that calls one is asked again as text.
An explicit end delimiter leaves the task and reflection request outside that data;
data lines starting with that delimiter are deterministically quoted.
`all` freezes sorted tool definitions for that run; `retrieve` keeps selecting tools
as before. The JSON result's `prefix_cache.schema_changes` counts changes to the
offered planning schemas within the run, not actual cache misses; an optional
`reported_cached_tokens` is the server-reported total for planning and reflection responses only. Providers
may not report caching, and byte-stable prefixes do not guarantee a cache hit.
The legacy `/v1` API and the default-off request bytes remain unchanged.

`[agent] window_eviction = "chunk"` or `"boundary"` (default `"slide"`; `--window-eviction`
overrides it) changes when working memory drops old turns so the retained history stays the same
for several steps and a server's prompt cache can reuse it. Experimental; `slide` leaves request
bodies and the JSON result as before. See [Memory](memory.md#working-memory-window).

Opt in to separate captured tool history with `[agent] captured_history = true` or
`--captured-history on`; `off` overrides the setting. The checked preview uses a 50,000-character
ceiling including truncation and image notices. History stores a distinct, pattern-redacted copy of
up to 200,000 characters per result (errors first), with exact omitted counts relative to the
redacted capture. Counts live in `capture_metadata`, not in-band markers in stored text.
`acquisition_omitted_chars` is null because the tool may already have limited its
output. Capture is not a complete file or unlimited stdout; redaction is pattern-based and can miss
secrets. The model's checked preview can still contain raw text; the `Message.tool_result`,
`tool.finished` event and trace use redacted copies. Existing history retention and permissions
apply. `Agent.run_tool()` does not write captured history and retains its existing result check.
The option is off by default.

## Embedding and long-term memory

Long-term memory needs an embedding backend; see
[Set up an embedding model](../guides/embedding.md). With
`embedding_backend = "none"`:

- nothing is recalled from long-term memory, and nothing is written to it;
- reflections are not promoted to long-term memory;
- the run loop, tools, working memory and history work as usual.

`GET /v1/health` on the gateway then shows `"embedding_backend": "none"`.
See [Memory](memory.md).

## Cloud API keys

- The config file stores the *name* of the environment variable
  (`api_key_env`), never the key.
- Export the variable in the shell that runs `melo`.
- If it is not set, `melo` stops before the run with exit code 1:

  ```text
    Configuration errors:
    - [openai] environment variable $OPENAI_API_KEY is not set. Export it before starting the gateway.
  ```

- `--base-url` sends no API key, so use a config file for cloud APIs.
- The macOS background service does not get your shell's variables; see
  [Run the gateway at login](../guides/gateway.md#run-the-gateway-at-login-macos).

## Extra request fields (extra_body)

Each backend table accepts an `extra_body` table. Its fields are added to
every chat request. Use it for server options such as `chat_template_kwargs`:

```toml
[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen/Qwen3-8B"

[vllm.extra_body.chat_template_kwargs]
enable_thinking = false
```

- The inline form works too:
  `extra_body = { chat_template_kwargs = { enable_thinking = false } }`.
- A flat `chat_template_kwargs = { … }` key in the table is read as part of
  `extra_body`; the wizard writes it back inside `extra_body`.
- localmelo sets `model`, `messages`, `tools`, `tool_choice` and `stream`
  itself. Setting one of them in `extra_body` is a configuration error:

  ```text
    - [vllm] extra_body must not set 'model'; localmelo manages that field
  ```

- For the `ollama` backend, the fields go into the body of Ollama's
  `/api/chat`. localmelo asks Ollama for the model's thinking by default;
  `extra_body = { think = false }` turns that off.
- `--base-url` does not read the config file, so `extra_body` does not apply
  there.

## MCP servers

`mcp_config = "~/mcp.json"` at the top of `config.toml` names the MCP servers JSON file the
interactive UI loads (the gateway will read the same key); `--mcp-config FILE` overrides it for
one run. Direct mode never reads it.

## Chat request timeout

A chat request waits up to 120 seconds for the server (300 seconds with the
`ollama` backend). To change this for every run, set `request_timeout_s` at
the top of the config file, before the first table:

```toml
chat_backend = "vllm"
embedding_backend = "none"
request_timeout_s = 600

[vllm]
chat_url = "http://127.0.0.1:8000/v1"
chat_model = "Qwen/Qwen3-8B"
```

- Below a `[table]` line, the key belongs to that table, and localmelo does
  not read it there.
- The value is a number of seconds greater than 0, such as `600` or `90.5`.
  Anything else stops `melo` with exit code 1, for example
  `request_timeout_s must be a finite number of seconds > 0, got '600'` for
  a quoted value.
- `--request-timeout SECONDS` overrides it for one direct run and does not
  change the file.
- It applies to chat requests only. Embedding requests always wait up to
  120 seconds.
- The timer restarts whenever the server sends data, so a server that sends
  keep-alive bytes while it generates never times out.
- A request that times out is retried twice, each time with the full
  timeout; see [Provider errors and retries](run-results.md#provider-errors-and-retries).
- To limit a whole run, use `--timeout`. See
  [Timeouts](../concepts/how-it-works.md#timeouts).

## Environment variables

| Variable | Effect |
|---|---|
| `LOCALMELO_PERSIST_MEMORY` | Any non-empty value, even `0`, saves history and long-term memory in SQLite files. |
| `LOCALMELO_MEMORY_DIR` | The directory for those files (default `~/.cache/localmelo/memory`). It is used exactly as given: `~` is not expanded. |
| `LOCALMELO_DETECT_URLS` | The URLs that the [first run](../guides/configure.md#first-run) asks for a model list, comma-separated, in place of ports 8000, 8400, 11434 and 30000 of `127.0.0.1`. An empty value turns the search off. |
| `LOCALMELO_THINKING_RETENTION_DAYS` | The days that `history.db` and the session store's `sessions.db` keep the model's reasoning (default 30): any number above 0, such as `7` or `0.5`. Any other value stops `melo` with exit code 2. See [Model calls](memory.md#model-calls). |
| The variable that `api_key_env` names | The API key of a cloud backend, such as `OPENAI_API_KEY`. |

See [Keep memory between runs](../guides/memory.md#keep-memory-between-runs).

## Files and directories

| Path | What it holds | Created |
|---|---|---|
| `~/.cache/localmelo/config.toml` | The configuration. | By the setup wizard. |
| `~/.cache/localmelo/memory/` | `history.db`, `procedures.db` (procedural memory, written at the end of every run), and `long_term.db` with an embedding backend. | When an agent starts with `LOCALMELO_PERSIST_MEMORY` set, with mode 0700. An existing one that is open to group or others is set to 0700. |
| The `--memory-dir` directory | The same files, for that run. | When the run starts, with mode 0700 when it is new. |
| `~/.cache/localmelo/playground/` | Legacy playground memory. | When the opt-in playground starts (security-1b in progress). |
| `~/.localmelo/logs/` | The background service's logs. | By `melo --daemon install`. |
| `~/Library/LaunchAgents/ai.localmelo.gateway.plist` | The background service's definition. | By `melo --daemon install`. |

## Upgrading from an older config

Older config files still load. localmelo reads:

- a single `backend` key, the oldest format;
- the backend name `mlc-llm`, now `mlc`;
- `[cloud_api]` and `[online]` tables, now one table per vendor;
- `chat_port` in `[mlc]`, now `chat_url`;
- a flat `chat_template_kwargs`, now inside `extra_body`.

Running the wizard again writes the file in the current format.
