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
| `chat_backend` | top level | not set | The chat backend; see [Backends](../guides/model-servers.md#backends). For oMLX, use `omlx`, which first-run setup writes for a server that names itself oMLX, or `vllm`: both load as `vllm` and are saved as written. Required. |
| `embedding_backend` | top level | not set | `mlc`, `ollama`, `vllm`, `sglang` or `none`; `omlx` is accepted as `vllm`. Required. |
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
| `fixed_prefix` | `[agent]` | `true` | `false` sends the requests from before the stable prefix became the default. See [Context accounting](#context-accounting). |
| `window_eviction` | `[agent]` | `"chunk"` | `chunk`, `boundary` or `slide`: when working memory drops old turns. See [Memory](memory.md#working-memory-window). |
| `brief_items` | `[agent]` | `false` | Experimental. `true` makes a `--brief` run name the brief's numbered items that no edit has touched yet; see [`--brief-items`](cli.md#brief-items). |
| `reflect_tool_guard` | `[agent]` | `true` | `false` sends reflection requests without the tools and the closing sentence. See [Context accounting](#context-accounting). |
| `confine_writes` | `[agent]` | `true` | On macOS, `--brief` worker runs and saved tasks start `shell_exec` and `python_exec` in a sandbox that refuses file writes outside the workspace; `false` turns it off (`-X agent.confine_writes=false` for one run). Other runs never confine. See [Writes stay in the workspace](cli.md#confined-writes). |
| `confine_direct_runs` | `[agent]` | `false` | Developer setting. `true` confines plain direct runs too, as `confine_writes` confines `--brief` runs: on macOS a task given on the command line and the `melo ""` loop start `shell_exec` and `python_exec` in a sandbox that refuses file writes outside the workspace (`--workspace`, else the current folder), and the result carries `confinement`. There is no flag; `-X agent.confine_direct_runs=true` sets it for one run. `--brief` runs follow `confine_writes`; the interactive UI, the console, the app and the gateway ignore it. See [Writes stay in the workspace](cli.md#confined-writes). |
| `answer_on_stop` | `[agent]` | `false` | Developer setting. `true` makes a direct run that ends `gate_stopped` or `budget_exhausted` with no answer ask the model once more, without tools, for the final answer from the evidence gathered (the working window and the latest reflection), in the format the task asks for, `unknown` allowed. The run keeps its status; `result` is the answer and the JSON record has `"answered_on_stop": true`. There is no flag; `-X agent.answer_on_stop=on` sets it for one run. The interactive UI, the console, the app and the gateway ignore it. See [Run results](run-results.md#statuses). `--reflection-gate patient` (`-X agent.reflection_gate=patient`) is the other developer setting of this kind: see [Reflection](../concepts/how-it-works.md#reflection). |
| `lean_output` | `[agent]` | `false` | Experimental. `true` makes a task given on the command line ask for fewer output tokens; there is no flag, and `-X agent.lean_output=true` sets it for one run. The interactive UI and the gateway ignore it. See [Context accounting](#context-accounting). |
| `tool_images` | `[agent]` | `"auto"` | Whether the pictures tools return reach the model: `on` sends the newest ones as `image_url` parts in each planning request, `off` names them only, and `auto` is `on` when the server says the model reads images (oMLX's `model_type` `vlm`, Ollama's `vision` capability). There is no flag; `-X agent.tool_images=on` sets it for one run. See [Pictures that tools return](../guides/tools-mcp.md#pictures-that-tools-return). |
| `tool_images_keep`, `tool_images_retire` | `[agent]` | `3`, `2` | With `tool_images` on: the newest `tool_images_keep` pictures keep their pixels, and the oldest `tool_images_retire` lose them together once more arrive, so a request carries 3 to 4 pictures by default. |
| `cache_layout` | `[agent]` | `"v3"` | `"v2"` sends the earlier requests, with a reflection rebuilt as the last message of every planning request; `"v3"`, with `fixed_prefix`, appends it once instead. See [Context accounting](#context-accounting). |
| `window_messages` | `[agent]` | `64` on a local server or a URL, `20` on a cloud vendor | The working window's message cap, 8 to 256; `20` sends the earlier cap. See [Context accounting](#context-accounting). |
| `retriever` | `[retrieval]` | `"auto"` | `auto`, `off`, `bm25`, `llm` or `embedding`: how recall ranks documents. See [Recall in the loop](../concepts/memory.md#recall-in-the-loop). |
| `recall_at` | `[retrieval]` | not set | A list of `start`, `attempt`, `step` and `tool`. Omit for `start`, `attempt` and `step`; `tool` is never on unless named. |
| `k` | `[retrieval]` | `3` | Task-local hits per recall, from 1 to 10. |
| `kinds` | `[retrieval]` | not set | A list of `note`, `draft`, `thought`, `digest`, `reflection` and `long`. Omit for all but `thought`. |
| `placement` | `[retrieval]` | `"tool"` | `tool` (append to the last tool result) or `message` (its own user message). `message` is refused in `replay` and `compress+replay`. |
| `min_score` | `[retrieval]` | `0.35` | The cosine floor for embedding and long-term cosine hits, from 0 to 1. Inert for the task-local `bm25` and `llm` ranking and for a long-term row that only BM25 scored. |
| `cross_task_ratio` | `[retrieval]` | `1.1` | Cross-task episodic digests need this multiple of the best same-task digest's score from the same long-store backend; 1–10. |
| `cross_task_cap` | `[retrieval]` | `1` | Maximum cross-task digests per recall, 0–2; the existing total long-hit limit remains two. |
| `cross_task_cold` | `[retrieval]` | `0` | Without a matching same-task digest, permit 0 or 1 cross-task digest; still subject to the cap and backend floor. General memories bypass this gate. |
| `across_chats` | `[memory]` | `true` | Memory across chats in the web console and the app: melo keeps a note when you ask it to remember something or judges what you tell it worth keeping, indexes every finished chat turn and task result, and brings back what matches each new message, with where it came from. `false` turns all of it off. Direct runs and workers never use it. See [Memory across chats](../concepts/memory.md#memory-across-chats). |
| `retriever` | `[memory]` | `"auto"` | How a console or app chat finds notes for a message: `"auto"` (words, and meaning too when an embedding model is set up), `"bm25"` (words only, no model request), `"embedding"` (the same as `auto` with a model; `melo serve` refuses to start without one) or `"llm"` (words pick up to 20 candidates and the chat model chooses among them within 4 s, falling back to words alone on a timeout, an error or an unreadable reply; no embedding calls). A developer setting for comparing the ways: `-X memory.retriever=llm` for one run, listed by `melo -h all`. Every recall's timings are in the `recall` object of `turn.memory`. See [How recall finds notes](../concepts/memory.md#memory-across-chats). |
| `admit_on_outcome` | `[memory]` | `false` | R6-lite: admit only the final attempt digest at the run's end, with the terminal status in its provenance. The `--admit-on-outcome` flag wins for one run; direct mode applies the table, the interactive UI and the gateway do not read it, and it is accepted and inert in `legacy`. |
| `api_key_env` | cloud backend | not set | The name of the environment variable that holds the API key. |
| `base_url` | cloud backend | the vendor's URL | The API's URL. |
| `host` | `[gateway]` | `"127.0.0.1"` | Loopback listen address; non-loopback binds are refused. |
| `port` | `[gateway]` | `8401` | The port the gateway listens on. |
| `workspace` | `[gateway]` | `""` | The default workspace folder for web console sessions (`~/localmelo-workspace` when empty); written only when set. |
| `stream` | `[cli]` | `true` | Live streaming in the interactive UI only, on unless turned off (`/stream off`, `-X cli.stream=false`, `melo config set cli.stream false`); runs without that UI never stream. `/new` puts `/stream` back to this value. An invalid value warns and falls back in the UI; direct runs and the gateway never fail on it. See [Streamed replies](../guides/terminal.md#streamed-replies). |
| `status_line` | `[cli]` | `"auto"` | `"auto"`: the interactive status bar shows the usage zone (tokens or cost, `ctx`, the cache and idle time or the rate limit) from events already received, and the gateway's tasks when it has some. `"off"`: only the run zone (the title, or a running turn's steps) and `ctx` of the last request. Advanced: no command-line flag. Any other value warns and falls back in the UI; direct runs and the gateway never fail on it. See [The status bar](../guides/terminal.md#the-status-bar). |
| `colors` | `[cli]` | `"auto"` | The interactive UI's colour depth: `"auto"` (from `COLORTERM` and `TERM`), `"16"`, `"256"` or `"truecolor"`. `NO_COLOR` still turns colour off. Developer setting: `-X cli.colors=256` for one run. Any other value warns and falls back. See [Terminal colours](../guides/terminal.md#terminal-colours). |
| `background` | `[cli]` | `"auto"` | The terminal's background for the interactive UI: `"auto"` (asks the terminal, then `COLORFGBG`), `"dark"` or `"light"` for a terminal that answers nothing (tmux, screen, ssh). Developer setting: `-X cli.background=light`. See [Terminal colours](../guides/terminal.md#terminal-colours). |
| `reasoning_tags` | `[cli]` | not set | A `MODEL = "explicit" \| "closing_only" \| "none"` table: the trusted framing declaration for a model's replies, which also splits the completed reply when `stream = false`. A model with no entry keeps the conservative rule (content is held until the reply ends). Set by hand: no onboarding prompt asks for it yet. |
| `mode` | `[ui]` | `"default"` | `"default"`: `melo --help` shows the user options. `"developer"`: it shows every option, grouped as user, advanced and experimental; `melo --developer --help` does the same for one run. The mode changes only what is shown (the help text, and the web console's developer views, which its settings toggle saves here), never what an option does. Any other value warns on stderr and shows the default help; runs and the gateway never fail on it. See [Default and developer mode](cli.md#default-and-developer-mode). |
| `resume_mode` | `[sessions]` | `"pairs"` | Terminal conversation restoration: `pairs` loads recent user/answer pairs; `window` loads the saved working window plus later pairs, falling back when no checkpoint is usable. Invalid values warn/fall back only in the UI; direct runs do not consume this setting. |
| `record_direct` | `[sessions]` | `true` | Record every direct run (`melo "<task>"`, `melo --brief FILE`) as a worker session in `~/.cache/localmelo/cli/sessions.db`, which the web console's Workers view and the macOS app read. Only `false` turns it off; `--no-save-session` does the same for one run, and `--save-session` or `--parent-run` records anyway. Written only when `false`. See [Worker histories](cli.md#worker-histories). |
| `policy` | `[scheduler]` | `"off"` | How the web workspace's calls reach the model server: `off` records only, `fcfs` admits first come and `class` admits P1 before P2 (and so on); see [Scheduler](../guides/scheduler.md#the-gate-on-chats-s2). |
| `n_srv` | `[scheduler]` | `8` | The model server's slots. |
| `r_f` | `[scheduler]` | `1` | Slots kept for the CLI and direct runs; forced to 0 when `n_srv <= 2`. |
| `n_fg` | `[scheduler]` | `1` | Background calls allowed in flight while a chat is focused. |
| `call_log` | `[scheduler]` | `true` | Write one JSONL call log per gateway run. |
| `call_log_dir` | `[scheduler]` | `""` | The log's parent directory; the default is `~/.cache/localmelo/scheduler`. |
| `gpu_lock_file` | `[scheduler]` | `""` | The shared GPU lock to read; empty means no lock reader. The battery reader is always on in the gateway. |
| `lock_self` | `[scheduler]` | `""` | This bench's own holder: the exact `holder:` value, or `holder:run` when the lock carries a `run:` line. Matching is exact and case-sensitive. |
| `driver` | `[browser]` | `"off"` | Whether melo may read web pages in a browser: `off` gives it no browser tools; `safari-automation` gives the terminal chat and the web console eight read-only tools that open pages in Safari's automation windows (asks once per site per chat; see [Browsing with melo](../guides/browsing.md)). `/browser` switches it in the terminal; `-X browser.driver=safari-automation` for one run. Direct runs, worker runs and saved tasks never get the tools. Any other value is a configuration error. |
| `lock_gate` | `[scheduler]` | `"record"` | `record` never changes a request; `wait` holds every workspace call, embeddings included, while a foreign lock is present. See [Scheduler](../guides/scheduler.md#the-gpu-lock-and-the-battery-s2b). |

A `[memory]` table that is all defaults is not written. Its keys,
`across_chats` and `admit_on_outcome`, must be bools, and `retriever` one of
the four values above; any other value is a configuration error (exit 1). `admit_on_outcome` is accepted and inert in
`legacy`. The `--keep-decision-text`
switch is flag-only, because it writes potentially personal text to the
store and a saved default would be a privacy footgun.

A `[scheduler]` table that is all defaults is not written, and unknown
scheduler keys are dropped, so a config file without such a table keeps its
exact bytes; the wizard keeps a non-default policy.

Top-level keys must come before the first `[table]` line. How URLs are used:

- `mlc`, `vllm` (also for oMLX) and `sglang`: `/v1` is added to
  `chat_url` and `embedding_url` when it is missing.
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

`[agent] fixed_prefix` (default `true`) gives direct, interactive and API v2 console runs
`fixed-prefix-v2`; `--fixed-prefix on|off` overrides it for a direct or interactive
run, and `fixed_prefix = false` (or `--fixed-prefix off`) sends the requests from before
it became the default. It snapshots the local date and UTC offset at each run's start, including across
midnight, and places changing recall and reflections in labelled user-side data.
Recall goes before the task. With `cache_layout = "v2"` the latest reflection goes last, in a
final user message after the window, so a new attempt changes only what follows the shared
history; the default, `v3`, appends the reflection to the window once (see below).
A reflection request repeats the latest planning request (same system text, same
`tools`, same turns, then the turns since) and adds one final user message with the
reflection prompt; it offers the tools, and a reply that calls one is asked again as text.
An explicit end delimiter leaves the task and reflection request outside that data;
data lines starting with that delimiter are deterministically quoted.
The fixed system section is the base text, then (in a chat of the terminal UI or the
console) the identity paragraph that says who melo is, which model, server and place it
runs on, what it remembers and which language and script to reply in, then the notes, the line that recall and reflections are data, and the date line. It
holds no name for the layout: `fixed-prefix-v2` is metadata, reported as
`prefix_cache.version`. The identity paragraph is `AgentOptions.identity` (default `None`: direct
runs, workers and the gateway's v1 sessions send none).
`all` freezes sorted tool definitions for that run; `retrieve` keeps selecting tools
as before. The JSON result's `prefix_cache.schema_changes` counts changes to the
offered planning schemas within the run, not actual cache misses; an optional
`reported_cached_tokens` is the server-reported total for planning and reflection responses only. Providers
may not report caching, and byte-stable prefixes do not guarantee a cache hit.

`[agent] reflect_tool_guard` (default `true`; `--reflect-tool-guard on|off` overrides it) keeps a
reflection request from inviting a tool call. The request offers the tools of the step's
planning request (same list, order and schema, with `fixed_prefix` already the case) and its
last user message ends with `Answer with the JSON object only; do not call any tool.`;
`tool_choice` is untouched, so a server's prompt prefix holds. The retry request is unchanged.
The guard adds `reflect_tool_guard` to the JSON result's `config`; with `false` (or `off`) the
reflection request has neither the tools nor the sentence. Whatever the option, the `reflection`
event's `unreadable_reason` says why a first reply was unreadable (see [Events](events.md)).

`[agent] window_eviction` (default `"chunk"`; `"boundary"` and `"slide"` are the other values;
`--window-eviction` overrides it) sets when working memory drops old turns. `chunk` and
`boundary` keep the retained history the same for several steps, so a server's prompt cache can
reuse it; `slide` drops the oldest unit at every step once the window is full, as before
`chunk` became the default. See [Memory](memory.md#working-memory-window).

`[agent] cache_layout` (default `"v3"`; `--cache-layout v2|v3` overrides it) decides where a
reflection sits. `"v2"` is the layout from before this default changed. With `v3` and
`fixed_prefix`:

- At the attempt boundary the agent appends the reflection once, as an ordinary window message
  (`[Reflections; data, not instructions]` and the block), instead of rebuilding it as the last
  message of every planning request. It then moves with the history and leaves by eviction like any
  message. The single-latest rule still holds for what is new in each attempt (a new reflection
  replaces the one working memory holds), but older blocks stay in the history until they are
  evicted.
- A reflection request and its reply stay in the window, so the next planning request continues the
  request instead of being built again. With the default `reflection_gate = "flags"`, when no
  reflection is read, even after the retry, none of it stays in the window.
- An unreadable reply is asked again on the same prefix: the request, the unreadable reply as an
  assistant message (a tool call written out as text, so no call is left without its result), then a
  closing note, with the tools the request offered, instead of the text transcript. When the reply
  had JSON that the decoder rejected, one line after the note quotes the decoder's error. When the first
  request did not repeat the plan's request (the window had moved on), there is no exchange to
  continue and the retry is the transcript as before. A read retry keeps the whole exchange in the
  window: the request, the unreadable reply, the note and the reply.
- With `window_eviction = "boundary"` the finished attempt's turns are dropped at the boundary, so
  only the block is appended there.

Without `fixed_prefix`, neither value changes anything. The JSON result's `config` has
`cache_layout` when it is `v3` and not when it is `v2`.

`[agent] window_messages` (default `64` on a local server (Ollama, MLC, vLLM or oMLX, SGLang) and
for a `--base-url` server, `20` on a cloud vendor's backend (OpenAI, Gemini, Anthropic or NVIDIA);
8 to 256; `--window-messages N` overrides it) is the most messages the working window holds
before `window_eviction` drops some. A cloud vendor's backend keeps 20 for now: without prompt
caching, a longer window costs input tokens on every request. A value you set stays as you set
it, on any backend.
With a large cap, the prompt-size fit does the dropping: past the 100,000-character cap it drops
whole units (a tool call with its results, never split) down to 70% of the cap under `chunk` and
`boundary`, so the history holds for several steps; under `slide` it drops a unit at every step.
A count drop takes at most half the window at once, so a cap below 16 drops fewer than eight
messages. See [Memory](memory.md#working-memory-window).
The JSON result's `config` has `window_messages` when it is not 20.

`cache_layout = "v3"` and the 64-message window became defaults after two benchmark rounds on
coding tasks with a Qwen model on oMLX: 46 of 48 runs passed against 44 of 48, and 86.8% of the
prompt tokens were served from the cache against 78.9%. `--cache-layout v2 --window-messages 20`
send the earlier requests. `melo --developer --help` lists both flags under Advanced.

`fixed_prefix`, `window_eviction` and `reflect_tool_guard` became defaults after a same-build A/B
on the benchmark's coding tasks with a local model. A key you set in `[agent]` stays as you set
it, for these three and for the two above; a file without the key has the default, and saving
writes a key only when it differs from the default (`window_messages` is written whenever it is
set). `melo --developer --help` lists the three flags under Experimental, and
`--fixed-prefix off --window-eviction slide --reflect-tool-guard off` send the requests of the
earlier defaults. API v2 console sessions, console workers and the legacy `/v1` sessions read
these five keys from the config; the legacy `/v1` API does not read the other `[agent]` keys or
`[retrieval]`.

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

Opt in to fewer output tokens in direct runs with `[agent] lean_output = true` or
`-X agent.lean_output=true` (experimental; there is no flag). A task given on the command line
(`melo "<task>"` or `melo worker BRIEF`, with `json` or `stream-json` output too) then sends a
system prompt that, in place of `Think step by step. Be concise.`, asks for no text between tool
calls, a final answer of at most three lines, relative paths without a `sys.path` setup, and
reading the files a task names before listing folders. With the default
`reflection_gate = "flags"` the reflection prompt also leaves out the five estimates,
`progress_made` and `new_information_gained` (only `utility` reads the estimates, and nothing
reads the other two) and asks for `best_effort_result` only with `done` or `stop`; the action and
the three flags the gate reads are still asked for, and a reply in the long form is read as
before. `--reflection-gate utility` keeps the full reflection prompt, and the compress modes keep
the system prompt's last line, because their notes are text between tool calls. The interactive
UI, the `melo ""` loop and the gateway ignore the key. Off, the default, every request is the one
sent before the option existed; on, the start of the requests stays the same on every turn of a run.

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

## Pricing

The interactive UI's status bar shows a cloud API session's cost only from
prices you set; melo bundles none, since prices change. One table per model
id, in dollars per million tokens:

```toml
[pricing."your-model-id"]   # the chat_model, as /status names it; example numbers
input = 2.0        # prompt tokens
output = 8.0       # completion tokens
cache_read = 0.5   # prompt tokens the server reported cached; input when absent
```

`input` and `output` are required, every value is a number of at least 0, and
no other key is read. A model the session used without a table makes the bar
show tokens (`↑ 100k ↓ 5`) instead of a cost. A local server is never priced.
An invalid table warns and the bar shows tokens; nothing else reads it, and
saving the config keeps it as written. See
[The status bar](../guides/terminal.md#the-status-bar).

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

A `[browser]` table whose driver is `off` is not written.

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
- A server that loads a model when the first request for it arrives, as
  oMLX does, can need more than 120 seconds for that request; `600` above
  gives it room.
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
