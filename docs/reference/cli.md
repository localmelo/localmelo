# Command-line reference

This page describes the `melo` command: its modes, options, output, exit
codes and signals, and how to use it from scripts. `melo-retrieve`, the
other command installed with localmelo, compares memory retrievers offline;
see [Offline retrieval](retrieval.md).

## Synopsis

```text
melo [options] -- "<task>"                   # run one task (direct mode)
melo [options]                               # the interactive UI (in a terminal)
melo --base-url URL [--chat-model NAME]      # the interactive UI on this server
melo [--base-url URL] [--chat-model NAME] "" # the plain conversation loop
melo --reconfigure                           # run setup, then the interactive UI
melo --serve [--host HOST] [--port PORT]     # start the gateway
melo --serve --reconfigure                   # run setup, then start the gateway
melo --import-session dsh-jsonl FILE [--title TITLE] # import worker history
melo --daemon install|uninstall|status       # macOS background service
```

Quote the task and put it after `--` so option-like text is unambiguous.
Several words without quotes are joined with spaces. Existing task command
lines, abbreviations and the `melo ""` loop keep their previous parsing.

## Modes

| Command | What it does |
|---|---|
| `melo -- "<task>"` | Direct mode: runs one task, prints the result and exits. |
| `melo` (no task) | Opens the [interactive UI](../guides/terminal.md#the-interactive-ui) in a terminal, on the saved backend. With no saved backend it first asks for a model server and a model, and saves them; see [First run](../guides/terminal.md#first-run). Without a terminal it exits with code 2 and says what to run instead. |
| `melo --base-url URL` (no task) | Opens the interactive UI on that server, with `--chat-model` or the first chat model the server lists. |
| `melo --reconfigure` | Runs the setup wizard, then opens the interactive UI. Without a terminal it runs the wizard, reading the answers from stdin, and exits. |
| `melo --serve` | Starts the gateway. Runs the setup wizard first if there is no configuration (in a terminal; without one it exits with code 1). |
| `melo ""` (empty task) | Opens the plain conversation loop: each line you type is a task for the same agent, and only the answer prints. `/help` lists its commands; `/exit`, `/quit`, `exit`, `quit` or Ctrl-D exits. |
| `melo --daemon …` | Manages the macOS background service that runs the gateway; see [Run the gateway at login](../guides/gateway.md#run-the-gateway-at-login-macos). |

In the `melo ""` loop, turns share one agent and its working memory, which
keeps each answer, so a turn sees the earlier ones. A turn that does not
complete prints its status line on stderr, and the loop asks for the next task.
`/exit`, `/quit`, `exit` or `quit` (in any letter case), or Ctrl-D, ends the
loop with exit code 0; Ctrl+C ends it at once (a shell shows 130). At start
it prints one line saying that `shell_exec` is off in direct mode and that
`melo "" --builtins all` reopens the loop with it on (the line is absent when
the tool is already on). The loop takes `--base-url`, `--chat-model` and
`--builtins`; the other direct-mode options are usage errors with it. Its
conversation is separate from the web workspace's chats.
On a terminal the prompt is edited with readline: Backspace deletes a whole
character, Chinese and other wide characters included, and the Up arrow
recalls earlier lines of this loop. A line that is not valid UTF-8 prints
`melo: that line was not valid UTF-8; type it again.` and the loop asks
again.

## Worker histories

`--save-session [TITLE]` records a direct task as a `worker` session in the
CLI sessions file. Put the task after `--`; without a title, the first task
line supplies one. `--parent-run ID` requires `--save-session` and links a
launcher. Without the flag, direct runs do not open the session store. Workers
are not offered by `-c`, `-r`, or the chat picker. Workspace and git metadata
use `--workspace` when supplied, otherwise the current working directory.

`--import-session dsh-jsonl FILE [--title TITLE]` imports a dsh JSONL log
without a task. Reimporting identical file contents is a no-op; unknown or
invalid lines are counted. The gateway can list and read worker history via
its read-only [worker API](api-v2.md#get-apisessions), but cannot continue or edit it.

Worker records are local, unencrypted, and stored with file mode 0600. Stored
text passes the existing secret redactor, and recognizable reads of sensitive
files are replaced by a path-labelled placeholder. Redaction is pattern-based:
arbitrary secrets printed by a command, echoed by the model, or loaded through
an unrecognized tool can still be retained. Review the data before sharing it.
Thinking is subject to the session retention policy.

## Options

### General

| Option | Default | Description |
|---|---|---|
| `query` (positional) | not set | The task to run. Giving one selects direct mode; an empty one (`""`) opens the plain loop; none opens the interactive UI. |
| `-h`, `--help` | | Show the help and exit. |
| `--base-url URL` | not set | Send chat requests to this OpenAI-compatible server instead of the configured backend. |
| `--chat-model NAME` | with `--base-url`, `default` for a task and the server's first listed model without one; else the configured model | The model to request from the `--base-url` server, or instead of the configured chat model. |

- `--base-url` is used exactly as given, so include `/v1`, as in
  `http://127.0.0.1:11434/v1`.
- `--base-url` sends no API key, and it does not read the config file, so
  `extra_body` and `request_timeout_s` do not apply.
- Without `--base-url`, `--chat-model` replaces the configured chat model for
  this run; the config file is not changed.
- Without a task, either option opens the interactive UI; see
  [Which model server](../guides/terminal.md#which-model-server). With `--serve` or `--daemon`,
  either is a usage error: the gateway uses the config file.

### Direct mode

These options apply to a task run without `--serve`; the interactive UI also
accepts the options named in [Interactive UI](#interactive-ui).

| Option | Default | Description |
|---|---|---|
| `--workspace DIR` | the current folder | The folder the run works in: the file tools stay inside it and `shell_exec` and `python_exec` start there; see [With a workspace root](../guides/tools-mcp.md#with-a-workspace-root). |
| `--context-window N`, `--output-reserve N`, `--safety-margin N` | not set | Opt in to per-request context accounting; all three must resolve to positive integers and the window must exceed the other two combined. |
| `--max-steps N` | `30` | The most planning steps the whole run may take. |
| `--attempt-policy reflect\|none` | `reflect` | `reflect`: attempts with a reflection after each. `none`: one attempt, no reflection. |
| `--reflection-gate flags\|utility` | `flags` | After a reflection, start the next attempt when it chooses `continue` and its three checks hold (`flags`), or also require its utility score to reach 0.1 (`utility`, the behavior before this option). |
| `--tool-exposure retrieve\|all` | `retrieve` | Which tools each step offers the model. |
| `--builtins all\|none\|NAME,...` | the mode's set | The built-in tools to register. Without the flag, a direct run gets `file_read`, `file_write` and `python_exec`; `all` adds `shell_exec` (the interactive UI keeps all four by default). |
| `--mcp-config PATH` | not set | Add the tools of the MCP servers listed in this JSON file. |
| `--mcp-only` | off | Use only the MCP tools: the same as `--builtins none --tool-exposure all`. |
| `--memory on\|off` | `on` | `off` turns long-term memory off for this run. |
| `--memory-dir DIR` | not set | Save history, long-term and procedural memory as SQLite files in `DIR`: `history.db`, `long_term.db` and `procedures.db`. |
| `--timeout SECONDS` | not set | End the run with status `timeout` after this many seconds. |
| `--request-timeout SECONDS` | `120` | How long each chat request may wait for the server. |
| `--output-format text\|json` | `text` | Print the result as text, or the run record as one JSON line. |
| `--embedding-base-url URL` | not set | An OpenAI-compatible embedding server for long-term memory; see [Set up an embedding model](../guides/embedding.md). |
| `--embedding-model NAME` | `default` | The model to request from the `--embedding-base-url` server. |
| `--progress auto\|off` | `auto` | Draw a [progress line](#progress-line) on stderr while the task runs: `auto` when stderr is a terminal and the output is text, `off` never. |
| `--routine NAME` | not set | Learn and recall this task's steps as the routine `NAME`, in any folder and whatever the task's wording. |
| `--reasoning legacy\|drop\|replay\|compress\|compress+replay` | `legacy` | How a reply's reasoning is sent back. `legacy` keeps the provider's own behavior (an OpenAI-compatible server gets none, Ollama native gets its own field); `drop` never sends it; `replay` sends it on the assistant turns in scope, under one key; `compress` sends none and keeps each reply's note and drafts in working memory; `compress+replay` does both. |
| `--reasoning-field auto\|reasoning_content\|reasoning` | not set | The request key `--reasoning replay` uses. `auto` follows the key the last reply's reasoning came from; the others fix it. Needed with `--base-url`, or set `reasoning_field` in the chat backend's config section. |
| `--replay-turns N` | every turn in scope | With `replay` or `compress+replay`, keep the newest `N` assistant turns' reasoning; older ones are sent without it. It is inert in `legacy`, `drop` and `compress`. |
| `--note-writer auto\|heuristic` | `auto` | In the compress modes, who writes each reply's note: `auto` reads the model's `<note>` block and falls back to the heuristic, `heuristic` always writes it from the reply's text and reasoning. Accepted and inert in the other modes. |
| `--retriever auto\|off\|bm25\|llm\|embedding` | `auto` | How recall ranks documents. `auto` uses `embedding` when an embedding provider is configured, else `llm` at `start`/`attempt` and `bm25` at `step`/`tool`. `off` turns recall off. See [Recall in the loop](../concepts/memory.md#recall-in-the-loop). |
| `--recall-at LIST` | `start,attempt,step` | The recall triggers: a comma list of `start`, `attempt`, `step` and `tool`. `tool` registers the `recall` tool and is never on unless named. |
| `--recall-k N` | `3` | Task-local hits per recall, from 1 to 10 (plus at most 2 long-term hits). |
| `--recall-kinds LIST` | all but `thought` | A comma list of `note`, `draft`, `thought`, `digest`, `reflection` and `long`. |
| `--recall-placement tool\|message` | `tool` | Where a step block goes: appended to the last tool result, or in its own user message. `message` is refused in `replay` and `compress+replay`. |
| `--recall-min-score F` | `0.35` | The cosine floor for embedding and long-term cosine hits, from 0 to 1. It is inert for the task-local `bm25` and `llm` ranking and for a long-term row that only BM25 scored. |
| `--retrieval-extra-body JSON` | not set | Extra top-level fields for `retrieve` requests only (the `llm` retriever), as a JSON object. Available for a task or the interactive UI with `llm`/`auto` outside `legacy`; never saved to the config file. |
| `--admit-on-outcome on\|off` | `off` | R6-lite: at the run's end, admit only the final attempt digest, with the terminal status in its provenance; off writes nothing unless a signal admits something. Accepted and inert in `legacy`, and accepted in the interactive UI. |
| `--keep-decision-text` | off | R6-lite: also store the checked text of each admission decision; the SHA-256 is always kept. An evaluation switch and refused in the interactive UI. |

- `--workspace`: without it a task runs in the current folder, exactly as
  before. With it, `~` is expanded and the path is resolved like the
  interactive UI's, and the run works in that folder: the file tools stay
  inside it and `shell_exec` and `python_exec` start there. It adds no
  approvals (see [With a workspace root](../guides/tools-mcp.md#with-a-workspace-root)).
  A path that is not a folder is a usage error (exit 2, one line); see
  [Usage errors](#usage-errors). `melo ""` and `--serve` refuse the flag.
- The three context flags override the matching `[agent]` keys independently, in direct mode and the
  interactive UI. They measure, but do not truncate or reject, planning and reflection requests; the
  output reserve is not a generation cap. See [context accounting](configuration.md#context-accounting).
- `--max-steps`: at least 1. With `reflect`, a run has attempts of up to 10
  steps, at most 5 attempts, and at most `--max-steps` steps in total. See
  [How localmelo works](../concepts/how-it-works.md#the-run-loop).
- `--reflection-gate`: see [Reflection](../concepts/how-it-works.md#reflection). It has
  no effect with `--attempt-policy none`.
- `--tool-exposure`: see [How tools are offered](tools.md#how-tools-are-offered).
- `--builtins`: `all`, `none`, or a comma-separated list of `file_read`,
  `file_write`, `python_exec` and `shell_exec`. A direct run's default set
  leaves `shell_exec` out: `--builtins all` (or naming `shell_exec`) turns
  it back on, and it then runs without an approval, because direct mode has
  no policy. `recall` is accepted only with `--recall-at tool`, which
  registers it; without that trigger it is a usage error.
- `--mcp-only` needs `--mcp-config` and cannot be combined with `--builtins`
  or `--tool-exposure`. See [Tools and MCP](../guides/tools-mcp.md#add-tools-from-mcp-servers).
- `--memory-dir` turns persistence on for this run, whatever the environment
  variables say. `~` is expanded. See
  [Memory](../guides/memory.md#keep-memory-between-runs).
- `--request-timeout`: the default is `request_timeout_s` from the config
  file, else 120 seconds (300 seconds for the `ollama` backend). With
  `--base-url` it is 120 seconds, also for an Ollama `/v1` URL. It does not
  change the config file. See
  [Chat request timeout](configuration.md#chat-request-timeout).
- `--embedding-base-url` is also used exactly as given: include `/v1`, as in
  `http://127.0.0.1:11434/v1`.
- `--embedding-base-url` needs `--base-url`, and `--embedding-model` needs
  `--embedding-base-url`. Without `--base-url`, the configured embedding
  backend is used.
- `--progress`: see [Progress line](#progress-line). `off` starts the run
  exactly as a task run without a terminal does.
- `--routine`: without it, a task is known by its wording (dates and times
  masked) and the folder it runs in. Runs given the same `NAME` are one
  routine, so `--routine daily-mail` joins differently worded requests run
  from any folder. A blank name is a usage error. See
  [Procedural memory](../concepts/procedures.md#how-it-works).
- `--reasoning`: `legacy` is the default and leaves the wire unchanged.
  `drop`, `replay`, `compress` and `compress+replay` do not read `[RECALL]`
  and remove an inline `<think>` block from an explicit-mode reply's text,
  so the result is the clean answer. `replay` sends the reasoning on
  assistant turns in scope, at most `40,000` characters per turn (a longer
  turn's reasoning is skipped and counted); a backend whose
  `requires_reasoning_replay` is true never has its reasoning cut, and
  `drop` is refused there. `legacy` is never refused, even on such a
  backend, where it can still draw an HTTP 400. `compress` sends no
  reasoning and asks the model for a short note before each tool call; the
  notes and drafts stay in working memory (see
  [Reasoning notes](../concepts/memory.md#reasoning-notes)). `compress+replay` is both.
  A backend whose `requires_reasoning_replay` is true runs `compress` as
  `compress+replay`, and the JSON line reports the effective mode.
  `--replay-turns` counts assistant turns that carry reasoning; tool results
  do not consume a turn. The default task scope is the current task's
  turns only: earlier turns of a `melo ""` loop or of a session are not
  replayed. On a backend whose `requires_reasoning_replay` is true, every
  assistant turn in the window is replayed, and neither the task scope nor
  `--replay-turns` applies. `--base-url` runs use these flags and do not
  read the config file.
- `--retriever`, `--recall-*` and `--retrieval-extra-body`: see
  [Recall in the loop](../concepts/memory.md#recall-in-the-loop). `--retriever`,
  `--recall-*` and `--retrieval-extra-body` are accepted in the interactive
  UI. They have no effect in `legacy`: every one of
  them is a usage error there (exit 2, one line), as are `--retriever
  embedding` without an embedding provider or with `--memory off`,
  `--recall-placement message` in `replay`/`compress+replay`, `--recall-at
  tool` with `--mcp-only` or a `--builtins` list without `recall`,
  `--builtins recall` without `--recall-at tool`, and
  `--retrieval-extra-body` when the resolved retriever is `off`, `bm25` or
  `embedding` (including a saved default). With `auto`, the extra body is
  used only if a recall actually makes an `llm` retrieve request. A provided
  tool named `recall` while the tool is registered is
  a setup error. The arms differ only in these flags:

  ```bash
  melo --base-url http://127.0.0.1:11434/v1 --chat-model MODEL \
    --reasoning compress --retriever bm25 -- "List the files"
  melo --base-url http://127.0.0.1:11434/v1 --chat-model MODEL \
    --reasoning compress --retriever off -- "List the files"
  ```
- `--admit-on-outcome` and `--keep-decision-text`: see
  [Long-term memory](memory.md#long-term-memory). They are accepted in
  `legacy` and do nothing there. `--keep-decision-text` writes potentially
  personal text to the store, so the interactive UI refuses it (exit 2, one
  line). `[memory] admit_on_outcome` is the saved default for the first
  in direct mode (the interactive UI and the gateway do not read the table);
  the second is flag-only.
- `--note-writer`: `auto` (default) parses the model's
  `<note>plan: … found: … open: … remember: …</note>` block and falls back
  to a heuristic when there is none; `heuristic` ignores the block. A note
  is at most about 1,200 characters, and `remember: yes` marks it a
  candidate for later memory. The flag is accepted in every mode and only
  the compress modes use it. In `config.toml` it is the `[agent]`
  `note_writer` key, and `[agent] reasoning` takes the same five values.
  In `compress` and `compress+replay` the default `--retriever auto` also
  recalls notes and drafts at `start`, `attempt` and `step`; see
  [Recall in the loop](../concepts/memory.md#recall-in-the-loop).
- `--reasoning-field`: `auto` resolves to the key the last reply's
  reasoning came from, and sends no key before the first such reply. The
  flag, and a backend's `reasoning_field`, are used only in `replay` and
  `compress+replay`; in `legacy`, `drop` and `compress` the wire is as if
  no field were set. The `ollama`
  backend fixes its field to `thinking`, so the flag is a usage error
  there. The same rule applies in the interactive UI, including after a
  backend switch.

### Interactive UI

These options apply to `melo` without a task.

| Option | Default | Description |
|---|---|---|
| `-c`, `--continue` | off | Resume the newest eligible chat in this workspace; with none, start new. |
| `-r`, `--resume [ID\|TITLE]` | off | Resume the named chat; without a value open the session picker. |
| `--yolo` | off | Run every tool without asking for approval. |

- The UI also takes `--context-window`, `--output-reserve`, `--safety-margin`,
  `--workspace`, `--max-steps`, `--attempt-policy`,
  `--reflection-gate`, `--builtins`, `--memory`, `--memory-dir`,
  `--request-timeout`, `--reasoning`, `--reasoning-field`, `--replay-turns`,
  `--note-writer`, `--retriever`, `--recall-*`, `--retrieval-extra-body` and
  the `--embedding-*` options, for every turn of the session. Explicit flags
  win over the saved `[agent]` and `[retrieval]` defaults; retrieval settings
  stay inert in `legacy`, and explicit retrieval flags there are errors.
  A `--base-url` session takes these values from flags, not saved defaults.
  `--tool-exposure` (the UI offers every tool), `--timeout`,
  `--output-format`, `--progress`, `--routine` and `--keep-decision-text`
  are usage errors; `--mcp-config` and `--mcp-only` are accepted.
- For example, `melo --reasoning compress --retriever bm25 --note-writer heuristic`
  starts a session with short notes and recall. There is no `/reasoning`
  command: choose the mode when starting the process. `/model` and `/backend`
  keep the mode, full conversation and RAM-only notes/drafts. A provider that requires replay
  refuses a `drop` session without changing its model, conversation or MCP
  state; `compress` on such a provider runs as `compress+replay`.
- Session replay covers the retained conversation window, not only the
  current task. On resume, `pairs` restores answers without thinking;
  `window` can replay the checkpoint's retained, redacted thinking. The
  current process's mode wins over the mode recorded on old turns. Neither
  path reconstructs the RAM-only note store. See [Resume](../guides/sessions.md#resume).
- `--workspace` is expanded and resolved here too, and a path that is not a
  folder gives the same one-line usage error as with a task; see
  [Usage errors](#usage-errors).

### Gateway and setup

| Option | Default | Description |
|---|---|---|
| `--serve` | off | Start the gateway: the HTTP API and the web workspace. |
| `--host HOST` | `127.0.0.1` | The address the gateway listens on. |
| `--port PORT` | `8401` | The port the gateway listens on. |
| `--allow-unconfigured` | off | With `--serve`, start the gateway even when no backend is configured or the config file has validation errors, so it can be configured over `/api/settings`. |
| `--reconfigure` | off | Run the setup wizard, then start the gateway (with `--serve`) or open the interactive UI (without). Without a terminal it reads the answers from stdin; without `--serve` it then exits. |
| `--daemon install\|uninstall\|status` | not set | Manage the macOS background service; `install` also takes `--port`. |

- The defaults of `--host` and `--port` come from the `[gateway]` table of
  the config file. The options change them for this run only.
- `--host`, `--port` and `--reconfigure` are ignored when you give a task.
- Without a task, `--host` and `--port` are usage errors; give `--serve`
  to start the gateway.
- Without a terminal and with no config file, `melo --serve` prints one
  line on stderr and exits with code 1, as under launchd:
  `melo: no backend is configured; run melo --reconfigure in a terminal`.
  `--reconfigure` still runs the wizard without a terminal and reads its
  answers from stdin, so a script can pipe them in.
  `--allow-unconfigured` skips both the wizard and startup validation so
  `melo --serve --allow-unconfigured` starts without a config file; it
  cannot be combined with `--reconfigure` or used without `--serve`.
- Ctrl-D at a wizard question prints `Setup cancelled.` and exits with
  code 0; Ctrl+C prints the same and exits with 130.
- The gateway has no authentication. Keep it on `127.0.0.1`; see
  [Keep the gateway private](../guides/gateway.md#keep-the-gateway-private).

## Usage errors

`melo` checks the command line before it runs anything. A problem prints
`melo: error: …` on stderr, usually after a usage summary, and exits with
code 2:

```text
melo: error: argument --max-steps: must be at least 1, got 0
```

These are usage errors:

- a value that is not valid, such as `--max-steps 0`, `--timeout 0`, an
  unknown built-in tool name, a blank `--routine`, or a `--memory-dir` that
  is a file;
- a `--workspace` that is not an existing folder, in either mode, reported
  with one line and no usage summary:
  `melo: --workspace: not a directory: DIR`;
- options that cannot be combined: `--mcp-only` without `--mcp-config`,
  `--mcp-only` with `--builtins` or `--tool-exposure`, `--embedding-model`
  without `--embedding-base-url`, `--embedding-base-url` without
  `--base-url`;
- a direct-mode option, `--workspace` included, with `--serve`, `--daemon`
  or with `melo ""` (except `--builtins`, which the loop takes for its
  turns), reported as
  `only for direct mode (a query without --serve): --max-steps`; without a
  task, only those the interactive UI cannot use (`--tool-exposure`,
  `--timeout`, `--output-format`, `--progress`, `--routine`);
- `-c` with `-r`, reported as `-c and -r cannot be combined`;
- `--yolo`, `-c` or `-r` with a task, `--serve` or `--daemon`, reported as
  `only for the interactive UI (melo without a query, --serve or --daemon): --yolo`;
- `--base-url` or `--chat-model` with `--serve` or `--daemon`, reported as
  `--base-url and --chat-model are for a task or the interactive UI; --serve uses the config file (melo --reconfigure)`;
- `--host` or `--port` without a task and without `--serve` or `--daemon`,
  reported as
  `--host and --port are for --serve and --daemon; bare melo opens the interactive UI`;
- `--reconfigure` with `--base-url` and no task, reported as
  `--reconfigure sets up the config file, which --base-url does not use; give one of them`;
- a recall flag in `legacy`, reported as
  `argument --retriever/--recall-*/--retrieval-*: retrieval is not available in legacy; use --reasoning compress, compress+replay, drop or replay`;
- `--retriever embedding` without an embedding provider or with
  `--memory off`;
- `--recall-placement message` with `--reasoning replay` or
  `compress+replay`;
- `--recall-at tool` with `--mcp-only`, without `recall` in `--builtins`,
  or with `--retriever off`;
- `--retrieval-extra-body` with `--retriever off`, `bm25` or `embedding`;
- `--reasoning replay` or `--reasoning compress+replay` with `--base-url`
  and no `--reasoning-field`, reported as
  `argument --reasoning: replay and compress+replay need a reasoning field: give --reasoning-field (auto, reasoning_content or reasoning) with --base-url, or set reasoning_field in the backend's config`;
- `--reasoning replay` or `--reasoning compress+replay` on a configured
  backend whose `reasoning_field` is unset, reported as
  `argument --reasoning: replay and compress+replay need a reasoning field: set reasoning_field in the chat backend's config section`;
- `--reasoning drop` on a backend with `requires_reasoning_replay = true`,
  reported as
  `argument --reasoning: this backend requires reasoning replay, so drop is not possible: use replay or compress+replay`;
- `--reasoning-field` with the `ollama` backend, reported as
  `argument --reasoning-field: the ollama backend always reads and sends the thinking field`;
- an MCP config file that cannot be read or is not valid;
- an invalid `LOCALMELO_THINKING_RETENTION_DAYS`, also with `--serve` or
  `melo ""`, which prints its one `melo: error:` line without the usage
  summary:
  `melo: error: LOCALMELO_THINKING_RETENTION_DAYS must be a number of days greater than 0, not 'abc'`.

## Output

**Text** (the default):

- The result goes to stdout.
- When the status is not `completed`, one line goes to stderr:

  ```text
  melo: run ended with status budget_exhausted: Max steps reached
  ```

  The error is joined into one line and cut to 500 characters. An empty
  error reads `no error message`.

**JSON** (`--output-format json`):

- stdout gets exactly one line: the run record, with the
  [fields of a run result](run-results.md#fields) in order, plus
  `localmelo_version`. There is no status line.
- `localmelo_version` is the output of `git describe --tags --always --dirty`
  when localmelo runs from a git checkout (for example `603f40d`), otherwise
  the installed package version (for example `0.1.0`), or `unknown`.
- Outside `--reasoning legacy`, the record also has `config.reasoning`,
  `config.replay_turns`, `config.note_writer`, `usage.reasoning_chars` (and
  `usage.reasoning_tokens` or `usage.cached_tokens` when the server reports
  them) and `memory_stats`, with the note counters zeroed in the modes that
  do not use them; see [Run results](run-results.md#fields). A `legacy`
  record has none of them.

**In both formats:**

- stdout carries only the report. Anything a tool writes to stdout by itself,
  such as a program that `python_exec` code starts without capturing its
  output, goes to stderr instead.
- Warnings go to stderr, one line each.
- A traceback appears only when the run ends as `crashed`, after Ctrl+C, or
  when the config file is not valid TOML.

## Progress line

With text output on a terminal, `melo -- "<task>"` shows where the task stands
in one line on stderr. The line is redrawn on each step and tool call and
every second, and erased before the result and the status line are
printed, so it never remains in the scrollback:

```text
step 4 · Verify (cycle 1) · pytest 0:41 · tests 3/8 · 1m03s
```

- **When.** With `--progress auto`, the default, only when stderr is a
  terminal, `TERM` is not `dumb` and the output is text. Nothing is drawn
  with `--output-format json` (on a terminal or not), when stderr is a pipe
  or a file (as in scripts and benchmarks), with `--progress off`, or in the
  `melo ""` loop; the run then starts exactly as it would without the
  option. Stdout is the same either way. While `melo` runs in the
  background (`melo "<task>" &`, or Ctrl+Z then `bg`), the line is not
  drawn, so it never covers the shell's prompt; it comes back with `fg`.
- **First runs: facts.** `step 4` counts planning steps; `attempt 2` appears
  from the second attempt on. The phase is read from the tool calls so far:
  `Explore` (before the first edit), `Write` (editing), `Verify` (running
  tests) or `Finishing` (after a test run passed), with `(cycle n)` from
  the first test run after an edit, n counting such edit-then-test rounds.
  Then the tool call under way and its time (`pytest 0:41`), the counts of
  the last test summary seen (`tests 3/8`: passed of total), and the run's
  time (`1m03s`).
- **Repeated tasks: past runs too.** Once a task's
  [routine](../concepts/procedures.md#how-it-works) has completed before on this model
  with the same options (or, as a hint, a similar task in the same folder
  has), the line adds what those runs did, never a percent:

  ```text
  step 4 · Write (cycle 1) · 1m03s · last time 6 steps, 2m10s: Read test_parser.py ✓ · Edit parser.py + pytest (×2) ◐ · git commit
  step 4 · Write (cycle 1) · 1m03s · past 5-8 steps, 2-3 min · last time: Read test_parser.py ✓ · Edit parser.py ◐ · git commit
  ```

  The steps are those of past runs: `✓` done, `◐` the current one, `–`
  skipped, no mark for one not reached yet (this run need not do them all);
  `(×2)` marks a pattern that repeated. A percent bar is not shown in this
  release. The step count is never shown against `--max-steps`, which is a
  limit, not a plan.

  A similar task's runs are a hint, so the line never says `last time` for
  them; they follow `like <routine title>:`, naming the task they are from
  (the routine's name, else the start of its query), and its steps
  `its steps:`:

  ```text
  step 4 · Write (cycle 1) · 1m03s · like Fix the parser tests: 6 steps, 2m10s: Read test_parser.py ✓ · Edit parser.py ◐ · git commit
  ```
- **Past runs need persistence.** Each `melo` is a new process, and
  procedural memory stays in RAM unless memory is kept between runs. Run
  with `LOCALMELO_PERSIST_MEMORY=1` or `--memory-dir DIR` (see
  [Memory](../guides/memory.md#keep-memory-between-runs)); otherwise every run shows
  facts only.
- **Narrow terminals.** When the line is wider than the terminal, it leaves
  out, in this order, the past runs' times, the test counts, the past
  runs' named steps and the attempt, and then cuts the rest with `…`. In a
  Chinese, Japanese or Korean locale (the first of `LC_ALL`, `LC_CTYPE`
  and `LANG` that is set starts with `zh`, `ja` or `ko`), marks such as
  `·` and `…` count two cells, as a terminal set to draw ambiguous-width
  characters wide draws them, so the line never wraps there.
- **Color.** The line is dim; set `NO_COLOR` to a non-empty value to draw
  it plain.
- **Warnings** print on a line of their own above it. Output that a tool
  writes straight to the terminal, such as a program that `python_exec`
  code starts without capturing its output, can overwrite the line until
  its next redraw, within a second.
- **Signals.** SIGTERM, SIGHUP and Ctrl+C erase the line before `melo`
  stops.
- **Paused output.** While the terminal's output is stopped (Ctrl+S), the
  task goes on without the line and a signal still stops `melo`; the last
  line drawn then stays on the screen.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | The run ended with status `completed`. |
| 1 | The run ended as `crashed`; or `melo` stopped before the run because no backend is configured or the config file is not valid; or the setup wizard needed a terminal and had none; or the interactive UI could not open its session file or start its agent. |
| 2 | The run ended with any other status; or a usage error; or the interactive UI had no terminal; or `--workspace` is not a folder, in either mode. |
| 3 | The gateway could not start, for example because its port is in use. |
| 129 | SIGHUP ended the run. |
| 130 | Ctrl+C (SIGINT) ended the run. |
| 143 | SIGTERM ended the run. |

129, 130 and 143 are what a shell reports for a process ended by that
signal. Cancelling the setup wizard, or `q` at the first-run questions,
exits with 0. Before the interactive UI opens, Ctrl-D at a setup question
exits with 0 and Ctrl+C with 130, both after `Setup cancelled.` Leaving the
UI with `/exit`, Ctrl+D or Ctrl+C twice exits with 0, whatever its turns
ended with.

## Signals

SIGTERM and SIGHUP end a direct run cleanly, within 8 seconds:

1. A running `shell_exec` command or `python_exec` child is killed together
   with its process group.
2. MCP servers are closed. A server still running 6 seconds after the signal
   is killed.
3. The task record is saved with status `crashed` and the error
   `Cancelled by SIGTERM` (or `Cancelled by SIGHUP`). Reflections are not
   promoted to long-term memory.
4. `melo` exits by the signal (a shell shows 143 or 129) and prints nothing on
   stdout: no result, status line or JSON line.

One exception:

- A signal that was ignored when `melo` started stays ignored. For example,
  `nohup` ignores SIGHUP.

Ctrl+C (SIGINT) stops a run at once. `melo` prints a traceback, saves the task
record as `crashed` with the error `CancelledError`, and a shell shows exit
code 130.

In the interactive UI, SIGTERM and SIGHUP (also a terminal window that
closes) stop a running turn, which is recorded as `cancelled` with the
error `Cancelled by SIGTERM`, close the tools within the same 8 seconds,
and end `melo` by the signal. Ctrl+C there is a key (see [Keys](../guides/terminal.md#keys)).

## Use `melo` in scripts

Read the JSON line for details, and the exit code for the outcome:

```bash
line=$(melo --output-format json --timeout 600 -- "List the files in this directory")
code=$?
if [ -n "$line" ]; then
  status=$(printf '%s\n' "$line" | python3 -c 'import json, sys; print(json.load(sys.stdin)["status"])')
  echo "status: $status, exit code: $code"
else
  echo "no run record, exit code: $code" >&2
fi
```

- Exit code 2 covers several statuses. Read `status` to tell them apart; see
  [Run results](run-results.md#statuses).
- Set `--timeout`. Without it, nothing bounds the whole run: a server that
  sends keep-alive bytes while it generates never times out.
- To stop `melo` from a supervisor, send SIGTERM and wait about 10 seconds
  before SIGKILL; the clean-up of `melo` takes at most 8 seconds.
