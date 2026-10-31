# Command-line reference

This page describes the `melo` command: its modes, options, output, exit
codes and signals, and how to use it from scripts. `melo-retrieve`, the
other command installed with localmelo, compares memory retrievers offline;
see [Offline retrieval](retrieval.md).

## Synopsis

```text
melo [options] -- "<task>"                   # run one task (direct mode)
melo [options]                               # the interactive UI (in a terminal)
melo -u URL [-m NAME]                        # the interactive UI on this server
melo [-u URL] [-m NAME] ""                   # the plain conversation loop
melo setup                                   # run setup, then the interactive UI
melo serve [-p PORT] [--host HOST]           # start the gateway (-p 0: a free port)
melo serve --playground                      # opt in to legacy playground (in progress)
melo import dsh-jsonl FILE [--title TITLE]   # import worker history
melo serve install|uninstall|status          # macOS background service
melo serve stop                              # stop the gateway that is running
melo -h all                                  # every option, command and setting
```

Quote the task and put it after `--` so option-like text is unambiguous.
Several words without quotes are joined with spaces.

Existing task command
lines, abbreviations and the `melo ""` loop keep their previous parsing.

<a id="commands-and-short-options"></a>
## Commands and short options

Eight command words name what `melo` does. A word counts only as the first
argument and only as an exact match, so `melo "setup the project"` (one quoted
argument) and `melo -- setup the project` are tasks, while `melo setup a venv`
prints a hint and exits 2 before any model is built.

A task that is one command
word after options, as in `melo -X gateway.port=9000 serve`, is refused the same
way, and the message says where the word goes (`melo serve -X gateway.port=9000`).
`melo -- serve` runs the task "serve"; an option value such as
`melo -w serve "fix it"` is not affected.

| Command | What it does | Options |
|---|---|---|
| `melo setup` | Choose the model server again, then open the chat. | |
| `melo config` | Show where the config file is and which settings differ from the defaults. | |
| `melo memory VERB` | See and change the notes melo keeps: `list`, `forget`, `status`, `reindex`, `import`. | `list -n N`, `forget ID`, `forget -y`, `import --undo BATCH` |
| `melo serve [install\|uninstall\|status\|stop]` | Run the gateway, manage it as a login service (macOS), or stop the one that is running. | `--host H`, `-p`/`--port N`, `--playground`, `--force` |
| `melo console [--reset]` | Open the web console in your browser, signed in; over SSH or when no browser starts, print a one-use link instead. `--reset` rotates the access token. | `--host H`, `-p`/`--port N` |
| `melo task VERB` | The `melo-task` client. | as `melo-task` |
| `melo worker BRIEF` | Run a worker brief. | `--inbox FILE`, `--inbox-fd N`, `--parent ID`, `--title T`, `-o json\|stream-json` |
| `melo import FORMAT FILE` | Import a worker history. | `--title T` |

`melo -h` lists the everyday options and three commands with what they do (`setup`,
`serve`, `console`), `melo -h all` lists every option, command and setting, and
`melo COMMAND -h` explains one command. A letter takes its value apart (`-s 5` or
`-s=5`); `--title T` names a task's worker session.

### Old spelling to new

Every old spelling still works and parses to the same result; `melo -h` leaves
them out. A setting is a `config.toml` key: `-X KEY=VALUE` changes it for one run
and `melo config set KEY VALUE` saves it (see [Settings](#settings)).

| Old spelling | New |
|---|---|
| `--chat-model` | `-m, --model` |
| `--base-url` | `-u, --url` |
| `--max-steps` | `-s, --steps` |
| `--output-format` | `-o, --output` |
| `--mcp-config` | `--mcp` |
| `--save-session` | `--title` |
| `--reconfigure` | `melo setup` |
| `--serve` | `melo serve` |
| `--daemon` | `melo serve install|uninstall|status|stop` |
| `--host` | `melo serve --host` |
| `--port` | `melo serve -p` |
| `--enable-playground` | `melo serve --playground` |
| `--allow-unconfigured` | `melo serve --force` |
| `--gateway-login` | `melo console` |
| `--gateway-rotate-token` | `melo console --reset` |
| `--brief` | `melo worker BRIEF` |
| `--inbox` | `melo worker BRIEF --inbox` |
| `--inbox-fd` | `melo worker BRIEF --inbox-fd` |
| `--parent-run` | `melo worker BRIEF --parent` |
| `--import-session` | `melo import FORMAT FILE` |
| `--attempt-policy` | `-X agent.attempt_policy=VALUE` |
| `--reflection-gate` | `-X agent.reflection_gate=VALUE` |
| `--fixed-prefix` | `-X agent.fixed_prefix=VALUE` |
| `--window-eviction` | `-X agent.window_eviction=VALUE` |
| `--reflect-tool-guard` | `-X agent.reflect_tool_guard=VALUE` |
| `--cache-layout` | `-X agent.cache_layout=VALUE` |
| `--window-messages` | `-X agent.window_messages=VALUE` |
| `--brief-items` | `-X agent.brief_items=VALUE` |
| `--reasoning` | `-X agent.reasoning=VALUE` |
| `--replay-turns` | `-X agent.replay_turns=VALUE` |
| `--note-writer` | `-X agent.note_writer=VALUE` |
| `--captured-history` | `-X agent.captured_history=VALUE` |
| `--context-window` | `-X agent.context_window=VALUE` |
| `--output-reserve` | `-X agent.output_reserve=VALUE` |
| `--safety-margin` | `-X agent.safety_margin=VALUE` |
| `--tool-exposure` | `-X agent.tool_exposure=VALUE` |
| `--retriever` | `-X retrieval.retriever=VALUE` |
| `--recall-at` | `-X retrieval.recall_at=VALUE` |
| `--recall-k` | `-X retrieval.k=VALUE` |
| `--recall-kinds` | `-X retrieval.kinds=VALUE` |
| `--recall-placement` | `-X retrieval.placement=VALUE` |
| `--recall-min-score` | `-X retrieval.min_score=VALUE` |
| `--retrieval-extra-body` | `-X retrieval.extra_body=VALUE` |
| `--admit-on-outcome` | `-X memory.admit_on_outcome=VALUE` |
| `--keep-decision-text` | `-X memory.keep_decision_text=true` |
| `--memory-dir` | `-X memory.dir=VALUE` |
| `--request-timeout` | `-X request_timeout_s=VALUE` |
| `--no-save-session` | `-X sessions.record_direct=false` |
| `--embedding-base-url` | `-X endpoint.embedding_url=VALUE` |
| `--embedding-model` | `-X endpoint.embedding_model=VALUE` |
| `--reasoning-field` | `-X endpoint.reasoning_field=VALUE` |

## Settings

A setting is a key of the [config file](configuration.md#the-config-file): `retrieval.k`
is `k` in the `[retrieval]` table, and a top-level key such as `request_timeout_s` has no
dot. `melo -h all` lists the keys with their defaults.

- `-X KEY=VALUE` (long form `--set KEY=VALUE`, as often as needed) changes one for this
  run. The value is read as TOML (`true`, `5`, `"v3"`), else taken as text; `on` and
  `off` also mean a boolean. `-X` beats the config file, which beats the default, and
  never writes the file. After `--` it is task text.
- A key that has an old option becomes that option before parsing (see the table
  above): `-X retrieval.k=5` is `--recall-k 5`, with the same checks and refusals. Any
  other key (`cli.stream`, `scheduler.policy`...) overrides the config this process
  loads, and a bad value exits 2 naming what the key allows. Nine settings are for one
  run only, so `-X` takes them and the file does not: `agent.attempt_policy`,
  `agent.reflection_gate`, `agent.tool_exposure`, `memory.dir`,
  `memory.keep_decision_text`, `retrieval.extra_body`, `endpoint.embedding_url`,
  `endpoint.embedding_model` and `endpoint.reasoning_field`.
- `agent.answer_on_stop` is on by default (`-X agent.answer_on_stop=off` turns it off): it
  also answers a run whose request the server refused for a
  budget or rate limit (HTTP 429, or an error that says a budget is used): the run still ends
  `provider_error`, but `result` holds the answer. See [Run results](run-results.md#statuses).
- `melo config` shows the file and what differs from the defaults. `melo config keys
  [FILTER]` lists every key with its value, default and source (default, file or `-X`),
  `melo config get KEY` prints one, and `melo config set KEY VALUE` checks a key and
  saves it. An unknown key names the nearest ones; a name that is an option,
  such as `agent.max_steps`, says to use that option (`-s N (--steps)`). An unset
  `agent.window_messages` shows the window it falls back to and why (`64 (local server)`,
  `20 (cloud)`) in `keys` and `get`.
- Whatever writes the file (`/model`, the console's settings page, `melo config set`)
  writes what the file had, never a `-X` value.

### Memory

`melo memory VERB` sees and changes the notes melo keeps about you, the same ones the console's
Memory tab lists (see [Memory across chats](../concepts/memory.md#memory-across-chats)). It reads and writes
`~/.cache/localmelo/memory/` and never calls a model.

| Verb | What it does |
|---|---|
| `melo memory list [-n N]` | Print the notes newest first: number, topic, text, and the chat and day each came from. `-n N` keeps the newest N. |
| `melo memory forget [ID] [-y]` | Forget note `ID`, or with no `ID` every note and the chat index (chats stay). It asks first: on a terminal a list with `Keep it` or `Keep everything` first, with piped input a numbered question read from stdin; Esc, end of input and Enter keep. `-y` does not ask. Exit 1 when nothing was forgotten. |
| `melo memory status` | Whether memory across chats is on, the number of notes and of indexed chat turns, how it searches (by meaning with the embedding model, or by words only), the folder and the last reason the embedding model was not used. |
| `melo memory import [-u BATCH]` | Offer the notes other assistants on this computer keep about you: a list of `Bring in N notes from M files`, `Show what they say` and `Not now` (piped input: a numbered question). The list starts on `Show what they say`. It writes nothing until the first row is chosen, then prints the batch; Esc, ctrl+c and end of input bring nothing in and record nothing (`Nothing brought in.`, exit 130 for Esc and ctrl+c, else 0). `-u BATCH` (`--undo`) takes that batch back and keeps notes you edited since (`Kept N notes you edited.`). Exit 1 when `--undo` finds no notes. |
| `melo memory reindex` | Rebuild the chat index from the chats and give every note a vector of the embedding model now set. Turns that ended before the last "forget everything" stay out. Exit 1 while `melo serve` or the app runs, since it owns the chats. |

```text
$ melo memory list
   3  Preferences  You prefer short answers.
      from chat “I prefer short answers”, Oct 30
$ melo memory forget 3 -y
Forgot note 3.
```

## Modes

| Command | What it does |
|---|---|
| `melo -- "<task>"` | Direct mode: runs one task, prints the result and exits. |
| `melo` (no task) | Opens the [interactive UI](../guides/terminal.md#the-interactive-ui) in a terminal, on the saved backend. With no saved backend it first asks for a model server and a model, and saves them; see [First run](../guides/terminal.md#first-run). Without a terminal it exits with code 2 and says what to run instead. |
| `melo --base-url URL` (no task) | Opens the interactive UI on that server, with `--chat-model`, else the server's only chat model or the one it says is loaded. With several chat models and none known to be loaded it exits 2, lists them and asks for `--chat-model <id>`. |
| `melo --base-url URL -- "<task>"` | Direct mode on that server, with `--chat-model`, else the model chosen the same way as in the row above, after a list request (see [General](#general)). With a list it cannot read, it asks for the model `default`. The same holds for `--brief` and `melo ""`. |
| `melo --reconfigure` | Runs the setup wizard, then opens the interactive UI; in default mode it first offers the server it finds (`Use this model server?`, a list) and shows the full backend menu only after `Use another server`, with `--developer` (`melo --developer --reconfigure`) or in developer mode (`[ui] mode = "developer"`), or with none found while a backend is saved (with nothing saved and none found, it prints the commands that start one). Without a terminal it runs the wizard, reading the answers from stdin, and exits. |
| `melo --serve` | Starts the gateway. Runs the setup wizard first if there is no configuration (in a terminal; without one it exits with code 1). In the in-progress security-1b changes, `--enable-playground` opts into authenticated legacy playground routes and is only valid with `--serve`. |
| `melo ""` (empty task) | Opens the plain conversation loop: each line you type is a task for the same agent, and only the answer prints. `/help` lists its commands; `/exit`, `/quit`, `exit`, `quit` or Ctrl-D exits. |
| `melo --daemon …` | Manages the macOS background service that runs the gateway; see [Run the gateway at login](../guides/gateway.md#run-the-gateway-at-login-macos). |

In the `melo ""` loop, turns share one agent and its working memory, which
keeps each answer, so a turn sees the earlier ones. A turn that does not
complete prints its status line on stderr, and the loop asks for the next task.

`/exit`, `/quit`, `exit` or `quit` (in any letter case), or Ctrl-D, ends the
loop with exit code 0; Ctrl+C ends it at once (a shell shows 130). At start
it prints one line saying that `shell_exec` is off in direct mode and that
`melo "" --builtins all` reopens the loop with it on (the line is absent when
the tool is already on).

The loop takes `--base-url`, `--chat-model` and
`--builtins`; the other direct-mode options are usage errors with it. Its
conversation is separate from the web workspace's chats.

On a terminal the prompt is edited with readline: Backspace deletes a whole
character, Chinese and other wide characters included, and the Up arrow
recalls earlier lines of this loop. A line that is not valid UTF-8 prints
`melo: that line was not valid UTF-8; type it again.` and the loop asks
again.

## Worker histories

Every direct run (`melo "<task>"` and `melo --brief FILE`) is recorded as a
`worker` session in the CLI sessions file, `~/.cache/localmelo/cli/sessions.db`,
so the web console's Workers view and the macOS app show it, a running one
included. Recording changes nothing the run sends or prints: the model's
requests, stdout, the JSON result and the exit code are the same, and a run
costs about 20 ms more before its first request.

Put the task after `--`; the
first task line is the title, and `--save-session TITLE` sets another.
`--parent-run ID` links a launcher and needs no other flag.

Opt out for one run
with `--no-save-session`, or for the whole home with `record_direct = false`
under `[sessions]` in the config (see the
[configuration reference](configuration.md)); an explicit `--save-session` or
`--parent-run` records regardless of the config. A store that cannot be opened
or written never fails the run: one line,
`melo: could not record this run (<reason>); the run is unchanged`, goes to
stderr and the run goes on unrecorded.

Workers are not offered by `-c`, `-r`,
or the chat picker. Workspace and git metadata use `--workspace` when supplied,
otherwise the current working directory.

`--import-session dsh-jsonl FILE [--title TITLE]` imports a dsh JSONL log
without a task. Reimporting identical file contents is a no-op; unknown or
invalid lines are counted.

The gateway can list and read worker history via
its read-only [worker API](api-v2.md#get-apisessions), but cannot continue or edit it.

Worker records are local, unencrypted, and stored with file mode 0600. Stored
text passes the existing secret redactor, and recognizable reads of sensitive
files are replaced by a path-labelled placeholder.

Redaction is pattern-based:
arbitrary secrets printed by a command, echoed by the model, or loaded through
an unrecognized tool can still be retained. Review the data before sharing it.

Thinking is subject to the session retention policy.

## Default and developer mode

`melo` has two modes. They change what `melo --help` shows, never what any
option does.

| Mode | How to choose it | What `--help` shows |
|---|---|---|
| Default | nothing, or `[ui] mode = "default"` in the [config file](configuration.md#the-config-file) | The short help (23 lines): the everyday options, three commands with a line on what each does, then where to find the rest. |
| Developer | `[ui] mode = "developer"`, or `--developer` for one run | Everything `melo -h all` prints: every option, command and setting; each experimental setting names the issue that decides it. |

Every option parses and works in both modes, from the command line and from
the config file, so scripts and the benchmark keep running: the mode hides an
option from help, it does not remove it. `--developer` takes effect for help
wherever it stands on the command line (`melo --help --developer` works), and
needs its full name.

A `[ui] mode` that is neither `"default"` nor
`"developer"` prints one line on stderr and shows the default help; it never
stops a run or the gateway.

### Option tiers

| Tier | Meaning |
|---|---|
| user | What a new user needs; no setup beyond these. Default-mode help shows exactly these. |
| advanced | Stable options for scripts, the gateway, workers and tuning. |
| experimental | Compares designs on the benchmark, may change or go away. Each names the issue whose result decides it; the winner becomes the default and the flag goes one release later. |

The table below is the whole classification, in the order developer help lists
it. It is the same table as `support/option_tiers.py`, and a test keeps the
two equal.

| Option | Tier | Decided in |
|---|---|---|
| `-h`, `--help` | user |  |
| `-V`, `--version` | user |  |
| `-m`, `--model` | user |  |
| `-u`, `--url` | user |  |
| `-w`, `--workspace` | user |  |
| `-s`, `--steps` | user |  |
| `-t`, `--timeout` | user |  |
| `-o`, `--output` | user |  |
| `-c`, `--continue` | user |  |
| `-r`, `--resume` | user |  |
| `-y`, `--yolo` | user |  |
| `--developer` | advanced |  |
| `--memory` | advanced |  |
| `--builtins` | advanced |  |
| `--mcp` | advanced |  |
| `--routine` | advanced |  |
| `--progress` | advanced |  |
| `--title` | advanced |  |
| `--mcp-only` | advanced |  |
| `--chat-model` | advanced |  |
| `--base-url` | advanced |  |
| `--max-steps` | advanced |  |
| `--output-format` | advanced |  |
| `--mcp-config` | advanced |  |
| `--brief` | advanced |  |
| `--serve` | advanced |  |
| `--gateway-login` | advanced |  |
| `--reconfigure` | advanced |  |
| `--gateway-rotate-token` | advanced |  |
| `--enable-playground` | advanced |  |
| `--host` | advanced |  |
| `--port` | advanced |  |
| `--allow-unconfigured` | advanced |  |
| `--daemon` | advanced |  |
| `--import-session` | advanced |  |
| `--save-session` | advanced |  |
| `--no-save-session` | advanced |  |
| `--parent-run` | advanced |  |
| `--inbox` | advanced |  |
| `--inbox-fd` | advanced |  |
| `--context-window` | advanced |  |
| `--output-reserve` | advanced |  |
| `--safety-margin` | advanced |  |
| `--cache-layout` | advanced |  |
| `--window-messages` | advanced |  |
| `--memory-dir` | advanced |  |
| `--request-timeout` | advanced |  |
| `--embedding-base-url` | advanced |  |
| `--embedding-model` | advanced |  |
| `--attempt-policy` | experimental | #185 |
| `--reflection-gate` | experimental | #185 |
| `--brief-items` | experimental | #185 |
| `--fixed-prefix` | experimental | #183 |
| `--window-eviction` | experimental | #183 |
| `--reflect-tool-guard` | experimental | #183 |
| `--reasoning` | experimental | #183 |
| `--reasoning-field` | experimental | #183 |
| `--replay-turns` | experimental | #183 |
| `--note-writer` | experimental | #183 |
| `--tool-exposure` | experimental | #3 |
| `--captured-history` | experimental | #4 |
| `--retriever` | experimental | #4 |
| `--recall-at` | experimental | #4 |
| `--recall-k` | experimental | #4 |
| `--recall-kinds` | experimental | #4 |
| `--recall-placement` | experimental | #4 |
| `--recall-min-score` | experimental | #4 |
| `--admit-on-outcome` | experimental | #4 |
| `--keep-decision-text` | experimental | #4 |
| `--retrieval-extra-body` | experimental | #4 |

A new option that compares designs starts as experimental, with an issue that
will remove it, in the same pull request that adds it. A new user option is a
deliberate change: a test pins the number of options default-mode help shows.

## Options

### General

| Option | Default | Description |
|---|---|---|
| `query` (positional) | not set | The task to run. Giving one selects direct mode; an empty one (`""`) opens the plain loop; none opens the interactive UI. |
| `-h`, `--help` | | Show the help and exit: the short help in default mode, everything in [developer mode](#default-and-developer-mode); `-h all` and `-h COMMAND` choose another. |
| `--version` | | Print `melo <version>` (the package version, then the short commit on a git checkout, e.g. `melo 0.1.0 (baab21e)`) and exit 0. |
| `--developer` | off | Show every option in `--help`, grouped by tier, for this run; the same as `[ui] mode = "developer"`. It changes only the help. |
| `--base-url URL` | not set | Send chat requests to this OpenAI-compatible server instead of the configured backend. |
| `--chat-model NAME` | with `--base-url`, the server's only chat model or its loaded one (see [Modes](#modes)); else the configured model | The model to request from the `--base-url` server, or instead of the configured chat model. |

- `--base-url` is used exactly as given, so include `/v1`, as in
  `http://127.0.0.1:11434/v1`.
- `--base-url` sends no API key, and it does not read the config file, so
  `extra_body` and `request_timeout_s` do not apply.
- Without `--chat-model`, `--base-url` asks the server for its models before the
  first chat request, in every mode (a task, `--brief`, `melo ""` and the
  interactive UI): `GET URL/models`, and between several chat models
  `GET URL/models/status` (oMLX) and `GET <root>/api/ps` (Ollama), none sending
  a key. The model is the only chat model (an id with `embed` or `rerank` is
  not one) or the one the server says is loaded. With several chat models and
  none known to be loaded (the status route is missing, answers 401, or names
  none or more than one), the model `default` could make the server load any of
  them, so a direct run ends before its first chat request with exit code 2:

  ```text
  melo: error: http://127.0.0.1:8000/v1 serves several models and it is not known which one is loaded:
    big-27B
    small-8B
  Pass --chat-model <id>.
  ```

  When the status route answered 401 (it needs a key), one more line comes before
  the hint: `The server did not say which models are loaded (it needs a key for that).`
  With `--brief` the same text, on one line, is the `error` of a `startup_error`
  result with `exit_code` 2 (see [Output](#output)). Plain `--output-format json`
  prints no record, as for any refusal before the run. A server whose list cannot
  be read (no answer, an HTTP error, no model listed) is asked for the model
  `default`, as before, and text output adds one line on stderr, such as
  `melo: warning: Could not list the models at URL (connection refused); asking for the model 'default'.`
  The JSON formats print no warning. With `--chat-model` nothing is listed.
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
| `--brief FILE` | not set | Read a direct task from a nonempty UTF-8 file, at most 1 MiB, relative to the invocation directory. Requires no positional task, defaults to worker JSON output and rejects text output. |
| `--inbox FILE` | not set | With `--brief`, accept complete appended instruction lines at planning boundaries. The path is explicit, relative to the invocation directory, and must resolve outside the workspace. |
| `--inbox-fd N` | not set | With `--brief`, accept the same lines from an inherited, readable pipe fd `N`. Requires the parent to keep the write end; the read end is close-on-exec before tools start. Cannot be combined with `--inbox`. |
| `--workspace DIR` | the current folder | The folder the run works in: the file tools stay inside it and `shell_exec` and `python_exec` start there; see [With a workspace root](../guides/tools-mcp.md#with-a-workspace-root). |
| `--fixed-prefix on\|off` | `on` | Use a versioned fixed system prefix with the local date frozen at run start; `off` sends the earlier requests. Overrides `[agent] fixed_prefix`. |
| `--window-eviction slide\|chunk\|boundary` | `chunk` | How working memory drops old turns: `chunk` eight messages at a time; `slide` one unit per step once full (the earlier default); `boundary` also drops a finished attempt's turns at the next attempt. Overrides `[agent] window_eviction`; see [Memory](memory.md#working-memory-window). |
| `--reflect-tool-guard on\|off` | `on` | With `on`, every reflection request offers the tools of that step's planning request (same list, order and schema, so a server's prompt prefix holds) and its last user message ends with `Answer with the JSON object only; do not call any tool.`; `tool_choice` is untouched. The retry request is unchanged with `--cache-layout v2`; with the default `v3` it continues the request. With `off`, the request has neither. Overrides `[agent] reflect_tool_guard`; see [Configuration](configuration.md#context-accounting). |
| `--cache-layout v2\|v3` | `v3` | With the fixed prefix, `v3` appends a reflection once at the attempt boundary as a window message, keeps the reflection request and its reply in the window, and asks an unreadable reflection again on the same prefix; `v2` sends the earlier requests. Without the fixed prefix neither changes anything. Overrides `[agent] cache_layout`; see [Configuration](configuration.md#context-accounting). |
| `--window-messages N` | `64` on a local server or a URL, `20` on a cloud vendor | The working window's message cap, 8 to 256; `20` sends the earlier cap. Overrides `[agent] window_messages`; see [Memory](memory.md#working-memory-window). |
| `--captured-history on\|off` | `off` | Store a bounded, pattern-redacted captured result separately from the model preview; see [configuration](configuration.md#context-accounting). |
| `--brief-items on\|off` | `off` | Experimental. With `--brief`, name the brief's numbered items that no edit has touched yet in each reflection request and two steps before the last attempt ends; see [Run a worker task](#brief-items). Overrides `[agent] brief_items`; inert without `--brief`. |
| `--context-window N`, `--output-reserve N`, `--safety-margin N` | not set | Opt in to per-request context accounting; all three must resolve to positive integers and the window must exceed the other two combined. |
| `--max-steps N` | `30` | The most planning steps the whole run may take. |
| `--attempt-policy reflect\|none` | `reflect` | `reflect`: attempts with a reflection after each. `none`: one attempt, no reflection. |
| `--reflection-gate flags\|utility\|patient` | `patient` | After a reflection, start the next attempt when it chooses `continue` and its three checks hold (`flags`), or also require its utility score to reach 0.1 (`utility`, the behavior before this option), or as `flags`, but once per run also when the next step is not novel (`patient`, the default). |
| `--tool-exposure retrieve\|all` | `retrieve` | Which tools each step offers the model. |
| `--builtins all\|none\|NAME,...` | the mode's set | The built-in tools to register. Without the flag, a direct run gets `file_read`, `file_write` and `python_exec`; `all` adds `shell_exec` and `list_files` (the interactive UI keeps all five by default). |
| `--mcp-config PATH` | not set | Add the tools of the MCP servers listed in this JSON file. |
| `--mcp-only` | off | Use only the MCP tools: the same as `--builtins none --tool-exposure all`. |
| `--memory on\|off` | `on` | `off` turns long-term memory off for this run. |
| `--memory-dir DIR` | not set | Save history, long-term and procedural memory as SQLite files in `DIR`: `history.db`, `long_term.db` and `procedures.db`. |
| `--timeout SECONDS` | not set | End the run with status `timeout` after this many seconds. |
| `--request-timeout SECONDS` | `120` | How long each chat request may wait for the server. |
| `--output-format text\|json\|stream-json` | `text` (`json` with `--brief`) | Text, one JSON result, or newline-delimited events and result (`stream-json` requires `--brief`). |
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
  interactive UI. Over-budget requests are fitted on a copy, or the run ends `context_budget_exceeded`;
  the output reserve is not a generation cap. See [context accounting](configuration.md#context-accounting).
- `--max-steps`: at least 1. With `reflect`, a run has attempts of up to 10
  steps, at most 5 attempts, and at most `--max-steps` steps in total. See
  [How localmelo works](../concepts/how-it-works.md#the-run-loop). To limit every
  model request instead (planning, reflection and the rest), set `-X agent.max_calls=N`
  (a developer setting, a whole number of 2 or more, no option of its own): the run
  stops itself before the request that would pass `N` and ends `budget_exhausted` with
  the error `Call budget reached: ...`. With `agent.answer_on_stop` on (the default) the last call
  is kept to ask once, without tools, for the final answer from the evidence, so the
  run sends `N - 1` requests and then that one. Both appear in `melo -h all`, and
  [Run results](run-results.md#statuses) says what the record holds.
- `--reflection-gate`: see [Reflection](../concepts/how-it-works.md#reflection). It has
  no effect with `--attempt-policy none`. `--reflection-gate flags` gives the gate from
  before `patient` became the default (`-X agent.reflection_gate=flags` for one run).
- `--tool-exposure`: see [How tools are offered](tools.md#how-tools-are-offered).
- `--builtins`: `all`, `none`, or a comma-separated list of `file_read`,
  `file_write`, `list_files`, `python_exec` and `shell_exec`. A direct run's default set
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
- Live reply streaming is on by default in the interactive UI: `/stream off`
  turns it off for later turns, `-X cli.stream=false` for one run and
  `melo config set cli.stream false` for good. A per-model `[cli.reasoning_tags]`
  hint tells the UI how the model frames reasoning; a model with no hint shows
  its reply when complete. The UI draws only redacted released text in the live
  area, labels provisional text that did not become the final answer, and prints
  the final answer once. A provider without streaming sends a complete reply,
  and so does a server that ignores a streamed request or closes the stream
  before sending anything: melo uses its whole reply, asks it once more without
  streaming when needed, and says so once. Direct mode and stdin-closed runs
  never stream. See [Streamed replies](../guides/terminal.md#streamed-replies).
- Text attachments in the interactive UI: `/attach PATH` stages one file
  (quote paths with spaces), `/attachments` lists staged files, and `/detach N`
  or `/detach all` removes them. Explicit selection from `@` file completion
  stages a file; typed `@file` prose alone does not. Workspace-only regular
  UTF-8 text, no symlinks, traversal, URLs, recognized secret files, binary or
  images; sanitized names are at most 200 characters. Hardlinks can refer to
  files outside the workspace; this is not a sandbox. Up to 8 files, 256 KiB
  each and 512 KiB total in original bytes,
  subject also to the composed prompt limit. Redaction is best effort; if
  cleaning or redaction changes a snapshot or path, review and confirm it.
  Snapshots persist unencrypted with the session and `/retry` uses the saved
  snapshot, not the live file. Pairs resume has a 32,000-character user/answer
  budget and may omit a turn; window resume can clip or omit it too. See
  [Attach text files](../guides/terminal.md#attach-text-files).
- Shell commands in the interactive UI: at an idle prompt `! COMMAND` runs
  COMMAND in `$SHELL` (else `/bin/sh`) as the user, in the workspace, with the
  terminal handed over (Ctrl+C and Esc go to the command). It is not a sandbox:
  no approval, checker or policy applies. The output stays in the terminal, never
  in model context, history or stored events; `/attach !` attaches a redacted
  snapshot of the last output to the next message. `!` alone shows help,
  `\! text` sends a literal leading `! `; during a turn, approval, switch or
  another command it is refused and the draft kept. Not saved in the input
  history. See [Run a shell command](../guides/terminal.md#run-a-shell-command).
- `--workspace` is expanded and resolved here too, and a path that is not a
  folder gives the same one-line usage error as with a task; see
  [Usage errors](#usage-errors).

### Gateway and setup

| Option | Default | Description |
|---|---|---|
| `--serve` | off | Start the gateway: the HTTP API and the web workspace. |
| `--host HOST` | `127.0.0.1` | The address the gateway listens on. |
| `--port PORT` | `8401` | The port the gateway listens on. With `--serve`, `0` asks the system for a free port: the startup lines print the real address, the sign-in line (printed when no browser opens) names the port for `melo --gateway-login --port PORT`, and `melo-task` needs `--gateway URL`. `0` is refused with `--daemon` and the gateway authentication commands, and in `[gateway] port`. |
| `--allow-unconfigured` | off | With `--serve`, start the gateway even when no backend is configured or the config file has validation errors, so it can be configured over `/api/settings`. |
| `--reconfigure` | off | Run the setup wizard (in default mode without `--serve`, first the found server and `Use this model server?`; the full menu after `Use another server`, with `--developer` or `[ui] mode = "developer"`, or with none found while a backend is saved), then start the gateway (with `--serve`) or open the interactive UI (without). Without a terminal it reads the answers from stdin; without `--serve` it then exits. |
| `--daemon install\|uninstall\|status` | not set | Manage the macOS background service; `install` also takes `--port`. |

- The defaults of `--host` and `--port` come from the `[gateway]` table of
  the config file. The options change them for this run only.
- `--host`, `--port` and `--reconfigure` are ignored when you give a task.
- Without a task, `--host` and `--port` require `--serve`, `--daemon`,
  `--gateway-login` or `--gateway-rotate-token`.
- Without a terminal and with no config file, `melo --serve` prints one
  line on stderr and exits with code 1, as under launchd:
  `melo: no backend is configured; run melo setup in a terminal`.
  `--reconfigure` still runs the wizard without a terminal and reads its
  answers from stdin, so a script can pipe them in.
  `--allow-unconfigured` skips both the wizard and startup validation so
  `melo --serve --allow-unconfigured` starts without a config file; it
  cannot be combined with `--reconfigure` or used without `--serve`.
- Ctrl-D at a wizard question prints `Setup cancelled.` and exits with
  code 0; Ctrl+C prints the same and exits with 130.
- `melo --gateway-login [--port PORT]` (`melo console`) opens the web console in
  the default browser when stdin and stdout are terminals on this computer (no
  `SSH_CONNECTION`, `SSH_CLIENT` or `SSH_TTY`; `BROWSER` picks the browser). It
  prints the one-use sign-in link, to a terminal only, when no browser starts
  or over SSH. Exit 2 when stdout is not a terminal (a message on stderr, nothing on stdout),
  exit 1 when nothing listens on the port.
  `melo --gateway-rotate-token` rotates the shared local gateway token.
  The gateway refuses non-loopback binds and requires authentication; see
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
- `--host` or `--port` without a task, `--serve`, `--daemon`, or a gateway
  authentication command, reported as
  `--host and --port are for --serve and --daemon; bare melo opens the interactive UI`;
- `--port 0` with `--daemon` or a gateway authentication command, reported as
  `--port 0 picks a free port for --serve only; …`;
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

**JSON without `--brief`** (`--output-format json`):

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

**With `--brief`:** JSON is a versioned worker result: `v`, `status`, `run_id`,
`exit_code`, `head_sha`, `git_dirty`, `tests`, `run` and `error`. HEAD and dirty
state come from the workspace, and are null when unavailable.

`run` is the run record without `config`, or null for startup or transport errors. Without a
`Check:` line in the brief, `tests` is `recorded` only for the last passing
recognized test check of the final attempt, otherwise `not_recorded`.

A recorded item contains the raw command or Python code, redacted and capped at
2,000 characters with `command_truncated`, its nullable exit code and
`tool_event` source. It is evidence from tool output, not a guarantee that a task is correct.

With a `Check:` line, `tests` is the harness's own result instead (see below):
`passed` or `failed`, with one item per command of the gate's last run, each
with `command`, `command_truncated`, `exit_code`, `duration_s` and the source
`brief_check`, and `output_tail` (the last 20 lines of the check's output,
redacted, at most 4,000 characters; absent from records of older versions),
plus `refused` (the reason) when the policy did not let the command run.

With `--brief --output-format stream-json`, stdout is NDJSON: each agent event
is wrapped as `{ "v": 1, "type": "event", "event": ... }`, followed by a
`{ "v": 1, "type": "result", "result": ... }` worker result when the run
finishes normally. A broken or undrained output pipe is a failure; signals,
process death or a broken pipe can leave no result.

Drain stdout and check
both the final record and the exit code. Agent construction failures after a
valid brief, including config-file values the agent would refuse and an
invalid TOML file, produce `startup_error` with `run: null` and exit 1.

A `--base-url` brief with no `--chat-model`, on a server that serves several chat
models and does not say which one is loaded, produces it too, with `exit_code` 2
and the list in `error` (see [General](#general)). For
event output failures the worker attempts a `transport_error` result with
`run: null` and exits 1; a failed or blocked output may prevent that record
from arriving.

A broken pipe while the startup record is written still exits
1. Usage errors remain resultless. Worker output uses pattern-based
redaction and omits some recognizable sensitive-file reads, including simple
shell and Python reads. A startup error's text is redacted too, and the
user's home directory is shown as `~`.

Arbitrary commands, code and model
output can still expose secrets: the stream is not a secret boundary. Review
before sharing.

Direct runs, this one included, are recorded by default (see above).

A `--brief` run also differs from a plain task in four ways. The query is the
brief followed by one fixed paragraph, the same on every request: make the
smallest change that satisfies the task as soon as you know where it goes, run
the task's check commands after each change, fix what you broke (tests that
assert the old behaviour included), read the lines you will edit first because
earlier tool output leaves the context after about ten steps, and say in the
final answer which checks ran and their result.

A reflection whose action is
`continue` starts the next attempt while steps and attempts remain, whatever
its three flags say (the `reflection` event still records them). When an
attempt boundary finds the workspace unchanged since the run started, the next
attempt's first request carries the line "Nothing has changed in the workspace
yet. Start this attempt by writing the part of the task you can already write,
then continue." and a `no_change` notice is emitted; a workspace that is not a
git tree gets neither.

Runs without `--brief` are unchanged.

Three more aids keep a brief run writing. After 6 steps in a row in which the
workspace digest has not changed since the last change seen (or the start), one
user line, "You have read for 6 steps without changing anything. Write the part
of the task you can already write now (the smallest item first), then
continue.", is added and a `no_change` notice with detail `{"scope": "step"}`
is emitted; the count restarts at each attempt and after each probe, so the line
comes at most once every 6 steps (the boundary line stays).

When the window or a
fit drops tool output, the run remembers one line per dropped result (tool, its
key argument shortened to 80 characters, output length, newest 8) and each
planning request ends with "Earlier tool output no longer in your context:
file_read support/sessions/runner.py (49,999 chars); shell_exec `grep -n ...`
(2,310 chars). Read again what you need before you edit." (not on a step that
carries an inbox line). The reflection request ends with a sentence that asks
for exact file paths with line numbers in `useful_evidence` and for the first
command or edit to run next as the first `next_promising_directions`; when the
boundary line above is added, it ends with "Start with: <that direction>"
(redacted, at most 300 characters).

The fourth difference is that the run checks itself. A brief names its checks on
lines that start with `Check:` (no leading space); the rest of each line is one
shell command, for example `Check: uv run pytest tests/cli -q`.

Before the run would end `completed`, the harness runs the commands in order, each as a
`shell_exec` call of the run's executor, whether or not the run offers
`shell_exec` to the model (in direct mode it does not, unless `--builtins all`
or `--builtins shell_exec` says so; `--builtins none` does not stop the check:
to skip it, remove the line). Only the harness's call gets past that filter:
the model's tool list and its own `shell_exec` calls stay as they are.

The checker's `pre_execute`, the workspace root (`--workspace`), the shell timeout
and the redaction apply as they do to the model's own calls. A check passes on
exit status 0.

When one fails, and the attempt has a step left
(or a `done` reflection can start another attempt), the run does not end: the
next planning request carries one user message with the command, its exit code
and the last 60 lines of its output, redacted, then "The task's check failed. Fix
what it shows, run it again, then finish." Each failing check costs one retry,
and a run has two. The first failure of a round stops the round.

A check that is
refused is not retried. When a check still fails at the end, the run ends
`completed` as before, but `tests` is `failed`, a `brief_check_failed` notice
says so, and the exit code is 4 (see [Exit codes](#exit-codes); stderr gets one
line, `melo: the brief's check failed: <command> (exit N)`, with `not run` for
a refused check and `no exit status` for a timeout); when every check of the last round
passes, `tests` is `passed`.

Each run of a check emits a `brief_check` notice
with its command, exit code and duration. When the run is in its last attempt with exactly two steps of its budget left,
the workspace has changed since the checks last ran (or since the run started),
and the gate has not run in that attempt, the harness runs the checks once
before the next planning request, so the model can still fix what they show.

A pass changes nothing; a failure sends the same message as the gate and uses one
of the two retries. `max_steps` stays a hard bound.

A brief with no `Check:`
line behaves as before, and the model's own account of its checks is still only that.

<a id="confined-writes"></a>

**Writes stay in the workspace (macOS; `[agent] confine_writes`, on by default).**
A worker's `shell_exec` and `python_exec` children, the checks included, start
under a `sandbox-exec` profile that refuses every file write outside the workspace
(`--workspace`, else the current folder), whatever the command does: `cd ..` and a
write, `uv run` in another checkout and a child `git` or Python are all refused.
Writes are allowed in the workspace, in `.melo-tmp` there (`TMPDIR`, `TMP` and
`TEMP` point at it; a `.gitignore` in it keeps it out of `git status`, and it stays
with the workspace), in your own temporary folder (`/var/folders/.../T`, where
macOS's `mktemp` and `mktemp -d` write whatever `TMPDIR` says; `/tmp` stays
refused), in uv's cache (`UV_CACHE_DIR`, else `~/.cache/uv`), black's
(`~/Library/Caches/black`) and npm's (`~/.npm`), in `/dev/null`, `/dev/zero`, the
terminal, the standard streams, `/dev/fd` and pseudo-terminals, and, when the
workspace is a linked git worktree, in its own git directory and the repository's
`objects`: `git status`, `diff` and `add` work, a commit to a branch is refused,
and the main checkout's files stay unwritable.

`uv run`, `uv sync`, `pytest`,
`ruff`, `black`, `mypy` and `npm` were measured to need nothing else. Reads and
the network are not confined.

A refused write keeps its error and gains a first line (after `[exit code N]`):

```text
melo: writing outside the workspace is not allowed in this run (refused: /Users/me/outside.txt); write in /Users/me/project, and put scratch files in $TMPDIR.
```

`run.confinement` in the result counts them: `{"writes": "confined",
"refused_writes": 1}`. When the profile cannot be applied (`sandbox-exec` is
missing, or melo runs inside a sandbox that refuses another), the worker stops
before its first command with a `startup_error` and the line `writes cannot be
confined to the workspace: <why>; set agent.confine_writes to false to run without
it (-X agent.confine_writes=false for one run)`.

On other systems the run goes on
unconfined, and `run.confinement` is `{"writes": "not_confined", "reason": ...}`.
`-X agent.confine_writes=false` turns it off for one run; then the result has no
`confinement` key.

Runs without `--brief`, the interactive UI, the web console and
the app are not confined, unless the developer setting `[agent] confine_direct_runs
= true` (`-X agent.confine_direct_runs=true` for one run; off by default) is on: a
task given on the command line and the `melo ""` loop then get the same
confinement, with the workspace the same, and a run that cannot be confined stops
before its first command with `melo: error: writes cannot be confined to the
workspace: <why>; set agent.confine_direct_runs to false to run without it
(-X agent.confine_direct_runs=false for one run)` and exit code 1.

<a id="brief-items"></a>

**Experimental: `--brief-items on` (`[agent] brief_items = true`, off by default).**
A brief with several numbered items and a weak check can end `completed passed`
with one item untouched. With the option on, the harness reads the brief's
top-level items (`1.`, `2)` and `- [ ]` lines outside code blocks; the first line
of each, at most 12, shortened to about 120 characters; a brief with fewer than
two gets nothing) and marks each from the files `git status` lists as changed
since the run started: `changed` when a file the item names (a word with a `/`
or a file extension) was edited, `not yet` when it names files and none was,
`no file named` otherwise.

A mark speaks of edits only; it never says an item is
done. The marks go into the reflection request at each attempt boundary and into
the check two steps before the last attempt ends, always as the same block,
`Items in the brief with no edit yet: 2, 3` followed by those item lines, so the
planning requests in between keep their prefix.

The block is left out when no
item is `not yet`. Each use emits a `brief_items` notice with detail
`{"scope": "reflection" | "deadline", "marks": {"1": "changed", ...}}`.

A workspace that is not a git tree gets no block. Without `--brief` the option
does nothing.

With an explicit `--inbox FILE`, a worker reads complete LF-terminated,
nonempty UTF-8 lines at the start of each planning step. An incomplete last
line waits for its newline.

The file may initially be absent, but must otherwise
be an append-only regular file (not a symlink), at most 1 MiB total and 16 KiB
per line. Serialize writers, append each full line in one write and use a fresh
inbox per run.

There is no default inbox: `--brief` alone reads no file, and the
inbox path is never put in the model's context.

With `--inbox-fd N`, the same lines, total and per-line limits and
acknowledgements come from an inherited, readable pipe file descriptor instead.
The parent opens the pipe, passes the read end (for example with
`pass_fds=(read_fd,)` in `subprocess.Popen`) and keeps the write end open; it
appends one complete line at a time.

The worker checks that `N` is a pipe before
the first model call and makes the read end close-on-exec there, so
`python_exec`, `shell_exec` and MCP server subprocesses get no copy and cannot
write the channel. This is the recommended inbox for untrusted tasks.

A closed fd or one that is not a pipe is refused at the same point as a bad file, as a
clean `setup_error` (exit 2) with no model call. An fd has no path, so the
workspace check below applies only to a file inbox.

`--inbox` and `--inbox-fd`
cannot be combined.

For a file inbox, the path must resolve outside the run's workspace, after
symlinks. A path inside it is refused before any model call, because the default
`file_write` and `python_exec` tools could rewrite it; so are a symlink, FIFO,
directory, oversized file and unreadable path, which are checked at the same
point.

An empty, non-UTF-8 or oversize message, or a file replaced or truncated
after a line was accepted, ends the run at the boundary that reads it. All of
these are `setup_error` runs (exit 2) with the plain message and no traceback,
not `crashed`.

The reader requires the bytes already read to stay a prefix of
the file: a same-inode rewrite that preserves them is accepted as appended
content, while a shorter file or an earlier changed byte fails. It cannot detect
a truncate-and-rewrite that keeps every previously read byte.

A file inbox is a steering channel, not a trust boundary: same-user processes,
and `python_exec`/`shell_exec` code with the user's permissions, can still write
it or any other file. Keeping it outside the workspace only stops the confined
file tools from rewriting it.

The fd channel closes that tool-write path, but it
is not authentication: a process that already holds the write end can steer the
run, so the parent must not pass that end to the worker's tools or to another
untrusted process.

In `stream-json`, `inbox.accepted` reports each inserted line's ending byte
`offset`, step and attempt, without its text. An acknowledgement means the
instruction entered the live run's context, not that the model received or
followed it.

Acknowledgements are never dropped: a slow or stalled reader makes
the worker wait for the writer instead of skipping records. Messages appended
after the final planning boundary may never be accepted.

Offsets are committed
once per live run, not across crashes or restarts; reconcile acknowledgements
and use a fresh file or pipe for a new run. A saved worker records the redacted
text in its event history.

Redaction can change that text, and session resume
does not reconstruct inbox events as conversation messages; do not treat saved
history as an exact replay.

**In both formats:**

- stdout carries only the report. Anything a tool writes to stdout by itself,
  such as a program that `python_exec` code starts without capturing its
  output, goes to stderr instead.
- Warnings go to stderr, one line each.
- A traceback appears only when the run ends as `crashed`, after Ctrl+C, or
  when the config file is not valid TOML and no `--brief` was given.

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
| 2 | The run ended with any other status; or a usage error; or the interactive UI had no terminal; or `--workspace` is not a folder, in either mode; or `melo --base-url URL`, with or without a task, served several chat models with no `--chat-model` and none known to be loaded (a `--brief` run reports it as a `startup_error` result whose `exit_code` is 2). |
| 3 | The gateway could not start, for example because its port is in use. |
| 4 | A `--brief` run ended `completed`, but its brief's check failed for good: `tests.state` is `failed`, a refused check included. `status` stays `completed` and the record's `exit_code` is 4. A brief without a `Check:` line never exits 4. |
| 129 | SIGHUP ended the run. |
| 130 | Ctrl+C (SIGINT) ended the run. |
| 143 | SIGTERM ended the run. |

129, 130 and 143 are what a shell reports for a process ended by that
signal. Cancelling the setup wizard, or `q` at the first-run questions,
exits with 0.

Before the interactive UI opens, Ctrl-D at a setup question
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
- With `--brief`, exit code 4 means the run completed but its `Check:` command
  failed: `melo --brief brief.md && next-step` does not run `next-step`. Read
  `tests.state` in the record for the detail.
- Set `--timeout`. Without it, nothing bounds the whole run: a server that
  sends keep-alive bytes while it generates never times out.
- To stop `melo` from a supervisor, send SIGTERM and wait about 10 seconds
  before SIGKILL; the clean-up of `melo` takes at most 8 seconds.
