# Command-line reference

This page describes the `melo` command: its modes, options, output, exit
codes and signals, and how to use it from scripts. `melo-retrieve`, the
other command installed with localmelo, compares memory retrievers offline;
see [Offline retrieval](retrieval.md).

## Synopsis

```text
melo [options] "<task>"                      # run one task (direct mode)
melo [options]                               # the interactive UI (in a terminal)
melo --base-url URL [--chat-model NAME]      # the interactive UI on this server
melo [--base-url URL] [--chat-model NAME] "" # the plain conversation loop
melo --reconfigure                           # run setup, then the interactive UI
melo --serve [--host HOST] [--port PORT]     # start the gateway
melo --serve --reconfigure                   # run setup, then start the gateway
melo --daemon install|uninstall|status       # macOS background service
```

Quote the task. Several words without quotes are joined with spaces.

## Modes

| Command | What it does |
|---|---|
| `melo "<task>"` | Direct mode: runs one task, prints the result and exits. |
| `melo` (no task) | Opens the [interactive UI](#the-interactive-ui) in a terminal, on the saved backend. With no saved backend it first asks for a model server and a model, and saves them; see [First run](#first-run). Without a terminal it exits with code 2 and says what to run instead. |
| `melo --base-url URL` (no task) | Opens the interactive UI on that server, with `--chat-model` or the first chat model the server lists. |
| `melo --reconfigure` | Runs the setup wizard, then opens the interactive UI. Without a terminal it runs the wizard, reading the answers from stdin, and exits. |
| `melo --serve` | Starts the gateway. Runs the setup wizard first if there is no configuration (in a terminal; without one it exits with code 1). |
| `melo ""` (empty task) | Opens the plain conversation loop: each line you type is a task for the same agent, and only the answer prints. `/help` lists its commands; `/exit`, `/quit`, `exit`, `quit` or Ctrl-D exits. |
| `melo --daemon …` | Manages the macOS background service that runs the gateway; see [Run the gateway at login](gateway.md#run-the-gateway-at-login-macos). |

In the `melo ""` loop, turns share one agent and its working memory, which
keeps each answer, so a turn sees the earlier ones. A turn that does not
complete prints its status line on stderr, and the loop asks for the next task.
`/exit`, `/quit`, `exit` or `quit` (in any letter case), or Ctrl-D, ends the
loop with exit code 0; Ctrl+C ends it at once (a shell shows 130). The loop
takes `--base-url` and `--chat-model`; the direct-mode options are usage
errors with it. Its conversation is separate from the web workspace's chats.
On a terminal the prompt is edited with readline: Backspace deletes a whole
character, Chinese and other wide characters included, and the Up arrow
recalls earlier lines of this loop. A line that is not valid UTF-8 prints
`melo: that line was not valid UTF-8; type it again.` and the loop asks
again.

## The interactive UI

`melo` without a task, in a terminal, opens a chat that shows its work as
it happens: each tool call as it finishes, a spinner with the step and the
time while the model works, and the answer as Markdown. What it prints
stays in the terminal's scrollback; the input box and a status bar stay at
the bottom.

```text
╭─ melo 0.1.0 ─────────────────────────────────────────────────────────────╮
│ model      Qwen3-8B · vLLM · http://127.0.0.1:8000/v1                    │
│ workspace  ~/demo-project                                                │
│ tools      4 built-in (3 ask first)                                      │
│ memory     no long-term memory (no embedding backend)                    │
│ session    new                                                           │
╰──────────────────────────────────────────────────────────────────────────╯
  / commands · esc stop · ctrl+d quit

› What is in this folder?

✓ shell_exec  ls -1 · 0.1s · allowed
  └ app.py
    notes.txt
    tests
    … +1 lines
● The folder holds app.py, notes.txt, tests and todo.md.
✻ Worked for 6.2s · 1 tool call · ↓ 412 tokens

────────────────────────────────────────────────────────────────────────────
› Ask melo · / for commands
────────────────────────────────────────────────────────────────────────────
  Qwen3-8B · ~/demo-project · ctx 2.1k · What is in this folder?         ask
```

It needs a terminal: stdin and stdout must both be terminals and `TERM`
must not be `dumb`. Otherwise `melo` prints this on stderr and exits with
code 2:

```text
melo: the interactive UI needs a terminal (stdin and stdout must be TTYs).
  Run one task:      melo "your task"
  Start the gateway: melo --serve
```

### Which model server

The first of these that applies:

1. `--reconfigure`: the setup wizard, then the backend it saved. Without a
   task it cannot be combined with `--base-url`.
2. `--base-url URL`: that server, without the config file. The model is
   `--chat-model`, else the first model that `GET URL/models` lists,
   skipping models whose id contains `embed` or `rerank` while another is
   listed. A URL without `/v1` is listed at `URL/v1/models`, and when that
   answers the UI uses `URL/v1` too. When the server does not answer the
   list, or lists no model, the model is `default` and a warning under the
   banner says why. With no config file it asks once
   `Save URL · MODEL as the default backend? [Y/n]`; Enter saves it, as the
   first run does.
3. The saved backend, with `--chat-model` in place of its model for this
   session. Nothing is checked before the UI opens: after the banner, melo
   lists the server's models once (`GET /models`) and warns when the server
   does not answer, or does not list the model:
   `⚠ Cannot reach the model server at URL (connection refused). Start it, then send your message.`
4. No saved backend: the [first-run](#first-run) questions, then the UI.

The banner names the model and the server; when they are not the saved
backend, a second line says where they come from (`from --base-url (not
saved)`, `saved as the default`) and, when melo took the first of several
models, `first of N; /model lists them`.

### Keys

| Key | What it does |
|---|---|
| Enter | Send the message. During a turn, queue it: it is sent when the turn completes. |
| Alt+Enter, Ctrl+J, or `\` then Enter | A new line in the message. |
| Esc | Close the completion menu; otherwise stop the turn. |
| Ctrl+C | Close the menu, else clear the draft, else clear a queued message, else stop the turn; a second Ctrl+C within 2 seconds quits. |
| Ctrl+D | Quit, when no turn runs and the draft is empty; with text, delete the character under the cursor. |
| ↑ ↓ | Move between lines, then through earlier messages. On an empty draft during a turn, ↑ takes back the queued message. |
| Ctrl+R | Search earlier messages. |
| Tab | Complete a command or a listed model; ↑ ↓ pick in the menu, and Enter takes the one picked. |
| Ctrl+L | Clear the screen. |
| Ctrl+Z | Nothing (melo does not suspend in this version). |

- **The queue.** One message waits at a time, shown as
  `↳ Queued for the next turn: …`. When the turn ends any other way than
  `completed` (stopped, failed), it goes back into the input box unsent.
  A message whose turn could not run at all
  (`✗ The turn could not run: …`) goes back too, in front of it.
- **Stopping.** Esc stops the turn at once; the turn ends `cancelled` and
  prints `■ Stopped`. After the answer has printed, melo spends up to 2
  seconds recording the run's steps (see
  [Procedural memory](procedures.md)); Esc then does nothing, and the next
  message waits for it.
- **History.** Messages and commands are kept in
  `~/.cache/localmelo/cli_history` (mode 0600, the newest 1,000 once it
  passes 1 MB), never the files' text. When that file cannot be opened,
  the session keeps its history in memory and says so.

### Commands

Type `/` and the commands show under the input box with what each does;
type more letters to narrow them, Tab or Enter to take one.

| Command | What it does |
|---|---|
| `/help [command]` | The commands and keys, or one command's details. |
| `/new` (also `/clear`) | Start a new session: the model forgets the conversation, and session grants end. |
| `/model [name\|number]` | Alone: the current model, and the models the server lists. With a listed id, or a number from that list (melo lists the models first if it has not yet): switch to it from the next message, and save it as the default. |
| `/backend [url]` | Switch to another model server: asks for its URL (Enter keeps the current one) and a model from its list, checks the server with `GET /models`, switches and saves both as the default. |
| `/exit` (also `/quit`) | Quit; a running turn is stopped first. |

- A command that runs is echoed in the scrollback as a message is
  (`› /model 2`), followed by its output. A refused one is not, and
  `/exit` prints nothing.
- `/new`, `/model` and `/backend` are refused while a turn runs: the
  draft stays and the hint says
  `/model is not available while a turn runs (esc stops it).`; press Enter
  again once the turn ends.
- `/model` switches only among the models the server lists. A name it
  does not list is refused, with nothing switched or saved:
  `✗ URL lists no model named 'NAME'. /model lists them.`; so is a number
  outside the list: `✗ Type a number from 1 to N, or a model id.` Only
  when there is no list (the server does not answer, or a cloud API)
  does `/model NAME` switch to it anyway, after
  `⚠ Could not list the models (REASON); switching anyway.` when the
  server did not answer.
- Saving: `/model` keeps the configured backend and changes its model.
  `/backend` saves the server as the `vllm` backend (`chat_backend`,
  `[vllm] chat_url` and `chat_model`) and keeps the rest of the config:
  embedding, `request_timeout_s` and `extra_body`; the session runs on
  the config as saved. In a `melo --base-url URL` session they switch for
  this session only, unless melo asked `Save URL · MODEL as the default
  backend?` (only when there is no config file yet) and you answered yes.
  After `s` (start anyway) at the first run's questions, `/model` and
  `/backend` save the server, as the first run would have.
- A model that does not start leaves the current one in place
  (`✗ Could not start NAME: …`). `/backend` keeps the current server when
  the new one does not answer or lists no models:
  `✗ Could not reach URL: connection refused. Still using URL.`
- A message that starts with `/` and a lowercase name, then a space or
  its end, is a command; an unknown name prints
  `Unknown command /NAME. /help lists the commands.` and keeps the draft.
  `/etc/hosts is broken` is a message. To send a message that starts like
  a command, start it with a space.

### A turn's output

- **The spinner** shows what melo does (`Thinking`, `Running`,
  `Waiting for your approval`), the step (`step 4`, never against
  `--max-steps`), the attempt from the second one, the time, the output
  tokens, and `esc to stop`. A running tool shows its own clock under it,
  and for `shell_exec` and `python_exec` their limit:
  `└ shell_exec  pytest -q tests · 0:25 of 10:00`.
- **Tool calls** print one block each when they finish: `✓` ran, `✗`
  failed, `⊘` blocked or denied, `◌` stopped or not run; the duration;
  `exit N` when a command ran and exited with another code than 0 (`grep`
  exits 1 without failing); how its approval ended; and up to 3 lines of
  its output, then `… +N lines` (a single line left prints instead).
  Escape sequences in the output are removed, so a tool cannot move the
  cursor, set the title or write the clipboard.
- **The answer** prints as Markdown after `●`. A run that did not complete
  prints its best effort, the [status line](#output) and a hint, such as
  `Is the model server running at URL?` for a server that does not answer.
- Thinking excerpts, retry notices and reflections come in a later release.

### Approvals

`shell_exec`, `python_exec` and `file_write` run only once you answer. The
panel takes the place of the input box, and your draft stays:

```text
╭─ Run a shell command? ───────────────────────────────────────────────────╮
│ shell_exec · in ~/demo-project                                           │
│   ls -1                                                                  │
╰──────────────────────────────────────────────────────────────────────────╯
❯ 1. Yes
  2. Yes, and allow every shell_exec call in this session
  3. No, and tell melo why
  "Allow every" lets any shell command run without asking for the rest of this session.
  Commands can reach files outside ~/demo-project.
  ↑↓ select · enter confirm · 1-3 · v view all · esc stop the turn
```

- `1` allows this call, `2` allows the tool for the rest of the session,
  and `3` opens a field for a reason (at most 500 characters) that the
  model reads; Enter sends it. ↑ ↓ and Enter choose too. No letter answers,
  and for 0.4 seconds after the panel opens only Esc and Ctrl+C act, so a
  message you were typing cannot answer it.
- Esc or Ctrl+C at the choices stops the turn, and a second Ctrl+C within
  2 seconds quits; in the reason field, Esc goes back to the choices.
- The command, the code or the file's new content shows exactly as it will
  run, its long lines wrapped, never cut: control characters and the
  characters that reorder or hide text show escaped (`\r`, `\x1b`,
  `\u{202e}`), with a warning line, so nothing can hide in it.
- The panel shows at most 8 lines of the body, fewer on a short terminal,
  then `… +N lines (v views all)`; `v` prints the body in full above it.
- `--yolo` runs every tool without asking, and the status bar says
  `⚠ YOLO: no approvals`.

### The workspace

melo works in the current folder, or in `--workspace DIR`. File tools
(`file_read`, `file_write`) stay in this folder. Commands and Python can
reach other files, so melo asks first. The banner warns when the workspace
is your home folder or `/`. `shell_exec` and `python_exec` stop after 10
minutes in the UI (30 and 60 seconds for a task).

### Chinese and other wide characters

The input box edits by character and places text by terminal cell: Chinese
and other wide characters type, delete and wrap whole, and deleting a
wrapped message leaves no blank rows. Known limits:

- A character built from several code points (an accent typed as a
  combining mark, an emoji sequence such as 👩‍💻) takes one Backspace per
  code point.
- Ctrl+W deletes back to the last space, which in Chinese text without
  spaces is the whole run.
- Text that an input method (such as macOS Pinyin) is still composing is
  drawn by the terminal, not by melo.

### Terminal settings

- The UI uses the 16 ANSI colors, so light and dark themes both work. Set
  `NO_COLOR` to a non-empty value to turn the colors off.
- On a terminal whose encoding is not UTF-8, or with `LOCALMELO_ASCII=1`,
  melo draws ASCII stand-ins for its symbols (`>` for `›`, `...` for `…`).
- melo rings the terminal's bell when an approval panel opens and when a
  turn of 30 seconds or more ends; `LOCALMELO_BELL=0` turns it off.

### Sessions and files

Each `melo` starts a new session; the first message creates it. Sessions
are kept in `~/.cache/localmelo/cli/sessions.db` (resuming them comes in a
later release): every turn, with your message, the answer and each tool
call with its output. `~/.cache/localmelo/cli_history` keeps your
messages. Both are plain text (mode 0600), not encrypted. The UI writes
its log to `~/.cache/localmelo/logs/cli.log`
(folder 0700, files 0600, 1 MB, two backups) and never prints a traceback
on the screen; a problem shows as one dim line naming the log. A config
file that is not valid TOML, or a log folder that cannot be made, stops
`melo` with one `melo: …` line and exit code 1. Procedural
memory follows [memory persistence](memory.md#keep-memory-between-runs), as
for a task.

### First run

In a terminal with no config file, `melo` asks for a model server and a
model, saves them, and opens the UI. Later runs go straight in.

```text
$ melo
Found a model server at http://127.0.0.1:8000/v1 (1 model).
Model server URL [http://127.0.0.1:8000/v1]:
✓ http://127.0.0.1:8000/v1 lists 1 model; using Qwen3-8B-4bit.
Saved to ~/.cache/localmelo/config.toml (melo --reconfigure changes it).
```

- melo looks for a server on this computer with `GET /v1/models` on
  `127.0.0.1` ports 8000 (vLLM, oMLX), 8400 (MLC), 11434 (Ollama) and 30000
  (SGLang), and offers the first that lists a model. Press Enter to accept
  the URL, or type another. Only the model list is requested: no chat
  request, so a server does not load a model for it. A server that lists
  a chat model comes first; one that lists only embedding or reranking
  models comes next. A server that answered without a list is named, as
  `A model server at URL answered, but HTTP 401: …`, and offered. When none
  answers, it prints
  `No model server answered on this machine (ports 8000, 8400, 11434, 30000).`
  and offers `http://127.0.0.1:8000/v1`. `LOCALMELO_DETECT_URLS` replaces the
  four URLs with a comma-separated list, and its first URL is then the one
  offered; an empty value turns the search off.
- A server that lists several models shows them as a numbered list and asks
  `Model [N]:`; type a number or an id. The default skips embedding and
  reranking models. A server that lists only such models is shown the same
  way, with a line saying that chat needs a chat model.
- The config file gets `chat_backend = "vllm"` (the key for any
  OpenAI-compatible server), its `[vllm]` table's `chat_url` and
  `chat_model`, and `embedding_backend = "none"`. Its other settings are
  kept. For embeddings, a cloud API or another backend key, run
  `melo --reconfigure`.
- When the server cannot be used, melo says why, as
  `✗ URL: connection refused.`, and asks
  `[r]etry, [s]tart anyway, [q]uit [r]:`. `r` asks for the URL again, with
  the one you typed as the default; `s` asks `Model name [default]:` and
  opens the UI without saving (`Not saved: melo asks again next time.`);
  `q` exits with code 0. The reasons are `connection refused`,
  `no answer in 5 s`, `HTTP 401` or `HTTP 403` with
  `the server requires an API key; localmelo sends none to local servers (oMLX: "Allow inference without API key")`,
  another `HTTP` status (a redirect names its target, which melo does not
  follow), `the reply is not a model list`, `the reply could not be read`,
  `not a URL` and `it lists no models`.
- An option is chosen by its first letter, so `s`, `start` and
  `start anyway` are the same answer; another answer asks again.
- Ctrl-D at a question prints `Setup cancelled.` and exits with code 0;
  Ctrl+C prints the same and exits with 130.

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
  [Which model server](#which-model-server). With `--serve` or `--daemon`,
  either is a usage error: the gateway uses the config file.

### Direct mode

These options apply to a task run without `--serve`.

| Option | Default | Description |
|---|---|---|
| `--max-steps N` | `30` | The most planning steps the whole run may take. |
| `--attempt-policy reflect\|none` | `reflect` | `reflect`: attempts with a reflection after each. `none`: one attempt, no reflection. |
| `--reflection-gate flags\|utility` | `flags` | After a reflection, start the next attempt when it chooses `continue` and its three checks hold (`flags`), or also require its utility score to reach 0.1 (`utility`, the behavior before this option). |
| `--tool-exposure retrieve\|all` | `retrieve` | Which tools each step offers the model. |
| `--builtins all\|none\|NAME,...` | `all` | The built-in tools to register. |
| `--mcp-config PATH` | not set | Add the tools of the MCP servers listed in this JSON file. |
| `--mcp-only` | off | Use only the MCP tools: the same as `--builtins none --tool-exposure all`. |
| `--memory on\|off` | `on` | `off` turns long-term memory off for this run. |
| `--memory-dir DIR` | not set | Save history, long-term and procedural memory as SQLite files in `DIR`: `history.db`, `long_term.db` and `procedures.db`. |
| `--timeout SECONDS` | not set | End the run with status `timeout` after this many seconds. |
| `--request-timeout SECONDS` | `120` | How long each chat request may wait for the server. |
| `--output-format text\|json` | `text` | Print the result as text, or the run record as one JSON line. |
| `--embedding-base-url URL` | not set | An OpenAI-compatible embedding server for long-term memory. |
| `--embedding-model NAME` | `default` | The model to request from the `--embedding-base-url` server. |
| `--progress auto\|off` | `auto` | Draw a [progress line](#progress-line) on stderr while the task runs: `auto` when stderr is a terminal and the output is text, `off` never. |
| `--routine NAME` | not set | Learn and recall this task's steps as the routine `NAME`, in any folder and whatever the task's wording. |

- `--max-steps`: at least 1. With `reflect`, a run has attempts of up to 10
  steps, at most 5 attempts, and at most `--max-steps` steps in total. See
  [How localmelo works](how-it-works.md#the-run-loop).
- `--reflection-gate`: see [Reflection](how-it-works.md#reflection). It has
  no effect with `--attempt-policy none`.
- `--tool-exposure`: see [How tools are offered](tools.md#how-tools-are-offered).
- `--builtins`: `all`, `none`, or a comma-separated list of `file_read`,
  `file_write`, `python_exec` and `shell_exec`.
- `--mcp-only` needs `--mcp-config` and cannot be combined with `--builtins`
  or `--tool-exposure`. See [Tools and MCP](tools.md#add-tools-from-mcp-servers).
- `--memory-dir` turns persistence on for this run, whatever the environment
  variables say. `~` is expanded. See
  [Memory](memory.md#keep-memory-between-runs).
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
  [Procedural memory](procedures.md#how-it-works).

### Interactive UI

These options apply to `melo` without a task.

| Option | Default | Description |
|---|---|---|
| `--workspace DIR` | the current folder | The folder melo works in; see [The workspace](#the-workspace). |
| `--yolo` | off | Run every tool without asking for approval. |

- The UI also takes `--max-steps`, `--attempt-policy`, `--reflection-gate`,
  `--builtins`, `--memory`, `--memory-dir`, `--request-timeout` and the
  `--embedding-*` options, for every turn of the session. `--tool-exposure`
  (the UI offers every tool), `--timeout`, `--output-format`, `--progress`
  and `--routine` are usage errors there, and so, for now, are
  `--mcp-config` and `--mcp-only`.
- `--workspace` that is not a folder exits with code 2:
  `melo: --workspace: not a directory: DIR`.

### Gateway and setup

| Option | Default | Description |
|---|---|---|
| `--serve` | off | Start the gateway: the HTTP API and the web workspace. |
| `--host HOST` | `127.0.0.1` | The address the gateway listens on. |
| `--port PORT` | `8401` | The port the gateway listens on. |
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
- Ctrl-D at a wizard question prints `Setup cancelled.` and exits with
  code 0; Ctrl+C prints the same and exits with 130.
- The gateway has no authentication. Keep it on `127.0.0.1`; see
  [Keep the gateway private](gateway.md#keep-the-gateway-private).

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
- options that cannot be combined: `--mcp-only` without `--mcp-config`,
  `--mcp-only` with `--builtins` or `--tool-exposure`, `--embedding-model`
  without `--embedding-base-url`, `--embedding-base-url` without
  `--base-url`;
- a direct-mode option with `--serve` or with `melo ""`, reported as
  `only for direct mode (a query without --serve): --max-steps`; without a
  task, only those the interactive UI cannot use (`--tool-exposure`,
  `--timeout`, `--output-format`, `--progress`, `--routine`);
- `--mcp-config` or `--mcp-only` without a task, reported as
  `--mcp-config and --mcp-only are not in the interactive UI yet; give them with a task (melo -- "<task>")`;
- `--workspace` or `--yolo` with a task, `--serve` or `--daemon`, reported as
  `only for the interactive UI (melo without a query, --serve or --daemon): --yolo`;
- `--base-url` or `--chat-model` with `--serve` or `--daemon`, reported as
  `--base-url and --chat-model are for a task or the interactive UI; --serve uses the config file (melo --reconfigure)`;
- `--host` or `--port` without a task and without `--serve` or `--daemon`,
  reported as
  `--host and --port are for --serve and --daemon; bare melo opens the interactive UI`;
- `--reconfigure` with `--base-url` and no task, reported as
  `--reconfigure sets up the config file, which --base-url does not use; give one of them`;
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

**In both formats:**

- stdout carries only the report. Anything a tool writes to stdout by itself,
  such as a program that `python_exec` code starts without capturing its
  output, goes to stderr instead.
- Warnings go to stderr, one line each.
- A traceback appears only when the run ends as `crashed`, after Ctrl+C, or
  when the config file is not valid TOML.

## Progress line

With text output on a terminal, `melo "<task>"` shows where the task stands
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
  [routine](procedures.md#how-it-works) has completed before on this model
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
  [Memory](memory.md#keep-memory-between-runs)); otherwise every run shows
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
| 2 | The run ended with any other status; or a usage error; or the interactive UI had no terminal, or `--workspace` is not a folder. |
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

1. A running `shell_exec` command is killed together with its process group.
2. MCP servers are closed. A server still running 6 seconds after the signal
   is killed.
3. The task record is saved with status `crashed` and the error
   `Cancelled by SIGTERM` (or `Cancelled by SIGHUP`). Reflections are not
   promoted to long-term memory.
4. `melo` exits by the signal (a shell shows 143 or 129) and prints nothing on
   stdout: no result, status line or JSON line.

There are two exceptions:

- `python_exec` code runs inside `melo`, so the signal takes effect only when
  that code returns.
- A signal that was ignored when `melo` started stays ignored. For example,
  `nohup` ignores SIGHUP.

Ctrl+C (SIGINT) stops a run at once. `melo` prints a traceback, saves the task
record as `crashed` with the error `CancelledError`, and a shell shows exit
code 130.

In the interactive UI, SIGTERM and SIGHUP (also a terminal window that
closes) stop a running turn, which is recorded as `cancelled` with the
error `Cancelled by SIGTERM`, close the tools within the same 8 seconds,
and end `melo` by the signal. Ctrl+C there is a key (see [Keys](#keys)).

## Use `melo` in scripts

Read the JSON line for details, and the exit code for the outcome:

```bash
line=$(melo --output-format json --timeout 600 "List the files in this directory")
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
