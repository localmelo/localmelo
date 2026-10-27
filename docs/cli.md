# Command-line reference

This page describes the `melo` command: its modes, options, output, exit
codes and signals, and how to use it from scripts. `melo-retrieve`, the
other command installed with localmelo, compares memory retrievers offline;
see [Offline retrieval](retrieval.md).

## Synopsis

```text
melo [options] "<task>"                      # run one task (direct mode)
melo [--base-url URL] [--chat-model NAME] "" # conversation loop in the terminal
melo --serve [--host HOST] [--port PORT]     # start the gateway
melo --serve --reconfigure                   # run setup, then start the gateway
melo --daemon install|uninstall|status       # macOS background service
```

Quote the task. Several words without quotes are joined with spaces.

## Modes

| Command | What it does |
|---|---|
| `melo "<task>"` | Direct mode: runs one task, prints the result and exits. |
| `melo --serve` | Starts the gateway. Runs the setup wizard first if there is no configuration. |
| `melo ""` (empty task) | Opens a conversation loop in the terminal: each line you type is a task for the same agent. `/help` lists the commands; `/exit`, `/quit`, `exit`, `quit` or Ctrl-D exits. |
| `melo` (no task) | The same as `melo --serve`. |
| `melo --reconfigure` | Runs the setup wizard, then starts the gateway (the same as `melo --serve --reconfigure`). |
| `melo --daemon …` | Manages the macOS background service that runs the gateway; see [Run the gateway at login](gateway.md#run-the-gateway-at-login-macos). |

> [!NOTE]
> This will change: a planned interactive CLI will open when you run `melo`
> without a task (see the
> [Roadmap](https://localmelo.github.io/localmelo/en/updates.html#roadmap)).

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

## Options

### General

| Option | Default | Description |
|---|---|---|
| `query` (positional) | not set | The task to run. Giving one selects direct mode; an empty one (`""`) opens the loop. |
| `-h`, `--help` | | Show the help and exit. |
| `--base-url URL` | not set | Send chat requests to this OpenAI-compatible server instead of the configured backend. |
| `--chat-model NAME` | `default` with `--base-url`; else the configured model | The model to request from the `--base-url` server, or instead of the configured chat model. |

- `--base-url` is used exactly as given, so include `/v1`, as in
  `http://127.0.0.1:11434/v1`.
- `--base-url` sends no API key, and it does not read the config file, so
  `extra_body` and `request_timeout_s` do not apply.
- Without `--base-url`, `--chat-model` replaces the configured chat model for
  this run; the config file is not changed. With `--serve`, or with no task
  (`melo` alone), both are ignored.

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
  exactly as `melo` without a terminal does.
- `--routine`: without it, a task is known by its wording (dates and times
  masked) and the folder it runs in. Runs given the same `NAME` are one
  routine, so `--routine daily-mail` joins differently worded requests run
  from any folder. A blank name is a usage error. See
  [Procedural memory](procedures.md#how-it-works).

### Gateway and setup

| Option | Default | Description |
|---|---|---|
| `--serve` | off | Start the gateway: the HTTP API and the web workspace. |
| `--host HOST` | `127.0.0.1` | The address the gateway listens on. |
| `--port PORT` | `8401` | The port the gateway listens on. |
| `--reconfigure` | off | Run the setup wizard, then start the gateway. |
| `--daemon install\|uninstall\|status` | not set | Manage the macOS background service; `install` also takes `--port`. |

- The defaults of `--host` and `--port` come from the `[gateway]` table of
  the config file. The options change them for this run only.
- `--host`, `--port` and `--reconfigure` are ignored when you give a task.
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
- a direct-mode option with `--serve` or without a task (`melo ""`
  included), reported as
  `only for direct mode (a query without --serve): --max-steps`;
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
- **Past runs need persistence.** Each `melo` is a new process, and
  procedural memory stays in RAM unless memory is kept between runs. Run
  with `LOCALMELO_PERSIST_MEMORY=1` or `--memory-dir DIR` (see
  [Memory](memory.md#keep-memory-between-runs)); otherwise every run shows
  facts only.
- **Narrow terminals.** When the line is wider than the terminal, it leaves
  out, in this order, the past runs' times, the test counts, the past
  runs' named steps and the attempt, and then cuts the rest with `…`.
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
| 1 | The run ended as `crashed`; or `melo` stopped before the run because no backend is configured or the config file is not valid. |
| 2 | The run ended with any other status; or a usage error. |
| 3 | The gateway could not start, for example because its port is in use. |
| 129 | SIGHUP ended the run. |
| 130 | Ctrl+C (SIGINT) ended the run. |
| 143 | SIGTERM ended the run. |

129, 130 and 143 are what a shell reports for a process ended by that
signal. In gateway mode, cancelling the setup wizard exits with 0.

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
