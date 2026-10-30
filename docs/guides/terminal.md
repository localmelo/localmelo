# Guide: Terminal

This guide explains the interactive terminal: how to chat, use its keys and commands, answer approvals and work in a workspace. Read it when you run `melo` without a task; the command-line options are in the [command-line reference](../reference/cli.md).

## The interactive UI

`melo` without a task, in a terminal, opens a chat that keeps a quiet record
of its work and makes the answer the loudest thing on screen: one line for
each thought, one row for each tool call with the first line of its output,
notices and reflections, and a live area that says what melo is doing, for
how long, and which tool runs. The answer prints as Markdown. What it prints
stays in the terminal's scrollback; the input box and a status bar stay at
the bottom. Developer mode keeps the whole record (see
[A turn's output](#a-turns-output)).

```text
melo 0.1.0 · Qwen3-8B (/model to change)
~/demo-project
Try: What is this project? · / commands · ctrl+d quit

› What is in this folder?

✓ shell_exec  ls -1 · 0.1s · allowed
  └ app.py · +3 lines (/expand 1.1)
● The folder holds app.py, notes.txt, tests and todo.md.
✻ Worked for 6.2s · 1 tool call · ↓ 412 tokens

────────────────────────────────────────────────────────────────────────────
› Ask melo · / for commands
────────────────────────────────────────────────────────────────────────────
  Qwen3-8B · ~/demo-project · context 2.1k tokens · What is in this folder?
```

The welcome is three rows: the version and the model, with how to change it;
the folder melo works in; and one thing to try, with the keys for commands and
for quitting. It shows the model's server in a row of its own only when the
model is not the saved default. Warnings follow it, and `/status` checks that
the model answers and shows the folder, session, approvals and context use. In
developer mode (`--developer`, or `[ui] mode = "developer"`) the welcome is a
box of every row.

It needs a terminal: stdin and stdout must both be terminals and `TERM`
must not be `dumb`. Otherwise `melo` prints this on stderr and exits with
code 2:

```text
melo: the interactive UI needs a terminal (stdin and stdout must be TTYs).
  Run one task:      melo "your task"
  Start the gateway: melo serve
```

### Which model server

The first of these that applies:

1. `melo setup`: the setup wizard, then the backend it saved. In default
   mode it first offers the model server it finds on this computer (`Found a
   model server at …`, `Use this model server?`) with the first run's questions; the
   full backend menu follows only after `Use another server`, in developer mode, or when none is
   found while a backend is saved (with nothing saved, it prints the commands
   that start a server). Without a task it cannot be combined with `--url`.
2. `--url URL`: that server, without the config file. The model is
   `--model`, else the only chat model that `GET URL/models` lists
   (ids containing `embed` or `rerank` are not chat models), else the one
   chat model the server says is loaded; with several and none known to be
   loaded it exits 2, lists them and asks for `--model <id>`. A task,
   `melo worker` and `melo ""` choose the model the same way before their first request
   (see the [command-line reference](../reference/cli.md#general)). A URL without `/v1` is listed at `URL/v1/models`, and when that
   answers the UI uses `URL/v1` too. When the server does not answer the
   list, or lists no model, the model is `default` and a warning under the
   banner says why. With no config file it asks once, as a
   [list](#questions-are-lists), `Save URL · MODEL as the default backend?`:
   `Save it` is selected, so Enter saves it, as the first run does; `Not now`
   or Esc saves nothing.
3. The saved backend, with `--model` in place of its model for this
   session. Nothing is checked before the UI opens: after the banner, melo
   lists the server's models once (`GET /models`) and warns when the server
   does not answer, or does not list the model:
   `⚠ Cannot reach the model server at URL (connection refused). Start it, then send your message.`
4. No saved backend: the [first-run](#first-run) questions, then the UI.

The welcome's first row shows the package version, and on a git checkout the
short commit (`melo 0.1.0 (baab21e)`); `melo --version` prints the same line.
When the model is not the saved default, a row after the folder gives the
server's URL and where it comes from (`from --url (not saved)`,
`not saved; melo asks again next time`) and, when melo took the one model the
server says is loaded out of several,
`the loaded one of N; /model lists them`. A server that does not answer gets
the warning below the welcome instead. `/status` calls the server `oMLX` when the config file says `omlx` or
a model listing of the server did, and otherwise `OpenAI-compatible server`
(Ollama, MLC LLM and SGLang keep their names). Developer mode's box has a
model row with that name; a saved server is listed after the box, so the box
shows what the file says, unless the run already listed the server (the first
run does).

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
| Ctrl+O | List the turn's steps (during a turn, or the last one after it); Esc closes the list. |
| Ctrl+T | List the tasks beside the chat (the gateway's); Enter on one lists its actions. |
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
  [Procedural memory](../concepts/procedures.md)); Esc then does nothing, and the next
  message waits for it.
- **History.** Messages and commands are kept in
  `~/.cache/localmelo/cli_history` (mode 0600, the newest 1,000 once it
  passes 1 MB), never the files' text. When that file cannot be opened,
  the session keeps its history in memory and says so.

### Commands

Type `/` and the commands show under the input box with what each does;
type more letters to narrow them, Tab or Enter to take one. The everyday
commands come first, and are the menu's first screen: `/model`, `/new`,
`/status`, `/help`, `/exit`, `/undo`, `/copy` and `/retry`. Each line says what
the command does and, where it needs one, its argument (`/attach PATH`).

| Command | What it does |
|---|---|
| `/help [command]` | The everyday commands in three groups (Conversation, Model, Help), the `!` line and how to see more. `/help all`, and `/help` in developer mode, list every command and the keys; `/help NAME` gives one command's details. A row longer than the terminal continues under its description. |
| `/attach PATH` | Stage one workspace text file; quote a path containing spaces. |
| `/attachments` | List staged files and their numbers. |
| `/detach N\|all` | Remove the numbered staged file or all staged files. |
| `/new` (also `/clear`) | Start a new chat: the model forgets the conversation, and this chat's grants end. A `── New chat ──` rule marks the start. |
| `/sessions [filter]` | Pick a saved chat in this folder; type to filter, Tab for all folders, ↑ ↓ to select, Enter to resume, Esc to close. |
| `/resume [id\|title]` | Resume by ID, ID prefix (at least four characters) or exact title; no argument opens the picker. |
| `/title [text]` | Show or set a 1–200-character title. Before the first message it is kept until the chat is saved. |
| `/undo` | Forget the newest finished turn and put its typed message in ↑ history. Files the tools changed stay changed. |
| `/usage` | Recorded steps, tool calls, tokens and time per turn, including undone work, with totals. `?` means not recorded. |
| `/memory` | Inspect the model's copied working window, its counts, the long-term backend and any restore limitation; no model call or memory write. |
| `/copy [turn]` | Copy a stored answer's Markdown, not the wrapped display. Without a number, use the latest finished, non-undone turn. |
| `/verbose [on\|off]` | Verbose rendering: 40 thinking rows, a tool call's arguments, 20 output lines, the reflection's summary and whole rationale; without an argument, flips it. |
| `/thinking [on\|off]` | The thinking excerpt on, or off (the default, except in developer mode), leaving only `Thought for 1.2s`; without an argument, flips it. |
| `/stream [on\|off]` | Live streamed replies for later turns on or off; without an argument, flips it. See [Streamed replies](#streamed-replies). |
| `/expand [turn.call]` | A tool call's whole output or error, including stored calls after resuming: the latest call, call N of the latest turn, or turn T's call C (≤ 200 lines, then the rest's count). |
| `/progress` | The latest progress facts of this turn: level, step, attempt, phase and cycle, tests, the past runs, every named step and the memory it rests on. |
| `/approvals [ask\|auto\|deny\|forget [tool]]` | The approval mode and this session's grants; switch it, or forget one tool's grant (all with `forget`). Under `--yolo`, says how to turn prompts on. |
| `/status [all]` | Five rows that say whether melo is ready: the model and its server with `✓ answers` (or `✗ does not answer (connection refused). Start it, then /status checks again.`), the folder, the session, what the approvals do in words, and how full the context is (`8% used (2.6k of 32k tokens)`, or `2.6k tokens used` when no limit is known). `/status all`, and `/status` in developer mode, print every row instead: the model and server with `reachable` or why not, the workspace and its fence, the session with its id, the approval mode and grants, tools, memory (with the window and why it has that size, `64 (local server)`), limits, a usage block (cache, context, steps, tasks: a row appears once it has data; developer mode shows every row), files and version. A row longer than the terminal continues under its text. Any other argument says `✗ Use /status or /status all.` |
| `/mcp` | The MCP servers, their tools and command lines, and the file they came from (`--mcp` or `mcp_config`); why MCP is off after a failure; how to name a file when there is none. |
| `/retry` | Re-run the newest finished turn, keeping its stored typed text even after a resume. Undo it first only if it is not already undone; earlier turns and files stay changed. |
| `/model [name\|number]` | Alone: a list of the server's chat models (an embedding or reranking model is left out), the current one marked `✓` and the ones in memory `(loaded)` where the server says. Type to filter, ↑ ↓ to move, Enter to switch and save the model as the default, Esc to keep the current one. With a listed id, or a number from the server's listing (melo lists the models first if it has not yet): switch to it from the next message, and save it as the default. With no list to pick from (a cloud API), `/model` says so and `/model NAME` switches. |
| `/backend [url]` | Switch to another model server: asks for its URL (Enter keeps the current one) and a model from its list, checks the server with `GET /models`, switches and saves both as the default (as `ollama` when the server answers like Ollama, `omlx` when its model listing names oMLX, else `vllm`). |
| `/exit` (also `/quit`) | Quit; a running turn is stopped first. |

- A command that runs is echoed in the scrollback as a message is
  (`› /model 2`), followed by its output. A refused one is not, and
  `/exit` prints nothing.
- `/new`, `/sessions`, `/resume`, `/undo`, `/retry`, `/model` and `/backend` are refused while a turn runs: the
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
  `/backend` saves the server as the `ollama` backend when it answers like
  Ollama (`GET /api/version`), as `omlx` when its model listing names oMLX,
  else as `vllm`; it writes `chat_backend` and
  that table's `chat_url` and `chat_model`, and keeps the rest of the
  config: embedding, `request_timeout_s` and `extra_body`; the session runs
  on the config as saved. In a `melo --url URL` session they switch for
  this session only, unless melo asked `Save URL · MODEL as the default
  backend?` (only when there is no config file yet) and you answered yes.
  After `s` (start without saving) at the first run's questions, `/model` and
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

### Attach text files

In an idle interactive chat, stage a file before sending your message:

```text
/attach "notes/my file.txt"
/attachments
/detach 1
/detach all
```

Alternatively, type `@` and a workspace-relative prefix in a message and
explicitly select a file from the completion menu. Merely typing `@file` in
prose does not attach it. The staged-file line identifies the files; send a
non-empty message to include them. Attachment commands cannot run during a
turn, and staged files must be detached before switching sessions.

Only regular UTF-8 text files within the current workspace are accepted;
absolute paths inside it work too. URLs, `..` traversal, symlinks (including
ones pointing inside the workspace), recognized secret files and binary data
are refused. Names after sanitization must fit 200 characters. Hardlinks can
refer to an inode outside the workspace; this is not a sandbox. Image attachments
are not supported. Limits are 8 files, 256 KiB per file and 512 KiB total,
measured in original bytes. The displayed byte count is the snapshot's UTF-8
size after cleaning and redaction. The composed message is also subject to the
agent's prompt limit, so files within these limits can still be rejected.

Each file is read once into an immutable snapshot; changing or deleting the
file afterwards does not update it. The model receives the snapshot in a
delimited block marked as untrusted data, not an instruction. Secret
redaction is best effort, not a guarantee: inspect files before attaching.
If redaction or cleaning changes the content or path, melo displays the
snapshot and asks for confirmation before staging it.

Submitted snapshots are saved with the session in
`~/.cache/localmelo/cli/sessions.db` (plain text, not encrypted); `/retry`
uses the saved snapshot rather than reopening the file. Persistence does not
mean full replay: `pairs` resume has a 32,000-character budget for user text
and answers and can omit an entire attachment turn; `window` resume may clip
or omit it when the retained checkpoint or tail does not contain it fully. Reattach
and send again if the model needs content that was not restored. Attachments
are not supported in the plain `melo ""` loop or direct mode.

### Run a shell command

At an idle prompt, a message that starts with `! ` runs the rest as a command in
your own shell (`$SHELL`, else `/bin/sh`), as you, in the workspace folder:

```text
! echo from-the-child; exit 3
```

```text
! user shell · ~/demo-project
$ echo from-the-child; exit 3
from-the-child
  └ exit 3 · 0.0s
  └ /attach ! shows this output (redacted, 15 bytes) and asks before attaching it
```

- **Not a sandbox.** The command is not a model tool: no approval, checker or
  workspace policy applies, it has your account's full permissions and it can
  leave the workspace. melo only gives it the terminal.
- **The terminal is the command's** while it runs: programs that read input,
  draw a screen or follow a resize work. Ctrl+C and Esc go to the command and
  never cancel the agent; afterwards the usual Ctrl+C, draft and queue rules
  apply. The block ends with the exit code (`exit 3`) or the signal
  (`killed by SIGTERM`) and the elapsed time. The cursor, colours, terminal
  modes and the prompt are restored once, however the command ends.
- **Output stays in the terminal.** It never reaches the model, the
  conversation or the events, and the command is not saved in the input
  history. `/attach !` attaches a redacted snapshot of the last output (its
  last 256 KiB, without text a terminal never showed: concealed text and
  anything before a screen clear; a cut start drops its partial first line) to
  your next message. It always prints the exact snapshot and asks
  `Attach this output?` (`Attach this redacted snapshot?` when something was
  redacted) as a [list](#questions-are-lists) that starts on `Don't attach
  it`, even when nothing was redacted; nothing is attached unless you choose
  `Attach it` (Esc does not attach). Once attached, the snapshot is stored with the session like any
  [`/attach PATH`](#attach-text-files) file, unencrypted.
- **Refused when busy.** During a turn, an approval, a session switch or another
  command, `! ...` is refused (`! runs only when melo is idle`) and the draft
  is kept.
- `!` alone prints this help. To send a message that starts with `! `, write
  `\! text`: the backslash is dropped.
- When melo exits, the command's process group gets SIGTERM; when the terminal
  closes, SIGHUP; then SIGKILL after 2 seconds. Only that group is signalled; a
  program that detaches itself from it (`setsid`, `nohup`) keeps running.
- Not available in direct mode (`melo "task"`) or the plain loop.

### Inspect, undo or copy a conversation

`/title` can name a chat before any message is sent; it does not create a
saved session on its own. The chosen title is applied when the first turn
creates that session. `/new` or a successful resume clears an unsaved title.
Titles may be changed during a turn, but not while switching sessions.

`/undo` removes the newest finished turn from the model's restored working
window and keeps the original typed message in input history (↑). It does
not reverse shell commands, Python code or file edits. In another window,
a running turn blocks undo; the command stays in the draft. Once undo is
committed, Esc does not reverse it: the operation settles and prints its
`Undid …` line, and the busy hint asks you to wait rather than claiming
Esc cancels it. The preview uses only the first typed line, clipped by
terminal cells, without adding a second period.

`/retry` re-runs the newest finished turn. If `/undo` already removed it,
retry leaves the previous live turn alone; otherwise it undoes the target
first. It sends the full stored message while retaining its typed text.

`/usage` includes undone rows because that work was performed. A running
turn or missing counter is `?`, not zero; a partial total ends in `+?`.
Below 72 columns, usage uses labeled, wrapped rows instead of clipping
numeric table cells.
`/memory` reads a copy of the window, so it and `/usage` may run during a
turn. Memory rows show user/assistant/tool messages, not hidden thinking;
the footer explains whether a resume restored pairs or a checkpoint with
later pairs. It does not read all persistent memories or ask the model to
summarize them.

`/copy` reads the **stored, redacted answer** after removing thinking, so
Markdown and code are not copied from the terminal's wrapped rendering;
leading code indentation and trailing newlines are preserved. An explicit
turn number reads just that turn; default selection reads backward in
bounded pages. An explicit turn number may select an undone answer. Empty answers,
running turns and status-only labels cannot be copied. On macOS outside
SSH it uses `/usr/bin/pbcopy` with a two-second timeout; under SSH, on other
platforms or if that utility fails it sends a base64-encoded OSC 52 request.
The latter says `Sent the answer …` rather than claiming success: the
terminal must support and allow clipboard requests. No clipboard operation
happens without the explicit `/copy` command.

### A turn's output

- **Approvals show what changes.** A `file_write` over an existing file shows a diff (removed
  lines in red, added in green, with the counts), and a write too large or not text says why;
  `/approvals` switches between `ask`, `auto` and `deny`, and the session's grants.
- **The live area.** The first row names what melo does (`Starting`,
  `Thinking`, `Replying`, `Running`, `Waiting for your approval`,
  `Retrying in 4s`, `Reflecting`, `Wrapping up`, `Stopping`), then the time;
  the output tokens and `esc to stop` follow (`Thinking… · 14s · ↓ 180 tokens
  · esc to stop`). Developer mode puts the run's facts before the time: the
  step (`step 4`, never against `--steps`), the attempt from the second
  one, the phase with its cycle and the tests seen. When the line is narrow
  the hint goes first, then the tokens, and in developer mode the facts are
  cut with `…`. The second row
  is the running tool, its own clock, and for `shell_exec` and
  `python_exec` their limit: `└ shell_exec  pytest -q tests · 0:25 of
  10:00`. For a task melo has run before, a third row shows the past runs the
  progress line would, with the steps they took: `last time 6 steps, 2m10s:
  Run ls ✓ · Read app.py ✓`. It shows on a session's first turn, and in
  developer mode on every turn.
- **Thinking.** `✻ Thought for 1.2s`: one line. `/thinking on` and developer
  mode add the first 3 rows of the thinking, `/verbose` 40, wrapped to the
  width. An intermediate message prints as Markdown after `●`; the
  last one is held, prints at the next step, and at the end only when it
  differs from the final answer.
- **Tool calls** print one block each when they finish: `✓` ran, `✗`
  failed, `⊘` blocked or denied, `◌` stopped or not run; then the tool's kind
  (`→` read, `±` write, `$` run, `@` network) and name; the duration;
  `exit N` when a command ran and exited with another code than 0, shown
  with `✗` in red although the call ran (`grep` exits 1 without failing);
  how its approval ended; and the first line of its output with
  `+N lines (/expand T.C)` on the same row (a second line prints instead of a
  count of one). The mark carries the outcome's color, the kind's glyph and
  the name carry the kind's (a read is muted, so reads fade), and the
  duration and tags are dim; see [Terminal colours](#terminal-colours). A
  call that failed, or that ran and exited with another code than 0, shows
  up to 3 lines, the first in red;
  developer mode shows 3 lines for every call and `/verbose` 20, with the
  call's arguments. Images print as `▪ image/png · 312 KB · not shown`. Escape
  sequences in the output are removed, so a tool cannot move the cursor,
  set the title or write the clipboard.
- **Notices and reflections.** `↻` marks a retry: one line a turn, which for
  a server that cannot be reached keeps its URL
  (`↻ Could not reach URL; trying again in 1 s`), with its delay counting
  down in the live area; `·` an empty reply; `⚠` a stuck or spent-budget
  notice. A reflection prints as `✻ Reflected after attempt 1: continue ·
  Read the test before editing. (2.3s)`; `/verbose` adds its summary and
  whole rationale, and `asked twice` when it was; an unreadable one says
  whether the run continues.
- **The answer** prints as Markdown after `●`. A run that did not complete
  prints its best effort, then what happened and what to do, in plain words:
  `✗ Could not reach the model server at URL.` and, under it,
  `Start it, then /retry.` The first line is the status's title from the
  shared table (`Ran out of steps (30).`,
  `Stopped: the same calls kept repeating.`) with the reason that helps (the
  URL, the model, the error of a failed check), and the second is the action,
  in the default color. The footer then reads `✻ Stopped after 14s` instead of
  `Worked for`, without a count of zero tool calls. Direct mode keeps its
  `melo: run ended with status …` line on stderr.
- **Changed files.** A turn that wrote files with `file_write` ends with one more
  quiet line under its footer, so you see what it touched without reading every
  block:

  ```text
  ✓ ± file_write  hello.py (15 chars) · 0.0s · allowed
    └ new file, 1 line
  ● I saved hello.py. Run it with python hello.py.
  ✻ Worked for 4.7s · 1 tool call · ↓ 50 tokens
    Changed + hello.py (+1)
  ```

  Several files read `Changed + hello.py (+1), ~ README.md (+3 −1)`: `+` a new file and
  `~` a modified one, paths relative to the
  folder melo works in, each with the lines the turn added and removed in it. At most
  three names show, then `and 2 more`; a narrow terminal shows fewer and never wraps. A
  file written twice counts once, the file as the turn found it against the file as it
  left it, and a file written back as it was is left out. A name has no counts when
  there is nothing to count or compare: a new empty file, a file melo could not compare
  (too large, or not text) and every file of a `--yolo` turn, where nothing read the old
  file first; the last two have no `+` or `~` either, since melo does not know whether
  the file was new. The line comes from the turn's own `file_write` calls: a
  write that was denied or refused (a path outside the folder) is not listed, and
  neither is a file that a shell command or a script changed, because melo never scans
  the folder. Nothing prints when the turn changed no file, or when you stop it (a
  stopped turn has no footer). Developer mode shows the same line.

### The status bar

The bar under the input box answers "how much has this used" and "what runs
beside it". The model and the folder are in the welcome and `/status`, not
here. Its zones keep their place, with a rule between them: usage, the work
beside the chat, and the run (the chat's title when no turn runs). On a local
server:

```text
  ↑ 3.0k ↓ 15 · ctx 2.6k/32k ▮▯▯▯▯ · cache 61% (avg 72%) · idle 2m │ Fix the parser
```

- `↑ 3.0k ↓ 15`: the tokens this session sent and received.
- `ctx 2.6k/32k ▮▯▯▯▯`: the last measured prompt over the input limit, with a
  meter in fifths. It turns amber from 70% and red from 90%. Without a limit it
  is the tokens alone (`ctx 2.6k`), with no meter.
- `cache 61% (avg 72%)`: the share of the last request's prompt the server
  reported as cached, then the session's average once two requests reported
  one. It is teal from 50%, and amber under 20% after the first request (the
  first is always cold). A server that reports no cached tokens shows no cache
  figure, never 0%.
- `idle 2m`: the time since the last request, from one minute on. A local
  server evicts its prefix cache by memory, not by time, so there is no
  countdown.

On a cloud API the usage zone says what the session cost instead, from the
prices you configured (see [`[pricing]`](../reference/configuration.md#pricing)),
and the tightest rate limit the last response reported:

```text
  $0.13 · ctx 100k · limit 42% ▮▮▯▯▯ ↻3m │ Fix the parser
```

`limit 42%` is the share of the window used, from the response's
`x-ratelimit-*` or `anthropic-ratelimit-*` headers (no extra request), and
`↻3m` is the time until it resets. A model with no configured price shows its
tokens (`↑ 100k ↓ 5`), and a server that sends no limit headers shows no limit.
melo bundles no prices.

- **The run.** While a turn runs, the run zone gives its steps in brief and the
  key that lists them:

  ```text
    ↑ 1.0k ↓ 5 · ctx 1.0k · cache 83% │ 2/~4 · ctrl+o steps
  ```

  `2/~4` is step 2 of about 4: the task was done before and took 4 steps then,
  so the total is an estimate and marked `~`. A task melo has not done before
  shows `step 2`, and so does a run that has gone past the estimate. (`2/8`
  without `~` is kept for a turn that brings its own plan; none does yet.)
  Ctrl+O opens the steps as a list: each tool call with its tool, target,
  status and time, then the steps the last run of the task still had after
  this point (`◌ Run the tests · to come`); Esc closes it. After the turn the
  same key lists the last turn's steps.
- **The work beside the chat.** When the gateway (`melo serve`) has tasks that
  have not finished, a zone counts them by state, with the top priority of
  those not paused and the key that lists them:

  ```text
    … │ 2 running · 1 queued · P1 · ctrl+t tasks │ Fix the parser
  ```

  `running` is moss and `held` (a task that waits for you, such as an
  approval) amber; `queued` and `paused` are muted. Ctrl+T lists the tasks
  (`P2  running  Bench the cache`); Enter on one lists what its state allows:
  Pause, or Resume for a paused one, Raise to and Lower to the next priority
  (P0 is the highest; lowering stops at P4, since P5 is paused), and Stop,
  which asks `Stop it` or `Keep it` first. Each goes to the gateway exactly as
  `melo task pause|resume|level|stop` does, and a hint says what happened
  (`Paused: Bench the cache`) or why the gateway refused. melo asks the
  gateway for its tasks every 5 seconds while the terminal is open, and only
  when a gateway token exists (`~/.cache/localmelo/gateway/auth`); without a
  gateway there is no zone, and Ctrl+T says how to add a task.

On a narrow terminal the title goes first, then the work's details (the
priority, then the key), then the usage's details (right to left: idle time,
average, meter), then a running turn's key; then the figures go right to left,
and a running turn's figure stays. The bar never wraps. The mode shows at the
right edge only when it is not `ask` (`auto`, `deny`, or the
`⚠ YOLO: no approvals` warning); developer mode always shows it. While the
completion menu or a list is open, a blank row sets the bar apart from it.

The figures come from events melo has already received; nothing asks the
model server. The bar redraws when an event arrives or you press a key, and
once a second while it shows a clock (the idle time, a limit's reset); nothing
else redraws it while idle. With `NO_COLOR` or `LOCALMELO_ASCII=1` the zones
are still apart (`│` or `|`), and ASCII says `in 3.0k out 15`, `##---` and
`reset 3m`. `/status` prints the figures in full. `[cli] status_line = "off"`
keeps only the run zone, the title and the last request's context
(`ctx 1.0k`). There is no command-line flag for it.

### Streamed replies

The interactive UI streams a model's reply as it is written, and does so by
default: it asks the provider for a streamed planning reply and draws only text
the release filter has already redacted, in the live area, at the app's redraw
rate; it never writes raw deltas to the terminal. `/stream off` turns it off
for later turns, `melo -X cli.stream=false` for one run, and
`melo config set cli.stream false` for good.

A model server that does not stream needs no setting. When it answers a
streamed request with one ordinary JSON body, or closes the stream before it
sends anything, melo uses that reply (or asks the same question once more,
without streaming) and does not ask that server to stream again until melo
quits. The reply then shows when it is complete, as it does with `/stream off`,
and melo says so once: `This server sends replies whole; melo shows them when
they are complete.` A stream that breaks after part of a reply has arrived is
still an error: `Could not get an answer from the model server: Streaming
response ended before completion`.

Whether a reply shows as it is written depends on how its model frames
reasoning, which a per-model hint in `config.toml` declares:

```toml
[cli.reasoning_tags]
"Qwen3-8B" = "closing_only"
```

The hint is `explicit` (a leading `<think>` starts reasoning), `closing_only`
(everything before `</think>` is reasoning) or `none` (tags are literal text).
A model with no hint keeps the conservative rule: content is held until the
reply ends, so the live area shows `✻ thinking…` rather than text. In
developer mode, after its first whole reply melo says so once: `This model's
replies appear when complete until its reasoning format is set (see Streamed
replies).` A new user never sees that note. A server
that sends reasoning in a field of its own streams without a hint, and that
field is never drawn as content; `/thinking on` shows its excerpt. The same
hint splits the completed reply, so the stream and the stored answer do not
disagree.

Streamed text is provisional. If the reply becomes a tool call, a retry,
a run that fails, or a stop, melo closes the live text and prints it once,
and in developer mode labels it `↳ partial reply (tool call)`, `(retrying)`,
`(not the final answer)` or `(stopped)`; the completed final answer prints once
after `●`. A provider
without streaming sends the complete reply, which prints as usual.
`/stream on|off` flips the setting for later turns without saving it, and
`/new` puts it back to `[cli] stream`. Runs without the interactive UI
(direct mode, workers, the gateway) never stream, so their output and request
bodies are unchanged. An invalid `[cli]` value never fails a run: the
interactive UI warns and uses the defaults, and direct mode and the gateway do
not read it.

### Approvals

`shell_exec`, `python_exec` and `file_write` run only once you answer, and
so do MCP tools marked for approval. `file_read` and `list_files` only read,
so they never ask: melo can look at your folder to answer "What is this
project?" without a panel. (The interactive UI loads the MCP
servers of `--mcp`, else of the top-level `mcp_config` key; a server
that fails to start turns MCP off for the session and `/mcp` says why.) The panel takes the place
of the input box, and your draft stays:

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
- The header names the tool and the folder. Developer mode adds the tool's
  permission tier after its name, and after the path for `file_write`
  (`shell_exec · exec · in ~/demo-project`; `write`, `exec` or `network` for an
  MCP tool, once the UI can load one). A read-only MCP tool would show `read`
  only when the server's `require_approval` forces it to ask; see
  [Permission tiers](tools-mcp.md#permission-tiers).
- Esc or Ctrl+C at the choices stops the turn, and a second Ctrl+C within
  2 seconds quits; in the reason field, Esc goes back to the choices.
- The command, the code or the file's new content shows exactly as it will
  run, its long lines wrapped, never cut: control characters and the
  characters that reorder or hide text show escaped (`\r`, `\x1b`,
  `\u{202e}`), with a warning line, so nothing can hide in it.
- The panel shows at most 8 lines of the body, fewer on a short terminal,
  then `… +N lines (v views all)`; `v` prints the body in full above it.
- `--yolo` runs every tool without asking (it means no policy, not
  read-only), and the status bar says `⚠ YOLO: no approvals`.

### The workspace

melo works in the current folder, or in `--workspace DIR`. File tools
(`file_read`, `file_write`, `list_files`) stay in this folder. Commands and Python can
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

### Terminal colours

Colour sorts what a turn prints, so the answer, a failure and the files that
changed stand out while reads fade. Every colour has a glyph or a place that
says the same without it, so `NO_COLOR` and `LOCALMELO_ASCII=1` lose nothing.

| What | Glyph (ASCII) | Colour |
|---|---|---|
| Your message | `›` (`>`) | moss mark, on a raised band across the row |
| The answer, its headings and bullets | `●` (`*`) | Melo's orange (navy headings on a light terminal), the only orange |
| A message that is not the answer; thinking | `●` (`*`); `✻` (`*`) | muted |
| A tool that reads | `→` (`>`) | muted, so reads fade |
| A tool that writes | `±` (`~`) | violet |
| A tool that runs a command or code | `$` | blue |
| A tool that reaches the network | `@` | indigo |
| Ran; failed or exited with another code than 0; refused; not run | `✓` (`+`); `✗` (`x`); `⊘` (`!`); `◌` (`-`) | teal; red, with `exit N`; amber; the whole row dim |
| A changed file: new; modified | `+`; `~` | teal; amber |
| Durations, counts, hints | `·` (`-`) | muted |

Inline code is set on a raised chip and a code block on a raised panel, its
keywords moss, strings gold and names navy.

Panels, lists and bars use the same colours:

| What | Look |
|---|---|
| The approval panel | an amber frame with its title bold; the chosen answer `❯ 1. Yes` on a moss band; a diff's added rows on a teal band and removed rows on a red one |
| Every list (`/model`, `/sessions`, the setup questions, the attach question, Ctrl+O's steps) | the title bold, the chosen row on a moss band with one `❯`, the current model's `✓` teal |
| A line of keys (`↑↓ select · enter confirm · esc stop the turn`) | the keys in your text colour, their words muted (plain at 16 colours, never dim) |
| The live row | the spinner moss, what melo does in your text colour, the time and tokens muted; while an approval waits, an amber `⚠` and bold words |
| The status bar | muted, with the cost in your text colour, a good cache teal, a filling context or limit amber then red, running work moss, held work amber, and a rule between the zones |
| A command in advice (`Start it, then /retry.`) | moss, bold |

Without the exact background the bands go and the chosen row is moss and bold;
with `NO_COLOR` it is in reverse video.

At 256 colours or more, in a Unicode terminal of 80 columns or more, the
welcome's rows stand beside Melo's head, five rows of half blocks with its two
upright ears, eyes, nose and cream muzzle, and `melo` is orange (the README's
screenshot shows it). The head is computed from the pet's face picture
(`macos/art/akita/face/idle@2x.png`), not drawn by hand;
`scripts/make_terminal_mark.py` makes it again when the picture changes. Under
80 columns, at 16 colours, with `NO_COLOR` or with `LOCALMELO_ASCII=1`, the
welcome is its three rows alone: fewer rows of half blocks lose the ears.

How melo chooses:

- **How many colours.** `NO_COLOR` (any value but empty) or `TERM=dumb`:
  none, only bold, dim and underline, with every glyph kept. Otherwise the
  `[cli] colors` setting, then `COLORTERM=truecolor` (or `24bit`) for 24-bit
  colour, then a `TERM` that ends in `256color` for 256 colours, else the 16
  named colours of your terminal's theme.
- **The background.** The `[cli] background` setting, then the terminal's own
  answer: when the interactive UI starts, it asks the terminal for its
  background and text colours (OSC 11 and OSC 10), waiting at most 100 ms
  (never in direct mode, with `--json`, or without a terminal), then
  `COLORFGBG`. With the exact colours, melo draws the band, the chips and the
  panel, each only where your text keeps a contrast of 4.5:1 on it, and moves
  a colour drawn on them until it reads. With only dark or light known, the
  colours stay and the fills go: inline code is bold and a code block has a
  `│` gutter. With nothing known, melo uses the 16 named colours.
- **When colours look wrong.** Some terminals, and tmux, screen or ssh, answer
  nothing. Say which background yours has: `-X cli.background=light` (or
  `dark`) for one run, or `melo config set cli.background light` to keep it.
  `-X cli.colors=16`, `256` or `truecolor` sets the number of colours.

### Terminal settings

- Colours: see [Terminal colours](#terminal-colours); set `NO_COLOR` to a
  non-empty value to turn them off.
  A sentence you must read or act on (a warning, the approval's warning and
  keys, the hint under the input box) is never muted; muted is for durations,
  counts and `(/expand T.C)`.
- On a terminal whose encoding is not UTF-8, or with `LOCALMELO_ASCII=1`,
  melo draws ASCII stand-ins for its symbols (`>` for `›`, `...` for `…`).
- melo rings the terminal's bell when an approval panel opens and when a
  turn of 30 seconds or more ends; `LOCALMELO_BELL=0` turns it off.

### Sessions and files

Bare `melo` starts a new chat; only the first message saves it. The status
bar names it `New chat` until it has a title.
Use `melo -c` (`--continue`) to resume the newest chat with a finished turn
in the current workspace, or `melo -r` (`--resume`) to pick one. With none,
either starts a new chat and says so; Esc in the startup picker also
starts new, without creating an empty row. `melo -r REF` resolves an ID,
a prefix of at least four characters, or an exact case-insensitive title;
multiple matches, or a title/message search fallback, open the picker.
An unknown reference exits 2. The startup choice, including the `-r` picker,
happens before the banner: the recap names a chosen chat (`◆ Resumed "TITLE"`,
and developer mode's box labels it `"TITLE" (resumed)`); Esc or an empty list
gives the new-chat banner. `-c` and `-r` cannot be
combined, or used with a task or `melo serve`.

The picker lists at most 50 chats with finished, non-undone turns, newest
first. Its header says what the keys do (`Resume a chat · ~/demo-project · tab:
this folder or all · esc close`); a folder too long for the row is cut from its
middle, so the keys stay. The filter updates after 150 ms; Tab toggles this
folder and all folders. Enter resumes, Esc or Ctrl+D closes it, and the
ordinary draft stays intact.
`/sessions FILTER` and `/resume REF` use the same picker and resolution.
Web and scheduler sessions are never offered. The web console's Workers view
shows your chats here read-only under **Terminal chats**, redacted as it reads
them (the stored copy is not redacted); it cannot change or send to them. A resume keeps the current
model and workspace, records them on the session and clears approval grants;
`/verbose`, `/thinking` and `/stream` preferences stay as they were. `/new`
resets those display preferences (`/stream` back to `[cli] stream`). A note says when the chat previously worked in another folder.

After the banner, the recap shows the last three eligible turns from stored
redacted copies, as plain text: your message after `›`, and each answer after
one `●`, with its Markdown markers off, its blank lines left out and at most
four rows, the rows under the mark indented. The line below its title says how
many turns are back in the model's memory (`1 turn is back in the model's
memory.`); developer mode adds what became of their tool calls. `[sessions]
resume_mode = "pairs"` (the default) restores recent user/answer pairs, without
tool steps; `"window"` restores a saved working-memory checkpoint plus later
pairs, falling back to pairs if no checkpoint is usable.
Set this in `config.toml`; an invalid value warns in the UI and uses pairs,
without adding warnings to direct-mode output. The recap's counts reflect
the restored window after eviction, not the total number of stored turns.

Two terminals may take turns on one CLI session. Before sending, melo reloads
changes from the other window; a busy or deleted session keeps the draft.
On exit, a chat with turns says how to come back to it:
`◆ Saved "What is this project?". Continue it with: melo -c` when it is the
newest chat of this folder and melo ran in that folder, and
`Resume it with: melo -r ID` otherwise.

Sessions are kept in `~/.cache/localmelo/cli/sessions.db`: every turn,
with your message, the answer and each tool call with its output.
`~/.cache/localmelo/cli_history` keeps your messages. Both are plain text (mode 0600), not encrypted. The UI writes
its log to `~/.cache/localmelo/logs/cli.log`
(folder 0700, files 0600, 1 MB, two backups) and never prints a traceback
on the screen; a problem shows as one dim line naming the log. A config
file that is not valid TOML, or a log folder that cannot be made, stops
`melo` with one `melo: …` line and exit code 1. Procedural
memory follows [memory persistence](memory.md#keep-memory-between-runs), as
for a task.

### First run

In a terminal with no config file, `melo` offers the model server it finds,
asks for a model, saves them, and opens the UI. When the server lists an embedding
model, it asks once more whether to use it. Later runs go straight in.

```text
$ melo
melo uses a model server on this computer (oMLX, Ollama, LM Studio…).
Found a model server at http://127.0.0.1:8000/v1 (1 model).
✓ http://127.0.0.1:8000/v1 lists 1 model; using Qwen3-8B-4bit.
Long-term memory is off; melo setup turns it on.
Saved to ~/.cache/localmelo/config.toml (melo setup changes it).
```

Between the second and third lines melo asks `Use this model server?` as a
list (see [Questions are lists](#questions-are-lists)); the list is erased once
you answer. With a server that lists an embedding model too, one more list
asks `Use nomic-embed-text for long-term memory?` (`Use it`, `Leave it off`).

#### Questions are lists

On a terminal (stdin and stdout, `TERM` not `dumb`) every question with a
fixed set of answers is a list, as the tool approval and `/model` are:

```text
 Use this model server?
 ❯ 1. Use it
   2. Use another server
 ↑↓ select · enter confirm · esc cancel
```

↑ ↓ move, Enter confirms the selected row, and a number key (1 to 9) chooses
that row at once. The safe or recommended answer is selected first, and Esc
gives the answer named at the end of the key line (`esc cancel` ends the
setup, as Ctrl+C does; `esc not now`, `esc leave it off` and `esc don't attach
it` decline). There is no filter field: the lists are short, and letters
typed in them do nothing. Ctrl+C prints `Setup cancelled.` and exits with 130,
Ctrl+D exits with 0, as at a typed question. The questions are:
`Use this model server?`; `Use <model> for long-term memory?`; `What now?` after
a URL that cannot be used (`Try again`, `Start without saving`, `Quit`; Esc
quits); `No model server found. What now?` (`Check again`, `Enter an address`
which then asks `Server URL […]:`, `Quit`; Esc quits); `Save URL · MODEL as
the default backend?`; the wizard's backend lists, `Save these settings?`
(`Save and continue`, `Cancel setup`) and `The check failed. What now?` (`Try
again`, `Save anyway`, `Cancel setup`, selected first, as `[N]` was); and, in
the UI, `Attach this output?`. Text questions (`Server URL`, `Model name`,
`Base URL`) stay text. Without a terminal (answers piped in, a script, a test)
no list is drawn and each question reads the typed answer it always did
(`Use it? [Y/n]:`, `Enter tries again · s starts without saving · q quits:`,
`[r]etry, [s]ave anyway, [N]o [N]:`, a number for a backend, Enter for the
default). A direct run (`melo "task"`, `--output-format json`) never asks a
question.

- melo looks for a server on this computer with `GET /v1/models` on
  `127.0.0.1` ports 8000 (vLLM, oMLX), 8400 (MLC), 11434 (Ollama) and 30000
  (SGLang), and offers the first that lists a model: Enter says yes to
  `Use this model server?` (Enter takes `Use it`; `Use another server` asks
  `Server URL […]:` with that URL, for another).
  The model list is requested; with several chat
  models, so are `GET /v1/models/status` (oMLX) and `GET /api/ps` (Ollama),
  which say which models are loaded. Saving the choice asks
  `GET /api/version` once to tell an Ollama server apart. No chat request
  is sent, so a server does not load a model for it. A server that
  lists a chat model comes first; one that lists only embedding or reranking
  models comes next. A server that answered without a list is named, as
  `A model server at URL answered, but HTTP 401: …`, and offered. When none
  answers, it prints the commands that start one (oMLX on a Mac, Ollama
  elsewhere, the ones in the [model servers guide](model-servers.md)) and
  waits:

  ```text
  No model server is running yet. To start one with oMLX:

    brew tap jundot/omlx https://github.com/jundot/omlx
    brew install jundot/omlx/omlx
    omlx serve --model-dir ~/models --port 8000

  Put a model in its own folder of ~/models first. The guide shows how:
  https://github.com/localmelo/localmelo/blob/main/docs/guides/model-servers.md
  ```

  A list follows, `No model server found. What now?`. `Check again` (selected)
  looks at the same ports again and, when a server answers, goes on as above
  (`✗ No model server found yet (looked at ports …).` when none does);
  `Enter an address` asks `Server URL [http://127.0.0.1:8000/v1]:` and checks
  what you type like a URL, below; `Quit` (or Esc) exits with code 0. With the
  answers piped in, the line is `Press Enter to check again, type an address,
  or q to quit:`.
  `LOCALMELO_DETECT_URLS` replaces the four URLs with a comma-separated
  list, which Enter looks at; an empty value turns the search off, and melo
  then asks `Server URL [http://127.0.0.1:8000/v1]:`.
- A server that lists several models is shown as a list to choose from. On a
  terminal (stdin and stdout, `TERM` not `dumb`) it is the same list as `/model`:
  type to filter, ↑ ↓ to move, Enter to choose, Esc or Ctrl+C to cancel the
  setup (`Setup cancelled.`). It holds the chat models only (embedding and
  reranking models are left out, unless the server lists nothing else). With
  several chat models, a loaded one is marked `(loaded)` and highlighted; when
  the server does not say which are loaded, nothing is highlighted, Enter does
  nothing until a key moves to a row, and a line first says that a model that
  is not loaded yet can take minutes and a lot of memory to load. Without a
  terminal (answers piped in) the models are a numbered list, chat models
  first, and the prompt is `Model [N]:`; type a number or an id. An embedding
  or reranking model is tagged `(embedding)` or `(reranking)` and is never the
  default. With several chat models, a loaded one is marked `(loaded)` and is
  the default; when the server does not say which are loaded, the prompt is
  `Model:` with no default and Enter alone asks again
  (`Type the number of the model to use. …`), and without a terminal an empty
  answer exits 2 with an error. `/backend` shows the same list (`Choose a model`; Esc changes nothing). A server that
  lists only embedding or reranking models is shown the same way, with a line
  saying that chat needs a chat model. The hints are wrapped to the terminal's
  width at word boundaries.
- An Ollama server, one that answers `GET /api/version` with
  `{"version": "…"}`, is saved as `chat_backend = "ollama"` with its
  `[ollama]` table's `chat_url` and `chat_model`; the banner then says
  Ollama, and chat requests use Ollama's native API with its 300 s timeout.
  An oMLX server (its model listing names oMLX as the owner of every model)
  is saved as `chat_backend = "omlx"`, and any other OpenAI-compatible server
  as `chat_backend = "vllm"` (the key for any OpenAI-compatible server); both
  use the `[vllm]` table's `chat_url` and `chat_model`. The file's other
  settings are kept.
- When the server lists an embedding model other than the chosen chat model,
  melo asks `Use <model> for long-term memory?` (piped: `[Y/n]:`), recommending
  Qwen3-Embedding first. `Use it` sends one `POST /v1/embeddings` probe and saves
  `embedding_backend = "<chat key>"` with that table's `embedding_url` (the
  chat URL) and `embedding_model`. `Leave it off` (or Esc), a failed probe, or no candidate
  saves `embedding_backend = "none"` and says that memory is off
  (`Long-term memory is off; melo setup turns it on.`); `melo setup` also names
  how to add a model, `ollama pull qwen3-embedding:0.6b` for an Ollama server
  and `Serve Qwen3-Embedding-0.6B on it` otherwise. See
  [Set up an embedding model](embedding.md).
- When the server cannot be used, melo says why, as
  `✗ URL: connection refused.`, and asks `What now?` (piped:
  `Enter tries again · s starts without saving · q quits:`). `Try again`
  (selected) asks for the URL again, with the one you typed as the default;
  `Start without saving` asks `Model name [default]:` and opens the UI without
  saving (`Not saved: melo asks again next time.`); `Quit` (or Esc) exits with
  code 0. The reasons are `connection refused`,
  `no answer in 5 s`, `HTTP 401` or `HTTP 403` with
  `the server requires an API key; localmelo sends none to local servers (oMLX: "Allow inference without API key")`,
  another `HTTP` status (a redirect names its target, which melo does not
  follow), `the reply is not a model list`, `the reply could not be read`,
  `not a URL` and `it lists no models`.
- Piped, an option is chosen by its first letter, so `s`, `start` and
  `start without saving` are the same answer (and `r` or `retry` is Enter);
  another answer asks again.
- Ctrl-D at a question prints `Setup cancelled.` and exits with code 0;
  Ctrl+C prints the same and exits with 130.
