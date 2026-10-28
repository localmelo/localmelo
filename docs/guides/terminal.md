# Guide: Terminal

## The interactive UI

`melo` without a task, in a terminal, opens a chat that shows its work as
it happens: the model's thinking, each tool call as it finishes, notices
and reflections, and a live area with the step, the phase, the running
tool and the run's progress facts while the model works. The answer
prints as Markdown. What it prints stays in the terminal's scrollback; the
input box and a status bar stay at the bottom.

```text
╭─ melo 0.1.0 ─────────────────────────────────────────────────────────────╮
│ model      Qwen3-8B · vLLM · http://127.0.0.1:8000/v1                    │
│ workspace  ~/demo-project                                                │
│ tools      4 built-in (3 ask first)                                      │
│ memory     no long-term memory · melo --reconfigure adds one             │
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
  [Procedural memory](../concepts/procedures.md)); Esc then does nothing, and the next
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
| `/attach PATH` | Stage one workspace text file; quote a path containing spaces. |
| `/attachments` | List staged files and their numbers. |
| `/detach N\|all` | Remove the numbered staged file or all staged files. |
| `/new` (also `/clear`) | Start a new session: the model forgets the conversation, and session grants end. |
| `/sessions [filter]` | Pick a saved chat in this folder; type to filter, Tab for all folders, ↑ ↓ to select, Enter to resume, Esc to close. |
| `/resume [id\|title]` | Resume by ID, ID prefix (at least four characters) or exact title; no argument opens the picker. |
| `/title [text]` | Show or set a 1–200-character title. Before the first message it is kept until the session is created. |
| `/undo` | Forget the newest finished turn and put its typed message in ↑ history. Files the tools changed stay changed. |
| `/usage` | Recorded steps, tool calls, tokens and time per turn, including undone work, with totals. `?` means not recorded. |
| `/memory` | Inspect the model's copied working window, its counts, the long-term backend and any restore limitation; no model call or memory write. |
| `/copy [turn]` | Copy a stored answer's Markdown, not the wrapped display. Without a number, use the latest finished, non-undone turn. |
| `/verbose [on\|off]` | Verbose rendering: 40 thinking rows, a tool call's arguments, 20 output lines, the reflection's summary and whole rationale; without an argument, flips it. |
| `/thinking [on\|off]` | The thinking excerpt on (the default) or off, leaving only `Thought for 1.2s`; without an argument, flips it. |
| `/stream [on\|off]` | Live streamed replies for later turns on or off; without an argument, flips it. See [Streamed replies](#streamed-replies). |
| `/expand [turn.call]` | A tool call's whole output or error, including stored calls after resuming: the latest call, call N of the latest turn, or turn T's call C (≤ 200 lines, then the rest's count). |
| `/progress` | The latest progress facts of this turn: level, step, attempt, phase and cycle, tests, the past runs, every named step and the memory it rests on. |
| `/approvals [ask\|auto\|deny\|forget [tool]]` | The approval mode and this session's grants; switch it, or forget one tool's grant (all with `forget`). Under `--yolo`, says how to turn prompts on. |
| `/status` | The model, server and its reachability, the workspace and its fence, the session, approvals, tools, memory, limits, files and version. |
| `/mcp` | The MCP servers, their tools and command lines, and the file they came from (`--mcp-config` or `mcp_config`); why MCP is off after a failure; how to name a file when there is none. |
| `/retry` | Re-run the newest finished turn, keeping its stored typed text even after a resume. Undo it first only if it is not already undone; earlier turns and files stay changed. |
| `/model [name\|number]` | Alone: the current model, and the models the server lists. With a listed id, or a number from that list (melo lists the models first if it has not yet): switch to it from the next message, and save it as the default. |
| `/backend [url]` | Switch to another model server: asks for its URL (Enter keeps the current one) and a model from its list, checks the server with `GET /models`, switches and saves both as the default (as `ollama` when the server answers like Ollama, else `vllm`). |
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
  Ollama (`GET /api/version`), else as `vllm`; it writes `chat_backend` and
  that table's `chat_url` and `chat_model`, and keeps the rest of the
  config: embedding, `request_timeout_s` and `extra_body`; the session runs
  on the config as saved. In a `melo --base-url URL` session they switch for
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
  `Thinking`, `Running`, `Waiting for your approval`, `Retrying in 4s`,
  `Reflecting`, `Wrapping up`, `Stopping`), then the run's facts: the step
  (`step 4`, never against `--max-steps`), the attempt from the second
  one, the phase with its cycle, the tests seen, and the time; the output
  tokens and `esc to stop` follow. When the line is narrow the hint goes
  first, then the tokens, and the facts are cut with `…`. The second row
  is the running tool, its own clock, and for `shell_exec` and
  `python_exec` their limit: `└ shell_exec  pytest -q tests · 0:25 of
  10:00`. On a session's first turn, a third row shows the past runs the
  progress line would: `last time 6 steps, 2m10s: Run ls ✓ · Read app.py
  ✓`.
- **Thinking.** `✻ Thought for 1.2s`, then the first 3 rows of the
  thinking (40 with `/verbose`, none with `/thinking off`), wrapped to
  the width. An intermediate message prints as Markdown after `●`; the
  last one is held, prints at the next step, and at the end only when it
  differs from the final answer.
- **Tool calls** print one block each when they finish: `✓` ran, `✗`
  failed, `⊘` blocked or denied, `◌` stopped or not run; the duration;
  `exit N` when a command ran and exited with another code than 0 (`grep`
  exits 1 without failing); how its approval ended; and up to 3 lines of
  its output (20 with `/verbose`), then `… +N lines (/expand T.C)` (a
  single line left prints instead). `/verbose` adds the call's arguments
  and images print as `▪ image/png · 312 KB · not shown`. Escape
  sequences in the output are removed, so a tool cannot move the cursor,
  set the title or write the clipboard.
- **Notices and reflections.** `↻` marks a retry, with its delay counting
  down in the live area; `·` an empty reply; `⚠` a stuck or spent-budget
  notice. A reflection prints as `✻ Reflected after attempt 1: continue ·
  Read the test before editing. (2.3s)`; `/verbose` adds its summary and
  whole rationale, and `asked twice` when it was; an unreadable one says
  whether the run continues.
- **The answer** prints as Markdown after `●`. A run that did not complete
  prints its best effort, the [status line](../reference/cli.md#output) and a hint, such as
  `Is the model server running at URL?` for a server that does not answer.

### Streamed replies

Streaming is off by default. Turn it on in `config.toml`:

```toml
[cli]
stream = true

[cli.reasoning_tags]
"Qwen3-8B" = "closing_only"
```

With it on, the interactive UI asks the provider for a streamed planning
reply and draws only text the release filter has already redacted, in the
live area, at the app's redraw rate; it never writes raw deltas to the
terminal. The per-model hint says how that model frames reasoning:
`explicit` (a leading `<think>` starts it), `closing_only` (everything
before `</think>` is reasoning) or `none` (tags are literal text). A model
with no hint keeps the conservative rule: content is held until the reply
ends, so the live area shows `✻ thinking…` rather than text. The same hint
splits the completed reply, so the stream and the stored answer do not
disagree. A model field the server sends separately is always treated as
reasoning and is never drawn as content; `/thinking off` hides its excerpt
as before.

Streamed text is provisional. If the reply becomes a tool call, a retry,
a run that fails, or a stop, melo closes the live text and prints it once
as `↳ partial reply (tool call)`, `(retrying)`, `(not the final answer)` or
`(stopped)`; the completed final answer prints once after `●`. A reply a
provider sends in one piece (or a provider without streaming) silently
falls back to the ordinary complete reply. `/stream on|off` flips the
setting for later turns without saving it. Direct mode and stdin-closed
runs never stream, so their output and request bodies are unchanged.

### Approvals

`shell_exec`, `python_exec` and `file_write` run only once you answer, and
so do MCP tools marked for approval. (The interactive UI loads the MCP
servers of `--mcp-config`, else of the top-level `mcp_config` key; a server
that fails to start turns MCP off for the session and `/mcp` says why.) The panel takes the place
of the input box, and your draft stays:

```text
╭─ Run a shell command? ───────────────────────────────────────────────────╮
│ shell_exec · exec · in ~/demo-project                                    │
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
- The header names the tool's permission tier after its name, and after the
  path for `file_write` (`exec` here; `write`, `exec` or `network` for an MCP
  tool, once the UI can load one). A read-only MCP tool would show `read`
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

Bare `melo` starts a new session; only the first message creates its row.
Use `melo -c` (`--continue`) to resume the newest chat with a finished turn
in the current workspace, or `melo -r` (`--resume`) to pick one. With none,
either starts a new session and says so; Esc in the startup picker also
starts new, without creating an empty row. `melo -r REF` resolves an ID,
a prefix of at least four characters, or an exact case-insensitive title;
multiple matches, or a title/message search fallback, open the picker.
An unknown reference exits 2. The startup choice, including the `-r` picker,
happens before the banner: a chosen chat is labelled `"TITLE" (resumed)`;
Esc or an empty list gives a new-session banner. `-c` and `-r` cannot be
combined, or used with a task, `--serve` or `--daemon`.

The picker lists at most 50 chats with finished, non-undone turns, newest
first. The filter updates after 150 ms; Tab toggles this folder/all folders.
Enter resumes, Esc or Ctrl+D closes it, and the ordinary draft stays intact.
`/sessions FILTER` and `/resume REF` use the same picker and resolution.
Web and scheduler sessions are never offered. A resume keeps the current
model and workspace, records them on the session and clears approval grants;
`/verbose` and `/thinking` preferences stay as they were. `/new` resets those
display preferences. A note says when the session previously worked in another folder.

After the banner, the recap shows the last three eligible turns from stored
redacted copies. The line below its title says what reached model memory:
`[sessions] resume_mode = "pairs"` (the default) restores recent user/answer
pairs, without tool steps; `"window"` restores a saved working-memory
checkpoint plus later pairs, falling back to pairs if no checkpoint is usable.
Set this in `config.toml`; an invalid value warns in the UI and uses pairs,
without adding warnings to direct-mode output. The recap's counts reflect
the restored window after eviction, not the total number of stored turns.

Two terminals may take turns on one CLI session. Before sending, melo reloads
changes from the other window; a busy or deleted session keeps the draft.
On exit, a session with turns prints its `melo -r ID` resume command.

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

In a terminal with no config file, `melo` asks for a model server and a
model, saves them, and opens the UI. When the server lists an embedding
model, it asks once more whether to use it. Later runs go straight in.

```text
$ melo
Found a model server at http://127.0.0.1:8000/v1 (1 model).
Model server URL [http://127.0.0.1:8000/v1]:
✓ http://127.0.0.1:8000/v1 lists 1 model; using Qwen3-8B-4bit.
No embedding model on this server. Serve Qwen3-Embedding-0.6B on it, then melo --reconfigure.
Saved to ~/.cache/localmelo/config.toml (melo --reconfigure changes it).
```

With a server that lists an embedding model too:

```text
Use nomic-embed-text for long-term memory? [Y/n]:
```

- melo looks for a server on this computer with `GET /v1/models` on
  `127.0.0.1` ports 8000 (vLLM, oMLX), 8400 (MLC), 11434 (Ollama) and 30000
  (SGLang), and offers the first that lists a model. Press Enter to accept
  the URL, or type another. Only the model list is requested; saving the
  choice asks `GET /api/version` once to tell an Ollama server apart. No
  chat request is sent, so a server does not load a model for it. A server that
  lists a chat model comes first; one that lists only embedding or reranking
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
- An Ollama server, one that answers `GET /api/version` with
  `{"version": "…"}`, is saved as `chat_backend = "ollama"` with its
  `[ollama]` table's `chat_url` and `chat_model`; the banner then says
  Ollama, and chat requests use Ollama's native API with its 300 s timeout.
  Any other OpenAI-compatible server is saved as `chat_backend = "vllm"`
  (the key for any OpenAI-compatible server), with its `[vllm]` table's
  `chat_url` and `chat_model`. The file's other settings are kept.
- When the server lists an embedding model other than the chosen chat model,
  melo asks `Use <model> for long-term memory? [Y/n]:`, recommending
  Qwen3-Embedding first. `y` sends one `POST /v1/embeddings` probe and saves
  `embedding_backend = "<chat key>"` with that table's `embedding_url` (the
  chat URL) and `embedding_model`. `n`, a failed probe, or no candidate
  saves `embedding_backend = "none"` and says why; the no-candidate line
  names `ollama pull qwen3-embedding:0.6b` for an Ollama server and
  `Serve Qwen3-Embedding-0.6B on it` otherwise. See
  [Set up an embedding model](embedding.md).
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
