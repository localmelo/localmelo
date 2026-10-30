# Guide: Web console

Use the web console to configure a localmelo gateway, work in chats, and inspect
saved worker runs. This guide is for someone using the console in a browser.

Before you start, install the gateway extra and have a model server ready; see
[Start the gateway](gateway.md#start-the-gateway). The gateway requires
authentication and only binds to loopback. Tools run with your
permissions and are not sandboxed; review a task before approving a tool call.

## Web workspace

Open `http://127.0.0.1:8401/` in a browser for the web workspace. It is the
console's page and talks only to `/api` ([API v2](../reference/api-v2.md)), so the gateway
must be running to serve it: `melo serve`. Run `melo console` in a
terminal and open its single-use link within 60 seconds (add `--port PORT`
for a nondefault port). Keep the link private; see [gateway security](gateway.md#keep-the-gateway-private).

On a first run, with no model set up, the welcome shows one card in place of the
starters. It looks for a model server on this Mac, as the terminal's first run
does, and shows what it found: **Found oMLX on this Mac** (or Ollama, or a model
server) with its model, **A model server needs an API key**, or **No model server
is running on this Mac**. **Use this model** saves the server, tests it and makes
the welcome ready without opening settings. With several chat models and none
known to be loaded, the card lists them, marks the loaded one and waits for you to
choose; it never picks one for you. With no server running, the
card shows the commands that start one (oMLX on a Mac, Ollama elsewhere, as in the
[model servers guide](model-servers.md)) in a block that one click selects, with a
**Copy commands** button for the keyboard, and asks
again every three seconds while it is on screen and the tab is visible, so the model
appears once the server answers. **Check again** asks at once, **Enter a server
address…** opens connection settings, and the composer
reads `Connect a model to start` until a model is set up. The card also shows when
the configured server stops answering, and it looks for a server only while it is
on screen ([`GET /api/setup/detect`](../reference/api-v2.md#get-apisetupdetect)).

The page opens in default mode, which leaves out what only a developer needs: the chat's
level chip, **Details**, the GPU-lock line, the **LOCAL** badge, the cache, step, phase and queue segments of the
status row, and the scheduler priority in Tasks. It says things plainly too: a running chat reads
`Working…` beside **Stop**, and the settings call the server **Model server** and its address
**Server address**. The header shows the running chip and a
model chip (the model's name, `Model offline` or `No model`) that opens settings.
**Workers** and **Tasks** join the sidebar once there is a run, a terminal chat or a task
to see in them. [Developer mode](#developer-mode) shows all of it. Everything stays in the
page, hidden.

A new chat shows where melo will work before you send anything. The chip in the composer reads
`In localmelo-workspace` and opens a panel with the default folder (`[gateway] workspace`, else
`~/localmelo-workspace`), the folders of your recent chats (newest first, four at most),
**Choose a folder…** and, in the [macOS app](macos-app.md), **Show in Finder**. In a browser
**Choose a folder…** opens a path field (a full path, or one that starts with `~/`), and in the app
the system's folder panel. File tools stay inside the folder; shell commands and Python start there.
The pick goes out with the first message as `workspace` in `POST /api/sessions` (a chat in the
default folder sends none), stays for the next new chat, and shows in the chat's header (`In ~/notes`;
developer mode adds it after the model). Three one-line starters under the composer (**Summarize
the files in this folder**, **Plan a small project with me**, **Explain a piece of code**) fill it
for you to send or change, and step aside once there is text.

1. Use **Settings** (**Connection settings** in developer mode) to choose the model
   server, its address, the model and the workspace (**Chat backend** and **Chat API URL**
   in developer mode). Changes apply to your next message. With nothing saved the dialog opens on a blank **Choose a server…**, and its
   fields, **Discover models**, **Test connection** and **Save** wait until you
   choose one. Local model discovery lists models without starting one. A cloud backend
   names an API-key environment variable and shows only whether it is set,
   never the key. Saving uses the loaded settings revision; if the file changed,
   reload the dialog and retry. **Test connection** probes the unsaved model
   settings; **Appearance** (**Theme** in developer mode) applies System, Light or Dark
   immediately and remembers it in this browser. Default mode stacks Model, with its test,
   Appearance and the **Developer mode** toggle in one dialog; developer mode keeps the
   Model, Test, Theme and Developer tabs and shows the settings file and its revision.
   You can also use `melo setup`.
2. Type a message and send. The first message creates the chat: the page
   answers your draft with the chat's first turn, and never creates a chat on
   its own. Answers are rendered as Markdown, with code blocks.
3. A turn reads like the terminal's: your message, a work group with the
   tool cards it ran, its reasoning, its notices and reflections, the live
   row while it runs, and the final answer with `Copy response`. Output that
   a tool printed is cleaned of escape sequences, and its images (PNG, JPEG,
   GIF, WebP) show inline.
4. A status row sits above the composer: the same segments as the terminal's
   developer-mode status bar (its default mode says them in words), `cache 67% idle 3m`, `ctx 1.0k/8.0k`, `step 3`, `tasks 1 running
   2 queued`, read from `GET /api/sessions/{id}/status` so both agree. Default mode draws the context
   segment only. The cache
   share reads `cache --` when the model server reports no cached tokens, never
   0%; a local server shows its idle time, not a countdown. The row is read at
   most once a second while events arrive and on the page's polls; on a narrow
   screen it drops segments from the right. An open worker run shows the same
   row.
5. A **N running** chip in the header counts what runs now: this console's
   chats with a turn in flight, tasks, direct runs and terminal chats (a run or
   terminal chat counts only while its process holds a live lease). It is
   hidden at 0 and stays on a narrow screen. Select it for a short list, each
   row naming the kind and the title and opening that chat, the task's card or
   the Workers detail; Esc or a click elsewhere closes it. The numbers come
   from `status` of `GET /api/health`, read with the page's 4-second poll; the
   list shows 20 rows at most and says how many more run.

The sidebar lists chats newest first, with search and switching, and closes a
chat. Under each title it names the chat's folder (developer mode puts the model
first), and a dot in the accent colour before the title marks a chat that waits
for your approval. Drafts stay with each chat in the current browser tab. The open chat's
turns arrive on its event stream (SSE); after a page reload the page restores
the chat, reads its last ten turns, and replays the stored events from there,
so nothing is lost or drawn twice. **Show earlier turns** loads older turns
without moving the current view. A stream the browser gives up on reopens
after 1, 2, 4, 8, 16 and then 30 seconds from the page's own cursor, and only
after the chat is read again: one that is gone leaves the page on a new chat
rather than retrying for ever. The list is re-read every 4 seconds while the
tab is visible, for changes from another browser. **Reconnect now** checks the
chat before reopening its stream. Independent banners show store, model and
connection problems without replacing other notices.

On a phone, or in a window 700 px wide or less, the page leaves out what a small
touch screen does not need. The header drops the model chip while it only names the
model, since the composer shows it, and keeps the chip when it warns (`Model offline`,
`No model`). The keyboard hints (`Enter to send · Shift + Enter for a new line` and
`⌘ K`) go, on touch screens and in narrow windows alike. The next steps of a failed
turn are buttons at least 44 px tall.

## Inspect workers

**Workers** (in default mode, in the sidebar once a run or a terminal chat exists) opens
read-only worker history from the CLI sessions file. Filter
by status, source or exact parent run, load more with the server's cursor, and
open a run to inspect its recorded brief, metadata, work and answer. Every
direct `melo` run is recorded there by default (`--no-save-session` or
`[sessions] record_direct = false` opts out; see
[Worker histories](../reference/cli.md#worker-histories)). The list shows runs
that are still going, and the list is read again every three seconds. An open
unfinished run is read once, then followed on its event stream: new steps, tool
calls and text appear within about a second, without reading the earlier work
again, and the stream stops when the run finishes. A hidden tab stops
following and catches up from the last event when shown. **Refresh** reads the
run again at once. A run that holds notes, accepted messages, memory
decisions or recalls shows each as one muted line. The view cannot send messages, approve calls or change grants; chat
close and level controls are disabled while Workers is open. Connection
settings remain available separately. **Chats** returns to the open chat.
Records are redacted by patterns, not guaranteed secret-free; review before sharing.

The **Terminal chats** chip beside **Worker runs** lists the chats of
interactive `melo` sessions from the same file, newest activity first. The
status filter stays; source and parent run are left out because a chat has
neither. A chat opens read-only, one section per turn, and is followed on its event
stream while a turn runs in its terminal; when that turn ends the page keeps
checking the chat and follows the next one. These chats never appear in the
sidebar's chat list, and the console cannot send to them. The terminal saves a
chat unredacted, so the console redacts each text of it as it reads: keys,
bearer tokens and private key blocks show as `[redacted]`, and the output of a
read of a secret file such as `.env` is replaced by a note. Images are not
redacted.

## Tasks

**Tasks** (`#/tasks`; in default mode, in the sidebar once a task exists) lists the
scheduler's tasks, newest update first, with state, priority (developer mode only) and
title. Open one (`#/tasks/<id>`) to see its state, the saved
recovery card (shown as plain text), the evidence status and the actions its
state allows, and the same status row as a chat (a task has no row until it has
run). Like Workers, the view replaces the chat while it is open.

- **Resume** appears for a parked or interrupted task. The page asks the
  server and confirms only when the server requires it: then a dialog says "A
  process from before the restart may still be running. Resuming may repeat
  effects. Confirmation does not prove the process stopped or effects happened
  only once." with **Resume anyway** and **Cancel**. Escape cancels. Nothing is
  recorded until you choose **Resume anyway**, and the confirmation lasts five
  minutes and does not survive a gateway restart.
- A task that asks to run a tool shows the state "Waiting for your approval"
  and the same approval card as a chat, with **Allow once**, **Allow for this
  task** and **Deny** (with an optional note). It is
  also where a `melo-task` task waits: nothing else shows it. The card
  disappears when you answer, and the task goes on (Allow) or gets the denial
  (Deny); pausing or stopping the task withdraws the card. While it is open,
  the view reads the task (or the list) again every 3 seconds and redraws only
  what changed, so a request that arrives later appears by itself.
- A started task shows a **Live** section below its approval card: its steps,
  tool calls and model text, drawn as the Workers view draws a run. The card
  reads the task's session once (`GET /api/sessions/<task id>/turns?events=true`)
  and, while the task runs or waits for approval, follows
  `GET /api/sessions/<task id>/events?after=<last id>`. New events are folded in
  and drawn together at most every 150 ms, without reading the earlier work
  again, and only the Live section is redrawn, so a reason being typed in the
  approval card keeps its focus. A hidden tab closes the stream and a shown one
  resumes after the last event. When the task is done, failed, cancelled,
  expired, interrupted or parked, the stream closes after one last read; a
  resumed task is followed again. A task that never started shows no Live
  section, and a finished task shows its work once. The console shows task
  events as it shows a chat's: the same authenticated loopback, no extra
  redaction.
- **Stop** uses the task's own stop, not the chat's. It is available until the
  task is finished and wins over a Resume in flight.
- A started task without a saved card shows "Resume unavailable: recovery
  information is missing." and **Start again as a new task**, which creates a
  separate task from the original objective. The effects of the earlier task are
  unknown and are not undone.

A button is disabled while its request is in flight. A refusal (a stale
confirmation, a finished task, Stop winning the race) shows the server's text.

## Inspect a chat

In developer mode, **Details** opens a session drawer with its tools and session grants, working
and long-term memory status, and finished runs with paging when a page supplies
a turn cursor. Granted tools can be
revoked there; the tools view may open a paused session. On narrow screens the
drawer stacks below the composer.

## Developer mode

**Connection settings → Developer** (in default mode **Settings**, where the toggle is
last) has one toggle, "Developer mode: show advanced settings and debug views". It is saved as `[ui] mode` in
`config.toml`, the setting `melo --help` reads, with the same revision check as
the other settings; a mode written in the file by hand wins the next time the
page reads it. It changes what is shown, never what runs. When it is on:

- the Developer tab lists the advanced and experimental `melo` options
  read-only, each with its tier and, for an experimental one, the issue that
  decides it (set them with the flag or in `config.toml`);
- **Details** gains an **Events** tab with the stored raw events of the latest
  turn, and its **Memory** tab adds the agent state and the long-term backend;
- the **Runs** tab adds each run's recorded counters: requests, tool calls,
  token usage and cache numbers;
- the header shows the chat's level chip, **Details**, the GPU-lock line and a chip with
  the page's state (`Ready`, `Working`, `Approval needed`), and the sidebar always lists
  **Workers** and **Tasks**;
- the status row adds the cache, step, phase and queue segments, and Tasks names each task's
  priority;
- the sidebar shows the **LOCAL** badge, a running chat reads `Running on your backend`, and the settings
  keep their technical words (**Chat backend**, **Chat API URL**, `Settings apply to subsequent turns…`).

When it is off these are hidden, not removed: turn it on again to see them. While a chat
waits on a foreign GPU lock, default mode shows a banner in place of the GPU line:
"Waiting for this Mac's GPU, your chat starts when it is free." Chats, approvals, the
Workers and Tasks views and what the settings save work the same in both modes.

## Recover a send

A send is named by a `client_request_id`. If its answer is lost the page asks
the store whether the message was accepted, and offers `Retry` only when it
was not; a retry repeats the same key, so one message makes one turn. A
message that was never accepted goes back to that chat's draft. The outbox is
also kept in the tab's session storage, including its key, and a send that
was in flight at reload is checked against the store before a retry. It says
`Not sent` (or `Outcome unknown` when the store could not be asked), with
`Retry` and `Cancel and keep draft`.

## Scheduling and stopping

In developer mode a chat's header has a level chip (P1 to P4, P2 by default) that sets how the
scheduler orders the chat's calls in front of other chats; it is kept in
memory and is not saved. Next to the title and the level chip, a machine line
shows a foreign GPU lock (`GPU lock: <holder> · <purpose>`, with
`(recorded)` or `(chats wait)`), `GPU lock: yours`, `GPU lock unreadable`,
`· ETA passed`, and the battery (`Battery 55% · discharging`). AC power with
no lock shows nothing. The level chip is a menu button: Enter or the Down arrow opens its
menu with focus on the chat's level, the arrow keys, Home and End move between the levels,
and Escape closes it and returns to the button. While a call waits (for a slot, the background cap or
a foreign lock), the header's status pill and the status label say
`Waiting: <reason>`, the sidebar says `● Waiting` and the live row says the
same; the reason is the scheduler's own code (`Resources`, `Priority`, …), except that a
wait on a foreign GPU lock (`Reservation`) reads "Waiting for this Mac's GPU", as the
banner does. These are the same facts the terminal shows, read from and
written to the one gate the gateway shares:
`level` and `held` on a session, `machine` on the list, and
`POST /api/sessions/{id}/level` and `POST /api/presence` to set them.

Stop cancels the run, and so does Escape while the cursor is in the message
box; Escape anywhere else only closes what is open, in Tasks and Workers as in a chat. A model request in flight is abandoned, although the
model server may finish computing it, and a running `python_exec` child is
killed with its process group. `python_exec` code runs
in a child process, so the gateway keeps answering other requests while it
runs. See [Tools and MCP](../reference/tools.md) for the tools' limits. The turn is saved
with the status `cancelled`, and the chat takes the next message once that
save is done. Each message sees the earlier answers of its chat, as in a
[session](gateway.md#sessions).

## Approvals and failures

A tool call that needs approval shows a card below its `Needs approval` tool.
Console sessions use `RememberingPolicy` and ask before `file_write` and
`python_exec`; legacy `/v1` has no approval policy. Approvals are not a
security boundary against a bearer holder or a malicious process under the same
OS account, which can answer them.

The card asks a plain question ("Run this command in demo-project?", "Replace
notes.txt (125 bytes)?", "Let melo use <tool>?") and shows exactly what would
run. **Allow once**, the filled button, runs this call only. **Allow for this
chat** (**Allow for this task** in Tasks) stops asking about that tool for the
rest of the chat; the line under the buttons says so, and for a command or
Python code it warns that it can reach files outside the workspace. **Deny**
opens an optional box, "Tell melo what to do instead", and sends the denial
with that text; **Back** or Escape returns to the answers. The controls wait
500 ms before they can be used; arrival does not move focus and no single-key
shortcut answers a request. When the card is outside the view, a bar above the
message box reads "melo is waiting for your answer", and **Review** opens its
group and moves to **Allow once**. If approval details cannot be loaded, or
arguments were truncated or converted, both allow buttons are disabled; you can
reload the details, deny, or stop the run.

A failed turn shows a card with a symbol (✗ for an error, ⚠ for a warning), a
plain title, the first line of the error and a hint. On the newest turn one next
step is the filled button and the other actions are links: **Open model
settings** after a model server error that came back as an answer (a refused
key, an unknown model), **Continue** after running out of steps
or timing out (**Try again** when no tool ran), **Try again** when the run was
stuck, failed or stopped after reflection, **Show tools** when tools could not
start, and **Copy details** after a crash inside localmelo. **Try again** and
**Continue** are offered on the newest finished turn only, and Continue needs a
turn that ran tools. **Copy error** copies one line and the run id for a bug
report; the technical fields stay under **Details**. Stopped or interrupted
turns use a neutral line (■). The banner shown when the page cannot reach the
gateway has **Try again**.

When the model server cannot be reached at all, the card says `Could not reach
the model server` and asks `Is your model server still running at
127.0.0.1:8000? Start it, then try again.`, naming the address that failed. It
does not repeat the error: the provider's own text stays under **Details**. The
filled button is **Try again** (**Continue** when the run had used tools, so the
work is picked up, not started over), and **Change the server address** is a
link that opens the settings. For a hosted model the hint reads `Check your
network connection, then try again.` and the link stays **Open model settings**,
since there is no address to change. A run that ended this way also leaves the
retry lines out of its work box, which the card already says; while it is still
retrying they show.

## Persistence and limits

The page's chats are sessions in the gateway's own session file,
`~/.cache/localmelo/gateway/sessions.db`; the terminal keeps its own in
`~/.cache/localmelo/cli/sessions.db`. They last until you close them: a
gateway restart keeps them, and a chat reopens with its transcript read from
the file, no Agent built yet (`not_loaded`); its Agent is built on the
conversation the file holds when the next turn or retry starts.

A chat runs in gateway mode: its Agent gets the gateway's default run options,
the three built-in tools `file_read`, `file_write` and `python_exec`, and no
MCP tools, except that it keeps its answers. It can read and write files and
run Python code; `shell_exec` is not offered, so there is no direct shell
command (Python code can still start a process). Its model is the gateway's
own, from the config file, and the page reads that file's `[scheduler]` table
too — `policy`, the lane sizes, the call log and the lock and battery keys.

In `lock_gate = "record"` (the default) a foreign lock is shown and logged but
nothing waits. In `wait`, every call a chat makes waits while a foreign lock is
present, embeddings included. A chat request waits up to 300 seconds, and a
turn is stopped after 600 seconds of active time: a call the scheduler holds,
including a foreign-lock hold, does not count against that bound.

The page does not use the disabled-by-default [legacy playground routes](../reference/http-v1.md#legacy-playground).

The workspace has these limits:

- It is for loopback use only. Sign in as described above and open it on this
  computer as `http://127.0.0.1:8401/` or
  `http://localhost:8401/`. Remote access, from another device or through a
  proxy, is not supported; see
  [Keep the gateway private](gateway.md#keep-the-gateway-private).
