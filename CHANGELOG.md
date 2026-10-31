# Changelog

All notable changes to localmelo are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

localmelo has no release yet. The v0.1.0 milestone is due 2026-12-20.

## [Unreleased]

### Changed

- A delegated task whose request says "today" or "今天" anywhere (and not "not today" or "今天不用") is Today level; before, only "by/before/due today" counted. The console takes its default folder from the same place as tasks, so both always agree. The embedding guide shows the setup picture, and the folder-listing terminal test checks a real file name.

### Security

- The approval card says why melo asks, and never offers a chat-wide allow for
  a high-impact call. In the web console and the macOS app, a call that
  installs software, downloads files, changes files outside the workspace or
  does anything else high-impact gets a card with a question that names it
  ("Install software?"), "Why melo asks" with up to three reasons, "You
  asked:" with the message that started the turn, and only **Allow once** and
  **Deny** (the filled button, the one **Review** moves to). The terminal's
  panel shows the same title and reasons before the code, has no session
  choice, starts on No, and numbers its options by position. Both say in one
  line that a session grant never covers installs, downloads or changes
  outside the folder; the routine grant line now says it covers routine calls
  and that installs, downloads and changes outside the workspace still ask.
  Answering "for this chat" to a high-impact request over the API allows that
  one call, replies `"remember": false` and no longer starts the grant's
  clock ([#294]).

- High-impact calls ask even inside a session grant. Before an approval
  policy decides, each `shell_exec`, `python_exec` and `file_write` call is
  reviewed by fixed rules (the shell command split, the Python code parsed;
  nothing runs) as routine, high-impact or undecided. Installing software or
  a project's packages, creating an environment, downloading installers,
  large files or anything outside the workspace, running downloaded code,
  writing or deleting outside the workspace, setting up something that runs
  later, using passwords or keys, changing system settings, opening a network
  service, sending files out and running code built at run time are
  high-impact. "Allow for this chat" now covers routine calls only: a
  high-impact or undecided call asks every time and is never granted, and the
  terminal's `auto` mode asks for them too. A call that would change
  localmelo itself (an install into the Python that runs it, a write under
  `~/.cache/localmelo`) is denied without asking. A denied high-impact call
  tells the model why melo asked. Direct, `--brief` and `--yolo` runs are not
  reviewed ([#290]).

- A worker's shell and Python write only inside its workspace on macOS. In a
  `--brief` worker run (`melo worker`) and a saved task, `shell_exec` and
  `python_exec`, the brief's checks included, start under a `sandbox-exec`
  profile that refuses file writes outside the workspace, whatever the command
  does; uv's, black's and npm's caches, the user's own temporary folder (where
  macOS's `mktemp` writes), a few devices and, in a linked git
  worktree, its own git directory and the object store stay writable, so
  `uv run`, `pytest`, `ruff`, `black`, `mypy`, `npm` and `git status`/`diff`/`add`
  work. `TMPDIR`, `TMP` and `TEMP` point at `.melo-tmp` in the workspace. Reads
  and the network are not confined. A refused write gains a line that says so,
  and `run.confinement` counts them. A run whose writes cannot be confined stops
  before its first command; other systems run unconfined and say so. Chats are
  unchanged; `-X agent.confine_writes=false` turns it off ([#274]).

- A developer setting confines plain direct runs too. `[agent]
  confine_direct_runs = true` (or `-X agent.confine_direct_runs=true` for one run;
  off by default, listed by `melo config keys` and `melo -h all`) gives a task
  given on the command line and the `melo ""` loop the same write confinement
  that `--brief` runs have, in the workspace (`--workspace`, else the current
  folder), and the run's JSON result carries `confinement`. `--brief` runs and
  saved tasks keep `confine_writes`; chats are unchanged. A run that cannot be
  confined stops before its first command with one line on stderr and exit
  code 1 ([#284]).

- Gateway authentication is in progress: loopback-only binding, a private bearer
  token for API clients, and a one-use terminal sign-in link for browser sessions.
  Existing unauthenticated API clients must migrate; see the gateway guide.
  Same-user processes and localhost services on other ports remain outside the
  isolation guarantee. ([#136])

- Gateway security-1b is in progress: disable the legacy playground by default;
  `melo --serve --enable-playground` opts in behind authentication and uses
  saved local backend/model settings. Authenticated settings and model probes
  accept typed HTTP/HTTPS endpoints without following redirects. Explicit
  close/delete refuses running or queued starts with 409 rather than cancelling
  them ([#141]).

- Redaction now covers PEM private key blocks (RSA, EC, DSA, OPENSSH, ENCRYPTED,
  plain PKCS#8 and PGP `PRIVATE KEY BLOCK`): the BEGIN and END lines stay, the body becomes `[redacted]`,
  and a block cut before its END line is redacted to the end of the text. Public
  keys and certificates are unchanged. Streamed display text holds an open block
  until its END line, or drops its body past 16 KiB, so a key never streams out
  ([#188]).

### Fixed

- When you update a fact, the newer value wins. Told "I live in Lisbon." in one chat
  and "I live in Utrecht." in another, a third chat asked "Where do I live?" recalled
  both: the Lisbon note had been replaced, but the Lisbon chat's turn came back first,
  and the model answered with the old city or asked which one was right (on a local
  model, 1 of 15 such questions right with an embedding model, 0 of 15 by words only).
  Now a note replaces an older note of the same subject whoever wrote either ("I live
  in", "my name is", "call me", "my weekly sync is on", "我在X工作", "我的X是Y", and, with an
  embedding model, a sentence that differs only in its value), in any topic; look-alikes
  such as "my flight to Porto" and "my flight to Faro" stay apart. The old note is kept
  but never recalled, and Undo brings it back. A turn whose fact was replaced comes
  after the newer note and reads `(older; replaced by a newer note)`, and the chat is
  told that when notes or earlier chats disagree, the newest one is true ([#322]).

- A fact you state is kept as a note, and melo no longer goes off to save it in a
  file. On a real model, "My group meeting is on Thursdays at 3 pm.", "Remember that Sam
  has the lab keys.", "Call me Ray." and "我的猫叫团子。" were already kept as notes, but
  the model then explored the folder and tried to save the fact itself with `file_write`
  or `python_exec`, which asked for approval and left the turn running. The chat's
  instructions now say that localmelo keeps the notes by itself, that the model never
  saves what you say about yourself in a file or with code, and that it acknowledges a
  fact in one short sentence without tools; and the message the model reads carries one
  line saying what was kept (`melo kept from this message: “Call me Ray.” (saved; nothing
  else to save)`), or that part of it looked like a password or key, was not kept and must
  not be written anywhere. The page shows nothing new. Direct runs, workers and the
  benchmark are unchanged ([#313]).

- Memory across chats keeps and finds more of what you say. "My reading group meets on
  Tuesdays." and "我的猫叫团子。" were not kept as notes, because only "my ... is" sentences
  counted; a sentence that starts with "my" or "our" and has a verb of habit or state
  (meets, starts, lives, plays, likes ...), or "我的X" with 叫, 姓, 住在, 喜欢 or 每周, is now
  a note. And a chat asked "which day does my reading group meet?" recalled nothing
  though an earlier turn said it: the words were compared as written, so "meet" did not
  match "meets" and only 2 of the 4 content words matched. A plural, "-ing" or "-ed"
  ending is now ignored when words are compared ([#310]).

- A chat says plainly what it remembers and answers in the script you write in. Asked about
  memory, melo had described how the system around it works ("retrieved context and
  reflections", "cannot write to long-term memory") and had answered in traditional characters
  to a user who wrote in simplified ones. The identity paragraph of terminal, console and app
  chats now says that melo remembers the recent messages of this chat and nothing from other
  chats, or, when memory is kept between runs and can be recalled, also the short notes
  localmelo keeps on this computer (in `legacy` reasoning what melo did with its tools and what
  it learned from failed attempts; in the explicit modes what you told it or asked it to
  remember and what it found or decided); that melo keeps no notes of its own and files in the
  folder are not memory; to answer a memory question in one or two plain sentences without
  describing internals; and to reply in the language and script you write in, simplified or
  traditional Chinese for example (this replaces "Answer in the user's language."). Direct runs,
  workers and the benchmark send no identity and are unchanged ([#305]).

- `/clear` clears the screen. In the terminal it was another spelling of `/new`: it started a
  new chat and printed a `New chat` rule under the whole page. Now `/clear` is a command of
  its own: after the same reset (conversation, approvals, display preferences) it wipes the
  screen and its scrollback and draws the welcome again at the top, with the input box below
  it; `/new` keeps the page and the rule as before. `/new` says in the menu and in `/help`
  what `/clear` adds, and `/clear` has its own row in the menu and in `/help all`. With
  `NO_COLOR`, `TERM=dumb` or output that is not a terminal, no escape code is written and
  `/clear` does what `/new` does ([#304]).

- Melo changes moves without a hitch. The desktop pet used to cut what was running when its
  mood changed (the breath snapped back to rest and began again, a hop was cut in the air)
  and dissolved one pose into the next in 150 ms. Now nothing that moves is cut: a breath
  the next mood shares carries on, one it changes eases to rest over a quarter of a second
  before the new one begins, the "z" of a sleeping Melo fades out, and a hop lands. A change
  waits at most 0.4 s for a rest point (the end of a breath, between blinks); then Melo
  squashes a little and rises, and the new picture fades in over the low point (250 ms) so
  the pose changes where it is hidden. Under Reduce Motion, Low Power Mode and heat nothing
  moves and a change is a plain 200 ms cross-fade. The mood still follows the app's 2 s
  health poll ([#306]).

- Waiting for the model is one row in the terminal, not two. A turn waiting on a model whose
  replies are held until they are complete showed `✢ Thinking… · 7s · esc to stop` and, under
  it, `✻ thinking…`, which said the same again. The spinner row alone says it now; rows that
  add something (the streamed text, the running tool, the past runs) stay ([#289]).

- A reply that is only the model's unfinished thinking is never shown as the
  answer. When a thinking model stopped before its end-of-thinking mark, a
  server could return the thinking as the reply's text, and a greeting got the
  model's notes to itself as its answer. localmelo now tells such a reply by
  facts, never by its wording: a `<think>` that is never closed, no answer
  with reasoning tokens counted, a streamed reply that repeats the server's
  reasoning, or no thinking and no `</think>` from a model whose thinking ends
  with a mark (declared in `[cli.reasoning_tags]`, or seen from a server that
  sends reasoning in a field of its own and counted none in this reply). The
  text becomes the reply's thinking, folded in the terminal and the console,
  and the model is asked once more (one extra request, with the notice
  `unfinished_answer`); a second reply without an answer ends the run
  `failed` with `melo did not finish an answer`, with `/retry` in the
  terminal and **Try again** in the console. Direct runs and workers report
  that error instead of returning the thinking as the result ([#288]).

- The console and the terminal no longer show a turn's time twice. A turn that answered
  without tool calls showed `Thought for 2.0s` over the answer and `2.0s · 1 step` under it,
  the step's time and the turn's time being the same. The console now shows no summary line for
  such a turn, or `Thought` over its reasoning when it has some (the reasoning is the body of
  that line, not a second `Thought` inside it); the terminal prints `✻ Thought` for a reply
  that ends a turn without tool calls. The footer keeps the turn's time. A turn with tool calls
  is unchanged ([#287]).

- The terminal's command list stays open while you delete characters. Typing
  `/exi` showed `/exit`, but deleting the `i` closed the list instead of showing
  the `/ex` matches, because prompt_toolkit completes only after text is
  inserted. Any deletion that removes text (Backspace, Delete, Ctrl+W, Ctrl+U,
  Ctrl+K) now asks for completions again on the text before the cursor: the
  commands of `/ex`, all of them at `/`, the names after an `@`, and nothing once
  the `/` or `@` is gone, which closes the list. Moving the cursor alone still
  opens nothing, Esc closes the list until the next edit, and Tab, Enter and the
  arrow keys take a command as before. ([#285])

- A reflection reply with one common slip in its JSON no longer costs a retry.
  A reply with a stray `]` after a string value (a `]` that no `[` opened) could
  not be decoded, so it was asked for again, and a second failure lost the
  reflection. With the default `reflection_gate flags`, a reply that holds no
  reflection now gets one repair of the `{...}` the decoder rejects, at the
  decoder's error and nowhere else: a `]` or `}` that no opener matches is
  dropped, a comma before a closer is dropped, or raw newline, carriage return
  and tab characters inside strings are escaped. The repaired text must still
  hold an object with a valid `recommended_action`, nothing is closed or
  inserted, and a reply that needs two repairs stays unreadable. The
  `reflection` event gains `repaired` (`unmatched_closer`, `trailing_comma`,
  `control_char` or `null`). When a reply with JSON that the decoder rejected
  is asked for again, the retry adds one line after its note that quotes the
  decoder's error and up to 60 characters around it. The repair changes no
  request whose first reply reads. ([#239])
- `melo --serve` writes its startup lines, the port that `--port 0` picked
  included, as soon as they are printed when stdout is a pipe or a file. They
  used to wait in the buffer until the server exited, so a script or a log
  reader could not find the port while it ran ([#236]).

- Setup never picks a model silently when several are served and none is known
  to be loaded. On a server whose status route needs a key (it answers 401) the
  first run used to default to the first listed model, which could be a large
  one that was not in memory, and Enter started loading it. First-run setup,
  `--reconfigure` and `/backend` now default only to a model the
  server says is loaded; otherwise they ask for a number or an id and Enter
  alone asks again (an empty answer with no terminal is an error).
  `melo --base-url URL` without `--chat-model` picks the one loaded model, or
  exits 2 with the list and `Pass --chat-model <id>`. ([#231])
- A direct run with `--base-url` and no `--chat-model` (`melo --base-url URL -- "task"`,
  `--brief` and `melo ""`) no longer asks a server for the model `default`, which a
  server that loads models on demand can read as any model, a large one that is not
  in memory included. It lists the server's models first and chooses as the interactive
  UI does: the only chat model, or the one the server says is loaded. With several and
  none known to be loaded it exits 2 with the list and `Pass --chat-model <id>` (and, when
  the status route answered 401, a line saying the server needs a key for that); with
  `--brief` that is a `startup_error` result whose `exit_code` is 2. A server that cannot
  be listed still gets `default`, with one warning line on stderr in text output. With
  `--chat-model` nothing changes: no list request, the same requests. ([#236])
- The welcome screen's "Connect a model" button shows only when no model is configured or the configured server cannot be reached. With a reachable model, the welcome screen shows just the suggestions ([#230]).

- `/memory` and `/status` in the terminal, and the console's details drawer, show the session's own working-window cap (`window_messages`) instead of always 20 ([#230]).

- Concurrent first runs on a new home no longer lose a recording. A process
  that read a new sessions file just as another finished its schema saw version
  0 with tables and called the file foreign (`melo: could not record this run
  (... is not a localmelo sessions file)`); the version and the tables are now
  read as one snapshot. A foreign file is still refused at once ([#222]).

- A brief's `Check:` command now runs in a default `melo --brief` run. A direct
  run offers no `shell_exec` to the model, so the harness's own check call was
  refused, the run ended `completed` with exit 0 and `tests` `failed`, and only
  the developer flag `--builtins all` made it run. The check now runs in every
  `--builtins` setting (`none` included); the model's tool list and its own
  `shell_exec` calls are unchanged, and `pre_execute`, the workspace root, the
  shell timeout and redaction still apply ([#205]).

- With `--window-eviction chunk` or `boundary`, the request fit of `--context-window`
  keeps its cut across requests: after it drops to 70% of the input limit it starts
  the next requests from the same unit and drops again only when a request would
  pass the limit, so a tight context window no longer moves the history every step
  ([#204]). `slide` is unchanged.
- `scripts/cache_report.py` runs under Python 3.9 again: the `zip(..., strict=True)`
  call that needed 3.10 is replaced by an explicit length check. `-h` and `--help`
  print its usage and exit 0, and no arguments print the usage to stderr and exit 2,
  instead of reading `--help` as a log path ([#198]).

- The console's **Tasks** view (`#/tasks`) shows `Tasks` in the header, as the
  Workers view does, instead of the open chat's title and model while the task list
  or a task card is open. Opening a chat again shows the chat's header ([#198]).

- `melo --brief` runs follow a worker protocol (#185): a fixed paragraph appended
  to the brief asks for the smallest change as soon as the target is known, the
  task's check commands after each change, a read of the lines before editing and
  a final answer that says which checks ran; a reflection that says `continue`
  starts the next attempt while steps and attempts remain, whatever its three
  flags say; and an attempt boundary that finds the git workspace unchanged since
  the run started adds a `no_change` notice and one line asking the next attempt
  to start by writing. Runs without `--brief` and request bodies outside brief
  runs are unchanged ([#193]).

- A `melo --brief` run keeps writing (#185): after 6 steps in a row without a
  workspace change it adds one "write what you can already write" line and a
  `no_change` notice with detail `{"scope": "step"}`; each planning request ends
  with a note of the tool output the window dropped (tool, key argument, length,
  newest 8); the reflection request asks for file paths with line numbers and the
  first command or edit to run next, and the boundary line ends with "Start with:
  <that direction>". Runs without `--brief` are unchanged ([#197]).

- A `melo --brief` run verifies itself (#185): the commands on the brief's
  `Check:` lines run in the workspace, as `shell_exec` calls under the same mode
  policy, workspace root and timeout, before the run ends `completed`. A failing
  check sends the model back to work with its command, exit code and last 60
  lines of output (redacted), at most twice. A check that still fails ends the
  run `completed` as before, but the worker result's `tests` is `failed` and a
  `brief_check_failed` notice says so; when every check passes `tests` is
  `passed`. Each run of a check is a `brief_check` notice and a `tests` item with
  `source: "brief_check"`. A brief with no `Check:` line, and runs without
  `--brief`, are unchanged ([#195]).

- The `reflection` event now records the decision's three flags (`flags`), the
  gate in force (`gate`, with `utility` under `"utility"`) and a `stop_reason`
  when the run does not continue, so a `gate_stopped` run shows which check
  stopped it. A blank rationale's `gate_stopped` error names it too. Decisions
  and request bodies are unchanged ([#187]).

- A turn cancelled before its run reported `run.finished`, such as a SIGHUP
  during setup, now saves the cancel reason in the stored result, so saved
  history shows why the turn ended instead of an empty error ([#168]).

- Prefer same-task episodic digests during recall; gate and cap cross-task
  digests, with an optional cold-start allowance. A digest key includes the
  run's workspace realpath, and a final evidence-less attempt keeps the last
  evidence-bearing attempt's digest. General memories are not gated. Do not
  admit digests lacking file or verification evidence ([#165]).

- Label Python checks in attempt digests with a recognizable runner and safe
  target basename when available, rather than the first line of code. Unknown
  checks use a neutral label. Results show counts, an exit code or `completed`
  instead of arbitrary output. Hidden paths and agent scratch notes are omitted;
  Python file writes are not inferred ([#149]).

- Direct workers report agent construction failures, including refused
  config-file values and invalid TOML, as `startup_error` with a redacted
  cause (`$HOME` shown as `~`), and attempt a `transport_error` result on
  event-output failure; a broken output pipe exits 1. ([#157])

- Correct the API v2 response formats and tools example, and update the bilingual
  quickstart to reflect the shipped console settings and approvals ([#122], [#124]).
  Clarify task validation and read-only worker details ([#119], [#123], [#130], [#133]).
- Clarify that explicit reasoning modes can use text-only long-term memory
  without an embedding backend, in the embedding guide and Chinese architecture
  boundary ([#132]).

- Attempt digests skip task headings and report verification only for passing
  checks, rather than treating trivial Python imports as tests ([#128]).

### Docs

- oMLX is documented as a supported local server, recommended on Apple silicon, next to Ollama: the README quick start (English and Chinese), the quickstart, the model servers guide, the configuration reference and the website home, quickstart and architecture pages say that its config key is `vllm` and explain which model first-run setup picks, and the first-run transcript and the `--base-url` model rule match the code ([#237]).
- Chinese (zh-CN) twins of the command-line, API v2 and troubleshooting references in `docs/zh-CN/reference/`, covered by the docs anchor and example tests ([#203]).
- Add zh-CN reference twins for events, retrieval and memory, with the English anchors and code examples ([#202]).
- Chinese (zh-CN) twins of the docs index, the evaluations page and the docs maintenance guide, and purpose lines on 14 pages that opened on a heading; the docs tests check the twins' anchors, code blocks and inline code ([#216]).

### Added

- The macOS app shows what melo is doing while its window is put away. A small
  pill takes Melo's place on the desktop and says in one plain line what runs
  ("Reading example.com", "Running the tests"), with "+1" when more run. It
  turns amber while an approval waits, shows "Done: …" for 6 seconds and
  "Stopped: …" for 10, and says "Model server not reachable" while the model
  server is down; then Melo comes back, wagging after a "Done". Click it to
  open the chat or task it names, drag it like Melo, or press **×** to hide it
  until the next task starts. **On the desktop**, in the menu-bar menu and in
  Melo's menu, chooses the pill with Melo (the default), Melo only or nothing.
  `GET /api/health` `status.recent` also says what the finished run was called
  (`title`) ([#326]).

- The main chat answers, or hands bigger work to a task by level. It answers a
  quick question itself and starts bigger work as a task it owns, with one line
  such as `Started: Plan the trip · Normal`; the result comes back as a card.
  `/task …` or **Run as a task** starts a task without a model call, and a short
  phrase that names a task (`pause the slides`, `暂停 Fix CI`, `先做 slides`)
  changes it at once, with **Undo**. The level comes from your own words: urgency
  or a deadline today is Today, "no rush" is Whenever, else Normal; **Change** on
  the line sets your own level. A message it cannot settle in 4 steps becomes a
  task (a question asks "Start this as a task?" first). Only your own message
  changes a task: a file, page or task report that says "stop" makes a one-tap
  suggestion. Levels read as words in default mode, in the console and in
  `melo task`, which also takes them as input. For API clients: turn bodies take
  `as_task`, a code-handled message answers 200 `{"echo"}`, the main chat's turns
  carry `echoes` (`attention.echo` events), and `POST /api/front/echoes/{id}`
  answers an echo's Undo, Apply, Start or level. See
  [Hand work to the main chat](docs/guides/web-console.md#hand-work-to-the-main-chat) ([#320]).

- One main chat, with side chats under it. The console and the app open on the
  **Main chat**, which is always there and cannot be closed; the chats you start
  with **New side chat** (`⌘ K`) are listed under it, newest first under a line
  for each day, and each names the main chat in its header, which goes back to it.
  Your existing chats become its side chats, unchanged. When a side chat's run
  ends, or a task finishes, fails or waits for your approval, the main chat shows
  a card with one plain line and a button that opens it; a card is not a message,
  so the model never reads it. The sidebar's **Chats | Workers** switch is gone:
  worker history is in the panel's **Workers** tab, and the empty main chat links
  it too. For API clients, every session has `main`, `GET /api/sessions` names the
  main chat in `main` with its newest card, the main chat's turns carry `cards`,
  `POST /api/sessions` still makes a chat (a side chat), `POST /api/tasks` takes an
  optional `front_id` (the chat that starts the task owns it; else the main chat),
  and deleting the main chat is 409 `main_chat`. See [The main chat and side chats](docs/guides/web-console.md#the-main-chat-and-side-chats) ([#317]).
- The panel follows the open chat, and approvals fold into Activity. A side chat's
  panel lists that chat's own activity and the workers it started; the main chat's,
  and the panel with no chat open, list the whole day, and Melo's status line says
  what the open chat is doing. A row that asked for approvals carries them folded
  under its result line (`2 approvals, 1 waiting`): open it to see each request in
  plain words, how it was answered and when, and answer a waiting one with **Allow
  once** or **Deny** right there. The **Approvals** tab is now **Workers**: the
  tasks the open chat started, running first, then those done today, each with its
  state in words and its result, opening to live steps or history with **Pause**,
  **Stop** and the priority (the main chat also lists worker runs); with none, it
  says when workers appear. Result lines in Activity and Workers are always plain
  words: `Stopped before answering`, `Made a picture`, a file's name, or the first
  sentence around code. `GET /api/activity` gains `session_id`, `scope` and, per
  row, `approvals` ([#314]).
- See and change what melo remembers. The panel's **Memory** tab lists the notes melo
  kept, newest first, each with its topic and the chat and day it came from; **Edit**
  rewords a note, **Delete** asks first, the search box filters, and **Forget everything…**
  names what goes (the notes and melo's search index of your chat turns; chats stay)
  before it removes them. A switch, **Remember across chats**, turns memory on or off and
  a line says whether melo searches by meaning or by words only. A note outlives its
  chat: deleting a chat keeps its notes (`From a deleted chat`), and a chat with notes
  asks "Also forget the 2 notes from this chat?" with **Keep the notes** first. A deleted
  or reworded note leaves no old text in the files, and a deleted note is never recalled
  again. In the terminal, `/memory` is now a list of the notes (Enter on one offers Back,
  Edit it or Delete it; `/memory window` is the old view of the model's working window),
  and `melo memory list`, `forget [ID] [-y]`, `status` and `reindex` do the same from the
  command line. The first run, `melo setup` and the console's Settings ask "Remember
  across chats with <model>?" when the server lists an embedding model and say plainly
  that melo searches by words only when it lists none; the setup lines that said
  "Long-term memory is off" now say so in these words. For clients, `GET
  /api/memory/records` adds `from`, `chat_gone` and `edited`, and there are `PATCH
  /api/memory/records/{id}`, `DELETE /api/memory/records?all=true`, `PUT /api/memory` and
  `POST /api/memory/reindex`; `DELETE /api/sessions/{id}?notes=forget`; `GET /api/models`
  lists `embedding_models`; `GET`/`PUT /api/settings` carry `across_chats`. See
  [Memory across chats](docs/concepts/memory.md#see-and-change-what-melo-remembers) ([#312]).
- Memory across chats: melo remembers what you tell it, in every chat of the web console
  and the app. When a message you type says something about you ("I prefer short
  answers", "my group meeting is on Thursdays", "call me Ray", "我习惯用简体中文") or asks
  melo to remember a fact, melo keeps it as a note, and a line under your message says
  "Remembered: you prefer short answers · Undo". A newer value replaces an older one;
  only what you type counts, never files, tool output, web pages, answers or task reports;
  text that looks like a password, key, code, card or ID number is not kept ("Not
  remembered"). Every completed chat turn and task result is also indexed. Before each
  message, melo brings back what matches it from other chats (at most 3 notes and 2
  earlier turns, quoted, with the chat and day each came from), and the page shows "Used 1
  memory ▾" under your message to list them. It searches by words, and by meaning too
  when an embedding model is set up on the same server; an embedding that is slow (over
  1.5 s) or fails falls back to words. Chats now say what they remember accordingly, list
  your About-you and Preferences notes in your words, and can look up more with
  `search_memory`. Deleting a chat deletes its index entries. The notes and the index live
  in `~/.cache/localmelo/memory/` (`records.db`, `index.db`); `[memory] across_chats =
  false` turns it all off. Direct runs, workers, the terminal chat and the benchmark are
  unchanged. See [Memory across chats](docs/concepts/memory.md#memory-across-chats) ([#309]).
- Choose how memory across chats finds notes, and time every recall. The developer
  setting `[memory] retriever` (`-X memory.retriever=...`, listed by `melo -h all`) is
  `auto` (the default: words, and meaning too with an embedding model), `bm25` (words
  only), `embedding` (the same as `auto` with a model; `melo serve` refuses to start
  without one) or `llm` (words pick up to 20 candidates and the chat model chooses among
  them within 4 s, falling back to words alone on a timeout, an error or an unreadable
  reply, with no embedding calls). Every console turn with memory on now carries a
  `recall` object in its `turn.memory` event (the retriever, why it fell back, the
  milliseconds spent embedding, searching and choosing, the tokens of the choice, the
  candidates and the items used), also when nothing is recalled, and the stored turn's
  result repeats it as `memory_recall`. A failed query embedding is no longer tried a
  second time within the same search. See
  [How recall finds notes](docs/concepts/memory.md#memory-across-chats) ([#310]).
- The panel beside a chat. The web console (and so the macOS app) shows Melo on
  the right of the chat: his portrait, which rests, works, waits or cheers with
  what runs, one status line (`Connected · <model> on <server>`, `Working on 2
  things`, `Waiting for you`, `Model server not reachable` with **Open
  settings**), and the tabs **Activity**, **Approvals**, **Scheduled** and
  **Memory** (the last three say what they will hold; arrow keys move between
  them, and the last one used reopens). Activity lists what runs now, in plain
  words (`Reading notes.md`, `Running the tests`, `Waiting for your approval`),
  and what finished today, newest first, with a one-line result and the time,
  across chats, tasks, worker runs and terminal chats. A row opens in place to
  its live steps and the latest output, with **Pause** or **Resume**, **Stop**
  and the priority in words for a task, and **Review** for one that waits. From
  700 to 1180 px the chat list is a rail of initials; below 900 px the panel is a
  sheet behind a face chip in the header. Default mode drops the header's
  running chip; developer mode opens **Details** in the panel. The app's window
  opens at 1280 × 800. For clients, `status.items` of `GET /api/health` gains
  `activity` and `icon`, `status.focus` names the item a glance shows, and
  `GET /api/activity?since=` lists the day ([#308]).
- Two developer settings for a run that stops with nothing to show, and a cap on
  a reflection's lists. `--reflection-gate patient` (`-X
  agent.reflection_gate=patient` for one run; off by default) is the default gate,
  except that a reflection that chooses `continue` with a feasible, concrete but
  not new next step still starts the next attempt, once per run; a second one
  stops the run as before. `[agent] answer_on_stop = true` (`-X
  agent.answer_on_stop=on`; off by default, listed by `melo config keys` and `melo
  -h all`) makes a run that ends `gate_stopped` or `budget_exhausted` with an
  empty result ask the model once more, without tools, for the final answer from
  what it gathered, in the format the task asks for (`unknown` is allowed): the
  status stays, `result` is that answer, and the JSON record gains
  `"answered_on_stop": true`. With either on, a reflection keeps at most 8 items
  of at most 300 characters in its evidence, failed-hypotheses and avoid lists.
  Everything else is unchanged ([#311]).

- Browsing, part 1: melo can read web pages in Safari. With `[browser] driver =
  "safari-automation"` (off by default; `/browser on` in the terminal chat, or
  `-X browser.driver=safari-automation` for one run) the terminal chat and the web console
  get eight read-only tools: `browser_open`, `browser_read`, `browser_find`,
  `browser_follow`, `browser_back`, `browser_tabs`, `browser_wait` and `browser_screenshot`.
  They open `http` and `https` pages in Safari's own automation windows (the sign-ins of
  your Safari are not used) through the `safaridriver --mcp` server that ships with Safari,
  started on the first call; nothing is installed. Safari needs Settings > Developer > "Allow
  remote automation" once, and melo says so in words when it is off. Nothing clicks, types,
  signs in or posts yet. Each site asks once per chat ("Open a website?"; "allow for this
  chat" is the only lasting answer, `/new` and `/approvals forget` take it back, a redirect
  asks about where it lands before the page is read, auto mode still asks and `--yolo` asks
  nothing). A page reaches the model as one block between two markers that carry the same
  random number, with a system note that text between them is data and never instructions;
  it is cut into parts of about 2,000 tokens (a Chinese character counts three Latin
  ones), shows at most 60 numbered elements (`browser_find` searches the rest), refuses a
  ref from an older reading, shows no value of a password, one-time-code or card field and
  does not read a page with payment fields. Direct runs, `--brief` workers, saved tasks and
  the benchmark never get the tools. A provider can now mark its tools as a group that
  `tool_exposure = "retrieve"` offers on every step, add a note to the system prompt, and
  ask the run's approval policy about something it finds while a tool runs. See [Browsing
  with melo](docs/guides/browsing.md) ([#297]).
- Melo moves, part 2: the resting pet blinks and flicks an ear. The lazy picture the
  pet rests in (the default when nothing runs) blinks twice every 11 seconds, as the
  calm pictures do, and the calm lying picture also flicks its right ear every 13
  seconds: the ear, cut from the picture without it, turns about its base (12 degrees
  one way, 4 the other, back in 420 ms) while the head stays. The other lying pictures
  do not flick an ear: their heads are separate drawings that do not lie on the
  no-ear picture. Both are Core Animation in the app (no timer) and
  stop under the same conditions as the other moves ([#296]).

- Pictures that tools return reach a model that can see. A screenshot from an MCP server used to
  reach the model only as a line such as `[1 image: sha256:... image/jpeg]`. With
  `[agent] tool_images = "on"` (or `-X agent.tool_images=on` for one run) each planning request
  now carries the newest three to four pictures as `image_url` data URLs, in a `user` message
  after the tool results that names the call (Ollama: its `images` field). The oldest two lose
  their pixels together, so the prompt prefix a server caches changes once every two pictures; a
  picture already sent is not sent again; one wider than 1,280 pixels is scaled to a 1,280-wide
  JPEG once (macOS `sips`); each counts 4,500 characters toward the prompt cap (pictures a user
  attaches now count too). The default, `auto`, sends pixels only when the server says the model
  reads images (oMLX's `model_type` `vlm`, Ollama's `vision` capability), asked once at the first
  picture. `tool_images_keep` and `tool_images_retire` set the counts. A run whose tools return no
  picture sends the same requests as before. An MCP result's `structuredContent` is no longer
  sent as text beside a picture, so a server that repeats a screenshot's base64 there sends the
  picture, not the base64 ([#295]).

- The terminal's status bar says what the session uses and what runs beside
  it, not the model and folder again (the welcome and `/status` keep those).
  On a local server it shows the tokens sent and received, the context with a
  small meter, the prompt cache's share of the last request (the session's
  average after it) and the idle time since the last request: `↑ 3.0k ↓ 15 ·
  ctx 2.6k/32k ▮▯▯▯▯ · cache 61% (avg 72%) · idle 2m`. On a cloud API it shows
  the session's cost from prices you configure in `[pricing.<model>]` (no
  prices are bundled; tokens without one) and the tightest rate limit the
  last response's headers reported, with its reset: `$0.13 · ctx 100k · limit
  42% ▮▮▯▯▯ ↻3m`. When the gateway has tasks, a zone counts them
  (`2 running · 1 queued · P1 · ctrl+t tasks`), and Ctrl+T lists them: Enter on
  one offers pause or resume, a higher or lower priority, and stop (asked
  first), each sent as `melo task` sends it. The figures are coloured by kind
  (a good cache teal, a filling context amber then red, held work amber), a
  narrow terminal drops the title, then details, before any figure, and the
  bar redraws once a second only while it shows a clock. A list open above the
  bar now ends with a blank row ([#293]).

- Melo moves. It blinks, wags its tail and sleeps with a "z". The calm portrait of the app's
  first window, the console's calm portrait and its calm face beside older replies blink twice
  every 11 seconds (an eyes-closed patch fades in over the eyes for a tenth of a second); the
  pet blinks on its calm lying picture too. When a run is done the pet's tail swings about its
  base, four times, instead of the whole picture swaying; and a sleeping Melo sends up a small
  "z". The patches and the tail are cut from pictures the art model redrew, only where they
  differ from the originals, and `scripts/build_pet_assets.py` stops with a message if a redrawn
  picture does not lie on its original. All of it is Core Animation in the app (no timer) and
  CSS in the console (no script), and stops under Reduce Motion (`prefers-reduced-motion` in the
  console), in Low Power Mode, when the Mac runs hot, while the window or Melo is hidden or
  covered and while the screen sleeps. ([#292])

- Chats say who they are. A model that is told nothing answers "who are you?" with the
  assistant it was trained as. The terminal chat, the web console and the macOS app now tell
  the model one short paragraph first: it is melo, the assistant built into localmelo, an
  open-source agent that runs on your computer, works in your folder with tools and asks before
  it changes anything; it runs on the model you chose, served by the server named as the banner
  names it (`oMLX`, `Ollama`, an `OpenAI-compatible server`, or a cloud provider), on this
  computer or at the server's host; it says so when asked who or what it is, does not present
  itself as the model's own assistant, answers in your language, and answers a greeting or a
  question it can answer directly without tools. Switching the model (`/model`, the console's
  model menu) names the new one from the next turn. It is the new `AgentOptions.identity`
  (default `None`: direct runs, workers and the gateway's v1 sessions send no identity).
  ([#287])

- Pictures a turn makes show under its answer in the web console and the app. Ask for a chart
  and the chart is there: the PNG, JPEG, GIF, WebP and SVG files the turn created or changed
  in the chat's folder (found by looking at the folder when the turn starts and when it ends, so
  a file written by `python_exec` or `shell_exec` counts), and any picture the answer names by a
  path inside that folder, appear as a row of previews with the file name and size. Four show;
  `+N more` shows the rest. Select one to see it full size; a file that is gone says `No longer
  in the folder`. The list is stored with the turn as a new `turn.pictures` event, so a reopened
  chat shows its pictures too. The pictures come from a new route, `GET
  /api/sessions/{id}/pictures?path=`, behind the console's sign-in: it serves a file only when
  its real path (links followed) is inside that chat's folder, its name and its content are a
  PNG, JPEG, GIF, WebP or SVG, and it is at most 20 MB, with `nosniff`, `Cache-Control: private`
  and, for SVG, `Content-Security-Policy: sandbox; default-src 'none'; style-src
  'unsafe-inline'`, so a picture never runs as a page. It writes and lists nothing. Direct runs
  and the terminal are unchanged. ([#291])

- Melo the Akita puppy is now the face of the app's first window and of the web console, in the
  app and in a browser alike. The window that shows while localmelo starts has Melo's sitting
  portrait instead of the framed print: working while the server starts and signs in, calm once
  it is ready, with a drooping ear when it cannot start. The console's empty chat shows the
  same portrait, every reply carries Melo's face (the newest reply shows how its turn is going:
  working, wide-eyed while an approval waits, happy when done, a drooping ear when it ended with
  a warning or an error; older replies and a turn you stopped keep the calm face), the approval
  bar shows the wide-eyed face, and the sidebar, the sign-in screen and the tab icon (16, 32 and
  64 px) show the calm one. The pictures are still, so Reduce Motion needs nothing, and they
  keep their outline on both themes and at phone width. Faces are decoration with empty alt text
  where words already say the state; a finished answer with no work box above it names its
  face `Done`. The console's nine pictures (150 KB) are copies of the app's own, written by the
  same script (`uv run --with pillow python scripts/build_pet_assets.py --console-only` copies
  them again), and the two logo files they replace are gone from the page's assets. ([#275])

- Melo the Akita lies on your desktop when you put the macOS app away. Close the window,
  minimize it or hide the app and a small Melo appears at the bottom right, above the Dock,
  over other windows and on every Space, without taking the keyboard. It rests lazily while
  nothing runs, works, waits for your approval, is happy when a run is done, looks sad when
  one did not finish and sleeps in a different pose after a quiet minute, all from the
  health check the app already makes. Click it to open the window (or the chat that waits
  for you), drag it anywhere (it remembers its place on each screen), right-click it for
  Open localmelo, Hide Melo and Quit localmelo; Hide Melo is remembered and **Show Melo** in
  the menu-bar menu brings it back. It breathes, bobs when work starts, hops when it waits
  and wags when a run is done, with Core Animation only: all of that stops under Reduce
  Motion, in Low Power Mode, when the Mac runs hot, and when Melo is covered or the screen
  sleeps. VoiceOver reads its mood when Melo is focused; the menu-bar item still announces a wait, Melo stays silent so it is not read twice. The Dock icon now opens the window
  again after it was closed. ([#272])

- Melo the Akita puppy gets pictures, a mood policy and a place in `GET /api/health`, the
  first step towards a desktop pet. `macos/art/akita` holds 14 transparent pictures, each at
  1x and `@2x` (1.17 MB): nine for a pet (lying, resting lazily, three sleeping poses), five
  sitting ones for the starting view, small faces and favicon sizes, with a thin cream rim so
  the outline shows on a dark desktop and an `anchors.json` that lets one pose replace
  another without a jump. `scripts/build_pet_assets.py` makes them from the art masters
  (`uv run --with pillow python scripts/build_pet_assets.py --picks picks.json`, the same
  bytes every run), and the macOS build copies them into the app's `Contents/Resources/akita`.
  `PetPolicy.swift` decides what the pet will do: a failure beats a wait, a wait beats work,
  a run that just finished shows as done for 6 seconds or as a problem for 10, then Melo
  rests, and after a quiet minute it sleeps in a pose that turns every 3 minutes; it also
  says when the pet may move (not under Reduce Motion, Low Power or heat, not hidden) and
  where its panel goes on the screen. `GET /api/health` gains `status.recent`: how the
  newest finished chat turn, task, run or terminal turn ended (`done`, `failed` or
  `stopped`) and how many seconds ago, or `null`. The app does not draw the pet yet.
  ([#271])

- The web console shows what a file write replaces before you allow it. The approval card
  for a write of text over an existing file now asks "Replace notes.txt?", counts the lines
  ("5 lines removed, 1 added") and lists them as numbered − and + rows with two lines of
  context, as the terminal's panel does, with a secret read as `[redacted]`. A file that is
  not text, cannot be read or is over 200 KB, a change of more than 1,000 rows and a private
  key are told in a line, and the card keeps its old title with the file's size and the new
  text. `GET /api/sessions/{id}` and the task routes carry the comparison as `target.diff` of a
  pending approval; nothing else in the approval changes, and no answer works differently.
  The terminal's file-edit helpers moved to `support/sessions/edits.py` for it, and
  `support/cli/edits.py` re-exports them. ([#265])

- A first question about the folder needs no approval. The interactive UI and the
  gateway (the web console's chats included) have a read-only `list_files` tool: a
  folder's folders (`name/`) and files with their sizes, 1 to 3 levels deep (`depth`),
  200 entries and then `... and N more`, `.git` left out. It never reads a file and never
  enters a symlink (shown as `name@`), its path is fenced to the workspace like
  `file_read`'s, and it is a `read` tool, so "What is this project?" no longer opens an
  approval for a shell command. A direct run, `melo --brief`, the Python API's default
  mode and task workers keep the three built-ins they had, so their requests do not
  change; `--builtins list_files` or `all` names it. `/status all` counts five built-ins
  in the terminal. ([#259])

- The terminal says which files a turn changed. A turn that wrote files with `file_write`
  ends with one more quiet line under its footer, `Changed hello.py (+1), README.md (+3
  −1)`: the paths relative to the folder melo works in, each with the lines the turn added
  and removed in it, at most three names and then `and 2 more` (fewer on a narrow
  terminal, never wrapped). A file written twice counts once, as the turn found it against
  as it left it; a file written back as it was is left out; a new file shows its line
  count. A file with no counts to show (a new empty file, a file too large or not text,
  any file of a run with `--yolo`, where nothing read the old file first) shows its name
  alone. Only the turn's own `file_write` calls count: a write that was denied or refused
  (a path outside the folder) is not listed, and neither is a file that a shell command or
  a script changed, since melo never scans the folder. A turn that changed no file prints
  nothing, and developer mode shows the same line. Direct mode's stdout and stderr, JSON,
  exit codes and requests are unchanged ([#258]).

- Developer mode shows the plan of a task melo has run before on every turn. The row of
  past runs and the steps they took (`last time 6 steps, 2m10s: Run ls ✓ · Read app.py ✓`)
  showed on the first turn of a session only; in developer mode it now shows on every turn
  that has past runs to show, and the default mode keeps it on the first turn. Direct
  mode's stderr line, JSON, exit codes and requests are unchanged ([#258]).

- Experimental `lean_output` setting for fewer output tokens in direct runs (#183), off by
  default and with no flag: `[agent] lean_output = true` in the config file, or
  `-X agent.lean_output=true` for one run (`melo config keys lean` lists it, and `melo -h all`
  names the issue). A task given on the command line, `melo worker BRIEF` and their `json` and
  `stream-json` runs then send a system prompt whose last line, "Think step by step. Be
  concise.", is replaced by three short lines: no text between tool calls and a final answer of
  at most three lines; relative paths without a `sys.path` setup; read the files a task names
  before listing folders. With the default `reflection_gate flags` the reflection prompt also
  leaves out the five estimates, `progress_made` and `new_information_gained`, which the gate
  does not read, and asks for `best_effort_result` only with `done` or `stop`; the action and the
  three flags are still asked for, and a reply in the earlier long form is read as before.
  `--reflection-gate utility` keeps the full reflection prompt, and the compress modes keep the
  old system line. The interactive UI, the `melo ""` loop and the gateway ignore the setting.
  Without it every request is the one sent before. ([#255])

- The first task in the web console shows the folder melo works in, and lets you pick
  another. In a new chat the composer has a chip, "In localmelo-workspace", that opens a
  panel with the default folder, the folders of your recent chats, "Choose a folder…" and,
  in the macOS app, "Show in Finder". In the app "Choose a folder…" opens the system's
  folder panel; in a browser it opens a path field. The pick goes out with the first
  message as `workspace` in `POST /api/sessions` (a chat in the default folder sends the
  same request as before), and a chat's folder shows in its header ("In ~/notes";
  developer mode adds it after the model). Three one-line starters under the composer
  ("Summarize the files in this folder", "Plan a small project with me", "Explain a piece
  of code") replace the three cards. The app answers the page through one message handler
  that accepts only the gateway's own page, returns only a path picked in the panel and
  shows in Finder only a folder of the gateway's chats or one picked there. ([#248])

- The web console connects the model server that already runs on this Mac in one
  step. While no model is set up (or the configured server is down) and no chat
  is open, the welcome shows a setup card in place of the banner and the starters:
  "Found oMLX on this Mac" (or Ollama, or a model server) with its model and
  **Use this model**, which saves it, tests it and makes the welcome ready without
  opening settings; "A model server needs an API key"; or "No model server is
  running on this Mac" with Check again, Enter a server address and a link to the
  quickstart. With several chat models and none known to be loaded the card lists
  them, marks the loaded one and waits for a pick. The composer reads "Connect a
  model to start" meanwhile. `GET /api/setup/detect` is the route behind it: it asks
  the servers the first run asks (`LOCALMELO_DETECT_URLS`) and answers `state`
  (`found`, `needs_key`, `none`), the server's `name` and `backend`, its chat
  `models`, the `loaded` ones, the `suggested` one (`null` when the user must
  choose) and a `problem`. Saving from the console now writes an oMLX server as
  `omlx`, as the first run does. The connection settings dialog opens on a blank
  "Choose a server..." entry when nothing is saved, instead of a preset (MLC at
  `http://localhost:8400/v1`), and no longer shows the terminal advice
  (`run 'melo setup'`).
  ([#243])

- The macOS app's menu bar says when melo needs you, and the app has an icon. The
  menu-bar item is Melo's head (a template image the system tints for light and dark)
  with the number of running things beside it, and a dot beside the head while a chat
  or a task waits for your answer to a tool approval; it shows at the next 2-second
  health check. VoiceOver and the tooltip say it in words (`localmelo, 3 running`,
  `localmelo, waiting for your approval`), and the start of a wait is announced. The
  menu starts with what runs and, while an approval waits, "Waiting for your approval",
  which opens that chat or task; "Settings…" (Command-comma) opens the console's
  settings. `GET /api/health` gains `status.waiting`, the chats and tasks with an
  approval to answer (`sessions.pending_approvals` counted open chats only and named
  none). The build draws `LocalMelo.icns` from one SVG mark, `macos/mark.svg`, with
  `sips` and `iconutil` and names it in the plist, so the Dock and every dialog show it
  instead of the generic icon. The item's text title (`localmelo · N`) is gone. ([#241])

- One command builds and installs the macOS app on this Mac. `uv run python
  scripts/build_macos_app.py --install` builds the app, signs it ad hoc (no Apple
  developer account), puts it in `~/Applications/LocalMelo.app` and links
  `~/.local/bin/melo` to its launcher, then says in three plain lines what happened
  and what to do next. An earlier copy is replaced only when its bundle id is
  `org.localmelo.app`; another app of that name is refused. The link is made only when
  no other `melo` is on `PATH` (the checkout's own does not count), and an existing
  `melo` is never replaced. `--uninstall` removes the app and the link. The verified
  Python archive is cached in `~/Library/Caches/LocalMelo` and hashed on every use, and
  build tools print only when they fail. The bundled Python is compiled to bytecode
  before signing (the standalone archive has none), so `melo --version` from the bundle
  takes 0.29 s instead of 0.96 s (median of three runs) and nothing writes bytecode into
  the signed bundle, at the cost of 63 MB (120 to 183 MB). ([#238])

- Experimental `--brief-items on|off` (`[agent] brief_items`, off by default; #185). With
  `--brief`, the harness marks the brief's top-level numbered items (`changed`,
  `not yet` or `no file named`) from the files `git status` shows edited since the
  run started, and names the items with no edit yet in each reflection request and
  in the check two steps before the last attempt ends. It records a `brief_items`
  notice each time. Without the option nothing changes. ([#232])

- Experimental `cache_layout` and `window_messages` options for prompt-cache reuse
  (#183), off by default. `--cache-layout v3` (`[agent] cache_layout`), with the fixed
  prefix, appends a reflection once at the attempt boundary as an ordinary window
  message instead of rebuilding it as the last message of every planning request,
  keeps the reflection request and its reply in the window so the next planning
  request continues it, and asks an unreadable reflection again on the same prefix
  instead of as a text transcript; older reflection blocks stay in the history until
  evicted. `--window-messages N` (`[agent] window_messages`, 8 to 256, default 20) sets
  the working window's message cap, so the prompt-size fit can be the only evictor.
  The default requests are unchanged. Each request's `prefix` now also has
  `fresh_chars` and `request_chars` (the characters that no earlier request of the run
  held, and all of them), and `scripts/cache_report.py` reports the reusable share
  `cached / (prompt - fresh)` per run and pooled beside the share of all prompt tokens;
  `fresh` is an estimate. ([#228])

- What runs now, in the web console header and the macOS app (#213).
  `GET /api/health` gains `status`: `running` counts the console's chats with a
  turn in flight, scheduler tasks in `running`, and, from the CLI file under a
  live lease, direct runs and terminal chats; `tasks` is the scheduler queue the
  status row shows (`running`, `queued`, `held`, `top_level`); `items` lists
  what runs, at most 20. The console header gets a "N running" chip, hidden at
  0 and kept on a narrow screen, that opens that list with a link to each chat,
  task card or Workers detail. The macOS app's menu-bar title reads `Melo · N`
  while N sessions run, and its quit dialog names what would stop ("2 tasks and
  1 chat are running"). One health call adds about 0.4 ms on a store of
  1,000 sessions and 1,000 tasks: the task counts read only the live states by
  their index (`TaskStore.list_tasks(states)`). ([#227])
- Terminal chats in the web console, read-only and redacted (#213). The Workers
  view has a Worker runs / Terminal chats chip pair; `GET /api/sessions?kind=terminal`
  lists the chats of interactive `melo` sessions from the CLI file (never in the
  console's own `kind=chat` list), and detail, turns, `/events` and `/status`
  serve them by id, read-only. A chat is stored unredacted, so every text of one
  is redacted as it is read, as a worker run is when it is written: secrets in
  titles, turn text, answers and event payloads, the output of a call that read a
  secret file replaced, the run `config` left out. An open chat is followed while
  a turn runs under its terminal's lease, and again when the next turn starts.
  Every write route still returns 404 for a chat id. ([#226])
- Live section on the task card (#213). A started task's card in the web console
  shows the task's steps, tool calls and model text beside its approval card,
  folded like a chat. The card reads the task's session once and, while the task
  runs or waits for approval, follows its events from the last one read, drawing
  at most every 150 ms and only the Live section; a hidden tab closes the stream
  and a shown one resumes. The stream closes after one last read when the task is
  done, failed, cancelled, expired, interrupted or parked. A task that never
  started shows nothing live. No server change. ([#224])
- Follow worker runs live in the web console (#213). `GET /api/sessions/{id}/events`
  and `GET /api/sessions/{id}/status` now also serve a worker run of the CLI
  file, with the same frames, `after`, `follow=false` and `Last-Event-ID`. A
  following stream reads the file another process writes about once a second and
  ends when the run holds no live lease and its last event is sent. The Workers
  view reads an unfinished run once and follows that stream, appending new steps,
  tool calls and text without reading the earlier work again, instead of
  re-reading the whole run every 3 seconds; a hidden tab stops and catches up from
  the last event, and the status row shows on a worker run. Worker runs stay
  read-only: no write route accepts a worker id. `context_budget`, `note`,
  `inbox.accepted`, `memory` and `recall` events no longer show "Some recorded
  event types cannot be shown": notes, accepted messages, memory decisions and
  recalls draw as one muted line each. ([#221])
- Status row in the web console (#176, #213). A chat shows one line above its
  composer, and a task's card shows it under its facts: the same segments as the
  terminal's status bar (`cache 67% idle 3m`, `ctx 1.0k/8.0k`, `step 3`,
  `tasks 1 running 2 queued`), from the new `GET /api/sessions/{id}/status`, which
  folds the session's stored events with the terminal's own code. A server that
  reports no cached tokens shows `cache --`, never 0%. The row refetches at most
  once a second while events arrive and on the page's existing polls; a narrow
  width drops segments right to left. A worker (CLI-store) session id is 404 until
  the console can read those events. ([#218])
- Every direct run (`melo "<task>"`, `melo --brief FILE`) is now recorded as a
  worker session in `~/.cache/localmelo/cli/sessions.db` by default, so the web
  console's Workers view and the macOS app show localmelo's own runs without a
  flag. Opt out with `--no-save-session` or `[sessions] record_direct = false`;
  `--parent-run` no longer needs `--save-session`. The model's requests, stdout,
  the JSON result and the exit code are unchanged; a store that cannot be written
  prints one stderr line and the run goes on unrecorded. The Workers list shows
  runs that are still going, and an open unfinished run re-reads itself every
  three seconds ([#217]).

- Developer mode in the web console (#186). **Connection settings** has a
  Developer tab with one toggle, "Developer mode: show advanced settings and
  debug views", saved as the same `[ui] mode` the CLI reads (revision-checked
  like the other settings; a config written by hand wins on reload). When on, the
  details drawer gains an Events tab (the stored raw events of the latest turn),
  the Memory tab shows the agent state and the long-term backend, the Runs tab
  shows each run's counters (requests, usage and cache numbers), and the Developer
  tab lists the advanced and experimental options read-only with their tier and
  deciding issue. When off they are hidden, not removed. `GET /api/settings` adds
  `mode` and `advanced_options`; `PUT /api/settings` takes an optional `mode`.
  The default mode's drawer no longer shows the Runs tab's raw result fields.
  Nothing changes what runs. ([#209])
- A task that waits for a tool approval can be answered ([#208]). The task JSON
  carries `pending_approvals` (the shape a chat session uses), the console's
  Tasks view shows the same approval card and says "Waiting for your approval",
  and `melo-task show` and `list` print `Waiting for your approval: <tool>
  <target> (answer it in the web console, Tasks)`; `Reason:` reads `approval`
  instead of `not recorded`. Answers use the existing `POST /api/approvals/{id}`.
- The interactive status bar shows usage at a glance (#176 SL2): after the model
  and workspace, `cache 83%` (the last request's cached share, `--` when the
  server reports none, plus `idle 3m` for a local server), `ctx 12k/32k`, the
  run's `step`/`attempt`/`phase` while it runs, and `tasks` when a task store is
  reachable. Each segment hides without data; a narrow terminal drops them right
  to left. It is built from events already received (no request, no timer).
  `[cli] status_line = "off"` keeps the old bar; there is no flag. ([#201])

- Default mode and developer mode (#186). Every `melo` option has a tier (user,
  advanced or experimental) in one table, `support/option_tiers.py`, copied in the
  CLI reference. `melo --help` now shows the 15 user options and one line pointing
  at developer mode; `melo --developer --help`, or `[ui] mode = "developer"` in
  the config file, shows every option under Options, Advanced and Experimental
  headings, each experimental one naming the issue that decides it. The mode
  changes only what help prints: every option still parses and works in both
  modes, so scripts and the benchmark are unaffected. `--window-eviction`, which
  was hidden outright, has help text and shows in developer mode. A `[ui] mode`
  that is not `"default"` or `"developer"` warns on stderr and shows the default
  help. ([#199])
- `/status` in the interactive UI gains a usage block, built only from events the
  session already received (no request, no polling of the model server): the
  last request's cached share and the session's average over the requests whose
  server reported `cached_tokens` (`--` when it sent none, never 0%), for a local
  backend the idle time since the last request (no countdown; a cloud backend's
  cache lifetime is not tracked yet), the last measured prompt tokens over the
  input limit when a `context_budget` event exists, the current or last run's
  `run.progress` facts (no percent), and `no task store in this session` where
  the session cannot reach one. Each row says in a few words why it has no value.
  #176 SL1. ([#194])
- `--window-eviction slide|chunk|boundary` (and `[agent] window_eviction`), an
  experimental option for the prompt-cache A/B: `chunk` drops the working-memory
  window in chunks of eight messages and the prompt-cap fit down to 70% of the
  cap, so the retained history stays the same for several steps; `boundary` does
  the same and also drops a finished attempt's turns when the next attempt starts
  after a reflection. The default, `slide`, is unchanged, and non-default modes
  add `window_eviction` to the JSON result's `config` ([#191]).

- `--reflect-tool-guard on|off`, an experimental option hidden from default help
  (no config key): with `on`, a reflection request offers the tools of that step's
  planning request and ends with `Answer with the JSON object only; do not call any
  tool.`, so a model that continues the window's tool turns answers in text and the
  server's prompt prefix holds. The default, `off`, is unchanged; `on` adds
  `reflect_tool_guard` to the JSON result's `config`. The `reflection` event has a new
  `unreadable_reason` (`tool_call`, `invalid`, `no_json` or `null`) that says why the
  first reply held no reflection, in every mode ([#210]).

- Each `assistant.message` event's `usage` now carries `cached_tokens` and
  `reasoning_tokens` when the server sent them (also on a streamed reply
  with `stream_options.include_usage`; a missing key means not reported), and
  a new `prefix` object says how much of the best earlier request of the run
  (planning or reflection) this one repeats, which request that is
  (`against`) and the role of the first message that changed, so a
  prompt-cache miss can be traced. The `reflection` event gets the same
  `cached_tokens`/`reasoning_tokens` in its `usage` and a `prefix` for its
  first request. Request bodies and the JSON result are unchanged.
  `scripts/cache_report.py LOG` prints the cache use per request (plan and
  reflect) from a `stream-json` log, with totals by kind. ([#190])

- The console has a **Tasks** view (`#/tasks`, `#/tasks/<id>`): the task list, and a card with the state, the saved recovery card, the evidence status, Resume (with a confirmation dialog only when the server asks for one), Stop and "Start again as a new task" for a started task without a recovery card. ([#189])

- `--fixed-prefix` is now `fixed-prefix-v2`: the latest reflection moves to the end of the
  planning request, in a final user message after the window, so a new attempt changes only
  what follows the history the requests share; and a reflection request starts with the bytes
  of the latest planning request (the same system text, the same `tools` array, the same
  turns, then the turns since) and adds one final user message with the reflection prompt.
  A reply that calls a tool instead of answering is asked again as before. The default path
  is unchanged. ([#192])

- Interactive `! COMMAND` runs a command in the user's own shell from an idle
  prompt, with the terminal handed over to it (Ctrl+C and Esc go to the command,
  never the agent) and restored once afterwards. It is not a sandbox. Its output
  stays in the terminal, out of model context, history and stored events;
  `/attach !` attaches a bounded, redacted snapshot of the last output
  (without concealed text or anything before a screen clear) to the next
  message, after showing the exact snapshot and asking `[y/N]`. `!` alone shows help and `\! text` sends a literal leading `! `.
  melo exiting ends only the command's own process group. [#180]

- Controlled task Resume: `POST /api/tasks/{id}/resume` (also
  `/api/v2/tasks/{id}/resume`) queues a recovered task only through the
  evidence decision table. Every create records a durable never-started proof
  before any launch, so a proved never-started task first-starts without a card
  or confirmation; a started task resumes without confirmation only with a
  trusted complete capture whose recorded groups are verifiably gone. A verified
  boot change counts as gone only on the host that recorded the group: captures
  now persist a durable host id (the macOS hardware UUID, kept only as a SHA-256
  digest, or `/etc/machine-id`), and another or unknown host asks for
  confirmation. Anything else returns 409 `confirmation_required` with a
  five-minute, single-use, in-memory token bound to the task, execution,
  evidence and CAS generation. Missing card, invalid workspace, stale identity,
  terminal state and unresolved recovery operations block Resume; Stop always
  wins and cancels a queued resumed turn before any provider or tool. An
  operation ID
  names one operation: a retry returns the committed row whatever its payload,
  without a second turn. A Resume committed before a restart is re-admitted on
  startup when its turn never started, even if the task had reached running, and a retry after that turn started
  returns 409 instead of replaying it. PATCH stays fenced for recovered work.
  ([#177])

- Direct workers can read steering instructions from an inherited pipe with
  `--inbox-fd N`: the same lines, size limits and acknowledgements as the file
  inbox, but the read end is close-on-exec before tools start. Python, shell
  and MCP subprocesses get no copy, so a tool cannot write the channel; use it
  for untrusted tasks ([#169]).
- An unsigned arm64 macOS development bundle wraps the local web console in a
  native window and menu-bar host. It owns its gateway child, checks the
  instance before sending any login code, binds the code to the launch instance
  and checks page responses, re-signs in after a token rotation, and offers
  "Sign In Again"; the check is not authentication ([#167]).
- Interactive terminal live streaming is opt-in with `[cli] stream = true` and a
  per-model `[cli.reasoning_tags]` hint (`explicit`, `closing_only` or `none`),
  with `/stream on|off` for later turns. Released, redacted text shows in the
  live area; provisional text that becomes a tool call, retry, failed run or
  stop is labelled, and the final answer prints once. A provider without
  streaming falls back to the complete reply; direct mode is unchanged, and an
  invalid `[cli]` value warns and falls back instead of failing a run. `/new`
  resets `/stream` to `[cli] stream`. ([#175])
- Direct workers can read an explicit append-only inbox at planning boundaries,
  with per-line acceptance events and redacted saved history. An inbox inside the
  workspace is refused as `setup_error` before any model call; a slow event
  reader is waited for rather than dropped; input errors are `setup_error`, not
  `crashed`. ([#164])
- Experimental Python opt-in typed memory assertions: conservative evidence
  checks (timezone-aware `timestamp`/`date` records), ordered same-task updates
  (record time or per-task user order), cancellation markers, exact-duplicate-only
  retention and wall-clock labelled recall with source-dated titles ([#162]).
- `melo-task` creates, inspects and controls gateway tasks, with authenticated
  loopback requests and an optional event follow stream that prints one readable
  line per event and a terminal snapshot. ([#159])
- Wire process capture into task launches: the gated launcher records the
  process group and start time before releasing a child, tool-provider starts
  and arbitrary-code tool entries mark capture incomplete, and a task-store
  outage retries for up to 10 seconds before refusing that one call. Resume
  stays fenced and nothing reads the evidence yet. [#166]
- Scheduler component v4 adds dormant process capture state, identity checks
  and a gated subprocess primitive; worker behavior and Resume are unchanged. [#158]

- Interactive terminal text attachments: `/attach PATH`, `/attachments`,
  `/detach N|all`, and explicit `@` file-completion selection stage bounded
  workspace snapshots. Best-effort redaction and cleaning require confirmation
  when they change a snapshot; saved snapshots support retry without rereading
  files, but resume budgets and retained windows can omit a turn. No image
  attachments; direct mode is unchanged ([#155]).
- Startup recovery classifies task rows without replay, retains pending pause
  and reconciles execution-linked terminal results. Recovered tasks with
  uncertain history remain fenced until controlled recovery is wired; Stop is
  available. Ordinary task creation does not yet establish first-start proof. [#152]
- Scheduler component v3 adds dormant recovery provenance, guarded durable
  intents and receipts, and process identity records. Interrupted recovery
  retains pending pause intent; explicit legacy adoption records orphan-risk
  acceptance. Process capture completeness and controlled Resume are deferred. [#147]
- Experimental opt-in provider streaming and OpenAI-compatible SSE assembly;
  no CLI or agent uses it yet ([#145]).
- With a trusted reasoning-framing declaration, the streaming display policy can
  release redacted complete lines; unknown framing still buffers the reply ([#153]).
- Direct workers can read a brief file and emit a versioned JSON result or an
  NDJSON event stream. ([#143])

- Opt-in fixed system prefix with a local-date snapshot and schema-change telemetry;
  dynamic recall and reflection remain labelled data ([#142]).
- Opt-in bounded, pattern-redacted captured tool-result history, separate from the
  model preview, with capture and acquisition omission metadata ([#144]).
- Heuristic notes retain a bounded set of explicit English negative and Chinese
  residence/cancellation assertions, and item attribution rejects conflicting
  event times, rejects opposing detected polarity and requires the full negative
  assertion ([#137]). Similar opposing assertions are retained as conflicts;
  structured polarity-aware supersession remains off by default.

- Worker park integration fences held provider and approval delivery, persists
  pause intent and evidence-based cards, releases cleanly parked worker slots,
  and re-asks withdrawn approvals on resume. Failed park writes retry and can
  interrupt with a card; unavailable storage keeps the fence closed, and failed
  cleanup quarantines the slot until restart. Restart replay and accounting are
  deferred. [#135]
- Opt-in context fitting shortens request copies and retries one classified overflow without rerunning tools; impossible requests end with a specific status ([#131]).
- Read-only worker history in the web workspace, with server-side filters,
  paging, recorded transcripts and explicit refresh for unfinished runs. ([#130])

- Opt-in context-window request accounting reports estimated and measured planning/reflection
  tokens in events and run results; it does not prune or enforce an output cap ([#126]).

- S3b.1a foundations: scheduler component v2 adds a durable pause priority
  snapshot and atomic v1 migration; durable generation and operation IDs guard
  transitions and identify committed receipts. The internal synchronous gate
  `try_park` fences an eligible pending call and cancels only its bound turn.
  An atomic component event transaction supports `ParkStore` hold/resume/park
  records, including the current pending-pause flag. Guarded terminal retries
  return their matching committed record. An evidence-only card builder derives
  bounded, redacted cards from the task spec and persisted events. Worker API
  pause behavior remains as in #119; there is no worker hold/park integration or
  new API in this slice.
  S3b.1b follows ([#125]).

- Workspace settings gain connection testing and an immediately applied theme;
  a session details drawer shows tools, grant revocation, memory and runs.
  Store, model and stream banners no longer replace other notices. ([#124])

- Optional direct worker-session recording and dsh JSONL import, with a
  read-only, filterable and paginated gateway worker-history API. [#123]

- Workspace approval cards with a 500 ms guard, explicit allow/deny choices,
  and a jump-to-approval control; failure and stopped states with retry,
  continuation when eligible, copy-error and details; older-turn paging;
  revision-checked connection settings; and per-tab outbox recovery. Session
  update events also carry a display-safe workspace label. ([#122])
- A documentation inventory, bilingual style guide and experimental offline
  preview builder lay the groundwork for a later site migration; published
  pages and existing URLs are unchanged ([#120]).

- S3a.2 adds `/api/tasks` creation, inspection and control and in-process
  workers sharing the console's scheduler gate. Creation is idempotent with
  `request_key`; workers disable memory, omit `shell_exec`, and have at most
  four resident at once. On startup, surviving nonterminal rows fail with
  `gateway restarted; recovery arrives in S3b`; full recovery and accounting
  are deferred, deadlines are rejected, and an invalid workspace fails at run
  or resume. Pause takes effect at the next model call; running tools finish
  and answered pending approvals wait until resume; a task paused before
  start starts no worker until resumed. Resume restores the pre-pause priority,
  or computed P2 if the pause log is missing (including a task created or set
  at P5) or pruned by retention; an unpaused task returns 409 `not_paused`.
  Retention discards old terminal gate identities while keeping their rows.
  PATCH rejects `first` until S6b
  ([#119]).

- Reasoning and recall options now work in the interactive UI, including
  `--reasoning-field` and `--retrieval-extra-body`; explicit flags override
  `[agent]`/`[retrieval]` defaults. Console sessions honor the configured
  reasoning mode (default `legacy`) without changing legacy `/v1`. Session
  replay uses the retained window; pairs resume without old thinking,
  while checkpoint resume can replay retained, redacted thinking using the
  current process's mode. Scheduler wrappers preserve replay capabilities;
  rejected mode switches keep the old Agent and MCP state and close new
  providers; successful switches retain the RAM-only notes/drafts.
  Explicit-mode requests also remove legacy inline thinking
  copies after resume without changing the stored window ([#115]).
- `[agent] tool_env` names extra environment variables inherited by
  shell/Python tools in config-backed direct runs, the UI and console.
  Names are validated and missing variables skipped; only names are saved.
  SSH-agent and proxy access are explicit capability grants; defaults and
  `--base-url` runs/sessions expose nothing new ([#115]).

- The gateway's `sessions.db` can hold the scheduler's `scheduler` component
  version 1: `tasks`, `task_deps`, `priority_events`, `questions`,
  `sched_rules`, `sched_log` and `idle_windows` are created in one
  transaction in that file only, `PRAGMA user_version` stays the session
  schema's, and a newer component is refused. A task and its `kind="task"`
  session are created together behind `UNIQUE(origin, request_key)`, so a
  repeated request returns its task and a reused key with a different
  payload is a conflict; actual level changes write `priority_events` and
  30-day retention clears the log and idle windows.
  The task boundary normalizes the `TaskSpec`, pins a user-selected level
  only through a trusted adapter, and resolves a worker's folder under the
  user's home or `[gateway] workspace`; a non-user task with no folder gets
  its own child workspace. Only `user` and
  `user_rule` origins get remembered approvals, unmarked MCP tools and session
  grants. `ChatGate` can register a worker task, so its calls are logged as
  `session_kind="task"`, its P0 is the user's single task P0, and only the
  runner ends it. S3a.1 supplied the store and task policy boundary ([#113]).

- The workspace page on API v2. `http://127.0.0.1:8401/` now talks only to
  `/api`: the session list with its machine snapshot, one chat's turns, its
  event stream (SSE, reopened from the last applied event id), keyed sends
  that the store is asked about before `Retry` is offered, stop, close, the
  level chip and the typing heartbeat. A refused or unknown send goes back
  to the chat's draft, so a reload keeps it, and `#/s/<id>` opens its chat
  even when the list's first page does not hold it. A turn reads as the
  terminal's: the
  user's message, a work group with one card per tool call (`Running`,
  `Waiting: <held>`, `Needs approval`, `[ok]` with a non-zero `exit`,
  `blocked` with the approval tag, `not run`, `Interrupted: may have partly
  run`), reasoning and reflections folded away, notices, the live row
  (`step 4 · attempt 2 · Verify (cycle 1) · tests 3/8`, and on a first turn
  the routine's `last time …` facts) and the answer with its duration,
  steps, tokens and `Copy response`. That needs the S2 facts on `/api`:
  `level` and `held` on a session, `machine` on the list,
  `POST /api/sessions/{id}/level` and `POST /api/presence` on the one gate
  both surfaces share, and a console turn sets the gate's turn facts as a
  chat's message does. The page is plain ES modules with no build step
  (`workspace.js`, the new `turns.mjs` holding the event reducer and the
  terminal's wording, `markdown.mjs`). The follow-up adds approval cards,
  the failure block's actions and the settings dialog.
  `/` and every `/v1/*` route keep their behaviour ([#111]).

- API v2 settings, model listing, guard body limits and unconfigured gateway
  startup: `GET /api/settings`, `PUT /api/settings`,
  `POST /api/settings/test`, `GET /api/settings/key-status` and
  `GET /api/models`; `[gateway] workspace` in `config.toml` (validated by
  `Config.validate()`, read by console sessions, and written only when
  non-empty) and atomic `config.save()` replacement via `os.replace`
  (preserving file permissions and following a symlinked `config.toml`);
  `onboard.backend_defaults()`; a 1 MiB request body cap (`413 too_large`),
  `Content-Type: application/json` enforcement on non-empty `/api` bodies
  (`415 unsupported_media_type`), UTF-8/lone-surrogate validation, the
  console's Content-Security-Policy on `/api` and baseline security headers
  on every response; and `melo --serve --allow-unconfigured` so the gateway
  can boot before a backend is configured ([#106]).

- Terminal session commands: `/title [text]` shows or sets a title, including
  before the first message; `/undo` forgets the latest finished turn and
  keeps its typed history without reversing file changes (a committed undo
  reports success even after Esc, and `/retry` re-runs that undone turn
  without removing the previous live one); `/usage` shows
  recorded work and totals including undone turns, with missing counters
  left unknown and a narrow-terminal layout; `/memory` inspects the copied working window and explains
  its actual restore mode. `/copy [turn]` copies stored, redacted Markdown
  without thinking, preserving indentation and newlines, via bounded turn
  reads and macOS `pbcopy` or an OSC 52 request when
  remote, unsupported locally or on other platforms. Clipboard requests
  only happen on an explicit command; direct runs are unchanged ([#101]).

- `--workspace DIR` in direct mode: a task run keeps its file tools inside
  the folder and starts `shell_exec` and `python_exec` there. Without the
  flag a run keeps the current folder, exactly as before; `~` is expanded
  and the path is resolved like the interactive UI's. `melo ""` and
  `--serve` refuse it, and a path that is not a folder exits 2 with one
  line ([#98]).
- A remember request (`Remember that …`, `记住…`) now also stores the fact
  it carries after the marker, admitted by `user_request` beside the run's
  episodic digest, instead of only the digest. The fact is classified as
  `semantic` unless it is first-person (`profile`), and the redaction,
  meta-noun and question rules apply to it; a to-do such as "remember to
  buy milk" still stores no fact ([#100]).
- R6-lite's typed long-term store: `long_term.db` migrates in place to
  `user_version` 1 with `kind`, `source`, `status`, `admitted_by`,
  provenance, task/session ids, links and uses beside the text and vector,
  plus a `decisions` table. A candidate is admitted only by a rehearsal
  (restatement, a successful call it acted on, a cited note), a user
  request or statement, or `--admit-on-outcome on` for the final digest;
  the policy picks `NEW`, `UPDATE`, `RELATED`, `CONFLICT` or `IGNORE`, and
  every stored text, provenance included, passes the candidate boundary and
  `redact_secrets` (with its task text, paths and commands bounded). A long
  entry's llm catalog line carries a bounded text snippet, a BM25 row is
  exempt from the cosine floor, two items of one note stay two entries, a
  restatement merges the new note's records and supersedes its candidate, a
  newer file keeps a legacy run alive with one WARNING, a user signal needs
  a declarative, non-meta sentence, `call me` is a profile, and the three
  store modes accept and ignore model flags.
  `--keep-decision-text` keeps a decision's text beside its hash, and the
  `[memory]` table saves the outcome default. The `start` recall reads
  active entries of other tasks of every kind, by cosine or BM25, with
  `long:<entry id>` doc ids; `memory_stats` gains `admitted`, `written` and
  `ignored`, and a `memory` event reports each decision. `melo-retrieve`
  gains `--export-store`, `--export-decisions` and `--route` for the
  bench. Legacy requests, JSON line, window, `tasks`/`steps` and search
  order are unchanged; an existing `long_term.db` migrates with its rows
  intact. ([#97])
- Resume saved terminal chats with `melo -c`, `melo -r [ID|TITLE]`,
  `/sessions [filter]` or `/resume [ID|TITLE]`. The inline picker filters
  eligible chats by folder or across folders before the startup banner,
  which names the chosen chat; the recap explains what
  was restored with `[sessions] resume_mode = "pairs"` (default) or
  `"window"`. Resuming keeps the current model/workspace and clears
  approval grants, while retaining `/verbose` and `/thinking` preferences.
  Changes in another terminal are reloaded before the
  next turn; a busy or deleted session keeps the draft. `/retry` preserves
  a resumed turn's typed text, `/expand` reads stored tool output across
  event pages, and quitting prints a saved session's resume command.
  Direct runs keep their legacy parsing and output ([#95]).
- Recall in the loop: `--retriever auto|off|bm25|llm|embedding`,
  `--recall-at`, `--recall-k`, `--recall-kinds`, `--recall-placement`,
  `--recall-min-score` and `--retrieval-extra-body`, plus the `[retrieval]`
  config table and the `recall` tool. The compress modes recall the task's
  notes, drafts and thought chunks, and earlier tasks' long-term entries,
  at `start`, `attempt` and `step`: a frozen boundary block on the task
  query, one sticky block after a turn's tool results (or in its own user
  message), and the model-requested tool. `auto` uses the embedding
  retriever when configured and otherwise `llm` at `start`/`attempt` and
  `bm25` at `step`/`tool`; every failure falls back to `bm25`. Blocks are
  delimited, framed as data, checked by the candidate boundary and bounded
  (9,500 per turn, 12,000 live, 4,000 boundary); the system prompt gains
  one static data note. Each block is recorded once in
  `llm_calls.recall_json` with its hits' `origin_task_ids`, and in the
  `recall` event and `memory_stats.recall_log`. A step's query carries the
  latest note's plan, what it found and its open questions and outcome, and
  its candidates include a superseded note whose turn the window no longer
  shows, so a later attempt's earlier thinking stays reachable; such a line
  names its attempt and is marked `(superseded)`, and `recall_json` records
  each hit's `stage`.
  `memory_stats.recall_empty` counts a triggered recall that found nothing.
  `history.db` is schema
  version 2: a v1 file gains the `recall_json` column in the same
  transaction, and the column is `NULL` in `legacy`. `legacy` keeps its
  requests, JSON line, window and `tasks`/`steps` rows byte-identical; the
  new flags, the block and the data note are off there (and `drop`/`replay`
  recall only when a retriever is named). The JSON line gains
  `config.retriever`/`retriever_effective`/`recall_*` and the recall
  counters ([#94]).
- Console session details over API v2, and the scheduler gate in front of
  a console turn: `GET /api/sessions/{id}/tools` lists each tool's source,
  whether it asks first and whether the session allows it, `/memory` the
  conversation the gateway holds (thinking stripped, every text redacted,
  text cut at 2,000 characters) and `/runs` each finished turn's `RunResult`
  without its answer, while `DELETE /api/sessions/{id}/grants/{tool}` and `/grants`
  take a session grant back. Every model call of a console turn now waits
  for its slot under `[scheduler]`, like a workspace chat's: the console
  shares the gateway's one `ChatGate` and wraps the Agent's providers
  with `GatedLLM`/`GatedEmbedding` through `build_session_agent`'s hooks,
  so the call log records the session's id and `lock_gate = "wait"` holds
  a turn while a foreign GPU lock is present. A create now makes its
  session row before it builds the Agent, because the gate needs the id;
  its key still lands in the create scope, so a client without the id finds
  it, and a create whose Agent or first turn never came up, or was
  cancelled, deletes the row it made. Console runners ask each turn for
  `run.progress`. `/`, the workspace page and every `/v1/*` route are
  unchanged ([#92]).
- Console turns over API v2: `POST /api/sessions` (with an optional first
  turn), `POST /api/sessions/{id}/turns`, `/retry` and `/turns/{id}/cancel`,
  the two cancel-by-key routes, `POST /api/approvals/{request_id}`, and
  `PATCH`/`DELETE` of a session. A turn starts in the background and its
  events arrive on the session's stream: `approval.requested` waits for an
  answer given through the API, and the turn finishes `turn.finished` with
  the result read from its row. A `client_request_id` makes a create or a
  turn safe to repeat, one turn runs per session (409 `session_busy`), and
  a stop during an approval wait withdraws the card. The console keeps the
  `ApprovalBroker` and its `RememberingPolicy` through `runner.observe`, so
  the approvals a console turn asks show on the stream. A console turn
  does not pass the scheduler gate yet; that arrives with the tools and
  grants change. `docs/api.md` gains the routes and a tested Python script
  that creates a session, follows the stream, answers an approval and reads
  the answer; every request string must be valid UTF-8 (a lone surrogate
  is 400 `invalid_request`), a session's workspace is stored as its real
  path, and `/`, the workspace page and every `/v1/*` route are unchanged
  ([#91]).
- Reasoning notes for the compress modes: `--reasoning compress` and
  `compress+replay`, `--note-writer auto|heuristic`, and the `[agent]`
  `note_writer` key. `compress` sends no reasoning; instead each reply is
  compressed into a short note (what it planned, found, decided and left
  open, plus what its tools returned), with its code drafts, in a RAM-only
  note store beside the working window. A pending targeted draft is pinned
  once after its turn's tool results; a successful later write of 60% of
  its lines marks it written. Notes are rehearsed when a later note
  restates them, a successful call acts on a path or command they name, or
  a reflection cites them, and each attempt gets a digest from its rehearsed
  notes. `compress+replay` also sends thinking as `replay` does; a backend
  that requires replay runs `compress` as `compress+replay` and the JSON
  line reports that. Every document passes the new
  `check_memory_candidate` boundary (redact, sanitize, cap) before it
  reaches the window or the store. Nothing is recalled from the notes yet
  (retrieval is a later release), no notes table or schema change reaches
  `history.db`, the explicit modes stop writing raw step summaries and
  promoting reflections to long-term memory, and a session turn's `note`
  column gets the final digest. Legacy is byte-identical, goldens included
  ([#85]).
- The gateway opens its console session store and serves API v2 reads under
  `/api`: health with the versions, limits, store state and a cached
  model-server check; the sessions list and one session, a pending
  `file_write` approval's `target` and each session grant's `granted_at`;
  a session's turns with their events; a Server-Sent Events stream with
  `Last-Event-ID` and one-shot catch-up; image blobs whose CSP merges into
  the gateway's framing rule; and submission lookups by `client_request_id`.
  The console holds each session's Agent, keeps its conversation when an
  idle MCP session or the open cap closes it, and resumes it on reopen.
  When the store cannot open, health still serves and the reads answer 503
  `store_unavailable` while the gateway retries the open every 5 s. `/api`
  is loopback-only, carries the one error envelope and `Cache-Control:
  no-store`, and is additive-only and documented in
  [docs/api.md](docs/api.md); `/` and every `/v1/*` route are unchanged, and
  no turn starts yet ([#84]).
- A reasoning mode for direct runs: `--reasoning legacy|drop|replay`
  (default `legacy`), `--reasoning-field auto|reasoning_content|reasoning`
  and `--replay-turns N`, or the same `[agent]` keys in `config.toml`.
  `legacy` is byte-identical to before everywhere, including a provider
  built from the config by the gateway, a session or the Python API; `drop`
  never sends reasoning, and `replay` sends it on the assistant turns in
  scope (by default only the current task's) under one key, with a
  40,000-character per-turn cap and strip-first eviction. A backend whose
  `requires_reasoning_replay` is true gets every assistant turn in the
  window and keeps the final answer's thinking with `record_answers`. The
  JSON line reports the mode, `usage.reasoning_chars`, and the
  `memory_stats` replay counters outside `legacy`. The interactive UI
  refuses `--reasoning-field` until R2b wires it ([#78]).
- Diffs in the approval panel: a `file_write` over an existing file shows
  the change (`+a -b lines`, removed lines in red and added in green),
  a new file its line count, and a file too large or not text says why.
  `/approvals` shows the mode and this session's grants and switches
  between `ask`, `auto` and `deny` (or forgets one tool's grant). The
  terminal UI accepts `--mcp-config`, and the config file's top-level
  `mcp_config` key names an MCP servers file the UI loads (direct mode
  never reads it); `/mcp` lists the servers and their tools and says why
  MCP is off after a failure, `/status` shows the model and server, the
  session, the tools, memory, limits and files, and `/retry` runs the
  newest turn again. The hints name them where they apply ([#75]).
- Setup can add the embedding model that long-term memory needs. The first
  run offers an embedding model the chat server already lists, the wizard
  appends such models as options `6+`, the banner names the configured
  backend and model (or the flag that adds one), and a stored vector from
  another model is skipped instead of failing recall. A new
  [guide](docs/embedding.md) covers Ollama, oMLX, vLLM, SGLang and MLC
  ([#74]).
- The terminal and the web console share one set of words for a session.
  `support/sessions/display.py` re-exports `split_thinking` and adds the
  status table `STATUS_TEXT`, keyed by status and by `(status, cause)` and
  read through `status_text` (each with its word, severity and title), and
  `approval_tag`, which gives the short tag and long text of how a call's
  approval ended. `build_session_agent` takes optional `wrap_llm` and
  `wrap_embedding` callables, called with the providers it builds before
  the Agent's `Chat` is made, so a surface can put its scheduler gate in
  front of every model call; with neither, the builder is unchanged. The
  terminal reads its status severities from `STATUS_TEXT`, and its rendered
  lines and goldens are unchanged ([#72]).
- The web workspace's chats go through the scheduler: each chat call is
  admitted at once or held according to `[scheduler] policy` (`off`
  by default, which only records), each embedding call is recorded and counts
  in flight but is never held, a chat's level chip (P1-P4), the focused
  chat's lane, the server's `n_srv` slots and the `r_f`/`n_fg` reserves. A
  held chat shows `Waiting: <reason>` in the page, and every call leaves one
  schema-1 line in `~/.cache/localmelo/scheduler/runs/<UTC>/calls.jsonl`.
  Direct runs, the CLI and `/v1/agent/run` are unchanged and gated nowhere
  ([#69]).
- The gateway reads the shared GPU lock and `pmset -g batt`, records both in
  `machine.jsonl` beside the call log, and shows the holder and the battery
  in the workspace header. `[scheduler] lock_gate` defaults to `record`,
  which changes no request; `wait` holds every workspace call, embeddings
  included, while a foreign lock is present, and shows
  `Waiting: Reservation`. `lock_self` is an exact holder (or `holder:run`);
  unreadable and stale locks are foreign, and localmelo never writes,
  renames or removes a lock. Direct runs, the CLI and `/v1/agent/run` are
  unchanged and read neither source ([#76]).
- Tools have a permission tier, and an approval policy asks by tier rather
  than by a fixed list of names. The built-ins keep their tiers (`file_read`
  read, `file_write` write, `shell_exec` and `python_exec` exec), and a
  provided tool's raw MCP `annotations` are kept on `ToolDef.annotations`
  and read cautiously: `openWorldHint` gives network, `readOnlyHint` gives
  read unless `destructiveHint` contradicts it, and anything else gives
  write. So an unannotated MCP tool now asks in a run with a policy, while
  `require_approval` only adds caution. The terminal approval panel names
  the tier in its header, `ToolInfo.tier` and `ApprovalRequest.tier` report
  it, and `--yolo` still means no policy, not read-only ([#66]).
- Memory can be forgotten from the Python API. `await hippo.forget(task_id)`
  deletes that task's `tasks`, `steps` and `llm_calls` rows and the
  long-term entries whose metadata names it (its reflections), and
  `await hippo.forget_all()` empties history and long-term memory, step
  summaries included. `Hippo` has the facade; `SqliteHistory`,
  `SqliteLongTerm`, `History` and `LongTerm` have the two methods. A forget
  runs in one `BEGIN IMMEDIATE` transaction under the write lock, overwrites
  what it deletes and truncates the WAL, and leaves working memory,
  procedural memory and sessions alone. There is no command or UI yet
  ([#64]).
- A session runner can ask its turns for progress events, and the progress
  line's past facts are available alone. `SessionRunner(..., progress=True)`
  asks each turn's run for `run.progress` and stores it with the turn; the
  flag stays on the runner, `replace_agent` included. `run_turn` and
  `start_turn` take `routine="NAME"` and pass it to the run only when
  given. `localmelo.support.progress_line.format_past(payload, width)`
  gives exactly the direct-mode line's past-facts-and-items segment, with
  the same wording and width rules. A session turn emits `run.progress`
  only when its runner was built with `progress=True`; the interactive UI
  does not ask yet (its render commit does) ([#60]).
- An interactive UI in the terminal: `melo` without a task opens a chat
  that shows its work as it happens. A banner names the model, the
  workspace and the tools; each tool call prints as a block with the first
  lines of its output, a spinner shows the step and the time while the
  model works, and the answer prints as Markdown. `/` completes the
  commands with their descriptions: `/help`, `/new`, `/model` (one of
  the models the server lists), `/backend` and `/exit`; `/model` and
  `/backend` save the choice to the config file, except in a `--base-url`
  session. `shell_exec`,
  `python_exec` and `file_write` ask first (yes, allow for the session, or
  no with a reason); `--yolo` runs them without asking, and
  `--workspace DIR` sets the folder the file tools stay in. The input box
  edits Chinese and other wide characters whole, recalls earlier messages
  (↑↓, Ctrl+R) and makes a new line with Alt+Enter or Ctrl+J; Esc stops a
  turn. Sessions are kept in `~/.cache/localmelo/cli/sessions.db`;
  resuming them comes later ([#53]).
- A turn's work in the interactive UI, as it happens: the model's
  thinking (`✻ Thought for 1.2s` and 3 rows; 40 with `/verbose`, none with
  `/thinking off`), intermediate text and one held message printed only
  when it differs from the answer, `↻` retries, `·` empty replies, `⚠`
  stuck and budget notices, and reflections (including the unreadable
  ones). Tool blocks take the call's arguments and 20 output lines under
  `/verbose` and print images as `▪ image/png · 312 KB · not shown`. The
  live area now draws the run's `run.progress` facts, the running tool's
  clock and limit, and on a session's first turn a dim row of the past
  runs (the direct-mode line's wording). `/expand [turn.call]` prints a
  call's whole output, `/progress` the latest facts with the memory they
  rest on ([#68]).
- A first run in a terminal: `melo` with no config file looks for a model
  server on this computer, asks for its URL and for a model from the
  server's list, saves them and opens the chat. It looks with
  `GET /v1/models` on ports 8000, 8400, 11434 and 30000 of `127.0.0.1`
  (`LOCALMELO_DETECT_URLS` replaces them), and sends the chosen server one
  `GET /api/version` when saving, to tell Ollama apart; a server that cannot be used is
  named with the reason, and you can retry, start without saving or quit
  ([#52]).
- A progress line in the terminal. While `melo "<task>"` runs with stderr
  on a terminal and text output, one dimmed line on stderr shows the step,
  the phase, the running tool and the time; once the task has run before
  with memory kept, it adds what the last run took and its steps (or a
  similar task's, marked as such). The line is erased before the answer,
  draws nothing in the background, never wraps, and never holds up the
  run. `--progress off` turns it off, and `--routine NAME` groups
  differently worded runs of one task. Pipes, files, `--output-format json`
  and the benchmark harness see no change ([#51]).
- Runs learn the steps of recurring tasks. Each run that ends normally
  records an episode in procedural memory, and a completed run appends a
  new version of its routine's step template: in `procedures.db` in the
  memory directory with persistence on, in RAM otherwise. In the Python
  API, `run_task(..., progress=True)` with a sink adds `run.progress`
  events (levels 0 and 1; the percent bar is off until replayed runs
  support it), `observe=` watches a run without making it interactive,
  `routine=` names a routine, and `BaseLLMProvider.model` names a chat
  provider's model. The query text and labels procedural memory keeps
  also mask URL passwords and the values of names such as `API_KEY` or
  `password`, and `melo-procedures forget` leaves nothing of a forgotten
  routine in the file. Request bodies, the
  output, the JSON line, exit codes and history rows are unchanged
  ([#46]).
- Procedural memory (experimental), `localmelo.melo.memory.procedures`, and
  the offline `melo-procedures` command (`episodes`, `replay`, `export`,
  `forget`). From the runs in a `history.db` it learns the steps of a task
  you run again and again: the classes of its tool calls, short labels, step
  counts and times, never the model's text. Templates are versioned and kept
  in `procedures.db`. `replay` measures the rules of a planned progress
  display on recorded runs. Runs, the gateway and the Python API do not use
  it yet ([#45]).
- With persistence on, `history.db` keeps every reply of the model with its
  reasoning, in a new `llm_calls` table: one row per planning or reflection
  reply, with its kind, outcome, step, tool calls, token usage and duration,
  and the tool steps it made. Text and reasoning pass a minimal redaction of
  secrets. Reasoning is kept 30 days, or the days in
  `LOCALMELO_THINKING_RETENTION_DAYS`, in `history.db` and `sessions.db`
  alike. `melo-retrieve --export-run` exports these calls from a run's
  directory ([#42]).
- Every chat request carries an `X-Localmelo-Call-Kind` header (`plan` or
  `reflect`; `retrieve` for `melo-retrieve`), so a proxy can tell the kinds
  of calls apart. Request bodies are unchanged ([#42]).
- Sessions in the Python API (experimental), `localmelo.support.sessions`.
  A session keeps a conversation in SQLite: each turn begins with a stored
  row, outlives the request that started it, and ends exactly once with a
  status and a cause, also when the process that ran it dies; a fresh agent
  can resume the conversation. The package has the store, a runner for an
  agent's turns, approval helpers for surfaces that ask a person, an event
  stream and a read-only change feed. Neither `melo` nor the gateway uses
  sessions yet ([#41]).
- `docs/for-agents.md`, one compact page that tells AI coding agents and LLM
  tools what localmelo is and how to use it, and `docs/llms.txt`, an index of
  the docs in the llms.txt format, served at
  <https://localmelo.github.io/localmelo/llms.txt> ([#40]).
- The scheduler's policy core and simulator, `localmelo.support.scheduler`
  (experimental). It decides which model calls reach a shared local model
  server, in what order and why a task waits, and its simulator compares the
  `fifo`, `class` and `las2` arms on synthetic workloads or imported bench
  traces. Nothing uses it yet: runs and the gateway are unchanged
  ([#39]).
- `melo-retrieve`, installed next to `melo`, compares the `bm25`, `llm` and
  `embedding` memory retrievers offline on labelled queries: recall@k, mean
  reciprocal rank, latency and token use. `--export-run` turns a run's model
  calls or a history database into a corpus of reasoning notes, code drafts,
  thought chunks and reflections ([#38]).
- The web workspace replaces the gateway's playground page, at
  `http://127.0.0.1:8401/` by default. It discovers your server's models,
  keeps a draft for each chat, searches chats, renders Markdown with code,
  tables and links, can stop a run, and fits narrow screens. Chats last while
  the gateway runs, and a reload restores them ([#35]).
- `melo ""` opens a conversation in the terminal, with the configured backend
  or the server given by `--base-url`. `/help` lists its commands, and
  `/exit` or Ctrl-D leaves ([#35]).
- Run events for Python code: `Agent.run_task(..., on_event=sink)` passes
  each run, step, model reply, tool call, notice and reflection to the sink
  as an `AgentEvent`, with secrets redacted ([#37]).
- Cancelling a run that has an event sink or an approval policy ends it as
  `cancelled`, with the interrupted step closed in history and in the
  conversation. `Agent.begin_exit()` bounds the clean-up before the process
  exits ([#37]).
- Approval policies: `run_task(..., approval=policy)` is asked before each
  call of `shell_exec`, `python_exec`, `file_write` and other tools marked
  as requiring approval. An MCP server can mark its tools with
  `require_approval` ([#37]).
- `AgentOptions(workspace_root=...)` keeps `file_read` and `file_write`
  inside a directory, where `shell_exec` and `python_exec` also start
  ([#37]).
- `AgentOptions(record_answers=True)`, `Agent.conversation()` and
  `Agent.load_conversation()`, to carry a conversation across runs and
  agents; `Agent.start()`, `Agent.list_tools()` and `Agent.run_tool()`
  ([#37]).
- Direct-mode run options `--max-steps`, `--attempt-policy`,
  `--tool-exposure`, `--builtins`, `--memory`, `--memory-dir` and
  `--timeout`, and `AgentOptions` in the Python API ([#30]).
- `--embedding-base-url` and `--embedding-model`, for long-term memory in a
  `--base-url` run ([#30]).
- Nine run statuses and the `RunResult` record. `--output-format json` prints
  the record as one JSON line, with `localmelo_version` ([#30]).
- Exit codes: 0 for `completed`, 1 for `crashed`, 2 for any other status or a
  usage error ([#30]).
- A reflection can end a run as `completed` with the `done` action ([#30]).
- Tools from stdio MCP servers: `McpToolProvider` ([#27]), and `--mcp-config`
  and `--mcp-only` in direct mode ([#30]).
- Tools can return images. History and working memory keep only references
  to them: MIME type, SHA-256 digest and size ([#27]).
- The chat request timeout: `--request-timeout` and the `request_timeout_s`
  config key ([#30]).
- SIGTERM and SIGHUP end a direct run within 8 seconds, with its task record
  saved ([#30]).
- `extra_body` in each backend's config table, for extra chat request fields
  ([#26]).
- `--reflection-gate flags|utility` and `AgentOptions.reflection_gate`. The
  JSON line's `config` gains `reflection_gate` ([#34]).
- The MIT license ([#28]).
- A uv development environment with a committed `uv.lock` ([#25]).

### Changed

- melo remembers two ways, and never asks you to confirm a note. When you ask ("remember
  that Sam has the lab keys", "keep in mind that…", "记住…", "别忘了…", "帮我记一下…"), the
  note is kept at once, as before, even if the model does nothing. Otherwise the chat's
  model keeps a note with a new `remember` tool (`text`, one sentence; `topic`) when you
  share something lasting about yourself or your work, or a finished task's result in the
  chat is worth keeping; a plain statement ("I prefer short answers", "call me Ray") is no
  longer kept by fixed rules, only when the model keeps it. Code checks each note the
  model keeps: a password or key is refused, and its words must come from what you typed
  in that chat (any turn not undone) or from a finished task's result recalled in it, so a
  fact only a web page, a file or a tool's output states is refused and the model is told
  it was not something you said. A newer note of the same subject replaces the older one
  and a repeat is not written twice. The note appears under your message as before
  (`Remembered: … · Undo`): the turn's `turn.memory` event is sent again, whole, with it.
  The chat's instructions now tell the model to keep notes with `remember`, never to ask
  whether to remember something, never to save notes in a file or with code, and to
  acknowledge a note in one short sentence. The line that tells the model a requested
  note is saved now reads `(saved; do not save it again or in a file)`, so it can still
  keep another fact from the same message. While the tool runs, the panel and the
  activity line say `Remembering`, then `Remembered`. The tool asks no approval. Direct runs,
  workers, the terminal chat and the benchmark have no `remember` tool and are unchanged
  ([#316]).

- The command list under the input box is in the terminal's colours. Typing `/`, `@` or a
  command's argument showed a grey block (prompt_toolkit's default, black text on grey bars);
  the list now has no background: names in your text colour, their descriptions muted, and the
  chosen row on the same moss band with one `❯` as every other list, with the names standing
  under the text you typed. The scrollbar is a slim rule-coloured bar and appears only when
  there are more rows than the eight shown. Without the exact background the chosen row is
  moss and bold, and with `NO_COLOR` it is in reverse video ([#289]).

- The fixed system section of a request no longer has the line `Prefix: fixed-prefix-v2`.
  The model read the layout's internal name as an instruction and spent a reply deciding
  whether to start its answers with it. The date line stays, and `prefix_cache.version` in a
  run's result still names the layout. Every request of a run with the fixed prefix (the
  default), direct runs included, differs from before by that one line, so a server's prompt
  cache warms once after the upgrade ([#287]).

- The terminal's new look, part 2: panels, lists, the status bar and the welcome. The approval
  panel has an amber frame with its title bold, the chosen answer sits on a moss band, and a
  diff's added and removed rows are banded. Every list (the setup questions, the attach
  question, `/model`, `/sessions`) draws its chosen row on a band with one `❯` and the current
  model's `✓` in teal; key lines name their keys in your text colour with the words muted.
  The live row's spinner is moss and its words your text colour. The status bar is split into
  zones that keep their place, model and folder │ the run │ cache and context; while a turn
  runs, the run zone gives its steps in brief, `2/~8 · ctrl+o steps` (`~`: the task took 8
  steps before) or `step 2`, and on a narrow terminal the cache and context go first. Ctrl+O
  lists the turn's steps (each call's tool, target, status and time, then the steps still to
  come), during the turn or after it. At 256 colours or more and 80 columns or more, the
  welcome stands beside Melo's head with its ears, in half blocks computed from the pet's
  face picture. A command in an error's advice (`/retry`) is drawn as a command. `NO_COLOR`
  and `LOCALMELO_ASCII=1` keep every meaning; direct mode, workers, JSON, stream-json, events
  and exit codes are unchanged.
  ([#273])
- The terminal's colour tells kinds of output apart (part 1 of its new look). Each tool row
  shows its kind before the name, in the kind's colour: `→` reads (muted, so they fade), `±`
  writes, `$` commands and code, `@` network. A command that exits with another code than 0
  now shows `✗` and `exit N` in red, though its event still says the call ran. The final
  answer's `●`, headings and bullets are Melo's orange, the only orange on screen; other
  messages and thinking are muted. The changed-files line marks each file `+` new or `~`
  modified (`Changed + SUMMARY.md (+3), ~ todo.md (+1 −1)`). melo draws in 24-bit or 256
  colours when the terminal has them (it was pinned to 16), and the interactive UI asks the
  terminal for its background (OSC 11/10, at most 100 ms); with the exact colours your message
  sits on a raised band and inline code and code blocks on a raised chip and panel, else code
  gets a `│` gutter. `[cli] colors` and `[cli] background` (developer settings, `-X
  cli.background=light`) cover terminals that answer nothing. `NO_COLOR` and
  `LOCALMELO_ASCII=1` keep every glyph. Direct mode, workers, JSON, stream-json, events and
  exit codes are unchanged. ([#268])
- Every terminal question with a fixed set of answers is a list you move through and
  confirm with Enter, not a letter to type. On a terminal, the first run's questions
  (`Use this model server?`, `Use <model> for long-term memory?`, `What now?` after a
  URL that cannot be used, the no-server screen's `Check again`, `Enter an address` and
  `Quit`, and `Save URL · MODEL as the default backend?`), the setup wizard's backend
  lists, `Save these settings?` and `The check failed. What now?`, and `/attach !`'s
  `Attach this output?` show their answers as rows of words, as the tool approval and
  `/model` do: ↑ ↓ move, a number key (1 to 9) chooses that row at once, Enter confirms,
  and Esc takes the safe answer named in the key line (cancel the setup, decline, do not
  attach). The recommended answer is selected first, and a failed check and the attach
  question start on `Cancel setup` and `Don't attach it`, as `[N]` and `[y/N]` did. A
  list has no filter field. With answers piped in (stdin is not a terminal), nothing
  is drawn and every question reads the typed answer it always did (`Use it? [Y/n]:`,
  `[r]etry, [s]ave anyway, [N]o [N]:`, a number for a backend, Enter for the default),
  so scripts keep working; a direct run, `--output-format json` and workers never ask.
  In the UI, `/backend`'s model question is the list `/model` shows (it was numbered
  rows and a typed number). The inline list the first run's model question uses now
  serves them all ([#266]).

- Quitting the macOS app asks only when something would stop. With no chat or task running and
  nothing waiting for your answer, **Quit localmelo** (or ⌘Q) quits at once. When a chat or task
  runs, something waits for an approval, or the app cannot tell what runs, the dialog is the
  same as before (`Quit and Stop` / `Cancel`, naming what would stop). Direct runs and terminal
  chats are other processes, so they do not make it ask. Direct mode's stdout and stderr, JSON,
  exit codes and requests are unchanged. ([#264])
- The web console opens in your browser. At a terminal on your computer, `melo serve` opens
  the console once the gateway listens, already signed in, and prints `Web console: opened in
  your browser (melo console opens it again).` in place of the `Web UI:` and `Sign in:` lines;
  `melo console` opens it the same way and says `Opened the web console in your browser.` Over
  SSH, or when no browser starts, both print the one-use link instead (`Open this link within 60
  seconds. Do not share it:` and the link on its own line). The app's gateway, a login service
  and a pipe never open a browser, and a pipe keeps the `Web UI:` and `Sign in:` lines.
  `melo console` with nothing listening on the port says `no gateway is running on port N` and
  exits 1, instead of printing a link to nothing. A console page opened without a link now says
  `Sign in to localmelo` and `In a terminal, run melo console. It opens this page signed in.`
  The `Gateway:` line, direct mode's stdout and stderr, JSON, exit codes and requests are
  unchanged. ([#264])
- The macOS app and its menu-bar item use the localmelo logo, Melo the mascot,
  instead of the placeholder mark. The app icon (Dock, Finder, Spotlight and every
  dialog) is the README's emblem, `docs/assets/melo.png`, whole, on a rounded tile of
  the macOS icon grid in the emblem's own paper colour, taken from its four corners;
  the emblem fades into the tile, so no edge shows, and every size is made from the
  picture itself. The menu-bar item is the round logo, `docs/assets/icon_small.png`,
  in colour instead of a template image, with the states it had: the logo alone while
  idle, the count beside it while something runs, and a red dot with a ring while an
  approval waits (a lighter red on a dark bar). VoiceOver, the tooltip and the
  announcement of a wait are unchanged. The icon and the logo are committed files in
  `macos/assets`, made by `uv run --with pillow python scripts/make_app_icons.py`
  (Pillow is fetched for that run; it is not a dependency), and the build only copies
  them, so replacing the art is regenerate and commit. `macos/mark.svg` is gone
  ([#263]).

- The first run and the first answer say only what helps. With a model server found, the
  first run asks `Use it? [Y/n]:` (Enter says yes; `n` asks for a URL) where it said it had
  found one and then asked for its URL. With no embedding model on the server it says
  `Long-term memory is off; melo setup turns it on.`; the instruction to serve
  Qwen3-Embedding-0.6B or run `ollama pull qwen3-embedding:0.6b` is now in `melo setup`.
  Outside developer mode the dim `↳ partial reply (…)` line before text that was streamed
  is gone (the text still prints once), the note after a model's first whole reply, `This
  model's replies appear when complete until its reasoning format is set`, shows in
  developer mode only, and the approval header names the tool and the folder
  (`shell_exec · in ~/demo-project`); developer mode keeps the permission tier
  (`shell_exec · exec · in ~/demo-project`). Direct mode's stdout and stderr, JSON, exit
  codes and requests are unchanged ([#260]).
- With no model server running, the first run says exactly how to start one. With nothing
  saved and nothing answering on the usual ports, the first run of `melo` offered
  `http://127.0.0.1:8000/v1` as the server URL, and Enter looped on `✗ …: connection
  refused.` with `s starts without saving`, a chat that could not answer. It now prints
  `No model server is running yet. To start one with oMLX:` and the commands of the model
  servers guide (`brew tap jundot/omlx …`, `brew install jundot/omlx/omlx`, `omlx serve
  --model-dir ~/models --port 8000`; on other systems `ollama serve` and `ollama pull
  qwen3:8b`), then asks `Press Enter to check again, type an address, or q to quit:`.
  Enter looks again at the same ports; an address typed instead is checked as a URL is,
  and its failure still offers `s starts without saving`. `melo serve` and `melo setup`
  with nothing saved do the same instead of opening the eight-way menu. The web console's
  setup card shows the same commands in a block that one click selects, each command on
  one line, with a Copy commands button, links the model servers guide in place of the
  quickstart, and asks `GET /api/setup/detect` every three seconds while it says that no
  server runs, so the model appears once the server starts (nothing is asked while the
  card is hidden or the tab is). `GET /api/setup/detect` gains `start` in every answer.
  The model servers guide gains a short Ollama section. Direct mode's stdout and stderr,
  JSON, exit codes and requests are unchanged ([#260]).
- `melo serve` before any setup asks what the first run asks. With no config file and
  a terminal it printed `Welcome to localmelo.` and opened the eight-way backend menu with
  MLC as the default, even while a model server ran on this computer. Outside developer
  mode it now says `melo uses a model server on this computer (…)`, prints `Found a model
  server at … (N models).` and asks `Use it? [Y/n]:`, as `melo setup` does; the full menu
  follows only after `n`, with `--developer` or `[ui] mode = "developer"`, or with no
  server found while a backend is saved. The startup lines name the model and its server
  (`Model: Qwen3-8B-4bit on oMLX`) where they named a backend key (`Chat: vllm`); the
  `Gateway:` line, which the macOS app reads, is unchanged. The gateway's API, direct
  mode's stdout and stderr, JSON, exit codes and requests are unchanged ([#260]).
- Coming back to a chat in the terminal reads plainly. The exit line says
  `◆ Saved "What is this project?". Continue it with: melo -c` when the chat is the
  newest of the folder melo ran in, and still gives `Resume it with: melo -r ID`
  otherwise. A resumed chat's recap shows each message and each answer with one
  mark, the rows under it indented: the answer reads as plain text (its Markdown
  markers off, no marked blank rows, at most four rows) where it was raw Markdown
  with a `●` on every line, and the note below the title says `1 turn is back in the
  model's memory.` Developer mode keeps the tool-call detail (`; its tool calls are
  not.`). The picker's header reads `Resume a chat · ~/demo-project · tab: this
  folder or all · esc close`, and a long folder is cut from its middle so the keys
  stay on the row. The terminal says "chat" where it said "session": `New chat` in the
  status bar and in the `/new` rule, the `chat` row of `/status`, the picker's footer,
  the messages around resuming, and the descriptions of `/new`, `/sessions`, `/resume`
  and `/title`. Command names (`/sessions`, `-c`, `-r`), developer mode's rows and
  machine output such as `session_id` keep theirs. Direct mode's stdout and stderr,
  JSON, exit codes and requests are unchanged ([#258]).

- `melo -h` says what its everyday commands do. It lists `setup` (choose the model
  server again), `serve` (run the web console and API) and `console` (open the web
  console), then `More commands and every setting: melo -h all`, where it listed seven
  bare words (`setup · config · serve · console · task · worker · import`) that a new
  user could not tell apart. `-s, --steps` reads `most model calls per task (default 30)`
  where it said `planning-call limit`. The help is 23 lines instead of 20; `melo -h all`
  still lists every command and setting, and `melo COMMAND -h` still explains one. Every
  option and its behavior are unchanged, as are direct mode's stdout and stderr, JSON,
  exit codes and requests ([#258]).

- The terminal's status bar says its figures in words in default mode: `context 8% used`
  (`context 2.6k tokens` when the server's input limit is not known) where it said
  `ctx 12k/32k`, and `83% cached` where it said `cache 83%`, with `, idle 3m` after it
  for a local server. The cache shows once the server has reported a value, where it
  showed `cache --` before. The last request's context figure drops whole on a narrow
  terminal, like a measured one, before the folder is cut. The words are never cut
  in the middle. Developer mode keeps the compact segments, and so do the web console's
  status row and `GET /api/sessions/{id}/status`: their payload is unchanged. Direct
  mode's stdout and stderr, JSON, exit codes and requests are unchanged ([#258]).
- The console is easier on a phone. Under 700 px (or on a touch screen) the header drops the
  model chip while it only names the model, which the composer already shows, so a chat's
  title is no longer cut to two words; the chip stays when it warns (`Model offline`, `No
  model`). `Enter to send · Shift + Enter for a new line` and `⌘ K` are hidden there, and the
  next steps of a failed turn (Try again, Change the server address, Copy error) are outlined
  buttons at least 44 px tall instead of text links. In the sidebar, a chat's second line is its
  folder (`localmelo-workspace`) instead of the model, which every chat shared, and an accent
  dot before the title marks a chat that waits for an approval; developer mode shows the model
  first and the folder after it. Direct mode's stdout and stderr, JSON, exit codes and requests
  are unchanged. ([#257])
- The console and the app use plain words by default. The settings call the server **Model
  server** and its address **Server address**, with `The address your model server prints when
  it starts, such as http://127.0.0.1:8000/v1.` and `Changes apply to your next message.`; the
  list of models says `Lists the models this server offers.` A running chat reads `Working…`
  beside Stop, the status row drops its `step N`, and the sidebar's **LOCAL** badge is gone. In
  the macOS app the quit dialog says `Quitting stops 1 chat.` (or `Quitting stops what is
  running: 2 tasks and 1 chat.`) and no longer mentions the gateway, and **Sign In Again** is
  left out of the menus until it helps: after four failed sign-in attempts in a row, while the
  console shows its Locked screen, and always in developer mode (the page now carries `data-mode`
  and `data-locked` on its root element for the app to read). Developer mode keeps every word it
  shows today: **Chat backend**, **Chat API URL**, `Running on your backend`, the badge and the
  step. Direct mode's stdout and stderr, JSON, exit codes and requests are unchanged. ([#257])
- The console and the app point the right way when the model server is down. In the console, a
  run that could not reach the server now says `Could not reach the model server` and asks
  `Is your model server still running at 127.0.0.1:8000? Start it, then try again.` once: the
  red line that repeated the error, and the work box's retry line, are gone, and the provider's
  own text stays under Details. The filled button is **Try again** (**Continue** when the run had
  used tools), and **Change the server address** is a link beside it; a hosted model reads
  `Check your network connection, then try again.` and keeps **Open model settings**. A server
  that answered with an error keeps its card as before. In the macOS app, a gateway that quits
  on its own shows `Its background service quit unexpectedly. Try Again starts it again; Open
  Log shows what happened.` instead of `The gateway was terminated by signal 9.`; the signal or
  exit status goes to `gateway.log`. The provider's error text, direct mode's stdout and stderr,
  JSON, exit codes and requests are unchanged. ([#257])
- The macOS app builds with the Command Line Tools when the full Xcode is not
  installed. The Swift host is compiled with the developer folder that
  `DEVELOPER_DIR` names when it is set, else `/Applications/Xcode.app`, else the
  folder `xcode-select -p` names when it has a Swift compiler. With none of them
  the build stops before it does any work and prints `xcode-select --install` as
  the fix. The build's summary says which tools built the app ([#254]).
- The macOS app is 36 MB smaller (183 to 147 MB). Before the dependencies go in, the
  build leaves out the parts of the standalone Python that the app never runs: IDLE,
  tkinter and turtle with the Tcl/Tk libraries, pip, setuptools, ensurepip and 2to3. A
  fixed list of paths below the runtime decides, never a pattern matched against the
  build folder's own path, and a listed path that leads out of the runtime through a
  link stops the build. `unittest`, `distutils`, `venv`, `pydoc` and everything the
  dependencies import stay: every module of localmelo and of the bundled dependencies
  still imports, the gateway starts and answers `GET /api/health`, the signature
  verifies, and `melo --version` from the bundle takes 0.28 s as before (median of five
  runs). Code that runs on the bundled interpreter, such as the `python_exec` tool,
  can no longer import tkinter or turtle or run pip, and `python -m venv` needs
  `--without-pip`. ([#251])
- `/help` and the `/` menu start with the everyday commands. `/help` prints eight of them
  in three groups (Conversation, Model, Help), the `!` line and `/help all lists every
  command and key.`; `/help all` (and `/help` in developer mode) lists every command and the
  keys, and `/help NAME` explains one. The menu opens on those eight (`/model`, `/new`,
  `/status`, `/help`, `/exit`, `/undo`, `/copy`, `/retry`) and shows each command's name
  alone, with its argument written into the description where one is needed: its name
  column used to be as wide as `/approvals [ask|auto|deny|forget [tool]]`, which cut
  every description at 80 columns. A row longer than the terminal continues under its
  description instead of at the first column. Hints that named a retired spelling name
  the new one: `--mcp FILE` in `/mcp`, `-X endpoint.embedding_url=URL` in the memory
  row of developer mode's banner, and `melo console --reset` when the gateway's token
  is invalid. Direct mode's stdout and stderr, JSON, exit codes and requests are
  unchanged ([#250]).

- `/status` says whether melo is ready, in five rows: the model and its server with
  `✓ answers` (or `✗ does not answer (connection refused). Start it, then /status checks
  again.`), the folder, the session, what the approvals do in words (`asks before running
  commands or changing files`) and how full the context is (`8% used (2.6k of 32k tokens)`).
  `/status all`, and `/status` in developer mode, print every row as before, with two
  differences: a row longer than the terminal continues under its text instead of at the
  first column, and the memory row names the window with its reason (`64 (local server)`).
  The status bar names the approval mode only when it is not `ask`, which gives the
  other segments that room. `melo config keys` and `melo config get` show an unset
  `agent.window_messages` as the window it falls back to (`64 (local server)` or
  `20 (cloud)`) instead of `unset`. Direct mode's stdout and stderr, JSON, exit codes and
  requests are unchanged ([#250]).

- Reflections survive two more one-character slips in their JSON. With the default
  `reflection_gate flags`, a reply that holds no reflection already gets one repair of
  the `{...}` the decoder rejects (a `]` or `}` that no opener matches, a comma before a
  closer, raw line breaks and tabs in strings). It now also gets one for a backslash
  before a character that JSON has no escape for, such as `\;` in a string (the
  backslash is doubled, so the text reads as the two characters `\;`), and for an array
  that the next key breaks off (a `]` is inserted before the comma that precedes that
  key). A repair changes the text at the decoder's error and nowhere else and counts
  only when the result holds an object with a valid `recommended_action`; a reply that
  needs two repairs, a second invalid escape included, stays unreadable. The
  `reflection` event's `repaired` can now also be `invalid_escape` or `unclosed_array`.
  In a replay of 72 reflection replies from a benchmark run, 4 stayed unreadable after
  the earlier repair and 2 do now (a tool call, and a list written without its opening
  bracket). Fewer reflections are asked for again; request bodies are unchanged. ([#249])
- A chat that waits on a foreign GPU lock says so in plain words in both modes of the
  web console: its work group, live row, tool card, header chip and composer read
  "Waiting for this Mac's GPU" where they read "Waiting: Reservation". Other waits keep
  the scheduler's code; the terminal and the API are unchanged. ([#248])

- `/model` is a list to pick from. Alone, it opens the server's chat models under
  the input box: type to filter, ↑ ↓ to move, Enter to switch and save the model as
  the default, Esc to keep the current one. `✓` marks the current model, `(loaded)`
  the ones the server says are in memory, and embedding or reranking models are
  left out. `/sessions` uses the same list. On a terminal the first run's
  model question is the same list (nothing is highlighted when several chat models
  are served and none is known to be loaded, so Enter does nothing until a key moves
  to one; Esc cancels the setup); with the answers piped in, the numbered
  `Model [N]:` question is unchanged. `/model NAME` and `/model N` work as before.
  ([#247])

- The web console starts with fewer choices. In default mode the header shows the
  running chip and a model chip (the model's name, "Model offline" or "No model")
  that opens settings. The chat's level (P2), Details and the GPU-lock line are for
  developer mode; a chat that waits on a foreign GPU lock gets a banner instead,
  "Waiting for this Mac's GPU, your chat starts when it is free." The status row
  drops the cache, phase and queue segments, Tasks stops naming priorities, and
  Workers and Tasks join the sidebar once there is a run, a terminal chat or a task
  to see. Settings stacks Model with its test, Appearance and one Developer mode
  toggle, and shows the settings file and its revision in developer mode only.
  Developer mode shows everything as before, and nothing here changes what runs.
  The page reads `[ui] mode` from `GET /api/settings` when it opens. The level chip
  is now a menu button: it opens on the chat's level, the arrow keys move between
  levels and Escape returns to the button. Escape closes what is open in Tasks and
  Workers too, and send, stop, the menu button and the chips are 44px on touch
  screens and phones. ([#246])

- The web console is easier to read, in light and dark: no text under 12px, text
  at 4.5:1 contrast or better, field, composer and focus edges at 3:1 or better,
  a 2px focus ring on the composer, muted lines in Tasks and Workers drawn muted,
  and undone turns in a readable grey instead of faded. Colours, sizes and radii
  come from tokens in `workspace.css`, and the dark palette is written once. The
  level and send buttons lose their native `title` tooltips and keep their
  `aria-label`. ([#243])

- The web console asks for an approval in plain words: a question such as
  "Replace notes.txt (125 bytes)?", one filled "Allow once", an outlined "Allow
  for this chat" that says what it covers, and a quiet "Deny" that asks what melo
  should do instead. Escape stops a running turn only from the message box. ([#243])

- The web console says what to do after a run ends badly: a ✗ or ⚠ symbol, a
  hint and one filled next step (Open model settings, Continue, Try again, Show
  tools or Copy details). Tool rows use ✓ ✗ ⊘ ◌ and a live dot, the offline banner
  has "Try again", and the selected chat has an accent bar. ([#243])

- The first run explains itself. With no config file, `melo` first says what it needs
  (`melo uses a model server on this computer (oMLX, Ollama, LM Studio…).`) and asks
  `Server URL` (it said `Model server URL`). The model list shows chat models first;
  an embedding or reranking model is tagged `(embedding)` or `(reranking)` and is never
  the default (there is still no default among several chat models when the server does
  not say which is loaded). The hints wrap to the terminal's width at word boundaries
  instead of breaking inside a word, a failed check asks `Enter tries again · s starts
  without saving · q quits:` (it asked `[r]etry, [s]tart without saving, [q]uit [r]:`),
  and the first banner no longer says how to add long-term memory right after you
  declined it. The saved config and every machine output are unchanged ([#240]).

- The welcome starts the conversation. The interactive UI opens with three rows,
  `melo 0.1.0 · MODEL (/model to change)`, the folder, and `Try: What is this project? ·
  / commands · ctrl+d quit`, with the warnings under them (it was a box of model,
  workspace, tools, memory and session). The model's server shows in a row of its own
  only when the model is not the saved default; tools, memory and session are in
  `/status`. A narrow terminal drops the least needed parts of a row whole instead of
  wrapping it. `--developer` (or `[ui] mode = "developer"`) keeps the box
  ([#240]).

- Errors in the terminal say what happened and what to do, in plain words. A run that does
  not complete used to end with `✗ melo: run ended with status provider_error: cannot reach
  the model server at …`, cut at 80 columns, and a dim hint under it. It now reads
  `✗ Could not reach the model server at URL.` with `Start it, then /retry.` under it in the
  default color; every other status uses its title with the reason that helps (`Ran out of
  steps (30).`, `Stopped: the same calls kept repeating.`) and one action. The footer after
  a failed turn reads `✻ Stopped after 5.0s` (it said `Worked for`, and `0 tool calls`),
  and a retry prints one line a turn that keeps the server's URL. Direct mode's stderr
  line, JSON, exit codes and requests are unchanged ([#240]).
- A turn's record is quiet and the answer is the loudest thing on screen. A turn of three
  tool calls printed 27 rows of thinking excerpts and output before its answer; it now
  prints 12 (11 without the label of text streamed before a call). Thinking is one line
  (`✻ Thought for 8.0s`; `/thinking on` shows the excerpt). A call that succeeded shows
  the first line of its output with `+N lines (/expand 1.1)` on the same row; a call that
  failed, or a command that exited with another code than 0, still shows three lines, the
  first of a failure in red. Only the mark of a tool row is colored (the whole row was
  green). The live row says what melo does and for how long, without `step N` and the
  phase, and the status bar leaves out `step` (`cache`, `ctx` and `tasks` stay).
  `--developer` (or `[ui] mode = "developer"`) and `/verbose` keep the detail. Direct
  mode's stdout and stderr, JSON, exit codes and requests are unchanged ([#245]).
- The terminal UI is readable on a light terminal. The sentences a person must read or
  act on were dim, which a light background draws at about 2.4:1: the approval's warning
  and key line, and the hint row under the input box are now default text, dim stays for
  durations, counts and `(/expand T.C)`. A warning is yellow only on a background
  `COLORFGBG` reports as dark and bold in the default color elsewhere, and the code in an
  answer uses rich's `ansi_light` theme on a light background. The terminal is never
  queried; start melo with `COLORFGBG='0;15' melo` where the variable is not set. Direct mode's
  stdout and stderr, JSON, exit codes and requests are unchanged ([#245]).
- Replies stream by default in the terminal UI. `[cli] stream` is `true` unless set (it
  was `false`), so the interactive UI's requests carry `stream: true` and a model whose
  reasoning format is declared in `[cli.reasoning_tags]`, or that sends its reasoning in
  a field of its own, shows its reply as it is written. A model with no hint still shows
  its reply when complete, and says so once after its first whole reply (`This model's
  replies appear when complete until its reasoning format is set (see Streamed
  replies).`). `/stream on` and `/stream off` answer `Streaming on.` and `Streaming off.`
  with that sentence (they said `Live streaming on · conservative until declared.`).
  `/stream off`, `melo -X cli.stream=false` and `melo config set cli.stream false` turn
  it off. A model server that does not stream still answers every turn: a reply it sends
  as one JSON body is used as it is, and a stream that closes before it sent anything is
  asked for again once without streaming. The provider then stops asking that server to
  stream, and melo says so once (`This server sends replies whole; melo shows them when
  they are complete.`). A stream that breaks after part of a reply arrived still fails the
  turn. Direct mode, workers and the gateway never stream: their requests, output and
  exit codes are unchanged ([#245]).
- The measured cache layout is now the default (part of #183 and #185).
  `[agent] cache_layout` is `"v3"` (was `"v2"`) on every backend: with the fixed prefix,
  a reflection is appended to the window once, at the attempt boundary, instead of being
  rebuilt as the last message of every planning request, and an unreadable reflection is
  asked again on the same prefix. `[agent] window_messages` is 64 (was 20) on a local
  server and for a `--base-url` server; a cloud backend keeps 20 for now, because without
  prompt caching a longer window costs input tokens. A key you set keeps its value on any
  backend, and a file without the key has the default. Two benchmark rounds on 12 coding
  tasks, two runs each, with a Qwen model on oMLX compared the earlier defaults with these
  two. With the new defaults 46 of 48 runs passed (earlier: 44 of 48); 86.8% of the prompt
  tokens were served from the cache (78.9%); the share of tokens that an earlier request of
  the same run already held and the cache reused rose from 92.9% to 99.7%, and in the worst
  run from 79.4% to 93.7%; the uncached tokens per run fell from 9,865 to 6,059; a run took
  8.3 steps (8.7) and 32 s (35 s). `--cache-layout v2 --window-messages 20`, or the same
  keys in `[agent]`, send the earlier requests. Both flags moved from the experimental to
  the advanced tier of `melo --developer --help`. A default run's JSON result now carries
  `config.cache_layout`, and `config.window_messages` on a local server. ([#239])

- The macOS app's window says what is happening. Until the console has loaded it
  shows Melo, `Starting localmelo…` and the step (`Starting the local server`,
  `Signing in`, `Opening your workspace`) instead of an empty web view, and its
  background matches the console's, so a dark window no longer flashes white. When
  the gateway cannot start, does not start within 20 seconds or stops, the same
  view gives the reason and offers Try Again (a new gateway), Open Log and Quit;
  these problems no longer show a dialog and close the app ([#238]).

- The macOS app starts its gateway with `--port 0` and reads the port from the
  gateway's startup line (`Gateway: http://127.0.0.1:PORT`) instead of using a
  fixed port. It can no longer collide with another gateway, the background
  service or the ports other work reserves, so `LOCALMELO_APP_PORT` and the
  "port in use" check before the start are gone. The gateway's output still goes
  to `~/Library/Logs/LocalMelo/gateway.log`, and the 20-second start deadline now
  also covers the wait for that line ([#238]).

- The macOS app calls itself localmelo wherever it names itself: the window title, the
  Dock and the app menu, the menu-bar item (`localmelo`, or `localmelo · 3` while three
  things run, where it read `Melo`), the menu entries, the quit dialog and the
  output of `--install` and `--uninstall`, as its starting view and failure view
  already did. `LocalMelo.app` stays the file name and the bundle id stays
  `org.localmelo.app` ([#238]).

- An oMLX server is called oMLX wherever a user looks. The setup wizard's menus and the web
  settings dropdown label the `vllm` entry `oMLX or vLLM (OpenAI-compatible, local)` (it said
  `vLLM (local endpoint)`). The banner, `/status` and the web console's model row call a `vllm`
  server `oMLX` when the config file says so or a model listing of the server did (oMLX names
  itself as the owner of every model it lists, so no request is added), and `OpenAI-compatible
  server` otherwise (they said `vLLM`); `--base-url` sessions use the same two names (the second
  was `OpenAI-compatible`), and so do the backend's error messages. `chat_backend = "omlx"` and
  `embedding_backend = "omlx"` are accepted as another spelling of `vllm`: they load as the same
  backend and are saved as written. First-run setup (and `--reconfigure` and `/backend`) saves
  `omlx` for a server whose model listing names oMLX, and its embedding model the same way, so
  every banner says oMLX from the start; an older version rejects that key in a config-based run
  (a `--base-url` run is unaffected). On macOS the no-server hint names `omlx serve` before
  `ollama serve`. A model that oMLX serves under an alias is marked `(loaded)` and can be the
  default under the alias, because its status reply names the alias (`model_alias`). The requests
  of `melo --base-url … --chat-model …` runs are unchanged ([#237]).

- The command line has command words and short options (part of #2). `melo setup`,
  `melo serve [install|uninstall|status]`, `melo console [--reset]`, `melo task ...`,
  `melo worker BRIEF`, `melo import FORMAT FILE` and `melo config` replace
  `--reconfigure`, `--serve`, `--daemon`, `--gateway-login`, `--gateway-rotate-token`,
  `--brief` and `--import-session`. `-m/--model`, `-u/--url`, `-s/--steps`,
  `-o/--output`, `--mcp`, `-w`, `-t`, `-y` and `-V` are more names for `--chat-model`,
  `--base-url`, `--max-steps`, `--output-format`, `--mcp-config`, `--workspace`,
  `--timeout`, `--yolo` and `--version`, and `--title T` names a task's worker
  session. `melo -h` is 20 lines; `melo -h all` and `melo COMMAND -h` show the rest.
  Every old spelling still works and parses to the same result, so scripts, the macOS
  app and the benchmark need no change. A first argument that is exactly a command
  word is the command: quote a task that starts with one (`melo "setup the project"`)
  or put `--` before it. `-X KEY=VALUE` (long form `--set`) changes one `config.toml`
  key for one run and never writes the file: a key that has an old option
  (`-X retrieval.k=5` is `--recall-k 5`) becomes that option, so the run is the same,
  and any other key (`cli.stream`, `scheduler.policy`) overrides the loaded config and
  is checked. `melo config keys`, `get` and `set` list, read and save keys; nine keys
  are for one run only. The hints name the new commands. A command word alone after
  options (`melo -X gateway.port=9000 serve`) is refused, and the message says where
  it goes; `melo -- serve` gives the task "serve". A key named like an option
  (`-X agent.max_steps=5`) is refused, and the message says to use `-s N (--steps)`.
  ([#242])

- Cache-friendly requests are now the defaults (part of #183). `[agent] fixed_prefix`
  is `true` (was `false`), `[agent] window_eviction` is `"chunk"` (was `"slide"`), and
  the reflection tool guard is on, with a new `[agent] reflect_tool_guard` key that
  defaults to `true`. A same-build A/B on the benchmark's coding tasks with a local
  model chose them: more runs passed, prompt-cache efficiency rose and fewer
  reflections were unreadable. Direct runs, the interactive UI, the gateway and
  `AgentOptions()` read the defaults; a config key you set keeps its value, a file
  without the key has the default, and saving writes a key only when it differs from
  the default. `--fixed-prefix off --window-eviction slide --reflect-tool-guard off`
  (hidden from default help, listed by `melo --developer --help`) send the requests of
  the earlier defaults byte for byte. A default run's JSON result now carries
  `prefix_cache` and the `config` keys `window_eviction` and `reflect_tool_guard`.
  ([#223])

- First-run setup prefers a model that is already in memory: when the server
  reports which models are loaded (oMLX `GET /v1/models/status`, Ollama
  `GET /api/ps`), they are marked `(loaded)` and the default answer is the first
  loaded chat model, so Enter does not load a large model. `melo --serve --port 0`
  now binds a free port and prints the real address. The quickstart says a worker
  task waits once per tool the model asks to use, and that `python_exec` can start
  programs. ([#219])

- A `--brief` run checks its work before the budget runs out: with `Check:` lines,
  exactly two steps left in the last attempt, and a workspace that changed since
  the checks last ran, the harness runs them once so the model can still fix a
  failure (it uses one of the two retries). The `brief_check` notice and each
  `tests.items` entry of a brief check gain `output_tail`, the last 20 lines of
  the check's output, redacted, at most 4,000 characters. Part of #185 ([#214]).

- Second-walkthrough polish (part of #200): the no-backend message says
  `Run melo in a terminal to set up`; the quickstart has a section on running a
  task through the gateway (`melo-task new`, `list`, `show`); `melo-task show`
  prints `Reason:` and `Waiting:` only when the task has one; in the guided
  `melo --reconfigure`, after `n` the backend menu defaults to the saved backend,
  else to the declined server's kind (Ollama or vLLM); the `--brief` help says
  that `Check:` lines run and that a failed check exits 4; the bare `/` list shows
  each command once, with `/clear` and `/quit` named in the `/new` and `/exit`
  rows ([#211]).

- First-look fixes for the default mode: `melo --version` prints the banner's
  version and exits 0, and a usage error prints a short usage instead of every
  developer flag (`--developer` keeps the long one). The banner and
  `--version` show `melo 0.1.0 (baab21e)` on a git checkout (the JSON record's
  and gateway's version fields are unchanged). `/status` hides the cache,
  context, steps and tasks rows until they have data. The `/` list starts with
  `/model`, `/backend`, `/new`, `/status`, `/help` and `/exit`. `melo-task
  stop` says `stop requested; …` while the task has not stopped. In default
  mode `melo --reconfigure` offers the server it finds before the full
  backend menu, and the wizard's `Start?` is `Save and continue?` ([#207]).
- The "no model server" screens now say what to do. A chat request that cannot
  connect ends the run with `cannot reach the model server at URL. Start it, or
  run melo to set one up.` (for a server beyond this computer and the local
  network: `Check the URL and the network.`) and the raw cause after it, in the status line and
  the JSON `error`; timeouts and HTTP errors keep their text. The first run with
  no server found prints `No model server found on this computer (looked at
  ports …)` before the URL question, and the retry prompt reads `[s]tart without
  saving`. `melo --serve` prints `Sign in: run melo --gateway-login --port N in
  another terminal` after `Web UI:`. Text only: exit codes and choices are
  unchanged ([#206]).
- A `melo --brief` run that ends `completed` while its brief's check failed for
  good (`tests` is `failed`, a refused check included) exits with the new code
  4, so `melo --brief f && next-step` stops. `status` stays `completed`, the
  record's `exit_code` is 4, and stderr gets
  `melo: the brief's check failed: <command> (exit N)`. Every other exit code is
  unchanged, and a brief without a `Check:` line never exits 4 ([#205]).

- Docs: the quickstart has a "Run a worker task" section (a brief with a
  `Check:` line, `tests.state` and the exit code), and says that direct tasks
  have no shell unless `--builtins all` ([#205]).

- Organize Markdown documentation into start, guides, reference, concepts and
  project sections. Old paths retain readable links for each historical
  fragment; the agent entry points and printed CLI paths remain stable ([#129]).

- An attempt's `compress` digest now says what the attempt did: the paths its
  successful `file_write` calls wrote (at most six, workspace-relative, then
  `+N`), the last `shell_exec` or `python_exec` check and its result (test
  counts, exit code or last output line), up to three decisions, and the
  task's first sentence or heading instead of a fixed 300-character cut.
  Empty or trivial observations (a bare boolean, `None`, an exit code or
  punctuation) no longer enter a digest or make one a rehearsal candidate.
  Over the limit, open questions drop first, then observed lines, decisions,
  facts and files, then the task shortens; the outcome and the verification
  stay. `legacy` and direct mode without memory are unchanged ([#116]).
- Direct mode and the Python API's default `python` mode no longer offer
  `shell_exec`: their default built-ins are `file_read`, `file_write` and
  `python_exec`, and an `exec` call that was registered anyway is refused
  with a reason naming the mode and the way to enable it. `--builtins all`
  (or naming `shell_exec`) restores the old set, and a named shell then
  runs. A call to a built-in the run does not offer returns
  `shell_exec is not available in this run. Available tools: …`, naming the
  run's tools; an unknown name keeps the registry message. `--builtins` is
  still a usage error with `--serve` and `--daemon`. The JSON line's
  `config` gains `mode` (`"direct"` in a direct run), and `config.builtins`
  and `config.tools` lose `shell_exec` there. The six-pattern command
  blocklist, `validate_executor_request` and `ExecutorRequest` are gone: the
  tool's tier and the mode decide, not the command's spelling. The `melo ""`
  loop says once at start that `shell_exec` is off and that
  `melo "" --builtins all` turns it on ([#110]).
- `shell_exec` and `python_exec` start their child with a minimal
  environment instead of inheriting melo's: `PATH`, `HOME`, `USER`,
  `LOGNAME`, `SHELL`, `TERM`, `LANG`, every `LC_*` variable and `TMPDIR`,
  then the new `AgentOptions(tool_env={...})`, and nothing else. An API key
  in melo's environment reaches no tool; `VIRTUAL_ENV`, `PYTHONPATH`,
  `SSH_AUTH_SOCK` and the proxy variables pass only through `tool_env`. MCP
  servers keep their minimal environment, which now takes every `LC_*`
  variable too, from the same list ([#81]).
- Tool hints come from the model's tool calls only. A tool named in the
  task, in a tool result, in recalled memory or in a reflection no longer
  offers that tool, so text in a run cannot put a tool the model never
  called back in front of it; the built-ins, the BM25 matches and the
  re-offer after a refused call are unchanged ([#80]).
- `python_exec` runs its code in a child Python in every mode, not only with
  a workspace root. Code that loops, sleeps or calls `os._exit()` no longer
  holds or ends `melo`: the tool's time limit (60 seconds by default, 10
  minutes in the interactive UI) and a run's `timeout_s` kill it with every
  process it started. The code reaches the child on its standard input, a
  pipe, never in its `-c` argument, so `ps` does not show it and a call has
  no `ARG_MAX` limit; once the fixed wrapper has read the code it points
  fd 0 at `/dev/null`, so `input()` gets EOF. The child gets a POSIX
  CPU-time limit of one hour and an open-file limit of 1,024. Success
  output is unchanged. A failure now reads
  `Python exited with code <n>:` and the end of the child's output instead
  of the exception's text, and `sys.exit(0)` is a success. The code runs as
  `__main__` with `sys.argv == ["-c"]`, in a fresh module namespace: module
  state no longer carries from one call to the next, and writes to file
  descriptor 1 or 2, from a subprocess too, are the call's output instead of
  melo's. The `tool.finished` event's `exit_code` now carries the child's
  status without a root too ([#79]).
- The README is short, and the website carries the documentation. Its header
  is two badge rows: one that goes to the website's pages (Website,
  Quickstart, Docs, Developers, 简体中文) and a quieter row of facts
  (pre-alpha, Python 3.11+, MIT, llms.txt for AI agents). Below it are what
  localmelo is, install, a few commands, a "Learn more" list of the
  website's pages and the security warning; everything else that was there
  is on the website or in `docs/`. `CONTRIBUTING.md` is short too, and links
  to a new developer guide, `docs/en/developers.html` and
  `docs/zh-CN/developers.html`, linked from every page's navigation: the
  repository layout and the `melo/` ↔ `support/` rule, the package layout,
  setup, tests, ruff/black/mypy, adding a backend, a built-in tool or a
  check, commits and pull requests, and the docs and website layout
  ([#67]).
- The `dev` extra installs `pytest-xdist` and `pytest-testmon`, so
  `uv run pytest -n auto -q` runs the offline suite in parallel before a
  pull request and `uv run pytest --testmon -q` runs only the tests a
  change affects while working. CONTRIBUTING and the agent docs name both,
  and `.testmondata` is ignored ([#58]).
- A plain `pytest` no longer opens sockets to local model servers while
  collecting `tests/smoke/test_backend_smoke.py`; its two real-backend tests
  are skipped unless `LOCALMELO_SMOKE=1` is set. With the switch on they probe
  and send requests as before, and
  `--ignore=tests/smoke/test_backend_smoke.py` is no longer needed for an
  offline run ([#57]).
- `melo` without a task, in a terminal, opens the interactive UI on the
  saved backend; it used to start the gateway, which `melo --serve` still
  does. `melo --base-url URL` and `melo --chat-model NAME` without a task
  open it too, where they used to be ignored; `--base-url` without
  `--chat-model` uses the first chat model the server lists.
  `melo --reconfigure` opens it after the wizard ([#52], [#53]).
- Without a terminal (stdin and stdout must be TTYs and `TERM` not
  `dumb`), bare `melo` says what to run instead and exits with code 2;
  `melo --reconfigure` runs the wizard and exits. The UI takes the run
  options, such as `--max-steps` and `--memory-dir`, for every turn;
  without a task, `--tool-exposure`, `--timeout`, `--output-format`,
  `--progress`, `--routine`, `--mcp-config` and `--mcp-only` are usage
  errors, and with a task, `--workspace` and `--yolo` are. `melo ""`
  keeps the plain conversation loop ([#53]).
- The setup wizard offers the config file's values as its defaults, and
  `None` for embeddings when none is set. A failed chat check can be
  retried with the values you typed; a failed embedding check saves with
  embedding off instead of cancelling the setup ([#52]).
- A memory directory that localmelo creates gets mode 0700, and a new
  `history.db` mode 0600. The default directory is tightened to 0700 when it
  is open to other users ([#42]).
- `--chat-model` without `--base-url` replaces the configured chat model for
  that run; it used to be ignored ([#35]).
- A follow-up question in a gateway session, a web chat or the `melo ""`
  loop sees the earlier answers, without their `<think>` blocks and cut to
  8,000 characters each ([#35]).
- A step runs only the tools it offered. A call to any other tool is refused
  and does not run ([#30]).
- A call to a built-in tool whose arguments have the wrong JSON type is
  refused before it runs ([#30]).
- A failed model request is retried when a retry may succeed. A request that
  still fails ends the run as `provider_error` ([#30]).
- In `retrieve` tool exposure, every step offers all registered built-in
  tools, then the extra tools found by hints and by Okapi BM25 with IDF, both
  sorted by name ([#32]).
- After a reflection, the next attempt starts when the reflection chooses
  `continue` and its three checks hold; its utility score no longer has to
  reach 0.1. A reply that still holds no reflection after the retry no longer
  ends a run that has attempts and steps left, and a reflection over 2,000
  characters is trimmed instead of ending the run. A reply whose only JSON is
  not a reflection, such as a tool call written out as text, is asked for
  again too. `--reflection-gate utility` keeps the previous behavior
  ([#34]).

### Fixed

- Correct documentation of retrieval option validation, persistent memory,
  interface event streams and current architecture status ([#121]).

- The heuristic note writer reads factual replies without a `<note>` into up
  to 12 `found` items, and an explicit note can also hold 12. Questions,
  options, plans and negated or cancelled statements are not promoted as
  positive facts. Self-work claims are filtered unless a keyed content
  clause follows the frame; `task:`/code markers and completed outcome
  digests remain exempt. Source matching scores typed keys independently,
  retains date years, currencies and whole phone numbers, rejects conflicts,
  and leaves ties and unmatched items unsourced even after admission. Bare
  slash dates use US order when ambiguous; quantity/count contexts, mixed
  numbers and `24/7` are excluded. Digit runs over 20 characters are ignored
  as keys.
  Factual extraction remains heuristic and does not extract CJK-only keys.
  ([#114])
- A note's items now take their `source_ids` from the observation that
  established them, even when it came in an earlier reply of the same task:
  the reply's own tool results first, then the results still in the window,
  then the rest of the task's (newest first, last 200 records). A record
  matches only on a strong key (a number, a date or time, an identifier
  containing a digit, or a capitalised name that is not sentence-initial)
  or two distinctive non-filler words, and a note with records but no
  match gets no item source, so "Done." no longer cites an unrelated
  "no harm done" record. This fills `provenance.source_ids` for facts
  copied from an earlier tool result and for the digest built from them;
  another task's records are never used ([#103]).
- The model-facing recall header no longer carries the retrieval latency:
  it is `(retriever bm25)`, not `(retriever bm25, 12 ms)`. Identical runs
  now produce identical prompt bytes, so a block kept in the window does
  not break the prefix cache. `ms` remains in `llm_calls.recall_json`, the
  `recall` event and `memory_stats` ([#103]).
- A thought chunk of an attempt is now superseded at the attempt boundary
  together with that attempt's notes, so a chunk recalled later renders
  with the `(superseded)` marker and names its attempt instead of passing
  for the current attempt's thinking ([#100]).
- `melo-retrieve --export-store DIR` and `--export-decisions DIR` print
  zero entries and exit 0 when the directory has no `long_term.db` yet,
  instead of failing with "No such file"; a path that is not a directory
  keeps the missing-file usage error ([#100]).
- A thought chunk recalled with `--recall-kinds thought` no longer replays
  the reply that produced it: its alias stays with that step's turn, so
  step recall and the `recall` tool return it only after the turn left the
  window, as they already did for notes and drafts ([#96]).
- The test suite no longer writes into the user's own `~/.cache/localmelo`:
  every test runs with `HOME` in a temporary directory, the defaults of the
  gateway's call log, the web workspace's memory and `config.toml` resolve
  when they are used instead of at import, and a session guard fails a test
  that resolves the real home or writes the real cache
  (`@pytest.mark.allow_real_cache` opts a test out of the home rule)
  ([#90]).
- `POST /v1/agent/run` answers a non-string `session_id` (a number, a list)
  with the route's own 400 `Invalid session_id format` instead of FastAPI's
  plain-text 500, and its 100,000-character limit counts the query after the
  surrounding whitespace is removed, which is the text the agent runs: a
  whitespace-padded query whose trimmed text fits is accepted ([#89]).
- A held chat's header still said `● Working` while the sidebar, status
  label and message area said it was waiting, and a chat that had just sent
  kept `Working on your request…` in the message area for the whole hold
  (the poll skipped the detail refresh while the send was pending). The
  header's status pill now says `Waiting: <reason>`, and while the send is
  pending the message area follows the polled hold as it changes; both
  return to Working when the hold ends ([#71]).
- The scheduler simulator's importer reads a chat call's kind from the
  `X-Localmelo-Call-Kind` header that localmelo sends, in the headers side
  file a proxy record names with `bodies.headers`. Retrieval and
  step-estimate calls, which have no tools, no longer replay as `reflect`;
  pre-R1 records and records without `--log-bodies` keep the old guess, an
  unknown header value keeps the guess and is counted, and the import
  report counts the records of each kind source ([#61]).
- `DELETE /v1/sessions/{id}` no longer closes a session whose task is still
  running: it answers 409 with the gateway's error body
  (`{"error", "session_id"}`) and leaves the task and the session alone;
  idle sessions close as before. The `/v1/agent` API still has no cancel
  route, so a caller waits for the task. Stopping the gateway still closes
  every session, a busy one included, and a task still running then ends as
  `crashed` ([#63]).
- A `shell_exec` command no longer inherits melo's standard input: it reads
  `/dev/null` in every mode. A shell whose stdin was a pty that was not
  melo's controlling terminal took that pty as its controlling terminal at
  start-up and hung it up when it exited, so `melo` ended a completed run
  with exit code 120 instead of 0 ([#59]).
- A server picked at the first run, at
  `Save URL · MODEL as the default backend?` or with `/backend` was always
  saved as the `vllm` backend. An Ollama server, saved that way, ran with
  vLLM's 120 s chat timeout instead of Ollama's 300 s, and the banner said
  vLLM. A server that answers `GET /api/version` like Ollama is saved as the
  `ollama` backend now; any other OpenAI-compatible server is still saved as
  `vllm` ([#55]).
- A similar task's past steps on the progress line said `like a similar
  task:`; they name the task they are from, `like <routine title>:` (the
  routine's name, else the start of its query) ([#56]).
- A `shell_exec` call whose `command` arrived as an argv array (a list of
  strings, which the tool refuses) was labelled `{command:` on the
  progress line, and procedural memory stored that label; the call is now
  classified and labelled by the command's words, joined with
  `shlex.join`, so a read command such as `["ls", "-la"]` is R rather
  than X ([#56]).
- `melo --serve` with no config file and no terminal, as under launchd,
  ended in an `EOFError` traceback, again every few seconds; it prints one
  line and exits with code 1. Ctrl-D or Ctrl+C in the setup wizard prints
  `Setup cancelled.` instead of a traceback ([#52]).
- The config file escapes quotation marks and backslashes in URLs, model
  names and other values; such a value made the file unreadable, so every
  later `melo`, `melo --reconfigure` included, failed ([#52]).
- An embedding URL that is not valid, such as a mistyped port, ended the
  setup wizard with a traceback after the chat check passed; the config is
  now saved with embedding off ([#52]).
- `--base-url` and `--chat-model` were silently ignored with `--serve` and
  `--daemon`; they are usage errors now ([#52]).
- An HTTP 401 or 403 from a local server during setup says that localmelo
  sends no API key to local servers (oMLX: "Allow inference without API
  key") ([#52]).
- In the `melo ""` loop, Backspace deletes whole characters: it no longer
  leaves part of a Chinese or other wide character on the screen, or in the
  line, which made the loop stop with a `UnicodeDecodeError`. The prompt is
  edited with readline on a terminal, so the Up arrow recalls earlier lines
  ([#50]).
- A request that disconnects during a run no longer leaves its gateway
  session busy for good ([#35]).
- Stopping the gateway closes the web workspace's model connections
  ([#35]).
- `--host '[::1]'` starts the gateway, and the startup lines show an IPv6
  address in brackets, as in `http://[::1]:8401/` ([#36]).
- Tool calls are sent to OpenAI-compatible models as real tool-call turns,
  and every call of a step runs ([#26]).
- Empty replies, oversized prompts and long tool results no longer end a run
  early or silently ([#30]).
- A reflection reply without a JSON object is asked for once more, without
  tools ([#31]).

### Security

- Gateway and console sessions no longer offer `shell_exec`: both run in the
  new `gateway` tool mode, which registers `file_read`, `file_write` and
  `python_exec` only: legacy `/v1` has no approval policy; the console asks for
  `file_write` and `python_exec` using `RememberingPolicy`, and
  the console's approvals arrive over the same unauthenticated loopback API,
  so any local process could approve its own request. `AgentOptions` gains
  `mode` (`direct`, `interactive`, `gateway` or `python`; `python` is the
  API default) and the CLI sets its own surface's mode; the interactive UI
  keeps `shell_exec` behind its existing approval panel. Direct mode and the
  library's `python` default follow in the entry above. An explicit
  `AgentOptions(builtins=…)` or `--builtins` still names `shell_exec` in
  `direct` and `python` mode, and the call then runs; `gateway` refuses it
  even when named ([#104]).
- Secrets no longer stay in stored memory. `tasks.query` and `tasks.result`,
  `steps.thought`, a tool call's arguments and a tool result's output and
  error, and the text long-term memory keeps (each step's summary and the
  promoted reflections) are written with the same redaction `llm_calls` and
  events already use: `sk-` keys, GitHub tokens, AWS access key ids, the
  token after `Bearer` and a one-time code next to its keyword. The run and
  the model keep the text: working memory, the requests a run sends, its
  events and its `result` are unchanged, and the stored images stay
  references ([#83]).
- On `localhost` or a loopback IP address, the gateway refuses requests that
  are not addressed to it, before any route runs. A `Host` other than
  `127.0.0.1`, `localhost` or `[::1]` with the gateway's port gets HTTP 421
  `misdirected_request`. A `POST`, `PUT`, `PATCH`, `DELETE` or `OPTIONS`
  request from a web page at another origin, and a `/v1/` request that the
  browser marks as cross-site or same-site, get HTTP 403 `foreign_origin`.
  Web pages could otherwise reach the gateway through DNS rebinding or
  cross-site requests and run tasks with its tools. The standalone
  playground server applies the same checks ([#36]).
- No web page can show the gateway's pages in a frame: every response
  carries `X-Frame-Options: DENY` and
  `Content-Security-Policy: frame-ancestors 'none'` ([#36]).
- On any other host, such as `0.0.0.0`, the gateway makes none of these
  checks and logs a warning when it starts ([#36]).
- A new `long_term.db` is created with mode 0600, like `history.db` and
  `procedures.db`, with its `-wal` and `-shm`; it got the default mode
  before, usually 0644, so other users of the computer could read the step
  summaries and reflections it holds. An existing file keeps its mode
  ([#62]).

Changes before these are not listed here. See the
[updates page](https://localmelo.github.io/localmelo/en/updates.html) for
earlier milestones.

[Unreleased]: https://github.com/localmelo/localmelo/commits/main
[#25]: https://github.com/localmelo/localmelo/pull/25
[#26]: https://github.com/localmelo/localmelo/pull/26
[#27]: https://github.com/localmelo/localmelo/pull/27
[#28]: https://github.com/localmelo/localmelo/pull/28
[#30]: https://github.com/localmelo/localmelo/pull/30
[#31]: https://github.com/localmelo/localmelo/pull/31
[#32]: https://github.com/localmelo/localmelo/pull/32
[#34]: https://github.com/localmelo/localmelo/pull/34
[#35]: https://github.com/localmelo/localmelo/pull/35
[#36]: https://github.com/localmelo/localmelo/pull/36
[#37]: https://github.com/localmelo/localmelo/pull/37
[#38]: https://github.com/localmelo/localmelo/pull/38
[#39]: https://github.com/localmelo/localmelo/pull/39
[#40]: https://github.com/localmelo/localmelo/pull/40
[#41]: https://github.com/localmelo/localmelo/pull/41
[#42]: https://github.com/localmelo/localmelo/pull/42
[#45]: https://github.com/localmelo/localmelo/pull/45
[#46]: https://github.com/localmelo/localmelo/pull/46
[#50]: https://github.com/localmelo/localmelo/pull/50
[#51]: https://github.com/localmelo/localmelo/pull/51
[#52]: https://github.com/localmelo/localmelo/pull/52
[#53]: https://github.com/localmelo/localmelo/pull/53
[#55]: https://github.com/localmelo/localmelo/pull/55
[#56]: https://github.com/localmelo/localmelo/pull/56
[#57]: https://github.com/localmelo/localmelo/pull/57
[#58]: https://github.com/localmelo/localmelo/pull/58
[#59]: https://github.com/localmelo/localmelo/pull/59
[#60]: https://github.com/localmelo/localmelo/pull/60
[#61]: https://github.com/localmelo/localmelo/pull/61
[#62]: https://github.com/localmelo/localmelo/pull/62
[#63]: https://github.com/localmelo/localmelo/pull/63
[#64]: https://github.com/localmelo/localmelo/pull/64
[#66]: https://github.com/localmelo/localmelo/pull/66
[#67]: https://github.com/localmelo/localmelo/pull/67
[#68]: https://github.com/localmelo/localmelo/pull/68
[#69]: https://github.com/localmelo/localmelo/pull/69
[#71]: https://github.com/localmelo/localmelo/pull/71
[#72]: https://github.com/localmelo/localmelo/pull/72
[#74]: https://github.com/localmelo/localmelo/pull/74
[#75]: https://github.com/localmelo/localmelo/pull/75
[#76]: https://github.com/localmelo/localmelo/pull/76
[#78]: https://github.com/localmelo/localmelo/pull/78
[#79]: https://github.com/localmelo/localmelo/pull/79
[#80]: https://github.com/localmelo/localmelo/pull/80
[#81]: https://github.com/localmelo/localmelo/pull/81
[#83]: https://github.com/localmelo/localmelo/pull/83
[#84]: https://github.com/localmelo/localmelo/pull/84
[#85]: https://github.com/localmelo/localmelo/pull/85
[#91]: https://github.com/localmelo/localmelo/pull/91
[#92]: https://github.com/localmelo/localmelo/pull/92
[#89]: https://github.com/localmelo/localmelo/pull/89
[#94]: https://github.com/localmelo/localmelo/pull/94
[#95]: https://github.com/localmelo/localmelo/pull/95
[#96]: https://github.com/localmelo/localmelo/pull/96
[#90]: https://github.com/localmelo/localmelo/pull/90
[#97]: https://github.com/localmelo/localmelo/pull/97
[#100]: https://github.com/localmelo/localmelo/pull/100
[#101]: https://github.com/localmelo/localmelo/pull/101
[#103]: https://github.com/localmelo/localmelo/pull/103
[#104]: https://github.com/localmelo/localmelo/pull/104
[#106]: https://github.com/localmelo/localmelo/pull/106
[#110]: https://github.com/localmelo/localmelo/pull/110
[#111]: https://github.com/localmelo/localmelo/pull/111
[#113]: https://github.com/localmelo/localmelo/pull/113
[#114]: https://github.com/localmelo/localmelo/pull/114
[#115]: https://github.com/localmelo/localmelo/pull/115
[#116]: https://github.com/localmelo/localmelo/pull/116
[#119]: https://github.com/localmelo/localmelo/pull/119
[#120]: https://github.com/localmelo/localmelo/pull/120
[#121]: https://github.com/localmelo/localmelo/pull/121
[#122]: https://github.com/localmelo/localmelo/pull/122
[#123]: https://github.com/localmelo/localmelo/pull/123
[#124]: https://github.com/localmelo/localmelo/pull/124
[#125]: https://github.com/localmelo/localmelo/pull/125
[#126]: https://github.com/localmelo/localmelo/pull/126
[#128]: https://github.com/localmelo/localmelo/pull/128
[#129]: https://github.com/localmelo/localmelo/pull/129
[#130]: https://github.com/localmelo/localmelo/pull/130
[#131]: https://github.com/localmelo/localmelo/pull/131
[#132]: https://github.com/localmelo/localmelo/pull/132
[#133]: https://github.com/localmelo/localmelo/pull/133
[#135]: https://github.com/localmelo/localmelo/pull/135
[#136]: https://github.com/localmelo/localmelo/pull/136
[#137]: https://github.com/localmelo/localmelo/pull/137
[#141]: https://github.com/localmelo/localmelo/pull/141
[#142]: https://github.com/localmelo/localmelo/pull/142
[#143]: https://github.com/localmelo/localmelo/pull/143
[#144]: https://github.com/localmelo/localmelo/pull/144
[#145]: https://github.com/localmelo/localmelo/pull/145
[#147]: https://github.com/localmelo/localmelo/pull/147
[#149]: https://github.com/localmelo/localmelo/pull/149
[#152]: https://github.com/localmelo/localmelo/pull/152
[#153]: https://github.com/localmelo/localmelo/pull/153
[#155]: https://github.com/localmelo/localmelo/pull/155
[#157]: https://github.com/localmelo/localmelo/pull/157
[#158]: https://github.com/localmelo/localmelo/pull/158
[#159]: https://github.com/localmelo/localmelo/pull/159
[#162]: https://github.com/localmelo/localmelo/pull/162
[#164]: https://github.com/localmelo/localmelo/pull/164
[#165]: https://github.com/localmelo/localmelo/pull/165
[#166]: https://github.com/localmelo/localmelo/pull/166
[#167]: https://github.com/localmelo/localmelo/pull/167
[#168]: https://github.com/localmelo/localmelo/pull/168
[#169]: https://github.com/localmelo/localmelo/pull/169
[#175]: https://github.com/localmelo/localmelo/pull/175
[#177]: https://github.com/localmelo/localmelo/pull/177
[#180]: https://github.com/localmelo/localmelo/pull/180
[#187]: https://github.com/localmelo/localmelo/pull/187
[#188]: https://github.com/localmelo/localmelo/pull/188
[#189]: https://github.com/localmelo/localmelo/pull/189
[#190]: https://github.com/localmelo/localmelo/pull/190
[#191]: https://github.com/localmelo/localmelo/pull/191
[#192]: https://github.com/localmelo/localmelo/pull/192
[#193]: https://github.com/localmelo/localmelo/pull/193
[#194]: https://github.com/localmelo/localmelo/pull/194
[#195]: https://github.com/localmelo/localmelo/pull/195
[#197]: https://github.com/localmelo/localmelo/pull/197
[#198]: https://github.com/localmelo/localmelo/pull/198
[#199]: https://github.com/localmelo/localmelo/pull/199
[#209]: https://github.com/localmelo/localmelo/pull/209
[#210]: https://github.com/localmelo/localmelo/pull/210
[#211]: https://github.com/localmelo/localmelo/pull/211
[#214]: https://github.com/localmelo/localmelo/pull/214
[#216]: https://github.com/localmelo/localmelo/pull/216
[#217]: https://github.com/localmelo/localmelo/pull/217
[#218]: https://github.com/localmelo/localmelo/pull/218
[#219]: https://github.com/localmelo/localmelo/pull/219
[#221]: https://github.com/localmelo/localmelo/pull/221
[#222]: https://github.com/localmelo/localmelo/pull/222
[#223]: https://github.com/localmelo/localmelo/pull/223
[#224]: https://github.com/localmelo/localmelo/pull/224
[#226]: https://github.com/localmelo/localmelo/pull/226
[#227]: https://github.com/localmelo/localmelo/pull/227
[#228]: https://github.com/localmelo/localmelo/pull/228
[#230]: https://github.com/localmelo/localmelo/pull/230
[#231]: https://github.com/localmelo/localmelo/pull/231
[#232]: https://github.com/localmelo/localmelo/pull/232
[#236]: https://github.com/localmelo/localmelo/pull/236
[#237]: https://github.com/localmelo/localmelo/pull/237
[#238]: https://github.com/localmelo/localmelo/pull/238
[#239]: https://github.com/localmelo/localmelo/pull/239
[#240]: https://github.com/localmelo/localmelo/pull/240
[#241]: https://github.com/localmelo/localmelo/pull/241
[#242]: https://github.com/localmelo/localmelo/pull/242
[#243]: https://github.com/localmelo/localmelo/pull/243
[#245]: https://github.com/localmelo/localmelo/pull/245
[#246]: https://github.com/localmelo/localmelo/pull/246
[#247]: https://github.com/localmelo/localmelo/pull/247
[#248]: https://github.com/localmelo/localmelo/pull/248
[#249]: https://github.com/localmelo/localmelo/pull/249
[#250]: https://github.com/localmelo/localmelo/pull/250
[#251]: https://github.com/localmelo/localmelo/pull/251
[#254]: https://github.com/localmelo/localmelo/pull/254
[#255]: https://github.com/localmelo/localmelo/pull/255
[#257]: https://github.com/localmelo/localmelo/pull/257
[#258]: https://github.com/localmelo/localmelo/pull/258
[#259]: https://github.com/localmelo/localmelo/pull/259
[#260]: https://github.com/localmelo/localmelo/pull/260
[#263]: https://github.com/localmelo/localmelo/pull/263
[#264]: https://github.com/localmelo/localmelo/pull/264
[#265]: https://github.com/localmelo/localmelo/pull/265
[#266]: https://github.com/localmelo/localmelo/pull/266
[#268]: https://github.com/localmelo/localmelo/pull/268
[#271]: https://github.com/localmelo/localmelo/pull/271
[#272]: https://github.com/localmelo/localmelo/pull/272
[#273]: https://github.com/localmelo/localmelo/pull/273
[#274]: https://github.com/localmelo/localmelo/pull/274
[#275]: https://github.com/localmelo/localmelo/pull/275
[#284]: https://github.com/localmelo/localmelo/pull/284
[#285]: https://github.com/localmelo/localmelo/pull/285
[#287]: https://github.com/localmelo/localmelo/pull/287
[#288]: https://github.com/localmelo/localmelo/pull/288
[#289]: https://github.com/localmelo/localmelo/pull/289
[#290]: https://github.com/localmelo/localmelo/pull/290
[#291]: https://github.com/localmelo/localmelo/pull/291
[#292]: https://github.com/localmelo/localmelo/pull/292
[#293]: https://github.com/localmelo/localmelo/pull/293
[#294]: https://github.com/localmelo/localmelo/pull/294
[#295]: https://github.com/localmelo/localmelo/pull/295
[#296]: https://github.com/localmelo/localmelo/pull/296
[#297]: https://github.com/localmelo/localmelo/pull/297
[#304]: https://github.com/localmelo/localmelo/pull/304
[#305]: https://github.com/localmelo/localmelo/pull/305
[#306]: https://github.com/localmelo/localmelo/pull/306
[#308]: https://github.com/localmelo/localmelo/pull/308
[#309]: https://github.com/localmelo/localmelo/pull/309
[#311]: https://github.com/localmelo/localmelo/pull/311
[#312]: https://github.com/localmelo/localmelo/pull/312
[#313]: https://github.com/localmelo/localmelo/pull/313
[#314]: https://github.com/localmelo/localmelo/pull/314
[#316]: https://github.com/localmelo/localmelo/pull/316
[#317]: https://github.com/localmelo/localmelo/pull/317
[#320]: https://github.com/localmelo/localmelo/pull/320
[#322]: https://github.com/localmelo/localmelo/pull/322
[#326]: https://github.com/localmelo/localmelo/pull/326
[#310]: https://github.com/localmelo/localmelo/pull/310
[#201]: https://github.com/localmelo/localmelo/pull/201
[#202]: https://github.com/localmelo/localmelo/pull/202
[#203]: https://github.com/localmelo/localmelo/pull/203
[#204]: https://github.com/localmelo/localmelo/pull/204
[#205]: https://github.com/localmelo/localmelo/pull/205
[#206]: https://github.com/localmelo/localmelo/pull/206
[#207]: https://github.com/localmelo/localmelo/pull/207
[#208]: https://github.com/localmelo/localmelo/pull/208
