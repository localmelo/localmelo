# Changelog

One line per change a person notices, in their words, under its area; no pull-request internals.
Each release cuts its own section. localmelo has no release yet: v0.1.0 is due 2026-12-20.

## Unreleased

### Chat

- One main chat, with side chats under it: it answers, or hands bigger work to a task by priority; the terminal, the console and the app open the same chat.
- A chat does the work you ask for itself, and hands what is left to one fresh task when the rest would fill its context; its turn names the task.
- A task a chat started runs once; more work on the same job (your message, Continue, Finish it, Allow once) goes on in one fresh task with the record as background.
- When a task a chat started fails, runs out of steps or needs you, the chat follows up once: it continues another way or asks one clear question.
- A finished task's answer comes back to the chat that started it.
- Auto mode: choose Ask or Auto for each chat. In Auto, the steps the goal plainly needs run, risky ones ask with the reason, and Allow once lets one through.
- In Auto, a card waits only while someone can see it, and your earlier words in the chat still count.
- The go-ahead Allow once sends is melo's own sentence: a chat shows it as a folded "Not typed by you" line, and a chat or a task no longer keeps it as words you typed; it never answers a question a task waits on.
- Text in a file or a paste that imitates the heading of melo's memory is not taken for recalled notes when a step is reviewed in Auto, and a password typed after such a line is hidden like any other.
- In Auto, a long task that goes well no longer runs out of reviews (a review that lets a high-impact step run on your words still counts), and a tool you allowed for the chat runs its routine calls without one.
- Approval cards say in one plain sentence what the step will do and why melo asks, and follow your chat's mode.
- A chat keeps whom a request is about, and a new chat sees a short digest of your other recent chats.
- A short follow-up such as "go look it up" or "continue" works from what you said just before.
- A chat replies in the language of your latest message, even after tool output in another language.
- Reminders: say "remind me to send the report on Friday at 9" and melo tells you at that time.
- "Remind me of/about …" with no time is a question melo answers, not a request for a reminder time.
- Chores that run on time, with a check-in every 30 minutes that says what finished or waits.
- Pictures and files in the console and the app, and words you add while melo works reach the turn.
- melo reads PDFs and public web pages in the chat, and can search the web with one switch for every chat and task.
- A thinking model's answer is no longer lost, and a reply that is only unfinished thinking or a broken tool call is never shown as the answer.
- A question that asks for a short answer gets one in at most two steps.
- A chat with an embedding model set up answers faster: the model server no longer reads the whole conversation again on most turns.
- Chats and tasks use the context the model server says it runs the model with, up to 131,072 tokens, so work that fits it is no longer cut at a fixed 100,000 characters and read again; work larger than that context still reads parts again.
- Long work that outgrows the model's context keeps what it learned: when the context is nearly full, melo writes down where the work stands and what it already read, and the oldest results make room.
- A chat's reply appears line by line as it is written, in the console, the app and the terminal; Stop ends it at once and keeps what was shown.
- A slow reply from a local model is no longer cut at two minutes: a chat request waits up to five minutes for the server.

### Tasks

- When one task needs you and asked last, your next message goes to it with Undo; a plain yes allows its one blocked step once, and its card has Allow this step.
- A typing or form step Auto blocked in a task shows on its page and in its chat with the site and the whole text, and Allow once runs that exact text once.
- A scheduled task can keep your answer to a blocked typing or sending step for its site: later runs type and send there without asking; listed with Remove on the Scheduled tab.
- A task says where it is in the same words on its page, in the side panel and in the terminal, and the reason it waits after a restart reads in your language.
- A task says honestly how it ended: Ready only with proof, otherwise Done, not finished, with the reason and a one-tap Finish it.
- Ready means done: a task asked to write a named file that is missing says so on its card.
- A task that runs out of steps reads Stopped, and Continue picks it up where it left off.
- A task that is stuck asks the chat that started it instead of stopping.
- A task starts with what its chat knows about you and the project, and reads the project's own instruction file.
- A task a chat starts reads what the chat understood of your message, so "retry today's task" names the right work; only your own words decide what is allowed.
- A task works in its chat's folder and asks before it reads or writes outside it.
- A task's approvals go through the chat that started it, and an Allow for this chat given there covers the task too.
- Priorities read P0 to P5 everywhere.
- A scheduled task can browse, and keeps its folder.
- `melo task` creates, lists, follows and controls the gateway's tasks from the terminal.
- One task that cannot be restored after a restart no longer keeps the console from opening; it waits with the reason on its page.
- After a restart, a task cut off in the middle of writing a file in its folder goes on and looks at that file first; any other half-done step waits for you.
- A task goes on by itself when localmelo starts again after a stop or a crash, and its page says so; only work that may be half done waits for you.
- Worker runs (`melo worker BRIEF`) run their `Check:` lines when the model is done, and a failed check exits 4.

### Browser

- melo browses in Chrome in the background, with no window: a live view in the console and the app shows the page as it works.
- Take over the page to sign in, type the password yourself and give it back; a take-over is a real pause in a real window.
- One click on a browser card shows the page big.
- melo clicks, types and chooses on pages, and finishes web forms from what you told it.
- A go-ahead such as "I authorize you" starts the work in your name; a sign-in is a step you take, not a stop.
- A window another app opens after melo's click asks you first; until you allow it, nothing about it reaches the model.
- `/browser on` and the Settings switch use Chrome in the background when it is installed, and Safari's automation windows (one tab per chat) otherwise; a browser you chose before is kept.
- When a click changes nothing on a page, melo is told so and tries another way; after it acts on a page it reads only what changed.
- The hosts of one site count as one site, so one allow covers them.
- A human check from a known provider asks you once per chat, in Chrome, in Ask and in Auto: Always in this chat covers such checks in that chat, its tasks and its chores (in the main chat, every chat). Other boxes and Safari ask as before.
- The browser tools never open this computer, its local network or melo's own console, except the test servers a developer lists in `[tools] web_fetch_allow_local`.

### Memory

- melo remembers what you tell it across chats, never asks you to confirm, and the newer value wins.
- The Memory tab lists your notes to read, edit or forget; `melo memory` does the same in the terminal.
- On first setup, or from the Memory tab, melo brings in what other coding assistants on this computer remember, with Undo.
- Tidying notes while you are away is built and off by default while it is tested; one tap undoes a tidy.
- A note melo keeps says "you", says who it is about, and melo never claims a save it did not make.
- melo keeps notes you write in Chinese or Japanese when it words them differently.
- With no embedding model set up, melo now finds a note you ask about in other words: the chat model picks among the notes that share a word with your message, within 4 seconds.
- The Memory tab, `melo memory status`, Settings and setup say how melo searches from the models you have set up, not from a fixed guess.
- A note from a page, a file or a task's result is never shown to the model as your words.
- What you ask melo to remember from a page or file you pointed to in the chat can now be kept, and the note says where it came from; before, such a note was refused.
- The note gate has one rule: `[memory] note_gate` and `note_min_cosine` are removed. A config file that still holds them loads with a note; nothing else changes for a default setup.
- A note melo takes from a page, a file or a task's result now replaces only an older note of that kind: one you typed, reworded or imported, or that sleep wrote, is kept beside it with its source, and your own words are listed first.
- An ask that points at something elsewhere ("remember her email and where she works") no longer leaves a note that only repeats the ask, and a note from "remember that …" no longer starts with those words.
- A password given after a short word, as in "pw: …" or "pwd = …", is not kept in melo's memory.
- A token on the line after the word "Bearer", and a one-time code in the sentence beside its keyword ("verification code") on the same line, are left out of melo's memory too.
- A run in the default mode no longer sends each tool step to the embedding model or opens its requests with earlier step summaries; notes, recall and search across chats are unchanged, and an existing long_term.db is left as it is.
- A note melo cannot keep no longer costs you the answer: after its last refused try the note tool is not offered again in that turn, and melo answers the rest of what you asked.
- A file you send, and the "[1 file]" or "[1 picture]" mark for a message with no words, no longer count as something you said when melo searches your chats or checks a note.
- A note that restates your fact with another form of your word ("kept" for "keep", "fluently" for "fluent") or a joining word you did not use is no longer refused.

### App and web console

- The web console and the macOS app speak 简体中文, with times and dates written the Chinese way.
- Melo the Akita lives in the app, the console and on your desktop: it wakes, types, rests and plays, and raises a paw when something waits for you.
- With the app's window put away, a small pill says what runs.
- One command replaces every picture of Melo from an art pack.
- Settings is organized by subject; Settings → Permissions lists every allow, Everywhere and Per chat, with Revoke.
- Every slash command of the terminal works in the console and the app.
- The console follows every session live: chats, tasks, worker runs and terminal chats.
- After an update, a browser tab or the app's window shows the new console at once instead of an old copy.
- One command builds and installs the macOS development app.
- The console reads in plain words, is easier on a phone and is readable in light and dark.
- The app's window has no title band, and the menu-bar item is Melo's head drawn in the bar's own colour.
- The console has one look for its chips, labels and code, and a calmer first screen in a narrow window.
- The app at rest no longer uses a full CPU core: the menu-bar item stops redrawing itself without pause.
- A calmer chat: a reply's time shows once, copying is an icon button, task cards and suggestions sit apart from the talk, and the panel's tasks have a line between them.
- Calmer task pages, Settings and details: short times, a task's facts on one line, plain words for each tool, and a failed task's card saying why in your language.
- A blocked step reads the same on its chat card and its task page, also after a reload; a run whose work went on in a newer task says so instead of Needs you everywhere; the Scheduled tab says how each last run really ended.

### Terminal

- `melo` opens the main chat in the terminal, with a live area, streaming answers and the files a turn changed.
- Command words and short options: `melo setup`, `melo serve`, `melo console`, `-m`, `-u`, `-w`; old spellings still work.
- `melo -h` shows the everyday options; `melo -h all` shows every option and setting.
- Come back to a chat with `melo -c` or `melo -r`, and name, undo or copy it.
- Shift+Tab switches the chat between Ask and Auto.
- In the terminal too, your next message goes to the one task that asked for you last, and a plain yes allows its blocked step once.
- Every question with fixed answers is a list you move through and confirm with Enter.
- Colour tells kinds of output apart, and `NO_COLOR` or `LOCALMELO_ASCII=1` lose nothing.
- Attach a text file with `/attach`, and run a shell command with `!`.
- The terminal keeps a running turn when its connection to the gateway breaks, and shows the rest of it when the connection is back.
- `melo` no longer starts tool servers it will not use when the gateway is already running.
- `melo task` says Did not finish where it said Failed, and Continued in a newer task (with no hint to continue it) for a run whose work went on.

### Setup

- The first run finds a model server that already runs, or on a Mac sets one up after one OK.
- The chat model is picked from what the server offers; when the saved model is gone and the server has one chat model, melo uses it and says so.
- Hosted model services in Settings and `melo setup`, chosen from a list, with the key in the keychain.
- melo_serve is a local model server, and its model can be switched from every surface.
- `melo serve` just works after the first run, and can run at login on macOS.
- oMLX is supported and recommended on Apple silicon.

### Security

- The gateway listens on loopback only, needs a bearer token or a browser session, and refuses requests from other sites.
- Direct runs and console chats offer no shell by default.
- A worker's shell and Python write only inside its workspace on macOS.
- High-impact calls (installing, downloading, changes outside the folder) always ask, even inside an allow.
- Allow once allows that one call, in chats, tasks and the terminal; only Allow for this chat is kept, and a card offers it only where melo may keep it.
- Secrets, private keys included, are redacted from tool output, events, logs and stored memory.
- A token after "Bearer" is hidden after a Chinese character too, and a private key with a second end line is hidden to the last one in stored, logged and finished text.
- A commit whose message credits melo or an AI is refused.
- New memory files are readable only by you.
- A password inside a text file you attach is cut out of the saved chat like one you type; an older chat may still hold one, and the console guide says how to delete it.
- A text file you send is never read as your words: not when it waits behind a running turn (it stays a file, as when sent at once), not on Try again, and not in a task a chat starts from it.
- A text file you send on to a task, running or finished, is data too: only the words you typed with it count as yours.
- A card number alone on a line, written with dots or unusual spaces, across one line break, or followed by its expiry or security code is kept out of what melo remembers; a chat's title loses such a number, a PIN or a password too.
- A password longer than 128 characters is cut out whole, not in part; hiding a message twice gives the same text, and a shell's `PWD=/folder` or `pwd /folder` is no longer taken for a password.

### Scripts and API

- A run ends with one of a fixed set of statuses and exit codes, and `-o json` prints the run as one JSON line.
- SIGTERM and SIGHUP end a direct run within 8 seconds, with its record saved.
- The web console's HTTP API (`/api`) runs chats, tasks, approvals and settings; `POST /v1/agent/run` stays.
- Every direct run is recorded, so the console's Workers view lists it.
- `melo-retrieve` and `melo-procedures` compare memory retrievers and replay recorded runs offline.
- The Python API (experimental) has events, stopping, approvals and sessions.
- `GET /api/sessions/{id}/browser/picture` gives a script one picture of the page a chat's browser is on, while the live view is on.
- Three experimental settings are gone (`--brief-items`, `agent.lean_output`, `agent.thinking_steps`): runs behave as before, and a script that still sets one exits 2 and says so.
- The MIT license.
