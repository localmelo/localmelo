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
- Approval cards say in one plain sentence what the step will do and why melo asks, and follow your chat's mode.
- A chat keeps whom a request is about, and a new chat sees a short digest of your other recent chats.
- A short follow-up such as "go look it up" or "continue" works from what you said just before.
- A chat replies in the language of your latest message, even after tool output in another language.
- Reminders: say "remind me to send the report on Friday at 9" and melo tells you at that time.
- Chores that run on time, with a check-in every 30 minutes that says what finished or waits.
- Pictures and files in the console and the app, and words you add while melo works reach the turn.
- melo reads PDFs and public web pages in the chat, and can search the web with one switch for every chat and task.
- A thinking model's answer is no longer lost, and a reply that is only unfinished thinking or a broken tool call is never shown as the answer.
- A question that asks for a short answer gets one in at most two steps.
- A chat with an embedding model set up answers faster: the model server no longer reads the whole conversation again on most turns.

### Tasks

- A task says honestly how it ended: Ready only with proof, otherwise Done, not finished, with the reason and a one-tap Finish it.
- Ready means done: a task asked to write a named file that is missing says so on its card.
- A task that runs out of steps reads Stopped, and Continue picks it up where it left off.
- A task that is stuck asks the chat that started it instead of stopping.
- A task starts with what its chat knows about you and the project, and reads the project's own instruction file.
- A task works in its chat's folder and asks before it reads or writes outside it.
- A task's approvals go through the chat that started it, and the same approved step is not asked twice.
- Priorities read P0 to P5 everywhere.
- A scheduled task can browse, and keeps its folder.
- `melo task` creates, lists, follows and controls the gateway's tasks from the terminal.
- One task that cannot be restored after a restart no longer keeps the console from opening; it waits with the reason on its page.
- Worker runs (`melo worker BRIEF`) run their `Check:` lines when the model is done, and a failed check exits 4.

### Browser

- melo browses in Chrome in the background, with no window: a live view in the console and the app shows the page as it works.
- Take over the page to sign in, type the password yourself and give it back; a take-over is a real pause in a real window.
- One click on a browser card shows the page big.
- melo clicks, types and chooses on pages, and finishes web forms from what you told it.
- A go-ahead such as "I authorize you" starts the work in your name; a sign-in is a step you take, not a stop.
- A window another app opens after melo's click asks you first; until you allow it, nothing about it reaches the model.
- Safari's automation windows work too (`/browser on`), one tab per chat.
- The hosts of one site count as one site, so one allow covers them.
- The browser tools never open this computer, its local network or melo's own console.

### Memory

- melo remembers what you tell it across chats, never asks you to confirm, and the newer value wins.
- The Memory tab lists your notes to read, edit or forget; `melo memory` does the same in the terminal.
- On first setup, or from the Memory tab, melo brings in what other coding assistants on this computer remember, with Undo.
- Tidying notes while you are away is built and off by default while it is tested; one tap undoes a tidy.
- A note melo keeps says "you", says who it is about, and melo never claims a save it did not make.
- melo keeps notes you write in Chinese or Japanese when it words them differently.
- A note from a page, a file or a task's result is never shown to the model as your words.
- Long-term memory can be kept between runs, with an embedding model the first run offers.

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

### Terminal

- `melo` opens the main chat in the terminal, with a live area, streaming answers and the files a turn changed.
- Command words and short options: `melo setup`, `melo serve`, `melo console`, `-m`, `-u`, `-w`; old spellings still work.
- `melo -h` shows the everyday options; `melo -h all` shows every option and setting.
- Come back to a chat with `melo -c` or `melo -r`, and name, undo or copy it.
- Shift+Tab switches the chat between Ask and Auto.
- Every question with fixed answers is a list you move through and confirm with Enter.
- Colour tells kinds of output apart, and `NO_COLOR` or `LOCALMELO_ASCII=1` lose nothing.
- Attach a text file with `/attach`, and run a shell command with `!`.
- The terminal keeps a running turn when its connection to the gateway breaks, and shows the rest of it when the connection is back.
- `melo` no longer starts tool servers it will not use when the gateway is already running.

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
- Secrets, private keys included, are redacted from tool output, events, logs and stored memory.
- A commit whose message credits melo or an AI is refused.
- New memory files are readable only by you.

### Scripts and API

- A run ends with one of a fixed set of statuses and exit codes, and `-o json` prints the run as one JSON line.
- SIGTERM and SIGHUP end a direct run within 8 seconds, with its record saved.
- The web console's HTTP API (`/api`) runs chats, tasks, approvals and settings; `POST /v1/agent/run` stays.
- Every direct run is recorded, so the console's Workers view lists it.
- `melo-retrieve` and `melo-procedures` compare memory retrievers and replay recorded runs offline.
- The Python API (experimental) has events, stopping, approvals and sessions.
- The MIT license.
