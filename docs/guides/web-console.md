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
must be running to serve it: `melo serve` opens it in your browser, signed in.
A page opened without a link says *Sign in to localmelo*: run `melo console`
in a terminal and it opens the page signed in (add `--port PORT` for a
nondefault port). Over SSH, or when no browser starts, it prints a single-use
link to open within 60 seconds instead. Keep the link private; see [gateway security](gateway.md#keep-the-gateway-private).

On a first run, with no model set up, the welcome shows one card in place of the
starters, in the main chat as in a new chat. It looks for a model server on this Mac, as the terminal's first run
does, and shows what it found: **Found oMLX on this Mac** (or Ollama, or a model
server) with its model, **A model server needs an API key**, or **No model server
is running on this Mac**. **Use this model** saves the server, tests it and makes
the welcome ready without opening settings. With several chat models and none
known to be loaded, the card lists them, marks the loaded one and waits for you to
choose; it never picks one for you. With no server running, a supported
Apple-silicon Mac sets one up from the card, as the terminal's first run does, with
[the same list and the same job](model-servers.md#first-run-setup): what this Mac can
run, each item with its size and where it goes, and the total to download. An item
that is optional has a box to turn it off, and one button (`Set up Qwen3 8B (6.6 GB)`)
starts it after that one OK; a list left on screen for more than 15 minutes is asked
for again first, and set up starts at once if it is the same, or waits for another OK if
it changed. Each step then shows
its own progress and its state in words, **Stop** keeps what was downloaded, and
**Continue** goes on after a stop or a failure. When the chat model
answers, the card connects it as **Use this model** does. Where setup is not offered
(another system, an Intel Mac, too little memory, or a check that failed) the card says
why, with the words of whoever said no in smaller type under it, and shows the commands that start a server (oMLX on a Mac, Ollama elsewhere, as in the
[model servers guide](model-servers.md)) in a block that one click selects, with a
**Copy commands** button for the keyboard (they are there while the card still checks too). It asks
again every three seconds while it is on screen and the tab is visible, so the model
appears once a server answers. **Check again** asks at once, and **Enter a server
address…** opens connection settings. The card names the other way too, under **Or use an account or a key**: **Choose a service…** opens
Settings → Model server, where a model that runs online is set up with your own key.
Until a model is set up the composer reads `Connect a model to start` and nothing is sent: words you typed stay in the box, and the banner
(**No model is configured yet.**) has a **Connect a model** button that opens the same settings. The card also shows when
the configured server stops answering, and it looks for a server only while it is
on screen ([`GET /api/setup/detect`](../reference/api-v2.md#get-apisetupdetect), with the
list from [`GET /api/setup/plan`](../reference/api-v2.md#get-apisetupplan)).

The page opens in default mode, which leaves out what only a developer needs: the chat's
level chip, **Details**, the GPU-lock line, the **LOCAL** badge, the cache, step, phase and queue segments of the
status row, the running chip (the [panel beside the chat](#the-panel-beside-a-chat) shows what runs), and the
priority line and menu of a task's page. It has no Workers and no priority codes to learn either: the panel's **Workers** tab, the welcome link to worker runs, **Run as a task** and the `P0` to `P5` chips and lists belong to developer mode, a task shows as what melo is doing (in Now and in the chat that asked), and a priority you changed reads in words (`today`). It says things plainly too: a running chat reads
`Working…` beside **Stop**, and the live row says what melo is doing and for how long (`Reading notes.md · 12s`), with no tool name, step number or token count (the work line reads `Worked for 14s` and the time shows nowhere else, a tool card reads `Wrote notes.md`, the run's phase is in words such as `Looking around`, and the panel names the model without the kind of server), and the settings call the server **Model server** and its address
**Server address**. The header shows a
model chip (the model's name, `Model offline` or `No model`) that opens settings. With melo_serve it shows the model and its backend and opens the list of models to switch to, as does the model button under the message box ([Switch the model](model-servers.md#switch-the-model)).
A task opens in the [panel beside the chat](#the-panel-beside-a-chat), from its row in Activity, and has no link of its own in the sidebar; worker runs are a developer-mode view, the
panel's **Workers** tab and the sidebar's **Workers** link ([Developer mode](#developer-mode)). Everything stays in the
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

1. Open **Settings** to connect a model server. It lists its sections on the left and shows
   the chosen one beside them; on a phone the list comes first and a section opens as a sheet
   (**‹ Settings** goes back). With nothing saved it opens on **Model server** with the form
   open on a blank **Choose a server…**. You can also use `melo setup`.
   [More on Settings](#settings).
2. Type a message and send. The page opens on the [main chat](#the-main-chat-and-side-chats),
   which is always there; in a new side chat the first message creates the chat: the page
   answers your draft with the chat's first turn, and never creates a chat on
   its own. Answers are rendered as Markdown, with code blocks.
3. A turn reads like the terminal's: your message, a work group with the
   tool cards it ran, its reasoning, its notices and reflections, the live
   row while it runs, and the final answer with its copy button (an icon; it says
   **Copied** for a moment). Output that
   a tool printed is cleaned of escape sequences, and its images (PNG, JPEG,
   GIF, WebP) show inline, and the pictures the turn made show under the answer
   (see [Pictures in the answer](#pictures-in-the-answer)). A turn that answers without tool calls shows its
   time once, beside the copy button under the answer; its reasoning, when it has some,
   opens under a `Thought` line. Chats tell the model that it is melo from
   localmelo, which model and server it runs on and what it remembers, so asking
   "who are you?" or "what do you remember?" is answered plainly (see [What melo says about itself](terminal.md#what-melo-says-about-itself)).
4. A status row sits above the composer: the same segments as the terminal's
   developer-mode status bar (its default mode says them in words), `cache 67% idle 3m`, `ctx 1.0k/8.0k`, `step 3`, `tasks 1 running
   2 queued`, read from `GET /api/sessions/{id}/status` so both agree. Default mode draws the context
   segment, and a sentence such as `1 task waits for you` while a task waits for your approval. The cache
   share reads `cache --` when the model server reports no cached tokens, never
   0%; a local server shows its idle time, not a countdown. The row is read at
   most once a second while events arrive and on the page's polls; on a narrow
   screen it drops segments from the right. An open worker run shows the same
   row.
5. In developer mode, a **N running** chip in the header counts what runs now: this console's
   chats with a turn in flight, tasks, direct runs and terminal chats (a run or
   terminal chat counts only while its process holds a live lease). It is
   hidden at 0 and stays on a narrow screen. Select it for a short list, each
   row naming the kind and the title and opening that chat, the task's card or
   the Workers detail; Esc or a click elsewhere closes it. The numbers come
   from `status` of `GET /api/health`, read with the page's list refresh (every 4 seconds while anything runs or changes, and for a few seconds after a run ends); the
   list shows 20 rows at most and says how many more run.

The sidebar pins the main chat on top and lists its side chats under it, newest
first under a line for each day (`Today`, `Yesterday`, `Oct 28`), with search and
switching, and closes a side chat. Under each title it names the chat's folder (developer mode puts the model
first), and a dot in the accent colour before the title marks a chat that waits
for your approval. Up and Down move through the list, the main chat first, and Home
and End go to its ends. Drafts stay with each chat in the current browser tab. The open chat's
turns arrive on its event stream (SSE); after a page reload the page restores
the chat, reads its last ten turns, and replays the stored events from there,
so nothing is lost or drawn twice. **Show earlier turns** loads older turns
without moving the current view. A stream the browser gives up on reopens
after 1, 2, 4, 8, 16 and then 30 seconds from the page's own cursor, and only
after the chat is read again: one that is gone leaves the page on a new chat
rather than retrying for ever. The list is re-read every 4 seconds while the
tab is visible and anything runs or changes, for changes from another browser;
with nothing new it backs off to every 16 seconds, and it is read at once when
you come back to the tab or act on the page. A tab nobody sees asks nothing. **Reconnect now** checks the
chat before reopening its stream. Independent banners show store, model and
connection problems without replacing other notices.

On a phone, or in a window 700 px wide or less, the page leaves out what a small
touch screen does not need. The header drops the model chip while it only names the
model, since the composer shows it, and keeps the chip when it warns (`Model offline`,
`No model`). The keyboard hints (`Enter to send · Shift + Enter for a new line · / for commands` and
`⌘ K`) go, on touch screens and in narrow windows alike. The next steps of a failed
turn are buttons at least 44 px tall.

## The main chat and side chats

The console has one **main chat**, always there and first in the sidebar: Melo's face,
the time of its last message, and a dot while something new has come back to it (amber
when it waits for you). It cannot be closed. The page opens on it, and a tab that had a
side chat open goes back to that one.

The main chat always works in the default folder: `[gateway] workspace`, else
`~/localmelo-workspace`, made on first start. Change the default in Settings and its next
turn works there; it has no folder chip of its own.

**New side chat** (`⌘ K`) starts a focused conversation with its own folder: the chip offers
the default, your recent folders and **Choose a folder…**, and the chat keeps its pick. It is listed
under the main chat. A side chat's header begins with **Main chat ›**, which goes back,
and closing the open side chat goes back to the main chat too. Chats made before the main
chat existed are its side chats, as they were.

What you allow in the main chat counts everywhere; in a side chat, only there (see [Permissions](#permissions)).

### Who "he" or "she" means

- **A new side chat knows what you were just discussing.** Its first message also reads up to five short lines on your other chats of the last 30 minutes: what each turn was about, and where. `read his résumé first` (`先读他的简历`) then means the friend you were talking about in the main chat. Your own words come first. Under your message, **From your other chats (last 30 min)** opens the lines melo read, and a chat's name opens that chat. `[chat] recent_digest_minutes = 0` turns it off ([Configuration](../reference/configuration.md)).
- **He, she or it means whom the chat is about.** Your own profile (a profile file in your folder, notes about you) counts only when you speak of yourself (`I`, `my`, `我`, `我的`). 他, 她 and 它 never switch the subject. When it stays unclear, melo asks one short question.
- **A short follow-up keeps the subject.** `go look it up` (`去查一下`) after a long search still answers about the person the chat is about, even once melo's working memory has dropped the earlier messages.

When a side chat's run ends, the main chat shows a card under the day it happened. So does a
task of the main chat that finishes, fails or waits for your approval; a side chat's own tasks
come back to that side chat ([below](#side-chats-start-tasks-too)). Cards and the chat's lines about tasks hang on one
thin line under melo's replies, apart from the talk; only a card that asks you something is a box. A card reads:

- `Done: Fix the list_files test` with the first two lines of the answer: what changed, the tests, the commit.
- `Did not finish: Fix CI` with why it ended, in a few words in the page's language (`Could not get an answer from the model server`, `Timed out`); the server's own text stays on the task's page.
- `Ready: Fix CI`, `Done, not finished: Fix CI` or `Needs you: Fix CI` for a task that finished: how code found it ended, not what its report says ([how it is checked](scheduler.md#tasks-from-the-main-chat)). A task not finished says why in amber (`No test ran after the last change · The changes are not committed`) and offers **Finish it: run the tests and commit** (or **Finish it: run the tests**), which continues it as **Continue** does. A task that needs you shows its question, and, when one step Auto blocked in its run can still run once, that step (`Blocked: Install software`) with **Allow this step** instead of the question that restates it. Once the step is answered, on this card, on the task's page or by your typed yes, the card says how in the line the task's page shows (`Allowed once · melo can run it until 14:32`, `Allowed once · ran at 14:05`) and offers nothing to press: both read the task's own record, also after a reload.
- `Stopped: Fix CI` for a task that used all its steps before it was done. It shows the sentence of its answer that says what is left (`Remaining: run the tests and commit.`) and offers **Continue**, which picks the task up where it stopped, with what it did before. The same task reads Stopped on its page and in the panel's Activity and Workers rows; **Continue** is on its page. A task that failed for another reason stays `Did not finish`.
- `Needs your approval: Fix CI` with what it asks: the step's one-sentence summary (`Needs your OK: Close the extra Safari tabs`) when it has one. Once you answer, the same card says how ("Allowed once", "Allowed for this task", "Denied") and no longer offers **Review**.
- A title is the one the main chat gave the task, or the first sentence of your request without a "please" or a level word at its start (a first line that is only "Hi" is skipped), a path as its file name, cut at a word.
- **Open side chat**, **Open task** or **Review** goes to the source. A card is not a message: the model never reads it.

Between two of your messages a busy side chat keeps only its newest card. A stop of your own, a shutdown or an approval answered before the main chat was looked at makes no card.

Between two of your messages, finished runs of one request (tasks with the same title that ended `Ready`, or finished with no ending) fold into one row, `Ready: Fix CI · 3 times`; a click opens a line for each run with its own **Open task**. A task that needs you, one that did not finish, a stopped one and a reminder never fold. What waits for you stands last: after the finished rows, and below everything else the turn whose approval waits, so it is the thing nearest the message box.

In developer mode the panel's **Workers** tab lists the workers of the open chat: a side chat's own, and in
the main chat all of today's. For runs of earlier days, an empty main chat links **See what
worker runs did**. Default mode has neither; Activity shows the same tasks as what melo did.

## Hand work to the main chat

![The main chat: a question answered, a task that came back done, a task with its priorities open, and a pause with Undo](../assets/img/main-chat-tasks.webp)

- **Ask, or ask for work.** A chat decides first who does the work. Short work it does itself, in its own turn: it answers, reads and changes files in its folder, runs Python, looks things up and uses the browser when browsing is on. You see its steps in the chat, and each one passes the same approvals as a task's. Long work (many steps or some minutes: building or testing something, going through many files or pages, research that ends in a report) it gives to a task, and the chat stays free for you.
- **It starts right away.** You say what you want, in your own words; the chat's model reads it and decides, and no list of verbs stands in its way (`去签到`, `order me a pizza`, `fix tests/cli/test_e2e_pty.py and commit`). It does not ask first, and it never answers with only what it is about to do: it does it, or says why it cannot. A go-ahead counts (`I authorize you to apply`, `我授权你去投简历`), and so does a request in steps (`look at this site and tell me the latest prices`, `分析一下这个仓库，告诉我它的结构`).
- **A task is for long work, and for work you want beside the chat.** The chat starts one by itself for long work, or ask for one (`/task …`, `run this as a task`, `do this in the background`): one line and **Stop**, `Started: Plan the trip` (developer mode adds `· P2`). Your message is the task's request, word for word, and its result comes back as a card.
- **Work that turns out long moves on.** When work the chat began itself turns out long, or what is left would fill its context with large tool or page output, it hands that to one fresh task and its turn ends with a line naming it: the task starts from your message, what the chat did and found (which calls worked, which failed, and which may have done part of their work) and what is left, and never redoes what is done. While the chat has a browser page open, a take-over card or an approval waiting, holds a password you gave in that message (typed, or the one in a file when your words say to sign in or name a password, give none of their own, and the files hold exactly one password, none of them too long to keep), or sits in a folder no task may use, the work stays in the chat; no sign-in moves with it. Secrets in what the chat wrote or its calls returned are redacted before the task gets them. **Retry** on that message finds the same task, and a Stop as the task is admitted shows it, and it runs.
- **What needs only an answer gets one.** A greeting, a short message with nothing to do in it (`thanks`, `今天天气不错`), a question (`why do the tests fail?`, `is it done?`), text to translate or explain. Words that say not to act (`just explain how`, `don't change anything`) are read by the model like any others, and a request too unclear to act on gets one short question back.
- **Say it in code's words.** `/task …`, or **Run as a task** beside the send button (developer mode), starts the message as a task at once; no model reads it to decide. A task started this way begins at once under the first words of your message, and is renamed in 3 to 8 words in the language of your message (`Hide the recovery card in default mode`) when the model answers one short request, made at the lowest priority so it never waits ahead of real work; a short message is its own title, a name you gave the task meanwhile stays, and the first words stay when the model does not answer in 6 seconds. `stop`, `pause`, `resume` or `first` with a task's name (`pause the slides`, `暂停 Fix CI`, `先做 slides`) changes that task at once, with **Undo**.
- **Priorities as P0 to P5.** In developer mode a line shows the code (`P1`); hover it for what it means. In default mode a task's start says nothing of it, and a level you change reads in words (`today`). Your words choose: "urgent", "asap", a deadline today (`by 5pm`, `今天下午三点前`) or "today" anywhere in the request (not "not today", "今天不用") make it P1 (today), "no rush" makes it P3 (when there is time), anything else is P2 (normal). In developer mode **Change** on the line opens the list (`P0 · runs first; others wait`, `P1 · today`, `P2 · normal`, `P3 · when there is time`, `P4 · only when the computer is idle`, `P5 · on hold until you start it`) and sets your own priority, which stays.
- **Follow up on a task.** `finish it`, `commit it` or `继续那个任务` goes to the task you mean: a running task reads it at its next step, and the same job of one that stopped before finishing goes on in a fresh task, with what it did before as background. The line says where it went: `Sent to: Slides`, `Continued: Fix CI`.
- **A task runs once.** A worker does one job in one run, which keeps its context small; when it has ended, it never runs again and its record stays. New work on what it finished starts a new task (`Started: Add the prices`), which works in the same folder and gets the old task's result and the files it wrote as background. What it left unfinished (it failed, ran out of steps or was stopped) goes on in a fresh task with your message, the calls it made and its last report; **Continue**, **Finish it** and **Allow once** do the same. A stopped task gets one fresh task: a second tap or message goes to that one, also one that arrives as the task ends. The stopped task's page says where its work goes on, with no **Continue**, and a **Continue** tapped on its card in the chat says the same instead of continuing it again.
- **Continue a stopped task by name.** `Continue the recovery card task where it stopped: run the tests` or `继续恢复卡片任务` continues the one stopped task whose title those words name, at once and with no suggestion to tap, with the whole message after its request. When several stopped tasks match, each is offered as a one-tap `Suggested: send your message to …`. One still working says so (`Fix CI is still working.`). A task that finished has nothing to continue, so the chat reads your whole message and answers or starts new work.
- **What got done.** The main chat reads each task's state and the first lines of its answer, so it can name the commits tasks made. Asked about one task (`is it done?`, `what did it find?`), it reads that task's whole answer and the files it wrote, and answers from them; it never needs the task's page.
- **Only your own words change a task.** What you type counts; an attached file, a web page or a task's report that says "stop the slides" changes nothing: the line reads `Suggested: stop Slides`, and one tap applies it.
- **The work runs in the chat.** Every chat, main or side, works the same way. There is no small step limit: it goes on while it makes real progress, to about 500 steps, and when it stalls or reaches that, it stops and says what is left. While it works, what you send [goes to it as a note](#add-words-while-melo-works), and Esc stops it. **Retry** on a message that started a task finds that task: it never starts a second one. It answers in the language of your message. While you type, finished results wait until you send or pause for 10 seconds; one that needs your approval shows at once.

## Side chats start tasks too

- **Same words, same priorities.** In a side chat, `/task …`, **Run as a task** (developer mode), `pause the slides` and the model's own judgment work as in the main chat, at the same P0 to P5 levels and under the same scheduler. The first message of a new side chat is read like any other.
- **A task comes back to the chat that started it.** Its line (`Started: Find hotels` with **Stop**; developer mode adds `· P2` and **Change**), its card, its approvals and its result appear in that side chat, which reads the tasks that ended before its next answer. The main chat gets no card for them.
- **The main chat still sees everything.** Activity and Workers list every task; one a side chat started reads `From Trip plan`.
- **Your own chat first.** `pause the slides` means this chat's slides task before another chat's. Words that fit several tasks of this chat show a one-tap `Suggested: pause …` for each, and never reach another chat's task. A task of another chat can still be named, by its id, its chat or words that fit none of this chat's own: its line reads `From Main chat` or `From Trip plan`. The same holds when the model picks the task.
- **Only your words start a task.** When the model starts a task, your own message is the task, never the model's wording (the model only names it); a file, a page or a task's report can give the model the idea but never the request. A turn melo began itself (a task's question, a follow-up) starts no task and changes nothing: it reads and answers.
- **Deleting a side chat** hands its unfinished tasks, chores and waiting approvals to the main chat in one step, including tasks that end while it goes. The main chat then shows their cards and reads their results; finished tasks keep their record.
- **A task that waits for you shows on the list.** The side chat's row gets the dot and reads `A task waits for you`; the main chat's row reads `1 waiting in a side chat`.
- **The main chat hears when a side chat finishes work.** With your next message it reads one line, the side chat's name and a short summary, when that chat answered or one of its tasks finished. It is information, not a message.
- **Chores and reminders belong to the chat that made them.** One saved in a side chat runs as that chat's task and reports there, not in the main chat's check-in.
- The terminal's `/task …` starts a task in a side chat too, once the chat has its first message.

## When a task is stuck, it asks its chat

- **The task asks the chat that started it, not you.** A row appears there: `Task “Fix tests” asks…`. **Show the question** opens what the task wrote: the background, where it stands, the problem, the likely causes, its options, the help it needs, and what to tell you if nobody can solve it. Its card and Activity read `Waiting on Main chat` (or the side chat's name).
- **melo answers when the chat is free.** It never interrupts your own message. It reads the question as information, can look things up with the chat's reading tools and its memory, and answers: `melo answered “Fix tests”`, with the answer quoted. Several questions to one chat are answered one at a time, oldest first.
- **Or it asks you.** When only you can decide, a row reads `“Fix tests” needs your answer`: what the task is doing, what happened, what was tried, the choices with their trade-offs, and the question. Type in its box and **Send answer**, or, when only this task waits for you in the chat, just send your answer as a message: it shows `Sent to “Fix tests” as your answer` with **Undo** for 10 seconds, then reaches the task (a question, a request for new work, a command or a control phrase stays the chat's). When several wait, use each box, or `/answer Fix tests: B`.
- **Answer a task that needs you where you are.** When one task of the chat waits for you because it needs you (one that ran out of steps has **Continue** instead), and its card, its blocked step or melo's follow-up on it is the last thing the chat showed (on your page when you sent: a result held back while you type does not count), your next message goes to it the same way, with **Undo**, without naming it. A plain yes (`allow`, `ok`, `go ahead`, `允许`, `好的` and a few more, as the whole message) allows the step its card offers once instead, as **Allow this step** does, when that step is the one step still blocked and of a kind your words may allow; the chat says `Allowed once: Install software; the task goes on with this step`. A step that only a click allows still needs the click. Two tasks waiting, a message of yours or another task's line since, a question, a command or words that name another task: the chat reads your message as usual. Only what you type counts: a page, a file or a task's report never answers for you. The task waits for you as long as you take; its card reads `Waiting on you`.
- **A message you send meanwhile waits its turn.** While melo answers a task, you can still type and send: your message shows as `Queued: this runs once melo has answered “Fix tests”` and runs right after, before the next question, even if the gateway restarts first.
- **No answer in 10 minutes.** If melo neither answers nor asks you within 10 minutes of starting on the question, the task goes on with its best option and says so in its report (`No answer for “Fix tests”; it goes on with its best option`). Stopping the task ends the wait at once and stops melo's work on it; a restart marks the question expired.
- **Answers never approve.** melo's answer is information for the task. A step that needs your approval still asks you (or Auto's review), as before. The task's words never count as yours: in Auto they allow no step melo blocked, melo never remembers them as something you said, and melo's turn on the question has no Retry.
- A task asks at most 3 times per run. A side chat that was deleted hands its tasks' questions to the main chat, marked `From <task title>`, a question that waits for your answer included.

## Approvals a task asks for

- **Asked in the chat that started it.** A task's step that needs your approval shows in that chat, inside its card (`Needs your approval: Fix CI`): the step as code shows it, why melo asks, and **Allow once**, **Allow for this task** and **Deny**. The task's page shows the same card.
- **melo explains, you decide.** For your own tasks and chores, under the step, **melo's explanation** says in a few sentences what the task is doing, why it needs this step, what could go wrong and what each answer does. melo writes it from the task's title and goal, the step and the review's reasons, never from what a tool printed; it takes up to 20 seconds, and the card works without it. It cannot approve anything: the step, the buttons and what your answer allows come from code. A task another program or melo itself started gets no explanation.
- **Allow once runs this call only.** The same step asks again later, in this task, in the chat's other tasks and in the chat itself. To stop the asking, choose **Allow for this task** (or **Allow for this chat** in the chat): that is a grant, listed in [Permissions](#permissions), where you take it back. A grant of the chat covers its tasks' routine calls of that tool; melo reads every file changed since the chat began, so a file an earlier task wrote is checked before the grant covers a step.
- **"Allow for this chat" covers its tasks.** A tool you allowed for a chat runs its routine steps in that chat's tasks too, as in the chat, after melo reads the files changed since the chat began. What you allowed in the main chat counts there too; another side chat's grants never do, and **Allow for this task** holds in that task only.
- **What always asks.** A high-impact step (installs, downloads, deletes, sends data out, changes outside the folder, uses your keys, changes melo itself) asks every time, however often you allowed it before. So does a step melo could not see all of (a file too big to read, a command built while it runs). A call of a tool melo's rules do not read (a web or MCP tool, such as one that pays an invoice) asks however often you allowed it once; only **Allow for this chat** covers its later calls. Chores, and tasks another program or melo itself started, never use a chat's grants; a chore's run asks every time, in the chat that saved it. The one exception is a human check: a chore follows its chat's **Always in this chat** for it. Auto's **Allow once** is one step, once. A task you create with your gateway token (`melo task new`) is yours, and uses the main chat's grants.
- **In Auto.** A step Auto stops in a task you started asks in its chat, on the same card, and the task waits. A step Auto blocked in a task shows in its chat too, with what was blocked, why, and **Allow once** while the job is unfinished. A job that finished offers none: the step is new work you ask the chat for.

## When a task it started stops, the chat follows up

- **When it finishes, the answer comes back.** When a task the chat started ends **Ready**, the chat takes one turn of its own: a row reads `melo follows up on “Plan the trip”: it finished`, marked `melo started this`, and the chat tells you the result first, in your language, then its key points and the files it wrote. A long answer is summarised, with the rest under **Open task**. It happens once per ending, is skipped if you wrote after the task ended (your turn read it), never continues the task, and Activity reads `Told you the answer: Plan the trip`.
- **The chat owns the outcome.** When a task a chat started fails, runs out of steps, ends `Done, not finished`, or needs you, that chat takes one turn of its own, right after the task's card: a row reads `melo follows up on “Fix CI”: it ran out of steps`, marked `melo started this`.
- **It does one thing, then says so.** It continues the work once in a fresh task, with a short note (`Continued: Fix CI`), works around a step that was blocked (for an install Auto blocked: use what is installed, or write it without the package), or asks you one clear question. Then it tells you in one or two lines what got done and what did not.
- **A blocked step never runs again.** Where the chat continued the work or handed it on, or you tapped **Continue** or **Finish it**, a step you, Auto or melo's rules blocked is refused before it is reviewed (`blocked earlier`), and so is one alike: the same command spelt another way (`pip3` for `pip`, other quoting, a quiet flag) or the same install. A site you or Auto refused when `web_fetch` or the browser asked about it counts too: that address, however it is written (spacing, capitals, `http` or `https`, another port, its international spelling), every other page of that site and its other hosts (`login.example.com` after `www.example.com`), even when your request names it, and an answer kept for the site does not run it. A command or script that fetches an address on that site is refused as well: `curl`, `wget`, Python's `urllib`, `requests` or `httpx`, an address kept in a variable or written into an f-string, an address given to `curl` or `wget` without `http://`, a connection opened to the site's name, the site behind a reader, an archive or a proxy, in capitals, with a port, percent-encoded or in its international spelling. While a site is refused, a step whose address is put together while it runs is reviewed, even one that would otherwise run as routine, and the review is told which sites were refused. The task must find another way. Your own message to the task, or **Allow once**, lifts that. A call that was cut off and may have done part of its work asks you on a card before it runs again as it was, in any task of the job, in Ask and in Auto, whoever moved it on, unless it only reads (a page action or a navigation does not): no grant or review lets it run, and your Allow runs it once. A question that was withdrawn (the gateway stopped) blocked nothing.
- **It continues at most once.** For each message of yours, the work is continued by its chat at most once, however many fresh tasks it went through; if it stops again, the chat only reports and asks. A third stop gets the card alone.
- **Your turn wins.** The follow-up never interrupts your message: it waits until the chat is free, and a message you send meanwhile is queued. If you write after the task ended, your turn reads the ending and the follow-up is skipped.
- **Stop stays while more can run.** The `Started` row keeps **Stop** while the task runs, and after it ends for as long as the chat's follow-up on it may still continue it. Stop then ends the follow-up too, waiting or running, so it never continues the task. Once nothing more will run (the follow-up asked, reported or was skipped), Stop goes. Stop on a task with nothing left to stop offers no **Undo**.
- **It survives a restart.** A follow-up still waiting when melo stops runs after the restart, with the same limits: once per ending, at most one continue per message of yours, and skipped if you wrote after the task ended. One whose turn had begun never runs twice.
- **Each card keeps its own report.** When a task given from the task page ran more than once, each `Did not finish` card shows the report of its own run.
- **It approves nothing.** A follow-up has only the task tools (no web, browser, file or shell tool) and widens no permission. What it tells the task reaches it as melo's note, never as your words: a step your own request did not ask for stays blocked in Auto, whatever the note says.
- **Only tasks the chat started.** A run of a chore the chat saved counts as one of them (a reminder's does not). Tasks started from the task page and other chats' tasks keep the card alone. Activity reads what the chat did: `Continued the task: Fix CI`, `Asked you: Should I split it?`.
- **Turn it off** with **When a task stops, its chat takes over** in Settings → Chats and folders (developer mode), or `[tasks] follow_up = false`.

<a id="slash-commands"></a>

## Slash commands

The commands of the [terminal](terminal.md#commands) work on this page and in the [app](macos-app.md). Type `/` at the start of the message box to see them.

- A menu opens above the box, as the folder and Ask/Auto menus do, and narrows as you type. ↑ and ↓ move, Enter or Tab picks, Esc closes it, and a click picks. A pick fills in `/name ` for you to finish; a command with no arguments runs on Enter.
- A command is never sent to the model. Its answer is a row under the conversation, which goes when you send your next message (`/clear` clears it at once).
- An unknown name answers `Unknown command /NAME — /help lists them`. `/etc/hosts is broken` and `//text` are messages, as in the terminal; to send a message that starts like a command, begin it with a space.
- A command that changes the chat is refused while a turn runs (`/new is not available while a turn runs (Esc stops it).`), and the message stays in the box.
- The list is the terminal's own, served by [`GET /api/commands`](../reference/api-v2.md#get-apicommands) in English and Chinese.

| Command | On this page |
|---|---|
| `/help [command]` | The everyday commands; `/help all` lists every one and `/help NAME` gives one command's details. |
| `/model [name]` | With a name: switch the model and save it, with the Settings checks. Alone: open Settings at the model server. With melo_serve: alone, the list of its chat models; with a name, a switch every chat follows. With a key for a hosted service: alone, the list of every model you can use now; with a name or `service:name`, a switch from the next message ([Hosted models](hosted-models.md#switch-between-models)). |
| `/backend [url]` | With a URL: switch the model server and save it. Alone: open Settings at the model server. |
| `/new`, `/clear`, `/side [title]` | A new chat, made by its first message and a side chat under the main chat; `/side` names it, and `/clear` also removes the rows shown. |
| `/chats`, `/sessions [filter]` | Open the chat list, filtered by the word. |
| `/resume [id\|title]` | Open the chat with that ID (or its start) or title; with several matches, the filtered list. |
| `/main` | Back to the main chat. |
| `/task TEXT` | Start a task in the open chat. In a chat not made yet it waits for the first message. |
| `/answer TASK TEXT` | Answer a task that waits for you. |
| `/status [all]` | The model and whether it answers, the folder, the chat, the approvals and the context; `all` (and developer mode) adds the version and the gateway. |
| `/usage` | The chat's turns, tool calls, tokens and time. |
| `/progress` | The step, attempt, phase and context of this turn. |
| `/mcp` | Says that MCP servers are not part of the console yet. |
| `/undo`, `/retry` | Forget the newest finished turn (its file changes stay), or run it again. |
| `/copy [turn]` | Copy an answer to the clipboard; without a number, the latest. |
| `/title [text]` | Show or set the chat's title. |
| `/memory [window\|import]` | Open the Memory tab; `window` shows the model's window; `import` also presses **Look for notes**, so the files other assistants keep are listed (a row says so when memory is off). |
| `/approvals [ask\|auto\|deny\|forget [tool]]` | Show the chat's mode and grants; `ask` and `auto` switch it and `forget` takes grants back. In a side chat, `forget NAME` on a grant from the main chat takes it back everywhere, and `forget` alone takes back only that chat's own (in the main chat: every grant that holds everywhere). `deny` is for the terminal. |
| `/grants [revoke N]` | What you allowed, numbered as **Settings › Permissions** lists it; `revoke N` takes the N-th back. |
| `/browser [on\|off\|chrome\|safari]` | Show or switch reading web pages, in Chrome in the background or in Safari's windows, saved in `[browser] driver`. |
| `/attach`, `/attachments`, `/detach` | `/attach` opens the picker for a picture or a text file; `/attachments` lists what waits and `/detach N` (or `/detach all`) removes it. |
| `/verbose`, `/thinking`, `/stream [on\|off]` | How the page shows a chat, kept in this browser: tool calls open, thinking open, or steps shown only when a turn ends. |
| `/expand [turn.call]` | Open a tool call with its whole output. |
| `/exit`, `/quit` | In the app: ask, then quit. In a browser tab: say to close the tab, since melo keeps running. |

## Chores that repeat

![The Scheduled tab: the check-in with its switch, two chores with their next run and how the last run went, and one chore open to Run now, Edit time and Delete](../assets/img/scheduled-tab.webp)

- **Say it once.** A message such as `every weekday at 8:30, brief me on my calendar` or `每天早上九点 总结我的邮件` is saved as it is, your whole message, with one line and **Undo**: `Saved: brief me on my calendar · every weekday at 8:30`.
- **Now, and on a schedule.** `download my invoice now and every day at 9` saves the schedule and runs it now in this chat, as any task you ask for. The chat reads which you mean. A message that points at earlier work (`do that every day at 9`) keeps all its words, and what it points at is saved beside them.
- **The card stays for what melo did not save.** When your message names a schedule and the chat saved nothing, a card asks `Run this every weekday at 8:30?` with **Save chore**, **Just once** and **Cancel**. So do words that say not to start yet. When the message started its work already, the card has only **Save chore** and **Cancel**: a message starts its work once, however it arrives.

  ![The main chat asks "Run this every weekday at 8:30?" with Save chore, Just once and Cancel](../assets/img/chore-proposal.webp)

- **Three schedules.** Every day at a time, every weekday at a time, or every 1 to 24 hours. With no time, a day part sets it (`every morning` is 8:00, `every evening` 18:00), else 9:00. The chat says so when you ask for something else (`every Monday`, `every 30 minutes`), and saves nothing.
- **Runs on time.** While the gateway runs, a chore starts as a task within three minutes after its time and never before. It works as a task you asked for in that chat does: it has the browser when browsing is on, and its approvals are the same cards in the chat that saved it. A run missed while the computer slept happens once when it wakes.
- **One folder, one run at a time.** A chore works in one folder of its own in your workspace, `chore-` and its id, and every run uses it: a later run finds what the last one left. The files in it are plain data a run may read, never instructions it follows. Runs never overlap: one place decides whether a chore may start, whether the time comes, **Run now** is pressed or a message saves the chore and runs it at once. While the last run has not ended (it may be waiting for you or paused), the next time is skipped, the chat that saved the chore says so once (“Skipped ‘Daily check-in’ this time: its last run has not finished, and it may be waiting for you.”), and **Run now** says the same instead of starting a second run. A run that has waited for you for an hour or more does not hold the chore for good: when the next time comes, melo ends that run with a plain line in the chat (“Ended the last run of ‘Daily check-in’: it had waited for you for over an hour, and the chore's next time came. This time's run starts now.”) and starts the new one; a run whose page you hold is yours and is never ended this way. The run that starts with the message (`… now and every day at 9`) works in the chore's folder when the message saves the schedule first; when the chat starts the work first and saves the schedule after, that one run stays in the chat's folder and the later runs use the chore's. Deleting the chore removes the folder with what its runs left; a run already going when you delete it is not cancelled but goes on and ends in place, the chat says where its folder is, and melo removes the folder once that run has ended, as long as melo has not been restarted by then (a restart forgets it, and the folder stays where the chat said). When a folder cannot be removed, the chat says so and where it is, and it stays. melo only ever removes the folder of a chore it deleted; it never removes a folder because of its name. A reminder's run only says the reminder: no browser, and a folder of its own like any run.
- **A run that waits for you** asks in the chat that saved it, with the same card as any task, and the check-in names it once it has waited a whole beat. A chore asks each time: it uses no chat's grant but its **Always in this chat** for a human check, and in Auto it blocks what it would ask. A picture check on a page is worked through by a chore's run itself, as in a task you started, with nobody at its side. What a repeating chore posts in your name asks each time, whatever the words of its request say: its schedule counts as well as the words. A password you gave in the message is typed by the run it was given for only: later runs rely on the sign-in melo's browser kept, or ask you.
- **When it ends,** the chat that saved the chore follows up as it does for a task it started (see below), except for a reminder, whose card says it.
- **The Scheduled tab.** The check-in on top, then each chore: its title, when it runs, the next run, how the last run went, and a switch. How the last run went reads as its task's page says it (`Last: Needs you · …` for a run that stopped on a blocked step, `Last: Ready · …`, `Did not finish`, `Stopped`, or `Running now` while it goes on) and opens that run. Under it, one line for each site its runs may type and send on without asking, with **Remove** (see [Auto mode](#auto-mode)). A row opens to its priority in developer mode (`Priority: P2 · normal`), **Run now**, **Edit time** and **Delete**, which asks first. The main chat lists every chore, a side chat its own; `#/scheduled` opens the tab.
- **Reminders.** A reminder is a chore that runs once. `remind me to send the report on Friday at 9`, `in 2 hours` or `提醒我明天九点给导师发邮件` makes a card, `Remind you on Fri Oct 30 at 9:00?`, with **Remind me** and **Cancel**; it always shows the date and time it understood. Relative words (`tomorrow`, `Friday`, `in 2 hours`, `下周一`, `两小时后`) become that date and time on your computer's clock. A day with no time is 9:00, a bare hour from 1 to 6 is the afternoon, and a time already past today is tomorrow unless you said `today`. `Friday` on a Friday is next week. With no usable time, the main chat asks `When should I remind you?` and never guesses: reply with just a time (`Friday at 9`). `remind me of …` or `remind me about …` with no time is an ordinary question for melo, not a reminder. At its time, melo says the reminder in a card, `Reminder: send the report`, also when the run did not finish. A reminder missed while the computer slept or the gateway was down runs when melo starts again and says how late it ran (`ran 2 h late`). On the Scheduled tab it shows its date and time, then `Reminded today at 9:01 AM`; **Edit time** takes a date and time and sets a done reminder again. A reminder is at most a year ahead; `every …` still makes a repeating chore.
- **The check-in.** Every 30 minutes it looks for something to tell you: a chore's finished run, or a task, of the main chat or of a side chat, that has waited a whole beat on your approval or your answer. With nothing, it does nothing; otherwise it posts one card in the main chat. melo writes the line from those facts (`Slides: waiting for the user; Photo sort: waiting for the user`), and when they do not fit one line it gives the first ones and `and 3 more`. It asks no model, so it never wakes the model server or waits behind a running task. Its switch turns it off.

## Add words while melo works

While melo answers, the message box stays open. Its placeholder reads `Add to what melo is doing…`, and Enter adds your words, and any pictures in the row above the box, to the turn that runs instead of waiting for it to end. Shift + Enter still starts a new line, and Esc still stops the turn.

- **melo reads them at its next step.** The model gets the words as your message, marked as added while it worked, so a correction (`use the 2024 file instead`) changes the rest of the turn, not only the next one. A turn takes at most 20 added messages of 8,000 characters each.
- **Sent, then seen.** The words show in the turn's work as `Sent. melo reads it at its next step.` and change to `Seen by melo at step 3` when the run has read them. They are stored with the turn, so a reopened chat shows them too.
- **Pictures go along.** Drop, paste or pick a picture while melo works, as for [any message](#pictures-and-files-in-a-message), and Send adds it to the turn with your words. A picture alone is enough. The note shows its pictures in the turn's work, and the model gets them beside the words at the next step. The same pictures, the same limit of 4 and the same refusal for a model that cannot read pictures apply. If the model turns out not to read pictures before the note is read, the model is told you attached them and that it cannot see them, and the pictures stay in the chat for you.
- **If the turn ends first**, they start nothing. Your words and pictures go back to the message box, whatever ended the turn, and the turn says `Not read: the turn ended first. Nothing runs from it until you send it again.` You press Send when you still want them. With no page open, the chat keeps a row with your words.
- **Everywhere, and never a second turn.** Send during a turn is a note in the console, the app, the terminal's main chat and the API. A message that cannot be a note keeps its place and runs next, up to 20 waiting. The terminal's local chat has no notes and keeps its own queue.
- **Never on their own.** A note that misses its turn is never run for you, also after a restart, when melo may have read it before it stopped: the chat says so, with your words.
- **After a restart, pictures stay behind.** A note that outlives a restart comes back without its picture. The chat says so, and you add it again.
- **Once, after an upgrade.** Messages and notes waiting from an older version do not run: each comes back the same way, and a real queued message needs one more Send.
- **What does not change.** Only a turn you started takes added words; the turn melo starts to answer a task's question does not, and a task's question is still answered as [described above](#when-a-task-is-stuck-it-asks-its-chat). Text files cannot join a turn that runs (`Text files cannot join a turn that is running. Send them when melo is done, or press Stop.`).
- The same words reach a client through [`POST /api/turns/{turn_id}/notes`](../reference/api-v2.md#post-apiturnsturn_idnotes).

## Pictures and files in a message

Add a picture or a text file to a message the same way in the browser and in the [macOS app](macos-app.md):

- **Drop** it anywhere on the page. A hint, `Drop pictures or text files to add them to your message`, shows while a file is over the window.
- **Paste** it with Cmd-V while the cursor is in the message box; a screenshot on the clipboard works. Pasted text is still text.
- **Pick** it with the paperclip beside the box, or type `/attach`, which opens the same picker. `/attachments` lists what waits and `/detach N` (or `/detach all`) removes it.

What you add shows as a row above the box: a thumbnail for a picture, a chip with the name for a file, each with a remove button. A click on a thumbnail opens the picture larger. A message may be only pictures or files.

- **Which pictures.** PNG, JPEG, GIF, WebP and HEIC. melo tells what a file is from its content, not its name. A HEIC picture is converted to JPEG, the first frame of a GIF (an animated one too) and a WebP picture to PNG, and a picture whose long side is over 1,280 pixels is scaled down to that, as the screenshots melo takes are, so a huge photo costs the model no more than a small one. melo reads a picture's size from its header before anything opens the file (every frame of a GIF, every image of a HEIC) and refuses one over 120 million pixels. A file that is not a picture and a damaged picture are refused too, each with a sentence that says what to do. When this Mac cannot scale or convert a picture, melo says so (`Save a smaller PNG or JPEG`) instead of sending it as it is.
- **Nothing hidden goes along.** Every picture is rewritten before the model gets it, also one that needed no conversion: the EXIF block (the camera, the time, often the GPS position), XMP, comments, text chunks and anything after the end of the picture are removed. Only the orientation and the few numbers a decoder needs for color are kept, not an embedded color profile. A picture melo cannot rewrite is refused, not sent.
- **How many.** Up to 4 pictures in one message, each at most 20 MB before scaling. A picture takes the model's context (roughly a thousand tokens at that size) and the models that run here have windows of 8,000 to 32,000 tokens, so more than four would crowd out the conversation.
- **Text files.** Up to 8 files, 256 KB each and 512 KB together, sent with the message as the terminal's `/attach` sends them. Any other kind of file answers `report.pdf: this kind of file is not supported yet. Pictures and text files work.`, and the page stays where it is: a file dropped beside the message box never opens in the window, in the browser or in the app.
- **A password in a file.** A password a text file gives (`password: …`) is cut out of it as one you type is: the chat, its search and the model hold a placeholder such as `[password-482913075]` in its place (its number is new each time melo starts), never the value. melo types it on a site your own words name when your words say to sign in (`sign in`, `log in`, `登录`) or name a password, the files hold exactly one password (none when a password in them was too long to keep) and your words give none of their own (a password in your words is used instead). It is not typed when the files hold two or more (melo hands you the page), for a message sent again with **Retry**, one that waited in the queue and started after a restart, or a task's, and a site that only the file names never counts: name the site in your message. melo cannot tell which site a file's password is for, so a message that says to sign in types the one it holds even when it belongs to something else; a placeholder such as `password: ********` is typed as written. A file is searched for a password only where the file itself says what it is (`password: …`); a file that holds nothing but the value is stored as it is. A key, token, key block or one-time code beside a password stays readable, unless cutting the password leaves a secret of that text unrecognised (for example, it took the keyword or first line, as in `password: Bearer …`, or the password is itself key-shaped): then every secret of that text is hidden with it. A file sent before this version may have left the password in the chat's text, and melo reads an old chat's text again whenever that chat continues. Search the sidebar for the word your file used (`password`, …), which looks in what you sent, and close the chat, which deletes it. The main chat cannot be closed, so change a password that sat in a file you sent there.
- **In the chat.** The message bubble shows the pictures and the file names. The pictures are stored with the turn, so a reloaded or reopened chat shows them again, and **Try again** sends them again.
- **A model that cannot read pictures.** When the chosen model is known not to read them, a line under the thumbnails says so and links to Settings → Model. melo never drops a picture silently: sending anyway is refused with `This model cannot read pictures. Choose a model that can in Settings → Model, or send the message without the picture.` When melo cannot tell, it sends the picture. If the model or its server then refuses the turn because it does not take the picture (an error that says so; a bare error, a message about length or a picture it could not decode keeps its own text), the turn ends with one line, `This model can't see pictures; choose one that can in Settings → Model`, and melo treats the model as text-only for the next minute or until you choose another model; after that it asks the model's server again and goes by its answer. While it lasts, the composer warns, and **Try again** on that turn is refused (`model_text_only`) instead of sending the same picture again.
- **Waiting pictures stay in memory.** A picture you added and did not send is kept in memory on this Mac for 30 minutes at most (24 pictures, 64 MB), and is gone when you remove it or close the chat. Nothing is written to disk until the message is sent.
- **Not a task.** A message with a picture goes to the model; no rule reads it, so it cannot start a task, and **Run as a task** and `/task` are refused with pictures (`Pictures cannot start a task. Send them as a message.`).
- **While melo works**, a picture is added the same way and goes to the turn as [added words](#add-words-while-melo-works) do; a text file is told it cannot.
- A client sends a picture with [`POST /api/uploads`](../reference/api-v2.md#post-apiuploads) and names it in the turn's `images`, or in the `images` of a [note](../reference/api-v2.md#post-apiturnsturn_idnotes).

## Pictures in the answer

When a turn makes a picture it shows under the answer, in the browser and in the
[macOS app](macos-app.md) alike, so a chart you asked for is there without opening the
folder.

- **Pictures melo took.** A picture a step took (a page melo looked at with
  `browser_screenshot`, another app's window) shows in its own row under the turn's steps,
  also when they are folded: "Picture 1", "Picture 2" and so on, each once, opening at full
  size like the others. These are the chat's own copies
  (`GET /api/sessions/{id}/blobs/<sha256>`), so they stay with the chat.
- **Which pictures.** The PNG, JPEG, GIF, WebP and SVG files the turn created or changed in
  the chat's folder, and the picture files its answer names by a path inside that folder.
  The gateway looks at the folder when the turn starts and when it ends, so a file written by
  `python_exec` counts although no tool listed it. The look skips hidden folders,
  `node_modules`, links and folders more than three levels down, stops after 4,000 entries,
  and a turn keeps 24 pictures at most.
- **The row.** Each preview shows the picture, its file name and its size (`updated` when the
  turn changed an existing file). Four show; `+N more` shows the rest and **Show fewer** folds
  them. A click, Enter or Space on a preview opens it at full size; Esc, **Close** or a click
  outside closes it. The previews work in both themes and at phone width.
- **Reopened chats.** The list is stored with the turn (the
  [`turn.pictures` event](../reference/events.md#session-events)), so a chat you reopen shows its
  pictures for as long as the files are in the folder. A preview whose file is gone says
  `No longer in the folder`; one that is no longer a picture says `Not a picture file`.
- **What the gateway serves.** `GET /api/sessions/{id}/pictures?path=` returns a file only
  when its real path, links followed, is inside that chat's folder, its name and its content
  are a PNG, JPEG, GIF, WebP or SVG, and it is at most 20 MB. A path with `..`, an absolute
  path outside the folder, a link that leads out and another chat's folder are refused (403);
  a text file named `chart.png` is refused (415); nothing is written or listed. The route
  needs the console's sign-in like every `/api` route. An SVG is sent with a policy that stops
  scripts and keeps it from opening as a page
  ([API v2](../reference/api-v2.md#get-apisessionsidpictures)).

## The panel beside a chat

On the right of the chat is Melo's panel. At the top is Melo's portrait, with a ring in the colour of what it is
doing (green while it works, amber while something waits for you), the name, and one status line: `Connected ·
<model>` (developer mode adds `on <server>`), `Working on 2 things`, `Stopping…` (from a Stop until the turn ends), `Waiting for you`, or `Model server not reachable` with **Open
settings**. The panel reads the gateway again as soon as a turn this page shows ends, so it stops saying a chat works when its answer or its Stop has come.

Melo in the portrait moves with what it does:

| Melo | Meaning |
|---|---|
| Wakes up, shakes and sits | The page has just opened. It sits for a few seconds, then rests. |
| Typing at its laptop | Something runs. |
| Tilting its head | The model is answering and no tool runs. |
| Raising a paw | Something waits for you. It shows at once. |
| Lying down, eyes closed | Nothing runs. Now and then it gets up and plays (at most every 5 minutes). |

- A run that just ended closes its eyes happily with its tail up, and an error puts its ears back and its head
  low; both are still pictures. While the console is connecting or cannot reach the gateway, Melo is curled up and
  faded.
- After three minutes of plain rest Melo lies still until something changes or it is time to play.
- With reduced motion each state is a still picture, and a hidden tab or a portrait scrolled out of view draws
  nothing.

Under it are the tabs **Activity**, **Scheduled** and **Memory**, and in developer mode **Workers**. Arrow keys, Home and
End move between them, and the panel opens the tab you used last.

The panel shows what belongs to the chat that is open. A side chat's panel lists that chat's own activity and the
workers it started; the main chat's panel, and the panel with no chat open, list everything of the day. Melo's status
line follows the chat too: in a side chat it says what that chat is doing. Opening another chat reads the lists again.

**Activity** has two parts, read from [`GET /api/activity`](../reference/api-v2.md#get-apiactivity) at the page's
health poll whenever what runs changes, and every 30 seconds:

- **Now** lists what runs or waits: this console's chats, tasks, worker runs and terminal chats. Each row says what
  it is doing in plain words (`Reading notes.md`, `Running the tests`, `Waiting for your approval`), with an icon for
  the kind of work. A row that waits for you is amber. With nothing running it reads `Nothing running`, and Melo
  rests.
- **Today** lists what finished today, newest first: the title, a one-line result (`All 412 tests passed.`) and the
  time. The first eight show; **Earlier today (n)** shows the rest.

A result line is always plain words. A run stopped before it answered reads `Stopped before answering`, a picture
reads `Made a picture`, a file the file's name, and an answer that is code the first sentence around the code (`Wrote
some code` when there is none). Markdown marks are removed (code keeps its words, as in `__init__.py`, struck
text goes, and `x**2` stays), a lead-in such as `In short:` is dropped, and the line ends at the first sentence. A
chat's line says first what the chat did, then its reply's first sentence: `Offered a task: Visit example.com ·
I can't open websites myself.`; what it did is `Started a task: …`, `Offered a task: …` or `Kept a note`, and a reply that only says it handed the work to a task is left out. The main
chat's row is named in the page's language.

A row that asked you for approvals carries them folded under its result line: `1 approval`, `2 approvals, 1
waiting`, or `1 approval waiting` when every request still waits. Select it to list each request in plain words (`Write slides.md`, `Run the tests`, `Run a command`, `Open example.com`, `Click or choose on example.com`; never
the command itself or an internal name), how it was answered (`Allowed once`, `Allowed for this chat`, `Denied`, `No answer in time`) and
when it was asked. A request that waits for you has **Allow once** and **Deny** right there, the same answers as the
approval card in the chat, and **Review**, which opens the chat or task with the card and the exact content. There is
no separate Approvals tab.

Select a task's row (it ends in a `›`) and the task's details open in the panel, in place of Melo's page; see
[Tasks](#tasks). Select the row of a chat, a worker run or a terminal chat and it opens in place: a running row shows
its steps as they happen (a mark, the step in words and its time) and the last lines of the latest output. **Open
chat** or **Open run** goes to the whole view; a row that waits offers **Review**, which opens the chat or task with
the approval card. One row is open at a time, and its steps follow the session's event stream while it runs.

**Workers** (developer mode) lists the workers of the scope above: the tasks the open chat started, running ones first, then those
that finished today. Each row has the title, its state in words (`Working`, `Waiting for your approval`, `Waiting to
start`, `Paused`, `Waiting for you`, `Done`, `Failed`, `Stopped`), the same words the task's own page and the terminal
use, the time and, for a finished worker, its result line. A task that waits for your Resume after a restart reads
`Waiting for you`, and its page says why in your language. Select a task
to open it in the panel, as in Activity; a worker run opens in place. The main chat's Workers also lists the worker runs recorded as sessions, such as `melo
worker`. With none, the tab says when they appear (`This chat has not started any workers. When it hands a task to
one, the worker is listed here, running first.`). **All tasks** at its foot opens the list of every task, today's and earlier, in the panel; **All worker runs** opens the Workers view.

The panel is 340 px wide from a 1180 px window up. From 700 to 1180 px the chat list becomes a rail of initials
(hover a circle for the chat's title), and the panel stays. Below 900 px the panel becomes a sheet: the face chip in
the header (`2 · !` when two things run and one waits) opens it from the bottom; Esc, the backdrop or a drag down on
its handle closes it (Esc first leaves a task shown in it). The [macOS app](macos-app.md) now opens at 1280 × 800 so the three columns fit.

**Memory** lists what melo remembers about you: the notes it kept from what you typed
(see [Memory across chats](../concepts/memory.md#memory-across-chats)).

- The switch **Remember across chats** turns memory on or off, and a line under it says how melo searches: `Searching
  by meaning with Qwen3-Embedding-0.6B, and by words.` or `Searching by words, and your chat model picks the notes that
  matter.` Settings asks `Remember across chats with <model>?` once the model list finds an embedding model on your
  server.
- The notes come newest first, each with its topic (About you, Preferences, People, Projects, Decisions) and where it
  came from: `From chat “Planning”, Oct 30`, or `From a deleted chat, Oct 30`, since a note outlives its chat. The search
  box filters them.
- **Edit** opens the note to reword; **Save** keeps the new wording (a wording that looks like a password or key is
  refused with the reason). **Delete** asks `Forget this note?` with **Keep it** and **Forget it**.
- **Forget everything…** says what goes before it goes: `This removes 12 notes and melo's search index of 48 chat turns.
  Your chats stay. It cannot be undone.` **Keep everything** comes first. In developer mode **Rebuild search index** reads
  the chats again.
- With nothing remembered, the tab says how memory starts: tell melo something about yourself in a chat, such as `I
  prefer short answers`, or ask it to remember something.

**From other assistants** (the foot of the Memory tab) brings in what other assistants on this Mac know about you.
**Look for notes** lists the files that hold notes, as in the first-run card below; **Bring them in** writes the checked
ones and the tab says `Brought in 4 notes.` Each batch stays listed as `4 notes brought in · Oct 31` with its own
**Undo**, and an imported note says `From the file CLAUDE.md, Oct 31`.

The first time the main chat opens with a model connected and memory on, it shows one card, `melo found notes on this Mac
from other assistants`, with a row per file: its path (`~/.claude/CLAUDE.md`), `3 notes`, a checkbox (checked), and
**See what is in it**, which shows the sample lines and what was left out (`2 left out: secrets, other people’s
details`). **Bring them in** writes the checked files and the card says `Brought in 6 notes · Undo`; **Undo** takes the
batch back and keeps any note you have edited since (`Taken back. Kept 1 note you edited.`). **Not now** closes the card, and it does not come back; if melo cannot save that answer, the card says so, and may ask again next time. Nothing is written until you press **Bring them
in**, and a search that finds nothing shows no card.

Closing a chat that made notes asks `Also forget the 2 notes from this chat?`: **Keep the notes** (first),
**Forget the 2 notes too**, or **Don’t delete the chat**.

In developer mode, **Details** opens in the panel, in place of the tabs, until you close it.

## The browser in the chat

While melo browses in Chrome in the background ([Browsing with melo](browsing.md#watching-melos-browser)), the chat that asked shows a card under the task's line, like a message: **Working for this chat**, the task's goal, the live page, what melo is doing ("Opening a page", "Pressing a button"), one state word (Working, Waiting for you, You're in control, Done or Stopped), **Open browser** and **Stop**. A click on the picture or on **Open browser** shows the page big at once ([The Browser panel](#the-browser-panel)). It works in the main chat and in a side chat alike, and a web search counts as browsing. You can keep typing under it, and the chat answers while the task works.

- The card shows only what the program wrote: the state, the kind of step and the site's name, and the task's title. It never holds a typed value, a full address or a query.
- **Stop** ends that task, while the run the card shows is the one going: a card left from an earlier run stops nothing. **Done**, which shows while you have the page, gives it back to that task's run. Neither reaches any other task of the chat.
- Approvals work as they do in a chat: each step asks or is reviewed as Ask or Auto says, and a question the task puts to its chat is never an approval. Deleting the chat moves the card to the main chat.
- Melo has one Chrome. While you have a page in a real window it restarts, so the cards of other chats and tasks say "Not updating" until it is back. Four tasks you hold pages for fill the four worker slots, and the next task waits its turn.
- When the card scrolls out of view, a small bar at the top of the chat keeps the page in sight (a picture, the state and the same button); a click on it goes back to the card.
- A picture that is not arriving never reads as live: the card says "Not updating", or "Reconnecting…" while the link is being made again, and dims the picture.
- When the run ends the card keeps the last picture this page showed, and says how it ended. A click on it shows that picture big, with no live controls.
- A chat that is open again, or a page that was reloaded, shows the same card again.

## The Browser panel

A click on the card's picture, on **Open browser**, or on the **Browser** button in the header shows the page big and live at once, in the **Browser** panel, and lets you take it over. It covers the chat and Melo's panel, so the picture fills the window; there is no second click.

- It never opens by itself. **Back to chat** in its header, or Esc, closes it and puts the keys back on the card you clicked. Closing it only stops watching: melo goes on, nothing is stopped or given back, and the card stays live in the chat.
- melo keeps working while you watch. The footer says what it is doing now ("Opening a page", "Pressing a button") and its state, as the card does, and **Stop** ends the run you see (a task's run, or the chat's own turn).
- A page that has ended shows its last picture big, with no Take over, Stop or Pop out. Open it from a finished card; a run that ends while you watch turns into that picture.
- After you switch the browser to Chrome in Settings, it opens at the chat's next browser step. A chat in the middle of a turn finishes it in the browser it has.
- **Pop out** shows the panel alone and large: in the app in a window of its own, in a browser in a new window. A task's browser pops out the same way, showing that run's page and no other.
- It shows the page's title and address, the picture, and a ring with "melo is browsing" while a step runs. When the picture stops it dims and reads "Not updating"; when it ends it says why.
- It fills the area beside the chat list; on a phone or a window 700 pixels wide or less it fills the screen.
- It sends nothing while it is hidden or its tab is in the background, and closing the tab ends the picture stream.
- **Take over** in its footer makes the page yours, in a real window of melo's Chrome: the panel says "You're in control", shows a live copy and offers **Show Chrome window**. melo is paused, with no time limit. **Done** gives it back (closing the window does too), and melo reads the page again before it acts ([Taking over melo's browser](browsing.md#taking-over-melos-browser)).
- When melo needs you on a page (a sign-in, a code, a QR code, a check that you are human), a card in the chat says "melo needs you to sign in on example.com" with **Take over** and **Not now**. A page with a form you began offers **Take over here** first, since a window would clear the form.
- When the page is in the panel (a form you began, or a window that could not open), your clicks, keys, input method and pastes go to it there, and Esc is the page's: Control-Esc moves to **Done**. A box the page puts up while you have it (an alert, a confirm, a question) shows in the panel.
- The chat's record keeps one line of how it ended: "Gave the page back to melo at 10:42.", "You closed the Chrome window, so the page went back to melo." or "melo was stopped while you had the page."
- Nothing you type is kept, and the picture is not saved anywhere.

## Melo in the page

The page shows Melo, the Akita puppy, where it already had a picture, in the
app and in a browser alike. The panel's portrait moves; the other pictures are
still, and all of them come from the same moves. Under
`prefers-reduced-motion: reduce` the page shows still pictures.

- **Empty chat.** Melo sits above *What would you like to work on?*.
- **The panel.** Melo's portrait sits at the top of the [panel](#the-panel-beside-a-chat) and moves with what it
  does: it wakes up, works, thinks, raises a paw while something waits for you, rests and plays now and then. It
  closes its eyes happily for a few seconds after a run ends well, and puts its ears back and its head low when the
  model server cannot be reached or a run just failed; those two are still pictures. While the page cannot reach
  localmelo, Melo is curled up and faded.
- **Replies.** Beside `localmelo` on each reply is Melo's face. On the newest
  reply it shows that turn's state: at its laptop while it runs, with a raised paw
  while an approval waits, eyes closed and happy when it is done, and with its ears
  back when it ended with a warning or an error. A turn you stopped is not a failure and
  keeps the calm face, and so does every older reply.
- **Approval bar.** The bar that reads "melo is waiting for your answer" shows the
  face with the raised paw.
- **Brand and tab.** The sidebar and the sign-in screen show Melo's calm face, and the browser tab uses it, cut
  closer, at 16, 32 and 64 px. The Main chat shows the smiling face, as the replies do.

The faces are decoration where words already say the state (the live row, the
work box, the failure card, the approval bar), so they have empty alt text. A
finished answer with no work box above it has nothing else saying it is done, so
its face reads `Done` to a screen reader. The empty chat's Melo keeps the alt text
`Melo, the localmelo Akita mascot`; the panel's moving portrait is decoration, as its status line says the same in
words.

The moves are served from `/assets/moves/`: one picture per state holding its frames, a still of each state
(`poster-<state>.webp`), the done, error and offline stills (`poster-done.webp`, `poster-error.webp`,
`poster-offline.webp`), Melo's head (`face-idle.png`, `face-brand.png`, `face-working.png`, `face-waiting.png`,
`face-done.png`, `face-error.png`) and the tab icons (`favicon-16.png`, `-32` and `-64`), and `manifest.json` with
each frame's duration: 1.6 MB in all, and the app shows the same files. The page loads the moves once, when it opens.

The pictures are transparent. On the light canvas the dark brown line round Melo
sets it off (about 15:1 against the console's light surfaces); on the dark canvas
its orange fur does (about 7.5:1 against the dark ones; 3:1 is the bar for a picture).

## Inspect workers

**Workers** (the sidebar's link and the panel's tab in developer mode; `#/workers` in the address
in either mode) opens
read-only worker history from the CLI sessions file. Filter
by status, source or exact parent run, load more with the server's cursor, and
open a run to inspect its recorded brief, metadata, work and answer. Every
direct `melo` run is recorded there by default (`--no-save-session` or
`[sessions] record_direct = false` opts out; see
[Worker histories](../reference/cli.md#worker-histories)). The list shows runs
that are still going, and the list is read again every three seconds. An open
unfinished run is read once, then followed on its event stream: new steps, tool
calls and text appear within about a second, without reading the earlier work
again, and the stream stops when the run finishes. A hidden tab keeps
following; a stream that drops opens again after 1, 2, 4, 8, 16 and then 30
seconds, from the last event. **Refresh** reads the
run again at once. A run that holds notes, accepted messages, memory
decisions or recalls shows each as one muted line. The view cannot send messages, approve calls or change grants; chat
close and level controls are disabled while Workers is open. Connection
settings remain available separately. A chat's row in the sidebar returns to it.
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

A task opens in the [panel beside the chat](#the-panel-beside-a-chat), in place of Melo's page. Select its row in
Activity or Workers; or a task's **Open task** or **Review** in the main chat, a task in the **N running** list, or
`#/tasks/<id>` in the address (old links and the macOS app use it). The chat and its header stay as they
were. Above the task's title a link, **Back to Melo**, returns to Melo's page, with focus on the control you came
from (the row, the card in the chat or the **All tasks** link); so does Esc, wherever focus is, unless you are typing in
a box or have a menu or a dialog open (the Resume confirmation takes its own Esc first), and so does closing the
sheet. After **Back to Melo**, the browser's Back button does not bring up a task you opened from the chat (a task opened from the list of all tasks goes back to the list, as any page does). Opening a task moves focus to its title, and a
phone's sheet opens for it, also when the window was made narrower while the task was shown. A waiting task's approval
card is drawn on its page while the panel shows it, and in the task's card in the chat otherwise, never in both.

The page shows the task's state, when it was updated and its folder on one line under the title (`Ready · updated
6:09 PM · in ~/notes`; the full date and time show on hover), the saved recovery card (shown as plain text), the
evidence status when there is some and the actions its state allows, and the same status row as a chat (a task has no row until it has run). A task
that runs or waits to run can be paused (**Pause**) and, in developer mode, given another priority (`P0` to `P4`, each with its meaning;
P5 is Pause), as the terminal's task list offers. A task that is running stops at its next safe point, so after
**Pause** the page says `Pausing…` until it is parked, and then `Paused.`, as the Activity row reads; a paused one
offers **Resume**. In default mode the page leaves out
the scheduler's words and an empty recovery card; developer mode adds them, with the priority line, and lists the
state, mode, time and folder as rows of the full record instead of the one line.

**All tasks**, at the foot of the panel's Activity tab (once a task exists) and Workers tab (developer mode), opens the list of every
task in the same place: newest update first, with state, priority (developer mode only), title and time, the earlier
days too, which Activity does not list. Rows are set apart by a thin line; a task that waits for you has an amber box.
A time reads as the rest of the page writes it (`6:09 PM`, `Yesterday`, a date), with the full stamp on hover. Select one to open it. `#/tasks` in the address opens that list. There is no
Tasks link in the sidebar, and the task page is not a view of its own.

- A task that ended opens with its answer, set as in a chat, above its facts. A task that
  did not complete says why in the same place. Its steps are folded under **Steps**.
- **What melo told the task** is folded on the page when the task began with something from you
  or its chat (see [Tasks start with them](memory.md#notes-melo-keeps-about-you)), each part under
  its own label: what you typed (when its request left it out), the chat's earlier turns, what the
  chat recalled for it, and your notes. They are information, not instructions. A task that got
  none of these shows no fold.

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
- A started task shows its steps below its approval card, under **Live** while it runs
  and under **Steps** otherwise: its steps, tool calls and model text, drawn as the
  Workers view draws a run. A `python_exec` step names the command its code runs
  (`grep -rn DEFAULT_WORKSPACE .`) rather than its first import. The card
  reads the task's session once (`GET /api/sessions/<task id>/turns?events=true`)
  and, while the task runs or waits for approval, follows
  `GET /api/sessions/<task id>/events?after=<last id>`. New events are folded in
  and drawn together at most every 150 ms, without reading the earlier work
  again, and only the Live section is redrawn, so a reason being typed in the
  approval card keeps its focus. A hidden tab keeps following, so a card Auto
  shows can still be answered when you come back; a stream that drops opens
  again after 1, 2, 4, 8, 16 and then 30 seconds, from the last event. When the
  task is done, failed, cancelled,
  expired, interrupted or parked, the stream closes after one last read; a
  resumed task is followed again. A task that never started shows no steps,
  and a finished task shows its work once. The console shows task
  events as it shows a chat's: the same authenticated loopback, no extra
  redaction.
- **Stop** uses the task's own stop, not the chat's. It is available until the
  task is finished and wins over a Resume in flight.
- A started task without a saved card shows "Resume unavailable: recovery
  information is missing." and **Start again as a new task**, which creates a
  separate task from the original objective. The effects of the earlier task are
  unknown and are not undone.

- **Its folder.** A task works in the folder of the chat that started it (the
  default for the main chat, a side chat's own) unless your request names one.
  The list and the Workers tab name it (`In ~/notes`); a row in the panel names it only when it is not the open
  chat's folder.
- **A folder melo may not use.** Outside your home folder, hidden or under
  `~/Library`: the task does not start and never runs somewhere else. Its card
  says why, and **Choose a folder** starts it again in the folder you pick.

![A task that could not use its folder offers Choose a folder](../assets/img/task-folder-refused.webp)

- **Only the commit left.** A task that ran out of steps before its commit says
  `Not committed. Run: <command>` on its card and its page.
- **How it ended.** A finished task's state is how code found it ended: **Ready**
  (green), **Done, not finished** and **Needs you** (amber), **Failed** (red),
  **Stopped** or **Not needed** (grey). **How it ended** says why, and **Tests it
  ran** lists the tests the task's own run started, each passed, failed or result
  not checked; a test its answer only mentions is not listed. **Ready** needs
  proof: when a test's result or the folder could not be checked, the task is not
  ready. Changed code owes a test; when **Finish it** then finds no test runner
  for the folder, the task reads **Ready** with **No tests apply**, and when it
  finds tests it cannot run, **melo found tests here but could not run them**.
  **Finish it** continues, with one tap, a task that still owes its tests or its
  commit, and is never offered again after its own run. A
  folder too large or unreadable to check offers it only while it can still run
  a test or commit; then it says why: **The folder was too large to check**. The panel's Activity and Workers rows say the same, and
  `GET /api/tasks/<id>` carries it as `ending`. A task asked to write or save
  a named file (`write the summary to notes.md`, `保存为 report.csv`, in its
  request or a later message) is **Ready** only when that file is in its
  folder; otherwise it reads **Done, not finished** with **A file it was asked to
  write is missing: notes.md**, and the chat's follow-up can continue it once to
  write the file.

A button is disabled while its request is in flight. A refusal (a stale
confirmation, a finished task, Stop winning the race) shows the server's text.

## Inspect a chat

In developer mode, **Details** opens the session's details in the [panel](#the-panel-beside-a-chat), in place of
its tabs: its tools and the grants that hold in it (a side chat shows the main chat's too, open or not), the working
window and the embedding model, and finished runs with paging when a page supplies
a turn cursor. Granted tools can be
revoked there, and a main chat grant revoked there is revoked everywhere; the tools view may open a paused session.
In default mode (a failure's **Show tools** opens it) the tools view names each tool by what it does and whether it is
allowed (`Read a file · Not allowed`); developer mode adds its name, where it comes from and the model's instructions. **Close details** or Esc returns to the tabs. On a narrow
screen the panel's sheet opens with them.

<a id="language"></a>

## Language

The console speaks English or Simplified Chinese (简体中文).

- **Automatic** follows your browser's languages, or your Mac's in the macOS app: the first that is English or Simplified Chinese, else English. Traditional Chinese reads English.
- To choose one, open **Settings → Appearance → Language**: **Automatic**, **English** or **简体中文**. The page reloads in it. In a terminal, `melo config set console.language zh-Hans` (or `en`, or `auto`) does the same; reload the page after it.
- Every screen follows: the main chat and its cards, the panel's tabs and the tasks it shows, Workers, Settings and the sign-in page.
- The gateway writes its own words for the page in the same language: error messages, the panel's rows, the main chat's cards and the lines it adds after your messages (`Started: …`, `Saved: …`). The terminal and other API clients stay English.
- Times and dates are written in the page's language, not your computer's: `2:06 AM` and `Oct 31` in English, `上午 2:06`, `10月31日` and `星期五` in Chinese. In English the clock follows your computer, so a 24-hour computer keeps its 24-hour clock.
- A failed run's details and the line **Copy details** copies read in the language; the status, the cause, the error text and the run id are copied as they are. The status row's words (the context, and in developer mode the cache, step and queue) read in it too, while the terminal prints the same figures in English.
- In Chinese, the card that asks about a high-impact step (Ask) or reports a blocked one (Auto) gives its question and its reasons in Chinese, by the kind of step; the step itself stays on the card as written, and so does anything your model wrote.
- A schedule's words (`every weekday at 8:30`) and what each row says it is doing (`Reading notes.md`, which the macOS pill shows too) follow the language, and the clock follows the page's format (`8:30 AM`, `上午 8:30`). The check-in's note is written in the language too: the setting, else the language the page asked for last.
- In the macOS app the menus, the pill and Melo's menu change the moment the page reloads in the new language; nothing needs a restart.
- Commands, paths, model names and the levels P0-P5 are never translated; the terms follow the [Chinese interface glossary](../zh-CN/STYLE.md#interface-glossary). Melo answers in the language you write in, whatever the setting.

## Settings

**Settings** in the sidebar (or **Settings…** in the macOS app's menu) has one section per
subject. Each change saves at once and says so at the foot (`Saved: …`) with **Undo**; the
save carries the revision it read, so a change made elsewhere meanwhile asks you to reload.

- **Model server**: the chat model and the embedding model, each with its state
  (`Connected · <model> · checked 12 s ago`, `Not reachable`, `Off`), the server's
  address, **Test connection** (it never shows a key) and **Change…**, the form that picks
  the server, its address and the model. The form asks the server which models it offers
  by itself, when Settings opens, when the form opens and when you enter the address, and
  the chat model is a list of its chat models (embedding models are kept apart; **Other**
  lets you type a name); **Refresh** asks again. A saved model the server no longer has is
  replaced by the one chat model it offers, with a line that says so, and **Save connection**
  keeps it; when it offers several, the line asks you to pick one from the list, and a chat
  that hit the missing model names it and points to the list. The form is the
  one change saved with a button, **Save connection**; **Cancel** puts the saved one back.
  Under **Hosted services** the list offers OpenAI, Google Gemini, Anthropic, NVIDIA,
  DeepSeek and Other (OpenAI format): the form then has an **API key** field with
  **Save key** (kept in your Mac's keychain, shown only as **Saved in the keychain**, with
  **Replace** and **Remove**) and, for a service with a switch for it, **Let the model think
  before it answers**. Once a key is saved the service lists its models by itself, when
  Settings opens, when you choose the service and after you save the key (one request,
  no chat); the first model it suggests is offered first, and a saved model it no longer
  offers is said at once, never changed for you. Each service you used keeps its last
  model, and a key goes only to its own service. **Test connection** tries one short chat
  with a tool. See [Hosted models](hosted-models.md).
- **Memory**: **Remember across chats**, how Melo finds notes (**By words and meaning** or
  **By words only**; developer mode adds the other two ways, with the key
  `memory.retriever`), and **Open Memory** for the notes themselves.
- **Chats and folders**: the default folder of a new chat (**Change…** opens the app's
  folder panel, or a path field in a browser), how new chats start (**New chats start in**),
  **Read web pages** (on by default; it lets chats and tasks read public pages with
  [`web_fetch`](../reference/tools.md#reading-a-public-web-page)), **Web search** (off by
  default; it lets chats and tasks look things up with
  [`web_search`](../reference/tools.md#searching-the-web), and the words searched for leave
  this computer), **Reflect when stuck after many steps** (on by default; a task still not
  done after 50 steps pauses once to reflect before it goes on, which costs one more model
  request; off, it keeps going to 100 steps; the key is `agent.reflection`),
  **Other folders melo may read** (the folders you chose **Always allow this folder** for, each with **Remove**; see
  [Folders outside the workspace](tools-mcp.md#folders-outside-the-workspace)) and the recent
  folders, each with **Make default**.
- **Permissions**: what you allowed, in two groups, **Everywhere** and **Per chat** (see [Permissions](#permissions)).
- **Scheduled**: the check-in switch with its next time, and **Open Scheduled** for the chores.
- **Appearance**: the theme (System, Light, Dark, kept in this browser), the
  [language](#language) (Automatic, English, 简体中文), **Melo on the desktop** in the macOS
  app (the same choice as its menu), and **Developer mode**.
- **Developer** (developer mode only): every setting `melo -h all` lists, grouped by section,
  with its value, its `config.toml` key in small type, the old option that sets it and, for a
  comparison, the issue that decides it; then the options that set one run. They are
  read-only here: `melo config set KEY VALUE` changes one. The section also names the
  settings file and its revision.

## Developer mode

**Settings → Appearance → Developer mode** is saved as `[ui] mode` in `config.toml`, the
setting `melo --help` reads, with the same revision check as the other settings; a mode
written in the file by hand wins the next time the page reads it. It changes what is shown,
never what runs. When it is on:

- Settings lists the **Developer** section (every setting with its key and value);
- **Details** gains an **Events** tab with the stored raw events of the latest
  turn, and its **Memory** tab adds the agent state and the long-term backend;
- the **Runs** tab adds each run's recorded counters: requests, tool calls,
  token usage and cache numbers;
- the header shows the chat's level chip, **Details**, the GPU-lock line, the **N running** chip and a chip with
  the page's state (`Ready`, `Working`, `Approval needed`), and the sidebar always lists
  **Workers** and the panel's **Workers** tab;
- the status row adds the cache, step, phase and queue segments, and a task's page names its
  priority;
- a running turn shows the tool's name and arguments on each tool card, its step, attempt, phase and test count, the number of tool calls on the work line and the steps and tokens under an answer, and a model server error keeps its own text on the card (default mode words each of these for a person and keeps the raw text under **Details**);
- the sidebar shows the **LOCAL** badge, a running chat reads `Running on your backend`, and the
  connection form keeps its technical words (**Chat backend**, **Chat API URL**).

When it is off these are hidden, not removed: turn it on again to see them. While a chat
waits on a foreign GPU lock, default mode shows a banner in place of the GPU line:
"Waiting for this Mac's GPU, your chat starts when it is free." Chats, approvals, the
Workers view, a task in the panel and what the settings save work the same in both modes.

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

In developer mode a chat's header has a level chip (P1 to P4, P2 by default; its menu says what each means) that sets how the
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
box; Escape anywhere else only closes what is open, in Workers as in a chat; it leaves a task shown in the panel. A model request in flight is abandoned, although the
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
notes.txt?", "Let melo use <tool>?") and shows exactly what would run. A command
or a Python script starts with one plain sentence about what the step will do,
which melo builds from the call and which names every effect it finds, in order
("Operate Safari, then delete files"; a command too long to show ends "and 2
more"). The model's own words (a `why` argument, or a comment at the top of the
code) always carry a label, "melo says: …". They show only in the language of your own recent messages: words the model wrote in another script (English under a Chinese chat, say) are left out and the card keeps melo's own sentence. When the rules find nothing to ask
about, the words are the sentence and the code sits under it in a fold, **Show
the code**. When the rules find something, the sentence stays melo's own, the
words go in a line below it, and the code is open, so the words can never make a
step look safer; the question and the reasons come from melo's rules either way. A file
write over an existing text file shows what it replaces: the line under the
question counts the lines ("5 lines removed, 1 added"), and the box lists the
changed lines, each with its number, a − or + sign and two lines of context on
each side. Removed lines are tinted red and added ones green, and the signs say
the same without the colours. A long change shows its first 20 rows; **Show all
N lines** shows the rest. A secret in a row reads `[redacted]`. When melo cannot
compare (the file is not text, cannot be read, is over 200 KB or holds a private
key, or the change runs past 1,000 rows) the card says so in one line, asks
"Replace logo.bin (480 bytes)?" with the file's size, and shows the new text
alone. The comparison is for you to read: the answers below work as they do for
any other file write. **Allow once**, the filled button, runs this call only.
**Allow for this
chat** (**Allow for this task** on a task's page) stops asking about the routine calls
of that tool in the chat (see [Permissions](#permissions) for how long and where it holds); the line under the buttons says so, and
for a command or Python code it adds that installs, downloads and changes
outside the workspace still ask. **Deny**
opens an optional box, "Tell melo what to do instead", and sends the denial
with that text; **Back** or Escape returns to the answers. The controls wait
500 ms before they can be used; arrival does not move focus and no single-key
shortcut answers a request. When the card is outside the view, a bar above the
message box reads "melo is waiting for your answer", and **Review** opens its
group and moves to **Allow once** (to **Deny** on a high-impact card). If
approval details cannot be loaded, or arguments were truncated or converted,
both allow buttons are disabled; you can reload the details, deny, or stop the
run.

A call that installs software, downloads files, changes files outside the
workspace or does anything else [high-impact](tools-mcp.md#high-impact-calls)
asks every time, with a different card. Its question names what the call does
("Install software?"), the line under it reads `python_exec · asks every
time`, and **Why melo asks** lists up to three reasons ("Installs playwright
with pip into .signin_venv (downloads about 42 MB)."). **You asked:** repeats
the message that started the turn, so you can compare it with what melo is
about to do. When the model's words and the code disagree about deleting
files or sending data out, the card shows a red warning that quotes the words
("melo described this as “Close the tabs”, but the code deletes files."). When the
words do say the right thing but name a file, folder or website and the code deletes or sends
to a place they do not name as well, it says where ("…but the code deletes build/."). A script
that operates one app by its plain name (AppleScript telling Safari to close
tabs, `open -a Notes`) asks **Control an app?** and names the app; a script that
runs a shell, presses keys, drives System Events, names a terminal or a path,
or runs a Shortcut stays **Change system settings?**, and so do a script that reaches macOS
without naming an app, a link that is not `http`, `https` or `file` (`open shortcuts://…`),
erasing a disk and stopping other programs. Discarding uncommitted work (`git checkout -- .`,
`git restore`, `git clean`) asks **Delete files that cannot be restored?**. The card has **Allow once** and **Deny** only, and **Deny** is the
filled button: there is no chat-wide allow. A call melo cannot read, or that it asks about for another
reason, shows the plain card.

When melo wants a file outside the chat's folder, the card asks about that folder
("Let melo read in ~/Documents/letters?") and shows its real path in a box, with what melo
wants to do there ("melo wants to read cover-letter.md in this folder."). A read offers **Allow once**,
**Allow this folder for this chat**, **Always allow this folder** and **Deny**: "this chat"
is a grant (see [Permissions](#permissions)), and "always" keeps the folder in Settings, under **Chats and folders**,
where you can remove it. A write there is a high-impact card with **Allow once** and **Deny**
only, every time. The folders a chat may read are listed in **Details › Tools**, each with
**Revoke**. Keys, melo's own data, system folders and your whole home folder are never
offered (see [Folders outside the workspace](tools-mcp.md#folders-outside-the-workspace)).

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

When the model server answers with an error, the card says `Could not get an answer from the model server` and one sentence about it with the next step (`The model server ran into a problem and could not answer. Try again; if it keeps happening, restart the server.`; there is one for a key it did not accept, a model it does not have, a busy server and a request it turned down). The server's own text, such as `HTTP 500: …`, stays under **Details**; developer mode shows it on the card too.

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

A reply that is only the model's unfinished thinking never shows as the
answer: it is a folded thinking item in the work box, with the line `The
model's reply ended inside its thinking; asking it for its answer.`, and the
model is asked once more. When the service reported the model's thinking, melo then
asks one last time with thinking off (where the service has that switch) and, if the
reply is still empty, shows the last lines of the thinking as the answer, ending with
`(from melo's thinking)`. Each of the two steps adds a line to the work box (`The model
ended its turn with only thinking; asking again with thinking off.` and `The model gave no
answer after thinking; the answer is the end of its thinking.`). When there is no thinking to use and the second reply has no
answer either, the card's
title is `melo did not finish an answer`, the hint `The model's reply held only
its thinking. Try again, or rephrase the request.`, and the filled button
**Try again**.

## Auto mode

In **Auto**, melo works on its own: routine steps run; before a step you did
not ask for, or one only you can allow, melo asks you on the usual card; the
riskiest steps are blocked and explained. **Ask**, the default, asks before
commands and file changes.

- **melo acts in your name; the cards ask.** What you ask for is done as you: applying with your resume, signing in, filling in and submitting a form, sending a message as you, booking. The model does not refuse it because it speaks in your name or cannot be undone, and it does not ask you to type a permission; which steps you confirm is decided by the cards in Ask and by Auto's review in Auto. Only what would harm other people is declined.
- **Switch.** The chip beside the model under the message box reads **Ask**
  or **Auto**; its menu, "How melo works in this chat", switches the chat
  from melo's next step (a card already waiting stays), and it reaches the
  tasks the chat started that are still running: Auto to Ask at their next
  decision, Ask to Auto at the one after the card that waits. A chore keeps
  the mode it was saved with. A new chat starts in
  the mode of Settings → Chats and folders → **New chats start in**
  (`[approvals] mode`); the main chat follows that setting until you pick
  its mode.
- **What runs.** Steps inside the chat's folder that install, download, send
  or delete nothing. A step melo cannot judge goes to your model first and
  runs only when it is routine and part of what you asked; a tool you allowed
  for the chat runs its routine calls without that review, as in Ask. A
  turn's reviews count only the steps that were stopped or failed, so a long
  task that goes well keeps being reviewed; a review that let a high-impact
  step run on your words still counts. A write outside
  the folder fails in Auto even when no rule caught it: its output says
  `writing outside the workspace is not allowed in auto mode`. A step in
  Auto can neither read melo's sign-in token nor reach melo's own API.
- **A reviewer, not a wall.** Auto stops risky steps and steps that stray
  from what you asked. Its sandbox keeps a step's files in the folder and
  away from melo's sign-in, but a step that hands work to another app or
  program (opening an app or a script, scripting another app, code for
  another language, a package fetched and run, a background job) is judged
  by the review, not walled off: such steps are blocked or checked by your
  model, never run as routine. The sandbox also keeps a step from talking to
  a container engine.
- **The goal decides.** On the site or app the chat is about, the steps your
  task plainly needs run without a card, named or not: opening and reading
  it, signing in the way you described, picking options, filling in and
  sending the task's own form, reading the result. "Check in on the forum for
  me" is enough for the sign-in clicks, the mood, the line and the send.
  melo judges each step, by what it clicks or types into, against all your
  messages in the chat, and against its recalled notes when you point to
  them ("do it the way I told you").
- **Your words only add.** A step your messages ask for, or whose target they
  name (a package, a file, a site, a box or a button on a page), also runs
  without asking; words that do not name a step never stop one the task
  needs. Only what you typed counts as your words, never melo's answers or a
  page.
- **What asks.** A step off the goal, and an install, download, change
  outside the folder, delete, data sent out or a message to other people
  beyond the task's own form, a network service, code melo cannot check, or
  a click inside a box from another site that melo cannot tell, shows the
  approval card in the chat and waits, as in Ask: **Allow once**, **Deny**
  with a note for melo, and, for a site you did not ask for, **Allow for this
  chat**. Paying and changing an account's email, password or sign-in always
  ask, even when you asked for them. melo reads what you chose and your note;
  after **Deny** it does not ask about that step again until your next
  message. melo asks you itself: it never asks you to type a permission.
- **Only while you watch.** The card waits only while the chat is open in the
  console, the app or the terminal. With nobody watching, the step is blocked,
  as before, unless it is a human check whose **Always in this chat** you kept. A card nobody answers within 1 minute, or that you close the chat
  on, counts as **Deny**: the step reads "No answer in time", with one line
  under it ("Not run: no answer within 1 minute").
- **Not allowed is not failed.** A step you denied, or that nobody answered,
  shows in amber with one plain line ("Not run: you declined it" and
  your note), never as a failure and never with the note melo reads. Ask's
  **Deny** shows the same way.
- **Allowed in this chat.** **Allow for this chat** on a site's card lets the
  same kind of step on that site (opening and reading it, or clicking and
  choosing on it) run in the chat. A site Auto let the chat open
  stays allowed for reading its pages in that chat too, also
  after you switch to Ask, and holds in that chat only, even the main chat; a click on it still asks in Ask. The chip's menu lists the sites the chat allowed, each with
  **Remove**. Typing and sending a form are never kept for a chat: Ask asks every time, and Auto checks each one against your request, runs what the task plainly needs and stops the rest (it asks you, or blocks a secret). A
  high-impact step never rides an allowance, and a new chat starts with only what you allowed in the main chat.
- **What is blocked.** Typing or reading a secret (a password, a one-time
  code, a key; clicking a sign-in button is none), system settings, scripts
  that operate another app, something that runs later and code from the
  internet are blocked without a card; a change to localmelo itself never runs; in a
  folder that holds the copy of localmelo that is running, a write under
  that copy is refused. The step's row reads `blocked: uses passwords or
  keys`, a card under the work says what and why and shows the step, and
  melo's answer gives the other ways.
- **Allow a blocked step.** **Allow once** on its card runs that exact step
  once, within 30 minutes, and sends "Go ahead with the step melo blocked:
  use your passwords or keys." to melo. It is melo's sentence, not your
  words: the chat shows it as a folded line (`Not typed by you`), and
  nothing, in a chat or in a task, stores, remembers or counts it as something you typed, and it
  never answers a question a task waits on: that still waits for you. While melo works the
  button waits; afterwards the card reads `Allowed once · ran at 14:02`. An
  allowance that ran out unused says `Allowance expired` beside **Allow
  once** again: the step still asks.
- **Tasks.** A task you started follows its chat's mode at each step: a switch
  reaches its next decision, and a card that waits stays for you. It asks in
  its chat, as in Ask, and waits for your answer. A task another program or
  melo made keeps the mode it was made with. A scheduled chore takes its chat's mode each time it runs, and in Auto,
  with nobody watching, it blocks the step instead: its page shows **Mode**
  and lists **Blocked steps**, each opening the same card; the task goes on
  without them, and **Allow once** there sends the message to the task
  while its job is unfinished. A task that ended on its blocked step says
  so once: its state reads **Needs you**, its report shows without the
  `Ending:` line (developer mode shows the report as written), and the
  step's box shows the step. Answered there or on the chat's card, both
  say the same line and neither offers a button; a run whose work went on
  in a newer task reads **Continued in a newer task** as its state, on its
  page, in the card's heading, on the Scheduled tab, in the panel and in
  `melo task`. Once it finished, the step is new work you
  ask the chat for. Typing on a page and sending a form are blocked steps
  like any other: the card says the site and the box and shows the whole
  text, or where the form goes and what it holds, and **Allow once** runs
  that exact text there once; the task is told exactly which step it may
  repeat. Another text, another box, another site or the form's send is a
  step of its own and is reviewed again. A text filed under passwords or
  keys, a step on a site refused earlier in the task, and a click, a choice
  or a page opened have no **Allow once**. Two boxes described by the same
  words on one page count as the same box. A step that types and sends in
  one go (typing, then Enter) asks two questions; **Allow once** covers one
  of them, so such a step may need a kept answer to finish.
- **Keep it for a scheduled task.** On a typing or sending step of a
  chore's run that the review found fits the task and stopped only because
  it sends data from your Mac, the card also offers **Always allow for this
  scheduled task on <site>**. It runs the step once, as **Allow once**
  does, and from then on the rest of that run and the chore's later runs
  type and send on that exact site (scheme, host and port) without asking:
  a check-in's send right after its note is not asked again (a run that
  had ended goes on in a fresh task, which holds the answer too). The click
  then says `Allowed for this scheduled task on example.com. This run goes
  on now.`, or `… Its later runs use it.` when the step can no longer go
  ahead (it ran already, or its job has finished); opened later, the step
  says `Allowed for this scheduled task on example.com.` Each step is
  still reviewed; what still stops, as before:
  passwords and keys (a secret in the text too), a text in which melo finds
  your phone number, address or name, a step the review finds does not fit
  the task (or is unsure of), a form sent to another site or where the page
  decides, a site refused in the run and every other site. melo finds a
  phone number or a street address only by its written shape, and your name
  only as your Mac account spells it, in the text the step shows: other
  details of yours, or these written another way, it does not see. A call that types and then
  sends runs when both questions pass. The run's page lists what it was
  allowed this way (`Allowed to type on example.com without asking: you
  allow this for this scheduled task`), and so does the terminal. The
  Scheduled tab lists it under the chore (`May type and send on example.com
  without asking · since Nov 10`) with **Remove**; changing the chore's
  request drops it (the tab says so) and deleting the chore deletes it. It
  has no end date. A chat and a one-time task never offer it.
- An **Allow once** on a blocked step lives in memory: restarting
  localmelo forgets it, and an allowance the retried turn did not use goes
  when that turn ends. Sites allowed for a chat are grants, which are kept
  (see [Permissions](#permissions)).

## Permissions

Everything you allow "for this chat" is a grant: a tool, a site, a folder outside the workspace, another app's window, an app a page may open or a human check. **Settings › Permissions** lists them, and they survive a restart of the gateway.

- **The rule:** what you allow in the main chat counts everywhere; in a side chat, only there. Everywhere means every side chat, the tasks you start and the terminal.
- **The card says so:** in the main chat the button that keeps a grant reads **Allow in every chat** (**Allow this folder in every chat**, **Allow Chat everywhere** for another app's window), with a line saying it counts in every chat, the tasks you start and the terminal until you revoke it here. A side chat's card says **this chat**, a task's **this task**.
- **In a side chat:** the chat's menu marks a grant that came from the main chat, **From the main chat**; removing it takes it away everywhere, and the chat says so. Clearing a side chat's grants (`/approvals forget`) takes back only its own, and says how many of the main chat's still hold there.
- **Only there:** a side chat's grant holds in that chat. A task's **Allow for this task** holds in that task. A site Auto let the chat read holds in that chat only, even in the main chat.
- **The page:** the rule above and the steps that always ask, in two sentences; how typing and sending a form are
  handled is folded under **Typing and sending a form**. Then two groups, **Everywhere** and **Per chat** (a small heading per chat). A row says what is allowed in plain words, the chat, who decided (You or Auto), when, and **Revoke**, which works at once, in chats that are running too. With none, the page says so. **Always allow this folder** is not a grant: it stays in **Chats and folders**.
- **`/grants`** lists the same rows with numbers (everywhere first, newest first, then each chat, the one with the newest grant first); `/grants revoke N` takes the N-th back.
- **Never kept, never listed:** **Allow once**, a click on a password, code or card field, a consent control or a box from another site, a press in another app's window, paying, changing an account's settings, and every high-impact step (in Auto, opening an app a page asked for too) ask every time. Typing and sending a form are never kept for a chat: Ask asks every time, and Auto checks each one against your request, runs what the task plainly needs and stops the rest (it asks you, or blocks a secret). A scheduled task's **Always allow for this scheduled task** is not a grant either: it belongs to that chore and is listed on the Scheduled tab.

Deleting a chat deletes its grants. Moving a side chat to another folder takes back its own (the main chat's still hold).

## Persistence and limits

The page's chats are sessions in the gateway's own session file,
`~/.cache/localmelo/gateway/sessions.db`; the terminal keeps its own in
`~/.cache/localmelo/cli/sessions.db`. They last until you close them, and so do
the grants you gave them: a
gateway restart keeps them, and a chat reopens with its transcript read from
the file, no Agent built yet (`not_loaded`); its Agent is built on the
conversation the file holds when the next turn or retry starts.

A chat runs in gateway mode: its Agent gets the gateway's default run options,
the built-in tools `file_read`, `file_write`, `list_files` and `python_exec`, and no
MCP tools, except that it keeps its answers. It can list folders, read and write
files and run Python code; `shell_exec` is not offered, so there is no direct shell
command (Python code can still start a process). Its model is the gateway's
own, from the config file, and the page reads that file's `[scheduler]` table
too — `policy`, the lane sizes, the call log and the lock and battery keys.

In `lock_gate = "record"` (the default) a foreign lock is shown and logged but
nothing waits. In `wait`, every call a chat makes waits while a foreign lock is
present, embeddings included. A chat request waits up to 300 seconds, and a
turn is stopped after 600 seconds of active time: a call the scheduler holds,
including a foreign-lock hold, does not count against that bound.

The workspace has these limits:

- It is for loopback use only. Sign in as described above and open it on this
  computer as `http://127.0.0.1:8401/` or
  `http://localhost:8401/`. Remote access, from another device or through a
  proxy, is not supported; see
  [Keep the gateway private](gateway.md#keep-the-gateway-private).
