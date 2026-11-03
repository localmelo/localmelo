# Searching the web with melo

Let every chat and every task look something up on the web, with one switch. Read what leaves your computer before you turn it on; the settings are in the [configuration reference](../reference/configuration.md).

- Web search is **off by default**.
- With it on, chats and tasks get a `web_search` tool: melo types words into a public search engine and reads back a list of results, each with a title, an address and a short snippet.
- It searches when it needs to check something: a word it does not know, a step that failed twice, or a fact that changes (a date, a price, a version, the news).
- It reads a result with `web_fetch`, as it reads any page, so a site you did not name still asks first.

## Turn it on

In the web console or the app, open **Settings → Chats and folders → Web search**. The line under the switch says what it does: the words melo searches for leave this computer.

In the terminal:

```bash
melo config set web.search true
```

For one run only, without saving:

```bash
melo -X web.search=true
```

- The setting applies to the terminal chat, every chat of the console and the app, and every task. A chat reads it when it opens, so open a new chat after changing it.
- Direct runs (a task given on the command line) never get it.
- Turn it off the same way: the switch, `melo config set web.search false`, or remove the `[web]` table.

## What leaves your computer

- Only the words of the search, at most 300 characters. A plain request carries no cookies and no sign-in, and melo's own name as its User-Agent.
- The engine sees the words and the network address of this Mac. Nothing else from your chat is sent.
- With the switch on, a search counts as asked for, so an ordinary one runs without a question.
- A search whose words look like a secret (a key, a token, a code, a card or ID number) or like pasted text (long, or on several lines, as text copied from a private note or file is) asks first, on the same card as other approvals. Your answer covers that search only, never a later one, even if you chose "allow for this chat".
- A task asks the chat that owns it. Where nobody can be asked, a task does not send such a search.

## Where the results come from

melo tries the ways below in order and stops at the first one that gives results that fit the words. The answer always names the way that answered, and says why the ones before it did not.

| Order | Way | When it is used |
|---|---|---|
| 1 | Bing's result feed, a plain request | Always. Fast, about a fifth of a second. |
| 2 | DuckDuckGo in Safari's automation window | Only when [browsing](browsing.md) is on. The best answers; a plain request is turned away. |
| 3 | Baidu, a plain request | Always. Works for Chinese and English words, about two seconds. |

- A way that asks for a check (a captcha, a wait page, an error) is skipped, and the next one is tried.
- A feed can answer an unknown client with results that read well and are about something else. melo checks that at least two of the first five results share words with the search, and does not take a way that fails the check for an answer. If no way passes, it gives the nearest results and says they may be about something else.
- It asks one engine at most once per search, and at least two seconds apart.
- Safari's window is used only with browsing on. Turn it on with `/browser on` (see [Browsing with melo](browsing.md)); without it, the second way is left out and nothing is said about it.

## How a search shows

In the chat and on the task page a search reads as one line, *Searching the web for “asyncio TaskGroup”* while it runs and *Searched the web for “asyncio TaskGroup”* afterwards, with the words (a secret in them hidden) and, once done, the engine and the number of results. The model reads:

```text
Searched the web for “python asyncio TaskGroup” with Bing: 3 results.
<<page id=s1 nonce=ab3218de type=search way=bing-rss results=3>>
1. asyncio — Asynchronous I/O
   https://docs.python.org/3/library/asyncio.html
   asyncio is a library to write concurrent code using the async/await syntax.
2. …
<<end page s1 nonce=ab3218de>>
[Bing asked for a check (HTTP 429); Baidu answered.]
```

- The results sit between the same markers as a page read with `web_fetch`. The closing marker carries a random number that no result can know, so a result cannot end its own block, and the model is told that text between the markers is data, never orders.
- The line before the block and the note after it are melo's own sentences. Nothing an engine writes outside its results reaches the model.
- A snippet is cut at 240 characters, an address at 300.

## Developer settings

Two more keys, under `[web]`, show in developer mode and in `melo config keys web`:

| Key | Default | Meaning |
|---|---|---|
| `search_order` | `[]` (the order above) | The ways to try, in order: `bing-rss`, `browser`, `baidu`. Leave a name out to never use it. |
| `search_count` | `6` | How many results one search shows, from 1 to 8. A call may ask for fewer. |

```toml
[web]
search = true
search_order = ["bing-rss", "browser", "baidu"]
search_count = 6
```

## If it does not work

- **"No search way gave results."** Every way was turned away or could not be reached. The message names each one. Try again in a minute, or turn browsing on so the Safari way can answer.
- **Results about something else.** The note under the block says so. Ask with other words, or open a result.
- **The tool is missing.** Check that `melo config get web.search` says `true`, and that the chat was opened after you changed it.

The exact rules are in [Tools](../reference/tools.md#searching-the-web).
