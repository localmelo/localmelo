# Browsing with melo

This guide shows how to let melo read web pages in Safari, what it can and cannot do there, and who decides which sites it may open. It is for people who chat with melo in the terminal or the web console on a Mac. Read the safety section before you turn it on; the settings are in the [configuration reference](../reference/configuration.md).

Browsing is **off by default** and **read-only in this first part**. melo opens pages in Safari's *automation windows*, follows links and reads what it finds. It does not click buttons, fill in forms, sign in, post or buy. Those come later (see [What is coming](#what-is-coming)).

## What you need

- A Mac with Safari 27 or later. melo uses the automation server that Safari ships, `/usr/bin/safaridriver --mcp`; nothing is installed.
- Two switches in Safari, turned on once (see [Set up Safari](#set-up-safari)).
- The terminal chat (`melo`) or the web console. Direct runs (`melo "task"`), worker runs and saved tasks never get the browser tools.

## Turn it on

In the terminal chat:

```text
/browser
```

shows whether browsing is on and, when it is, checks that Safari answers. It then offers a list: **Turn browsing on** or **Leave it off**. `/browser on` and `/browser off` do the same without the list. The choice is saved as `[browser] driver` in `~/.cache/localmelo/config.toml` when you have a saved configuration, and the chat continues in a new Agent, so nothing you said is lost.

To turn it on for one run only, without saving:

```bash
melo -X browser.driver=safari-automation
```

In the config file:

```toml
[browser]
driver = "safari-automation"
```

`"off"` is the default.

The web console reads the same setting when it opens a chat; change it with `melo config set browser.driver safari-automation` and open a new chat.

## Set up Safari

Safari has to allow outside programs to open automation windows. In Safari:

1. Open **Settings**, then the **Developer** tab. If there is no Developer tab, open the **Advanced** tab and tick **Show features for web developers** first.
2. Tick **Allow remote automation** (its full label may read "Allow remote automation and external agents").

When it is off, melo says so in its answer to the first browsing call and in `/browser`:

```text
Safari is not set up for melo yet. In Safari, open Settings > Developer and turn on "Allow remote automation" ...
```

Turn the switch on and ask again; melo tries to connect again on every call, so you do not restart it. If the check still fails after both switches are on, Apple's one-time WebDriver setup is `safaridriver --enable` in Terminal (it asks for your password); melo never runs it for you.

melo starts Safari's automation server only on the first browsing call (or `/browser`). A chat that never browses opens nothing.

## What the automation windows are

Safari opens them apart from your own windows, with an orange address field. They start from a clean slate, like a private window: **none of your sign-ins, cookies or AutoFill are available**, nothing is kept afterwards, and only one automation session runs at a time. A pane over the window catches your mouse and keyboard; breaking it offers to stop the session. So melo can read pages that need no sign-in. Pages that only show for a signed-in person, and sites that challenge automated browsers, may not work in this part.

## What melo can do

| Tool | What it does |
|---|---|
| `browser_open` | Open an `http` or `https` address (in a new tab, or the current one) and read it. |
| `browser_read` | Read the current page again, or the next part of a long page (`part`). |
| `browser_find` | Search the elements of the last reading by words in their label or address. |
| `browser_follow` | Open the address of a link from the last reading. It opens the address; it does not click. |
| `browser_back` | Go back one page in the tab. |
| `browser_tabs` | List melo's tabs, switch to one, or close one. |
| `browser_wait` | Wait (30 seconds at most) for the page to load, or for some text to appear or go. |
| `browser_screenshot` | Take a picture of the page. You see it in the chat; the model gets only a note that it was taken. |

A page reaches the model as one bounded block:

```text
<<page id=s1 nonce=ab3218de tab=w2 origin=https://example.com title="Example Home">>
Welcome to the example.
Second line, with 中文。
elements:
[e1 link "News" -> /news]
[e2 textbox "Search" value="cats"]
[e3 password field: melo never types here]
[e4 button "Go"]
<<end page s1 nonce=ab3218de>>
```

- A block holds about 6,000 characters of text; a Chinese character counts as three Latin ones, so a Chinese page is cut at about 2,000 characters. A longer page is cut at a paragraph or sentence, with `part=1/3` in the first line and a line that says which `browser_read part=2` continues.
- At most 60 elements show. `browser_find` searches all of them.
- Element refs (`e1`) are numbered across the chat and belong to the newest reading of their tab. A ref from an older page is refused, not guessed.
- The values of password, one-time-code and card fields are never read, and a page with payment fields is not read at all.

## Who decides which sites melo opens

- **Each site asks once per chat.** Before melo loads a page, you see "Open a website?" with the site's name and the answers *Yes*, *Yes, and allow this site for the rest of this chat*, and *No*. In the web console it is the usual approval card. "For the rest of this chat" is the only lasting answer; there is no "always" in this part. `/new`, and `/approvals forget`, take the grants back.
- **A redirect asks again.** If a link or a redirect lands on another site, melo asks about that site before it reads the page. If you say no, the window goes back to a blank page and the model reads that you refused.
- **Only `http` and `https`.** `file:`, `about:`, `javascript:`, `data:` and the browser's own pages are never opened.
- **`auto` mode still asks about sites.** Auto mode runs commands and file changes without asking; it does not skip this question. `--yolo` skips every question, this one included.
- **Page text is data.** A page can say anything. melo's system prompt tells the model that text between the page markers is content of a web page, never instructions, and the closing marker carries a random number that no page can know, so a page cannot end its own block. This lowers the risk of a page steering the model; it does not remove it. Do not allow sites you do not trust, and do not turn browsing on in a chat that holds secrets you would not paste into a web page.

## What is coming

This is part 1 of browsing. Planned next, in this order: acting in your own signed-in browser (a Chrome extension; clicks and typing, with a confirmation that `auto` and `--yolo` cannot skip for sending, posting, buying and signing in), then saved chores that run on a schedule (for example a daily check-in) with steps you approved once. None of that exists yet.

## Troubleshooting

| You see | What to do |
|---|---|
| `Safari is not set up for melo yet` | Turn on the two switches in [Set up Safari](#set-up-safari). |
| `Safari's automation helper did not start` | Check that Safari 27 or later is installed and the switches are on; `/browser` runs the check again. |
| `This Safari's automation server lacks the tools melo needs` | Update macOS and Safari. |
| A page reads as empty | It may need a sign-in or be built by scripts after loading; try `browser_wait`, or open it yourself. |
| melo says it did not open a site | You answered no to the question about it; ask again and allow the site. |
