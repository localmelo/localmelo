# Browsing with melo

Let melo read web pages in Safari, and click, type and choose on them, on a Mac, from the terminal or the web console. Read the safety section before you turn it on; the settings are in the [configuration reference](../reference/configuration.md).

![A terminal chat in which melo opens a web page and says what it contains.](../assets/img/browser.webp)

- Browsing is **off by default**.
- melo opens pages in Safari's *automation windows*, follows links and reads what it finds. It works in one tab, like a person would.
- It can click a button or link, type into a text box and choose from a list, all **inside melo's own browser window**, with Safari's own clicks and keystrokes, and it says what changed after each one. It never types a password, a card number or a one-time code, never reads your cookies, and asks every time before it types a text and before it sends a form. Once a text is in a page, the page itself may send it: melo cannot stop every such send (see [Clicking, typing and choosing](#clicking-typing-and-choosing)).
- It does not sign in for you, post or buy; moving the mouse on your desktop is a different feature and is not part of this (see [What is coming](#what-is-coming)).

## Read a page without the browser

Every chat can read a public page without Safari, with the `web_fetch` tool. It is **on by default** and needs no setup.

- Ask about a website or paste its address: melo reads the page as text (title, headings, main text and links). Plain text and JSON come back as they are.
- It only reads: one request, no cookies, no sign-ins, no forms. PDFs and other files are refused for now.
- It never reaches this Mac or your local network. `localhost`, private and link-local addresses, `.local` names, this Mac's own addresses (also a public IPv6 one) and the network around them, and a redirect or DNS answer that points to one of them are refused.
- A site your own recent messages name reads at once: that exact site, with or without `www.`. A site under it (`docs.example.com` when you named `example.com`) and any other site ask once per chat ("Let melo read example.com?", with the full address), and one "allow for this chat" covers the browser too. A file name such as `README.md` does not name a site.
- A page that needs a sign-in, blocks automated readers or is built by scripts says so. With browsing on, melo can try it in Safari instead.
- Tasks have it too. Turn it off in Settings → Chats and folders → **Read web pages**, or with `melo config set tools.web_fetch false`. The limits and the exact rules are in [Tools](../reference/tools.md#reading-a-public-web-page).

The rest of this page is about browsing in Safari, which also follows links, clicks, types and runs a page's scripts.

## What you need

- A Mac with Safari 27 or later. melo uses the automation server that Safari ships, `/usr/bin/safaridriver --mcp`; nothing is installed.
- Two switches in Safari, turned on once (see [Set up Safari](#set-up-safari)).
- Optional, for [windows from other apps](#windows-from-other-apps): the desktop control helper, [Cua Driver](https://cua.ai/docs/cua-driver), installed as `/Applications/CuaDriver.app` (melo also finds `cua-driver` on your `PATH`). Give it Accessibility in System Settings → Privacy & Security; Screen Recording adds a picture of the window. melo itself needs neither.
- The terminal chat (`melo`) or the web console. Direct runs (`melo "task"`), worker runs and saved tasks never get the browser tools.

## Turn it on

In the terminal chat:

```text
/browser
```

- It shows whether browsing is on and, when it is, checks that Safari answers; then a list offers **Turn browsing on** or **Leave it off**.
- `/browser on` and `/browser off` do the same without the list.
- The choice is saved as `[browser] driver` in `~/.cache/localmelo/config.toml` when you have a saved configuration, and the chat goes on in a new Agent, so nothing you said is lost.

For one run only, without saving:

```bash
melo -X browser.driver=safari-automation
```

In the config file:

```toml
[browser]
driver = "safari-automation"
keep_tabs = false   # true leaves the tabs melo opened in the window
app_windows = true  # false: melo no longer looks for other apps' windows
```

`"off"` is the default. The web console and the app read the same setting when they open a chat, the main chat included: change it with `/browser on` there, or `melo config set browser.driver safari-automation`, and open a new chat.

## Set up Safari

Safari has to allow outside programs to open automation windows. In Safari:

1. Open **Settings**, then the **Developer** tab. With no Developer tab, open **Advanced** and tick **Show features for web developers** first.
2. Tick **Allow remote automation** (its full label may read "Allow remote automation and external agents").

When it is off, melo says so in its first browsing answer and in `/browser`:

```text
Safari is not set up for melo yet. In Safari, open Settings > Developer and turn on "Allow remote automation" ...
```

- Turn the switch on and ask again; melo connects again on every call, so you do not restart it.
- If the check still fails with both switches on, Apple's one-time setup is `safaridriver --enable` in Terminal (it asks for your password); melo never runs it for you.
- melo starts Safari's automation server only on the first browsing call (or `/browser`); a chat that never browses opens nothing.

## What the automation windows are

- They open apart from your own windows, with an orange address field.
- They start clean, like a private window: **none of your sign-ins, cookies or AutoFill are available**, and nothing is kept afterwards.
- Only one automation session runs at a time; a pane over the window catches your mouse and keyboard, and breaking it offers to stop the session.
- So melo reads pages that need no sign-in; pages for signed-in people, and sites that challenge automated browsers, may not work yet.

## What melo can do

| Tool | What it does |
|---|---|
| `browser_open` | Open an `http` or `https` address and read it. It reuses the tab melo opened in this chat; `tab="new"` opens another, and a tab id such as `t2` names one. |
| `browser_read` | Read the current page again, or the next part of a long page (`part`). |
| `browser_find` | Read the page again and search its elements (links, buttons, menu items, fields) by words in their visible text, `aria-label` or address, so its refs are always those of the page as it is now. It says how many things on the page it left out. |
| `browser_follow` | Open the address of a link from the last reading. It opens the address; it does not click. |
| `browser_click` | Click a button, link, menu item or checkbox from the last reading. The answer says how it clicked and what changed, then reads the page again. |
| `browser_type` | Type into a text box from the last reading, at most 1,000 characters (`submit=true` presses Enter afterwards, which sends the form). melo asks about every text and shows all of it. Never a password, card or one-time-code field. |
| `browser_select` | Choose an option in a select list from the last reading, by its text. |
| `browser_back` | Go back one page in the tab. |
| `browser_tabs` | List melo's tabs, switch to one, or close one. A tab whose page melo has not checked shows as *(not checked)*. |
| `browser_wait` | Wait (30 seconds at most) for the page melo opened to load, or for some text to appear or go. A page with no body yet, such as a bot check, counts as still loading. The wait ends early when another app puts a window on the screen. |
| `browser_screenshot` | Take a picture of the page. The picture is kept with the chat, and the web console shows it; the terminal does not draw images. A model that reads images also receives it, in the next step (see `tool_images` in the [configuration reference](../reference/configuration.md)). |
| `browser_click_point` | Click a point of the last picture (`x` and `y` from 0 to 1000 across and down it), for what the element list lacks: a drawn button, a box from another site. The answer has a new picture with the point marked. |
| `window_read` | Read a window another app opened after melo's own step (see [Windows from other apps](#windows-from-other-apps)): its text, its buttons and a picture of only that window. |
| `window_press` | Press a button in that window by its name, or, when that does not go through, a point of its picture. melo asks you first, every time. |

A page reaches the model as one bounded block:

```text
<<page id=s1 nonce=ab3218de tab=t1 origin=https://example.com title="Example Home">>
Welcome to the example.
Second line, with 中文。
elements:
[e1 link "News" -> /news]
[e2 textbox "Search" value="cats"]
[e3 password field: melo never types here]
[e4 button "Go" sends a form -> /search]
<<end page s1 nonce=ab3218de>>
```

- A block holds about 6,000 characters of text; a Chinese character counts as three Latin ones, so a Chinese page is cut at about 2,000 characters. A longer page is cut at a paragraph or sentence, with `part=1/3` in the first line and a line that says which `browser_read part=2` continues.
- At most 60 elements show. `browser_find` searches all of them.
- Element refs (`e1`) are numbered across the chat and belong to the newest reading of their tab. A ref from an older page is refused, not guessed.
- The values of password, one-time-code and card fields are never read, and a page with payment fields is not read at all.
- A button that would send a form says so (`sends a form -> /search`), with the address the form goes to.
- Buttons are not only `<button>` elements: `role="button"` elements, menu items, tabs, links that run a script and other clickable elements are listed too, and `browser_find` matches their visible text and their `aria-label`. When nothing matches, the answer says how many things the page reader skipped (hidden or disabled ones, or past its limit of 400).

## Tabs

- **One tab per chat.** `browser_open` loads the next page into the tab melo opened, so a long task does not pile up windows. Ask for `tab="new"` only when two pages must stay open.
- **Short ids.** Tabs are called `t1`, `t2` and so on, in the order melo opened them, and every tool that takes a `tab` takes these. An id is never cut and never reused. Safari's own long ids never reach the model, and an id that melo did not hand out is refused.
- **Closing works.** `browser_tabs close` closes a tab of this chat. If Safari lost a tab meanwhile, the list leaves it out and says so (`t3 is gone: the browser no longer has it.`).
- **Which tab is which.** melo names a tab only by the handle Safari gave it: a tab it opened, or one it saw appear while it was acting. It never picks a tab by its place in Safari's list or because it is the active one, and the address it uses for a tab is the one Safari reports for that handle, never one the page says.
- **Your own tabs are left alone.** A tab melo did not open is never listed, read or closed; the list only says how many other tabs the window has. A tab that appears while melo clicks (a link that opens in a new tab) is checked like the others and can be read, up to three; any more are not looked at and not touched. It is not melo's to close: melo closes only the tabs it made with `browser_open`, and a tab that appeared is left open, and let go of, when the chat ends. A plain `browser_open` never loads a page into it.
- **Tidying up.** The tabs `browser_open` made are closed when the chat ends and after ten minutes without a browser call; the last one is kept, empty, so Safari's automation session goes on, and nothing in it stays approved. `/new` does the same. Set `[browser] keep_tabs = true` to leave them all open.

<a id="clicking-typing-and-choosing"></a>

## Clicking, typing and choosing

`browser_click`, `browser_type` and `browser_select` act on an element of the last reading by its ref (`e12`), the way a person would. They work inside melo's browser window; they do not move your mouse or touch other apps.

```text
Clicked e12 (via browser).
<<page id=s4 nonce=1c9e04f2 tab=t1 origin=https://example.com title="Daily">>
what changed:
- new text: 'Checked in: day 12'
- the element acted on: now disabled
...
```

- **Safari's own input first.** melo finds the element in Safari's own reading of the page and Safari clicks, types or chooses on it: real events, the same as yours, so a page that ignores script clicks still works. When Safari cannot reach the element, a page script does it instead. The first line names the way: `via browser`, `via page-script` with the reason, or `via point`. A site where Safari's own input did not work starts with the page script for the rest of the chat.
- **What changed.** melo watches the page from just before the action. It waits until the page is quiet (no change for about 0.3 seconds, and after a click up to 2 seconds for the site's answer), then reports new text such as a notice or an error (even one that is gone again), a new address, and the clicked element's new state. When nothing changed, the first line says so. The page is read again, so the next step needs no `browser_read`.
- **A page that redrew itself.** If the page rebuilt its elements since melo read it, without loading again, melo reads it again and uses the one element with the same kind, name, address and form that still belongs where it did (the same row or heading and the same text next to it, a linked title included), with every check again. It does nothing when two match, when the page had two such elements when melo read it (a "Delete" in each row, say), or when nothing on the page says where the element belongs; where a real control and a look-alike match, it uses the control.
- **A page that loaded again.** If the page reloaded at the same address since melo read it, melo reads it again and uses the one element with the same kind and name that still belongs where it did (the same row or heading and the same text next to it), and says so under *what changed*. It does nothing when two match, when nothing says where the element belongs, when the page moved to another address, or when the action sends something (a form's button, Enter in a box): what the new page would send was not asked about. So the "Delete" of a record the reload replaced with another is never pressed.
- **Frames.** A frame from the same site is read and acted in as part of the page. In a box from another site (often a sign-in box) Safari's own reading lists the buttons, links and fields, as `[e9 button "Continue" in a box from accounts.example]`; melo cannot check inside it, so each click or text there asks you, every time, and a field whose type Safari does not name counts as a secret one. A box with nothing melo can reach shows as `[a box from accounts.example that melo can't reach ...]`.
- **A point of a picture, as the last way.** For a drawn button or a box from another site, the model takes a `browser_screenshot` and names a point with `browser_click_point`. melo checks what is there first: a label moves to its field, and a point on nothing clickable is refused. A point on a password, code or card field, on a control that agrees to terms, consent or a sign-in with another account, or inside a box from another site asks you every time. Link and send checks are the same as for a click. A red ring shows where melo clicks, and the next picture marks the point. melo cannot see inside a box from another site, so the new picture is how you and the model see what a click there did.

- **Only the page that was read.** The action is done only if the tab still shows the same address and the same document as when the page was read (a reload, or a page a script replaced, is another document) and the element is still the one that was read: same place, kind, name and address, and for a text box the same description as a field. Where a link leads and where a form is sent are read from the page again at that moment; if either changed since the reading, nothing is done. Otherwise nothing happens and melo asks the model to read the page again. A ref from an older page is refused before the browser is called.
- **What melo does while it acts, and what it cannot stop.** melo asks before it types a text and before it presses send. While it acts it also tries to stop a form the page sends that melo did not ask about, a link whose address changed when it was pressed or clicked, and a real click or key press that lands on another element than the one melo checked (the answer tells the model what was stopped), and it drops a link's tracking `ping`. That is a best-effort guard, not a promise: it holds only while melo is acting, and a page can get around it (a send from a timer, a request from a script, a reference it saved earlier). **Once a text is in a page, the page itself may send it, and melo cannot stop every such send.** Between melo's calls the page runs its own scripts as it would for you.
- **After each action** melo takes the address Safari reports for every tab the action touched (the tab it acted on and up to three tabs that appeared meanwhile; any further new tab is neither checked, read nor closed) and checks them all before it reads any: never this Mac, your network or melo's console, and a site it has not asked about is asked about before the page is read. A tab on a local or private address, or one whose address melo cannot tell, is blanked and nothing from it reaches the model. Blanking replaces the page with an empty one in the browser's history, so Back does not return to it. If the replacement does not take within a moment, melo opens a blank page the browser's own way instead, which adds an entry, and Back can then reach the refused page: `browser_back` checks where it lands like any read, so melo refuses that page, blanks the tab again and reads nothing from it. If blanking fails altogether, melo closes a tab it made with `browser_open`, and only lets go of any other. A page that melo reads is also compared with the address Safari reported for the tab just before: a page that moved in between is not read. A failed action is checked the same way, and the readings of its page are dropped. If melo cannot list the window's tabs, it does not act. Then it reads the page again; its refs are new.
- **What that does not do.** melo itself never opens a local or private address, and a link or form that points at one is refused before it is used. The one way melo's own action can reach one is `browser_back`, when an earlier page in the tab's history was refused and could not be replaced (see above); it checks where it lands, refuses that page and blanks the tab again. A page can also send its own tab to such an address with its own script; melo then notices, blanks the tab and reads nothing from it. That is a check on what melo reads, not a network block: Safari still makes the requests a page asks for (images, scripts, background calls), and a form a page sends from its own script can reach an address before melo can blank the tab.
- **Never a secret field.** melo does not type into a password, card or one-time-code field, and says why. When it is unsure, the field counts as secret. It decides from the field's `type`, its `autocomplete` values (`current-password`, `new-password`, `one-time-code`, `cc-*`), text the page masks with CSS, the words that name it (name, id, `aria-label`, the text `aria-labelledby` and `aria-describedby` point at, title, placeholder, its labels, its fieldset's legend and the text just before it, in English, Chinese and a few other languages: password, PIN (also written into a word, as in "PINcode"), card number, expiry, CVV or CVC2, "code" (also at the end of a longer word, as in "Bestätigungscode"), token, verification, 2-step and so on, but not a postal, promo or country code) and the shape of a code (a box for one character, a short numeric or telephone field, a digits-only `pattern` or an all-9s `max`, or a short box whose names say nothing). It decides when it reads the page, right after the field gets focus, after the text is in and after the page's own handlers ran; a field that turned into a protected one is cleared and melo says so. A card expiry list counts as a payment field, and a page with a payment field is not read at all. Ask the person to fill such a field in the window themselves. melo cannot see where a page's scripts send what is typed in an ordinary field, or what they do with it: type nothing there that you would not put on that page yourself.
- **Clicks and choices ask apart from reading.** The first time melo wants to click or choose on a site in a chat you see "Let melo click and choose on example.com?"; "allow for this chat" covers that site's clicks and choices only. Opening the site does not allow it.
- **Typing asks every time, and shows all of the text.** Each `browser_type` asks "Let melo type on example.com?" with the whole text, however long (up to 1,000 characters; a longer text is refused, so it is typed in parts). A text is refused if it has a control, format (zero-width space, direction override, byte-order mark, tag), surrogate, private-use or unassigned character, a line or paragraph separator, a Hangul filler (U+3164, U+FFA0), the blank Braille cell (U+2800), the combining grapheme joiner (U+034F) or a variation selector (U+FE00-FE0F, U+E0100-E01EF), because the question could not show them; that includes an emoji written with U+FE0F. New lines, tabs and the joiners U+200C and U+200D are allowed. The answers are *Yes* (this one text) and *No*: typing has no "allow for this chat", in the terminal, the console or the API.
- **A form that is sent asks by itself, every time.** A click on a button that submits a form, and `browser_type` with `submit=true` (Enter), ask "Let melo send a form from example.com?" with the address the form goes to and what the form holds, as melo last read it (the page can change a value before it sends); values that look secret or like a token show as `(hidden)`. When Enter is pressed outside any form, melo cannot say where it goes and the question says the page decides, and the page may then send any form on it. The answers are *Yes* (this one send) and *No*: a send has no "allow for this chat", in the terminal, the console or the API. The address is checked first, like any other: a form that posts to this Mac or your network is refused outright, and one that posts to another site asks about that site before anything is sent.
- **A link's address is checked before the click**, the same way, and again at the mouse press. A click that lands on a local address anyway (a script sent it there) is blanked and refused.
- **Your session stays yours.** melo never reads cookies or page storage, and the automation windows have none of your sign-ins.

<a id="windows-from-other-apps"></a>

## Windows from other apps

A page melo acts on can make another app open a window, such as a chat app's sign-in window that asks you to authorize. melo looks for one after each of its own steps in the browser.

- **With the desktop control helper.** The helper is a separate app that holds the Mac's Accessibility and Screen Recording permissions, so melo never needs them. melo names the new window and its buttons in that step's answer, can read it with a picture of only that window, and can press one of its buttons. With Accessibility alone it reads the window's text and buttons and presses by name, without a picture.
- **Pressing asks.** In Ask mode you see a card for each press. In Auto mode a press the task needs, such as authorizing the sign-in you asked for, goes ahead; anything else stops.
- **Without the helper.** melo cannot read the window. It still notices that another app came to the front and tells you which one, so you can answer it there.
- **Only what melo's step opened.** melo never looks at your other windows and never types into other apps. A panel of an app that only came to the front is read only when melo listed that app's windows before its step; otherwise melo names it and asks you to answer it.
- **The setting.** Settings → Chats and folders → **Other apps' windows**, shown while browsing is on. It is on by default; `[browser] app_windows = false` turns it off.

## Who decides which sites melo opens

- **Each site asks once per chat.** Before melo loads a page, you see "Open a website?" with the site's name and the answers *Yes*, *Yes, and allow this site for the rest of this chat*, and *No*. In the web console it is the usual approval card. "For the rest of this chat" is the only lasting answer; there is no "always" in this part. `/new`, and `/approvals forget`, take the grants back.
- **A redirect asks again.** If a link or a redirect lands on another site, melo asks about that site before it reads the page. If you say no, the window goes back to a blank page and the model reads that you refused.
- **Only `http` and `https`.** `file:`, `about:`, `javascript:`, `data:` and the browser's own pages are never opened. Nor is an address with a user name or password in it (`user@site`).
- **Never this Mac, your network or melo's console.** As with `web_fetch`, melo does not open `localhost`, private, link-local or carrier-shared addresses, `.local` names, names without a dot, an address written as one number, or a name whose DNS answer is one of these. melo's own console is refused too, wherever it listens. A redirect or a page that moves to such an address is blanked before anything is read from it; this is checked before any question about the site, and it is a check on what melo reads, not a network block. For a task's result, the main chat reads the task itself (`task_get`).
- **In auto mode only what you did not ask for asks.** A site your own recent messages name opens (that site itself, with or without `www.`, not a site under it); another one opens when your model finds that it serves what you asked (checked once per site per chat), and asks you otherwise, on the same card, with *Allow once*, *Allow for this chat* and *Deny*. A click or a choice on a site is reviewed like opening it, and a box or a button your words name counts as asked for. Typing is reviewed by itself, even on a site you named, and so is every form that is sent; the verdict on one text or one send is not kept for the next, and neither is your answer. `--yolo` skips every question.
- **Page text is data.** A page can say anything. melo's system prompt tells the model that text between the page markers is content of a web page, never instructions, and the closing marker carries a random number that no page can know, so a page cannot end its own block. This lowers the risk of a page steering the model; it does not remove it. Do not allow sites you do not trust, and do not turn browsing on in a chat that holds secrets you would not paste into a web page.

## What is coming

Browsing is built in parts; none of the next two exists yet.

- Acting in your own signed-in browser (a Chrome extension): clicks and typing there, with a confirmation that `auto` and `--yolo` cannot skip for sending, posting, buying and signing in. The clicks on this page happen only inside melo's own automation window; moving your mouse on the desktop (computer use, for apps outside the browser) is a separate feature and the next way after a point of a picture.
- Saved chores that run on a schedule (for example a daily check-in), with steps you approved once.

## Troubleshooting

| You see | What to do |
|---|---|
| `Safari is not set up for melo yet` | Turn on the two switches in [Set up Safari](#set-up-safari). |
| `Safari's automation helper did not start` | Check that Safari 27 or later is installed and the switches are on; `/browser` runs the check again. |
| `This Safari's automation server lacks the tools melo needs` | Update macOS and Safari. |
| A page reads as empty | It may need a sign-in or be built by scripts after loading; try `browser_wait`, or open it yourself. |
| melo says it did not open a site | You answered no to the question about it; ask again and allow the site. |
| `... is not a tab of this chat` | Use a tab id such as `t1` from `browser_tabs list`; ids are never cut or guessed. |
| `That is a password field: melo never types ...` | By design. melo does not type passwords, card numbers or one-time codes; fill the field in yourself, in the automation window. |
| `The page is not the one that was read` or `That element is no longer on the page` | The page changed after it was read. `browser_read` gives current refs. |
| `The page tried to send a form that melo had not asked about` | melo stopped it. To send a form, click its button: melo asks first. |
| `(via page-script: ...)` after an action | Safari could not reach that element itself, so a page script acted. A page that ignores script clicks may then do nothing: take a `browser_screenshot` and use `browser_click_point`. |
| `A real click or key press landed on another element` | Something covered the element between melo's check and the click, so melo stopped it. `browser_read` and try again. |
| `... came to the front: it may show a window melo cannot see` | Another app opened a window and the desktop control helper is not installed or lacks its permissions. Answer the window in that app yourself. |
| `melo cannot read other apps' windows: ...` | Install the desktop control helper (see [What you need](#what-you-need)), or give it Accessibility in System Settings → Privacy & Security. |
| `no Screen Recording permission` | The helper reads text and buttons and presses by name, but has no picture. Give it Screen Recording for pictures and point clicks. |
| `a box from ... that melo can't reach` | A frame from another site, often a sign-in box, with nothing Safari lists in it. Fill it in yourself, or let melo click a point of a picture of it. |
| `Take a browser_screenshot of this tab first` or `The page scrolled since the picture` | A point belongs to the last picture of the same page and view; take a new picture. |
| `The page moved while melo was reading it` | The tab changed address between the check and the reading, so nothing was read. `browser_read` tries again. |
| `That text is N characters; melo types at most 1000` | Type it in parts: the question shows all of each text. |
| `melo could not set up its safe page checks` | The page blocked the hidden frame melo's checks come from. melo then reads the text only and does nothing on the page. |
| `melo could not list the browser's tabs` | melo does not act in a window it cannot list, since it would not know which tabs the action opens. Try again, or `/browser` to check Safari. |
| `melo's browser does not open local or private addresses` | The address is this Mac, your network or melo's console. That is on purpose; open it yourself if you need it. The developer setting that lets `web_fetch` read a test server (`web_fetch_allow_local`) does not change this. |

To see what Safari's automation server answers, with no model involved, run this from the checkout. It opens an automation window, reads one page, lists the tabs, takes a picture and closes; each line shows what came back, and a failure says what to turn on:

```bash
uv run python -m localmelo.support.browser.probe
```
