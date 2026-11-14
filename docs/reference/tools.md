# Tools and MCP

This page covers the tools the model can call: the built-in tools, tools
from MCP servers, which tools each step offers, and what the checks do.

## Built-in tools

| Tool | What it does | Limits |
|---|---|---|
| `shell_exec` | Runs a shell command with `/dev/null` as standard input and returns its output. Takes `command` and an optional `why`, one short sentence that starts the approval card; it changes nothing about what runs. | Killed with every process it started after 30 seconds. Only `interactive` mode offers it by default: a direct run and a Python caller need an explicit `builtins` set that names it; a gateway or console session does not offer it by default and refuses it even when named. |
| `python_exec` | Runs Python code in a child process and returns what it prints. Takes `code` and an optional `why`, as `shell_exec` does. | Killed with every process it started after 60 seconds. `sys.exit()` ends only the child. |
| `file_read` | Reads a text file. Takes `path` and, to read a part, `start_line` and `end_line` (1-based, inclusive). | Any file your account can read. A part that does not show the whole file ends with a note of how many lines the file has and the `start_line` to go on from (`... [the file has 1,240 lines; showing 1-400; read more with start_line=401]`). A file that starts with `%PDF-` is read as a PDF (see [Reading a PDF](#reading-a-pdf)). |
| `list_files` | Lists a folder: its folders first, each ending in `/`, then its files with their sizes; a link ends in `@`. Takes `path` (default: the workspace folder) and `depth` (1 to 3, default 1). Never reads a file, never follows a link and skips `.git`. | Shows 200 entries, then `... and N more`. Read-only, so it never asks for approval. The interactive UI and gateway and console sessions offer it by default; a direct run and the Python API's default mode do not (`--builtins list_files` or `all` names it). |
| `file_write` | Writes text to a file, creating or replacing it. | Any file your account can write. |
| `file_edit` | Replaces one exact piece of a file (`old`, which must occur once) with `new`. | Files up to 1 MB. `old` is checked first: an edit whose text is missing or occurs more than once fails at once with a plain error, and asks no one. Otherwise it asks as `file_write` does; the approval card shows the change as a diff. Only a task's worker offers it (name it in `builtins` elsewhere). |
| `git_status` | Reads the workspace's git state: `status`, `branch`, `log` (`count` up to 50), `diff` (`stat`, `staged`, `paths`) or `show` (`rev`). | Read-only: git's own argv, no shell, no hooks, pager or external diff, no index refresh. Paths (and a `REV:path`'s path) must stay inside the workspace and may not name a file that may hold keys (`.env`, `*.pem`, `id_rsa`…). Never asks. Output over 20,000 characters is cut. Only a task's worker offers it. |

`--builtins all` names the first five; `file_edit` and `git_status` are a task's own and are named one by one.

A `file_read`, `file_write` or `list_files` call may take up to 60 seconds, and an MCP tool
call up to its server's `timeout`. Output longer than 50,000 characters is
cut.

The `python_exec` code reaches the child on its standard input, a pipe,
never in its `-c` argument: `ps` does not show the code, and a call has no
`ARG_MAX` limit (about 1 MiB of argv and environment on macOS). The child
runs a fixed wrapper; once that has read the code it points fd 0 at
`/dev/null`, so `input()` gets EOF.

The child may have at most 1,024 open
files and one hour of CPU time (the POSIX `RLIMIT_NOFILE` and `RLIMIT_CPU`
limits). The call's time limit normally ends it first; the CPU limit only
stops code that outlives melo or that spends an hour of CPU.

An address-space limit is not set: macOS refuses a finite one.

`shell_exec` and `python_exec` start their child with a minimal
environment: `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `LANG`,
every `LC_*` variable and `TMPDIR`, then whatever
`AgentOptions(tool_env={...})` adds, and nothing else. Config-backed runs
and sessions can additionally inherit names listed in `[agent] tool_env`;
see [Tool environment](configuration.md#tool-environment).

An API key in melo's own environment reaches no tool unless explicitly named or mapped.
`VIRTUAL_ENV`, `PYTHONPATH`, `SSH_AUTH_SOCK` and proxy variables are not
inherited by default.

Exposing `SSH_AUTH_SOCK` permits tools to use your SSH
agent, and exposing a proxy variable also exposes any credentials in it.

This is how the built-in tools work in a direct run without `--workspace`
and in the gateway. A direct run with `--workspace DIR` and a Python agent
with `workspace_root` keep the file tools (`file_read`, `file_write` and `list_files`) inside a directory and start
`shell_exec` and `python_exec` there (see
[With a workspace root](../guides/tools-mcp.md#with-a-workspace-root)).

The interactive UI (bare
`melo`) runs with its workspace as the root too, and asks before each
`shell_exec`, `python_exec` and `file_write` call, unless `--yolo`; `file_read` and
`list_files` only read, so they never ask (see
[Permission tiers](../guides/tools-mcp.md#permission-tiers)); direct mode has no approval policy
and offers neither `shell_exec` nor `list_files`, with or without a root. The interactive UI loads the MCP
servers of `--mcp-config`, else of the top-level `mcp_config` key; a server
that fails to start turns MCP off for that session (`/mcp` says why).

A Python run with an approval policy asks its own policy about an MCP tool by
the same tiers.

## Reading a public web page

`web_fetch` reads one public `http` or `https` page and returns it as text. Every chat
(the terminal's, the web console's and the app's) and every console task has it while
`[tools] web_fetch` is on, the default; direct runs never have it. Its tier is `read`, so the
call itself asks nothing; each site passes the site gate below first.

- **Site gate**, before the site's name is looked up, for the first address and for a redirect
  to another host:
  - a site your own recent messages name is read at once, in every mode. Recent means your
    last 3 messages, the first 800 characters of each. Named means that host itself, or it
    with `www.` added or removed: `example.com` does not name `docs.example.com`, so naming
    a shared domain such as `vercel.app` names none of the sites under it. A full `http://`
    or `https://` address names its host; a bare word such as `README.md` or `config.yaml`
    whose last part is a common file extension names no site. Text you paste counts as your
    own words, so a site in a pasted log or page names it too;
  - any other site asks once per chat with the browser's site question, worded "Let melo read
    example.com?", showing the full address it asks for (cut at 300 characters): one "allow
    for this chat" covers `web_fetch` and the browser tools on all the hosts of that site (see
    [Browsing](../guides/browsing.md)), holds in every chat when given in the main chat, and
    `/grants revoke N`, Settings → Permissions or `/approvals forget` takes it back;
  - in auto mode, an unnamed site is reviewed by your model, as a site the browser would open is.
    A verdict that lets it read covers that site's other addresses with no query (`?`), no
    fragment (`#`), no user name and a path of at most 120 characters, up to 10 of them until
    your next message, so its plain pages are not reviewed one by one. Any other address, which
    can carry more data out, is reviewed by itself, and so is one longer than 300 characters or
    one that an earlier review did not let through. A verdict that blocks covers only its own
    address. melo keeps the last 512 verdicts of a chat;
  - a task asks the chat that owns it; a task with nobody to ask reads no site.
  The address (its path and query, and its name through your resolver) is the one thing
  `web_fetch` sends out, which is why a site you did not name asks first.

- **Input.** `url`, and `part` (from 1) for a later part of a long page.
- **Output.** One line with the final address after redirects and the kind of page, then the
  page in the block the browser tools use,
  `<<page id=wN nonce=N url=URL type=html|text|json|pdf title="…" [part=i/n]>>` …
  `<<end page wN nonce=N>>`, then melo's notes: `[…cut: …]` when more parts follow or the
  download reached its limit, and a note when a page has little text without its scripts.
- **HTML.** The title, headings (`# …`), the main text (only `<main>` or `<article>` when it
  holds enough text), list items, table cells joined with ` | `, and up to 40 links. Scripts,
  styles and hidden text (`hidden`, `aria-hidden`, `display: none`) are left out. `text/*`,
  JSON and XML come back as they are; a PDF is read as text (see [Reading a PDF](#reading-a-pdf)); images and other files are refused with their type.
- **Server text.** Nothing a server writes in its status line or headers reaches the model: an
  error says the status code only, and a media type or an encoding is named only when it is a
  plain token (else "an unknown type"). Runs of `<` or `>` in a page are spaced out, so a page
  cannot write a marker.
- **Limits.** 8 seconds to connect, 15 seconds per read, 30 seconds in all; 2 MB downloaded,
  counted after gzip or deflate is undone; about 6,000 text units a part (a CJK character is 3);
  at most 5 redirects, and never from `https` to `http`; addresses of at most 2,048 characters.
  Name lookups run on four threads of their own. The time you take to answer a site question is
  not counted.
- **What it sends.** A `GET` with `User-Agent: localmelo-web-fetch/1.0
  (+https://github.com/localmelo/localmelo)`. No cookies (one a server sets is dropped), no
  `Authorization`, no proxy or `.netrc` from the environment.
- **What it refuses**, before any connection and again at every redirect:
  - schemes other than `http` and `https`, a user name or password in the address, spaces,
    control characters and backslashes;
  - `localhost`, names under `.localhost`, `.local`, `.localdomain`, `.internal`, `.home.arpa`,
    `.home`, `.lan` and `.corp`, and a name without a dot;
  - a number written as one integer, in octal or hex, or in fewer than four parts;
  - loopback, private (RFC 1918), shared carrier (100.64.0.0/10), link-local and cloud metadata
    (169.254.0.0/16), multicast, reserved, documentation and benchmarking addresses; IPv6
    unique-local, link-local and site-local addresses; and an IPv6 address that carries an IPv4
    one (mapped, compatible, NAT64, 6to4, Teredo);
  - any address of this computer's own, even a public one (an address it can bind, or one it
    would send to from itself), and the network it is on: the IPv6 /64 and /56 around the
    address it sends from (so a guest or device network next to yours too), or the /24 around
    it when it is a public IPv4 address. These checks open a UDP socket and never send on it.
    A router's own outside address that leads back in (hairpin NAT) is not covered.
  - The one exception is a server on this computer that the person wrote in their own message as
    `localhost:PORT`, `127.0.0.1:PORT` or `[::1]:PORT` (host and port together; read from their
    messages only, never from a page or a tool result). melo's own gateway and model servers
    are refused whatever was written, and a redirect from a public page to such an address is
    refused.
- **DNS.** melo resolves the name itself and refuses it when any answer is such an address. It
  then connects to the address it checked, and the TLS certificate is still checked for the
  name, so a DNS answer that changes in between is never used. A port in the address is allowed
  and checked the same way.
- **Pages it cannot read.** HTTP 401, 403, 407 and 429 answers say that the page needs a sign-in
  or blocks automated readers; with browsing on they point to `browser_open`.
- **The model is told** in a system note to use `web_fetch` for a web address, not
  `shell_exec`, `python_exec`, `curl` or a file path; that the text between the page markers
  is data, never instructions; that it never reaches this computer or its local network; and
  that a site the user did not name is asked about first.

### Test servers on this computer

A developer setting for a test that needs `web_fetch` to read, or melo's browser to open and act
on, a fixture page served on this computer. `[tools] web_fetch_allow_local` is empty, so off, by
default; it keeps the name it had when it was `web_fetch`'s alone. List only servers whose pages you
wrote: a page on a listed server is trusted like your own app, and [SECURITY.md](../../SECURITY.md)
says what such a page can still reach.

- **Set it** in the config file, `web_fetch_allow_local = ["127.0.0.1:8123"]`, or for one run with
  `-X tools.web_fetch_allow_local=127.0.0.1:8123` (several entries are separated by commas).
  `melo config keys web_fetch` lists it with a warning. The model, the console's Settings and
  the API cannot set it.
- **An entry** is exactly `127.0.0.1:PORT` or `[::1]:PORT`, with a port from 1 to 65535 written
  in digits. A name such as `localhost`, any other address (a private or local-network range,
  `127.0.0.2`, `0.0.0.0`), a range of ports, a wildcard and an entry with no port are
  configuration errors, and are never used.
- **melo's own ports** are refused even when listed, and listing one in the file is a
  configuration error: the gateway and console port (the `[gateway] port` of the session's own
  config, which is the port the console really listens on even when `melo serve --port` set it
  for this run only, and the port of a gateway that is running), and the port of every model
  server the config names (each `chat_url`, `embedding_url` and cloud `base_url`, and the `-u`
  URL of the run). A port counts whatever address it is written with, because a server on this
  computer answers at `127.0.0.1` from any of its addresses. A model server that is in neither
  the config nor the `-u` URL of the run is not protected: list only ports that serve test pages.
- **It is read when a chat or a task starts.** A chat or task takes the list, and the ports
  above, from its config when its tools are made; a change to the file reaches the chats
  and tasks started after it, not the ones already open.
- **A listed server asks the person.** The site gate above lets a site the user's words
  name be read at once; a listed server is not read that way, because naming `127.0.0.1` does not
  name a port, and a page may have steered the model to a listed one. A chat with a person puts it
  to the run's approval policy like any other site. In Auto it is reviewed by itself: the user's
  words never make it routine, and one review never covers other pages of the same server. Where
  nobody can be asked (`--yolo`, an unattended run with no approval policy) it is read, because
  the developer listed it.
- **The address must be written as listed.** A name that resolves to a listed address
  (`localhost`, `pages.test`) and another way of writing it (`127.1`, `[::ffff:127.0.0.1]`) are
  refused, as is every other local address and every other port.
- **Redirects meet the same list.** A listed address may redirect to another listed address and
  never to an unlisted one, and a page that is not itself a listed address cannot send melo to
  one. The site gate above still applies.
- **The browser opens a listed server only as melo's own step.** `browser_open` reaches it, and
  so does a page already on a listed server (its links, clicks, frames, forms, its own
  requests). A request of any other page to it stays refused as before: a public page's
  link, redirect, frame or background request, a redirect of the address melo opened to
  another listed server, and a new tab a page opens. In Chrome in the background a page's own
  request is judged by the page that sent it (its `Origin`, else its `Referer`), and one that
  carries neither is blocked; Safari does not check a page's own requests, so there only where
  a page lands and its frames are checked. The site question is asked as for any site.
- **A call cannot set it.** A `shell_exec` or `python_exec` call that runs
  `melo config set tools.web_fetch_allow_local …`, or passes the key with `-X` or `--set`, is a
  change to localmelo itself: it asks every time, is never granted, and Auto mode refuses it. A
  key the call builds at run time is undecided.
- **It is marked.** A fetch or a browser step the list allows adds `allowed_local: true` to the call's
  `tool.finished` event (see [Events](events.md#toolfinished)) and the line `[This address is a test
  server on this computer, which [tools] web_fetch_allow_local lists.]` to what the call shows,
  an error included. While the list has an entry, the model's note about `web_fetch` gains one
  sentence that names the exception.

## Reading a PDF

`file_read` and `web_fetch` read a PDF through one reader. A file or a download is a PDF when its first bytes are `%PDF-`, whatever its name or type says; `web_fetch` takes one served as `application/pdf`, as a plain file (`application/octet-stream`), with no type, or even as text or HTML: its first bytes decide. Images, audio and video are still refused before they are downloaded.

- **Text.** The pages in order, each under a line `--- page N of M ---`. A row stays whole: a place and a date stay on the line of their school or employer. `file_read` counts the lines of that text (`start_line`, `end_line`) and cuts a long one as it cuts any file; `web_fetch` gives it in parts (`part`), as for a page of plain text, between the same markers with `type=pdf`, cleaned as a page's text is; a part holds at most 3 pages with no text, so each part brings its own pictures and `part=N` reads on.
- **Pages with no text.** A page with fewer than 20 characters, spaces left out (a scan, a drawing), reads `[page N: no text, picture attached]` and the result carries a picture of it: a PNG at 150 dpi, at most 1,280 pixels wide and 2,048 tall, the size of a screenshot. A page whose picture would take more than about 3 MB is drawn smaller to fit, down to 480 pixels wide; a page that cannot be made so reads `picture failed`. At most 3 pictures come with one result, in page order, and only for pages in the lines or the part the result shows. Another such page reads `picture not sent`; a note names it and gives the `start_line` or `part` to read on from. A page that cannot be drawn reads `picture failed`. A model that is not sent pictures (`tool_images`) is told so as for any picture, and the note asks it to say which pages could not be read. An empty page never reads as an empty document.
- **The child process.** The bytes go to a child Python started with `-I` on its standard input, and nothing else goes with them. pypdfium2 is imported only in the child, so a crash of the library ends the child and not the gateway. The child has a process group of its own that a cancellation, the time limit or a memory limit kills, the environment `child_env` gives, and no output but one JSON object. That answer is not trusted: it is read up to about 4 MB for text and 12 MB for pictures, and its counts, sizes, text and each picture's whole PNG structure (checksums included, never decoded in melo) are checked before any of it is used; one that does not fit is the same plain line as a damaged file. The child also ends itself, by an alarm after 25 seconds of any kind, so it does not outlive a melo that is gone.
- **Limits.** `file_read`: 50 MB a file. `web_fetch`: its 2 MB download limit, and a longer PDF is refused with a line that says so. Both: the first 300 pages and 1,000,000 characters (a note names the pages left unread), 20 seconds on the clock and 25 of CPU for the child, and 1 GB of resident memory for the child, looked at every quarter of a second. That limit is a sample that stops ordinary runaway memory, not a sandbox: a burst between two looks can pass it before it is seen. If the memory cannot be read three looks in a row, the child is stopped. One PDF is read at a time in a process, so the limit is for all PDF reads together.
- **Refusals.** One plain line, never a traceback: password-protected, damaged or not a PDF, larger than the limit, longer than the time limit, needed more memory than the limit, melo could not check the memory, "the PDF reader stopped on it", or no PDF reader installed.
- **Checks first.** `file_read`'s path checks and `web_fetch`'s site gate run exactly as for any other file or page, before a byte is parsed.

## Searching the web

`web_search` looks words up on public search engines and returns a list of results. Every chat
(the terminal's, the web console's and the app's) and every console task has it while
`[web] search` is on; the default is off, and direct runs never have it. Its tier is `read`,
so the call itself asks nothing; the one case that asks is below. The guide is
[Searching the web with melo](../guides/web-search.md).

- **Input.** `query` (the words) and `count` (1 to 8; the default is `[web] search_count`, 6).
  Whitespace in `query` is collapsed, and at most 300 characters are sent.
- **Output.** One line, `Searched the web for “…” with <engine>: N results.`, then the results in
  a block, `<<page id=sN nonce=N type=search way=KEY results=N>>` … `<<end page sN nonce=N>>`,
  numbered, each with its title, address and snippet (at most 120, 300 and 240 characters, cleaned
  as page text is), then melo's notes: `[<engine> <reason>; … <engine> answered.]` for the ways
  that did not answer, and a note when no way gave results that fit the words.
- **Ways**, tried in `[web] search_order` (default `bing-rss`, `browser`, `baidu`) until one gives
  results that fit:
  - `bing-rss`: `https://www.bing.com/search?format=rss`, read with `web_fetch`'s fetcher (one `GET`,
    no cookies, `User-Agent: localmelo-web-fetch/1.0`, the same address checks and redirect checks);
  - `browser`: `html.duckduckgo.com/html/` loaded in the Safari automation window of the browser
    tools, in a tab of its own that is closed afterwards (the chat's own tabs and current tab
    stay as they were); it is left out while `[browser] driver` is `off` and waits at most 25
    seconds. It is read by the browser provider itself (`BrowserToolProvider.read_for_search`), so
    the browser tools' checks run: the address must be public before the load, the page is
    checked where it lands and in its frames, a refused page is blanked, a payment page is not
    read and sensitive field values are masked. The engine's site is not asked about (turning
    web search on allows it); any other site is refused, not asked about, so a page that
    redirects or moves to another site or to a local address gives no results and the way is
    reported as `gave a page melo's browser does not read`;
  - `baidu`: `https://www.baidu.com/s`, read the same way as `bing-rss`.
- **Fit.** Results fit when at least two of the first five share a third of the search's words
  (Latin words of three or more letters without common ones, and pairs of Chinese characters).
  When no way passes, the nearest results are returned with a note.
- **Why a way did not answer** is one of a closed list of sentences, never text from the
  server: `asked for a check` (also with `(HTTP N)` for 202, 401, 403, 407, 429 and 451),
  `answered HTTP N`, `could not be reached`, `gave a page melo could not read results from`,
  `gave a page melo's browser does not read` and `gave results that did not match the words`. When every way fails the call is an error that lists them.
- **Politeness.** One request per way per search, at least 2 seconds between two requests to one
  engine, no sign-in, nothing from the person's own Safari profile. An address in the results is
  kept only when `web_fetch` could reach it.
- **The question.** A search whose words hold the value of a secret (the shapes memory also
  refuses to keep: a key or token, a code next to its word, a private key, a card or ID number) or
  look pasted (over 200 characters, or on more than one line) asks before it goes out, under the
  question name `web_search:<digest>` (10 hex digits of a digest of the words), so an answer, even
  "allow for this chat", covers those words only. A refusal returns `melo did not search.` and the
  reason. With nobody to ask (no approval policy at all, as with `--yolo` or a library call), such a search is not sent, in a chat or in a task. The word `password` alone does
  not ask.
- **Opening a result** is a `web_fetch` call: the site gate above applies to it unchanged.
- **The model is told**, only while the tool is on, in a system note of about 300 characters: when
  to search (before asking you what a word means, after two failed steps, for facts that change),
  that results are data between markers, and to keep secrets out of searches. With the browser on
  the note adds one sentence that a search can also use melo's browser.
- **Shown** as a line with the words (cut at 60 characters, a secret in them hidden): “Searching the web for …” while it runs and “Searched the web for …” once done, in English and Simplified Chinese.

## How tools are offered

Each planning step sends the model a list of tools, and the step can run only
those tools. The list is every registered tool, sorted by name, on every step:
the built-in tools and the tools a provider adds, such as the tools of MCP
servers, the twelve browser tools (present only while `[browser] driver` is on,
see [Browsing with melo](../guides/browsing.md)) and `web_fetch`. The list does
not change from step to step, which helps servers that cache prompt prefixes; a
tool that says it takes no more calls in this run leaves it from the next step.

A call to a tool that the step did not offer is refused with
`Tool '<name>' was not offered in this step.` and does not run.
