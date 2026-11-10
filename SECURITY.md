# Security policy

## Supported versions

localmelo is pre-alpha and has no releases yet. Security fixes land on the
`main` branch. After v0.1.0, the latest release will receive them.

## Report a vulnerability

Do not report security problems in public issues or pull requests.

- Open the repository's **Security** tab and choose
  **Report a vulnerability**.
- If that option is not shown, open an issue that asks for a private
  contact, without any details of the problem.

Include a description of the problem and its impact, the commit or the
`localmelo_version` from `--output-format json`, and the steps to reproduce.

## Security model

In localmelo, a language model chooses which tools to call. Plan for what
those tools can do.

- **Tools run with your account's permissions.** `shell_exec` runs shell
  commands, and `python_exec` runs Python code inside the localmelo process.
  `file_read` and `file_write` reach any file you can: no directory
  restriction is active, and `list_files`, which lists a folder without
  reading a file, reaches any folder you can wherever it is offered (the
  interactive UI and the gateway). This holds in direct mode and the gateway, and in
  the Python API unless an agent sets the options below. The interactive UI
  (bare `melo`) keeps the file tools inside its workspace, the current
  folder or `--workspace DIR`.
- **A PDF is parsed in a child process; its limits are not a sandbox.**
  `file_read` and `web_fetch` read a PDF (a file or a download that starts with
  `%PDF-`) with pypdfium2, a native library that parses untrusted bytes. The
  bytes go to a child Python (`python -I`, a script run by path, so melo's own
  modules are not loaded) on its standard input and nothing else goes with
  them: no path, no descriptor, no secret in the environment (only the
  variables `child_env` names). A PDF that makes the library fail or exit costs
  the child, not the gateway, and the model gets one plain line. The child has
  a process group of its own that a cancellation, its 20-second time limit or
  its memory limit kills, and it ends itself after 25 seconds of CPU or of wall
  clock (an alarm in the child, so also when the process that started it is
  gone). The memory limit is a sampled one: melo reads the child's resident
  memory every 0.25 seconds and stops it over 1 GB. That stops ordinary runaway
  memory and promises no more: a burst of allocation between two looks can pass
  the limit before it is seen, memory not yet touched does not count, only the
  child's own process is counted, a page's text is whole in the child before
  the character limit cuts it, and there is no address-space limit on macOS. If
  the memory cannot be read three looks in a row while the child runs (no
  `ps`, no answer in time), melo stops the child rather than leave it
  unwatched. One PDF is read at a time in a process, so the limit holds for all
  PDF reads together. The child's answer is not trusted: it is read up to about
  4 MB for text and 12 MB for pictures and checked before any of it is used
  (types, counts, sizes, the text cleaned again, and for each picture the
  whole PNG structure with its checksums, never decoded in melo). A file over
  50 MB is never sent (`web_fetch` keeps its 2 MB download limit), and only the
  first 300 pages and 1,000,000 characters are read. The child is not
  sandboxed: it can read what you can, and the library is not memory-safe, so a
  PDF that exploits it runs with your permissions. `uv` never builds the
  library from source (`no-build-package`) and the app installs it from hashed
  wheels; a plain `pip install` of the source tree is not covered by that
  setting. The path checks of `file_read` and the site gate of `web_fetch` run
  before any byte is parsed. A PDF's words and pictures are untrusted like any
  file's or page's: `web_fetch` frames and cleans the text as it does a page's,
  and the note gate treats it as a page.
- **Approvals apply only with a policy.** In the Python API, an approval
  policy passed to `Agent.run_task` decides whether each call of
  `shell_exec`, `python_exec`, `file_write` and other tools marked as
  requiring approval may run. It is asked after every other check, and a
  policy that fails denies the call. Without a policy, as in direct mode and
  the gateway, these tools run without asking. The interactive UI asks in
  its approval panel before each such call, unless it was started with
  `--yolo`.
- **The fence applies only with a workspace root.**
  `AgentOptions(workspace_root=DIR)` keeps `file_read`, `file_write` and
  `list_files` inside that directory: paths are resolved before the check, a symlink
  that points out is refused, and so is anything but a regular file.
  Without a workspace root there is no fence.
- **`shell_exec` and `python_exec` are not fenced in chats.** With a
  workspace root they start in it, and `python_exec` runs in a child process,
  but both can reach any path you can. An approval policy is their only
  control. The terminal UI and other direct runs work this way; console and
  app chats write only in their folder, as below.
- **Worker runs write only in their workspace (macOS).** In a worker run
  (`melo worker BRIEF`, `--brief`) with a workspace, in a saved task, and in a
  console or app chat, whose folder is its workspace,
  `shell_exec` and `python_exec` start under a `sandbox-exec` profile that
  refuses every file write outside the workspace, whatever the command does
  (`cd` elsewhere, a child `git` or Python). The exceptions are the caches
  the project's tools write (uv's cache, black's cache, npm's cache), your own
  temporary folder (`/var/folders/.../T`, where macOS's `mktemp` writes; `/tmp`
  stays refused), a few devices (`/dev/null`, the terminal, the standard streams), and, when the
  workspace is a linked git worktree, its own git directory and the
  repository's object store, so `git status`, `git diff` and `git add` work
  while a commit to a branch and every file of the main checkout stay out of
  reach. `TMPDIR`, `TMP` and `TEMP` point at `.melo-tmp` in the workspace.
  Reads and the network are **not** confined: a worker can still read any
  file you can and reach any host. A refused write keeps the system's error
  (`Operation not permitted`) and gains a line that says writing outside the
  workspace is not allowed in this run; the run's record counts refused
  writes. When the profile cannot be applied (no `sandbox-exec`, or melo
  already runs inside a sandbox that refuses another), the run stops before
  its first command. On other systems a worker runs unconfined, and its
  record says so. `-X agent.confine_writes=false` (or `[agent]
  confine_writes = false`) turns confinement off.
- **Auto mode confines writes (macOS).** In a chat set to Auto,
  `shell_exec` and `python_exec` start under the same kind of profile: a step
  cannot write outside the chat's folder, and it cannot read or change melo's
  own sign-in token and session store. Other reads and the network are not
  confined. Auto is a reviewer, not a wall: a step that hands work to another
  app or program is judged by the review (blocked, or checked by your model),
  not walled off by the sandbox.
- **A chat acts only on a turn a person began.** A console chat now does the
  work itself, so it has the tools a task has, and the fences go with them:
  every step passes the same checker and approvals, its writes are confined as
  above, and in a folder a task may not use (outside your home, hidden, under
  `~/Library`) it reads and answers and has no tool that changes a file. A turn
  melo began itself, to answer a task's question or to follow one up, gets
  only tools that read, public web reads included, and no browser, also not
  through a search. That is a check on the run, made by code: the other tools
  are not offered and a call of one is refused, whatever the prompt or a page
  says. A note that misses its turn never starts work by itself, after a Stop
  or a restart included: it goes back to you, and you press Send. After a
  restart only a row stored as a message runs. Any other row, one stored by
  an older version included, goes back the same way, without its picture.
- **Work a chat hands on keeps its limits.** A chat's turn can hand the
  rest of its work to one fresh task (#571). Only the person's own turn can,
  only as the one call of its step (a step that mixes it with another call runs
  none of them), and the task is admitted in one commit before anything of it
  runs: a hand-off that is not admitted runs nothing, and a retry finds the same
  task. The task's request is the person's message; what the chat wrote about
  the work and what its calls returned reach it framed as data, never as the
  person's words, so they add no file, site or post the person did not ask for,
  and every text a call or a model gave is redacted before it is cut or kept.
  When the hand-off fails after the commit, the task is looked up before the
  chat reads anything, so the chat never goes on beside a task that holds the
  work. From the commit on, the task is registered and scheduled whatever stops
  the call: a Stop that lands then shows the task it admitted, and that task
  runs, with its own Stop. It has
  its chat's mode, folder and grants, and the same reviews. No sign-in moves:
  the request's password handles do not work in the task, and a turn that holds
  a password the person gave keeps its work. While the chat holds a browser
  page, a take-over card or an approval waiting, the work stays in the chat; a
  page is never closed and opened again to move it. A chat whose folder no task
  may use keeps its work too. A task a chat started runs
  once: more work on its job is one fresh task per ended task, admitted under a
  key of the ended one, so a second trigger reaches it instead of starting
  another; a message that reaches a worker as it ends goes on in that fresh task,
  never in the ended one's session, with a step the person allowed once and
  that did not run yet. When that fresh task cannot start, the chat says the
  work did not go on. The job keeps its follow-up count (the chat
  continues it at most once per message of the person's in that chat), the
  "asks before each post" of a repeating chore, the steps refused in it,
  including in the chat's turn that handed it on, and the calls whose effect is
  unknown, from every turn of every task of the job. Every call keeps the key
  of the live call from its start, whatever the store keeps of its arguments,
  so a call cut off with no result kept is known by it too. A call whose effect
  is unknown needs a fresh decision in every task of the job, whoever moved the
  job on, unless its tool only reads (a page action, a navigation included,
  does not): run again as it was, it is asked about whatever its tool, and only
  the person decides, on a card in Ask and in Auto. No grant, earlier approval
  or review by a model lets it run; in Auto with nobody watching it is blocked,
  and the person may allow that block once. Their Allow runs it that one time. A step refused in the job is refused
  before any review until the person's own message moves it on; then each gets
  a fresh decision.
  Stop stays final for the task it stopped. Step budgets do not carry: each
  task gets the configured step budget, and what bounds the chain is that a
  hand-off happens at most once per message of the person's and the chat
  continues a job by itself at most once per message of the person's in that
  chat.
- **A chat's mode decides for the tasks it started.** A task you started
  follows the Ask or Auto of its chat while it runs, so a switch you make in
  the chat is not ignored by work that began earlier. Auto to Ask applies at
  the task's next decision. Ask to Auto applies at the next decision after a card
  that waits: that card stays for you, and nothing answers it. The mode only
  chooses between the two policies the task already had; it adds no
  permission, and Auto's own limits (the write sandbox, the review, the
  steps it never allows) are the same as in a chat. A chore, and a task
  another program or melo made, keep the mode they were made with: a switch
  of the chat never makes one of them ask less or ask nobody. A task reads
  its chat's mode each time it decides and keeps no copy to go stale: an
  overlapping switch, one cut short, a restart, or a chat deleted while the task
  runs (the main chat then owns it, and its mode decides) cannot leave the task
  in another mode than its chat's. A task that ended keeps the mode it ended
  with. Fixing the mode when a task started was a choice for its page to show
  one value, not a rule an approval relied on; the page now shows the mode the
  task has now.
- **A chat's answer to a task never approves.** When a task asks its chat
  for help, the chat answers in a turn of its own (`origin` `ask_owner`).
  That turn has no tool that grants an approval, and the question and the
  answer reach each side framed as data, not instructions. A step that needs
  approval is still decided by you (or Auto's review), never by one agent
  for another.
- **A task reuses an approval only by an exact match in code.** In the web
  console, a task's approval goes through the chat that started it. A step
  you allowed in that chat (or in one of its tasks) runs again in that
  chat's later tasks without asking only when code finds the same tool, the
  same step key (the same command or Python syntax tree, the same file and
  content, the same folder after resolving links, and the files melo's
  rules read for it unchanged) and melo's rules read the step themselves
  and found it routine, both times. That covers `python_exec`, `shell_exec`,
  `file_write` and `file_edit` only. A tool allowed for the chat covers its
  tasks' routine calls of that tool, as in the chat. Such a task's steps
  are reviewed with every file changed since that chat began, not only
  since the task began, so code an earlier task of the chat wrote is read
  before an approval or grant covers a step; when that start cannot be
  read, neither applies and the step asks. Each such step leaves a
  row with Revoke, and the approval card says when an Allow will be kept.
  Recorded approvals live in memory only: a restart, Revoke or deleting the
  chat drops them.
- **What never reuses an approval:** a high-impact step (any review
  category, `self` and the click-only ones included); an undecided step
  (melo could not see all it runs: a file too big or not read, a command
  built at run time), a step a model settled, and a step nothing reviewed;
  a tool the rules do not read (a network or MCP tool, or any other tool:
  its calls are routine to the rules unread); a chore, or a task another
  agent, a channel or melo itself started, which neither uses nor records
  approvals; another chat's approvals; Auto's Allow once, which is one step,
  once, and never recorded; an Allow answered after its chat was deleted;
  a revoked approval. A task created with your gateway token (`POST
  /api/tasks`, `melo task new`) counts as yours and belongs to the main
  chat, so it uses the main chat's approvals: a token holder acts as you.
- **No model decides a match.** melo's explanation on an approval card is
  written for your own tasks and chores only, from the task's title and
  goal, the step and the review's reasons as data, never from tool output,
  and nothing reads it back; the step, the buttons and what an answer
  allows come from code. Auto's blocks are never lifted by an earlier
  approval: Auto blocks only steps that are not routine, and those are never
  kept.
- **The checks are guardrails, not a sandbox:**
  - a step runs only the tools it offered;
  - the arguments of built-in tools must have the declared types;
  - six destructive shell command patterns are blocked, which is easy to get
    around;
  - tool output is cut to 50,000 characters, and memory writes are limited;
  - a `shell_exec` command is killed with its process group when it times out
    or the run is cancelled.
- **To limit what a task can do,** use `--builtins none` or a subset of the
  tools, `--mcp-only`, a container or virtual machine, or a separate user
  account.
- **Pages in melo's background Chrome do not reach this Mac.** Chrome's
  Local Network Access check stays on and melo grants no permission, so a
  page's requests (its frames' included) cannot reach this computer or its
  network (melo's gateway, a model server, a dev server). The one exception
  is a sign-in app melo lists in code (today WeChat's quick login): melo
  sends those requests itself, only from that app's own sign-in page (by the
  `Origin` Chrome sets), only to its fixed ports at `127.0.0.1` with the
  certificate checked for its name, never to melo's own ports, with no
  cookie, and follows no redirect; only content, caching and CORS headers
  come back. Chrome does not check where a page sends its own tab: melo
  stops that before it is sent when the address is written as local (a user
  part, `x@127.0.0.1`, included) or is a name of one word, but a name that
  only resolves to this Mac (DNS rebinding, `/etc/hosts`) is checked by
  neither, and melo only blanks that tab after it lands and reads nothing
  from it. Safari's automation windows have no such check. The developer
  setting `[tools] web_fetch_allow_local`, empty by default, lets melo's own
  navigation and the pages already on a listed `127.0.0.1:PORT` server reach
  the listed servers (never melo's own ports, as an address a page or melo
  names); a request of any other page to them stays blocked, judged by the
  `Origin` or `Referer` Chrome sets. Limits: a page on a listed server is
  trusted like the developer's own app, because Chrome's own check does not
  stop its WebSockets or its requests to names that lead to this computer,
  and melo's rule sees only addresses written as such, so list only servers
  whose pages you wrote; a page's own requests are judged by its origin only,
  so a server both written by the person and listed is treated as listed;
  while melo's navigation to a listed site loads, any GET of the main frame
  to that site passes, one the previous page starts or one a redirect chain
  leads back to included; and Safari tells only where a page landed, so a
  public page passed through inside one step is not seen.
- **The hosts of one site share a kept answer.** What the person keeps for
  opening a site, or for clicking and choosing on it, covers the site's other
  hosts where that answer holds, because a sign-in moves between them. One site
  is the same scheme and the same registrable domain, read offline from a
  public suffix list bundled with melo, its private section included, so each
  account of a listed hosting service stays a site of its own. So a kept answer
  trusts everyone who runs a host under that domain. The list is as new as the
  installed package: a hosting service added since is read as one site. An
  address by number, a local or one-word name, a host that is itself a public
  suffix, an unlisted suffix and a port that is not the scheme's own stay the
  exact origin, and `https` never covers `http` or the reverse. Nothing else
  widened: an Allow once, typing, a form sent, a file attached, a click at a
  point, a page's own question, a take-over, a page's request for another app
  and high-impact steps stay as they were; a link followed on a page opened by
  an Allow once asks on another host; what Auto kept for a chat stays the host
  it read, and Auto keeps no click. A password given in a message goes where it
  always went, which is not one exact origin: a password box on an `https` page
  of a host the person's own words name, `www.` aside, whatever the port. One
  site adds nothing to that: another host of the site gets no password. Every
  host still passes the check for a public address.
- **A refusal covers the whole site.** The person's refusal of a site takes
  back what that chat or task keeps for it, on either scheme, what a task holds
  from its chat, and a main chat's answer that would cover the site where they
  refused, so a later turn asks again, in the main chat too. A browser question
  refused takes the approval of every tab on the site, so a tab asks again in a
  later turn as well. A refusal of any kind then stands for the rest of the
  run, and in a task continued later, over all that would run with no answer
  from the person: a kept answer, a site the request names, Auto, a page
  approved before it, which is left for an empty one, and the rest of a page
  read before it.
  Only the person's newer answer stands over it, for what they allowed: that
  question, or the whole site on that scheme when they keep the answer. Stop
  while a refusal is being saved ends only the wait: the answers and the tabs
  go all the same. Four limits remain. An automatic block leaves kept answers alone, since it can come
  from nobody being there to ask. The terminal's own chat reads the main chat's
  answers from a file it does not write: a refusal there stands for the run, and
  the answer holds again from the next turn. A refusal outside the browser does
  not reach every tab: after `web_fetch`'s question is refused, a tab of the
  site not touched in that run can be read in a later turn, and a refused shell
  or Python step is one refused step, not a refused site, so it takes nothing
  back and stops no tab. A take-back the sessions file refuses still stands for
  the run and in that process, the person is told, and it is tried again with
  the next grant change and at close; if it never lands, the answer holds again
  after a restart.
- **melo's Chrome keeps its sign-ins on disk.** Its profile
  (`~/.cache/localmelo/browser/chrome`, mode 0700) restores its last session
  each time Chrome starts, so that a page can move to a window for the person
  and back: cookies a site means for one session stay on disk too, like
  lasting ones. Chrome does not use the Keychain there, so they are protected
  by the folder's permissions and FileVault only. To clear them, quit melo and
  delete that folder.
- **The person's window is a full Chrome on that profile.** It offers to save
  no password, address or card, so what they type is not kept at rest there.
  What else they change in it (an extension, a setting) stays for melo's later
  background browsing, and melo's other open pages, from every chat, come back
  in it as minimized windows.
- **A saved chore's run browses like a task you asked for.** A chore made from
  your request (not a one-time reminder) has the browser, in the chat that
  saved it, so it can use any page a Chrome sign-in on that profile opens, with
  nobody at its side. Nothing is wider than a task you started: its steps ask
  in that chat's cards on every run, as before, use and record no earlier
  approval or chat grant, Auto blocks what it would ask, a post, send or
  purchase is reviewed from your saved words as they stand, and, as the
  chore's schedule reaches that review from code, what a repeating chore posts
  in your name asks each time, whatever the request's words say. A chore saved
  from your own request is a task you started, so its run works through a
  picture check on a page itself, **with nobody at its side**, every time it
  fires; so a daily run can solve a check while you are away. The limits of a
  task you started hold: a model that reads pictures, a frame proved a check's
  by what Chrome says (a page's words never turn it on), melo's own model only
  and no outside service, 3 tries and then the take-over card, a step in the
  record that says plainly what it pressed, and nothing that hides that a
  program is pressing. A reminder has no browser at all. Each chore keeps
  one work folder, `chore-<id>` inside the gateway workspace, which every run
  shares and deleting the chore removes (a link
  there is unlinked, never followed). What is in that folder is plain data, an
  untrusted file like any a page gave: it is never read as the project's
  instructions, whatever it is named. One guard per chore decides whether a
  run may start, for the timer, Run now, a message that saves a chore and runs
  it at once, and delete: a run that has not ended (paused ones too) makes the
  next slot skip and refuses Run now, a chore deleted a moment ago starts
  nothing and makes no folder, so two runs never share that folder at once.
  A run that has waited on you for over an hour does not hold the chore for
  good: the next slot ends it with a plain line in the chat, never answering
  what it waited on, and starts its own run; a run whose page you hold is
  never ended this way. A run already going when you delete the chore is not
  cancelled: it ends in place, the chat says where its folder is, and melo
  removes the folder after it ends, from what it kept when it deleted that
  chore; if melo restarts first, or the removal fails, the folder stays where
  the chat was told it is. melo never removes a folder because of its name: it
  removes only the folder of a chore it deleted. A
  password you gave in the message that saved the chore is typed by the run
  that message started only; a later run does not get it, so it relies on the
  sign-in melo's browser kept, or asks you. What a run leaves in its folder,
  and a page it keeps open, stay until then. The saved text is what a run is
  told to do whenever it fires: edit or delete the chore when it should stop.
- **Gateway:** The gateway accepts only loopback binds and requires a bearer token or browser session cookie for protected APIs. A token holder can act with your permissions. The private token file does not protect against malicious processes running as the same OS user, browser extensions, or compromised same-origin code. Browser cookies are not isolated by port: a port-specific name is not a security boundary against another service on the same host. Do not expose the gateway through a proxy.
  See [gateway authentication](docs/guides/gateway.md#keep-the-gateway-private)
  for sign-in and token rotation. Host, Origin and fetch-metadata checks remain
  important browser defenses; authentication is not a sandbox.
- **MCP servers** run as local processes, with a minimal environment plus
  the `env` you give them.
- **Secrets.** The config file stores the *name* of the API-key variable,
  not the key. `history.db` stores tasks, the model's text, tool calls and
  their results as plain text. With persistence on, it also holds the
  model's reasoning for every call, in its `llm_calls` table. There,
  one-time codes, key shapes and bearer tokens become `[redacted]`, and
  reasoning older than 30 days (`LOCALMELO_THINKING_RETENTION_DAYS`) is
  cleared, while tasks and steps keep the text as typed. A new memory
  directory is created with mode 0700, the default one is set to 0700 when
  it is open to group or others, and a new `history.db` is created with
  mode 0600. `procedures.db`, also mode 0600, keeps each task's normalized
  query and its tool calls' labels (file and program names), with key
  shapes, bearer tokens, one-time codes, URL passwords and the values of
  `NAME=value` secrets masked; see [Privacy](docs/memory.md#privacy).
- **Sessions.** The session store ([Sessions](docs/sessions.md)), which the
  interactive UI uses for every turn in `~/.cache/localmelo/cli/sessions.db`
  (its messages also go to `cli_history`), keeps transcripts and tool
  output in `sessions.db` as plain text, its window checkpoints included, in
  `0600`
  files inside `0700` directories. The stored answers and checkpoints pass
  only the minimal redaction of events: well-known key shapes, bearer tokens
  and one-time codes; anything else stays as written. A withdrawn message
  keeps only the SHA-256 digest of its text. A browser's drafts, a queued
  message and its outbox of messages not yet confirmed stay in the browser.

[What the checks do](docs/tools.md#what-the-checks-do) describes the checks
in more detail.

## Scope

In scope:

- getting around the per-step tool allowlist, the shell command blocklist or
  the filtering of MCP server environments;
- problems in the gateway or its web workspace;
- secrets that leak into history, logs or output.

Out of scope:

- a model that misuses a tool you enabled, within the permissions described
  above.
