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
  control. The terminal UI, the web console, the app and other direct runs
  work this way.
- **Worker runs write only in their workspace (macOS).** In a worker run
  (`melo worker BRIEF`, `--brief`) with a workspace, and in a saved task,
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
