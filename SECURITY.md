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
  restriction is active. This holds in direct mode and the gateway, and in
  the Python API unless an agent sets the options below.
- **Approvals apply only with a policy.** In the Python API, an approval
  policy passed to `Agent.run_task` decides whether each call of
  `shell_exec`, `python_exec`, `file_write` and other tools marked as
  requiring approval may run. It is asked after every other check, and a
  policy that fails denies the call. Without a policy, as in direct mode and
  the gateway, these tools run without asking.
- **The fence applies only with a workspace root.**
  `AgentOptions(workspace_root=DIR)` keeps `file_read` and `file_write`
  inside that directory: paths are resolved before the check, a symlink
  that points out is refused, and so is anything but a regular file.
  Without a workspace root there is no fence.
- **`shell_exec` and `python_exec` are not fenced.** With a workspace root
  they start in it, and `python_exec` runs in a child process, but both can
  reach any path you can. An approval policy is their only control.
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
- **The gateway has no authentication.** It listens on `127.0.0.1:8401` by
  default, where any program on your computer can reach it and run tasks
  with your tools, and the web UI makes the gateway fetch any URL that a
  client sends.
  - On `localhost` or a loopback IP address, such as `127.0.0.1` or `::1`,
    the gateway refuses requests whose `Host` header is not `127.0.0.1`,
    `localhost` or `[::1]` with its port (HTTP 421), which stops web pages
    that reach it through DNS rebinding. It also refuses requests from web
    pages at another origin (HTTP 403): `POST`, `PUT`, `PATCH`, `DELETE` and
    `OPTIONS` requests whose `Origin` is not the gateway's, and requests
    under `/v1/` that the browser marks as cross-site or same-site. A link
    from another site still opens the web UI, but no page can show it in a
    frame.
  - On any other host, such as `0.0.0.0` or a host name other than
    `localhost`, none of this applies: anything that can reach the port, web
    pages included, can run tasks. Never bind it to other interfaces.
- **MCP servers** run as local processes, with a minimal environment plus
  the `env` you give them.
- **Secrets.** The config file stores the *name* of the API-key variable,
  not the key. `history.db` stores tasks, the model's text, tool calls and
  their results as plain text.
- **Sessions.** The session store of the Python API
  ([Sessions](docs/sessions.md)) keeps transcripts and tool output in
  `sessions.db` as plain text, its window checkpoints included, in `0600`
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
- problems in the gateway or its web UI;
- secrets that leak into history, logs or output.

Out of scope:

- a model that misuses a tool you enabled, within the permissions described
  above.
