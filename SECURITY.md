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
  restriction is active.
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
  default. Anyone who can reach it can run tasks with your tools, and the web
  UI makes the gateway fetch any URL that a client sends. Never bind it to
  other interfaces.
- **MCP servers** run as local processes, with a minimal environment plus
  the `env` you give them.
- **Secrets.** The config file stores the *name* of the API-key variable,
  not the key. `history.db` stores tasks, the model's text, tool calls and
  their results as plain text.

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
