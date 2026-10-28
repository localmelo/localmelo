# Tools and MCP

This page covers the tools the model can call: the built-in tools, tools
from MCP servers, which tools each step offers, and what the checks do.

## Built-in tools

| Tool | What it does | Limits |
|---|---|---|
| `shell_exec` | Runs a shell command with `/dev/null` as standard input and returns its output. | Killed with every process it started after 30 seconds. Only `interactive` mode offers it by default: a direct run and a Python caller need an explicit `builtins` set that names it; a gateway or console session does not offer it by default and refuses it even when named. |
| `python_exec` | Runs Python code in a child process and returns what it prints. | Killed with every process it started after 60 seconds. `sys.exit()` ends only the child. |
| `file_read` | Reads a text file. | Any file your account can read. |
| `file_write` | Writes text to a file, creating or replacing it. | Any file your account can write. |

A `file_read` or `file_write` call may take up to 60 seconds, and an MCP tool
call up to its server's `timeout`. Output longer than 50,000 characters is
cut.

The `python_exec` code reaches the child on its standard input, a pipe,
never in its `-c` argument: `ps` does not show the code, and a call has no
`ARG_MAX` limit (about 1 MiB of argv and environment on macOS). The child
runs a fixed wrapper; once that has read the code it points fd 0 at
`/dev/null`, so `input()` gets EOF. The child may have at most 1,024 open
files and one hour of CPU time (the POSIX `RLIMIT_NOFILE` and `RLIMIT_CPU`
limits). The call's time limit normally ends it first; the CPU limit only
stops code that outlives melo or that spends an hour of CPU. An
address-space limit is not set: macOS refuses a finite one.

`shell_exec` and `python_exec` start their child with a minimal
environment: `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `LANG`,
every `LC_*` variable and `TMPDIR`, then whatever
`AgentOptions(tool_env={...})` adds, and nothing else. An API key in melo's
own environment therefore reaches no tool; `VIRTUAL_ENV`, `PYTHONPATH`,
`SSH_AUTH_SOCK` and the proxy variables pass only through `tool_env` (see
[The Python API](python-api.md#work-in-a-directory)).

This is how the built-in tools work in a direct run without `--workspace`
and in the gateway. A direct run with `--workspace DIR` and a Python agent
with `workspace_root` keep the file tools inside a directory and start
`shell_exec` and `python_exec` there (see
[With a workspace root](#with-a-workspace-root)). The interactive UI (bare
`melo`) runs with its workspace as the root too, and asks before each
`shell_exec`, `python_exec` and `file_write` call, unless `--yolo` (see
[Permission tiers](#permission-tiers)); direct mode has no approval policy
and does not offer `shell_exec`, with or without a root. The interactive UI loads the MCP
servers of `--mcp-config`, else of the top-level `mcp_config` key; a server
that fails to start turns MCP off for that session (`/mcp` says why). A
Python run with an approval policy asks its own policy about an MCP tool by
the same tiers.

## Choose the built-in tools

```bash
melo --builtins none "What is 6*7?"
melo --builtins file_read,shell_exec "How many lines does notes.txt have?"
```

- `--builtins` takes `all` (every built-in, `shell_exec` included), `none`,
  or a comma-separated list of tool names. An unknown name is a usage error.
  Without the flag, the mode decides: `file_read`, `file_write` and
  `python_exec` in a direct run, the gateway and the Python API; all four in
  the interactive UI. `--builtins all` is the way to keep the old direct-run
  set, and `--builtins shell_exec` names only the shell.
- `config.builtins` and `config.tools` in the JSON line show what was
  registered.
- In Python, use `AgentOptions(builtins=...)`; see
  [AgentOptions](python-api.md#agentoptions).

## Add tools from MCP servers

localmelo can use the tools of [Model Context Protocol](https://modelcontextprotocol.io)
servers that run as local processes (stdio servers). List the servers in a
JSON file, in the `mcpServers` format that other MCP clients use:

```json
{
  "mcpServers": {
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time"]
    }
  }
}
```

This example uses the MCP project's reference time server, which `uvx`
downloads on first use. Save the file, for example as `mcp.json`, and pass it
with `--mcp-config`:

```bash
melo --mcp-config mcp.json "What time is it in Tokyo?"
```

The server's tools are registered as `mcp__time__<tool>`. To see them, add
`--output-format json` and read `config.tools`. To use only the MCP tools,
add `--mcp-only`.

A server does not inherit melo's environment: it gets `PATH`, `HOME`,
`USER`, `LOGNAME`, `SHELL`, `TERM`, `LANG`, every `LC_*` variable and
`TMPDIR`, plus the `env` of its config entry, and nothing else. An API key
in melo's environment reaches no server; a server that needs one must be
given the literal value in its `env`.

### MCP config file

Every key of a server entry:

```json
{
  "mcpServers": {
    "notes": {
      "command": "python3",
      "args": ["-m", "my_notes_server"],
      "env": {"NOTES_DIR": "/path/to/notes"},
      "cwd": "/path/to/project",
      "timeout": 120,
      "startup_timeout": 30,
      "include_tools": ["search", "read_note"],
      "require_approval": false,
      "disabled": false
    }
  }
}
```

| Key | Default | Description |
|---|---|---|
| `command` | required | The program that starts the server. |
| `args` | `[]` | Its arguments. |
| `env` | `{}` | Environment variables for the server. Values are used as written: `${VAR}` is not expanded. |
| `cwd` | the working directory of `melo` | The directory the server starts in. |
| `timeout` | `120` | Seconds each tool call may take. |
| `startup_timeout` | `30` | Seconds to start the server and list its tools. |
| `include_tools` | all tools | Register only these tools, by the names the server uses. |
| `require_approval` | `false` | `true`: each call of the server's tools asks the run's approval policy first, even a read-only one. A list of tool names, spelled as in `include_tools`: only calls of those tools ask. It is an extra mark on top of the tool's tier (see [Permission tiers](#permission-tiers)) and has an effect only in a run with a policy. |
| `disabled` | `false` | `true` skips the server. |

- `servers` is accepted in place of `mcpServers`. Other keys are ignored,
  except `type` (or `transport`): when present, it must be `stdio`.
- `require_approval` must be `true`, `false` or a list of strings; any other
  value is a usage error. A listed name that the server does not offer is
  logged as a warning.
- Only stdio servers work. Any other `type` is a usage error, such as
  `only stdio servers are supported (got type 'sse')`. So is an entry with a
  `url` and no `command`:
  `'command' must be a non-empty string (HTTP/SSE servers are not supported)`.
- A server does not inherit your environment. It gets `HOME`, `LOGNAME`,
  `PATH`, `SHELL`, `TERM`, `USER`, `LANG`, `LC_ALL`, `LC_CTYPE` and
  `TMPDIR`, plus its `env`.
- Tools are named `mcp__<server>__<tool>`. Characters other than letters,
  digits, `_` and `-` become `_`. Names longer than 64 characters are
  shortened to 64 and end with a short hash; `config.tools` in the JSON line
  shows the names used.
- A tool's `annotations` object is kept as the server gives it, and decides
  the tool's permission tier when it is not a built-in (see
  [Permission tiers](#permission-tiers)). A malformed value is ignored: the
  tool still lists.
- The servers start when the run starts. A server that cannot start ends the
  run with status `setup_error`, for example
  `MCP server 'notes' exited (code 1)`.
- MCP servers work in direct mode and in the
  [Python API](python-api.md#tool-providers-and-mcp). Gateway sessions have
  no MCP tools.

## How tools are offered

Each planning step sends the model a list of tools, and the step can run only
those tools. `--tool-exposure` chooses the list:

- **`retrieve`** (the default): every registered built-in tool, sorted by
  name, then the extra tools for the step, also sorted by name. The extra
  tools are chosen among the other tools, such as MCP tools:
  - the tools the model has already called in this conversation, and
  - the 3 best matches for the task, by a keyword search (Okapi BM25 with
    IDF) over each tool's name, description and tags.

  Only the model's tool calls are hints: a tool name in the task, in a tool
  result or in recalled memory adds nothing.
- **`all`**: every registered tool, sorted by name, on every step. The list
  never changes, which helps servers that cache prompt prefixes.

A call to a tool that the step did not offer is refused with
`Tool '<name>' was not offered in this step.` and does not run. The refusal
stays in the conversation, so a later step can offer that tool.

## What the checks do

Before a tool runs, localmelo checks the call:

- The step must have offered the tool.
- The arguments must be a JSON object. For built-in tools, each argument must
  also have the type the tool declares; a call such as `shell_exec` with a
  list for `command` is refused with
  `Arguments for shell_exec: 'command' must be of type string, not array`.
- The tool's tier and the mode decide, never the command's spelling. A mode
  with an approver (the interactive UI) offers `shell_exec` and asks about
  each `exec` call (see [Permission tiers](#permission-tiers)); under
  `--yolo` nothing asks and the call runs. The modes with no approver
  (direct, the gateway, the console and the Python API's default `python`
  mode) leave `shell_exec` out of their default set, so the model never sees
  it, and the checker refuses an `exec` call that was registered anyway,
  unless a `direct` or `python` caller named it in `builtins`. A named
  `shell_exec` is the caller's opt-in there and runs; `gateway` still
  refuses it, named or not.
- With a workspace root, the path of `file_read` and `file_write` must be
  inside it (see [With a workspace root](#with-a-workspace-root)).
- With an approval policy, a tool whose tier is not `read`, or that is
  explicitly marked, waits for the policy's decision, and with a workspace
  root, a file path is then checked again (see
  [Permission tiers](#permission-tiers) and
  [With an approval policy](#with-an-approval-policy)).

After a tool runs:

- Output longer than 50,000 characters is cut.
- What goes into memory is checked too, and each write is limited to 50,000
  characters.
- A `shell_exec` command that times out, or whose run is cancelled, is killed
  together with every process it started.

A refused call does not run. The model sees the refusal as the tool's error
and can try something else.

The checks are guardrails, not a sandbox:

- Tools run with your account's permissions.
- Without a workspace root, as in a direct run without `--workspace` and in
  the gateway, file tools reach any file you can. `python_exec` runs in a
  child process of melo, not in a sandbox: it reaches any file you can too.
- With a workspace root, `shell_exec` and `python_exec` still reach any
  file you can: only the file tools are kept inside the root.
- Approvals apply only to a run with an approval policy, and in the
  interactive UI (unless `--yolo`). Direct mode, `melo ""` and the gateway
  have none.
- A mode with no approver offers no shell at all, so there is no command
  to get around: direct, the gateway, the console and the Python API's
  default `python` mode do not register `shell_exec`. The interactive UI
  keeps it behind its panel, and `--yolo` means no panel.

To limit what a task can do, use `--builtins none` or a subset of tools,
`--mcp-only`, a container or virtual machine, or a separate user account.
See the [security policy](https://github.com/localmelo/localmelo/blob/main/SECURITY.md).

### Permission tiers

Every tool has one of four tiers. A tier says what a call can reach; it is
not a denial. For a built-in or provided tool in a run with an approval
policy, `read` is the only tier that runs without asking.

| Tier | Meaning | Asks with a policy |
|---|---|---|
| `read` | Reads local data; no side effect outside the process. | No, unless the tool is explicitly marked. |
| `write` | Creates, changes or deletes local files or state. | Yes. |
| `exec` | Runs a command, code or another program with your privileges. | Yes. |
| `network` | Sends data to, or acts on, something outside this machine: an HTTP API, a chat, a payment, a GUI. | Yes. |

The built-ins are fixed by name: `file_read` is `read`, `file_write` is
`write`, and `shell_exec` and `python_exec` are `exec`. The table wins over a
tool's own claims, so a provided or custom tool that reuses a built-in's name
inherits that tier.

The mode decides which built-ins are offered at all. The interactive UI
runs in `interactive` mode and gets all four; it asks before an `exec` call
unless it runs with `--yolo`. A direct run, a gateway or console session and
the Python API's default `python` mode get `file_read`, `file_write` and
`python_exec`: `shell_exec` is not registered, so the model never sees it,
and the checker refuses one that was registered anyway unless the caller
named it in `builtins`. A named `shell_exec` (`--builtins all`,
`--builtins shell_exec`, `AgentOptions(builtins={"shell_exec"})`) is the
caller's opt-in in `direct` and `python` mode and runs; `gateway` refuses it
even when named. `AgentOptions(builtins=…)` or `--builtins` always wins over
the mode's default set.

Every other tool's tier comes from the MCP `annotations` object that the
server lists with it, read cautiously:

- `openWorldHint: true` makes it `network`, whether or not it also claims to
  be read-only.
- `readOnlyHint: true` makes it `read`, unless `destructiveHint: true`
  contradicts it; then the cautious reading wins and it is `write`.
- Everything else is `write`: no annotations, an annotation that is not an
  object, a hint that is not a boolean, and the `idempotentHint` and `title`
  keys. A tool is not read-only unless it says so.

Annotations are claims by the server, not proof: a server can say
`readOnlyHint: true` for a destructive tool. localmelo trusts the hint only
after you chose to start that server. A tool registered by Python keeps its
explicit `requires_approval`: the tier is only a label for it, and a
`register` call without the mark does not ask.

An MCP server's `require_approval` is an extra mark on top of the tier:
`true` or a list of tool names makes even a `read` tool ask. `false` means no
extra mark; it cannot stop a `write`, `exec` or `network` tool from asking.

### With a workspace root

A direct run with `--workspace DIR` and an agent built in Python with
`AgentOptions(workspace_root=DIR)` change the built-in tools (see
[Work in a directory](python-api.md#work-in-a-directory)). A direct run
without the flag and the gateway never set a workspace root.

- **The fence.** `file_read` and `file_write` resolve a relative path against
  the root and reach only files whose real path is inside it, after `..` and
  symlinks are resolved. A path outside is refused with
  `Path '<path>' is outside allowed workspace root`, and an empty path, or
  one with a NUL character, with
  `Argument 'path' for <tool> must be a non-empty string without NUL characters.`
  The tool gets the resolved path. `~` is not expanded, a path spelled in
  another letter case than the root is refused, and hard links are not
  detected.
- **Opening files.** The file tools open the resolved path without following
  symlinks, so a symlink that takes the file's place after the check is
  refused. They refuse anything but a regular file, such as a FIFO, with
  `Not a regular file: <path>`. `file_read` reads at most 200,000 bytes. A
  file over 200,000 bytes, or whose text is longer than 50,000 characters,
  is cut to 50,000 characters, which end with
  `... [truncated: the file has <size> bytes]`.
- **Commands.** `shell_exec` and `python_exec` always run in a child process
  with a process group of its own, which a timeout or a cancellation kills
  with every process it started; a root only sets the directory they start
  in. A shell command's standard input is `/dev/null`; `python_exec` gets its
  code on a pipe (never in argv) and then points fd 0 at `/dev/null`, so
  `input()` gets EOF and neither reads the terminal melo runs in. These two
  tools are not fenced: they can reach any path.
- **Environment.** The child of `shell_exec` and `python_exec` does not
  inherit melo's environment: it gets `PATH`, `HOME`, `USER`, `LOGNAME`,
  `SHELL`, `TERM`, `LANG`, every `LC_*` variable and `TMPDIR`, plus
  `AgentOptions(tool_env={...})`. Direct mode and the interactive UI pass
  nothing else; the Python API can pass any variable in `tool_env`. An MCP
  server gets the same base environment plus its `env`.
- **Exit status.** A `shell_exec` command that exits with a non-zero status
  starts its text with the line `[exit code <n>]` (`-<signal>` for a command
  that a signal ended), and the call still succeeds. The line comes first,
  so it stays when the output is cut. Python code that exits with a non-zero
  status fails with `Python exited with code <n>`, followed by the last
  4,000 characters of its output.
- **Time limits.** `AgentOptions(tool_timeout_s=...)` sets the time limit of
  `shell_exec` and `python_exec`, in seconds. Without it they keep 30
  seconds for a shell command and 60 seconds for Python. A shell command
  that runs too long fails with `Command timed out after <n> s`; a Python
  call whose child is still running ends the call as a timeout with
  `Execution timed out after <n>s`, and kills it.

### With an approval policy

A run started in Python with `approval=policy` asks the policy before each
call of a built-in or provided tool whose tier is not `read` (see
[Permission tiers](#permission-tiers)): the built-ins it offers
(`python_exec` and `file_write` in the default `python` mode; `shell_exec`
too when the caller names it),
a provided or MCP tool that is `write`, `exec` or `network`, including an MCP
tool with no annotations, which is `write`. The interactive UI always has
such a policy, unless `--yolo`; direct mode and the gateway's `/v1` and
workspace sessions have none. A gateway session therefore does not offer
`shell_exec` at all: there is no one to ask. The console has a policy, but
its approvals arrive over the same unauthenticated loopback API, so it is
gateway mode too. A tool is also asked about when it is explicitly marked: a
provided tool with `requires_approval=True`, a custom tool whose `register`
call sets it, or the tools of an MCP server that sets `require_approval` (see
[Approve tool calls](python-api.md#approve-tool-calls)). A custom tool's
tier is only a label: it asks when its `register` call sets the mark.

- **After the other checks.** The policy is asked only about a call that
  every check above lets through, so it sees exactly what will run, the
  resolved file path included. A call whose arguments do not match the
  tool's parameters is refused with
  `Arguments for <tool> do not match its parameters: <message>`, and the
  policy is not asked.
- **Outside the time limit.** The tool's time limit starts once the policy
  allows the call.
- **A denial** refuses the call. The model reads why, for example
  `The user denied this call. Do not run it again or reach the same result another way unless the user asks.`
- **It fails closed.** A policy that raises an error, or returns no decision,
  refuses the call with
  `Approval failed, so the call did not run: <ExceptionType>: <message>`.
- **The re-check.** Once the policy allows a file tool's call under a
  workspace root, the path is resolved again. A path that now resolves
  elsewhere, as when a symlink took the file's place while the policy
  decided, is refused with
  `Path '<path>' changed while waiting for approval; the call did not run.`
- Without a policy, these tools run without asking.
