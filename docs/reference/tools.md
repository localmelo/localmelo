# Tools and MCP

This page covers the tools the model can call: the built-in tools, tools
from MCP servers, which tools each step offers, and what the checks do.

## Built-in tools

| Tool | What it does | Limits |
|---|---|---|
| `shell_exec` | Runs a shell command with `/dev/null` as standard input and returns its output. | Killed with every process it started after 30 seconds. Only `interactive` mode offers it by default: a direct run and a Python caller need an explicit `builtins` set that names it; a gateway or console session does not offer it by default and refuses it even when named. |
| `python_exec` | Runs Python code in a child process and returns what it prints. | Killed with every process it started after 60 seconds. `sys.exit()` ends only the child. |
| `file_read` | Reads a text file. | Any file your account can read. |
| `list_files` | Lists a folder: its folders first, each ending in `/`, then its files with their sizes; a link ends in `@`. Takes `path` (default: the workspace folder) and `depth` (1 to 3, default 1). Never reads a file, never follows a link and skips `.git`. | Shows 200 entries, then `... and N more`. Read-only, so it never asks for approval. The interactive UI and gateway and console sessions offer it by default; a direct run and the Python API's default mode do not (`--builtins list_files` or `all` names it). |
| `file_write` | Writes text to a file, creating or replacing it. | Any file your account can write. |
| `file_edit` | Replaces one exact piece of a file (`old`, which must occur once) with `new`. | Files up to 1 MB. Asks as `file_write` does; the approval card shows the change as a diff. Only a task's worker offers it (name it in `builtins` elsewhere). |
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

A provider can mark its tools as a group (`ProvidedTool.group`); with `retrieve`, every tool of a group is offered on every step, like the built-ins, after them. The eight browser tools are such a group, present only while `[browser] driver` is on (see [Browsing with melo](../guides/browsing.md)).

A call to a tool that the step did not offer is refused with
`Tool '<name>' was not offered in this step.` and does not run. The refusal
stays in the conversation, so a later step can offer that tool.
