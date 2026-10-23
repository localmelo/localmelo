# Tools and MCP

This page covers the tools the model can call: the built-in tools, tools
from MCP servers, which tools each step offers, and what the checks do.

## Built-in tools

| Tool | What it does | Limits |
|---|---|---|
| `shell_exec` | Runs a shell command and returns its output. | Killed with every process it started after 30 seconds. Six destructive command patterns are refused. |
| `python_exec` | Runs Python code inside the localmelo process and returns what it prints. | No time limit: it cannot be interrupted. |
| `file_read` | Reads a text file. | Any file your account can read. |
| `file_write` | Writes text to a file, creating or replacing it. | Any file your account can write. |

A `file_read` or `file_write` call may take up to 60 seconds, and an MCP tool
call up to its server's `timeout`. Output longer than 50,000 characters is
cut.

## Choose the built-in tools

```bash
melo --builtins none "What is 6*7?"
melo --builtins file_read,shell_exec "How many lines does notes.txt have?"
```

- `--builtins` takes `all` (the default), `none`, or a comma-separated list
  of tool names. An unknown name is a usage error.
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
| `disabled` | `false` | `true` skips the server. |

- `servers` is accepted in place of `mcpServers`. Other keys are ignored,
  except `type` (or `transport`): when present, it must be `stdio`.
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
  - the tools that the conversation has already named or called, and
  - the 3 best matches for the task, by a keyword search (Okapi BM25 with
    IDF) over each tool's name, description and tags.
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
- `shell_exec` refuses commands that match one of six patterns: `rm -rf /`,
  `mkfs`, `dd if=`, a fork bomb, `shutdown` and `reboot`.

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
- File tools are not limited to a directory: they reach any file you can.
- The command blocklist is easy to get around.
- `python_exec` runs code inside the localmelo process.

To limit what a task can do, use `--builtins none` or a subset of tools,
`--mcp-only`, a container or virtual machine, or a separate user account.
See the [security policy](https://github.com/localmelo/localmelo/blob/main/SECURITY.md).
