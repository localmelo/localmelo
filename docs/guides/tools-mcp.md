# Guide: Tools MCP

This guide shows how to choose the built-in tools, add tools from MCP servers, and understand the checks and permission tiers that guard them. Read it before you let a run read files or run commands; the tool list and options are in the [tools reference](../reference/tools.md).

## Choose the built-in tools

```bash
melo --builtins none "What is 6*7?"
melo --builtins file_read,shell_exec "How many lines does notes.txt have?"
```

Without `--builtins`, the mode decides:

| Where you run | Built-ins you get |
|---|---|
| A direct run, the Python API | `file_read`, `file_write`, `python_exec` |
| The gateway | Those three and `list_files` |
| The interactive UI (`interactive` mode) | All five, `shell_exec` included |

- `--builtins` takes `all` (every built-in, `shell_exec` included), `none` or a comma-separated list of names; an unknown name is a usage error.
- `--builtins shell_exec` names only the shell.
- `config.builtins` and `config.tools` in the JSON line show what was registered.
- In Python, use `AgentOptions(builtins=...)`; see [AgentOptions](../reference/python-api.md#agentoptions).

## Add tools from MCP servers

localmelo can use the tools of [Model Context Protocol](https://modelcontextprotocol.io) servers that run as local processes (stdio servers). List the servers in a JSON file, in the `mcpServers` format that other MCP clients use:

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

This example uses the MCP project's reference time server, which `uvx` downloads on first use. Save the file, for example as `mcp.json`, and pass it with `--mcp`:

```bash
melo --mcp mcp.json "What time is it in Tokyo?"
```

- The server's tools are registered as `mcp__time__<tool>`; add `--output json` and read `config.tools` to see them.
- `--mcp-only` uses only the MCP tools.
- A server does not inherit melo's environment: it gets `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `LANG`, every `LC_*` variable and `TMPDIR`, plus the `env` of its entry.
- An API key in melo's environment reaches no server; a server that needs one gets the literal value in its `env`.

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

What else the file and the servers do:

| Topic | What happens |
|---|---|
| File keys | `servers` is accepted in place of `mcpServers`. Other keys are ignored, except `type` (or `transport`): when present, it must be `stdio`. |
| `require_approval` | Must be `true`, `false` or a list of strings, else a usage error. A listed name the server does not offer is logged as a warning. |
| Transport | Only stdio servers work. Another `type` is a usage error: `only stdio servers are supported (got type 'sse')`. So is an entry with a `url` and no `command`: `'command' must be a non-empty string (HTTP/SSE servers are not supported)`. |
| Tool names | `mcp__<server>__<tool>`. Characters other than letters, digits, `_` and `-` become `_`. A name over 64 characters is cut to 64 and ends with a short hash; `config.tools` shows the names used. |
| `annotations` | Kept as the server gives them; they decide the tier of a tool that is not a built-in (see [Permission tiers](#permission-tiers)). A malformed value is ignored and the tool still lists. |
| Start | The servers start with the run. One that cannot start ends it as `setup_error`, for example `MCP server 'notes' exited (code 1)`. |
| Where | Direct mode and the [Python API](../reference/python-api.md#tool-providers-and-mcp). Gateway sessions have no MCP tools. |
| `structuredContent` | Sent as text only when the result has no text and no picture: a screenshot server that repeats the base64 there sends the picture, not its base64. |

### Pictures that tools return

A tool can return pictures, such as an MCP server's screenshot.

- Each picture is checked (at most 8 per result, 20 MiB each; PNG, JPEG, GIF or WebP).
- It is named at the end of the tool's message, for example `[1 image: sha256:3f2a9c0d1b7e image/png]`.
- Whether its pixels reach the model is the setting `[agent] tool_images`:

| Value | What the model gets |
|---|---|
| `auto` (default) | The pixels when the server says the model reads images: oMLX's `GET /v1/models/status` gives the model `model_type` `vlm`, or Ollama's `POST /api/show` lists `vision` in its `capabilities`. localmelo asks once, when a tool first returns a picture. Any other answer, or none (vLLM, SGLang and MLC do not say), counts as `off`. |
| `on` | The pixels, without asking. |
| `off` | Only the line that names each picture, as before this setting. |

Set it for one run with `-X`:

```bash
melo -X agent.tool_images=on --mcp browser.json --mcp-only "Describe the page that is open"
```

With pixels on, the tool's text stays in its `role: "tool"` message. The pictures follow the run of tool results, one `user` message per call, as `image_url` data URLs (Ollama: the `images` field), labelled with the call. The second request of `tests/cli/test_tool_images_e2e.py` ends:

```json
{"role": "tool", "tool_call_id": "call_1", "content": "[1 image: sha256:497790947d46 image/png]"},
{"role": "user", "content": [
  {"type": "text", "text": "Output of tool call call_1 (mcp__fake__screenshot): [1 image: sha256:497790947d46 image/png]"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoA..."}}
]}
```

| Topic | What happens |
|---|---|
| Which pictures keep pixels | The newest `tool_images_keep` (default 3). When one more arrives past `tool_images_keep + tool_images_retire - 1`, the oldest `tool_images_retire` (default 2) lose them together: a request carries 3 to 4 pictures, and the prompt prefix a server caches changes once every two pictures, not on every step. |
| A repeated picture | One with the digest of one already sent is not sent again; its tool message ends with `[same image as an earlier tool result; its pixels are above]`. |
| A wide picture | Wider than 1,280 pixels, it is scaled to 1,280 wide as a JPEG once, when it is kept (with macOS `sips`; elsewhere it is sent as it is). The label then names the JPEG. |
| The prompt cap | Each picture sent counts 4,500 characters toward the 100,000-character prompt cap, and its pixel size toward a context budget ([Context accounting](../reference/configuration.md#context-accounting)); older turns leave the window to make room. |
| What is stored | The copy the model gets lives in memory only: history and the working window keep the line that names the picture (a session still stores the picture for the console). After a session is reopened, no picture is sent until a tool returns a new one. |
| No pictures | A run whose tools return none sends the same requests in every mode. |

## Browsing

On a Mac, melo can read web pages in Safari. It is off by default and read-only; [the browsing guide](browsing.md) has the setup, what melo can and cannot do, and troubleshooting.

- Turn it on with `/browser on` in the terminal chat, or `melo -X browser.driver=safari-automation` for one run.
- In Safari, open Settings > Developer and turn on "Allow remote automation" once.
- The terminal chat and the web console then get eight tools: `browser_open`, `browser_read`, `browser_find`, `browser_follow`, `browser_back`, `browser_tabs`, `browser_wait` and `browser_screenshot`.
- They work in Safari's own automation windows, which start without any of your sign-ins.
- melo asks "Open a website?" before it loads a page, once per site and chat (`auto` mode still asks); it does not click, type, sign in or post. Direct runs, workers and saved tasks never get these tools.

## What the checks do

Before a tool runs, localmelo checks the call:

| Check | What it requires |
|---|---|
| The step | The step must have offered the tool. |
| The arguments | A JSON object. A built-in's arguments also need the declared types: `shell_exec` with a list for `command` is refused with `Arguments for shell_exec: 'command' must be of type string, not array`. |
| The tier and the mode | They decide, never the command's spelling (details below). |
| The workspace root | With one, the path of `file_read` and `file_write` must be inside it (see [With a workspace root](#with-a-workspace-root)). |
| The approval policy | With one, a tool whose tier is not `read`, or that is marked, waits for its decision, and a file path is checked again (see [Permission tiers](#permission-tiers) and [With an approval policy](#with-an-approval-policy)). |

How the mode treats `shell_exec`:

- A mode with an approver (the interactive UI) offers it and asks about each `exec` call; under `--yolo` nothing asks and the call runs.
- The modes with no approver (direct, the gateway, the console and the Python API's default `python` mode) leave it out of their default set, so the model never sees it.
- There the checker refuses an `exec` call that was registered anyway, unless a `direct` or `python` caller named it in `builtins`; naming it is the caller's opt-in and it runs.
- `gateway` refuses it, named or not.

After a tool runs:

- Output longer than 50,000 characters is cut.
- What goes into memory is checked too, and each write is limited to 50,000 characters.
- A `shell_exec` command that times out, or whose run is cancelled, is killed together with every process it started.

A refused call does not run. The model sees the refusal as the tool's error and can try something else.

The checks are guardrails, not a sandbox:

| Where | What a tool can reach |
|---|---|
| Any run | Everything your account can; tools run with your permissions. |
| No workspace root (a direct run without `--workspace`, the gateway) | Any file you can reach, through the file tools and through `python_exec`, a child process of melo. |
| A workspace root | Only the file tools stay inside it; `shell_exec` and `python_exec` still reach any file you can. |
| A worker run (`melo worker`) or a saved task on macOS | Writes are confined to the workspace; reads and the network are not (see [Writes stay in the workspace](../reference/cli.md#confined-writes)). |
| Direct mode, `melo ""` and the legacy gateway API | No approval policy: nothing asks. |
| The interactive UI and console sessions | An approval policy, unless `--yolo`. |

- Direct and Python modes do not offer `shell_exec` by default, but an explicit `--builtins all` or `--builtins shell_exec` (Python: `AgentOptions(builtins=frozenset({"shell_exec"}))`) enables it without an approval policy.
- Gateway and console modes never offer it; the interactive UI keeps it behind its approval panel, unless `--yolo`.
- To limit what a task can do, use `--builtins none` or a subset of tools, `--mcp-only`, a container or virtual machine, or a separate user account; see the [security policy](https://github.com/localmelo/localmelo/blob/main/SECURITY.md).

### Permission tiers

Every tool has one of four tiers: what a call can reach, not a denial. For a built-in or provided tool in a run with an approval policy, `read` is the only tier that runs without asking.

| Tier | Meaning | Asks with a policy |
|---|---|---|
| `read` | Reads local data; no side effect outside the process. | No, unless the tool is explicitly marked. |
| `write` | Creates, changes or deletes local files or state. | Yes. |
| `exec` | Runs a command, code or another program with your privileges. | Yes. |
| `network` | Sends data to, or acts on, something outside this machine: an HTTP API, a chat, a payment, a GUI. | Yes. |

- The built-ins are fixed by name: `file_read` and `list_files` are `read`, `file_write` is `write`, and `shell_exec` and `python_exec` are `exec`.
- The table wins over a tool's own claims, so a provided or custom tool that reuses a built-in's name inherits that tier.
- The mode decides which built-ins are offered at all (see [Choose the built-in tools](#choose-the-built-in-tools)); a gateway or console session gets `list_files` as well as the usual three.
- In every mode but the interactive UI, `shell_exec` is not registered, so a model that calls it gets `shell_exec is not available in this run. Available tools: …`, naming the run's tools (an unknown non-built-in name keeps the `Tool not found in registry` message).
- `AgentOptions(builtins=…)` or `--builtins` always wins over the mode's default set, except in gateway mode.

Every other tool's tier comes from the MCP `annotations` object that the server lists with it, read cautiously:

| Annotation | Tier |
|---|---|
| `openWorldHint: true` | `network`, whether or not it also claims to be read-only. |
| `readOnlyHint: true` | `read`, unless `destructiveHint: true` contradicts it; then the cautious reading wins and it is `write`. |
| Anything else | `write`: no annotations, an annotation that is not an object, a hint that is not a boolean, and the `idempotentHint` and `title` keys. A tool is not read-only unless it says so. |

- Annotations are claims by the server, not proof: a server can say `readOnlyHint: true` for a destructive tool. localmelo trusts the hint only after you chose to start that server.
- A tool registered by Python keeps its explicit `requires_approval`: the tier is only a label for it, and a `register` call without the mark does not ask.
- An MCP server's `require_approval` is an extra mark on top of the tier: `true` or a list of tool names makes even a `read` tool ask. `false` means no extra mark; it cannot stop a `write`, `exec` or `network` tool from asking.

### With a workspace root

A direct run with `--workspace DIR` and an agent built in Python with `AgentOptions(workspace_root=DIR)` change the built-in tools (see [Work in a directory](python.md#work-in-a-directory)). A direct run without the flag and the gateway never set a root.

| Topic | What happens |
|---|---|
| The fence | `file_read` and `file_write` resolve a relative path against the root and reach only files whose real path is inside it, after `..` and symlinks are resolved. A path outside is refused with `Path '<path>' is outside allowed workspace root`; an empty path, or one with a NUL character, with `Argument 'path' for <tool> must be a non-empty string without NUL characters.` The tool gets the resolved path. |
| What the fence misses | `~` is not expanded, a path spelled in another letter case than the root is refused, and hard links are not detected. |
| `list_files` | Fenced the same way; no path (or an empty one) lists the root. It opens a folder without following a symlink and never enters a link inside it, so a link that points out of the root shows as `name@` and its target stays unlisted. |
| Opening files | The file tools open the resolved path without following symlinks, so a symlink that takes the file's place after the check is refused. Anything but a regular file, such as a FIFO, is refused with `Not a regular file: <path>`. |
| Size | `file_read` reads at most 200,000 bytes. A file over that, or whose text is longer than 50,000 characters, is cut to 50,000 characters, which end with `... [truncated: the file has <size> bytes]`. |
| Commands | `shell_exec` and `python_exec` run in a child process with a process group of its own, which a timeout or a cancellation kills with every process it started; a root only sets the directory they start in. They are not fenced: they reach any path, except that a worker run or a saved task on macOS refuses their writes outside the workspace (`AgentOptions(confine_writes=True)`). |
| Input | A shell command's standard input is `/dev/null`; `python_exec` gets its code on a pipe (never in argv) and then points fd 0 at `/dev/null`, so `input()` gets EOF and neither reads the terminal melo runs in. |
| Environment | The child gets `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `TERM`, `LANG`, every `LC_*` variable and `TMPDIR`, plus `AgentOptions(tool_env={...})`. Config-backed direct runs, the UI and console sessions also resolve the extra names in `[agent] tool_env`; `--url` runs ignore that setting. An MCP server gets the same base plus its own `env`, not the agent's extra names. |
| Exit status | A `shell_exec` command that exits non-zero starts its text with `[exit code <n>]` (`-<signal>` for a command a signal ended), and the call still succeeds; the line comes first, so it stays when the output is cut. Python that exits non-zero fails with `Python exited with code <n>`, followed by the last 4,000 characters of its output. |
| Time limits | `AgentOptions(tool_timeout_s=...)` sets the limit of `shell_exec` and `python_exec`, in seconds; without it, 30 for a shell command and 60 for Python. A shell command fails with `Command timed out after <n> s`; a Python call ends as a timeout with `Execution timed out after <n>s`, and the child is killed. |

### With an approval policy

![The terminal asks "Change a file?" with the diff, and offers Yes, Yes and allow every call in this session, or No and tell melo why.](../assets/img/terminal-approval.webp)

A run started in Python with `approval=policy` asks the policy before each call of a built-in or provided tool whose tier is not `read` (see [Permission tiers](#permission-tiers)):

- the built-ins it offers (`python_exec` and `file_write` in the default `python` mode; `shell_exec` too when the caller names it);
- a provided or MCP tool that is `write`, `exec` or `network`, including an MCP tool with no annotations, which is `write`;
- a tool that is explicitly marked: a provided tool with `requires_approval=True`, a custom tool whose `register` call sets it, or the tools of an MCP server that sets `require_approval` (see [Approve tool calls](python.md#approve-tool-calls)).

Who has a policy:

| Run | Policy |
|---|---|
| The interactive UI | Always, unless `--yolo`. |
| The console | Yes, but a credential holder or a same-user process can answer approvals, so it counts as gateway mode. |
| Direct mode, the gateway's `/v1` and workspace sessions | None. A gateway session does not offer `shell_exec` at all: there is no one to ask. |

A custom tool's tier is only a label: it asks when its `register` call sets the mark. Without a policy, these tools run without asking.

| Step | What happens |
|---|---|
| After the other checks | The policy sees only a call that every check above lets through, so it sees exactly what will run, the resolved file path included. A call whose arguments do not match the tool's parameters is refused with `Arguments for <tool> do not match its parameters: <message>`, and the policy is not asked. |
| Outside the time limit | The tool's time limit starts once the policy allows the call. |
| A denial | Refuses the call. The model reads why, for example `The user denied this call. Do not run it again or reach the same result another way unless the user asks.` |
| It fails closed | A policy that raises an error, or returns no decision, refuses the call with `Approval failed, so the call did not run: <ExceptionType>: <message>`. |
| The re-check | Once the policy allows a file tool's call under a workspace root, the path is resolved again. A path that now resolves elsewhere, as when a symlink took the file's place while the policy decided, is refused with `Path '<path>' changed while waiting for approval; the call did not run.` |

### High-impact calls

Before the policy decides, the checker reviews each `shell_exec`, `python_exec` and `file_write` call with fixed rules: a command is split as the shell splits it, Python code is parsed (nothing runs), and a local script the call runs is read once. A call is **routine**, **high-impact** or **undecided**.

It is high-impact when it:

- installs software or packages, restores a project's packages (`uv sync`, `npm ci`, `pip install -r`), or creates a programming environment (`python -m venv`, `uv venv`, `conda create`);
- downloads an installer, a large file (an archive, a wheel, model weights) or anything to a place outside the workspace, or runs code downloaded from the internet (`curl … | sh`, `get-pip.py`);
- writes or deletes files outside the workspace, deletes the workspace or its `.git`, or discards work (`git reset --hard`, `git clean -x`, a force-push);
- sets up something that runs later (launchd, `crontab`, shell startup files, git hooks, `.envrc`, `nohup`), uses passwords, keys, cookies or the keychain (reads too), changes system settings or uses `sudo`, opens a network service other computers can reach, or sends a file from outside the workspace or an email;
- changes localmelo itself, or runs code built at run time (`exec` of computed text, `eval "$x"`).

A call is undecided when the rules cannot read what it does: a command, program or path built at run time, a POST that sends data, `git push`.

A high-impact call gets its own card in the web console and its own panel in the terminal (see [the web console](web-console.md#approvals-and-failures) and [the terminal](terminal.md#approvals)). Each:

- names what the call does as a question ("Install software?", "Change files outside demo-project?");
- lists up to three reasons and offers no session choice;
- starts on No in the terminal, with one line saying that a session grant never covers installs, downloads or changes outside the folder;
- shows the message that started the turn (the console card).

What else is decided:

| Topic | What happens |
|---|---|
| A session grant | "Allow for this chat" (in the terminal, "Yes, and allow every … call in this session") covers routine calls only. A high-impact or undecided call of the same tool asks every time, and allowing it grants only that call. The terminal's `auto` mode runs routine calls and asks before the others. |
| localmelo itself | A call that installs into the Python that runs localmelo, or writes under `~/.cache/localmelo` or the app, is denied without asking: `The approval policy denied this call: melo never lets a call change its own runtime, settings or keys. …` |
| What the model reads | A denial says what the call does, for example `The user denied this call: not now. melo asked because it installs httpx with pip into .venv. Do not run it again or reach the same result another way unless the user asks. If the task needs it, say in your answer what for and ask the user.` |
| Limits | The rules read the literal text of a call; code that builds what it runs at run time is undecided, and a call the rules miss still runs under a grant. Direct runs, `--brief` runs and `--yolo` have no policy, so nothing is reviewed there. |
