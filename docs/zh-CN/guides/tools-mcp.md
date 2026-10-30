<a id="guide-tools-mcp"></a>

# 指南：工具与 MCP

本指南说明如何选择内置工具、从 MCP 服务器添加工具，以及理解保护这些工具的检查和权限级别。允许一次运行读取文件或执行命令之前请阅读本指南；工具列表和选项见[工具参考](../reference/tools.md)。

<a id="choose-the-built-in-tools"></a>

## 选择内置工具

```bash
melo --builtins none "What is 6*7?"
melo --builtins file_read,shell_exec "How many lines does notes.txt have?"
```

- `--builtins` 接受 `all`（所有内置工具，包括 `shell_exec`）、`none` 或以逗号分隔的工具名称列表。未知名称属于用法错误。未指定该选项时由模式决定：直接运行和 Python API 使用 `file_read`、`file_write` 和 `python_exec`；网关在这三个之外再加 `list_files`；交互式界面使用全部五个工具。`--builtins all` 指定所有内置工具，而 `--builtins shell_exec` 只指定 Shell 工具。
- JSON 行中的 `config.builtins` 和 `config.tools` 显示已注册的工具。
- 在 Python 中使用 `AgentOptions(builtins=...)`；见 [AgentOptions](../reference/python-api.md#agentoptions)。

<a id="add-tools-from-mcp-servers"></a>

## 从 MCP 服务器添加工具

localmelo 可以使用以本地进程运行的 [Model Context Protocol（英文）](https://modelcontextprotocol.io)服务器（stdio 服务器）提供的工具。在 JSON 文件中使用其他 MCP 客户端也使用的 `mcpServers` 格式列出服务器：

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

此示例使用 MCP 项目的参考时间服务器，`uvx` 首次使用时会下载它。保存文件，例如保存为 `mcp.json`，然后通过 `--mcp` 传入：

```bash
melo --mcp mcp.json "What time is it in Tokyo?"
```

服务器的工具注册为 `mcp__time__<tool>`。要查看它们，添加 `--output json` 并读取 `config.tools`。要仅使用 MCP 工具，添加 `--mcp-only`。

服务器不继承 melo 的环境：它只获得 `PATH`、`HOME`、`USER`、`LOGNAME`、`SHELL`、`TERM`、`LANG`、所有 `LC_*` 变量、`TMPDIR` 和其配置项中的 `env`，没有其他变量。melo 环境中的 API 密钥不会传给任何服务器；需要密钥的服务器必须在其 `env` 中获得字面值。

<a id="mcp-config-file"></a>

### MCP 配置文件

服务器配置项的所有键：

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

| 键 | 默认值 | 说明 |
|---|---|---|
| `command` | 必填 | 启动服务器的程序。 |
| `args` | `[]` | 程序的参数。 |
| `env` | `{}` | 服务器的环境变量。值按原样使用：`${VAR}` 不会展开。 |
| `cwd` | `melo` 的工作目录 | 服务器启动时所在的目录。 |
| `timeout` | `120` | 每次工具调用允许的秒数。 |
| `startup_timeout` | `30` | 启动服务器并列出其工具允许的秒数。 |
| `include_tools` | 所有工具 | 仅注册这些工具，名称使用服务器提供的名称。 |
| `require_approval` | `false` | `true`：每次调用该服务器的工具，包括只读工具，都先询问本次运行的审批策略。也可指定工具名称列表，拼写与 `include_tools` 相同：只有这些工具的调用会询问。这是在工具的权限级别之上的额外标记（见[权限级别](#permission-tiers)），仅在运行具有策略时生效。 |
| `disabled` | `false` | `true` 跳过该服务器。 |

- 可用 `servers` 代替 `mcpServers`。其他键会被忽略，`type`（或 `transport`）除外：如果提供，它必须是 `stdio`。
- `require_approval` 必须是 `true`、`false` 或字符串列表；其他值属于用法错误。列表中的名称若不是服务器提供的工具，会记录一条警告。
- 仅支持 stdio 服务器。其他 `type` 属于用法错误，例如 `only stdio servers are supported (got type 'sse')`。提供 `url` 却没有 `command` 的配置项也是用法错误：`'command' must be a non-empty string (HTTP/SSE servers are not supported)`。
- 服务器不继承你的环境。它获得 `HOME`、`LOGNAME`、`PATH`、`SHELL`、`TERM`、`USER`、`LANG`、`LC_ALL`、`LC_CTYPE` 和 `TMPDIR`，以及自己的 `env`。
- 工具命名为 `mcp__<server>__<tool>`。字母、数字、`_` 和 `-` 以外的字符会变为 `_`。超过 64 个字符的名称会缩短至 64 个字符，并以短哈希结尾；JSON 行中的 `config.tools` 显示实际使用的名称。
- 工具的 `annotations` 对象按服务器提供的原样保留；非内置工具的权限级别由它决定（见[权限级别](#permission-tiers)）。格式错误的值会被忽略，工具仍会列出。
- 服务器在运行开始时启动。无法启动的服务器会使运行以 `setup_error` 状态结束，例如 `MCP server 'notes' exited (code 1)`。
- MCP 服务器适用于直接模式和 [Python API](../reference/python-api.md#tool-providers-and-mcp)。网关会话没有 MCP 工具。
- 结果的 `structuredContent` 只在结果既没有文本也没有图片时才作为文本发送：截图服务器若在其中重复图片的 base64，发送的是图片本身，而不是它的 base64。

<a id="pictures-that-tools-return"></a>

### 工具返回的图片

工具可以返回图片，例如 MCP 服务器的截图。每张图片都会经过检查（每个结果最多 8 张，每张最多 20 MiB；PNG、JPEG、GIF 或 WebP），并在工具消息末尾注明，例如 `[1 image: sha256:3f2a9c0d1b7e image/png]`。像素是否发给模型由设置 `[agent] tool_images` 决定：

| 值 | 模型得到的内容 |
|---|---|
| `auto`（默认） | 服务器表明模型能读图时发送像素：oMLX 的 `GET /v1/models/status` 给出该模型的 `model_type` 为 `vlm`，或 Ollama 的 `POST /api/show` 在 `capabilities` 中列出 `vision`。localmelo 在工具第一次返回图片时询问一次。其他回答或没有回答（vLLM、SGLang 和 MLC 不提供此信息）都按 `off` 处理。 |
| `on` | 不询问，直接发送像素。 |
| `off` | 只发送注明每张图片的那一行，与引入此设置之前相同。 |

用 `-X` 为一次运行设置：

```bash
melo -X agent.tool_images=on --mcp browser.json --mcp-only "Describe the page that is open"
```

开启像素后，工具的文本仍留在它的 `role: "tool"` 消息中；图片跟在这一组工具结果之后，每次调用一条 `user` 消息，以 `image_url` data URL 发送（Ollama 使用 `images` 字段），并标明对应的调用。`tests/cli/test_tool_images_e2e.py` 的第二个请求以如下内容结尾：

```json
{"role": "tool", "tool_call_id": "call_1", "content": "[1 image: sha256:497790947d46 image/png]"},
{"role": "user", "content": [
  {"type": "text", "text": "Output of tool call call_1 (mcp__fake__screenshot): [1 image: sha256:497790947d46 image/png]"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoA..."}}
]}
```

- 最新的 `tool_images_keep` 张图片（默认 3）保留像素。当图片数超过 `tool_images_keep + tool_images_retire - 1` 时，最旧的 `tool_images_retire` 张（默认 2）一起失去像素：每个请求带 3 到 4 张图片，服务器缓存的提示前缀每两张图片才变化一次，而不是每一步都变。
- 与已发送图片摘要相同的图片不会再次发送；它的工具消息以 `[same image as an earlier tool result; its pixels are above]` 结尾。
- 宽于 1,280 像素的图片在保存时一次性缩放为 1,280 像素宽的 JPEG（使用 macOS 的 `sips`；其他系统按原样发送）。此时标签注明的是该 JPEG。
- 每张发送的图片在 100,000 字符的提示上限中计 4,500 字符，并按像素尺寸计入上下文预算（[上下文记账](../reference/configuration.md#context-accounting)）；较早的轮次会移出窗口以腾出空间。
- 发给模型的副本只保存在内存中：历史记录和工作窗口只保留注明图片的那一行（会话仍像以前一样为控制台保存图片）。会话重新打开后，在工具返回新图片之前不会发送任何图片。
- 工具不返回图片的运行，在每种模式下发送的请求都完全相同。

<a id="what-the-checks-do"></a>

## 检查的作用

工具运行前，localmelo 会检查调用：

- 本步必须提供了该工具。
- 参数必须是 JSON 对象。对于内置工具，每个参数还必须符合工具声明的类型；例如 `shell_exec` 的 `command` 使用列表时，会以 `Arguments for shell_exec: 'command' must be of type string, not array` 拒绝。
- 由工具的权限级别和模式决定，而非命令的写法。有审批者的模式（交互式界面）提供 `shell_exec` 并对每次 `exec` 调用请求审批（见[权限级别](#permission-tiers)）；使用 `--yolo` 时不请求审批，调用直接运行。没有审批者的模式（直接模式、网关、控制台及 Python API 默认的 `python` 模式）默认不提供 `shell_exec`，因此模型看不到它；即使 `exec` 工具已注册，检查器也会拒绝其调用，除非 `direct` 或 `python` 调用方在 `builtins` 中指定了它。在这些模式中指定 `shell_exec` 表示调用方明确启用它，调用可以运行；`gateway` 无论是否指定都会拒绝。
- 设置工作区根目录时，`file_read` 和 `file_write` 的路径必须在其中（见[设置工作区根目录时](#with-a-workspace-root)）。
- 设置审批策略时，级别不是 `read` 或明确标记的工具会等待策略决策；若还设置了工作区根目录，文件路径之后会再次检查（见[权限级别](#permission-tiers)和[设置审批策略时](#with-an-approval-policy)）。

工具运行后：

- 超过 50,000 字符的输出会被截断。
- 写入记忆的内容也会被检查，每次写入上限为 50,000 字符。
- 超时或其运行被取消的 `shell_exec` 命令会连同它启动的所有进程一起被终止。

被拒绝的调用不会运行。模型将拒绝信息视为工具错误，可以尝试其他方法。

这些检查是防护措施，不是沙箱：

- 工具以你的账户权限运行。
- 未设置工作区根目录时，例如不带 `--workspace` 的直接运行和网关中，文件工具可以访问你能访问的任何文件。`python_exec` 在 melo 的子进程中运行，而非沙箱中，也能访问你能访问的任何文件。
- 设置工作区根目录后，`shell_exec` 和 `python_exec` 仍能访问你能访问的任何文件：只有文件工具被限制在根目录内。例外是 macOS 上的 worker 运行（`melo worker`）和保存的任务：它们的写入被限制在工作区内（读取和网络不受限制）；见[写入留在工作区内（英文）](../../reference/cli.md#confined-writes)。
- 审批适用于有审批策略的运行，包括交互式界面（除非使用 `--yolo`）和控制台会话。直接模式、`melo ""` 和旧版网关 API 没有审批策略。
- 直接模式和 Python 模式默认不提供 `shell_exec`，但明确指定 `--builtins all` 或 `--builtins shell_exec`（Python：`AgentOptions(builtins=frozenset({"shell_exec"}))`）会在没有审批策略的情况下启用它。网关和控制台模式从不提供它。交互式界面将它置于审批面板之后，除非使用 `--yolo`。

要限制任务能做的事，请使用 `--builtins none` 或工具子集、`--mcp-only`、容器或虚拟机，或独立用户账户。见[安全策略（英文）](https://github.com/localmelo/localmelo/blob/main/SECURITY.md)。

<a id="permission-tiers"></a>

### 权限级别

每个工具都有四种级别之一。级别说明调用能够访问什么，并不表示拒绝。对于具有审批策略的运行中的内置工具或提供的工具，只有 `read` 级别无需询问即可运行。

| 级别 | 含义 | 有策略时询问 |
|---|---|---|
| `read` | 读取本地数据；没有进程外的副作用。 | 否，除非工具被明确标记。 |
| `write` | 创建、修改或删除本地文件或状态。 | 是。 |
| `exec` | 以你的权限运行命令、代码或其他程序。 | 是。 |
| `network` | 向本机外的对象发送数据或对其执行操作：HTTP API、聊天、支付、图形界面等。 | 是。 |

内置工具的级别按名称固定：`file_read` 和 `list_files` 是 `read`，`file_write` 是 `write`，`shell_exec` 和 `python_exec` 是 `exec`。该表优先于工具自身的声明，因此复用内置工具名称的提供的工具或自定义工具会继承该级别。

模式决定实际提供哪些内置工具。交互式界面以 `interactive` 模式运行，获得全部五个工具；除非使用 `--yolo`，它在 `exec` 调用前会询问。网关或控制台会话在 `file_read`、`file_write` 和 `python_exec` 之外还获得 `list_files`；直接运行以及 Python API 默认的 `python` 模式只获得这三个。除交互式界面外，各模式都不注册 `shell_exec`，因此模型看不到它；即使注册了，检查器也会拒绝它，除非调用方在 `builtins` 中指定。模型仍然调用它时会收到 `shell_exec is not available in this run. Available tools: …`，其中列出本次运行的工具（未知的非内置名称仍使用 `Tool not found in registry` 消息）。指定 `shell_exec`（`--builtins all`、`--builtins shell_exec`、`AgentOptions(builtins={"shell_exec"})`）表示调用方在 `direct` 和 `python` 模式中明确启用它，调用可以运行；`gateway` 即使在指定时也会拒绝。除了网关模式，`AgentOptions(builtins=…)` 或 `--builtins` 始终优先于模式的默认工具集合。

所有其他工具的级别来自服务器列出工具时提供的 MCP `annotations` 对象，按保守原则解读：

- `openWorldHint: true` 使其成为 `network`，无论它是否也声明只读。
- `readOnlyHint: true` 使其成为 `read`，除非 `destructiveHint: true` 与之矛盾；此时采用更保守的解读，级别为 `write`。
- 其他情况均为 `write`：没有注解、注解不是对象、提示项不是布尔值，以及 `idempotentHint` 和 `title` 键。工具只有声明只读才被视为只读。

注解是服务器的声明，不是证明：服务器可能对破坏性工具声明 `readOnlyHint: true`。localmelo 只在你选择启动该服务器后才信任这一提示。通过 Python 注册的工具保留其明确的 `requires_approval`：级别只是标签，不带该标记的 `register` 调用不会请求审批。

MCP 服务器的 `require_approval` 是级别之上的额外标记：`true` 或工具名称列表会使 `read` 工具也请求审批。`false` 表示没有额外标记；它不能阻止 `write`、`exec` 或 `network` 工具请求审批。

<a id="with-a-workspace-root"></a>

### 设置工作区根目录时

使用 `--workspace DIR` 的直接运行和通过 Python 使用 `AgentOptions(workspace_root=DIR)` 创建的智能体会改变内置工具（见[在目录中工作](python.md#work-in-a-directory)）。不带该选项的直接运行和网关从不设置工作区根目录。

- **范围限制。** `file_read` 和 `file_write` 相对于根目录解析相对路径，在解析 `..` 和符号链接后，只能访问真实路径位于根目录内的文件。外部路径会以 `Path '<path>' is outside allowed workspace root` 拒绝，空路径或包含 NUL 字符的路径会以 `Argument 'path' for <tool> must be a non-empty string without NUL characters.` 拒绝。工具获得解析后的路径。`~` 不会展开，与根目录字母大小写不同的路径会被拒绝，硬链接不会被检测。`list_files` 受同样的限制，不带路径（或路径为空）时列出根目录。
- **打开文件。** 文件工具打开解析后的路径时不跟随符号链接，因此检查后取代文件的符号链接会被拒绝。非普通文件，例如 FIFO，会以 `Not a regular file: <path>` 拒绝。`list_files` 打开文件夹时不跟随符号链接，也不会进入其中的链接，因此指向根目录之外的链接显示为 `name@`，其目标不会被列出。`file_read` 最多读取 200,000 字节。超过 200,000 字节或文本超过 50,000 字符的文件会截断至 50,000 字符，末尾为 `... [truncated: the file has <size> bytes]`。
- **命令。** `shell_exec` 和 `python_exec` 始终在拥有独立进程组的子进程中运行，超时或取消会终止它及其启动的所有进程；根目录只决定它们的启动目录。Shell 命令的标准输入是 `/dev/null`；`python_exec` 通过管道接收代码（绝不通过 argv），随后将 fd 0 指向 `/dev/null`，因此 `input()` 得到 EOF，两者都不读取 melo 所在的终端。这两个工具不受范围限制：它们可以访问任何路径；但在 macOS 上，worker 运行和保存的任务会拒绝它们在工作区之外的写入（`AgentOptions(confine_writes=True)`）。
- **环境。** `shell_exec` 和 `python_exec` 的子进程不继承 melo 的环境：它获得 `PATH`、`HOME`、`USER`、`LOGNAME`、`SHELL`、`TERM`、`LANG`、所有 `LC_*` 变量和 `TMPDIR`，以及 `AgentOptions(tool_env={...})`。使用配置的直接运行、界面和控制台会话也会解析 `[agent] tool_env` 中的额外名称；使用 `--url` 的会话和运行会忽略该设置。Python API 可以在其映射中传入任何变量。MCP 服务器获得相同的基础环境以及自己的 `env`，而不是智能体的额外名称。
- **退出状态。** `shell_exec` 命令以非零状态退出时，其文本以 `[exit code <n>]` 行开头（被信号终止的命令使用 `-<signal>`），调用仍然成功。该行位于开头，因此输出被截断时仍会保留。Python 代码以非零状态退出时，调用以 `Python exited with code <n>` 失败，后面附上输出的最后 4,000 个字符。
- **时限。** `AgentOptions(tool_timeout_s=...)` 以秒为单位设置 `shell_exec` 和 `python_exec` 的时限。不设置时，Shell 命令保持 30 秒，Python 保持 60 秒。运行过久的 Shell 命令以 `Command timed out after <n> s` 失败；子进程仍在运行的 Python 调用以超时结束，提示 `Execution timed out after <n>s`，并终止子进程。

<a id="with-an-approval-policy"></a>

### 设置审批策略时

通过 Python 使用 `approval=policy` 启动的运行，在每次调用级别不是 `read` 的内置工具或提供的工具前都会询问策略（见[权限级别](#permission-tiers)）：包括运行提供的内置工具（默认 `python` 模式下的 `python_exec` 和 `file_write`；调用方指定时也包括 `shell_exec`），以及级别为 `write`、`exec` 或 `network` 的提供的工具或 MCP 工具，包括没有注解因而为 `write` 的 MCP 工具。交互式界面始终有这样的策略，除非使用 `--yolo`；直接模式以及网关的 `/v1` 和工作区会话没有。因此网关会话完全不提供 `shell_exec`：没有人可以询问。控制台有策略，但持有凭据的人或同一用户的进程都能回答审批，因此也属于网关模式。工具被明确标记时也会请求审批：例如 `requires_approval=True` 的提供的工具、在 `register` 调用中设置该标记的自定义工具，或设置了 `require_approval` 的 MCP 服务器的工具（见[审批工具调用](python.md#approve-tool-calls)）。自定义工具的级别只是标签：只有其 `register` 调用设置了标记才会询问。

- **在其他检查之后。** 策略只会收到上述所有检查均通过的调用，因此能看到将要运行的确切内容，包括解析后的文件路径。参数不符合工具参数定义的调用会以 `Arguments for <tool> do not match its parameters: <message>` 拒绝，不会请求策略决策。
- **不计入工具时限。** 工具时限从策略允许调用后开始计算。
- **拒绝**会阻止调用。模型读取拒绝原因，例如 `The user denied this call. Do not run it again or reach the same result another way unless the user asks.`
- **故障时默认拒绝。** 引发错误或不返回决策的策略会以 `Approval failed, so the call did not run: <ExceptionType>: <message>` 拒绝调用。
- **再次检查。** 在工作区根目录下，策略允许文件工具调用后会再次解析路径。如果此时路径解析到其他位置，例如策略决策期间符号链接取代了文件，会以 `Path '<path>' changed while waiting for approval; the call did not run.` 拒绝。
- 没有策略时，这些工具无需询问即可运行。

<a id="high-impact-calls"></a>

### 高影响调用

在策略决策之前，检查器会用固定规则审查每次 `shell_exec`、`python_exec` 和 `file_write` 调用：命令按 Shell 的方式拆分，Python 代码只解析不运行，调用运行的本地脚本会读取一次。调用分为**常规**、**高影响**和**无法判定**三类。以下调用属于高影响：

- 安装软件或软件包，恢复项目的软件包（`uv sync`、`npm ci`、`pip install -r`），或创建编程环境（`python -m venv`、`uv venv`、`conda create`）；
- 下载安装程序、大文件（压缩包、wheel、模型权重）或把任何文件下载到工作区以外，或运行从互联网下载的代码（`curl … | sh`、`get-pip.py`）；
- 在工作区以外写入或删除文件，删除工作区本身或其 `.git`，或丢弃工作（`git reset --hard`、`git clean -x`、强制推送）；
- 设置以后会自行运行的内容（launchd、`crontab`、Shell 启动文件、git 钩子、`.envrc`、`nohup`），使用密码、密钥、Cookie 或钥匙串（读取也算），更改系统设置或使用 `sudo`，打开其他计算机可以访问的网络服务，或发送工作区以外的文件或电子邮件；
- 更改 localmelo 本身，或运行在运行时拼出的代码（对计算出的文本使用 `exec`、`eval "$x"`）。

规则无法读出调用的作用时，调用属于无法判定：在运行时拼出的命令、程序或路径，发送数据的 POST，`git push`。

- **会话授权只覆盖常规调用。** “Allow for this chat”（终端中为 “Yes, and allow every … call in this session”）让该工具的常规调用无需询问即可运行。同一工具的高影响或无法判定的调用每次都会询问；对它选择会话授权只允许这一次调用，不会授权该工具。终端的 `auto` 模式运行常规调用，其余调用会先询问。
- **永不更改 localmelo 本身。** 安装到运行 localmelo 的 Python、或写入 `~/.cache/localmelo` 或应用程序的调用会被直接拒绝，不会询问：`The approval policy denied this call: melo never lets a call change its own runtime, settings or keys. …`
- **模型会读到原因。** 拒绝高影响调用时，文本会说明调用的作用，例如 `The user denied this call: not now. melo asked because it installs httpx with pip into .venv. Do not run it again or reach the same result another way unless the user asks. If the task needs it, say in your answer what for and ask the user.`
- **限制。** 规则读取调用的字面文本；在运行时拼出所运行内容的代码属于无法判定，规则漏掉的调用在授权下仍会运行。直接运行、`--brief` 运行和 `--yolo` 没有策略，因此不做审查。审批卡片和终端面板暂时还不显示原因。
