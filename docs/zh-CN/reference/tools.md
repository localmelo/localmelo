<a id="tools-and-mcp"></a>

# 工具与 MCP

本页介绍模型可以调用的工具：内置工具、来自 MCP 服务器的工具、每一步提供哪些工具，以及检查做些什么。

<a id="built-in-tools"></a>

## 内置工具

| 工具 | 作用 | 限制 |
|---|---|---|
| `shell_exec` | 以 `/dev/null` 作为标准输入运行 shell 命令，并返回它的输出。 | 30 秒后连同它启动的所有进程一起终止。只有 `interactive` 模式默认提供它：直接运行和 Python 调用方需要显式点名它的 `builtins` 集合；网关或控制台会话默认不提供它，即使点名也会拒绝。 |
| `python_exec` | 在子进程中运行 Python 代码，并返回它打印的内容。 | 60 秒后连同它启动的所有进程一起终止。`sys.exit()` 只结束子进程。 |
| `file_read` | 读取文本文件。接受 `path`；只读一部分时还可用 `start_line` 和 `end_line`（从 1 开始计数，含两端）。 | 你的账户能读取的任何文件。没有显示整个文件的读取，末尾会注明文件共有多少行，以及接着读时用的 `start_line`（`... [the file has 1,240 lines; showing 1-400; read more with start_line=401]`）。 |
| `list_files` | 列出一个文件夹：先是文件夹（以 `/` 结尾），然后是带大小的文件；符号链接以 `@` 结尾。接受 `path`（默认：工作区文件夹）和 `depth`（1 到 3，默认 1）。从不读取文件，从不跟随符号链接，并跳过 `.git`。 | 显示 200 个条目，之后是 `... and N more`。只读，因此从不请求审批。交互式界面以及网关和控制台会话默认提供它；直接运行和 Python API 的默认模式不提供（`--builtins list_files` 或 `all` 可指定它）。 |
| `file_write` | 把文本写入文件，创建或替换它。 | 你的账户能写入的任何文件。 |
| `file_edit` | 把文件里一段确切的文本（`old`，必须只出现一次）替换为 `new`。 | 文件最大 1 MB。先检查 `old`：文本找不到或出现多次的编辑会立刻失败并给出明白的错误，不会询问任何人。否则像 `file_write` 一样会询问；批准卡片把改动显示为 diff。只有任务的 worker 提供它（其他地方需在 `builtins` 中写明）。 |
| `git_status` | 读取工作区的 git 状态：`status`、`branch`、`log`（`count` 最多 50）、`diff`（`stat`、`staged`、`paths`）或 `show`（`rev`）。 | 只读：由 melo 自己组装 git 参数，不经过 shell，不运行钩子、分页器或外部 diff，也不刷新索引。路径（包括 `REV:path` 中的路径）必须在工作区内，且不能是可能存有密钥的文件（`.env`、`*.pem`、`id_rsa`…）。从不询问。超过 20,000 个字符的输出会被截断。只有任务的 worker 提供它。 |

`--builtins all` 指前五个；`file_edit` 和 `git_status` 属于任务，需要逐个写明。

一次 `file_read`、`file_write` 或 `list_files` 调用最长可能用 60 秒，一次 MCP 工具调用最长用其服务器的 `timeout`。超过 50,000 个字符的输出会被截断。

`python_exec` 的代码通过标准输入（一个管道）到达子进程，绝不出现在它的 `-c` 参数中：`ps` 看不到代码，调用也不受 `ARG_MAX` 限制（macOS 上约为 1 MiB 的 argv 和环境）。子进程运行一个固定的包装程序；它读完代码后就把文件描述符 0 指向 `/dev/null`，因此 `input()` 会得到 EOF。

子进程最多可有 1,024 个打开的文件和一小时的 CPU 时间（POSIX 的 `RLIMIT_NOFILE` 和 `RLIMIT_CPU` 限制）。通常调用的时间限制会先结束它；CPU 限制只会停止比 melo 存活更久、或耗满一小时 CPU 的代码。

不设置地址空间限制：macOS 不接受有限的地址空间限制。

`shell_exec` 和 `python_exec` 以最小环境启动子进程：`PATH`、`HOME`、`USER`、`LOGNAME`、`SHELL`、`TERM`、`LANG`、所有 `LC_*` 变量和 `TMPDIR`，再加上 `AgentOptions(tool_env={...})` 添加的内容，别无其他。基于配置的运行和会话还可以另外继承 `[agent] tool_env` 中列出的名称；见[工具环境](configuration.md#tool-environment)。

melo 自身环境中的 API 密钥不会到达任何工具，除非被显式点名或映射。`VIRTUAL_ENV`、`PYTHONPATH`、`SSH_AUTH_SOCK` 和代理变量默认不会被继承。

暴露 `SSH_AUTH_SOCK` 会允许工具使用你的 SSH 代理，暴露代理变量也会暴露其中的任何凭据。

在不带 `--workspace` 的直接运行和网关中，内置工具就是这样工作的。带 `--workspace DIR` 的直接运行和设置了 `workspace_root` 的 Python 智能体把文件工具（`file_read`、`file_write` 和 `list_files`）限制在一个目录内，并在那里启动 `shell_exec` 和 `python_exec`（见[设置工作区根目录](../guides/tools-mcp.md#with-a-workspace-root)）。

交互式界面（单独的 `melo`）也以它的工作区为根目录，并在每次 `shell_exec`、`python_exec` 和 `file_write` 调用前询问，除非用了 `--yolo`；`file_read` 和 `list_files` 只读取，从不询问（见[权限级别](../guides/tools-mcp.md#permission-tiers)）；直接模式没有审批策略，既不提供 `shell_exec` 也不提供 `list_files`，有没有根目录都一样。交互式界面加载 `--mcp-config` 指定的 MCP 服务器，否则加载顶层 `mcp_config` 键指定的那些；启动失败的服务器会让该会话关闭 MCP（`/mcp` 会说明原因）。

带审批策略的 Python 运行按同样的权限级别，用自身的策略处理 MCP 工具。

<a id="how-tools-are-offered"></a>

## 如何提供工具

每个规划步骤向模型发送一份工具列表，该步只能运行列表中的工具。`--tool-exposure` 选择列表：

- **`retrieve`**（默认值）：所有已注册的内置工具，按名称排序，然后是这一步的额外工具，同样按名称排序。额外的工具在其他工具（例如 MCP 工具）中选出：
  - 模型在本次对话中已经调用过的工具，以及
  - 与任务最匹配的 3 个工具，依据是对每个工具的名称、描述和标签做关键词搜索（带 IDF 的 Okapi BM25）。

  只有模型的工具调用算作提示：任务中、工具结果中或召回的记忆中出现的工具名称不增添任何内容。
- **`all`**：每一步都提供所有已注册的工具，按名称排序。列表从不变化，这对缓存提示前缀的服务器有帮助。

提供者可以把自己的工具标成一组（`ProvidedTool.group`）；在 `retrieve` 下，一组中的每个工具都像内置工具一样在每一步提供，排在内置工具之后。八个浏览工具就是这样的一组，只有 `[browser] driver` 开启时才存在（见[用 melo 浏览网页](../guides/browsing.md)）。

调用这一步没有提供的工具会被拒绝，错误为 `Tool '<name>' was not offered in this step.`，也不会运行。这次拒绝会留在对话中，因此后续步骤可以再提供该工具。
