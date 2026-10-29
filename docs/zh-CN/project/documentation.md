<a id="maintaining-the-docs"></a>

# 维护文档

本页说明文档和网站如何组织、发布、修改和翻译。页面索引见[文档索引](../README.md)；开发环境的设置见 [CONTRIBUTING.md](https://github.com/localmelo/localmelo/blob/main/CONTRIBUTING.md)。

<a id="layout"></a>

## 目录结构

```text
docs/
  README.md       the index of pages; GitHub shows it for the docs/ folder
  start/          onboarding sources
  guides/         task-oriented guides
  reference/      interfaces, options and exact behavior
  concepts/       explanations of components
  project/        project and documentation maintenance
  *.md            compatibility stubs at former page paths, plus the style guide
  for-agents.md   the compact entry for AI agents
  llms.txt        the llms.txt index for AI agents (llmstxt.org format)
  index.html      the website's root page; it redirects to en/index.html
  en/             website pages in English
  zh-CN/          Chinese twins of the Markdown pages, at the same paths; website pages in Chinese
  assets/         the website's stylesheet and images
```

- 每个页面在 `docs/zh-CN/` 下都有相同路径的中文对应页，但 `for-agents.md`、`llms.txt` 和 `index.html` 除外（见[翻译](#translations)）。根目录下的兼容存根没有中文对应页。
- 网站每种语言各有六个页面：首页、快速开始、架构、路线图、开发者和记忆设计。每个英文页面都有一个中文对应页，其中元素 `id` 相同，章节顺序相同，代码块完全一致。
- 记忆设计页逐字保留维护者的记忆设计文本，英文和中文版本都是如此。需要时可以重新排版，但只有维护者才能改动其中的文字。
- 根目录下的兼容存根让旧路径和片段上的链接继续可用；`README.md`、`for-agents.md` 和 `llms.txt` 仍然是有用的入口。
- 网站的快速开始页与 `start/quickstart.md` 保持一致。网站的其他页面链接到 GitHub 上的 Markdown 页面。

<a id="publishing"></a>

## 发布

GitHub Pages 发布 `main` 分支的 `docs/` 文件夹。Pages 还会用 Jekyll 渲染 `docs/` 中的 Markdown 文件。`docs/` 内页面之间的链接是相对链接；从文档链接到 `docs/` 之外的文件时，使用绝对 URL（`https://github.com/localmelo/localmelo/blob/main/…`），因此它们在 GitHub 上和渲染后的副本中都能工作。

要预览网站：

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory docs
```

然后打开 `http://127.0.0.1:8080/en/`。

源 Markdown 页面及其旧路径存根都在仓库中。实验性构建器是以后迁移网站的基础；它不会取代已发布的 HTML 网站。执行 `uv sync --extra dev` 之后，在仓库根目录运行以下命令，构建到一个新的临时文件夹，并将第二次构建与之比较：

```bash
SITE="$(mktemp -d)"
uv run --offline --locked --extra dev python scripts/build_docs.py --output "$SITE"
uv run --offline --locked --extra dev python scripts/build_docs.py --output "$SITE" --check
```

输出目录必须为空，并且位于检出目录之外。`--check` 会逐字节比较整个隔离的输出树，包括资源文件，并报告过时或多余的文件。预览会原样复制已有的 HTML，并渲染 Markdown、转义原始 HTML；它还不是完整的替代网站。具体来说，旧路径重定向、拆分页面的片段路由，以及复制的 HTML 中的链接，目前尚未迁移到预览中。旧的 Markdown 路径则提供了可读的存根，每个历史片段都有一个链接；这些存根无需 JavaScript 即可工作。缺少的中文页面不会被生成。现有的英文和中文快速开始 HTML 页面保留各自的 URL 和 ID；将来转换时，必须把每个旧片段映射到两种语言中已有的目标。具体地说，`#json-output` 和 `#serve-a-model-with-omlx` 不能都原样映射到新的英文快速开始页。不要提交预览。请参见[英文风格指南](../../STYLE.md)和[中文风格指南](../STYLE.md)。

文档测试会检查页面清单（manifest）以及所有受跟踪的链接和锚点。它们还会检查 `tests/docs/test_manifest.py` 中列出的每个中文对应页是否保留英文的锚点、代码块和行内代码。用 `uv run pytest tests/docs -q` 运行这些测试。

在 `docs/manifest.json` 中，页面的 `fragments` 把每个旧源路径映射到其迁移规则。`"same"` 把每个片段送到该语言的输出页面，并保留相同的 ID；而单个输出路径则把每个片段都送到该页面，并保留相同的 ID。两种规则都不需要为新标题编辑清单。拆分的页面要使用一份显式的“旧 ID 到 `page.html#ID`”映射：增加标题时，必须选择它的目标并把它加入该映射。文档测试会报告来源和缺失的锚点；已退役的 ID 也要保留在映射中。`tests/docs/legacy_fragments.json` 记录移动前的 Markdown 锚点，以及每个存根链接应指向何处；只有在要保留另一个历史片段时才应有意地更新它，而不应因为仅在未拆分的页面中增加标题就改动它。

<a id="add-or-change-a-page"></a>

## 新增或修改页面

- 在 `docs/README.md` 和 `docs/zh-CN/README.md` 的索引中加入新的 Markdown 页面，并在同一个 pull request 中把它的中文对应页加到 `docs/zh-CN/` 下的同一路径。两者都要列入 `docs/manifest.json`。只有新用户需要时，才把它加入顶层 `README.md` 和 `README.zh-CN.md` 的文档表格；该表格保持简短，并以指向索引的链接结束。
- 新增网站页面时，两种语言都要加上，并加入每个网站页面的导航。记忆设计是例外：它是设计记录，架构页和路线图页改为链接到它。
- 写入文档的每条命令都要先实际运行一遍。
- 每次改变行为、选项、输出或功能状态时，都要同步更新 `for-agents.md`，并把新页面加入 `llms.txt`。这两个文件是写给智能体看的：紧凑、精确、简短。

<a id="translations"></a>

## 翻译

- 英文是原文。修改英文页面时，要在同一个 pull request 中修改它的中文对应页。
- Markdown 页面的中文对应页位于 `docs/zh-CN/`。网站的中文页面是 `docs/zh-CN/*.html`，`README.zh-CN.md` 是中文的项目 README。
- `for-agents.md` 和 `llms.txt` 按设计只保留英文：它们是写给 AI 智能体的简明入口。`index.html` 同样只保留英文：它跳转到 `en/index.html`，并链接到中文首页 `zh-CN/index.html`。
- 中文页面中指向仅有英文版页面的链接，用（英文）标注。
- 两种语言中的代码块保持一致；共用代码块中的注释使用英文。

<a id="translation-glossary"></a>

## 翻译术语表

代码字体中的值在两种语言中都保持英文。术语在页面中首次出现时，可以在括号中附上英文，例如：嵌入（embedding）。

**产品与组件**

| 英文 | 中文 |
|---|---|
| localmelo (project), `melo` (command) | localmelo，`melo` 命令 |
| local personal agent | 本地个人智能体 |
| agent | 智能体 |
| task | 任务 |
| run | 运行（一次运行） |
| interactive run (Python API) | 交互式运行 |
| event | 事件 |
| direct mode | 直接模式 |
| gateway, gateway mode | 网关，网关模式 |
| session | 会话 |
| turn | 轮次 |
| web UI (playground) | 网页界面（playground） |
| web workspace | 网页工作区 |
| conversation loop (`melo ""`) | 对话循环（`melo ""`） |
| interactive CLI | 交互式命令行 |
| setup wizard | 配置向导 |
| config file | 配置文件 |
| background service (macOS) | 后台服务（macOS） |
| model server | 模型服务器 |
| cloud API | 云端 API |
| backend, chat backend, embedding backend | 后端，对话后端，嵌入后端 |
| embedding | 嵌入 |
| provider | 提供方 |
| tool, built-in tool, tool call | 工具，内置工具，工具调用 |
| tool provider | 工具提供方 |
| MCP server | MCP 服务器 |
| tool exposure | 工具暴露方式 |
| retrieve mode | 检索模式 |
| permission tier | 权限级别 |
| offered tools (of a step) | 本步提供的工具 |
| checker, check | 检查器，检查 |
| executor | 执行器 |
| allowlist, blocklist | 允许列表，拦截列表 |
| sandbox | 沙箱 |
| workspace | 工作区 |
| workspace root | 工作区根目录 |
| approval, approval policy | 审批，审批策略 |
| memory | 记忆 |
| working memory | 工作记忆 |
| long-term memory | 长期记忆 |
| history | 历史记录 |
| tool registry | 工具注册表 |
| persistence, persist | 持久化 |
| recall | 召回 |
| promote to long-term memory | 提升为长期记忆 |
| step, attempt | 步，尝试 |
| step budget | 步数上限 |
| reflection | 反思 |
| continuation gate | 继续判定 |
| stuck detection | 卡住检测 |
| run result, status | 运行结果，状态 |
| exit code, usage error | 退出码，用法错误 |
| JSON line | JSON 行 |
| chat request timeout | 对话请求超时 |
| run time limit (`--timeout`) | 运行时限 |
| retry, signal, process group | 重试，信号，进程组 |
| smoke test | 冒烟测试 |
| evaluation, percentage point | 评测，百分点 |
| pre-alpha | 早期预览版（pre-alpha） |
| roadmap, milestone, phase | 路线图，里程碑，阶段 |
| quickstart, reference, troubleshooting | 快速开始，参考，故障排查 |
| changelog, contributing guide, security policy | 更新日志，贡献指南，安全策略 |
| architecture, components | 架构，组件 |

**运行状态**（值保持英文）

| 状态 | 英文释义 | 中文释义 |
|---|---|---|
| `completed` | completed | 已完成 |
| `failed` | failed | 失败 |
| `gate_stopped` | stopped after reflection | 反思后停止 |
| `budget_exhausted` | budget exhausted | 预算用尽 |
| `stuck` | stuck | 卡住 |
| `provider_error` | model-server error | 模型服务错误 |
| `setup_error` | tool or inbox setup failed | 工具或收件箱准备失败 |
| `timeout` | timed out | 超时 |
| `crashed` | crashed | 异常终止 |
| `cancelled` (interactive runs only) | cancelled | 已取消 |
| `running` (task records only) | running | 运行中 |

**选项**（选项名保持英文）

| 选项 | 概念 | 中文 |
|---|---|---|
| `query` (positional) | task | 任务 |
| `--base-url URL` | server URL for this run | 本次运行的模型服务地址 |
| `--chat-model NAME` | chat model | 对话模型 |
| `--max-steps N` | step budget | 步数上限 |
| `--attempt-policy reflect\|none` | attempt policy | 尝试策略 |
| `--reflection-gate flags\|utility` | continuation gate | 继续判定 |
| `--tool-exposure retrieve\|all` | tool exposure | 工具暴露方式 |
| `--builtins all\|none\|NAME,...` | built-in tools to register | 启用的内置工具 |
| `--mcp-config PATH` | MCP config file | MCP 配置文件 |
| `--mcp-only` | MCP tools only | 仅使用 MCP 工具 |
| `--memory on\|off` | long-term memory | 长期记忆开关 |
| `--memory-dir DIR` | memory directory | 记忆目录 |
| `--timeout SECONDS` | run time limit | 运行时限 |
| `--request-timeout SECONDS` | chat request timeout | 对话请求超时 |
| `--output-format text\|json` | output format | 输出格式 |
| `--embedding-base-url URL` | embedding server URL | 嵌入服务地址 |
| `--embedding-model NAME` | embedding model | 嵌入模型 |
| `--serve` | start the gateway | 启动网关 |
| `--host HOST`, `--port PORT` | gateway address, port | 网关监听地址，端口 |
| `--reconfigure` | run the setup wizard again | 重新运行配置向导 |
| `--daemon install\|uninstall\|status` | background service (macOS) | 后台服务（macOS） |
| `-h`, `--help` | help | 帮助 |

以下内容始终保持英文，并使用代码字体：配置键（`chat_backend`、`request_timeout_s`、`extra_body`、…）、环境变量（`LOCALMELO_PERSIST_MEMORY`、`LOCALMELO_MEMORY_DIR`）、后端键和 `none`、工具名、`RunResult` 字段名、HTTP 路由、文件名和路径。后端显示名：MLC LLM（`mlc`）、Ollama（`ollama`）、vLLM（`vllm`）、SGLang（`sglang`）、OpenAI（`openai`）、Anthropic（`anthropic`）、Google Gemini（`gemini`）、NVIDIA（`nvidia`）。

**网站标签**

| 英文 | 中文 |
|---|---|
| Quickstart | 快速开始 |
| Architecture | 架构 |
| Roadmap | 路线图 |
| Developers | 开发者 |
| Reference ↗ | 参考文档（英文）↗ |
| GitHub ↗ | GitHub ↗ |
| On this page | 本页内容 |
| Skip to content | 跳到正文 |
| Get started | 开始使用 |
| Experimental (tag) | 实验性 |
| Offline only (tag) | 仅离线 |

**中文排版**

- 中文与拉丁字母或数字之间加空格，例如“使用 uv 安装”和“120 秒”。
- 中文句子中使用全角标点，标点前后不加空格。代码保留其 ASCII 标点。
- 产品名的大小写保持不变。
- 除标识符、产品名以及上文中在括号内附英文的术语外，不要在中文句子中混用英文名词。
- 不要在 Markdown 中把一个中文段落拆成多行：换行会显示为空格。
- 在中文网站页面中，用 `lang="en"` 标记完整的英文句子。
- 称呼读者为“你”，句子简短自然。
