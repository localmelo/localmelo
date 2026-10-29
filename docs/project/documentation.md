# Maintaining the docs

How the documentation and the website are laid out, published, changed and
translated. For the index of pages, see [the documentation](../README.md); for
the development setup, see
[CONTRIBUTING.md](https://github.com/localmelo/localmelo/blob/main/CONTRIBUTING.md).

## Layout

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

- Each page has a Chinese twin at the same path under `docs/zh-CN/`, except
  `for-agents.md`, `llms.txt` and `index.html` (see [Translations](#translations)).
  Root-level stubs have none.
- The website has six pages in each language: Home, Quickstart,
  Architecture, Roadmap, Developers and Memory design. Each English page has
  a Chinese twin with the same element `id`s, the same sections in the same
  order, and identical code blocks.
- Memory design keeps the maintainer's memory design word for word, in
  English and in Chinese. Restyle it if you need to, but change its words
  only with the maintainer.
- Root-level compatibility stubs keep links at old paths and fragments;
  `README.md`, `for-agents.md` and `llms.txt` remain useful entry points.
- The website's Quickstart mirrors `start/quickstart.md`. The other website
  pages link to the Markdown pages on GitHub.

## Publishing

GitHub Pages publishes the `docs/` folder of the `main` branch. Pages also
renders the Markdown files in `docs/` with Jekyll. Links between pages in
`docs/` are relative; links from documentation to files outside `docs/` use
absolute URLs (`https://github.com/localmelo/localmelo/blob/main/…`), so they
work on GitHub and on the rendered copies alike.

To preview the website:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory docs
```

Then open `http://127.0.0.1:8080/en/`.

The source Markdown pages and their legacy stubs are in the repository. The
experimental builder is the foundation for a later site migration; it does
not replace the published HTML website. After `uv sync --extra dev`,
run the following from the repository root to build into a new temporary
folder and compare a second build with it:

```bash
SITE="$(mktemp -d)"
uv run --offline --locked --extra dev python scripts/build_docs.py --output "$SITE"
uv run --offline --locked --extra dev python scripts/build_docs.py --output "$SITE" --check
```

The output must be empty and outside the checkout. `--check` compares the
entire isolated output tree byte for byte, including assets, and reports stale
or extra files. The preview copies existing HTML unchanged and renders Markdown
with raw HTML escaped; it is not a complete replacement site. In particular,
old-path redirects, split-page fragment routing and links in copied HTML are
not migrated into the preview yet. The old Markdown paths instead have readable
stubs with a link for each historical fragment; these work without JavaScript.
Missing Chinese pages are not generated. The existing English and Chinese
quickstart HTML pages keep their own URLs and IDs; their future conversion must
map each legacy fragment to an existing destination in both languages. In
particular, `#json-output` and `#serve-a-model-with-omlx` do not both map
unchanged to the new English quickstart. Do not commit the
preview. See the [English style guide](../STYLE.md) and
[Chinese style guide](../zh-CN/STYLE.md).

The docs tests check the manifest and every tracked link and anchor. They also
check that each Chinese twin listed in `tests/docs/test_manifest.py` keeps its
English anchors, code blocks and inline code. Run them with
`uv run pytest tests/docs -q`.

In `docs/manifest.json`, a page's `fragments` maps each old source path to
its migration rule. `"same"` sends every fragment to that language's output
page with the same ID; a single output path instead sends every fragment to
that page with the same ID. Neither rule needs an edit for a new heading.
A split page uses an explicit old-ID-to-`page.html#ID` map: adding a heading
requires choosing its destination and adding it to that map. The docs tests
report the source and missing anchors; keep known retired IDs in the map too.
`tests/docs/legacy_fragments.json` records the pre-move Markdown anchors and
where each stub link must lead; update it deliberately when preserving another
historical fragment, not when merely adding a new heading to an unsplit page.

## Add or change a page

- Add a new Markdown page to the index in `docs/README.md` and
  `docs/zh-CN/README.md`, and add its Chinese twin at the same path under
  `docs/zh-CN/` in the same pull request. List both in `docs/manifest.json`.
  Add it to the documentation table of the top-level `README.md` and
  `README.zh-CN.md` only if a new user needs it; the table stays short and
  ends with a link to the index.
- Add a new website page in both languages, and to the navigation of every
  website page. Memory design is the exception: it is a design record, and
  the Architecture and Roadmap pages link to it instead.
- Check every command you document by running it.
- Keep `for-agents.md` in step with every change to behaviour, options,
  output or feature status, and add new pages to `llms.txt`. It is written
  for agents: dense, exact and short.

## Translations

- English is the source. Change the Chinese counterpart in the same pull
  request as the English page.
- The Chinese twins of the Markdown pages are in `docs/zh-CN/`. The website's
  Chinese pages are `docs/zh-CN/*.html`, and `README.zh-CN.md` is the Chinese
  project README.
- `for-agents.md` and `llms.txt` stay English-only by design: they are terse
  entries for AI agents. `index.html` also stays English-only: it redirects to
  `en/index.html` and links to `zh-CN/index.html`, the Chinese home page.
- Chinese pages mark links to English-only pages with （英文）.
- Keep code blocks identical in both languages; comments in shared code
  blocks are in English.

## Translation glossary

Values in code font stay in English in both languages. On its first use on a
page, a term may carry the English in parentheses, as in 嵌入（embedding）.

**Product and parts**

| English | 中文 |
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

**Run statuses** (the value stays in English)

| Status | English gloss | 中文释义 |
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

**Options** (the option stays in English)

| Option | Concept | 中文 |
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

These always stay in English, in code font: config keys (`chat_backend`,
`request_timeout_s`, `extra_body`, …), environment variables
(`LOCALMELO_PERSIST_MEMORY`, `LOCALMELO_MEMORY_DIR`), backend keys and
`none`, tool names, `RunResult` field names, HTTP routes, file names and
paths. Backend display names: MLC LLM (`mlc`), Ollama (`ollama`), vLLM
(`vllm`), SGLang (`sglang`), OpenAI (`openai`), Anthropic (`anthropic`),
Google Gemini (`gemini`), NVIDIA (`nvidia`).

**Website labels**

| English | 中文 |
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

**Chinese typography**

- Put a space between Chinese text and Latin letters or digits, as in
  “使用 uv 安装” and “120 秒”.
- Use full-width punctuation in Chinese sentences, with no spaces around it.
  Code keeps its ASCII punctuation.
- Keep the capitalization of product names.
- Do not mix English nouns into Chinese sentences, except identifiers,
  product names and the terms above that carry the English in parentheses.
- Do not break a Chinese paragraph across lines in Markdown: the line break
  shows as a space.
- In a Chinese website page, mark a full English sentence with `lang="en"`.
- Address the reader as 你, and write short, natural sentences.
