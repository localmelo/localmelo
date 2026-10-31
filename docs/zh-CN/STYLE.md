# 文档风格指南

目的：编写便于查找、遵循和验证的 localmelo 文档。
适用对象：编写或审阅面向用户的文档的贡献者。

## 页面模板

以一个一级标题、一句目的和一句读者说明开头。在第一个操作之前说明前提条件和安全约束。按任务组织正文：选择、配置、运行、检查和排错。列出选项前先解释操作的作用；展示成功和失败的情况，再链接下一项任务及确切的参考页。API 契约、选项默认值、错误文本和限制应放在对应参考页，不要在每个指南中复制表格。不要把示例或基准测试观察结果写成效果保证。

## 给人阅读的页面

网站页面（`docs/en/*.html`、`docs/zh-CN/*.html`）、两份 README、指南（`docs/guides/`）和概念页（`docs/concepts/`）面向快速浏览的读者。参考页仍按上面的页面模板写：表格和列表保持精确，正文不写 PR 或 issue 编号。

### 版式

- 一节是一个标题加最多五条短要点，每条一行；指南把精确的细节放在表格或代码块里，不放在段落里。
- 段落不超过两句话。
- 用平实的词，从读者的角度写（“你可以……”）；只有讲内部结构的页面才写内部细节。
- 路线图是未完成工作的看板，分三列：现在、接下来、以后，下方有一条“已完成”区；每张卡片有标题、一行说明和状态标签（开发中、测试中或已完成），尚未开始的卡片没有标签。
- 更新按日期倒序：每天一个标题，加三到六条读者能看到的变化。
- 中文要点与英文要点一一对应，不增删事实；术语按[维护文档](../project/documentation.md)（英文）中的术语表。

### 图片和编号

- 命令放进终端块，不用裸代码块：有标题和提示符，命令、选项、取值和字符串各用一种颜色，复制按钮只复制命令本身。终端块由 `scripts/usage_blocks.py` 生成；请修改 `scripts/usage_blocks.toml`，不要改生成的标记。
- 先图后文：每个有截图、短动图或示意图的功能小节都放一张，放在标题下面；比较用表格。
- 指南改名或删除标题时，把旧标题保留为锚点（`<a id="old-heading"></a>`），使链接和清单仍然有效。
- 图片放在 `docs/assets/img/`，WebP 格式，宽度不超过 1280 像素，带替代文字；每页图片总量不超过 3 MB，中英文页面用同一组图片。
- 这些页面不写 PR 或 issue 编号；读者可能想看细节的地方，放一个指向更新日志的小链接“变更详情”。
- `docs/for-agents.md`、`docs/llms.txt` 和更新日志保持精确、简洁，保留 PR 编号。

## 示例与安全

添加或修改命令和代码示例时，逐条运行。在 PR 中记录命令、环境、预期输出和退出状态；使用临时 HOME 和工作区、端口 0 的模拟服务器，不使用凭据、真实模型、网络服务或剪贴板。标明需要真实服务器或人工操作的示例，并在可能时提供安全的模拟检查。不得掩盖工具以用户权限运行且没有沙箱的事实。

## 链接和语言

使用有意义的链接文字和稳定标题；保留现有 URL 和片段标识符。段落要简短，术语要一致，平台注意事项要准确。英文页面是原文；在同一 PR 更新中文对应页，否则加上可见的带日期过期说明，注明英文原文的修订、受影响的章节、负责人和后续 issue。代码、选项、ID 和 URL 保持一致。由另一位审阅者检查语义、术语及事实的增删，维护者再抽查。既有术语见[维护文档](../project/documentation.md)（英文）中的术语表。Docs-1 结束之前每个对应页都必须经过审阅。

<a id="interface-glossary"></a>

## 界面术语

网页控制台、macOS App 和网关写给页面的文字都用下表的译法，文档也用同样的词。词库是 `support/gateway/static/i18n/zh-Hans.json`，键与 `en.json` 一一对应。

- 不翻译：命令（如 `melo console`）、路径、模型名、级别 P0-P5、HTTP 路由、状态值和配置键。语言选项用各自语言的名字：English、简体中文。
- 计数写成数字加量词：“3 个任务”“2 条笔记”“1 次工具调用”；中文不分单复数。
- 按钮用动词：“保存”“取消”“重试”；macOS 菜单沿用系统的译法：拷贝、粘贴、全选、撤销、重做。
- 中文与数字或拉丁字母之间加空格，句中用全角标点；见[维护文档](../project/documentation.md)（英文）中的排版规则。

| English | 中文 |
|---|---|
| main chat, side chat | 主对话，旁支对话 |
| chat, task, terminal chat | 对话，任务，终端对话 |
| worker run, Workers | worker 运行，Worker |
| panel, tab, sidebar | 面板，标签页，侧边栏 |
| Details | 详情 |
| Memory (tab), note | 记忆，笔记 |
| About you, Preferences, People, Projects, Decisions | 关于你，偏好，人物，项目，决定 |
| Remember across chats, Forget, Undo | 跨对话记忆，忘记，撤销 |
| approval, Waiting for your approval | 审批，在等你批准 |
| allowed, denied, session grant | 已允许，已拒绝，会话授权 |
| tool call, step, attempt, reflection | 工具调用，步，尝试，反思 |
| Done, Stopped, Failed, Interrupted, Timed out | 完成，已停止，失败，已中断，超时 |
| Working, Thinking, Running | 工作中，正在思考，运行中 |
| model server, embedding model | 模型服务器，嵌入模型 |
| folder, default folder, workspace | 文件夹，默认文件夹，工作区 |
| Settings, developer mode | 设置，开发者模式 |
| sign in, gateway | 登录，网关 |
| menu-bar item, activity pill, Melo on your desktop | 菜单栏项，活动胶囊，桌面上的 Melo |
| Scheduled (tab), chore, check-in | 定时，定时事项，例行查看 |
| priority, chat level | 优先级，对话级别 |
| Appearance, Theme, Language, Automatic | 外观，主题，语言，自动 |
| Copy, Try again | 拷贝，重试 |
