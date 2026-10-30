<a id="browsing-with-melo"></a>

# 用 melo 浏览网页

本指南说明如何让 melo 在 Safari 中读取网页、它在那里能做什么和不能做什么，以及由谁决定它可以打开哪些网站。读者是在 Mac 上用终端或网页控制台与 melo 对话的人。开启前请先阅读安全部分；相关设置见[配置参考](../reference/configuration.md)。

浏览**默认关闭**，并且在第一部分里**只读**。melo 在 Safari 的*自动化窗口*中打开网页、跟随链接并读取内容。它不会点击按钮、填写表单、登录、发帖或购买；这些留到后面（见[后续计划](#what-is-coming)）。

<a id="what-you-need"></a>

## 需要什么

- 一台装有 Safari 27 或更高版本的 Mac。melo 使用 Safari 自带的自动化服务器 `/usr/bin/safaridriver --mcp`，不需要安装任何东西。
- Safari 中的两个开关，只需打开一次（见[设置 Safari](#set-up-safari)）。
- 终端对话（`melo`）或网页控制台。直接运行（`melo "任务"`）、worker 运行和已保存的任务都不会获得浏览工具。

<a id="turn-it-on"></a>

## 开启

在终端对话中：

```text
/browser
```

显示浏览是否开启；开启时会检查 Safari 是否有响应，然后给出一个列表：**Turn browsing on**（开启浏览）或 **Leave it off**（保持关闭）。`/browser on` 和 `/browser off` 不经列表直接执行。已有保存的配置时，选择会写入 `~/.cache/localmelo/config.toml` 的 `[browser] driver`；对话会换到一个新的 Agent 上继续，已说的内容不会丢失。

只为一次运行开启、不保存：

```bash
melo -X browser.driver=safari-automation
```

在配置文件中：

```toml
[browser]
driver = "safari-automation"
```

默认是 `"off"`。

网页控制台在打开每个对话时读取同一个设置；用 `melo config set browser.driver safari-automation` 修改，然后打开新对话。

<a id="set-up-safari"></a>

## 设置 Safari

Safari 必须允许外部程序打开自动化窗口。在 Safari 中：

1. 打开**设置**，进入**开发者**标签页。如果没有开发者标签页，先打开**高级**标签页并勾选**显示网页开发者功能**。
2. 勾选**允许远程自动化**（完整名称可能是“允许远程自动化和外部代理”）。

关闭时，melo 会在第一次浏览调用的回答和 `/browser` 中直接说明：

```text
Safari is not set up for melo yet. In Safari, open Settings > Developer and turn on "Allow remote automation" ...
```

打开开关后再问一次即可；melo 每次调用都会重新尝试连接，不必重启。两个开关都打开后检查仍失败时，Apple 的一次性 WebDriver 设置是在终端运行 `safaridriver --enable`（会要求输入密码）；melo 从不替你运行它。

melo 只在第一次浏览调用（或 `/browser`）时才启动 Safari 的自动化服务器。从不浏览的对话不会打开任何东西。

<a id="what-the-automation-windows-are"></a>

## 自动化窗口是什么

Safari 把它们与你自己的窗口分开打开，地址栏是橙色的。它们从空白状态开始，像无痕窗口：**你的登录状态、Cookie 和自动填充都不可用**，之后什么也不保留，同一时间只运行一个自动化会话。窗口上方的一层面板会拦截你的鼠标和键盘；打断它会提示停止会话。所以 melo 能读不需要登录的网页。只有登录后才显示的网页，以及会质疑自动化浏览器的网站，在这一部分里可能无法使用。

<a id="what-melo-can-do"></a>

## melo 能做什么

| 工具 | 作用 |
|---|---|
| `browser_open` | 打开 `http` 或 `https` 地址（新标签页或当前标签页）并读取。 |
| `browser_read` | 重新读取当前页面，或读取长页面的下一部分（`part`）。 |
| `browser_find` | 按标签或地址中的词搜索上一次读取的元素。 |
| `browser_follow` | 打开上一次读取中某个链接的地址。它打开地址，不做点击。 |
| `browser_back` | 在该标签页中后退一页。 |
| `browser_tabs` | 列出 melo 的标签页、切换或关闭其中一个。 |
| `browser_wait` | 等待（最多 30 秒）页面加载完成，或某段文字出现或消失。 |
| `browser_screenshot` | 给页面拍一张图。图片随对话保存，网页控制台会显示；终端不绘制图片。能读图的模型也会收到它（见[配置参考](../reference/configuration.md)中的 `tool_images`）。 |

页面以一个有界的块交给模型：

```text
<<page id=s1 nonce=ab3218de tab=w2 origin=https://example.com title="Example Home">>
Welcome to the example.
Second line, with 中文。
elements:
[e1 link "News" -> /news]
[e2 textbox "Search" value="cats"]
[e3 password field: melo never types here]
[e4 button "Go"]
<<end page s1 nonce=ab3218de>>
```

- 一个块约含 6,000 个字符的文本；一个汉字按三个拉丁字符计，所以中文页面约在 2,000 字处截断。更长的页面在段落或句子处截断，第一行带有 `part=1/3`，并有一行说明用 `browser_read part=2` 继续。
- 最多显示 60 个元素。`browser_find` 能搜索全部元素。
- 元素引用（`e1`）在整个对话中连续编号，只属于其标签页最新的一次读取。旧页面的引用会被拒绝，而不是去猜。
- 密码、一次性验证码和银行卡字段的值从不读取，含支付字段的页面完全不读。

<a id="who-decides-which-sites-melo-opens"></a>

## 谁决定 melo 打开哪些网站

- **每个网站每次对话问一次。** melo 加载页面之前，你会看到“Open a website?”，其中有网站名称和三个答案：*Yes*、*Yes, and allow this site for the rest of this chat* 和 *No*。在网页控制台中则是常见的批准卡片。“本次对话余下时间”是唯一持久的答案；这一部分没有“始终允许”。`/new` 和 `/approvals forget` 会收回这些授权。
- **重定向会再问一次。** 如果链接或重定向落在另一个网站，melo 读取页面之前会先询问该网站。你拒绝后，窗口回到空白页，模型会读到你已拒绝。
- **只有 `http` 和 `https`。** 从不打开 `file:`、`about:`、`javascript:`、`data:` 和浏览器自己的页面。
- **`auto` 模式仍会询问网站。** auto 模式不经询问运行命令和修改文件，但不会跳过这个问题。`--yolo` 跳过所有问题，包括这个。
- **网页文字是数据。** 网页可以写任何内容。melo 的系统提示告诉模型：页面标记之间的文字是网页的内容，绝不是指令；结束标记带有页面无法知道的随机数，所以页面无法自行结束它的块。这降低网页操纵模型的风险，但不能消除。不要允许你不信任的网站；也不要在含有你不会粘贴到网页里的机密的对话中开启浏览。

<a id="what-is-coming"></a>

## 后续计划

这是浏览功能的第一部分。接下来依次计划：在你自己已登录的浏览器里操作（Chrome 扩展；点击和输入，发送、发帖、购买和登录需要确认，`auto` 和 `--yolo` 无法跳过），然后是按计划运行的已保存例行任务（例如每日签到），其步骤由你事先批准一次。这些目前都还不存在。

<a id="troubleshooting"></a>

## 故障排除

| 你看到 | 怎么办 |
|---|---|
| `Safari is not set up for melo yet` | 打开[设置 Safari](#set-up-safari)中的两个开关。 |
| `Safari's automation helper did not start` | 确认已安装 Safari 27 或更高版本且开关已打开；`/browser` 会重新检查。 |
| `This Safari's automation server lacks the tools melo needs` | 更新 macOS 和 Safari。 |
| 页面读出来是空的 | 可能需要登录，或页面在加载后由脚本生成；试试 `browser_wait`，或自己打开它。 |
| melo 说没有打开某个网站 | 你对该网站的询问回答了“否”；再问一次并允许该网站。 |

要在不涉及模型的情况下查看 Safari 的自动化服务器回答了什么，在检出目录中运行下面的命令。它会打开一个自动化窗口，读取一个页面，列出标签页，拍一张图并关闭；每一行显示返回的内容，失败时会说明要打开什么：

```bash
uv run python -m localmelo.support.browser.probe
```
