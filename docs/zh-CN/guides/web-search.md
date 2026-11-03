<a id="searching-the-web-with-melo"></a>

# 用 melo 搜索网页

用一个开关，让每个对话和每个任务都能在网上查资料。开启前请先看清哪些内容会离开你的电脑；相关设置见[配置参考](../reference/configuration.md)。

- 网页搜索**默认关闭**。
- 开启后，对话和任务会获得 `web_search` 工具：melo 把关键词交给公开的搜索引擎，读回一份结果列表，每条有标题、地址和一小段摘要。
- 它需要核实时才会搜索：遇到不认识的词、同一步失败两次，或者会变化的事实（日期、价格、版本、新闻）。
- 要读某条结果时，它用 `web_fetch`，和读任何网页一样，所以你没提到过的网站仍然会先问你。

<a id="turn-it-on"></a>

## 开启

在网页控制台或 App 里，打开 **设置 → 对话和文件夹 → 网页搜索**。开关下面的一行说明了它的作用：melo 搜索用的关键词会离开这台电脑。

在终端里：

```bash
melo config set web.search true
```

只在这一次运行中开启，不保存：

```bash
melo -X web.search=true
```

- 这个设置对终端对话、控制台和 App 的每个对话以及每个任务都有效。对话在打开时读取设置，所以改完后请新开一个对话。
- 直接运行（在命令行里直接给出任务）从不获得它。
- 关闭的方式相同：用开关、`melo config set web.search false`，或删掉 `[web]` 表。

<a id="what-leaves-your-computer"></a>

## 哪些内容会离开你的电脑

- 只有搜索的关键词，最多 300 个字符。普通请求不带 cookie、不登录，User-Agent 里是 melo 自己的名字。
- 搜索引擎看得到关键词和这台 Mac 的网络地址。对话里的其他内容都不会发出去。
- 开关打开后，搜索视为你已经同意，所以普通搜索不会再问。
- 关键词看起来像机密（密钥、令牌、验证码、银行卡号或证件号），或者像粘贴来的文字（很长，或分成多行，像从私人笔记或文件里复制出来的）时，会先在审批卡片上问你。你的回答只对这一次搜索有效，之后的搜索不受影响，即使你选了“在这个对话里允许”。
- 任务会问拥有它的那个对话。没有人可问时（完全没有批准策略，例如用 `--yolo` 或库调用），这样的搜索不会发出，对话和任务都一样。
- 浏览器方式同样会把关键词发出去：Safari 加载搜索引擎结果页时，关键词就在页面地址里，所以每种方式下关键词都会离开这台电脑。

<a id="where-the-results-come-from"></a>

## 结果从哪里来

melo 按下表的顺序逐个尝试，用第一个给出贴合关键词的结果的方式。回答里总会写明是哪种方式给出的，以及前面的方式为什么没有给出。

| 顺序 | 方式 | 何时使用 |
|---|---|---|
| 1 | Bing 的结果订阅源，普通请求 | 始终。很快，约五分之一秒。 |
| 2 | Safari 自动化窗口里的 DuckDuckGo | 只在[浏览](browsing.md)开启时。结果最好；普通请求会被它拒绝。 |
| 3 | 百度，普通请求 | 始终。中英文关键词都可以，约两秒。 |

- 要求验证（验证码、等待页、报错）的方式会被跳过，改试下一个。
- 订阅源可能对不认识的客户端返回读起来正常、却答非所问的结果。melo 会检查前五条结果里至少有两条和关键词有共同的词，通不过检查的方式不会被当成答案。如果所有方式都通不过，它给出最接近的结果，并说明这些结果可能答非所问。
- 同一次搜索里，一个搜索引擎最多问一次，两次请求至少间隔两秒。
- Safari 方式用浏览工具自己的检查来读取页面：跳到另一个网站或本地地址、或页面里有本地地址的框架，都会被拒绝且不读取，也不会为此询问你。搜索引擎自己的网站同样不会询问，因为开启搜索就是允许它。
- 只有开启浏览时才会用 Safari 窗口。用 `/browser on` 开启（见[用 melo 浏览网页](browsing.md)）；没有开启时，第二种方式直接略过，也不会提到它。

<a id="how-a-search-shows"></a>

## 搜索如何显示

在对话和任务页面里，一次搜索只显示一行：进行中是*正在搜索网页：“asyncio TaskGroup”*，完成后是*搜索了网页：“asyncio TaskGroup”*，里面是关键词（其中的机密会被隐藏），完成后还有搜索引擎和结果数量。模型读到的是：

```text
Searched the web for “python asyncio TaskGroup” with Bing: 3 results.
<<page id=s1 nonce=ab3218de type=search way=bing-rss results=3>>
1. asyncio — Asynchronous I/O
   https://docs.python.org/3/library/asyncio.html
   asyncio is a library to write concurrent code using the async/await syntax.
2. …
<<end page s1 nonce=ab3218de>>
[Bing asked for a check (HTTP 429); Baidu answered.]
```

- 结果放在和 `web_fetch` 读取的网页相同的标记之间。结束标记带有任何结果都无从知道的随机数，所以结果不能自己结束这个块；模型被告知标记之间的文字是数据，绝不是命令。
- 块前面的一行和后面的说明是 melo 自己写的句子。搜索引擎在结果之外写的任何内容都不会到达模型。
- 摘要最多保留 240 个字符，地址最多 300 个。

<a id="developer-settings"></a>

## 开发者设置

`[web]` 下还有两个键，在开发者模式和 `melo config keys web` 中可见：

| 键 | 默认值 | 含义 |
|---|---|---|
| `search_order` | `[]`（即上面的顺序） | 要尝试的方式及其顺序：`bing-rss`、`browser`、`baidu`。不写某个名字，就永远不会用它。 |
| `search_count` | `6` | 一次搜索显示多少条结果，1 到 8。一次调用可以要求更少。 |

```toml
[web]
search = true
search_order = ["bing-rss", "browser", "baidu"]
search_count = 6
```

<a id="if-it-does-not-work"></a>

## 不工作时

- **“No search way gave results.”** 所有方式都被拒绝或连不上，消息里会逐个写明。过一分钟再试，或者开启浏览，让 Safari 这条路来回答。
- **结果答非所问。** 块下面的说明会这样告诉你。换个说法再问，或者打开一条结果看看。
- **工具不见了。** 确认 `melo config get web.search` 显示 `true`，并且对话是在改完之后打开的。

具体规则见[工具](../reference/tools.md#searching-the-web)。
