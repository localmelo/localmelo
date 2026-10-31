<a id="macos-development-app"></a>

<a id="the-macos-app-development-build"></a>

# macOS 应用（开发版本）

控制台外面的一个原生窗口、一个菜单栏项，以及桌面上的 Melo。它是供构建它的这台 Mac 使用的本地原型：使用临时签名（ad hoc），因此不需要 Apple 开发者账号；它没有经过公证，也不是 Homebrew cask。

![应用在自己的窗口里显示的控制台：一个对话，旁边是面板。](../../assets/img/panel.webp)

<a id="install"></a>

## 安装

- 你需要 macOS、`uv`、构建时的网络访问，以及一个 Swift 编译器：位于 `/Applications/Xcode.app` 的正式版 Xcode，没有它时用 Command Line Tools（`xcode-select --install`）。
- 两者都没有时，构建会立即停止并只打印一行说明；环境中设置的 `DEVELOPER_DIR` 会选择另一个开发者目录。
- 构建不会安装、启动或下载任何模型服务器，也不会更改任何签名身份或 Xcode 设置。

在名为 `localmelo` 的检出目录中，一条命令即可构建应用并为当前用户安装，不需要管理员权限：

```bash
uv run python scripts/build_macos_app.py --install
```

- 从 Spotlight 或 `~/Applications` 打开 localmelo：应用是 `~/Applications/LocalMelo.app`，只有旧副本的 bundle id 是 `org.localmelo.app` 时才会替换它。
- 只有 `PATH` 上没有其他 `melo` 时（检出目录自己的 `.venv/bin/melo` 不算），才会把 `~/.local/bin/melo` 链接到应用的启动器；已有的 `melo` 绝不会被替换。
- 升级时先退出 localmelo，再运行同一命令；固定版本的 Python 缓存在 `~/Library/Caches/LocalMelo`。

若要移除应用和链接（仅此而已，设置和对话仍保留在 `~/.cache/localmelo`）：

```bash
uv run python scripts/build_macos_app.py --uninstall
```

只构建、不安装时，给出一个新路径；构建器从不覆盖已存在的路径：

```bash
uv run python scripts/build_macos_app.py --output /tmp/LocalMelo.app
```

固定版本的 Python 归档和 wheel 都会按哈希校验。构建会去掉 Python 运行时中用不到的部分（36 MB；应用约 147 MB）。在内置解释器上运行的代码（例如 `python_exec` 工具）无法导入它们，`python -m venv` 需要加上 `--without-pip`。

<a id="the-window"></a>

## 窗口

- 启动视图显示 Melo、`Starting localmelo…` 和当前步骤：`Starting the local server`、`Signing in`、`Opening your workspace`。
- 网关在 20 秒内无法启动，或之后停止，视图会说明原因，并提供 **Try Again**、**Open Log** 和 **Quit**。
- 日志是 `~/Library/Logs/LocalMelo/gateway.log`；问题不会弹出对话框，也不会关闭应用。
- 首次设置在控制台的设置对话框中完成；应用使用 `~/.cache/localmelo` 里常规的配置和对话。

<a id="the-menu-bar-item"></a>

## 菜单栏项

| 你看到 | 含义 |
|---|---|
| 只有标志 | 没有东西在运行。 |
| 标志旁边有一个数字 | 有这么多个对话、任务、直接运行和终端对话在运行。 |
| 带圈的红点 | 有对话或任务在等你回答审批。 |

- 菜单列出正在运行的内容、**Waiting for your approval**（打开那个对话或任务）、**Open localmelo**、**Settings…** 和 **Quit localmelo**。
- **Sign In Again** 只在它能帮上忙时才出现：连续四次登录失败之后、显示锁定画面时，以及开发者模式下始终出现。
- 应用每 2 秒检查一次，因此审批会在一次轮询之内显示；VoiceOver 和悬停提示用文字说出同样的内容。

<a id="melo-on-your-desktop"></a>

## 桌面上的 Melo

![秋田犬 Melo 从休息中起身，开始工作。](../../assets/img/pet-smooth.webp)

关闭窗口、把它最小化或隐藏应用（Command-H），屏幕右下角、程序坞上方就会躺着一个小小的 Melo；窗口回来时它就消失。

| Melo | 含义 |
|---|---|
| 瘫在肚皮上，眼睛半闭 | 休息：没有任何东西在运行。 |
| 抬眼望着，吐着舌头 | 工作中：有对话、任务、运行或终端对话在运行。 |
| 睁大闪亮的眼睛，轻轻一跳 | 在等你批准。 |
| 开心地闭着眼睛，尾巴摇起来 | 完成：一次运行刚刚顺利结束（持续 6 秒）。 |
| 眉毛耷拉着 | 上一次运行没有完成（持续 10 秒），或 localmelo 已停止。 |
| 睡着了，头顶飘起一个小小的 “z” | 一分钟内什么都没发生；碰一下 Melo 就会醒。 |

- 点击 Melo 会打开窗口，或打开在等你的那个对话或任务；可以把它拖到任何地方，它会为每块屏幕记住自己的位置。
- 右键（或按住 Control 点击）会出现 **Open localmelo**、**Hide Melo**、**On the desktop** 和 **Quit localmelo**；菜单栏菜单里的 **Show Melo** 会把它请回来。
- 它停在其他窗口之上，出现在每一个空间里，不在程序坞和 Command-Tab 中，也从不抢走键盘。
- 它的状态保存在 `~/Library/Application Support/LocalMelo/pet.json`。
- 在“减弱动态效果”下、低电量模式下、Mac 发热时或 Melo 被隐藏时，它不再动，只显示静态图片。

<a id="the-activity-pill"></a>

## 活动胶囊

![Melo 的活动胶囊：工作中、有多个在运行、在等你批准、完成、已停止、模型服务器连不上。](../../assets/img/pill.webp)

窗口收起、而又有东西在运行或在等你时，一个小小的胶囊会取代 Melo，用一行话说 Melo 正在做什么。

- 它用平实的话说正在运行的内容（“Reading example.com”“Running the tests”）；**+1** 表示还有几个在运行。有审批在等你时，它会变成琥珀色，并写着“Waiting for your approval”。
- 它显示“Done: …” 6 秒、“Stopped: …” 10 秒，模型服务器连不上时显示“Model server not reachable”。然后 Melo 回来，“Done” 之后会摇尾巴。
- 点击它会打开它所说的对话或任务（审批则是在等你的那个）。像拖 Melo 一样拖它：它占用 Melo 的位置并保存。指向它并点 **×**，就在下一个任务开始之前把它隐藏。
- 菜单栏菜单和 Melo 的菜单里的 **On the desktop** 可选 **Melo and the activity pill**（默认）、**Melo only** 或 **Nothing**。**Hide Melo** 等于 **Nothing**，**Show Melo** 会恢复你原来的选择。
- 它只在工作时呼吸；在“减弱动态效果”下、低电量模式下、Mac 发热时或被盖住时保持不动。VoiceOver 读出一个按钮：“Melo is reading example.com. Opens the chat.”

<a id="quit-and-the-gateway"></a>

## 退出与网关

- 打开应用会用 `melo --serve --port 0` 启动它自己的网关：一个空闲端口，从不占用其他工作保留的端口，也不会接管已有的监听者。
- 关闭窗口会让网关继续运行；只有在有东西会被停止时，退出才会询问，例如 `Quitting stops what is running: 2 tasks and 1 chat.`
- 直接运行和终端对话属于其他进程，会继续运行：`Not affected: 1 run and 1 terminal chat.`
- 如果另一个网关占用了控制台存储，请先停止它；后台服务用 `melo serve status` 检查，用 `melo serve uninstall` 移除。
- 移除应用包之前请先退出应用；做隔离的手工测试时，请使用临时 `HOME`。

<a id="safety"></a>

## 安全

- 应用发给网关的启动标识符是实例检查，**不是身份认证**：任何本地进程都可以读取并重放它。
- 网关仍然要求登录：应用获取短期有效、只能使用一次的登录码，既不记录它们，也不把 bearer 令牌放进 URL。
- 同一用户的进程和其他本地主机服务不在这条边界之内，工具以你的身份运行；不要把网关暴露到网络上，也不要用于不受信任的任务。
- 控制台的文件夹标签通过一个处理器 `localmeloFolder` 与应用通信，只接受来自网关自己地址的页面；**Show in Finder** 只显示网关列出的或你选择过的文件夹。

<a id="for-developers"></a>

## 开发者

应用图标、菜单栏标志和桌面图片都是提交进仓库的文件（`macos/assets`、`macos/art/akita`），构建只是复制它们；源图片改动后，在 Mac 上重新生成：

```bash
uv run --with pillow python scripts/make_app_icons.py
```

```bash
uv run --with pillow python scripts/build_pet_assets.py --picks /path/to/picks.json
```

- `picks.json` 的形式是 `{"root": "DIR", "picks": {"lie-idle": "path/below/DIR.png", ...}}`，列出脚本中的 14 个状态；Pillow 只在这一次运行中获取。
- 网页控制台显示同样的图片：`--console-only` 会不用美术母版，直接把它们再复制到 `support/gateway/static/akita`（见[网页控制台指南](web-console.md#melo-in-the-page)）。
- `GET /api/health` 还在 `status.recent` 中说明上一次运行是怎么结束的（见 [API 参考](../reference/api-v2.md)）；Melo 会读取它。

WebKit 集成测试会打开原生窗口，需要图形化的 macOS 会话，因此默认关闭；请用临时 HOME 运行：

```bash
HOME="$(mktemp -d)" LOCALMELO_MACOS_UI_TESTS=1 nice -n 19 uv run pytest -q tests/test_macos_webview_integration.py
```
