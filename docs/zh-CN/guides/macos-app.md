<a id="macos-development-app"></a>

<a id="the-macos-app-development-build"></a>

# macOS 应用（开发版本）

arm64 开发应用是供构建它的这台 Mac 使用的本地原型，不是经过公证的发行版，也不是 Homebrew cask。它使用临时签名（ad hoc），因此不需要 Apple 开发者账号。构建它需要 macOS、一个 Swift 编译器、`uv`，以及在构建时获取固定版本独立 Python 和锁定依赖所需的网络访问。Swift 编译器优先使用位于 `/Applications/Xcode.app` 的正式版 Xcode；没有它时，使用 `xcode-select -p` 指向的 Command Line Tools（`xcode-select --install` 可安装它）。两者都没有时，构建会在做任何工作之前停止，并只打印一行说明。环境中设置的 `DEVELOPER_DIR` 会选择另一个开发者目录。它不会安装、启动或下载任何模型服务器。不会更改任何签名身份或 Xcode 设置。

在名为 `localmelo` 的检出目录中，一条命令即可构建应用并为当前用户安装，不需要管理员权限：

```bash
uv run python scripts/build_macos_app.py --install
```

命令会用平实的语言说明做了什么、下一步该做什么：从 Spotlight 或 `~/Applications` 打开 localmelo。窗口、程序坞和菜单都称这个应用为 localmelo；`LocalMelo.app` 只是文件名。应用被放进 `~/Applications/LocalMelo.app`，只有旧副本的 bundle id 是 `org.localmelo.app` 时才会替换它；同名的其他应用会被拒绝并保持原样。只有 `PATH` 上没有其他 `melo` 时（检出目录自己的 `.venv/bin/melo` 不算，`uv run` 会把它放在 `PATH` 最前面），才会把 `~/.local/bin/melo` 链接到应用的启动器；已有的 `melo` 绝不会被替换，如果 `~/.local/bin` 不在 `PATH` 中，输出会给出需要添加的那一行。退出 localmelo 后再次运行同一命令即可原地升级。固定版本的 Python 归档缓存在 `~/Library/Caches/LocalMelo`，每次使用都会对照其 SHA-256 校验。若要移除应用和链接（仅此而已，设置和对话仍保留在 `~/.cache/localmelo`）：

```bash
uv run python scripts/build_macos_app.py --uninstall
```

只构建、不安装时，给出一个新路径：

```bash
uv run python scripts/build_macos_app.py --output /tmp/LocalMelo.app
```

构建器拒绝覆盖已存在的输出。它把 Python 运行时和网关额外依赖复制进应用包，用上面所说的 Swift 编译器编译 Swift 宿主（摘要会说明用了哪些工具，例如 `Built with the Command Line Tools.`），把内置的 Python 编译为字节码，先对其中每个 Mach-O 文件、再对整个应用包签名，并包含可重定位的 `Contents/Resources/runtime/bin/melo`。字节码基于哈希且从不检查源文件，启动器和直接运行内置解释器都不会再写入字节码，因此签名保持有效。精确的独立解释器归档已固定版本，解压前会校验 SHA-256；wheel 的哈希则对照锁文件检查。构建工具静默运行，只在失败时才输出。

在装入依赖之前，构建器会先精简它复制来的 Python 运行时：去掉 IDLE、tkinter 和 turtle（连同它们加载的 Tcl/Tk 库）、pip、setuptools、ensurepip 和 2to3，共 36 MB，因此应用约为 147 MB，而不是 183 MB。去掉什么由运行时之下的一份固定路径清单决定（`scripts/build_macos_app.py` 中的 `UNUSED`）；`unittest`、`distutils`、`venv` 和 `pydoc` 保留，依赖会导入的所有内容也保留。清单中的路径若经由链接指向运行时之外，构建会中止。在内置解释器上运行的代码（例如 `python_exec` 工具）无法导入被去掉的部分，`python -m venv` 需要加上 `--without-pip`。

在控制台加载完成之前，窗口显示一个原生的启动视图：Melo 的画像、`Starting localmelo…`，以及在旋转指示器旁说明当前步骤的一行（`Starting the local server`、`Signing in`、`Opening your workspace`）。如果网关在 20 秒的期限内无法启动，或之后停止，同一个视图会在 `localmelo could not start` 或 `localmelo stopped` 之下说明原因，并提供 **Try Again**（启动一个新的网关）、**Open Log**（打开 `~/Library/Logs/LocalMelo/gateway.log`）和 **Quit**。网关自行退出时，视图用平实的话说明（`Its background service quit unexpectedly. Try Again starts it again; Open Log shows what happened.`）；说明原因的信号或退出状态写入日志。问题不会弹出对话框，也不会关闭应用。窗口和网页视图使用控制台的画布颜色，因此深色窗口不会闪白。

菜单栏项是 localmelo 的标志，即 Melo 的圆形图像（`docs/assets/icon_small.png`；网页控制台显示的是同一个标志），以彩色显示。它不是模板图像，系统不会给它着色，所以在浅色和深色菜单栏上看起来一样。空闲时只显示标志。有内容运行时，标志旁边显示正在运行的数量，统计控制台的对话、任务、直接运行和终端对话（与控制台页眉标签的数字相同）。当某个对话或任务在等你回答工具审批时，标志旁边会出现一个带圈的红点；深色菜单栏上的红色更亮，所以红点在两种菜单栏上都醒目。这些全部来自应用每 2 秒一次的健康检查，因此审批会在一次轮询之内显示出来。VoiceOver 和悬停提示用文字说出状态（`localmelo, nothing running`、`localmelo, 3 running` 或 `localmelo, waiting for your approval`），等待开始时还会朗读出来。菜单以正在运行的内容（`Nothing running`、`3 running`）开头；有审批在等待时，还有 **Waiting for your approval**，它会让窗口打开到那个对话或任务。其后是 **Open localmelo**、**Settings…**（应用菜单中也是 Command-逗号，打开控制台的设置）和 **Quit localmelo**。**Sign In Again** 只在它能帮上忙时才出现：连续四次登录尝试失败之后、控制台显示锁定画面时，以及开发者模式下始终出现（应用会询问页面，页面会说明自己的模式和是否被锁定）。如果应用包中缺少这个标志，菜单栏项改为显示文字 `localmelo`。

应用图标是 Melo，即 localmelo 的吉祥物，和 README 中展示的一样。它是一个提交进仓库的文件 `macos/assets/AppIcon.icns`，构建只是把它作为 `LocalMelo.icns` 复制进应用包。plist 中的 `CFBundleIconFile` 指向它，因此程序坞、Spotlight 和所有对话框（包括退出对话框）显示的都是它，而不是通用图标。菜单栏项的标志以同样的方式提交为 `macos/assets/MenuBarLogo.png` 和 `MenuBarLogo@2x.png`（18 和 36 px，圆角），同样被复制；构建中没有任何图像处理代码。

这三个文件由 `docs/assets` 中的图片生成。图标来自 `docs/assets/melo.png`（一张 400 x 534 的图片）：整张图片缩放到 macOS 图标网格中圆角底板（1024 px 中的 824 px）的高度，并在边缘渐隐进一块与图片纸张同色的底板，底板颜色取自图片的四个角，因此看不到图片的边缘。从 16 到 1024 px 的每个尺寸都用 Lanczos 滤波器直接由这张图片制作。1024 px 的尺寸把图片放大约 1.5 倍，在有更大的原图之前会略显柔和。标志来自 `docs/assets/icon_small.png`。图片改动后，在 Mac 上重新生成这些文件并提交：

```bash
uv run --with pillow python scripts/make_app_icons.py
```

Pillow 只在这一次运行中获取；它不是 localmelo 的依赖，由 macOS 自带的 `iconutil` 打包图标。

Melo，即 localmelo 的吉祥物，是一只秋田犬幼崽；把应用收起来时，Melo 会趴在你的桌面上。关闭窗口、把它最小化或隐藏应用（Command-H），屏幕右下角、程序坞上方就会出现一个小小的 Melo。它停在其他窗口之上，出现在每一个空间里（它不会要求出现在全屏应用之上），不在程序坞和 Command-Tab 中，也从不抢走你正在使用的应用的键盘。窗口回来时它就消失，无论你是从程序坞、菜单栏、聚焦搜索打开窗口，还是点击 Melo。

Melo 的样子说明 localmelo 在做什么，来自菜单栏图标用的同一个每 2 秒一次的健康检查：

| Melo | 含义 |
|---|---|
| 瘫在肚皮上，眼睛半闭 | 休息：没有任何东西在运行。这是默认状态。 |
| 抬眼望着，吐着舌头，开始工作时轻轻颠一下 | 工作中：有对话、任务、运行或终端对话在运行。 |
| 睁大闪亮的眼睛，嘴巴圆圆的，开始时轻轻跳一下 | 在等你批准。 |
| 开心地闭着眼睛、笑得张开了嘴，摇一小会儿 | 完成：一次运行刚刚顺利结束（持续 6 秒）。 |
| 眉毛耷拉着，一脸愁容 | 上一次运行没有完成（持续 10 秒），或 localmelo 已停止。 |
| 侧着、趴着或仰着睡 | 一分钟内什么都没发生；睡姿每 3 分钟换一次，碰一下 Melo 就会醒。 |

点击 Melo 会打开窗口；它在等你批准时，则打开需要你回答的那个对话或任务。可以把它拖到任何地方；它会为每块屏幕记住自己的位置，放在靠近底边处时会吸附到底边，屏幕变化时会回到屏幕内。右键（或按住 Control 点击）会出现 **Open localmelo**、**Hide Melo** 和 **Quit localmelo**。**Hide Melo** 会让它一直隐藏，重启应用后也一样，此时菜单栏菜单里会出现 **Show Melo**，这是唯一的回来方式（下次窗口被收起时，Melo 就会再出现）。它记住的内容保存在 `~/Library/Application Support/LocalMelo/pet.json`：是否隐藏、上次所在的屏幕，以及它在每块屏幕上的位置。

Melo 趴着时会慢慢呼吸，开始工作时颠一下，等待时跳一下，运行完成时摇一摇，换图片时有短暂的淡入淡出（睡姿之间更慢）。这些全部由 Core Animation 完成，应用自己不做任何绘制：没有为 Melo 运行的定时器，它的状态在收到健康检查回复、被触碰或窗口变化时重新计算。在“减弱动态效果”下（此时图片直接切换，没有淡入淡出）、低电量模式下、Mac 发热时、Melo 被遮住或隐藏时，以及屏幕休眠时，Melo 会停止动作，每种状态只显示它的静态图片。VoiceOver 会把 Melo 读作一个按钮，并说出它的状态（`Melo, resting. Nothing running.`、`Melo, working. 2 running.`、`Melo, waiting for your approval.`、`Melo, done.`、`Melo, the last run did not finish.`、`Melo, localmelo stopped.`），开始等待时它不会自己出声播报：菜单栏图标已经播报了，因此 VoiceOver 不会读两遍。

图片在应用包的 `Contents/Resources/akita` 下：九张给桌面宠物用（趴着、慵懒地趴着，以及三种睡姿），五张坐姿的用于启动视图，另有小头像和网站图标的各种尺寸，每张都有 1x 和 `@2x`。和图标一样，它们是提交进仓库的文件，位于 `macos/art/akita`（1.17 MB，透明 PNG，带一圈很细的奶油色外沿，让轮廓在深色桌面上依然看得见），构建会原样复制这个文件夹。`anchors.json` 以点为单位给出每张图片的画布、地面线和中心，因此换姿势时不会跳动。美术母版留在仓库之外；要在 Mac 上根据它们重新生成这些图片：

```bash
uv run --with pillow python scripts/build_pet_assets.py --picks /path/to/picks.json
```

`picks.json` 的形式是 `{"root": "DIR", "picks": {"lie-idle": "path/below/DIR.png", ...}}`，列出脚本中的 14 个状态。Pillow 只在这一次运行中获取，两次运行得到的字节完全相同。`GET /api/health` 现在还在 `status.recent` 中说明上一次运行是怎么结束的（见 [API 参考](../reference/api-v2.md)）；Melo 会读取它，用来显示刚结束的运行。

打开应用会用 `melo --serve --port 0` 启动它自己的网关。网关绑定一个空闲端口，并在启动信息的 `Gateway: http://127.0.0.1:PORT` 一行中打印它，应用从网关的输出中读取这一行；整个启动过程的期限为 20 秒。应用从不自己选择端口，因此不会与其他网关冲突，也不会占用其他工作保留的端口，并且不会接管已有的监听者。如果另一个网关占用了控制台存储，请先停止那个网关。做隔离的手工测试时，请使用临时 `HOME`。关闭窗口会让网关继续运行；只有在有东西会被停止时才会询问：有对话或任务正在运行或在等你回答，或应用无法判断什么在运行；没有东西可停止时，应用直接退出。退出对话框会写明将被停止的内容，例如 `Quitting stops what is running: 2 tasks and 1 chat.`（只有一项时是 `Quitting stops 1 chat.`）；直接运行和终端对话属于其他进程，会继续运行，对话框会补充 `Not affected: 1 run and 1 terminal chat.`。如果另一个网关是 melo 后台服务，`melo serve status` 可以检查它，`melo serve uninstall` 会停止并移除它。子进程输出写入 `~/Library/Logs/LocalMelo/gateway.log`。这个原型没有日志轮转。

应用会向子进程发送一个随机的启动标识符，在 `GET /` 探测时检查它，并在主页面响应时再次检查，以发现意外的端口冲突。在发送每个登录码之前，它还会重新确认子进程仍然持有该端口，因此在探测之后才抢占端口的监听者收不到任何登录码；登录码与该标识符绑定，因此只有应用自己的网关会接受它。应用的登录请求携带该标识符，所以由应用启动的网关会拒绝标识符不匹配的交换。**这是实例检查，不是身份认证：**任何本地进程都可以读取响应头并重放它。网关另外要求身份认证：对于它自己的健康检查和网页视图，应用会获取短期有效、只能使用一次的登录码，既不记录这些码，也不把 bearer 令牌放进 URL。浏览器中的代码放在 URL 片段里，控制台登录后会移除该片段。应用会验证健康 cookie，并重试没有存下任何 cookie 的登录。令牌轮换之后，它会重新登录并刷新页面；应用菜单中的 **Sign In Again** 操作（登录失败时才显示）可按需做同样的事。这些登录码来自包内的一个私有代码片段；有文档的 `melo` 命令用于此事属于 A1b 的工作，目前尚不可用。同一用户的进程和其他本地主机服务不在身份认证边界之内，工具以用户身份运行。不要把它暴露到网络上，也不要用于不受信任的任务。

控制台的文件夹标签通过一个消息处理器 `localmeloFolder` 与应用通信。应用只接受来自网关自己页面的消息（它的主框架，通过 `http` 从 `127.0.0.1` 和网关的端口发出），并且只接受它认识的请求；页面内的框架、任何其他页面和任何其他请求都会被拒绝。**Choose a folder…** 会打开系统的文件夹面板，并只回答用户选择的路径，绝不回答页面发来的路径。**Show in Finder** 只在文件夹确实存在、并且网关把它列为某个对话的文件夹（`GET /api/sessions`）或用户在面板中选择过它时才会显示；任何其他路径都会被拒绝。

应用和控制台使用常规的 localmelo 配置与会话存储，位于 `~/.cache/localmelo` 下；首次设置在网页控制台的设置对话框中完成。应用不安装也不移除单独的 launchd 服务。移除应用包之前，请先退出应用。

WebKit 集成测试会打开原生窗口，需要图形化的 macOS 会话。它们默认关闭，因此普通测试运行不会抢走焦点；请用临时 HOME 运行：

```bash
HOME="$(mktemp -d)" LOCALMELO_MACOS_UI_TESTS=1 nice -n 19 uv run pytest -q tests/test_macos_webview_integration.py
```
