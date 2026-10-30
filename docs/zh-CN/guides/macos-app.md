<a id="macos-development-app"></a>

# macOS 开发应用

arm64 开发应用是供构建它的这台 Mac 使用的本地原型，不是经过公证的发行版，也不是 Homebrew cask。它使用临时签名（ad hoc），因此不需要 Apple 开发者账号。构建它需要 macOS、位于 `/Applications/Xcode.app` 的正式版 Xcode、`uv`，以及在构建时获取固定版本独立 Python 和锁定依赖所需的网络访问。它不会安装、启动或下载任何模型服务器。不会更改任何签名身份或 Xcode 设置。

在名为 `localmelo` 的检出目录中，一条命令即可构建应用并为当前用户安装，不需要管理员权限：

```bash
uv run python scripts/build_macos_app.py --install
```

命令会用平实的语言说明做了什么、下一步该做什么：从 Spotlight 或 `~/Applications` 打开 LocalMelo。应用被放进 `~/Applications/LocalMelo.app`，只有旧副本的 bundle id 是 `org.localmelo.app` 时才会替换它；同名的其他应用会被拒绝并保持原样。只有 `PATH` 上没有其他 `melo` 时，才会把 `~/.local/bin/melo` 链接到应用的启动器；已有的 `melo` 绝不会被替换，如果 `~/.local/bin` 不在 `PATH` 中，输出会给出需要添加的那一行。退出 LocalMelo 后再次运行同一命令即可原地升级。固定版本的 Python 归档缓存在 `~/Library/Caches/LocalMelo`，每次使用都会对照其 SHA-256 校验。若要移除应用和链接（仅此而已，设置和对话仍保留在 `~/.cache/localmelo`）：

```bash
uv run python scripts/build_macos_app.py --uninstall
```

只构建、不安装时，给出一个新路径：

```bash
uv run python scripts/build_macos_app.py --output /tmp/LocalMelo.app
```

构建器拒绝覆盖已存在的输出。它把 Python 运行时和网关额外依赖复制进应用包，用正式版 Xcode 编译 Swift 宿主，把内置的 Python 编译为字节码，先对其中每个 Mach-O 文件、再对整个应用包签名，并包含可重定位的 `Contents/Resources/runtime/bin/melo`。字节码基于哈希且从不检查源文件，启动器和直接运行内置解释器都不会再写入字节码，因此签名保持有效。精确的独立解释器归档已固定版本，解压前会校验 SHA-256；wheel 的哈希则对照锁文件检查。构建工具静默运行，只在失败时才输出。

打开应用会在 `127.0.0.1:8401` 上启动它自己的网关；它不会接管已有的监听者。先停止占用控制台存储的冲突监听者或网关。在为其他工作保留了 8400-8799 的机器上，那段工作期间不得运行该应用。做隔离的手工测试时，可用 `LOCALMELO_APP_PORT` 选择另一个可用端口；请使用临时 `HOME`，并避开其他工作占用的端口。关闭窗口会让网关继续运行；退出时总会先询问，再停止网关和正在运行的任务。菜单栏项在没有任何内容运行时显示 `Melo`，有三项在运行时显示 `Melo · 3`，统计控制台的对话、任务、直接运行和终端对话（与控制台页眉标签的数字相同），随应用每 2 秒一次的健康检查读取。退出对话框会写明将被停止的内容，例如 `2 tasks and 1 chat are running. Quitting stops the gateway and everything running in it.`；直接运行和终端对话属于其他进程，会继续运行，对话框会补充 `Not affected: 1 run and 1 terminal chat.`。如果监听者是 melo 后台服务，`melo --daemon status` 可以检查它，`melo --daemon uninstall` 会停止并移除它。子进程输出写入 `~/Library/Logs/LocalMelo/gateway.log`。这个原型没有日志轮转。

应用会向子进程发送一个随机的启动标识符，在 `GET /` 探测时检查它，并在主页面响应时再次检查，以发现意外的端口冲突。在发送每个登录码之前，它还会重新确认子进程仍然持有该端口，因此在探测之后才抢占端口的监听者收不到任何登录码；登录码与该标识符绑定，因此只有应用自己的网关会接受它。应用的登录请求携带该标识符，所以由应用启动的网关会拒绝标识符不匹配的交换。**这是实例检查，不是身份认证：**任何本地进程都可以读取响应头并重放它。网关另外要求身份认证：对于它自己的健康检查和网页视图，应用会获取短期有效、只能使用一次的登录码，既不记录这些码，也不把 bearer 令牌放进 URL。浏览器中的代码放在 URL 片段里，控制台登录后会移除该片段。应用会验证健康 cookie，并重试没有存下任何 cookie 的登录。令牌轮换之后，它会重新登录并刷新页面；应用菜单中的 **Sign In Again** 操作可按需做同样的事。这些登录码来自包内的一个私有代码片段；有文档的 `melo` 命令用于此事属于 A1b 的工作，目前尚不可用。同一用户的进程和其他本地主机服务不在身份认证边界之内，工具以用户身份运行。不要把它暴露到网络上，也不要用于不受信任的任务。

应用和控制台使用常规的 localmelo 配置与会话存储，位于 `~/.cache/localmelo` 下；首次设置在网页控制台的设置对话框中完成。应用不安装也不移除单独的 launchd 服务。移除应用包之前，请先退出应用。

WebKit 集成测试会打开原生窗口，需要图形化的 macOS 会话。它们默认关闭，因此普通测试运行不会抢走焦点；请用临时 HOME 运行：

```bash
HOME="$(mktemp -d)" LOCALMELO_MACOS_UI_TESTS=1 nice -n 19 uv run pytest -q tests/test_macos_webview_integration.py
```
