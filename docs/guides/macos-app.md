# macOS development app

The arm64 development app is a local prototype for the Mac that builds it, not a notarized release or a Homebrew cask. It is signed ad hoc, so no Apple developer account is needed. It needs macOS, a Swift compiler, `uv`, and network access to obtain the pinned standalone Python and locked dependencies at build time. The Swift compiler is the release Xcode at `/Applications/Xcode.app` when that is installed, and otherwise the Command Line Tools that `xcode-select -p` names (`xcode-select --install` installs them). With neither, the build stops before it does any work and prints one line that says so. A `DEVELOPER_DIR` set in the environment chooses another developer folder. It does not install, start or download a model server. No signing identities or Xcode settings are changed.

From a checkout named `localmelo`, one command builds the app and installs it for your user, with no admin rights:

```bash
uv run python scripts/build_macos_app.py --install
```

It says in plain words what it did and what to do next: open localmelo from Spotlight or from `~/Applications`. The window, the Dock and the menus call the app localmelo; `LocalMelo.app` is only the file name. The app goes to `~/Applications/LocalMelo.app`, replacing an earlier copy only when its bundle id is `org.localmelo.app`; any other app of that name is refused and left alone. `~/.local/bin/melo` is linked to the app's launcher only when no other `melo` is on `PATH` (the checkout's own `.venv/bin/melo`, which `uv run` puts first on `PATH`, does not count); an existing `melo` is never replaced, and if `~/.local/bin` is not on your `PATH` the output gives the line to add. Run the command again, after quitting localmelo, to upgrade in place. The pinned Python archive is cached in `~/Library/Caches/LocalMelo` and checked against its SHA-256 on every use. To remove the app and the link, and nothing else (your settings and chats stay in `~/.cache/localmelo`):

```bash
uv run python scripts/build_macos_app.py --uninstall
```

To build without installing, give a new path:

```bash
uv run python scripts/build_macos_app.py --output /tmp/LocalMelo.app
```

The builder refuses to overwrite an existing output. It copies a Python runtime and the gateway extra into the bundle, compiles the Swift host with the compiler named above (the summary says which tools it used, for example `Built with the Command Line Tools.`), compiles the bundled Python to bytecode, signs every Mach-O file inside it and then the bundle, and includes a relocatable `Contents/Resources/runtime/bin/melo`. The bytecode is hash-based and never checks the source, and neither the launcher nor a direct run of the bundled interpreter writes more of it, so the signature stays valid. The exact standalone interpreter archive is pinned and SHA-256 checked before extraction; wheel hashes are checked against the lock file. Build tools run quietly and print only when one fails.

Before the dependencies go in, the builder trims the Python runtime it copied. IDLE, tkinter and turtle with the Tcl/Tk libraries they load, pip, setuptools, ensurepip and 2to3 are left out, 36 MB in all, so the app is about 147 MB instead of 183 MB. A fixed list of paths below the runtime decides (`UNUSED` in `scripts/build_macos_app.py`); `unittest`, `distutils`, `venv` and `pydoc` stay, and so does everything the dependencies import. A listed path that leads out of the runtime through a link stops the build. Code that runs on the bundled interpreter, such as the `python_exec` tool, cannot import what was left out, and `python -m venv` needs `--without-pip`.

Until the console has loaded, the window shows a native starting view: Melo's picture, `Starting localmelo…` and a
spinner beside a line that names the step (`Starting the local server`, `Signing in`, `Opening your workspace`). If the
gateway cannot start within its 20-second deadline, or stops later, the same view says why, under `localmelo could not
start` or `localmelo stopped`, and offers **Try Again**, which starts a new gateway, **Open Log**, which opens
`~/Library/Logs/LocalMelo/gateway.log`, and **Quit**. When the gateway quits on its own, the view says so in plain
words (`Its background service quit unexpectedly. Try Again starts it again; Open Log shows what happened.`); the signal
or exit status that tells why goes to the log. A problem never shows a dialog and never closes the app. The
window and the web view use the console's canvas colour, so a dark window does not flash white.

The menu-bar item is the localmelo logo, Melo's round picture (`docs/assets/icon_small.png`; the web console shows the same logo), in colour. It is not a template image, so the system does not tint it and it looks the same on a light bar and a dark bar. Idle, it is the logo alone. While something runs, the number running sits beside it, counting the console's chats, tasks, direct runs and terminal chats (the count of the console's header chip). While a chat or a task waits for your answer to a tool approval, a red dot with a ring appears beside the logo; the red is lighter on a dark bar, so the dot stands out on both. All of it comes from the app's 2-second health check, so an approval shows within one poll. VoiceOver and the hover tooltip say it in words (`localmelo, nothing running`, `localmelo, 3 running` or `localmelo, waiting for your approval`), and the start of a wait is announced. The menu starts with what runs (`Nothing running`, `3 running`) and, while an approval waits, **Waiting for your approval**, which opens the window on that chat or task. Below them are **Open localmelo**, **Settings…** (Command-comma in the app menu too; it opens the console's settings) and **Quit localmelo**. **Sign In Again** joins them only when it can help: after four sign-in attempts in a row have failed, while the console shows its Locked screen, and always in developer mode (the app asks the page, which says its mode and whether it is locked). If the logo is missing from the bundle, the item shows the text `localmelo` instead.

The app's icon is Melo, the localmelo mascot, as the README shows it. It is a committed file, `macos/assets/AppIcon.icns`, and the build only copies it into the bundle as `LocalMelo.icns`. The plist's `CFBundleIconFile` names it, so the Dock, Spotlight and every dialog, the quit dialog included, show it instead of the generic icon. The menu-bar item's logo is committed the same way, as `macos/assets/MenuBarLogo.png` and `MenuBarLogo@2x.png` (18 and 36 px, rounded corners), and copied too; the build has no image code.

The three files come from the pictures in `docs/assets`. The icon is `docs/assets/melo.png`, a 400 x 534 picture: the whole picture, scaled to the height of the rounded tile of the macOS icon grid (824 of 1024 px) and fading into a tile of its own paper colour, which comes from the picture's four corners, so no edge of the picture shows. Every size from 16 to 1024 px is made from the picture itself with a Lanczos filter. The 1024-px size shows the picture about 1.5 times larger than it is, so it is a little soft until larger artwork arrives. The logo is `docs/assets/icon_small.png`. When a picture changes, make the files again on a Mac and commit them:

```bash
uv run --with pillow python scripts/make_app_icons.py
```

Pillow is fetched for that one run; it is not a dependency of localmelo, and `iconutil`, which ships with macOS, packs the icon.

Opening the app starts its own gateway with `melo --serve --port 0`. The gateway binds a free port and prints it in its
startup line, `Gateway: http://127.0.0.1:PORT`, which the app reads from the gateway's output; the whole start has a
20-second deadline. The app never chooses a port, so it cannot collide with another gateway or with the ports that other work
reserves, and it does not adopt an existing listener. If another gateway holds the console store, stop that gateway
first. For isolated manual tests, use a temporary `HOME`. Closing the window leaves the gateway running;
quitting always asks before stopping the gateway and any running tasks. The quit dialog names what would stop, for example
`Quitting stops what is running: 2 tasks and 1 chat.` (`Quitting stops 1 chat.` for one); direct runs and terminal chats
belong to other processes and go on, which the dialog adds as `Not affected: 1 run and 1 terminal chat.` If the other gateway is the melo background
service, `melo serve status` checks it and `melo serve uninstall` stops and removes it. Child output goes to
`~/Library/Logs/LocalMelo/gateway.log`. This prototype has no log rotation.

The app sends a random launch identifier to its child and checks it on a `GET /` probe, and again on main-frame page
responses, to catch accidental port collisions. It also re-checks that the child still owns the port immediately
before it sends each login code, so a listener that takes the port after the probe receives none, and the code is
bound to the launch identifier, so only the app's own gateway accepts it. Its login request carries the identifier,
so a gateway started by the app refuses a mismatched exchange. **This is an instance check, not
authentication:** any local process can read the response header and replay it. The gateway separately requires
authentication: for its own health checks and web view the app obtains short-lived, single-use login codes without
logging them or placing the bearer token in a URL. The browser code is carried in a URL fragment, which the console
removes after sign-in. It verifies the health cookie and retries a login that stores none. After a token rotation it
signs in again and reloads; the app menu's **Sign In Again** action, which shows when sign-in has failed, does the same on demand. The codes come from a
private in-bundle snippet; a documented `melo` command for this is A1b work, not yet available. Same-user processes and other localhost services remain outside the authentication boundary, and tools run as the user. Do not expose it on a network or use it with untrusted tasks.

The console's folder chip reaches the app through one message handler, `localmeloFolder`. The app takes a
message only from the gateway's own page (its main frame, over `http` from `127.0.0.1` on the gateway's port)
and only a request it knows; a frame in the page, any other page and any other request is refused.
**Choose a folder…** opens the system's folder panel and answers with the path the person picked, never one
from the page. **Show in Finder** shows an existing folder only when the gateway lists it as a chat's
(`GET /api/sessions`) or the person picked it in the panel; any other path is refused.

The app and console use the regular localmelo config and session store under `~/.cache/localmelo`;
first-run setup is the web console's settings dialog. The app does not install or remove the
separate launchd service. Quit the app before removing its bundle.

The WebKit integration tests open native windows and require a graphical macOS session. They
are opt-in so a normal test run never steals focus; run them with a temporary HOME:

```bash
HOME="$(mktemp -d)" LOCALMELO_MACOS_UI_TESTS=1 nice -n 19 uv run pytest -q tests/test_macos_webview_integration.py
```
