# macOS development app

The arm64 development app is a local prototype for the Mac that builds it, not a notarized release or a Homebrew cask. It is signed ad hoc, so no Apple developer account is needed. It needs macOS, the release Xcode at `/Applications/Xcode.app`, `uv`, and network access to obtain the pinned standalone Python and locked dependencies at build time. It does not install, start or download a model server. No signing identities or Xcode settings are changed.

From a checkout named `localmelo`, one command builds the app and installs it for your user, with no admin rights:

```bash
uv run python scripts/build_macos_app.py --install
```

It says in plain words what it did and what to do next: open LocalMelo from Spotlight or from `~/Applications`. The window, the Dock and the menus call the app localmelo; `LocalMelo.app` is only the file name. The app goes to `~/Applications/LocalMelo.app`, replacing an earlier copy only when its bundle id is `org.localmelo.app`; any other app of that name is refused and left alone. `~/.local/bin/melo` is linked to the app's launcher only when no other `melo` is on `PATH`; an existing `melo` is never replaced, and if `~/.local/bin` is not on your `PATH` the output gives the line to add. Run the command again, after quitting LocalMelo, to upgrade in place. The pinned Python archive is cached in `~/Library/Caches/LocalMelo` and checked against its SHA-256 on every use. To remove the app and the link, and nothing else (your settings and chats stay in `~/.cache/localmelo`):

```bash
uv run python scripts/build_macos_app.py --uninstall
```

To build without installing, give a new path:

```bash
uv run python scripts/build_macos_app.py --output /tmp/LocalMelo.app
```

The builder refuses to overwrite an existing output. It copies a Python runtime and the gateway extra into the bundle, compiles the Swift host with release Xcode, compiles the bundled Python to bytecode, signs every Mach-O file inside it and then the bundle, and includes a relocatable `Contents/Resources/runtime/bin/melo`. The bytecode is hash-based and never checks the source, and neither the launcher nor a direct run of the bundled interpreter writes more of it, so the signature stays valid. The exact standalone interpreter archive is pinned and SHA-256 checked before extraction; wheel hashes are checked against the lock file. Build tools run quietly and print only when one fails.

Until the console has loaded, the window shows a native starting view: Melo's picture, `Starting localmelo…` and a
spinner beside a line that names the step (`Starting the local server`, `Signing in`, `Opening your workspace`). If the
gateway cannot start within its 20-second deadline, or stops later, the same view says why, under `localmelo could not
start` or `localmelo stopped`, and offers **Try Again**, which starts a new gateway, **Open Log**, which opens
`~/Library/Logs/LocalMelo/gateway.log`, and **Quit**. A problem never shows a dialog and never closes the app. The
window and the web view use the console's canvas colour, so a dark window does not flash white.

Opening the app starts its own gateway with `melo --serve --port 0`. The gateway binds a free port and prints it in its
startup line, `Gateway: http://127.0.0.1:PORT`, which the app reads from the gateway's output; the whole start has a
20-second deadline. The app never chooses a port, so it cannot collide with another gateway or with the ports that other work
reserves, and it does not adopt an existing listener. If another gateway holds the console store, stop that gateway
first. For isolated manual tests, use a temporary `HOME`. Closing the window leaves the gateway running;
quitting always asks before stopping the gateway and any running tasks. The menu-bar item reads `localmelo` when nothing runs
and `localmelo · 3` when three things run, counting the console's chats, tasks, direct runs and terminal chats (the count of the
console's header chip), read with the app's 2-second health check. The quit dialog names what would stop, for example
`2 tasks and 1 chat are running. Quitting stops the gateway and everything running in it.`; direct runs and terminal chats
belong to other processes and go on, which the dialog adds as `Not affected: 1 run and 1 terminal chat.` If the other gateway is the melo background
service, `melo --daemon status` checks it and `melo --daemon uninstall` stops and removes it. Child output goes to
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
signs in again and reloads; the app menu's **Sign In Again** action does the same on demand. The codes come from a
private in-bundle snippet; a documented `melo` command for this is A1b work, not yet available. Same-user processes and other localhost services remain outside the authentication boundary, and tools run as the user. Do not expose it on a network or use it with untrusted tasks.

The app and console use the regular localmelo config and session store under `~/.cache/localmelo`;
first-run setup is the web console's settings dialog. The app does not install or remove the
separate launchd service. Quit the app before removing its bundle.

The WebKit integration tests open native windows and require a graphical macOS session. They
are opt-in so a normal test run never steals focus; run them with a temporary HOME:

```bash
HOME="$(mktemp -d)" LOCALMELO_MACOS_UI_TESTS=1 nice -n 19 uv run pytest -q tests/test_macos_webview_integration.py
```
