# macOS development app

The arm64 development bundle is an unsigned local prototype, not a notarized release or a
Homebrew cask. It needs macOS, the release Xcode at `/Applications/Xcode.app`, `uv`, and network
access to obtain the pinned standalone Python and locked dependencies at build time. It does not
install, start or download a model server. No signing identities or Xcode settings are changed.

From a checkout named `localmelo`, build to a new path:

```bash
uv run python scripts/build_macos_app.py --output /tmp/LocalMelo.app
```

The builder refuses to overwrite an existing output. It copies a Python runtime and the gateway
extra into the bundle, compiles the Swift host with release Xcode, and includes a relocatable
`Contents/Resources/runtime/bin/melo`. The exact standalone interpreter archive is pinned and
SHA-256 checked before extraction; wheel hashes are checked against the lock file.

Opening the app starts an owned gateway on `127.0.0.1:8401`; it does not adopt an existing
listener. Stop a conflicting listener or gateway holding the console store first. For isolated
manual tests, `LOCALMELO_APP_PORT` can select another available port; use a temporary `HOME` and
avoid ports reserved by other work. Closing the window leaves the gateway running; quitting always
asks before stopping the gateway and any running tasks. If the listener is a melo background service,
`melo --daemon status` checks it and `melo --daemon uninstall` stops and removes it. Child output goes to
`~/Library/Logs/LocalMelo/gateway.log`. This prototype has no log rotation.

The app sends a random launch identifier to its child and checks it on main-frame page responses
to catch accidental port collisions. **This is an instance check, not authentication:** any local
process can read the response header and replay it. Malicious same-user processes are outside this
development prototype's threat model; the gateway has no authentication and tools run as the
user. Do not expose it on a network or use it with untrusted tasks.

The app and console use the regular localmelo config and session store under `~/.cache/localmelo`;
first-run setup is the web console's settings dialog. The app does not install or remove the
separate launchd service. Quit the app before removing its bundle.

The WebKit integration tests open native windows and require a graphical macOS session. They
are opt-in so a normal test run never steals focus; run them with a temporary HOME:

```bash
HOME="$(mktemp -d)" LOCALMELO_MACOS_UI_TESTS=1 nice -n 19 uv run pytest -q tests/test_macos_webview_integration.py
```
