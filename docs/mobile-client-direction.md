# Apple client direction

Status: architecture evaluation, not an implemented mobile product. The current
web workspace has desktop and responsive browser coverage. No native iOS or
macOS application exists, and physical iPhone behavior has not been tested.

## Recommendation

Start with a mobile web/PWA client using the existing workspace, then build
SwiftUI clients for iPhone and Mac once the remote-control API is stable. Keep
one authoritative localmelo runtime on the Mac: model inference, tools, tasks,
conversation records, and execution permissions belong to the gateway. Clients
submit work, display progress, and request cancellation.

A PWA reuses the current interface and can be added to the iPhone Home Screen.
Home Screen web apps support Web Push on iOS/iPadOS 16.4 and later after an
explicit user interaction and notification permission. These features still
require implementation; a responsive page alone does not provide them.
[WebKit: Home Screen web apps and Web Push][web-push]

SwiftUI is the longer-term direction for platform-specific navigation, Mac
windows and keyboard commands, sharing, and other system integrations. An Apple
multiplatform project can share its API client and state model while adapting
views and interactions to each platform.
[Apple: Configuring a multiplatform app][swiftui]

## Staged implementation

1. **Durable control API.** Persist conversations, messages, run status, and
   results in SQLite. Submit a task with an idempotency key and return
   `202 Accepted` with a `run_id`; run it independently of the initiating HTTP
   request. Reconnecting or retrying must not execute the same request twice.
   On gateway restart, mark interrupted work explicitly rather than implying
   that execution resumed automatically.
2. **Shared live state.** Use REST for commands and SSE for ordered task, tool,
   and output events. Persist event IDs and support cursor-based replay after a
   disconnect. Keep polling as a fallback, refresh on foreground activation,
   and display connection state. Add a CLI gateway-client mode so commands that
   should appear on other devices use the same conversation and run IDs.
3. **Mobile web/PWA.** Add the manifest, icons, installation guidance, safe-area
   layout, keyboard handling, and optional completion notifications. Test on a
   physical iPhone, including lock/unlock, network changes, reconnection, and
   duplicate submission. Any offline shell should show its disconnected state;
   local model execution still requires the Mac.
4. **SwiftUI clients.** Share the protocol and state model across iOS and macOS,
   beginning with pairing, chat history, submission, cancellation, and recovery.
   Add platform integrations after these core flows are dependable.

Phone background execution must not keep a server task alive. Apple does not
promise timely or guaranteed delivery of background notifications; notifications
should prompt a state refresh, not serve as the authoritative task record.
[Apple: Background updates][background]

## Connectivity and access

Use a private Tailscale network with Tailscale Serve HTTPS as the default remote
connection. This gives enrolled devices one address on local Wi-Fi, cellular,
and other networks while the gateway can remain bound to localhost. Restrict
access to the intended user/devices through tailnet access rules. Serve is a
private sharing mechanism; public exposure is outside this proposal.
[Tailscale: Serve][serve]

Pair and authenticate clients before enabling remote execution. Support device
revocation and authorize sessions and operations by workspace. Native clients
should keep credentials in Keychain; browser clients should use protected
session cookies with appropriate origin/CSRF checks. If trusting Serve identity
headers, accept them only through the trusted local proxy path, as its docs
require. Keep the oMLX model endpoint private to the Mac.

Optional LAN-only discovery can advertise a specific Bonjour service. A native
client needs a local-network usage description and declared Bonjour service
types. Discovery identifies a candidate server; pairing must still establish
trust. [Apple: Local network privacy][local-network]

The Mac must remain online and available for inference. macOS offers an option
to prevent automatic sleep while connected to power with the display off.
Network wake supports shared resources and should not be treated as a guarantee
that arbitrary gateway or model workloads can be reached while asleep. No power
or network settings have been changed. [Apple: Sleep and wake settings][sleep]

## Gaps in the current implementation

- The CLI constructs its own Agent; it does not share web conversations.
- Web conversation metadata and transcripts live in memory. Reload recovery
  works while the gateway runs, but does not reconstruct chats after restart.
- The playground still has a 12-session capacity and idle eviction. Durable
  conversation records need a lifecycle separate from live Agent resources.
- There is no client authentication, device pairing, or session ownership model.
- Run requests wait for execution to finish; there is no detached submission API,
  durable event log, SSE stream, or idempotent retry protocol.
- The web client currently polls every 2.5 seconds while visible. Native clients,
  PWA installation assets, and completion push delivery are not implemented.

[web-push]: https://webkit.org/blog/13878/web-push-for-web-apps-on-ios-and-ipados/
[swiftui]: https://developer.apple.com/documentation/xcode/configuring-a-multiplatform-app-target
[background]: https://developer.apple.com/documentation/usernotifications/pushing-background-updates-to-your-app
[serve]: https://tailscale.com/docs/features/tailscale-serve
[local-network]: https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy
[sleep]: https://support.apple.com/en-gb/guide/mac-help/mchle41a6ccd/mac
