import Foundation

struct GatewayHealth {
    let store: String
    let error: String
    let reason: String
    let running: Int
}

func gatewayHealth(_ data: Data) -> GatewayHealth? {
    guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          body["api"] as? Int == 2,
          let sessions = body["sessions"] as? [String: Any],
          let store = sessions["store"] as? String,
          let error = sessions["store_error"] as? String,
          let running = sessions["running"] as? Int, running >= 0 else { return nil }
    return GatewayHealth(store: store, error: error, reason: sessions["store_reason"] as? String ?? "", running: running)
}

/// What runs now, from `status.running` of `GET /api/health`.
struct RunningCounts {
    let chats: Int
    let tasks: Int
    let runs: Int
    let terminal: Int
    var total: Int { chats + tasks + runs + terminal }
}

func runningCounts(_ data: Data) -> RunningCounts? {
    guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let status = body["status"] as? [String: Any],
          let running = status["running"] as? [String: Any],
          let chats = running["chats"] as? Int, chats >= 0,
          let tasks = running["tasks"] as? Int, tasks >= 0,
          let runs = running["runs"] as? Int, runs >= 0,
          let terminal = running["terminal"] as? Int, terminal >= 0 else { return nil }
    return RunningCounts(chats: chats, tasks: tasks, runs: runs, terminal: terminal)
}

/// What the menu-bar item says: nothing known yet, idle, working, or waiting for the person.
enum MenuBarState: Equatable {
    case unknown
    case idle
    case running(Int)
    case waiting(running: Int)
}

/// A chat or task that waits for an answer to a tool approval, from `status.waiting` of `GET /api/health`.
struct WaitingItem: Equatable {
    let kind: String
    let id: String

    /// The console route that opens it.
    var route: String { kind == "task" ? "#/tasks/\(id)" : "#/s/\(id)" }
}

private let routeCharacters = CharacterSet(
    charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-")

/// The list is empty when it is missing or damaged. An entry whose kind or id the console would not
/// open is left out, so a route is always safe to hand to the page.
func waitingItems(_ data: Data) -> [WaitingItem] {
    guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let status = body["status"] as? [String: Any],
          let list = status["waiting"] as? [[String: Any]] else { return [] }
    return list.compactMap { entry in
        guard let kind = entry["kind"] as? String, kind == "chat" || kind == "task",
              let id = entry["id"] as? String, (1...128).contains(id.count),
              id.unicodeScalars.allSatisfy(routeCharacters.contains) else { return nil }
        return WaitingItem(kind: kind, id: id)
    }
}

func waitingRoute(_ data: Data) -> String? { waitingItems(data).first?.route }

/// Waiting wins over running. A held task is not running, so a wait can come with a count of 0.
func menuBarState(_ data: Data) -> MenuBarState {
    guard let counts = runningCounts(data) else { return .unknown }
    if !waitingItems(data).isEmpty { return .waiting(running: counts.total) }
    return counts.total > 0 ? .running(counts.total) : .idle
}

/// The count beside the mark: what the console's header chip counts as running, and nothing at 0.
func menuTitle(_ state: MenuBarState) -> String {
    switch state {
    case .running(let n), .waiting(running: let n): return n > 0 ? "\(n)" : ""
    case .unknown, .idle: return ""
    }
}

/// What VoiceOver reads for the item, and what hovering shows.
func menuBarLabel(_ state: MenuBarState) -> String {
    switch state {
    case .unknown: return "localmelo"
    case .idle: return "localmelo, nothing running"
    case .running(let n): return "localmelo, \(n) running"
    case .waiting: return "localmelo, waiting for your approval"
    }
}

/// The first line of the item's menu; nil while nothing is known.
func menuStatusLine(_ state: MenuBarState) -> String? {
    switch state {
    case .unknown: return nil
    case .idle: return "Nothing running"
    case .running(let n), .waiting(running: let n): return n > 0 ? "\(n) running" : "Nothing running"
    }
}

private func counted(_ n: Int, _ noun: String) -> String {
    "\(n) \(noun)\(n == 1 ? "" : "s")"
}

/// The quit dialog's text: what quitting stops, in the words people use for it. Quitting stops the
/// gateway and so its chats and tasks; runs and terminal chats belong to other processes and go on.
func quitMessage(_ counts: RunningCounts?, statusKnown: Bool) -> String {
    let generic = "Quitting stops anything that is running."
    guard let counts = counts else {
        return statusKnown ? generic : "Could not check what is running. " + generic
    }
    let stops = [(counts.tasks, "task"), (counts.chats, "chat")]
        .filter { n, _ in n > 0 }.map { n, noun in counted(n, noun) }
    let goes = [(counts.runs, "run"), (counts.terminal, "terminal chat")]
        .filter { n, _ in n > 0 }.map { n, noun in counted(n, noun) }
    var text = generic
    if stops.count == 1 {
        text = "Quitting stops \(stops[0])."
    } else if stops.count > 1 {
        text = "Quitting stops what is running: \(stops.joined(separator: " and "))."
    }
    if !goes.isEmpty { text += " Not affected: \(goes.joined(separator: " and "))." }
    return text
}

/// What the console page says about itself that the app's menu needs: `data-mode` and `data-locked` on
/// its root element (set by `refreshMode` and `lock` in workspace.js).
struct PageState: Equatable {
    var developer = false
    var locked = false
}

let pageStateScript = """
JSON.stringify({mode: document.documentElement.dataset.mode || '', locked: document.documentElement.dataset.locked === '1'})
"""

/// The page's state from `pageStateScript`'s answer; anything else (no page yet, damaged text) is the default.
func pageState(_ json: String?) -> PageState {
    guard let data = json?.data(using: .utf8),
          let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return PageState() }
    return PageState(developer: body["mode"] as? String == "developer", locked: body["locked"] as? Bool == true)
}

/// Sign-in attempts that may fail in a row (they retry twice a second) before the menu offers "Sign In Again".
let signInFailureLimit = 4

/// "Sign In Again" is for a sign-in that failed: the app's own attempts keep failing, or the console shows
/// its Locked screen. Developer mode always has it.
func offersSignIn(failures: Int, page: PageState) -> Bool {
    page.developer || page.locked || failures >= signInFailureLimit
}

func ownsListener(_ pids: [Int32]?, child: Int32) -> Bool {
    guard let pids = pids else { return false }
    return !pids.isEmpty && pids.allSatisfy { $0 == child }
}

func acceptsInstance(_ header: String?, expected: String, childRunning: Bool) -> Bool {
    childRunning && header == expected
}

/// The port in the gateway's startup line `Gateway: http://HOST:PORT`, where
/// HOST may be an IPv6 address in brackets. Any other line gives nil.
func gatewayPort(fromLine line: String) -> Int? {
    let marker = "Gateway: http://"
    let text = line.trimmingCharacters(in: .whitespacesAndNewlines)
    guard text.hasPrefix(marker) else { return nil }
    let address = text.dropFirst(marker.count)
    guard let colon = address.lastIndex(of: ":") else { return nil }
    let host = address[..<colon], digits = address[address.index(after: colon)...]
    guard !host.isEmpty, host.hasPrefix("[") ? host.hasSuffix("]") : !host.contains(":"),
          !digits.isEmpty, digits.count <= 5, digits.allSatisfy({ ("0"..."9").contains($0) }),
          let port = Int(digits), port > 0, port <= 65535 else { return nil }
    return port
}

/// The child's output as it arrives. The startup line can span two reads, so
/// it keeps the unfinished line; it reports the port once and then ignores the rest.
struct StartupOutput {
    private var pending = ""
    private var done = false

    mutating func append(_ data: Data) -> Int? {
        guard !done else { return nil }
        let lines = (pending + String(decoding: data, as: UTF8.self)).components(separatedBy: .newlines)
        pending = lines.last.map { $0.utf8.count > 4096 ? "" : $0 } ?? ""
        guard let port = lines.dropLast().lazy.compactMap({ gatewayPort(fromLine: $0) }).first else { return nil }
        done = true
        pending = ""
        return port
    }
}

func isGatewayURL(_ url: URL, port: Int) -> Bool {
    url.scheme == "http" && url.host == "127.0.0.1" && url.port == port
        && url.user == nil && url.password == nil
}

func isGatewayPage(_ url: URL, port: Int) -> Bool {
    isGatewayURL(url, port: port) && url.path == "/"
}

func storeFailureMessage(_ health: GatewayHealth) -> String {
    let hint = health.reason == "owned"
        ? " Stop the other gateway and try again."
        : " Check the gateway log and session store permissions, then try again."
    return "The gateway session store is unavailable: \(health.error).\(hint)"
}

/// The line under "Starting localmelo…", from what the app knows so far.
func startStepText(portKnown: Bool, ready: Bool) -> String {
    if ready { return "Opening your workspace" }
    return portKnown ? "Signing in" : "Starting the local server"
}

/// What the window offers when the gateway cannot start or stops, in button order.
enum FailureAction: CaseIterable {
    case tryAgain, openLog, quit

    var title: String {
        switch self {
        case .tryAgain: return "Try Again"
        case .openLog: return "Open Log"
        case .quit: return "Quit"
        }
    }
}

struct FailureNotice {
    let title: String
    let reason: String
    let actions: [FailureAction]
}

/// The window's failure text. `ready` is true once the console had loaded.
func failureNotice(reason: String, ready: Bool) -> FailureNotice {
    FailureNotice(title: ready ? "localmelo stopped" : "localmelo could not start",
                  reason: reason, actions: FailureAction.allCases)
}

/// What the window says when the gateway quits on its own: no signal or status, which mean nothing to
/// most people and go to the log. `detail` is a fact worth saying, such as who holds the port now.
func gatewayExitMessage(detail: String = "") -> String {
    "Its background service quit unexpectedly.\(detail) Try Again starts it again; Open Log shows what happened."
}

/// The same exit as the log records it.
func gatewayExitLogLine(status: Int32, signalled: Bool) -> String {
    signalled ? "The gateway was terminated by signal \(status)." : "The gateway exited (status \(status))."
}

/// What the console's folder chip may ask of the app.
enum FolderRequest: Equatable {
    case choose
    case reveal(String)
}

/// The request a page message holds, or nil: an unknown action, or a path that is not absolute or `~/`, has a NUL or is over 4096 bytes.
func folderRequest(_ body: Any) -> FolderRequest? {
    guard let message = body as? [String: Any], let action = message["action"] as? String else { return nil }
    switch action {
    case "choose":
        return .choose
    case "reveal":
        guard let path = message["path"] as? String, path.utf8.count <= 4096, !path.contains("\0"),
              path.hasPrefix("/") || path.hasPrefix("~/") else { return nil }
        return .reveal(path)
    default:
        return nil
    }
}

/// Whether a message came from the gateway's own page: its main frame, over http, from 127.0.0.1 and the gateway's port.
func acceptsBridgeMessage(scheme: String, host: String, port: Int, isMainFrame: Bool, gatewayPort: Int) -> Bool {
    isMainFrame && gatewayPort > 0 && scheme == "http" && host == "127.0.0.1" && port == gatewayPort
}

/// The `workspace` of every chat in `GET /api/sessions`: the folders that count as a chat's own.
func chatFolders(_ data: Data) -> [String] {
    guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let sessions = body["sessions"] as? [[String: Any]] else { return [] }
    return sessions.compactMap { $0["workspace"] as? String }
}

/// The folder to show in Finder for `requested`: an existing directory that is one of `allowed`, as real paths.
func revealTarget(_ requested: String, allowed: [String]) -> URL? {
    func real(_ path: String) -> String {
        let home = ProcessInfo.processInfo.environment["HOME"] ?? NSHomeDirectory()
        return URL(fileURLWithPath: path.hasPrefix("~/") ? home + path.dropFirst() : path).resolvingSymlinksInPath().path
    }
    var isDirectory: ObjCBool = false
    let target = real(requested)
    guard allowed.map(real).contains(target),
          FileManager.default.fileExists(atPath: target, isDirectory: &isDirectory), isDirectory.boolValue else { return nil }
    return URL(fileURLWithPath: target, isDirectory: true)
}
