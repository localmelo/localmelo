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

/// The console route of a chat or task, nil for any other kind and for an id the console would not open, so a
/// route is always safe to hand to the page.
func consoleRoute(kind: String, id: String) -> String? {
    guard kind == "chat" || kind == "task", (1...128).contains(id.count),
          id.unicodeScalars.allSatisfy(routeCharacters.contains) else { return nil }
    return WaitingItem(kind: kind, id: id).route
}

/// The list is empty when it is missing or damaged. An entry whose kind or id the console would not
/// open is left out.
func waitingItems(_ data: Data) -> [WaitingItem] {
    guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let status = body["status"] as? [String: Any],
          let list = status["waiting"] as? [[String: Any]] else { return [] }
    return list.compactMap { entry in
        guard let kind = entry["kind"] as? String, let id = entry["id"] as? String,
              consoleRoute(kind: kind, id: id) != nil else { return nil }
        return WaitingItem(kind: kind, id: id)
    }
}

/// `model_server.reachable` of `GET /api/health`: false when the settings' local model server did not answer, nil
/// while it is not known or for a cloud model.
func modelServerReachable(_ data: Data) -> Bool? {
    guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let server = body["model_server"] as? [String: Any] else { return nil }
    return server["reachable"] as? Bool
}

/// A line of the health reply's text on one line: runs of white space (and line breaks) become one space, the ends
/// are trimmed and a long one is cut to `limit` characters.
func plainLine(_ text: String?, limit: Int) -> String {
    let words = (text ?? "").split(whereSeparator: { $0.isWhitespace || $0.isNewline })
    return String(words.joined(separator: " ").prefix(limit))
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
    case .idle: return L10n.t("app.bar.idle", "localmelo, nothing running")
    case .running(let n): return L10n.t("app.bar.running", "localmelo, {n} running", ["n": String(n)])
    case .waiting: return L10n.t("app.bar.waiting", "localmelo, waiting for your approval")
    }
}

/// The first line of the item's menu; nil while nothing is known.
func menuStatusLine(_ state: MenuBarState) -> String? {
    switch state {
    case .unknown: return nil
    case .idle: return L10n.t("app.bar.nothing", "Nothing running")
    case .running(let n), .waiting(running: let n):
        return n > 0 ? L10n.t("app.bar.count", "{n} running", ["n": String(n)]) : L10n.t("app.bar.nothing", "Nothing running")
    }
}

/// The quit dialog's text: what quitting stops, in the words people use for it. Quitting stops the
/// gateway and so its chats and tasks; runs and terminal chats belong to other processes and go on.
func quitMessage(_ counts: RunningCounts?, statusKnown: Bool) -> String {
    let generic = L10n.t("app.quit.generic", "Quitting stops anything that is running.")
    guard let counts = counts else {
        return statusKnown ? generic : L10n.t("app.quit.unknown", "Could not check what is running. {generic}", ["generic": generic])
    }
    let and = L10n.t("app.quit.and", " and ")
    var stops: [String] = [], goes: [String] = []
    if counts.tasks > 0 { stops.append(L10n.plural("app.quit.tasks", counts.tasks, one: "{n} task", other: "{n} tasks")) }
    if counts.chats > 0 { stops.append(L10n.plural("app.quit.chats", counts.chats, one: "{n} chat", other: "{n} chats")) }
    if counts.runs > 0 { goes.append(L10n.plural("app.quit.runs", counts.runs, one: "{n} run", other: "{n} runs")) }
    if counts.terminal > 0 {
        goes.append(L10n.plural("app.quit.terminal", counts.terminal, one: "{n} terminal chat", other: "{n} terminal chats"))
    }
    var text = generic
    if stops.count == 1 {
        text = L10n.t("app.quit.stopsOne", "Quitting stops {what}.", ["what": stops[0]])
    } else if stops.count > 1 {
        text = L10n.t("app.quit.stopsAll", "Quitting stops what is running: {what}.", ["what": stops.joined(separator: and)])
    }
    if !goes.isEmpty {
        text += L10n.t("app.quit.notAffected", " Not affected: {what}.", ["what": goes.joined(separator: and)])
    }
    return text
}

/// Whether quitting needs the person's say: a chat or task would stop, something waits for an answer, or
/// the health reply did not say. Runs and terminal chats belong to other processes and go on.
func quitNeedsConfirmation(_ health: Data?) -> Bool {
    guard let data = health, let counts = runningCounts(data) else { return true }
    return counts.chats + counts.tasks > 0 || !waitingItems(data).isEmpty
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

// A gateway the app did not start has no app instance: its pages send no header.
func acceptsInstance(_ header: String?, expected: String, childRunning: Bool) -> Bool {
    childRunning && (header ?? "") == expected
}

/// The gateway the app talks to: its own child, or a running one it attached to.
protocol GatewayChild: AnyObject {
    var isRunning: Bool { get }
    var processIdentifier: Int32 { get }
    /// Started by this app: stopped on Quit and on a failure. An attached one never is.
    var owned: Bool { get }
}

extension Process: GatewayChild {
    var owned: Bool { true }
}

final class AttachedGateway: GatewayChild {
    let processIdentifier: Int32
    init(pid: Int32) { processIdentifier = pid }
    var isRunning: Bool { kill(processIdentifier, 0) == 0 || errno == EPERM }
    var owned: Bool { false }
}

/// What `~/.cache/localmelo/gateway/endpoint.json` says: the running gateway that holds
/// the main chat. Only one on 127.0.0.1, which the app's pages can reach, is used.
struct GatewayEndpoint: Equatable {
    let pid: Int32
    let port: Int
    let instance: String
    let id: String
}

func gatewayEndpoint(_ data: Data?) -> GatewayEndpoint? {
    guard let data = data,
          let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let pid = body["pid"] as? Int, pid > 0, pid <= Int(Int32.max),
          let port = body["port"] as? Int, port > 0, port <= 65535,
          body["host"] as? String == "127.0.0.1",
          let id = body["id"] as? String, !id.isEmpty, id.count <= 64 else { return nil }
    let instance = body["instance"] as? String ?? ""
    return GatewayEndpoint(pid: Int32(pid), port: port, instance: instance, id: id)
}

enum GatewayStart: Equatable {
    case attach(GatewayEndpoint)
    case start
}

/// Attach to the gateway the file names when its process lives; else start one.
func gatewayStart(_ endpoint: GatewayEndpoint?, alive: (Int32) -> Bool) -> GatewayStart {
    guard let endpoint = endpoint, alive(endpoint.pid) else { return .start }
    return .attach(endpoint)
}

/// `GET /` of that port must carry the file's id: else the file is stale.
func attachConfirmed(status: Int?, gatewayHeader: String?, endpoint: GatewayEndpoint) -> Bool {
    status == 200 && gatewayHeader == endpoint.id
}

func quitTitle(attached: Bool) -> String {
    attached ? L10n.t("app.menu.quitAttached", "Quit localmelo (the gateway keeps running)")
        : L10n.t("app.menu.quit", "Quit localmelo")
}

/// An attached gateway that ends was stopped elsewhere (`melo serve stop`, say).
var attachedStoppedMessage: String {
    L10n.t("app.fail.attachedStopped", "The gateway it was using has stopped. Try Again starts a new one.")
}

/// Another process listens on the port the gateway was to use.
func portHeldMessage(_ port: Int) -> String {
    L10n.t("app.fail.portHeld", "Port {port} is held by another process. Stop the existing listener and try again.",
           ["port": String(port)])
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
        ? L10n.t("app.fail.storeOwned", " Stop the other gateway and try again.")
        : L10n.t("app.fail.storeCheck", " Check the gateway log and session store permissions, then try again.")
    return L10n.t("app.fail.store", "The gateway session store is unavailable: {error}.{hint}", ["error": health.error, "hint": hint])
}

/// The line under "Starting localmelo…", from what the app knows so far.
func startStepText(portKnown: Bool, ready: Bool) -> String {
    if ready { return L10n.t("app.start.opening", "Opening your workspace") }
    return portKnown ? L10n.t("app.start.signingIn", "Signing in") : L10n.t("app.start.server", "Starting the local server")
}

/// What the window offers when the gateway cannot start or stops, in button order.
enum FailureAction: CaseIterable {
    case tryAgain, openLog, quit

    var title: String {
        switch self {
        case .tryAgain: return L10n.t("app.action.tryAgain", "Try Again")
        case .openLog: return L10n.t("app.action.openLog", "Open Log")
        case .quit: return L10n.t("app.action.quit", "Quit")
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
    FailureNotice(title: ready ? L10n.t("app.fail.stopped", "localmelo stopped") : L10n.t("app.fail.couldNotStart", "localmelo could not start"),
                  reason: reason, actions: FailureAction.allCases)
}

/// What the window says when the gateway quits on its own: no signal or status, which mean nothing to
/// most people and go to the log. `detail` is a fact worth saying, such as who holds the port now.
func gatewayExitMessage(detail: String = "") -> String {
    L10n.t("app.fail.exited", "Its background service quit unexpectedly.{detail} Try Again starts it again; Open Log shows what happened.",
           ["detail": detail])
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
