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

/// The menu-bar title: the plain name at 0 or when the count is unknown.
func menuTitle(_ counts: RunningCounts?) -> String {
    guard let counts = counts, counts.total > 0 else { return "Melo" }
    return "Melo · \(counts.total)"
}

private func counted(_ n: Int, _ noun: String) -> String {
    "\(n) \(noun)\(n == 1 ? "" : "s")"
}

/// The quit dialog's text. Quitting stops the gateway and so its chats and
/// tasks; runs and terminal chats belong to other processes and go on.
func quitMessage(_ counts: RunningCounts?, statusKnown: Bool) -> String {
    let generic = "Quitting stops the gateway and any running tasks."
    guard let counts = counts else {
        return statusKnown ? generic : "The task status is unavailable. " + generic
    }
    let stops = [(counts.tasks, "task"), (counts.chats, "chat")]
        .filter { n, _ in n > 0 }.map { n, noun in counted(n, noun) }
    let goes = [(counts.runs, "run"), (counts.terminal, "terminal chat")]
        .filter { n, _ in n > 0 }.map { n, noun in counted(n, noun) }
    var text = generic
    if !stops.isEmpty {
        let verb = counts.tasks + counts.chats == 1 ? "is" : "are"
        text = "\(stops.joined(separator: " and ")) \(verb) running. "
            + "Quitting stops the gateway and everything running in it."
    }
    if !goes.isEmpty { text += " Not affected: \(goes.joined(separator: " and "))." }
    return text
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

func gatewayExitMessage(status: Int32, signalled: Bool) -> String {
    signalled ? "The gateway was terminated by signal \(status)." : "The gateway exited (status \(status))."
}
