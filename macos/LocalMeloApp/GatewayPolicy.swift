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

func ownsListener(_ pids: [Int32]?, child: Int32) -> Bool {
    guard let pids = pids else { return false }
    return !pids.isEmpty && pids.allSatisfy { $0 == child }
}

func acceptsInstance(_ header: String?, expected: String, childRunning: Bool) -> Bool {
    childRunning && header == expected
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

func gatewayExitMessage(status: Int32, signalled: Bool) -> String {
    signalled ? "The gateway was terminated by signal \(status)." : "The gateway exited (status \(status))."
}
