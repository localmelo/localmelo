import Foundation

struct GatewayHealth {
    let store: String
    let error: String
    let running: Int
}

func gatewayHealth(_ data: Data) -> GatewayHealth? {
    guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          body["api"] as? Int == 2,
          let sessions = body["sessions"] as? [String: Any],
          let store = sessions["store"] as? String,
          let error = sessions["store_error"] as? String,
          let running = sessions["running"] as? Int, running >= 0 else { return nil }
    return GatewayHealth(store: store, error: error, running: running)
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
