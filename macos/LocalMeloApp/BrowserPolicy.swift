import Foundation

/// What the console's Browser panel may ask of the app.
enum BrowserRequest: Equatable {
    case popOut(String)
}

/// The characters of a chat id, as the console route takes them.
private let chatIDCharacters = CharacterSet(
    charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-")

private func isChatID(_ id: String) -> Bool {
    (1...128).contains(id.count) && id.unicodeScalars.allSatisfy(chatIDCharacters.contains)
}

/// The request a page message holds, or nil: `{"action": "popOut", "session": "<chat id>"}` and nothing else.
func browserRequest(_ body: Any) -> BrowserRequest? {
    guard let message = body as? [String: Any], message.count == 2, message["action"] as? String == "popOut",
          let session = message["session"] as? String, isChatID(session) else { return nil }
    return .popOut(session)
}

/// The page that shows one chat's browser alone, or nil for an id the console would not open or a port that is none.
func browserLiveURL(session: String, port: Int) -> URL? {
    guard isChatID(session), (1...65535).contains(port),
          let encoded = session.addingPercentEncoding(withAllowedCharacters: chatIDCharacters) else { return nil }
    return URL(string: "http://127.0.0.1:\(port)/#/live/\(encoded)")
}

/// Whether the pop-out window's web view may go to *url*. Its main frame stays on the gateway's own page and its
/// subframes on the gateway; it never goes to another site, and a link is never opened elsewhere on its behalf.
func browserWindowAllows(_ url: URL, port: Int, isMainFrame: Bool) -> Bool {
    guard port > 0 else { return false }
    return isMainFrame ? isGatewayPage(url, port: port) : isGatewayURL(url, port: port)
}
