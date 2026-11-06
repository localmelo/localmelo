import Foundation

/// What the console's Browser panel may ask of the app.
enum BrowserRequest: Equatable {
    case popOut(BrowserSource)
}

/// The characters of a chat id, as the console route takes them.
private let chatIDCharacters = CharacterSet(
    charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-")

/// The request a page message holds, or nil: `{"action": "popOut", "session": "<id>"}` for a chat's browser, with `gen`,
/// `run` and `owner` added for a running task's (its exact source), and nothing else.
func browserRequest(_ body: Any) -> BrowserRequest? {
    guard let message = body as? [String: Any], message["action"] as? String == "popOut",
          let session = message["session"] as? String else { return nil }
    let parts = ["gen", "run", "owner"].filter { message[$0] != nil }
    guard message.count == 2 + parts.count, parts.allSatisfy({ message[$0] is String }),
          let source = BrowserSource(session: session, owner: message["owner"] as? String, gen: message["gen"] as? String,
                                     run: message["run"] as? String) else { return nil }
    return .popOut(source)
}

/// The page that shows one source's browser alone (`#/live/<id>`, with the source's `gen`, `run` and `owner` for a task's), or
/// nil for a port that is none.
func browserLiveURL(_ source: BrowserSource, port: Int) -> URL? {
    guard (1...65535).contains(port),
          let encoded = source.session.addingPercentEncoding(withAllowedCharacters: chatIDCharacters) else { return nil }
    return URL(string: "http://127.0.0.1:\(port)/#/live/\(encoded)" + (source.query.isEmpty ? "" : "?" + source.query))
}

/// Whether the pop-out window's web view may go to *url*. Its main frame stays on the gateway's own page and its
/// subframes on the gateway; it never goes to another site, and a link is never opened elsewhere on its behalf.
func browserWindowAllows(_ url: URL, port: Int, isMainFrame: Bool) -> Bool {
    guard port > 0 else { return false }
    return isMainFrame ? isGatewayPage(url, port: port) : isGatewayURL(url, port: port)
}
