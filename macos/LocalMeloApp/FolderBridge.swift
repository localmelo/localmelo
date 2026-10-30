import WebKit

/// The console's one native hand: the page can ask for a folder, and for a folder to be shown in Finder. A message
/// counts only from the gateway's own page and only if the policy knows its request; anything else is refused.
/// The answers come from `choose` and `reveal`, never from the message.
final class FolderBridge: NSObject, WKScriptMessageHandlerWithReply {
    static let name = "localmeloFolder"
    private let gatewayPort: () -> Int
    private let choose: (@escaping (String?) -> Void) -> Void
    private let reveal: (String, @escaping (Bool) -> Void) -> Void

    init(gatewayPort: @escaping () -> Int, choose: @escaping (@escaping (String?) -> Void) -> Void,
         reveal: @escaping (String, @escaping (Bool) -> Void) -> Void) {
        self.gatewayPort = gatewayPort
        self.choose = choose
        self.reveal = reveal
    }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage,
                               replyHandler: @escaping (Any?, String?) -> Void) {
        let origin = message.frameInfo.securityOrigin
        guard acceptsBridgeMessage(scheme: origin.protocol, host: origin.host, port: origin.port,
                                   isMainFrame: message.frameInfo.isMainFrame, gatewayPort: gatewayPort()),
              let request = folderRequest(message.body) else {
            replyHandler(nil, "refused")
            return
        }
        switch request {
        case .choose: choose { replyHandler($0 ?? NSNull(), nil) }
        case .reveal(let path): reveal(path) { replyHandler($0, nil) }
        }
    }
}
