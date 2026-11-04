import AppKit
import WebKit

/// The console's Browser panel in a window of its own, which floats beside other work. It shows the page the gateway
/// serves for one chat (`browserLiveURL`) and nothing else: it signs in with the cookie the main window's web view
/// already holds (they share the default website data store), and it never goes to another site. Closing the window
/// takes its web view down, so no stream stays open behind it.
final class BrowserWindowController: NSObject, NSWindowDelegate, WKNavigationDelegate {
    private var window: NSWindow?
    private var content: NSView?
    private var webView: WKWebView?
    // What the web view shows now; the same chat again only brings the window forward.
    private var shown: (session: String, port: Int)?
    // The gateway's port, which the navigation policy checks every address against.
    private var port = 0

    /// Shows *session*'s browser and brings the window to the front. Another chat loads in place of the one shown.
    func show(session: String, port: Int) {
        guard let url = browserLiveURL(session: session, port: port) else { return }
        let window = self.window ?? makeWindow()
        if webView == nil || shown?.session != session || shown?.port != port {
            // A fresh web view for each chat: the page starts from its address, and the old chat's stream ends.
            tearDownWebView()
            self.port = port
            load(url)
            shown = (session, port)
        }
        if window.isMiniaturized { window.deminiaturize(nil) }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Closes the window and takes its web view down; the gateway it showed is gone, or the app is done with it.
    func close() {
        window?.close()
        tearDownWebView()
    }

    /// The window's title in the language now in use.
    func languageChanged() {
        window?.title = L10n.t("app.browser.title", "Browser")
    }

    func windowWillClose(_ notification: Notification) { tearDownWebView() }

    private func makeWindow() -> NSWindow {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 960, height: 600),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable],
                              backing: .buffered, defer: false)
        window.title = L10n.t("app.browser.title", "Browser")
        window.backgroundColor = Palette.canvas
        window.minSize = NSSize(width: 480, height: 300)
        window.isReleasedWhenClosed = false
        window.level = .floating
        window.hidesOnDeactivate = false
        let content = NSView(frame: NSRect(x: 0, y: 0, width: 960, height: 600))
        window.contentView = content
        window.delegate = self
        window.center()
        self.window = window
        self.content = content
        return window
    }

    private func load(_ url: URL) {
        guard let content = content else { return }
        let configuration = WKWebViewConfiguration()
        // The page picks its language as the console's window did, from the Mac's languages.
        configuration.userContentController.addUserScript(
            WKUserScript(source: languagesScript(Locale.preferredLanguages), injectionTime: .atDocumentStart,
                         forMainFrameOnly: true))
        let view = ConsoleWebView(frame: content.bounds, configuration: configuration)
        view.autoresizingMask = [.width, .height]
        view.navigationDelegate = self
        view.underPageBackgroundColor = Palette.canvas
        content.addSubview(view)
        webView = view
        view.load(URLRequest(url: url))
    }

    private func tearDownWebView() {
        guard let view = webView else { return }
        view.navigationDelegate = nil
        view.stopLoading()
        // An empty page ends the old page's streams now, not whenever WebKit lets go of the view.
        view.loadHTMLString("", baseURL: nil)
        view.configuration.userContentController.removeAllUserScripts()
        view.removeFromSuperview()
        webView = nil
        shown = nil
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = action.request.url else { decisionHandler(.cancel); return }
        // A request without a target frame (a new window) counts as the main frame, which is the stricter rule.
        let allowed = browserWindowAllows(url, port: port, isMainFrame: action.targetFrame?.isMainFrame ?? true)
        decisionHandler(allowed ? .allow : .cancel)
    }

    func webView(_ webView: WKWebView, decidePolicyFor response: WKNavigationResponse,
                 decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        guard response.isForMainFrame else { decisionHandler(.allow); return }
        guard let url = response.response.url, isGatewayPage(url, port: port) else {
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }
}
