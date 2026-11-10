import AppKit

/// The console's window, with no title band: the page reaches the top edge and the window's buttons sit over its
/// sidebar. The title stays for the Window menu, Mission Control and VoiceOver; the page moves the window from its top
/// strip (`WindowBridge`).
func consoleWindow() -> NSWindow {
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 800),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                          backing: .buffered, defer: false)
    window.title = "localmelo"
    window.titleVisibility = .hidden
    window.titlebarAppearsTransparent = true
    return window
}

/// The strip at the top of the console's window: its height and how far from the left the window's buttons reach, in
/// points. Both are zero in full screen, where the buttons are gone.
struct WindowBar: Equatable {
    var height: Double = 0
    var buttons: Double = 0
}

extension WindowBar {
    /// The title bar of `window` as the system lays it out (its height differs between versions of macOS).
    init(titleBarOf window: NSWindow) {
        height = window.frame.height - window.contentLayoutRect.height
        buttons = window.standardWindowButton(.zoomButton).map { $0.convert($0.bounds, to: nil).maxX } ?? 0
    }
}

/// The script that gives the console page the strip's size: `--window-bar` and `--window-buttons` on its root, and
/// `data-window-bar`, which turns the page's app-only rules on. Only whole numbers go into it.
func windowBarScript(_ bar: WindowBar) -> String {
    func points(_ value: Double) -> Int { value.isFinite ? max(0, Int(value.rounded(.up))) : 0 }
    return "(() => { const root = document.documentElement; root.dataset.windowBar = '';"
        + " root.style.setProperty('--window-bar', '\(points(bar.height))px');"
        + " root.style.setProperty('--window-buttons', '\(points(bar.buttons))px'); })();"
}

/// What a press on the console's top strip asks of the window.
enum WindowRequest: Equatable {
    /// Move the window with the press.
    case drag
    /// Do what a double click on a title bar does.
    case doubleClick
}

/// The request a page message holds: `{"action": "drag"}` or `{"action": "double-click"}` and nothing else.
func windowRequest(_ body: Any) -> WindowRequest? {
    guard let message = body as? [String: Any], message.count == 1 else { return nil }
    switch message["action"] as? String {
    case "drag": return .drag
    case "double-click": return .doubleClick
    default: return nil
    }
}

enum DoubleClickAction: Equatable {
    case zoom, minimize, none
}

/// What a double click on a title bar does, as the system setting says: `AppleActionOnDoubleClick` in the global
/// domain, else the older `AppleMiniaturizeOnDoubleClick`. Anything else ("Maximize", "Fill") zooms.
func doubleClickAction(setting: String?, minimizes: Bool) -> DoubleClickAction {
    switch setting {
    case "Minimize": return .minimize
    case "None": return .none
    case nil: return minimizes ? .minimize : .zoom
    default: return .zoom
    }
}

extension WindowRequest {
    /// Runs the request on `window`. The drag starts from the press itself (`press`, the web view's last mouse down),
    /// and only while the button is still down: a click that has ended moves nothing.
    func perform(on window: NSWindow, press: NSEvent?) {
        switch self {
        case .drag:
            guard let press = press, press.window === window, NSEvent.pressedMouseButtons & 1 != 0 else { return }
            window.performDrag(with: press)
        case .doubleClick:
            let defaults = UserDefaults.standard
            switch doubleClickAction(setting: defaults.string(forKey: "AppleActionOnDoubleClick"),
                                     minimizes: defaults.bool(forKey: "AppleMiniaturizeOnDoubleClick")) {
            case .zoom: window.performZoom(nil)
            case .minimize: window.performMiniaturize(nil)
            case .none: break
            }
        }
    }
}
