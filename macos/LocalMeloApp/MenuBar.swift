import AppKit

/// The menu-bar item. Melo's head is a template image, the count of what runs sits beside it, and a
/// dot appears beside the head while something waits for the person. The menu starts with what runs
/// and with the way to what waits, so the chat that needs an answer is one click away.
final class MenuBar {
    let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    /// The first two lines of the menu, which the integration tests read and click: what runs, and the
    /// way to the chat or task that waits.
    let runningLine = NSMenuItem.sectionHeader(title: "")
    let waitingLine: NSMenuItem
    private let divider = NSMenuItem.separator()
    private let plain: NSImage?
    private let badged: NSImage?

    /// `menu` is the app menu's copy; the item puts its two first lines in front of its entries.
    init(menu: NSMenu, waiting: Selector) {
        let mark = Bundle.main.url(forResource: "mark", withExtension: "svg").flatMap { NSImage(contentsOf: $0) }
        waitingLine = NSMenuItem(title: "Waiting for your approval", action: waiting, keyEquivalent: "")
        plain = MenuBar.image(mark, badge: false)
        badged = MenuBar.image(mark, badge: true)
        for item in [runningLine, waitingLine, divider].reversed() { menu.insertItem(item, at: 0) }
        statusItem.menu = menu
        statusItem.button?.imagePosition = .imageLeft
        show(.unknown)
    }

    func show(_ state: MenuBarState) {
        var waiting = false
        if case .waiting = state { waiting = true }
        let count = menuTitle(state), label = menuBarLabel(state), line = menuStatusLine(state)
        if let button = statusItem.button {
            if let image = waiting ? badged : plain {
                button.image = image
                button.title = count
            } else {
                // Without the mark the item must still be there: its name stands in for it.
                button.image = nil
                button.title = count.isEmpty ? "localmelo" : "localmelo · \(count)"
            }
            button.setAccessibilityLabel(label)
            button.toolTip = label
        }
        runningLine.title = line ?? ""
        runningLine.isHidden = line == nil
        waitingLine.isHidden = !waiting
        divider.isHidden = runningLine.isHidden && waitingLine.isHidden
    }

    /// The mark, 18 points tall. The badged image is wider and holds a dot beside the right ear, so
    /// the head itself is never cut. Both are templates: the system tints them for light, dark and
    /// the open menu.
    private static func image(_ mark: NSImage?, badge: Bool) -> NSImage? {
        guard let mark = mark else { return nil }
        let image = NSImage(size: NSSize(width: badge ? 23 : 18, height: 18), flipped: false) { _ in
            mark.draw(in: NSRect(x: 0, y: 0, width: 18, height: 18))
            if badge {
                NSColor.black.setFill()
                NSBezierPath(ovalIn: NSRect(x: 17, y: 11.6, width: 5.6, height: 5.6)).fill()
            }
            return true
        }
        image.isTemplate = true
        return image
    }
}
