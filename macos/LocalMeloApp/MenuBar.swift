import AppKit

/// The menu-bar item. The localmelo logo is shown in colour, the count of what runs sits beside it, and
/// a red dot appears beside the logo while something waits for the person. The menu starts with what
/// runs and with the way to what waits, so the chat that needs an answer is one click away.
final class MenuBar {
    let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    /// The first two lines of the menu, which the integration tests read and click: what runs, and the
    /// way to the chat or task that waits.
    let runningLine = NSMenuItem.sectionHeader(title: "")
    let waitingLine: NSMenuItem
    /// Brings Melo back after "Hide Melo": shown only while Melo is hidden.
    let showPetLine: NSMenuItem
    private let divider = NSMenuItem.separator()
    private let plain: NSImage?
    private let badged: NSImage?
    private var wasWaiting = false

    /// `menu` is the app menu's copy; the item puts its two first lines in front of its entries.
    init(menu: NSMenu, waiting: Selector, showPet: Selector) {
        let logo = MenuBar.logo()
        waitingLine = NSMenuItem(title: "Waiting for your approval", action: waiting, keyEquivalent: "")
        showPetLine = NSMenuItem(title: "Show Melo", action: showPet, keyEquivalent: "")
        showPetLine.isHidden = true
        menu.insertItem(showPetLine, at: (menu.items.firstIndex { $0.title == "Open localmelo" } ?? -1) + 1)
        plain = MenuBar.image(logo, badge: false)
        badged = MenuBar.image(logo, badge: true)
        for item in [runningLine, waitingLine, divider].reversed() { menu.insertItem(item, at: 0) }
        statusItem.menu = menu
        statusItem.button?.imagePosition = .imageLeft
        show(.unknown)
    }

    /// The "On the desktop" choice, in the menu after "Show Melo".
    func offerDesktop(_ item: NSMenuItem) {
        guard let menu = statusItem.menu else { return }
        menu.insertItem(item, at: menu.index(of: showPetLine) + 1)
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
                // Without the logo the item must still be there: its name stands in for it.
                button.image = nil
                button.title = count.isEmpty ? "localmelo" : "localmelo · \(count)"
            }
            // Title and help say the words too: left alone they are the bare count and the tooltip, which
            // VoiceOver would read on top of the label.
            button.setAccessibilityLabel(label)
            button.setAccessibilityTitle(label)
            button.setAccessibilityHelp("")
            button.toolTip = label
        }
        runningLine.title = line ?? ""
        runningLine.isHidden = line == nil
        waitingLine.isHidden = !waiting
        divider.isHidden = runningLine.isHidden && waitingLine.isHidden
        // Someone working in another app cannot see the dot: the start of a wait is said aloud, as a failure is.
        if waiting && !wasWaiting {
            NSAccessibility.post(element: NSApp as Any, notification: .announcementRequested,
                                 userInfo: [.announcement: label, .priority: NSAccessibilityPriorityLevel.high.rawValue])
        }
        wasWaiting = waiting
    }

    /// The logo, 18 points square: the two files the build copies, 18 and 36 px with their corners already
    /// rounded, so the system never scales it. Nil when neither is in the bundle.
    private static func logo() -> NSImage? {
        let logo = NSImage(size: NSSize(width: 18, height: 18))
        for name in ["MenuBarLogo", "MenuBarLogo@2x"] {
            guard let url = Bundle.main.url(forResource: name, withExtension: "png"),
                  let data = try? Data(contentsOf: url), let rep = NSBitmapImageRep(data: data) else { continue }
            rep.size = logo.size
            logo.addRepresentation(rep)
        }
        return logo.representations.isEmpty ? nil : logo
    }

    /// The logo in colour. The badged image is wider and holds a red dot beside the logo's top-right corner,
    /// so the logo itself is never covered. Neither image is a template, so the system leaves their colours
    /// alone: the dot and its ring come in a light and a dark variant, drawn for the appearance of the bar
    /// they are on.
    private static func image(_ logo: NSImage?, badge: Bool) -> NSImage? {
        guard let logo = logo else { return nil }
        let image = NSImage(size: NSSize(width: badge ? 23 : 18, height: 18), flipped: false) { _ in
            logo.draw(in: NSRect(x: 0, y: 0, width: 18, height: 18))
            if badge {
                Palette.badgeRing.setFill()
                NSBezierPath(ovalIn: NSRect(x: 15, y: 10, width: 8, height: 8)).fill()
                Palette.badge.setFill()
                NSBezierPath(ovalIn: NSRect(x: 16, y: 11, width: 6, height: 6)).fill()
            }
            return true
        }
        image.isTemplate = false
        return image
    }
}
