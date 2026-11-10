import AppKit

/// The menu-bar item. Melo's head is drawn as lines in the bar's colours (MenuBarMark.swift), the count of what runs
/// sits beside it, and a red dot appears beside the head while something waits for the person. The menu starts with what
/// runs and with the way to what waits, so the chat that needs an answer is one click away.
final class MenuBar {
    let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    /// The first two lines of the menu, which the integration tests read and click: what runs, and the
    /// way to the chat or task that waits.
    let runningLine = NSMenuItem.sectionHeader(title: "")
    let waitingLine: NSMenuItem
    /// Brings Melo back after "Hide Melo": shown only while Melo is hidden.
    let showPetLine: NSMenuItem
    /// melo_serve's chat models (ModelMenu.swift): shown only while melo_serve serves the chats.
    let modelItem: NSMenuItem
    private let models: NSMenu
    private let menuOpened = MenuOpened()
    private let modelsOpened = MenuOpened()
    private var modelRows: [ModelMenuRow] = []
    private let choose: Selector
    private let cancel: Selector
    private let divider = NSMenuItem.separator()
    private let mark = MenuBarMark(folder: Bundle.main.resourceURL)
    private var appearanceWatch: NSKeyValueObservation?
    private var wasWaiting = false
    private var state = MenuBarState.unknown

    /// `menu` is the app menu's copy; the item puts its two first lines in front of its entries. *chooseModel* and
    /// *cancelSwitch* are what a pick of the Model submenu and its Cancel Switch send; *menuOpens* runs as the menu
    /// opens, so the picks are read again before the submenu shows.
    init(menu: NSMenu, waiting: Selector, showPet: Selector, chooseModel: Selector, cancelSwitch: Selector,
         menuOpens: @escaping () -> Void) {
        waitingLine = NSMenuItem(title: "", action: waiting, keyEquivalent: "").word(.waiting)
        showPetLine = NSMenuItem(title: "", action: showPet, keyEquivalent: "").word(.showPet)
        showPetLine.isHidden = true
        choose = chooseModel
        cancel = cancelSwitch
        let models = NSMenu()
        self.models = models
        let modelItem = NSMenuItem(title: "", action: nil, keyEquivalent: "")
        modelItem.submenu = models
        modelItem.word(.model)
        modelItem.isHidden = true
        self.modelItem = modelItem
        let open = MenuWord.open.rawValue
        menu.insertItem(showPetLine, at: (menu.items.firstIndex { $0.identifier?.rawValue == open } ?? -1) + 1)
        menu.insertItem(modelItem, at: menu.index(of: showPetLine) + 1)
        for item in [runningLine, waitingLine, divider].reversed() { menu.insertItem(item, at: 0) }
        statusItem.menu = menu
        statusItem.button?.imagePosition = .imageLeft
        // The mark's colours are the bar's: a change of the bar's appearance (a light or a dark wallpaper behind
        // it, or the system's setting) draws it again.
        appearanceWatch = statusItem.button?.observe(\.effectiveAppearance) { [weak self] _, _ in
            DispatchQueue.main.async { self?.refresh() }
        }
        models.autoenablesItems = false
        menuOpened.opened = menuOpens
        menu.delegate = menuOpened
        models.delegate = modelsOpened
        show(.unknown)
    }

    /// The "On the desktop" choice, in the menu after "Model".
    func offerDesktop(_ item: NSMenuItem) {
        guard let menu = statusItem.menu else { return }
        menu.insertItem(item, at: menu.index(of: modelItem) + 1)
    }

    /// The Model submenu for *list* (nil: melo_serve does not serve the chats, and the item hides), with the switch
    /// under way. Each item shows its words as a visible subtitle.
    func showModels(_ list: ServeList?, switching: ServeSwitch?) {
        modelItem.isHidden = list == nil
        guard let list = list else { return }
        modelItem.subtitle = modelMenuSubtitle(list, switching: switching)
        let rows = modelMenuRows(list, switching: switching)
        guard modelMenuNeedsDrawing(rows, shown: modelRows, open: modelsOpened.open) else { return }
        modelRows = rows
        models.removeAllItems()
        for row in rows {
            switch row {
            case .status(let title, let subtitle):
                let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
                item.subtitle = subtitle.isEmpty ? nil : subtitle
                item.isEnabled = false
                models.addItem(item)
            case .cancel:
                let item = NSMenuItem(title: "", action: cancel, keyEquivalent: "").word(.cancelSwitch)
                models.addItem(item)
            case .separator:
                models.addItem(.separator())
            case .pick(let id, let title, let subtitle, let checked, let enabled):
                let item = NSMenuItem(title: title, action: choose, keyEquivalent: "")
                item.subtitle = subtitle
                item.representedObject = id
                item.state = checked ? .on : .off
                item.isEnabled = enabled
                models.addItem(item)
            case .foot(let text):
                let item = NSMenuItem(title: text, action: nil, keyEquivalent: "")
                item.isEnabled = false
                models.addItem(item)
            }
        }
    }

    /// Writes the lines again, in the language now in use.
    func refresh() { show(state) }

    func show(_ state: MenuBarState) {
        self.state = state
        var waiting = false
        if case .waiting = state { waiting = true }
        let count = menuTitle(state), label = menuBarLabel(state), line = menuStatusLine(state)
        if let button = statusItem.button {
            if let mark = mark {
                button.image = mark.image(for: button.effectiveAppearance, badge: waiting)
                button.title = count
            } else {
                // Without the mark the item must still be there: its name stands in for it.
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
}

/// Runs a closure as the menu it is the delegate of opens.
final class MenuOpened: NSObject, NSMenuDelegate {
    var opened: () -> Void = {}
    private(set) var open = false

    func menuWillOpen(_ menu: NSMenu) {
        open = true
        opened()
    }

    func menuDidClose(_ menu: NSMenu) { open = false }
}
