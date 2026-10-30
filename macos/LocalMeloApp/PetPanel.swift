import AppKit
import ImageIO

/// What the pet's panel shows and hears: one picture on a layer, a press that is a click or a drag,
/// the pointer coming and going, and VoiceOver. It decides nothing itself.
final class PetView: NSView {
    var onTouch: (() -> Void)?
    var onClick: (() -> Void)?
    var onDrop: (() -> Void)?
    private var pressedAt: NSPoint?
    private var panelOrigin = NSPoint.zero
    private var dragged = false

    init(picture: CALayer) {
        super.init(frame: NSRect(origin: .zero, size: petPanelSize))
        wantsLayer = true
        layer?.addSublayer(picture)
        setAccessibilityElement(true)
        setAccessibilityRole(.button)
    }

    required init?(coder: NSCoder) { nil }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override var mouseDownCanMoveWindow: Bool { false }

    func describe(_ mood: PetMood) {
        setAccessibilityLabel(mood.label)
        setAccessibilityHelp(mood.help)
        toolTip = mood.label
    }

    override func accessibilityPerformPress() -> Bool {
        onClick?()
        return true
    }

    override func updateTrackingAreas() {
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(rect: .zero, options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect],
                                       owner: self, userInfo: nil))
        super.updateTrackingAreas()
    }

    override func mouseEntered(with event: NSEvent) { onTouch?() }
    override func mouseExited(with event: NSEvent) { onTouch?() }

    // The press is followed by hand, not by the window server, so that a press that stays within a few points is a
    // click and anything more is a drag; the panel follows the pointer by the distance it moved.
    override func mouseDown(with event: NSEvent) {
        guard let window = window else { return }
        onTouch?()
        pressedAt = window.convertPoint(toScreen: event.locationInWindow)
        panelOrigin = window.frame.origin
        dragged = false
    }

    override func mouseDragged(with event: NSEvent) {
        guard let window = window, let start = pressedAt else { return }
        let here = window.convertPoint(toScreen: event.locationInWindow)
        let dx = here.x - start.x, dy = here.y - start.y
        if !dragged && petIsClick(moved: hypot(dx, dy)) { return }
        dragged = true
        window.setFrameOrigin(NSPoint(x: panelOrigin.x + dx, y: panelOrigin.y + dy))
    }

    override func mouseUp(with event: NSEvent) {
        guard pressedAt != nil else { return }
        pressedAt = nil
        onTouch?()
        if dragged { onDrop?() } else { onClick?() }
        dragged = false
    }
}

final class PetPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

/// Melo on the desktop. A small borderless panel above other windows, on every Space, that shows while the app's
/// window is put away (closed, minimized or the app hidden) and the person has not hidden Melo. The app's health
/// poll feeds it (`update`); it adds no timer: its mood is worked out again on each reply, each touch and each
/// change of window.
final class PetController: NSObject {
    let panel: PetPanel
    let view: PetView
    let picture = CALayer()
    /// What a click opens: the window, or the chat or task that waits.
    var openWindow: () -> Void = {}
    var openWaiting: () -> Void = {}
    /// Told when the person hides Melo or brings it back.
    var hiddenChanged: (Bool) -> Void = { _ in }
    private(set) var mood = PetMood.unknown
    private(set) var settings: PetSettings
    private let settingsURL: URL
    private let pictures: URL?
    private var images: [PetState: CGImage] = [:]
    private var imageScale = 0
    private var health: Data?
    private var stopped = false
    private var windowShown = true
    private var touched = Date()
    private var poseSeed = Int.random(in: 0..<PetState.sleepPoses.count)
    private var shownState: PetState?

    var hiddenByPerson: Bool { settings.hidden }
    var isShown: Bool { panel.isVisible }

    /// `settingsURL` is the file the pet's hidden flag and spots live in; `pictures` is the `akita` folder.
    init(settingsURL: URL, pictures: URL?) {
        self.settingsURL = settingsURL
        self.pictures = pictures
        settings = petSettings(try? Data(contentsOf: settingsURL))
        picture.frame = CGRect(origin: .zero, size: petPanelSize)
        picture.contentsGravity = .resize
        picture.actions = ["contents": NSNull()]
        view = PetView(picture: picture)
        panel = PetPanel(contentRect: NSRect(origin: .zero, size: petPanelSize),
                         styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        super.init()
        panel.contentView = view
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle]
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.hidesOnDeactivate = false
        panel.canHide = false
        panel.isMovable = false
        panel.isReleasedWhenClosed = false
        panel.isExcludedFromWindowsMenu = true
        panel.title = "Melo"
        view.menu = menu()
        view.onTouch = { [weak self] in self?.touch() }
        view.onClick = { [weak self] in self?.click() }
        view.onDrop = { [weak self] in self?.drop() }
        observe()
    }

    // MARK: What the app tells it

    /// The last health reply (nil: none yet), and whether the app itself stopped.
    func update(health: Data?, stopped: Bool) {
        self.health = health
        self.stopped = stopped
        if isShown { refresh() }
    }

    /// Whether the main window is in front of the person.
    func windowShown(_ shown: Bool) {
        windowShown = shown
        reconcile()
    }

    /// The person hid Melo (Hide Melo) or brought it back (Show Melo); remembered.
    func setHidden(_ hidden: Bool) {
        guard settings.hidden != hidden else { return }
        settings.hidden = hidden
        save()
        hiddenChanged(hidden)
        reconcile()
    }

    // MARK: Showing and hiding

    private func reconcile() {
        let wanted = petShouldShow(windowShown: windowShown, hiddenByPerson: settings.hidden)
        if wanted && !panel.isVisible {
            guard picture(for: .lieIdle) != nil, place() else { return }
            touched = Date()
            poseSeed = Int.random(in: 0..<PetState.sleepPoses.count)
            shownState = nil
            panel.orderFrontRegardless()
            refresh()
        } else if !wanted && panel.isVisible {
            panel.orderOut(nil)
        }
    }

    // MARK: Mood and picture

    private func touch() {
        touched = Date()
        if case .sleeping(let left) = mood, let index = PetState.sleepPoses.firstIndex(of: left) {
            poseSeed = (index + Int.random(in: 1..<PetState.sleepPoses.count)) % PetState.sleepPoses.count
        }
        if isShown { refresh() }
    }

    private func refresh() {
        let old = mood
        mood = petMood(health: health, stopped: stopped, sinceTouch: Date().timeIntervalSince(touched),
                       poseSeed: poseSeed)
        if shownState == nil || mood != old { view.describe(mood) }
        if mood.picture != shownState, let image = picture(for: mood.picture) {
            self.picture.contentsScale = CGFloat(image.width) / petPanelSize.width
            self.picture.contents = image
            shownState = mood.picture
        }
        if mood == .waiting && old != .waiting {
            NSAccessibility.post(element: view, notification: .announcementRequested,
                                 userInfo: [.announcement: mood.label, .priority: NSAccessibilityPriorityLevel.high.rawValue])
        }
    }

    /// The picture for the panel's screen scale, decoded once; nil when the file is missing.
    private func picture(for state: PetState) -> CGImage? {
        let scale = Int((panel.screen ?? NSScreen.main)?.backingScaleFactor ?? 2) >= 2 ? 2 : 1
        if scale != imageScale { images = [:]; imageScale = scale }
        if let image = images[state] { return image }
        guard let url = pictures?.appendingPathComponent("\(state.folder)/\(state.fileName(scale: scale))"),
              let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        images[state] = image
        return image
    }

    // MARK: Clicks, menu, drops

    func click() {
        switch petClickTarget(mood) {
        case .waiting: openWaiting()
        case .window: openWindow()
        }
    }

    private func menu() -> NSMenu {
        let menu = NSMenu()
        for (title, action) in [("Open localmelo", #selector(menuOpen)), ("Hide Melo", #selector(menuHide))] {
            menu.addItem(withTitle: title, action: action, keyEquivalent: "").target = self
        }
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit localmelo", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "")
            .target = NSApp
        return menu
    }

    @objc private func menuOpen() { openWindow() }
    @objc private func menuHide() { setHidden(true) }

    /// The pet was dropped: its paws snap onto the bottom edge if they are near it, the panel stays on the screen it
    /// is mostly on, and the spot is saved for that screen.
    private func drop() {
        let screens = PetController.screens()
        guard let screen = screens.max(by: { overlap($0.visible) < overlap($1.visible) }) else { return }
        let snapped = petSnapped(panel.frame, belowPaws: petBelowPaws, in: screen.visible)
        let frame = petClamped(snapped, belowPaws: petBelowPaws, in: screen.visible)
        panel.setFrame(frame, display: true)
        settings.remember(screen: screen.key, spot: petSpot(of: frame, on: screen.visible))
        save()
    }

    private func overlap(_ area: CGRect) -> CGFloat {
        let common = area.intersection(panel.frame)
        return common.isNull ? 0 : common.width * common.height
    }

    // MARK: Where, and what is remembered

    static func screens() -> [PetScreen] {
        NSScreen.screens.map { screen in
            let number = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID ?? 0
            var key = String(number)
            if let uuid = CGDisplayCreateUUIDFromDisplayID(number)?.takeRetainedValue(),
               let text = CFUUIDCreateString(nil, uuid) as String? { key = text }
            return PetScreen(key: key, visible: screen.visibleFrame)
        }
    }

    /// Puts the panel at its saved spot, or at the bottom right of the main screen. False without a screen.
    @discardableResult
    private func place() -> Bool {
        guard let placement = petPlacement(settings, screens: PetController.screens()) else { return false }
        panel.setFrame(placement.frame, display: false)
        return true
    }

    private func save() {
        try? FileManager.default.createDirectory(at: settingsURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? petSettingsData(settings).write(to: settingsURL, options: .atomic)
    }

    // MARK: What the system says

    private func observe() {
        NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification, object: nil,
                                               queue: .main) { [weak self] _ in
            if self?.isShown == true { self?.place() }
        }
    }
}
