import AppKit

/// What the pet's panel shows and hears: the moves' canvas (`PetMovePlayer`), a press that is a click or a drag, the
/// pointer coming and going, and VoiceOver. It decides nothing itself.
final class PetView: NSView {
    var onTouch: (() -> Void)?
    var onClick: (() -> Void)?
    var onDrop: (() -> Void)?
    private var pressedAt: NSPoint?
    private var panelOrigin = NSPoint.zero
    private var dragged = false

    init() {
        super.init(frame: NSRect(origin: .zero, size: petPanelSize))
        wantsLayer = true
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
/// change of window, and its motion is Core Animation's. Everything it shows is `moves`, the animated states: a mood
/// with moves (`petMove`) plays them, and only a mood with a still (`petStill`: nothing known yet, a run that just
/// finished, a failure, the app stopped) or Reduce Motion holds a poster.
final class PetController: NSObject {
    let panel: PetPanel
    let view: PetView
    /// The animated states; nil when their files cannot be read. Then the pet shows the offline still alone
    /// (`standIn`) if that can be read, and never shows if not; either is logged once.
    let moves: PetMovePlayer?
    let standIn: CALayer?
    /// What a click opens: the window, or the chat or task that waits.
    var openWindow: () -> Void = {}
    var openWaiting: () -> Void = {}
    /// Told when the person hides Melo or brings it back.
    var hiddenChanged: (Bool) -> Void = { _ in }
    /// The "On the desktop" choice: the submenus of the pet's menu and the menu bar's, and who hears a new choice.
    let desktop = DesktopMenu()
    var desktopChanged: (DesktopMode) -> Void = { _ in }
    private(set) var mood = PetMood.unknown
    private(set) var settings: PetSettings
    var environment: PetEnvironment { gate.environment }
    private let settingsURL: URL
    private var gate: MotionGate!
    private var health: Data?
    private var stopped = false
    private var windowShown = true
    private var touched = Date()
    private var described = false
    // The activity pill is on the desktop in the pet's place; and until when the pet shows "done" after a "Done" of
    // the pill.
    private var pillUp = false
    private var doneUntil: Date?
    // Which move shows: the wake-up first, a play now and then.
    private var picker = PetMovePicker()

    var hiddenByPerson: Bool { settings.hidden }
    var isShown: Bool { panel.isVisible }

    /// `settingsURL` is the file the pet's hidden flag and spots live in; `movesFolder` is the `moves` folder.
    init(settingsURL: URL, movesFolder: URL?) {
        self.settingsURL = settingsURL
        settings = petSettings(try? Data(contentsOf: settingsURL))
        view = PetView()
        moves = movesFolder.flatMap { PetMovePlayer(folder: $0, side: petMoveSide) }
        standIn = moves != nil ? nil : movesFolder.flatMap { petDecoded($0.appendingPathComponent(PetStill.offline.file)) }
            .map { image in
                let layer = CALayer()
                layer.contents = image
                layer.contentsGravity = .resize
                layer.opacity = PetStill.offline.opacity
                layer.frame = petMoveFrame(side: petMoveSide, anchor: petStillAnchor, pixels: petStillPixels,
                                           in: petPanelSize, belowPaws: petBelowPaws)
                return layer
            }
        if let moves = moves {
            let manifest = moves.manifest
            moves.layer.frame = petMoveFrame(side: petMoveSide, anchor: manifest.anchor, pixels: manifest.pixels,
                                             in: petPanelSize, belowPaws: petBelowPaws)
            view.layer?.addSublayer(moves.layer)
        } else if let standIn = standIn {
            view.layer?.addSublayer(standIn)
            NSLog("localmelo: Melo's moves cannot be read in %@; the pet shows its offline picture",
                  movesFolder?.path ?? "none")
        } else {
            NSLog("localmelo: Melo's moves cannot be read in %@; the pet stays hidden", movesFolder?.path ?? "none")
        }
        panel = PetPanel(contentRect: NSRect(origin: .zero, size: petPanelSize),
                         styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        super.init()
        desktop.show(settings.mode)
        desktop.choose = { [weak self] in self?.setDesktop($0) }
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
        gate = MotionGate(window: { [weak self] in self?.panel }, visible: { [weak self] in
            guard let panel = self?.panel else { return false }
            return panel.isVisible && panel.occlusionState.contains(.visible)
        })
        gate.changed = { [weak self] _ in
            guard let self = self, self.isShown else { return }
            self.applyMoves()
        }
        moves?.onEnded = { [weak self] move in
            self?.picker.ended(move, now: Date())
            if self?.isShown == true { self?.refresh() }
            // The pet sits after the wake-up, then lies down: the next look comes when the sitting ends.
            if move == .startup {
                DispatchQueue.main.asyncAfter(deadline: .now() + petSitFor + 0.05) { [weak self] in
                    if self?.isShown == true { self?.refresh() }
                }
            }
        }
        NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification,
                                               object: nil, queue: .main) { [weak self] _ in
            if self?.isShown == true { self?.place() }
        }
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

    /// The console's language changed: the pet's label is written again.
    func languageChanged() {
        described = false
        if isShown { refresh() }
    }

    /// The person hid Melo (Hide Melo) or brought it back (Show Melo); remembered. Showing again brings back what the
    /// person had: Melo with the pill, or Melo alone.
    func setHidden(_ hidden: Bool) {
        setDesktop(hidden ? .nothing : settings.pill ? .pill : .pet)
    }

    /// The person chose what is on the desktop; remembered.
    func setDesktop(_ mode: DesktopMode) {
        guard settings.mode != mode else { return }
        let hidden = settings.hidden
        settings.mode = mode
        save()
        desktop.show(mode)
        if hidden != settings.hidden { hiddenChanged(settings.hidden) }
        desktopChanged(mode)
        reconcile()
    }

    /// The activity pill appeared or went. While it is there the pet is not; when it goes the pet comes back, and
    /// shows "done" for a few seconds if the pill's "Done" ran out.
    func pillShowing(_ showing: Bool, showsDone: Bool) {
        pillUp = showing
        if showsDone { doneUntil = Date().addingTimeInterval(petDoneAfterPill) }
        reconcile()
    }

    /// The pill was dropped at `spot` of the screen `key`: the pet comes back there.
    func rememberSpot(screen key: String, spot: PetSpot) {
        settings.remember(screen: key, spot: spot)
        save()
    }

    // MARK: Showing and hiding

    private func reconcile() {
        let wanted = petShouldShow(windowShown: windowShown, hiddenByPerson: settings.hidden) && !pillUp
        if wanted && !panel.isVisible {
            guard moves != nil || standIn != nil, place() else { return }
            touched = Date()
            described = false
            panel.orderFrontRegardless()
            gate.refresh()
            refresh()
        } else if !wanted && panel.isVisible {
            panel.orderOut(nil)
            moves?.run(false)
            gate.refresh()
        }
    }

    // MARK: Mood and moves

    private func touch() {
        touched = Date()
        if isShown { refresh() }
    }

    private func refresh() {
        gate.refresh()  // each reply also reads the system again, so a missed notification cannot stick
        let old = mood
        mood = petMood(health: health, stopped: stopped, sinceTouch: Date().timeIntervalSince(touched))
        if let until = doneUntil, mood == .resting { mood = Date() < until ? .done : mood }
        if let until = doneUntil, Date() >= until { doneUntil = nil }
        // The words are written when the mood or the language changes, not on every reply.
        if !described || mood != old { view.describe(mood) }
        described = true
        showMoves()
    }

    /// Shows what the mood looks like: its still (`petStill`), held, or its move, played. The words (`describe`) are
    /// the mood's either way. The picker hears every mood, so a still ends a rest, a play and the sitting.
    private func showMoves() {
        guard let moves = moves else { return }
        let still = !petMovesMayMove(environment)
        moves.setStill(still)  // before the state, so it starts as it should show
        let pick = picker.pick(mood, thinking: health.map(petThinking) ?? false, still: still, now: Date())
        if let picture = petStill(mood) {
            // A folder made before the stills: the moves' own poster stands in.
            if !moves.hold(picture) { moves.set(picture.standIn, poster: true) }
        } else if let (move, once) = pick {
            moves.set(move, once: once)
        }
        applyMoves()
    }

    /// The moves follow the gate: posters under Reduce Motion; stopped where they are while the pet cannot be seen.
    private func applyMoves() {
        guard let moves = moves else { return }
        moves.setStill(!petMovesMayMove(environment))
        moves.run(isShown && petMovesRun(environment))
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
        for (word, action) in [(MenuWord.open, #selector(menuOpen)), (.hidePet, #selector(menuHide))] {
            menu.addItem(withTitle: "", action: action, keyEquivalent: "").word(word).target = self
        }
        menu.addItem(desktop.makeItem())
        menu.addItem(.separator())
        menu.addItem(withTitle: "", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "").word(.quit).target = NSApp
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
}
