import AppKit
import ImageIO

/// The pill's colours, light then dark, in the console's palette: moss at work, amber for a wait (the one colour that
/// asks for the person), green for done, red for a failure and a model server that is down; the mark and words say it too.
enum PillColors {
    static let fill = Palette.pair(0xf6f6f3, 0x1e211c)
    static let edge = Palette.pair(0xd9dbd2, 0x3a3e36)
    static let chip = Palette.pair(0xf1f2ec, 0x2d312a)
    static let disc = Palette.pair(0xf2efdc, 0xf2efdc)
    static let working = Palette.pair(0x3f5a36, 0xb5cf9f)
    static let waiting = Palette.pair(0x865300, 0xffc861)
    static let done = Palette.pair(0x2b6a30, 0x8fd694)
    static let failed = Palette.pair(0xb3261e, 0xff8a80)

    static func ring(_ state: PillState) -> NSColor {
        [.working: working, .waiting: waiting, .done: done][state] ?? failed
    }
}

/// The "×" at the pill's corner that shows while the pointer is over it.
final class CloseMark: NSView {
    var onPress: (() -> Void)?

    override init(frame: NSRect) {
        super.init(frame: frame)
        isHidden = true
        setAccessibilityElement(false)
    }

    required init?(coder: NSCoder) { nil }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override var mouseDownCanMoveWindow: Bool { false }
    override func mouseDown(with event: NSEvent) { onPress?() }

    override func draw(_ dirtyRect: NSRect) {
        let disc = NSBezierPath(ovalIn: bounds.insetBy(dx: 0.5, dy: 0.5))
        PillColors.fill.setFill()
        disc.fill()
        PillColors.edge.setStroke()
        disc.stroke()
        let mark = NSAttributedString(string: "\u{00D7}", attributes: [
            .font: NSFont.systemFont(ofSize: 13, weight: .bold), .foregroundColor: Palette.ink])
        let size = mark.size()
        mark.draw(at: NSPoint(x: (bounds.width - size.width) / 2, y: (bounds.height - size.height) / 2))
    }
}

/// The capsule's tint, edge and "+1" chip, drawn over the blur and under the rest. The edge is amber while it waits.
final class PillBackdrop: NSView {
    var model: PillModel? { didSet { needsDisplay = true } }

    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    override func draw(_ dirtyRect: NSRect) {
        guard let model = model else { return }
        let body = bounds.insetBy(dx: pillMargin, dy: pillMargin)
        let waiting = model.state == .waiting
        let width: CGFloat = waiting ? 2 : 1
        let shape = NSBezierPath(roundedRect: body.insetBy(dx: width / 2, dy: width / 2),
                                 xRadius: pillHeight / 2, yRadius: pillHeight / 2)
        PillColors.fill.withAlphaComponent(0.88).setFill()
        shape.fill()
        (waiting ? PillColors.waiting : PillColors.edge).setStroke()
        shape.lineWidth = width
        shape.stroke()
        if model.more > 0 {
            PillColors.chip.setFill()
            NSBezierPath(roundedRect: NSRect(x: body.maxX - 14 - pillChip, y: body.minY + 14, width: pillChip, height: 20),
                         xRadius: 10, yRadius: 10).fill()
        }
    }
}

/// The round portrait, in layers of its own: a ring in the state's colour round the picture. It is the pill's only
/// moving part (`PillView.play`), so its layers are hosted here, where AppKit leaves them as they are.
final class PortraitHost: NSView {
    let disc = CALayer()
    let picture = CALayer()

    init() {
        super.init(frame: NSRect(x: 0, y: 0, width: 36, height: 36))
        layer = CALayer()
        wantsLayer = true
        disc.frame = bounds
        disc.cornerRadius = 18
        disc.borderWidth = 2
        picture.frame = bounds.insetBy(dx: 2, dy: 2)
        picture.cornerRadius = 16
        picture.masksToBounds = true
        picture.contentsGravity = .resizeAspectFill
        disc.addSublayer(picture)
        layer?.addSublayer(disc)
        setAccessibilityElement(false)
    }

    required init?(coder: NSCoder) { nil }

    override func hitTest(_ point: NSPoint) -> NSView? { nil }
}

/// What the pill's panel draws and hears: the capsule, the round portrait with its ring, the mark and the one line,
/// the "+1" chip, a press that is a click or a drag, the pointer coming and going, and VoiceOver. It decides nothing.
final class PillView: NSView {
    var onClick: (() -> Void)?
    var onDrop: (() -> Void)?
    var onDismiss: (() -> Void)?
    private(set) var capsuleWidth: CGFloat = 0
    private(set) var model: PillModel?
    /// The system's blur behind the capsule: it shows only on a screen, never when the view is drawn into a bitmap.
    let blur = NSVisualEffectView()
    private let backdrop = PillBackdrop()
    let host = PortraitHost()
    var disc: CALayer { host.disc }
    let mark = PillView.label(13, .bold)
    let line = PillView.label(14, .medium)
    let more = PillView.label(11, .semibold)
    let close = CloseMark(frame: NSRect(x: 0, y: 0, width: 18, height: 18))
    private var pressedAt: NSPoint?
    private var panelOrigin = NSPoint.zero
    private var dragged = false

    init() {
        super.init(frame: .zero)
        wantsLayer = true
        blur.material = .hudWindow
        blur.blendingMode = .behindWindow
        blur.state = .active
        blur.wantsLayer = true
        blur.layer?.masksToBounds = true
        addSubview(blur)
        addSubview(backdrop)
        addSubview(host)
        more.alignment = .center
        for label in [mark, line, more] { addSubview(label) }
        addSubview(close)
        close.onPress = { [weak self] in self?.onDismiss?() }
        setAccessibilityElement(true)
        setAccessibilityRole(.button)
        describeActions()
    }

    required init?(coder: NSCoder) { nil }

    /// The one custom action's name, in the language now in use.
    func describeActions() {
        setAccessibilityCustomActions([NSAccessibilityCustomAction(name: L10n.t("app.pill.hide", "Hide until the next task")) { [weak self] in
            self?.onDismiss?()
            return true
        }])
    }

    private static func label(_ size: CGFloat, _ weight: NSFont.Weight) -> NSTextField {
        let field = NSTextField(labelWithString: "")
        field.font = .systemFont(ofSize: size, weight: weight)
        field.textColor = Palette.ink
        field.lineBreakMode = .byTruncatingTail
        field.wantsLayer = true
        field.setAccessibilityElement(false)
        return field
    }

    var capsule: NSRect { bounds.insetBy(dx: pillMargin, dy: pillMargin) }

    /// Shows `model` and works out the capsule's width. `fade` is a 150 ms cross-fade of the line from the one before.
    func show(_ model: PillModel, picture image: CGImage?, fade: Bool) {
        if fade && (self.model?.text != model.text || self.model?.glyph != model.glyph) {
            let dissolve = CATransition()
            dissolve.type = .fade
            dissolve.duration = 0.15
            for label in [mark, line] { label.layer?.add(dissolve, forKey: "dissolve") }
        }
        self.model = model
        line.stringValue = model.text
        mark.stringValue = model.glyph
        more.stringValue = model.more > 0 ? "+\(model.more)" : ""
        more.isHidden = model.more == 0
        host.picture.contents = image
        backdrop.model = model
        capsuleWidth = pillWidth(text: ceil(line.cell?.cellSize.width ?? line.intrinsicContentSize.width), more: model.more)
        setAccessibilityLabel(model.label)
        toolTip = model.text
        needsLayout = true
        recolor()
    }

    override func layout() {
        super.layout()
        let body = capsule
        blur.frame = body
        blur.layer?.cornerRadius = body.height / 2
        backdrop.frame = bounds
        host.frame.origin = CGPoint(x: body.minX + 6, y: body.minY + 6)
        mark.frame = NSRect(x: body.minX + 50, y: body.minY + 14, width: 24, height: 20)
        let tail = (model?.more ?? 0) > 0 ? 8 + pillChip + 14 : pillTail
        let wide = min(body.width - pillLead - tail, pillTextMost)
        line.frame = NSRect(x: body.minX + pillLead, y: body.minY + 14, width: max(wide, 0), height: 20)
        more.frame = NSRect(x: body.maxX - 14 - pillChip, y: body.minY + 14, width: pillChip, height: 18)
        close.frame = NSRect(x: body.minX + 4 - 9, y: body.maxY - 4 - 9, width: 18, height: 18)
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        recolor()
    }

    // Layers hold plain colours: they are resolved for the appearance the view has now.
    private func recolor() {
        guard let model = model else { return }
        effectiveAppearance.performAsCurrentDrawingAppearance {
            disc.borderColor = PillColors.ring(model.state).cgColor
            disc.backgroundColor = PillColors.disc.cgColor
        }
        mark.textColor = model.state == .working ? Palette.ink : PillColors.ring(model.state)
        more.textColor = Palette.ink
    }

    /// Breathing while it works, and three pulses of the ring when a wait begins (`entering`); nothing else moves,
    /// and nothing at all when `animate` is false.
    func play(_ state: PillState, entering: Bool, animate: Bool) {
        disc.removeAnimation(forKey: "breath")
        if animate && state == .working {
            let breath = CABasicAnimation(keyPath: "transform.scale")
            breath.fromValue = 1
            breath.toValue = 1.04
            breath.duration = 1.6
            breath.autoreverses = true
            breath.repeatCount = .infinity
            breath.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            disc.add(breath, forKey: "breath")
        }
        if animate && state == .waiting && entering {
            let pulse = CABasicAnimation(keyPath: "borderWidth")
            pulse.fromValue = 2
            pulse.toValue = 4
            pulse.duration = 0.4
            pulse.autoreverses = true
            pulse.repeatCount = 3
            disc.add(pulse, forKey: "pulse")
        } else if !(animate && state == .waiting) {
            disc.removeAnimation(forKey: "pulse")
        }
    }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override var mouseDownCanMoveWindow: Bool { false }

    override func hitTest(_ point: NSPoint) -> NSView? {
        let local = convert(point, from: superview)
        if !close.isHidden && close.frame.contains(local) { return close }
        return capsule.contains(local) ? self : nil
    }

    override func updateTrackingAreas() {
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(rect: .zero, options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect],
                                       owner: self, userInfo: nil))
        super.updateTrackingAreas()
    }

    override func mouseEntered(with event: NSEvent) { close.isHidden = false }
    override func mouseExited(with event: NSEvent) { close.isHidden = true }

    // The press is followed by hand, as the pet's is: a press that stays within a few points is a click, anything
    // more is a drag, and the panel follows the pointer by the distance it moved.
    override func mouseDown(with event: NSEvent) {
        guard let window = window else { return }
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
        if dragged { onDrop?() } else { onClick?() }
        dragged = false
    }

    override func accessibilityPerformPress() -> Bool {
        onClick?()
        return true
    }
}

final class PillPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

/// The "On the desktop" submenu of each menu that offers it (the menu bar's and the pet's), with one shared check mark.
final class DesktopMenu: NSObject {
    var choose: (DesktopMode) -> Void = { _ in }
    private var items: [NSMenuItem] = []
    private var current = DesktopMode.pill

    func makeItem() -> NSMenuItem {
        let menu = NSMenu(title: desktopMenuTitle)
        for mode in DesktopMode.allCases {
            let item = menu.addItem(withTitle: "", action: #selector(pick(_:)), keyEquivalent: "").word(mode.menuWord)
            item.target = self
            item.representedObject = mode.rawValue
            items.append(item)
        }
        let entry = NSMenuItem(title: desktopMenuTitle, action: nil, keyEquivalent: "")
        entry.submenu = menu
        entry.word(.desktop)
        show(current)
        return entry
    }

    func show(_ mode: DesktopMode) {
        current = mode
        for item in items { item.state = item.representedObject as? String == mode.rawValue ? .on : .off }
    }

    @objc private func pick(_ item: NSMenuItem) {
        if let raw = item.representedObject as? String, let mode = DesktopMode(rawValue: raw) { choose(mode) }
    }
}

/// Melo's activity pill: a borderless, non-activating panel above other windows on every Space that takes the pet's place
/// (same spot, right edge kept) while the window is put away and something runs or waits. The health poll feeds it; its
/// only timer is the one `PillPolicy` asks for, and its motion is Core Animation's.
final class PillController: NSObject {
    let panel: PillPanel
    let view = PillView()
    var openWindow: () -> Void = {}
    var openWaiting: () -> Void = {}
    var openRoute: (String) -> Void = { _ in }
    /// Told when the pill appears or goes, and whether the pet that takes its place shows "done" (a "Done" ran out).
    var showingChanged: (_ showing: Bool, _ showsDone: Bool) -> Void = { _, _ in }
    var environment: PetEnvironment { gate.environment }
    var isShown: Bool { panel.isVisible }
    private var policy = PillPolicy()
    private var health: Data?
    private var healthAt: TimeInterval = 0
    private var stopped = false
    private var windowIsShown = true
    private var mode = DesktopMode.pill
    private var model: PillModel?
    private var applied: (state: PillState, animate: Bool)?
    private var timer: Timer?
    private var gate: MotionGate!
    private var images: [String: CGImage] = [:]
    private let pictures: URL?
    private let settings: () -> PetSettings
    private let remember: (String, PetSpot) -> Void
    private let clock: () -> TimeInterval

    /// `pictures` is the `moves` folder; `settings` and `remember` are the pet's saved spots, which the pill shares.
    init(pictures: URL?, settings: @escaping () -> PetSettings, remember: @escaping (String, PetSpot) -> Void,
         clock: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }) {
        self.pictures = pictures
        self.settings = settings
        self.remember = remember
        self.clock = clock
        panel = PillPanel(contentRect: NSRect(origin: .zero, size: pillPanelSize(width: 240)),
                          styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        super.init()
        panel.contentView = view
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle]
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.hidesOnDeactivate = false
        panel.canHide = false
        panel.isMovable = false
        panel.isReleasedWhenClosed = false
        panel.isExcludedFromWindowsMenu = true
        panel.title = "Melo"
        view.onClick = { [weak self] in self?.click() }
        view.onDrop = { [weak self] in self?.drop() }
        view.onDismiss = { [weak self] in
            self?.policy.dismiss()
            self?.evaluate()
        }
        gate = MotionGate(window: { [weak self] in self?.panel }, visible: { [weak self] in
            guard let panel = self?.panel else { return false }
            return panel.isVisible && panel.occlusionState.contains(.visible)
        })
        gate.changed = { [weak self] _ in self?.applyMotion(entering: false) }
        NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification,
                                               object: nil, queue: .main) { [weak self] _ in
            if self?.isShown == true { self?.place() }
        }
    }

    /// The last health reply (nil: none yet), and whether the app itself stopped.
    func update(health: Data?, stopped: Bool) {
        self.health = health
        healthAt = clock()
        self.stopped = stopped
        evaluate()
    }

    /// Whether the main window is in front of the person.
    func windowShown(_ shown: Bool) {
        windowIsShown = shown
        evaluate()
    }

    /// The console's language changed: the line and the labels are written again from the last health reply.
    func languageChanged() {
        view.describeActions()
        evaluate()
    }

    func desktop(_ mode: DesktopMode) {
        self.mode = mode
        evaluate()
    }

    private func evaluate() {
        let decision = policy.update(health: health, healthAt: healthAt, stopped: stopped, windowShown: windowIsShown,
                                     mode: mode, now: clock())
        timer?.invalidate()
        timer = decision.recheckIn.map { seconds in
            Timer.scheduledTimer(withTimeInterval: seconds, repeats: false) { [weak self] _ in self?.evaluate() }
        }
        let was = panel.isVisible
        guard let next = decision.pill else {
            model = nil
            if was {
                panel.orderOut(nil)
                view.close.isHidden = true  // no mouse-exit comes to a panel that goes
                view.play(.done, entering: false, animate: false)
                applied = nil
                gate.refresh()
            }
            if was || decision.petShowsDone { showingChanged(false, decision.petShowsDone) }
            return
        }
        let before = model
        model = next
        view.show(next, picture: picture(for: next.portrait), fade: was && before != nil)
        let size = pillPanelSize(width: view.capsuleWidth)
        if !was {
            place()
            panel.orderFrontRegardless()
            gate.refresh()
            showingChanged(true, false)
        } else if panel.frame.size != size, let screen = screen(of: panel.frame) {
            panel.setFrame(pillResized(panel.frame, size: size, in: screen.visible), display: true)
        }
        panel.invalidateShadow()
        applyMotion(entering: !was || before?.state != next.state)
    }

    // The animations run on their own; they start again only when the state or what may move changed.
    private func applyMotion(entering: Bool) {
        guard let state = model?.state else { return }
        let animate = shouldAnimate(environment)
        if !entering, let done = applied, done.state == state, done.animate == animate { return }
        applied = (state, animate)
        view.play(state, entering: entering, animate: animate)
    }

    private func picture(for portrait: PillPortrait) -> CGImage? {
        let file = pillPictureFile(portrait)
        if let image = images[file] { return image }
        guard let url = pictures?.appendingPathComponent(file),
              let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        images[file] = image
        return image
    }

    private func click() {
        guard let target = model?.click else { return }
        switch target {
        case .waiting: openWaiting()
        case .route(let route): openRoute(route)
        case .window: openWindow()
        }
    }

    /// Dropped: snapped to the bottom edge if near it, kept on the screen it is mostly on, the spot saved as the pet's.
    private func drop() {
        guard let screen = screen(of: panel.frame) else { return }
        let dropped = pillDropped(panel.frame, in: screen.visible)
        panel.setFrame(dropped.frame, display: true)
        remember(screen.key, dropped.spot)
    }

    private func screen(of frame: CGRect) -> PetScreen? {
        func overlap(_ area: CGRect) -> CGFloat {
            let common = area.intersection(frame)
            return common.isNull ? 0 : common.width * common.height
        }
        return PetController.screens().max(by: { overlap($0.visible) < overlap($1.visible) })
    }

    private func place() {
        let size = pillPanelSize(width: view.capsuleWidth)
        guard let placement = pillPlacement(settings(), size: size, screens: PetController.screens()) else { return }
        panel.setFrame(placement.frame, display: false)
    }
}
