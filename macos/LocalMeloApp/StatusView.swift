import AppKit

/// The console's colours, so the window around the console reads as part of it.
/// Each pair is light, then dark: the console's canvas and ink tokens.
enum Palette {
    static func pair(_ light: UInt32, _ dark: UInt32) -> NSColor {
        NSColor(name: nil) { appearance in
            let hex = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? dark : light
            return NSColor(srgbRed: CGFloat(hex >> 16 & 0xff) / 255, green: CGFloat(hex >> 8 & 0xff) / 255,
                           blue: CGFloat(hex & 0xff) / 255, alpha: 1)
        }
    }

    static let canvas = pair(0xfbfbf9, 0x171916)
    static let ink = pair(0x1d1f1b, 0xeef0e9)
    static let secondary = pair(0x4d5149, 0xc2c7ba)
    static let onInk = pair(0xffffff, 0x1e211c)
    // The menu-bar dot and the ring that sets it off from the logo and the bar. The red is at least 3:1
    // against the ring and against a light or a dark bar of the system's own greys.
    static let badge = pair(0xd70015, 0xff453a)
    static let badgeRing = pair(0xffffff, 0x1c1c1e)
}

/// Melo's portrait in the mood the window is in: a still of the moves (`startPortrait`). The pictures are transparent,
/// so they stand on the console's canvas in light and dark with no frame or tile. It is decoration: the title below
/// says the same in words, so VoiceOver skips it.
final class PortraitView: NSImageView {
    private let pictures: URL?
    private var images: [PetMove: NSImage] = [:]
    private(set) var state: PetMove = .startup

    /// `pictures` is the `moves` folder of the bundle's resources.
    init(pictures: URL?) {
        self.pictures = pictures
        super.init(frame: .zero)
        imageScaling = .scaleProportionallyUpOrDown
        setAccessibilityElement(false)
        show(.startup)
    }

    required init?(coder: NSCoder) { nil }

    /// The state's poster at the canvas's size; with no picture (a run outside the bundle) the portrait takes no room.
    func show(_ next: PetMove) {
        state = next
        if images[next] == nil, let url = pictures?.appendingPathComponent("poster-\(next.rawValue).webp"),
           let image = NSImage(contentsOf: url) {
            image.size = portraitSize
            images[next] = image
        }
        image = images[next]
        isHidden = image == nil
    }
}

/// What the window shows until the console has loaded, and when it cannot load:
/// Melo's portrait and a title, then either a spinner with the current step or the
/// reason with the actions that can fix it. It paints the console's canvas, so the
/// window never shows an empty web view or a white frame.
final class StatusView: NSView {
    var onAction: ((FailureAction) -> Void)?
    /// The buttons of the failure view, by action; the integration tests click them.
    private(set) var buttons: [FailureAction: NSButton] = [:]
    private let title = NSTextField(labelWithString: "")
    private let step = NSTextField(labelWithString: "")
    private let reason = NSTextField(wrappingLabelWithString: "")
    private let spinner = NSProgressIndicator()
    private let working = NSStackView()
    private let actions = NSStackView()

    private let portraitView: PortraitView

    /// The still on show.
    var portrait: PetMove { portraitView.state }

    init(pictures: URL?) {
        portraitView = PortraitView(pictures: pictures)
        super.init(frame: .zero)
        title.font = .systemFont(ofSize: 22, weight: .semibold)
        title.textColor = Palette.ink
        step.font = .systemFont(ofSize: 13)
        step.textColor = Palette.secondary
        reason.font = .systemFont(ofSize: 14)
        reason.textColor = Palette.secondary
        reason.alignment = .center
        reason.preferredMaxLayoutWidth = 420
        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.isIndeterminate = true
        spinner.setAccessibilityLabel(L10n.t("app.start.label", "Starting localmelo"))
        working.spacing = 8
        working.addArrangedSubview(spinner)
        working.addArrangedSubview(step)
        actions.spacing = 8
        let column = NSStackView(views: [portraitView, title, working, reason, actions])
        column.orientation = .vertical
        column.alignment = .centerX
        column.spacing = 8
        column.setCustomSpacing(24, after: portraitView)
        column.setCustomSpacing(24, after: reason)
        column.translatesAutoresizingMaskIntoConstraints = false
        addSubview(column)
        // Anchored by its top, so Melo and the title stay put when the reason and
        // the buttons replace the spinner.
        NSLayoutConstraint.activate([
            column.centerXAnchor.constraint(equalTo: centerXAnchor),
            column.topAnchor.constraint(equalTo: centerYAnchor, constant: -150),
            column.leadingAnchor.constraint(greaterThanOrEqualTo: leadingAnchor, constant: 32),
            column.trailingAnchor.constraint(lessThanOrEqualTo: trailingAnchor, constant: -32),
        ])
        showStarting(step: startStepText(portKnown: false, ready: false), ready: false)
    }

    required init?(coder: NSCoder) { nil }

    override var isOpaque: Bool { true }

    override func draw(_ dirtyRect: NSRect) {
        Palette.canvas.setFill()
        dirtyRect.fill()
    }

    /// `ready` is true once the gateway is up and the console is loading; the portrait is the same either way.
    func showStarting(step text: String, ready: Bool) {
        portraitView.show(startPortrait(failed: false))
        title.stringValue = L10n.t("app.start.title", "Starting localmelo…")
        step.stringValue = text
        working.isHidden = false
        reason.isHidden = true
        actions.isHidden = true
        spinner.startAnimation(nil)
        isHidden = false
    }

    func hide() {
        spinner.stopAnimation(nil)
        isHidden = true
    }

    func showFailure(_ notice: FailureNotice) {
        portraitView.show(startPortrait(failed: true))
        spinner.stopAnimation(nil)
        title.stringValue = notice.title
        reason.stringValue = notice.reason
        working.isHidden = true
        reason.isHidden = false
        actions.isHidden = false
        actions.arrangedSubviews.forEach { $0.removeFromSuperview() }
        buttons = [:]
        for (index, action) in notice.actions.enumerated() {
            let button = NSButton(title: action.title, target: self, action: #selector(clicked(_:)))
            button.bezelStyle = .rounded
            button.controlSize = .large
            if index == 0 {
                // The one primary button is filled with ink, as in the console.
                button.keyEquivalent = "\r"
                button.bezelColor = Palette.ink
                button.attributedTitle = NSAttributedString(string: action.title,
                                                            attributes: [.foregroundColor: Palette.onInk])
            }
            buttons[action] = button
            actions.addArrangedSubview(button)
        }
        isHidden = false
        NSAccessibility.post(element: self, notification: .announcementRequested,
                             userInfo: [.announcement: "\(notice.title). \(notice.reason)",
                                        .priority: NSAccessibilityPriorityLevel.high.rawValue])
    }

    @objc private func clicked(_ sender: NSButton) {
        if let action = buttons.first(where: { $0.value === sender })?.key { onAction?(action) }
    }
}
