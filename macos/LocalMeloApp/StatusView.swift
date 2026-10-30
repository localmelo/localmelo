import AppKit

/// The console's colours, so the window around the console reads as part of it.
/// Each pair is light, then dark: the console's canvas and ink tokens.
enum Palette {
    private static func pair(_ light: UInt32, _ dark: UInt32) -> NSColor {
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
}

/// Melo's picture as a framed print: its paper tile reads as the print, not as a
/// bright patch, when the window is dark.
final class PortraitView: NSView {
    private let image: NSImage

    init(_ image: NSImage) {
        self.image = image
        super.init(frame: .zero)
        setAccessibilityElement(false)
    }

    required init?(coder: NSCoder) { nil }

    override var intrinsicContentSize: NSSize { NSSize(width: 96, height: 128) }

    override func draw(_ dirtyRect: NSRect) {
        let frame = NSBezierPath(roundedRect: bounds.insetBy(dx: 0.5, dy: 0.5), xRadius: 12, yRadius: 12)
        NSGraphicsContext.saveGraphicsState()
        frame.addClip()
        image.draw(in: bounds)
        NSGraphicsContext.restoreGraphicsState()
        NSColor.separatorColor.setStroke()
        frame.stroke()
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

    init(portrait: NSImage?) {
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
        spinner.setAccessibilityLabel("Starting localmelo")
        working.spacing = 8
        working.addArrangedSubview(spinner)
        working.addArrangedSubview(step)
        actions.spacing = 8
        var views: [NSView] = portrait.map { [PortraitView($0)] } ?? []
        views += [title, working, reason, actions]
        let column = NSStackView(views: views)
        column.orientation = .vertical
        column.alignment = .centerX
        column.spacing = 8
        if let first = views.first as? PortraitView { column.setCustomSpacing(24, after: first) }
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
        showStarting(step: startStepText(portKnown: false, ready: false))
    }

    required init?(coder: NSCoder) { nil }

    override var isOpaque: Bool { true }

    override func draw(_ dirtyRect: NSRect) {
        Palette.canvas.setFill()
        dirtyRect.fill()
    }

    func showStarting(step text: String) {
        title.stringValue = "Starting localmelo…"
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
