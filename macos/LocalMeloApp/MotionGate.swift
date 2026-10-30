import AppKit

/// What stops a loop on one view: Reduce Motion, Low Power Mode, heat, its window covered or hidden, the screen
/// asleep or the session switched out (`PetEnvironment`; `shouldAnimate` decides). `changed` hears each change of the
/// environment; `refresh()` reads everything again, so a notification that never came cannot leave a loop stuck.
final class MotionGate {
    private(set) var environment = PetEnvironment()
    var changed: (PetEnvironment) -> Void = { _ in }
    private let window: () -> NSWindow?
    private let visible: () -> Bool
    private var screenOn = true
    private var sessionActive = true
    private var observers: [(NotificationCenter, NSObjectProtocol)] = []

    /// `window` is the window the view is in (for its occlusion) and `visible` whether the view can be seen at all:
    /// the window is ordered in and uncovered and the view is not hidden.
    init(window: @escaping () -> NSWindow?, visible: @escaping () -> Bool) {
        self.window = window
        self.visible = visible
        let center = NotificationCenter.default, workspace = NSWorkspace.shared.notificationCenter
        let reread: (Notification) -> Void = { [weak self] _ in self?.refresh() }
        let occluded: (Notification) -> Void = { [weak self] note in
            if let window = self?.window(), note.object as? NSWindow === window { self?.refresh() }
        }
        watch(center, NSWindow.didChangeOcclusionStateNotification, occluded)
        for name in [ProcessInfo.thermalStateDidChangeNotification, Notification.Name.NSProcessInfoPowerStateDidChange] {
            watch(center, name, reread)
        }
        watch(workspace, NSWorkspace.accessibilityDisplayOptionsDidChangeNotification, reread)
        // The screen being off and the session being switched out stop the loops the same way.
        let awake: [(Notification.Name, (MotionGate) -> Void)] = [
            (NSWorkspace.screensDidSleepNotification, { $0.screenOn = false }),
            (NSWorkspace.screensDidWakeNotification, { $0.screenOn = true }),
            (NSWorkspace.sessionDidResignActiveNotification, { $0.sessionActive = false }),
            (NSWorkspace.sessionDidBecomeActiveNotification, { $0.sessionActive = true })]
        for (name, set) in awake {
            watch(workspace, name) { [weak self] _ in
                guard let self = self else { return }
                set(self)
                self.refresh()
            }
        }
        refresh()
    }

    deinit { observers.forEach { $0.0.removeObserver($0.1) } }

    private func watch(_ center: NotificationCenter, _ name: Notification.Name, _ body: @escaping (Notification) -> Void) {
        observers.append((center, center.addObserver(forName: name, object: nil, queue: .main, using: body)))
    }

    func refresh() {
        var now = PetEnvironment()
        now.reduceMotion = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        now.lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        now.thermal = ProcessInfo.processInfo.thermalState
        now.visible = visible()
        now.screenAwake = screenOn && sessionActive
        guard now != environment else { return }
        environment = now
        changed(now)
    }
}
