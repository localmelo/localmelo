import CoreGraphics
import Foundation

/// The activity pill: a capsule that takes the pet's place on the desktop while the window is put away and something
/// runs or waits, and says in one plain line what Melo is doing. Like `PetPolicy.swift`, this file decides everything:
/// what it shows from the health reply, when it is hidden, when the pet comes back, and where its panel goes.

/// Seconds a line stays before another may replace it; seconds the pet wags when it comes back after "Done"; and how old
/// a health reply may be before the pill stops believing it (the gateway may have hung).
let pillTextEvery: TimeInterval = 1.5
let pillWagFor: TimeInterval = 3
let pillStaleAfter: TimeInterval = 12

enum PillState: Equatable {
    case working, waiting, done, failed, serverDown

    /// Something runs or waits now (as against an ended run, which only shows for a few seconds).
    var isActive: Bool { self != .done && self != .failed }
}

/// What a running thing is doing, as the gateway names it (`ActivityText.icon`); an unknown name reads as a tool.
enum PillIcon: String {
    case wait, held, paused, read, write, run, web, memory, think, tool

    init(name: String?) { self = name.flatMap(PillIcon.init(rawValue:)) ?? .tool }

    /// The same glyphs as the console's panel, so a row reads the same everywhere.
    var glyph: String {
        let glyphs: [PillIcon: String] = [.wait: "!", .held: "\u{25F7}", .paused: "\u{2016}", .read: "\u{25A4}",
                                          .write: "\u{270E}", .run: ">_", .web: "\u{2295}", .memory: "\u{25CE}",
                                          .think: "\u{2026}", .tool: "\u{2699}"]
        return glyphs[self] ?? "\u{2699}"
    }
}

/// Which picture of Melo the pill's round portrait shows. The four working and waiting ones are new art, still to be
/// made (`slot`, a file under `pill/` of the pictures folder); until a file is there each falls back to today's face
/// (`face`, a file under `face/`).
enum PillPortrait: Equatable, Hashable {
    case workLaptop, workRead, workThink, waitAlert, done, error

    var slot: String? {
        let slots: [PillPortrait: String] = [.workLaptop: "work-laptop", .workRead: "work-read",
                                             .workThink: "work-think", .waitAlert: "wait-alert"]
        return slots[self]
    }

    var face: String {
        switch self {
        case .workLaptop, .workRead, .workThink: return "working"
        case .waitAlert: return "waiting"
        case .done: return "done"
        case .error: return "error"
        }
    }

    /// The working pose for what Melo does: typing, reading a page, or thinking.
    static func working(_ icon: PillIcon) -> PillPortrait {
        switch icon {
        case .read, .web: return .workRead
        case .think, .memory, .held, .paused, .wait: return .workThink
        case .write, .run, .tool: return .workLaptop
        }
    }
}

/// The file of a portrait for a screen scale, relative to the pictures folder: the new art's slot if `exists` says it
/// is there, else today's face.
func pillPictureFile(_ portrait: PillPortrait, scale: Int, exists: (String) -> Bool) -> String {
    func file(_ folder: String, _ name: String) -> String { "\(folder)/\(name)\(scale >= 2 ? "@2x" : "").png" }
    if let slot = portrait.slot, exists(file("pill", slot)) { return file("pill", slot) }
    return file("face", portrait.face)
}

/// Where a press on the pill goes: to the chat or task that waits, to the console route of the focus chat or task, or
/// just to the window (a run or terminal chat has no page of its own).
enum PillClick: Equatable {
    case waiting, route(String), window
}

/// What the pill shows.
struct PillModel: Equatable {
    var state: PillState
    var text: String
    var icon: PillIcon
    /// How many more run beside the one it names ("+1"); 0 shows no chip.
    var more: Int
    var portrait: PillPortrait
    var click: PillClick

    /// The mark before the line: the kind of work while it runs, else the end's own.
    var glyph: String {
        switch state {
        case .working, .waiting: return icon.glyph
        case .done: return "\u{2713}"
        case .failed: return "\u{2717}"
        case .serverDown: return "\u{2298}"
        }
    }

    /// What VoiceOver reads: the one line of the pill, in a sentence, and where a press goes.
    var label: String {
        let line = text.prefix(1).lowercased() + text.dropFirst()
        let subject: String
        switch state {
        case .working, .waiting, .done: subject = "Melo is \(line)."
        case .failed: subject = "Melo \(line)."
        case .serverDown: subject = "Melo: \(line)."
        }
        if case .route(let route) = click, route.hasPrefix("#/tasks/") { return subject + " Opens the task." }
        return subject + " Opens the chat."
    }
}

/// `status.focus` of `GET /api/health`: the one running thing a glance shows, and what it is doing in plain words.
struct PillFocus: Equatable {
    let kind: String
    let id: String
    let activity: String
    let icon: PillIcon
}

func pillFocus(_ data: Data) -> PillFocus? {
    guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let status = body["status"] as? [String: Any],
          let entry = status["focus"] as? [String: Any],
          let kind = entry["kind"] as? String, ["chat", "task", "run", "terminal"].contains(kind),
          let id = entry["id"] as? String, (1...128).contains(id.count) else { return nil }
    let activity = plainLine(entry["activity"] as? String, limit: 200)
    guard !activity.isEmpty else { return nil }
    return PillFocus(kind: kind, id: id, activity: activity, icon: PillIcon(name: entry["icon"] as? String))
}

/// The ids of everything `status.items` lists as running.
func runningIDs(_ data: Data) -> Set<String> {
    guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let status = body["status"] as? [String: Any],
          let items = status["items"] as? [[String: Any]] else { return [] }
    return Set(items.compactMap { $0["id"] as? String })
}

/// What a health reply `age` seconds old shows, the ids it is about (what "×" hides) and an ended run's seconds left.
struct PillSnapshot: Equatable {
    var model: PillModel
    var ids: Set<String>
    var expiresIn: TimeInterval?
}

func pillSnapshot(health data: Data, age: TimeInterval) -> PillSnapshot? {
    guard let counts = runningCounts(data) else { return nil }
    let more = max(0, counts.total - 1)
    let waits = waitingItems(data)
    let focus = pillFocus(data)
    let ids = runningIDs(data).union(waits.map { $0.id }).union(focus.map { [$0.id] } ?? [])
    let route = focus.flatMap { consoleRoute(kind: $0.kind, id: $0.id) }.map(PillClick.route) ?? .window
    func active(_ state: PillState, _ text: String, _ icon: PillIcon, _ portrait: PillPortrait, _ click: PillClick)
        -> PillSnapshot {
        PillSnapshot(model: PillModel(state: state, text: text, icon: icon, more: more, portrait: portrait, click: click),
                     ids: ids.isEmpty ? ["running"] : ids)
    }
    if !waits.isEmpty { return active(.waiting, "Waiting for your approval", .wait, .waitAlert, .waiting) }
    if counts.total > 0 {
        let icon = focus?.icon ?? .think
        if modelServerReachable(data) == false {
            return active(.serverDown, "Model server not reachable", icon, .error, route)
        }
        return active(.working, focus?.activity ?? "Working", icon, .working(icon), route)
    }
    guard let recent = recentOutcome(data) else { return nil }
    let ended: (PillState, String, TimeInterval)
    switch recent.outcome {
    case .done: ended = (.done, "Done", petDoneFor)
    case .failed: ended = (.failed, "Stopped", petFailedFor)
    case .stopped: return nil
    }
    let (state, word, shownFor) = ended
    let left = shownFor - (recent.endedAgo + age)
    guard left > 0 else { return nil }
    let text = recent.title.isEmpty ? word : "\(word): \(recent.title)"
    let target = consoleRoute(kind: recent.kind, id: recent.id).map(PillClick.route) ?? .window
    return PillSnapshot(model: PillModel(state: state, text: text, icon: .tool, more: 0,
                                         portrait: state == .done ? .done : .error, click: target),
                        ids: [recent.id], expiresIn: left)
}

/// What the pill does for one health reply.
struct PillDecision: Equatable {
    /// What to show; nil: no pill, and the pet lies there.
    var pill: PillModel?
    /// The pill's "Done" just ended: the pet comes back wagging.
    var petWags = false
    /// Seconds until the policy wants to be asked again (an ended run's time is up, a held line may change).
    var recheckIn: TimeInterval?
}

/// The pill's memory: what it shows, when its line last changed, and what "×" hid. It is asked on each health reply,
/// each change of window or menu choice, and when it asked to be (`recheckIn`); `now` is any clock in seconds.
struct PillPolicy {
    private var shown: PillModel?
    private var changedAt = -TimeInterval.infinity
    private var hidden: Set<String>?
    private var quietSinceHidden = false
    private var last: PillSnapshot?

    /// "×": hide the pill until the next task starts: something that is not running now, or anything once all has stopped.
    mutating func dismiss() {
        hidden = last?.ids ?? []
        quietSinceHidden = false
        shown = nil
    }

    mutating func update(health: Data?, healthAt: TimeInterval, stopped: Bool, windowShown: Bool, mode: DesktopMode,
                         now: TimeInterval) -> PillDecision {
        let age = now - healthAt
        let believed = !stopped && age >= 0 && age < pillStaleAfter
        let live = believed ? health.flatMap { pillSnapshot(health: $0, age: age) } : nil
        last = live
        trackDismissal(live)
        let previous = shown
        guard mode == .pill, !windowShown, hidden == nil, var model = live?.model else {
            shown = nil
            // The wag is for a "Done" that ran out on the desktop, not one the person's window or choice cut short.
            let wags = previous?.state == .done && mode == .pill && !windowShown && hidden == nil && !stopped
            return PillDecision(pill: nil, petWags: wags)
        }
        var waits = [live?.expiresIn, pillStaleAfter - age].compactMap { $0 }
        if let held = previous, held.state == model.state, held.text != model.text, now - changedAt < pillTextEvery {
            // A line that changed a moment ago stays; the new one comes when the time is up.
            (model.text, model.icon, model.portrait) = (held.text, held.icon, held.portrait)
            waits.append(changedAt + pillTextEvery - now)
        } else if previous?.text != model.text || previous?.state != model.state {
            changedAt = now
        }
        shown = model
        return PillDecision(pill: model, recheckIn: max(0.05, waits.min() ?? pillStaleAfter))
    }

    private mutating func trackDismissal(_ live: PillSnapshot?) {
        guard let gone = hidden else { return }
        guard let seen = live else { quietSinceHidden = true; return }
        let new = !seen.ids.isSubset(of: gone)
        if new || (seen.model.state.isActive && quietSinceHidden) {
            hidden = nil
        } else if !seen.model.state.isActive {
            quietSinceHidden = true
        }
    }
}

/// The capsule's height and the clear strip around it in the panel (room for the "×" and the shadow); the line is cut
/// with an ellipsis at `pillTextMost`; points before the line (portrait and mark) and after it; the "+1" chip's width.
let pillHeight: CGFloat = 48
let pillMargin: CGFloat = 5
let pillTextMost: CGFloat = 320
let pillLead: CGFloat = 74
let pillTail: CGFloat = 18
let pillChip: CGFloat = 30

/// The capsule's width for a line of `text` points and the "+1" chip when `more` is above 0.
func pillWidth(text: CGFloat, more: Int) -> CGFloat {
    pillLead + min(max(text, 0), pillTextMost) + (more > 0 ? 8 + pillChip + 14 : pillTail)
}

/// The panel around a capsule of `width`.
func pillPanelSize(width: CGFloat) -> CGSize {
    CGSize(width: width + 2 * pillMargin, height: pillHeight + 2 * pillMargin)
}

/// The panel's frame at the pet's saved spot (or first one), right edges together, clamped so it is never over the menu
/// bar or the Dock.
func pillPlacement(_ settings: PetSettings, size: CGSize, screens: [PetScreen]) -> (screen: PetScreen, frame: CGRect)? {
    guard let screen = screens.first(where: { $0.key == settings.screen }) ?? screens.first else { return nil }
    if let spot = settings.spots[screen.key] {
        return (screen, petFrame(size: size, spot: spot, belowPaws: 0, on: screen.visible))
    }
    return (screen, petDefaultFrame(size: size, belowPaws: 0, in: screen.visible))
}

/// A panel whose line changed: the new `size`, the right edge where it was, kept on the screen.
func pillResized(_ frame: CGRect, size: CGSize, in visible: CGRect) -> CGRect {
    petClamped(CGRect(x: frame.maxX - size.width, y: frame.minY, width: size.width, height: size.height),
               belowPaws: 0, in: visible)
}

/// A dropped panel: snapped to the bottom edge when near it, kept on the screen, and the spot to save (the pet's).
func pillDropped(_ frame: CGRect, in visible: CGRect) -> (frame: CGRect, spot: PetSpot) {
    let dropped = petClamped(petSnapped(frame, belowPaws: 0, in: visible), belowPaws: 0, in: visible)
    return (dropped, petSpot(of: dropped, on: visible))
}
