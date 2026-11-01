import CoreGraphics
import Foundation

/// What Melo holds when nothing moves for a reason, beside the moves (`poster-<name>.webp`, from the stills of
/// `scripts/build_moves.py`): a run that ended well (eyes closed and happy, tail up), a failure (ears back, head low),
/// and nothing known or no connection (curled up, shown at `petOfflineOpacity`).
enum PetStill: String, CaseIterable {
    case done, error, offline

    var file: String { "poster-\(rawValue).webp" }
    /// The opacity it is shown at: the offline still is faded, as the console's is.
    var opacity: Float { self == .offline ? petOfflineOpacity : 1 }
    /// The poster of the moves that stands in when a folder has no stills (one made before them).
    var standIn: PetMove {
        switch self {
        case .done: return .startup
        case .error: return .thinking
        case .offline: return .idleRest
        }
    }
}

let petOfflineOpacity: Float = 0.6

/// The picture of the app's first window, by its name among the moves' posters (`poster-<name>.webp`): the end of the
/// wake-up (Melo sitting up, smiling) while the server starts and once it is ready; the error still (ears back, head
/// low) when it cannot go on, so a failure never reads as a healthy start.
func startPortrait(failed: Bool) -> String { failed ? PetStill.error.rawValue : PetMove.startup.rawValue }

/// The points the portrait is shown at: the moves' square canvas, so the 256 px poster is sharp on a Retina screen.
let portraitSize = CGSize(width: 128, height: 128)

/// How the newest finished run or chat turn ended: `done`, `failed` (anything the console words as a warning or an
/// error) or `stopped` (the person stopped it).
enum RunOutcome: String {
    case done, failed, stopped
}

/// `status.recent` of `GET /api/health`: what finished last, and how many seconds ago.
struct RecentOutcome: Equatable {
    let kind: String
    let id: String
    let outcome: RunOutcome
    let endedAgo: TimeInterval
    /// What it was called; empty when the gateway did not say.
    var title = ""
}

/// The newest outcome, or nil when none finished yet, the field is missing (an older gateway) or damaged.
func recentOutcome(_ data: Data) -> RecentOutcome? {
    guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let status = body["status"] as? [String: Any],
          let entry = status["recent"] as? [String: Any],
          let kind = entry["kind"] as? String, ["chat", "task", "run", "terminal"].contains(kind),
          let id = entry["id"] as? String, (1...128).contains(id.count),
          let word = entry["outcome"] as? String, let outcome = RunOutcome(rawValue: word),
          let ago = entry["ended_s_ago"] as? Double, ago.isFinite, ago >= 0 else { return nil }
    return RecentOutcome(kind: kind, id: id, outcome: outcome, endedAgo: ago,
                         title: plainLine(entry["title"] as? String, limit: 200))
}

/// Seconds a finished run shows as done, and a failed one as an error, before the pet goes back to rest: the console
/// panel's 8 s for both.
let petDoneFor: TimeInterval = 8
let petFailedFor: TimeInterval = 8
/// Seconds of quiet before the pet falls asleep.
let petSleepAfter: TimeInterval = 60

/// What the pet is doing. What it looks like is `petMove` (a move) or `petStill` (a still); the words differ where two
/// moods look the same.
enum PetMood: Equatable {
    case unknown, resting, sleeping, working(Int), waiting, done, failed, stopped
    /// No model is set up (`configured` false), or the model server does not answer (`model_server.reachable`
    /// false): the console's header says the same and shows its error picture.
    case noModel, unreachable

    /// What VoiceOver reads, and what hovering shows.
    var label: String {
        switch self {
        case .unknown: return "Melo"
        case .resting: return L10n.t("app.pet.resting", "Melo, resting. Nothing running.")
        case .sleeping: return L10n.t("app.pet.sleeping", "Melo, sleeping. Nothing running.")
        case .working(let n): return L10n.t("app.pet.working", "Melo, working. {n} running.", ["n": String(n)])
        case .waiting: return L10n.t("app.pet.waiting", "Melo, waiting for your approval.")
        case .done: return L10n.t("app.pet.done", "Melo, done.")
        case .failed: return L10n.t("app.pet.failed", "Melo, the last run did not finish.")
        case .stopped: return L10n.t("app.pet.stopped", "Melo, localmelo stopped.")
        case .noModel: return L10n.t("app.pet.noModel", "Melo, no model is set up yet.")
        case .unreachable: return L10n.t("app.pet.unreachable", "Melo, the model server is not reachable.")
        }
    }

    /// What pressing the pet does.
    var help: String {
        self == .waiting ? L10n.t("app.pet.helpWaiting", "Opens what waits for your approval")
            : L10n.t("app.pet.help", "Opens localmelo")
    }
}

/// The pet's mood from the last health reply (nil: none yet, or the gateway did not answer). First match wins, in the
/// console header's order: the app's own failure, then no model or a model server that does not answer, then a wait,
/// then work, then a run that just finished, then rest. `sinceTouch` is the
/// seconds since the person last touched the pet (hover, click, drag) or since it appeared; the quiet that puts
/// it to sleep is the shorter of that and the age of the newest finished run, so a touch wakes it into rest.
func petMood(health: Data?, stopped: Bool, sinceTouch: TimeInterval) -> PetMood {
    if stopped { return .stopped }
    guard let data = health, let counts = runningCounts(data) else { return .unknown }
    if healthConfigured(data) == false { return .noModel }
    if modelServerReachable(data) == false { return .unreachable }
    if !waitingItems(data).isEmpty { return .waiting }
    if counts.total > 0 { return .working(counts.total) }
    let recent = recentOutcome(data)
    if let recent = recent {
        if recent.outcome == .done && recent.endedAgo < petDoneFor { return .done }
        if recent.outcome == .failed && recent.endedAgo < petFailedFor { return .failed }
    }
    let quiet = min(sinceTouch, recent?.endedAgo ?? .infinity)
    guard !quiet.isNaN, quiet >= petSleepAfter else { return .resting }
    return .sleeping
}

/// `configured` of `GET /api/health`: false while no model is set up, nil when the reply does not say.
func healthConfigured(_ data: Data) -> Bool? {
    guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
    return body["configured"] as? Bool
}

/// Melo's moves: the animated states the desktop pet plays (`PetMoves.swift`), the files and the rules the console's
/// portrait shares (`support/gateway/static/moves`, `moves.mjs`).
enum PetMove: String, CaseIterable {
    case startup, thinking, working, attention
    case idleRest = "idle_rest", idlePlay = "idle_play"
}

/// The move a mood shows; nil where the pet shows a still (`petStill`) with its words. A wait raises a paw, work types
/// at the laptop (the model alone answering, with no tool, approval or hold open: `thinking`), rest and sleep lie
/// down.
func petMove(_ mood: PetMood, thinking: Bool) -> PetMove? {
    switch mood {
    case .waiting: return .attention
    case .working: return thinking ? .thinking : .working
    case .resting, .sleeping: return .idleRest
    case .unknown, .done, .failed, .stopped, .noModel, .unreachable: return nil
    }
}

/// The still a mood shows; nil where the mood has moves. Nothing known yet is the offline still (curled up, faded); a
/// run that just finished is done (eyes closed and happy, tail up); a failure, a stopped app, no model and a model
/// server that does not answer are the error still (ears back, head low), so they never look like a healthy rest.
/// The words say which it is.
func petStill(_ mood: PetMood) -> PetStill? {
    switch mood {
    case .unknown: return .offline
    case .done: return .done
    case .failed, .stopped, .noModel, .unreachable: return .error
    case .resting, .sleeping, .working, .waiting: return nil
    }
}

/// Whether everything that runs waits on the model: every item of `status.items` is `answering` (its newest step
/// is a model call, with no tool, approval or hold open), and the list leaves nothing out. False for an older
/// gateway without the mark: the pet never guesses that it thinks.
func petThinking(_ data: Data) -> Bool {
    guard let counts = runningCounts(data), counts.total > 0,
          let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let status = body["status"] as? [String: Any],
          let items = status["items"] as? [[String: Any]] else { return false }
    return items.count >= counts.total && items.allSatisfy { ($0["answering"] as? Bool) == true }
}

/// While the pet rests it plays once (one loop of `idle_play`, 6 s) after `petPlayAfter` seconds of rest, at most
/// once every `petPlayEvery`; asleep it only rests. The console's portrait plays by the same numbers.
let petPlayAfter: TimeInterval = 20
let petPlayEvery: TimeInterval = 300
/// Seconds the pet sits after the wake-up, on its last frame, before it lies down to rest (`SIT_MS` of the console).
let petSitFor: TimeInterval = 8

/// The state a move shows, and whether a looping state plays once. The app's first move is the wake-up (`startup`,
/// once) if the pet is at rest then; it sits on its last frame for `petSitFor` seconds, then rests; a play begins by
/// the rule above and goes on to its end. Any other mood ends the wake-up, the sitting, the rest and a play at once.
/// `ended` hears a state that played once reach its end.
struct PetMovePicker: Equatable {
    private(set) var startup = true
    private(set) var sitUntil: Date?
    private(set) var restSince: Date?
    private(set) var playedAt = Date.distantPast
    private(set) var playing = false

    /// `still`: nothing may move (posters only), so there is no play. Nil for a mood without moves (a still).
    mutating func pick(_ mood: PetMood, thinking: Bool, still: Bool, now: Date) -> (move: PetMove, once: Bool)? {
        guard let move = petMove(mood, thinking: thinking) else {
            (sitUntil, restSince, playing) = (nil, nil, false)
            return nil
        }
        guard move == .idleRest else {
            (startup, sitUntil, restSince, playing) = (false, nil, nil, false)
            return (move, false)
        }
        if startup { return (.startup, false) }
        if let until = sitUntil {
            if now < until { return (.startup, false) }
            sitUntil = nil
        }
        let since = restSince ?? now
        restSince = since
        guard mood == .resting, !still else {
            playing = false
            return (.idleRest, false)
        }
        if !playing && now.timeIntervalSince(since) >= petPlayAfter && now.timeIntervalSince(playedAt) >= petPlayEvery {
            (playing, playedAt) = (true, now)
        }
        return playing ? (.idlePlay, true) : (.idleRest, false)
    }

    mutating func ended(_ move: PetMove, now: Date) {
        if move == .startup && startup { (startup, sitUntil) = (false, now.addingTimeInterval(petSitFor)) }
        if move == .idlePlay { playing = false }
    }
}

/// Points of the side of the moves' square canvas on the pet's panel; nothing drawn is cut by the panel (the art
/// keeps a clear margin round the dog).
let petMoveSide: CGFloat = 100
/// The moves' canvas (pixels) and where the paws rest on it, for a still shown when the manifest cannot be read.
let petStillPixels: CGFloat = 256
let petStillAnchor = CGPoint(x: 128, y: 244)

/// The frame of the moves' canvas on a panel of `size` (points, y up): its anchor (`anchor`, in pixels of a canvas
/// `pixels` wide, from the top left: where the paws rest) on the panel's ground line, `belowPaws` above the bottom,
/// at the middle.
func petMoveFrame(side: CGFloat, anchor: CGPoint, pixels: CGFloat, in size: CGSize, belowPaws: CGFloat) -> CGRect {
    let unit = side / pixels
    return CGRect(x: size.width / 2 - anchor.x * unit, y: belowPaws - (pixels - anchor.y) * unit, width: side, height: side)
}

/// The moves the bundled gateway serves (`static/moves` of the localmelo package in the bundle's Python, under
/// `resources`), so the pet, the start window, the pill and the console show the same files; nil when there are none.
func runtimeMovesFolder(resources: URL) -> URL? {
    let lib = resources.appendingPathComponent("runtime/bin/python/lib")
    guard let names = try? FileManager.default.contentsOfDirectory(atPath: lib.path) else { return nil }
    for name in names.sorted() where name.hasPrefix("python3.") {
        let folder = lib.appendingPathComponent("\(name)/site-packages/localmelo/support/gateway/static/moves")
        if FileManager.default.fileExists(atPath: folder.appendingPathComponent("manifest.json").path) { return folder }
    }
    return nil
}

/// What the system says that decides whether the pet may move.
struct PetEnvironment: Equatable {
    var reduceMotion = false
    var lowPower = false
    var thermal: ProcessInfo.ThermalState = .nominal
    /// The pet's panel can be seen: not hidden by the person, not covered or off the screen.
    var visible = true
    /// The screen is on and this user's session is in front.
    var screenAwake = true
}

/// Whether the activity pill's loops (the portrait's breath and the ring's pulse) may run. Each stop leaves the pill
/// still.
func shouldAnimate(_ environment: PetEnvironment) -> Bool {
    let hot = environment.thermal == .serious || environment.thermal == .critical
    return !environment.reduceMotion && !environment.lowPower && !hot
        && environment.visible && environment.screenAwake
}

/// Whether the moves play (else each shows its poster): only Reduce Motion stills them. Low Power Mode and heat leave
/// them playing: a frame is a swap of the part of a decoded sheet that shows, done by Core Animation, so the pet stays
/// alive in every state that has moves. A pet that cannot be seen (`petMovesRun`) stops instead, where it is, and
/// goes on from there.
func petMovesMayMove(_ environment: PetEnvironment) -> Bool { !environment.reduceMotion }

/// Whether the moves' clock runs: the panel can be seen and the screen is on.
func petMovesRun(_ environment: PetEnvironment) -> Bool { environment.visible && environment.screenAwake }

/// Points between the pet's panel and the right edge of the screen's usable area when it first appears.
let petEdgeInset: CGFloat = 16
/// A drop whose paws are this close to the bottom edge of the usable area puts them on it.
let petSnapDistance: CGFloat = 12

/// The panel's frame (screen coordinates, y up) at first: at the bottom right of `visible`, the usable area of the
/// screen, with the paws on its bottom edge. `belowPaws` is the clear strip under the paws on the panel.
func petDefaultFrame(size: CGSize, belowPaws: CGFloat, in visible: CGRect) -> CGRect {
    let frame = CGRect(x: visible.maxX - petEdgeInset - size.width, y: visible.minY - belowPaws,
                       width: size.width, height: size.height)
    return petClamped(frame, belowPaws: belowPaws, in: visible)
}

/// `frame` moved the least that puts it inside `visible`, except that the strip under the paws may hang below the
/// bottom edge so the paws can rest on it. A panel larger than `visible` keeps its bottom-left corner.
func petClamped(_ frame: CGRect, belowPaws: CGFloat, in visible: CGRect) -> CGRect {
    CGRect(x: max(visible.minX, min(frame.minX, visible.maxX - frame.width)),
           y: max(visible.minY - belowPaws, min(frame.minY, visible.maxY - frame.height)),
           width: frame.width, height: frame.height)
}

/// A dropped panel: within `petSnapDistance` of the bottom edge, the paws go onto it; otherwise as it is.
func petSnapped(_ frame: CGRect, belowPaws: CGFloat, in visible: CGRect) -> CGRect {
    guard abs(frame.minY + belowPaws - visible.minY) <= petSnapDistance else { return frame }
    return CGRect(x: frame.minX, y: visible.minY - belowPaws, width: frame.width, height: frame.height)
}

/// Where to save a panel for its screen: how far its bottom-right corner is from the bottom-right corner of the
/// screen's usable area. A change of Dock size or of the screen's resolution moves the pet with that corner.
struct PetSpot: Equatable, Codable {
    let dx: CGFloat
    let dy: CGFloat
}

func petSpot(of frame: CGRect, on visible: CGRect) -> PetSpot {
    PetSpot(dx: frame.maxX - visible.maxX, dy: frame.minY - visible.minY)
}

/// The panel's frame for a saved spot on the screen's current usable area, clamped onto it.
func petFrame(size: CGSize, spot: PetSpot, belowPaws: CGFloat, on visible: CGRect) -> CGRect {
    petClamped(CGRect(x: visible.maxX + spot.dx - size.width, y: visible.minY + spot.dy,
                      width: size.width, height: size.height), belowPaws: belowPaws, in: visible)
}

/// The panel's size in points, round the moves' canvas (`petMoveSide`, its anchor on the ground line), and the clear
/// strip under the paws.
let petPanelSize = CGSize(width: 132, height: 97)
let petBelowPaws: CGFloat = 3
/// A press that moved no further than this many points is a click, not the start of a drag.
let petClickSlop: CGFloat = 3

func petIsClick(moved distance: CGFloat) -> Bool { distance <= petClickSlop }

/// Whether the app's main window is in front of the person: ordered in, not minimized, the app not hidden.
func petWindowShown(visible: Bool, minimized: Bool, appHidden: Bool) -> Bool { visible && !minimized && !appHidden }

/// The pet lies on the desktop only while the window is put away and the person has not hidden it.
func petShouldShow(windowShown: Bool, hiddenByPerson: Bool) -> Bool { !windowShown && !hiddenByPerson }

/// Where a click on the pet goes: to the chat or task that waits for an approval, else to the window.
enum PetClick: Equatable {
    case waiting, window
}

func petClickTarget(_ mood: PetMood) -> PetClick { mood == .waiting ? .waiting : .window }

/// What is on the desktop while the window is put away: Melo and the activity pill (the default), Melo alone, or
/// nothing. Chosen in the menu bar's menu and the pet's menu.
enum DesktopMode: String, CaseIterable {
    case pill, pet, nothing

    var title: String {
        switch self {
        case .pill: return L10n.t("app.desktop.pill", "Melo and the activity pill")
        case .pet: return L10n.t("app.desktop.pet", "Melo only")
        case .nothing: return L10n.t("app.desktop.nothing", "Nothing")
        }
    }
}

/// The menu that offers the choices above.
var desktopMenuTitle: String { L10n.t("app.desktop.menu", "On the desktop") }

/// What the console's Settings may ask of the app about the desktop: the choice now, or a new one.
enum DesktopRequest: Equatable {
    case get
    case set(DesktopMode)
}

/// The request a page message holds, or nil: an unknown action, or a mode the menu does not offer.
func desktopRequest(_ body: Any) -> DesktopRequest? {
    guard let message = body as? [String: Any], let action = message["action"] as? String else { return nil }
    switch action {
    case "get":
        return .get
    case "set":
        guard let raw = message["mode"] as? String, let mode = DesktopMode(rawValue: raw) else { return nil }
        return .set(mode)
    default:
        return nil
    }
}

/// What the app remembers about the pet between launches: whether the person hid it, whether the activity pill may
/// take its place while work runs, the screen it was last on, and its spot on each screen. A missing, damaged or
/// partly damaged file gives what is left of it.
struct PetSettings: Equatable, Codable {
    var hidden = false
    var pill = true
    var screen: String?
    var spots: [String: PetSpot] = [:]

    init() {}

    /// The menu's choice: hidden is "Nothing"; else the pill's flag says Melo with or without the pill. Choosing
    /// Nothing keeps the pill's flag, so Show Melo brings back what was there.
    var mode: DesktopMode {
        get { hidden ? .nothing : pill ? .pill : .pet }
        set {
            hidden = newValue == .nothing
            if newValue != .nothing { pill = newValue == .pill }
        }
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: Keys.self)
        hidden = (try? values.decode(Bool.self, forKey: .hidden)) ?? false
        pill = (try? values.decode(Bool.self, forKey: .pill)) ?? true
        screen = try? values.decode(String.self, forKey: .screen)
        let raw = (try? values.decode([String: PetSpot].self, forKey: .spots)) ?? [:]
        spots = raw.filter { $0.value.dx.isFinite && $0.value.dy.isFinite && $0.key.count <= 128 }
    }

    private enum Keys: String, CodingKey { case hidden, pill, screen, spots }

    /// `key` is where the pet was dropped, at `spot`. Other screens keep theirs, up to 16 of them.
    mutating func remember(screen key: String, spot: PetSpot) {
        screen = key
        spots[key] = spot
        while spots.count > 16, let extra = spots.keys.first(where: { $0 != key }) { spots[extra] = nil }
    }
}

func petSettings(_ data: Data?) -> PetSettings {
    data.flatMap { try? JSONDecoder().decode(PetSettings.self, from: $0) } ?? PetSettings()
}

func petSettingsData(_ settings: PetSettings) -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
    return (try? encoder.encode(settings)) ?? Data("{}".utf8)
}

/// One screen the pet may lie on: a key that stays the same across launches, and its usable area.
struct PetScreen: Equatable {
    let key: String
    let visible: CGRect
}

/// Where the panel goes: on the screen it was last on if that is still there, else the first screen (the one with
/// the menu bar); at that screen's saved spot, else at its bottom right. Nil when there is no screen.
func petPlacement(_ settings: PetSettings, screens: [PetScreen]) -> (screen: PetScreen, frame: CGRect)? {
    guard let screen = screens.first(where: { $0.key == settings.screen }) ?? screens.first else { return nil }
    if let spot = settings.spots[screen.key] {
        return (screen, petFrame(size: petPanelSize, spot: spot, belowPaws: petBelowPaws, on: screen.visible))
    }
    return (screen, petDefaultFrame(size: petPanelSize, belowPaws: petBelowPaws, in: screen.visible))
}
