import CoreGraphics
import Foundation

/// Melo the Akita puppy's 14 pictures, named as their files under `Resources/akita` (`scripts/build_pet_assets.py`).
/// The five sitting ones are the portrait of the starting view and the empty chat; the lying and sleeping nine are
/// the desktop pet's.
enum PetState: String, CaseIterable {
    case sitIdle = "sit-idle", sitWorking = "sit-working", sitWaiting = "sit-waiting"
    case sitDone = "sit-done", sitError = "sit-error"
    case lieIdle = "lie-idle", lieWorking = "lie-working", lieWaiting = "lie-waiting"
    case lieDone = "lie-done", lieError = "lie-error"
    case restLazy = "rest-lazy", sleepFront = "sleep-front", sleepSide = "sleep-side", sleepBack = "sleep-back"

    /// The three sleeping poses, in the order the pet turns through them.
    static let sleepPoses: [PetState] = [.sleepFront, .sleepSide, .sleepBack]

    var isAsleep: Bool { Self.sleepPoses.contains(self) }

    /// The folder of the picture under `Resources/akita`.
    var folder: String { rawValue.hasPrefix("sit-") ? "portrait" : "pet" }

    /// The file for a screen scale: `lie-idle.png`, and `lie-idle@2x.png` from 2.
    func fileName(scale: Int) -> String { pieceFile(rawValue, scale: scale) }

    /// A file of the picture's family (`lie-idle-blink`) for a screen scale.
    func pieceFile(_ name: String, scale: Int) -> String { scale >= 2 ? "\(name)@2x.png" : "\(name).png" }

    /// The eyes-closed patch laid over the picture for a blink, nil where the picture has none. A patch fits only the
    /// picture it was cut from: the sleeping dogs have their eyes shut already, and the working, waiting, done and
    /// error faces have no patch.
    var blinkPatch: String? {
        switch self {
        case .sitIdle: return "sit-idle-blink"
        case .lieIdle: return "lie-idle-blink"
        case .restLazy: return "rest-lazy-blink"
        default: return nil
        }
    }

    /// The picture without its tail, and the tail alone, for the wag; nil where the picture has no tail to wag.
    var tailPieces: (body: String, tail: String)? { self == .lieDone ? ("lie-done-body", "lie-done-tail") : nil }

    /// The picture without its right ear, and the ear alone, for the twitch; nil where the picture has no ear piece. Only
    /// the idle dog has them: the other lying pictures' heads do not lie on its no-ear head (the alignment check of
    /// `scripts/build_pet_assets.py` fails for them), and the lazy dog's head is another drawing.
    var earPieces: (body: String, ear: String)? { self == .lieIdle ? ("lie-idle-body", "lie-idle-ear") : nil }

    /// Where a sleeping dog's "z" starts, in points from the top left of the pet's canvas (`z` in anchors.json).
    var sleepZ: CGPoint? {
        switch self {
        case .sleepFront: return CGPoint(x: 86, y: 24)
        case .sleepSide: return CGPoint(x: 62, y: 22)
        case .sleepBack: return CGPoint(x: 30, y: 40)
        default: return nil
        }
    }
}

/// Where the lying dog's tail joins its back, in points from the top left of the pet's canvas: the tail wags about it
/// (`tail.pivot` of `lie-done` in anchors.json).
let petTailPivot = CGPoint(x: 101.8, y: 53)

/// Where the idle dog's right ear joins its head, in the same points: the ear twitches about it (`ear.pivot` of
/// `lie-idle` in anchors.json).
let petEarPivot = CGPoint(x: 64.98, y: 19.82)

/// The sitting picture of the app's first window: the working puppy while the server starts and signs in, the calm
/// one once the gateway is ready and the console is on its way, the one with the drooping ear when it cannot go on.
func startPortrait(failed: Bool, ready: Bool) -> PetState {
    failed ? .sitError : ready ? .sitIdle : .sitWorking
}

/// The points the portrait is shown at: the canvas of the sitting pictures (`anchors.json`), so a 2x file is
/// sharp on a Retina screen and shown whole on any other.
let portraitSize = CGSize(width: 108, height: 140)

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
    return RecentOutcome(kind: kind, id: id, outcome: outcome, endedAgo: ago)
}

/// Seconds a finished run shows as done, and a failed one as an error, before the pet goes back to rest.
let petDoneFor: TimeInterval = 6
let petFailedFor: TimeInterval = 10
/// Seconds of quiet before the pet falls asleep, and between one sleeping pose and the next.
let petSleepAfter: TimeInterval = 60
let petRotateEvery: TimeInterval = 180

/// What the pet is doing. `picture` is what it looks like; the words differ where two moods share a picture.
enum PetMood: Equatable {
    case unknown, resting, sleeping(PetState), working(Int), waiting, done, failed, stopped

    var picture: PetState {
        switch self {
        case .unknown: return .lieIdle
        case .resting: return .restLazy
        case .sleeping(let pose): return pose
        case .working: return .lieWorking
        case .waiting: return .lieWaiting
        case .done: return .lieDone
        case .failed, .stopped: return .lieError
        }
    }

    /// What VoiceOver reads, and what hovering shows.
    var label: String {
        switch self {
        case .unknown: return "Melo"
        case .resting: return "Melo, resting. Nothing running."
        case .sleeping: return "Melo, sleeping. Nothing running."
        case .working(let n): return "Melo, working. \(n) running."
        case .waiting: return "Melo, waiting for your approval."
        case .done: return "Melo, done."
        case .failed: return "Melo, the last run did not finish."
        case .stopped: return "Melo, localmelo stopped."
        }
    }

    /// What pressing the pet does.
    var help: String { self == .waiting ? "Opens what waits for your approval" : "Opens localmelo" }
}

/// The pet's mood from the last health reply (nil: none yet, or the gateway did not answer). First match wins:
/// the app's own failure, then a wait, then work, then a run that just finished, then rest. `sinceTouch` is the
/// seconds since the person last touched the pet (hover, click, drag) or since it appeared; the quiet that puts
/// it to sleep is the shorter of that and the age of the newest finished run, so a touch wakes it into rest and
/// nothing changes its pose while it is touched. `poseSeed` picks the first sleeping pose of a quiet spell.
func petMood(health: Data?, stopped: Bool, sinceTouch: TimeInterval, poseSeed: Int) -> PetMood {
    if stopped { return .stopped }
    guard let data = health, let counts = runningCounts(data) else { return .unknown }
    if !waitingItems(data).isEmpty { return .waiting }
    if counts.total > 0 { return .working(counts.total) }
    let recent = recentOutcome(data)
    if let recent = recent {
        if recent.outcome == .done && recent.endedAgo < petDoneFor { return .done }
        if recent.outcome == .failed && recent.endedAgo < petFailedFor { return .failed }
    }
    let quiet = min(sinceTouch, recent?.endedAgo ?? .infinity)
    guard !quiet.isNaN, quiet >= petSleepAfter else { return .resting }
    let turns = Int(min((quiet - petSleepAfter) / petRotateEvery, 1_000_000))
    let poses = PetState.sleepPoses
    return .sleeping(poses[((poseSeed % poses.count + turns) % poses.count + poses.count) % poses.count])
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

/// Whether the pet's loops (breathing, the sleeping drift) may run. Each stop leaves the state's still picture.
func shouldAnimate(_ environment: PetEnvironment) -> Bool {
    let hot = environment.thermal == .serious || environment.thermal == .critical
    return !environment.reduceMotion && !environment.lowPower && !hot
        && environment.visible && environment.screenAwake
}

/// Seconds of the plain cross-fade between two pictures when the pet may not move (Reduce Motion, Low Power Mode, heat):
/// a change is a dissolve, never a motion.
let petCrossfade: TimeInterval = 0.2

/// Seconds the change from one picture to the next takes: none for the first picture; a slow 0.9 s between two
/// sleeping poses while the pet may move, so the turn reads as rolling over; 0.25 s from one pose to another while it
/// may move (the squash of `petDips` covers it); else the plain cross-fade.
func petFade(from old: PetState?, to new: PetState, environment: PetEnvironment) -> TimeInterval {
    guard let old = old, old != new else { return 0 }
    guard shouldAnimate(environment) else { return petCrossfade }
    return old.isAsleep && new.isAsleep ? 0.9 : 0.25
}

/// Whether a change of picture hides in a short squash and rise of the whole pet (the cross-fade centred on its low
/// point): only while the pet may move, and not for the slow turn between sleeping poses.
func petDips(from old: PetState?, to new: PetState, environment: PetEnvironment) -> Bool {
    guard let old = old, old != new, shouldAnimate(environment) else { return false }
    return !(old.isAsleep && new.isAsleep)
}

/// Points between the pet's panel and the right edge of the screen's usable area when it first appears.
let petEdgeInset: CGFloat = 16
/// A drop whose paws are this close to the bottom edge of the usable area puts them on it.
let petSnapDistance: CGFloat = 12

/// The panel's frame (screen coordinates, y up) at first: at the bottom right of `visible`, the usable area of the
/// screen, with the paws on its bottom edge. `belowPaws` is the clear strip under the paws on the panel's canvas
/// (its height minus `ground` in anchors.json).
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

/// The panel's size in points: the canvas every lying and sleeping picture shares (`canvas` in anchors.json), and
/// the clear strip under the paws (its height minus `ground`).
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

/// What the app remembers about the pet between launches: whether the person hid it, the screen it was last on,
/// and its spot on each screen. A missing, damaged or partly damaged file gives what is left of it.
struct PetSettings: Equatable, Codable {
    var hidden = false
    var screen: String?
    var spots: [String: PetSpot] = [:]

    init() {}

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: Keys.self)
        hidden = (try? values.decode(Bool.self, forKey: .hidden)) ?? false
        screen = try? values.decode(String.self, forKey: .screen)
        let raw = (try? values.decode([String: PetSpot].self, forKey: .spots)) ?? [:]
        spots = raw.filter { $0.value.dx.isFinite && $0.value.dy.isFinite && $0.key.count <= 128 }
    }

    private enum Keys: String, CodingKey { case hidden, screen, spots }

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
