import AppKit
import ImageIO
import QuartzCore

/// Melo's moves on the desktop: the animated states the bundled gateway serves (`runtimeMovesFolder`), the console's
/// own files (`scripts/build_moves.py`). Each state is a sheet of its frames (`columns` x `rows` cells of one square canvas, left
/// to right and top to bottom) with a duration per frame; `manifest.json` says which plays once, the frame that stands
/// for it when nothing may move (its poster) and its pose.
///
/// The rules are the console's (`moves.mjs`):
/// - A change cross-fades from what is on the screen, a frame or the mix of a fade still going: 160 ms between two
///   states of one pose, 220 ms between two poses, 120 ms into `attention`, which interrupts anything at once.
/// - The latest request wins: a state that played once and ends late never brings itself back.
/// - The same state asked again does not restart. `startup` plays once and holds its last frame.
/// - When nothing may move (`still`) each state is its poster, and a change is an 80 ms fade.
/// - A state asked for as a still (`poster`) shows its poster and holds it, with the fade of any other change.
/// - A state whose sheet cannot be read shows its poster file (`poster-<state>.webp`) instead, held; with neither the
///   pet keeps what it shows.
/// - A still (`hold`, `PetStill`) is its own picture beside the sheets, held at its opacity, with the fade of a change
///   of pose; the state asked for next fades from it.
/// - A pet that cannot be seen stops where it is (`run(false)`) and goes on from there.
///
/// The frames are a key-frame animation of the layer's `contentsRect` over the state's sheet, run by Core Animation;
/// the app does no work per frame.

/// One state's timeline: its sheet, its pose, whether it plays once, its poster frame and each frame's seconds.
struct PetMoveTimeline: Equatable {
    let file: String
    let pose: String
    let once: Bool
    let poster: Int
    let seconds: [Double]

    var length: Double { seconds.reduce(0, +) }
}

/// `manifest.json` of the moves: the canvas (pixels) and its anchor (pixels from the top left: where the paws rest),
/// the sheet's grid, the fades (seconds) and the six timelines.
struct PetMovesManifest: Equatable {
    let pixels: CGFloat
    let anchor: CGPoint
    let columns: Int
    let rows: Int
    let samePose: Double
    let poseChange: Double
    let attention: Double
    let still: Double
    let states: [PetMove: PetMoveTimeline]

    /// The cross-fade from `old` to `new`: none for the first state.
    func fade(from old: PetMove?, to new: PetMove, still isStill: Bool) -> Double {
        guard let old = old else { return 0 }
        if isStill { return still }
        if new == .attention { return attention }
        return states[old]?.pose == states[new]?.pose ? samePose : poseChange
    }

    /// The part of a sheet frame `index` takes, in the unit square of the layer's contents (y up, as Core Animation
    /// reads `contentsRect` on the Mac: the first row of the sheet is its top).
    func cell(_ index: Int) -> CGRect {
        let width = 1 / CGFloat(columns), height = 1 / CGFloat(rows)
        let column = index % columns, row = index / columns
        return CGRect(x: CGFloat(column) * width, y: 1 - CGFloat(row + 1) * height, width: width, height: height)
    }
}

/// The manifest, or nil when it is not one: all six states, each with one positive duration per frame that fits
/// the sheet, and a poster among its frames.
func petMovesManifest(_ data: Data) -> PetMovesManifest? {
    guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let canvas = body["canvas"] as? [String: Any], let size = canvas["size"] as? Double, size > 0,
          let anchor = canvas["anchor"] as? [Double], anchor.count == 2,
          let sheet = body["sheet"] as? [String: Any], let columns = sheet["columns"] as? Int,
          let rows = sheet["rows"] as? Int, columns > 0, rows > 0,
          let fade = body["fade_ms"] as? [String: Double],
          let same = fade["same_pose"], let change = fade["pose_change"], let urgent = fade["attention"],
          let reduced = fade["reduced_motion"],
          let entries = body["states"] as? [String: [String: Any]] else { return nil }
    var states: [PetMove: PetMoveTimeline] = [:]
    for move in PetMove.allCases {
        guard let entry = entries[move.rawValue], let file = entry["file"] as? String,
              !file.contains("/"), file.hasSuffix(".webp"),
              let pose = entry["pose"] as? String, let playback = entry["playback"] as? String,
              let poster = entry["poster"] as? Int, let durations = entry["durations_ms"] as? [Double],
              (1...columns * rows).contains(durations.count), durations.allSatisfy({ $0 > 0 && $0.isFinite }),
              (0..<durations.count).contains(poster) else { return nil }
        states[move] = PetMoveTimeline(file: file, pose: pose, once: playback == "once", poster: poster,
                                       seconds: durations.map { $0 / 1000 })
    }
    return PetMovesManifest(pixels: CGFloat(size), anchor: CGPoint(x: anchor[0], y: anchor[1]), columns: columns,
                            rows: rows, samePose: same / 1000, poseChange: change / 1000, attention: urgent / 1000,
                            still: reduced / 1000, states: states)
}

/// The key-frame animation of a timeline's frames on `contentsRect`: each frame from the start of its duration,
/// once or for ever; one that plays once stays on its last frame.
func petFramesAnimation(_ timeline: PetMoveTimeline, manifest: PetMovesManifest, once: Bool) -> CAKeyframeAnimation {
    let animation = CAKeyframeAnimation(keyPath: "contentsRect")
    animation.values = timeline.seconds.indices.map { NSValue(rect: manifest.cell($0)) }
    var elapsed = 0.0
    var times: [NSNumber] = []
    for seconds in timeline.seconds {
        times.append(NSNumber(value: elapsed / timeline.length))
        elapsed += seconds
    }
    animation.keyTimes = times + [1]  // discrete: one more time than values, the last one 1
    animation.calculationMode = .discrete
    animation.duration = timeline.length
    animation.repeatCount = once ? 1 : .infinity
    animation.fillMode = .forwards
    animation.isRemovedOnCompletion = false
    return animation
}

/// A WebP (or any image) decoded in full now, so drawing it later costs nothing; nil when it cannot be read.
func petDecoded(_ url: URL) -> CGImage? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
}

/// Plays the moves on one layer (`layer`, the canvas, placed by its owner). Only this player draws on it.
final class PetMovePlayer: NSObject, CAAnimationDelegate {
    let layer = CALayer()
    let manifest: PetMovesManifest
    /// Hears a state that played once reach its end (only the state still asked for).
    var onEnded: (PetMove) -> Void = { _ in }
    private(set) var wanted: PetMove?
    private(set) var once = false
    /// The state asked for shows its poster, held.
    private(set) var poster = false
    /// The still on show (`hold`), in place of any state.
    private(set) var holding: PetStill?
    private(set) var still = false
    private let incoming = CALayer()
    private let outgoing = CALayer()
    private let folder: URL
    private var sheets: [PetMove: CGImage] = [:]
    private var posters: [PetMove: CGImage] = [:]
    private var stillImages: [PetStill: CGImage] = [:]
    private var generation = 0

    /// `folder` holds manifest.json, the sheets and the posters; nil when the manifest cannot be read or `attention`'s
    /// sheet cannot be decoded. `attention` is decoded at once, so the state that interrupts never waits on the disk;
    /// the other five are decoded off the main thread right after, so asking for one later costs no hitch.
    init?(folder: URL, side: CGFloat) {
        guard let data = try? Data(contentsOf: folder.appendingPathComponent("manifest.json")),
              let manifest = petMovesManifest(data) else { return nil }
        self.folder = folder
        self.manifest = manifest
        super.init()
        layer.bounds = CGRect(x: 0, y: 0, width: side, height: side)
        layer.actions = ["position": NSNull(), "bounds": NSNull(), "hidden": NSNull(), "opacity": NSNull()]
        for piece in [outgoing, incoming] {
            piece.frame = layer.bounds
            piece.contentsGravity = .resize
            piece.minificationFilter = .trilinear
            piece.actions = ["contents": NSNull(), "contentsRect": NSNull(), "opacity": NSNull()]
            layer.addSublayer(piece)
        }
        outgoing.opacity = 0
        guard sheet(.attention) != nil else { return nil }
        let files = PetMove.allCases.filter { $0 != .attention }.compactMap { move in
            manifest.states[move].map { (move, folder.appendingPathComponent($0.file)) }
        }
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let decoded = files.compactMap { move, url in petDecoded(url).map { (move, $0) } }
            DispatchQueue.main.async {
                guard let self = self else { return }
                for (move, image) in decoded where self.sheets[move] == nil { self.sheets[move] = image }
            }
        }
    }

    /// Whether the sheet of `move` is decoded and ready (the background decode, or a first request, did it).
    func isReady(_ move: PetMove) -> Bool { sheets[move] != nil }

    /// The sheet of `move`, decoded once: by the background decode, or here if it is asked for first.
    private func sheet(_ move: PetMove) -> CGImage? {
        if let image = sheets[move] { return image }
        guard let timeline = manifest.states[move], let image = petDecoded(folder.appendingPathComponent(timeline.file))
        else { return nil }
        sheets[move] = image
        return image
    }

    /// The picture of a still, decoded once.
    private func stillImage(_ still: PetStill) -> CGImage? {
        if let image = stillImages[still] { return image }
        guard let image = petDecoded(folder.appendingPathComponent(still.file)) else { return nil }
        stillImages[still] = image
        return image
    }

    /// Shows `still` and holds it, at its opacity, fading from what is on the screen. The same still again does
    /// nothing. False when its file cannot be read (the pet keeps what it shows).
    @discardableResult
    func hold(_ still: PetStill) -> Bool {
        guard holding != still else { return true }
        guard let image = stillImage(still) else { return false }
        let frozen = wanted == nil && holding == nil ? nil : capture()
        generation += 1
        (wanted, holding, poster, once) = (nil, still, false, false)
        for piece in [outgoing, incoming] { piece.removeAllAnimations() }
        let whole = CGRect(x: 0, y: 0, width: 1, height: 1)
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        outgoing.contents = frozen
        outgoing.contentsRect = whole
        outgoing.opacity = 0
        incoming.contents = image
        incoming.contentsRect = whole
        incoming.opacity = still.opacity
        CATransaction.commit()
        if frozen != nil { blend(self.still ? manifest.still : manifest.poseChange, to: still.opacity) }
        return true
    }

    /// The cross-fade of `seconds` from the frozen picture to the incoming one, which ends at `opacity`.
    private func blend(_ seconds: Double, to opacity: Float) {
        guard seconds > 0 else { return }
        for (piece, from, to) in [(outgoing, Float(1), Float(0)), (incoming, Float(0), opacity)] {
            let fade = CABasicAnimation(keyPath: "opacity")
            (fade.fromValue, fade.toValue, fade.duration) = (from, to, seconds)
            fade.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            piece.add(fade, forKey: "fade")
        }
    }

    /// The poster of `move` alone (`poster-<state>.webp`), for a state whose sheet cannot be read.
    private func posterFile(_ move: PetMove) -> CGImage? {
        if let image = posters[move] { return image }
        guard let image = petDecoded(folder.appendingPathComponent("poster-\(move.rawValue).webp")) else { return nil }
        posters[move] = image
        return image
    }

    /// Shows `move`. The state asked for last does nothing (unless `restart`); `once` plays a looping state a single
    /// time and holds its last frame; `poster` shows its poster and holds it. False when its sheet cannot be read (the
    /// pet keeps what it shows).
    @discardableResult
    func set(_ move: PetMove, once playOnce: Bool = false, poster held: Bool = false, restart: Bool = false) -> Bool {
        guard restart || move != wanted || held != poster else { return true }
        guard let timeline = manifest.states[move] else { return false }
        // A sheet that cannot be read gives way to the state's poster file, shown whole and held.
        let sheetImage = sheet(move)
        guard let image = sheetImage ?? posterFile(move) else { return false }
        let whole = sheetImage == nil
        let old = wanted, wasHolding = holding != nil
        // What the person sees now, frozen, before any animation is taken off: the next state fades from it.
        let frozen = old == nil && !wasHolding ? nil : capture()
        generation += 1
        wanted = move
        holding = nil
        poster = held
        once = playOnce || timeline.once
        for piece in [outgoing, incoming] { piece.removeAllAnimations() }
        // From a still, the fade of a change of pose (or the still one).
        let fade = wasHolding ? (still ? manifest.still : manifest.poseChange) : manifest.fade(from: old, to: move, still: still)
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        outgoing.contents = frozen
        outgoing.contentsRect = CGRect(x: 0, y: 0, width: 1, height: 1)
        outgoing.opacity = 0
        incoming.contents = image
        incoming.opacity = 1
        // The model value is where the state rests: its poster when still or held, else the frame it holds at its end
        // (one that plays once) or starts from; the whole picture for a poster file.
        incoming.contentsRect = whole ? CGRect(x: 0, y: 0, width: 1, height: 1)
            : manifest.cell(still || held ? timeline.poster : once ? timeline.seconds.count - 1 : 0)
        CATransaction.commit()
        if frozen != nil { blend(fade, to: 1) }
        let token = generation
        if held { return true }
        if still || whole {
            // A still state that plays once has ended as soon as it shows: its poster is where it holds.
            if once { DispatchQueue.main.async { [weak self] in self?.ended(move, token) } }
            return true
        }
        let frames = petFramesAnimation(timeline, manifest: manifest, once: once)
        if once {
            frames.delegate = self
            frames.setValue(token, forKey: "generation")
            frames.setValue(move.rawValue, forKey: "move")
        }
        incoming.add(frames, forKey: "frames")
        return true
    }

    func animationDidStop(_ animation: CAAnimation, finished: Bool) {
        guard finished, let token = animation.value(forKey: "generation") as? Int,
              let move = (animation.value(forKey: "move") as? String).flatMap(PetMove.init(rawValue:)) else { return }
        ended(move, token)
    }

    private func ended(_ move: PetMove, _ token: Int) {
        guard token == generation, wanted == move else { return }
        onEnded(move)
    }

    /// Posters only (`true`) or moving: the state shown starts again, after the short fade (a held poster stays).
    func setStill(_ value: Bool) {
        guard value != still else { return }
        still = value
        if let move = wanted, !poster {
            set(move, once: once && manifest.states[move]?.once == false, restart: true)
        }
    }

    /// Stops the clock where it is (`false`: the pet cannot be seen) or lets it go on from there.
    func run(_ running: Bool) {
        if !running {
            guard layer.speed != 0 else { return }
            let now = layer.convertTime(CACurrentMediaTime(), from: nil)
            layer.speed = 0
            layer.timeOffset = now
        } else {
            guard layer.speed == 0 else { return }
            let paused = layer.timeOffset
            layer.speed = 1
            layer.timeOffset = 0
            layer.beginTime = 0
            layer.beginTime = layer.convertTime(CACurrentMediaTime(), from: nil) - paused
        }
    }

    var running: Bool { layer.speed != 0 }

    /// What the person sees, as one picture: the layers as they are drawn now (a frame, or the mix of a fade), at
    /// twice the canvas's points.
    private func capture() -> CGImage? {
        let scale: CGFloat = 2
        let width = Int(layer.bounds.width * scale), height = Int(layer.bounds.height * scale)
        guard width > 0, height > 0,
              let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        context.scaleBy(x: scale, y: scale)
        (layer.presentation() ?? layer).render(in: context)
        return context.makeImage()
    }
}
