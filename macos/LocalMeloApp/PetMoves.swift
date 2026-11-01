import AppKit
import ImageIO
import QuartzCore

/// Melo's moves on the desktop: the animated states of `Resources/moves`, the console's own files
/// (`scripts/build_moves.py`). Each state is a sheet of its frames (`columns` x `rows` cells of one square canvas, left
/// to right and top to bottom) with a duration per frame; `manifest.json` says which plays once, the frame that stands
/// for it when nothing may move (its poster) and its pose.
///
/// The rules are the console's (`moves.mjs`):
/// - A change cross-fades from what is on the screen, a frame or the mix of a fade still going: 160 ms between two
///   states of one pose, 220 ms between two poses, 120 ms into `attention`, which interrupts anything at once.
/// - The latest request wins: a state that played once and ends late never brings itself back.
/// - The same state asked again does not restart. `startup` plays once and holds its last frame.
/// - When nothing may move (`still`) each state is its poster, and a change is an 80 ms fade.
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

/// Plays the moves on one layer (`layer`, the canvas, placed by its owner). Only this player draws on it.
final class PetMovePlayer: NSObject, CAAnimationDelegate {
    let layer = CALayer()
    let manifest: PetMovesManifest
    /// Hears a state that played once reach its end (only the state still asked for).
    var onEnded: (PetMove) -> Void = { _ in }
    private(set) var wanted: PetMove?
    private(set) var once = false
    private(set) var still = false
    private let incoming = CALayer()
    private let outgoing = CALayer()
    private let folder: URL
    private var sheets: [PetMove: CGImage] = [:]
    private var generation = 0

    /// `folder` holds manifest.json and the sheets; nil when the manifest cannot be read or `attention`'s sheet is
    /// missing. `attention` is decoded at once, so the state that interrupts never waits on the disk.
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
    }

    /// The sheet of `move`, decoded once.
    private func sheet(_ move: PetMove) -> CGImage? {
        if let image = sheets[move] { return image }
        guard let timeline = manifest.states[move],
              let source = CGImageSourceCreateWithURL(folder.appendingPathComponent(timeline.file) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
        else { return nil }
        sheets[move] = image
        return image
    }

    /// Shows `move`. The state asked for last does nothing (unless `restart`); `once` plays a looping state a single
    /// time and holds its last frame. False when its sheet cannot be read (the pet keeps what it shows).
    @discardableResult
    func set(_ move: PetMove, once playOnce: Bool = false, restart: Bool = false) -> Bool {
        guard restart || move != wanted else { return true }
        guard let timeline = manifest.states[move], let image = sheet(move) else { return false }
        let old = wanted
        // What the person sees now, frozen, before any animation is taken off: the next state fades from it.
        let frozen = old == nil ? nil : capture()
        generation += 1
        wanted = move
        once = playOnce || timeline.once
        for piece in [outgoing, incoming] { piece.removeAllAnimations() }
        let fade = manifest.fade(from: old, to: move, still: still)
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        outgoing.contents = frozen
        outgoing.contentsRect = CGRect(x: 0, y: 0, width: 1, height: 1)
        outgoing.opacity = 0
        incoming.contents = image
        incoming.opacity = 1
        // The model value is where the state rests: its poster when still, else the frame it holds at its end (one
        // that plays once) or starts from.
        incoming.contentsRect = manifest.cell(still ? timeline.poster : once ? timeline.seconds.count - 1 : 0)
        CATransaction.commit()
        if fade > 0, frozen != nil {
            for (piece, from, to) in [(outgoing, 1.0, 0.0), (incoming, 0.0, 1.0)] {
                let blend = CABasicAnimation(keyPath: "opacity")
                (blend.fromValue, blend.toValue, blend.duration) = (from, to, fade)
                blend.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                piece.add(blend, forKey: "fade")
            }
        }
        let token = generation
        if still {
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

    /// Posters only (`true`) or moving: the state shown starts again, after the short fade.
    func setStill(_ value: Bool) {
        guard value != still else { return }
        still = value
        if let move = wanted { set(move, once: once && manifest.states[move]?.once == false, restart: true) }
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
