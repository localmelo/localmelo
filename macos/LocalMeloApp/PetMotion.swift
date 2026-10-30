import AppKit
import QuartzCore

/// How the desktop pet moves. Every motion is a Core Animation animation added once to a layer: the render server
/// runs it, so the app does no per-frame work (no timer, no `draw`). The pictures are bitmaps, so the motions are
/// transforms of the whole picture, cross-fades between pictures, an eyes-closed patch that fades in over the eyes
/// (the blink), the tail and the ear as layers of their own that swing about their base, and the "z" the app draws
/// itself. All of
/// them stop (the state's still picture stays) when `shouldAnimate` says so; a change of picture still cross-fades
/// unless Reduce Motion asks for a plain swap (`petFade`).

/// The slow scale of a lying body: `amount` is the scale on the vertical axis at the top of a breath, about the
/// ground line, and `seconds` the whole breath, in and out.
struct PetBreath: Equatable {
    let amount: CGFloat
    let seconds: TimeInterval
}

/// What a mood does once, as it begins.
enum PetBeat: Equatable {
    /// A work started: the picture lifts 3 pt and settles.
    case bob
    /// Something waits for the person: the same, 6 pt.
    case hop
    /// A run finished well: the tail swings from side to side about its base and settles.
    case wag
}

struct PetMotionPlan: Equatable {
    var breath: PetBreath?
    var beat: PetBeat?
    /// The eyes blink now and then (the mood's picture has an eyes-closed patch).
    var blinks = false
    /// A "z" drifts up from the sleeping dog.
    var drifts = false
    /// The ear twitches now and then (the mood's picture has an ear piece).
    var twitches = false
}

/// The pieces of a picture's family that move over it; each is nil where the picture has none.
struct PetParts {
    var blink: CGImage?
    var body: CGImage?
    var tail: CGImage?
    /// Where the "z" starts, in points from the top left of the canvas.
    var zStart: CGPoint?
    /// The picture without its ear, and the ear alone.
    var earBody: CGImage?
    var ear: CGImage?
}

/// What a mood does: it breathes unless nothing is known yet or the app stopped; the sleeping breath is slower and
/// a little deeper, and a sleeping dog has its "z". `beat` is for the moment the mood begins and is not repeated. A
/// picture with an eyes-closed patch blinks, and one with an ear piece twitches it, even before anything is known.
func petMotionPlan(_ mood: PetMood) -> PetMotionPlan {
    var plan: PetMotionPlan
    switch mood {
    case .unknown, .stopped: plan = PetMotionPlan(breath: nil, beat: nil)
    case .sleeping: plan = PetMotionPlan(breath: PetBreath(amount: 1.03, seconds: 4.4), beat: nil)
    case .resting, .failed: plan = PetMotionPlan(breath: PetBreath(amount: 1.02, seconds: 3.2), beat: nil)
    case .working: plan = PetMotionPlan(breath: PetBreath(amount: 1.02, seconds: 3.2), beat: .bob)
    case .waiting: plan = PetMotionPlan(breath: PetBreath(amount: 1.02, seconds: 3.2), beat: .hop)
    case .done: plan = PetMotionPlan(breath: PetBreath(amount: 1.02, seconds: 3.2), beat: .wag)
    }
    plan.blinks = mood.picture.blinkPatch != nil
    plan.drifts = mood.picture.sleepZ != nil
    plan.twitches = mood.picture.earPieces != nil
    return plan
}

/// The pet's layers. `moving` carries the beats (lift) and the "z"; `picture` the breathing and the bitmap, with the
/// blink patch and the tail on it, so they breathe with it. `moving` and `picture` turn about the point on the
/// ground line under the centre of the picture, so the paws stay where they are; the tail turns about its base.
final class PetMotion {
    let moving = CALayer()
    let picture = CALayer()
    let blink = CALayer()
    let tail = CALayer()
    let ear = CALayer()
    let drift = CATextLayer()
    private var still: CGImage?
    private var body: CGImage?
    private var earBody: CGImage?
    private var zStart: CGPoint?
    private let size: CGSize

    /// `size` is the panel's canvas and `anchor` the point the motions turn about, in points from the canvas's
    /// bottom-left (the centre of the ground line).
    init(size: CGSize, anchor: CGPoint) {
        self.size = size
        let bounds = CGRect(origin: .zero, size: size)
        let fraction = CGPoint(x: anchor.x / size.width, y: anchor.y / size.height)
        for layer in [moving, picture] {
            layer.anchorPoint = fraction
            layer.bounds = bounds
            layer.position = anchor
            layer.contentsGravity = .resize
            layer.actions = ["contents": NSNull(), "transform": NSNull(), "position": NSNull()]
        }
        moving.addSublayer(picture)
        picture.position = anchor
        // The patch, the tail and the ear are canvas-sized layers over the picture, so each lies where its picture
        // does. The tail's and the ear's anchor point is their base (the pivot; layers are y-up).
        for layer in [blink, tail, ear] {
            layer.bounds = bounds
            layer.contentsGravity = .resize
            layer.position = CGPoint(x: size.width / 2, y: size.height / 2)
            layer.actions = ["contents": NSNull(), "transform": NSNull(), "opacity": NSNull(), "hidden": NSNull()]
            picture.addSublayer(layer)
        }
        for (layer, point) in [(tail, petTailPivot), (ear, petEarPivot)] {
            let pivot = CGPoint(x: point.x, y: size.height - point.y)
            layer.position = pivot
            layer.anchorPoint = CGPoint(x: pivot.x / size.width, y: pivot.y / size.height)
        }
        blink.opacity = 0
        ear.opacity = 0
        tail.isHidden = true
        // The sleeping dog's "z": a navy letter with a cream outline, as the pictures have a cream rim, so it reads on a
        // light and a dark desktop. Clear (opacity 0) except while its drift plays.
        let navy = NSColor(srgbRed: 0.06, green: 0.13, blue: 0.19, alpha: 1)
        let cream = NSColor(srgbRed: 0.97, green: 0.96, blue: 0.91, alpha: 1)
        drift.string = NSAttributedString(string: "z", attributes: [
            .font: NSFont.systemFont(ofSize: 13, weight: .heavy), .foregroundColor: navy,
            .strokeColor: cream, .strokeWidth: -4])
        drift.alignmentMode = .center
        drift.bounds = CGRect(x: 0, y: 0, width: 14, height: 15)
        drift.contentsScale = 2
        drift.opacity = 0
        drift.actions = ["transform": NSNull(), "opacity": NSNull(), "position": NSNull()]
        moving.addSublayer(drift)
    }

    /// Shows `image` (as many pixels as `points` times its scale) with the `parts` that move over it; `fade` seconds of
    /// cross-fade, 0 for a swap. The tail stays hidden until its wag plays, and the ear until its twitch.
    func show(_ image: CGImage, parts: PetParts = PetParts(), fade: TimeInterval) {
        picture.contentsScale = CGFloat(image.width) / picture.bounds.width
        if fade > 0 {
            let transition = CATransition()
            transition.type = .fade
            transition.duration = fade
            transition.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            picture.add(transition, forKey: kCATransition)
        }
        still = image
        picture.contents = image
        for (layer, piece) in [(blink, parts.blink), (tail, parts.tail), (ear, parts.ear)] {
            layer.contents = piece
            layer.contentsScale = CGFloat(piece?.width ?? 1) / size.width
        }
        tail.isHidden = true
        body = parts.body
        earBody = parts.earBody
        zStart = parts.zStart
        if let start = parts.zStart { drift.position = CGPoint(x: start.x, y: size.height - start.y) }
    }

    /// Starts what the plan says, or stops everything and leaves the still picture. `entering` is true on the
    /// change of mood, the only time a beat plays.
    func play(_ plan: PetMotionPlan, entering: Bool, animate: Bool) {
        stop()
        guard animate else { return }
        if let breath = plan.breath { picture.add(Self.breathing(breath), forKey: "breath") }
        if plan.blinks && blink.contents != nil { blink.add(Self.blinking(), forKey: "blink") }
        if plan.drifts && zStart != nil { drift.add(Self.drifting(), forKey: "drift") }
        if plan.twitches, let still = still, let earBody = earBody, ear.contents != nil {
            // The picture shows its ear-less body for the twitch only, and the ear turns over it; the rest of the
            // 13 s the still picture shows, so nothing is cut at rest.
            picture.add(Self.swapping(to: earBody, from: still), forKey: "twitch")
            ear.add(Self.twitching(), forKey: "twitch")
        }
        guard entering, let beat = plan.beat else { return }
        switch beat {
        case .bob: moving.add(Self.lift(points: 3), forKey: "beat")
        case .hop: moving.add(Self.lift(points: 6), forKey: "beat")
        case .wag:
            // The picture without its tail, and the tail over it: laid together they are the still picture.
            guard let body = body, tail.contents != nil else { return }
            picture.contents = body
            tail.isHidden = false
            tail.add(Self.wag(), forKey: "beat")
        }
    }

    func stop() {
        moving.removeAllAnimations()
        for layer in [blink, tail, ear, drift] { layer.removeAllAnimations() }
        picture.removeAnimation(forKey: "breath")
        picture.removeAnimation(forKey: "twitch")
        tail.isHidden = true
        if let still = still { picture.contents = still }
    }

    /// A breath: the vertical scale rises to `amount` and returns, `seconds` in all, forever. The easing is the
    /// sine in-out curve, so the body slows at the top and the bottom of each breath.
    static func breathing(_ breath: PetBreath) -> CABasicAnimation {
        let animation = CABasicAnimation(keyPath: "transform.scale.y")
        animation.fromValue = 1.0
        animation.toValue = breath.amount
        animation.duration = breath.seconds / 2
        animation.autoreverses = true
        animation.repeatCount = .infinity
        animation.timingFunction = CAMediaTimingFunction(controlPoints: 0.37, 0, 0.63, 1)
        return animation
    }

    /// A lift of `points` and back in 0.35 s, once: up on the first half with ease-out, down on the second with
    /// ease-in. (Layers are y-up.)
    static func lift(points: CGFloat) -> CAKeyframeAnimation {
        let animation = CAKeyframeAnimation(keyPath: "transform.translation.y")
        animation.values = [0, points, 0]
        animation.keyTimes = [0, 0.5, 1]
        animation.timingFunctions = [CAMediaTimingFunction(name: .easeOut), CAMediaTimingFunction(name: .easeIn)]
        animation.duration = 0.35
        return animation
    }

    /// Two blinks in an 11 s loop, at 4.2 s and 9.6 s: the patch over the eyes fades in over 60 ms, holds 30 ms and
    /// fades out over 70 ms, so a blink is 160 ms. The patch is clear the rest of the time.
    static func blinking() -> CAKeyframeAnimation {
        let animation = CAKeyframeAnimation(keyPath: "opacity")
        animation.values = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        animation.keyTimes = [0, 4.2, 4.26, 4.29, 4.36, 9.6, 9.66, 9.69, 9.76, 11].map { NSNumber(value: $0 / 11) }
        animation.duration = 11
        animation.repeatCount = .infinity
        return animation
    }

    /// A tail wag: the tail turns 16 degrees to one side in 0.08 s, swings four times to the other side and back (a
    /// swing is 0.32 s, each half-swing eases in and out), then settles in 0.24 s: 1.6 s, once.
    static func wag() -> CAKeyframeAnimation {
        let angle = 16 * CGFloat.pi / 180
        let animation = CAKeyframeAnimation(keyPath: "transform.rotation.z")
        animation.values = [0, angle, -angle, angle, -angle, angle, -angle, angle, -angle, angle, 0]
        animation.keyTimes = [0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1]
        animation.timingFunctions = Array(repeating: CAMediaTimingFunction(name: .easeInEaseOut), count: 10)
        animation.duration = 1.6
        return animation
    }

    /// The ear's twitch window in its 13 s loop: it starts at 7 s (between the blinks, which are at 4.2 s and 9.6 s of
    /// their 11 s loop) and takes 0.42 s.
    static let twitchLoop: TimeInterval = 13
    static let twitchAt: TimeInterval = 7
    static let twitchFor: TimeInterval = 0.42

    /// The ear's twitch, every 13 s: it turns 12 degrees one way in 0.12 s, 4 degrees the other way in 0.16 s and back
    /// to rest in 0.14 s, 420 ms in all (layers are y-up: the first turn is clockwise, the ear's tip going outward).
    /// The ear shows only while it turns.
    static func twitching() -> CAAnimationGroup {
        let angle = CGFloat.pi / 180
        let loop = twitchLoop, start = twitchAt, end = twitchAt + twitchFor
        let turn = CAKeyframeAnimation(keyPath: "transform.rotation.z")
        turn.values = [0, 0, -12 * angle, 4 * angle, 0, 0]
        turn.keyTimes = [0, start, start + 0.12, start + 0.28, end, loop].map { NSNumber(value: $0 / loop) }
        turn.timingFunctions = [CAMediaTimingFunction(name: .linear), CAMediaTimingFunction(name: .easeOut),
                                CAMediaTimingFunction(name: .easeInEaseOut), CAMediaTimingFunction(name: .easeOut),
                                CAMediaTimingFunction(name: .linear)]
        let show = CAKeyframeAnimation(keyPath: "opacity")
        show.values = [0, 1, 0]
        show.keyTimes = [0, start, end].map { NSNumber(value: $0 / loop) }
        show.calculationMode = .discrete
        let group = CAAnimationGroup()
        group.animations = [turn, show]
        for animation in [turn, show, group] { animation.duration = loop }
        group.repeatCount = .infinity
        return group
    }

    /// The picture's own contents for the 13 s loop, with the ear-less `body` in place of the `still` while the ear
    /// turns (discrete: the picture jumps, it does not blend).
    static func swapping(to body: CGImage, from still: CGImage) -> CAKeyframeAnimation {
        let animation = CAKeyframeAnimation(keyPath: "contents")
        animation.values = [still, body, still]
        animation.keyTimes = [0, twitchAt, twitchAt + twitchFor].map { NSNumber(value: $0 / twitchLoop) }
        animation.calculationMode = .discrete
        animation.duration = twitchLoop
        animation.repeatCount = .infinity
        return animation
    }

    /// The sleeping dog's "z": it fades in, drifts 4 pt up and fades out, 3 s, again and again.
    static func drifting() -> CAAnimationGroup {
        let rise = CABasicAnimation(keyPath: "transform.translation.y")
        rise.fromValue = 0
        rise.toValue = 4
        let fade = CAKeyframeAnimation(keyPath: "opacity")
        fade.values = [0, 1, 1, 0]
        fade.keyTimes = [0, 0.25, 0.6, 1]
        let group = CAAnimationGroup()
        group.animations = [rise, fade]
        for animation in [rise, fade, group] { animation.duration = 3 }
        group.repeatCount = .infinity
        return group
    }
}
