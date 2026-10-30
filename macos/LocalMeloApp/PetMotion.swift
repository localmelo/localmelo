import QuartzCore

/// How the desktop pet moves. Every motion is a Core Animation animation added once to a layer: the render server
/// runs it, so the app does no per-frame work (no timer, no `draw`), and the pictures are whole bitmaps, so the
/// motions are transforms of the whole picture and cross-fades between pictures. All of them stop (the state's
/// still picture stays) when `shouldAnimate` says so; a change of picture still cross-fades unless Reduce Motion
/// asks for a plain swap (`petFade`).

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
    /// A run finished well: the picture swings from side to side and settles, as a tail wag.
    case wiggle
}

struct PetMotionPlan: Equatable {
    var breath: PetBreath?
    var beat: PetBeat?
}

/// What a mood does: it breathes unless nothing is known yet or the app stopped; the sleeping breath is slower and
/// a little deeper. `beat` is for the moment the mood begins and is not repeated.
func petMotionPlan(_ mood: PetMood) -> PetMotionPlan {
    switch mood {
    case .unknown, .stopped: return PetMotionPlan(breath: nil, beat: nil)
    case .sleeping: return PetMotionPlan(breath: PetBreath(amount: 1.03, seconds: 4.4), beat: nil)
    case .resting, .failed: return PetMotionPlan(breath: PetBreath(amount: 1.02, seconds: 3.2), beat: nil)
    case .working: return PetMotionPlan(breath: PetBreath(amount: 1.02, seconds: 3.2), beat: .bob)
    case .waiting: return PetMotionPlan(breath: PetBreath(amount: 1.02, seconds: 3.2), beat: .hop)
    case .done: return PetMotionPlan(breath: PetBreath(amount: 1.02, seconds: 3.2), beat: .wiggle)
    }
}

/// The pet's two layers. `moving` carries the beats (lift, wiggle) and `picture` the breathing and the bitmap; both
/// turn about the point on the ground line under the centre of the picture, so the paws stay where they are.
final class PetMotion {
    let moving = CALayer()
    let picture = CALayer()

    /// `size` is the panel's canvas and `anchor` the point the motions turn about, in points from the canvas's
    /// bottom-left (the centre of the ground line).
    init(size: CGSize, anchor: CGPoint) {
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
    }

    /// Shows `image` (as many pixels as `points` times its scale); `fade` seconds of cross-fade, 0 for a swap.
    func show(_ image: CGImage, fade: TimeInterval) {
        picture.contentsScale = CGFloat(image.width) / picture.bounds.width
        if fade > 0 {
            let transition = CATransition()
            transition.type = .fade
            transition.duration = fade
            transition.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            picture.add(transition, forKey: kCATransition)
        }
        picture.contents = image
    }

    /// Starts what the plan says, or stops everything and leaves the still picture. `entering` is true on the
    /// change of mood, the only time a beat plays.
    func play(_ plan: PetMotionPlan, entering: Bool, animate: Bool) {
        stop()
        guard animate else { return }
        if let breath = plan.breath { picture.add(Self.breathing(breath), forKey: "breath") }
        guard entering, let beat = plan.beat else { return }
        switch beat {
        case .bob: moving.add(Self.lift(points: 3), forKey: "beat")
        case .hop: moving.add(Self.lift(points: 6), forKey: "beat")
        case .wiggle: moving.add(Self.wiggle(), forKey: "beat")
        }
    }

    func stop() {
        moving.removeAllAnimations()
        picture.removeAnimation(forKey: "breath")
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

    /// A tail wag from a lying body: the whole picture turns 3 degrees to one side in 0.08 s, swings four times to
    /// the other side and back (a swing is 0.32 s, each half-swing eases in and out), then settles in 0.24 s: 1.6 s,
    /// once.
    static func wiggle() -> CAKeyframeAnimation {
        let angle = 3 * CGFloat.pi / 180
        let animation = CAKeyframeAnimation(keyPath: "transform.rotation.z")
        animation.values = [0, angle, -angle, angle, -angle, angle, -angle, angle, -angle, angle, 0]
        animation.keyTimes = [0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1]
        animation.timingFunctions = Array(repeating: CAMediaTimingFunction(name: .easeInEaseOut), count: 10)
        animation.duration = 1.6
        return animation
    }
}
