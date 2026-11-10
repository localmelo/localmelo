import AppKit

/// The menu-bar item's mark: Melo's head as lines, derived from the head picture by scripts/make_menu_bar_mark.py.
/// Two masks per scale (18 and 36 px), each black with the mask in its alpha: the lines in the bar's text colour
/// (MenuBarMark.png) and the ears and crown in Melo's fur colour (MenuBarMarkFur.png). The image is coloured for the
/// appearance of the bar it is on, and made again when that changes, with the waiting dot when something waits.
final class MenuBarMark {
    /// The masks' coverage by scale, row by row from the top: line, then fur.
    private let masks: [Int: (line: [CGFloat], fur: [CGFloat])]
    private var images: [String: NSImage] = [:]

    /// Nil unless the folder holds the four masks at their sizes.
    init?(folder: URL?) {
        var masks: [Int: (line: [CGFloat], fur: [CGFloat])] = [:]
        for (scale, suffix) in [(1, ""), (2, "@2x")] {
            let side = 18 * scale
            func coverage(_ name: String) -> [CGFloat]? {
                guard let url = folder?.appendingPathComponent("\(name)\(suffix).png"),
                      let data = try? Data(contentsOf: url), let rep = NSBitmapImageRep(data: data),
                      rep.pixelsWide == side, rep.pixelsHigh == side else { return nil }
                return (0..<side * side).map { rep.colorAt(x: $0 % side, y: $0 / side)?.alphaComponent ?? 0 }
            }
            guard let line = coverage("MenuBarMark"), let fur = coverage("MenuBarMarkFur") else { return nil }
            masks[scale] = (line, fur)
        }
        self.masks = masks
    }

    /// The item's image on a bar of `appearance`: 18 points square, or 23 wide with the red dot beside the head's top
    /// right, so the head is never covered. It is not a template: its colours are the bar's label colour, the fur pair
    /// and the dot's pair, as `appearance` resolves them.
    func image(for appearance: NSAppearance, badge: Bool) -> NSImage {
        let key = "\(appearance.name.rawValue) \(badge)"
        if let image = images[key] { return image }
        var line = NSColor.white, fur = NSColor.white, ring = NSColor.white, dot = NSColor.red
        appearance.performAsCurrentDrawingAppearance {
            line = NSColor.labelColor.usingColorSpace(.sRGB) ?? .white
            fur = Palette.fur.usingColorSpace(.sRGB) ?? .orange
            ring = Palette.badgeRing.usingColorSpace(.sRGB) ?? .white
            dot = Palette.badge.usingColorSpace(.sRGB) ?? .red
        }
        let size = NSSize(width: badge ? 23 : 18, height: 18)
        let image = NSImage(size: size)
        for scale in [1, 2] {
            guard let rep = draw(scale: scale, width: Int(size.width), line: line, fur: fur,
                                 dot: badge ? (ring, dot) : nil) else { continue }
            rep.size = size
            image.addRepresentation(rep)
        }
        image.isTemplate = false
        images[key] = image
        return image
    }

    /// The image at `scale` in an sRGB bitmap: the head's pixels take the colours its masks cover, by their share
    /// (premultiplied, written byte by byte), then the dot, drawn in points.
    private func draw(scale: Int, width: Int, line: NSColor, fur: NSColor, dot: (ring: NSColor, fill: NSColor)?)
        -> NSBitmapImageRep? {
        guard let mask = masks[scale], let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: width * scale, height: 18 * scale, bitsPerComponent: 8,
                                      bytesPerRow: width * scale * 4, space: space,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let bytes = context.data?.assumingMemoryBound(to: UInt8.self) else { return nil }
        let side = 18 * scale, stride = width * scale * 4
        let colours = [line, fur].map { [$0.redComponent, $0.greenComponent, $0.blueComponent, $0.alphaComponent] }
        for index in 0..<side * side {
            let shares = [mask.line[index] * colours[0][3], mask.fur[index] * colours[1][3]]
            let at = index / side * stride + index % side * 4
            for channel in 0..<3 {
                let value = shares[0] * colours[0][channel] + shares[1] * colours[1][channel]
                bytes[at + channel] = UInt8(min(255, (value * 255).rounded()))
            }
            bytes[at + 3] = UInt8(min(255, ((shares[0] + shares[1]) * 255).rounded()))
        }
        if let dot = dot {
            context.scaleBy(x: CGFloat(scale), y: CGFloat(scale))
            context.setFillColor(dot.ring.cgColor)
            context.fillEllipse(in: CGRect(x: 15, y: 10, width: 8, height: 8))
            context.setFillColor(dot.fill.cgColor)
            context.fillEllipse(in: CGRect(x: 16, y: 11, width: 6, height: 6))
        }
        return context.makeImage().map { NSBitmapImageRep(cgImage: $0) }
    }
}
