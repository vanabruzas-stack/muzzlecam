import CoreGraphics
import Foundation

@main
struct RegistrationCheck {
    static func main() {
        let side = 96
        // A non-periodic signal with texture at several frequencies.
        func signal(_ x: Double, _ y: Double) -> Float {
            let u = x - 43
            let v = y - 51
            let blob = exp(-(u * u + v * v) / 470)
            let wave = 0.09 * sin(x * 0.32 + y * 0.12) * cos(y * 0.43 - x * 0.26)
            return Float(0.38 + 0.33 * blob + wave + 0.10 * sin(x * y * 0.013))
        }
        func patch(_ dx: Double, _ dy: Double, exposure: Float = 1) -> GrayPatch {
            let samples = (0..<(side * side)).map {
                signal(Double($0 % side) - dx, Double($0 / side) - dy) * exposure
            }
            return GrayPatch(size: side, values: samples)
        }
        let reference = patch(0, 0)
        guard let match = PatchRegistration.match(reference, patch(3.4, -4.2, exposure: 0.57), radius: 12)
        else { fatalError("Textured image was rejected") }
        precondition(abs(match.offset.x - 3.4) < 0.45 && abs(match.offset.y + 4.2) < 0.45,
                     "Incorrect translation: \(match.offset)")
        let flat = GrayPatch(size: side, values: [Float](repeating: 0.5, count: side * side))
        precondition(PatchRegistration.match(reference, flat, radius: 12) == nil)
        precondition(PatchRegistration.match(flat, reference, radius: 12) == nil)

        let rect = CGRect(x: 0, y: 0, width: 3024, height: 4032)
        let center = CGPoint(x: 0.9, y: 0.1)
        let crop = CameraGeometry.crop(in: rect, center: center, digital: 100, aspect: 0.46)
        let selected = CameraGeometry.point(in: crop, at: CGPoint(x: 0.5, y: 0.5), in: rect)
        precondition(rect.contains(crop) && abs(crop.width / crop.height - 0.46) < 1e-6)
        precondition(abs(selected.x - center.x) < 1e-6 && abs(selected.y - center.y) < 1e-6)

        let native = CGRect(x: 0, y: 0, width: 6048, height: 8064)
        let anchor = CGPoint(x: 0.5, y: 0.5)
        precondition(CameraGeometry.crop(in: native, center: anchor, digital: 1, aspect: 0.75) == native)
        var stabilized = ReferenceCropTracker()
        var peakError = 0.0
        for index in 0..<90 {
            let drift = 0.008 * sin(Double(index) * 0.43)
            let observed = CGPoint(x: anchor.x + drift, y: anchor.y - drift)
            precondition(stabilized.update(estimate: observed, digital: 3, sourceWidth: 1440))
            peakError = max(peakError, hypot(stabilized.center.x - observed.x, stabilized.center.y - observed.y))
        }
        precondition(peakError < 0.002, "Synthetic jitter was not reduced enough")
        let held = stabilized.center
        for _ in 0..<60 {
            precondition(!stabilized.update(estimate: nil, digital: 50, sourceWidth: 1440))
            precondition(stabilized.center == held)
        }
        let outlier = CGPoint(x: held.x + 0.10, y: held.y)
        precondition(!stabilized.update(estimate: outlier, digital: 50, sourceWidth: 1440))
        precondition(stabilized.center == held, "A single outlier moved the image")
        precondition(stabilized.update(estimate: outlier, digital: 50, sourceWidth: 1440))
        precondition(abs(stabilized.center.x - held.x - 0.24 / 50) < 1e-9)

        stabilized.reset(to: anchor, digital: 50)
        let target = CGPoint(x: 0.503, y: 0.497)
        for _ in 0..<300 { stabilized.update(estimate: target, digital: 50, sourceWidth: 1440) }
        precondition(hypot(stabilized.center.x - target.x, stabilized.center.y - target.y) < 0.0002,
                     "A stationary reference drifted")
        print("Registration, framing, stationary reference and dropout: OK")
    }
}
