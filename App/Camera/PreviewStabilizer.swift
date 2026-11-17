import CoreMotion
import CoreVideo
import Foundation
import simd

struct TrackingSample {
    let patch: GrayPatch
    let center: CGPoint
    let widthFraction: CGFloat
}

final class PreviewStabilizer {
    struct Reading {
        let center: CGPoint
        let tracking: Bool
        let highlights: Bool
        let sample: TrackingSample?
    }

    private let motion = CMMotionManager()
    private var prior: GrayPatch?
    private var priorCenter = CGPoint(x: 0.5, y: 0.5)
    private var center = CGPoint(x: 0.5, y: 0.5)
    private var lastTime = 0.0
    private var previousHardware: CGFloat = 0
    private var previousZoom: Double = 0
    private var resetID = -1
    private var lastMatch = 0.0
    private var highlights = false
    private var lastMeter = 0.0

    var gyroAvailable: Bool { motion.isDeviceMotionAvailable }

    func start() {
        guard motion.isDeviceMotionAvailable, !motion.isDeviceMotionActive else { return }
        motion.deviceMotionUpdateInterval = 1 / 100
        motion.startDeviceMotionUpdates()
    }

    func stop() {
        motion.stopDeviceMotionUpdates()
        prior = nil
        resetID = -1
    }

    func update(_ buffer: CVPixelBuffer, time: Double, controls: ZoomControls,
                hardware: CGFloat, fieldOfView: CGFloat) -> Reading {
        let digital = max(1, CGFloat(controls.zoom) / hardware)
        let size = CGSize(width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer))
        let area = CameraGeometry.sourceArea(in: CGRect(origin: .zero, size: size))
        let areaTop = size.height - area.maxY
        let patchWidth = min(area.width * 0.46, max(area.width * 0.16, area.width / digital * 1.3))
        let roiSpan = max(2, patchWidth)
        let viewport = CameraGeometry.crop(in: CGRect(origin: .zero, size: size),
                                           center: center, digital: digital, aspect: controls.aspect)
        if time - lastMeter >= 0.4 || time < lastMeter {
            lastMeter = time
            let meter = CGRect(x: viewport.minX, y: size.height - viewport.maxY,
                               width: viewport.width, height: viewport.height)
            highlights = luma(buffer, rect: meter, size: 28)?.clippedHighlights ?? false
        }

        defer {
            lastTime = time
            previousHardware = hardware
            previousZoom = controls.zoom
        }
        let requested = controls.locked ? controls.anchor : CGPoint(x: 0.5, y: 0.5)
        if resetID != controls.trackingReset || abs(previousHardware - hardware) > 0.02 ||
            time - lastTime > 0.23 || time <= lastTime {
            center = CameraGeometry.bounded(requested, digital: digital)
            prior = nil
            lastMatch = 0
            resetID = controls.trackingReset
        }
        guard digital > 1.08 else {
            center = CameraGeometry.bounded(requested, digital: digital)
            prior = nil
            return Reading(center: center, tracking: false, highlights: highlights, sample: nil)
        }

        let searchCenter = center
        let rect = CGRect(x: area.minX + searchCenter.x * area.width - roiSpan / 2,
                          y: areaTop + searchCenter.y * area.height - roiSpan / 2,
                          width: roiSpan, height: roiSpan)
        let patch = luma(buffer, rect: rect, size: 60)
        var predicted = CGPoint.zero
        if let sample = motion.deviceMotion, time - lastTime < 0.12 {
            let focal = size.height * hardware / (2 * tan(max(0.15, fieldOfView) / 2))
            let dt = max(0, time - lastTime)
            predicted = CGPoint(x: sample.rotationRate.y * dt * focal / area.width,
                                y: sample.rotationRate.x * dt * focal / area.height)
        }
        let match: PatchRegistration.Match?
        if let prior, let patch, abs(previousZoom / controls.zoom - 1) < 0.025 {
            match = PatchRegistration.match(prior, patch, radius: 8, threshold: 0.83)
        } else {
            match = nil
        }

        var next = searchCenter
        if let match, let patch {
            lastMatch = time
            let scale = roiSpan / CGFloat(patch.size)
            next = CGPoint(x: priorCenter.x + CGFloat(match.offset.x) * scale / area.width,
                           y: priorCenter.y + CGFloat(match.offset.y) * scale / area.height)
        } else if time - lastTime < 0.12 {
            next.x += max(-0.011, min(0.011, predicted.x))
            next.y += max(-0.011, min(0.011, predicted.y))
        }

        // When unlocked, let deliberate pans lead the crop. While locked, follow
        // the neighborhood with a slow drift correction; clamp at sensor edges.
        let recenter: CGFloat = controls.locked ? (time - lastMatch > 0.5 ? 0.014 : 0.001) : 0.21
        next.x += (requested.x - next.x) * recenter
        next.y += (requested.y - next.y) * recenter
        center = CameraGeometry.bounded(next, digital: digital)

        prior = patch
        priorCenter = searchCenter
        let sample = patch.map { TrackingSample(patch: $0, center: searchCenter, widthFraction: roiSpan / area.width) }
        return Reading(center: center, tracking: time - lastMatch < 0.4,
                       highlights: highlights, sample: sample)
    }

    private func luma(_ buffer: CVPixelBuffer, rect: CGRect, size: Int) -> GrayPatch? {
        guard CVPixelBufferIsPlanar(buffer), CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess
        else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return nil }
        let src = base.assumingMemoryBound(to: UInt8.self)
        let width = CVPixelBufferGetWidthOfPlane(buffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let rowBytes = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        var values = [Float](repeating: 0, count: size * size)
        let videoRange = CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        for y in 0..<size {
            let sy = min(height - 1, max(0, Int(rect.minY + (CGFloat(y) + 0.5) / CGFloat(size) * rect.height)))
            for x in 0..<size {
                let sx = min(width - 1, max(0, Int(rect.minX + (CGFloat(x) + 0.5) / CGFloat(size) * rect.width)))
                let sample = Float(src[sy * rowBytes + sx])
                values[y * size + x] = videoRange ? max(0, min(1, (sample - 16) / 219)) : sample / 255
            }
        }
        return GrayPatch(size: size, values: values)
    }
}
