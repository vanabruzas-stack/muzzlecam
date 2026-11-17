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

    func update(_ buffer: CVPixelBuffer, time: Double, controls: ZoomControls, hardware: CGFloat) -> Reading {
        let digital = max(1, CGFloat(controls.zoom) / hardware)
        let size = CGSize(width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer))
        let area = CameraGeometry.sourceArea(in: CGRect(origin: .zero, size: size))
        let areaTop = size.height - area.maxY
        let patchWidth = min(area.width * 0.38, max(area.width * 0.13, area.width / digital * 1.2))
        let roiSpan = max(2, patchWidth)
        let viewport = CameraGeometry.crop(in: CGRect(origin: .zero, size: size),
                                           center: center, digital: digital, aspect: CameraGeometry.aspect)
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
        let elapsed = time - lastTime
        if resetID != controls.trackingReset || abs(previousHardware - hardware) > 0.015 {
            center = CameraGeometry.bounded(requested, digital: digital)
            prior = nil
            lastMatch = 0
            resetID = controls.trackingReset
        }
        if elapsed > 0.23 || elapsed <= 0 { prior = nil }
        guard controls.zoom >= ZoomControls.minimumLockZoom, digital > 1.005 else {
            center = CameraGeometry.bounded(requested, digital: digital)
            prior = nil
            return Reading(center: center, tracking: false, highlights: highlights, sample: nil)
        }

        let searchCenter = center
        let rect = CGRect(x: area.minX + searchCenter.x * area.width - roiSpan / 2,
                          y: areaTop + searchCenter.y * area.height - roiSpan / 2,
                          width: roiSpan, height: roiSpan)
        let patch = luma(buffer, rect: rect, size: 76)
        let match: PatchRegistration.Match?
        if let prior, let patch, abs(previousZoom / controls.zoom - 1) < 0.025 {
            match = PatchRegistration.match(prior, patch, radius: 10, threshold: 0.86)
        } else {
            match = nil
        }

        var movement = CGPoint.zero
        if let match, let patch {
            lastMatch = time
            let scale = roiSpan / CGFloat(patch.size)
            movement = CGPoint(x: priorCenter.x - searchCenter.x + CGFloat(match.offset.x) * scale / area.width,
                               y: priorCenter.y - searchCenter.y + CGFloat(match.offset.y) * scale / area.height)
        }

        // OIS already corrects motion before the frame arrives. Use the gyro
        // to detect panning without injecting a second correction into the crop.
        let rate = motion.deviceMotion?.rotationRate
        let panning = hypot(rate?.x ?? 0, rate?.y ?? 0) > 0.40
        let recentlyTracked = time - lastMatch < 0.18
        let followTime: Double = controls.locked ? (recentlyTracked ? 20 : 2.4) : (panning ? 0.16 : 0.72)
        let hold = match == nil && recentlyTracked
        let dt = hold ? 0 : min(0.075, max(0, elapsed))
        center = CameraGeometry.stabilizedCenter(searchCenter, toward: requested, movement: movement,
                                                digital: digital, elapsed: dt, followTime: followTime)

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
            let sy = min(CGFloat(height - 1), max(0, rect.minY + (CGFloat(y) + 0.5) * rect.height / CGFloat(size) - 0.5))
            let y0 = Int(sy)
            let y1 = min(height - 1, y0 + 1)
            let fy = Float(sy - CGFloat(y0))
            for x in 0..<size {
                let sx = min(CGFloat(width - 1), max(0, rect.minX + (CGFloat(x) + 0.5) * rect.width / CGFloat(size) - 0.5))
                let x0 = Int(sx)
                let x1 = min(width - 1, x0 + 1)
                let fx = Float(sx - CGFloat(x0))
                let top = Float(src[y0 * rowBytes + x0]) * (1 - fx) + Float(src[y0 * rowBytes + x1]) * fx
                let bottom = Float(src[y1 * rowBytes + x0]) * (1 - fx) + Float(src[y1 * rowBytes + x1]) * fx
                let sample = top * (1 - fy) + bottom * fy
                values[y * size + x] = videoRange ? max(0, min(1, (sample - 16) / 219)) : sample / 255
            }
        }
        return GrayPatch(size: size, values: values)
    }
}
