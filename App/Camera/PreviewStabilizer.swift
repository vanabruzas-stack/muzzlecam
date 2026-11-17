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

    private struct Keyframe {
        let wide: GrayPatch
        let detail: GrayPatch?
        let span: CGFloat
        let created: Double
    }

    private let motion = CMMotionManager()
    private var tracker = ReferenceCropTracker()
    private var keyframe: Keyframe?
    private var lastTime = 0.0
    private var previousHardware: CGFloat = 0
    private var previousZoom: Double = 0
    private var resetID = -1
    private var lastMatch: Double?
    private var consecutiveMatches = 0
    private var panStarted: Double?
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
        clearReference()
        lastTime = 0
        panStarted = nil
        resetID = -1
    }

    func update(_ buffer: CVPixelBuffer, time: Double, controls: ZoomControls, hardware: CGFloat) -> Reading {
        let digital = max(1, CGFloat(controls.zoom) / hardware)
        let extent = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer))
        let area = CameraGeometry.sourceArea(in: extent)
        let requested = controls.locked ? controls.anchor : CGPoint(x: 0.5, y: 0.5)
        let elapsed = time - lastTime
        let dt = max(0, min(0.07, elapsed))
        defer {
            lastTime = time
            previousHardware = hardware
            previousZoom = controls.zoom
        }
        if resetID != controls.trackingReset || abs(previousHardware - hardware) > 0.015 {
            tracker.reset(to: requested, digital: digital)
            resetID = controls.trackingReset
            clearReference()
            panStarted = nil
        }
        if elapsed <= 0 || elapsed > 0.28 || abs(previousZoom / controls.zoom - 1) > 0.04 {
            clearReference()
        }

        if time - lastMeter >= 0.4 || time < lastMeter {
            lastMeter = time
            let view = CameraGeometry.crop(in: extent, center: tracker.center, digital: digital,
                                           aspect: CameraGeometry.aspect)
            let meter = CGRect(x: view.minX, y: extent.height - view.maxY, width: view.width, height: view.height)
            highlights = luma(buffer, rect: meter, size: 28)?.clippedHighlights ?? false
        }

        guard controls.softwareStabilization, controls.zoom >= ZoomControls.minimumLockZoom, digital > 1.005 else {
            tracker.reset(to: requested, digital: digital)
            clearReference()
            return Reading(center: tracker.center, tracking: false, highlights: highlights, sample: nil)
        }

        // The gyro detects sustained pans. Sensor-shift OIS has already acted on
        // the image, so gyro data never moves the crop directly during tracking.
        let rate = motion.deviceMotion?.rotationRate
        if !controls.locked && hypot(rate?.x ?? 0, rate?.y ?? 0) > 0.34 {
            if panStarted == nil { panStarted = time }
        } else {
            panStarted = nil
        }
        if let panStarted, time - panStarted > 0.15 {
            clearReference()
            tracker.pan(toward: requested, digital: digital, elapsed: dt)
            return Reading(center: tracker.center, tracking: false, highlights: highlights, sample: nil)
        }

        let searchCenter = CameraGeometry.bounded(tracker.center, digital: digital)
        let wideSpan = keyframe?.span ?? area.width * min(0.34, max(0.12, 1 / digital))
        func patch(at point: CGPoint, span: CGFloat, size: Int) -> GrayPatch? {
            let rect = CGRect(x: area.minX + point.x * area.width - span / 2,
                              y: extent.height - area.maxY + point.y * area.height - span / 2,
                              width: span, height: span)
            return luma(buffer, rect: rect, size: size)
        }
        func shifted(_ point: CGPoint, by shift: PatchRegistration.Match, span: CGFloat, size: Int) -> CGPoint {
            let pixelsPerSample = span / CGFloat(size)
            return CGPoint(x: point.x + CGFloat(shift.offset.x) * pixelsPerSample / area.width,
                           y: point.y + CGFloat(shift.offset.y) * pixelsPerSample / area.height)
        }

        let widePatch = patch(at: searchCenter, span: wideSpan, size: 72)
        let detailSpan = max(30, wideSpan * 0.38)
        if keyframe == nil, let widePatch {
            let detail = digital >= 4 ? patch(at: searchCenter, span: detailSpan, size: 56) : nil
            keyframe = Keyframe(wide: widePatch, detail: detail, span: wideSpan, created: time)
        }

        var estimate: CGPoint?
        if let keyframe, let widePatch, time > keyframe.created,
           let coarse = PatchRegistration.match(keyframe.wide, widePatch, radius: 10, threshold: 0.89) {
            let offset = shifted(searchCenter, by: coarse, span: wideSpan, size: widePatch.size)
            if let detail = keyframe.detail,
               let current = patch(at: offset, span: detailSpan, size: detail.size),
               let fine = PatchRegistration.match(detail, current, radius: 4, threshold: 0.91,
                                                  minimumPeakGap: 0.012) {
                estimate = shifted(offset, by: fine, span: detailSpan, size: detail.size)
            } else if digital < 16 || coarse.score >= 0.96 {
                estimate = offset
            }
        }

        if tracker.update(estimate: estimate, digital: digital, sourceWidth: area.width) {
            lastMatch = time
            consecutiveMatches = min(consecutiveMatches + 1, 3)
        } else {
            consecutiveMatches = 0
            // A lost lock holds its last crop and waits for the subject or a tap.
            // Auto stabilization can refresh at the same crop without a jump.
            if let keyframe, !controls.locked, time - (lastMatch ?? keyframe.created) > 0.27 {
                clearReference()
            }
        }
        let tracking = consecutiveMatches >= 2
        let sample = tracking ? widePatch.map {
            TrackingSample(patch: $0, center: searchCenter, widthFraction: wideSpan / area.width)
        } : nil
        return Reading(center: tracker.center, tracking: tracking, highlights: highlights, sample: sample)
    }

    private func clearReference() {
        keyframe = nil
        lastMatch = nil
        consecutiveMatches = 0
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
