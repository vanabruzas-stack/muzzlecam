import CoreImage
import Foundation
import Metal
import QuartzCore

final class LockedValue<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value

    init(_ value: Value) { self.value = value }

    func read() -> Value {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func update(_ change: (inout Value) -> Void) {
        lock.lock()
        defer { lock.unlock() }
        change(&value)
    }
}

struct ViewfinderFrame {
    let id: UInt64
    let image: CIImage
    let arrived: Double
    let rendering: PreviewStatus
    let center: CGPoint
    let crop: CGRect
    let sourceArea: CGRect
    let sourceExtent: CGRect
    let controls: ZoomControls
    let hardware: CGFloat
    let tracking: Bool
    let highlights: Bool
    let sample: TrackingSample?
}

enum PreviewStatus: Equatable {
    case standard, neural, limited, modelUnavailable
}

struct CameraHardware {
    var zoom: CGFloat = 1
    var constrained = false
}

final class PreviewPipeline {
    let queue = DispatchQueue(label: "MuzzleZoom.Viewfinder", qos: .userInteractive)
    let controls = LockedValue(ZoomControls())
    let hardware = LockedValue(CameraHardware())
    let latest = LockedValue<ViewfinderFrame?>(nil)
    let device: MTLDevice?
    var gyroAvailable: Bool { stabilizer.gyroAvailable }

    private let context: CIContext
    private let stabilizer = PreviewStabilizer()
    private let modelLoader = DispatchQueue(label: "MuzzleZoom.PreviewModel", qos: .userInitiated)
    private var neural: NeuralScaler?
    private var modelLoading = false
    private var sequence: UInt64 = 0
    private var frameTime = 0.0
    private var limitedUntil = 0.0
    private var modelFailed = false

    init(device: MTLDevice?) {
        self.device = device
        context = device.map { CIContext(mtlDevice: $0, options: [.cacheIntermediates: false]) }
            ?? CIContext(options: [.cacheIntermediates: false])
    }

    func start() {
        queue.async { self.stabilizer.start() }
    }

    func stop() {
        queue.async {
            self.stabilizer.stop()
            self.latest.update { $0 = nil }
            self.frameTime = 0
            self.limitedUntil = 0
        }
    }

    // AVCaptureVideoDataOutput invokes this method on queue and discards late frames.
    func consume(_ buffer: CVPixelBuffer, timestamp: Double) {
        let controls = controls.read()
        let hardware = hardware.read()
        let thermal = ProcessInfo.processInfo.thermalState
        let hot = thermal == .serious || thermal == .critical || hardware.constrained
        let wantsNeural = controls.neural && controls.mode != .quality
        if wantsNeural && !hot && neural == nil && !modelFailed && !modelLoading {
            loadModel()
        }
        let canRunNeural = wantsNeural && !hot && !modelFailed && timestamp >= limitedUntil
        let activeModel = canRunNeural ? neural : nil
        let fps: Double = hot ? 18 : (activeModel != nil ? 20 : 30)
        guard timestamp < frameTime || timestamp - frameTime >= 0.97 / fps else { return }
        frameTime = timestamp

        let reading = stabilizer.update(buffer, time: timestamp, controls: controls,
                                        hardware: hardware.zoom)
        let image = CIImage(cvPixelBuffer: buffer)
        let area = CameraGeometry.sourceArea(in: image.extent)
        let crop = CameraGeometry.crop(in: image.extent, center: reading.center,
                                       digital: CGFloat(controls.zoom) / hardware.zoom,
                                       aspect: CameraGeometry.aspect)
        let grid = max(4, (crop.height / 4).rounded(.toNearestOrEven))
        let cropped = ImageTools.sample(image, rect: crop, width: grid * 3, height: grid * 4)
        var output = cropped
        var status: PreviewStatus = wantsNeural ? (modelFailed ? .modelUnavailable : .limited) : .standard
        let strength: Float = controls.mode == .extreme ? 0.12 : 0.27
        if let activeModel {
            let start = CACurrentMediaTime()
            do {
                let enriched = try activeModel.live(cropped, context: context)
                // Preserve source detail when the neural path uses a smaller input.
                let height = max(cropped.extent.height, enriched.extent.height)
                let baseline = ImageTools.resized(cropped, width: height * cropped.extent.width / cropped.extent.height,
                                                  height: height, highQuality: false)
                output = NeuralScaler.blend(enriched, with: baseline, amount: strength)
                status = .neural
            } catch {
                status = .modelUnavailable
                modelFailed = true
            }
            if CACurrentMediaTime() - start > 0.052 {
                limitedUntil = timestamp + 3
            }
        }
        if status != .neural, controls.zoom >= 5, !hot {
            output = ImageTools.filtered(cropped, noise: 0.017, sharpness: 0.15)
        }

        sequence &+= 1
        let frame = ViewfinderFrame(id: sequence, image: output, arrived: CACurrentMediaTime(), rendering: status,
                                    center: reading.center, crop: crop, sourceArea: area,
                                    sourceExtent: image.extent, controls: controls,
                                    hardware: hardware.zoom, tracking: reading.tracking,
                                    highlights: reading.highlights, sample: reading.sample)
        latest.update { $0 = frame }
    }

    private func loadModel() {
        modelLoading = true
        modelLoader.async {
            let candidate = NeuralScaler(.preview)
            let ready = candidate.prepare()
            self.queue.async {
                self.modelLoading = false
                self.modelFailed = !ready
                self.neural = ready ? candidate : nil
            }
        }
    }
}
