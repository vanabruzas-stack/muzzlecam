import CoreImage
import CoreImage.CIFilterBuiltins
import CoreML
import OSLog

/// Each pipeline owns a model instance and uses it from its serial worker queue.
final class NeuralScaler {
    enum Use {
        case preview, photo
        var name: String { self == .preview ? "MuzzlePreviewSR" : "MuzzlePhotoSR" }
        var size: Int { self == .preview ? 48 : 128 }
    }

    private let use: Use
    private let log = Logger(subsystem: "com.muzzlezoom.camera", category: "CoreML")
    private var model: MLModel?
    private var attempted = false
    private var input: CVPixelBuffer?

    init(_ use: Use) { self.use = use }

    @discardableResult
    func prepare() -> Bool {
        if attempted { return model != nil }
        attempted = true
        do {
            guard let url = Bundle.main.url(forResource: use.name, withExtension: "mlmodelc"),
                  let input = ImageTools.pixelBuffer(width: use.size, height: use.size) else {
                throw CameraFailure.model
            }
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            let loaded = try MLModel(contentsOf: url, configuration: configuration)
            let source = loaded.modelDescription.inputDescriptionsByName["image"]?.imageConstraint
            let result = loaded.modelDescription.outputDescriptionsByName["enhanced"]?.imageConstraint
            guard source?.pixelsWide == use.size, source?.pixelsHigh == use.size,
                  result?.pixelsWide == 4 * use.size, result?.pixelsHigh == 4 * use.size else {
                throw CameraFailure.model
            }
            model = loaded
            self.input = input
            return true
        } catch {
            log.error("Model load failed: \(error.localizedDescription, privacy: .public)")
            return false
        }
    }

    private func predict(_ tile: CIImage, context: CIContext) throws -> CIImage {
        guard prepare(), let model, let input else { throw CameraFailure.model }
        context.render(tile, to: input, bounds: tile.extent, colorSpace: ImageTools.sRGB)
        let features = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)])
        let prediction = try model.prediction(from: features)
        guard let pixels = prediction.featureValue(for: "enhanced")?.imageBufferValue else {
            throw CameraFailure.model
        }
        return CIImage(cvPixelBuffer: pixels, options: [.colorSpace: ImageTools.sRGB])
    }

    func live(_ image: CIImage, context: CIContext) throws -> CIImage {
        let side = CGFloat(use.size)
        let factor = side / max(image.extent.width, image.extent.height)
        let width = max(1, floor(image.extent.width * factor))
        let height = max(1, floor(image.extent.height * factor))
        let source = ImageTools.resized(image, width: width, height: height, highQuality: false)
        let tile = source.clampedToExtent().cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
        return try predict(tile, context: context)
            .cropped(to: CGRect(x: 0, y: 0, width: width * 4, height: height * 4))
    }

    func still(_ image: CIImage, context: CIContext) throws -> CIImage {
        // SRVGG uses 34 stacked 3×3 convolutions. Discard 34 pixels of halo
        // so zero-padding inside the model cannot create seams between tiles.
        let halo = 34
        let strideSize = use.size - 2 * halo
        let width = Int(image.extent.width.rounded(.down))
        let height = Int(image.extent.height.rounded(.down))
        let source = ImageTools.normalized(image).clampedToExtent()
        let fullExtent = CGRect(x: 0, y: 0, width: width * 4, height: height * 4)
        var result = CIImage(color: .clear).cropped(to: fullExtent)
        for y in stride(from: 0, to: height, by: strideSize) {
            for x in stride(from: 0, to: width, by: strideSize) {
                let inputRect = CGRect(x: x - halo, y: y - halo, width: use.size, height: use.size)
                let tile = ImageTools.normalized(source.cropped(to: inputRect))
                let predicted = try predict(tile, context: context)
                let center = CGRect(x: halo * 4, y: halo * 4,
                                    width: min(strideSize, width - x) * 4,
                                    height: min(strideSize, height - y) * 4)
                let placed = predicted.cropped(to: center)
                    .transformed(by: CGAffineTransform(translationX: CGFloat((x - halo) * 4),
                                                        y: CGFloat((y - halo) * 4)))
                result = placed.composited(over: result)
            }
        }
        return result.cropped(to: fullExtent)
    }

    static func blend(_ neural: CIImage, with evidence: CIImage, amount: Float) -> CIImage {
        let filter = CIFilter.dissolveTransition()
        filter.inputImage = evidence
        filter.targetImage = ImageTools.resized(neural, width: evidence.extent.width, height: evidence.extent.height)
        filter.time = amount
        return (filter.outputImage ?? evidence).cropped(to: evidence.extent)
    }
}
