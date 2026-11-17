import CoreImage
import Metal
import simd

struct AlignedExposure {
    let image: CIImage
    let offset: CGPoint
    let darkGain: Float?
    let weight: Float
}

final class BurstFusion {
    private struct Geometry {
        var origin: SIMD2<Float>
        var samplesPerPixel: SIMD2<Float>
    }

    private let device: MTLDevice
    private let commands: MTLCommandQueue
    private let reconstruction: MTLComputePipelineState

    init?(device: MTLDevice?) {
        guard let device, let commands = device.makeCommandQueue(),
              let function = device.makeDefaultLibrary()?.makeFunction(name: "reconstructBurst"),
              let reconstruction = try? device.makeComputePipelineState(function: function) else { return nil }
        self.device = device
        self.commands = commands
        self.reconstruction = reconstruction
    }

    func reconstruct(_ reference: CIImage, crop: CGRect, exposures: [AlignedExposure],
                     scale: Int, context: CIContext) -> CIImage? {
        guard !exposures.isEmpty else { return nil }
        let upload = crop.insetBy(dx: -3, dy: -3).integral
        let width = Int(crop.width * CGFloat(scale))
        let height = Int(crop.height * CGFloat(scale))
        guard width > 0, height > 0, width * height <= 12_000_000,
              let command = commands.makeCommandBuffer(),
              let main = texture(width: Int(upload.width), height: Int(upload.height)),
              let output = texture(width: width, height: height) else { return nil }

        func render(_ image: CIImage, rect: CGRect, into texture: MTLTexture) {
            let source = ImageTools.normalized(image.clampedToExtent().cropped(to: rect))
            context.render(source, to: texture, commandBuffer: command,
                           bounds: source.extent, colorSpace: ImageTools.p3)
        }
        render(reference, rect: upload, into: main)
        var inputs = [main, main]
        var parameters = [SIMD4<Float>(repeating: 0), SIMD4<Float>(repeating: 0)]
        for (index, exposure) in exposures.prefix(2).enumerated() {
            guard let texture = self.texture(width: main.width, height: main.height) else { return nil }
            let x = exposure.offset.x.rounded()
            let y = exposure.offset.y.rounded()
            render(exposure.image, rect: upload.offsetBy(dx: x, dy: y), into: texture)
            inputs[index] = texture
            parameters[index] = SIMD4(Float(exposure.offset.x - x), Float(y - exposure.offset.y),
                                      exposure.darkGain ?? exposure.weight, exposure.darkGain == nil ? 1 : 2)
        }

        guard let encoder = command.makeComputeCommandEncoder() else { return nil }
        var geometry = Geometry(origin: SIMD2(Float(crop.minX - upload.minX), Float(upload.maxY - crop.maxY)),
                                samplesPerPixel: SIMD2(Float(crop.width) / Float(width),
                                                       Float(crop.height) / Float(height)))
        encoder.setComputePipelineState(reconstruction)
        encoder.setTexture(main, index: 0)
        encoder.setTexture(inputs[0], index: 1)
        encoder.setTexture(inputs[1], index: 2)
        encoder.setTexture(output, index: 3)
        encoder.setBytes(&geometry, length: MemoryLayout<Geometry>.stride, index: 0)
        parameters.withUnsafeBytes {
            if let address = $0.baseAddress { encoder.setBytes(address, length: $0.count, index: 1) }
        }
        let threads = reconstruction.threadExecutionWidth
        let rows = max(1, min(8, reconstruction.maxTotalThreadsPerThreadgroup / threads))
        encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: threads, height: rows, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { return nil }
        return CIImage(mtlTexture: output, options: [.colorSpace: ImageTools.p3])
    }

    private func texture(width: Int, height: Int) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                                  width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        descriptor.storageMode = .private
        return device.makeTexture(descriptor: descriptor)
    }
}
