import CoreImage
import CoreImage.CIFilterBuiltins
import CoreVideo

enum ImageTools {
    static let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
    static let p3 = CGColorSpace(name: CGColorSpace.displayP3)!

    static func normalized(_ image: CIImage) -> CIImage {
        image.transformed(by: CGAffineTransform(translationX: -image.extent.minX,
                                                y: -image.extent.minY))
    }

    static func resized(_ image: CIImage, width: CGFloat, height: CGFloat,
                        highQuality: Bool = true) -> CIImage {
        let source = normalized(image)
        let sx = width / source.extent.width
        let sy = height / source.extent.height
        let target = CGRect(x: 0, y: 0, width: width, height: height)
        if !highQuality {
            return source.clampedToExtent()
                .transformed(by: CGAffineTransform(scaleX: sx, y: sy)).cropped(to: target)
        }
        let filter = CIFilter.lanczosScaleTransform()
        filter.inputImage = source.clampedToExtent()
        filter.scale = Float(sy)
        filter.aspectRatio = Float(sx / sy)
        return (filter.outputImage ?? source.transformed(by: CGAffineTransform(scaleX: sx, y: sy)))
            .cropped(to: target)
    }

    static func patch(_ image: CIImage, rect: CGRect, size: Int, context: CIContext) -> GrayPatch {
        let patch = resized(image.clampedToExtent().cropped(to: rect), width: CGFloat(size),
                            height: CGFloat(size), highQuality: false)
        var gray = [Float](repeating: 0, count: size * size)
        guard let buffer = pixelBuffer(width: size, height: size) else {
            return GrayPatch(size: size, values: gray)
        }
        context.render(patch, to: buffer, bounds: patch.extent, colorSpace: sRGB)
        guard CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else {
            return GrayPatch(size: size, values: gray)
        }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let bytes = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self) else {
            return GrayPatch(size: size, values: gray)
        }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<size {
            for x in 0..<size {
                let offset = y * rowBytes + x * 4
                gray[y * size + x] = (0.2126 * Float(bytes[offset + 2]) + 0.7152 * Float(bytes[offset + 1])
                    + 0.0722 * Float(bytes[offset])) / 255
            }
        }
        return GrayPatch(size: size, values: gray)
    }

    static func filtered(_ image: CIImage, noise: Float, sharpness: Float) -> CIImage {
        let denoise = CIFilter.noiseReduction()
        denoise.inputImage = image.clampedToExtent()
        denoise.noiseLevel = noise
        denoise.sharpness = 0.32
        let sharpen = CIFilter.sharpenLuminance()
        sharpen.inputImage = denoise.outputImage ?? image
        sharpen.sharpness = sharpness
        return (sharpen.outputImage ?? image).cropped(to: image.extent)
    }

    static func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var result: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                          kCVPixelBufferMetalCompatibilityKey as String: true] as [String: Any]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                        kCVPixelFormatType_32BGRA, attributes as CFDictionary, &result)
        return status == kCVReturnSuccess ? result : nil
    }
}
