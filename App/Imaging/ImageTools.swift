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

    /// Transform the full source before clipping to a fixed output grid.
    /// Cropping first can quantize the origin on narrow, high-zoom windows.
    static func sample(_ image: CIImage, rect: CGRect, width: CGFloat, height: CGFloat) -> CIImage {
        let sx = width / rect.width
        let sy = height / rect.height
        let mapping = CGAffineTransform(a: sx, b: 0, c: 0, d: sy,
                                       tx: -rect.minX * sx, ty: -rect.minY * sy)
        return image.clampedToExtent().transformed(by: mapping)
            .cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
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
        let patch = sample(image, rect: rect, width: CGFloat(size), height: CGFloat(size))
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
