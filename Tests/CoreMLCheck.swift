import CoreML
import CoreVideo
import Foundation

@main
struct CoreMLCheck {
    static func main() throws {
        let app = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        for (name, side) in [("MuzzlePreviewSR", 48), ("MuzzlePhotoSR", 128)] {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly
            let model = try MLModel(contentsOf: app.appendingPathComponent(name + ".mlmodelc"), configuration: config)
            var buffer: CVPixelBuffer?
            precondition(CVPixelBufferCreate(nil, side, side, kCVPixelFormatType_32BGRA, nil, &buffer) == kCVReturnSuccess)
            let input = buffer!
            CVPixelBufferLockBaseAddress(input, [])
            let pixels = CVPixelBufferGetBaseAddress(input)!.assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(input)
            for y in 0..<side {
                for x in 0..<side {
                    let offset = y * stride + x * 4
                    let value = UInt8(20 + (x + y) * 100 / (side - 1))
                    pixels[offset] = value
                    pixels[offset + 1] = value
                    pixels[offset + 2] = value
                    pixels[offset + 3] = 255
                }
            }
            CVPixelBufferUnlockBaseAddress(input, [])
            let features = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)])
            let result = try model.prediction(from: features).featureValue(for: "enhanced")!.imageBufferValue!
            precondition(CVPixelBufferGetWidth(result) == side * 4 && CVPixelBufferGetHeight(result) == side * 4)
            precondition(CVPixelBufferGetPixelFormatType(result) == kCVPixelFormatType_32BGRA)
            CVPixelBufferLockBaseAddress(result, .readOnly)
            let output = CVPixelBufferGetBaseAddress(result)!.assumingMemoryBound(to: UInt8.self)
            let rowBytes = CVPixelBufferGetBytesPerRow(result)
            let dark = output[side * rowBytes + side * 4]
            let bright = output[3 * side * rowBytes + 3 * side * 4]
            CVPixelBufferUnlockBaseAddress(result, .readOnly)
            precondition(Int(bright) - Int(dark) > 50, "Unexpected neural output for \(name)")
            print("\(name): \(side) → \(side * 4), inference OK")
        }
    }
}
