import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import ImageIO
import Metal
import UIKit

struct ProcessedPhoto {
    let data: Data
    let fileName: String
    let thumbnail: UIImage?
    let notes: [String]
}

final class PhotoProcessor {
    private let queue = DispatchQueue(label: "MuzzleZoom.PhotoProcessing", qos: .userInitiated)
    private let context: CIContext
    private let neural = NeuralScaler(.photo)
    private let extremeNeural = NeuralScaler(.extreme)
    private let device: MTLDevice?
    private lazy var fusion = BurstFusion(device: device)

    init(device: MTLDevice?) {
        self.device = device
        context = device.map { CIContext(mtlDevice: $0, options: [.cacheIntermediates: false]) }
            ?? CIContext(options: [.cacheIntermediates: false])
    }

    func develop(_ ticket: CaptureTicket) async throws -> ProcessedPhoto {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                continuation.resume(with: Result {
                    try autoreleasepool { try self.developSync(ticket) }
                })
            }
        }
    }

    private func developSync(_ ticket: CaptureTicket) throws -> ProcessedPhoto {
        defer { context.clearCaches() }
        guard let first = ticket.frames.first(where: { !$0.dark }), let reference = decode(first)
        else { throw CameraFailure.decoding }

        let digital = max(1, CGFloat(ticket.controls.zoom) / ticket.hardware)
        let center = registeredCenter(in: reference, ticket: ticket)
        let nativeShot = ticket.controls.mode == .quality && ticket.controls.zoom <= 1.001
        let crop = nativeShot ? reference.extent : CameraGeometry.crop(
            in: reference.extent, center: center, digital: digital, aspect: CameraGeometry.aspect)
        guard crop.width >= 2, crop.height >= 2 else { throw CameraFailure.processing }

        var notes = [String]()
        if let captureNote = ticket.captureNote { notes.append(captureNote) }
        if ticket.reduced { notes.append("Щадящая обработка из-за нагрева") }
        let gridWidth = max(2, crop.width.rounded())
        let gridHeight = max(2, crop.height.rounded())
        let base = nativeShot ? reference : ImageTools.sample(reference, rect: crop,
                                                               width: gridWidth, height: gridHeight)
        var evidence = base

        let candidates = ticket.frames.filter { $0.dark } + Array(ticket.frames.filter { !$0.dark }.dropFirst())
        var aligned = candidates.compactMap { source -> AlignedExposure? in
            guard let image = decode(source), image.extent.size == reference.extent.size else { return nil }
            return align(image, with: reference, crop: crop, dark: source.dark)
        }
        var anchor = reference
        var anchorCrop = crop
        if let sharpest = aligned.indices.filter({ aligned[$0].darkGain == nil })
            .max(by: { aligned[$0].weight < aligned[$1].weight }),
           aligned[sharpest].weight >= 1.14 {
            let best = aligned.remove(at: sharpest)
            anchor = best.image
            anchorCrop = crop.offsetBy(dx: best.offset.x, dy: best.offset.y)
            aligned = aligned.map {
                AlignedExposure(image: $0.image,
                                offset: CGPoint(x: $0.offset.x - best.offset.x, y: $0.offset.y - best.offset.y),
                                darkGain: $0.darkGain, weight: min(1, $0.weight / best.weight))
            }
            aligned.insert(AlignedExposure(image: reference, offset: CGPoint(x: -best.offset.x, y: -best.offset.y),
                                           darkGain: nil, weight: 1 / best.weight), at: 0)
            evidence = ImageTools.sample(anchor, rect: anchorCrop, width: gridWidth, height: gridHeight)
        }
        if !candidates.isEmpty {
            // Zoomed photos benefit from a denser sampling grid. HDR brackets
            // at low zoom use the native grid within a bounded memory budget.
            let scale = digital >= 2 && crop.height <= 1100 ? 2 : 1
            if let reconstructed = fusion?.reconstruct(anchor, crop: anchorCrop, exposures: aligned,
                                                       scale: scale, context: context) {
                evidence = ImageTools.normalized(reconstructed)
                if aligned.contains(where: { $0.darkGain != nil }) { notes.append("HDR из двух экспозиций") }
                else { notes.append("Совмещено кадров: \(aligned.count + 1)") }
            } else {
                notes.append("Использован один кадр с обработкой")
            }
        }

        let mode = ticket.controls.mode
        let target = CameraGeometry.exportSize(for: crop.size, mode: mode, nativeShot: nativeShot)
        // At 1× the ISP has already processed the full-resolution photo.
        // Keep its original grid and avoid losing fine texture to a second denoise pass.
        var result = evidence
        if !nativeShot {
            let noise: Float = mode == .quality ? 0.011 : (aligned.isEmpty ? 0.024 : 0.012)
            result = ImageTools.filtered(evidence, noise: noise, sharpness: mode == .quality ? 0.19 : 0.11)
        }

        var reconstructedWithAI = false
        if ticket.controls.neural && mode != .quality && !ticket.reduced {
            if isHot {
                notes.append("ИИ приостановлен из-за нагрева")
            } else {
                do {
                    if mode == .extreme, base.extent.height <= 112,
                       let enhanced = try? extremeNeural.restoreCrop(evidence, context: context) {
                        result = NeuralScaler.mergePhotoDetail(enhanced, with: result, amount: 0.76)
                        notes.append("ИИ-реконструкция · x4plus")
                    } else {
                        // Preserve additional samples from burst fusion within a bounded model tile grid.
                        let maxSide: CGFloat = mode == .extreme ? 360 : 300
                        let ratio = min(1, maxSide / max(evidence.extent.width, evidence.extent.height))
                        let w = max(2, floor(evidence.extent.width * ratio))
                        let h = max(2, floor(evidence.extent.height * ratio))
                        let input = ImageTools.resized(evidence, width: w, height: h)
                        let enhanced = try neural.still(input, context: context)
                        result = NeuralScaler.mergePhotoDetail(enhanced, with: result,
                                                               amount: mode == .extreme ? 0.63 : 0.43)
                        notes.append("ИИ-реконструкция")
                    }
                    reconstructedWithAI = true
                } catch {
                    notes.append("ИИ недоступен · снимок обработан без модели")
                }
            }
        }

        let tone = CIFilter.highlightShadowAdjust()
        tone.inputImage = result
        tone.highlightAmount = mode == .quality ? 0.93 : 0.96
        tone.shadowAmount = 0.08
        let color = CIFilter.colorControls()
        color.inputImage = tone.outputImage ?? result
        color.contrast = mode == .quality ? 1.025 : 1.035
        color.saturation = mode == .quality ? 1.015 : 1.01
        guard let graded = color.outputImage?.cropped(to: result.extent) else { throw CameraFailure.processing }
        let output = nativeShot ? graded : ImageTools.resized(graded, width: target.width, height: target.height)
        let data: Data
        let extensionName: String
        let quality = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        if let heif = context.heifRepresentation(of: output, format: .RGBA8, colorSpace: ImageTools.p3,
                                                options: [quality: 0.94]) {
            data = heif
            extensionName = "heic"
        } else if let jpeg = context.jpegRepresentation(of: output, colorSpace: ImageTools.p3,
                                                        options: [quality: 0.96]) {
            data = jpeg
            extensionName = "jpg"
        } else { throw CameraFailure.processing }
        let small = ImageTools.resized(output, width: 144, height: 144 * output.extent.height / output.extent.width)
        let thumbnail = context.createCGImage(small, from: small.extent).map { UIImage(cgImage: $0) }
        let marker = reconstructedWithAI ? "-AI" : ""
        let file = "MuzzleZoom\(marker)-\(UUID().uuidString.prefix(12)).\(extensionName)"
        if mode == .extreme {
            notes.append("Исходный кроп \(Int(gridWidth)) × \(Int(gridHeight))")
        }
        if let source = CGImageSourceCreateWithData(data as CFData, nil),
           let metadata = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
           let w = metadata[kCGImagePropertyPixelWidth] as? Int,
           let h = metadata[kCGImagePropertyPixelHeight] as? Int {
            notes.insert("\(w) × \(h) · " + (Double(w * h) / 1_000_000)
                .formatted(.number.precision(.fractionLength(1))) + " Мп", at: 0)
        }
        return ProcessedPhoto(data: data, fileName: file, thumbnail: thumbnail, notes: notes)
    }

    private var isHot: Bool {
        let thermal = ProcessInfo.processInfo.thermalState
        return thermal == .serious || thermal == .critical
    }

    private func decode(_ source: SourcePhoto) -> CIImage? {
        guard var image = CIImage(data: source.data, options: [.applyOrientationProperty: true]) else { return nil }
        if image.extent.width > image.extent.height { image = image.oriented(.right) }
        return ImageTools.normalized(image)
    }

    private func registeredCenter(in reference: CIImage, ticket: CaptureTicket) -> CGPoint {
        guard let sample = ticket.sample, !ticket.reduced else { return ticket.center }
        let area = CameraGeometry.sourceArea(in: reference.extent)
        let side = sample.widthFraction * area.width
        let rect = CGRect(x: area.minX + sample.center.x * area.width - side / 2,
                          y: area.maxY - sample.center.y * area.height - side / 2,
                          width: side, height: side)
        let current = ImageTools.patch(reference, rect: rect, size: sample.patch.size, context: context)
        guard let match = PatchRegistration.match(sample.patch, current, radius: 10, threshold: 0.91)
        else { return ticket.center }
        let scale = side / CGFloat(sample.patch.size)
        return CGPoint(x: ticket.center.x + CGFloat(match.offset.x) * scale / area.width,
                       y: ticket.center.y + CGFloat(match.offset.y) * scale / area.height)
    }

    private func align(_ image: CIImage, with reference: CIImage, crop: CGRect,
                       dark: Bool) -> AlignedExposure? {
        let span = max(144, min(reference.extent.width * 0.6, crop.width * 1.75))
        let region = CGRect(x: crop.midX - span / 2, y: crop.midY - span / 2, width: span, height: span)
        let first = ImageTools.patch(reference, rect: region, size: 100, context: context)
        let second = ImageTools.patch(image, rect: region, size: 100, context: context)
        guard let rough = PatchRegistration.match(first, second, radius: 14, threshold: dark ? 0.78 : 0.87)
        else { return nil }
        let one = region.width / CGFloat(first.size)
        var shift = CGPoint(x: CGFloat(rough.offset.x) * one, y: -CGFloat(rough.offset.y) * one)

        let fineSpan = max(72, min(190, crop.width))
        let fine = CGRect(x: crop.midX - fineSpan / 2, y: crop.midY - fineSpan / 2,
                          width: fineSpan, height: fineSpan)
        let fineReference = ImageTools.patch(reference, rect: fine, size: 94, context: context)
        let fineCandidate = ImageTools.patch(image, rect: fine.offsetBy(dx: shift.x, dy: shift.y),
                                             size: 94, context: context)
        let fineMatch = PatchRegistration.match(fineReference, fineCandidate, radius: 8,
                                                threshold: dark ? 0.79 : 0.86)
        if let fineMatch {
            shift.x += CGFloat(fineMatch.offset.x) * fineSpan / CGFloat(fineReference.size)
            shift.y -= CGFloat(fineMatch.offset.y) * fineSpan / CGFloat(fineReference.size)
        } else if !dark && one > 2.8 {
            return nil
        }

        let shifted = crop.offsetBy(dx: shift.x, dy: shift.y)
        guard image.extent.contains(shifted) else { return nil }
        let gain = dark ? exposureGain(reference: first, dark: ImageTools.patch(
            image, rect: region.offsetBy(dx: shift.x, dy: shift.y), size: first.size, context: context)) : nil
        if dark && gain == nil { return nil }

        let ratio = fineMatch == nil ? second.sharpness / max(1, first.sharpness)
            : fineCandidate.sharpness / max(1, fineReference.sharpness)
        guard dark || ratio > 0.58 else { return nil }
        let weight = dark ? 1 : min(1.2, max(0.45, ratio))
        return AlignedExposure(image: image, offset: shift, darkGain: gain, weight: weight)
    }

    private func exposureGain(reference: GrayPatch, dark: GrayPatch) -> Float? {
        func linear(_ encoded: Float) -> Float {
            encoded <= 0.04045 ? encoded / 12.92 : pow((encoded + 0.055) / 1.055, 2.4)
        }
        let ranges = zip(reference.values, dark.values).compactMap { normal, dim -> Float? in
            guard normal > 0.27, normal < 0.79, dim > 0.1, dim < normal else { return nil }
            return linear(normal) / linear(dim)
        }.sorted()
        guard ranges.count >= 40 else { return nil }
        return min(3.8, max(1.12, ranges[ranges.count / 2]))
    }
}
