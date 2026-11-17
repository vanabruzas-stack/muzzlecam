import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
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
        let crop = CameraGeometry.crop(in: reference.extent, center: center, digital: digital,
                                       aspect: ticket.controls.aspect)
        guard crop.width >= 2, crop.height >= 2 else { throw CameraFailure.processing }

        var notes = [String]()
        if let captureNote = ticket.captureNote { notes.append(captureNote) }
        if ticket.reduced { notes.append("Щадящая обработка из-за нагрева") }
        let base = ImageTools.normalized(reference.cropped(to: crop))
        var evidence = base

        let candidates = ticket.frames.filter { $0.dark } + Array(ticket.frames.filter { !$0.dark }.dropFirst())
        let aligned = candidates.compactMap { source -> AlignedExposure? in
            guard let image = decode(source), image.extent.size == reference.extent.size else { return nil }
            return align(image, with: reference, crop: crop, dark: source.dark)
        }
        if !candidates.isEmpty {
            // Zoomed photos benefit from a denser sampling grid. HDR brackets
            // at low zoom use the native grid within a bounded memory budget.
            let scale = digital >= 2 && crop.height <= 1100 ? 2 : 1
            if let reconstructed = fusion?.reconstruct(reference, crop: crop, exposures: aligned,
                                                       scale: scale, context: context) {
                evidence = ImageTools.normalized(reconstructed)
                if aligned.contains(where: { $0.darkGain != nil }) { notes.append("HDR из двух экспозиций") }
                else { notes.append("Совмещено кадров: \(aligned.count + 1)") }
            } else {
                notes.append("Использован один кадр с обработкой")
            }
        }

        let referenceHeight = base.extent.height
        let mode = ticket.controls.mode
        let cap: CGFloat = mode == .quality ? (ticket.reduced ? 3456 : 5760) :
            (mode == .superResolution ? 3072 : 1792)
        let targetHeight = max(2, min(cap, max(referenceHeight, mode == .quality && digital < 2 ? 0 : 1280)))
            .rounded(.down)
        let targetWidth = max(2, floor(targetHeight * crop.width / crop.height / 2) * 2)
        let evidenceInput = ImageTools.filtered(evidence, noise: aligned.isEmpty ? 0.024 : 0.012,
                                                sharpness: mode == .quality ? 0.21 : 0.11)
        var result = ImageTools.resized(evidenceInput, width: targetWidth, height: targetHeight)

        if ticket.controls.neural && mode != .quality && !ticket.reduced {
            if isHot {
                notes.append("ИИ приостановлен из-за нагрева")
            } else {
                do {
                    // Limit inference to a bounded tile grid. The layer is blended
                    // with the independently reconstructed camera evidence.
                    let maxSide: CGFloat = 300
                    let ratio = min(1, maxSide / max(base.extent.width, base.extent.height))
                    let w = max(2, floor(base.extent.width * ratio))
                    let h = max(2, floor(base.extent.height * ratio))
                    let srInput = ImageTools.resized(evidence, width: w, height: h)
                    let sr = try neural.still(srInput, context: context)
                    result = NeuralScaler.blend(sr, with: result, amount: mode == .extreme ? 0.16 : 0.36)
                    notes.append("ИИ-реконструкция")
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
        guard let output = color.outputImage?.cropped(to: result.extent) else { throw CameraFailure.processing }
        let data: Data
        let extensionName: String
        if let heif = context.heifRepresentation(of: output, format: .RGBA8, colorSpace: ImageTools.p3,
                                                options: [.lossyCompressionQuality: 0.94]) {
            data = heif
            extensionName = "heic"
        } else if let jpeg = context.jpegRepresentation(of: output, colorSpace: ImageTools.p3,
                                                        options: [.lossyCompressionQuality: 0.96]) {
            data = jpeg
            extensionName = "jpg"
        } else { throw CameraFailure.processing }
        let small = ImageTools.resized(output, width: 144, height: 144 * output.extent.height / output.extent.width)
        let thumbnail = context.createCGImage(small, from: small.extent).map { UIImage(cgImage: $0) }
        let file = "MuzzleZoom-\(UUID().uuidString.prefix(12)).\(extensionName)"
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
        guard let match = PatchRegistration.match(sample.patch, current, radius: 10, threshold: 0.87)
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

        let ratio = second.sharpness / max(1, first.sharpness)
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
