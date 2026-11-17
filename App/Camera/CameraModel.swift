import AVFoundation
import Combine
import Metal
import Photos
import PhotosUI
import SwiftUI
import UIKit
import VideoToolbox

@MainActor
final class CameraModel: ObservableObject {
    @Published private(set) var controls = ZoomControls()
    @Published private(set) var state: CaptureEngine.State = .idle
    @Published private(set) var denied = false
    @Published private(set) var phase: Phase?
    @Published private(set) var tracking = false
    @Published private(set) var previewStatus: PreviewStatus = .standard
    @Published private(set) var focusMark: CGPoint?
    @Published private(set) var exposure: Float = 0
    @Published private(set) var notice: Notice?
    @Published private(set) var thumbnail: UIImage?
    @Published private(set) var hardwareZoom: CGFloat = 1
    @Published private(set) var hasViewfinderFrame = false

    enum Phase: String {
        case capturing = "Съёмка"
        case processing = "Обработка"
        case saving = "Сохранение"
    }

    struct Notice {
        let text: String
        let error: Bool
    }

    let engine: CaptureEngine
    var preview: PreviewPipeline { engine.preview }
    private let processor: PhotoProcessor
    private var foreground = false
    private var requestedPermission = false
    private var noticeTimer: Task<Void, Never>?
    private var focusTimer: Task<Void, Never>?

    init() {
        let device = MTLCreateSystemDefaultDevice()
        engine = CaptureEngine(device: device)
        processor = PhotoProcessor(device: device)
        engine.notifications.update { sink in
            sink = { [weak self] state in
                Task { @MainActor in self?.state = state }
            }
        }
    }

    var canCapture: Bool {
        if case .running = state { return phase == nil && !denied && preview.device != nil && hasViewfinderFrame }
        return false
    }

    var digitalZoom: Double { max(1, controls.zoom / Double(hardwareZoom)) }

    func setForeground(_ active: Bool) {
        foreground = active
        if active { requestCamera() }
        else { engine.setRunning(false) }
    }

    func requestCamera() {
        guard foreground else { return }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            denied = false
            engine.setRunning(true)
        case .notDetermined:
            guard !requestedPermission else { return }
            requestedPermission = true
            Task {
                let allowed = await AVCaptureDevice.requestAccess(for: .video)
                requestedPermission = false
                denied = !allowed
                if allowed && foreground { engine.setRunning(true) }
            }
        default:
            denied = true
        }
    }

    func pollViewfinder() {
        let latest = preview.latest.read()
        if hasViewfinderFrame != (latest != nil) { hasViewfinderFrame = latest != nil }
        guard let latest else { return }
        if tracking != latest.tracking { tracking = latest.tracking }
        if previewStatus != latest.rendering { previewStatus = latest.rendering }
        if hardwareZoom != latest.hardware { hardwareZoom = latest.hardware }
    }

    func select(_ mode: ZoomMode) {
        let zoom = min(mode.range.upperBound, max(mode.range.lowerBound, controls.zoom))
        applyZoom(zoom, mode: mode)
    }

    func setPreset(_ factor: Double) { applyZoom(factor, mode: ZoomMode.forPreset(factor)) }

    func slide(to factor: Double) { applyZoom(factor, mode: controls.mode) }

    private func applyZoom(_ factor: Double, mode: ZoomMode) {
        guard phase != .capturing else { return }
        let value = min(mode.range.upperBound, max(mode.range.lowerBound, factor))
        guard controls.mode != mode || abs(controls.zoom - value) > 0.001 else { return }
        if value < ZoomControls.minimumLockZoom && controls.locked {
            controls.locked = false
            controls.trackingReset += 1
        }
        controls.mode = mode
        controls.zoom = value
        engine.update(controls)
    }

    func setNeural(_ enabled: Bool) {
        guard phase != .capturing else { return }
        controls.neural = enabled
        engine.update(controls)
    }

    func toggleLock() {
        guard controls.zoom >= ZoomControls.minimumLockZoom, phase != .capturing else { return }
        controls.locked.toggle()
        controls.anchor = preview.latest.read()?.center ?? CGPoint(x: 0.5, y: 0.5)
        controls.trackingReset += 1
        engine.update(controls)
    }

    func focus(at unitPoint: CGPoint) {
        guard let frame = preview.latest.read(), phase != .capturing else { return }
        let point = CameraGeometry.point(in: frame.crop, at: unitPoint, in: frame.sourceArea)
        let actual = CameraGeometry.point(in: frame.crop, at: unitPoint, in: frame.sourceExtent)
        let uncropped = CGPoint(x: 0.5 + (actual.x - 0.5) / frame.hardware,
                                y: 0.5 + (actual.y - 0.5) / frame.hardware)
        engine.focus(at: uncropped)
        if controls.locked {
            controls.anchor = point
            controls.trackingReset += 1
            engine.update(controls)
        }
        focusTimer?.cancel()
        focusMark = unitPoint
        focusTimer = Task {
            try? await Task.sleep(for: .milliseconds(850))
            if !Task.isCancelled { focusMark = nil }
        }
    }

    func setExposure(_ value: Float) {
        exposure = value
        engine.exposure(value)
    }

    func shutter() {
        guard canCapture else { return }
        let snapshot = controls
        phase = .capturing
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        Task {
            defer { phase = nil }
            do {
                let authorization = PHPhotoLibrary.authorizationStatus(for: .addOnly)
                let permission = authorization == .notDetermined
                    ? await PHPhotoLibrary.requestAuthorization(for: .addOnly) : authorization
                guard permission == .authorized || permission == .limited else { throw CameraFailure.saveDenied }
                let frames = try await engine.capture(snapshot)
                phase = .processing
                let processed = try await processor.develop(frames)
                phase = .saving
                try await save(processed)
                thumbnail = processed.thumbnail
                let detail = processed.notes.isEmpty ? "" : "\n" + processed.notes.joined(separator: " · ")
                show("Сохранено в «Фото»" + detail)
            } catch {
                show(error.localizedDescription, error: true)
            }
        }
    }

    private func save(_ photo: ProcessedPhoto) async throws {
        do {
            try await PHPhotoLibrary.shared().performChanges {
                let create = PHAssetCreationRequest.forAsset()
                let options = PHAssetResourceCreationOptions()
                options.originalFilename = photo.fileName
                create.addResource(with: .photo, data: photo.data, options: options)
            }
        } catch { throw CameraFailure.saving }
    }

    func show(_ message: String, error: Bool = false) {
        noticeTimer?.cancel()
        notice = Notice(text: message, error: error)
        noticeTimer = Task {
            try? await Task.sleep(for: .seconds(error ? 6 : 4))
            if !Task.isCancelled { notice = nil }
        }
    }

    nonisolated static func videoToolboxSummary() -> String {
        guard VTLowLatencySuperResolutionScalerConfiguration.isSupported else {
            return "VideoToolbox SR недоступен"
        }
        let factors = VTLowLatencySuperResolutionScalerConfiguration.supportedScaleFactors(frameWidth: 640, frameHeight: 480)
        if factors.isEmpty { return "VideoToolbox SR недоступен для 640 × 480" }
        let scales = factors.map { String(format: "%g×", $0) }.joined(separator: ", ")
        return "VideoToolbox SR для 640 × 480: " + scales
    }
}
