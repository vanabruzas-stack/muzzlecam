import AVFoundation
import CoreMedia
import Foundation
import Metal
import QuartzCore

struct SourcePhoto {
    let data: Data
    let dark: Bool
}

struct CaptureTicket {
    let controls: ZoomControls
    let center: CGPoint
    let hardware: CGFloat
    let reduced: Bool
    let frames: [SourcePhoto]
    let sample: TrackingSample?
    let captureNote: String?
}

final class CaptureEngine: NSObject {
    enum State {
        case idle, running, interrupted, failure(String)
    }

    let preview: PreviewPipeline
    let notifications = LockedValue<((State) -> Void)?>(nil)
    private let queue = DispatchQueue(label: "MuzzleZoom.Camera", qos: .userInitiated)
    private let session = AVCaptureSession()
    private let frames = AVCaptureVideoDataOutput()
    private let stills = AVCapturePhotoOutput()
    private var device: AVCaptureDevice?
    private var photoSize: CMVideoDimensions?
    private var hardwareLimit: CGFloat = 1
    private var delegates: [Int64: PhotoDelegate] = [:]
    private var observers: [NSObjectProtocol] = []
    private var pressureObserver: NSKeyValueObservation?
    private var configured = false
    private var wantsRunning = false
    private var capturing = false

    init(device: MTLDevice?) {
        preview = PreviewPipeline(device: device)
        super.init()
        let center = NotificationCenter.default
        observers = [
            center.addObserver(forName: AVCaptureSession.wasInterruptedNotification, object: session, queue: nil) {
                [weak self] _ in self?.queue.async { self?.publish(.interrupted) }
            },
            center.addObserver(forName: AVCaptureSession.interruptionEndedNotification, object: session, queue: nil) {
                [weak self] _ in self?.queue.async { self?.startIfNeeded() }
            },
            center.addObserver(forName: AVCaptureSession.runtimeErrorNotification, object: session, queue: nil) {
                [weak self] note in
                let reset = (note.userInfo?[AVCaptureSessionErrorKey] as? AVError)?.code == .mediaServicesWereReset
                self?.queue.async {
                    guard let self, self.wantsRunning else { return }
                    if reset { self.startIfNeeded() }
                    else { self.publish(.failure("Сбой камеры. Попробуйте запустить её снова.")) }
                }
            }
        ]
    }

    deinit {
        observers.forEach { NotificationCenter.default.removeObserver($0) }
    }

    func setRunning(_ running: Bool) {
        queue.async {
            self.wantsRunning = running
            if running {
                self.startIfNeeded()
            } else {
                if self.session.isRunning { self.session.stopRunning() }
                self.preview.stop()
                self.publish(.idle)
            }
        }
    }

    func update(_ settings: ZoomControls) {
        preview.controls.update { $0 = settings }
        queue.async { self.applyZoom() }
    }

    func focus(at point: CGPoint) {
        queue.async {
            guard let device = self.device, !self.capturing else { return }
            // AVFoundation uses landscape coordinates with origin at the upper left.
            let sensorPoint = CGPoint(x: min(1, max(0, point.y)), y: min(1, max(0, 1 - point.x)))
            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }
                if device.isFocusPointOfInterestSupported, device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusPointOfInterest = sensorPoint
                    device.focusMode = .continuousAutoFocus
                }
                if device.isExposurePointOfInterestSupported, device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposurePointOfInterest = sensorPoint
                    device.exposureMode = .continuousAutoExposure
                }
            } catch { self.publish(.running) }
        }
    }

    func exposure(_ bias: Float) {
        queue.async {
            guard let device = self.device, !self.capturing else { return }
            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }
                device.setExposureTargetBias(min(device.maxExposureTargetBias, max(device.minExposureTargetBias, bias)),
                                             completionHandler: nil)
            } catch { self.publish(.running) }
        }
    }

    func capture(_ controls: ZoomControls) async throws -> CaptureTicket {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                guard self.session.isRunning, !self.session.isInterrupted, !self.capturing,
                      let connection = self.stills.connection(with: .video), connection.isActive else {
                    continuation.resume(throwing: CameraFailure.notReady)
                    return
                }
                self.applyZoom()
                self.capturing = true
                let recent = self.preview.latest.read()
                let hardware = self.preview.hardware.read()
                let recentIsValid = recent.map {
                    abs($0.controls.zoom - controls.zoom) < 0.04 &&
                    $0.controls.trackingReset == controls.trackingReset &&
                    abs($0.hardware - hardware.zoom) < 0.03 &&
                    CACurrentMediaTime() - $0.arrived < 0.4
                } ?? false
                let center = (recentIsValid ? recent?.center : nil) ??
                    (controls.locked ? controls.anchor : CGPoint(x: 0.5, y: 0.5))
                let thermal = ProcessInfo.processInfo.thermalState
                let reduced = hardware.constrained || thermal == .serious || thermal == .critical
                let hdr = !reduced && recentIsValid && recent?.highlights == true &&
                    controls.mode == .quality && abs(self.device?.exposureTargetBias ?? 0) < 0.15 &&
                    self.stills.maxBracketedCapturePhotoCount >= 2
                let count = !reduced && (controls.zoom >= 5 || controls.mode != .quality) ? 3 : 1
                self.sequence(remaining: count, bracket: hdr, collected: [], note: nil) { result in
                    self.capturing = false
                    self.applyZoom()
                    continuation.resume(with: result.map { burst in
                        CaptureTicket(controls: controls, center: center, hardware: hardware.zoom,
                                      reduced: reduced, frames: burst.frames,
                                      sample: recentIsValid && controls.zoom >= 5 ? recent?.sample : nil,
                                      captureNote: burst.note)
                    })
                }
            }
        }
    }

    private func publish(_ state: State) { notifications.read()?(state) }

    private func startIfNeeded() {
        guard wantsRunning else { return }
        do {
            if !configured { try configure() }
            if !session.isRunning { session.startRunning() }
            if session.isRunning && !session.isInterrupted {
                preview.start()
                publish(.running)
            } else {
                publish(.interrupted)
            }
        } catch { publish(.failure(error.localizedDescription)) }
    }

    private func configure() throws {
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        else { throw CameraFailure.unavailable }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }
        if session.canSetSessionPreset(.photo) { session.sessionPreset = .photo }
        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else { throw CameraFailure.unavailable }
        session.addInput(input)

        try camera.lockForConfiguration()
        if let format = bestFormat(for: camera) {
            camera.activeFormat = format
            let frameDuration = CMTime(value: 1, timescale: 30)
            camera.activeVideoMinFrameDuration = frameDuration
            camera.activeVideoMaxFrameDuration = frameDuration
        }
        if camera.activeFormat.supportedColorSpaces.contains(.P3_D65) {
            session.automaticallyConfiguresCaptureDeviceForWideColor = false
            camera.activeColorSpace = .P3_D65
        }
        if camera.isFocusModeSupported(.continuousAutoFocus) { camera.focusMode = .continuousAutoFocus }
        if camera.isExposureModeSupported(.continuousAutoExposure) { camera.exposureMode = .continuousAutoExposure }
        camera.isSubjectAreaChangeMonitoringEnabled = false
        camera.unlockForConfiguration()

        guard session.canAddOutput(stills) else { throw CameraFailure.unavailable }
        session.addOutput(stills)
        stills.maxPhotoQualityPrioritization = .quality
        let largest = camera.activeFormat.supportedMaxPhotoDimensions.max { Self.pixels($0) < Self.pixels($1) }
        if let largest {
            stills.maxPhotoDimensions = largest
            photoSize = largest
        }

        guard session.canAddOutput(frames) else { throw CameraFailure.unavailable }
        session.addOutput(frames)
        frames.alwaysDiscardsLateVideoFrames = true
        frames.automaticallyConfiguresOutputBufferDimensions = false
        frames.deliversPreviewSizedOutputBuffers = true
        let types = frames.availableVideoPixelFormatTypes
        let preferred = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        let fallback = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        if types.contains(preferred) || types.contains(fallback) {
            frames.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                types.contains(preferred) ? preferred : fallback]
        }

        for output in [stills as AVCaptureOutput, frames as AVCaptureOutput] {
            if let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        }
        frames.setSampleBufferDelegate(self, queue: preview.queue)
        device = camera
        let format = camera.activeFormat
        let secondary = format.secondaryNativeResolutionZoomFactors.filter { $0 <= 2.02 }.max() ?? 1
        let nativeLimit = max(format.videoZoomFactorUpscaleThreshold, secondary)
        hardwareLimit = max(1, min(2, camera.maxAvailableVideoZoomFactor, nativeLimit))
        let fieldOfView = CGFloat(camera.activeFormat.videoFieldOfView) * .pi / 180
        preview.hardware.update { $0.fieldOfView = fieldOfView }
        pressureObserver = camera.observe(\.systemPressureState, options: [.initial, .new]) { [weak self] camera, _ in
            let level = camera.systemPressureState.level
            self?.preview.hardware.update {
                $0.constrained = level == .serious || level == .critical || level == .shutdown
            }
        }
        configured = true
        applyZoom()
    }

    private static func pixels(_ size: CMVideoDimensions) -> Int {
        Int(size.width) * Int(size.height)
    }

    private func bestFormat(for device: AVCaptureDevice) -> AVCaptureDevice.Format? {
        let candidates = device.formats.filter { format in
            let size = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let ratio = Double(size.width) / Double(size.height)
            return !format.supportedMaxPhotoDimensions.isEmpty && size.width >= 1280 &&
                size.width <= 4032 && abs(ratio - (4.0 / 3.0)) < 0.05 &&
                format.videoSupportedFrameRateRanges.contains { $0.minFrameRate <= 30 && $0.maxFrameRate >= 30 }
        }
        func rank(_ format: AVCaptureDevice.Format) -> (Int, Int, Int, Int) {
            let photo = format.supportedMaxPhotoDimensions.map(Self.pixels).max() ?? 0
            let video = Self.pixels(CMVideoFormatDescriptionGetDimensions(format.formatDescription))
            return (photo, format.isHighestPhotoQualitySupported ? 1 : 0,
                    format.isHighPhotoQualitySupported ? 1 : 0, min(video, 3_000_000))
        }
        return candidates.max { rank($0) < rank($1) }
    }

    private func applyZoom() {
        guard let device, !capturing else { return }
        let desired = max(device.minAvailableVideoZoomFactor,
                          min(hardwareLimit, device.maxAvailableVideoZoomFactor, CGFloat(preview.controls.read().zoom)))
        guard abs(device.videoZoomFactor - desired) > 0.0002 else {
            preview.hardware.update { $0.zoom = device.videoZoomFactor }
            return
        }
        do {
            try device.lockForConfiguration()
            device.videoZoomFactor = desired
            device.unlockForConfiguration()
            preview.hardware.update { $0.zoom = desired }
        } catch { return }
    }

    private typealias Burst = (frames: [SourcePhoto], note: String?)

    private func sequence(remaining: Int, bracket: Bool, collected: [SourcePhoto], note: String?,
                          completion: @escaping (Result<Burst, Error>) -> Void) {
        guard remaining > 0, wantsRunning, session.isRunning, !session.isInterrupted,
              stills.connection(with: .video)?.isActive == true else {
            if collected.isEmpty { completion(.failure(CameraFailure.interrupted)) }
            else { completion(.success((collected, note ?? "Серия прервалась"))) }
            return
        }
        let codec: AVVideoCodecType = stills.availablePhotoCodecTypes.contains(.hevc) ? .hevc : .jpeg
        let format: [String: Any] = [AVVideoCodecKey: codec]
        let settings: AVCapturePhotoSettings
        let useBracket = bracket && stills.maxBracketedCapturePhotoCount >= 2
        if useBracket {
            let exposures = [Float(0), Float(-1.5)].map {
                AVCaptureAutoExposureBracketedStillImageSettings.autoExposureSettings(exposureTargetBias: $0)
            }
            let pair = AVCapturePhotoBracketSettings(rawPixelFormatType: 0, processedFormat: format,
                                                     bracketedSettings: exposures)
            if stills.isLensStabilizationDuringBracketedCaptureSupported { pair.isLensStabilizationEnabled = true }
            settings = pair
            settings.photoQualityPrioritization = .speed
        } else {
            settings = AVCapturePhotoSettings(format: format)
            settings.photoQualityPrioritization = .quality
            if let photoSize { settings.maxPhotoDimensions = photoSize }
        }

        let id = settings.uniqueID
        let delegate = PhotoDelegate { [weak self] result in
            guard let self else { completion(.failure(CameraFailure.capture)); return }
            self.queue.async {
                self.delegates[id] = nil
                switch result {
                case .success(let photos) where photos.contains(where: { !$0.dark }):
                    if useBracket {
                        completion(.success((photos, photos.count == 2 ? nil : "Доступна одна экспозиция")))
                    } else if remaining == 1 {
                        completion(.success((collected + photos, note)))
                    } else {
                        self.sequence(remaining: remaining - 1, bracket: false, collected: collected + photos,
                                      note: note, completion: completion)
                    }
                default:
                    if useBracket {
                        self.sequence(remaining: remaining, bracket: false, collected: collected,
                                      note: "HDR-серия недоступна", completion: completion)
                    } else if collected.isEmpty {
                        completion(.failure(CameraFailure.capture))
                    } else {
                        completion(.success((collected, "Камера вернула неполную серию")))
                    }
                }
            }
        }
        delegates[id] = delegate
        stills.capturePhoto(with: settings, delegate: delegate)
    }
}

extension CaptureEngine: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let stamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        preview.consume(pixels, timestamp: stamp.isFinite ? stamp : CACurrentMediaTime())
    }
}

private final class PhotoDelegate: NSObject, AVCapturePhotoCaptureDelegate {
    private let completion: (Result<[SourcePhoto], Error>) -> Void
    private var photos: [SourcePhoto] = []

    init(completion: @escaping (Result<[SourcePhoto], Error>) -> Void) {
        self.completion = completion
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        guard error == nil, let data = photo.fileDataRepresentation() else { return }
        let bias = (photo.bracketSettings as? AVCaptureAutoExposureBracketedStillImageSettings)?.exposureTargetBias ?? 0
        photos.append(SourcePhoto(data: data, dark: bias < -0.1))
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings,
                     error: Error?) {
        if let error, photos.isEmpty { completion(.failure(error)) }
        else { completion(.success(photos)) }
    }
}
