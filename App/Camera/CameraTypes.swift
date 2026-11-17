import CoreGraphics
import Foundation

enum ZoomMode: Int, CaseIterable, Identifiable {
    case quality, superResolution, extreme

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .quality: "Quality"
        case .superResolution: "AI Super"
        case .extreme: "Extreme"
        }
    }

    var range: ClosedRange<Double> {
        switch self {
        case .quality: 1...10
        case .superResolution: 10...30
        case .extreme: 30...200
        }
    }

    static func forPreset(_ factor: Double) -> Self {
        if factor <= 10 { return .quality }
        if factor <= 30 { return .superResolution }
        return .extreme
    }
}

struct ZoomControls {
    static let minimumLockZoom = 2.05

    var mode: ZoomMode = .quality
    var zoom = 1.0
    var neural = true
    var softwareStabilization = true
    var locked = false
    var anchor = CGPoint(x: 0.5, y: 0.5)
    var trackingReset = 0
}

struct CameraGeometry {
    static let aspect: CGFloat = 0.75
    static let exportLongEdge: CGFloat = 4096

    /// Coordinates are normalized in the common, upright 3:4 source area.
    static func sourceArea(in extent: CGRect) -> CGRect {
        let width = min(extent.width, extent.height * aspect)
        let height = width / aspect
        return CGRect(x: extent.midX - width / 2, y: extent.midY - height / 2,
                      width: width, height: height)
    }

    static func bounded(_ center: CGPoint, digital: CGFloat) -> CGPoint {
        let half = 0.5 / max(1, digital)
        return CGPoint(x: min(1 - half, max(half, center.x)),
                       y: min(1 - half, max(half, center.y)))
    }

    static func crop(in extent: CGRect, center: CGPoint, digital: CGFloat, aspect: CGFloat) -> CGRect {
        let area = sourceArea(in: extent)
        let center = bounded(center, digital: digital)
        let width = min(area.width, area.height * aspect) / max(1, digital)
        let height = min(area.height, area.width / aspect) / max(1, digital)
        return CGRect(x: area.minX + center.x * area.width - width / 2,
                      y: area.maxY - center.y * area.height - height / 2,
                      width: width, height: height)
    }

    static func point(in crop: CGRect, at position: CGPoint, in area: CGRect) -> CGPoint {
        let x = crop.minX + position.x * crop.width
        let y = crop.maxY - position.y * crop.height
        return CGPoint(x: (x - area.minX) / area.width, y: (area.maxY - y) / area.height)
    }

    static func exportSize(for crop: CGSize, mode: ZoomMode, nativeShot: Bool) -> CGSize {
        if nativeShot { return crop }
        let height = mode == .quality ? max(exportLongEdge, crop.height.rounded(.down)) : exportLongEdge
        let evenHeight = max(2, floor(height / 2) * 2)
        let evenWidth = max(2, floor(evenHeight * aspect / 2) * 2)
        return CGSize(width: evenWidth, height: evenHeight)
    }
}

struct ReferenceCropTracker {
    private(set) var center = CGPoint(x: 0.5, y: 0.5)
    private var pending: CGPoint?

    mutating func reset(to point: CGPoint, digital: CGFloat) {
        center = CameraGeometry.bounded(point, digital: digital)
        pending = nil
    }

    /// The observation is an absolute crop center in the current sensor image.
    /// A missed match leaves the composition in place. A large change must
    /// agree with the next frame before it can move the visible window.
    @discardableResult
    mutating func update(estimate: CGPoint?, digital: CGFloat, sourceWidth: CGFloat) -> Bool {
        center = CameraGeometry.bounded(center, digital: digital)
        guard let estimate else { pending = nil; return false }
        let target = CameraGeometry.bounded(estimate, digital: digital)
        let frameWidth = 1 / max(1, digital)
        guard hypot(estimate.x - target.x, estimate.y - target.y) <= frameWidth * 0.12 else {
            pending = nil
            return false
        }
        let dx = target.x - center.x
        let dy = target.y - center.y
        let distance = hypot(dx, dy)
        if distance > frameWidth * 0.24 {
            let confirmed = pending.map { hypot(target.x - $0.x, target.y - $0.y) <= frameWidth * 0.18 } ?? false
            pending = target
            guard confirmed else { return false }
        } else {
            pending = nil
        }
        let deadZone = 0.20 / max(1, sourceWidth)
        if distance > deadZone {
            let gain = min(0.90, frameWidth * 0.24 / distance)
            center = CGPoint(x: center.x + gain * dx, y: center.y + gain * dy)
        }
        return true
    }

    mutating func pan(toward point: CGPoint, digital: CGFloat, elapsed: Double) {
        let blend = 1 - exp(-elapsed / 0.18)
        reset(to: CGPoint(x: center.x + (point.x - center.x) * blend,
                          y: center.y + (point.y - center.y) * blend), digital: digital)
    }
}

enum CameraFailure: LocalizedError {
    case unavailable, notReady, capture, decoding, processing, saveDenied, saving, model, interrupted

    var errorDescription: String? {
        switch self {
        case .unavailable: "Основная камера недоступна."
        case .notReady: "Камера ещё готовится к съёмке."
        case .capture: "Камера не вернула снимок."
        case .decoding: "Не удалось прочитать кадр камеры."
        case .processing: "Не удалось обработать снимок. Фото не сохранено."
        case .saveDenied: "Разрешите добавление снимков в «Фото» в настройках iOS."
        case .saving: "Не удалось добавить обработанный снимок в «Фото»."
        case .model: "Модель Core ML недоступна. Используется обработка без ИИ."
        case .interrupted: "Камера временно недоступна."
        }
    }
}
