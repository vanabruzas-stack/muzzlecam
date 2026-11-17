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
    var locked = false
    var anchor = CGPoint(x: 0.5, y: 0.5)
    var trackingReset = 0
}

struct CameraGeometry {
    static let aspect: CGFloat = 0.75

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

    static func stabilizedCenter(_ center: CGPoint, toward anchor: CGPoint, movement: CGPoint,
                                 digital: CGFloat, elapsed: Double, followTime: Double) -> CGPoint {
        let frameFraction = 0.22 / max(digital, 1)
        let strength = min(1, frameFraction / max(0.00001, hypot(movement.x, movement.y)))
        let tracked = CGPoint(x: center.x + movement.x * strength, y: center.y + movement.y * strength)
        let recenter = 1 - exp(-max(0, elapsed) / followTime)
        return bounded(CGPoint(x: tracked.x + (anchor.x - tracked.x) * recenter,
                               y: tracked.y + (anchor.y - tracked.y) * recenter), digital: digital)
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
