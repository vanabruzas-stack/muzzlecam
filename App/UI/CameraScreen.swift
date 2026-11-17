import Combine
import ImageIO
import PhotosUI
import SwiftUI
import UIKit

struct CameraScreen: View {
    @StateObject private var model = CameraModel()
    @Environment(\.scenePhase) private var scene
    @State private var picker = false
    @State private var selectedPhoto: PhotosPickerItem?
    @State private var viewedPhoto: ViewedPhoto?
    @State private var exposurePanel = false
    @State private var info = false
    @State private var pinchBase: Double?
    @State private var vtSupport: String?
    @State private var pickerTask: Task<Void, Never>?

    private let accent = Color(red: 0.84, green: 0.98, blue: 0.42)
    private let presets: [Int] = [1, 2, 5, 10, 30, 50, 100, 120, 150, 200]
    private let clock = Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()

    var body: some View {
        ZStack {
            VStack(spacing: 0) {
                header
                    .padding(.horizontal, 18)
                cameraCanvas
                    .frame(maxHeight: .infinity)
                if let notice = model.notice { message(notice).padding(.horizontal, 18) }
                if exposurePanel { brightness.padding(.horizontal, 18) }
                controls
                    .padding(.horizontal, 18)
            }
            .padding(.top, 9)
            .padding(.bottom, 7)
            statusOverlay
        }
        .background(.black)
        .statusBarHidden(true)
        .persistentSystemOverlays(.hidden)
        .preferredColorScheme(.dark)
        .onAppear { model.setForeground(scene == .active) }
        .onChange(of: scene) { _, phase in
            if phase != .inactive { model.setForeground(phase == .active) }
        }
        .onReceive(clock) { _ in model.pollViewfinder() }
        .photosPicker(isPresented: $picker, selection: $selectedPhoto, matching: .images)
        .onChange(of: selectedPhoto) { _, item in
            guard let item else { return }
            pickerTask?.cancel()
            pickerTask = Task {
                do {
                    guard let data = try await item.loadTransferable(type: Data.self) else { throw CameraFailure.decoding }
                    let image = await Task.detached(priority: .userInitiated) { Self.galleryImage(data) }.value
                    guard !Task.isCancelled else { return }
                    guard let image else { throw CameraFailure.decoding }
                    viewedPhoto = ViewedPhoto(image: image)
                } catch {
                    if !Task.isCancelled { model.show("Не удалось открыть фото.", error: true) }
                }
                if !Task.isCancelled { selectedPhoto = nil }
            }
        }
        .sheet(isPresented: $info) { information }
        .fullScreenCover(item: $viewedPhoto) { item in
            NavigationStack {
                Image(uiImage: item.image)
                    .resizable().scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.black)
                    .accessibilityLabel("Выбранная фотография")
                    .navigationTitle("Фото")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button("Готово") { viewedPhoto = nil }
                                .accessibilityLabel("Закрыть фотографию")
                        }
                    }
                    .toolbarBackground(.visible, for: .navigationBar)
                }
                .statusBarHidden(false)
                .persistentSystemOverlays(.visible)
        }
    }

    private var cameraCanvas: some View {
        GeometryReader { geometry in
            let width = max(1, min(geometry.size.width, geometry.size.height * CameraGeometry.aspect))
            let size = CGSize(width: width, height: width / CameraGeometry.aspect)
            ZStack {
                Viewfinder(pipeline: model.preview)
                if model.controls.zoom >= ZoomControls.minimumLockZoom || model.focusMark != nil {
                    let mark = model.controls.locked ? CGPoint(x: 0.5, y: 0.5)
                        : (model.focusMark ?? CGPoint(x: 0.5, y: 0.5))
                    Reticle(locked: model.controls.locked, strong: model.focusMark != nil)
                        .stroke(model.controls.locked || model.focusMark != nil ? accent : .white.opacity(0.57),
                                style: StrokeStyle(lineWidth: 1.15, lineCap: .round))
                        .frame(width: model.controls.locked ? 110 : 62, height: model.controls.locked ? 110 : 62)
                        .position(x: size.width * mark.x, y: size.height * mark.y)
                        .allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
            }
            .frame(width: size.width, height: size.height)
            .contentShape(Rectangle())
            .gesture(SpatialTapGesture().onEnded { value in
                model.focus(at: CGPoint(x: value.location.x / size.width, y: value.location.y / size.height))
            })
            .simultaneousGesture(MagnifyGesture().onChanged { value in
                let base = pinchBase ?? model.controls.zoom
                pinchBase = base
                model.setPreset(min(200, max(1, base * value.magnification)))
            }.onEnded { _ in pinchBase = nil })
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Muzzle Zoom").font(.system(size: 26, weight: .bold, design: .default)).tracking(-1.1)
                HStack(spacing: 5) {
                    Circle().fill(model.tracking ? accent : .white.opacity(0.48)).frame(width: 6, height: 6)
                    Text(stabilizationLabel).font(.system(size: 10, weight: .semibold, design: .monospaced))
                }
                .foregroundStyle(.white.opacity(0.82))
                .accessibilityElement(children: .combine)
            }
            Spacer()
            icon("info", label: "О режимах и устройстве", active: false) { info = true }
            icon("hand.raised", label: model.controls.softwareStabilization
                 ? "Отключить программную стабилизацию" : "Включить программную стабилизацию",
                 active: model.controls.softwareStabilization) { model.toggleStabilization() }
            icon("plus.forwardslash.minus", label: "Компенсация экспозиции", active: exposurePanel) {
                withAnimation(.easeInOut(duration: 0.18)) { exposurePanel.toggle() }
            }
        }
        .foregroundStyle(.white)
    }

    private var stabilizationLabel: String {
        if !model.controls.softwareStabilization { return "ПРОГРАММНАЯ СТАБ. ВЫКЛ." }
        if model.controls.locked { return model.tracking ? "ZOOM LOCK · ЗАХВАТ" : "ZOOM LOCK · ПОИСК" }
        if model.tracking { return "СТАБИЛИЗАЦИЯ · АКТИВНА" }
        if model.controls.zoom >= ZoomControls.minimumLockZoom { return "СТАБИЛИЗАЦИЯ · ПОИСК" }
        return "ОСНОВНАЯ КАМЕРА"
    }

    private var controls: some View {
        VStack(spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(zoomLabel)
                    .font(.system(size: 37, weight: .semibold, design: .monospaced))
                    .tracking(-3).foregroundStyle(accent)
                    .contentTransition(.numericText())
                    .accessibilityLabel("Увеличение " + zoomLabel)
                Spacer()
                if let phase = model.phase {
                    ProgressView().controlSize(.mini).tint(accent)
                    Text(phase.rawValue).font(.caption.weight(.medium))
                } else {
                    Text(imageLabel).font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .tracking(0.6).foregroundStyle(.white.opacity(0.78))
                }
            }
            .padding(.horizontal, 4)
            slider
            presetStrip
            modeStrip
            ZStack {
                shutterButton
                HStack(spacing: 5) {
                    Button {
                        picker = true
                    } label: {
                        ZStack {
                            Circle().fill(.black.opacity(0.52))
                            if let image = model.thumbnail {
                                Image(uiImage: image).resizable().scaledToFill()
                                    .frame(width: 44, height: 44).clipShape(Circle())
                            } else {
                                Image(systemName: "photo.on.rectangle").font(.system(size: 17, weight: .semibold))
                            }
                        }
                        .frame(width: 50, height: 50)
                        .overlay(Circle().strokeBorder(.white.opacity(0.28), lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Открыть недавние фотографии")
                    Spacer(minLength: 85)
                    icon("scope", label: model.controls.locked ? "Отключить Zoom Lock" : "Включить Zoom Lock",
                         active: model.controls.locked) { model.toggleLock() }
                        .disabled(!model.controls.softwareStabilization || model.controls.zoom < ZoomControls.minimumLockZoom)
                        .opacity(model.controls.softwareStabilization && model.controls.zoom >= ZoomControls.minimumLockZoom ? 1 : 0.46)
                    icon("sparkles", label: model.controls.neural ? "Отключить ИИ" : "Включить ИИ",
                         active: model.controls.neural && model.controls.mode != .quality) {
                        model.setNeural(!model.controls.neural)
                    }
                    .disabled(model.controls.mode == .quality)
                    .opacity(model.controls.mode == .quality ? 0.46 : 1)
                }
            }
        }
        .padding(12)
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 25))
        .overlay(RoundedRectangle(cornerRadius: 25).strokeBorder(.white.opacity(0.13), lineWidth: 1))
    }

    private var shutterButton: some View {
        Button { model.shutter() } label: {
            ZStack {
                Circle().strokeBorder(model.canCapture ? accent : .white.opacity(0.5), lineWidth: 2)
                Circle().fill(model.canCapture ? .white : .white.opacity(0.35)).padding(6)
            }
            .frame(width: 64, height: 64)
        }
        .buttonStyle(.plain)
        .disabled(!model.canCapture)
        .accessibilityLabel("Сделать снимок")
    }

    private var slider: some View {
        let range = model.controls.mode.range
        return Slider(value: Binding(get: { log(model.controls.zoom) },
                                    set: { model.slide(to: exp($0)) }),
                      in: log(range.lowerBound)...log(range.upperBound))
            .tint(accent)
            .accessibilityLabel("Увеличение")
            .accessibilityValue(zoomLabel)
    }

    private var presetStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(presets, id: \.self) { value in
                    let selected = abs(model.controls.zoom - Double(value)) < 0.05
                    Button { model.setPreset(Double(value)) } label: {
                        Text("\(value)×").font(.system(size: 12, weight: .semibold, design: .monospaced))
                            .padding(.horizontal, 10).frame(height: 40)
                            .foregroundStyle(selected ? .black : .white)
                            .background(selected ? accent : .white.opacity(0.09), in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(selected ? .isSelected : [])
                }
            }
        }
    }

    private var modeStrip: some View {
        HStack(spacing: 3) {
            ForEach(ZoomMode.allCases) { mode in
                let selected = mode == model.controls.mode
                Button { model.select(mode) } label: {
                    Text(mode.title)
                        .font(.system(size: 13, weight: selected ? .bold : .medium))
                        .lineLimit(1)
                        .frame(maxWidth: .infinity, minHeight: 43)
                        .foregroundStyle(selected ? .white : .white.opacity(0.69))
                        .background(selected ? .white.opacity(0.16) : .clear,
                                    in: RoundedRectangle(cornerRadius: 11))
                }
                .buttonStyle(.plain)
                .accessibilityLabel(mode.title + " \(Int(mode.range.lowerBound))–\(Int(mode.range.upperBound))×")
                .accessibilityAddTraits(selected ? .isSelected : [])
            }
        }
        .padding(3)
        .background(.black.opacity(0.34), in: RoundedRectangle(cornerRadius: 14))
    }

    private var brightness: some View {
        HStack(spacing: 12) {
            Button { model.setExposure(0) } label: {
                Text(String(format: "%+.1f EV", model.exposure))
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .frame(minWidth: 72, minHeight: 40)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Сбросить экспозицию")
            Slider(value: Binding(get: { model.exposure }, set: { model.setExposure($0) }), in: -2...2, step: 0.1)
                .tint(accent).accessibilityLabel("Экспозиция")
        }
        .disabled(model.phase == .capturing)
        .padding(.horizontal, 12).padding(.vertical, 4)
        .background(.black.opacity(0.76), in: RoundedRectangle(cornerRadius: 17))
        .padding(.bottom, 6)
    }

    @ViewBuilder
    private var statusOverlay: some View {
        if model.denied {
            statusCard("Доступ к камере", detail: "Разрешите доступ в настройках iOS.",
                       button: "Открыть настройки") {
                if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
            }
        } else {
            switch model.state {
            case .running:
                if model.preview.device == nil {
                    statusCard("Metal недоступен", detail: "Для видоискателя нужно устройство с Metal.",
                               button: nil, action: {})
                } else if !model.hasViewfinderFrame {
                    ProgressView().tint(accent).accessibilityLabel("Подготовка видоискателя")
                }
            case .idle:
                ProgressView().tint(accent).accessibilityLabel("Запуск камеры")
            case .interrupted:
                statusCard("Камера приостановлена", detail: "Вернитесь в кадр, когда камера освободится.",
                           button: "Повторить", action: model.requestCamera)
            case .failure(let detail):
                statusCard("Камера недоступна", detail: detail, button: "Повторить", action: model.requestCamera)
            }
        }
    }

    private func statusCard(_ title: String, detail: String, button: String?, action: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title).font(.headline)
            Text(detail).font(.subheadline).foregroundStyle(.secondary)
            if let button {
                Button(button, action: action).buttonStyle(.borderedProminent).tint(accent).foregroundStyle(.black)
            }
        }
        .padding(19)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 22))
        .padding(.horizontal, 28)
    }

    private func icon(_ symbol: String, label: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(active ? accent : .white)
                .frame(width: 44, height: 44)
                .background(.black.opacity(0.61), in: Circle())
                .overlay(Circle().strokeBorder((active ? accent : .white).opacity(0.36), lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }

    private func message(_ notice: CameraModel.Notice) -> some View {
        Label {
            Text(notice.text).font(.footnote.weight(.medium)).fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: notice.error ? "exclamationmark.circle" : "checkmark.circle")
                .foregroundStyle(notice.error ? .orange : accent)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.black.opacity(0.86), in: RoundedRectangle(cornerRadius: 15))
        .padding(.bottom, 6)
    }

    private var zoomLabel: String {
        let value = model.controls.zoom
        return abs(value.rounded() - value) < 0.04 ? "\(Int(value.rounded()))×"
            : value.formatted(.number.precision(.fractionLength(1))) + "×"
    }

    private var imageLabel: String {
        switch model.previewStatus {
        case .neural: "ИИ · РЕКОНСТРУКЦИЯ"
        case .limited: "ИИ · ТОЛЬКО ФОТО / ПАУЗА"
        case .modelUnavailable: "ИИ · МОДЕЛЬ НЕДОСТУПНА"
        case .standard: model.digitalZoom > 1.08 ? "ЦИФРОВОЕ КАДРИРОВАНИЕ" : "FUSION MAIN"
        }
    }

    private var information: some View {
        NavigationStack {
            List {
                Section("Режимы") {
                    LabeledContent("Quality Zoom", value: "1×–10×")
                    LabeledContent("AI Super Zoom", value: "10×–30×")
                    LabeledContent("Extreme Zoom", value: "30×–200×")
                }
                Section("Съёмка") {
                    Text("Коснитесь объекта для фокуса и замера экспозиции. При включённом Zoom Lock касание переносит точку удержания.")
                    Text("В Extreme мелкие детали исходного кадра ограничены. Результат работы нейросети представляет собой реконструкцию.")
                }
                Section("Устройство") {
                    LabeledContent("Гироскоп", value: model.preview.gyroAvailable ? "Доступен" : "Недоступен")
                    Text(vtSupport ?? "Проверяем VideoToolbox…").font(.footnote)
                }
            }
            .navigationTitle("Muzzle Zoom")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Готово") { info = false } } }
        }
        .task {
            vtSupport = await Task.detached(priority: .utility) { CameraModel.videoToolboxSummary() }.value
        }
    }

    nonisolated private static func galleryImage(_ data: Data) -> UIImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 2048
              ] as CFDictionary) else { return nil }
        return UIImage(cgImage: image)
    }
}

private struct ViewedPhoto: Identifiable {
    let id = UUID()
    let image: UIImage
}

private struct Reticle: Shape {
    let locked: Bool
    let strong: Bool

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let length = rect.width * (locked ? 0.19 : 0.24)
        for x in [rect.minX, rect.maxX] {
            for y in [rect.minY, rect.maxY] {
                path.move(to: CGPoint(x: x, y: y + (y == rect.minY ? length : -length)))
                path.addLine(to: CGPoint(x: x, y: y))
                path.addLine(to: CGPoint(x: x + (x == rect.minX ? length : -length), y: y))
            }
        }
        if strong || locked {
            path.move(to: CGPoint(x: rect.midX - 5, y: rect.midY))
            path.addLine(to: CGPoint(x: rect.midX + 5, y: rect.midY))
            path.move(to: CGPoint(x: rect.midX, y: rect.midY - 5))
            path.addLine(to: CGPoint(x: rect.midX, y: rect.midY + 5))
        }
        return path
    }
}
