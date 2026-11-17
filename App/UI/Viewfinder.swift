import CoreImage
import MetalKit
import SwiftUI

struct Viewfinder: UIViewRepresentable {
    let pipeline: PreviewPipeline

    func makeCoordinator() -> Renderer { Renderer(pipeline: pipeline) }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: pipeline.device)
        view.isOpaque = true
        view.framebufferOnly = false
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0.025, green: 0.029, blue: 0.026, alpha: 1)
        view.preferredFramesPerSecond = 30
        view.delegate = context.coordinator
        view.isAccessibilityElement = false
        return view
    }

    func updateUIView(_ view: MTKView, context: Context) {}

    final class Renderer: NSObject, MTKViewDelegate {
        let pipeline: PreviewPipeline
        private let commands: MTLCommandQueue?
        private let context: CIContext?
        private let inFlight = DispatchSemaphore(value: 2)
        private var rendered: UInt64?
        private var hasCleared = false

        init(pipeline: PreviewPipeline) {
            self.pipeline = pipeline
            commands = pipeline.device?.makeCommandQueue()
            context = pipeline.device.map { CIContext(mtlDevice: $0, options: [.cacheIntermediates: false]) }
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            rendered = nil
            hasCleared = false
        }

        func draw(in view: MTKView) {
            let latest = pipeline.latest.read()
            if let latest, latest.id == rendered { return }
            if latest == nil && hasCleared { return }
            guard let commands, let context,
                  inFlight.wait(timeout: .now()) == .success else { return }
            guard let drawable = view.currentDrawable, let command = commands.makeCommandBuffer() else {
                inFlight.signal()
                return
            }
            let rect = CGRect(x: 0, y: 0, width: drawable.texture.width, height: drawable.texture.height)
            let image: CIImage
            if let latest {
                let source = latest.image
                let scale = max(rect.width / source.extent.width, rect.height / source.extent.height)
                let scaled = source.clampedToExtent()
                    .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                    .transformed(by: CGAffineTransform(translationX: (rect.width - source.extent.width * scale) / 2,
                                                        y: (rect.height - source.extent.height * scale) / 2))
                image = scaled.cropped(to: rect)
                rendered = latest.id
                hasCleared = false
            } else {
                image = CIImage(color: CIColor(red: 0.025, green: 0.029, blue: 0.026)).cropped(to: rect)
                hasCleared = true
            }
            context.render(image, to: drawable.texture, commandBuffer: command,
                           bounds: rect, colorSpace: ImageTools.sRGB)
            let semaphore = inFlight
            command.addCompletedHandler { _ in semaphore.signal() }
            command.present(drawable)
            command.commit()
        }
    }
}
