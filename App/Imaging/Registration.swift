import Foundation

/// Small, grayscale windows in sensor order (origin at the top left).
struct GrayPatch {
    let size: Int
    let values: [Float]

    var clippedHighlights: Bool {
        let brights = values.filter { $0 > 0.96 }.count
        let darks = values.filter { $0 < 0.29 }.count
        return brights * 45 > values.count && darks * 6 > values.count
    }

    var sharpness: Float {
        var total: Float = 0
        for y in stride(from: 2, to: size - 2, by: 2) {
            for x in stride(from: 2, to: size - 2, by: 2) {
                let p = y * size + x
                let edge = 4 * values[p] - values[p - 1] - values[p + 1] - values[p - size] - values[p + size]
                total += min(abs(edge), 0.18)
            }
        }
        return total
    }
}

enum PatchRegistration {
    struct Match {
        /// The same feature appears at reference position + offset in the current image.
        let offset: SIMD2<Float>
        let score: Float
    }

    static func match(_ reference: GrayPatch, _ current: GrayPatch,
                      radius: Int, threshold: Float = 0.78) -> Match? {
        guard reference.size == current.size, radius >= 2 else { return nil }
        let size = reference.size
        let border = radius + 3
        guard size > border * 2 + 8 else { return nil }

        var samples: [(Int, Float)] = []
        var firstSum: Float = 0
        var firstSquares: Float = 0
        for y in stride(from: border, to: size - border, by: 3) {
            for x in stride(from: border, to: size - border, by: 3) {
                let index = y * size + x
                let value = reference.values[index]
                samples.append((index, value))
                firstSum += value
                firstSquares += value * value
            }
        }
        let count = Float(samples.count)
        let variance = firstSquares - firstSum * firstSum / count
        guard variance > count * 0.00025 else { return nil }

        func score(_ x: Int, _ y: Int) -> Float {
            let offset = y * size + x
            var sum: Float = 0
            var squares: Float = 0
            var product: Float = 0
            for (index, value) in samples {
                let other = current.values[index + offset]
                sum += other
                squares += other * other
                product += value * other
            }
            let otherVariance = squares - sum * sum / count
            guard otherVariance > count * 0.00015 else { return -1 }
            return (product - firstSum * sum / count) / sqrt(variance * otherVariance)
        }

        var bestX = 0
        var bestY = 0
        var best = score(0, 0)
        var coarse: [(x: Int, y: Int, score: Float)] = []
        for y in stride(from: -radius, through: radius, by: 2) {
            for x in stride(from: -radius, through: radius, by: 2) {
                let value = score(x, y)
                coarse.append((x, y, value))
                if value > best {
                    (bestX, bestY, best) = (x, y, value)
                }
            }
        }

        let coarseX = bestX
        let coarseY = bestY
        for y in max(-radius, coarseY - 2)...min(radius, coarseY + 2) {
            for x in max(-radius, coarseX - 2)...min(radius, coarseX + 2) {
                let value = score(x, y)
                if value > best {
                    (bestX, bestY, best) = (x, y, value)
                }
            }
        }
        let competing = coarse.lazy
            .filter { abs($0.x - bestX) > 4 || abs($0.y - bestY) > 4 }
            .map(\.score).max() ?? -1
        guard best >= threshold, best - competing > 0.025,
              abs(bestX) < radius, abs(bestY) < radius else { return nil }

        func refine(_ left: Float, _ right: Float) -> Float {
            let curvature = left - 2 * best + right
            guard curvature < -0.0001 else { return 0 }
            return max(-0.5, min(0.5, 0.5 * (left - right) / curvature))
        }

        return Match(offset: SIMD2(
            Float(bestX) + refine(score(bestX - 1, bestY), score(bestX + 1, bestY)),
            Float(bestY) + refine(score(bestX, bestY - 1), score(bestX, bestY + 1))
        ), score: best)
    }
}
