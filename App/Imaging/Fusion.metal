#include <metal_stdlib>
using namespace metal;

struct FusionGeometry {
    float2 origin;
    float2 samplesPerPixel;
};

constexpr sampler linearSampler(coord::pixel, address::clamp_to_edge, filter::linear);
constexpr sampler pointSampler(coord::pixel, address::clamp_to_edge, filter::nearest);

static float luminance(float3 rgb) {
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

static float3 linearRGB(float3 rgb) {
    return select(rgb / 12.92, pow((rgb + 0.055) / 1.055, float3(2.4)), rgb > 0.04045);
}

static float3 encodedRGB(float3 rgb) {
    rgb = clamp(rgb, 0.0, 1.0);
    return select(12.92 * rgb, 1.055 * pow(rgb, float3(1.0 / 2.4)) - 0.055, rgb > 0.0031308);
}

// Gather original samples around the output coordinate. A half-pixel shift
// in a supporting frame supplies measurements between the reference samples.
static void accumulate(texture2d<float> image, float2 p, float3 anchor,
                       float validity, thread float3 &sum, thread float &weight) {
    float2 first = floor(p - 0.5) + 0.5;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            float2 location = first + float2(dx, dy);
            float2 distance = abs(location - p);
            float2 window = max(float2(0.0), float2(1.0) - distance / 0.84);
            float3 sample = image.sample(pointSampler, location).rgb;
            float difference = max(abs(luminance(sample) - luminance(anchor)),
                                   length(sample - anchor) * 0.36);
            float robust = 1.0 - smoothstep(0.08, 0.23, difference);
            float contribution = window.x * window.y * robust * validity;
            sum += linearRGB(sample) * contribution;
            weight += contribution;
        }
    }
}

static void support(texture2d<float> source, float2 p, float4 frame, float3 anchor,
                    thread float3 &sum, thread float &weight, thread float3 &hdr,
                    thread float &hdrWeight) {
    if (frame.w < 0.5) { return; }
    float2 position = p + frame.xy;
    float3 other = source.sample(linearSampler, position).rgb;
    if (frame.w > 1.5) {
        float3 recovered = linearRGB(other) * frame.z;
        // Roll off the shoulder before clipping. Apply exposure fusion to
        // bright regions only and reject motion in the overlap band.
        float peak = max(recovered.r, max(recovered.g, recovered.b));
        float shoulder = peak <= 0.72 ? peak : 0.72 + 0.28 * (1.0 - exp(-(peak - 0.72) / 0.28));
        float3 rendered = encodedRGB(recovered * (shoulder / max(0.001, peak)));
        float anchorPeak = max(anchor.r, max(anchor.g, anchor.b));
        float overlap = 1.0 - smoothstep(0.09, 0.24, length(rendered - anchor) * 0.5);
        hdrWeight = smoothstep(0.67, 0.96, anchorPeak) * overlap;
        hdr = linearRGB(rendered);
    } else {
        float mismatch = max(abs(luminance(other) - luminance(anchor)), length(other - anchor) * 0.38);
        float validity = (1.0 - smoothstep(0.035, 0.14, mismatch)) * frame.z;
        accumulate(source, position, anchor, validity, sum, weight);
    }
}

kernel void reconstructBurst(texture2d<float, access::sample> reference [[texture(0)]],
                             texture2d<float, access::sample> second [[texture(1)]],
                             texture2d<float, access::sample> third [[texture(2)]],
                             texture2d<float, access::write> result [[texture(3)]],
                             constant FusionGeometry &geometry [[buffer(0)]],
                             constant float4 *frames [[buffer(1)]],
                             uint2 position [[thread_position_in_grid]]) {
    if (position.x >= result.get_width() || position.y >= result.get_height()) { return; }
    float2 p = geometry.origin + (float2(position) + 0.5) * geometry.samplesPerPixel;
    float3 anchor = reference.sample(linearSampler, p).rgb;
    float3 sum = float3(0.0);
    float weight = 0.0;
    float hdrWeight = 0.0;
    float3 hdr = float3(0.0);
    support(second, p, frames[0], anchor, sum, weight, hdr, hdrWeight);
    support(third, p, frames[1], anchor, sum, weight, hdr, hdrWeight);

    // The reference keeps dominant weight if the burst is sparsely sampled
    // or the scene moves. Blend in linear radiance, then return a P3 SDR pixel.
    float3 anchorLinear = linearRGB(anchor);
    float3 fused = (anchorLinear * 0.62 + sum) / (0.62 + weight);
    fused = mix(fused, hdr, hdrWeight);
    result.write(float4(encodedRGB(fused), 1.0), position);
}
