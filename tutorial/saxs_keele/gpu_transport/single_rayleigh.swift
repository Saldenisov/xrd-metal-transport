// Metal feasibility benchmark for the *single-Rayleigh* part of the Keele
// photon transport problem. This is not a Geant4 replacement: Compton,
// photoelectric secondaries, and multiple scattering are not yet transported.
import Foundation
import Metal

let shader = #"""
#include <metal_stdlib>
using namespace metal;

inline float random01(thread uint &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (float(state >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

kernel void single_rayleigh(
    device const float *cdf [[buffer(0)]],
    constant float *p [[buffer(1)]],
    constant uint &seed [[buffer(2)]],
    device uint *result [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    uint state = (id + 1u) * 747796405u + seed;
    float a = 6.28318530718f * random01(state);
    float r = p[4] * sqrt(random01(state));
    float x = r * cos(a), y = r * sin(a);
    float3 direction = normalize(float3(-x, -y, p[5]));
    float total = p[6], rayleigh = p[7];
    float distance = -log(random01(state)) / total;
    float z = distance * direction.z;
    bool scattered = z < p[0];
    if (scattered && random01(state) >= rayleigh / total) {
        result[id] = 0x20000000u; // another process at first interaction
        return;
    }
    float theta = 0.0f;
    if (scattered) {
        x += distance * direction.x;
        y += distance * direction.y;
        float u = random01(state);
        uint lo = 0u, hi = 4096u;
        while (lo < hi) {
            uint mid = (lo + hi) >> 1;
            if (cdf[mid] < u) lo = mid + 1u;
            else hi = mid;
        }
        float width = cdf[lo] - cdf[lo - 1u];
        float fraction = width > 0.0f ? (u - cdf[lo - 1u]) / width : 0.5f;
        theta = (float(lo - 1u) + clamp(fraction, 0.0f, 1.0f))
                * (3.14159265359f / 4096.0f);
        float phi = 6.28318530718f * random01(state);
        float3 e1 = normalize(float3(direction.z, 0.0f, -direction.x));
        float3 e2 = cross(direction, e1);
        direction = cos(theta) * direction + sin(theta) *
                    (cos(phi) * e1 + sin(phi) * e2);
        if (direction.z <= 0.0f) {
            result[id] = 0x30000000u;
            return;
        }
        float remaining = (p[0] - z) / direction.z;
        if (random01(state) > exp(-total * remaining)) {
            result[id] = 0x40000000u; // at least two sample interactions
            return;
        }
    } else {
        z = 0.0f;
    }
    if (random01(state) > exp(-p[8] * p[1] / direction.z)) {
        result[id] = 0x50000000u; // air interaction
        return;
    }
    float detectorZ = p[0] + p[1];
    float flight = (detectorZ - z) / direction.z;
    float hitX = x + direction.x * flight;
    float hitY = y + direction.y * flight;
    if (abs(hitX) >= p[2] || abs(hitY) >= p[2]) {
        result[id] = 0x30000000u;
        return;
    }
    if (!scattered) {
        result[id] = 0x00000000u;
        return;
    }
    float q = p[9] * sin(theta * 0.5f);
    uint bin = min(255u, uint(q / p[3] * 256.0f));
    result[id] = 0x10000000u | bin;
}
"""#

struct Settings {
    let photons: Int
    let thickness: Float
    let gap: Float
    let detectorHalfWidth: Float
    let qMaximum: Float
    let beamRadius: Float
    let focusFromEntry: Float
    let muTotal: Float
    let muRayleigh: Float
    let muAir: Float
    let energyKeV: Float
    let seed: UInt32
    let ffPath: String

    var parameters: [Float] {
        [thickness, gap, detectorHalfWidth, qMaximum, beamRadius,
         focusFromEntry, muTotal, muRayleigh, muAir,
         4 * .pi * energyKeV / 1.239841984]
    }
}

func formFactorTable(_ path: String) throws -> ([Float], [Float]) {
    let content = try String(contentsOfFile: path, encoding: .utf8)
    var qs = [Float](), amplitudes = [Float]()
    for line in content.split(whereSeparator: \.isNewline) {
        let fields = line.split(whereSeparator: \.isWhitespace)
        if fields.count >= 2, let q = Float(fields[0]), let f = Float(fields[1]) {
            qs.append(q)
            amplitudes.append(f)
        }
    }
    guard qs.count >= 2 else { throw NSError(domain: "FF", code: 1) }
    return (qs, amplitudes)
}

func interpolate(_ x: Float, _ xs: [Float], _ ys: [Float]) -> Float {
    var lo = 0, hi = xs.count - 1
    while lo + 1 < hi {
        let mid = (lo + hi) / 2
        if xs[mid] <= x { lo = mid } else { hi = mid }
    }
    if x <= xs[0] { return ys[0] }
    if x >= xs[hi] { return ys[hi] }
    let w = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] * (1 - w) + ys[hi] * w
}

func angularCDF(_ settings: Settings) throws -> [Float] {
    let (qs, ff) = try formFactorTable(settings.ffPath)
    var cumulative = [Float](repeating: 0, count: 4097)
    for i in 1...4096 {
        let theta = (Float(i) - 0.5) * .pi / 4096
        let qG4 = 2 * settings.energyKeV / 510.99895 * sin(theta / 2)
        let f = interpolate(qG4, qs, ff)
        let cosine = cos(theta)
        let kernel = f * f * (1 + cosine * cosine) * sin(theta)
        cumulative[i] = cumulative[i - 1] + max(0, kernel)
    }
    let norm = cumulative[4096]
    guard norm > 0 else { throw NSError(domain: "FF", code: 2) }
    return cumulative.map { $0 / norm }
}

@inline(__always) func random01(_ state: inout UInt32) -> Float {
    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5
    return Float(state >> 8) * (1.0 / 16777216.0) + (0.5 / 16777216.0)
}

func simulateCPU(_ s: Settings, cdf: [Float]) -> [UInt32] {
    var result = [UInt32](repeating: 0, count: s.photons)
    for id in 0..<s.photons {
        var state = (UInt32(truncatingIfNeeded: id) &+ 1) &* 747796405 &+ s.seed
        let a = 2 * Float.pi * random01(&state)
        let r = s.beamRadius * sqrt(random01(&state))
        var x = r * cos(a), y = r * sin(a)
        var dx = -x, dy = -y, dz = s.focusFromEntry
        let norm = sqrt(dx * dx + dy * dy + dz * dz)
        dx /= norm; dy /= norm; dz /= norm
        let distance = -log(random01(&state)) / s.muTotal
        var z = distance * dz
        let scattered = z < s.thickness
        if scattered && random01(&state) >= s.muRayleigh / s.muTotal {
            result[id] = 0x20000000
            continue
        }
        var theta: Float = 0
        if scattered {
            x += distance * dx; y += distance * dy
            let u = random01(&state)
            var lo = 0, hi = 4096
            while lo < hi {
                let mid = (lo + hi) / 2
                if cdf[mid] < u { lo = mid + 1 } else { hi = mid }
            }
            let width = cdf[lo] - cdf[lo - 1]
            let fraction = width > 0 ? (u - cdf[lo - 1]) / width : 0.5
            theta = (Float(lo - 1) + min(1, max(0, fraction))) * .pi / 4096
            let phi = 2 * Float.pi * random01(&state)
            let norm1 = sqrt(dz * dz + dx * dx)
            let e1x = dz / norm1, e1z = -dx / norm1
            let e2x = dy * e1z, e2y = dz * e1x - dx * e1z, e2z = -dy * e1x
            let c = cos(theta), sn = sin(theta)
            let cp = cos(phi), sp = sin(phi)
            let nx = c * dx + sn * (cp * e1x + sp * e2x)
            let ny = c * dy + sn * sp * e2y
            let nz = c * dz + sn * (cp * e1z + sp * e2z)
            dx = nx; dy = ny; dz = nz
            if dz <= 0 { result[id] = 0x30000000; continue }
            let remaining = (s.thickness - z) / dz
            if random01(&state) > exp(-s.muTotal * remaining) {
                result[id] = 0x40000000
                continue
            }
        } else { z = 0 }
        if random01(&state) > exp(-s.muAir * s.gap / dz) {
            result[id] = 0x50000000
            continue
        }
        let flight = (s.thickness + s.gap - z) / dz
        let hitX = x + dx * flight, hitY = y + dy * flight
        if abs(hitX) >= s.detectorHalfWidth || abs(hitY) >= s.detectorHalfWidth {
            result[id] = 0x30000000
            continue
        }
        if scattered {
            let q = 4 * Float.pi * s.energyKeV / 1.239841984 * sin(theta / 2)
            let bin = min(255, UInt32(max(0, q / s.qMaximum * 256)))
            result[id] = 0x10000000 | bin
        }
    }
    return result
}

func simulateMetal(_ s: Settings, cdf: [Float]) throws -> [UInt32] {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw NSError(domain: "Metal", code: 1)
    }
    let library = try device.makeLibrary(source: shader, options: nil)
    let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "single_rayleigh")!)
    let queue = device.makeCommandQueue()!
    let cdfBuffer = cdf.withUnsafeBufferPointer {
        device.makeBuffer(bytes: $0.baseAddress!, length: cdf.count * 4, options: .storageModeShared)!
    }
    let parameters = s.parameters
    let paramsBuffer = parameters.withUnsafeBufferPointer {
        device.makeBuffer(bytes: $0.baseAddress!, length: parameters.count * 4, options: .storageModeShared)!
    }
    var seed = s.seed
    let seedBuffer = device.makeBuffer(bytes: &seed, length: 4, options: .storageModeShared)!
    let output = device.makeBuffer(length: s.photons * 4, options: .storageModeShared)!
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(cdfBuffer, offset: 0, index: 0)
    encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
    encoder.setBuffer(seedBuffer, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)
    let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
    encoder.dispatchThreads(MTLSize(width: s.photons, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else { throw command.error ?? NSError(domain: "Metal", code: 2) }
    let ptr = output.contents().bindMemory(to: UInt32.self, capacity: s.photons)
    return Array(UnsafeBufferPointer(start: ptr, count: s.photons))
}

func main() throws {
    let a = CommandLine.arguments
    guard a.count == 15, (a[1] == "cpu" || a[1] == "gpu"),
          let n = Int(a[2]), let t = Float(a[3]), let gap = Float(a[4]),
          let half = Float(a[5]), let qMax = Float(a[6]), let radius = Float(a[7]),
          let focus = Float(a[8]), let muT = Float(a[9]), let muR = Float(a[10]),
          let muAir = Float(a[11]), let energy = Float(a[12]), let seed = UInt32(a[13]),
          n > 0, t > 0, gap >= 0, half > 0, qMax > 0, radius >= 0,
          focus > t + gap, muT > 0, muR >= 0, muR <= muT, muAir >= 0 else {
        fputs("usage: single_rayleigh cpu|gpu photons thickness_mm gap_mm detector_half_mm qmax_nm-1 beam_radius_mm focus_from_entry_mm mu_total_mm-1 mu_rayleigh_mm-1 mu_air_mm-1 energy_keV seed ff.dat\n", stderr)
        exit(2)
    }
    let s = Settings(photons: n, thickness: t, gap: gap, detectorHalfWidth: half,
                     qMaximum: qMax, beamRadius: radius, focusFromEntry: focus,
                     muTotal: muT, muRayleigh: muR, muAir: muAir,
                     energyKeV: energy, seed: seed, ffPath: a[14])
    let cdf = try angularCDF(s)
    let start = CFAbsoluteTimeGetCurrent()
    let outcomes = try a[1] == "gpu" ? simulateMetal(s, cdf: cdf) : simulateCPU(s, cdf: cdf)
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    var classes = [Int](repeating: 0, count: 6)
    var profile = [Int](repeating: 0, count: 256)
    for outcome in outcomes {
        let category = Int(outcome >> 28)
        classes[category] += 1
        if category == 1 { profile[Int(outcome & 255)] += 1 }
    }
    let payload: [String: Any] = ["mode": a[1], "photons": n, "seconds": elapsed,
                                  "classes": classes, "q_bins": profile]
    let json = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    print(String(decoding: json, as: UTF8.self))
}

// Positional CLI: mode and form-factor path follow all scalar parameters.
// Kept as a plain executable so it can be compared with Geant4 without Python GPU dependencies.
do { try main() } catch { fputs("\(error)\n", stderr); exit(1) }
