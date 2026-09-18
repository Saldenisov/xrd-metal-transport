// Metal photon histories for a homogeneous Keele MIFF slab and ideal 2D detector.
// Physics baseline: G4-derived macroscopic XS, MIFF Rayleigh, measured Penelope
// Compton angle CDF, repeated scattering, air and photoelectric absorption.
// This is an independently validated accelerator prototype, NOT Geant4 itself.
import Foundation
import Metal

private func shaderSource() throws -> String {
    let executable = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    let path = executable.deletingLastPathComponent().appendingPathComponent("multi_transport.metal")
    return try String(contentsOf: path, encoding: .utf8)
}

private struct Physics: Decodable {
    let energy_kev: [Float]
    let source_energy_kev: Float?
    let upstream_air_mm: Float?
    let sample_xs_mm_inv: [[Float]]
    let air_xs_mm_inv: [[Float]]
    let compton_cdf: [Float]
    let sample_compton_shells: [[Float]]?
    let air_compton_shells: [[Float]]?
    let sample_lateral_mm: Float?

    var flattenedXS: [Float] {
        energy_kev + sample_xs_mm_inv.flatMap { $0 } + air_xs_mm_inv.flatMap { $0 }
    }
}

private struct TransportConfig {
    let photons: Int
    let thickness: Float
    let gap: Float
    let half: Float
    let pitch: Float
    let radius: Float
    let focus: Float
    let density: Float
    let photoScale: Float
    let comptonScale: Float
    let rayleighScale: Float
    let seed: UInt32
    let physicsPath: String
    let formFactorPath: String
    let imagePath: String?

    init(_ a: [String]) {
        guard (a.count == 15 || a.count == 16), let photons = Int(a[1]),
              let thickness = Float(a[2]), let gap = Float(a[3]),
              let half = Float(a[4]), let pitch = Float(a[5]),
              let radius = Float(a[6]), let focus = Float(a[7]),
              let density = Float(a[8]), let photoScale = Float(a[9]),
              let comptonScale = Float(a[10]), let rayleighScale = Float(a[11]),
              let seed = UInt32(a[12]), photons > 0, photons <= Int(UInt32.max),
              thickness > 0, gap >= 0, half > 0, pitch > 0, radius >= 0,
              focus > thickness + gap, density > 0, photoScale >= 0,
              comptonScale >= 0, rayleighScale >= 0 else {
            fputs("usage: multi_transport photons thickness_mm gap_mm detector_half_mm pixel_pitch_mm beam_radius_mm focus_from_entry_mm density_scale phot_scale compton_scale rayleigh_scale seed physics.json ff.dat [image.raw]\n", stderr)
            exit(2)
        }
        self.photons = photons
        self.thickness = thickness
        self.gap = gap
        self.half = half
        self.pitch = pitch
        self.radius = radius
        self.focus = focus
        self.density = density
        self.photoScale = photoScale
        self.comptonScale = comptonScale
        self.rayleighScale = rayleighScale
        self.seed = seed
        self.physicsPath = a[13]
        self.formFactorPath = a[14]
        self.imagePath = a.count == 16 ? a[15] : nil
    }

    var pixels: Int { Int((2 * half / pitch).rounded()) }
}

private func validate(_ physics: Physics, config: TransportConfig) throws {
    guard (physics.sample_lateral_mm ?? 120.0) > 0 else {
        throw NSError(domain: "SampleGeometry", code: 1)
    }
    guard physics.energy_kev.count == 6,
          zip(physics.energy_kev, physics.energy_kev.dropFirst()).allSatisfy({ $0.0 < $0.1 }),
          physics.sample_xs_mm_inv.count == 3,
          physics.air_xs_mm_inv.count == 3,
          physics.sample_xs_mm_inv.allSatisfy({ $0.count == 6 }),
          physics.air_xs_mm_inv.allSatisfy({ $0.count == 6 }),
          physics.compton_cdf.count == 1025 else {
        throw NSError(domain: "PhysicsTable", code: 1)
    }
    for shells in [physics.sample_compton_shells, physics.air_compton_shells] {
        if let shells = shells, (shells.isEmpty || shells.count > 64 ||
            !shells.allSatisfy({ $0.count == 3 && $0[0] > 0 && $0[1] >= 0 && $0[2] > 0 &&
                                 $0.allSatisfy(\.isFinite) })) {
            throw NSError(domain: "PenelopeOscillators", code: 1)
        }
    }
    let pixels = config.pixels
    guard pixels > 0, pixels * pixels < 4_194_304,
          abs(Float(pixels) * config.pitch - 2 * config.half) < 1e-4 else {
        throw NSError(domain: "DetectorGeometry", code: 1)
    }
}

private func kernelParameters(_ config: TransportConfig, physics: Physics) -> [Float] {
    // Indices are shared with multi_transport.metal. Keep this order stable.
    [config.thickness, config.gap, config.half, config.pitch,
     config.radius, config.focus, config.density, config.photoScale,
     config.comptonScale, config.rayleighScale,
     physics.source_energy_kev ?? 22.162917, physics.upstream_air_mm ?? 0,
     Float(physics.sample_compton_shells?.count ?? 0),
     Float(physics.air_compton_shells?.count ?? 0),
     (physics.sample_lateral_mm ?? 120.0) * 0.5]
}

private func formFactors(_ path: String) throws -> ([Float], [Float]) {
    let content = try String(contentsOfFile: path, encoding: .utf8)
    var qs: [Float] = [], ff: [Float] = []
    for line in content.split(whereSeparator: \.isNewline) {
        let parts = line.split(whereSeparator: \.isWhitespace)
        if parts.count >= 2, let q = Float(parts[0]), let f = Float(parts[1]) {
            qs.append(q); ff.append(f)
        }
    }
    guard qs.count >= 2 else { throw NSError(domain: "MIFF", code: 1) }
    return (qs, ff)
}

private func interpolate(_ x: Float, _ xs: [Float], _ ys: [Float]) -> Float {
    if x <= xs[0] { return ys[0] }
    if x >= xs[xs.count - 1] { return ys[ys.count - 1] }
    var lo = 0, hi = xs.count - 1
    while lo + 1 < hi {
        let mid = (lo + hi) / 2
        if xs[mid] <= x { lo = mid } else { hi = mid }
    }
    let w = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] * (1 - w) + ys[hi] * w
}

private func rayleighCDF(_ physics: Physics, ffPath: String) throws -> [Float] {
    let (qs, ff) = try formFactors(ffPath)
    var all = [Float]()
    all.reserveCapacity(6 * 4097)
    for energy in physics.energy_kev {
        var cdf = [Float](repeating: 0, count: 4097)
        for i in 1...4096 {
            let theta = (Float(i) - 0.5) * .pi / 4096
            let q = 2 * energy / 510.99895 * sin(theta / 2)
            let f = interpolate(q, qs, ff)
            let c = cos(theta)
            cdf[i] = cdf[i - 1] + max(0, f * f * (1 + c * c) * sin(theta))
        }
        guard cdf[4096] > 0 else { throw NSError(domain: "MIFF", code: 2) }
        all += cdf.map { $0 / cdf[4096] }
    }
    return all
}

private func atomicFormFactor(_ path: URL) throws -> ([Float], [Float]) {
    let content = try String(contentsOf: path, encoding: .utf8)
    var qs: [Float] = [], ff: [Float] = []
    for line in content.split(whereSeparator: \.isNewline).dropFirst() {
        let parts = line.split(whereSeparator: \.isWhitespace)
        if parts.count >= 2, let q = Float(parts[0]), let f = Float(parts[1]) {
            qs.append(q)
            ff.append(f)
        }
    }
    guard qs.count >= 2 && zip(qs, qs.dropFirst()).allSatisfy({ $0 < $1 }) else {
        throw NSError(domain: "AtomicFormFactor", code: 1)
    }
    return (qs, ff)
}

private func airRayleighCDF(_ physics: Physics) throws -> [Float] {
    // The Geant4 Air material is 70% N and 30% O by mass. Its Penelope MI
    // fallback uses atomic form-factor additivity, not the sample's water MIFF.
    guard let led = ProcessInfo.processInfo.environment["G4LEDATA"] else {
        throw NSError(domain: "G4LEDATA is required for air Rayleigh tables", code: 1)
    }
    let root = URL(fileURLWithPath: led)
    let dir = root.standardizedFileURL.appendingPathComponent("penelope/rayleigh")
    let (nQ, nF) = try atomicFormFactor(dir.appendingPathComponent("pdaff07.p08"))
    let (oQ, oF) = try atomicFormFactor(dir.appendingPathComponent("pdaff08.p08"))
    let oxygenStoichiometry = Float((0.3 / 15.9994) / (0.7 / 14.0067))
    var all = [Float]()
    all.reserveCapacity(6 * 4097)
    for energy in physics.energy_kev {
        var cdf = [Float](repeating: 0, count: 4097)
        for i in 1...4096 {
            let theta = (Float(i) - 0.5) * .pi / 4096
            let q = 2 * energy / 510.99895 * sin(theta / 2)
            let nitrogen = interpolate(q, nQ, nF)
            let oxygen = interpolate(q, oQ, oF)
            let c = cos(theta)
            cdf[i] = cdf[i - 1] + (nitrogen * nitrogen + oxygenStoichiometry * oxygen * oxygen)
                     * (1 + c * c) * sin(theta)
        }
        guard cdf[4096] > 0 else { throw NSError(domain: "AirRayleigh", code: 1) }
        all += cdf.map { $0 / cdf[4096] }
    }
    return all
}

private func buffer<T>(device: MTLDevice, values: [T]) -> MTLBuffer {
    values.withUnsafeBufferPointer {
        device.makeBuffer(bytes: $0.baseAddress!, length: values.count * MemoryLayout<T>.stride,
                          options: .storageModeShared)!
    }
}

private func probeCompton(_ a: [String]) throws {
    guard a.count == 8, let count = Int(a[2]), count > 0, count <= 2_000_000,
          let seed = UInt32(a[3]), let energy = Float(a[4]), energy > 0,
          a[6] == "sample" || a[6] == "air" else {
        fputs("usage: multi_transport --probe-compton count seed energy_keV physics.json sample|air output.raw\n", stderr)
        exit(2)
    }
    let physics = try JSONDecoder().decode(Physics.self,
                                           from: Data(contentsOf: URL(fileURLWithPath: a[5])))
    let entries = a[6] == "sample" ? physics.sample_compton_shells : physics.air_compton_shells
    guard let entries = entries, !entries.isEmpty, entries.count <= 64,
          entries.allSatisfy({ $0.count == 3 }) else {
        throw NSError(domain: "PenelopeOscillators", code: 2)
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw NSError(domain: "Metal", code: 1) }
    let library = try device.makeLibrary(source: shaderSource(), options: nil)
    let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "probe_compton")!)
    let queue = device.makeCommandQueue()!
    let shells = buffer(device: device, values: entries.flatMap { $0 })
    var n = UInt32(entries.count), e = energy, key = seed
    let nbuf = device.makeBuffer(bytes: &n, length: 4, options: .storageModeShared)!
    let ebuf = device.makeBuffer(bytes: &e, length: 4, options: .storageModeShared)!
    let sbuf = device.makeBuffer(bytes: &key, length: 4, options: .storageModeShared)!
    let out = device.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride,
                                options: .storageModeShared)!
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(shells, offset: 0, index: 0)
    encoder.setBuffer(nbuf, offset: 0, index: 1)
    encoder.setBuffer(ebuf, offset: 0, index: 2)
    encoder.setBuffer(sbuf, offset: 0, index: 3)
    encoder.setBuffer(out, offset: 0, index: 4)
    let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
    encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else { throw command.error ?? NSError(domain: "Metal", code: 2) }
    try Data(bytes: out.contents(), count: count * MemoryLayout<SIMD2<Float>>.stride)
        .write(to: URL(fileURLWithPath: a[7]))
    print("{\"photons\":\(count),\"seconds_gpu_kernel\":\(command.gpuEndTime - command.gpuStartTime)}")
}

private func main() throws {
    let a = CommandLine.arguments
    let config = TransportConfig(a)
    let physics = try JSONDecoder().decode(Physics.self,
                                           from: Data(contentsOf: URL(fileURLWithPath: config.physicsPath)))
    try validate(physics, config: config)
    let pixels = config.pixels
    let ffCDF = try rayleighCDF(physics, ffPath: config.formFactorPath)
    let airCDF = try airRayleighCDF(physics)
    guard let device = MTLCreateSystemDefaultDevice() else { throw NSError(domain: "Metal", code: 1) }
    let library = try device.makeLibrary(source: shaderSource(), options: nil)
    let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "transport")!)
    let queue = device.makeCommandQueue()!
    let xs = buffer(device: device, values: physics.flattenedXS)
    let ray = buffer(device: device, values: ffCDF)
    let airRay = buffer(device: device, values: airCDF)
    let compt = buffer(device: device, values: physics.compton_cdf)
    let sampleShells = buffer(device: device, values:
        physics.sample_compton_shells?.flatMap { $0 } ?? [Float(0)])
    let airShells = buffer(device: device, values:
        physics.air_compton_shells?.flatMap { $0 } ?? [Float(0)])
    let params = kernelParameters(config, physics: physics)
    let pbuf = buffer(device: device, values: params)
    var mutableSeed = config.seed
    let sbuf = device.makeBuffer(bytes: &mutableSeed, length: 4, options: .storageModeShared)!
    let batch = min(2_000_000, config.photons)
    let out = device.makeBuffer(length: batch * 4, options: .storageModeShared)!
    var image = [UInt32](repeating: 0, count: pixels * pixels)
    var classes = [Int64](repeating: 0, count: 8)
    var airDetected: Int64 = 0
    let started = CFAbsoluteTimeGetCurrent()
    var kernelSeconds = 0.0
    for offset in stride(from: 0, to: config.photons, by: batch) {
        let count = min(batch, config.photons - offset)
        var offsetValue = UInt32(offset)
        let obuf = device.makeBuffer(bytes: &offsetValue, length: 4, options: .storageModeShared)!
        let command = queue.makeCommandBuffer()!
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(xs, offset: 0, index: 0)
        encoder.setBuffer(ray, offset: 0, index: 1)
        encoder.setBuffer(compt, offset: 0, index: 2)
        encoder.setBuffer(pbuf, offset: 0, index: 3)
        encoder.setBuffer(sbuf, offset: 0, index: 4)
        encoder.setBuffer(obuf, offset: 0, index: 5)
        encoder.setBuffer(out, offset: 0, index: 6)
        encoder.setBuffer(airRay, offset: 0, index: 7)
        encoder.setBuffer(sampleShells, offset: 0, index: 8)
        encoder.setBuffer(airShells, offset: 0, index: 9)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw command.error ?? NSError(domain: "Metal", code: 2)
        }
        kernelSeconds += command.gpuEndTime - command.gpuStartTime
        let ptr = out.contents().bindMemory(to: UInt32.self, capacity: count)
        for id in 0..<count {
            let packed = ptr[id]
            let category = Int(packed >> 28)
            classes[category] += 1
            if category <= 3 {
                if (packed & 0x0800_0000) != 0 { airDetected += 1 }
                image[Int(packed & 0x07ff_ffff)] += 1
            }
        }
    }
    let elapsed = CFAbsoluteTimeGetCurrent() - started
    if let imagePath = config.imagePath {
        let data = image.withUnsafeBytes { Data($0) }
        try data.write(to: URL(fileURLWithPath: imagePath))
    }
    let summary: [String: Any] = ["photons": config.photons, "seconds_total": elapsed,
                                   "seconds_gpu_kernel": kernelSeconds,
                                   "detector_pixels": pixels, "classes": classes,
                                   "air_interaction_hits": airDetected,
                                   "counts_on_detector": classes.prefix(4).reduce(0, +),
                                   "interaction_cap_count": classes[7],
                                   "physics_scope": physics.sample_compton_shells != nil &&
                                       physics.air_compton_shells != nil
                                       ? "G4 XS + sample MIFF and Air IAM Rayleigh + Penelope-2008 shell/Doppler Compton; Philox4x32-10; no fluorescence or detector response"
                                       : "G4 XS + sample MIFF and Air IAM Rayleigh + calibrated Compton angle; Philox4x32-10; free-electron energy shift; no fluorescence or detector response"]
    let json = try JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys])
    print(String(decoding: json, as: UTF8.self))
}

do {
    if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "--probe-compton" {
        try probeCompton(CommandLine.arguments)
    } else {
        try main()
    }
} catch { fputs("\(error)\n", stderr); exit(1) }
