import Foundation

enum TransportError: LocalizedError {
    case message(String)

    var errorDescription: String? {
        switch self { case .message(let text): return text }
    }
}

struct Physics: Decodable {
    let energy_kev: [Float]
    let source_energy_kev: Float?
    let upstream_air_mm: Float?
    let sample_xs_mm_inv: [[Float]]
    let air_xs_mm_inv: [[Float]]
    let compton_cdf: [Float]
    let sample_compton_shells: [[Float]]?
    let air_compton_shells: [[Float]]?
    let sample_lateral_mm: Float?

    var flattenedCrossSections: [Float] {
        energy_kev + sample_xs_mm_inv.flatMap { $0 } + air_xs_mm_inv.flatMap { $0 }
    }

    var scope: String {
        if sample_compton_shells != nil && air_compton_shells != nil {
            return "G4 XS + sample MIFF and Air IAM Rayleigh + Penelope-2008 shell/Doppler Compton; Philox4x32-10; no fluorescence or detector response"
        }
        return "G4 XS + sample MIFF and Air IAM Rayleigh + calibrated Compton angle; Philox4x32-10; free-electron energy shift; no fluorescence or detector response"
    }
}

enum OutputMode {
    case image(URL?)
    case radial(map: URL, output: URL)

    var name: String {
        switch self {
        case .image: return "detector_image"
        case .radial: return "gpu_sparse_radial"
        }
    }
}

struct TransportConfig {
    static let usage = "usage: multi_transport photons thickness_mm gap_mm detector_half_mm pixel_pitch_mm beam_radius_mm focus_from_entry_mm density_scale phot_scale compton_scale rayleigh_scale seed physics.json ff.dat [image.raw | --radial map.bin radial.raw]"

    let photons: Int
    let thickness: Float
    let gap: Float
    let detectorHalf: Float
    let pixelPitch: Float
    let beamRadius: Float
    let focusFromEntry: Float
    let densityScale: Float
    let photoScale: Float
    let comptonScale: Float
    let rayleighScale: Float
    let seed: UInt32
    let physicsURL: URL
    let formFactorURL: URL
    let output: OutputMode

    init(arguments: [String]) throws {
        let radial = arguments.count == 18 && arguments[15] == "--radial"
        guard arguments.count == 15 || arguments.count == 16 || radial,
              let photons = Int(arguments[1]),
              let thickness = Float(arguments[2]), let gap = Float(arguments[3]),
              let half = Float(arguments[4]), let pitch = Float(arguments[5]),
              let radius = Float(arguments[6]), let focus = Float(arguments[7]),
              let density = Float(arguments[8]), let photo = Float(arguments[9]),
              let compton = Float(arguments[10]), let rayleigh = Float(arguments[11]),
              let seed = UInt32(arguments[12]),
              photons > 0, photons <= Int(UInt32.max), thickness > 0, gap >= 0,
              half > 0, pitch > 0, radius >= 0, focus > thickness + gap,
              density > 0, photo >= 0, compton >= 0, rayleigh >= 0 else {
            throw TransportError.message(Self.usage)
        }
        self.photons = photons
        self.thickness = thickness
        self.gap = gap
        self.detectorHalf = half
        self.pixelPitch = pitch
        self.beamRadius = radius
        self.focusFromEntry = focus
        self.densityScale = density
        self.photoScale = photo
        self.comptonScale = compton
        self.rayleighScale = rayleigh
        self.seed = seed
        self.physicsURL = URL(fileURLWithPath: arguments[13])
        self.formFactorURL = URL(fileURLWithPath: arguments[14])
        self.output = radial
            ? .radial(map: URL(fileURLWithPath: arguments[16]),
                      output: URL(fileURLWithPath: arguments[17]))
            : .image(arguments.count == 16 ? URL(fileURLWithPath: arguments[15]) : nil)
    }

    var pixels: Int { Int((2 * detectorHalf / pixelPitch).rounded()) }

    func kernelParameters(physics: Physics) -> [Float] {
        // Keep this order synchronized with ParameterIndex in Metal/Kernels.metal.
        [thickness, gap, detectorHalf, pixelPitch, beamRadius, focusFromEntry,
         densityScale, photoScale, comptonScale, rayleighScale,
         physics.source_energy_kev ?? 22.162917, physics.upstream_air_mm ?? 0,
         Float(physics.sample_compton_shells?.count ?? 0),
         Float(physics.air_compton_shells?.count ?? 0),
         (physics.sample_lateral_mm ?? 120.0) * 0.5]
    }
}

func validate(_ physics: Physics, for config: TransportConfig) throws {
    guard (physics.sample_lateral_mm ?? 120.0) > 0 else {
        throw TransportError.message("sample_lateral_mm must be positive")
    }
    guard physics.energy_kev.count == 6,
          zip(physics.energy_kev, physics.energy_kev.dropFirst()).allSatisfy({ $0.0 < $0.1 }),
          physics.sample_xs_mm_inv.count == 3, physics.air_xs_mm_inv.count == 3,
          physics.sample_xs_mm_inv.allSatisfy({ $0.count == 6 }),
          physics.air_xs_mm_inv.allSatisfy({ $0.count == 6 }),
          physics.compton_cdf.count == 1025 else {
        throw TransportError.message("invalid six-node physics table")
    }
    for shells in [physics.sample_compton_shells, physics.air_compton_shells] {
        if let shells, shells.isEmpty || shells.count > 64 || !shells.allSatisfy({
            $0.count == 3 && $0[0] > 0 && $0[1] >= 0 && $0[2] > 0 && $0.allSatisfy(\.isFinite)
        }) {
            throw TransportError.message("invalid Penelope oscillator table")
        }
    }
    guard config.pixels > 0, config.pixels * config.pixels < 4_194_304,
          abs(Float(config.pixels) * config.pixelPitch - 2 * config.detectorHalf) < 1e-4 else {
        throw TransportError.message("detector width must be an exact integer number of pixels")
    }
}
