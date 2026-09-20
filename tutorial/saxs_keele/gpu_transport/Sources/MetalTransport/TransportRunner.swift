import Foundation
import Metal

private struct TransportBuffers {
    let crossSections: MTLBuffer
    let sampleRayleigh: MTLBuffer
    let comptonCDF: MTLBuffer
    let parameters: MTLBuffer
    let seed: MTLBuffer
    let airRayleigh: MTLBuffer
    let sampleShells: MTLBuffer
    let airShells: MTLBuffer

    init(context: MetalContext, config: TransportConfig, physics: Physics) throws {
        crossSections = try makeBuffer(
            device: context.device, values: physics.flattenedCrossSections
        )
        sampleRayleigh = try makeBuffer(
            device: context.device,
            values: sampleRayleighCDF(physics: physics, formFactorURL: config.formFactorURL)
        )
        airRayleigh = try makeBuffer(
            device: context.device, values: airRayleighCDF(physics: physics)
        )
        comptonCDF = try makeBuffer(device: context.device, values: physics.compton_cdf)
        parameters = try makeBuffer(
            device: context.device, values: config.kernelParameters(physics: physics)
        )
        sampleShells = try makeBuffer(
            device: context.device,
            values: physics.sample_compton_shells?.flatMap { $0 } ?? [Float(0)]
        )
        airShells = try makeBuffer(
            device: context.device,
            values: physics.air_compton_shells?.flatMap { $0 } ?? [Float(0)]
        )
        var mutableSeed = config.seed
        guard let seed = context.device.makeBuffer(
            bytes: &mutableSeed, length: 4, options: .storageModeShared
        ) else { throw TransportError.message("cannot allocate random-seed buffer") }
        self.seed = seed
    }

    func bind(to encoder: MTLComputeCommandEncoder, offset: MTLBuffer, result: MTLBuffer) {
        let ordered = [crossSections, sampleRayleigh, comptonCDF, parameters, seed,
                       offset, result, airRayleigh, sampleShells, airShells]
        for (index, buffer) in ordered.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
    }
}

func runTransport(arguments: [String]) throws {
    let config = try TransportConfig(arguments: arguments)
    let physics = try JSONDecoder().decode(Physics.self, from: Data(contentsOf: config.physicsURL))
    try validate(physics, for: config)

    let context = try MetalContext(executable: arguments[0])
    let transportPipeline = try context.pipeline("transport")
    let reducePipeline: MTLComputePipelineState?
    switch config.output {
    case .image: reducePipeline = nil
    case .radial: reducePipeline = try context.pipeline("reduce_radial")
    }
    let buffers = try TransportBuffers(context: context, config: config, physics: physics)
    let batch = min(2_000_000, config.photons)
    let output = try OutputCollector(
        device: context.device, mode: config.output, pixels: config.pixels, batch: batch
    )

    let started = CFAbsoluteTimeGetCurrent()
    var kernelSeconds = 0.0
    for offset in stride(from: 0, to: config.photons, by: batch) {
        let count = min(batch, config.photons - offset)
        var offsetValue = UInt32(offset)
        guard let offsetBuffer = context.device.makeBuffer(
            bytes: &offsetValue, length: 4, options: .storageModeShared
        ), let command = context.queue.makeCommandBuffer(),
           let encoder = command.makeComputeCommandEncoder() else {
            throw TransportError.message("cannot create Metal transport command")
        }
        encoder.setComputePipelineState(transportPipeline)
        buffers.bind(to: encoder, offset: offsetBuffer, result: output.result)
        let width = min(transportPipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        if let reducePipeline {
            try output.encodeReduction(
                command: command, pipeline: reducePipeline,
                count: count, device: context.device
            )
        }
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw command.error ?? TransportError.message("Metal transport failed")
        }
        kernelSeconds += command.gpuEndTime - command.gpuStartTime
        output.collect(count: count)
    }
    try output.write()

    let summary: [String: Any] = [
        "photons": config.photons,
        "seconds_total": CFAbsoluteTimeGetCurrent() - started,
        "seconds_gpu_kernel": kernelSeconds,
        "metal_device_index": context.deviceIndex,
        "metal_device_count": context.deviceCount,
        "metal_device_name": context.device.name,
        "detector_pixels": config.pixels,
        "classes": output.classes,
        "air_interaction_hits": output.airInteractionHits,
        "direct_count": output.directCount,
        "radial_bins": output.radialBins,
        "output_mode": config.output.name,
        "counts_on_detector": output.classes.prefix(4).reduce(0, +),
        "interaction_cap_count": output.classes[7],
        "physics_scope": physics.scope,
    ]
    let json = try JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys])
    print(String(decoding: json, as: UTF8.self))
}
