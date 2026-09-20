import Foundation
import Metal

func runComptonProbe(arguments: [String]) throws {
    guard arguments.count == 8, let count = Int(arguments[2]),
          count > 0, count <= 2_000_000,
          let seed = UInt32(arguments[3]),
          let energy = Float(arguments[4]), energy > 0,
          arguments[6] == "sample" || arguments[6] == "air" else {
        throw TransportError.message(
            "usage: multi_transport --probe-compton count seed energy_keV physics.json sample|air output.raw"
        )
    }
    let physics = try JSONDecoder().decode(
        Physics.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[5]))
    )
    let selected = arguments[6] == "sample"
        ? physics.sample_compton_shells : physics.air_compton_shells
    guard let shells = selected, !shells.isEmpty, shells.count <= 64,
          shells.allSatisfy({ $0.count == 3 }) else {
        throw TransportError.message("physics table has no valid Penelope oscillators")
    }

    let context = try MetalContext(executable: arguments[0])
    let pipeline = try context.pipeline("probe_compton")
    let shellBuffer = try makeBuffer(device: context.device, values: shells.flatMap { $0 })
    var shellCount = UInt32(shells.count)
    var mutableEnergy = energy
    var mutableSeed = seed
    guard let countBuffer = context.device.makeBuffer(
        bytes: &shellCount, length: 4, options: .storageModeShared
    ), let energyBuffer = context.device.makeBuffer(
        bytes: &mutableEnergy, length: 4, options: .storageModeShared
    ), let seedBuffer = context.device.makeBuffer(
        bytes: &mutableSeed, length: 4, options: .storageModeShared
    ), let output = context.device.makeBuffer(
        length: count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared
    ), let command = context.queue.makeCommandBuffer(),
       let encoder = command.makeComputeCommandEncoder() else {
        throw TransportError.message("cannot allocate Compton probe resources")
    }
    encoder.setComputePipelineState(pipeline)
    for (index, buffer) in [shellBuffer, countBuffer, energyBuffer, seedBuffer, output].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
    }
    let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
    encoder.dispatchThreads(
        MTLSize(width: count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        throw command.error ?? TransportError.message("Compton probe failed")
    }
    let bytes = count * MemoryLayout<SIMD2<Float>>.stride
    try Data(bytes: output.contents(), count: bytes)
        .write(to: URL(fileURLWithPath: arguments[7]))
    print("{\"photons\":\(count),\"seconds_gpu_kernel\":\(command.gpuEndTime - command.gpuStartTime)}")
}
