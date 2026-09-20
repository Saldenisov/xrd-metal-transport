import Foundation
import Metal

let shaderFiles = ["Random.metal", "Scattering.metal", "Geometry.metal", "Kernels.metal"]

func shaderSource(executable: String) throws -> String {
    let directory = URL(fileURLWithPath: executable).resolvingSymlinksInPath()
        .deletingLastPathComponent().appendingPathComponent("Metal")
    return try shaderFiles.map { name in
        let url = directory.appendingPathComponent(name)
        return "\n// MARK: \(name)\n" + (try String(contentsOf: url, encoding: .utf8))
    }.joined(separator: "\n")
}

func selectedMetalDevice() throws -> (device: MTLDevice, index: Int, count: Int) {
    let devices = MTLCopyAllDevices()
    let requested = ProcessInfo.processInfo.environment["KEELE_METAL_DEVICE_INDEX"]
        .flatMap(Int.init) ?? 0
    guard devices.indices.contains(requested) else {
        throw TransportError.message(
            "KEELE_METAL_DEVICE_INDEX=\(requested), but available indices are 0..<\(devices.count)"
        )
    }
    return (devices[requested], requested, devices.count)
}

func listMetalDevices() throws {
    let rows: [[String: Any]] = MTLCopyAllDevices().enumerated().map { index, device in
        ["index": index, "name": device.name, "registry_id": device.registryID]
    }
    let json = try JSONSerialization.data(withJSONObject: rows, options: [.sortedKeys])
    print(String(decoding: json, as: UTF8.self))
}

struct MetalContext {
    let device: MTLDevice
    let deviceIndex: Int
    let deviceCount: Int
    let library: MTLLibrary
    let queue: MTLCommandQueue

    init(executable: String) throws {
        let selected = try selectedMetalDevice()
        guard let queue = selected.device.makeCommandQueue() else {
            throw TransportError.message("cannot create Metal command queue")
        }
        self.device = selected.device
        self.deviceIndex = selected.index
        self.deviceCount = selected.count
        self.library = try selected.device.makeLibrary(
            source: shaderSource(executable: executable), options: nil
        )
        self.queue = queue
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw TransportError.message("Metal function not found: \(name)")
        }
        return try device.makeComputePipelineState(function: function)
    }
}

func makeBuffer<T>(device: MTLDevice, values: [T]) throws -> MTLBuffer {
    try values.withUnsafeBufferPointer { pointer in
        guard let buffer = device.makeBuffer(
            bytes: pointer.baseAddress!,
            length: values.count * MemoryLayout<T>.stride,
            options: .storageModeShared
        ) else { throw TransportError.message("cannot allocate Metal buffer") }
        return buffer
    }
}

func makeBuffer(device: MTLDevice, data: Data) throws -> MTLBuffer {
    try data.withUnsafeBytes { pointer in
        guard let buffer = device.makeBuffer(
            bytes: pointer.baseAddress!, length: data.count, options: .storageModeShared
        ) else { throw TransportError.message("cannot allocate Metal data buffer") }
        return buffer
    }
}
