import Foundation
import Metal

struct RadialMap {
    let pixels: UInt32
    let bins: UInt32
    let indices: Data
    let weights: Data
    let directMask: Data

    init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard data.count >= 16 else {
            throw TransportError.message("radial map header is truncated")
        }
        func word(_ offset: Int) -> UInt32 {
            data.withUnsafeBytes { raw in
                UInt32(littleEndian: raw.loadUnaligned(
                    fromByteOffset: offset, as: UInt32.self
                ))
            }
        }
        let magic = word(0)
        let pixels = word(4)
        let bins = word(8)
        let links = word(12)
        guard magic == 0x52414431, pixels > 0, bins > 0, links == 3 else {
            throw TransportError.message("invalid radial map header")
        }
        let entries = Int(pixels) * Int(pixels) * Int(links)
        let indexStart = 16
        let weightStart = indexStart + entries * MemoryLayout<Int16>.stride
        let directStart = weightStart + entries * MemoryLayout<Float>.stride
        guard data.count == directStart + Int(pixels) * Int(pixels) else {
            throw TransportError.message("radial map size does not match its header")
        }
        self.pixels = pixels
        self.bins = bins
        self.indices = data.subdata(in: indexStart..<weightStart)
        self.weights = data.subdata(in: weightStart..<directStart)
        self.directMask = data.subdata(in: directStart..<data.count)
    }
}

final class OutputCollector {
    let result: MTLBuffer
    let mode: OutputMode
    private let pixels: Int
    private let radialMap: RadialMap?
    private let mapIndices: MTLBuffer?
    private let mapWeights: MTLBuffer?
    private let directMask: MTLBuffer?
    private let radialBuffer: MTLBuffer?
    private let classBuffer: MTLBuffer?
    private let directBuffer: MTLBuffer?
    private let airBuffer: MTLBuffer?
    private var image: [UInt32]
    private(set) var radialCounts: [Double]
    private(set) var classes = [Int64](repeating: 0, count: 8)
    private(set) var directCount: Int64 = 0
    private(set) var airInteractionHits: Int64 = 0

    init(device: MTLDevice, mode: OutputMode, pixels: Int, batch: Int) throws {
        self.mode = mode
        self.pixels = pixels
        self.image = [UInt32](repeating: 0, count: pixels * pixels)
        switch mode {
        case .image:
            self.radialMap = nil
            self.radialCounts = []
            self.mapIndices = nil; self.mapWeights = nil; self.directMask = nil
            self.radialBuffer = nil; self.classBuffer = nil
            self.directBuffer = nil; self.airBuffer = nil
            guard let result = device.makeBuffer(length: batch * 4, options: .storageModeShared) else {
                throw TransportError.message("cannot allocate detector result buffer")
            }
            self.result = result
        case .radial(let mapURL, _):
            let map = try RadialMap(url: mapURL)
            guard Int(map.pixels) == pixels else {
                throw TransportError.message("radial map detector size differs from transport")
            }
            self.radialMap = map
            self.radialCounts = [Double](repeating: 0, count: Int(map.bins))
            self.mapIndices = try makeBuffer(device: device, data: map.indices)
            self.mapWeights = try makeBuffer(device: device, data: map.weights)
            self.directMask = try makeBuffer(device: device, data: map.directMask)
            self.radialBuffer = device.makeBuffer(
                length: Int(map.bins) * MemoryLayout<Float>.stride, options: .storageModeShared
            )
            self.classBuffer = device.makeBuffer(length: 8 * 4, options: .storageModeShared)
            self.directBuffer = device.makeBuffer(length: 4, options: .storageModeShared)
            self.airBuffer = device.makeBuffer(length: 4, options: .storageModeShared)
            guard let result = device.makeBuffer(length: batch * 4, options: .storageModePrivate),
                  radialBuffer != nil, classBuffer != nil,
                  directBuffer != nil, airBuffer != nil else {
                throw TransportError.message("cannot allocate radial reduction buffers")
            }
            self.result = result
        }
    }

    var radialBins: Int { Int(radialMap?.bins ?? 0) }

    func encodeReduction(
        command: MTLCommandBuffer, pipeline: MTLComputePipelineState, count: Int,
        device: MTLDevice
    ) throws {
        guard let map = radialMap else { return }
        memset(radialBuffer!.contents(), 0, radialBuffer!.length)
        memset(classBuffer!.contents(), 0, classBuffer!.length)
        memset(directBuffer!.contents(), 0, directBuffer!.length)
        memset(airBuffer!.contents(), 0, airBuffer!.length)
        var countValue = UInt32(count)
        var binsValue = map.bins
        guard let countBuffer = device.makeBuffer(
            bytes: &countValue, length: 4, options: .storageModeShared
        ), let binsBuffer = device.makeBuffer(
            bytes: &binsValue, length: 4, options: .storageModeShared
        ), let encoder = command.makeComputeCommandEncoder() else {
            throw TransportError.message("cannot create radial reduction encoder")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in [result, countBuffer, mapIndices!, mapWeights!, directMask!,
                                binsBuffer, radialBuffer!, classBuffer!, directBuffer!, airBuffer!]
            .enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    func collect(count: Int) {
        if let map = radialMap {
            let radial = radialBuffer!.contents().bindMemory(
                to: Float.self, capacity: Int(map.bins)
            )
            for bin in 0..<Int(map.bins) { radialCounts[bin] += Double(radial[bin]) }
            let categories = classBuffer!.contents().bindMemory(to: UInt32.self, capacity: 8)
            for category in 0..<8 { classes[category] += Int64(categories[category]) }
            directCount += Int64(directBuffer!.contents().bindMemory(to: UInt32.self, capacity: 1)[0])
            airInteractionHits += Int64(airBuffer!.contents().bindMemory(to: UInt32.self, capacity: 1)[0])
            return
        }
        let packedResults = result.contents().bindMemory(to: UInt32.self, capacity: count)
        for id in 0..<count {
            let packed = packedResults[id]
            let category = Int(packed >> 28)
            classes[category] += 1
            if category <= 3 {
                if (packed & 0x0800_0000) != 0 { airInteractionHits += 1 }
                image[Int(packed & 0x07ff_ffff)] += 1
            }
        }
    }

    func write() throws {
        switch mode {
        case .image(let output):
            if let output { try image.withUnsafeBytes { Data($0) }.write(to: output) }
        case .radial(_, let output):
            try radialCounts.withUnsafeBytes { Data($0) }.write(to: output)
        }
    }
}
