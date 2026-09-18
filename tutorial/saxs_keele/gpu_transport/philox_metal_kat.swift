// Metal known-answer test for Philox4x32-10, from D. E. Shaw Research
// Random123 test vectors (https://github.com/DEShawResearch/random123).
// This standalone test does not yet change multi_transport.swift.
//
// Copyright 2010-2012, D. E. Shaw Research. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions, and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions, and the following disclaimer in documentation
//   and/or other materials provided with the distribution.
// * Neither the name of D. E. Shaw Research nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
import Foundation
import Metal

private let shader = #"""
#include <metal_stdlib>
using namespace metal;

inline uint4 philox4x32_10(uint4 counter, uint2 key) {
    for (uint round = 0u; round < 10u; ++round) {
        ulong p0 = ulong(0xD2511F53u) * ulong(counter.x);
        ulong p1 = ulong(0xCD9E8D57u) * ulong(counter.z);
        counter = uint4(uint(p1 >> 32) ^ counter.y ^ key.x,
                        uint(p1),
                        uint(p0 >> 32) ^ counter.w ^ key.y,
                        uint(p0));
        key += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return counter;
}

kernel void known_answers(device uint4 *result [[buffer(0)]],
                          uint id [[thread_position_in_grid]]) {
    if (id == 0u) result[id] = philox4x32_10(uint4(0u), uint2(0u));
    else if (id == 1u) result[id] = philox4x32_10(uint4(0xffffffffu),
                                                   uint2(0xffffffffu));
    else result[id] = philox4x32_10(
        uint4(0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u),
        uint2(0xa4093822u, 0x299f31d0u));
}
"""#

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
let library = try device.makeLibrary(source: shader, options: nil)
let kernel = try device.makeComputePipelineState(function: library.makeFunction(name: "known_answers")!)
let output = device.makeBuffer(length: 3 * 4 * MemoryLayout<UInt32>.stride,
                               options: .storageModeShared)!
let queue = device.makeCommandQueue()!
let command = queue.makeCommandBuffer()!
let encoder = command.makeComputeCommandEncoder()!
encoder.setComputePipelineState(kernel)
encoder.setBuffer(output, offset: 0, index: 0)
encoder.dispatchThreads(MTLSize(width: 3, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 3, height: 1, depth: 1))
encoder.endEncoding()
command.commit()
command.waitUntilCompleted()
guard command.status == .completed else { throw command.error! }
let values = output.contents().bindMemory(to: UInt32.self, capacity: 12)
let expected: [[UInt32]] = [
    [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
    [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
    [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
]
for row in 0..<3 {
    for column in 0..<4 {
        if values[4 * row + column] != expected[row][column] {
            fatalError("Philox KAT mismatch at \(row), \(column)")
        }
    }
}
print("Philox4x32-10 Metal: 3/3 Random123 known-answer vectors passed")
