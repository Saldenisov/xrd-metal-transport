// Counter-based random stream: one Philox key/counter sequence per photon.
#include <metal_stdlib>
using namespace metal;

inline uint4 philox4x32_10(uint4 counter, uint2 key) {
    for (uint round = 0u; round < 10u; ++round) {
        ulong p0 = ulong(0xD2511F53u) * ulong(counter.x);
        ulong p1 = ulong(0xCD9E8D57u) * ulong(counter.z);
        counter = uint4(uint(p1 >> 32) ^ counter.y ^ key.x,
                        uint(p1), uint(p0 >> 32) ^ counter.w ^ key.y,
                        uint(p0));
        key += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return counter;
}

struct RNGState {
    uint4 counter;
    uint2 key;
    uint4 words;
    uint lane;
};

inline float rand01(thread RNGState &state) {
    if (state.lane == 4u) {
        state.words = philox4x32_10(state.counter, state.key);
        ++state.counter.y;
        state.lane = 0u;
    }
    uint word = state.lane == 0u ? state.words.x : state.lane == 1u ? state.words.y :
                state.lane == 2u ? state.words.z : state.words.w;
    ++state.lane;
    // The half-unit offset keeps every exponential-flight draw inside (0, 1).
    return min((float(word >> 8) + 0.5f) * (1.0f / 16777216.0f),
               0.9999999403953552f);
}
