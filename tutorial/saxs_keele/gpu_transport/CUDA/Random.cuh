#pragma once

#include <cuda_runtime.h>
#include <cmath>

#define XRD_DEVICE __device__ __forceinline__

struct U2 { unsigned int x, y; };
struct U4 { unsigned int x, y, z, w; };

struct RNGState {
    U4 counter;
    U2 key;
    U4 words;
    unsigned int lane;
};

XRD_DEVICE U4 philox4x32_10(U4 counter, U2 key) {
    for (unsigned int round = 0; round < 10; ++round) {
        const unsigned long long p0 = 0xD2511F53ull * counter.x;
        const unsigned long long p1 = 0xCD9E8D57ull * counter.z;
        const U4 next = {
            static_cast<unsigned int>(p1 >> 32) ^ counter.y ^ key.x,
            static_cast<unsigned int>(p1),
            static_cast<unsigned int>(p0 >> 32) ^ counter.w ^ key.y,
            static_cast<unsigned int>(p0),
        };
        counter = next;
        key.x += 0x9E3779B9u;
        key.y += 0xBB67AE85u;
    }
    return counter;
}

XRD_DEVICE float rand01(RNGState* state) {
    if (state->lane == 4) {
        state->words = philox4x32_10(state->counter, state->key);
        ++state->counter.y;
        state->lane = 0;
    }
    const unsigned int word = state->lane == 0 ? state->words.x
        : state->lane == 1 ? state->words.y
        : state->lane == 2 ? state->words.z : state->words.w;
    ++state->lane;
    return fminf((static_cast<float>(word >> 8) + 0.5f) * (1.0f / 16777216.0f),
                 0.9999999403953552f);
}
