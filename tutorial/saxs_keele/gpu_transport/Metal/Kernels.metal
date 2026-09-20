// Photon transport and optional detector-to-radial-profile reduction.
// Packed result: category in bits 28-31, air-interaction flag in bit 27,
// detector-pixel index in bits 0-26.

kernel void probe_compton(device const float *shells [[buffer(0)]],
                          constant uint &n [[buffer(1)]],
                          constant float &energy [[buffer(2)]],
                          constant uint &seed [[buffer(3)]],
                          device float2 *outcomes [[buffer(4)]],
                          uint id [[thread_position_in_grid]]) {
    RNGState state;
    state.counter = uint4(id, 0u, 0u, 0u);
    state.key = uint2(seed, 0xC0FFEE11u);
    state.lane = 4u;
    outcomes[id] = penelope_compton(shells, n, energy, state);
}

// Must match TransportConfig.kernelParameters(physics:) in TransportModels.swift.
enum ParameterIndex {
    THICKNESS = 0, GAP = 1, DETECTOR_HALF = 2, PIXEL_PITCH = 3,
    BEAM_RADIUS = 4, FOCUS = 5, DENSITY = 6,
    PHOTO_SCALE = 7, COMPTON_SCALE = 8, RAYLEIGH_SCALE = 9,
    SOURCE_ENERGY = 10, UPSTREAM_AIR = 11,
    SAMPLE_SHELL_COUNT = 12, AIR_SHELL_COUNT = 13, SAMPLE_HALF = 14
};

kernel void transport(device const float *xs [[buffer(0)]],
                      device const float *rayCDF [[buffer(1)]],
                      device const float *comptonCDF [[buffer(2)]],
                      constant float *p [[buffer(3)]],
                      constant uint &seed [[buffer(4)]],
                      constant uint &offset [[buffer(5)]],
                      device uint *result [[buffer(6)]],
                      device const float *airRayCDF [[buffer(7)]],
                      device const float *sampleShells [[buffer(8)]],
                      device const float *airShells [[buffer(9)]],
                      uint id [[thread_position_in_grid]]) {
    RNGState state;
    state.counter = uint4(offset + id, 0u, 0u, 0u);
    state.key = uint2(seed, 0xC0FFEE11u);
    state.lane = 4u;
    float az = 6.28318530718f * rand01(state);
    float radius = p[BEAM_RADIUS] * sqrt(rand01(state));
    float3 pos = float3(radius * cos(az), radius * sin(az), -p[UPSTREAM_AIR]);
    float3 dir = normalize(float3(-pos.x, -pos.y, p[FOCUS]));
    float energy = p[SOURCE_ENERGY];
    float3 sampleLo = float3(-p[SAMPLE_HALF], -p[SAMPLE_HALF], 0.0f);
    float3 sampleHi = float3(p[SAMPLE_HALF], p[SAMPLE_HALF], p[THICKNESS]);
    float3 worldLo = float3(-5000.0f);
    float3 worldHi = float3(5000.0f);
    // Geant4 SAXSSteppingAction counts only *phantom* interactions in the
    // detector channel labels. Air still changes the trajectory and energy.
    uint nRay = 0u, nCompton = 0u, nAir = 0u;
    uint region = p[UPSTREAM_AIR] > 0.0f ? 0u : 1u; // 0: air; 1: finite sample box
    for (uint interaction = 0u; interaction < 128u; ++interaction) {
        if (region == 1u) {
            float travel = box_exit_distance(pos, dir, sampleLo, sampleHi);
            if (travel < 0.0f) { result[id] = 0x60000000u; return; }
            float muP = xs_at(xs, 6u, energy) * p[DENSITY] * p[PHOTO_SCALE];
            float muC = xs_at(xs, 12u, energy) * p[DENSITY] * p[COMPTON_SCALE];
            float muR = xs_at(xs, 18u, energy) * p[DENSITY] * p[RAYLEIGH_SCALE];
            float total = muP + muC + muR;
            float flight = -log(rand01(state)) / total;
            if (flight >= travel) {
                pos += (travel + 1.0e-5f) * dir;
                region = 0u;
                continue;
            }
            pos += flight * dir;
            float event = rand01(state) * total;
            if (event < muP) { result[id] = 0x50000000u; return; }
            if (event < muP + muC) {
                float theta;
                if (p[SAMPLE_SHELL_COUNT] > 0.0f) {
                    float2 outcome = penelope_compton(sampleShells, uint(p[SAMPLE_SHELL_COUNT]), energy, state);
                    if (outcome.x < 0.0f) { result[id] = 0x70000000u; return; }
                    theta = outcome.x;
                    energy = outcome.y;
                } else {
                    theta = sample_cdf(comptonCDF, 1024u, state);
                    energy /= 1.0f + energy / 510.99895f * (1.0f - cos(theta));
                }
                dir = rotate_photon(dir, theta, state);
                ++nCompton;
            } else {
                float theta = rayleigh_angle(rayCDF, xs, energy, state);
                dir = rotate_photon(dir, theta, state);
                ++nRay;
            }
        } else {
            float sampleTravel = box_entry_distance(pos, dir, sampleLo, sampleHi);
            float detectorTravel = dir.z > 0.0f ?
                (p[THICKNESS] + p[GAP] - pos.z) / dir.z : 1.0e30f;
            if (detectorTravel <= 0.0f) detectorTravel = 1.0e30f;
            float worldTravel = box_exit_distance(pos, dir, worldLo, worldHi);
            float travel = min(sampleTravel, min(detectorTravel, worldTravel));
            if (travel < 0.0f || travel >= 1.0e30f) {
                result[id] = 0x60000000u; return;
            }
            float muP = xs_at(xs, 24u, energy);
            float muC = xs_at(xs, 30u, energy);
            float muR = xs_at(xs, 36u, energy);
            float total = muP + muC + muR;
            float flight = -log(rand01(state)) / total;
            if (flight >= travel) {
                if (sampleTravel <= detectorTravel && sampleTravel <= worldTravel) {
                    pos += (travel + 1.0e-5f) * dir;
                    region = 1u;
                    continue;
                }
                if (worldTravel < detectorTravel) {
                    result[id] = 0x60000000u; return;
                }
                pos += travel * dir;
                int ix = int(floor((pos.x + p[DETECTOR_HALF]) / p[PIXEL_PITCH]));
                int iy = int(floor((pos.y + p[DETECTOR_HALF]) / p[PIXEL_PITCH]));
                uint pixels = uint(rint(2.0f * p[DETECTOR_HALF] / p[PIXEL_PITCH]));
                if (ix < 0 || iy < 0 || ix >= int(pixels) || iy >= int(pixels)) {
                    result[id] = 0x60000000u; return;
                }
                uint category = nCompton > 0u ? 3u : nRay > 1u ? 2u :
                                nRay == 1u ? 1u : 0u;
                // Channels follow Geant4 sample-only stepping counters.
                // Bit 27 separately marks any air interaction for diagnostics.
                result[id] = (category << 28) | ((nAir > 0u ? 1u : 0u) << 27) |
                             (uint(iy) * pixels + uint(ix));
                return;
            }
            pos += flight * dir;
            float event = rand01(state) * total;
            if (event < muP) { result[id] = 0x50000000u; return; }
            if (event < muP + muC) {
                float theta;
                if (p[AIR_SHELL_COUNT] > 0.0f) {
                    float2 outcome = penelope_compton(airShells, uint(p[AIR_SHELL_COUNT]), energy, state);
                    if (outcome.x < 0.0f) { result[id] = 0x70000000u; return; }
                    theta = outcome.x;
                    energy = outcome.y;
                } else {
                    theta = sample_cdf(comptonCDF, 1024u, state);
                    energy /= 1.0f + energy / 510.99895f * (1.0f - cos(theta));
                }
                dir = rotate_photon(dir, theta, state);
            } else {
                dir = rotate_photon(dir, rayleigh_angle(airRayCDF, xs, energy, state), state);
            }
            ++nAir;
        }
    }
    result[id] = 0x70000000u; // explicit diagnostic: interaction cap
}

// Exact sparse pyFAI/XRD-preprocessing reduction of packed detector hits.
// Each detector pixel contributes to at most three radial bins. Keeping this
// step on GPU avoids copying and classifying one UInt32 per incident photon.
kernel void reduce_radial(device const uint *result [[buffer(0)]],
                          constant uint &count [[buffer(1)]],
                          device const short *mapBins [[buffer(2)]],
                          device const float *mapWeights [[buffer(3)]],
                          device const uchar *directMask [[buffer(4)]],
                          constant uint &nBins [[buffer(5)]],
                          device atomic_float *radial [[buffer(6)]],
                          device atomic_uint *classes [[buffer(7)]],
                          device atomic_uint *direct [[buffer(8)]],
                          device atomic_uint *airDetected [[buffer(9)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint packed = result[id];
    uint category = packed >> 28;
    atomic_fetch_add_explicit(classes + min(category, 7u), 1u,
                              memory_order_relaxed);
    if (category > 3u) return;
    if ((packed & 0x08000000u) != 0u)
        atomic_fetch_add_explicit(airDetected, 1u, memory_order_relaxed);
    uint pixel = packed & 0x07ffffffu;
    if (directMask[pixel] != 0u)
        atomic_fetch_add_explicit(direct, 1u, memory_order_relaxed);
    uint start = 3u * pixel;
    for (uint link = 0u; link < 3u; ++link) {
        short bin = mapBins[start + link];
        if (bin >= 0 && uint(bin) < nBins) {
            atomic_fetch_add_explicit(radial + uint(bin),
                                      mapWeights[start + link],
                                      memory_order_relaxed);
        }
    }
}
