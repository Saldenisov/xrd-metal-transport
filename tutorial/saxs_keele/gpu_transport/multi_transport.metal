// Metal kernel for independent X-ray photon histories.
// See docs/PHYSICS.md and docs/PROVENANCE.md for scope and source mapping.
// The Penelope Compton final-state algorithm is adapted from Geant4 11.4.2
// G4PenelopeComptonModel.cc. Geant4 license and attribution: ../../../GEANT4_LICENSE.
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

inline float rand01(thread RNGState &s) {
    if (s.lane == 4u) {
        s.words = philox4x32_10(s.counter, s.key);
        ++s.counter.y;
        s.lane = 0u;
    }
    uint word = s.lane == 0u ? s.words.x : s.lane == 1u ? s.words.y :
                s.lane == 2u ? s.words.z : s.words.w;
    ++s.lane;
    // Float rounding can otherwise turn the largest 24-bit value into 1.0,
    // producing a zero exponential flight even for vanishing cross sections.
    return min((float(word >> 8) + 0.5f) * (1.0f / 16777216.0f),
               0.9999999403953552f);
}

inline float xs_at(device const float *xs, uint start, float energy) {
    uint i = 0;
    while (i < 4u && energy > xs[i + 1u]) ++i;
    float t = clamp((energy - xs[i]) / (xs[i + 1u] - xs[i]), 0.0f, 1.0f);
    float y0 = max(xs[start + i], 1.0e-30f);
    float y1 = max(xs[start + i + 1u], 1.0e-30f);
    return exp(mix(log(y0), log(y1), t));
}

inline float sample_cdf(device const float *cdf, uint bins, thread RNGState &state) {
    float u = rand01(state);
    uint lo = 1u, hi = bins;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (cdf[mid] < u) lo = mid + 1u; else hi = mid;
    }
    float width = cdf[lo] - cdf[lo - 1u];
    float within = width > 0.0f ? (u - cdf[lo - 1u]) / width : 0.5f;
    return (float(lo - 1u) + clamp(within, 0.0f, 1.0f)) * (3.14159265359f / float(bins));
}

inline float rayleigh_angle(device const float *rayCDF, device const float *xs,
                            float energy, thread RNGState &state) {
    uint i = 0;
    while (i < 4u && energy > xs[i + 1u]) ++i;
    float w = clamp((energy - xs[i]) / (xs[i + 1u] - xs[i]), 0.0f, 1.0f);
    if (rand01(state) < w) ++i;
    return sample_cdf(rayCDF + i * 4097u, 4096u, state);
}

inline float3 rotate_photon(float3 d, float theta, thread RNGState &state) {
    float phi = 6.28318530718f * rand01(state);
    float3 e1 = abs(d.z) > 0.1f ? normalize(float3(d.z, 0.0f, -d.x))
                                  : normalize(float3(-d.y, d.x, 0.0f));
    float3 e2 = cross(d, e1);
    return normalize(cos(theta) * d + sin(theta) * (cos(phi) * e1 + sin(phi) * e2));
}

inline float box_exit_distance(float3 pos, float3 dir, float3 lo, float3 hi) {
    float distance = 1.0e30f;
    for (uint axis = 0u; axis < 3u; ++axis) {
        if (dir[axis] > 0.0f)
            distance = min(distance, (hi[axis] - pos[axis]) / dir[axis]);
        else if (dir[axis] < 0.0f)
            distance = min(distance, (lo[axis] - pos[axis]) / dir[axis]);
    }
    return distance;
}

inline float box_entry_distance(float3 pos, float3 dir, float3 lo, float3 hi) {
    float nearDistance = -1.0e30f;
    float farDistance = 1.0e30f;
    for (uint axis = 0u; axis < 3u; ++axis) {
        if (abs(dir[axis]) < 1.0e-12f) {
            if (pos[axis] < lo[axis] || pos[axis] > hi[axis]) return 1.0e30f;
        } else {
            float first = (lo[axis] - pos[axis]) / dir[axis];
            float last = (hi[axis] - pos[axis]) / dir[axis];
            nearDistance = max(nearDistance, min(first, last));
            farDistance = min(farDistance, max(first, last));
        }
    }
    return farDistance > max(nearDistance, 0.0f) && nearDistance > 0.0f
               ? nearDistance : 1.0e30f;
}

inline float penelope_profile(float pz) {
    float arg = pz > 0.0f ? 0.70710678118f + 1.41421356237f * pz
                           : 0.70710678118f - 1.41421356237f * pz;
    float value = 0.5f * exp(0.5f - arg * arg);
    return pz > 0.0f ? 1.0f - value : value;
}

// Photon final state translated from Geant4 11.4.2 G4PenelopeComptonModel
// (Penelope-2008 oscillator/impulse model). The parallel-history architecture
// follows FDA MC-GPU; its PENELOPE-2006 shell tables are deliberately NOT used.
// shells: consecutive [oscillator strength, ionisation energy keV, Hartree factor].
inline float2 penelope_compton(device const float *shells, uint n, float energy,
                              thread RNGState &state) {
    if (n == 0u || n > 64u) return float2(-1.0f, energy);
    const float me = 510.99895f;
    float ek = energy / me;
    float ek2 = 2.0f * ek + 1.0f;
    float eks = ek * ek;
    float ek1 = eks - ek2 - 1.0f;
    float taumin = 1.0f / ek2;
    float a1 = log(ek2);
    float a2 = a1 + 2.0f * ek * (1.0f + ek) / (ek2 * ek2);
    float s0 = 0.0f;
    for (uint i = 0u; i < n; ++i) {
        float ion = shells[3u * i + 1u];
        if (energy <= ion) continue;
        float aux2 = 2.0f * energy * (energy - ion);
        float pz = shells[3u * i + 2u] * (aux2 - me * ion) /
                   (me * sqrt(2.0f * aux2 + ion * ion));
        float rn = penelope_profile(pz);
        s0 += shells[3u * i] * rn;
    }
    if (s0 <= 0.0f) return float2(-1.0f, energy);
    thread float rn[64];
    float tau = 0.0f, cosTheta = 1.0f, s = 0.0f;
    bool accepted = false;
    for (uint attempt = 0u; attempt < 4096u; ++attempt) {
        if (a2 * rand01(state) < a1)
            tau = pow(taumin, rand01(state));
        else
            tau = sqrt(1.0f + rand01(state) * (taumin * taumin - 1.0f));
        tau = min(tau, 0.99999994f);
        float cdt1 = clamp((1.0f - tau) / (ek * tau), 0.0f, 2.0f);
        s = 0.0f;
        for (uint i = 0u; i < n; ++i) {
            float ion = shells[3u * i + 1u];
            rn[i] = 0.0f;
            if (energy <= ion) continue;
            float aux = energy * (energy - ion) * cdt1;
            float pz = shells[3u * i + 2u] * (aux - me * ion) /
                       (me * sqrt(2.0f * aux + ion * ion));
            rn[i] = penelope_profile(pz);
            s += shells[3u * i] * rn[i];
        }
        float tst = s * (1.0f + tau * (ek1 + tau * (ek2 + tau * eks))) /
                    (eks * tau * (1.0f + tau * tau));
        if (rand01(state) * s0 <= tst) {
            cosTheta = 1.0f - cdt1;
            accepted = true;
            break;
        }
    }
    if (!accepted || s <= 0.0f) return float2(-1.0f, energy);

    float pz = 0.0f;
    accepted = false;
    for (uint attempt = 0u; attempt < 4096u; ++attempt) {
        float target = s * rand01(state);
        float cumulative = 0.0f;
        uint selected = n - 1u;
        for (uint i = 0u; i < n; ++i) {
            cumulative += shells[3u * i] * rn[i];
            if (cumulative > target) { selected = i; break; }
        }
        float hartree = shells[3u * selected + 2u];
        float A = clamp(rand01(state) * rn[selected], 1.0e-30f, 0.99999994f);
        if (A < 0.5f)
            pz = (0.70710678118f - sqrt(0.5f - log(2.0f * A))) /
                 (1.41421356237f * hartree);
        else
            pz = (sqrt(0.5f - log(2.0f - 2.0f * A)) - 0.70710678118f) /
                 (1.41421356237f * hartree);
        if (pz < -1.0f) continue;
        float xqc = 1.0f + tau * (tau - 2.0f * cosTheta);
        float af = sqrt(max(xqc, 1.0e-30f)) *
                   (1.0f + tau * (tau - cosTheta) / max(xqc, 1.0e-30f));
        float fpzmax = 1.0f + abs(af) * 0.2f;
        float fpz = 1.0f + af * clamp(pz, -0.2f, 0.2f);
        if (fpzmax * rand01(state) <= fpz) { accepted = true; break; }
    }
    if (!accepted) return float2(-1.0f, energy);
    float t = pz * pz;
    float b1 = 1.0f - t * tau * tau;
    float b2 = 1.0f - t * tau * cosTheta;
    float root = sqrt(abs(b2 * b2 - b1 * (1.0f - t)));
    float epsilon = tau / b1 * (b2 + (pz > 0.0f ? root : -root));
    if (!isfinite(epsilon) || epsilon <= 0.0f)
        return float2(-1.0f, energy);
    return float2(acos(clamp(cosTheta, -1.0f, 1.0f)), energy * epsilon);
}

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

// Must match kernelParameters(_:, physics:) in multi_transport.swift.
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
