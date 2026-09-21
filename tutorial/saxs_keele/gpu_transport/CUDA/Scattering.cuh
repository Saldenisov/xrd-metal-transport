#pragma once

#include "Random.cuh"

struct ScatterResult { float angle, energy; };

XRD_DEVICE float xs_at(const float* xs, unsigned int start, float energy) {
    unsigned int i = 0;
    while (i < 4 && energy > xs[i + 1]) ++i;
    const float t = fminf(fmaxf((energy - xs[i]) / (xs[i + 1] - xs[i]), 0.0f), 1.0f);
    const float y0 = fmaxf(xs[start + i], 1.0e-30f);
    const float y1 = fmaxf(xs[start + i + 1], 1.0e-30f);
    return expf((1.0f - t) * logf(y0) + t * logf(y1));
}

XRD_DEVICE float sample_cdf(const float* cdf, unsigned int bins, RNGState* state) {
    const float u = rand01(state);
    unsigned int lo = 1, hi = bins;
    while (lo < hi) {
        const unsigned int mid = (lo + hi) >> 1;
        if (cdf[mid] < u) lo = mid + 1; else hi = mid;
    }
    const float width = cdf[lo] - cdf[lo - 1];
    const float raw = width > 0.0f ? (u - cdf[lo - 1]) / width : 0.5f;
    const float within = fminf(fmaxf(raw, 0.0f), 1.0f);
    return (static_cast<float>(lo - 1) + within) * (3.14159265359f / bins);
}

XRD_DEVICE float rayleigh_angle(const float* ray_cdf, const float* xs,
                                float energy, RNGState* state) {
    unsigned int i = 0;
    while (i < 4 && energy > xs[i + 1]) ++i;
    const float w = fminf(fmaxf((energy - xs[i]) / (xs[i + 1] - xs[i]), 0.0f), 1.0f);
    if (rand01(state) < w) ++i;
    return sample_cdf(ray_cdf + i * 4097, 4096, state);
}

XRD_DEVICE float penelope_profile(float pz) {
    const float arg = pz > 0.0f ? 0.70710678118f + 1.41421356237f * pz
                                : 0.70710678118f - 1.41421356237f * pz;
    const float value = 0.5f * expf(0.5f - arg * arg);
    return pz > 0.0f ? 1.0f - value : value;
}

// Final-state sampler translated from Geant4 11.4.2
// G4PenelopeComptonModel.cc (Penelope-2008 oscillator/impulse model).
XRD_DEVICE ScatterResult penelope_compton(const float* shells, unsigned int count,
                                          float energy, RNGState* state) {
    if (count == 0 || count > 64) return {-1.0f, energy};
    const float electron_mass = 510.99895f;
    const float ek = energy / electron_mass;
    const float ek2 = 2.0f * ek + 1.0f;
    const float eks = ek * ek;
    const float ek1 = eks - ek2 - 1.0f;
    const float tau_min = 1.0f / ek2;
    const float a1 = logf(ek2);
    const float a2 = a1 + 2.0f * ek * (1.0f + ek) / (ek2 * ek2);
    float s0 = 0.0f;
    for (unsigned int i = 0; i < count; ++i) {
        const float ion = shells[3*i + 1];
        if (energy <= ion) continue;
        const float aux2 = 2.0f * energy * (energy - ion);
        const float pz = shells[3*i + 2] * (aux2 - electron_mass * ion) /
                         (electron_mass * sqrtf(2.0f * aux2 + ion * ion));
        s0 += shells[3*i] * penelope_profile(pz);
    }
    if (s0 <= 0.0f) return {-1.0f, energy};

    float shell_weights[64];
    float tau = 0.0f, cos_theta = 1.0f, weight_sum = 0.0f;
    bool accepted = false;
    for (unsigned int attempt = 0; attempt < 4096; ++attempt) {
        if (a2 * rand01(state) < a1)
            tau = powf(tau_min, rand01(state));
        else
            tau = sqrtf(1.0f + rand01(state) * (tau_min * tau_min - 1.0f));
        tau = fminf(tau, 0.99999994f);
        const float cdt1 = fminf(fmaxf((1.0f - tau) / (ek * tau), 0.0f), 2.0f);
        weight_sum = 0.0f;
        for (unsigned int i = 0; i < count; ++i) {
            const float ion = shells[3*i + 1];
            shell_weights[i] = 0.0f;
            if (energy <= ion) continue;
            const float aux = energy * (energy - ion) * cdt1;
            const float pz = shells[3*i + 2] * (aux - electron_mass * ion) /
                             (electron_mass * sqrtf(2.0f * aux + ion * ion));
            shell_weights[i] = penelope_profile(pz);
            weight_sum += shells[3*i] * shell_weights[i];
        }
        const float test = weight_sum *
            (1.0f + tau * (ek1 + tau * (ek2 + tau * eks))) /
            (eks * tau * (1.0f + tau * tau));
        if (rand01(state) * s0 <= test) {
            cos_theta = 1.0f - cdt1;
            accepted = true;
            break;
        }
    }
    if (!accepted || weight_sum <= 0.0f) return {-1.0f, energy};

    float pz = 0.0f;
    accepted = false;
    for (unsigned int attempt = 0; attempt < 4096; ++attempt) {
        const float target = weight_sum * rand01(state);
        float cumulative = 0.0f;
        unsigned int selected = count - 1;
        for (unsigned int i = 0; i < count; ++i) {
            cumulative += shells[3*i] * shell_weights[i];
            if (cumulative > target) { selected = i; break; }
        }
        const float hartree = shells[3*selected + 2];
        const float a = fminf(fmaxf(rand01(state) * shell_weights[selected], 1.0e-30f),
                              0.99999994f);
        if (a < 0.5f)
            pz = (0.70710678118f - sqrtf(0.5f - logf(2.0f * a))) /
                 (1.41421356237f * hartree);
        else
            pz = (sqrtf(0.5f - logf(2.0f - 2.0f * a)) - 0.70710678118f) /
                 (1.41421356237f * hartree);
        if (pz < -1.0f) continue;
        const float xqc = 1.0f + tau * (tau - 2.0f * cos_theta);
        const float safe_xqc = fmaxf(xqc, 1.0e-30f);
        const float af = sqrtf(safe_xqc) *
                         (1.0f + tau * (tau - cos_theta) / safe_xqc);
        const float fpz_max = 1.0f + fabsf(af) * 0.2f;
        const float fpz = 1.0f + af * fminf(fmaxf(pz, -0.2f), 0.2f);
        if (fpz_max * rand01(state) <= fpz) { accepted = true; break; }
    }
    if (!accepted) return {-1.0f, energy};
    const float t = pz * pz;
    const float b1 = 1.0f - t * tau * tau;
    const float b2 = 1.0f - t * tau * cos_theta;
    const float root = sqrtf(fabsf(b2 * b2 - b1 * (1.0f - t)));
    const float epsilon = tau / b1 * (b2 + (pz > 0.0f ? root : -root));
    if (!isfinite(epsilon) || epsilon <= 0.0f) return {-1.0f, energy};
    return {acosf(fminf(fmaxf(cos_theta, -1.0f), 1.0f)), energy * epsilon};
}
