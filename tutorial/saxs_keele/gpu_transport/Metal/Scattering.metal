// Cross-section interpolation and Rayleigh/Compton final-state sampling.
// Penelope Compton is adapted from Geant4 11.4.2 G4PenelopeComptonModel.cc.
// See GEANT4_LICENSE and docs/PROVENANCE.md.

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
