#pragma once

#include "Random.cuh"

struct Vec3 { float x, y, z; };

XRD_DEVICE Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
XRD_DEVICE Vec3 mul(float s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; }
XRD_DEVICE float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
XRD_DEVICE Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
XRD_DEVICE Vec3 normalize3(Vec3 a) {
    const float inverse = 1.0f / sqrtf(fmaxf(dot(a, a), 1.0e-30f));
    return mul(inverse, a);
}

XRD_DEVICE Vec3 rotate_photon(Vec3 direction, float theta, RNGState* state) {
    const float phi = 6.28318530718f * rand01(state);
    const Vec3 basis = fabsf(direction.z) > 0.1f
        ? Vec3{direction.z, 0.0f, -direction.x}
        : Vec3{-direction.y, direction.x, 0.0f};
    const Vec3 e1 = normalize3(basis);
    const Vec3 e2 = cross(direction, e1);
    return normalize3(add(mul(cosf(theta), direction),
                          mul(sinf(theta), add(mul(cosf(phi), e1),
                                               mul(sinf(phi), e2)))));
}

XRD_DEVICE float axis_exit(float position, float direction, float lower, float upper) {
    if (direction > 0.0f) return (upper - position) / direction;
    if (direction < 0.0f) return (lower - position) / direction;
    return 1.0e30f;
}

XRD_DEVICE float box_exit_distance(Vec3 p, Vec3 d, Vec3 lower, Vec3 upper) {
    return fminf(axis_exit(p.x, d.x, lower.x, upper.x),
                 fminf(axis_exit(p.y, d.y, lower.y, upper.y),
                       axis_exit(p.z, d.z, lower.z, upper.z)));
}

XRD_DEVICE bool update_slab(float position, float direction, float lower, float upper,
                            float* near_distance, float* far_distance) {
    if (fabsf(direction) < 1.0e-12f) return position >= lower && position <= upper;
    const float first = (lower - position) / direction;
    const float last = (upper - position) / direction;
    *near_distance = fmaxf(*near_distance, fminf(first, last));
    *far_distance = fminf(*far_distance, fmaxf(first, last));
    return true;
}

XRD_DEVICE float box_entry_distance(Vec3 p, Vec3 d, Vec3 lower, Vec3 upper) {
    float near_distance = -1.0e30f;
    float far_distance = 1.0e30f;
    if (!update_slab(p.x, d.x, lower.x, upper.x, &near_distance, &far_distance) ||
        !update_slab(p.y, d.y, lower.y, upper.y, &near_distance, &far_distance) ||
        !update_slab(p.z, d.z, lower.z, upper.z, &near_distance, &far_distance)) {
        return 1.0e30f;
    }
    return far_distance > fmaxf(near_distance, 0.0f) && near_distance > 0.0f
        ? near_distance : 1.0e30f;
}
