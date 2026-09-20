// Geometry supported by the GPU backend: one finite axis-aligned box in air.

inline float3 rotate_photon(float3 direction, float theta, thread RNGState &state) {
    float phi = 6.28318530718f * rand01(state);
    float3 e1 = abs(direction.z) > 0.1f
        ? normalize(float3(direction.z, 0.0f, -direction.x))
        : normalize(float3(-direction.y, direction.x, 0.0f));
    float3 e2 = cross(direction, e1);
    return normalize(cos(theta) * direction
        + sin(theta) * (cos(phi) * e1 + sin(phi) * e2));
}

inline float box_exit_distance(float3 position, float3 direction,
                               float3 lower, float3 upper) {
    float distance = 1.0e30f;
    for (uint axis = 0u; axis < 3u; ++axis) {
        if (direction[axis] > 0.0f)
            distance = min(distance, (upper[axis] - position[axis]) / direction[axis]);
        else if (direction[axis] < 0.0f)
            distance = min(distance, (lower[axis] - position[axis]) / direction[axis]);
    }
    return distance;
}

inline float box_entry_distance(float3 position, float3 direction,
                                float3 lower, float3 upper) {
    float nearDistance = -1.0e30f;
    float farDistance = 1.0e30f;
    for (uint axis = 0u; axis < 3u; ++axis) {
        if (abs(direction[axis]) < 1.0e-12f) {
            if (position[axis] < lower[axis] || position[axis] > upper[axis])
                return 1.0e30f;
        } else {
            float first = (lower[axis] - position[axis]) / direction[axis];
            float last = (upper[axis] - position[axis]) / direction[axis];
            nearDistance = max(nearDistance, min(first, last));
            farDistance = min(farDistance, max(first, last));
        }
    }
    return farDistance > max(nearDistance, 0.0f) && nearDistance > 0.0f
        ? nearDistance : 1.0e30f;
}
