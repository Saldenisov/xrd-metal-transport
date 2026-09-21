// CUDA implementation of the restricted XRD photon transport model.
//
// The same source is compiled by nvcc on NVIDIA GPUs and translated to MSL by
// CuMetal on Apple silicon. Physics and packed-output conventions intentionally
// match Metal/{Random,Geometry,Scattering,Kernels}.metal.

#include "Geometry.cuh"
#include "Scattering.cuh"
#include <cstdint>

enum ParameterIndex {
    THICKNESS = 0, GAP = 1, DETECTOR_HALF = 2, PIXEL_PITCH = 3,
    BEAM_RADIUS = 4, FOCUS = 5, DENSITY = 6,
    PHOTO_SCALE = 7, COMPTON_SCALE = 8, RAYLEIGH_SCALE = 9,
    SOURCE_ENERGY = 10, UPSTREAM_AIR = 11,
    SAMPLE_SHELL_COUNT = 12, AIR_SHELL_COUNT = 13, SAMPLE_HALF = 14
};

extern "C" __global__ void xrd_transport(
    const float* xs, const float* sample_rayleigh, const float* compton_cdf,
    const float* parameters, unsigned int seed, unsigned int offset,
    unsigned int count, unsigned int* result, const float* air_rayleigh,
    const float* sample_shells, const float* air_shells) {
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    const float* p = parameters;
    RNGState state{{offset + id, 0, 0, 0}, {seed, 0xC0FFEE11u}, {0, 0, 0, 0}, 4};
    const float azimuth = 6.28318530718f * rand01(&state);
    const float radius = p[BEAM_RADIUS] * sqrtf(rand01(&state));
    Vec3 position{radius * cosf(azimuth), radius * sinf(azimuth), -p[UPSTREAM_AIR]};
    Vec3 direction = normalize3({-position.x, -position.y, p[FOCUS]});
    float energy = p[SOURCE_ENERGY];
    const Vec3 sample_lower{-p[SAMPLE_HALF], -p[SAMPLE_HALF], 0.0f};
    const Vec3 sample_upper{p[SAMPLE_HALF], p[SAMPLE_HALF], p[THICKNESS]};
    const Vec3 world_lower{-5000.0f, -5000.0f, -5000.0f};
    const Vec3 world_upper{5000.0f, 5000.0f, 5000.0f};
    unsigned int rayleigh_count = 0, compton_count = 0, air_count = 0;
    unsigned int region = p[UPSTREAM_AIR] > 0.0f ? 0 : 1;

    for (unsigned int interaction = 0; interaction < 128; ++interaction) {
        if (region == 1) {
            const float travel = box_exit_distance(position, direction, sample_lower, sample_upper);
            if (travel < 0.0f) { result[id] = 0x60000000u; return; }
            const float mu_photo = xs_at(xs, 6, energy) * p[DENSITY] * p[PHOTO_SCALE];
            const float mu_compton = xs_at(xs, 12, energy) * p[DENSITY] * p[COMPTON_SCALE];
            const float mu_rayleigh = xs_at(xs, 18, energy) * p[DENSITY] * p[RAYLEIGH_SCALE];
            const float total = mu_photo + mu_compton + mu_rayleigh;
            const float flight = -logf(rand01(&state)) / total;
            if (flight >= travel) {
                position = add(position, mul(travel + 1.0e-5f, direction));
                region = 0;
                continue;
            }
            position = add(position, mul(flight, direction));
            const float event = rand01(&state) * total;
            if (event < mu_photo) { result[id] = 0x50000000u; return; }
            if (event < mu_photo + mu_compton) {
                float theta;
                if (p[SAMPLE_SHELL_COUNT] > 0.0f) {
                    const ScatterResult outcome = penelope_compton(
                        sample_shells, static_cast<unsigned int>(p[SAMPLE_SHELL_COUNT]),
                        energy, &state);
                    if (outcome.angle < 0.0f) { result[id] = 0x70000000u; return; }
                    theta = outcome.angle;
                    energy = outcome.energy;
                } else {
                    theta = sample_cdf(compton_cdf, 1024, &state);
                    energy /= 1.0f + energy / 510.99895f * (1.0f - cosf(theta));
                }
                direction = rotate_photon(direction, theta, &state);
                ++compton_count;
            } else {
                direction = rotate_photon(
                    direction, rayleigh_angle(sample_rayleigh, xs, energy, &state), &state);
                ++rayleigh_count;
            }
        } else {
            const float sample_travel = box_entry_distance(
                position, direction, sample_lower, sample_upper);
            float detector_travel = direction.z > 0.0f
                ? (p[THICKNESS] + p[GAP] - position.z) / direction.z : 1.0e30f;
            if (detector_travel <= 0.0f) detector_travel = 1.0e30f;
            const float world_travel = box_exit_distance(
                position, direction, world_lower, world_upper);
            const float travel = fminf(sample_travel, fminf(detector_travel, world_travel));
            if (travel < 0.0f || travel >= 1.0e30f) {
                result[id] = 0x60000000u; return;
            }
            const float mu_photo = xs_at(xs, 24, energy);
            const float mu_compton = xs_at(xs, 30, energy);
            const float mu_rayleigh = xs_at(xs, 36, energy);
            const float total = mu_photo + mu_compton + mu_rayleigh;
            const float flight = -logf(rand01(&state)) / total;
            if (flight >= travel) {
                if (sample_travel <= detector_travel && sample_travel <= world_travel) {
                    position = add(position, mul(travel + 1.0e-5f, direction));
                    region = 1;
                    continue;
                }
                if (world_travel < detector_travel) {
                    result[id] = 0x60000000u; return;
                }
                position = add(position, mul(travel, direction));
                const int ix = static_cast<int>(floorf(
                    (position.x + p[DETECTOR_HALF]) / p[PIXEL_PITCH]));
                const int iy = static_cast<int>(floorf(
                    (position.y + p[DETECTOR_HALF]) / p[PIXEL_PITCH]));
                const unsigned int pixels = static_cast<unsigned int>(
                    rintf(2.0f * p[DETECTOR_HALF] / p[PIXEL_PITCH]));
                if (ix < 0 || iy < 0 || ix >= static_cast<int>(pixels) ||
                    iy >= static_cast<int>(pixels)) {
                    result[id] = 0x60000000u; return;
                }
                const unsigned int category = compton_count > 0 ? 3
                    : rayleigh_count > 1 ? 2 : rayleigh_count == 1 ? 1 : 0;
                result[id] = (category << 28) | ((air_count > 0 ? 1u : 0u) << 27) |
                             (static_cast<unsigned int>(iy) * pixels +
                              static_cast<unsigned int>(ix));
                return;
            }
            position = add(position, mul(flight, direction));
            const float event = rand01(&state) * total;
            if (event < mu_photo) { result[id] = 0x50000000u; return; }
            if (event < mu_photo + mu_compton) {
                float theta;
                if (p[AIR_SHELL_COUNT] > 0.0f) {
                    const ScatterResult outcome = penelope_compton(
                        air_shells, static_cast<unsigned int>(p[AIR_SHELL_COUNT]),
                        energy, &state);
                    if (outcome.angle < 0.0f) { result[id] = 0x70000000u; return; }
                    theta = outcome.angle;
                    energy = outcome.energy;
                } else {
                    theta = sample_cdf(compton_cdf, 1024, &state);
                    energy /= 1.0f + energy / 510.99895f * (1.0f - cosf(theta));
                }
                direction = rotate_photon(direction, theta, &state);
            } else {
                direction = rotate_photon(
                    direction, rayleigh_angle(air_rayleigh, xs, energy, &state), &state);
            }
            ++air_count;
        }
    }
    result[id] = 0x70000000u;
}

#undef XRD_DEVICE

#ifndef XRDCUDA_DEVICE_ONLY

#include "HostCommon.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
T* copy_to_device(const std::vector<T>& values) {
    T* device = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device), values.size() * sizeof(T)),
               "cudaMalloc");
    check_cuda(cudaMemcpy(device, values.data(), values.size() * sizeof(T),
                          cudaMemcpyHostToDevice), "cudaMemcpy host to device");
    return device;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) {
            throw std::runtime_error(
                "usage: cuda_transport photons seed input.bundle [image.raw]");
        }
        const std::uint64_t photons = xrd_cuda::parse_positive_u64(argv[1], "photon count");
        if (photons > UINT32_MAX) throw std::runtime_error("Photon count exceeds UInt32");
        const std::uint32_t seed = xrd_cuda::parse_u32(argv[2], "seed");
        const xrd_cuda::InputBundle input = xrd_cuda::read_bundle(argv[3]);
        const std::size_t batch = std::min<std::size_t>(
            xrd_cuda::kBatchHistories, static_cast<std::size_t>(photons));

        int device_index = 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDevice(&device_index), "cudaGetDevice");
        check_cuda(cudaGetDeviceProperties(&properties, device_index), "cudaGetDeviceProperties");

        float* d_xs = copy_to_device(input.cross_sections);
        float* d_sample_rayleigh = copy_to_device(input.sample_rayleigh);
        float* d_compton_cdf = copy_to_device(input.compton_cdf);
        float* d_parameters = copy_to_device(input.parameters);
        float* d_air_rayleigh = copy_to_device(input.air_rayleigh);
        float* d_sample_shells = copy_to_device(input.sample_shells);
        float* d_air_shells = copy_to_device(input.air_shells);
        std::uint32_t* d_result = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_result),
                              batch * sizeof(std::uint32_t)), "cudaMalloc result");
        std::vector<std::uint32_t> host_result(batch);
        xrd_cuda::OutputCollector output(input.pixels);

        cudaEvent_t start_event{}, stop_event{};
        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        const auto started = std::chrono::steady_clock::now();
        double kernel_seconds = 0.0;
        for (std::uint64_t offset = 0; offset < photons; offset += batch) {
            const std::uint32_t count = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(batch, photons - offset));
            const unsigned int blocks = (count + 255u) / 256u;
            check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");
            xrd_transport<<<blocks, 256>>>(
                d_xs, d_sample_rayleigh, d_compton_cdf, d_parameters, seed,
                static_cast<std::uint32_t>(offset), count, d_result,
                d_air_rayleigh, d_sample_shells, d_air_shells);
            check_cuda(cudaGetLastError(), "xrd_transport launch");
            check_cuda(cudaEventRecord(stop_event), "cudaEventRecord stop");
            check_cuda(cudaEventSynchronize(stop_event), "xrd_transport synchronization");
            float milliseconds = 0.0f;
            check_cuda(cudaEventElapsedTime(&milliseconds, start_event, stop_event),
                       "cudaEventElapsedTime");
            kernel_seconds += milliseconds * 1.0e-3;
            check_cuda(cudaMemcpy(host_result.data(), d_result,
                                  count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                       "cudaMemcpy device to host");
            output.collect(host_result.data(), count);
        }
        const double total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        output.write_image(argc == 5 ? argv[4] : "");
        xrd_cuda::print_summary("cuda", properties.name, photons, total_seconds,
                                kernel_seconds, output);

        cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
        cudaFree(d_xs); cudaFree(d_sample_rayleigh); cudaFree(d_compton_cdf);
        cudaFree(d_parameters); cudaFree(d_air_rayleigh); cudaFree(d_sample_shells);
        cudaFree(d_air_shells); cudaFree(d_result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#endif
