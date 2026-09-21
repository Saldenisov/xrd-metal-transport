// Host launcher for MSL generated from Transport.cu by CuMetal.
// CuMetal implements the CUDA Driver API on top of the public Metal framework.

#include "HostCommon.hpp"
#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &description);
    throw std::runtime_error(std::string(operation) + ": " +
                             (name ? name : "CUDA error") + " (" +
                             (description ? description : "no description") + ")");
}

CUdeviceptr copy_to_device(const std::vector<float>& values) {
    CUdeviceptr device = 0;
    check_cuda(cuMemAlloc(&device, values.size() * sizeof(float)), "cuMemAlloc");
    check_cuda(cuMemcpyHtoD(device, values.data(), values.size() * sizeof(float)),
               "cuMemcpyHtoD");
    return device;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 6) {
            throw std::runtime_error(
                "usage: cumetal_transport kernel.metal photons seed input.bundle [image.raw]");
        }
        const std::string kernel_path = argv[1];
        const std::uint64_t photons = xrd_cuda::parse_positive_u64(argv[2], "photon count");
        if (photons > UINT32_MAX) throw std::runtime_error("Photon count exceeds UInt32");
        std::uint32_t seed = xrd_cuda::parse_u32(argv[3], "seed");
        const xrd_cuda::InputBundle input = xrd_cuda::read_bundle(argv[4]);
        const std::size_t batch = std::min<std::size_t>(
            xrd_cuda::kBatchHistories, static_cast<std::size_t>(photons));

        check_cuda(cuInit(0), "cuInit");
        CUdevice device = 0;
        check_cuda(cuDeviceGet(&device, 0), "cuDeviceGet");
        char device_name[256]{};
        check_cuda(cuDeviceGetName(device_name, sizeof(device_name), device), "cuDeviceGetName");
        CUcontext context = nullptr;
        check_cuda(cuCtxCreate(&context, 0, device), "cuCtxCreate");
        CUmodule module = nullptr;
        check_cuda(cuModuleLoad(&module, kernel_path.c_str()), "cuModuleLoad");
        CUfunction kernel = nullptr;
        check_cuda(cuModuleGetFunction(&kernel, module, "xrd_transport"),
                   "cuModuleGetFunction");

        CUdeviceptr d_xs = copy_to_device(input.cross_sections);
        CUdeviceptr d_sample_rayleigh = copy_to_device(input.sample_rayleigh);
        CUdeviceptr d_compton_cdf = copy_to_device(input.compton_cdf);
        CUdeviceptr d_parameters = copy_to_device(input.parameters);
        CUdeviceptr d_air_rayleigh = copy_to_device(input.air_rayleigh);
        CUdeviceptr d_sample_shells = copy_to_device(input.sample_shells);
        CUdeviceptr d_air_shells = copy_to_device(input.air_shells);
        CUdeviceptr d_result = 0;
        check_cuda(cuMemAlloc(&d_result, batch * sizeof(std::uint32_t)),
                   "cuMemAlloc result");
        std::vector<std::uint32_t> host_result(batch);
        xrd_cuda::OutputCollector output(input.pixels);

        // CuMetal compiles MSL lazily on first launch. Exclude compilation from
        // transport timing with one discarded history. CuMetal 0.5.0 events do
        // not reliably bracket this generated kernel, so synchronized host wall
        // time is used below and cross-checked with CUMETAL_TRACE_GPU.
        std::uint32_t warm_offset = 0;
        std::uint32_t warm_count = 1;
        void* warm_parameters[] = {
            &d_xs, &d_sample_rayleigh, &d_compton_cdf, &d_parameters,
            &seed, &warm_offset, &warm_count, &d_result, &d_air_rayleigh,
            &d_sample_shells, &d_air_shells, nullptr,
        };
        check_cuda(cuLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0,
                                  nullptr, warm_parameters, nullptr),
                   "CuMetal warm-up launch");
        check_cuda(cuCtxSynchronize(), "CuMetal warm-up synchronization");

        const auto started = std::chrono::steady_clock::now();
        double kernel_seconds = 0.0;
        for (std::uint64_t offset64 = 0; offset64 < photons; offset64 += batch) {
            std::uint32_t offset = static_cast<std::uint32_t>(offset64);
            std::uint32_t count = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(batch, photons - offset64));
            void* parameters[] = {
                &d_xs, &d_sample_rayleigh, &d_compton_cdf, &d_parameters,
                &seed, &offset, &count, &d_result, &d_air_rayleigh,
                &d_sample_shells, &d_air_shells, nullptr,
            };
            const unsigned int blocks = (count + 255u) / 256u;
            const auto kernel_started = std::chrono::steady_clock::now();
            check_cuda(cuLaunchKernel(kernel, blocks, 1, 1, 256, 1, 1, 0,
                                      nullptr, parameters, nullptr),
                       "cuLaunchKernel");
            check_cuda(cuCtxSynchronize(), "cuCtxSynchronize");
            kernel_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - kernel_started).count();
            check_cuda(cuMemcpyDtoH(host_result.data(), d_result,
                                    count * sizeof(std::uint32_t)), "cuMemcpyDtoH");
            output.collect(host_result.data(), count);
        }
        const double total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        output.write_image(argc == 6 ? argv[5] : "");
        xrd_cuda::print_summary("cuda-cumetal", device_name, photons, total_seconds,
                                kernel_seconds, output);

        cuMemFree(d_xs); cuMemFree(d_sample_rayleigh); cuMemFree(d_compton_cdf);
        cuMemFree(d_parameters); cuMemFree(d_air_rayleigh); cuMemFree(d_sample_shells);
        cuMemFree(d_air_shells); cuMemFree(d_result);
        cuModuleUnload(module);
        cuCtxDestroy(context);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
