#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace xrd_cuda {

constexpr std::size_t kBatchHistories = 2'000'000;

struct BundleHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t pixels;
    std::uint32_t cross_sections;
    std::uint32_t sample_rayleigh;
    std::uint32_t compton_cdf;
    std::uint32_t parameters;
    std::uint32_t air_rayleigh;
    std::uint32_t sample_shells;
    std::uint32_t air_shells;
};

static_assert(sizeof(BundleHeader) == 44, "Unexpected input-bundle header layout");

struct InputBundle {
    std::uint32_t pixels = 0;
    std::vector<float> cross_sections;
    std::vector<float> sample_rayleigh;
    std::vector<float> compton_cdf;
    std::vector<float> parameters;
    std::vector<float> air_rayleigh;
    std::vector<float> sample_shells;
    std::vector<float> air_shells;
};

inline std::vector<float> read_floats(std::ifstream& stream, std::uint32_t count,
                                      const char* label) {
    std::vector<float> values(count);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!stream) throw std::runtime_error(std::string("Truncated ") + label + " table");
    return values;
}

inline InputBundle read_bundle(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open input bundle: " + path);
    BundleHeader header{};
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    const char expected[8] = {'X', 'R', 'D', 'C', 'U', '0', '1', '\0'};
    if (!stream || std::memcmp(header.magic, expected, sizeof(expected)) != 0 ||
        header.version != 1) {
        throw std::runtime_error("Invalid CUDA input-bundle header");
    }
    if (header.pixels == 0 || header.cross_sections != 42 ||
        header.sample_rayleigh != 6 * 4097 || header.compton_cdf != 1025 ||
        header.parameters != 15 || header.air_rayleigh != 6 * 4097 ||
        header.sample_shells == 0 || header.air_shells == 0) {
        throw std::runtime_error("Unsupported CUDA input-bundle dimensions");
    }
    InputBundle bundle;
    bundle.pixels = header.pixels;
    bundle.cross_sections = read_floats(stream, header.cross_sections, "cross-section");
    bundle.sample_rayleigh = read_floats(stream, header.sample_rayleigh, "sample Rayleigh");
    bundle.compton_cdf = read_floats(stream, header.compton_cdf, "Compton CDF");
    bundle.parameters = read_floats(stream, header.parameters, "parameter");
    bundle.air_rayleigh = read_floats(stream, header.air_rayleigh, "air Rayleigh");
    bundle.sample_shells = read_floats(stream, header.sample_shells, "sample shell");
    bundle.air_shells = read_floats(stream, header.air_shells, "air shell");
    if (stream.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("CUDA input bundle contains trailing bytes");
    }
    return bundle;
}

class OutputCollector {
  public:
    explicit OutputCollector(std::uint32_t pixels)
        : pixels_(pixels), image_(static_cast<std::size_t>(pixels) * pixels, 0) {}

    void collect(const std::uint32_t* packed, std::size_t count) {
        for (std::size_t id = 0; id < count; ++id) {
            const std::uint32_t category = packed[id] >> 28;
            classes_[category < classes_.size() ? category : classes_.size() - 1]++;
            if (category <= 3) {
                if ((packed[id] & 0x08000000u) != 0) ++air_interaction_hits_;
                const std::uint32_t pixel = packed[id] & 0x07ffffffu;
                if (pixel >= image_.size()) {
                    throw std::runtime_error("Kernel returned an invalid detector pixel");
                }
                ++image_[pixel];
            }
        }
    }

    void write_image(const std::string& path) const {
        if (path.empty()) return;
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("Cannot create detector image: " + path);
        stream.write(reinterpret_cast<const char*>(image_.data()),
                     static_cast<std::streamsize>(image_.size() * sizeof(std::uint32_t)));
        if (!stream) throw std::runtime_error("Failed while writing detector image: " + path);
    }

    std::uint64_t detector_count() const {
        return classes_[0] + classes_[1] + classes_[2] + classes_[3];
    }

    const std::array<std::uint64_t, 8>& classes() const { return classes_; }
    std::uint64_t air_interaction_hits() const { return air_interaction_hits_; }
    std::uint32_t pixels() const { return pixels_; }

  private:
    std::uint32_t pixels_;
    std::vector<std::uint32_t> image_;
    std::array<std::uint64_t, 8> classes_{};
    std::uint64_t air_interaction_hits_ = 0;
};

inline void print_summary(const char* backend, const std::string& device,
                          std::uint64_t photons, double total_seconds,
                          double kernel_seconds, const OutputCollector& output) {
    std::cout << std::setprecision(9)
              << "{\"air_interaction_hits\":" << output.air_interaction_hits()
              << ",\"backend\":\"" << backend << "\",\"classes\":[";
    for (std::size_t i = 0; i < output.classes().size(); ++i) {
        if (i) std::cout << ',';
        std::cout << output.classes()[i];
    }
    std::cout << "],\"counts_on_detector\":" << output.detector_count()
              << ",\"detector_pixels\":" << output.pixels()
              << ",\"device\":\"" << device
              << "\",\"interaction_cap_count\":" << output.classes()[7]
              << ",\"output_mode\":\"detector_image\""
              << ",\"photons\":" << photons
              << ",\"physics_scope\":\"G4 XS + sample MIFF and Air IAM Rayleigh + "
                 "Penelope-2008 shell/Doppler Compton; Philox4x32-10; no fluorescence "
                 "or detector response\""
              << ",\"seconds_gpu_kernel\":" << kernel_seconds
              << ",\"seconds_total\":" << total_seconds << "}\n";
}

inline std::uint64_t parse_positive_u64(const char* text, const char* label) {
    std::size_t used = 0;
    const std::string value(text);
    const auto parsed = std::stoull(value, &used, 10);
    if (used != value.size() || parsed == 0) {
        throw std::runtime_error(std::string("Invalid ") + label + ": " + value);
    }
    return parsed;
}

inline std::uint32_t parse_u32(const char* text, const char* label) {
    std::size_t used = 0;
    const std::string value(text);
    const auto parsed = std::stoull(value, &used, 10);
    if (used != value.size() || parsed > UINT32_MAX) {
        throw std::runtime_error(std::string("Invalid ") + label + ": " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}

}  // namespace xrd_cuda
