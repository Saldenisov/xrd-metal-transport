# CUDA and CUDA-to-Metal backend

## Purpose and boundary

The CUDA backend makes the restricted XRD transport model available on NVIDIA
GPUs under Windows and Linux. The same CUDA device source can also be translated
to Metal Shading Language by [CuMetal](https://github.com/Lulzx/cuda-metal) for
comparison on Apple silicon.

This backend does not run Geant4 on a GPU. Geant4 remains the CPU physics
reference and table producer. CUDA and native Metal separately implement the
same finite-box subset: Philox random streams, analytic box navigation,
photoelectric absorption, molecular Rayleigh sampling, Penelope-2008 Compton
final states, air transport and an ideal plane detector.

| Platform | Recommended path | Status |
|---|---|---|
| Apple silicon | Native Metal and Swift | Primary, validated implementation |
| Apple silicon comparison | CUDA source translated by CuMetal | Experimental cross-compiler check |
| Windows/Linux with NVIDIA GPU | Native CUDA built with `nvcc` | Supported source path; requires validation on the target GPU |
| Windows without NVIDIA CUDA hardware | None | CUDA does not provide a portable backend for other GPU vendors |

## Source map

| File | Responsibility |
|---|---|
| `CUDA/Random.cuh` | Philox4x32-10 and uniform draws |
| `CUDA/Geometry.cuh` | Vector rotation and finite-box intersections |
| `CUDA/Scattering.cuh` | Cross-section interpolation, Rayleigh CDF and Penelope Compton sampling |
| `CUDA/Transport.cu` | Photon-history kernel and native CUDA executable |
| `CUDA/HostCommon.hpp` | Validated binary input, image tally and JSON summary |
| `CUDA/CuMetalRunner.cpp` | CUDA Driver API launcher backed by CuMetal on macOS |
| `prepare_cuda_input.py` | JSON/form-factor conversion to the shared binary input |
| `compare_cuda_metal.py` | Same-seed detector and timing comparison against native Metal |

The binary input contains only little-endian 32-bit values and compact float
tables. Python performs JSON parsing and builds the same sample and air
Rayleigh CDFs as the Swift host. This avoids adding a C++ JSON library and keeps
the NVIDIA executable independent of Geant4 at runtime.

## Build native CUDA

With a current NVIDIA CUDA toolkit on Linux:

```bash
./scripts/build_cuda.sh --cuda-only
```

The equivalent direct command, including on Windows, is:

```bash
nvcc -O3 -std=c++17 tutorial/saxs_keele/gpu_transport/CUDA/Transport.cu -o cuda_transport
```

No NVIDIA toolkit or GPU was available for the recorded Apple M4 Pro
validation. The `nvcc` path must therefore be compiled and checked on an
NVIDIA host before its timing or numerical behavior is cited.

## Build the CUDA source through CuMetal

Install CuMetal, or build it from source, then expose its compiler, headers and
library to the project script:

```bash
export CUMETAL_PREFIX="$(brew --prefix cumetal)"
./scripts/build_cuda.sh --cumetal-only
```

For a source checkout:

```bash
export CUMETAL_SOURCE=/path/to/cuda-metal
export CUMETAL_BUILD=/path/to/cuda-metal/build
./scripts/build_cuda.sh --cumetal-only
```

The build emits `xrd_transport.cumetal.metal`, its explicit argument ABI and
`cumetal_transport`. CuMetal is an experimental compiler and runtime with a
tested CUDA subset. A successful MSL emission alone is insufficient; the
comparison command below also executes the kernel on the Apple GPU.

## Prepare and run one input

Source the Geant4 environment first so `G4LEDATA` identifies G4EMLOW:

```bash
source /path/to/geant4-install/bin/geant4.sh

python3 tutorial/saxs_keele/gpu_transport/prepare_cuda_input.py \
  50 110 50 0.1 0.1 320 1 1 1 1 \
  tutorial/saxs_keele/gpu_transport/results/multi_transport/g50_22p163_shell_joint.json \
  tutorial/saxs_keele/data/keele_breast_g50.dat \
  build/g50.bundle

# NVIDIA
tutorial/saxs_keele/gpu_transport/cuda_transport \
  1000000 20260917 build/g50.bundle build/cuda.raw

# Apple silicon through CuMetal
tutorial/saxs_keele/gpu_transport/cumetal_transport \
  tutorial/saxs_keele/gpu_transport/xrd_transport.cumetal.metal \
  1000000 20260917 build/g50.bundle build/cumetal.raw
```

Both executables write the same `uint32` detector-image format as native
Metal. Existing Python profile and pyFAI/XRD-preprocessing scripts can therefore
analyze CUDA output without a new scientific analysis path.

## Same-seed validation against native Metal

```bash
python3 tutorial/saxs_keele/gpu_transport/compare_cuda_metal.py \
  --photons 10000000 \
  --output build/cuda-metal-validation.json
```

The maintained Apple M4 Pro record used CuMetal 0.5.0 at commit `fee009f`,
macOS 26.6.2, the g50 50 mm sample, 110 mm downstream air and seed 20260917.
Native Metal and CUDA-to-Metal produced identical images for the first 1,000
histories. At 10,000,000 histories, 196 pixels differed by one count each;
detected histories were 1,015,960 and 1,015,952, and the detector-image L1
difference was 0.0193%. The eight event-category differences were
`[0, 0, 0, -8, 0, +6, +2, 0]`. These rare branch changes can arise from both
Python-versus-Swift CDF rounding and floating-point differences between the two
compiled kernels; this check does not isolate those causes and is not a
statistical Geant4 validation.

The saved records are
[`cuda_cumetal_validation_10m.json`](../tutorial/saxs_keele/gpu_transport/results/cuda_cumetal_validation_10m.json)
and a shorter
[`cuda_cumetal_validation_1m.json`](../tutorial/saxs_keele/gpu_transport/results/cuda_cumetal_validation_1m.json).
For 10 million histories, the CuMetal synchronized kernel interval was 3.914 s
and native Metal GPU time was 0.0460 s, a ratio of 85.1. CuMetal tracing gave
the same order of runtime. Complete timed transport was 4.004 s versus 0.133 s.
The generated MSL is therefore a functional portability check, not a useful
replacement for the hand-written Metal kernel on this workload.

## Current limitations

- CUDA currently writes detector images. GPU-side sparse radial reduction
  remains native-Metal-only; CUDA images use the shared Python integration.
- Geometry remains one homogeneous axis-aligned box in air with an ideal plane
  detector. The CUDA backend does not add arbitrary Geant4 solids or detector
  response.
- The CUDA source shares equations and Philox mapping with native Metal, so a
  same-seed comparison is sensitive to compiler-level floating-point branch
  changes. Geant4 comparisons must still use independent Monte Carlo samples.
- Native CUDA performance and numerical results are not inferred from CuMetal
  measurements. They require an NVIDIA run with recorded driver, toolkit and
  GPU metadata.
