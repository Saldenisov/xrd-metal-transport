# XRD photon transport on Apple Metal

Research prototype for **X-ray diffraction from homogeneous water, fat and
collagen mixtures**. A Geant4 11.4.2 SAXS application provides the independent
CPU reference and exports material cross sections and Penelope Compton shell
tables. A Metal kernel transports independent photon histories through a finite
sample box, air and an ideal pixel detector. The radial signal is analyzed by
the same XRD-preprocessing/pyFAI path on both backends.

This product includes software developed by Members of the Geant4 Collaboration
(http://cern.ch/geant4). The adapted SAXS example remains distinguishable from
the Geant4 toolkit; see [provenance and license](docs/PROVENANCE.md).

## What is in this repository

| Path | Role |
|---|---|
| `tutorial/saxs_keele/src`, `include` | Geant4 SAXS example with finite-box geometry, image-only scoring and physics-table export |
| `tutorial/saxs_keele/gpu_transport/Metal` | Short Metal modules for RNG, finite-box geometry, scattering physics and transport kernels |
| `tutorial/saxs_keele/gpu_transport/Sources/MetalTransport` | Swift input validation, table preparation, device selection, dispatch and output reduction |
| `tutorial/saxs_keele/gpu_transport/geometry.py` | One supported slab-geometry specification that emits matched Geant4 and Metal inputs |
| `tutorial/saxs_keele/gpu_transport/metal_radial.py` | Exact sparse pyFAI operator export and optional GPU-side radial reduction |
| `tutorial/saxs_keele/prepare_keele_ff.py` | Converts the six-column component table to Geant4 MI form factors |
| `tutorial/saxs_keele/gpu_transport/results` | Small physics fixtures and historical JSON validation records; no raw histories or detector images |

Read [architecture](docs/ARCHITECTURE.md), [physics and units](docs/PHYSICS.md), [reproducible examples](docs/EXAMPLES.md),
and [validation with unresolved differences](docs/VALIDATION.md) before using
the output as a reference.

## Quick start

Requires macOS with Apple Metal, Swift, CMake, **Geant4 11.4.2 with G4EMLOW**,
and Python with NumPy, SciPy, pandas, matplotlib, pyFAI, uproot and
[`xrd-preprocessing`](https://github.com/Eos-Dx/XRD-preprocessing).

```bash
# From repository root; point CMake to a Geant4 11.4.2 installation.
export CMAKE_PREFIX_PATH=/path/to/geant4-install
source /path/to/geant4-install/bin/geant4.sh
./scripts/build.sh

# Independent 100 mm water CPU/Metal run; 2 million photons per backend.
PYTHON=python3 PHOTONS=2000000 THREADS=8 ./scripts/quick_compare.sh
```

The 2-million-photon command is a **smoke comparison**, not a high-precision
parity claim. The saved large-run records and their exact conditions are in
[validation](docs/VALIDATION.md). `build/` and generated images are ignored.

## Published example suite

The [`examples/xrd_suite`](examples/xrd_suite) workflow runs matched Geant4
and Metal calculations for pure water, fat and collagen; all three binary
mixtures; two ternary compositions; and 10–100 mm thickness series. The
maintained publication dataset uses 100 million independent photon histories
per backend and case. It includes detector maps, azimuthal integrations,
channel comparisons, timing and full-covariance profile parity tests.

[Open the result gallery](examples/xrd_suite/results_100m/RESULTS.md).

![Geant4 and Metal XRD profile overview](examples/xrd_suite/results_100m/profile_overview.png)

## Scope

The model uses a finite homogeneous sample, a monochromatic photon source,
sample molecular-interference form factors, air interactions and an ideal
entrance-crossing detector. It does not model a voxelized patient, electron
transport, fluorescence or a measured detector response. The current Metal
Rayleigh sampler uses a tabulated angular CDF rather than Geant4's RITA
sampler. High-statistics residuals remain unresolved. No clinical or dose
equivalence is claimed.

The repository is being prepared for a later public release. Before that
release, confirm redistribution rights for `data/xrd_components.txt` and
derived Keele form-factor files; see [release checks](docs/RELEASE.md).
