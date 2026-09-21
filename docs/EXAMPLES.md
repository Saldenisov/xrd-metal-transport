# Reproducible examples

Commands are relative to repository root. The examples require macOS with
Metal, Geant4 **11.4.2** and its physics datasets. Source the installed
`geant4.sh` so `G4LEDATA` points to the G4EMLOW data. Install Python packages
from `requirements-analysis.txt` plus SciPy, pandas, joblib, pytest and
[`xrd-preprocessing`](https://github.com/Eos-Dx/XRD-preprocessing). The latter
must be importable in the selected Python environment.

```bash
export CMAKE_PREFIX_PATH=/path/to/geant4-install
source /path/to/geant4-install/bin/geant4.sh
./scripts/build.sh
```

Inspect GPUs and select one when several Metal devices are available:

```bash
tutorial/saxs_keele/gpu_transport/multi_transport --list-devices
export KEELE_METAL_DEVICE_INDEX=0
```

## 1. Inspect and convert material form factors

```bash
python3 tutorial/saxs_keele/prepare_keele_ff.py \
  --input data/xrd_components.txt \
  --geant4-miff-dir "$G4LEDATA/penelope/rayleigh/MIFF" \
  --output-dir build/converted-ff
```

The input has three `(q, I)` pairs for fat, collagen and water. The output
has Geant4 `q/(mₑc), F/√W` pairs. Existing `tutorial/saxs_keele/data/`
files are frozen inputs used for the saved benchmark. Compare regenerated
files with them before changing a validation configuration.

## 2. Run the Geant4 CPU reference

The following macro uses a 1 mm water cylinder, 20 keV photons and the
project's MIFF. It is a short tutorial run, **not** the 100 mm parity geometry.

```bash
cd build/saxs_keele
./saxs keele_custom_water.mac 4
cd ../..
```

For the exact finite-box CPU reference and Metal comparison, run the next
command. The Python script creates temporary Geant4 macros, image-only CSV
output and Metal raw images, integrates them, writes JSON and deletes the
temporary images. It does not retain ROOT files.

## 3. Compare independent 100 mm water histories

```bash
PYTHON=python3 PHOTONS=2000000 THREADS=8 ./scripts/quick_compare.sh
```

The script uses the frozen 22.0220601 keV Penelope-shell physics fixture.
Outputs appear in `build/quick_compare/`. Use 200 M or more per backend for
a profile comparison similar in precision to the recorded validation, and
specify independent seed offsets when repeating runs. Runtime and outcome
depend on hardware and statistics; a small run is a functional check.

## 4. Probe Compton final states separately

```bash
python3 tutorial/saxs_keele/gpu_transport/validate_penelope_compton.py \
  --physics tutorial/saxs_keele/gpu_transport/results/multi_transport/water_22p022_shell_exact_xs_joint.json \
  --output-dir build/compton-probe
```

This samples the same number of independent Metal events as the Geant4
reference histogram stored in the physics fixture. It compares angle,
outgoing energy and their joint distribution. A non-small p-value only
states that the chosen histograms did not reveal a difference at that count.

## 5. Test a changed water/fat/collagen material

```bash
python3 tutorial/saxs_keele/gpu_transport/validate_multi_transport.py \
  --photons 20000000 --threads 14 --fractions 0.45 0.45 0.10 \
  --output-dir build/mixture-45-45-10
```

`--fractions` means **water, fat, collagen** in this command. The converter's
`--fractions` uses **fat, water, collagen**; the option names in each script
and this example make that difference explicit. The validator creates the
mixture MIFF, obtains a new Geant4 composition-dependent shell table,
simulates independent histories and compares the same detector observable.

## 6. Optional external water-reference comparison

```bash
python3 tutorial/saxs_keele/gpu_transport/compare_colleague_water_cpu_gpu.py \
  --colleague-file /path/to/master_water_20_30_40.joblib \
  --photons 50000000 --threads 14 \
  --output-dir build/colleague-water
```

The private joblib is deliberately absent. Its detector geometry and flux
are incompletely specified; see [validation](VALIDATION.md) before
interpreting agreement with this reference.

## 7. GPU-side radial output

`metal_radial.py` serializes a frozen sparse pyFAI operator and invokes the
same transport kernel with `--radial map.bin radial.raw`. This path returns
radial sums directly and avoids copying one packed result per photon to the
CPU. Keep image mode for detector-level validation; use radial mode for
repeated inverse-model trials after the operator and detector mask are frozen.

The reducer is exact for operators with at most three nonzero bin links per
pixel, matching the tested pyFAI pixel-splitting operator. Its output has been
checked against multiplication of the same operator by the full Metal image.

## 8. Build and compare the CUDA source on Apple silicon

After installing CuMetal, build the CUDA source as MSL and its macOS launcher:

```bash
export CUMETAL_PREFIX="$(brew --prefix cumetal)"
./scripts/build_cuda.sh --cumetal-only

python3 tutorial/saxs_keele/gpu_transport/compare_cuda_metal.py \
  --photons 1000000 \
  --output build/cuda-metal-validation.json
```

This uses the same g50 sample, geometry, seed and detector for native Metal and
CUDA-to-Metal. It checks event classes and every detector pixel. It is a
compiler/backend consistency test and does not replace the independent Geant4
comparison. Native CUDA build and run commands for NVIDIA Windows/Linux hosts
are in [CUDA.md](CUDA.md).
