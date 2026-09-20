# X-ray diffraction example suite

This suite compares independent Geant4 11.4.2 CPU and Apple Metal GPU photon
histories for molecular-interference X-ray diffraction. It covers pure water,
fat and collagen; every binary 50/50 mixture; an equal ternary mixture; the
reference g50 ternary mixture; and 10–100 mm thickness series.

Each maintained result uses **100,000,000 incident photons per backend**. All
cases use 22.162917 keV photons, a 120 × 120 mm finite sample, a focused
200 µm diameter source, 150 mm from the mean scattering plane to an ideal
1000 × 1000 detector, and 100 µm pixels. Changing thickness changes the
downstream air gap to retain the 150 mm mean-plane distance.

## Run

```bash
source /path/to/geant4-install/bin/geant4.sh
./scripts/build.sh

python examples/xrd_suite/run_suite.py \
  --photons 100000000 \
  --workers 2 \
  --threads-per-case 7 \
  --output examples/xrd_suite/results_100m
```

Completed cases are resumable. Select individual cases with repeated
`--case NAME` options. Use `--keep-arrays` to retain compressed detector
arrays locally; these large intermediate arrays are excluded from Git.

Regenerate the aggregate report and figures without rerunning transport:

```bash
python examples/xrd_suite/compare_suite.py \
  --results examples/xrd_suite/results_100m
```

## Outputs

Each case directory contains:

- `summary.json`: geometry, composition, channel counts, timing and parity;
- `profiles.csv`: Geant4 and Metal azimuthal sums on the same q grid;
- `detector_comparison.png`: log detector maps, radial profiles and residuals;
- `input.json`: frozen case definition and random seeds.

The result root contains `RESULTS.md`, `summary.csv`, `profile_overview.png`
and `parity_overview.png`. The figures show independent Monte Carlo agreement
for the stated model. They do not establish detector-response, dose or clinical
equivalence. Metal currently implements a finite homogeneous box and ideal
entrance-counting detector; see [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).
