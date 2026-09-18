#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${G4LEDATA:?Source your Geant4 environment before running this comparison}"
python="${PYTHON:-python3}"
photons="${PHOTONS:-2000000}"
threads="${THREADS:-8}"

"$python" "$root/tutorial/saxs_keele/gpu_transport/validate_water_100mm_cpu_metal.py" \
  --photons "$photons" --repetitions 1 --threads "$threads" \
  --metal-physics "$root/tutorial/saxs_keele/gpu_transport/results/multi_transport/water_22p022_shell_exact_xs_joint.json" \
  --output-dir "$root/build/quick_compare"
