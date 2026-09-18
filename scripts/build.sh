#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cmake -S "$root/tutorial/saxs_keele" -B "$root/build/saxs_keele" \
  -DWITH_GEANT4_UIVIS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build/saxs_keele" -j "$(sysctl -n hw.ncpu)"

swiftc -O -framework Metal \
  "$root/tutorial/saxs_keele/gpu_transport/multi_transport.swift" \
  -o "$root/tutorial/saxs_keele/gpu_transport/multi_transport"
swiftc -O -framework Metal \
  "$root/tutorial/saxs_keele/gpu_transport/single_rayleigh.swift" \
  -o "$root/tutorial/saxs_keele/gpu_transport/single_rayleigh"

echo "Built Geant4 SAXS reference and Metal transport in $root"
