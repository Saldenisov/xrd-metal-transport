#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_dir="$root/tutorial/saxs_keele/gpu_transport/CUDA"
output_dir="$root/tutorial/saxs_keele/gpu_transport"
mode="${1:---all}"

build_nvidia() {
  if ! command -v nvcc >/dev/null 2>&1; then
    if [[ "$mode" == "--cuda-only" ]]; then
      echo "nvcc was not found" >&2
      return 1
    fi
    echo "Skipping native CUDA build: nvcc was not found"
    return 0
  fi
  nvcc -O3 -std=c++17 "$cuda_dir/Transport.cu" -o "$output_dir/cuda_transport"
  echo "Built NVIDIA CUDA transport: $output_dir/cuda_transport"
}

build_cumetal() {
  local compiler="${CUMETALC:-}"
  local include_dir="${CUMETAL_INCLUDE:-}"
  local library_dir="${CUMETAL_LIB:-}"
  if [[ -z "$compiler" && -n "${CUMETAL_BUILD:-}" ]]; then
    compiler="$CUMETAL_BUILD/cumetalc"
    library_dir="${library_dir:-$CUMETAL_BUILD}"
  fi
  if [[ -z "$include_dir" && -n "${CUMETAL_SOURCE:-}" ]]; then
    include_dir="$CUMETAL_SOURCE/runtime/api"
  fi
  if [[ -z "$compiler" ]] && command -v cumetalc >/dev/null 2>&1; then
    compiler="$(command -v cumetalc)"
  fi
  if [[ -z "$include_dir" || -z "$library_dir" ]]; then
    local prefix="${CUMETAL_PREFIX:-}"
    if [[ -z "$prefix" ]] && command -v brew >/dev/null 2>&1; then
      prefix="$(brew --prefix cumetal 2>/dev/null || true)"
    fi
    if [[ -n "$prefix" ]]; then
      include_dir="${include_dir:-$prefix/include}"
      library_dir="${library_dir:-$prefix/lib}"
    fi
  fi
  if [[ -z "$compiler" || ! -x "$compiler" || -z "$include_dir" ||
        ! -f "$include_dir/cuda.h" || -z "$library_dir" ||
        ! -f "$library_dir/libcumetal.dylib" ]]; then
    if [[ "$mode" == "--cumetal-only" ]]; then
      echo "CuMetal compiler, headers, or library were not found" >&2
      return 1
    fi
    echo "Skipping CuMetal build: set CUMETALC, CUMETAL_INCLUDE, and CUMETAL_LIB"
    return 0
  fi

  local metal="$output_dir/xrd_transport.cumetal.metal"
  "$compiler" "$cuda_dir/Transport.cu" -DXRDCUDA_DEVICE_ONLY=1 \
    --backend=cumetal-ir --emit=msl -o "$metal" --overwrite
  cat > "$metal.cumetal-abi" <<'EOF'
CUMETAL_ABI_V2
kernel xrd_transport
arg buffer 8
arg buffer 8
arg buffer 8
arg buffer 8
arg bytes 4
arg bytes 4
arg bytes 4
arg buffer 8
arg buffer 8
arg buffer 8
arg buffer 8
shared 0
EOF
  clang++ -O3 -std=c++17 "$cuda_dir/CuMetalRunner.cpp" -I "$include_dir" \
    -L "$library_dir" -lcumetal -Wl,-rpath,"$library_dir" \
    -o "$output_dir/cumetal_transport"
  echo "Built CUDA-to-Metal transport: $output_dir/cumetal_transport"
}

case "$mode" in
  --all) build_nvidia; build_cumetal ;;
  --cuda-only) build_nvidia ;;
  --cumetal-only) build_cumetal ;;
  *) echo "usage: $0 [--all|--cuda-only|--cumetal-only]" >&2; exit 2 ;;
esac
