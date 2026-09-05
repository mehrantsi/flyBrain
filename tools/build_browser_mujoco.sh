#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
emsdk_dir="$root_dir/work/browser/emsdk"
source_dir="$root_dir/work/browser/mujoco-3.9.0"
build_dir="$root_dir/work/browser/build-mujoco-3.9.0"

source "$emsdk_dir/emsdk_env.sh" >/dev/null
emcmake cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS=-msimd128 \
  -DCMAKE_CXX_FLAGS=-msimd128 \
  -DMUJOCO_BUILD_TESTS_WASM=OFF \
  -DMUJOCO_WASM_THREADS=OFF \
  -DMUJOCO_BUILD_STUDIO=OFF \
  -DMUJOCO_USE_FILAMENT=OFF \
  -DMUJOCO_ENABLE_AVX=OFF \
  -DMUJOCO_ENABLE_AVX_INTRINSICS=OFF
cmake --build "$build_dir" --target mujoco --parallel 4
