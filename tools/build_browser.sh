#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
source work/browser/emsdk/emsdk_env.sh
export MUJOCO_STATIC_LINK_DIR="$project_dir/work/browser/build-mujoco-3.9.0/lib"
export CARGO_TARGET_DIR="$project_dir/work/browser/rust-target"
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128"
exports='["_main","_fb_init","_fb_step","_fb_frame","_fb_command","_fb_set_vision","_fb_update_retina_display","_fb_scene","_fb_response_ptr","_fb_response_len","_fb_poses_ptr","_fb_poses_len","_fb_vision_ptr","_fb_display_ptr","_fb_metrics_ptr"]'
cargo rustc --locked --release --target wasm32-unknown-emscripten --bin flybrain-browser -- \
  -C link-arg=-Wl,--whole-archive -C "link-arg=$MUJOCO_STATIC_LINK_DIR/libmujoco.a" -C link-arg=-Wl,--no-whole-archive \
  -C link-arg=--js-library -C "link-arg=$project_dir/web/emscripten-gpu.js" \
  -C link-arg=-sASYNCIFY=1 -C 'link-arg=-sASYNCIFY_IMPORTS=["fly_gpu_create","fly_gpu_window"]' \
  -C 'link-arg=-sASYNCIFY_REMOVE=["mj_*","mju_*","mjd_*"]' \
  -C link-arg=-sASYNCIFY_STACK_SIZE=1048576 \
  -C link-arg=-sALLOW_MEMORY_GROWTH=1 -C link-arg=-sMAXIMUM_MEMORY=2147483648 \
  -C link-arg=-sSTACK_SIZE=8388608 -C link-arg=-sINITIAL_MEMORY=67108864 \
  -C link-arg=-sMODULARIZE=1 -C link-arg=-sEXPORT_ES6=1 -C link-arg=-sEXPORT_NAME=createFlyBrain \
  -C link-arg=-sENVIRONMENT=web,worker,node -C link-arg=-sFORCE_FILESYSTEM=1 \
  -C "link-arg=-sEXPORTED_FUNCTIONS=$exports" \
  -C 'link-arg=-sEXPORTED_RUNTIME_METHODS=["FS","ccall","HEAPU8","HEAPF32"]'
mkdir -p web/dist
.venv/bin/python tools/package_browser_assets.py
cp "$CARGO_TARGET_DIR/wasm32-unknown-emscripten/release/flybrain-browser.js" web/dist/flybrain.js
cp "$CARGO_TARGET_DIR/wasm32-unknown-emscripten/release/flybrain_browser.wasm" web/dist/flybrain_browser.wasm
printf 'Browser WASM built. Run: cd web && npm start\n'
