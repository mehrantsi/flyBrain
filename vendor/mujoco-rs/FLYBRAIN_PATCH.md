# FlyBrain MuJoCo binding patches

This vendored copy is `mujoco-rs` 5.0.0 for MuJoCo 3.9.0. FlyBrain changes
the macOS offscreen context path. The upstream winit/glutin fallback creates a
context on this machine that MuJoCo rejects for lacking
`ARB_framebuffer_object`. FlyBrain instead opens an invisible GLFW context,
matching the official MuJoCo Python package's macOS strategy. Other platforms
retain the upstream paths.

The GLFW dynamic library comes from the pinned FlyGym/MuJoCo Python environment
and is linked into `work/mujoco/lib` by `tools/setup_mujoco_runtime.py`.

## Emscripten linkage

The browser build uses Emscripten 4.0.10 and a static MuJoCo 3.9.0 archive.
MuJoCo registers its STL and OBJ mesh decoders through static constructors.
Cargo's native-library metadata travels through Rust rlibs, where the
`+whole-archive` modifier does not reliably reach the final Emscripten command.
For `wasm32-unknown-emscripten`, `build.rs` therefore does not emit the MuJoCo
archive as a dependency library. `tools/build_browser.sh` supplies the absolute
`libmujoco.a` itself between `-Wl,--whole-archive` and
`-Wl,--no-whole-archive`, matching MuJoCo's own `wasm/CMakeLists.txt`.

The remaining static dependencies are still emitted by this build script. This
is required for the real `c_thorax.stl` body asset to load; removing the final
archive wrapping can build a smaller WASM module that fails at runtime with no
STL decoder.

`MjRenderer::scene_mut()` also exposes the synchronized scene for the shared
live/offscreen room presentation. RGB rendering adjusts observer clip planes and
hides obstructing room-shell visuals after `sync_data`; physics and eye scenes are
separate. The underlying renderer's depth conversion still uses model clip planes,
so callers overriding scene clip planes must not use its cached depth conversion.
