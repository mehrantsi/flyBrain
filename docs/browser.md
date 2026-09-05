# Browser runtime

The browser runtime advances the same Rust `SimulationStepper` and MuJoCo body
model as the native world. It runs Rust and MuJoCo 3.9.0 as one
`wasm32-unknown-emscripten` module, then runs the full MaleCNS neural update on
WebGPU. It can run locally or from the [Cloudflare static deployment](cloudflare.md).
Neither mode is a pre-recorded or server-streamed simulation.

`Start full CNS` loads `outputs/packs/male_cns_v1`; `World-only diagnostic`
passes no pack to the same world/controller stack and deliberately disables the
neural bridge. The latter is a physics, asset, and rendering diagnostic only;
it is not evidence of neural control.

## Pinned inputs

- Rust crates are pinned by `Cargo.lock`; the crate requires Rust 1.85.
- Emscripten SDK is 4.0.10, checkout
  `62a853cd3b3134398ce85cde8bb5cbb2ef0194cb`.
- MuJoCo source is exactly 3.9.0, checkout
  `237c17e48539b6c90bf90d3161547cbdcbfaa1e0`.
- The vendored binding is `mujoco-rs` 5.0.0 for MuJoCo 3.9.0; local deltas are
  described in [the patch record](../vendor/mujoco-rs/FLYBRAIN_PATCH.md).
- The browser scene uses Three.js 0.180.0, pinned in `web/package-lock.json`.
- Runtime body/room assets and each connectome file are SHA-256 verified before
  writing to the Emscripten filesystem.

The MaleCNS pack download is currently 145,348 KiB: 97,877,776 bytes of
destinations, 48,938,952 bytes of signed counts, plus IDs and CSR offsets. The
browser packs pairs of signed `i16` counts into storage words and sign-extends
them for integer atomic accumulation. Destination and weight storage together
are about 147 MB before neural state, delay-ring, telemetry, and browser/WASM
memory. The QA page checks the standard 128 MiB storage-binding and 256 MiB
buffer limits; the approximately 98 MB destination and 49 MB weight arrays are
separate bindings.

## Build and run locally

The checked-out development prerequisites live under `work/browser`: Emscripten
SDK 4.0.10 at `work/browser/emsdk` and the exact MuJoCo checkout at
`work/browser/mujoco-3.9.0`. Rebuild the static MuJoCo archive, then the Rust
WASM module and browser assets:

```bash
tools/build_browser_mujoco.sh
tools/build_browser.sh
cd web
npm start
```

If the bundle is already built, only run:

```bash
cd web
npm start
```

Open the local address printed by `npm start`. The server deliberately serves
only loopback and enables cross-origin isolation for the worker. The worker
verifies the fetched body/room and connectome file hashes before writing them
to the Emscripten filesystem. This is not deployment configuration.
`web/neural-test.html` is a separate WebGPU QA page for the checked-in tiny
parity fixture.

## Viewer controls

The observer starts 36 mm from the fly. Drag to orbit; two-finger trackpad
scroll or pinch zooms in and out. Chase follows the fly's translation while
preserving your zoom and angle. Room resets to the room overview; Orbit leaves
the camera independent of the fly. These observer controls do not change the
sensory eye cameras.

The bottom transport provides pause, reset, sugar placement, and camera modes.
Inspector, Retina and Field open by default on desktop when a simulation starts.
On mobile (a viewport at most 760 px wide or a touch-first device), only Retina
opens by default. All three can still be toggled independently; resizing does
not override those choices. Retina
shows the left and right processed compound-eye inputs side by side. Field is a
network-activity proxy, not a measured biological EEG. Inspector opens detailed
neural/sensory telemetry and runtime counters; Hide UI leaves a restore control
in the top bar.

Grooming pose test in Inspector is explicitly procedural. Brain-connected runs
do not invoke it on an idle timer or after feeding. Candidate grooming neurons
are in the pack, but a validated direct grooming pathway and actuator mapping
remain missing; see [the neural grooming boundary](neural-grooming.md).

Full-CNS browser checks after optimization run around 0.40× with rendering
on the local M3 Max, versus 0.10× for the matched preserved runtime. The 0.5×
live-view target is not yet met. The UI reports the actual ratio. See the
[measurement and touchdown report](browser-performance.md) for test scope,
correctness checks and remaining bottlenecks; these are local observations,
not a claim of performance parity with native Metal.

Wall touchdowns use a wall-relative attitude and contact/adhesion handoff; see
[wall-landing verification](wall-landing.md). Reload the page after updating the
WASM build; Reset restarts the already loaded runtime.

`tools/build_browser.sh` links the exact MuJoCo archive between
`--whole-archive` and `--no-whole-archive`, preserving the static STL and OBJ
decoder registrations required by the articulated body assets.

For a headless world-only smoke check:

```bash
node web/smoke-test.mjs outputs/world/browser-physics-smoke.json
```

It loads the real body and room, advances ten 2 ms controller windows, checks a
deterministic reset, and runs both retina paths. It does not load or exercise a
connectome.

## Current numerical scope

The browser smoke trace was compared to
`outputs/world/browser-physics-native.json`, generated by the same 11-frame,
world-only command sequence. The raw `qpos` and `qvel` vectors have 133 and 132
entries respectively. The table gives the maximum absolute component difference
for each frame; units remain MuJoCo's model units (millimetres and derived
velocity units in this model).

| Window | Time (s) | max abs Δqpos | max abs Δqvel |
|---:|---:|---:|---:|
| 0 | 0.000 | 0 | 0 |
| 1 | 0.002 | 1.11e-16 | 9.97e-14 |
| 2 | 0.004 | 3.33e-16 | 9.66e-14 |
| 3 | 0.006 | 6.66e-16 | 9.53e-14 |
| 4 | 0.008 | 3.76e-16 | 9.37e-14 |
| 5 | 0.010 | 4.68e-16 | 9.24e-14 |
| 6 | 0.012 | 5.44e-12 | 2.24e-8 |
| 7 | 0.014 | 4.70e-11 | 2.55e-8 |
| 8 | 0.016 | 7.15e-11 | 7.88e-9 |
| 9 | 0.018 | 7.47e-11 | 3.53e-9 |
| 10 | 0.020 | 7.38e-11 | 2.19e-9 |

Across all frames the largest `qpos` delta is 7.47e-11 at window 9 and the
largest `qvel` delta is 2.55e-8 at window 7. This is a short, no-neural-input
trajectory comparison, not an assertion that browser and native trajectories
will remain bitwise equal, or within these bounds, under long contacts or full
CNS activity. It does establish that the browser is advancing the actual model
rather than a scene substitute.

Neural state is intentionally `f32` in WebGPU, just as the native compute path
is Metal `f32`; threshold timing and reduction order must therefore be assessed
with explicit fixture tolerances rather than a claim of bit parity. The browser
has no path to the Apple Neural Engine/NPU. Its physics, Rust controllers,
MuJoCo model, body assets, and controller timing are shared with native; only
the neural compute backend and browser presentation differ.

The scene renderer produces binocular raw eye images, but the worker limits
retina transfer and processing to 15 Hz wall time. Native viewer sampling is
scheduled by its own rendering loop, so browser and native visual sampling are
not a matched neural experiment. The current brain embodiment also continues to
use the documented engineered sensory and motor interfaces; this port does not
claim to recover biological vision, navigation, landing, or feeding circuitry.
