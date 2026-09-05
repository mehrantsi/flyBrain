# Browser performance and touchdown verification

Measured locally on the M3 Max in the Codex Chromium browser, September 5, 2026.
These are local observations, not cross-machine guarantees.

## Performance

The preserved runtime and optimized runtime both advance the complete imported
MaleCNS model at the existing neural timestep, with the same MuJoCo body, 0.1 ms
physics timestep and 500 Hz sensor/controller exchange. No neurons, contacts,
physics steps or controller updates were removed to improve throughput.

| Measurement | Realtime factor |
|---|---:|
| Preserved runtime, rendered, 1,000 control windows | 0.100× |
| Final runtime, rendered, 1,000 windows, GPU timestamps enabled | 0.407× |
| Final runtime, rendered, 10,000 windows, timestamps disabled | 0.400× |
| Final runtime, no observer or supplied retina frames, 1,000 windows | 0.562× |

The short rendered comparison is approximately 4× faster. **The 0.5× sustained live-view
target has not been reached.** The no-observer number is not a live-rendering
claim. The two-second trajectory is before the landing correction changes the
body trajectory: both versions have 2,071,162 spikes and exactly the same
SHA-256 of `JSON.stringify([qpos,qvel,total_spikes])`:
`cea2324cc5a5923854f279c41e07ea48487836f9531882132836d7133a6125da`.
That check is a trajectory regression, not proof of general numerical or
biological equivalence.

The restored normal viewer reported 0.39–0.44× at 14–33 simulated seconds,
including a 0.44× observation with both Retina and Field panels enabled.
Those HUD observations are separate from the controlled benchmark table; they
are not a sustained 0.5× claim.

The final 20-second corrected run averaged 4.88 ms per 2 ms simulation window:
2.36 ms in the brain bridge, including 2.15 ms GPU submission/readback wait.
Approximately 2.53 ms remains in the physical world and other controller work.
JavaScript command encoding averaged 0.044 ms. Retina sampling averaged
0.066 ms per window, amortized over its existing wall-clock cadence.
The separate timestamp-profiled two-second run measured 0.66 ms of GPU compute;
GPU compute is contained in the wait, not additional to it. Profiling is optional
and disabled in the normal viewer and the final sustained benchmark.
The remaining cost is not Tokio, a Python bridge, or locks: neither Tokio nor
Python participates in this browser loop.

Implemented changes:

- Build both Rust and MuJoCo with WASM SIMD. Exclude synchronous `mj_*`, `mju_*`
  and `mjd_*` physics functions from Asyncify instrumentation. They must remain
  synchronous: do not introduce suspending GPU calls in MuJoCo callbacks.
- Queue only delayed firing sources with outgoing edges; dispatch a workgroup
  per source, sharing its outgoing edges among 256 lanes. Ping-pong queues
  preserve continuity across odd-length windows. Grid-stride processing handles
  more than 65,535 simultaneous sources without exceeding WebGPU dispatch limits.
- Load source metadata once per workgroup through synchronized workgroup memory,
  instead of issuing redundant atomic loads from every lane. This final change
  reduced short-run GPU compute from approximately 2.09 ms to 0.66 ms and improved
  the unprofiled 20-second rendered run from 0.325× to 0.400×, retaining its final
  state checksum and spike count.
- Keep exact signed 16-bit synapse counts packed in 32-bit storage words,
  saving about 49 MB relative to widening every count to `i32`. Arrivals still
  use integer atomic accumulation. Fuse external-event application with
  propagation; skip arithmetic only at exact rest/zero-arrival states.
- Generate frame JSON at presentation cadence instead of every 500 Hz window.
  Read benchmark timing through a small fixed WASM buffer.
- Precompute retinal sampling taps without changing floating-point summation
  order. Hidden retina panels no longer generate/copy display images; both eyes
  still supply neural sensory input. Read both eye buffers asynchronously,
  reusing one shadow pass for the binocular pair.
- Redraw only changed observer frames, reuse identical materials/textures, and
  avoid revalidating/copying all actuator controls when updating a validated
  subset.

Additional experiments were rejected rather than silently retained: switching
the observer to Three.js WebGPU did not remove the compute penalty; larger
512-lane source tiles did not improve over 256; native WASM stack switching
hit `SuspendError` through this pinned Rust/Emscripten JS exception trampoline.
Rust's alternative [Emscripten WASM exception ABI](https://doc.rust-lang.org/stable/unstable-book/compiler-flags/emscripten-wasm-eh.html)
is a separate compiler/toolchain change, not an enabled runtime fallback.

## Landing correction

The original trace enters `Landing` around 5.0 s, bounces near the surface with
wing amplitude returning toward one, and reaches `Grounded` only around 8.2 s.
The previous support-transfer integrator reversed on every contact gap and
retained 20% wing amplitude even at full transfer. The instantaneous
velocity/contact guard then struggled to classify a vibrating body as grounded.

The corrected handoff requires at least two contacted feet to start transferring
support, tolerates up to 20 ms of contact chatter, and tapers wing amplitude to
zero over the existing 80 ms transfer interval. Sustained loss of support restores
wing power. The existing three-foot, low-velocity criteria still determine
`Grounded`. A single-foot brush does not shut the wings off. No new feeding,
grooming, or navigation routine was added, and landing intent still comes through
the existing neural/foraging integration. This support adapter is engineered,
not a claim of a recovered biological landing circuit.

In the full-CNS rendered repeat, `Grounded` occurs at approximately 5.4 s and
feeding begins at 6.6 s. All 30 sampled feeding frames have zero wing amplitude
and retracted targets. The largest measured wing-joint speed during those
feeding samples is 0.030 rad/s, versus roughly 1,000–2,000 rad/s while flying.
After feeding, the fly takes off and reaches 186 mm altitude within the room.
All MuJoCo warning counters remain zero.

The trace now records contacts, wing commands, actual wing joint positions and
velocities, landing drive and flight motor power in `flight_diagnostics`.
Native `cns-check` reports include the corresponding wing fields too.

## Reproduction and checks

Start the local server and open `/perf-test.html`. Stop other simulations,
including paused tabs that may still animate. Choose 1,000 or 10,000 windows,
then `Full CNS, rendered`; the page exposes timing distributions, a trajectory
checksum and a 100 ms sampled trace. The baseline checkbox loads the preserved
WASM and neural engine under `/baseline/`, with the same current renderer.
Leave `GPU timestamps (profiling overhead)` unchecked for throughput measurements;
enable it separately to measure GPU execution. Absent or disabled GPU
instrumentation is reported as zero by the diagnostic harness, not evidence that
GPU cost is zero. Retina sampling is wall-clock based,
so rendered runs are not guaranteed to have identical visual schedules.

The WebGPU QA page checks phase, refractory timing, zero/nonzero delays,
inhibition, silencing, split-window continuity, high fan-out, empty ticks,
65,537 simultaneous firing sources with uneven fan-out and the full signed
`i16` range. The extreme-weight burst uses an exactly representable 0.25 mV
scale to isolate scheduling/accumulation from float32 versus float64 rounding;
the ordinary fixtures retain their original parameters and tolerances. Retinal
tests check every ommatidium against the original sampling order as well as the
pinned FlyGym display hash. Scene tests exercise asynchronous left/right
pairing, row orientation, pending-frame retention and in-flight disposal.
Landing/contact-loss tests retain the pre-existing physical stability limits.

Compact measured artifacts are in `outputs/performance/`, particularly
`browser-rendered-baseline.json`, `browser-rendered-shared-metadata.json`,
`browser-rendered-final.json`, `landing-before.json` and `landing-after.json`.
`browser-rendered-optimized.json` records the earlier optimization checkpoint,
before the final shared-metadata improvement.
The earlier exploratory `browser-baseline.json` had competing rendering and
must not be used as the matched baseline.
