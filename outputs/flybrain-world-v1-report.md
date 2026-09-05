# FlyBrain world v1

The first native embodied run completed successfully on an Apple M3 Max.

## Render

- 3.0 biological seconds, 30,000 physics/brain ticks
- 90 frames, 960×720, H.264/yuv420p, 30 fps
- 5.88 seconds total wall time
- 2.41 seconds spent in the v783 Metal brain windows
- 97,206,574 bytes of explicit Metal buffers
- video SHA-256: `9b72c7b944e3215b0670368f9fcfdf4e7ac0f0fe728823018aac53c0aec0bb06`

## Runtime path

1. Rust loads the hashed FlyWire v783 CSR pack and runs 138,639 point neurons on Metal.
2. Rust runs the exported NeuroMechFly body in MuJoCo 3.9 at the same 0.1 ms timestep.
3. A 500 Hz bridge converts geometric food contact into deterministic taste events for 20 published
   right sugar GRNs.
4. DNa02R/L spike rates are filtered and exposed as differential steering input.
5. Rust interpolates FlyGym's recorded seven-joint stepping trajectories and adhesion states.
6. A native hidden GLFW/OpenGL context renders the fixed tracking camera and streams RGB to ffmpeg.

## Verification

- 26 Rust tests pass, including Brian-validated Metal fixtures and window-state continuity.
- 34 independent Python/Brian2/MLX tests pass.
- A 1,000-step Rust/Python MuJoCo replay is exact for time, qpos, qvel, contact values, and hashes.
- ffprobe reports H.264, 960×720, yuv420p, 30 fps, exactly 90 decoded frames and 3.0 seconds.

## Scientific boundary

The geometric taste encoder and DNa02-to-gait decoder are explicit engineering hypotheses, not
validated biological interfaces. DNa02R/L produced zero spikes in this sugar trial, so the visible
walking comes from the FlyGym-derived baseline VNC surrogate. This demonstrates a reproducible
closed-loop platform; it does not demonstrate an uploaded fly or recovered whole-animal behavior.
