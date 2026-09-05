# FlyBrain world v2: corrected sugar-to-feeding pathway

The silent motor output was a decoder-selection error, not a neural-engine failure. The original
world stimulated sugar gustatory receptor neurons but monitored DNa02L/R, locomotor steering
neurons from a different experimental context. DNa02 remained silent in v630, v783, and the
upstream Brian2 sugar results even though the input IDs, Metal event delivery, recurrent state, and
probe readback were correct.

The published sugar experiment instead validates contralateral MN9
(`720575940660219265`), a rostrum motor neuron used as a readout of proboscis extension. A native
150 Hz v783 diagnostic produced 100 MN9 spikes/s while both DNa02 neurons remained at zero. This is
the pathway implemented in world v2.

## Corrected runtime path

1. Rust loads the FlyWire v783 CSR pack and runs 138,639 point neurons on Metal.
2. MuJoCo runs the exported NeuroMechFly body at the same 0.1 ms timestep.
3. Mouth-to-food proximity drives the 20 available published right sugar GRNs at 150 Hz.
4. The full recurrent network activates contralateral MN9.
5. MN9 spikes enter a bounded leaky motor command that actuates rostrum pitch.
6. Feeding slows and freezes the FlyGym-derived gait phase so the fly remains at the food.
7. The brain-actuated rostrum and haustellum are blue-highlighted in the render.

## Final render

- 3.0 biological seconds and 30,000 coupled brain/physics ticks
- 138,639-neuron v783 brain; 97,206,574 bytes of explicit Metal buffers
- 8,940 external sugar events and 291 MN9 spikes
- first taste window: 16 ms
- first MN9 spike and feeding command: 48 ms
- measured taste-to-MN9 response latency: 32 ms
- peak rostrum command: 1.0; final command: 0.938
- final mouth-to-food distance: 0.516, inside the 0.75 taste radius
- Metal brain time: 2.47 seconds; complete wall time: 5.67 seconds
- 90 frames, 960×720, H.264/yuv420p, 30 fps
- video SHA-256: `6328de539587a3d3f7aef288ea2b4437810a595ceb05c6783ae0ed8b9cef481b`

## Verification

- 27 Rust tests passed, including Brian-validated Metal fixtures and streaming-state continuity.
- 34 independent Python/Brian2/MLX tests passed.
- Rust clippy passed with warnings denied.
- A 1,000-step Rust/Python MuJoCo replay matched exactly for time, qpos, qvel, contacts, and hashes.
- The sensorimotor verifier passed exact-ID, event, MN9-response, response-ordering, actuator, and
  final food-proximity checks.
- ffprobe reported H.264, 960×720, yuv420p, 30 fps, exactly 90 frames and 3.0 seconds.

## Scientific boundary

Sugar-to-MN9 is the readout validated by the published computational and experimental work. The
geometric taste transducer, spike-to-rostrum transfer function, gait arbitration, and FlyGym gait
remain explicit engineering components. This is a working, auditable sensorimotor experiment; it
does not recover the complete VNC, muscles, or donor fly state.
