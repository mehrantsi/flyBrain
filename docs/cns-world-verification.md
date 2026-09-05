# MaleCNS world gate

`tools/verify_cns_world.py` runs the headless `cns-check` command twice using
the live `SimulationStepper`: once intact and once with
`--disconnect-motor-outputs`.

```sh
.venv/bin/python tools/verify_cns_world.py \
  --rust-bin target/release/flybrain-world \
  --cns-pack outputs/packs/male_cns_v1 \
  --duration-seconds 2 \
  --control-hz 500 \
  --start-food-distance 40 \
  --output outputs/world/cns-world-verification.json
```

There is no seed argument. The world and fractional-rate sensory encoder use
the deterministic seeds in the runtime. `--parameters` is optional and is
passed to both runs. For the longer wall-release regression, use
`--duration-seconds 30 --start-food-distance 40`.

Already-completed reports can be checked without rerunning the world:

```sh
.venv/bin/python tools/verify_cns_world.py \
  --cns-pack outputs/packs/male_cns_v1 \
  --duration-seconds 30 --control-hz 500 \
  --intact-report outputs/cns/world-verified/intact.json \
  --disconnected-report outputs/cns/world-verified/disconnected.json
```

`--intact-report` and `--disconnected-report` must be supplied together. An
existing verifier output is rejected rather than overwritten.

## Report contract

The Rust command writes schema `flybrain.cns-world-check`. The gate consumes
these fields from that report:

- `brain.model`, `brain.neurons`, `brain.sensory_neurons`, and
  `brain.motor_outputs_connected` identify the whole MaleCNS and output state.
- `initial_state` is the runtime-provided initial-state description. Its
  `summary.initial_state_sha256` binds the pack array hashes, neural-I/O hash,
  world asset hashes, initial body position, parameters, and sensory setup.
  The motor-output toggle is excluded so the two runs are matched.
- `summary.population_spikes` and `summary.motor_output_spikes` are whole-CNS
  and annotated motor-pool counts. `summary.motor_output_source` must be
  `whole-cns-spikes` or `disconnected` for the corresponding run.
- `summary.forward_flight_distance_mm` is integrated body-forward velocity
  only while physically airborne above the 3 mm takeoff margin. Altitude and
  boundary metrics are computed by the verifier from
  `summary.minimum_position_mm`/`maximum_position_mm` and top-level
  `room_bounds_mm` (`[-300,-220,0]` to `[300,220,220]` in the default room).
- Every sample must carry `time_seconds`, `root_position`, `flight_mode`,
  `population_spike_delta`, `collision_reflex_active`, and the serialized
  `cns_motor` readout. The latter includes `spike_delta`, bilateral rate
  arrays, bounded activations/steering, and `outputs_connected`.

The intact run must contain whole-CNS spikes, motor-pool spikes and positive
CNS motor activation, enter `TAKEOFF` or `CRUISE`, move forward while
airborne, vary altitude, and remain inside the room. The verifier also limits
one continuous collision-reflex episode to five seconds, catching a trapped
wall-release controller despite an initially successful flight. The lesion
run must retain neural and motor-pool spikes while its body mapping reports
zero activation.

The paired comparison requires equal `initial_state_sha256`, equal initial
body position, and a final-position change of at least 0.25 mm after motor
outputs are disconnected. Closed-loop sensory traces are allowed—and
expected—to diverge once the bodies take different trajectories; they are not
compared for equality. The gate does not require a complete biological flight,
landing, feeding, or retinal-transduction model.
