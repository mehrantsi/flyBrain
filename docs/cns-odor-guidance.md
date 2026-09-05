# CNS odor-guidance integration

This work addresses the failed autonomous food-discovery check recorded in
[the earlier calibration report](cns-foraging-calibration.md). It is an engineered
sensory-to-motor decoder, **not** a recovered biological food-search circuit.
The default-start paired engineering acceptance gate passes. This does not establish
general room-foraging reliability or biological behavioral accuracy; the development
failures below remain part of the record.

## Verified default-start trial

`outputs/cns/odor-guidance/verification-v3.json` checks a 20-second intact run against
12-second odor-evoked-input and motor-output disconnection controls. All three use
the same executable, assets, initial pose and normalized parameter hash.

| Condition | CNS population spikes | Contiguous supported feeding | Flight |
|---|---:|---:|---|
| Intact | 21,644,339 | 2.976 s, from 8.858–11.834 s | Departure at 13.298 s; peak altitude 189.874 mm |
| Odor-evoked input disconnected | 12,585,087 | None; guidance never active | Other sensory/motor pathways remain active |
| Motor output disconnected | 12,571,427 | None | No flight; motor commands remain zero |

Every accepted feeding sample has six supporting feet, physical taste, a feeding
state, MN9 activity and extension above 0.1. The minimum body up-axis component is
0.928768. A brief taste, an upside-down posture, or a terminal upright pose alone
cannot pass the verifier. The controls cover the intact run's first feeding interval
but not its full 20-second duration.

The promoted defaults are approach speed scale 0.02 and close-odor threshold 1 ppm
in the existing receptor model's units. These are engineering calibration values.

```sh
target/release/flybrain-world cns-check --duration-seconds 20 --output outputs/cns/intact.json
target/release/flybrain-world cns-check --duration-seconds 12 --disconnect-olfactory-evoked-inputs --output outputs/cns/odor-control.json
target/release/flybrain-world cns-check --duration-seconds 12 --disconnect-motor-outputs --output outputs/cns/motor-control.json
.venv/bin/python tools/verify_cns_odor_guidance.py \
  --intact-report outputs/cns/intact.json \
  --odor-evoked-disconnected-report outputs/cns/odor-control.json \
  --motor-output-disconnected-report outputs/cns/motor-control.json \
  --output outputs/cns/odor-verification.json
```

Output paths must not already exist. Restart the viewer to load the rebuilt binary
and current defaults: `target/release/flybrain-world view`.

## Changed-heading check and limits

The same calibration with `--initial-yaw-deg 15` initially exposed a meal-clock bug:
tasting during descent spent most of the meal before grounding. The preserved
`heldout-yaw15-v1.json` run has only 118 ms of supported feeding. After separating
airborne sensory taste from the supported meal timer, `heldout-yaw15-v4.json` has a
2.988-second contiguous feeding bout (8.330–11.318 s), 2–4 supporting feet, minimum
body up-axis 0.974350, departure at 12.782 s, and a 188.541 mm peak altitude. No
heading-specific gains or food coordinates were introduced. This is a second start,
not a statistical validation across room layouts, winds, hunger states or donors.

The earlier `heldout-yaw15-v2.json` run emitted ten MuJoCo EPA face-capacity warnings
at 12.6503–12.6505 seconds, all from `banana_plate` against `fly/lh_tarsus1` during
post-meal takeoff. The captured three-state replay reproduces 1, 4 and 5 warnings at
the MuJoCo default of 35 CCD iterations. MuJoCo sizes the EPA face pool to six times
[`ccd_iterations`](https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_collision_gjk.c#L2303-L2312),
so the old pool held 210 faces. The configured value of 100 provides 600 faces while
retaining the 1e-6 tolerance. Both 100 and 200 iterations produce zero warnings and
matching contact distances, positions and normals within 1e-9 in the saved-state
regression. The full changed-heading trial then completes without the warning.

The final source passes 259 Rust unit/binary/collision tests and 37 focused Python tests.
Seven physical flight/altitude/touchdown/walking tests and two full-CNS olfactory
assays also pass. The failed neural-pathway and strict CPU-f64/Metal final-state
reports from the earlier work remain failed.

## Neural boundary

The unchanged 166,700-neuron MaleCNS simulation now exposes 127 matched DM1/DM2 ORNs
already present in its sensory interface. Left/right means are normalized per cell,
then weighted equally by glomerulus, so different population sizes do not create a
steering bias. Four population rates are filtered over 200 ms. Their measured spikes
come from the running full CNS, not from copying the requested input rates.
The filter is normalized by its accumulated weight to remove zero-initialization bias;
guidance cannot acquire before 200 ms of actual observation. This does not add spikes
or substitute an assumed firing rate. A further 500 ms acquisition interval suppresses
steering and landing while the concentration decoder settles.

Total ORN rate is not monotonic with concentration under the existing adaptation
model. A baseline-subtracted DM2/DM1 ratio cancels the shared gain. Its inverse Hill
readout estimates concentration in the existing receptor model's ppm units. This
inversion assumes that model; it is not a new animal measurement. Near the spontaneous
floor the estimate is noisy, and a finite ratio cap bounds saturation.

The guidance decoder supplies bounded steering, an odor-context landing gate, and an
approach-height target. It has no access to resource IDs, food coordinates, nearest-food
distance, or a path planner. It bypasses unvalidated central food-heading circuitry.
Wing and walking activation still require their actual motor-neuron readouts; landing
also requires the existing DN/tibial readout. Physical taste and MN9 spikes are required
for feeding. Collision stabilization, gait, wing kinematics and meal timing remain
engineered, as before.

CNS walking activation now sets gait cadence without also shrinking stride geometry.
Obstacle braking scales translation independently of cadence; bilateral signed strides
retain physical pivot authority. Zero activation stops the gait. During supported
feeding, MN9 extension above 0.1 holds the supported legs and gait phase; stride
excursion is not used as the feeding hold mechanism. The older FAFB mapping is retained.

The landing-to-walking handoff is calibrated against physical support. Food-surface
contacts on the non-room-wall surfaces `banana_plate` and `resource_banana` are
checked against explicit `ground_plane` contacts for the same fly geom, including
contact dimension, friction, solver reference/impedance, and inclusion margin. The
wing load transfer is reversible over 80 ms. The leg control targets and actuator
preload are preserved; wing fold is rate-limited. While airborne, gait is paused.
At touchdown, gait phase resets and sensed contacts are held for 40 ms. The following
120 ms joint blend uses the gait's own adhesion mask, so swing feet are not pinned
while their targets move. Odor context is preserved through Feed, but steering is
held during supported feeding. The meal clock starts only after grounding; physical
taste still reaches the CNS during descent.

The HUD explicitly labels this boundary `ORN guidance (engineered)`. Native reports
separate `cns_olfactory`, `odor_guidance`, raw CNS motor readouts, and physical feeding.
Their feeding-duration summary now excludes residual post-meal proboscis extension.

## Neural assays

Run the ignored integration tests with Metal available:

```sh
cargo test --locked --release --test cns_olfaction -- --ignored --nocapture
```

- Swapping left/right odor activation reverses the measured steering contrast,
  including a +/-0.3% perturbation. Balanced/no-odor mean contrast is near zero.
- Actual receptor dynamics plus whole-CNS spiking recover balanced concentrations:

| Input ppm | Decoded left ppm | Decoded right ppm |
|---:|---:|---:|
| 0.2 | 0.185 | 0.189 |
| 1 | 1.000 | 1.000 |
| 4 | 4.000 | 4.000 |
| 12 | 12.021 | 12.084 |

These verify this model's interface and sign, not animal behavior or the biological
validity of every simulated neuron.

## Development traces

Saved under `outputs/cns/odor-guidance/`; none is substituted for final acceptance:

- `candidate-1.json`: fast approach and premature landing; no feeding.
- `candidate-2.json`: reduced approach speed, but total ORN rate incorrectly used as
  closeness; landed in a plume and crawled with doubly attenuated gait; no feeding.
- `candidate-3.json`: ratio-based concentration, slower approach; passed within
  5.4 mm of the banana but above it; no feeding.
- `candidate-4.json`: corrected gait and lower concentration gate; touchdown continued
  translating past the food; no feeding.
- `candidate-5.json`: continuous wingbeat phase exposed an unstable yaw controller;
  circling despite the opposite steering command; no feeding.
- `candidate-6.json`: corrected yaw/speed tracking, but altitude bias kept the fly above
  the plume; no feeding.
- `candidate-7.json`: corrected altitude tracking; touched down in flight posture and
  flipped during the transition to walking; no feeding.
- `candidate-8.json`: level landing posture, but the old landing-height target kept the
  feet above the floor; no feeding.
- `candidate-9.json`: touchdown worked, but the flight obstacle policy overrode neural
  walking underneath the table; no feeding.
- `candidate-10.json`: separate ground navigation exposed cadence braking that also
  suppressed obstacle turning; no feeding.
- `candidate-11.json`: the corrected gait walks upright, but follows an artificial odor
  maximum downwind of the banana; no feeding in 60 seconds.
- `candidate-12.json`: lowering approach height to 5 mm caused collisions with the fruit
  before a landing command; no feeding. The 8 mm acquisition floor is retained.
- `candidate-13.json`: continuous odor field, but zero-initialized rate estimates delayed
  acquisition until the fly had climbed above the food; no feeding.
- `candidate-14.json`: bias-corrected odor readout reaches the banana, but only 6 ms of
  feeding before an overly fast touchdown destabilizes it. Its 10-second odor-evoked
  disconnection control has zero guidance/close samples and zero feeding extension,
  while the CNS produces 10,473,299 population spikes.
- `candidate-15.json`: 58 ms of feeding; the flight-to-gait handoff still moves the legs
  toward the gait library's unrelated neutral pose, destabilizing the food perch.
- `candidate-16.json`: no feeding; close odor was reached but sparse landing-DN bursts
  did not satisfy the old continuous 80 ms threshold.
- `candidate-17.json`: 2.916 seconds of contiguous upright, supported, neural feeding,
  followed by flight departure. This used an overly broad 2.6 mm capsule-surface taste
  margin and is not final acceptance.
- `candidate-18.json` and `candidate-19.json`: the corrected 1.7 mm surface margin
  exposes a touchdown-to-walking instability; neither sustains feeding. Early taste
  in candidate 17 stopped the gait and masked that handoff failure.
- `candidate-20.json`: no feeding because a compatible-tripod requirement blocked
  gait startup on a supported, non-tripod stance. That requirement was removed.
- `candidate-21.json`: only 2 ms of feeding; a brief Feed transition cleared odor
  context while a partial MN9 command pinned the feet and the gait continued moving.
- `candidate-22.json`: 2.976 seconds of contiguous supported feeding from 8.858 to
  11.834 seconds, with all six contacts and minimum up-axis 0.928768, followed by
  flight departure at about 13.3 seconds. Its peak altitude was 189.874 mm. This is
  the final default-start run reproduced it and passed the paired gate above.

The CNS landing readout now accumulates activation with a 250 ms leaky integrator.
The trigger dose is the configured landing threshold times 80 ms. Close odor brakes
the approach while this evidence accumulates. Odor context loss clears the evidence;
zero landing activity still cannot produce a descent command. This is an explicit
motor-interface calibration, not newly identified synaptic connectivity.

A temporal odor-recovery experiment failed to complete its return turn in the
physical walking fixture, including a higher-authority pivot variant. It was removed
from the production decoder rather than enabled on the strength of unit tests alone.

## Odor-field correction

The previous piecewise field had a discontinuity at the source's crosswind plane. In
`candidate-10`, an antenna crossed from about 1.04 to 0.112 ppm in one control window.
At 8 mm altitude, its broadening Gaussian peaked about 41 mm downwind of the banana.
It also omitted spreading dilution: widening the plume increased integrated odor.
Neither artifact should be interpreted as neural food-search behavior.

The replacement is a deterministic, finite-core regularization of a steady
advection–diffusion–decay field. For source offset `d`, airflow `v`, odor length `L`,
and core `a=2 mm`:

```text
D = 35 mm/s × 0.18 L
p_ref = 35 mm/s / (2D)
q = (2 p_ref + 1/L) / L
b = sqrt(|v|²/(4D²) + q)
r = sqrt(|d|² + a²)
C(d) = C_source × (a/r) × exp(v·d/(2D) − b(r−a))
```

The underlying uniform-flow transport model is described in
[MIT's advection–diffusion notes](https://ocw.mit.edu/courses/1-061-transport-processes-in-the-environment-fall-2008/c96f9a75576238ac2c4e0e73b84e8072_lec_05.pdf).
The finite core, diffusivity and decay calibration above are our engineering choices,
not measured room chemistry. `C_source` is concentration at the source center, not an
emission-flux parameter. This model is continuous through zero wind and the source
plane, includes inverse-radius dilution, and does not solve airflow around furniture
or turbulent intermittency. It does not claim actual-fly exposure accuracy.

Odor acquisition holds its initial altitude (minimum 8 mm); noisy rate/concentration
peaks no longer invent a vertical source location. Existing CNS altitude control still
operates outside food approach. Steering only receives actual ORN readouts, never
the source locations used to construct this field.

## Body-controller regressions

These are engineering tests of the MuJoCo embodiment, not biological validation:

- Wingbeat phase integrates elapsed time at the previous frequency. Changing frequency
  no longer jumps the wing phase by multiplying absolute simulation time.
- Horizontal and vertical disturbance integrators reject outward windup and reset when
  their controllers are disabled. Horizontal authority is now bounded at twice body
  weight; vertical authority remains 0.8 times body weight.
- A yaw-rate gain of 5 produced a spontaneous -1.226 rad/s turn with zero steering in
  the physical fixture. Gain 1 restores neutral tracking and gives about 0.461 rad/s
  for a 0.5 rad/s target.
- Pitch aerodynamic feed-forward scales with the squared wingbeat frequency/amplitude
  command, preserving the intended amplitude dependence.
- Gain 200 with the old 0.4-weight horizontal limit saturated on wingbeat fluctuations.
  With a 2-weight bound, it tracks 15 mm/s within 0.5 mm/s and reduces landing drift
  to less than 2.1 mm in the 0/15 mm/s touchdown fixtures.
- The previous altitude PD loop tracked a 20 mm target at about 24.35 mm. Bounded
  integral compensation reduces the mean error below 0.1 mm at both 8 and 20 mm.
- Wall-heading commands now use the same attitude controller and a bounded yaw-rate
  target. The separate all-axis heading controller was producing about 109 mm/s for
  a 15 mm/s command in the half-turn fixture.

`tests/flight_control.rs` checks neutral/signed steering, braking, and a level half-turn.
`tests/altitude_control_diagnostic.rs` asserts altitude tracking and bounded force.
`tests/touchdown_control.rs` asserts upright landing followed by walking.
`tests/walking_control.rs` asserts neutral/signed turns and bilateral pivots.
The world-level food-surface fixture additionally checks a rate-limited landing target,
horizontal braking during landing, and a low-speed upright support handoff. Feeding
preserves the contact-supported control targets and actuator preload; it does not
command the gait library's zero-excursion pose, which differs from the body's nominal
pose by up to 2.35 radians and is not a safe stop command.

The added `cns-check` controls include `--disconnect-olfactory-evoked-inputs` (retains
spontaneous ORN firing and other senses), `--disconnect-odor-guidance`, and
`--initial-yaw-deg`. Existing motor/landing disconnect controls remain available.
Paired hashes include the initial pose and runtime executable SHA-256 and normalize
the named intervention only. Controls from different builds cannot be paired.
