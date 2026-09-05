# Wall landing handoff

The previous `Landing` path only targeted the surface below the fly and leveled
the body. It also disabled tarsal adhesion throughout flight, including landing.
Disabling the level-pitch target after two wall contacts did not solve this:
the flight stabilizer then returned to its normal airborne pitch target. There
was no wall-relative approach or load-transfer controller.

The corrected mechanical adapter:

- Acquires a nearby vertical room-wall collision face only after the existing
  flight controller enters `Landing`. The face comes from MuJoCo geometry,
  including wall thickness and front-wall segments, not the observer camera.
- Holds that surface target and approach altitude through touchdown. A quaternion
  attitude target rotates the body head-up, with its feet toward the wall,
  without Euler-angle singularities at a vertical pitch.
- Brakes approach velocity and leaves rotation clearance while the body aligns.
  The existing bounded body-force surrogate supplies gravity compensation during
  this vertical maneuver; it is disabled when wing support is shut off.
- Arms tarsal adhesion once body-to-wall alignment is adequate. MuJoCo applies
  adhesion only at actual contacts, at the physics rate; a 500 Hz contact mask
  must not miss intervening touches. Wall load transfer requires at least three
  contacted feet before the existing chatter-tolerant handoff parks the wings.
- Clears the wall target on subsequent takeoff and releases adhesion. The existing
  head-first wall-escape controller handles departure.

This is an engineered sensorimotor adapter, not a newly reconstructed neural
landing circuit. No connectome weights, landing-decision thresholds, feeding
routines, or scheduled sitting interval were added. The head-up attitude is an
explicit actuator-controller target, not a claim that the brain selected it.
Ceiling landing is not implemented by this change.

## Wall penetration correction

The original wall-landing checks verified support and wing parking but did not
bound penetration. A new mesh-to-wall-plane check failed on the previous model:
1.613411 mm of penetration at the first grounded sample. Walls had been explicitly
excluded from the calibrated habitat contact material, leaving soft default
contacts that adhesion could pull through.

All collidable habitat surfaces now use the same priority-1 contact parameters
as the explicit floor pairs (`solref=0.0002`, calibrated impedance and friction).
The generator, checked-in model and both runtime hash manifests agree. The
native contact-material test checks all seven wall/ceiling segments as well as
food surfaces. No render offset, pose constraint or connectome change is involved.

Hard contacts also exposed premature full wing-load transfer from only a pair
of feet. Requiring three wall contacts and letting the physics-rate adhesion
catch each touch preserves stable close-wall landings without soft penetration.

## Verification

The native MuJoCo regression approaches all four vertical walls from flight,
plus two close-wall horizontal-body cases (one facing sideways). All six cases
establish four-foot support, keep zero wing amplitude for at least 0.5 simulated
seconds, and subsequently take off and clear the wall head-first. The ordinary
approaches touch down at about 1.094 s; the close-wall cases at 0.732 s and 0.722 s.
The maximum sampled penetration across approach and support is 0.008332 mm,
below the 0.01 mm regression limit. Every collidable fly mesh vertex is checked
against the actual target wall plane at each 2 ms controller window.
No body pose is teleported during these transitions; only test initialization
sets the starting pose. All MuJoCo warning counters remain zero.

The complete native library suite passes 257 tests, including the existing
food-landing, contact-loss, wing-parking, and wall-escape tests. Both the native
world binary and optimized browser WASM were rebuilt.
Results are recorded in `outputs/performance/wall-contact-verification.json`.

The rebuilt full-CNS browser completed 60 simulated seconds with binocular
sensory rendering and zero MuJoCo warnings, including food touchdown and
subsequent flight. It averaged 0.380× realtime, not a matched performance
comparison. No wall landing was requested in this run, so it does not independently
validate wall touchdown. Results are in `outputs/performance/wall-contact-full-cns.json`.
The earlier `wall-landing-full-cns.json` trace used soft walls and is historical;
its six-foot support observations did not establish nonpenetrating contact.

The WASM load/step/reset/binocular smoke check and renderer tests also pass.
The initial 11-frame ground trajectory remains exactly equal to the preceding
WASM smoke output in both positions and velocities.

Browser and native CNS traces now include the wall surface point, inward normal,
body alignment and wall-contact count in `flight_diagnostics`.

Reproduce the physical regression with:

```sh
cargo test --lib wall_landing_aligns_attaches_and_parks_wings -- --nocapture
```

Reload the browser page to replace the previously loaded WASM runtime; resetting
the simulation alone does not load the new binary.
