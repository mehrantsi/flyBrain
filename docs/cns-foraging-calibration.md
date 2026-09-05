# CNS foraging interface calibration — 2026-09-04

This is the preserved first-pass calibration report. The subsequent
[CNS odor-guidance integration](cns-odor-guidance.md) supersedes its runtime settings
and adds passing paired food-approach/feeding checks. The failed trial below remains
failed and is not relabeled as a success.

This fixes decoder/arbitration mismatches, not the missing biological food-search policy.
The final 60-second free-roaming run still did **not** find food. Do not present the
changes or the controlled sugar assay as validation of autonomous fly foraging.

## Changes

- MaleCNS wing activation (typically 0.86–0.91) no longer has to fall below the
  FAFB-specific 0.10 flight-drive threshold before food approach is allowed.
- During CNS odor approach, the engineered altitude target moves toward the existing
  28 mm cruise setpoint at the existing bounded slew rate. Tonic wing power no longer
  continually commands ~185 mm during approach. Outside approach, the CNS power and
  candidate altitude decoders retain control of the bounded target.
- CNS landing uses an adapted bilateral odor sum of 0.40, above the observed initial
  diffuse background (~0.32), plus the existing DN/tibial-motor landing readout >=0.02
  for 80 ms. Actual airborne support contact remains another landing context.
  These numbers are engineering calibration, not fitted animal thresholds.
- A consumed CNS landing request rearms after odor falls below its 0.20 release
  threshold for 0.6 s, allowing a new plume encounter despite diffuse background odor.
  Ground search retains the short physical-touchdown dwell, with 0.20 odor / 0.01
  neural-drive release thresholds. Taste and post-meal arbitration remain separate.
- CNS approach speed scale is 0.60 instead of the FAFB scale 0.24. The latter produced
  a small orbit with the present CNS turn readout and body dynamics.
- MN9 spikes can drive CNS proboscis extension only with positive taste context;
  extension decays after contact/refractory release. Without this context gate,
  tonic MN9 activity extended the proboscis and suppressed walking even without food.
  Raw MN9 spikes and rates remain observable. Food context alone cannot generate
  extension without MN9 spikes, and motor-output disconnection removes extension.
- Motor-output disconnection now also zeros the landing command. A separate
  `--disconnect-landing-output` intervention preserves the running CNS and flight
  motors while disconnecting only the landing readout.

The native graph, sensory neuron selection, synaptic arrays and LIF parameters are
unchanged. No destination coordinates, path planner, new edges, or forced feeding
events were added. Odor approach/landing arbitration, the taste context gate, gait,
flight stabilization, three-second meal bout and post-meal refractory remain
engineered components, not newly recovered neural programs.

The CNS-only settings are serialized under `parameters.cns_foraging`; existing
parameter artifacts load the explicit defaults. FAFB foraging thresholds are retained.
All changes are constant-time synchronous arithmetic/state updates in the native
control loop, without new threads, locks, allocations, or Python crossings.

## Verification

Reports use the same release binary, SHA-256
`a3d2a47f3b7c20b2dac387da6b127946619506ea90531078c2a5db3e16c81f79`.

| Final check | Result |
|---|---|
| [Free roaming, 60 s](../outputs/cns/foraging-calibration/intact-rearmed.json) | Four descent/touchdown episodes; 6.17 m horizontal path; height up to 184.92 mm; zero food contact/feeding; no false proboscis extension |
| [Landing readout disconnected, 10 s](../outputs/cns/foraging-calibration/landing-disconnected-rearmed.json) | CNS and flight remain active; no descent requests; landing drive zero |
| [Stationary sugar 0.6 mm ahead, 8 s](../outputs/cns/foraging-calibration/sugar-contact-rearmed.json) | Grounded taste + MN9 + extension approximately 0.036–3.008 s; post-meal state at 3.018 s; flight resumes at 4.418 s |
| [Same sugar, motor outputs disconnected](../outputs/cns/foraging-calibration/sugar-motor-disconnected-rearmed.json) | Taste and MN9 activity remain; extension, landing command and flight are zero |

The sugar is placed once before stepping; it is neither moved to follow the mouth
nor teleported under the fly during the assay. This tests physical food-contact
integration and departure, not odor-guided food discovery. The older native
`summary.feeding_seconds` counts physical taste plus residual extension, including
post-meal decay. For actual feeding, additionally require `behavior_mode == "FEED"`
and `foraging_mode == "FEED"`; the quoted intervals use that stricter definition.

The 60-second intact and 10-second landing-disconnected runs share their initial
configuration hash. Their trajectories are identical until the first intact landing
request (~6.66 s), after which sensory feedback and neural activity can diverge.

Earlier candidates are preserved: `candidate-1.json` orbited without landing;
`candidate-2.json` reached yeast and fed but predates the MN9 taste-context correction.
Its successful encounter is **not** a passing free-foraging result for the final build.

Rust regression: 211 library tests and seven CLI tests passed with Metal available;
20 targeted Python tests passed. Formatting, Clippy and targeted Ruff passed.
The [final foraging gate](../outputs/cns/foraging-calibration/final-foraging-gate.json)
passes eight interface/control checks and fails the two food-discovery/departure checks.
The sampled feeding predicate excludes post-meal decay and requires actual food
contact, the FEED state, MN9 activity and extension. Recheck without simulating:

```sh
.venv/bin/python tools/verify_cns_foraging.py \
  --intact-report outputs/cns/foraging-calibration/intact-rearmed.json \
  --landing-disconnected-report outputs/cns/foraging-calibration/landing-disconnected-rearmed.json \
  --output /tmp/new-foraging-check.json
```

Exit status 1 is expected for these saved free-roaming reports. It must not be
changed to a pass using the separate controlled sugar assay.

## Run

Close the old live window and launch the rebuilt binary; an already-running process
does not pick up decoder changes:

```sh
cd /Users/mehran/dev/flyBrain
target/release/flybrain-world view
```

To reproduce the controlled food-contact case, add `--start-food-distance 0.6`.
Pure sugar has zero odor in this habitat; fruit/fermentation plumes drive odor input.
