# MaleCNS live embodiment — 2026-09-04

For current food-approach calibration and paired checks, see
[CNS odor guidance](cns-odor-guidance.md). The failed first-pass search trial remains
in [CNS foraging calibration](cns-foraging-calibration.md).
The 30-second movement results below predate both calibrations.

The live default is the MaleCNS v1.0 graph, not FAFB plus an invented nerve cord.
All 166,700 model neurons and 24,469,412 signed neuron-pair edges advance in Rust/Metal.
The imported arrays are unchanged by embodiment; see the [pack manifest](../outputs/packs/male_cns_v1/manifest.json).

```sh
cargo build --locked --release --bin flybrain-world
target/release/flybrain-world view
```

The default places movable sugar 40 mm ahead. The window runs indefinitely;
`5` selects the room camera, `1` restores chase, `V` toggles both eyes, `B` toggles the
EEG-like network proxy, `R` resets, and `Esc` closes it. The HUD identifies MaleCNS,
motor-pool rates, decoded commands, current/target height, and sensory proxies.
Use `--pack outputs/packs/flywire_v783` to select the earlier brain-only baseline.

## What is connected

The [I/O artifact](../assets/neuromechfly/male_cns_v1_neural_io.json) is bound to the
annotation SHA-256 and all four native array hashes. Its 28 groups contain 2,459
memberships / 2,447 distinct neuron IDs, all present. There are 2,324 externally
driven model neurons. Selection details, exclusions, transmitter counts, and actual
directed-path witnesses are in the [evidence artifact](../assets/neuromechfly/male_cns_v1_neural_io_evidence.json).

| World/body interface | Annotated CNS populations | Remaining engineered interpretation |
|---|---|---|
| Food odor | Left/right typed antenna ORNs; eight food-response channels | Concentration-to-rate model and fruit odor approximations |
| Sugar contact | 85 labellar LB3 candidates; MN9 feeding readout | Contact valence to input events; MN9 spikes to proboscis extension |
| Motion and approach | ACh HS/VS-family visual projections; MeVP24 | Ray/velocity-derived bilateral motion and looming proxies, not retinal transduction |
| Rotation feedback | 148 SApp haltere sensory-ascending neurons | Scalar angular-speed rate channel, not resolved haltere mechanics |
| Wing power | 12 DLM/DVM motor neurons per side | Mean rate to flight activation, speed and height target |
| Wing steering | Three b1/b2/b3 motor neurons per side | Bilateral rate difference to steering |
| Walking | Tibial/trochanter extensor motor pools | Population drive gates the existing gait; descending activity supplies turn intent |
| Landing and altitude candidates | DNp07/DNp10 plus tibial motors; DNg02/DNg07 | Candidate rate decoders and contact/food arbitration, not recovered programs |

Motor activation is `rate / (rate + 20 Hz)` after the existing 50 ms rate filter.
The 20 Hz half-activation scale is an engineering parameter, not a fitted biological
constant. This avoids clipping the observed wing-motor rates into a constant command.
Outside food approach, current wing-motor activation controls the bounded altitude target; it no longer uses
the old historical-maximum flight-drive rule. DNg02/DNg07 add an explicitly hypothetical
signed altitude contribution. DNg07 is not an experimentally isolated altitude controller.
During CNS food approach, engineered arbitration instead selects the neutral cruise
height. The MN9 extension decoder now also requires positive taste context.

No motor or descending neuron is directly driven to force takeoff. Events enter the
selected sensory/projection populations and propagate through the imported CNS edges.
The body still uses engineered gait/wing waveforms, attitude and velocity stabilization,
collision escape, feeding arbitration and grooming. The visual mini-views are displays;
the new CNS visual channel consumes the documented kinematic proxies, not those images.

## Observed results

The [30-second paired gate](../outputs/cns/world-verified/acceptance.json) passes:

| Measurement | Intact | Motor outputs disconnected |
|---|---:|---:|
| Physical horizontal path | 4,050.50 mm | 1.66 mm (passive settling/drift) |
| Forward airborne distance | 3,997.75 mm | 0 mm |
| Flight-mode time | 29.822 s | 0 s |
| Maximum height | 187.24 mm | 2.10 mm initial height |
| Whole-CNS spikes | 32,060,348 | 31,453,554 |
| Unique motor-pool spikes | 133,621 | 126,005 |
| Longest collision-escape episode | 1.38 s | 0 s |

The intact trajectory spans x = -296.91 to 297.15 mm and y = -216.39 to 216.82 mm,
remaining within the room. The two conditions share their hash-bound initial setup.
Disconnecting the motor readout leaves the neurons running; subsequent sensory traces
diverge because the body trajectories differ. Passive settling is not neural locomotion.

The [four-second no-sensory-input control](../outputs/cns/world-verified/no-sensory-input.json)
has zero whole-CNS spikes, zero motor-pool spikes, zero flight time and zero airborne
distance. Its small displacement is the same initial passive settling.

An earlier run crossed the room but then remained in one collision-escape episode for
17.388 s. The escape controller had retained a heading from a previous wall. It now
replans when another wall blocks that heading, including a consistent diagonal at corners.
The [rejected trace](../outputs/cns/world-final/rejected-corner-summary.json) is preserved;
the gate rejects it and accepts the corrected run. This is an engineered controller fix,
not evidence for a newly recovered neural avoidance circuit.

Recheck the saved pair without simulating again:

```sh
.venv/bin/python tools/verify_cns_world.py --duration-seconds 30 \
  --intact-report outputs/cns/world-verified/intact.json \
  --disconnected-report outputs/cns/world-verified/disconnected.json \
  --min-forward-displacement-mm 500 --min-altitude-range-mm 20 \
  --min-lesion-world-delta-mm 50
```

For a fresh pair, omit both report arguments. Run the no-input control using
`flybrain-world cns-check --disconnect-sensory-inputs --duration-seconds 4 --output NEW.json`.
The report-producing native binary SHA-256 was
`e5c483d74ff2277cbc9eb6b16bcaa0091150ac1836f0653dfaf47b8e1881f395`;
the subsequent live-default CLI and recording-metadata changes do not alter simulation stepping.
The intact headless run took 94.04 seconds wall time for 30 simulated seconds on this
M3 Max, including brain telemetry and physics but excluding loading. This is not a
standalone neural-engine benchmark.

## What this does not establish

These are software-integration and causal-disconnection checks, not behavioral validation
against recorded animals. The tested fly remained airborne after takeoff; this does not
establish successful food search, feeding, natural flight/landing statistics or altitude choice.
Its height stays near the upper part of the room once the current rate decoder reaches
equilibrium. The Shiu LIF parameters and transmitter-wide sign priors remain unvalidated for
this complete CNS, and unsupported transmitter effects remain omitted rather than invented.

The earlier [MeVP24–DNp10 landing assay](male-cns-pathway.md) still fails its biological
acceptance criterion. The strict 100 ms f64/Metal state-tolerance gate also remains failed;
the diagnostic attributes it to f32/FMA threshold timing rather than a synchronization bug.
Neither negative result was reclassified as a pass to enable the experimental world interface.

Regression checks: 199 Rust library tests, seven native CLI tests, and 66 Python tests pass.
Rust formatting, Clippy and targeted Ruff checks pass. GPU-dependent tests require Metal access;
the sandbox's no-device failures disappear when run with that access.
