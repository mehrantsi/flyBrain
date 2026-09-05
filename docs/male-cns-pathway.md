# MaleCNS pathway experiment

This document records the original, separate CNS pathway assay. The subsequent
[world integration](cns-world-verification.md) uses the full CNS graph with explicitly engineered
sensory/motor interfaces; it does not claim this candidate landing pathway passed validation.
It preserves neuron identity within the MaleCNS specimen; no male nerve cord is grafted
onto the female FAFB graph. The native Rust/Metal engine advances the imported graph.

## Result — 2026-09-04

The import and software regression checks passed. The candidate landing-model acceptance
gate **failed**; the live controller was not changed at that stage.

The audited pack contains 166,700 neurons, 24,469,412 signed directed edges, and 120,260,398
anatomical contacts. The selected annotated graph before transmitter filtering has
25,582,938 edges and 124,177,617 contacts; 1,113,526 edges / 3,917,219 contacts have unsupported
transmitter effects and are omitted explicitly. All four native array hashes verify.

Total tibial-extensor spikes in 200 ms, with all other model settings fixed:

| Intervention | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| MeVP24 stimulation, intact CNS | 26 | 25 | 17 |
| No input | 0 | 0 | 0 |
| MeVP24 outputs disconnected | 0 | 0 | 0 |
| DNp10 outputs disconnected | 25 | 25 | 19 |
| Direct DNp10 stimulation, MeVP24 outputs disconnected | 14 | 16 | 15 |

Both DNp10 cells respond in all intact runs. Input-output disconnection eliminates every
downstream spike while preserving input activity. However, disconnecting DNp10 does not
consistently reduce extensor activity, and no intact run activates all six motor pools.
Direct DNp10 stimulation also reaches motors but is not a complete six-leg response.
This is evidence of signal propagation and alternative network routes, not recovery of a
landing program. It does not disprove DNp10's experimentally established biological role.

CPU f64 and Metal have identical cumulative per-neuron spike counts across all 166,700
neurons at both 20 and 100 ms. Maximum final-state errors:

| Window | Voltage error, mV | Conductance error, mV |
|---|---:|---:|
| 20 ms | 0.000042912 | 0.000014935 |
| 100 ms | 0.028072425 | 0.005969828 |

The predeclared 0.001 mV final-state gate fails at 100 ms. The shorter window passes.
Identical cumulative counts do not establish identical spike timing; the longer-window
state difference remains a numerical-validation limitation, not a reason to relax the gate.

Follow-up numerical diagnosis located the first spike-timing divergence at tick 342 (ID 802738):
CPU f64 reaches -45.000001615 mV, just below the strict threshold, while Metal's f32/FMA arithmetic
crosses it. An explicit f32/FMA CPU mirror matches the complete 1,000-tick Metal spike mask, with
final-state error around 0.0000305 mV. Chunk sizes 1, 18 and 256 give identical Metal results.
No production arithmetic or tolerance was changed. Reproduce the diagnostic with:

```sh
cargo run --locked --release --bin cns-numerics -- \
  outputs/packs/male_cns_v1 outputs/cns/mevp24-dnp10-bilateral.json 1000 150 1 weighted-fma
```

191 Rust library tests, four native CLI tests, and 55 Python tests passed. Rust formatting,
Clippy, and targeted Ruff checks passed. The existing FAFB v783 pack and Brian2 golden
fixture also pass the native audit/parity checks. These are software regression results;
the experimental acceptance failure above is separate and intentionally returns a nonzero
exit status after preserving the reports.

Final machine-readable results: [summary](../outputs/cns/verification-final/summary.json),
[bilateral pathway](../outputs/cns/mevp24-dnp10-bilateral.json), and
[pack provenance](../outputs/packs/male_cns_v1/manifest.json). The prior exploratory outputs
are retained; use `verification-final` for the consistent final artifact set.

## Evidence and boundaries

The [MaleCNS v1.0 release](https://male-cns.janelia.org/download/) includes brain and ventral
nerve cord, with downloadable connectivity and transmitter annotations under CC BY 4.0.
The raw connectivity table includes segmentation fragments. Our node-selection rule is
annotation `superclass != null`, not every segment with a synapse. This selection includes
different proofreading statuses; it does not claim uniform completeness.

The first assay stimulates both annotated MeVP24 visual-projection neurons, records both
DNp10 descending neurons, and reads all 12 annotated tibial-extensor motor neurons, grouped
by the six legs. The source data contains these direct connections:

| Presynaptic body | Postsynaptic body | Anatomical contacts |
|---|---|---:|
| MeVP24 L: 10142 | DNp10 L: 10425 | 93 |
| MeVP24 R: 10723 | DNp10 R: 10433 | 90 |

[Ache et al.](https://www.nature.com/articles/s41593-019-0413-4) provide functional evidence
that DNp10 activation evokes landing-like leg extension, and that its visual response
depends on behavioral state. The [BANC study](https://www.nature.com/articles/s41586-026-10735-w)
provides a separate female-specimen anatomical hypothesis connecting DNp10 with tibial
extension and sensory feedback. BANC connections are not imported into this male model.

MeVP24 is an annotation/connectivity-based candidate input, **not a validated looming
detector**. The experiment injects events into its soma rather than converting retinal
images into biological spike trains. Anatomical paths are deterministic shortest witnesses
through DNp10 and at most two VNC interneurons. They audit connectivity; they do not prune
the simulated graph or prove that activity follows only those paths.

The model uses the existing Shiu LIF parameters, excitatory ACh and inhibitory GABA/Glu
priors. These parameters and source-wide signs are not validated CNS physiology.
Unsupported neurotransmitters have no outgoing modeled connections and are explicitly
counted by the importer. Their neurons remain in the node table. In particular, unknown
transmitter labels on motor neurons do not erase their incoming connections, but we do not
invent neuromuscular effects for them. No flexor-inhibition claim can be inferred from a
silent flexor baseline.

## Reproduce the pathway assay

The three source files are already downloaded under `work/upstream/male-cns/v1.0`.
To compile them again, use a new output path:

```sh
.venv/bin/python tools/import_male_cns.py \
  --annotations work/upstream/male-cns/v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --neurotransmitters work/upstream/male-cns/v1.0/body-neurotransmitters-male-cns-v1.0.feather \
  --connectivity work/upstream/male-cns/v1.0/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --output outputs/packs/male_cns_v1-rebuild
```

Using the existing audited pack:

```sh
cargo build --locked --release --bin flybrain-rs
.venv/bin/python -m flybrain.cns_pathway \
  --pack outputs/packs/male_cns_v1 \
  --annotations work/upstream/male-cns/v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --output outputs/cns/mevp24-dnp10-rerun.json
.venv/bin/python tools/verify_male_cns_pathway.py \
  --pack outputs/packs/male_cns_v1 \
  --pathway outputs/cns/mevp24-dnp10-rerun.json \
  --output outputs/cns/verification-rerun
```

Output files/directories must not already exist; choose new paths for a rerun.
The native loader verifies all pack arrays and binds the pathway to their exact hashes.
Every selected neuron and anatomical witness edge must exist in that pack.

The experiment runs matched 150 Hz Bernoulli input schedules for 200 ms with seeds 1, 2,
and 3. All runs start at rest, without background input. Input neurons use the existing
zero-refractory stimulation convention; no light or opsin is simulated, and the intensity
is not calibrated to visual input. The five conditions are:

- Intact network.
- No external input.
- MeVP24 outgoing synapses disabled, preserving input events and soma activity.
- DNp10 outgoing synapses disabled, preserving input events and relay soma activity.
- Direct DNp10 activation with MeVP24 outgoing synapses disabled.

Each report contains per-neuron motor counts, input/relay counts, full population totals,
model parameters, source/array hashes, device/memory/timing, and result hashes. A separate
20 and 100 ms runs compare native Metal against the CPU f64 implementation on the same full graph
and event schedule. This is a numerical implementation check, not biological validation.

Software checks require a quiet no-input run and no downstream spikes when input outputs
are disconnected. Separately, the candidate-pathway check requires motor responses in the
intact graph and reduced motor responses when DNp10 outputs are disconnected, in every seed.
This checks any motor endpoint response, not coordinated activation of all motor neurons.
Per-pool and per-cell participation are reported separately. Numerical acceptance also
requires exact spike counts and maximum final-state errors no greater than 0.001 mV at
100 ms. This distinguishes observed network effects from assertions imposed by a decoder.

Even a positive pathway result does not validate retinal transduction, flight-state gating,
muscle force, joint motion, or landing behavior. Those are required before enabling this
pathway in the live body controller.
