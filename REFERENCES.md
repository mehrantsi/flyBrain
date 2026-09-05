# Sources and scientific references

Provenance checked on 2026-09-05. This distinguishes the project's original
foundation from the later MaleCNS integration. License scope and retained notices
are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Original foundation, before MaleCNS

### Brain dynamics: Shiu et al. and Brian2

Shiu, P. K. et al. (2024). **A Drosophila computational brain model reveals
sensorimotor processing.** Nature 634, 210–219.
[Paper](https://doi.org/10.1038/s41586-024-07763-9) ·
[Source code and input tables](https://github.com/philshiu/Drosophila_brain_model).

This was the starting neural model: leaky integrate-and-fire neurons, signed
connectome weights, synaptic delays, refractory behavior, and taste-to-MN9
experiments. The upstream implementation uses Brian2, not a trained behavioral
policy. FlyBrain reimplements its numerical model in Rust/Metal and maintains
independent Python/Brian2/NumPy/MLX checks; the browser adds a WebGPU implementation.
These ports and the body interface are FlyBrain work, not upstream results.

The local upstream checkout is pinned to
`91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`. Its README links the earlier 2023
preprint; the citation above is the final 2024 publication.

### Wiring and annotations: female FAFB/FlyWire

Dorkenwald, S. et al. (2024). **Neuronal wiring diagram of an adult brain.**
Nature 634, 124–138. [Paper](https://doi.org/10.1038/s41586-024-07558-y).

Schlegel, P. et al. (2024). **Whole-brain annotation and multi-connectome cell
typing of Drosophila.** Nature 634, 139–152.
[Paper](https://doi.org/10.1038/s41586-024-07686-5) ·
[Annotation repository](https://github.com/flyconnectome/flywire_annotations).

Credit the Princeton FlyWire team, Murthy and Seung labs, the FlyWire Consortium,
and the reconstruction and annotation contributors identified in these papers.
The initial graph came from the model repository's processed FlyWire tables,
not from a new EM reconstruction performed here:

| Local pack | Upstream model inputs | Model neurons | Directed neuron-pair edges | Anatomical contacts |
|---|---|---:|---:|---:|
| `flywire_v630` | `2023_03_23_completeness_630_final.csv`, `2023_03_23_connectivity_630_final.parquet` | 127,400 | 14,687,178 | 52,793,639 |
| `flywire_v783` | `Completeness_783.csv`, `Connectivity_783.parquet` | 138,639 | 15,091,983 | 54,492,922 |

These are counts in our imported simulator tables, not a claim that every neuron
in the broader annotated connectome is represented. The v630 pack was the original
reproduction target; v783 was the subsequent FAFB embodiment baseline. Their
`outputs/packs/<pack>/manifest.json` files retain source and array SHA-256 hashes.
FlyBrain compiles the tables into neuron-ID-indexed signed CSR arrays; it does not
recover axon geometry, receptor physiology, or the donor's state from those arrays.

The candidate sensory/descending-neuron groups in
[flywire_v783_neural_io.json](assets/neuromechfly/flywire_v783_neural_io.json)
come from annotation release `v2.1.0`, commit
`ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0`. That artifact records the exact TSV URL,
hash, selection rules, and IDs missing from the simulator pack.

### Body and senses: NeuroMechFly / FlyGym

Lobato-Rios, V. et al. (2022). **NeuroMechFly, a neuromechanical model of adult
Drosophila melanogaster.** Nature Methods 19, 620–627.
[Paper](https://doi.org/10.1038/s41592-022-01466-7).

Wang-Chen, S. et al. (2024). **NeuroMechFly v2: simulating embodied sensorimotor
control in adult Drosophila.** Nature Methods 21, 2353–2362.
[Paper](https://doi.org/10.1038/s41592-024-02497-y) ·
[FlyGym source](https://github.com/NeLy-EPFL/flygym).

FlyGym `v2.1.0`, commit `ca65a510c2afe6ac61c51df4f274c8d190c2f95f`, supplied
the articulated body, MJCF/mesh assets, recorded leg trajectories, and compound-eye
sampling assets. FlyBrain exports/adapts these for its Rust MuJoCo world, contact
handling, room, and viewer. The retina has 721 ommatidia per eye; its fisheye
implementation has an additional [Gil-Mor/iFish](https://github.com/Gil-Mor/iFish)
MIT lineage acknowledged by FlyGym.

Exact versions, transformations, and file hashes are in
[the body manifest](assets/neuromechfly/manifest.json) and
[the retina manifest](assets/neuromechfly/vision/manifest.json).

### Flight components added before MaleCNS: FlyBody

Vaxenburg, R. et al. (2025). **Whole-body physics simulation of fruit fly
locomotion.** Nature 643, 1312–1320.
[Paper](https://doi.org/10.1038/s41586-025-09029-4) ·
[FlyBody source](https://github.com/TuragaLab/flybody) ·
[Dataset v4](https://doi.org/10.25378/janelia.25309105.v4).

The pinned code revision is `d015e9bfe441bd90ae431bac24c55cb74bdbce26`.
FlyBrain uses/adapts flight pose, wing geometry and fluid-model parameters, and
fits the published `wing_pattern_fmech.npy` to a compact Fourier waveform.
The downloaded flight trajectories also supply system-identification targets;
the repository contains a separately exported pretrained flight policy and
verification fixture. These are not recovered connectome pathways or proof that
the neural model reproduces biological flight.

See [aerodynamic provenance](assets/neuromechfly/aerodynamics.json),
[flight targets](assets/neuromechfly/flight_system_id_targets_v1.json), and
[policy provenance](assets/neuromechfly/flybody_flight_policy_v1.json).
The code and downloaded datasets have different licenses.

## Later integration: MaleCNS, following the Google article

The Google Research article **A connectomics milestone: Mapping the complete male
fruit fly brain** (2026-09-03) led to the subsequent integration, not the original
FlyWire model. [Article](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/).

The underlying work is **Sexual dimorphism in the complete connectome of the
Drosophila male central nervous system**, Cell (2026).
[Publication](https://www.cell.com/cell/fulltext/S0092-8674%2826%2900942-6) ·
[MaleCNS project](https://male-cns.janelia.org/) ·
[v1.0 data](https://male-cns.janelia.org/download/).

Credit the MaleCNS collaboration: FlyEM at HHMI Janelia, University of Cambridge
Department of Zoology, MRC Laboratory of Molecular Biology, and Google Research.
The imported v1.0 annotations, neurotransmitter predictions, and connectivity
come from one male CNS specimen, including its nerve cord. This is a separate
dataset, not additional connections grafted onto the female FlyWire IDs.

The current pack selects 166,700 annotated neurons and 24,469,412 known-transmitter
directed edges. `outputs/packs/male_cns_v1/manifest.json` records the source URLs,
hashes, inclusion rules, and omitted edges. Source files, not this summary, define
the release. See [integration limits](docs/cns-embodiment.md) and
[pathway assays, including failed gates](docs/male-cns-pathway.md).

## Runtime and project-specific work

MuJoCo 3.9.0 supplies physics; the vendored `mujoco-rs` 5.0.0 binding is patched as
described in [FLYBRAIN_PATCH.md](vendor/mujoco-rs/FLYBRAIN_PATCH.md). Three.js
0.180.0 supplies browser rendering. These are software dependencies, not brain
datasets. Rust/Metal/WebGPU execution, packing and integrity checks, numerical
verification, neural/body interfaces, the room, HUD, and browser deployment are
project-specific engineering. The room texture's generated origin is documented
in [the material record](assets/materials/README.md).

The project is not a port of Eon's unpublished brain-to-body integration. It
does not preserve the scanned animal's memories or identity. Literature-based
candidate pathways, engineered sensor/motor decoders, and validated biological
circuits must remain distinct. Task-specific scientific references are retained
in the linked manifests and the [grooming](docs/neural-grooming.md),
[odor guidance](docs/cns-odor-guidance.md), and pathway documentation.
