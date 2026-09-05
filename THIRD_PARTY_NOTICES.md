# Third-party licenses and attribution

Checked against pinned local sources and upstream release metadata on 2026-09-05.
Scientific provenance is in [REFERENCES.md](REFERENCES.md).

The root [MIT license](LICENSE) covers original FlyBrain code and documentation.
It does not relicense third-party code, connectomes, annotations, model assets,
datasets, or learned weights. This directory is a mixed-license distribution,
not an assertion that every included file is MIT or commercially reusable.

## Component inventory

| Component / local material | Upstream terms | Notice / scope |
|---|---|---|
| Shiu and Spiller's `Drosophila_brain_model` code; model reimplementation lineage | MIT | [Original copyright and license](licenses/Drosophila_brain_model.txt), Philip Shiu and Nico Spiller, 2023. Separate from the FlyWire data terms. |
| FlyWire public v783 connectivity and annotations; `outputs/packs/flywire_v783/`, `assets/neuromechfly/flywire_v783_neural_io.json` | CC BY-NC 4.0 | [License](licenses/CC-BY-NC-4.0.txt), [data-use and citation guidelines](https://home.flywire.ai/guidelines). Credit the FlyWire team, Consortium, Murthy and Seung labs, and the Dorkenwald/Schlegel papers listed in REFERENCES. |
| Earlier FlyWire v630 simulator tables; `outputs/packs/flywire_v630/` | Release-specific FlyWire terms require confirmation | Obtained from Shiu's model repository. Its MIT code license alone is not evidence of a separate v630 data grant; current FlyWire guidelines explicitly identify public v783. Verify the earlier release's redistribution/commercial permissions before distributing that pack. |
| MaleCNS v1.0 data; `outputs/packs/male_cns_v1/`, `assets/neuromechfly/male_cns_v1_neural_io*.json` | CC BY 4.0 | [License](licenses/CC-BY-4.0.txt), [release terms](https://male-cns.janelia.org/download/). Credit FlyEM/HHMI Janelia, Cambridge, MRC LMB, Google Research, and the MaleCNS collaboration/publication. |
| FlyGym / NeuroMechFly body, gait, and retina assets in `assets/neuromechfly/` | Apache-2.0, with separate inherited components below | [Retained FlyGym license](assets/neuromechfly/LICENSE-FLYGYM). Cite both NeuroMechFly papers. The world manifest's source license describes FlyGym, not all subsequently combined data. |
| Gil-Mor/iFish fisheye implementation, via FlyGym Retina | MIT | [License and copyright](licenses/iFish.txt). Applies to the implementation lineage, not a claim about biological visual perception. |
| FlyBody code, wing geometry, and fluid-model lineage | Apache-2.0 | [Retained FlyBody license](licenses/FlyBody.txt); [pinned source](https://github.com/TuragaLab/flybody/tree/d015e9bfe441bd90ae431bac24c55cb74bdbce26). Does not cover the separately released Figshare datasets by implication. |
| FlyBody v4 flight-imitation dataset and trained policies; waveform coefficients in `aerodynamics.json` and `manifest.json`, `flight_system_id_targets_v1.json`, `flybody_flight_policy_v1.{json,f32le}`, `flybody_flight_policy_fixture_v1.json` | GPL-3.0-or-later (upstream label: GPL 3.0+) | [GPL text](licenses/GPL-3.0.txt), [dataset DOI](https://doi.org/10.25378/janelia.25309105.v4), [versioned publisher metadata](https://api.figshare.com/v2/articles/25309105/versions/4). Credit Roman Vaxenburg and coauthors. Preserve these terms for dataset-derived material; do not label it Apache merely because the code repository is Apache. |
| MuJoCo 3.9.0 | Apache-2.0 | [Retained license](licenses/MuJoCo.txt), [source](https://github.com/google-deepmind/mujoco). |
| Vendored `mujoco-rs` 5.0.0 | MIT OR Apache-2.0; MuJoCo-derived C/C++ portions Apache-2.0 | [Upstream notice](vendor/mujoco-rs/LICENSE), [local modifications](vendor/mujoco-rs/FLYBRAIN_PATCH.md). |
| Three.js 0.180.0 | MIT | [Retained license](licenses/Three.js.txt), [source](https://github.com/mrdoob/three.js). |

## Modifications and distribution boundaries

FlyWire input tables are converted to signed CSR arrays with neuron-ID mappings
and source/array hashes. Public annotations are selected into candidate I/O groups.
MaleCNS tables are filtered to annotated nodes and supported neurotransmitter
labels before CSR conversion; omitted counts and source hashes are in the pack
manifest. The male and female identifiers remain separate. These transformations
do not transfer ownership of the source data to FlyBrain.

FlyGym assets are exported, simplified, and adapted for MuJoCo/Rust; contact,
camera, appearance, and room details are modified here. FlyBody quantities are
converted to the project's body units and actuator conventions. Its downloaded
wing waveform is fitted to twelve Fourier harmonics; flight trajectories are
summarized into paired calibration targets; pretrained policy tensors are exported
to float32 arrays with fixtures. The relevant exporters and manifests identify
the changes and retain source hashes.

The current browser runtime includes the MaleCNS pack, legacy FlyWire I/O
annotations, and the dataset-derived flight waveform. It does not ship either
full FlyWire pack or the standalone pretrained flight-policy weight file.
That omission does not make the browser bundle solely MIT/Apache: the annotations
and waveform retain their own source-license considerations.

GPL-derived material needs the applicable GPL notices and corresponding-source
arrangements when distributed; adding a license text alone is not a completed
binary-distribution compliance audit. In particular, do not treat this inventory
as clearance to relicense the combined runtime or its datasets. Resolve the
distribution scope and v630 permission gap before describing a release as wholly
permissively licensed or suitable for commercial reuse.

## Other software and presentation assets

`Cargo.lock`, `uv.lock`, and `web/package-lock.json` record software dependency
versions. Brian2, NumPy, and MLX are reference/verification backends, not additional
connectome sources. Their packages retain their own license notices.

The Cloudflare packager includes notices for the resolved browser Rust dependency
graph and MuJoCo's libccd, LodePNG, miniz, Qhull, tinyobjloader, and TinyXML-2
dependencies. A packaged build's `licenses/Rust-dependencies.txt` and other
`licenses/` files describe that build, not every optional native or Python tool.
Retain the appropriate dependency notices when distributing those other builds.

The oak texture is AI-generated project presentation material, not a third-party
photograph or measured biological asset; see [its provenance](assets/materials/README.md).
Room details and cosmetic fly geometry are artistic/engineering additions, not
new scientific measurements. Citations and credit do not imply upstream
endorsement or biological validation of FlyBrain.
