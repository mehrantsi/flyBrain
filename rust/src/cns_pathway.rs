use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::pack::ConnectomePack;
use crate::parameters::ModelParameters;
use crate::stimulus::EventSchedule;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CnsPathway {
    pub schema_version: u32,
    pub name: String,
    pub materialization: String,
    pub pack_array_sha256: BTreeMap<String, String>,
    pub evidence: Vec<String>,
    pub interpretation: String,
    pub stimulus_ids: Vec<u64>,
    pub relay_ids: Vec<u64>,
    pub readout_groups: BTreeMap<String, Vec<u64>>,
    pub anatomical_paths: Vec<Vec<u64>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum PathwayControl {
    Intact,
    NoInput,
    InputDisconnected,
    RelayDisconnected,
    RelayDriven,
}

#[derive(Debug)]
pub struct ResolvedPathway {
    pub stimulus_indices: Vec<u32>,
    pub relay_indices: Vec<u32>,
    pub readout_indices: BTreeMap<String, Vec<u32>>,
    pub anatomical_edge_count: usize,
}

impl CnsPathway {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_slice(&bytes).context("decoding CNS pathway")
    }

    pub fn fingerprint(&self) -> Result<String> {
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }

    pub fn resolve(&self, pack: &ConnectomePack) -> Result<ResolvedPathway> {
        if self.schema_version != 1 || self.name.is_empty() || self.interpretation.is_empty() {
            bail!("CNS pathway needs schema version 1, a name, and an interpretation")
        }
        if self.evidence.is_empty() || self.evidence.iter().any(|value| value.is_empty()) {
            bail!("CNS pathway must cite its evidence")
        }
        if self.materialization != pack.materialization() {
            bail!("CNS pathway materialization does not match the pack")
        }
        let expected_arrays = [
            "neuron_ids.npy",
            "row_ptr.npy",
            "destinations.npy",
            "signed_counts.npy",
        ];
        if self.pack_array_sha256.len() != expected_arrays.len()
            || expected_arrays.iter().any(|name| {
                self.pack_array_sha256.get(*name).is_none_or(|digest| {
                    digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit())
                })
            })
            || self.pack_array_sha256 != pack.manifest.array_sha256
        {
            bail!("CNS pathway must match all four audited pack array hashes")
        }
        let index_by_id: BTreeMap<u64, u32> = pack
            .neuron_ids
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, index as u32))
            .collect();
        let stimulus_indices = resolve_group("stimulus", &self.stimulus_ids, &index_by_id)?;
        let relay_indices = resolve_group("relay", &self.relay_ids, &index_by_id)?;
        if self
            .stimulus_ids
            .iter()
            .any(|id| self.relay_ids.contains(id))
        {
            bail!("stimulus and relay neurons must be disjoint")
        }
        if self.readout_groups.is_empty() {
            bail!("CNS pathway needs motor readout groups")
        }
        let mut readout_indices = BTreeMap::new();
        let mut readout_ids = BTreeSet::new();
        for (name, ids) in &self.readout_groups {
            if name.is_empty() {
                bail!("CNS pathway readout group name is empty")
            }
            let indices = resolve_group(name, ids, &index_by_id)?;
            for id in ids {
                if self.stimulus_ids.contains(id)
                    || self.relay_ids.contains(id)
                    || !readout_ids.insert(*id)
                {
                    bail!("motor readouts must be disjoint from inputs, relays, and each other")
                }
            }
            readout_indices.insert(name.clone(), indices);
        }
        let mut witnessed_readouts = BTreeSet::new();
        let mut witnessed_relays = BTreeSet::new();
        let mut witnessed_edges = BTreeSet::new();
        for path in &self.anatomical_paths {
            if path.len() < 3
                || path.len() > 5
                || !self.stimulus_ids.contains(&path[0])
                || !readout_ids.contains(path.last().unwrap())
                || !self.relay_ids.contains(&path[1])
                || path.iter().copied().collect::<BTreeSet<_>>().len() != path.len()
            {
                bail!(
                    "each anatomical path must start input -> relay and reach a motor readout through at most two intermediates without cycles"
                )
            }
            for pair in path.windows(2) {
                let source = *index_by_id
                    .get(&pair[0])
                    .context("path source ID missing")?;
                let destination = *index_by_id
                    .get(&pair[1])
                    .context("path target ID missing")?;
                let edges = pack.row_ptr[source as usize] as usize
                    ..pack.row_ptr[source as usize + 1] as usize;
                if !edges.into_iter().any(|edge| {
                    pack.destinations[edge] == destination && pack.signed_counts[edge] != 0
                }) {
                    bail!(
                        "anatomical path contains absent edge {} -> {}",
                        pair[0],
                        pair[1]
                    )
                }
                witnessed_edges.insert((pair[0], pair[1]));
            }
            witnessed_readouts.insert(*path.last().unwrap());
            witnessed_relays.insert(path[1]);
        }
        if witnessed_readouts != readout_ids {
            bail!("every motor readout needs an anatomical path through a selected relay")
        }
        if witnessed_relays != self.relay_ids.iter().copied().collect() {
            bail!("every selected relay needs an anatomical path to a motor readout")
        }
        Ok(ResolvedPathway {
            stimulus_indices,
            relay_indices,
            readout_indices,
            anatomical_edge_count: witnessed_edges.len(),
        })
    }
}

impl ResolvedPathway {
    pub fn schedule(
        &self,
        control: PathwayControl,
        steps: usize,
        rate_hz: f64,
        seed: u64,
        parameters: ModelParameters,
        neuron_count: usize,
    ) -> Result<EventSchedule> {
        parameters.validate()?;
        if steps == 0 {
            bail!("pathway experiment requires at least one step")
        }
        let targets = if control == PathwayControl::RelayDriven {
            &self.relay_indices
        } else {
            &self.stimulus_indices
        };
        let schedule = EventSchedule::bernoulli(
            targets.clone(),
            steps,
            rate_hz,
            parameters.dt_ms,
            seed,
            neuron_count,
        )?;
        if control == PathwayControl::NoInput {
            Ok(EventSchedule::empty(steps, neuron_count))
        } else {
            Ok(schedule)
        }
    }

    pub fn silenced_sources(&self, control: PathwayControl) -> &[u32] {
        match control {
            PathwayControl::Intact | PathwayControl::NoInput => &[],
            PathwayControl::InputDisconnected => &self.stimulus_indices,
            PathwayControl::RelayDisconnected => &self.relay_indices,
            PathwayControl::RelayDriven => &self.stimulus_indices,
        }
    }
}

fn resolve_group(name: &str, ids: &[u64], indices: &BTreeMap<u64, u32>) -> Result<Vec<u32>> {
    if ids.is_empty() || ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        bail!("CNS pathway {name} IDs must be nonempty, unique, and sorted")
    }
    ids.iter()
        .map(|id| {
            indices
                .get(id)
                .copied()
                .with_context(|| format!("{name} neuron {id} missing from pack"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::CpuEngine;

    fn fixture() -> (ConnectomePack, CnsPathway) {
        let mut pack =
            ConnectomePack::from_arrays([10, 20, 30], [0, 1, 2, 2], [1, 2], [1000, 1000]).unwrap();
        pack.manifest.materialization = "test-cns".into();
        pack.manifest.array_sha256 = [
            "neuron_ids.npy",
            "row_ptr.npy",
            "destinations.npy",
            "signed_counts.npy",
        ]
        .into_iter()
        .map(|name| (name.into(), "a".repeat(64)))
        .collect();
        let pathway = CnsPathway {
            schema_version: 1,
            name: "synthetic chain".into(),
            materialization: "test-cns".into(),
            pack_array_sha256: pack.manifest.array_sha256.clone(),
            evidence: vec!["synthetic test fixture, not biological data".into()],
            interpretation: "test of assay mechanics".into(),
            stimulus_ids: vec![10],
            relay_ids: vec![20],
            readout_groups: BTreeMap::from([("motor".into(), vec![30])]),
            anatomical_paths: vec![vec![10, 20, 30]],
        };
        (pack, pathway)
    }

    #[test]
    fn pathway_requires_exact_dataset_identity_and_real_edges() {
        let (pack, mut pathway) = fixture();
        assert_eq!(pathway.resolve(&pack).unwrap().anatomical_edge_count, 2);
        pathway.materialization = "another-cns".into();
        assert!(pathway.resolve(&pack).is_err());
        pathway.materialization = "test-cns".into();
        pathway.anatomical_paths = vec![vec![10, 30, 20, 30]];
        assert!(pathway.resolve(&pack).is_err());
    }

    #[test]
    fn pathway_rejects_hash_mismatch_missing_ids_and_overlapping_roles() {
        let (pack, pathway) = fixture();
        let mut bad = pathway.clone();
        bad.pack_array_sha256
            .insert("signed_counts.npy".into(), "b".repeat(64));
        assert!(bad.resolve(&pack).is_err());
        let mut bad = pathway.clone();
        bad.stimulus_ids = vec![11];
        assert!(bad.resolve(&pack).is_err());
        let mut bad = pathway;
        bad.relay_ids = vec![10];
        assert!(bad.resolve(&pack).is_err());
    }

    #[test]
    fn controls_separate_input_activity_from_synaptic_propagation() {
        let (pack, pathway) = fixture();
        let resolved = pathway.resolve(&pack).unwrap();
        let parameters = ModelParameters::default();
        for control in [
            PathwayControl::Intact,
            PathwayControl::NoInput,
            PathwayControl::InputDisconnected,
            PathwayControl::RelayDisconnected,
        ] {
            let schedule = resolved
                .schedule(control, 1000, 150.0, 1, parameters, 3)
                .unwrap();
            let mut engine = CpuEngine::new(
                &pack,
                parameters,
                None,
                None,
                &resolved.stimulus_indices,
                resolved.silenced_sources(control),
            )
            .unwrap();
            engine.run_schedule(&schedule, false).unwrap();
            let counts = engine.spike_counts();
            assert_eq!(counts[2] > 0, control == PathwayControl::Intact);
            assert_eq!(counts[0] > 0, control != PathwayControl::NoInput);
            if control == PathwayControl::RelayDisconnected {
                assert!(counts[1] > 0);
            }
        }
    }

    #[test]
    fn direct_relay_drive_bypasses_input_and_preserves_motor_outputs() {
        let (pack, pathway) = fixture();
        let resolved = pathway.resolve(&pack).unwrap();
        let parameters = ModelParameters::default();
        let control = PathwayControl::RelayDriven;
        let schedule = resolved
            .schedule(control, 1000, 150.0, 1, parameters, 3)
            .unwrap();
        assert_eq!(schedule.targets(), resolved.relay_indices);
        let mut engine = CpuEngine::new(
            &pack,
            parameters,
            None,
            None,
            &resolved.relay_indices,
            resolved.silenced_sources(control),
        )
        .unwrap();
        engine.run_schedule(&schedule, false).unwrap();
        assert_eq!(engine.spike_counts()[0], 0);
        assert!(engine.spike_counts()[1] > 0);
        assert!(engine.spike_counts()[2] > 0);
    }
}
