use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::pack::ConnectomePack;

pub const MALE_CNS_MATERIALIZATION: &str = "male-cns-v1.0-superclass-non-null-known-nt";

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct NeuralIoArtifact {
    pub schema_version: u64,
    pub dataset: NeuralIoDataset,
    pub groups: BTreeMap<String, NeuralIoGroup>,
    #[serde(default)]
    pub food_olfaction: Option<FoodOlfactionProfiles>,
    #[serde(default)]
    pub summary: Option<NeuralIoSummary>,
    #[serde(default)]
    pub pack_resolution: Option<NeuralIoPackResolution>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct FoodOlfactionProfiles {
    pub schema: String,
    pub reference_odor: String,
    pub concentration_unit: String,
    pub evidence_source: String,
    pub annotation_field: String,
    pub selection_rule: String,
    pub channels: BTreeMap<String, FoodOlfactionChannel>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct FoodOlfactionChannel {
    pub glomerulus: String,
    pub side: String,
    pub response_band: String,
    pub root_ids: Vec<u64>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct NeuralIoDataset {
    pub name: String,
    pub materialization: String,
    pub annotation_release: String,
    pub annotation_commit: String,
    pub annotation_source: String,
    pub annotation_sha256: String,
    pub pack_neuron_ids_sha256: String,
    #[serde(default)]
    pub pack_array_sha256: BTreeMap<String, String>,
    pub pack_path: String,
    pub selection_provenance: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct NeuralIoGroup {
    pub evidence_category: String,
    pub selector: String,
    pub biological_scope: String,
    #[serde(default)]
    pub side: Option<String>,
    pub root_ids: Vec<u64>,
    #[serde(default)]
    pub pack_resolution: Option<NeuralIoGroupPackResolution>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct NeuralIoSummary {
    pub group_count: usize,
    pub selected_root_ids: usize,
    pub present_root_ids: usize,
    pub missing_root_ids: usize,
    pub category_counts: BTreeMap<String, NeuralIoSummaryCategory>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct NeuralIoSummaryCategory {
    pub groups: usize,
    pub selected_root_ids: usize,
    pub present_root_ids: usize,
    pub missing_root_ids: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct NeuralIoGroupPackResolution {
    pub selected_count: usize,
    pub present_count: usize,
    pub missing_root_ids: Vec<u64>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct NeuralIoPackResolution {
    pub materialization: String,
    pub pack_neuron_count: usize,
    pub pack_neuron_ids_sha256: String,
    pub groups: BTreeMap<String, NeuralIoGroupPackResolution>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeuralIoResolution {
    pub artifact: NeuralIoArtifact,
    pub groups: BTreeMap<String, ResolvedNeuralIoGroup>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedNeuralIoGroup {
    pub selected_root_ids: Vec<u64>,
    pub engine_indices: Vec<u32>,
    pub missing_root_ids: Vec<u64>,
}

impl NeuralIoArtifact {
    pub fn is_male_cns(&self) -> bool {
        self.dataset.materialization == MALE_CNS_MATERIALIZATION
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_bytes(&bytes).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let artifact: Self = serde_json::from_slice(bytes).context("decoding neural I/O JSON")?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn resolve(&self, pack: &ConnectomePack) -> Result<NeuralIoResolution> {
        self.validate()?;
        if self.dataset.materialization != pack.materialization() {
            bail!(
                "neural I/O materialization {} does not match connectome materialization {}",
                self.dataset.materialization,
                pack.materialization()
            )
        }
        if let Some(expected_hash) = pack.manifest.array_sha256.get("neuron_ids.npy") {
            if expected_hash != &self.dataset.pack_neuron_ids_sha256 {
                bail!(
                    "neural I/O neuron ID SHA256 {} does not match pack {}",
                    self.dataset.pack_neuron_ids_sha256,
                    expected_hash
                )
            }
        }
        if self.is_male_cns()
            && (self.dataset.pack_array_sha256.len() != 4
                || self.dataset.pack_array_sha256 != pack.manifest.array_sha256
                || pack.manifest.source_sha256.get("annotations")
                    != Some(&self.dataset.annotation_sha256))
        {
            bail!(
                "MaleCNS neural I/O must match the source annotation and all four audited pack arrays"
            )
        }

        let index_by_id: HashMap<u64, u32> = pack
            .neuron_ids
            .iter()
            .enumerate()
            .map(|(index, &root_id)| {
                (
                    root_id,
                    u32::try_from(index).expect("connectome index validated as uint32"),
                )
            })
            .collect();
        let mut groups = BTreeMap::new();
        for (name, group) in &self.groups {
            let mut engine_indices = Vec::with_capacity(group.root_ids.len());
            let mut missing_root_ids = Vec::new();
            for &root_id in &group.root_ids {
                match index_by_id.get(&root_id) {
                    Some(&index) => engine_indices.push(index),
                    None => missing_root_ids.push(root_id),
                }
            }

            let resolved = ResolvedNeuralIoGroup {
                selected_root_ids: group.root_ids.clone(),
                engine_indices,
                missing_root_ids,
            };
            validate_static_group_resolution(
                name,
                group,
                &resolved,
                self.pack_resolution.as_ref(),
            )?;
            groups.insert(name.clone(), resolved);
        }
        if let Some(pack_resolution) = &self.pack_resolution {
            if pack_resolution.materialization != pack.materialization() {
                bail!(
                    "neural I/O pack resolution materialization {} does not match connectome {}",
                    pack_resolution.materialization,
                    pack.materialization()
                )
            }
            if pack_resolution.pack_neuron_count != pack.neuron_count() {
                bail!(
                    "neural I/O pack resolution has {} neurons, connectome has {}",
                    pack_resolution.pack_neuron_count,
                    pack.neuron_count()
                )
            }
            if pack_resolution.pack_neuron_ids_sha256 != self.dataset.pack_neuron_ids_sha256 {
                bail!("neural I/O pack resolution hash disagrees with dataset hash")
            }
            if pack_resolution.groups.len() != self.groups.len()
                || self
                    .groups
                    .keys()
                    .any(|name| !pack_resolution.groups.contains_key(name))
            {
                bail!("neural I/O pack resolution groups do not match selected groups")
            }
        }

        Ok(NeuralIoResolution {
            artifact: self.clone(),
            groups,
        })
    }

    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            bail!(
                "unsupported neural I/O schema version {}; expected 1",
                self.schema_version
            )
        }
        if self.dataset.materialization != "783" && !self.is_male_cns() {
            bail!(
                "unsupported neural I/O materialization {}; expected 783 or MaleCNS v1.0",
                self.dataset.materialization
            )
        }
        for (name, value) in [
            ("annotation_sha256", &self.dataset.annotation_sha256),
            (
                "pack_neuron_ids_sha256",
                &self.dataset.pack_neuron_ids_sha256,
            ),
        ] {
            if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                bail!("neural I/O dataset has an invalid {name}")
            }
        }
        if self.dataset.name.is_empty()
            || self.dataset.annotation_release.is_empty()
            || self.dataset.annotation_commit.is_empty()
            || self.dataset.annotation_source.is_empty()
            || self.dataset.pack_path.is_empty()
            || self.dataset.selection_provenance.is_empty()
        {
            bail!("neural I/O dataset provenance fields must not be empty")
        }
        if self.groups.is_empty() {
            bail!("neural I/O artifact must contain at least one group")
        }
        for (name, group) in &self.groups {
            if name.is_empty()
                || group.evidence_category.is_empty()
                || group.selector.is_empty()
                || group.biological_scope.is_empty()
            {
                bail!("neural I/O group {name:?} has an empty metadata field")
            }
            if let Some(side) = &group.side {
                if !matches!(side.as_str(), "left" | "right" | "both") {
                    bail!("neural I/O group {name:?} has unsupported side {side:?}")
                }
            }
            if group.root_ids.is_empty() {
                bail!("neural I/O group {name:?} is empty")
            }
            let mut ids = HashSet::with_capacity(group.root_ids.len());
            if group.root_ids.iter().any(|root_id| !ids.insert(*root_id)) {
                bail!("neural I/O group {name:?} contains duplicate root IDs")
            }
            if group
                .root_ids
                .windows(2)
                .any(|window| window[1] <= window[0])
            {
                bail!("neural I/O group {name:?} root IDs must be strictly sorted")
            }
            if let Some(resolution) = &group.pack_resolution {
                validate_group_pack_resolution(name, group.root_ids.len(), resolution)?;
            }
        }
        if let Some(profiles) = &self.food_olfaction {
            self.validate_food_olfaction(profiles)?;
        }
        if let Some(pack_resolution) = &self.pack_resolution {
            if pack_resolution.materialization != self.dataset.materialization {
                bail!("neural I/O pack resolution materialization disagrees with dataset")
            }
            if pack_resolution.pack_neuron_ids_sha256 != self.dataset.pack_neuron_ids_sha256 {
                bail!("neural I/O pack resolution hash disagrees with dataset")
            }
            if pack_resolution.groups.len() != self.groups.len()
                || self
                    .groups
                    .keys()
                    .any(|name| !pack_resolution.groups.contains_key(name))
            {
                bail!("neural I/O pack resolution groups do not match selected groups")
            }
            for (name, resolution) in &pack_resolution.groups {
                let selected_count = self
                    .groups
                    .get(name)
                    .expect("pack resolution group keys validated")
                    .root_ids
                    .len();
                validate_group_pack_resolution(name, selected_count, resolution)?;
            }
        }
        if let Some(summary) = &self.summary {
            if summary.group_count != self.groups.len() {
                bail!("neural I/O summary group count does not match groups")
            }
            let mut selected_root_ids = 0;
            let mut present_root_ids = 0;
            let mut missing_root_ids = 0;
            let mut category_counts = BTreeMap::new();
            for group in self.groups.values() {
                let resolution = group
                    .pack_resolution
                    .as_ref()
                    .context("neural I/O summary requires group pack resolutions")?;
                selected_root_ids += resolution.selected_count;
                present_root_ids += resolution.present_count;
                missing_root_ids += resolution.missing_root_ids.len();
                let category = category_counts
                    .entry(group.evidence_category.clone())
                    .or_insert((0_usize, 0_usize, 0_usize, 0_usize));
                category.0 += 1;
                category.1 += resolution.selected_count;
                category.2 += resolution.present_count;
                category.3 += resolution.missing_root_ids.len();
            }
            if summary.selected_root_ids != selected_root_ids
                || summary.present_root_ids != present_root_ids
                || summary.missing_root_ids != missing_root_ids
            {
                bail!("neural I/O summary totals do not match groups")
            }
            if summary.category_counts.len() != category_counts.len()
                || summary.category_counts.iter().any(|(name, value)| {
                    category_counts.get(name)
                        != Some(&(
                            value.groups,
                            value.selected_root_ids,
                            value.present_root_ids,
                            value.missing_root_ids,
                        ))
                })
            {
                bail!("neural I/O summary categories do not match groups")
            }
        }
        Ok(())
    }

    fn validate_food_olfaction(&self, profiles: &FoodOlfactionProfiles) -> Result<()> {
        if profiles.schema != "flybrain-food-olfaction-v1"
            || profiles.reference_odor != "apple_cider_vinegar"
            || profiles.concentration_unit != "isobutylene-equivalent ppm"
            || profiles.evidence_source.is_empty()
            || profiles.annotation_field
                != if self.is_male_cns() {
                    "type"
                } else {
                    "hemibrain_type"
                }
            || profiles.selection_rule.is_empty()
            || profiles.channels.is_empty()
        {
            bail!("food-olfaction profile metadata is invalid")
        }
        let left = &self
            .groups
            .get("olfaction_left")
            .context("food-olfaction profile requires olfaction_left")?
            .root_ids;
        let right = &self
            .groups
            .get("olfaction_right")
            .context("food-olfaction profile requires olfaction_right")?
            .root_ids;
        let mut selected_ids = HashSet::new();
        for (name, channel) in &profiles.channels {
            if name.is_empty()
                || channel.glomerulus.is_empty()
                || !matches!(channel.side.as_str(), "left" | "right")
                || !matches!(
                    channel.response_band.as_str(),
                    "attractive" | "core" | "high_concentration" | "aversive_high_concentration"
                )
                || channel.root_ids.is_empty()
            {
                bail!("food-olfaction channel {name:?} has invalid metadata")
            }
            if channel
                .root_ids
                .windows(2)
                .any(|window| window[1] <= window[0])
            {
                bail!("food-olfaction channel {name:?} root IDs must be strictly sorted")
            }
            let side_population = if channel.side == "left" { left } else { right };
            for &root_id in &channel.root_ids {
                if side_population.binary_search(&root_id).is_err() {
                    bail!(
                        "food-olfaction channel {name:?} contains root {root_id} outside its side population"
                    )
                }
                if !selected_ids.insert(root_id) {
                    bail!("food-olfaction root {root_id} appears in more than one channel")
                }
            }
        }
        Ok(())
    }
}

impl NeuralIoResolution {
    pub fn group(&self, name: &str) -> Option<&ResolvedNeuralIoGroup> {
        self.groups.get(name)
    }

    pub fn group_indices(&self, name: &str) -> Option<&[u32]> {
        self.group(name)
            .map(|group| group.engine_indices.as_slice())
    }

    pub fn group_missing_root_ids(&self, name: &str) -> Option<&[u64]> {
        self.group(name)
            .map(|group| group.missing_root_ids.as_slice())
    }
}

fn validate_group_pack_resolution(
    name: &str,
    selected_count: usize,
    resolution: &NeuralIoGroupPackResolution,
) -> Result<()> {
    if resolution.selected_count != selected_count {
        bail!(
            "neural I/O group {name:?} selected count {} does not match {} root IDs",
            resolution.selected_count,
            selected_count
        )
    }
    if resolution.present_count > resolution.selected_count {
        bail!("neural I/O group {name:?} present count exceeds selected count")
    }
    if resolution
        .missing_root_ids
        .windows(2)
        .any(|window| window[1] <= window[0])
    {
        bail!("neural I/O group {name:?} missing IDs must be strictly sorted")
    }
    if resolution.present_count + resolution.missing_root_ids.len() != resolution.selected_count {
        bail!("neural I/O group {name:?} resolution counts do not add up")
    }
    Ok(())
}

fn validate_static_group_resolution(
    name: &str,
    group: &NeuralIoGroup,
    resolved: &ResolvedNeuralIoGroup,
    pack_resolution: Option<&NeuralIoPackResolution>,
) -> Result<()> {
    if let Some(static_resolution) = &group.pack_resolution {
        if static_resolution.present_count != resolved.engine_indices.len()
            || static_resolution.missing_root_ids != resolved.missing_root_ids
        {
            bail!("neural I/O group {name:?} static resolution does not match pack")
        }
    }
    if let Some(pack_resolution) = pack_resolution {
        let static_resolution = pack_resolution
            .groups
            .get(name)
            .with_context(|| format!("neural I/O pack resolution is missing group {name:?}"))?;
        if static_resolution.present_count != resolved.engine_indices.len()
            || static_resolution.missing_root_ids != resolved.missing_root_ids
        {
            bail!("neural I/O group {name:?} pack resolution does not match pack")
        }
    }
    if group.root_ids.len() != resolved.engine_indices.len() + resolved.missing_root_ids.len() {
        bail!("neural I/O group {name:?} resolution does not cover all root IDs")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::pack::ConnectomePack;

    fn artifact_json(group_root_ids: &str) -> String {
        format!(
            r#"{{
                "schema_version": 1,
                "dataset": {{
                    "name": "test",
                    "materialization": "783",
                    "annotation_release": "test",
                    "annotation_commit": "test",
                    "annotation_source": "https://example.test/annotations.tsv",
                    "annotation_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "pack_neuron_ids_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "pack_path": "test",
                    "selection_provenance": "test"
                }},
                "groups": {{
                    "test_group": {{
                        "evidence_category": "test",
                        "selector": "test",
                        "biological_scope": "test",
                        "root_ids": [{group_root_ids}]
                    }}
                }}
            }}"#
        )
    }

    fn test_pack() -> ConnectomePack {
        let mut pack =
            ConnectomePack::from_arrays([11_u64, 22_u64], [0_u32, 0_u32, 0_u32], [], []).unwrap();
        pack.manifest.materialization = "783".to_string();
        pack
    }

    #[test]
    fn resolves_root_ids_to_engine_indices_and_reports_missing_ids() {
        let artifact = NeuralIoArtifact::from_bytes(artifact_json("11, 99, 22").as_bytes());
        assert!(artifact.is_err());

        let artifact =
            NeuralIoArtifact::from_bytes(artifact_json("11, 22, 99").as_bytes()).unwrap();
        let resolution = artifact.resolve(&test_pack()).unwrap();
        assert_eq!(
            resolution.group_indices("test_group"),
            Some([0, 1].as_slice())
        );
        assert_eq!(
            resolution.group_missing_root_ids("test_group"),
            Some([99_u64].as_slice())
        );
    }

    #[test]
    fn rejects_materialization_mismatch() {
        let mut pack = test_pack();
        pack.manifest.materialization = "630".to_string();
        let artifact = NeuralIoArtifact::from_bytes(artifact_json("11, 22").as_bytes()).unwrap();
        let error = artifact.resolve(&pack).unwrap_err().to_string();
        assert!(error.contains("does not match connectome materialization"));
    }

    #[test]
    fn rejects_unsorted_or_duplicate_group_ids() {
        let unsorted = NeuralIoArtifact::from_bytes(artifact_json("22, 11").as_bytes());
        assert!(
            unsorted
                .unwrap_err()
                .to_string()
                .contains("strictly sorted")
        );

        let duplicate = NeuralIoArtifact::from_bytes(artifact_json("11, 11").as_bytes());
        assert!(duplicate.unwrap_err().to_string().contains("duplicate"));
    }

    #[test]
    fn pinned_v783_artifact_resolves_against_local_pack() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let artifact =
            NeuralIoArtifact::load(root.join("assets/neuromechfly/flywire_v783_neural_io.json"))
                .unwrap();
        let pack = ConnectomePack::open(root.join("outputs/packs/flywire_v783")).unwrap();
        let resolution = artifact.resolve(&pack).unwrap();

        let food_olfaction = resolution.artifact.food_olfaction.as_ref().unwrap();
        assert_eq!(food_olfaction.channels.len(), 16);
        assert_eq!(
            food_olfaction
                .channels
                .values()
                .map(|channel| channel.root_ids.len())
                .sum::<usize>(),
            401
        );

        assert_eq!(
            resolution.group_indices("olfaction_left").unwrap().len(),
            1038
        );
        assert_eq!(
            resolution
                .group_missing_root_ids("olfaction_left")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            resolution.group_indices("olfaction_right").unwrap().len(),
            1052
        );
        assert_eq!(
            resolution.group_indices("visual_left_r1_6").unwrap().len(),
            4044
        );
        assert_eq!(
            resolution
                .group_missing_root_ids("visual_left_r1_6")
                .unwrap()
                .len(),
            379
        );
        assert_eq!(
            resolution
                .group_indices("walking_dna02_left")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            resolution
                .group_indices("walking_dna02_right")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            resolution.group_indices("flight_dng02_left").unwrap().len(),
            13
        );
        assert_eq!(
            resolution
                .group_indices("flight_dng02_right")
                .unwrap()
                .len(),
            12
        );
        assert_eq!(
            resolution.group_indices("flight_dng07_left").unwrap().len(),
            8
        );
        assert_eq!(
            resolution
                .group_indices("flight_dng07_right")
                .unwrap()
                .len(),
            8
        );
        assert_eq!(
            resolution.artifact.groups["landing_dnp07_left"].root_ids,
            vec![720575940617034565]
        );
        assert_eq!(
            resolution
                .group_indices("landing_dnp07_left")
                .unwrap()
                .len(),
            1
        );
        assert!(
            resolution
                .group_missing_root_ids("landing_dnp07_left")
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            resolution.artifact.groups["landing_dnp07_right"].root_ids,
            vec![720575940622550089]
        );
        assert_eq!(
            resolution
                .group_indices("landing_dnp07_right")
                .unwrap()
                .len(),
            1
        );
        assert!(
            resolution
                .group_missing_root_ids("landing_dnp07_right")
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            resolution.artifact.groups["landing_dnp10_left"].root_ids,
            vec![720575940620153765]
        );
        assert_eq!(
            resolution
                .group_indices("landing_dnp10_left")
                .unwrap()
                .len(),
            1
        );
        assert!(
            resolution
                .group_missing_root_ids("landing_dnp10_left")
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            resolution.artifact.groups["landing_dnp10_right"].root_ids,
            vec![720575940619678366]
        );
        assert_eq!(
            resolution
                .group_indices("landing_dnp10_right")
                .unwrap()
                .len(),
            1
        );
        assert!(
            resolution
                .group_missing_root_ids("landing_dnp10_right")
                .unwrap()
                .is_empty()
        );
        for group in [
            "flight_state_msahn_left",
            "flight_state_msahn_right",
            "flight_state_mtahn_left",
            "flight_state_mtahn_right",
        ] {
            assert_eq!(resolution.group_indices(group).unwrap().len(), 1);
            assert!(resolution.group_missing_root_ids(group).unwrap().is_empty());
        }
    }

    #[test]
    fn pinned_male_cns_artifact_resolves_every_group_against_the_pack() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let artifact =
            NeuralIoArtifact::load(root.join("assets/neuromechfly/male_cns_v1_neural_io.json"))
                .unwrap();
        let pack = ConnectomePack::open(root.join("outputs/packs/male_cns_v1")).unwrap();
        let resolution = artifact.resolve(&pack).unwrap();

        assert!(artifact.is_male_cns());
        assert_eq!(resolution.groups.len(), artifact.groups.len());
        assert_eq!(resolution.groups.len(), 28);
        for name in artifact.groups.keys() {
            let group = resolution.group(name).expect("artifact group resolves");
            assert_eq!(
                group.selected_root_ids.len(),
                artifact.groups[name].root_ids.len()
            );
            assert_eq!(group.engine_indices.len(), group.selected_root_ids.len());
            assert!(
                group.missing_root_ids.is_empty(),
                "group {name} is incomplete"
            );
        }
        assert_eq!(
            resolution.group("feeding_mn9").unwrap().selected_root_ids,
            [10331]
        );
        assert_eq!(
            resolution
                .group("taste_sugar")
                .unwrap()
                .engine_indices
                .len(),
            85
        );
    }

    #[test]
    fn male_cns_rejects_a_tampered_connectivity_array_hash_with_matching_neuron_ids() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut artifact =
            NeuralIoArtifact::load(root.join("assets/neuromechfly/male_cns_v1_neural_io.json"))
                .unwrap();
        let pack = ConnectomePack::open(root.join("outputs/packs/male_cns_v1")).unwrap();
        let neuron_ids_hash = artifact.dataset.pack_neuron_ids_sha256.clone();
        artifact
            .dataset
            .pack_array_sha256
            .insert("destinations.npy".to_string(), "f".repeat(64));

        assert_eq!(artifact.dataset.pack_neuron_ids_sha256, neuron_ids_hash);
        let error = artifact
            .resolve(&pack)
            .expect_err("tampered connectivity hash must fail closed")
            .to_string();
        assert!(error.contains("all four audited pack arrays"));
    }

    #[test]
    fn male_cns_rejects_an_annotation_hash_not_bound_to_the_pack() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut artifact =
            NeuralIoArtifact::load(root.join("assets/neuromechfly/male_cns_v1_neural_io.json"))
                .unwrap();
        let pack = ConnectomePack::open(root.join("outputs/packs/male_cns_v1")).unwrap();
        artifact.dataset.annotation_sha256 = "0".repeat(64);

        let error = artifact
            .resolve(&pack)
            .expect_err("wrong annotation binding must fail closed")
            .to_string();
        assert!(error.contains("source annotation"));
    }
}
