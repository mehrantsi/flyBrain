use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::neural_io::{FoodOlfactionChannel, NeuralIoResolution};
use crate::pack::ConnectomePack;

pub const CNS_OLFACTION_FILTER_TAU_MS: f64 = 200.0;

const MATCHED_CHANNELS: [(&str, &str); 2] = [("DM1", "attractive"), ("DM2", "core")];
#[cfg(test)]
use crate::olfaction::ORN_SPONTANEOUS_RATE_HZ;
const FOOD_ODOR_HILL_EXPONENT: f64 = 1.42;
const DM2_HALF_MAX_PPM: f64 = 3.0;
const DM1_EVOKED_RATE_THRESHOLD_HZ: f64 = 1.0;
const CONCENTRATION_RATIO_UPPER_BOUND_FOR_FINITE_INVERSION: f64 = 0.999;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct CnsOlfactoryReadout {
    pub observed_seconds: f64,
    pub rate_hz: [f64; 2],
    pub band_rate_hz: [[f64; 2]; 2],
    pub concentration_ppm: [f64; 2],
    pub contrast: f64,
    pub spike_delta: u64,
}

#[derive(Debug)]
pub struct CnsOlfactoryProbes {
    positions: [[Vec<usize>; 2]; 2],
    unique_positions: Vec<usize>,
    filtered_band_rates_hz: [[f64; 2]; 2],
    filter_weight: f64,
    observed_seconds: f64,
}

impl CnsOlfactoryProbes {
    pub fn new(
        pack: &ConnectomePack,
        resolution: &NeuralIoResolution,
        probe_indices: &mut Vec<u32>,
        position_by_index: &mut BTreeMap<u32, usize>,
    ) -> Result<Self> {
        let profiles = resolution
            .artifact
            .food_olfaction
            .as_ref()
            .context("MaleCNS neural I/O is missing food-olfaction profiles")?;
        let mut channels: [[Option<&FoodOlfactionChannel>; 2]; 2] = [[None, None], [None, None]];
        for channel in profiles.channels.values() {
            let Some((side, glomerulus)) = matched_channel(channel) else {
                continue;
            };
            if channels[side][glomerulus].replace(channel).is_some() {
                bail!(
                    "food-olfaction has duplicate {} {} channel on {} side",
                    MATCHED_CHANNELS[glomerulus].0,
                    MATCHED_CHANNELS[glomerulus].1,
                    side_name(side),
                )
            }
        }

        let mut positions: [[Vec<usize>; 2]; 2] = Default::default();
        let mut unique_positions = BTreeSet::new();
        for side in 0..2 {
            let group_name = format!("olfaction_{}", side_name(side));
            let group = resolution
                .group(&group_name)
                .with_context(|| format!("neural I/O artifact is missing group {group_name}"))?;
            for glomerulus in 0..MATCHED_CHANNELS.len() {
                let channel = channels[side][glomerulus].with_context(|| {
                    format!(
                        "food-olfaction is missing matched {} {} channel on {} side",
                        MATCHED_CHANNELS[glomerulus].0,
                        MATCHED_CHANNELS[glomerulus].1,
                        side_name(side),
                    )
                })?;
                if channel.root_ids.is_empty() {
                    bail!(
                        "food-olfaction matched {} {} channel on {} side is empty",
                        MATCHED_CHANNELS[glomerulus].0,
                        MATCHED_CHANNELS[glomerulus].1,
                        side_name(side),
                    )
                }
                let mut channel_positions = BTreeSet::new();
                for &root_id in &channel.root_ids {
                    let index = group.engine_indices.iter().find_map(|&index| {
                        pack.neuron_ids
                            .get(index as usize)
                            .filter(|&&candidate| candidate == root_id)
                            .map(|_| index)
                    });
                    let index = match index {
                        Some(index) => index,
                        None if !pack.neuron_ids.contains(&root_id) => {
                            bail!(
                                "food-olfaction matched root {root_id} is missing from the connectome pack"
                            )
                        }
                        None => {
                            bail!(
                                "food-olfaction matched root {root_id} has no resolved pack index"
                            )
                        }
                    };
                    if !group.selected_root_ids.contains(&root_id) {
                        bail!(
                            "food-olfaction root {root_id} is outside the {} side ORN group",
                            side_name(side),
                        )
                    }
                    let position = *position_by_index.entry(index).or_insert_with(|| {
                        let position = probe_indices.len();
                        probe_indices.push(index);
                        position
                    });
                    channel_positions.insert(position);
                    unique_positions.insert(position);
                }
                positions[side][glomerulus] = channel_positions.into_iter().collect();
            }
        }

        Ok(Self {
            positions,
            unique_positions: unique_positions.into_iter().collect(),
            filtered_band_rates_hz: [[0.0; 2]; 2],
            filter_weight: 0.0,
            observed_seconds: 0.0,
        })
    }

    pub fn update(
        &mut self,
        deltas: &[u32],
        window_ms: f64,
        baseline_rate_hz: f64,
    ) -> CnsOlfactoryReadout {
        assert!(
            window_ms.is_finite() && window_ms > 0.0,
            "CNS olfactory window must be finite and positive"
        );
        let alpha = 1.0 - (-window_ms / CNS_OLFACTION_FILTER_TAU_MS).exp();
        self.filter_weight += alpha * (1.0 - self.filter_weight);
        self.observed_seconds += window_ms / 1000.0;
        for side in 0..2 {
            for glomerulus in 0..MATCHED_CHANNELS.len() {
                let rate_hz = population_rate(deltas, &self.positions[side][glomerulus], window_ms);
                self.filtered_band_rates_hz[side][glomerulus] +=
                    alpha * (rate_hz - self.filtered_band_rates_hz[side][glomerulus]);
            }
        }
        let band_rate_hz = self
            .filtered_band_rates_hz
            .map(|rates| rates.map(|rate| rate / self.filter_weight));
        let rate_hz =
            band_rate_hz.map(|[dm1_rate_hz, dm2_rate_hz]| 0.5 * (dm1_rate_hz + dm2_rate_hz));
        let [left_rate_hz, right_rate_hz] = rate_hz;
        let denominator = left_rate_hz + right_rate_hz;
        let contrast = if denominator > f64::EPSILON {
            ((left_rate_hz - right_rate_hz) / denominator).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let spike_delta = self
            .unique_positions
            .iter()
            .map(|&position| u64::from(deltas[position]))
            .sum();
        CnsOlfactoryReadout {
            observed_seconds: self.observed_seconds,
            rate_hz,
            band_rate_hz,
            concentration_ppm: band_rate_hz
                .map(|rates| estimate_concentration_ppm(rates, baseline_rate_hz)),
            contrast,
            spike_delta,
        }
    }
}

fn matched_channel(channel: &FoodOlfactionChannel) -> Option<(usize, usize)> {
    let side = match channel.side.as_str() {
        "left" => 0,
        "right" => 1,
        _ => return None,
    };
    let glomerulus = MATCHED_CHANNELS
        .iter()
        .position(|&(name, band)| channel.glomerulus == name && channel.response_band == band)?;
    Some((side, glomerulus))
}

fn side_name(side: usize) -> &'static str {
    match side {
        0 => "left",
        1 => "right",
        _ => unreachable!("CNS olfactory side is binary"),
    }
}

fn population_rate(deltas: &[u32], positions: &[usize], window_ms: f64) -> f64 {
    if positions.is_empty() {
        return 0.0;
    }
    positions
        .iter()
        .map(|&position| f64::from(deltas[position]))
        .sum::<f64>()
        * 1000.0
        / (window_ms * positions.len() as f64)
}

fn estimate_concentration_ppm([dm1_rate_hz, dm2_rate_hz]: [f64; 2], baseline_rate_hz: f64) -> f64 {
    let dm1_evoked_hz = dm1_rate_hz - baseline_rate_hz;
    if !dm1_evoked_hz.is_finite()
        || !dm2_rate_hz.is_finite()
        || dm1_evoked_hz <= DM1_EVOKED_RATE_THRESHOLD_HZ
    {
        return 0.0;
    }
    let dm2_evoked_hz = dm2_rate_hz - baseline_rate_hz;
    let ratio = (dm2_evoked_hz / dm1_evoked_hz).clamp(
        1.0 / DM2_HALF_MAX_PPM.powf(FOOD_ODOR_HILL_EXPONENT),
        CONCENTRATION_RATIO_UPPER_BOUND_FOR_FINITE_INVERSION,
    );
    let concentration_power =
        (DM2_HALF_MAX_PPM.powf(FOOD_ODOR_HILL_EXPONENT) * ratio - 1.0) / (1.0 - ratio);
    concentration_power
        .max(0.0)
        .powf(1.0 / FOOD_ODOR_HILL_EXPONENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_io::{
        FoodOlfactionProfiles, NeuralIoArtifact, NeuralIoDataset, ResolvedNeuralIoGroup,
    };

    fn test_resolution(
        pack: &ConnectomePack,
        channels: &[(&str, &str, &str, &str, u64)],
    ) -> NeuralIoResolution {
        let food_channels = channels
            .iter()
            .map(|&(name, glomerulus, side, response_band, root_id)| {
                (
                    name.to_string(),
                    FoodOlfactionChannel {
                        glomerulus: glomerulus.to_string(),
                        side: side.to_string(),
                        response_band: response_band.to_string(),
                        root_ids: vec![root_id],
                    },
                )
            })
            .collect();
        let food_olfaction = FoodOlfactionProfiles {
            schema: "flybrain-food-olfaction-v1".to_string(),
            reference_odor: "apple_cider_vinegar".to_string(),
            concentration_unit: "isobutylene-equivalent ppm".to_string(),
            evidence_source: "test".to_string(),
            annotation_field: "type".to_string(),
            selection_rule: "test".to_string(),
            channels: food_channels,
        };
        let group = |side: &str| {
            let selected_root_ids = pack
                .neuron_ids
                .iter()
                .copied()
                .filter(|root_id| {
                    channels
                        .iter()
                        .any(|channel| channel.2 == side && channel.4 == *root_id)
                })
                .collect::<Vec<_>>();
            let engine_indices = selected_root_ids
                .iter()
                .map(|root_id| {
                    pack.neuron_ids
                        .iter()
                        .position(|candidate| candidate == root_id)
                        .unwrap() as u32
                })
                .collect();
            ResolvedNeuralIoGroup {
                selected_root_ids,
                engine_indices,
                missing_root_ids: Vec::new(),
            }
        };
        NeuralIoResolution {
            artifact: NeuralIoArtifact {
                schema_version: 1,
                dataset: NeuralIoDataset {
                    name: "test".to_string(),
                    materialization: "unknown".to_string(),
                    annotation_release: "test".to_string(),
                    annotation_commit: "test".to_string(),
                    annotation_source: "test".to_string(),
                    annotation_sha256: "0".repeat(64),
                    pack_neuron_ids_sha256: "0".repeat(64),
                    pack_array_sha256: BTreeMap::new(),
                    pack_path: "test".to_string(),
                    selection_provenance: "test".to_string(),
                },
                groups: BTreeMap::new(),
                food_olfaction: Some(food_olfaction),
                summary: None,
                pack_resolution: None,
            },
            groups: BTreeMap::from([
                ("olfaction_left".to_string(), group("left")),
                ("olfaction_right".to_string(), group("right")),
            ]),
        }
    }

    fn test_pack() -> ConnectomePack {
        ConnectomePack::from_arrays(
            [11_u64, 12, 21, 22],
            [0_u32; 5],
            Vec::<u32>::new(),
            Vec::<i16>::new(),
        )
        .unwrap()
    }

    fn matched_channels() -> [(&'static str, &'static str, &'static str, &'static str, u64); 4] {
        [
            ("dm1_left", "DM1", "left", "attractive", 11),
            ("dm2_left", "DM2", "left", "core", 12),
            ("dm1_right", "DM1", "right", "attractive", 21),
            ("dm2_right", "DM2", "right", "core", 22),
        ]
    }

    #[test]
    fn equal_glomerulus_weighting_and_left_positive_contrast() {
        let pack = test_pack();
        let channels = matched_channels();
        let resolution = test_resolution(&pack, &channels);
        let mut probe_indices = Vec::new();
        let mut position_by_index = BTreeMap::new();
        let mut probes = CnsOlfactoryProbes::new(
            &pack,
            &resolution,
            &mut probe_indices,
            &mut position_by_index,
        )
        .unwrap();
        let readout = probes.update(
            &[10_000, 2_000, 4_000, 2_000],
            100_000.0,
            ORN_SPONTANEOUS_RATE_HZ,
        );
        assert_eq!(probe_indices, vec![0, 1, 2, 3]);
        assert_eq!(readout.band_rate_hz, [[100.0, 20.0], [40.0, 20.0]]);
        assert_eq!(readout.rate_hz, [60.0, 30.0]);
        assert_eq!(readout.contrast, 1.0 / 3.0);
        assert_eq!(readout.spike_delta, 18_000);
    }

    #[test]
    fn rate_filter_corrects_zero_initialization_bias_without_inventing_spikes() {
        let pack = test_pack();
        let resolution = test_resolution(&pack, &matched_channels());
        let mut probes =
            CnsOlfactoryProbes::new(&pack, &resolution, &mut Vec::new(), &mut BTreeMap::new())
                .unwrap();
        for step in 1..=10 {
            let readout = probes.update(&[1; 4], 100.0, 8.0);
            assert!(
                readout
                    .rate_hz
                    .iter()
                    .all(|rate| (rate - 10.0).abs() < 1e-12)
            );
            assert_eq!(readout.spike_delta, 4);
            assert!((readout.observed_seconds - step as f64 * 0.1).abs() < 1e-12);
        }
    }

    fn shared_gain_rates(concentration_ppm: f64, gain_hz: f64) -> [f64; 2] {
        let concentration_power = concentration_ppm.powf(FOOD_ODOR_HILL_EXPONENT);
        [
            ORN_SPONTANEOUS_RATE_HZ + gain_hz * concentration_power / (concentration_power + 1.0),
            ORN_SPONTANEOUS_RATE_HZ
                + gain_hz * concentration_power
                    / (concentration_power + DM2_HALF_MAX_PPM.powf(FOOD_ODOR_HILL_EXPONENT)),
        ]
    }

    #[test]
    fn inverse_hill_recovers_concentration_with_shared_gain() {
        for &(concentration_ppm, gain_hz) in &[(0.5, 70.0), (3.0, 120.0), (40.0, 50.0)] {
            let estimate = estimate_concentration_ppm(
                shared_gain_rates(concentration_ppm, gain_hz),
                ORN_SPONTANEOUS_RATE_HZ,
            );
            assert!(
                (estimate - concentration_ppm).abs() < 1e-9,
                "estimate={estimate} expected={concentration_ppm}"
            );
        }
    }

    #[test]
    fn concentration_estimate_is_zero_without_meaningful_dm1_evocation() {
        assert_eq!(estimate_concentration_ppm([9.0, 100.0], 8.0), 0.0);
        let shifted = shared_gain_rates(3.0, 120.0).map(|rate| rate + 4.0);
        assert!((estimate_concentration_ppm(shifted, 12.0) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn missing_matched_side_fails_closed() {
        let pack = test_pack();
        let channels = [
            ("dm1_left", "DM1", "left", "attractive", 11),
            ("dm2_left", "DM2", "left", "core", 12),
            ("dm1_right", "DM1", "right", "attractive", 21),
        ];
        let resolution = test_resolution(&pack, &channels);
        let error =
            CnsOlfactoryProbes::new(&pack, &resolution, &mut Vec::new(), &mut BTreeMap::new())
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("matched DM2 core channel on right")
        );
    }

    #[test]
    fn missing_pack_root_fails_closed() {
        let pack = ConnectomePack::from_arrays(
            [11_u64, 12, 21],
            [0_u32; 4],
            Vec::<u32>::new(),
            Vec::<i16>::new(),
        )
        .unwrap();
        let channels = [
            ("dm1_left", "DM1", "left", "attractive", 11),
            ("dm2_left", "DM2", "left", "core", 12),
            ("dm1_right", "DM1", "right", "attractive", 21),
            ("dm2_right", "DM2", "right", "core", 22),
        ];
        let resolution = test_resolution(&pack, &channels);
        let error =
            CnsOlfactoryProbes::new(&pack, &resolution, &mut Vec::new(), &mut BTreeMap::new())
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("missing from the connectome pack")
        );
    }
}
