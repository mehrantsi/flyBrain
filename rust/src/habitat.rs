use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

const HABITAT_FILE: &str = "habitat.json";
const MOVABLE_SUGAR_ID: &str = "sugar_drop";
const ODOR_SOURCE_CORE_MM: f64 = 2.0;
const ODOR_REFERENCE_AIR_SPEED_MM_S: f64 = 35.0;
const ODOR_DIFFUSION_LENGTH_FRACTION: f64 = 0.18;

#[derive(Clone, Debug, Deserialize)]
pub struct Habitat {
    schema: String,
    units: HabitatUnits,
    room: RoomSpec,
    airflow_mm_s: [f64; 3],
    resources: Vec<Resource>,
    sensory_model: SensoryModel,
}

#[derive(Clone, Debug, Deserialize)]
struct HabitatUnits {
    length: String,
    time: String,
    mass: String,
    odor_concentration: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RoomSpec {
    pub half_extents_mm: [f64; 3],
    pub open_ceiling: bool,
    pub front_doorway_width_mm: f64,
    #[serde(default = "default_flight_altitude_bounds_mm")]
    pub flight_altitude_bounds_mm: [f64; 2],
}

fn default_flight_altitude_bounds_mm() -> [f64; 2] {
    [5.0, 208.0]
}

#[derive(Clone, Debug, Deserialize)]
pub struct Resource {
    pub id: String,
    pub kind: String,
    pub geom: String,
    pub position: [f64; 3],
    pub movable: bool,
    pub taste_radius_mm: f64,
    pub odor_source_ppm: f64,
    pub odor_length_mm: f64,
    pub taste_valence: f64,
    pub nutrition: f64,
    pub hydration: f64,
    #[serde(default)]
    pub taste_capsules: Vec<TasteCapsule>,
    #[serde(default)]
    pub taste_margin_mm: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TasteCapsule {
    pub from_mm: [f64; 3],
    pub to_mm: [f64; 3],
    pub radius_mm: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct SensoryModel {
    odor: String,
    taste: String,
    vision: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HabitatSample {
    pub odor_left_ppm: f64,
    pub odor_right_ppm: f64,
    pub tasted_resource: Option<usize>,
    pub taste_valence: f64,
    pub nearest_resource: Option<usize>,
    pub nearest_distance_mm: f64,
}

impl Habitat {
    pub fn load(assets_dir: impl AsRef<Path>) -> Result<Self> {
        let path = assets_dir.as_ref().join(HABITAT_FILE);
        let bytes = fs::read(&path)
            .with_context(|| format!("reading habitat specification {}", path.display()))?;
        let habitat: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing habitat specification {}", path.display()))?;
        habitat.validate()?;
        Ok(habitat)
    }

    pub fn room(&self) -> &RoomSpec {
        &self.room
    }

    pub fn airflow_mm_s(&self) -> [f64; 3] {
        self.airflow_mm_s
    }

    pub fn resources(&self) -> &[Resource] {
        &self.resources
    }

    pub fn sample(
        &self,
        left_antenna: [f64; 3],
        right_antenna: [f64; 3],
        mouth: [f64; 3],
        movable_sugar_position: [f64; 3],
        movable_sugar_enabled: bool,
    ) -> HabitatSample {
        let odor_left_ppm =
            self.odor_at(left_antenna, movable_sugar_position, movable_sugar_enabled);
        let odor_right_ppm =
            self.odor_at(right_antenna, movable_sugar_position, movable_sugar_enabled);
        let mut tasted_resource = None;
        let mut taste_valence = 0.0_f64;
        let mut nearest_resource = None;
        let mut nearest_distance_mm = f64::INFINITY;
        for (index, resource) in self.resources.iter().enumerate() {
            let enabled = !resource.movable || movable_sugar_enabled;
            if !enabled {
                continue;
            }
            let position = self.resource_position(resource, movable_sugar_position);
            let center_distance = distance(mouth, position);
            if center_distance < nearest_distance_mm {
                nearest_distance_mm = center_distance;
                nearest_resource = Some(index);
            }
            if resource.taste_distance_mm(mouth, position) <= resource.taste_limit_mm()
                && resource.taste_valence > taste_valence
            {
                tasted_resource = Some(index);
                taste_valence = resource.taste_valence;
            }
        }
        HabitatSample {
            odor_left_ppm,
            odor_right_ppm,
            tasted_resource,
            taste_valence,
            nearest_resource,
            nearest_distance_mm,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.schema != "flybrain-habitat-v2"
            || self.units.length != "millimeter"
            || self.units.time != "second"
            || self.units.mass != "gram"
            || self.units.odor_concentration != "isobutylene-equivalent ppm"
        {
            bail!("unsupported habitat schema or unit system")
        }
        if self
            .room
            .half_extents_mm
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
            || !self.room.front_doorway_width_mm.is_finite()
            || self.room.front_doorway_width_mm < 0.0
            || self.room.front_doorway_width_mm >= self.room.half_extents_mm[0] * 2.0
            || self
                .room
                .flight_altitude_bounds_mm
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || self.room.flight_altitude_bounds_mm[0] >= self.room.flight_altitude_bounds_mm[1]
            || self.room.flight_altitude_bounds_mm[1] > self.room.half_extents_mm[2] * 2.0
            || self.airflow_mm_s.iter().any(|value| !value.is_finite())
        {
            bail!("habitat room or airflow values are invalid")
        }
        if self.resources.len() < 2 {
            bail!("habitat must contain multiple resources")
        }
        let mut ids = BTreeSet::new();
        let mut geoms = BTreeSet::new();
        let mut movable_count = 0;
        for resource in &self.resources {
            if resource.id.is_empty()
                || resource.kind.is_empty()
                || resource.geom.is_empty()
                || resource.id.contains('\0')
                || resource.geom.contains('\0')
                || !ids.insert(resource.id.as_str())
                || !geoms.insert(resource.geom.as_str())
                || resource.position.iter().any(|value| !value.is_finite())
                || !resource.taste_radius_mm.is_finite()
                || resource.taste_radius_mm <= 0.0
                || !resource.odor_source_ppm.is_finite()
                || resource.odor_source_ppm < 0.0
                || !resource.odor_length_mm.is_finite()
                || resource.odor_length_mm <= 0.0
                || !(-1.0..=1.0).contains(&resource.taste_valence)
                || !(0.0..=1.0).contains(&resource.nutrition)
                || !(0.0..=1.0).contains(&resource.hydration)
                || resource.taste_capsules.iter().any(|capsule| {
                    capsule.from_mm.iter().any(|value| !value.is_finite())
                        || capsule.to_mm.iter().any(|value| !value.is_finite())
                        || !capsule.radius_mm.is_finite()
                        || capsule.radius_mm <= 0.0
                        || !norm_squared(subtract(capsule.to_mm, capsule.from_mm)).is_finite()
                })
                || resource
                    .taste_margin_mm
                    .is_some_and(|margin| !margin.is_finite() || margin <= 0.0)
                || (!resource.taste_capsules.is_empty() && resource.taste_margin_mm.is_none())
            {
                bail!("habitat resource {} has invalid values", resource.id)
            }
            if resource.movable {
                movable_count += 1;
                if resource.id != MOVABLE_SUGAR_ID {
                    bail!("only the interactive sugar source may be movable")
                }
            }
        }
        if movable_count != 1
            || self.sensory_model.odor.is_empty()
            || self.sensory_model.taste.is_empty()
            || self.sensory_model.vision.is_empty()
        {
            bail!("habitat sensory contract is incomplete")
        }
        Ok(())
    }

    fn odor_at(
        &self,
        sample_position: [f64; 3],
        movable_sugar_position: [f64; 3],
        movable_sugar_enabled: bool,
    ) -> f64 {
        self.resources
            .iter()
            .filter(|resource| !resource.movable || movable_sugar_enabled)
            .map(|resource| {
                let source = self.resource_position(resource, movable_sugar_position);
                let delta = subtract(sample_position, source);
                resource.odor_source_ppm
                    * odor_transport_gain(delta, self.airflow_mm_s, resource.odor_length_mm)
            })
            .sum()
    }

    fn resource_position(&self, resource: &Resource, movable_sugar_position: [f64; 3]) -> [f64; 3] {
        if resource.movable {
            movable_sugar_position
        } else {
            resource.position
        }
    }
}

impl Resource {
    fn taste_limit_mm(&self) -> f64 {
        if self.taste_capsules.is_empty() {
            return self.taste_radius_mm;
        }
        self.taste_margin_mm
            .expect("validated capsule taste margin")
    }

    fn taste_distance_mm(&self, mouth: [f64; 3], position: [f64; 3]) -> f64 {
        if self.taste_capsules.is_empty() {
            return distance(mouth, position);
        }
        self.taste_capsules
            .iter()
            .map(|capsule| {
                (point_to_segment_distance(mouth, capsule.from_mm, capsule.to_mm)
                    - capsule.radius_mm)
                    .max(0.0)
            })
            .fold(f64::INFINITY, f64::min)
    }
}

fn odor_transport_gain(delta: [f64; 3], airflow: [f64; 3], length_mm: f64) -> f64 {
    let diffusion_mm2_s =
        ODOR_REFERENCE_AIR_SPEED_MM_S * ODOR_DIFFUSION_LENGTH_FRACTION * length_mm;
    let reference_advection_per_mm = ODOR_REFERENCE_AIR_SPEED_MM_S / (2.0 * diffusion_mm2_s);
    let decay_per_mm2 = (2.0 * reference_advection_per_mm + 1.0 / length_mm) / length_mm;
    let attenuation_per_mm =
        (norm_squared(airflow) / (4.0 * diffusion_mm2_s.powi(2)) + decay_per_mm2).sqrt();
    let radius_mm = (norm_squared(delta) + ODOR_SOURCE_CORE_MM.powi(2)).sqrt();
    let transport = dot(airflow, delta) / (2.0 * diffusion_mm2_s)
        - attenuation_per_mm * (radius_mm - ODOR_SOURCE_CORE_MM);
    ODOR_SOURCE_CORE_MM / radius_mm * transport.exp()
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn norm_squared(vector: [f64; 3]) -> f64 {
    dot(vector, vector)
}

fn norm(vector: [f64; 3]) -> f64 {
    norm_squared(vector).sqrt()
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    norm(subtract(left, right))
}

fn point_to_segment_distance(
    point: [f64; 3],
    segment_start: [f64; 3],
    segment_end: [f64; 3],
) -> f64 {
    let segment = subtract(segment_end, segment_start);
    let segment_length_squared = norm_squared(segment);
    let projection = if segment_length_squared > 0.0 {
        (dot(subtract(point, segment_start), segment) / segment_length_squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest = std::array::from_fn(|axis| segment_start[axis] + projection * segment[axis]);
    distance(point, closest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn habitat() -> Habitat {
        let habitat = Habitat {
            schema: "flybrain-habitat-v2".to_owned(),
            units: HabitatUnits {
                length: "millimeter".to_owned(),
                time: "second".to_owned(),
                mass: "gram".to_owned(),
                odor_concentration: "isobutylene-equivalent ppm".to_owned(),
            },
            room: RoomSpec {
                half_extents_mm: [300.0, 220.0, 110.0],
                open_ceiling: false,
                front_doorway_width_mm: 0.0,
                flight_altitude_bounds_mm: [5.0, 208.0],
            },
            airflow_mm_s: [35.0, 8.0, 0.0],
            resources: vec![
                Resource {
                    id: "sugar_drop".to_owned(),
                    kind: "sugar".to_owned(),
                    geom: "food_patch".to_owned(),
                    position: [1.1, 0.0, 0.25],
                    movable: true,
                    taste_radius_mm: 0.75,
                    odor_source_ppm: 0.0,
                    odor_length_mm: 40.0,
                    taste_valence: 1.0,
                    nutrition: 0.8,
                    hydration: 0.1,
                    taste_capsules: Vec::new(),
                    taste_margin_mm: None,
                },
                Resource {
                    id: "banana".to_owned(),
                    kind: "fruit".to_owned(),
                    geom: "banana_food".to_owned(),
                    position: [32.0, 18.0, 1.6],
                    movable: false,
                    taste_radius_mm: 1.2,
                    odor_source_ppm: 12.0,
                    odor_length_mm: 80.0,
                    taste_valence: 0.9,
                    nutrition: 0.9,
                    hydration: 0.4,
                    taste_capsules: Vec::new(),
                    taste_margin_mm: None,
                },
                Resource {
                    id: "water".to_owned(),
                    kind: "water".to_owned(),
                    geom: "water_dish".to_owned(),
                    position: [-40.0, -25.0, 0.6],
                    movable: false,
                    taste_radius_mm: 1.0,
                    odor_source_ppm: 0.0,
                    odor_length_mm: 20.0,
                    taste_valence: 0.5,
                    nutrition: 0.0,
                    hydration: 1.0,
                    taste_capsules: Vec::new(),
                    taste_margin_mm: None,
                },
            ],
            sensory_model: SensoryModel {
                odor: "bilateral_antenna_finite_core_advection_diffusion".to_owned(),
                taste: "mouth_distance_threshold".to_owned(),
                vision: "flygym_721_ommatidia_per_eye".to_owned(),
            },
        };
        habitat.validate().expect("valid inline habitat");
        habitat
    }

    #[test]
    fn validates_multiple_pinned_resources() {
        let habitat = habitat();
        assert_eq!(habitat.resources().len(), 3);
        assert_eq!(habitat.room().half_extents_mm, [300.0, 220.0, 110.0]);
        assert_eq!(habitat.room().flight_altitude_bounds_mm, [5.0, 208.0]);
        assert!(!habitat.room().open_ceiling);
        assert_eq!(habitat.room().front_doorway_width_mm, 0.0);
    }

    #[test]
    fn pinned_room_is_closed() {
        let habitat = Habitat::load(crate::world::DEFAULT_ASSETS_DIR).unwrap();
        assert!(!habitat.room().open_ceiling);
        assert_eq!(habitat.room().front_doorway_width_mm, 0.0);
    }

    #[test]
    fn bilateral_plume_is_local_and_directional() {
        let habitat = habitat();
        let source = [32.0, 18.0, 1.6];
        let wind_speed = norm(habitat.airflow_mm_s);
        let wind = habitat.airflow_mm_s.map(|value| value / wind_speed);
        let crosswind = [-wind[1], wind[0], 0.0];
        let centerline: [f64; 3] = std::array::from_fn(|axis| source[axis] + 25.0 * wind[axis]);
        let left: [f64; 3] = std::array::from_fn(|axis| centerline[axis] + 0.8 * crosswind[axis]);
        let right: [f64; 3] = std::array::from_fn(|axis| centerline[axis] - 5.0 * crosswind[axis]);
        let downwind = habitat.sample(left, right, [200.0; 3], [1.1, 0.0, 0.25], false);
        assert!(downwind.odor_left_ppm > downwind.odor_right_ppm);
        assert!(downwind.odor_left_ppm > 0.0);
    }

    #[test]
    fn odor_transport_is_continuous_across_source_plane_and_zero_wind() {
        for height in [0.0, 6.4, 20.0] {
            let before = odor_transport_gain([-1e-5, 0.0, height], [35.0, 0.0, 0.0], 95.0);
            let after = odor_transport_gain([1e-5, 0.0, height], [35.0, 0.0, 0.0], 95.0);
            assert!((before - after).abs() < 1e-6);
        }
        for airflow in [[0.0; 3], [35.0, 8.0, 0.0], [-35.0, 0.0, 0.0]] {
            assert_eq!(odor_transport_gain([0.0; 3], airflow, 95.0), 1.0);
        }
        let still = odor_transport_gain([20.0, 0.0, 0.0], [0.0; 3], 95.0);
        let transverse = odor_transport_gain([0.0, 20.0, 0.0], [0.0; 3], 95.0);
        assert_eq!(still, transverse);
        assert!(
            (still - odor_transport_gain([20.0, 0.0, 0.0], [1e-7, 0.0, 0.0], 95.0)).abs() < 1e-9
        );
    }

    #[test]
    fn plume_spreading_dilutes_concentration_without_a_distant_false_peak() {
        let wind = [35.0, 0.0, 0.0];
        let peak_x =
            (-100..=1000)
                .map(|step| step as f64 * 0.1)
                .max_by(|left, right| {
                    odor_transport_gain([*left, 0.0, 6.4], wind, 95.0)
                        .total_cmp(&odor_transport_gain([*right, 0.0, 6.4], wind, 95.0))
                })
                .unwrap();
        assert!((0.0..5.0).contains(&peak_x), "peak_x={peak_x}");
        let near = odor_transport_gain([100.0, 0.0, 0.0], wind, 95.0);
        let far = odor_transport_gain([200.0, 0.0, 0.0], wind, 95.0);
        let dilution_ratio = far / near * (100.0_f64 / 95.0).exp();
        assert!((dilution_ratio - 0.5).abs() < 0.01);
        assert!(near > odor_transport_gain([-100.0, 0.0, 0.0], wind, 95.0));
    }

    #[test]
    fn taste_selects_the_contacted_resource() {
        let habitat = habitat();
        let sample = habitat.sample(
            [0.0; 3],
            [0.0; 3],
            [32.0, 18.0, 1.6],
            [1.1, 0.0, 0.25],
            true,
        );
        let resource = &habitat.resources()[sample.tasted_resource.expect("taste")];
        assert_eq!(resource.id, "banana");
        assert_eq!(sample.taste_valence, 0.9);
    }

    #[test]
    fn banana_taste_follows_all_capsule_segments() {
        let habitat = Habitat::load(crate::world::DEFAULT_ASSETS_DIR).unwrap();
        let banana_index = habitat
            .resources()
            .iter()
            .position(|resource| resource.id == "banana")
            .unwrap();
        for mouth in [
            [25.0, 15.0, 1.4],
            [29.0, 17.0, 1.8],
            [34.0, 18.5, 2.0],
            [39.0, 18.0, 1.8],
            [43.0, 15.5, 1.4],
        ] {
            let sample = habitat.sample([0.0; 3], [0.0; 3], mouth, [1.1, 0.0, 0.25], false);
            assert_eq!(
                sample.tasted_resource,
                Some(banana_index),
                "mouth={mouth:?}"
            );
        }
        let near_surface = habitat.sample(
            [0.0; 3],
            [0.0; 3],
            [43.0, 15.5, 3.9],
            [1.1, 0.0, 0.25],
            false,
        );
        assert_eq!(near_surface.tasted_resource, Some(banana_index));
        let outside_surface = habitat.sample(
            [0.0; 3],
            [0.0; 3],
            [43.0, 15.5, 5.3],
            [1.1, 0.0, 0.25],
            false,
        );
        assert_ne!(outside_surface.tasted_resource, Some(banana_index));
    }
}
