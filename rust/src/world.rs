use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use mujoco_rs::prelude::{MjData, MjModel, MjtObj};
use serde::Deserialize;

use crate::embodiment::{JOINTS_PER_LEG, LEG_COUNT, SensorySample, SixLegVncCommand};

pub const DEFAULT_ASSETS_DIR: &str = "assets/neuromechfly";
pub const DEFAULT_MODEL_PATH: &str = "assets/neuromechfly/fly.xml";
pub const DEFAULT_MANIFEST_PATH: &str = "assets/neuromechfly/manifest.json";
pub const JOINT_ACTUATOR_COUNT: usize = 42;
pub const ADHESION_ACTUATOR_COUNT: usize = 6;
pub const FEEDING_ACTUATOR_COUNT: usize = 2;
pub const FEEDING_ACTUATOR_START: usize = JOINT_ACTUATOR_COUNT + ADHESION_ACTUATOR_COUNT;
pub const WING_ACTUATOR_COUNT: usize = 6;
pub const WING_ACTUATOR_START: usize = FEEDING_ACTUATOR_START + FEEDING_ACTUATOR_COUNT;
pub const ACTUATOR_COUNT: usize = WING_ACTUATOR_START + WING_ACTUATOR_COUNT;
pub const GROUND_CONTACT_SENSOR_COUNT: usize = LEG_COUNT;

const ROOT_BODY_NAME: &str = "fly/c_thorax";
const FEEDING_ACTUATOR_NAME: &str = "fly/c_head-c_rostrum-pitch-feeding-position";
const FEEDING_HAUSTELLUM_ACTUATOR_NAME: &str = "fly/c_rostrum-c_haustellum-pitch-feeding-position";
const FEEDING_JOINT_NAME: &str = "fly/c_head-c_rostrum-pitch";
const FEEDING_HAUSTELLUM_JOINT_NAME: &str = "fly/c_rostrum-c_haustellum-pitch";
const FEEDING_ACTUATOR_NAMES: [&str; FEEDING_ACTUATOR_COUNT] =
    [FEEDING_ACTUATOR_NAME, FEEDING_HAUSTELLUM_ACTUATOR_NAME];
const FEEDING_JOINT_NAMES: [&str; FEEDING_ACTUATOR_COUNT] =
    [FEEDING_JOINT_NAME, FEEDING_HAUSTELLUM_JOINT_NAME];
pub const WING_ACTUATOR_NAMES: [&str; WING_ACTUATOR_COUNT] = [
    "fly/c_thorax-l_wing-yaw-flight-position",
    "fly/c_thorax-l_wing-pitch-flight-position",
    "fly/c_thorax-l_wing-roll-flight-position",
    "fly/c_thorax-r_wing-yaw-flight-position",
    "fly/c_thorax-r_wing-pitch-flight-position",
    "fly/c_thorax-r_wing-roll-flight-position",
];
pub const WING_JOINT_NAMES: [&str; WING_ACTUATOR_COUNT] = [
    "fly/c_thorax-l_wing-yaw",
    "fly/c_thorax-l_wing-pitch",
    "fly/c_thorax-l_wing-roll",
    "fly/c_thorax-r_wing-yaw",
    "fly/c_thorax-r_wing-pitch",
    "fly/c_thorax-r_wing-roll",
];
const VNC_ACTUATOR_INDICES: [[usize; JOINTS_PER_LEG]; LEG_COUNT] = [
    [0, 1, 5],
    [7, 8, 12],
    [14, 15, 19],
    [21, 22, 26],
    [28, 29, 33],
    [35, 36, 40],
];
const VNC_ACTUATOR_NAMES: [[&str; JOINTS_PER_LEG]; LEG_COUNT] = [
    [
        "fly/c_thorax-lf_coxa-yaw-position",
        "fly/c_thorax-lf_coxa-pitch-position",
        "fly/lf_trochanterfemur-lf_tibia-pitch-position",
    ],
    [
        "fly/c_thorax-lm_coxa-yaw-position",
        "fly/c_thorax-lm_coxa-pitch-position",
        "fly/lm_trochanterfemur-lm_tibia-pitch-position",
    ],
    [
        "fly/c_thorax-lh_coxa-yaw-position",
        "fly/c_thorax-lh_coxa-pitch-position",
        "fly/lh_trochanterfemur-lh_tibia-pitch-position",
    ],
    [
        "fly/c_thorax-rf_coxa-yaw-position",
        "fly/c_thorax-rf_coxa-pitch-position",
        "fly/rf_trochanterfemur-rf_tibia-pitch-position",
    ],
    [
        "fly/c_thorax-rm_coxa-yaw-position",
        "fly/c_thorax-rm_coxa-pitch-position",
        "fly/rm_trochanterfemur-rm_tibia-pitch-position",
    ],
    [
        "fly/c_thorax-rh_coxa-yaw-position",
        "fly/c_thorax-rh_coxa-pitch-position",
        "fly/rh_trochanterfemur-rh_tibia-pitch-position",
    ],
];
const VNC_JOINT_NAMES: [[&str; JOINTS_PER_LEG]; LEG_COUNT] = [
    [
        "fly/c_thorax-lf_coxa-yaw",
        "fly/c_thorax-lf_coxa-pitch",
        "fly/lf_trochanterfemur-lf_tibia-pitch",
    ],
    [
        "fly/c_thorax-lm_coxa-yaw",
        "fly/c_thorax-lm_coxa-pitch",
        "fly/lm_trochanterfemur-lm_tibia-pitch",
    ],
    [
        "fly/c_thorax-lh_coxa-yaw",
        "fly/c_thorax-lh_coxa-pitch",
        "fly/lh_trochanterfemur-lh_tibia-pitch",
    ],
    [
        "fly/c_thorax-rf_coxa-yaw",
        "fly/c_thorax-rf_coxa-pitch",
        "fly/rf_trochanterfemur-rf_tibia-pitch",
    ],
    [
        "fly/c_thorax-rm_coxa-yaw",
        "fly/c_thorax-rm_coxa-pitch",
        "fly/rm_trochanterfemur-rm_tibia-pitch",
    ],
    [
        "fly/c_thorax-rh_coxa-yaw",
        "fly/c_thorax-rh_coxa-pitch",
        "fly/rh_trochanterfemur-rh_tibia-pitch",
    ],
];
const ADHESION_ACTUATOR_NAMES: [&str; ADHESION_ACTUATOR_COUNT] = [
    "fly/lf_tarsus5-adhesion",
    "fly/lm_tarsus5-adhesion",
    "fly/lh_tarsus5-adhesion",
    "fly/rf_tarsus5-adhesion",
    "fly/rm_tarsus5-adhesion",
    "fly/rh_tarsus5-adhesion",
];
const CONTACT_SENSOR_NAMES: [&str; GROUND_CONTACT_SENSOR_COUNT] = [
    "ground_contact_lf_leg",
    "ground_contact_lm_leg",
    "ground_contact_lh_leg",
    "ground_contact_rf_leg",
    "ground_contact_rm_leg",
    "ground_contact_rh_leg",
];
const ADHESIVE_FOOT_GEOM_NAMES: [&str; LEG_COUNT] = [
    "fly/lf_tarsus5",
    "fly/lm_tarsus5",
    "fly/lh_tarsus5",
    "fly/rf_tarsus5",
    "fly/rm_tarsus5",
    "fly/rh_tarsus5",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldCounts {
    pub qpos: usize,
    pub dofs: usize,
    pub bodies: usize,
    pub joints: usize,
    pub actuators: usize,
    pub sensors: usize,
    pub cameras: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorldMetadata {
    pub schema: String,
    pub model: String,
    pub physics: String,
    pub timestep_seconds: f64,
    pub counts: WorldCounts,
    pub environment: WorldEnvironment,
    pub brain_body_interface: BrainBodyInterface,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorldEnvironment {
    pub food_center: [f64; 3],
    pub taste_radius: f64,
    pub taste_source_body: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrainBodyInterface {
    pub feeding_actuator: String,
    pub feeding_joint: String,
    pub feeding_actuators: Vec<String>,
    pub feeding_joints: Vec<String>,
    pub control_range: [f64; 2],
    pub full_extension_control: f64,
    pub neural_readout: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActuatorMetadata {
    pub index: usize,
    pub name: String,
    pub joint_name: String,
    pub control_range: [f64; 2],
    pub neutral_control: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SensorMetadata {
    pub index: usize,
    pub name: String,
    pub address: usize,
    pub dimension: usize,
    pub sensor_type: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootPose {
    pub position: [f64; 3],
    pub quaternion: [f64; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObstacleSample {
    pub forward_clearance_mm: f64,
    pub left_clearance_mm: f64,
    pub right_clearance_mm: f64,
    pub up_clearance_mm: f64,
    pub down_clearance_mm: f64,
    pub overhead_geom_id: Option<usize>,
    pub nearest_geom_id: Option<usize>,
    pub environment_contact_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct JointAddress {
    qpos: usize,
    qvel: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct Manifest {
    schema: String,
    model: String,
    physics: String,
    timestep_seconds: f64,
    counts: ManifestCounts,
    neutral_qpos: Vec<f64>,
    neutral_control: Vec<f64>,
    actuators: Vec<ManifestActuator>,
    sensors: Vec<ManifestSensor>,
    cameras: Vec<String>,
    environment: ManifestEnvironment,
    brain_body_interface: ManifestBrainBodyInterface,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
struct ManifestCounts {
    qpos: usize,
    dofs: usize,
    bodies: usize,
    joints: usize,
    actuators: usize,
    sensors: usize,
    cameras: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestActuator {
    index: usize,
    name: String,
    joint_index: usize,
    joint_name: String,
    control_range: [f64; 2],
    neutral_control: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestSensor {
    index: usize,
    name: String,
    address: usize,
    dimension: usize,
    r#type: i32,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestEnvironment {
    food_center: [f64; 3],
    taste_radius: f64,
    taste_source_body: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestBrainBodyInterface {
    feeding_actuator: String,
    feeding_joint: String,
    feeding_actuators: Vec<String>,
    feeding_joints: Vec<String>,
    control_range: [f64; 2],
    full_extension_control: f64,
    neural_readout: String,
}

pub struct MuJoCoWorld {
    data: MjData<Box<MjModel>>,
    metadata: WorldMetadata,
    neutral_qpos: Box<[f64]>,
    neutral_control: [f64; ACTUATOR_COUNT],
    actuators: Box<[ActuatorMetadata]>,
    sensors: Box<[SensorMetadata]>,
    root_body_id: usize,
    vnc_actuator_indices: [[usize; JOINTS_PER_LEG]; LEG_COUNT],
    vnc_joint_addresses: [[JointAddress; JOINTS_PER_LEG]; LEG_COUNT],
    adhesion_actuator_indices: [usize; ADHESION_ACTUATOR_COUNT],
    feeding_actuator_indices: [usize; FEEDING_ACTUATOR_COUNT],
    wing_actuator_indices: [usize; WING_ACTUATOR_COUNT],
    contact_sensor_addresses: [usize; GROUND_CONTACT_SENSOR_COUNT],
}

impl MuJoCoWorld {
    pub fn new() -> Result<Self> {
        Self::load(DEFAULT_MODEL_PATH, DEFAULT_MANIFEST_PATH)
    }

    pub fn from_assets_dir(path: impl AsRef<Path>) -> Result<Self> {
        let dir = path.as_ref();
        Self::load(dir.join("fly.xml"), dir.join("manifest.json"))
    }

    pub fn load(model_path: impl AsRef<Path>, manifest_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        let manifest_path = manifest_path.as_ref();
        let manifest = load_manifest(manifest_path)?;
        let model = MjModel::from_xml(model_path)
            .with_context(|| format!("loading MuJoCo model {}", model_path.display()))?;
        let counts = model_counts(&model);
        validate_manifest(&manifest, &model, counts)?;

        let metadata = WorldMetadata {
            schema: manifest.schema,
            model: manifest.model,
            physics: manifest.physics,
            timestep_seconds: manifest.timestep_seconds,
            counts,
            environment: WorldEnvironment {
                food_center: manifest.environment.food_center,
                taste_radius: manifest.environment.taste_radius,
                taste_source_body: manifest.environment.taste_source_body,
            },
            brain_body_interface: BrainBodyInterface {
                feeding_actuator: manifest.brain_body_interface.feeding_actuator,
                feeding_joint: manifest.brain_body_interface.feeding_joint,
                feeding_actuators: manifest.brain_body_interface.feeding_actuators,
                feeding_joints: manifest.brain_body_interface.feeding_joints,
                control_range: manifest.brain_body_interface.control_range,
                full_extension_control: manifest.brain_body_interface.full_extension_control,
                neural_readout: manifest.brain_body_interface.neural_readout,
            },
        };
        let neutral_qpos = manifest.neutral_qpos.into_boxed_slice();
        let neutral_control: [f64; ACTUATOR_COUNT] = manifest
            .neutral_control
            .try_into()
            .map_err(|_| anyhow::anyhow!("manifest neutral_control has the wrong shape"))?;
        let actuators = manifest
            .actuators
            .into_iter()
            .map(|actuator| ActuatorMetadata {
                index: actuator.index,
                name: actuator.name,
                joint_name: actuator.joint_name,
                control_range: actuator.control_range,
                neutral_control: actuator.neutral_control,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let sensors = manifest
            .sensors
            .into_iter()
            .map(|sensor| SensorMetadata {
                index: sensor.index,
                name: sensor.name,
                address: sensor.address,
                dimension: sensor.dimension,
                sensor_type: sensor.r#type,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let root_body_id = model
            .name_to_id(MjtObj::mjOBJ_BODY, ROOT_BODY_NAME)
            .ok_or_else(|| anyhow::anyhow!("model is missing root body {ROOT_BODY_NAME}"))?;
        let mut vnc_joint_addresses =
            [[JointAddress { qpos: 0, qvel: 0 }; JOINTS_PER_LEG]; LEG_COUNT];
        for leg in 0..LEG_COUNT {
            for joint in 0..JOINTS_PER_LEG {
                let joint_id = model
                    .name_to_id(MjtObj::mjOBJ_JOINT, VNC_JOINT_NAMES[leg][joint])
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "model is missing VNC joint {}",
                            VNC_JOINT_NAMES[leg][joint]
                        )
                    })?;
                let qpos = model.jnt_qposadr()[joint_id];
                let qvel = model.jnt_dofadr()[joint_id];
                if qpos < 0 || qvel < 0 {
                    bail!(
                        "VNC joint {} has an invalid address",
                        VNC_JOINT_NAMES[leg][joint]
                    );
                }
                vnc_joint_addresses[leg][joint] = JointAddress {
                    qpos: qpos as usize,
                    qvel: qvel as usize,
                };
            }
        }
        let vnc_actuator_indices = VNC_ACTUATOR_INDICES;
        let adhesion_actuator_indices = std::array::from_fn(|leg| JOINT_ACTUATOR_COUNT + leg);
        let feeding_actuator_indices =
            std::array::from_fn(|offset| FEEDING_ACTUATOR_START + offset);
        let wing_actuator_indices = std::array::from_fn(|offset| WING_ACTUATOR_START + offset);
        let contact_sensor_addresses = std::array::from_fn(|leg| sensors[leg].address);

        let data = MjData::try_new(Box::new(model)).context("allocating MuJoCo simulation data")?;
        let mut world = Self {
            data,
            metadata,
            neutral_qpos,
            neutral_control,
            actuators,
            sensors,
            root_body_id,
            vnc_actuator_indices,
            vnc_joint_addresses,
            adhesion_actuator_indices,
            feeding_actuator_indices,
            wing_actuator_indices,
            contact_sensor_addresses,
        };
        world.reset()?;
        Ok(world)
    }

    pub fn metadata(&self) -> &WorldMetadata {
        &self.metadata
    }

    pub fn counts(&self) -> WorldCounts {
        self.metadata.counts
    }

    pub fn timestep_seconds(&self) -> f64 {
        self.metadata.timestep_seconds
    }

    pub fn timestep(&self) -> f64 {
        self.timestep_seconds()
    }

    pub fn time(&self) -> f64 {
        self.data.time()
    }

    pub fn model(&self) -> &MjModel {
        self.data.model()
    }

    pub fn data(&self) -> &MjData<Box<MjModel>> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut MjData<Box<MjModel>> {
        &mut self.data
    }

    pub fn neutral_qpos(&self) -> &[f64] {
        &self.neutral_qpos
    }

    pub fn neutral_control(&self) -> &[f64] {
        &self.neutral_control
    }

    pub fn neutral_controls(&self) -> &[f64] {
        self.neutral_control()
    }

    pub fn actuators(&self) -> &[ActuatorMetadata] {
        &self.actuators
    }

    pub fn actuator_names(&self) -> Vec<&str> {
        self.actuators
            .iter()
            .map(|actuator| actuator.name.as_str())
            .collect()
    }

    pub fn sensors(&self) -> &[SensorMetadata] {
        &self.sensors
    }

    pub fn actuator_name(&self, index: usize) -> Result<&str> {
        self.actuators
            .get(index)
            .map(|actuator| actuator.name.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!("actuator index {index} is outside 0..{}", ACTUATOR_COUNT)
            })
    }

    pub fn actuator_neutral_control(&self, index: usize) -> Result<f64> {
        self.actuators
            .get(index)
            .map(|actuator| actuator.neutral_control)
            .ok_or_else(|| {
                anyhow::anyhow!("actuator index {index} is outside 0..{}", ACTUATOR_COUNT)
            })
    }

    pub fn actuator_control_range(&self, index: usize) -> Result<[f64; 2]> {
        self.actuators
            .get(index)
            .map(|actuator| actuator.control_range)
            .ok_or_else(|| {
                anyhow::anyhow!("actuator index {index} is outside 0..{}", ACTUATOR_COUNT)
            })
    }

    pub fn qpos(&self) -> &[f64] {
        self.data.qpos()
    }

    pub fn raw_qpos(&self) -> &[f64] {
        self.qpos()
    }

    pub fn qvel(&self) -> &[f64] {
        self.data.qvel()
    }

    pub fn raw_qvel(&self) -> &[f64] {
        self.qvel()
    }

    pub fn controls(&self) -> &[f64] {
        self.data.ctrl()
    }

    pub fn root_pose(&self) -> RootPose {
        RootPose {
            position: self.data.xpos()[self.root_body_id],
            quaternion: self.data.xquat()[self.root_body_id],
        }
    }

    pub fn root_position(&self) -> [f64; 3] {
        self.root_pose().position
    }

    pub fn root_quaternion(&self) -> [f64; 4] {
        self.root_pose().quaternion
    }

    pub fn root_velocity(&self) -> [f64; 6] {
        self.data
            .object_velocity(MjtObj::mjOBJ_BODY, self.root_body_id, false)
    }

    pub fn obstacle_sample(&mut self, cutoff_mm: f64) -> Result<ObstacleSample> {
        if !cutoff_mm.is_finite() || cutoff_mm <= 0.0 {
            bail!("obstacle ray cutoff must be finite and positive")
        }
        let position = self.root_position();
        let [w, x, y, z] = self.root_quaternion();
        let forward = [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z)];
        let forward_norm = forward[0].hypot(forward[1]).max(1e-9);
        let forward = [forward[0] / forward_norm, forward[1] / forward_norm];
        let left = [-forward[1], forward[0]];
        let ray_origin = [
            position[0] + 2.5 * forward[0],
            position[1] + 2.5 * forward[1],
            position[2],
        ];
        let (forward_geom, forward_clearance_mm) =
            self.environment_ray(ray_origin, [forward[0], forward[1], 0.0], cutoff_mm, false)?;
        let (left_geom, left_clearance_mm) =
            self.environment_ray(ray_origin, [left[0], left[1], 0.0], cutoff_mm, false)?;
        let (right_geom, right_clearance_mm) =
            self.environment_ray(ray_origin, [-left[0], -left[1], 0.0], cutoff_mm, false)?;
        let (overhead_geom_id, up_clearance_mm) =
            self.environment_ray(position, [0.0, 0.0, 1.0], cutoff_mm, false)?;
        let (_, down_clearance_mm) =
            self.environment_ray(position, [0.0, 0.0, -1.0], cutoff_mm, true)?;
        let nearest_geom_id = [
            (forward_geom, forward_clearance_mm),
            (left_geom, left_clearance_mm),
            (right_geom, right_clearance_mm),
            (overhead_geom_id, up_clearance_mm),
        ]
        .into_iter()
        .filter_map(|(geom, distance)| geom.map(|geom| (geom, distance)))
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(geom, _)| geom);
        Ok(ObstacleSample {
            forward_clearance_mm,
            left_clearance_mm,
            right_clearance_mm,
            up_clearance_mm,
            down_clearance_mm,
            overhead_geom_id,
            nearest_geom_id,
            environment_contact_count: self.environment_contact_count(),
        })
    }

    pub fn geom_name(&self, geom_id: Option<usize>) -> &str {
        geom_id
            .and_then(|id| self.model().id_to_name(MjtObj::mjOBJ_GEOM, id))
            .unwrap_or("none")
    }

    fn environment_ray(
        &mut self,
        origin: [f64; 3],
        direction: [f64; 3],
        cutoff_mm: f64,
        include_ground: bool,
    ) -> Result<(Option<usize>, f64)> {
        let mut origin = origin;
        let mut travelled = 0.0;
        for _ in 0..16 {
            let (geom_id, distance) = self.data.ray(
                &origin,
                &direction,
                None,
                true,
                Some(self.root_body_id),
                None,
            );
            let Some(geom_id) = geom_id else {
                return Ok((None, cutoff_mm));
            };
            if !distance.is_finite() || distance < 0.0 {
                return Ok((None, cutoff_mm));
            }
            let total_distance = travelled + distance;
            if total_distance > cutoff_mm {
                return Ok((None, cutoff_mm));
            }
            let name = self
                .model()
                .id_to_name(MjtObj::mjOBJ_GEOM, geom_id)
                .unwrap_or("");
            let collidable_furniture = self.model().geom_conaffinity()[geom_id] != 0;
            let ignored = name.starts_with("fly/")
                || name.starts_with("room_wall_")
                || !(collidable_furniture || include_ground && name == "ground_plane");
            if !ignored {
                return Ok((Some(geom_id), total_distance));
            }
            let advance = distance + 0.25;
            for axis in 0..3 {
                origin[axis] += direction[axis] * advance;
            }
            travelled += advance;
            if travelled >= cutoff_mm {
                return Ok((None, cutoff_mm));
            }
        }
        Ok((None, cutoff_mm))
    }

    fn environment_contact_count(&self) -> usize {
        self.data
            .contact()
            .iter()
            .filter(|contact| {
                let Ok(geom1) = usize::try_from(contact.geom1) else {
                    return false;
                };
                let Ok(geom2) = usize::try_from(contact.geom2) else {
                    return false;
                };
                let name1 = self
                    .model()
                    .id_to_name(MjtObj::mjOBJ_GEOM, geom1)
                    .unwrap_or("");
                let name2 = self
                    .model()
                    .id_to_name(MjtObj::mjOBJ_GEOM, geom2)
                    .unwrap_or("");
                let fly1 = name1.starts_with("fly/");
                let fly2 = name2.starts_with("fly/");
                fly1 != fly2 && name1 != "ground_plane" && name2 != "ground_plane"
            })
            .count()
    }

    pub fn body_position(&self, name: &str) -> Result<[f64; 3]> {
        if name.is_empty() || name.contains('\0') {
            bail!("body name must be non-empty and contain no NUL bytes");
        }
        let body_id = self
            .model()
            .name_to_id(MjtObj::mjOBJ_BODY, name)
            .ok_or_else(|| anyhow::anyhow!("model is missing body {name}"))?;
        self.data
            .xpos()
            .get(body_id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("body {name} index is outside the model"))
    }

    pub fn reset(&mut self) -> Result<()> {
        self.data
            .reset_keyframe(0)
            .context("resetting MuJoCo data to keyframe 0")?;
        self.data.ctrl_mut().copy_from_slice(&self.neutral_control);
        self.data.forward();
        self.validate_state()
    }

    pub fn set_controls(&mut self, controls: &[f64]) -> Result<()> {
        if controls.len() != ACTUATOR_COUNT {
            bail!(
                "controls must have exactly {ACTUATOR_COUNT} values, got {}",
                controls.len()
            );
        }
        validate_control_values(controls, &self.actuators)?;
        self.data.ctrl_mut().copy_from_slice(controls);
        Ok(())
    }

    pub fn set_joint_controls(&mut self, controls: &[f64]) -> Result<()> {
        if controls.len() != JOINT_ACTUATOR_COUNT {
            bail!(
                "joint controls must have exactly {JOINT_ACTUATOR_COUNT} values, got {}",
                controls.len()
            );
        }
        validate_control_values(controls, &self.actuators[..JOINT_ACTUATOR_COUNT])?;
        self.data.ctrl_mut()[..JOINT_ACTUATOR_COUNT].copy_from_slice(controls);
        Ok(())
    }

    pub fn set_adhesion_controls(&mut self, controls: &[f64]) -> Result<()> {
        if controls.len() != ADHESION_ACTUATOR_COUNT {
            bail!(
                "adhesion controls must have exactly {ADHESION_ACTUATOR_COUNT} values, got {}",
                controls.len()
            );
        }
        let end = JOINT_ACTUATOR_COUNT + ADHESION_ACTUATOR_COUNT;
        validate_control_values(controls, &self.actuators[JOINT_ACTUATOR_COUNT..end])?;
        self.data.ctrl_mut()[JOINT_ACTUATOR_COUNT..end].copy_from_slice(controls);
        Ok(())
    }

    pub fn set_feeding_extension(&mut self, extension: f64) -> Result<()> {
        if !extension.is_finite() || !(0.0..=1.0).contains(&extension) {
            bail!("feeding extension must be finite and inside [0, 1]");
        }
        let interface = &self.metadata.brain_body_interface;
        let mut next = [0.0; ACTUATOR_COUNT];
        next.copy_from_slice(self.data.ctrl());
        for feeding_actuator_index in self.feeding_actuator_indices {
            let neutral = self.neutral_control[feeding_actuator_index];
            let control = neutral + extension * (interface.full_extension_control - neutral);
            next[feeding_actuator_index] = control;
        }
        self.set_controls(&next)
    }

    pub fn set_wing_controls(&mut self, controls: [f64; WING_ACTUATOR_COUNT]) -> Result<()> {
        validate_control_values(
            &controls,
            &self.actuators[WING_ACTUATOR_START..WING_ACTUATOR_START + WING_ACTUATOR_COUNT],
        )?;
        let next = self.data.ctrl_mut();
        for (value, index) in controls.into_iter().zip(self.wing_actuator_indices) {
            next[index] = value;
        }
        Ok(())
    }

    pub fn wing_joint_positions(&self) -> [f64; WING_ACTUATOR_COUNT] {
        std::array::from_fn(|axis| {
            let joint = self.model().actuator_trnid()[self.wing_actuator_indices[axis]][0] as usize;
            self.data.qpos()[self.model().jnt_qposadr()[joint] as usize]
        })
    }

    pub fn wing_joint_velocities(&self) -> [f64; WING_ACTUATOR_COUNT] {
        std::array::from_fn(|axis| {
            let joint = self.model().actuator_trnid()[self.wing_actuator_indices[axis]][0] as usize;
            self.data.qvel()[self.model().jnt_dofadr()[joint] as usize]
        })
    }

    pub fn set_leg_adhesion(&mut self, leg_index: usize, control: f64) -> Result<()> {
        if leg_index >= LEG_COUNT {
            bail!("leg index {leg_index} is outside 0..{LEG_COUNT}");
        }
        if !control.is_finite() {
            bail!("adhesion control must be finite");
        }
        let mut next = [0.0; ACTUATOR_COUNT];
        next.copy_from_slice(self.data.ctrl());
        next[self.adhesion_actuator_indices[leg_index]] = control;
        self.set_controls(&next)
    }

    pub fn set_adhesion(&mut self, leg_index: usize, control: f64) -> Result<()> {
        self.set_leg_adhesion(leg_index, control)
    }

    pub fn apply_vnc_command(&mut self, command: &SixLegVncCommand) -> Result<()> {
        if !command.phase_rad.is_finite()
            || !command.forward_gain.is_finite()
            || !command.turn_gain.is_finite()
        {
            bail!("VNC command scalars must be finite");
        }
        if command
            .joint_angles_rad
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            bail!("VNC command joint angles must be finite");
        }
        let mut controls = self.neutral_control;
        for leg in 0..LEG_COUNT {
            for joint in 0..JOINTS_PER_LEG {
                let actuator = self.vnc_actuator_indices[leg][joint];
                let value = self.neutral_control[actuator] + command.joint_angles_rad[leg][joint];
                if !value.is_finite() {
                    bail!("VNC command produces a non-finite control at leg {leg}, joint {joint}");
                }
                controls[actuator] = value;
            }
        }
        self.set_controls(&controls)
    }

    pub fn apply_command(&mut self, command: &SixLegVncCommand) -> Result<()> {
        self.apply_vnc_command(command)
    }

    pub fn step(&mut self) -> Result<()> {
        self.data.step();
        self.validate_state()
    }

    pub fn ground_contact_sensor_readings(&self) -> Result<[f64; GROUND_CONTACT_SENSOR_COUNT]> {
        let sensor_data = self.data.sensordata();
        let mut readings = [0.0; GROUND_CONTACT_SENSOR_COUNT];
        for (leg, address) in self.contact_sensor_addresses.iter().copied().enumerate() {
            let value = *sensor_data.get(address).ok_or_else(|| {
                anyhow::anyhow!("ground contact sensor address {address} is out of range")
            })?;
            if !value.is_finite() {
                bail!("ground contact sensor {leg} is non-finite");
            }
            readings[leg] = value;
        }
        Ok(readings)
    }

    pub fn ground_contact_readings(&self) -> Result<[f64; GROUND_CONTACT_SENSOR_COUNT]> {
        self.ground_contact_sensor_readings()
    }

    pub fn ground_contacts(&self) -> Result<[f64; GROUND_CONTACT_SENSOR_COUNT]> {
        self.ground_contact_sensor_readings()
    }

    pub fn non_ground_foot_contacts(&self) -> [bool; LEG_COUNT] {
        self.non_ground_foot_contacts_filtered(false)
    }

    pub fn wall_foot_contacts(&self) -> [bool; LEG_COUNT] {
        self.non_ground_foot_contacts_filtered(true)
    }

    fn non_ground_foot_contacts_filtered(&self, walls_only: bool) -> [bool; LEG_COUNT] {
        let mut contacts = [false; LEG_COUNT];
        for contact in self.data.contact() {
            let Ok(geom1) = usize::try_from(contact.geom1) else {
                continue;
            };
            let Ok(geom2) = usize::try_from(contact.geom2) else {
                continue;
            };
            let name1 = self
                .model()
                .id_to_name(MjtObj::mjOBJ_GEOM, geom1)
                .unwrap_or("");
            let name2 = self
                .model()
                .id_to_name(MjtObj::mjOBJ_GEOM, geom2)
                .unwrap_or("");
            if name1 == "ground_plane" || name2 == "ground_plane" {
                continue;
            }
            if let Some(leg) = support_leg_from_geom_name(name1)
                && is_support_surface(name2, walls_only)
            {
                contacts[leg] = true;
            }
            if let Some(leg) = support_leg_from_geom_name(name2)
                && is_support_surface(name1, walls_only)
            {
                contacts[leg] = true;
            }
        }
        contacts
    }

    pub fn support_contacts(&self) -> Result<[bool; LEG_COUNT]> {
        let ground = self.ground_contact_sensor_readings()?;
        let non_ground = self.non_ground_foot_contacts();
        Ok(std::array::from_fn(|leg| {
            ground[leg] > 0.0 || non_ground[leg]
        }))
    }

    pub fn sensory_sample(&self) -> Result<SensorySample> {
        let mut sample = SensorySample {
            timestamp_ms: self.time() * 1000.0,
            ..SensorySample::default()
        };
        for leg in 0..LEG_COUNT {
            for joint in 0..JOINTS_PER_LEG {
                let address = self.vnc_joint_addresses[leg][joint];
                sample.joint_angles_rad[leg][joint] = *self
                    .qpos()
                    .get(address.qpos)
                    .ok_or_else(|| anyhow::anyhow!("VNC qpos address is out of range"))?;
                sample.joint_velocities_rad_s[leg][joint] = *self
                    .qvel()
                    .get(address.qvel)
                    .ok_or_else(|| anyhow::anyhow!("VNC qvel address is out of range"))?;
            }
        }
        sample.foot_contacts = self.support_contacts()?;
        sample.validate()?;
        Ok(sample)
    }

    pub fn extract_sensory_sample(&self) -> Result<SensorySample> {
        self.sensory_sample()
    }

    fn validate_state(&self) -> Result<()> {
        if !self.time().is_finite() {
            bail!("MuJoCo simulation time is non-finite");
        }
        if self.qpos().iter().any(|value| !value.is_finite()) {
            bail!("MuJoCo qpos contains a non-finite value");
        }
        if self.qvel().iter().any(|value| !value.is_finite()) {
            bail!("MuJoCo qvel contains a non-finite value");
        }
        self.ground_contact_sensor_readings()?;
        Ok(())
    }
}

fn support_leg_from_geom_name(name: &str) -> Option<usize> {
    ADHESIVE_FOOT_GEOM_NAMES
        .iter()
        .position(|foot_name| name == *foot_name)
}

fn is_support_surface(name: &str, walls_only: bool) -> bool {
    !name.starts_with("fly/") && (!walls_only || name.starts_with("room_wall_"))
}

fn load_manifest(path: &Path) -> Result<Manifest> {
    let bytes =
        fs::read(path).with_context(|| format!("reading world manifest {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing world manifest {}", path.display()))
}

fn model_counts(model: &MjModel) -> WorldCounts {
    WorldCounts {
        qpos: model.nq() as usize,
        dofs: model.nv() as usize,
        bodies: model.nbody() as usize,
        joints: model.njnt() as usize,
        actuators: model.nu() as usize,
        sensors: model.nsensor() as usize,
        cameras: model.ncam() as usize,
    }
}

fn validate_manifest(manifest: &Manifest, model: &MjModel, counts: WorldCounts) -> Result<()> {
    if manifest.schema != "flybrain-world-v2"
        || manifest.model != "NeuroMechFly"
        || manifest.physics != "MuJoCo"
    {
        bail!("unsupported NeuroMechFly manifest schema, model, or physics");
    }
    let expected_counts = WorldCounts {
        qpos: 133,
        dofs: 132,
        bodies: 71,
        joints: 127,
        actuators: ACTUATOR_COUNT,
        sensors: GROUND_CONTACT_SENSOR_COUNT,
        cameras: 4,
    };
    if manifest.counts != ManifestCounts::from(expected_counts) || counts != expected_counts {
        bail!("NeuroMechFly manifest and model counts do not match the bounded world layout");
    }
    if !manifest.timestep_seconds.is_finite() || manifest.timestep_seconds <= 0.0 {
        bail!("manifest timestep_seconds must be finite and positive");
    }
    if !model.opt().timestep.is_finite()
        || (model.opt().timestep - manifest.timestep_seconds).abs() > 1e-12
    {
        bail!("manifest timestep_seconds does not match the model timestep");
    }
    if manifest
        .environment
        .food_center
        .iter()
        .any(|value| !value.is_finite())
        || !manifest.environment.taste_radius.is_finite()
        || manifest.environment.taste_radius <= 0.0
        || manifest.environment.taste_source_body.is_empty()
        || manifest.environment.taste_source_body.contains('\0')
    {
        bail!("manifest environment has invalid food or taste values");
    }
    if model
        .name_to_id(MjtObj::mjOBJ_BODY, &manifest.environment.taste_source_body)
        .is_none()
    {
        bail!(
            "manifest taste source body {} is missing from the model",
            manifest.environment.taste_source_body
        );
    }
    let interface = &manifest.brain_body_interface;
    let feeding_actuators_match = interface.feeding_actuators.len() == FEEDING_ACTUATOR_COUNT
        && interface
            .feeding_actuators
            .iter()
            .zip(FEEDING_ACTUATOR_NAMES)
            .all(|(actual, expected)| actual == expected);
    let feeding_joints_match = interface.feeding_joints.len() == FEEDING_ACTUATOR_COUNT
        && interface
            .feeding_joints
            .iter()
            .zip(FEEDING_JOINT_NAMES)
            .all(|(actual, expected)| actual == expected);
    if interface.feeding_actuator != FEEDING_ACTUATOR_NAME
        || interface.feeding_joint != FEEDING_JOINT_NAME
        || !feeding_actuators_match
        || !feeding_joints_match
        || interface.neural_readout != "contralateral MN9 spikes"
        || interface
            .control_range
            .iter()
            .any(|value| !value.is_finite())
        || interface.control_range[0] > interface.control_range[1]
        || !interface.full_extension_control.is_finite()
        || interface.full_extension_control < interface.control_range[0]
        || interface.full_extension_control > interface.control_range[1]
    {
        bail!("manifest brain-body feeding interface is invalid");
    }
    if manifest.neutral_qpos.len() != counts.qpos
        || manifest.neutral_control.len() != counts.actuators
        || manifest.neutral_qpos.iter().any(|value| !value.is_finite())
        || manifest
            .neutral_control
            .iter()
            .any(|value| !value.is_finite())
    {
        bail!("manifest neutral state has an invalid shape or non-finite value");
    }
    let key_qpos = model.key_qpos();
    if model.nkey() < 1 || key_qpos.len() < counts.qpos {
        bail!("model does not contain keyframe 0 with a full qpos state");
    }
    if manifest
        .neutral_qpos
        .iter()
        .zip(&key_qpos[..counts.qpos])
        .any(|(expected, actual)| (expected - actual).abs() > 1e-9)
    {
        bail!("manifest neutral_qpos does not match model keyframe 0");
    }
    if manifest.actuators.len() != ACTUATOR_COUNT {
        bail!("manifest must contain exactly {ACTUATOR_COUNT} actuators");
    }
    let mut seen_actuators = [false; ACTUATOR_COUNT];
    for actuator in &manifest.actuators {
        if actuator.index >= ACTUATOR_COUNT || seen_actuators[actuator.index] {
            bail!("manifest actuator indices must be unique and bounded");
        }
        seen_actuators[actuator.index] = true;
        let model_name = model
            .id_to_name(MjtObj::mjOBJ_ACTUATOR, actuator.index)
            .ok_or_else(|| anyhow::anyhow!("model is missing actuator {}", actuator.index))?;
        if model_name != actuator.name {
            bail!(
                "manifest actuator {} does not match model name",
                actuator.index
            );
        }
        if !actuator.neutral_control.is_finite()
            || !actuator.control_range[0].is_finite()
            || !actuator.control_range[1].is_finite()
            || actuator.control_range[0] > actuator.control_range[1]
            || actuator.neutral_control < actuator.control_range[0]
            || actuator.neutral_control > actuator.control_range[1]
        {
            bail!(
                "manifest actuator {} has an invalid control range",
                actuator.index
            );
        }
        let model_range = model.actuator_ctrlrange()[actuator.index];
        if model_range
            .iter()
            .zip(actuator.control_range)
            .any(|(actual, expected)| (actual - expected).abs() > 1e-12)
        {
            bail!(
                "manifest actuator {} control range does not match model",
                actuator.index
            );
        }
        if (manifest.neutral_control[actuator.index] - actuator.neutral_control).abs() > 1e-12 {
            bail!("manifest neutral_control disagrees with actuator entry");
        }
        if actuator.index < JOINT_ACTUATOR_COUNT {
            let joint_id = model
                .name_to_id(MjtObj::mjOBJ_JOINT, &actuator.joint_name)
                .ok_or_else(|| {
                    anyhow::anyhow!("model is missing actuator joint {}", actuator.joint_name)
                })?;
            if joint_id != actuator.joint_index
                || model.actuator_trnid()[actuator.index][0] != actuator.joint_index as i32
            {
                bail!(
                    "manifest joint mapping for actuator {} does not match model",
                    actuator.index
                );
            }
        }
    }
    if seen_actuators.iter().any(|seen| !seen) {
        bail!("manifest actuator indices are incomplete");
    }
    for leg in 0..LEG_COUNT {
        for joint in 0..JOINTS_PER_LEG {
            let index = VNC_ACTUATOR_INDICES[leg][joint];
            let entry = manifest
                .actuators
                .iter()
                .find(|actuator| actuator.index == index)
                .ok_or_else(|| anyhow::anyhow!("missing VNC actuator {index}"))?;
            if entry.name != VNC_ACTUATOR_NAMES[leg][joint]
                || entry.joint_name != VNC_JOINT_NAMES[leg][joint]
            {
                bail!("VNC actuator mapping is not explicit in the manifest");
            }
        }
    }
    for (offset, expected_name) in ADHESION_ACTUATOR_NAMES.iter().enumerate() {
        let index = JOINT_ACTUATOR_COUNT + offset;
        let entry = manifest
            .actuators
            .iter()
            .find(|actuator| actuator.index == index)
            .ok_or_else(|| anyhow::anyhow!("missing adhesion actuator {index}"))?;
        if entry.name != *expected_name {
            bail!("adhesion actuator mapping is not explicit in the manifest");
        }
    }
    for (offset, (expected_name, expected_joint)) in FEEDING_ACTUATOR_NAMES
        .iter()
        .zip(FEEDING_JOINT_NAMES)
        .enumerate()
    {
        let feeding_index = FEEDING_ACTUATOR_START + offset;
        let feeding = manifest
            .actuators
            .iter()
            .find(|actuator| actuator.index == feeding_index)
            .ok_or_else(|| anyhow::anyhow!("missing feeding actuator {feeding_index}"))?;
        if feeding.name != *expected_name
            || feeding.joint_name != *expected_joint
            || feeding.control_range != interface.control_range
            || model.actuator_trnid()[feeding_index][0] != feeding.joint_index as i32
        {
            bail!("feeding actuator mapping is not explicit in the manifest");
        }
    }
    for (offset, (expected_name, expected_joint)) in
        WING_ACTUATOR_NAMES.iter().zip(WING_JOINT_NAMES).enumerate()
    {
        let wing_index = WING_ACTUATOR_START + offset;
        let wing = manifest
            .actuators
            .iter()
            .find(|actuator| actuator.index == wing_index)
            .ok_or_else(|| anyhow::anyhow!("missing wing actuator {wing_index}"))?;
        if wing.name != *expected_name
            || wing.joint_name != *expected_joint
            || model.actuator_trnid()[wing_index][0] != wing.joint_index as i32
        {
            bail!("wing actuator mapping is not explicit in the manifest");
        }
    }
    if manifest.sensors.len() != GROUND_CONTACT_SENSOR_COUNT {
        bail!("manifest must contain exactly {GROUND_CONTACT_SENSOR_COUNT} sensors");
    }
    let mut seen_sensors = [false; GROUND_CONTACT_SENSOR_COUNT];
    for sensor in &manifest.sensors {
        if sensor.index >= GROUND_CONTACT_SENSOR_COUNT || seen_sensors[sensor.index] {
            bail!("manifest sensor indices must be unique and bounded");
        }
        seen_sensors[sensor.index] = true;
        let model_name = model
            .id_to_name(MjtObj::mjOBJ_SENSOR, sensor.index)
            .ok_or_else(|| anyhow::anyhow!("model is missing sensor {}", sensor.index))?;
        if model_name != sensor.name
            || sensor.name != CONTACT_SENSOR_NAMES[sensor.index]
            || model.sensor_adr()[sensor.index] < 0
            || model.sensor_adr()[sensor.index] as usize != sensor.address
            || model.sensor_dim()[sensor.index] < 1
            || model.sensor_dim()[sensor.index] as usize != sensor.dimension
            || model.sensor_type()[sensor.index] as i32 != sensor.r#type
        {
            bail!("manifest ground contact sensor mapping does not match model");
        }
        if sensor.dimension != 16 {
            bail!("ground contact sensors must expose the bounded 16-value contact record");
        }
    }
    if seen_sensors.iter().any(|seen| !seen) {
        bail!("manifest sensor indices are incomplete");
    }
    if manifest.cameras.len() != counts.cameras {
        bail!("manifest camera list has the wrong shape");
    }
    for (index, name) in manifest.cameras.iter().enumerate() {
        let model_name = model
            .id_to_name(MjtObj::mjOBJ_CAMERA, index)
            .ok_or_else(|| anyhow::anyhow!("model is missing camera {index}"))?;
        if model_name != name {
            bail!("manifest camera {index} does not match model name");
        }
    }
    Ok(())
}

impl From<WorldCounts> for ManifestCounts {
    fn from(counts: WorldCounts) -> Self {
        Self {
            qpos: counts.qpos,
            dofs: counts.dofs,
            bodies: counts.bodies,
            joints: counts.joints,
            actuators: counts.actuators,
            sensors: counts.sensors,
            cameras: counts.cameras,
        }
    }
}

fn validate_control_values(controls: &[f64], actuators: &[ActuatorMetadata]) -> Result<()> {
    if controls.len() != actuators.len() {
        bail!("control shape does not match actuator metadata");
    }
    for (offset, (&value, actuator)) in controls.iter().zip(actuators).enumerate() {
        if !value.is_finite() {
            bail!("control {offset} must be finite");
        }
        if value < actuator.control_range[0] || value > actuator.control_range[1] {
            bail!(
                "control {} for actuator {} is outside [{}, {}]",
                actuator.index,
                actuator.name,
                actuator.control_range[0],
                actuator.control_range[1]
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obstacle_avoidance::{NavigationObservation, NavigationPolicy};

    fn world() -> MuJoCoWorld {
        MuJoCoWorld::new().expect("actual NeuroMechFly model should load")
    }

    #[derive(Debug, PartialEq)]
    struct ContactMaterial {
        dim: i32,
        friction: [f64; 5],
        solref: [f64; 2],
        solimp: [f64; 5],
        includemargin: f64,
    }

    fn place_geom_center(world: &mut MuJoCoWorld, geom_name: &str, position: [f64; 3]) {
        let geom_id = world
            .model()
            .name_to_id(MjtObj::mjOBJ_GEOM, geom_name)
            .expect("contact geom is present");
        let current = world.data().geom_xpos()[geom_id];
        let root = world.root_position();
        world.data_mut().qpos_mut()[0..3].copy_from_slice(&[
            root[0] + position[0] - current[0],
            root[1] + position[1] - current[1],
            root[2] + position[2] - current[2],
        ]);
        world.data_mut().forward();
    }

    fn contact_material(
        world: &MuJoCoWorld,
        fly_geom_name: &str,
        surface_geom_name: &str,
    ) -> Option<ContactMaterial> {
        let fly_geom = world
            .model()
            .name_to_id(MjtObj::mjOBJ_GEOM, fly_geom_name)? as i32;
        let surface_geom = world
            .model()
            .name_to_id(MjtObj::mjOBJ_GEOM, surface_geom_name)? as i32;
        world.data().contact().iter().find_map(|contact| {
            if !((contact.geom1 == fly_geom && contact.geom2 == surface_geom)
                || (contact.geom1 == surface_geom && contact.geom2 == fly_geom))
            {
                return None;
            }
            Some(ContactMaterial {
                dim: contact.dim,
                friction: contact.friction,
                solref: contact.solref,
                solimp: contact.solimp,
                includemargin: contact.includemargin,
            })
        })
    }

    #[test]
    fn generated_habitat_contacts_match_explicit_ground_contact_parameters() {
        const FLY_GEOM: &str = "fly/c_thorax";
        let mut ground_world = world();
        place_geom_center(&mut ground_world, FLY_GEOM, [0.0, 0.0, 0.0]);
        let ground = contact_material(&ground_world, FLY_GEOM, "ground_plane")
            .expect("the explicit ground contact should be generated");

        for (surface, position) in [
            ("banana_plate", [32.0, 18.0, 0.7]),
            ("resource_banana", [31.5, 17.75, 2.9]),
            ("room_wall_right", [298.0, 0.0, 100.0]),
            ("room_wall_left", [-298.0, 0.0, 100.0]),
            ("room_wall_back", [0.0, 218.0, 100.0]),
            ("room_wall_front_window", [0.0, -218.0, 100.0]),
            ("room_wall_front_left", [-205.0, -218.0, 100.0]),
            ("room_wall_front_right", [205.0, -218.0, 100.0]),
            ("room_wall_ceiling", [0.0, 0.0, 218.0]),
        ] {
            let mut food_world = world();
            let geom_id = food_world
                .model()
                .name_to_id(MjtObj::mjOBJ_GEOM, FLY_GEOM)
                .unwrap();
            let radius = food_world.model().geom_rbound()[geom_id];
            let food = (0..=32)
                .find_map(|step| {
                    let center = [
                        position[0],
                        position[1],
                        position[2] + radius * (1.0 - step as f64 / 16.0),
                    ];
                    place_geom_center(&mut food_world, FLY_GEOM, center);
                    contact_material(&food_world, FLY_GEOM, surface)
                })
                .unwrap_or_else(|| panic!("{surface} contact should be generated"));
            assert_eq!(food, ground, "{surface} contact parameters differ");
        }
    }

    #[test]
    fn loads_actual_model_and_manifest_counts() {
        let world = world();
        assert_eq!(world.counts().qpos, 133);
        assert_eq!(world.counts().dofs, 132);
        assert_eq!(world.counts().bodies, 71);
        assert_eq!(world.counts().joints, 127);
        assert_eq!(world.counts().actuators, ACTUATOR_COUNT);
        assert_eq!(world.counts().sensors, 6);
        assert_eq!(world.counts().cameras, 4);
        assert!((world.timestep_seconds() - 0.0001).abs() < 1e-15);
        assert_eq!(
            world.metadata().environment.taste_source_body,
            "fly/c_haustellum"
        );
        assert_eq!(world.metadata().environment.food_center, [1.1, 0.0, 0.25]);
        assert_eq!(world.metadata().environment.taste_radius, 0.75);
        assert_eq!(world.actuators().len(), ACTUATOR_COUNT);
        assert_eq!(world.actuator_name(42).unwrap(), ADHESION_ACTUATOR_NAMES[0]);
        assert_eq!(world.actuator_name(48).unwrap(), FEEDING_ACTUATOR_NAME);
        assert_eq!(
            world.actuator_name(49).unwrap(),
            FEEDING_HAUSTELLUM_ACTUATOR_NAME
        );
        assert_eq!(world.actuator_name(50).unwrap(), WING_ACTUATOR_NAMES[0]);
        assert_eq!(world.actuator_name(55).unwrap(), WING_ACTUATOR_NAMES[5]);
        assert_eq!(world.actuator_neutral_control(0).unwrap(), -0.296706);
        for name in ["room_wall_front_window", "room_wall_ceiling"] {
            let geom_id = world
                .model()
                .name_to_id(MjtObj::mjOBJ_GEOM, name)
                .expect("closed room surface is present");
            assert_eq!(world.model().geom_conaffinity()[geom_id], 1);
        }
    }

    #[test]
    fn reset_and_neutral_step_remain_finite() {
        let mut world = world();
        assert_eq!(world.time(), 0.0);
        assert_eq!(world.qpos().len(), 133);
        assert_eq!(world.qvel().len(), 132);
        assert!(world.qpos().iter().all(|value| value.is_finite()));
        assert!(world.qvel().iter().all(|value| value.is_finite()));
        world.step().unwrap();
        assert!((world.time() - world.timestep_seconds()).abs() < 1e-15);
        assert!(world.qpos().iter().all(|value| value.is_finite()));
        assert!(world.qvel().iter().all(|value| value.is_finite()));
        assert!(
            world
                .ground_contacts()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        world.reset().unwrap();
        assert_eq!(world.time(), 0.0);
    }

    #[test]
    fn only_tarsal_geometries_count_as_surface_support() {
        assert_eq!(support_leg_from_geom_name("fly/lf_tarsus5"), Some(0));
        assert_eq!(support_leg_from_geom_name("fly/rh_tarsus5"), Some(5));
        assert_eq!(support_leg_from_geom_name("fly/rh_tarsus2"), None);
        assert_eq!(support_leg_from_geom_name("fly/lf_tibia"), None);
        assert_eq!(support_leg_from_geom_name("table_top"), None);
    }

    #[test]
    fn adhesion_holds_a_multi_leg_wall_support_pose() {
        let mut world = world();
        for _ in 0..5_000 {
            world.step().unwrap();
        }
        let data = world.data_mut();
        let [x, y, z] = data.qpos()[..3].try_into().unwrap();
        let [qw, qx, qy, qz] = data.qpos()[3..7].try_into().unwrap();
        let half = std::f64::consts::FRAC_1_SQRT_2;
        data.qpos_mut()[..3].copy_from_slice(&[298.0 - z, y, 100.0 + x]);
        data.qpos_mut()[3..7].copy_from_slice(&[
            half * (qw + qy),
            half * (qx - qz),
            half * (qy - qw),
            half * (qz + qx),
        ]);
        data.qvel_mut().fill(0.0);
        data.forward();
        assert!(world.wall_foot_contacts().into_iter().filter(|&contact| contact).count() >= 4);
        world.set_adhesion_controls(&[1.0; LEG_COUNT]).unwrap();
        for _ in 0..10_000 {
            world.step().unwrap();
        }
        assert!(world.root_position()[2] > 99.0, "{:?}", world.root_position());
        assert!(world.wall_foot_contacts().into_iter().filter(|&contact| contact).count() >= 4);
        assert!(world.data().contact().iter().all(|contact| contact.dist >= -0.01));
    }

    #[test]
    fn obstacle_rays_ignore_the_fly_and_report_ground_clearance() {
        let mut world = world();
        let sample = world.obstacle_sample(180.0).unwrap();
        assert!(sample.forward_clearance_mm.is_finite());
        assert!(sample.left_clearance_mm.is_finite());
        assert!(sample.right_clearance_mm.is_finite());
        assert!(sample.up_clearance_mm.is_finite());
        assert!(sample.down_clearance_mm > 0.0);
        assert!(sample.down_clearance_mm < 5.0);
        assert_eq!(sample.environment_contact_count, 0);
    }

    #[test]
    fn upward_ray_detects_the_table_overhang() {
        let mut world = world();
        world.data.qpos_mut()[0..3].copy_from_slice(&[120.0, 70.0, 28.0]);
        world.data.forward();
        let sample = world.obstacle_sample(180.0).unwrap();
        assert_eq!(world.geom_name(sample.overhead_geom_id), "table_top");
        assert!((sample.up_clearance_mm - 17.0).abs() < 1.0);
        assert!((sample.down_clearance_mm - 28.0).abs() < 1.0);
    }

    #[test]
    fn navigation_leaves_the_real_table_footprint_without_orbiting() {
        let mut world = world();
        let mut policy = NavigationPolicy::default();
        let mut position = [120.0, 70.0, 28.0];
        let mut forward: [f64; 2] = [1.0, 0.0];
        let mut escape_direction = None;
        let mut previous_progress = 0.0;
        let mut escaped = false;
        let mut released = false;

        for _ in 0..240 {
            let yaw = forward[1].atan2(forward[0]);
            world.data.qpos_mut()[0..3].copy_from_slice(&position);
            world.data.qpos_mut()[3..7].copy_from_slice(&[
                (yaw * 0.5).cos(),
                0.0,
                0.0,
                (yaw * 0.5).sin(),
            ]);
            world.data.forward();
            let sample = world.obstacle_sample(180.0).unwrap();
            let overhead = sample.overhead_geom_id.is_some() && sample.up_clearance_mm <= 55.0;
            let decision = policy
                .update(NavigationObservation {
                    position_mm: position,
                    forward_xy: forward,
                    room_half_extents_mm: [300.0, 220.0, 110.0],
                    forward_clearance_mm: sample.forward_clearance_mm,
                    left_clearance_mm: sample.left_clearance_mm,
                    right_clearance_mm: sample.right_clearance_mm,
                    up_clearance_mm: sample.up_clearance_mm,
                    overhead,
                    dt_seconds: 0.02,
                })
                .unwrap();
            if decision.escape_active {
                let chosen = *escape_direction.get_or_insert(decision.direction_xy);
                assert!((decision.direction_xy[0] - chosen[0]).abs() < 1e-12);
                assert!((decision.direction_xy[1] - chosen[1]).abs() < 1e-12);
                let progress = (position[0] - 120.0) * chosen[0] + (position[1] - 70.0) * chosen[1];
                assert!(progress + 1e-9 >= previous_progress);
                previous_progress = progress;
            } else if escape_direction.is_some() {
                released = true;
                break;
            }
            forward = decision.direction_xy;
            position[0] += forward[0] * 1.2;
            position[1] += forward[1] * 1.2;
            escaped |=
                !(48.0..=192.0).contains(&position[0]) || !(18.0..=122.0).contains(&position[1]);
        }

        assert!(escape_direction.is_some());
        assert!(escaped);
        assert!(released);
    }

    #[test]
    fn vnc_offsets_preserve_neutral_controls_and_map_three_channels_per_leg() {
        let mut world = world();
        let mut command = SixLegVncCommand {
            phase_rad: 0.0,
            forward_gain: 0.0,
            turn_gain: 0.0,
            joint_angles_rad: [[0.0; JOINTS_PER_LEG]; LEG_COUNT],
        };
        command.joint_angles_rad[0] = [0.1, -0.2, 0.3];
        world.apply_vnc_command(&command).unwrap();
        for (leg, actuator_indices) in VNC_ACTUATOR_INDICES.iter().enumerate() {
            for (joint, &index) in actuator_indices.iter().enumerate() {
                let expected =
                    world.neutral_control()[index] + command.joint_angles_rad[leg][joint];
                assert_eq!(world.controls()[index], expected);
            }
        }
        for index in 0..ACTUATOR_COUNT {
            if !VNC_ACTUATOR_INDICES
                .iter()
                .flatten()
                .any(|mapped| *mapped == index)
            {
                assert_eq!(world.controls()[index], world.neutral_control()[index]);
            }
        }
        world.set_leg_adhesion(3, 0.5).unwrap();
        assert_eq!(world.controls()[45], 0.5);
        assert_eq!(world.controls()[42], 0.0);
        world.set_feeding_extension(0.75).unwrap();
        assert_eq!(world.controls()[48], -0.75);
        assert_eq!(world.controls()[49], -0.75);
        assert_eq!(world.controls()[45], 0.5);
        world
            .set_wing_controls([0.4, -0.8, 0.6, 0.4, -0.8, -0.6])
            .unwrap();
        assert_eq!(
            &world.controls()[50..56],
            &[0.4, -0.8, 0.6, 0.4, -0.8, -0.6]
        );
        assert_eq!(world.controls()[48], -0.75);
    }
}
