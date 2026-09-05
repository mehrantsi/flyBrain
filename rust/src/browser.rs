use std::cell::RefCell;
use std::path::Path;

use anyhow::{Context, Result, bail};
use mujoco_rs::prelude::*;
use serde_json::{Value, json};

use crate::retina::{FlyGymRetina, RETINA_HEIGHT, RETINA_WIDTH};
use crate::world_sim::{SimulationParameters, SimulationStepper};

const ASSETS: &str = "/data/assets";
const EYE_BYTES: usize = RETINA_WIDTH * RETINA_HEIGHT * 3;

struct BrowserState {
    simulation: SimulationStepper,
    retinas: [FlyGymRetina; 2],
    vision: Vec<u8>,
    display: Vec<u8>,
    poses: Vec<f32>,
    total_spikes: u64,
    metrics: [f64; 4],
}

thread_local! {
    static STATE: RefCell<Option<BrowserState>> = const { RefCell::new(None) };
    static RESPONSE: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn respond(value: Value) {
    RESPONSE.with(|response| *response.borrow_mut() = serde_json::to_vec(&value).unwrap());
}

fn action(f: impl FnOnce(&mut BrowserState) -> Result<()>) -> i32 {
    let result = STATE.with(|state| {
        let mut state = state.borrow_mut();
        f(state.as_mut().context("simulation is not initialized")?)
    });
    match result {
        Ok(()) => 1,
        Err(error) => {
            respond(json!({"error": format!("{error:#}")}));
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_init(brain: i32) -> i32 {
    let result = (|| -> Result<BrowserState> {
        let parameters = if Path::new("/data/parameters.json").exists() {
            SimulationParameters::load("/data/parameters.json")?
        } else {
            SimulationParameters::default()
        };
        let mut simulation = SimulationStepper::new_with_parameters(
            ASSETS,
            (brain != 0).then_some(Path::new("/data/pack")),
            500.0,
            0.5,
            parameters,
        )?;
        simulation.set_brain_telemetry_enabled(brain != 0);
        simulation.place_food_ahead(40.0)?;
        Ok(BrowserState {
            simulation,
            retinas: [FlyGymRetina::load(ASSETS)?, FlyGymRetina::load(ASSETS)?],
            vision: vec![0; 2 * EYE_BYTES],
            display: vec![0; 2 * EYE_BYTES],
            poses: Vec::new(),
            total_spikes: 0,
            metrics: [0.0; 4],
        })
    })();
    match result {
        Ok(state) => {
            STATE.with(|slot| *slot.borrow_mut() = Some(state));
            fb_scene()
        }
        Err(error) => {
            respond(json!({"error": format!("{error:#}")}));
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_step(windows: u32) -> i32 {
    action(|state| {
        if !(1..=25).contains(&windows) {
            bail!("control-window batch must be between 1 and 25")
        }
        for _ in 0..windows {
            state.simulation.step_window()?;
            let snapshot = state.simulation.snapshot();
            state.total_spikes += snapshot.population_spike_delta;
            state.metrics = [
                snapshot.time_seconds,
                snapshot.brain_wall_seconds,
                snapshot.brain_encoding_seconds,
                snapshot.brain_engine_seconds,
            ];
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_metrics_ptr() -> *const f64 {
    STATE.with(|state| {
        state.borrow().as_ref().map_or(std::ptr::null(), |s| s.metrics.as_ptr())
    })
}

fn update_frame(state: &mut BrowserState) {
    let world = state.simulation.world();
    let data = world.data();
    state.poses.clear();
    for (position, quaternion) in data.xpos().iter().zip(data.xquat()) {
        state
            .poses
            .extend(position.iter().chain(quaternion).map(|v| *v as f32));
    }
    for (position, rotation) in data.cam_xpos().iter().zip(data.cam_xmat()) {
        state
            .poses
            .extend(position.iter().chain(rotation).map(|v| *v as f32));
    }
    let s = state.simulation.snapshot();
    let flight_diagnostics = json!({
        "contact_count": s.contact_count,
        "environment_contact_count": s.environment_contact_count,
        "wall_support_leg_count": s.wall_support_leg_count,
        "wall_landing_normal": s.wall_landing_target.map(|target| target.inward_xy),
        "wall_landing_surface_point_mm": s.wall_landing_target.map(|target| target.surface_point_mm),
        "wall_landing_alignment": s.wall_landing_alignment,
        "amplitude_scale": s.flight_amplitude_scale,
        "down_clearance_mm": s.flight_down_clearance_mm,
        "vertical_force_to_weight": s.flight_vertical_force_to_weight,
        "wing_controls": &data.ctrl()[crate::world::WING_ACTUATOR_START..],
        "wing_joint_positions": world.wing_joint_positions(),
        "wing_joint_velocities": world.wing_joint_velocities(),
        "brain_landing_drive": s.brain_landing_drive,
        "flight_power": s.cns_motor.map(|motor| motor.flight_activation),
        "root_velocity": world.root_velocity(),
    });
    respond(json!({
        "time_seconds": s.time_seconds, "root_position": s.root_position,
        "horizontal_speed_mm_s": s.horizontal_speed_mm_s, "body_pitch_deg": s.body_pitch_deg,
        "flight_mode": format!("{:?}", s.flight_mode), "behavior_mode": format!("{:?}", s.behavior_mode),
        "foraging_mode": format!("{:?}", s.foraging_mode),
        "food_center": s.food_center, "food_enabled": s.food_enabled,
        "food_distance": s.food_distance, "flight_allowed": s.flight_allowed,
        "taste_active": s.taste_active, "tasted_resource": state.simulation.resource_label(s.tasted_resource),
        "odor_left_ppm": s.odor_left_ppm, "odor_right_ppm": s.odor_right_ppm,
        "visual_left": s.visual_left, "visual_right": s.visual_right,
        "visual_event_delta": s.visual_event_delta,
        "brain_field_potential_mv": s.brain_field_potential_mv,
        "brain_field_sample_sequence": s.brain_field_sample_sequence,
        "brain_field_dominant_frequency_hz": s.brain_field_dominant_frequency_hz,
        "population_spike_delta": s.population_spike_delta, "total_spikes": state.total_spikes,
        "filtered_population_rate_hz": s.filtered_population_rate_hz,
        "cumulative_spiking_neuron_count": s.cumulative_spiking_neuron_count,
        "brain_flight_drive": s.brain_flight_drive, "brain_altitude_control": s.brain_altitude_control,
        "brain_walking_drive": s.brain_walking_drive, "brain_walking_steering": s.brain_walking_steering,
        "brain_flight_steering": s.brain_flight_steering,
        "filtered_mn9_rate_hz": s.filtered_mn9_rate_hz, "mn9_spike_delta": s.mn9_spike_delta,
        "grooming_active": s.grooming_active,
        "flight_target_height_mm": s.flight_target_height_mm, "feeding_extension": s.feeding_extension,
        "brain_wall_seconds": s.brain_wall_seconds,
        "flight_diagnostics": flight_diagnostics,
        "qpos": world.qpos(), "qvel": world.qvel(),
        "physics_warnings": data.warning().iter().map(|w| w.number).collect::<Vec<_>>(),
    }));
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_frame() -> i32 {
    action(|state| {
        update_frame(state);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_command(command: u32) -> i32 {
    action(|state| {
        match command {
            0 => {
                state.simulation.reset()?;
                state.total_spikes = 0;
            }
            1 => state.simulation.drop_food_below_fly()?,
            2 => state.simulation.request_grooming(),
            3 => state.simulation.toggle_flight(),
            4 => state.simulation.toggle_food()?,
            _ => bail!("unknown simulation command"),
        }
        update_frame(state);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_set_vision() -> i32 {
    action(|state| {
        for eye in 0..2 {
            state.retinas[eye]
                .sample_top_down(&state.vision[eye * EYE_BYTES..(eye + 1) * EYE_BYTES])?;
        }
        state
            .simulation
            .set_retina_summaries(state.retinas.each_ref().map(|eye| eye.summary()))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_update_retina_display() -> i32 {
    action(|state| {
        for eye in 0..2 {
            state.display[eye * EYE_BYTES..(eye + 1) * EYE_BYTES]
                .copy_from_slice(state.retinas[eye].display());
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_scene() -> i32 {
    action(|state| {
        let m = state.simulation.world().model();
        let name = |kind, id| m.id_to_name(kind, id).unwrap_or("").to_string();
        let meshes: Vec<_> = (0..m.nmesh() as usize)
            .map(|i| {
                let v = m.mesh_vertadr()[i] as usize;
                let f = m.mesh_faceadr()[i] as usize;
                let n = m.mesh_normaladr()[i] as usize;
                json!({"vertices": &m.mesh_vert()[v..v + m.mesh_vertnum()[i] as usize],
                "faces": &m.mesh_face()[f..f + m.mesh_facenum()[i] as usize],
                "normals": &m.mesh_normal()[n..n + m.mesh_normalnum()[i] as usize],
                "faceNormals": &m.mesh_facenormal()[f..f + m.mesh_facenum()[i] as usize]})
            })
            .collect();
        let geoms: Vec<_> = (0..m.ngeom() as usize).map(|i| {
            let material = m.geom_matid()[i];
            let mut color = m.geom_rgba()[i];
            let mut repeat = [1.0_f32; 2];
            if material >= 0 {
                color = m.mat_rgba()[material as usize];
                repeat = m.mat_texrepeat()[material as usize];
                let texture = m.mat_texid()[material as usize][MjtTextureRole::mjTEXROLE_RGB as usize];
                if texture >= 0 {
                    let t = texture as usize;
                    let start = m.tex_adr()[t] as usize;
                    let pixels = m.tex_width()[t] as usize * m.tex_height()[t] as usize;
                    let channels = m.tex_nchannel()[t] as usize;
                    let bytes = &m.tex_data()[start..start + pixels * channels];
                    let mut sums = [0_u64; 3];
                    for pixel in bytes.chunks_exact(channels) {
                        for channel in 0..3 { sums[channel] += u64::from(pixel[channel.min(channels - 1)]); }
                    }
                    let material_name = name(MjtObj::mjOBJ_MATERIAL, material as usize);
                    if !matches!(material_name.as_str(), "grid" | "habitat/wood" | "habitat/darkwood") {
                        for channel in 0..3 { color[channel] *= sums[channel] as f32 / (pixels as f32 * 255.0); }
                    }
                }
            }
            json!({"id":i, "name":name(MjtObj::mjOBJ_GEOM,i), "body":m.geom_bodyid()[i],
                "type":m.geom_type()[i] as i32, "size":m.geom_size()[i], "pos":m.geom_pos()[i],
                "quat":m.geom_quat()[i], "mesh":m.geom_dataid()[i], "rgba":color,
                "material":if material >= 0 {name(MjtObj::mjOBJ_MATERIAL, material as usize)} else {String::new()},
                "group":m.geom_group()[i], "texrepeat":repeat})
        }).collect();
        let cameras: Vec<_> = (0..m.ncam() as usize)
            .map(|i| {
                json!({
                    "name":name(MjtObj::mjOBJ_CAMERA,i), "body":m.cam_bodyid()[i],
                    "pos":m.cam_pos()[i], "quat":m.cam_quat()[i], "fovy":m.cam_fovy()[i],
                })
            })
            .collect();
        respond(
            json!({"bodyCount":m.nbody(), "meshCount":m.nmesh(), "geoms":geoms,
            "meshes":meshes,"cameras":cameras,"brain":{
                "neurons":state.simulation.brain_neuron_count(),
                "model":state.simulation.brain_model_name(),
                "backend":state.simulation.brain_device_name(),
            }}),
        );
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_response_ptr() -> *const u8 {
    RESPONSE.with(|response| response.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_response_len() -> usize {
    RESPONSE.with(|response| response.borrow().len())
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_poses_ptr() -> *const f32 {
    STATE.with(|state| state.borrow().as_ref().unwrap().poses.as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_poses_len() -> usize {
    STATE.with(|state| state.borrow().as_ref().unwrap().poses.len())
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_vision_ptr() -> *mut u8 {
    STATE.with(|state| state.borrow_mut().as_mut().unwrap().vision.as_mut_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn fb_display_ptr() -> *const u8 {
    STATE.with(|state| state.borrow().as_ref().unwrap().display.as_ptr())
}
