use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use flybrain_engine::aerodynamics::Vec3;
use flybrain_engine::flight::{
    FlightCommand, FlightDynamicsParameters, FlightRuntime, FlightStabilizer,
};
use flybrain_engine::flight_behavior::FlightMode;
use flybrain_engine::flight_system_id_world::FlightCalibrationEvaluator;
use flybrain_engine::flight_targets::FlightTargetAsset;
use flybrain_engine::gait::GaitLibrary;
use flybrain_engine::live_viewer::{LiveRenderOptions, LiveViewer};
use flybrain_engine::render::{FlyRenderer, VideoRecorder};
use flybrain_engine::system_id::{
    MetricTarget, OptimizerConfig, ParameterVector, Split, TrainObjective, TrialRecord,
    optimize_train,
};
use flybrain_engine::world::{DEFAULT_ASSETS_DIR, MuJoCoWorld};
use flybrain_engine::world_sim::{
    SimulationParameterArtifact, SimulationParameters, SimulationSnapshot, SimulationStepper,
};
use serde_json::json;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(name = "flybrain-world", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    CnsCheck(CnsCheckOptions),
    Inspect {
        #[arg(long, default_value = DEFAULT_ASSETS_DIR)]
        assets: PathBuf,
    },
    Verify {
        #[arg(long, default_value = DEFAULT_ASSETS_DIR)]
        assets: PathBuf,
        #[arg(long, default_value_t = 1000)]
        steps: usize,
    },
    FlightCheck {
        #[arg(long, default_value = DEFAULT_ASSETS_DIR)]
        assets: PathBuf,
        #[arg(long, default_value_t = 0.2)]
        duration_seconds: f64,
        #[arg(long, default_value_t = 1.0)]
        amplitude: f64,
        #[arg(long, default_value_t = 1.0)]
        frequency_scale: f64,
        #[arg(long, default_value_t = 0.0)]
        steering: f64,
        #[arg(long, default_value_t = 0.0)]
        pitch_bias_rad: f64,
        #[arg(long, default_value_t = 0.0)]
        roll_bias_rad: f64,
        #[arg(long, default_value_t = 0.0)]
        initial_pitch_deg: f64,
        #[arg(long, default_value_t = 0.0)]
        initial_forward_speed_mm_s: f64,
        #[arg(long)]
        initial_height_mm: Option<f64>,
        #[arg(long)]
        tethered: bool,
        #[arg(long)]
        stabilized: bool,
        #[arg(long)]
        parameters: Option<PathBuf>,
    },
    IdentifyFlight {
        #[arg(long, default_value = DEFAULT_ASSETS_DIR)]
        assets: PathBuf,
        #[arg(
            long,
            default_value = "assets/neuromechfly/flight_system_id_targets_v1.json"
        )]
        targets: PathBuf,
        #[arg(
            long,
            default_value = "assets/neuromechfly/simulation_parameters_baseline_v1.json"
        )]
        baseline_parameters: PathBuf,
        #[arg(
            long,
            default_value = "outputs/system_id/simulation_parameters_flight_v1.json"
        )]
        output: PathBuf,
        #[arg(long, default_value = "outputs/system_id/flight_fit_report_v1.json")]
        report: PathBuf,
        #[arg(long, default_value_t = 16)]
        iterations: usize,
        #[arg(long, default_value_t = 20260816)]
        seed: u64,
        #[arg(long)]
        force: bool,
    },
    EvaluateFlight {
        #[arg(long, default_value = DEFAULT_ASSETS_DIR)]
        assets: PathBuf,
        #[arg(
            long,
            default_value = "assets/neuromechfly/flight_system_id_targets_v1.json"
        )]
        targets: PathBuf,
        #[arg(
            long,
            default_value = "outputs/system_id/simulation_parameters_flight_v1.json"
        )]
        parameters: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        force: bool,
    },
    Render {
        #[arg(long, default_value = DEFAULT_ASSETS_DIR)]
        assets: PathBuf,
        #[arg(long, default_value = "outputs/packs/flywire_v783")]
        pack: PathBuf,
        #[arg(long, default_value = "outputs/world/flybrain-world.mp4")]
        output: PathBuf,
        #[arg(long)]
        preview: Option<PathBuf>,
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long, default_value_t = 3.0)]
        duration_seconds: f64,
        #[arg(long, default_value_t = 30)]
        fps: u32,
        #[arg(long, default_value_t = 960)]
        width: u32,
        #[arg(long, default_value_t = 720)]
        height: u32,
        #[arg(long, default_value_t = 500.0)]
        control_hz: f64,
        #[arg(long, default_value_t = 0.5)]
        settle_seconds: f64,
        #[arg(long, default_value = "fly/trackingcam")]
        camera: String,
        #[arg(long)]
        no_brain: bool,
        #[arg(long)]
        parameters: Option<PathBuf>,
        #[arg(long)]
        force: bool,
    },
    View {
        #[arg(long, default_value = DEFAULT_ASSETS_DIR)]
        assets: PathBuf,
        #[arg(long, default_value = "outputs/packs/male_cns_v1")]
        pack: PathBuf,
        #[arg(long, default_value_t = 1280)]
        width: u32,
        #[arg(long, default_value_t = 800)]
        height: u32,
        #[arg(long, default_value_t = 30)]
        fps: u32,
        #[arg(long, default_value_t = 500.0)]
        control_hz: f64,
        #[arg(long, default_value_t = 0.5)]
        settle_seconds: f64,
        #[arg(long, default_value_t = 1.0)]
        speed: f64,
        #[arg(long, default_value_t = 40.0)]
        start_food_distance: f64,
        #[arg(long, default_value = "chase")]
        camera: String,
        #[arg(long)]
        max_seconds: Option<f64>,
        #[arg(long)]
        no_brain: bool,
        #[arg(long)]
        parameters: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::CnsCheck(options) => cns_world_check(options),
        Command::Inspect { assets } => inspect(assets),
        Command::Verify { assets, steps } => verify(assets, steps),
        Command::FlightCheck {
            assets,
            duration_seconds,
            amplitude,
            frequency_scale,
            steering,
            pitch_bias_rad,
            roll_bias_rad,
            initial_pitch_deg,
            initial_forward_speed_mm_s,
            initial_height_mm,
            tethered,
            stabilized,
            parameters,
        } => flight_check(FlightCheckOptions {
            assets,
            duration_seconds,
            amplitude,
            frequency_scale,
            steering,
            pitch_bias_rad,
            roll_bias_rad,
            initial_pitch_deg,
            initial_forward_speed_mm_s,
            initial_height_mm,
            tethered,
            stabilized,
            parameters,
        }),
        Command::IdentifyFlight {
            assets,
            targets,
            baseline_parameters,
            output,
            report,
            iterations,
            seed,
            force,
        } => identify_flight(FlightIdentificationOptions {
            assets,
            targets,
            baseline_parameters,
            output,
            report,
            iterations,
            seed,
            force,
        }),
        Command::EvaluateFlight {
            assets,
            targets,
            parameters,
            output,
            force,
        } => evaluate_flight(assets, targets, parameters, output, force),
        Command::Render {
            assets,
            pack,
            output,
            preview,
            manifest,
            duration_seconds,
            fps,
            width,
            height,
            control_hz,
            settle_seconds,
            camera,
            no_brain,
            parameters,
            force,
        } => render_world(RenderOptions {
            assets,
            pack,
            output,
            preview,
            manifest,
            duration_seconds,
            fps,
            width,
            height,
            control_hz,
            settle_seconds,
            camera,
            with_brain: !no_brain,
            parameters,
            force,
        }),
        Command::View {
            assets,
            pack,
            width,
            height,
            fps,
            control_hz,
            settle_seconds,
            speed,
            start_food_distance,
            camera,
            max_seconds,
            no_brain,
            parameters,
        } => view_world(ViewOptions {
            assets,
            pack,
            width,
            height,
            fps,
            control_hz,
            settle_seconds,
            speed,
            start_food_distance,
            camera,
            max_seconds,
            with_brain: !no_brain,
            parameters,
        }),
    }
}

#[derive(Debug, clap::Args)]
struct CnsCheckOptions {
    #[arg(long, default_value = DEFAULT_ASSETS_DIR)]
    assets: PathBuf,
    #[arg(long, default_value = "outputs/packs/male_cns_v1")]
    pack: PathBuf,
    #[arg(long, default_value_t = 10.0)]
    duration_seconds: f64,
    #[arg(long, default_value_t = 500.0)]
    control_hz: f64,
    #[arg(long, default_value_t = 0.5)]
    settle_seconds: f64,
    #[arg(long, default_value_t = 40.0)]
    start_food_distance: f64,
    #[arg(long)]
    disconnect_motor_outputs: bool,
    #[arg(long)]
    disconnect_landing_output: bool,
    #[arg(long)]
    disconnect_sensory_inputs: bool,
    #[arg(long)]
    disconnect_olfactory_evoked_inputs: bool,
    #[arg(long)]
    disconnect_odor_guidance: bool,
    #[arg(long, default_value_t = 0.0)]
    initial_yaw_deg: f64,
    #[arg(long)]
    parameters: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}

fn cns_world_check(options: CnsCheckOptions) -> Result<()> {
    if !options.duration_seconds.is_finite() || options.duration_seconds <= 0.0 {
        bail!("duration_seconds must be finite and positive")
    }
    if options.output.exists() {
        bail!("output already exists: {}", options.output.display())
    }
    let mut parameters = options
        .parameters
        .as_deref()
        .map(SimulationParameters::load)
        .transpose()?
        .unwrap_or_default();
    parameters.brain.cns_motor_outputs_enabled = !options.disconnect_motor_outputs;
    parameters.brain.cns_landing_output_enabled = !options.disconnect_landing_output;
    let paired_olfactory_input_rate_hz = parameters.brain.olfactory_input_rate_hz;
    let paired_odor_guidance_enabled = parameters.odor_guidance.enabled;
    if options.disconnect_odor_guidance {
        parameters.odor_guidance.enabled = false;
    }
    if options.disconnect_olfactory_evoked_inputs {
        parameters.brain.olfactory_input_rate_hz = 0.0;
    }
    if options.disconnect_sensory_inputs {
        parameters.brain.taste_input_rate_hz = 0.0;
        parameters.brain.olfactory_baseline_rate_hz = 0.0;
        parameters.brain.olfactory_input_rate_hz = 0.0;
        parameters.brain.visual_baseline_rate_hz = 0.0;
        parameters.brain.visual_input_rate_hz = 0.0;
        parameters.brain.flight_state_input_rate_hz = 0.0;
    }
    let mut simulation = SimulationStepper::new_with_parameters(
        &options.assets,
        Some(&options.pack),
        options.control_hz,
        options.settle_seconds,
        parameters,
    )?;
    if simulation.brain_materialization()
        != Some(flybrain_engine::neural_io::MALE_CNS_MATERIALIZATION)
    {
        bail!("cns-check requires the MaleCNS pack")
    }
    simulation.set_initial_yaw(options.initial_yaw_deg.to_radians())?;
    simulation.place_food_ahead(options.start_food_distance)?;
    simulation.set_brain_telemetry_enabled(true);
    let started = Instant::now();
    let initial_position = simulation.snapshot().root_position;
    let habitat = flybrain_engine::habitat::Habitat::load(&options.assets)?;
    let room = habitat.room().half_extents_mm;
    let room_bounds_mm = [[-room[0], -room[1], 0.0], [room[0], room[1], 2.0 * room[2]]];
    let pack_manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(options.pack.join("manifest.json"))?)?;
    let io_sha256 = format!(
        "{:x}",
        Sha256::digest(fs::read(
            options
                .assets
                .join(flybrain_engine::brain_bridge::MALE_CNS_NEURAL_IO_FILE)
        )?)
    );
    let mut asset_hashes = std::collections::BTreeMap::new();
    for name in [
        "manifest.json",
        "fly.xml",
        "habitat.json",
        "aerodynamics.json",
        "tripod_gait.json",
    ] {
        asset_hashes.insert(
            name,
            format!("{:x}", Sha256::digest(fs::read(options.assets.join(name))?)),
        );
    }
    let mut paired_parameters = parameters;
    paired_parameters.brain.cns_motor_outputs_enabled = true;
    paired_parameters.brain.cns_landing_output_enabled = true;
    paired_parameters.odor_guidance.enabled = paired_odor_guidance_enabled;
    if options.disconnect_olfactory_evoked_inputs && !options.disconnect_sensory_inputs {
        paired_parameters.brain.olfactory_input_rate_hz = paired_olfactory_input_rate_hz;
    }
    let runtime_sha256 = format!("{:x}", Sha256::digest(fs::read(std::env::current_exe()?)?));
    let initial_state = json!({
        "runtime_sha256": runtime_sha256,
        "initial_position_mm": initial_position,
        "initial_quaternion": simulation.world().root_quaternion(),
        "pack_arrays": pack_manifest["array_sha256"], "io_sha256": io_sha256,
        "assets": asset_hashes, "parameters": paired_parameters,
        "control_hz": options.control_hz, "settle_seconds": options.settle_seconds,
        "start_food_distance": options.start_food_distance,
        "sensory_encoder": "deterministic fractional-rate accumulator",
    });
    let initial_state_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&initial_state)?));
    let mut previous_position = initial_position;
    let mut minimum_position = initial_position;
    let mut maximum_position = initial_position;
    let mut path_length_mm = 0.0;
    let mut flight_seconds = 0.0;
    let mut feeding_seconds = 0.0;
    let mut maximum_speed_mm_s = 0.0_f64;
    let mut maximum_abs_pitch_deg = 0.0_f64;
    let mut population_spikes = 0_u64;
    let mut motor_output_spikes = 0_u64;
    let mut forward_flight_distance_mm = 0.0;
    let mut samples = Vec::new();
    let period = simulation.control_period().as_secs_f64();
    let mut next_sample_time = 0.0;
    let mut next_progress_time = 1.0;
    while simulation.snapshot().time_seconds + period * 0.5 < options.duration_seconds {
        let snapshot = simulation.step_window()?;
        if snapshot
            .root_position
            .iter()
            .any(|value| !value.is_finite())
        {
            bail!("non-finite CNS world position")
        }
        for axis in 0..3 {
            minimum_position[axis] = minimum_position[axis].min(snapshot.root_position[axis]);
            maximum_position[axis] = maximum_position[axis].max(snapshot.root_position[axis]);
        }
        path_length_mm += (snapshot.root_position[0] - previous_position[0])
            .hypot(snapshot.root_position[1] - previous_position[1]);
        previous_position = snapshot.root_position;
        population_spikes += snapshot.population_spike_delta;
        motor_output_spikes += snapshot.cns_motor.map_or(0, |motor| motor.spike_delta);
        if snapshot.flight_mode != FlightMode::Grounded {
            flight_seconds += period;
            if snapshot.root_position[2] > initial_position[2] + 3.0 {
                forward_flight_distance_mm += snapshot.forward_speed_mm_s.max(0.0) * period;
            }
        }
        let [_, body_x, body_y, _] = simulation.world().root_quaternion();
        let body_up_z = 1.0 - 2.0 * (body_x * body_x + body_y * body_y);
        if snapshot.taste_active
            && snapshot.behavior_mode == flybrain_engine::behavior::BehaviorMode::Feed
            && snapshot.foraging_mode == flybrain_engine::foraging::ForagingMode::Feed
            && snapshot.flight_mode == FlightMode::Grounded
            && snapshot.filtered_mn9_rate_hz > 0.0
            && snapshot.feeding_extension > 0.1
            && snapshot.contact_count >= 2
            && body_up_z > 0.8
        {
            feeding_seconds += period;
        }
        maximum_speed_mm_s = maximum_speed_mm_s.max(snapshot.horizontal_speed_mm_s);
        maximum_abs_pitch_deg = maximum_abs_pitch_deg.max(snapshot.body_pitch_deg.abs());
        if snapshot.time_seconds >= next_sample_time {
            let flight_diagnostics = json!({
                "wall_support_leg_count": snapshot.wall_support_leg_count,
                "wall_landing_normal": snapshot.wall_landing_target.map(|target| target.inward_xy),
                "wall_landing_surface_point_mm": snapshot.wall_landing_target.map(|target| target.surface_point_mm),
                "wall_landing_alignment": snapshot.wall_landing_alignment,
                "amplitude_scale": snapshot.flight_amplitude_scale,
                "wing_controls": &simulation.world().controls()[flybrain_engine::world::WING_ACTUATOR_START..],
                "wing_joint_positions": simulation.world().wing_joint_positions(),
                "wing_joint_velocities": simulation.world().wing_joint_velocities(),
            });
            let mut trace_sample = json!({
                "time_seconds": snapshot.time_seconds,
                "root_position": snapshot.root_position,
                "flight_mode": snapshot.flight_mode.label(),
                "foraging_mode": snapshot.foraging_mode.label(),
                "behavior_mode": snapshot.behavior_mode.label(),
                "brain_flight_drive": snapshot.brain_flight_drive,
                "brain_altitude_control": snapshot.brain_altitude_control,
                "flight_target_height_mm": snapshot.flight_target_height_mm,
                "brain_flight_steering": snapshot.brain_flight_steering,
                "brain_walking_drive": snapshot.brain_walking_drive,
                "brain_landing_drive": snapshot.brain_landing_drive,
                "cns_motor": snapshot.cns_motor,
                "cns_olfactory": snapshot.cns_olfactory,
                "odor_guidance": snapshot.odor_guidance,
                "forward_speed_mm_s": snapshot.forward_speed_mm_s,
                "root_quaternion": simulation.world().root_quaternion(),
                "root_velocity_world": simulation.world().root_velocity(),
                "flight_steering": snapshot.flight_steering,
                "walking_activation": snapshot.forward_gain,
                "walking_turn_gain": snapshot.walking_turn_gain,
                "walking_translation_scale": snapshot.walking_translation_scale,
                "horizontal_speed_mm_s": snapshot.horizontal_speed_mm_s,
                "body_pitch_deg": snapshot.body_pitch_deg,
                "taste_active": snapshot.taste_active,
                "tasted_resource": snapshot.tasted_resource,
                "nearest_resource_distance_mm": snapshot.nearest_resource_distance,
                "contact_count": snapshot.contact_count,
                "feeding_extension": snapshot.feeding_extension,
                "mn9_spike_delta": snapshot.mn9_spike_delta,
                "mn9_rate_hz": snapshot.filtered_mn9_rate_hz,
                "odor_left": snapshot.odor_left,
                "odor_right": snapshot.odor_right,
                "odor_left_ppm": snapshot.odor_left_ppm,
                "odor_right_ppm": snapshot.odor_right_ppm,
                "visual_event_delta": snapshot.visual_event_delta,
                "olfactory_event_delta": snapshot.olfactory_event_delta,
                "taste_event_delta": snapshot.taste_event_delta,
                "population_rate_hz": snapshot.filtered_population_rate_hz,
                "population_spike_delta": snapshot.population_spike_delta,
                "boundary_avoidance": snapshot.flight_boundary_avoidance,
                "collision_reflex_active": snapshot.flight_escape_active,
            });
            trace_sample["flight_diagnostics"] = flight_diagnostics;
            samples.push(trace_sample);
            next_sample_time = snapshot.time_seconds + 0.01;
        }
        if snapshot.time_seconds >= next_progress_time {
            eprintln!(
                "CNS {:.1}s: {} pos {:?}, ORN {:.2?}ppm turn {:.3}, {}",
                snapshot.time_seconds,
                snapshot.flight_mode.label(),
                snapshot.root_position,
                snapshot.cns_olfactory.map(|odor| odor.concentration_ppm),
                snapshot.odor_guidance.steering,
                snapshot.foraging_mode.label()
            );
            next_progress_time += 1.0;
        }
    }
    let report = json!({
        "schema": "flybrain.cns-world-check", "schema_version": 1,
        "runtime_sha256": runtime_sha256,
        "brain": {"model": simulation.brain_model_name(),
            "neurons": simulation.brain_neuron_count(),
            "sensory_neurons": simulation.brain_sensory_neuron_count(),
            "materialization": simulation.brain_materialization(),
            "motor_outputs_connected": !options.disconnect_motor_outputs,
            "landing_output_connected": !options.disconnect_landing_output,
            "sensory_inputs_connected": !options.disconnect_sensory_inputs,
            "olfactory_evoked_inputs_connected": !options.disconnect_sensory_inputs && !options.disconnect_olfactory_evoked_inputs,
            "odor_guidance_enabled": parameters.odor_guidance.enabled,
            "allocated_bytes": simulation.brain_allocated_bytes()},
        "pack": options.pack,
        "duration_seconds": options.duration_seconds, "control_hz": options.control_hz,
        "initial_state": initial_state,
        "room_bounds_mm": room_bounds_mm,
        "parameters": parameters,
        "sensory_interface": "odor receptor proxy; ray/velocity-derived optic-flow and looming proxy; no retinal transduction claim",
        "motor_interface": "annotated CNS motor population rate decoder; engineered gait, wing waveform and stabilizer",
        "odor_guidance_interface": "engineered chemotaxis decoder of simulated DM1/DM2 ORN firing; bypasses unvalidated central food-heading circuitry; no food coordinates",
        "feeding_metric": "grounded FEED state with physical taste, MN9 activity and proboscis extension; excludes post-meal decay",
        "summary": {"duration_seconds": simulation.snapshot().time_seconds,
            "initial_state_sha256": initial_state_sha256,
            "population_spikes": population_spikes, "motor_output_spikes": motor_output_spikes,
            "motor_output_source": if options.disconnect_motor_outputs { "disconnected" } else { "whole-cns-spikes" },
            "forward_flight_distance_mm": forward_flight_distance_mm,
            "initial_position_mm": initial_position, "final_position_mm": previous_position,
            "minimum_position_mm": minimum_position, "maximum_position_mm": maximum_position,
            "path_length_mm": path_length_mm, "flight_seconds": flight_seconds,
            "feeding_seconds": feeding_seconds, "maximum_speed_mm_s": maximum_speed_mm_s,
            "maximum_abs_pitch_deg": maximum_abs_pitch_deg,
            "elapsed_seconds": started.elapsed().as_secs_f64()},
        "samples": samples,
    });
    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent)?;
    }
    use std::io::Write;
    let mut output = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&options.output)?;
    writeln!(output, "{}", serde_json::to_string_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report["summary"])?);
    Ok(())
}

struct ViewOptions {
    assets: PathBuf,
    pack: PathBuf,
    width: u32,
    height: u32,
    fps: u32,
    control_hz: f64,
    settle_seconds: f64,
    speed: f64,
    start_food_distance: f64,
    camera: String,
    max_seconds: Option<f64>,
    with_brain: bool,
    parameters: Option<PathBuf>,
}

fn view_world(options: ViewOptions) -> Result<()> {
    validate_view_options(&options)?;
    eprintln!("Loading the body and connectome for live simulation...");
    let pack = options.with_brain.then_some(options.pack.as_path());
    let parameters = options
        .parameters
        .as_deref()
        .map(SimulationParameters::load)
        .transpose()?
        .unwrap_or_default();
    let mut simulation = SimulationStepper::new_with_parameters(
        &options.assets,
        pack,
        options.control_hz,
        options.settle_seconds,
        parameters,
    )?;
    simulation.set_brain_telemetry_enabled(options.with_brain);
    simulation.place_food_ahead(options.start_food_distance)?;
    let mut viewer = LiveViewer::new(
        simulation.world().model(),
        &options.assets,
        options.width,
        options.height,
        &options.camera,
    )?;
    let brain_device = simulation
        .brain_device_name()
        .unwrap_or("disabled")
        .to_string();
    let brain_model = simulation
        .brain_model_name()
        .unwrap_or("disabled")
        .to_string();
    let full_neural_io = simulation.full_neural_io_enabled();
    let brain_neuron_count = simulation.brain_neuron_count();
    let brain_sensory_neuron_count = simulation.brain_sensory_neuron_count();
    let neural_io_stats = simulation.neural_io_stats();
    eprintln!(
        "Live window opened with {brain_neuron_count} simulated neurons ({brain_model} on {brain_device}); full neural I/O {full_neural_io}, {}/{}/{} selected/present/missing. V toggles both eye views, B toggles the EEG-like network field, G toggles autonomous flight, H requests front-leg grooming, and ESC quits.",
        neural_io_stats.selected_root_ids,
        neural_io_stats.present_root_ids,
        neural_io_stats.missing_root_ids,
    );

    let mut paused = false;
    let mut show_eye_view = true;
    let mut show_brain_graph = simulation.brain_enabled();
    simulation.set_brain_telemetry_enabled(show_brain_graph);
    let mut anchor_wall = Instant::now();
    let mut anchor_sim = simulation.world().time();
    let mut stats_wall = Instant::now();
    let mut stats_sim = simulation.world().time();
    let mut realtime_factor = 0.0;
    let mut snapshot = simulation.snapshot();
    let mut last_brain_field_sample_sequence = snapshot.brain_field_sample_sequence;
    let mut logged_flight_mode = snapshot.flight_mode;
    let control_seconds = simulation.control_period().as_secs_f64();
    let frame_period = Duration::from_secs_f64(1.0 / f64::from(options.fps));
    let mut last_frame = Instant::now() - frame_period;
    let mut last_title = Instant::now() - Duration::from_secs(1);

    while viewer.is_open() {
        let input = viewer.poll_input(simulation.world().model());
        if input.quit {
            break;
        }
        if input.toggle_pause {
            paused = !paused;
            anchor_wall = Instant::now();
            anchor_sim = simulation.world().time();
        }
        if input.reset {
            simulation.reset()?;
            viewer.clear_brain_history();
            anchor_wall = Instant::now();
            anchor_sim = simulation.world().time();
            stats_wall = anchor_wall;
            stats_sim = anchor_sim;
            realtime_factor = 0.0;
            snapshot = simulation.snapshot();
            last_brain_field_sample_sequence = snapshot.brain_field_sample_sequence;
            logged_flight_mode = snapshot.flight_mode;
        }
        if input.toggle_food {
            simulation.toggle_food()?;
            snapshot = simulation.snapshot();
        }
        if input.toggle_flight {
            simulation.toggle_flight();
            snapshot = simulation.snapshot();
        }
        if input.request_grooming {
            simulation.request_grooming();
        }
        if input.place_food_at_mouth {
            simulation.drop_food_below_fly()?;
            snapshot = simulation.snapshot();
        }
        if input.toggle_eye_view {
            show_eye_view = !show_eye_view;
        }
        if input.toggle_brain_graph && simulation.brain_enabled() {
            show_brain_graph = !show_brain_graph;
            simulation.set_brain_telemetry_enabled(show_brain_graph);
            if show_brain_graph {
                viewer.clear_brain_history();
                last_brain_field_sample_sequence = 0;
            }
        }
        if input.food_motion != [0.0; 2] {
            simulation.move_food([input.food_motion[0], input.food_motion[1], 0.0])?;
            snapshot = simulation.snapshot();
        }

        if !paused {
            let target_sim = anchor_sim + anchor_wall.elapsed().as_secs_f64() * options.speed;
            let mut windows = 0;
            while simulation.world().time() + control_seconds * 0.5 < target_sim && windows < 25 {
                snapshot = simulation.step_window()?;
                if show_brain_graph
                    && snapshot.brain_field_sample_sequence != last_brain_field_sample_sequence
                {
                    viewer.record_brain_field_sample(
                        snapshot.time_seconds,
                        snapshot.brain_field_potential_mv,
                        snapshot.brain_field_dominant_frequency_hz,
                    );
                    last_brain_field_sample_sequence = snapshot.brain_field_sample_sequence;
                }
                if snapshot.flight_mode != logged_flight_mode {
                    eprintln!(
                        "Flight transition at {:.3}s: {} -> {}, z {:.2} mm, target {:.2} mm, altitude command {:+.3}, nose-up {:.1} deg, speed {:.1} mm/s (forward {:.1}), flight drive {:.3}, landing DN {:.1} Hz/{:.3}, odor ppm L/R {:.2}/{:.2}, perceived {:.3}/{:.3}.",
                        snapshot.time_seconds,
                        logged_flight_mode.label(),
                        snapshot.flight_mode.label(),
                        snapshot.root_position[2],
                        snapshot.flight_target_height_mm,
                        snapshot.brain_altitude_control,
                        -snapshot.body_pitch_deg,
                        snapshot.horizontal_speed_mm_s,
                        snapshot.forward_speed_mm_s,
                        snapshot.brain_flight_drive,
                        snapshot.landing_dn_rate_hz,
                        snapshot.brain_landing_drive,
                        snapshot.odor_left_ppm,
                        snapshot.odor_right_ppm,
                        snapshot.odor_left,
                        snapshot.odor_right,
                    );
                    logged_flight_mode = snapshot.flight_mode;
                }
                windows += 1;
            }
            if windows == 25 && simulation.world().time() + control_seconds < target_sim {
                anchor_wall = Instant::now();
                anchor_sim = simulation.world().time();
            }
        }

        if stats_wall.elapsed() >= Duration::from_millis(500) {
            realtime_factor =
                (simulation.world().time() - stats_sim) / stats_wall.elapsed().as_secs_f64();
            stats_wall = Instant::now();
            stats_sim = simulation.world().time();
        }
        if last_frame.elapsed() < frame_period {
            std::thread::sleep((frame_period - last_frame.elapsed()).min(Duration::from_millis(2)));
            continue;
        }
        last_frame = Instant::now();
        let nearest_resource = simulation.resource_label(snapshot.nearest_resource);
        let tasted_resource = simulation.resource_label(snapshot.tasted_resource);
        let nearest_obstacle = simulation.obstacle_label(snapshot.flight_nearest_obstacle_geom_id);
        let status = live_status(LiveStatusContext {
            snapshot,
            paused,
            realtime_factor,
            brain_neuron_count,
            brain_sensory_neuron_count,
            nearest_resource,
            tasted_resource,
            nearest_obstacle,
        });
        if last_title.elapsed() >= Duration::from_millis(500) {
            let title = format!(
                "FlyBrain live — {:.1}s — {} — {:.2}x",
                snapshot.time_seconds,
                if paused { "PAUSED" } else { "RUNNING" },
                realtime_factor
            );
            viewer.set_title(&title)?;
            last_title = Instant::now();
        }
        let capture_vision = simulation.brain_enabled();
        viewer.render(
            simulation.world_mut().data_mut(),
            LiveRenderOptions {
                food_center: snapshot.food_center,
                food_enabled: snapshot.food_enabled,
                status: &status,
                show_eye_view,
                show_brain_graph,
                capture_vision,
                flight_allowed: snapshot.flight_allowed,
            },
        )?;
        simulation.set_retina_summaries(viewer.retina_summaries())?;
        if options
            .max_seconds
            .is_some_and(|limit| simulation.world().time() >= limit)
        {
            break;
        }
    }
    eprintln!(
        "Live simulation stopped at {:.3}s: position [{:.3}, {:.3}, {:.3}], nose-up {:.1} deg, horizontal speed {:.1} mm/s (forward {:.1}), flight {}, ever-spiking neurons {}.",
        snapshot.time_seconds,
        snapshot.root_position[0],
        snapshot.root_position[1],
        snapshot.root_position[2],
        -snapshot.body_pitch_deg,
        snapshot.horizontal_speed_mm_s,
        snapshot.forward_speed_mm_s,
        snapshot.flight_mode.label(),
        snapshot.cumulative_spiking_neuron_count,
    );
    Ok(())
}

struct LiveStatusContext<'a> {
    snapshot: SimulationSnapshot,
    paused: bool,
    realtime_factor: f64,
    brain_neuron_count: usize,
    brain_sensory_neuron_count: usize,
    nearest_resource: &'a str,
    tasted_resource: &'a str,
    nearest_obstacle: &'a str,
}

fn live_status(context: LiveStatusContext<'_>) -> String {
    let LiveStatusContext {
        snapshot,
        paused,
        realtime_factor,
        brain_neuron_count,
        brain_sensory_neuron_count,
        nearest_resource,
        tasted_resource,
        nearest_obstacle,
    } = context;
    if let Some(motor) = snapshot.cns_motor {
        return format!(
            "{} t={:.1}s {:.2}x {}\np [{:.0},{:.0},{:.1}]mm v {:.0} pitch {:+.0}\nMaleCNS {}n/{}sens outputs {}\nMN power {:.1}/{:.1} walk {:.1}/{:.1} Hz\nMN steer {:.1}/{:.1} Hz cmd {:+.2}\nflight {:.2} altitude {:.1}>{:.1}mm DNg {:+.2}\nodor {:.2}/{:.2}ppm taste {}\nORN guidance (engineered) {} turn {:+.2}\nvision motion/loom proxies | collision {}\nfield {:+.4}mV {:.1}Hz | food {} {:.0}mm",
            if paused { "PAUSED" } else { "RUNNING" },
            snapshot.time_seconds,
            realtime_factor,
            snapshot.flight_mode.label(),
            snapshot.root_position[0],
            snapshot.root_position[1],
            snapshot.root_position[2],
            snapshot.horizontal_speed_mm_s,
            -snapshot.body_pitch_deg,
            brain_neuron_count,
            brain_sensory_neuron_count,
            if motor.outputs_connected { "ON" } else { "OFF" },
            motor.flight_power_hz[0],
            motor.flight_power_hz[1],
            motor.walking_hz[0],
            motor.walking_hz[1],
            motor.wing_steering_hz[0],
            motor.wing_steering_hz[1],
            motor.steering,
            motor.flight_activation,
            snapshot.root_position[2],
            snapshot.flight_target_height_mm,
            snapshot.brain_altitude_control,
            snapshot.odor_left_ppm,
            snapshot.odor_right_ppm,
            if snapshot.taste_active { "YES" } else { "no" },
            if snapshot.odor_guidance.active {
                "ON"
            } else {
                "OFF"
            },
            snapshot.odor_guidance.steering,
            if snapshot.flight_escape_active {
                "YES"
            } else {
                "no"
            },
            snapshot.brain_field_potential_mv,
            snapshot.brain_field_dominant_frequency_hz,
            nearest_resource,
            snapshot.nearest_resource_distance,
        );
    }
    format!(
        "{} t={:5.1}s {:4.2}x {:?}/{} {}\np [{:4.0},{:4.0},{:5.1}]mm v {:4.0} pitch {:+3.0}\nbrain {}n/{}sens field {:+.4}mV {:4.1}Hz\nDN W {:.1}/{:.1} d{:.2} F {:.1}/{:.1} d{:.2}\nsens odor ppm {:.2}/{:.2} perceived {:.3}/{:.3} v {:.2}/{:.2} taste {} {}\nturn o{:+.2} w{:+.2} b{:+.2} x{:+.2} ={:+.2} esc {}\nalt {:5.1}>{:5.1} raw{:+.2} motor{:+.1} DNg {:.1}/{:.1} land {:.1}/{:.2}\nflow {:.1}/s hold {} obs {} F/U/D {:3.0}/{:3.0}/{:3.0} contact {} wall {} perch {} food {} {:3.0}mm",
        if paused { "PAUSED" } else { "RUNNING" },
        snapshot.time_seconds,
        realtime_factor,
        snapshot.flight_mode,
        snapshot.behavior_mode.label(),
        snapshot.foraging_mode.label(),
        snapshot.root_position[0],
        snapshot.root_position[1],
        snapshot.root_position[2],
        snapshot.horizontal_speed_mm_s,
        -snapshot.body_pitch_deg,
        brain_neuron_count,
        brain_sensory_neuron_count,
        snapshot.brain_field_potential_mv,
        snapshot.brain_field_dominant_frequency_hz,
        snapshot.walking_dn_left_rate_hz,
        snapshot.walking_dn_right_rate_hz,
        snapshot.brain_walking_drive,
        snapshot.flight_dn_left_rate_hz,
        snapshot.flight_dn_right_rate_hz,
        snapshot.brain_flight_drive,
        snapshot.odor_left_ppm,
        snapshot.odor_right_ppm,
        snapshot.odor_left,
        snapshot.odor_right,
        snapshot.visual_left,
        snapshot.visual_right,
        if snapshot.taste_active { "YES" } else { "no" },
        tasted_resource,
        snapshot.flight_odor_steering,
        snapshot.flight_wander_steering,
        snapshot.flight_brain_steering_contribution,
        snapshot.flight_obstacle_avoidance,
        snapshot.flight_steering,
        if snapshot.flight_escape_active {
            "YES"
        } else {
            "no"
        },
        snapshot.root_position[2],
        snapshot.flight_target_height_mm,
        snapshot.brain_altitude_control,
        snapshot.neural_altitude_contribution_mm_s,
        snapshot.flight_power_increase_rate_hz,
        snapshot.flight_power_decrease_rate_hz,
        snapshot.landing_dn_rate_hz,
        snapshot.brain_landing_drive,
        snapshot.ventral_optic_flow_rad_s,
        if snapshot.flight_altitude_hold {
            "YES"
        } else {
            "no"
        },
        nearest_obstacle,
        snapshot.flight_forward_clearance_mm,
        snapshot.flight_up_clearance_mm,
        snapshot.flight_down_clearance_mm,
        snapshot.environment_contact_count,
        snapshot.wall_support_leg_count,
        if snapshot.perched_on_wall {
            "YES"
        } else {
            "no"
        },
        nearest_resource,
        snapshot.nearest_resource_distance,
    )
}

fn validate_view_options(options: &ViewOptions) -> Result<()> {
    if options.width == 0 || options.height == 0 || options.fps == 0 {
        bail!("viewer dimensions and fps must be positive")
    }
    if !options.control_hz.is_finite() || options.control_hz <= 0.0 {
        bail!("control_hz must be finite and positive")
    }
    if !options.settle_seconds.is_finite() || options.settle_seconds < 0.0 {
        bail!("settle_seconds must be finite and non-negative")
    }
    if !options.speed.is_finite() || options.speed <= 0.0 {
        bail!("speed must be finite and positive")
    }
    if !options.start_food_distance.is_finite() || options.start_food_distance <= 0.0 {
        bail!("start_food_distance must be finite and positive")
    }
    if options
        .max_seconds
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        bail!("max_seconds must be finite and positive")
    }
    Ok(())
}

fn inspect(assets: PathBuf) -> Result<()> {
    let world = MuJoCoWorld::from_assets_dir(&assets)?;
    let gait = GaitLibrary::open(assets.join("tripod_gait.json"))?;
    let metadata = world.metadata();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": metadata.schema,
            "model": metadata.model,
            "physics": metadata.physics,
            "timestep_seconds": metadata.timestep_seconds,
            "counts": {
                "qpos": metadata.counts.qpos,
                "dofs": metadata.counts.dofs,
                "bodies": metadata.counts.bodies,
                "joints": metadata.counts.joints,
                "actuators": metadata.counts.actuators,
                "sensors": metadata.counts.sensors,
                "cameras": metadata.counts.cameras,
            },
            "environment": {
                "food_center": metadata.environment.food_center,
                "taste_radius": metadata.environment.taste_radius,
                "taste_source_body": metadata.environment.taste_source_body,
            },
            "brain_body_interface": {
                "feeding_actuator": metadata.brain_body_interface.feeding_actuator,
                "feeding_joint": metadata.brain_body_interface.feeding_joint,
                "feeding_actuators": metadata.brain_body_interface.feeding_actuators,
                "feeding_joints": metadata.brain_body_interface.feeding_joints,
                "control_range": metadata.brain_body_interface.control_range,
                "full_extension_control": metadata.brain_body_interface.full_extension_control,
                "neural_readout": metadata.brain_body_interface.neural_readout,
            },
            "gait": {
                "schema": gait.schema,
                "source": gait.source,
                "sample_count": gait.sample_count,
                "cycle_frequency_hz": gait.cycle_frequency_hz,
            },
        }))?
    );
    Ok(())
}

fn verify(assets: PathBuf, steps: usize) -> Result<()> {
    if steps == 0 {
        bail!("verification steps must be positive")
    }
    let mut world = MuJoCoWorld::from_assets_dir(&assets)?;
    for _ in 0..steps {
        world.step()?;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "flybrain-world-rust-trace-v1",
            "steps": steps,
            "time_seconds": world.time(),
            "qpos": world.qpos(),
            "qvel": world.qvel(),
            "contacts": world.ground_contacts()?,
            "qpos_sha256_f64le": sha256_f64(world.qpos()),
            "qvel_sha256_f64le": sha256_f64(world.qvel()),
        }))?
    );
    Ok(())
}

struct FlightCheckOptions {
    assets: PathBuf,
    duration_seconds: f64,
    amplitude: f64,
    frequency_scale: f64,
    steering: f64,
    pitch_bias_rad: f64,
    roll_bias_rad: f64,
    initial_pitch_deg: f64,
    initial_forward_speed_mm_s: f64,
    initial_height_mm: Option<f64>,
    tethered: bool,
    stabilized: bool,
    parameters: Option<PathBuf>,
}

fn flight_check(options: FlightCheckOptions) -> Result<()> {
    let FlightCheckOptions {
        assets,
        duration_seconds,
        amplitude,
        frequency_scale,
        steering,
        pitch_bias_rad,
        roll_bias_rad,
        initial_pitch_deg,
        initial_forward_speed_mm_s,
        initial_height_mm,
        tethered,
        stabilized,
        parameters,
    } = options;
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        bail!("flight-check duration must be finite and positive")
    }
    if !initial_pitch_deg.is_finite() || !(-89.0..=89.0).contains(&initial_pitch_deg) {
        bail!("flight-check initial pitch must be finite and inside [-89, 89] degrees")
    }
    if !initial_forward_speed_mm_s.is_finite() {
        bail!("flight-check initial forward speed must be finite")
    }
    if !steering.is_finite() || !(-1.0..=1.0).contains(&steering) {
        bail!("flight-check steering must be finite and inside [-1, 1]")
    }
    if initial_height_mm.is_some_and(|height| !height.is_finite() || height <= 0.0) {
        bail!("flight-check initial height must be finite and positive")
    }
    let command = FlightCommand {
        enabled: amplitude > 0.0,
        amplitude,
        steering,
        wing_steering_scale: 1.0,
        horizontal_speed_scale: 1.0,
        heading_target_xy: None,
        planar_velocity_direction: None,
        altitude_target_mm: None,
        body_pitch_target_rad: None,
        wall_landing: None,
        frequency_scale,
        pitch_bias_rad,
        roll_bias_rad,
        differential_pitch_rad: 0.0,
        differential_roll_rad: 0.0,
    }
    .validate()?;
    let dynamics = parameters
        .as_deref()
        .map(SimulationParameters::load)
        .transpose()?
        .map(|parameters| parameters.flight_dynamics)
        .unwrap_or_default();
    let stabilizer = FlightStabilizer::from_dynamics(dynamics)?;
    let mut world = MuJoCoWorld::from_assets_dir(&assets)?;
    let mut flight = FlightRuntime::new_with_parameters(&assets, &world, dynamics)?;
    let timestep = world.timestep_seconds();
    let settle_steps = rounded_positive_ratio(0.1, timestep, "flight-check settle period")?;
    let flight_steps = rounded_positive_ratio(duration_seconds, timestep, "flight-check duration")?;
    for _ in 0..settle_steps {
        flight.advance(&mut world, FlightCommand::default(), [0.0; 3])?;
    }
    if initial_pitch_deg != 0.0 || initial_height_mm.is_some() {
        if let Some(height) = initial_height_mm {
            world.data_mut().qpos_mut()[2] = height;
        }
        let half_angle = initial_pitch_deg.to_radians() * 0.5;
        world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
            half_angle.cos(),
            0.0,
            half_angle.sin(),
            0.0,
        ]);
        world.data_mut().qvel_mut()[..6].fill(0.0);
        world.data_mut().forward();
    }
    world.data_mut().qvel_mut()[0] = initial_forward_speed_mm_s;
    world.data_mut().forward();
    let initial_position = world.root_position();
    let tether_qpos = world.qpos()[..7].to_vec();
    let mut mean_vertical_force_to_weight = 0.0;
    let mut mean_force_g_mm_s2 = Vec3::ZERO;
    let mut mean_moment_g_mm2_s2 = Vec3::ZERO;
    let mut mean_root_qfrc_fluid = [0.0; 6];
    let mut mean_wing_forces_g_mm_s2 = [Vec3::ZERO; 2];
    let mut peak_vertical_force_to_weight = f64::NEG_INFINITY;
    let mut peak_strip_speed_mm_s = 0.0_f64;
    let mut minimum_root_height_mm = initial_position[2];
    let mut maximum_root_height_mm = initial_position[2];
    let initial_root_euler_rad = root_euler_rad(world.root_quaternion());
    let mut maximum_abs_roll_rad = initial_root_euler_rad[0].abs();
    let mut maximum_abs_pitch_rad = initial_root_euler_rad[1].abs();
    let mut first_effective_command = None;
    let mut last_effective_command = None;
    for step in 0..flight_steps {
        let step_command = if stabilized && command.enabled {
            let ramp = ((step as f64 * timestep) / 0.02).clamp(0.0, 1.0);
            stabilizer.command_with_base_limited(
                world.root_quaternion(),
                world.root_velocity(),
                command,
                ramp,
                flight.config(),
            )?
        } else {
            command
        };
        if first_effective_command.is_none() {
            first_effective_command = Some(step_command);
        }
        last_effective_command = Some(step_command);
        let telemetry = flight.advance(&mut world, step_command, [0.0; 3])?;
        mean_vertical_force_to_weight += telemetry.vertical_force_to_weight;
        mean_force_g_mm_s2 = mean_force_g_mm_s2 + telemetry.total_force_g_mm_s2;
        mean_moment_g_mm2_s2 = mean_moment_g_mm2_s2 + telemetry.total_moment_g_mm2_s2;
        mean_root_qfrc_fluid = std::array::from_fn(|index| {
            mean_root_qfrc_fluid[index] + telemetry.root_qfrc_fluid[index]
        });
        for (mean, wing) in mean_wing_forces_g_mm_s2.iter_mut().zip(&telemetry.wings) {
            *mean = *mean + wing.force_g_mm_s2;
        }
        peak_vertical_force_to_weight =
            peak_vertical_force_to_weight.max(telemetry.vertical_force_to_weight);
        peak_strip_speed_mm_s = peak_strip_speed_mm_s.max(
            telemetry
                .wings
                .iter()
                .map(|wing| wing.peak_strip_speed_mm_s)
                .fold(0.0_f64, f64::max),
        );
        minimum_root_height_mm = minimum_root_height_mm.min(world.root_position()[2]);
        maximum_root_height_mm = maximum_root_height_mm.max(world.root_position()[2]);
        let euler = root_euler_rad(world.root_quaternion());
        maximum_abs_roll_rad = maximum_abs_roll_rad.max(euler[0].abs());
        maximum_abs_pitch_rad = maximum_abs_pitch_rad.max(euler[1].abs());
        if tethered {
            world.data_mut().qpos_mut()[..7].copy_from_slice(&tether_qpos);
            world.data_mut().qvel_mut()[..6].fill(0.0);
            world.data_mut().qvel_mut()[0] = initial_forward_speed_mm_s;
            world.data_mut().forward();
        }
    }
    mean_vertical_force_to_weight /= flight_steps as f64;
    mean_force_g_mm_s2 = mean_force_g_mm_s2 / flight_steps as f64;
    mean_moment_g_mm2_s2 = mean_moment_g_mm2_s2 / flight_steps as f64;
    mean_root_qfrc_fluid = mean_root_qfrc_fluid.map(|force| force / flight_steps as f64);
    mean_wing_forces_g_mm_s2 = mean_wing_forces_g_mm_s2.map(|force| force / flight_steps as f64);
    let final_position = world.root_position();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "flybrain-flight-check-v1",
            "model": flight.config().model.name,
            "steps": flight_steps,
            "duration_seconds": duration_seconds,
            "amplitude": amplitude,
            "frequency_scale": frequency_scale,
            "steering": steering,
        "pitch_bias_rad": pitch_bias_rad,
            "roll_bias_rad": roll_bias_rad,
            "initial_pitch_deg": initial_pitch_deg,
            "initial_forward_speed_mm_s": initial_forward_speed_mm_s,
            "initial_height_mm": initial_height_mm,
        "differential_pitch_rad": command.differential_pitch_rad,
        "differential_roll_rad": command.differential_roll_rad,
            "tethered": tethered,
            "stabilized": stabilized,
            "effective_command_first": first_effective_command.map(command_json),
            "effective_command_last": last_effective_command.map(command_json),
            "initial_root_position": initial_position,
            "final_root_position": final_position,
            "root_displacement": std::array::from_fn::<_, 3, _>(|axis| final_position[axis] - initial_position[axis]),
            "mean_vertical_force_to_weight": mean_vertical_force_to_weight,
            "mean_force_g_mm_s2": mean_force_g_mm_s2.0,
            "mean_moment_g_mm2_s2": mean_moment_g_mm2_s2.0,
            "mean_root_qfrc_fluid": mean_root_qfrc_fluid,
            "mean_wing_forces_g_mm_s2": mean_wing_forces_g_mm_s2.map(|force| force.0),
            "peak_vertical_force_to_weight": peak_vertical_force_to_weight,
            "peak_strip_speed_mm_s": peak_strip_speed_mm_s,
            "minimum_root_height_mm": minimum_root_height_mm,
            "maximum_root_height_mm": maximum_root_height_mm,
            "initial_root_euler_rad": initial_root_euler_rad,
            "final_root_euler_rad": root_euler_rad(world.root_quaternion()),
            "maximum_abs_roll_rad": maximum_abs_roll_rad,
            "maximum_abs_pitch_rad": maximum_abs_pitch_rad,
            "final_root_quaternion": world.root_quaternion(),
            "final_root_velocity": world.root_velocity(),
            "force_model_limitations": flight.config().limitations,
        }))?
    );
    Ok(())
}

struct FlightIdentificationOptions {
    assets: PathBuf,
    targets: PathBuf,
    baseline_parameters: PathBuf,
    output: PathBuf,
    report: PathBuf,
    iterations: usize,
    seed: u64,
    force: bool,
}

struct FlightTrainingObjective<'a> {
    evaluator: &'a FlightCalibrationEvaluator,
    metrics: &'a [MetricTarget],
    base: FlightDynamicsParameters,
    failed_evaluations: &'a mut usize,
}

impl TrainObjective for FlightTrainingObjective<'_> {
    fn evaluate(&mut self, parameters: &ParameterVector, trials: &[TrialRecord]) -> Result<f64> {
        match self
            .evaluator
            .objective(parameters, trials, self.metrics, self.base)
        {
            Ok(score) => Ok(score),
            Err(_) => {
                *self.failed_evaluations += 1;
                Ok(1.0e6)
            }
        }
    }
}

fn identify_flight(options: FlightIdentificationOptions) -> Result<()> {
    if options.iterations == 0 {
        bail!("flight identification iterations must be positive")
    }
    if options.output == options.report {
        bail!("flight parameter output and report paths must differ")
    }
    for path in [&options.output, &options.report] {
        if path.exists() && !options.force {
            bail!(
                "output already exists: {}; pass --force to replace it",
                path.display()
            )
        }
    }

    let target_asset = FlightTargetAsset::load(&options.targets)?;
    let dataset = target_asset.to_system_id_dataset()?;
    let baseline_parameters = SimulationParameters::load(&options.baseline_parameters)?;
    let baseline_vector =
        FlightCalibrationEvaluator::parameter_vector(baseline_parameters.flight_dynamics)?;
    let training_trials = dataset.train_trials();
    let evaluator =
        FlightCalibrationEvaluator::from_training_trials(&options.assets, &training_trials)?;
    let baseline_train = evaluator.objective(
        &baseline_vector,
        &training_trials,
        &dataset.metrics,
        baseline_parameters.flight_dynamics,
    )?;
    let validation_trials = dataset.trials_for_split(Split::Validation);
    let test_trials = dataset.trials_for_split(Split::Test);
    let baseline_validation = evaluator.objective(
        &baseline_vector,
        &validation_trials,
        &dataset.metrics,
        baseline_parameters.flight_dynamics,
    )?;
    let baseline_test = evaluator.objective(
        &baseline_vector,
        &test_trials,
        &dataset.metrics,
        baseline_parameters.flight_dynamics,
    )?;

    let optimizer = OptimizerConfig::default()
        .with_iterations(options.iterations)
        .with_learning_rate(0.01)
        .with_perturbation(0.04)
        .with_top_k(5)
        .with_checkpoint_every((options.iterations / 4).max(1))
        .with_seed(options.seed);
    let mut failed_candidate_evaluations = 0_usize;
    let optimization = optimize_train(
        &dataset,
        &baseline_vector,
        optimizer,
        FlightTrainingObjective {
            evaluator: &evaluator,
            metrics: &dataset.metrics,
            base: baseline_parameters.flight_dynamics,
            failed_evaluations: &mut failed_candidate_evaluations,
        },
    )?;
    let best_vector = optimization.best_parameters();
    let best_dynamics = FlightCalibrationEvaluator::dynamics_from_vector(
        baseline_parameters.flight_dynamics,
        best_vector,
    )?;
    let best_train = evaluator.objective(
        best_vector,
        &training_trials,
        &dataset.metrics,
        baseline_parameters.flight_dynamics,
    )?;
    let best_validation = evaluator.objective(
        best_vector,
        &validation_trials,
        &dataset.metrics,
        baseline_parameters.flight_dynamics,
    )?;
    let best_test = evaluator.objective(
        best_vector,
        &test_trials,
        &dataset.metrics,
        baseline_parameters.flight_dynamics,
    )?;

    let mut identified_parameters = baseline_parameters;
    identified_parameters.flight_dynamics = best_dynamics;
    let artifact = SimulationParameterArtifact {
        schema: "flybrain.simulation-parameters".to_string(),
        schema_version: 1,
        profile_id: "flybody-flight-identified-v1".to_string(),
        status: "flight-dynamics-identified; neural-io-and-behavior-unfitted".to_string(),
        topology_sha256: None,
        source_dataset_sha256: Some(target_asset.source.sha256.clone()),
        parameters: identified_parameters,
    };
    artifact.validate()?;
    let artifact_bytes = serde_json::to_vec_pretty(&artifact)?;
    let artifact_sha256 = format!("{:x}", Sha256::digest(&artifact_bytes));
    let target_asset_sha256 = sha256_file(&options.targets)?;
    let report = json!({
        "schema": "flybrain.flight-system-identification-report",
        "schema_version": 1,
        "method": "deterministic-spsa; fixed measured-anatomy boundary",
        "dataset": {
            "path": options.targets,
            "asset_sha256": target_asset_sha256,
            "source_hdf5_sha256": target_asset.source.sha256,
            "train_trajectories": training_trials.len(),
            "validation_trajectories": validation_trials.len(),
            "test_trajectories": test_trials.len(),
            "split_rule": "original/reflected pairs remain in one split",
        },
        "fit_scope": {
            "fitted": [
                "target body pitch",
                "horizontal speed",
                "yaw response",
                "attitude response",
                "horizontal force response"
            ],
            "fixed": [
                "FlyWire neuron IDs and CSR topology",
                "FlyWire signed contact counts",
                "wing geometry and waveform",
                "neural sensory and descending-neuron mappings",
                "behavior state machine"
            ],
            "interpretation": "effective low-level flight dynamics, not recovered VNC synapses"
        },
        "scores": {
            "baseline": {
                "train": baseline_train,
                "validation": baseline_validation,
                "test": baseline_test
            },
            "identified": {
                "train": best_train,
                "validation": best_validation,
                "test": best_test
            }
        },
        "optimizer": optimization,
        "failed_candidate_evaluations": failed_candidate_evaluations,
        "output": {
            "path": options.output,
            "sha256": artifact_sha256,
            "status": artifact.status
        }
    });
    write_json_output(&options.output, &artifact_bytes)?;
    write_json_output(&options.report, &serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "parameter_output": options.output,
            "report_output": options.report,
            "train_score": best_train,
            "validation_score": best_validation,
            "test_score": best_test,
            "baseline_train_score": baseline_train,
            "baseline_validation_score": baseline_validation,
            "baseline_test_score": baseline_test,
            "evaluations": optimization.evaluations,
            "failed_candidate_evaluations": failed_candidate_evaluations,
            "parameter_sha256": artifact_sha256,
        }))?
    );
    Ok(())
}

fn evaluate_flight(
    assets: PathBuf,
    targets: PathBuf,
    parameters_path: PathBuf,
    output: Option<PathBuf>,
    force: bool,
) -> Result<()> {
    if output.as_ref().is_some_and(|path| path.exists() && !force) {
        bail!("evaluation output exists; pass --force to replace it")
    }
    let target_asset = FlightTargetAsset::load(&targets)?;
    let dataset = target_asset.to_system_id_dataset()?;
    let parameters = SimulationParameters::load(&parameters_path)?;
    let vector = FlightCalibrationEvaluator::parameter_vector(parameters.flight_dynamics)?;
    let train = dataset.train_trials();
    let validation = dataset.trials_for_split(Split::Validation);
    let test = dataset.trials_for_split(Split::Test);
    let evaluator = FlightCalibrationEvaluator::from_training_trials(&assets, &train)?;
    let train_score = evaluator.objective(
        &vector,
        &train,
        &dataset.metrics,
        parameters.flight_dynamics,
    )?;
    let validation_score = evaluator.objective(
        &vector,
        &validation,
        &dataset.metrics,
        parameters.flight_dynamics,
    )?;
    let test_score =
        evaluator.objective(&vector, &test, &dataset.metrics, parameters.flight_dynamics)?;
    let predictions = [
        evaluator.simulate(parameters.flight_dynamics, "evasion", "original")?,
        evaluator.simulate(parameters.flight_dynamics, "evasion", "reflected")?,
        evaluator.simulate(parameters.flight_dynamics, "saccade", "original")?,
        evaluator.simulate(parameters.flight_dynamics, "saccade", "reflected")?,
    ];
    let report = json!({
        "schema": "flybrain.flight-system-identification-evaluation",
        "schema_version": 1,
        "parameter_path": parameters_path,
        "parameter_sha256": sha256_file(&parameters_path)?,
        "target_path": targets,
        "target_sha256": sha256_file(&targets)?,
        "source_hdf5_sha256": target_asset.source.sha256,
        "scores": {
            "train": train_score,
            "validation": validation_score,
            "test": test_score
        },
        "metric_targets_derived_from_train_only": dataset.metrics,
        "representative_predictions": predictions,
        "controls": {
            "mirrored_pairs_simulated_separately": true,
            "connectome_modified": false,
            "held_out_parameters_refit": false
        },
        "interpretation": "low-level body flight calibration; not a neural/VNC identification result"
    });
    let bytes = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = output {
        write_json_output(&path, &bytes)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn write_json_output(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

fn command_json(command: FlightCommand) -> serde_json::Value {
    json!({
        "enabled": command.enabled,
        "amplitude": command.amplitude,
        "steering": command.steering,
        "planar_velocity_direction": command.planar_velocity_direction,
        "altitude_target_mm": command.altitude_target_mm,
        "frequency_scale": command.frequency_scale,
        "pitch_bias_rad": command.pitch_bias_rad,
        "roll_bias_rad": command.roll_bias_rad,
        "differential_pitch_rad": command.differential_pitch_rad,
        "differential_roll_rad": command.differential_roll_rad,
    })
}

fn root_euler_rad(quaternion: [f64; 4]) -> [f64; 3] {
    let [w, x, y, z] = quaternion;
    [
        (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y)),
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin(),
        (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z)),
    ]
}

struct RenderOptions {
    assets: PathBuf,
    pack: PathBuf,
    output: PathBuf,
    preview: Option<PathBuf>,
    manifest: Option<PathBuf>,
    duration_seconds: f64,
    fps: u32,
    width: u32,
    height: u32,
    control_hz: f64,
    settle_seconds: f64,
    camera: String,
    with_brain: bool,
    parameters: Option<PathBuf>,
    force: bool,
}

fn render_world(options: RenderOptions) -> Result<()> {
    validate_render_options(&options)?;
    let started = Instant::now();
    let gait = GaitLibrary::open(options.assets.join("tripod_gait.json"))?;
    let pack = options.with_brain.then_some(options.pack.as_path());
    let parameters = options
        .parameters
        .as_deref()
        .map(SimulationParameters::load)
        .transpose()?
        .unwrap_or_default();
    let mut simulation = SimulationStepper::new_with_parameters(
        &options.assets,
        pack,
        options.control_hz,
        options.settle_seconds,
        parameters,
    )?;
    let timestep = simulation.world().timestep_seconds();
    let total_steps = rounded_positive_ratio(options.duration_seconds, timestep, "duration")?;
    let control_steps =
        rounded_positive_ratio(1.0 / options.control_hz, timestep, "control period")?;
    let total_frames = (options.duration_seconds * f64::from(options.fps)).round() as u64;
    if total_frames == 0 {
        bail!("render duration is shorter than one video frame")
    }

    let preview_path = options
        .preview
        .clone()
        .unwrap_or_else(|| options.output.with_extension("png"));
    let manifest_path = options
        .manifest
        .clone()
        .unwrap_or_else(|| options.output.with_extension("json"));
    for path in [&preview_path, &manifest_path] {
        if path.exists() && !options.force {
            bail!(
                "output already exists: {}; pass --force to replace it",
                path.display()
            )
        }
    }

    let initial_pose = simulation.world().root_pose();
    let mut renderer = FlyRenderer::fixed_camera(
        simulation.world().model(),
        &options.camera,
        options.width,
        options.height,
    )?;
    let mut video = VideoRecorder::new(
        &options.output,
        options.width,
        options.height,
        options.fps,
        options.force,
    )?;
    video.write_frame(renderer.render(simulation.world_mut().data_mut())?)?;
    let mut rendered_frames = 1_u64;

    let brain_enabled = simulation.brain_enabled();
    let brain_model = simulation.brain_model_name().map(str::to_string);
    let brain_device = simulation.brain_device_name().map(str::to_string);
    let brain_allocated_bytes = simulation.brain_allocated_bytes();
    let brain_materialization = simulation.brain_materialization().map(str::to_string);
    let sensory_neuron_ids = simulation.brain_sensory_neuron_ids().to_vec();
    let motor_neuron_id = simulation.brain_motor_neuron_id();
    let neural_io_stats = simulation.neural_io_stats();

    let mut physics_step = 0_usize;
    let mut total_external_events = 0_u64;
    let mut total_mn9_spikes = 0_u64;
    let mut final_mn9_rate_hz = 0.0;
    let mut peak_mn9_rate_hz = 0.0_f64;
    let mut peak_feeding_extension = 0.0_f64;
    let mut feeding_active_windows = 0_u64;
    let mut first_taste_window_ms = None;
    let mut first_mn9_spike_window_ms = None;
    let mut first_feeding_window_ms = None;
    let mut brain_elapsed = Duration::ZERO;
    let mut brain_encoding_elapsed = Duration::ZERO;
    let mut brain_engine_elapsed = Duration::ZERO;
    let mut taste_windows = 0_u64;
    let mut contact_windows = 0_u64;
    let mut total_population_spikes = 0_u64;
    let mut final_population_rate_hz = 0.0;
    let mut peak_population_rate_hz = 0.0_f64;
    let mut minimum_root_altitude_mm = initial_pose.position[2];
    let mut maximum_root_altitude_mm = initial_pose.position[2];
    let mut minimum_cruise_target_mm: Option<f64> = None;
    let mut maximum_cruise_target_mm: Option<f64> = None;
    let mut peak_abs_altitude_control = 0.0_f64;
    let mut peak_flight_power_increase_rate_hz = 0.0_f64;
    let mut peak_flight_power_decrease_rate_hz = 0.0_f64;
    let mut altitude_target_clamped_windows = 0_u64;

    while physics_step < total_steps {
        let window_steps = control_steps.min(total_steps - physics_step);
        let snapshot = simulation.step_window_steps(window_steps)?;
        physics_step += window_steps;
        if snapshot.taste_active && first_taste_window_ms.is_none() {
            first_taste_window_ms = Some(snapshot.time_seconds * 1000.0);
        }
        taste_windows += u64::from(snapshot.taste_active);
        contact_windows += u64::from(snapshot.contact_count > 0);
        total_external_events += snapshot.taste_event_delta
            + snapshot.olfactory_event_delta
            + snapshot.visual_event_delta
            + snapshot.flight_state_event_delta;
        total_mn9_spikes += u64::from(snapshot.mn9_spike_delta);
        total_population_spikes += snapshot.population_spike_delta;
        brain_elapsed += Duration::from_secs_f64(snapshot.brain_wall_seconds);
        brain_encoding_elapsed += Duration::from_secs_f64(snapshot.brain_encoding_seconds);
        brain_engine_elapsed += Duration::from_secs_f64(snapshot.brain_engine_seconds);
        minimum_root_altitude_mm = minimum_root_altitude_mm.min(snapshot.root_position[2]);
        maximum_root_altitude_mm = maximum_root_altitude_mm.max(snapshot.root_position[2]);
        if snapshot.flight_mode == FlightMode::Cruise {
            minimum_cruise_target_mm = Some(
                minimum_cruise_target_mm.map_or(snapshot.flight_target_height_mm, |current| {
                    current.min(snapshot.flight_target_height_mm)
                }),
            );
            maximum_cruise_target_mm = Some(
                maximum_cruise_target_mm.map_or(snapshot.flight_target_height_mm, |current| {
                    current.max(snapshot.flight_target_height_mm)
                }),
            );
        }
        peak_abs_altitude_control =
            peak_abs_altitude_control.max(snapshot.brain_altitude_control.abs());
        peak_flight_power_increase_rate_hz =
            peak_flight_power_increase_rate_hz.max(snapshot.flight_power_increase_rate_hz);
        peak_flight_power_decrease_rate_hz =
            peak_flight_power_decrease_rate_hz.max(snapshot.flight_power_decrease_rate_hz);
        altitude_target_clamped_windows += u64::from(snapshot.flight_altitude_target_clamped);
        if snapshot.mn9_spike_delta > 0 && first_mn9_spike_window_ms.is_none() {
            first_mn9_spike_window_ms = Some(snapshot.time_seconds * 1000.0);
        }
        final_mn9_rate_hz = snapshot.filtered_mn9_rate_hz;
        peak_mn9_rate_hz = peak_mn9_rate_hz.max(snapshot.filtered_mn9_rate_hz);
        final_population_rate_hz = snapshot.filtered_population_rate_hz;
        peak_population_rate_hz = peak_population_rate_hz.max(snapshot.filtered_population_rate_hz);
        peak_feeding_extension = peak_feeding_extension.max(snapshot.feeding_extension);
        feeding_active_windows += u64::from(snapshot.feeding_extension > 0.05);
        if snapshot.feeding_extension > 0.05 && first_feeding_window_ms.is_none() {
            first_feeding_window_ms = Some(snapshot.time_seconds * 1000.0);
        }
        while rendered_frames < total_frames
            && simulation.world().time() + timestep * 0.5
                >= rendered_frames as f64 / f64::from(options.fps)
        {
            video.write_frame(renderer.render(simulation.world_mut().data_mut())?)?;
            rendered_frames += 1;
        }
    }
    while rendered_frames < total_frames {
        video.write_frame(renderer.render(simulation.world_mut().data_mut())?)?;
        rendered_frames += 1;
    }
    renderer.save_png(&preview_path)?;
    let video_summary = video.finish()?;
    let final_pose = simulation.world().root_pose();
    let environment = &simulation.world().metadata().environment;
    let final_food_distance = distance(
        simulation
            .world()
            .body_position(&environment.taste_source_body)?,
        environment.food_center,
    );
    let video_sha256 = sha256_file(&video_summary.path)?;
    let preview_sha256 = sha256_file(&preview_path)?;

    let manifest = json!({
        "schema": "flybrain-render-v2",
        "video": {
            "path": video_summary.path,
            "sha256": video_sha256,
            "bytes": video_summary.bytes,
            "frames": video_summary.frames,
            "fps": options.fps,
            "width": options.width,
            "height": options.height,
            "camera": options.camera,
        },
        "preview": {
            "path": preview_path,
            "sha256": preview_sha256,
        },
        "simulation": {
            "requested_duration_seconds": options.duration_seconds,
            "actual_duration_seconds": simulation.world().time(),
            "physics_steps": total_steps,
            "physics_timestep_seconds": timestep,
            "control_hz": options.control_hz,
            "control_window_steps": control_steps,
            "wall_seconds": started.elapsed().as_secs_f64(),
            "initial_root_position": initial_pose.position,
            "initial_root_quaternion": initial_pose.quaternion,
            "final_root_position": final_pose.position,
            "final_root_quaternion": final_pose.quaternion,
        },
        "body": {
            "model": simulation.world().metadata().model,
            "physics": simulation.world().metadata().physics,
            "source_manifest": options.assets.join("manifest.json"),
            "actuators": simulation.world().counts().actuators,
            "sensors": simulation.world().counts().sensors,
            "feeding_actuator": simulation.world().metadata().brain_body_interface.feeding_actuator,
            "feeding_actuators": simulation.world().metadata().brain_body_interface.feeding_actuators,
        },
        "environment": {
            "food_center": environment.food_center,
            "taste_radius": environment.taste_radius,
            "taste_source_body": environment.taste_source_body,
            "final_food_distance": final_food_distance,
            "taste_windows": taste_windows,
            "first_taste_window_ms": first_taste_window_ms,
            "contact_windows": contact_windows,
        },
        "controller": {
            "source": gait.source,
            "gait_samples": gait.sample_count,
            "cycle_frequency_hz": gait.cycle_frequency_hz,
            "runtime_interpolation": gait.runtime_interpolation,
            "settle_seconds": options.settle_seconds,
        },
        "flight": {
            "final_mode": simulation.snapshot().flight_mode.label(),
            "minimum_root_altitude_mm": minimum_root_altitude_mm,
            "maximum_root_altitude_mm": maximum_root_altitude_mm,
            "final_target_height_mm": simulation.snapshot().flight_target_height_mm,
            "minimum_cruise_target_mm": minimum_cruise_target_mm,
            "maximum_cruise_target_mm": maximum_cruise_target_mm,
            "target_altitude_bounds_mm": simulation.snapshot().flight_altitude_bounds_mm,
            "final_brain_altitude_control": simulation.snapshot().brain_altitude_control,
            "peak_abs_brain_altitude_control": peak_abs_altitude_control,
            "final_dng02_rate_hz": simulation.snapshot().flight_power_increase_rate_hz,
            "peak_dng02_rate_hz": peak_flight_power_increase_rate_hz,
            "final_dng07_rate_hz": simulation.snapshot().flight_power_decrease_rate_hz,
            "peak_dng07_rate_hz": peak_flight_power_decrease_rate_hz,
            "target_clamped_windows": altitude_target_clamped_windows,
            "decoder_mapping_status": if simulation.snapshot().cns_motor.is_some() {
                "Current DLM/DVM motor population activity selects a bounded altitude target. DNg02/DNg07 provide a hypothetical signed adjustment; the isolated DNg07 sign is unresolved. This is an engineered decoder, not a recovered altitude circuit."
            } else {
                "Peak activity in the broader published flight-DN population selects a bounded altitude setpoint. DNg02 bilateral activity is a separate published power-increase candidate; DNg07 is a candidate negative arm whose isolated sign is not experimentally resolved. Their signed rate difference can adjust the retained target. Both mappings are engineered decoders, not a recovered altitude circuit."
            }
        },
        "brain": {
            "enabled": brain_enabled,
            "model": brain_model,
            "materialization": brain_materialization,
            "embodiment_mode": if simulation.snapshot().cns_motor.is_some() {
                "MaleCNS brain and nerve cord with partial engineered sensory/motor embodiment"
            } else if simulation.brain_materialization() == Some("783") {
                "full v783 connectome with partial engineered embodiment"
            } else if brain_enabled {
                "published connectome with partial engineered embodiment"
            } else {
                "disabled"
            },
            "simulated_neuron_count": simulation.brain_neuron_count(),
            "neural_io": {
                "enabled": simulation.full_neural_io_enabled(),
                "groups": neural_io_stats.groups,
                "selected_root_ids": neural_io_stats.selected_root_ids,
                "present_root_ids": neural_io_stats.present_root_ids,
                "missing_root_ids": neural_io_stats.missing_root_ids,
            },
            "retinal_input_mode": if simulation.snapshot().cns_motor.is_some() {
                "ray/velocity-derived motion and looming projection proxies; not retinal camera transduction"
            } else if brain_enabled {
                "zero summaries; offscreen recorder does not capture the eye cameras"
            } else {
                "disabled with neural engine"
            },
            "device": brain_device,
            "allocated_bytes": brain_allocated_bytes,
            "elapsed_seconds": brain_elapsed.as_secs_f64(),
            "encoding_elapsed_seconds": brain_encoding_elapsed.as_secs_f64(),
            "metal_elapsed_seconds": brain_engine_elapsed.as_secs_f64(),
            "population_spike_count": total_population_spikes,
            "cumulative_spiking_neuron_count": simulation.snapshot().cumulative_spiking_neuron_count,
            "final_filtered_population_rate_hz": final_population_rate_hz,
            "peak_filtered_population_rate_hz": peak_population_rate_hz,
            "sensory_mapping": if simulation.snapshot().cns_motor.is_some() {
                "hash-bound MaleCNS LB3 taste, typed antennal ORNs, motion/loom projection proxies and SApp angular-speed feedback; engineered rate transduction"
            } else {
                "geometric food contact -> bounded engineered meal/taste adaptation -> 150 Hz deterministic rate encoder -> published right sugar GRNs"
            },
            "sensory_neuron_ids": sensory_neuron_ids.iter().map(u64::to_string).collect::<Vec<_>>(),
            "external_event_count": total_external_events,
            "motor_mapping": if simulation.snapshot().cns_motor.is_some() {
                "annotated DLM/DVM, wing-steering and leg motor spikes -> engineered rate decoder, gait/wing waveform and body stabilizer; MN9 -> proboscis extension"
            } else {
                "published contralateral MN9 spikes -> bounded leaky coordinated proboscis command; finite meal state freezes gait and post-meal release restores locomotion"
            },
            "final_cns_motor_readout": simulation.snapshot().cns_motor,
            "motor_neuron_id": motor_neuron_id.map(|id| id.to_string()),
            "mn9_spike_count": total_mn9_spikes,
            "final_filtered_mn9_rate_hz": final_mn9_rate_hz,
            "peak_filtered_mn9_rate_hz": peak_mn9_rate_hz,
            "peak_feeding_extension": peak_feeding_extension,
            "final_feeding_extension": simulation.snapshot().feeding_extension,
            "feeding_active_windows": feeding_active_windows,
            "first_mn9_spike_window_ms": first_mn9_spike_window_ms,
            "first_feeding_window_ms": first_feeding_window_ms,
            "mapping_status": if simulation.snapshot().cns_motor.is_some() {
                "Experimental MaleCNS sensorimotor interface; no claim of recovered natural flight, landing, feeding or grooming programs"
            } else {
                "sugar-to-MN9 is the published model readout; meal duration, taste adaptation, coordinated rostrum/haustellum actuation, and gait arbitration are explicit embodiment mappings"
            },
        },
    });
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("writing {}", manifest_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn validate_render_options(options: &RenderOptions) -> Result<()> {
    if !options.duration_seconds.is_finite() || options.duration_seconds <= 0.0 {
        bail!("duration_seconds must be finite and positive")
    }
    if !options.control_hz.is_finite() || options.control_hz <= 0.0 {
        bail!("control_hz must be finite and positive")
    }
    if !options.settle_seconds.is_finite() || options.settle_seconds < 0.0 {
        bail!("settle_seconds must be finite and non-negative")
    }
    if options.fps == 0 || options.width == 0 || options.height == 0 {
        bail!("fps and render dimensions must be positive")
    }
    Ok(())
}

fn rounded_positive_ratio(numerator: f64, denominator: f64, name: &str) -> Result<usize> {
    let ratio = numerator / denominator;
    if !ratio.is_finite() || ratio < 1.0 {
        bail!("{name} must contain at least one physics step")
    }
    let rounded = ratio.round();
    if (ratio - rounded).abs() > 1e-8 * ratio.max(1.0) {
        bail!("{name} must be an integer multiple of the physics timestep")
    }
    Ok(rounded as usize)
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn sha256_f64(values: &[f64]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::{Cli, Command, LiveStatusContext, live_status};
    use clap::Parser;
    use flybrain_engine::world_sim::SimulationSnapshot;

    #[test]
    fn live_defaults_to_male_cns_roaming_configuration() {
        let cli = Cli::parse_from(["flybrain-world", "view"]);
        match cli.command {
            Command::View {
                pack,
                start_food_distance,
                no_brain,
                ..
            } => {
                assert_eq!(pack.to_str(), Some("outputs/packs/male_cns_v1"));
                assert_eq!(start_food_distance, 40.0);
                assert!(!no_brain);
            }
            _ => panic!("expected live view"),
        }
    }

    #[test]
    fn live_status_fits_the_mujoco_overlay_limit() {
        let status = live_status(LiveStatusContext {
            snapshot: SimulationSnapshot::default(),
            paused: false,
            realtime_factor: 1.0,
            brain_neuron_count: 138_639,
            brain_sensory_neuron_count: 13_111,
            nearest_resource: "fermenting_juice",
            tasted_resource: "fermenting_juice",
            nearest_obstacle: "room_wall_front_right",
        });
        assert!(status.len() < 500, "overlay has {} bytes", status.len());
    }

    #[test]
    fn cns_live_status_fits_the_mujoco_overlay_limit() {
        let status = live_status(LiveStatusContext {
            snapshot: SimulationSnapshot {
                cns_motor: Some(flybrain_engine::brain_bridge::CnsMotorReadout {
                    flight_power_hz: [195.0; 2],
                    walking_hz: [120.0; 2],
                    wing_steering_hz: [120.0; 2],
                    outputs_connected: true,
                    ..Default::default()
                }),
                root_position: [-299.0, -219.0, 208.0],
                time_seconds: 12345.0,
                nearest_resource_distance: 700.0,
                ..Default::default()
            },
            paused: false,
            realtime_factor: 0.25,
            brain_neuron_count: 166_700,
            brain_sensory_neuron_count: 2_324,
            nearest_resource: "fermenting_juice",
            tasted_resource: "fermenting_juice",
            nearest_obstacle: "room_wall_front_right",
        });
        assert!(status.len() < 500, "overlay has {} bytes", status.len());
        assert!(status.contains("MaleCNS"));
        assert!(status.contains("proxies"));
    }
}
