#![cfg(target_os = "macos")]

use flybrain_engine::cns_olfaction::CnsOlfactoryReadout;
use flybrain_engine::flight::{FlightCommand, FlightRuntime};
use flybrain_engine::gait::GaitLibrary;
use flybrain_engine::habitat::Habitat;
use flybrain_engine::odor_guidance::{OdorGuidance, OdorGuidanceParameters};
use flybrain_engine::world::{JOINT_ACTUATOR_COUNT, MuJoCoWorld};

const ASSETS_DIR: &str = "assets/neuromechfly";
const CONTROL_HZ: f64 = 500.0;
const SETTLE_SECONDS: f64 = 0.5;
const WALK_SECONDS: f64 = 24.0;
const GAIT_SPEED_GAIN: f64 = 0.5;

#[derive(Clone, Copy, Debug)]
struct WalkingCase {
    start_xy: [f64; 2],
    translation_slowdown: bool,
}

#[derive(Debug)]
struct WalkingResult {
    case: WalkingCase,
    first_taste_seconds: Option<f64>,
    minimum_mouth_distance_mm: f64,
    final_root_position: [f64; 3],
    path_length_mm: f64,
    maximum_planar_speed_mm_s: f64,
    minimum_up_z: f64,
    guidance_active_seconds: f64,
    maximum_abs_turn: f64,
}

fn ideal_readout(left_ppm: f64, right_ppm: f64) -> CnsOlfactoryReadout {
    let rate = [left_ppm, right_ppm]
        .map(|concentration| 8.0 + 32.0 * concentration / (concentration + 1.0));
    let denominator = left_ppm + right_ppm;
    let contrast = if denominator > f64::EPSILON {
        (left_ppm - right_ppm) / denominator
    } else {
        0.0
    };
    CnsOlfactoryReadout {
        observed_seconds: 1.0,
        rate_hz: rate,
        concentration_ppm: [left_ppm, right_ppm],
        contrast,
        spike_delta: 1,
        ..CnsOlfactoryReadout::default()
    }
}

fn run_case(case: WalkingCase) -> WalkingResult {
    let mut world = MuJoCoWorld::from_assets_dir(ASSETS_DIR).unwrap();
    let gait = GaitLibrary::open(format!("{ASSETS_DIR}/tripod_gait.json")).unwrap();
    let habitat = Habitat::load(ASSETS_DIR).unwrap();
    let mut flight = FlightRuntime::new(ASSETS_DIR, &world).unwrap();
    let banana_index = habitat
        .resources()
        .iter()
        .position(|resource| resource.id == "banana")
        .unwrap();
    let timestep = world.timestep_seconds();
    let control_steps = (1.0 / CONTROL_HZ / timestep).round() as usize;
    assert_eq!(control_steps, 20);
    let window_seconds = control_steps as f64 * timestep;
    let settle_windows = (SETTLE_SECONDS / window_seconds).round() as usize;
    let walk_windows = (WALK_SECONDS / window_seconds).round() as usize;

    {
        let data = world.data_mut();
        data.qpos_mut()[0] = case.start_xy[0];
        data.qpos_mut()[1] = case.start_xy[1];
        data.qpos_mut()[2] = 2.1;
        data.qpos_mut()[3..7].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        data.qvel_mut()[..6].fill(0.0);
        data.forward();
    }
    let neutral_joint_controls: [f64; JOINT_ACTUATOR_COUNT] = world
        .neutral_control()
        .iter()
        .take(JOINT_ACTUATOR_COUNT)
        .copied()
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    world.set_joint_controls(&neutral_joint_controls).unwrap();
    world.set_adhesion_controls(&[0.0; 6]).unwrap();

    let mut phase_rad = 0.0;
    for window in 0..settle_windows {
        let ramp = ((window as f64 * window_seconds) / SETTLE_SECONDS).clamp(0.0, 1.0);
        let gait_command = gait.sample_bilateral(phase_rad, [ramp, ramp]).unwrap();
        let mut joint_controls = gait_command.joint_controls;
        for (index, control) in joint_controls.iter_mut().enumerate() {
            let [minimum, maximum] = world.actuator_control_range(index).unwrap();
            *control = control.clamp(minimum, maximum);
        }
        world.set_joint_controls(&joint_controls).unwrap();
        world
            .set_adhesion_controls(if ramp >= 0.5 {
                &gait_command.adhesion
            } else {
                &[0.0; 6]
            })
            .unwrap();
        for _ in 0..control_steps {
            flight
                .advance(&mut world, FlightCommand::default(), [0.0; 3])
                .unwrap();
        }
        phase_rad = gait
            .advance_phase(phase_rad, window_seconds, GAIT_SPEED_GAIN)
            .unwrap();
    }
    assert!(world.root_position()[2].is_finite());
    let mut guidance = OdorGuidance::default();
    let parameters = OdorGuidanceParameters {
        close_concentration_ppm: 2.0,
        ..OdorGuidanceParameters::default()
    };
    let mut first_taste_seconds = None;
    let mut minimum_mouth_distance_mm = f64::INFINITY;
    let mut path_length_mm = 0.0;
    let mut maximum_planar_speed_mm_s = 0.0_f64;
    let mut minimum_up_z = f64::INFINITY;
    let mut guidance_active_seconds = 0.0;
    let mut maximum_abs_turn = 0.0_f64;
    let mut previous_position = world.root_position();

    for _ in 0..walk_windows {
        let left_antenna = world.body_position("fly/l_funiculus").unwrap();
        let right_antenna = world.body_position("fly/r_funiculus").unwrap();
        let mouth = world
            .body_position(&world.metadata().environment.taste_source_body)
            .unwrap();
        let habitat_sample =
            habitat.sample(left_antenna, right_antenna, mouth, [1.1, 0.0, 0.25], true);
        let readout = ideal_readout(habitat_sample.odor_left_ppm, habitat_sample.odor_right_ppm);
        let command = guidance.update(
            readout,
            world.root_position()[2],
            window_seconds,
            true,
            parameters,
        );
        let translation = if case.translation_slowdown {
            1.0 - command.steering.abs()
        } else {
            1.0
        };
        let side_drive = [
            translation - 0.5 * command.steering,
            translation + 0.5 * command.steering,
        ];
        let gait_command = gait.sample_bilateral(phase_rad, side_drive).unwrap();
        let mut joint_controls = gait_command.joint_controls;
        for (index, control) in joint_controls.iter_mut().enumerate() {
            let [minimum, maximum] = world.actuator_control_range(index).unwrap();
            *control = control.clamp(minimum, maximum);
        }
        world.set_joint_controls(&joint_controls).unwrap();
        world.set_adhesion_controls(&gait_command.adhesion).unwrap();
        for _ in 0..control_steps {
            flight
                .advance(&mut world, FlightCommand::default(), [0.0; 3])
                .unwrap();
            let velocity = world.root_velocity();
            maximum_planar_speed_mm_s =
                maximum_planar_speed_mm_s.max(velocity[3].hypot(velocity[4]));
            let [_, x, y, _] = world.root_quaternion();
            minimum_up_z = minimum_up_z.min(1.0 - 2.0 * (x * x + y * y));
        }
        phase_rad = gait
            .advance_phase(phase_rad, window_seconds, GAIT_SPEED_GAIN)
            .unwrap();

        let position = world.root_position();
        path_length_mm +=
            (position[0] - previous_position[0]).hypot(position[1] - previous_position[1]);
        previous_position = position;
        let mouth = world
            .body_position(&world.metadata().environment.taste_source_body)
            .unwrap();
        let tasted = habitat.sample(
            world.body_position("fly/l_funiculus").unwrap(),
            world.body_position("fly/r_funiculus").unwrap(),
            mouth,
            [1.1, 0.0, 0.25],
            true,
        );
        let banana_distance = mouth
            .iter()
            .zip(habitat.resources()[banana_index].position)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>()
            .sqrt();
        minimum_mouth_distance_mm = minimum_mouth_distance_mm.min(banana_distance);
        if tasted.tasted_resource == Some(banana_index) && first_taste_seconds.is_none() {
            first_taste_seconds = Some(world.time());
        }
        if command.active {
            guidance_active_seconds += window_seconds;
        }
        maximum_abs_turn = maximum_abs_turn.max(command.steering.abs());
    }

    WalkingResult {
        case,
        first_taste_seconds,
        minimum_mouth_distance_mm,
        final_root_position: world.root_position(),
        path_length_mm,
        maximum_planar_speed_mm_s,
        minimum_up_z,
        guidance_active_seconds,
        maximum_abs_turn,
    }
}

#[test]
#[ignore = "requires a long MuJoCo walking fixture run"]
fn ideal_odor_guidance_walking_fixture_compares_translation_policies() {
    let mut results = Vec::new();
    for start_xy in [[20.0, 10.0], [45.0, 20.0]] {
        for translation_slowdown in [false, true] {
            results.push(run_case(WalkingCase {
                start_xy,
                translation_slowdown,
            }));
        }
    }
    for result in &results {
        eprintln!(
            "ideal_decoder_fixture start={:?} slowdown={} taste={:?} min_mouth_distance={:.3} final={:?} path={:.3} max_speed={:.3} min_up_z={:.3} active={:.3}s max_turn={:.3}",
            result.case.start_xy,
            result.case.translation_slowdown,
            result.first_taste_seconds,
            result.minimum_mouth_distance_mm,
            result.final_root_position,
            result.path_length_mm,
            result.maximum_planar_speed_mm_s,
            result.minimum_up_z,
            result.guidance_active_seconds,
            result.maximum_abs_turn,
        );
        assert!(result.minimum_up_z.is_finite());
        assert!(result.maximum_planar_speed_mm_s.is_finite());
        assert!(result.path_length_mm.is_finite());
    }
    assert!(
        results
            .iter()
            .any(|result| result.first_taste_seconds.is_some())
    );
}
