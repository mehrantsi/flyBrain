#![cfg(target_os = "macos")]

use flybrain_engine::flight::{
    FlightCommand, FlightDynamicsParameters, FlightRuntime, FlightStabilizer,
};
use flybrain_engine::gait::GaitLibrary;
use flybrain_engine::world::{JOINT_ACTUATOR_COUNT, MuJoCoWorld};

const ASSETS_DIR: &str = "assets/neuromechfly";
const LANDING_HEIGHT_MM: f64 = 0.6;
const GAIT_SPEED_GAIN: f64 = 0.5;

fn up_axis_z(quaternion: [f64; 4]) -> f64 {
    let [_, x, y, _] = quaternion;
    1.0 - 2.0 * (x * x + y * y)
}

struct TouchdownResult {
    initial_position: [f64; 3],
    touchdown_position: [f64; 3],
    final_position: [f64; 3],
    touchdown_up_z: f64,
    minimum_up_z: f64,
    maximum_contact_count: usize,
    touchdown_time_seconds: f64,
}

fn run_touchdown(initial_speed_mm_s: f64) -> TouchdownResult {
    let mut world = MuJoCoWorld::from_assets_dir(ASSETS_DIR).unwrap();
    let dynamics = FlightDynamicsParameters::default();
    let half_pitch = 0.5 * dynamics.target_pitch_rad;
    world.data_mut().qpos_mut()[2] = 8.0;
    world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
        half_pitch.cos(),
        0.0,
        half_pitch.sin(),
        0.0,
    ]);
    world.data_mut().qvel_mut()[..6].fill(0.0);
    world.data_mut().qvel_mut()[0] = initial_speed_mm_s;
    world.data_mut().forward();
    let initial_position = world.root_position();
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

    let mut flight = FlightRuntime::new_with_parameters(ASSETS_DIR, &world, dynamics).unwrap();
    let stabilizer = FlightStabilizer::from_dynamics(dynamics).unwrap();
    let gait = GaitLibrary::open(format!("{ASSETS_DIR}/tripod_gait.json")).unwrap();
    let timestep = world.timestep_seconds();
    let landing_amplitude_scale = 0.94 * 0.87_f64.sqrt();
    let mut maximum_contact_count = 0_usize;
    let mut minimum_height = f64::INFINITY;
    let mut maximum_height = f64::NEG_INFINITY;
    let mut touchdown_position = None;
    for _ in 0..10_000 {
        world.set_joint_controls(&neutral_joint_controls).unwrap();
        world.set_adhesion_controls(&[0.0; 6]).unwrap();
        let velocity = world.root_velocity();
        let height_error = LANDING_HEIGHT_MM - world.root_position()[2];
        let frequency_scale =
            (1.25 + 0.004 * height_error - 0.0005 * velocity[5]).clamp(1.05, 1.45);
        let base_command = FlightCommand {
            enabled: true,
            amplitude: 1.0,
            horizontal_speed_scale: 0.0,
            altitude_target_mm: Some(LANDING_HEIGHT_MM),
            body_pitch_target_rad: Some(0.0),
            frequency_scale,
            ..FlightCommand::default()
        };
        let command = stabilizer
            .command_with_base_limited(
                world.root_quaternion(),
                velocity,
                base_command,
                landing_amplitude_scale,
                flight.config(),
            )
            .unwrap();
        flight
            .advance(&mut world, command, [35.0, 8.0, 0.0])
            .unwrap();
        let position = world.root_position();
        minimum_height = minimum_height.min(position[2]);
        maximum_height = maximum_height.max(position[2]);
        let contact_count = world
            .support_contacts()
            .unwrap()
            .into_iter()
            .filter(|&contact| contact)
            .count();
        maximum_contact_count = maximum_contact_count.max(contact_count);
        if contact_count >= 2 {
            touchdown_position = Some(position);
            break;
        }
    }
    let touchdown_position = touchdown_position.unwrap_or_else(|| {
        panic!(
            "speed={initial_speed_mm_s} no touchdown t={} z={} vz={} up_z={} contacts_max={} z_range=[{minimum_height},{maximum_height}]",
            world.time(),
            world.root_position()[2],
            world.root_velocity()[5],
            up_axis_z(world.root_quaternion()),
            maximum_contact_count,
        )
    });
    let touchdown_time_seconds = world.time();
    let touchdown_up_z = up_axis_z(world.root_quaternion());

    let mut phase_rad = 0.0;
    let mut minimum_up_z = touchdown_up_z;
    for _ in 0..40_000 {
        let gait_command = gait.sample(phase_rad, 1.0, 0.0).unwrap();
        world
            .set_joint_controls(&gait_command.joint_controls)
            .unwrap();
        world.set_adhesion_controls(&gait_command.adhesion).unwrap();
        world.set_feeding_extension(0.0).unwrap();
        flight
            .advance(&mut world, FlightCommand::default(), [0.0; 3])
            .unwrap();
        phase_rad = gait
            .advance_phase(phase_rad, timestep, GAIT_SPEED_GAIN)
            .unwrap();
        minimum_up_z = minimum_up_z.min(up_axis_z(world.root_quaternion()));
    }
    TouchdownResult {
        initial_position,
        touchdown_position,
        final_position: world.root_position(),
        touchdown_up_z,
        minimum_up_z,
        maximum_contact_count,
        touchdown_time_seconds,
    }
}

#[test]
fn landing_handoff_retracts_flight_and_walks_upright() {
    for initial_speed_mm_s in [0.0, 15.0] {
        let result = run_touchdown(initial_speed_mm_s);
        let displacement = (result.final_position[0] - result.touchdown_position[0])
            .hypot(result.final_position[1] - result.touchdown_position[1]);
        eprintln!(
            "speed={initial_speed_mm_s} touchdown_time={} touchdown={:?} final={:?} contacts_max={} touchdown_up_z={} minimum_up_z={} displacement={displacement}",
            result.touchdown_time_seconds,
            result.touchdown_position,
            result.final_position,
            result.maximum_contact_count,
            result.touchdown_up_z,
            result.minimum_up_z,
        );
        let landing_shift = (result.touchdown_position[0] - result.initial_position[0])
            .hypot(result.touchdown_position[1] - result.initial_position[1]);
        assert!(landing_shift < 2.1, "landing shift={landing_shift}");
        assert!(result.touchdown_up_z > 0.9);
        assert!(result.minimum_up_z > 0.9);
        assert!(
            displacement > 2.0,
            "speed={initial_speed_mm_s} displacement={displacement}"
        );
    }
}
